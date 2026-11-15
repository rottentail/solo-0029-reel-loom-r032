import logging
import threading
from .errors import LoomError
from .probe import inspect_media
from .timebase import seconds

logger = logging.getLogger(__name__)


class Renderer:
    def __init__(self, settings, blobs, tickets, planner, executor, artifacts):
        self.settings, self.blobs, self.tickets = settings, blobs, tickets
        self.planner, self.executor, self.artifacts = planner, executor, artifacts
        self.stopping = threading.Event()
        self.threads = []

    def execute(self, ticket):
        ticket_id = ticket["id"]
        try:
            task = self.planner.compile(ticket["snapshot"], lambda key: self.blobs.resolve(key, verify=True))
            with self.executor.perform(task,
                    cancelled=lambda: self.stopping.is_set() or self.tickets.cancellation_requested(ticket_id),
                    progress=lambda value: self.tickets.progress(ticket_id, value)) as done:
                rendition = done.artifact("rendition")
                metadata = inspect_media(rendition, self.settings)
                actual = seconds(metadata["duration_tick"])
                if abs(actual-task.duration) > task.tolerance:
                    raise LoomError("rendition_duration", "Encoded duration differs from the frozen reel", 422,
                                    {"expected": task.duration, "actual": actual})
                return self.artifacts.commit_render(ticket_id, rendition,
                                                  {**metadata, "fingerprint": ticket["fingerprint"]})
        except LoomError as exc:
            self.tickets.fail(ticket_id, exc)
            return None
        except Exception:
            logger.exception("Render failed unexpectedly: %s", ticket_id)
            self.tickets.fail(ticket_id, LoomError("internal_render", "Unexpected rendering failure", 500))
            return None

    def run_once(self):
        ticket = self.tickets.claim()
        if ticket is None:
            return None
        self.execute(ticket)
        return self.tickets.get(ticket["id"])

    def loop(self):
        while not self.stopping.is_set():
            if self.run_once() is None:
                self.stopping.wait(.2)

    def start(self):
        self.stopping.clear()
        for index in range(self.settings.workers):
            thread = threading.Thread(target=self.loop, name=f"loom-render-{index}", daemon=True)
            thread.start()
            self.threads.append(thread)

    def close(self):
        self.stopping.set()
        for thread in self.threads:
            thread.join(timeout=5)
        self.threads.clear()
