"""A bounded production walkthrough using generated audiovisual media and HTTP."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from .config import Settings
from .application import Application
from .api import Api
from .http import MediaHttpServer
from .client import Client
from .errors import LoomError, require
from .inproc import InProcessExecutor

WORKFLOWS = ("ingest", "revisions", "looks", "render", "signal", "waveform", "loudness",
             "sheet", "captions", "imprint", "cancel-retry", "recovery", "resources", "matrix", "conformance", "lineage",
             "boundary", "executor")


def expect_error(action, code):
    try:
        action()
    except LoomError as exc:
        require(exc.code == code, "walkthrough_error", "Unexpected workflow rejection", expected=code, actual=exc.code)
        return exc.as_dict()
    raise RuntimeError("The walkthrough expected an operation to be rejected: " + code)


def synthesize(settings, output, duration=3, width=320, height=180, frequency=659):
    """Generate bounded sample media with the local FFmpeg executable."""
    command = [settings.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate=25:duration={duration}",
               "-f", "lavfi", "-i", f"sine=frequency={frequency}:sample_rate=48000:duration={duration}",
               "-af", "volume=enable='between(t,1,1.5)':volume=0",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(output)]
    completed = subprocess.run(command, capture_output=True, timeout=120)
    require(completed.returncode == 0, "walkthrough_sample", "Sample synthesis failed",
            stderr=completed.stderr.decode(errors="replace")[-600:])
    return output


class Walkthrough:
    def __init__(self, settings, executor=None):
        self.settings = settings
        self.app = Application(settings, start_workers=False, executor=executor)
        self.server = MediaHttpServer(("127.0.0.1", 0), Api(self.app))
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .02}, daemon=True)
        self.thread.start()
        self.client = Client(f"http://127.0.0.1:{self.server.server_port}/loom/v1")

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.app.close()

    def setup(self):
        with self.app.blobs.temporary(".mp4") as sample:
            synthesize(self.settings, sample)
            self.content = sample.read_bytes()
        self.asset = self.client.ingest(self.content, "harbor-tone")
        self.slices = [{"asset": self.asset["id"], "in": .2, "out": 1.2, "label": "开场"},
                       {"asset": self.asset["id"], "in": 1.6, "out": 2.8, "label": "收束"}]
        self.reel = self.client.create_reel("回声样片", self.slices, "demo-reel")
        self.look = self.client.put_look("preview", {"kind": "video", "width": 320, "height": 180})
        self.audio_look = self.client.put_look("listening", {"kind": "audio", "sample_rate": 48000})
        self.ticket = self.client.render(self.reel["id"], "preview", key="demo-render")
        return self

    def finish(self):
        ticket = self.client.request("GET", "/tickets/"+self.ticket["id"])
        if ticket["state"] == "queued":
            self.app.renderer.run_once()
        ticket = self.client.wait(self.ticket["id"], timeout=5)
        require(ticket["state"] == "succeeded", "walkthrough_render", "Walkthrough render did not succeed", ticket=ticket)
        return ticket

    def run(self, name):
        c, a = self.client, self.asset["id"]
        if name == "ingest":
            duplicate = c.ingest(self.content, "duplicate-name")
            require(duplicate["id"] == a, "walkthrough_dedup", "Identical ready media should be reused")
            failure = expect_error(lambda: c.ingest(b"not-media", "bad"), "unreadable_media")
            return {"asset": self.asset, "deduplicated": True, "failure": failure}
        if name == "revisions":
            updated = c.replace_reel(self.reel["id"], 1, "回声样片·修订", list(reversed(self.slices)))
            old = c.request("GET", "/reels/"+self.reel["id"]+"?revision=1")
            require(old["hash"] == self.reel["hash"] and updated["revision"] == 2,
                    "walkthrough_revision", "Historical reel changed")
            failure = expect_error(lambda: c.replace_reel(self.reel["id"], 1, "旧稿", self.slices), "revision_conflict")
            return {"head": updated["head"], "historical_hash": old["hash"], "failure": failure}
        if name == "looks":
            new = c.put_look("preview", {"kind": "video", "width": 640, "height": 360}, 1)
            ticket = c.request("GET", "/tickets/"+self.ticket["id"])
            require(ticket["look_version"] == 1 and new["version"] == 2, "walkthrough_look", "Queued look drifted")
            failure = expect_error(lambda: c.put_look("preview", {"kind":"video"}, 1), "revision_conflict")
            return {"new_version": 2, "frozen_version": ticket["look_version"], "failure": failure}
        if name == "render":
            ticket = self.finish()
            body = {"reel": self.reel["id"], "look": "preview"}
            replay = c.request("POST", "/tickets", body, {"Idempotency-Key":"demo-render"})
            require(replay["id"] == ticket["id"], "walkthrough_idempotency", "Render replay changed identity")
            failure = expect_error(lambda: c.request("POST", "/tickets", {**body,"look":"listening"},
                                                   {"Idempotency-Key":"demo-render"}), "idempotency_conflict")
            audio = c.render(self.reel["id"], "listening")
            self.app.renderer.run_once()
            audio = c.wait(audio["id"], 5)
            require(audio["state"] == "succeeded", "walkthrough_audio", "Audio rendition did not succeed")
            return {"video": ticket, "audio": audio, "failure": failure}
        if name in ("signal", "waveform", "loudness"):
            result = c.request("POST", f"/assets/{a}/{name}", {})
            replay = c.request("POST", f"/assets/{a}/{name}", {})
            require(result["id"] == replay["id"], "walkthrough_analysis", "Analysis cache should be content/settings stable")
            failure = expect_error(lambda: c.request("POST", f"/assets/{a}/{name}", {"unsupported": 1}), "unknown_fields")
            return {"analysis": result, "cached": True, "failure": failure}
        if name == "sheet":
            result = c.request("POST", f"/assets/{a}/contact-sheet", {"count": 6, "columns": 3})
            media = c.request("GET", "/artifacts/"+result["id"]+"/content", raw=True)
            require(media.startswith(b"\xff\xd8"), "walkthrough_sheet", "Contact sheet is not JPEG")
            failure = expect_error(lambda: c.request("POST", f"/assets/{a}/contact-sheet", {"count":1,"columns":3}), "sheet_layout")
            return {"artifact": result, "bytes": len(media), "failure": failure}
        if name == "captions":
            track = c.request("POST", f"/assets/{a}/cues", {"language":"zh-CN",
                              "srt":"1\n00:00:00,000 --> 00:00:02,900\n潮声\n"})
            projected = c.request("POST", f'/reels/{self.reel["id"]}/captions', {"revision":1,"tracks":[track["id"]]})
            require(len(projected["cues"]) == 2, "walkthrough_captions", "A cue spanning cuts must project to both slices")
            failure = expect_error(lambda: c.request("POST", f"/assets/{a}/cues", {"language":"zh","srt":"broken"}), "cue_format")
            return {"projection": projected, "track": track, "failure": failure}
        if name == "imprint":
            failure = expect_error(lambda: c.imprint(self.ticket["id"]), "imprint_state")
            ticket = self.finish()
            imprint = c.imprint(ticket["id"])
            verified = c.request("GET", f'/imprints/{imprint["id"]}/verify')
            self.client.replace_reel(self.reel["id"], 1, "新的草剪", list(reversed(self.slices)))
            unchanged = c.imprint(ticket["id"])
            require(imprint["id"] == unchanged["id"], "walkthrough_imprint", "Upstream editing changed frozen imprint")
            return {"imprint": imprint, "verified": verified, "failure": failure}
        if name == "cancel-retry":
            cancelled = c.request("POST", f'/tickets/{self.ticket["id"]}/cancel')
            require(cancelled["state"] == "cancelled", "walkthrough_cancel", "Queued render did not cancel")
            retried = c.request("POST", f'/tickets/{self.ticket["id"]}/retry')
            ticket = self.finish()
            failure = expect_error(lambda: c.request("POST", f'/tickets/{self.ticket["id"]}/retry'), "retry_state")
            require(retried["attempt"] == 2, "walkthrough_attempt", "Retry attempt was not incremented")
            return {"ticket": ticket, "failure": failure}
        if name == "recovery":
            claimed = self.app.tickets.claim()
            self.close()
            with Application(self.settings, start_workers=False) as restarted:
                recovered = restarted.tickets.get(claimed["id"])
                require(recovered["state"] == "failed" and recovered["error"]["error"] == "worker_interrupted",
                        "walkthrough_recovery", "Interrupted render was not recovered")
                restarted.tickets.retry(claimed["id"])
                finished = restarted.renderer.run_once()
                require(finished["state"] == "succeeded", "walkthrough_recovery", "Recovered render could not retry")
                return {"interrupted": recovered, "retried_state": finished["state"]}
        if name == "resources":
            c.request("POST", f"/assets/{a}/retire")
            failure = expect_error(lambda: c.create_reel("退役来源", self.slices), "asset_retired")
            ticket = self.finish()
            sweep = c.request("POST", "/resources/sweep", {"dry_run":False})
            status = c.request("GET", "/resources")
            require(status["consistent"] and not sweep["removed"], "walkthrough_resources", "Referenced source was collected")
            return {"reconcile":status,"sweep":sweep,"render_state":ticket["state"],"failure":failure}
        if name == "matrix":
            body = {"reel":self.reel["id"],"looks":[{"id":"preview"},{"id":"listening"}]}
            matrix = c.request("POST", "/matrices", body, {"Idempotency-Key":"demo-matrix"})
            before = len(c.request("GET", "/tickets")["items"])
            failure = expect_error(lambda: c.request("POST", "/matrices", {**body,"looks":[{"id":"preview"},{"id":"missing-look"}]}), "not_found")
            require(len(c.request("GET", "/tickets")["items"]) == before, "walkthrough_matrix", "Invalid matrix committed partial work")
            while self.app.renderer.run_once() is not None:
                pass
            finished = c.request("GET", "/matrices/"+matrix["id"])
            require(finished["state"] == "succeeded", "walkthrough_matrix", "Matrix did not complete")
            return {"matrix": finished,"failure":failure}
        if name == "conformance":
            ticket = self.finish()
            artifact = ticket["artifacts"][0]
            result = c.request("GET", f'/artifacts/{artifact["id"]}/conformance')
            require(result["valid"], "walkthrough_conformance", "Rendered media violates frozen look", report=result)
            failure = expect_error(lambda: c.request("GET", "/artifacts/missing/conformance"), "not_found")
            return {"conformance":result,"failure":failure}
        if name == "lineage":
            self.finish()
            imprint = c.imprint(self.ticket["id"])
            result = c.request("GET", f'/tickets/{self.ticket["id"]}/lineage')
            require(any(n["id"] == imprint["id"] for n in result["nodes"]), "walkthrough_lineage", "Imprint missing from lineage")
            failure = expect_error(lambda: c.request("GET", "/tickets/missing/lineage"), "not_found")
            return {"lineage":result,"failure":failure}
        if name == "boundary":
            return self.run_boundary()
        if name == "executor":
            return self.run_executor()
        raise ValueError("Unknown walkthrough: " + name)

    def run_boundary(self):
        """Drive every FFmpeg-backed domain flow through the in-process executor."""
        c, a = self.client, self.asset["id"]
        executor = self.app.executor
        require(isinstance(executor, InProcessExecutor),
                "walkthrough_boundary", "The boundary walkthrough requires the in-process executor")
        require(self.app.renderer.executor is executor and self.app.analysis.executor is executor
                and self.app.frames.executor is executor,
                "walkthrough_boundary", "Analysis and rendering must share one execution backend")
        video = self.finish()
        artifact = video["artifacts"][0]
        conformance = c.request("GET", f'/artifacts/{artifact["id"]}/conformance')
        require(conformance["valid"], "walkthrough_boundary", "In-process rendition violates the frozen look",
                report=conformance)
        audio = c.render(self.reel["id"], "listening")
        self.app.renderer.run_once()
        audio = c.wait(audio["id"], 5)
        require(audio["state"] == "succeeded", "walkthrough_boundary", "In-process audio rendition did not succeed")
        analyses = {}
        for kind in ("signal", "waveform", "loudness"):
            result = c.request("POST", f"/assets/{a}/{kind}", {})
            replay = c.request("POST", f"/assets/{a}/{kind}", {})
            require(result["id"] == replay["id"], "walkthrough_boundary", "Analysis cache drifted on the swapped executor")
            analyses[kind] = result
        silence = analyses["signal"]["result"]["silence"]
        require(any(item["end_tick"]-item["start_tick"] >= 400_000 for item in silence),
                "walkthrough_boundary", "In-process decode lost the muted interval", silence=silence)
        require(analyses["loudness"]["result"]["integrated_lufs"] is not None,
                "walkthrough_boundary", "In-process loudness report is not finite")
        sheet = c.request("POST", f"/assets/{a}/contact-sheet", {"count": 6, "columns": 3})
        media = c.request("GET", "/artifacts/"+sheet["id"]+"/content", raw=True)
        require(media.startswith(b"\xff\xd8"), "walkthrough_boundary", "In-process sheet is not JPEG")
        return {"executor": type(executor).__name__, "subprocesses": False,
                "video": video["state"], "audio": audio["state"], "conformance_valid": conformance["valid"],
                "analyses": {kind: analyses[kind]["id"] for kind in analyses},
                "loudness": analyses["loudness"]["result"], "sheet_bytes": len(media)}

    def run_executor(self):
        """Prove capacity, cancellation, and timeout semantics of the execution boundary."""
        c = self.client
        c.request("POST", f'/tickets/{self.ticket["id"]}/cancel')
        with self.app.blobs.temporary(".mp4") as sample:
            synthesize(self.settings, sample, duration=20, width=1280, height=720, frequency=440)
            long_content = sample.read_bytes()
        long_asset = c.ingest(long_content, "long-take")
        long_reel = c.create_reel("长镜头", [{"asset": long_asset["id"], "in": 1.0, "out": 19.0}], "demo-long-reel")
        c.put_look("big-screen", {"kind": "video", "width": 1280, "height": 720})
        executor = self.app.executor
        resolve = lambda key: self.app.blobs.resolve(key, verify=True)

        # Cancellation while waiting for capacity must not consume or leak a slot.
        holder_ticket = c.render(long_reel["id"], "big-screen")
        task = self.app.planner.compile(self.app.tickets.get(holder_ticket["id"])["snapshot"], resolve)
        holder_cancel, waiter_cancel = threading.Event(), threading.Event()
        holder_errors, waiter_errors = [], []

        def occupy():
            try:
                with executor.perform(task, cancelled=holder_cancel.is_set):
                    pass
            except LoomError as exc:
                holder_errors.append(exc)

        holders = [threading.Thread(target=occupy, name=f"loom-hold-{i}", daemon=True) for i in range(2)]
        for thread in holders:
            thread.start()
        deadline = time.monotonic()+10
        while executor.busy < 2 and time.monotonic() < deadline:
            time.sleep(.01)
        require(executor.busy == 2, "walkthrough_executor", "Holders did not occupy execution capacity")

        def wait_for_capacity():
            try:
                with executor.perform(task, cancelled=waiter_cancel.is_set):
                    pass
            except LoomError as exc:
                waiter_errors.append(exc)

        waiter = threading.Thread(target=wait_for_capacity, name="loom-wait", daemon=True)
        waiter.start()
        time.sleep(.3)
        require(waiter.is_alive() and executor.busy == 2,
                "walkthrough_executor", "The waiter unexpectedly acquired capacity")
        waiter_cancel.set()
        waiter.join(timeout=5)
        require(waiter_errors and waiter_errors[0].code == "render_cancelled",
                "walkthrough_executor", "Cancelling a capacity wait did not raise RenderCancelled")
        require(executor.busy == 2, "walkthrough_executor", "A cancelled wait consumed execution capacity")
        holder_cancel.set()
        for thread in holders:
            thread.join(timeout=15)
        require(len(holder_errors) == 2 and all(exc.code == "render_cancelled" for exc in holder_errors),
                "walkthrough_executor", "Cancelling a running task did not raise RenderCancelled")
        require(executor.busy == 0, "walkthrough_executor", "Execution capacity leaked after cancellation")
        require(not list((self.settings.root/"scratch").glob("loom-*")),
                "walkthrough_executor", "Scratch files leaked after cancellation")
        released = self.app.renderer.run_once()
        require(released["id"] == holder_ticket["id"] and released["state"] == "succeeded",
                "walkthrough_executor", "A queued render could not complete after capacity was released")

        # A running task that exceeds its deadline fails without registering an artifact.
        timed_ticket = c.render(long_reel["id"], "big-screen")
        self.close()
        tight = replace(self.settings, render_timeout=1)
        with Application(tight, start_workers=False) as limited:
            finished = limited.renderer.run_once()
            require(finished["id"] == timed_ticket["id"] and finished["state"] == "failed"
                    and finished["error"]["error"] == "encoder_timeout",
                    "walkthrough_executor", "A running task did not fail on its deadline", ticket=finished)
            require(finished["artifacts"] == [], "walkthrough_executor", "A failed task registered an artifact")
            require(limited.executor.busy == 0, "walkthrough_executor", "A deadline kill leaked execution capacity")
            require(not list((self.settings.root/"scratch").glob("loom-*")),
                    "walkthrough_executor", "Scratch files leaked after a deadline kill")

        # The interrupted ticket retries on its frozen snapshot and completes.
        with Application(self.settings, start_workers=False) as restored:
            restored.tickets.retry(timed_ticket["id"])
            finished = restored.renderer.run_once()
            require(finished["state"] == "succeeded" and finished["artifacts"],
                    "walkthrough_executor", "An interrupted render could not complete after retry", ticket=finished)
            conformance = restored.conformance.inspect(finished["artifacts"][0]["id"])
            require(conformance["valid"], "walkthrough_executor", "Retried rendition violates the frozen look",
                    report=conformance)
        return {"wait_cancelled": waiter_errors[0].code, "running_cancelled": [exc.code for exc in holder_errors],
                "capacity_released": released["state"], "timeout_state": "failed", "timeout_artifacts": 0,
                "retried_state": finished["state"], "retried_conformance": conformance["valid"]}


def main():
    parser = argparse.ArgumentParser(description="Run a bounded audiovisual production walkthrough")
    parser.add_argument("--workflow", choices=WORKFLOWS, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="reel-loom-demo-") as folder:
        settings = replace(Settings.from_env(), root=Path(folder), render_timeout=30)
        executor = InProcessExecutor(settings) if args.workflow == "boundary" else None
        demo = Walkthrough(settings, executor=executor)
        try:
            demo.setup()
            result = demo.run(args.workflow)
            print(json.dumps({"workflow":args.workflow,"ok":True,"result":result}, ensure_ascii=False))
        finally:
            if args.workflow not in ("recovery", "executor"):
                demo.close()


if __name__ == "__main__":
    main()
