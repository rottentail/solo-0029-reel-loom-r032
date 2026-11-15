"""Execution boundary between media domain flows and media processing.

Domain flows (rendering, audio analysis, contact sheets) declare tasks:
the frozen media they consume, the artifacts they expect, and the
completion conditions. The executor owns concrete commands, scratch
files, shared capacity, cancellation, deadlines, and — for the FFmpeg
backend — process lifecycle. Domain flows never build command lines, so
any backend honoring the task contract (including the in-process one in
`inproc.py`) can drive the same flows.
"""
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
from .errors import LoomError, RenderCancelled, require
from .probe import select_stream
from .timebase import decimal, seconds


@dataclass(frozen=True)
class Source:
    """Frozen media an execution consumes."""
    blob: str
    path: Path
    metadata: dict


@dataclass(frozen=True)
class ArtifactSpec:
    """A product the execution must leave in the task workspace."""
    role: str
    suffix: str
    missing_code: str = ""
    missing_detail: str = ""


@dataclass(frozen=True)
class RenderTask:
    """Encode a frozen reel snapshot into one rendition artifact."""
    snapshot: dict
    sources: tuple          # one Source per slice, aligned with the document
    product: ArtifactSpec
    duration: float         # expected rendition seconds
    tolerance: float        # allowed duration drift in seconds

    @property
    def consumes(self):
        return self.sources

    @property
    def produces(self):
        return (self.product,)

    @property
    def progress_duration(self):
        return self.duration


@dataclass(frozen=True)
class DecodeTask:
    """Downmix a frozen source to mono PCM for level/waveform analysis."""
    source: Source
    product: ArtifactSpec
    sample_rate: int

    @property
    def consumes(self):
        return (self.source,)

    @property
    def produces(self):
        return (self.product,)

    @property
    def progress_duration(self):
        return None


@dataclass(frozen=True)
class LoudnessTask:
    """Measure a frozen source; the R128 report arrives in the task log."""
    source: Source

    @property
    def consumes(self):
        return (self.source,)

    @property
    def produces(self):
        return ()

    @property
    def progress_duration(self):
        return None


@dataclass(frozen=True)
class SheetTask:
    """Tile sampled frames of a frozen source into one JPEG sheet."""
    source: Source
    product: ArtifactSpec
    count: int
    columns: int
    rows: int
    cell_width: int
    duration: float         # source duration in seconds

    @property
    def consumes(self):
        return (self.source,)

    @property
    def produces(self):
        return (self.product,)

    @property
    def progress_duration(self):
        return None


class Execution:
    """Outcome of a finished task; the workspace is reclaimed on close."""

    def __init__(self, workspace, artifacts, log):
        self._workspace = workspace
        self._artifacts = artifacts
        self.log = log

    def artifact(self, role):
        return self._artifacts[role]

    def close(self):
        if self._workspace is not None:
            self._workspace.cleanup()
            self._workspace = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Executor:
    """Capacity, cancellation, deadline, and scratch ownership for tasks.

    Analysis and rendering share one executor per service, so every task
    draws from the same bounded capacity. Cancellation while waiting and
    deadlines while running both release the slot and reclaim scratch;
    a failed task leaves no registered artifact behind.
    """

    name = "executor"

    def __init__(self, settings):
        self.settings = settings
        self.scratch = settings.root / "scratch"
        self.slots = threading.BoundedSemaphore(settings.workers + 1)
        self._busy = 0
        self._busy_lock = threading.Lock()

    @property
    def busy(self):
        with self._busy_lock:
            return self._busy

    def _timeout(self):
        return LoomError("encoder_timeout", f"{self.name} exceeded the processing deadline", 504)

    def _acquire(self, cancelled, deadline):
        while True:
            if cancelled():
                raise RenderCancelled()
            require(time.monotonic() < deadline, "encoder_capacity", "Encoder capacity wait expired", 503)
            if self.slots.acquire(timeout=0.1):
                with self._busy_lock:
                    self._busy += 1
                return

    def _release(self):
        with self._busy_lock:
            self._busy -= 1
        self.slots.release()

    def perform(self, task, cancelled=None, progress=None):
        """Run one task to completion and return its Execution handle."""
        cancelled = cancelled or (lambda: False)
        progress = progress or (lambda value: None)
        deadline = time.monotonic() + self.settings.render_timeout
        self._acquire(cancelled, deadline)
        try:
            workspace = tempfile.TemporaryDirectory(prefix="loom-", dir=self.scratch)
            try:
                folder = Path(workspace.name)
                targets = {spec.role: folder / (spec.role + spec.suffix) for spec in task.produces}
                log = self._invoke(task, folder, targets, cancelled, progress, deadline) or ""
                for spec in task.produces:
                    if spec.missing_code:
                        require(targets[spec.role].is_file(), spec.missing_code, spec.missing_detail)
                if cancelled():
                    raise RenderCancelled()
                progress(1)
                return Execution(workspace, targets, log)
            except BaseException:
                workspace.cleanup()
                raise
        finally:
            self._release()

    def _invoke(self, task, workspace, targets, cancelled, progress, deadline):
        raise NotImplementedError


class FfmpegExecutor(Executor):
    """Production backend: runs FFmpeg in bounded, killable process groups."""

    name = "FFmpeg"

    def _invoke(self, task, workspace, targets, cancelled, progress, deadline):
        command = [self.settings.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                   "-threads", "1", "-filter_threads", "1", "-filter_complex_threads", "1",
                   "-progress", "pipe:1", "-nostats"] + self._argv(task, workspace, targets)
        process = None
        with tempfile.TemporaryFile() as errors, tempfile.TemporaryFile() as status:
            try:
                process = subprocess.Popen(command, stdout=status, stderr=errors,
                                           start_new_session=os.name == "posix")
                last = 0
                duration = task.progress_duration
                while process.poll() is None:
                    if cancelled():
                        self._stop(process)
                        raise RenderCancelled()
                    if time.monotonic() >= deadline:
                        self._stop(process)
                        raise self._timeout()
                    end = status.tell()
                    if end != last and duration:
                        status.seek(0)
                        lines = status.read().decode(errors="replace").splitlines()
                        values = [line.split("=", 1)[1] for line in lines if line.startswith("out_time_us=")]
                        if values and values[-1].isdigit():
                            progress(min(0.98, int(values[-1])/1_000_000/duration))
                        status.seek(0, 2)
                        last = end
                    time.sleep(0.05)
                errors.seek(0, 2)
                size = errors.tell()
                errors.seek(max(0, size-6000))
                detail = errors.read().decode(errors="replace")
                require(process.returncode == 0, "encoder_failed", "FFmpeg could not produce this rendition",
                        422, stderr=detail)
                return detail
            finally:
                if process is not None:
                    self._stop(process)

    def _stop(self, process):
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=3)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)

    def _argv(self, task, workspace, targets):
        if isinstance(task, RenderTask):
            return self._render_argv(task, targets)
        if isinstance(task, DecodeTask):
            return ["-protocol_whitelist", "file,pipe", "-i", str(task.source.path),
                    "-vn", "-map", "0:a:0", "-ac", "1", "-ar", str(task.sample_rate),
                    "-c:a", "pcm_s16le", "-f", "s16le", str(targets["pcm"])]
        if isinstance(task, LoudnessTask):
            return ["-loglevel", "info", "-protocol_whitelist", "file,pipe",
                    "-i", str(task.source.path), "-map", "0:a:0", "-vn",
                    "-af", "loudnorm=I=-23:TP=-1:LRA=7:print_format=json",
                    "-f", "null", str(workspace / "sink.null")]
        if isinstance(task, SheetTask):
            interval = task.duration / task.count
            filter_spec = (f"fps=fps={task.count}/{task.duration}:start_time={interval/2},"
                           f"scale={task.cell_width}:-2,"
                           f"tile={task.columns}x{task.rows}:nb_frames={task.count}:padding=4:margin=4")
            return ["-protocol_whitelist", "file,pipe", "-i", str(task.source.path),
                    "-an", "-vf", filter_spec, "-frames:v", "1", "-q:v", "3", str(targets["sheet"])]
        raise LoomError("execution_task", "Unsupported media task", 500)

    def _render_argv(self, task, targets):
        snapshot = task.snapshot
        spec = snapshot["look"]["spec"]
        args, filters, segments = [], [], []
        for i, (item, source) in enumerate(zip(snapshot["document"]["slices"], task.sources)):
            args += ["-protocol_whitelist", "file,pipe", "-i", str(source.path)]
            a, b = decimal(item["in_tick"]), decimal(item["out_tick"])
            duration = seconds(item["out_tick"]-item["in_tick"])
            audio = select_stream(source.metadata, "audio")
            if spec["kind"] == "video":
                video = select_stream(source.metadata, "video")
                if spec["fit"] == "contain":
                    sizing = (f'scale={spec["width"]}:{spec["height"]}:force_original_aspect_ratio=decrease,'
                              f'pad={spec["width"]}:{spec["height"]}:(ow-iw)/2:(oh-ih)/2')
                else:
                    sizing = (f'scale={spec["width"]}:{spec["height"]}:force_original_aspect_ratio=increase,'
                              f'crop={spec["width"]}:{spec["height"]}')
                filters.append(f'[{i}:{video["index"]}]trim=start={a}:end={b},setpts=PTS-STARTPTS,'
                               f'{sizing},setsar=1,fps={spec["frame_rate"]},format=yuv420p[v{i}]')
            layout = "mono" if spec["channels"] == 1 else "stereo"
            if audio:
                filters.append(f'[{i}:{audio["index"]}]atrim=start={a}:end={b},asetpts=PTS-STARTPTS,'
                               f'aresample={spec["sample_rate"]},aformat=sample_fmts=fltp:channel_layouts={layout},'
                               f'apad,atrim=duration={duration:.6f},volume={spec["gain_db"]}dB[a{i}]')
            else:
                filters.append(f'anullsrc=r={spec["sample_rate"]}:cl={layout},atrim=duration={duration:.6f}[a{i}]')
            segments.append((f"[v{i}]" if spec["kind"] == "video" else "") + f"[a{i}]")
        count = len(segments)
        if spec["kind"] == "video":
            filters.append("".join(segments) + f"concat=n={count}:v=1:a=1[vjoined][aout]")
            filters.append(f'[vjoined]fps={spec["frame_rate"]},tpad=stop_mode=clone:stop_duration=0.1,'
                           f'trim=duration={seconds(snapshot["document"]["duration_tick"]):.6f}[vout]')
            maps = ["-map", "[vout]", "-map", "[aout]", "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", str(spec["crf"]), "-c:a", "aac", "-movflags", "+faststart"]
        else:
            filters.append("".join(segments) + f"concat=n={count}:v=0:a=1[aout]")
            maps = ["-map", "[aout]", "-c:a", "pcm_s16le"]
        return args + ["-filter_complex", ";".join(filters)] + maps + [str(targets["rendition"])]
