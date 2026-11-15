"""In-process execution backend: fulfills media tasks without subprocesses.

This backend exists to verify the domain flows against the execution
boundary: it honors the same task contract (frozen inputs, expected
artifacts, completion conditions, capacity/cancellation/deadline
semantics from the Executor base) but decodes and encodes through PyAV
inside the service process. Products are real media bytes; loudness
figures are approximations, so the FFmpeg backend remains the
production path for broadcast measurement.
"""
from array import array
from fractions import Fraction
import json
import math
import time
import av
from .errors import LoomError, RenderCancelled, require
from .execution import DecodeTask, Executor, LoudnessTask, RenderTask, SheetTask


class InProcessExecutor(Executor):
    """Verification backend that never starts a child process."""

    name = "In-process executor"

    def _poll(self, cancelled, deadline):
        if cancelled():
            raise RenderCancelled()
        if time.monotonic() >= deadline:
            raise self._timeout()

    def _invoke(self, task, workspace, targets, cancelled, progress, deadline):
        if isinstance(task, RenderTask):
            return self._render(task, targets, cancelled, progress, deadline)
        if isinstance(task, DecodeTask):
            self._decode_samples(task, targets["pcm"], cancelled, deadline)
            return ""
        if isinstance(task, LoudnessTask):
            return self._loudness(task, cancelled, deadline)
        if isinstance(task, SheetTask):
            return self._sheet(task, targets, cancelled, deadline)
        raise LoomError("execution_task", "Unsupported media task", 500)

    def _decode_samples(self, task, output, cancelled, deadline):
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=task.sample_rate)
        chunks = []
        with av.open(str(task.source.path), mode="r", options={"protocol_whitelist": "file,pipe"}) as container:
            stream = next(item for item in container.streams if item.type == "audio")
            for frame in container.decode(stream):
                self._poll(cancelled, deadline)
                for converted in resampler.resample(frame):
                    chunks.append(bytes(converted.planes[0])[:converted.samples*2])
            for converted in resampler.resample(None):
                chunks.append(bytes(converted.planes[0])[:converted.samples*2])
        with output.open("wb") as target:
            target.write(b"".join(chunks))
        return chunks

    def _loudness(self, task, cancelled, deadline):
        chunks = []
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=16000)
        with av.open(str(task.source.path), mode="r", options={"protocol_whitelist": "file,pipe"}) as container:
            stream = next(item for item in container.streams if item.type == "audio")
            for frame in container.decode(stream):
                self._poll(cancelled, deadline)
                for converted in resampler.resample(frame):
                    chunks.append(bytes(converted.planes[0])[:converted.samples*2])
        samples = array("h")
        samples.frombytes(b"".join(chunks))
        if samples:
            rms = math.sqrt(sum((value/32768)**2 for value in samples)/len(samples))
            peak = max(abs(value) for value in samples)/32768
            integrated = 20*math.log10(max(rms, 1e-12)) - 0.691
            true_peak = 20*math.log10(max(peak, 1e-12))
            report = {"input_i": f"{integrated:.2f}", "input_tp": f"{true_peak:.2f}",
                      "input_lra": "0.00", "input_thresh": f"{integrated-10:.2f}"}
        else:
            report = {"input_i": "-inf", "input_tp": "-inf", "input_lra": "0.00", "input_thresh": "-inf"}
        return json.dumps(report)

    def _render(self, task, targets, cancelled, progress, deadline):
        spec = task.snapshot["look"]["spec"]
        kind = spec["kind"]
        rate = spec["sample_rate"]
        channels = spec["channels"]
        layout = "mono" if channels == 1 else "stereo"
        total_samples = max(1, round(task.duration*rate))
        gain = 10**(spec["gain_db"]/20)
        container = av.open(str(targets["rendition"]), mode="w")
        try:
            vstream = None
            total_frames = 0
            if kind == "video":
                codec = "libx264" if "libx264" in av.codecs_available else "mpeg4"
                fps = Fraction(spec["frame_rate"])
                vstream = container.add_stream(codec, rate=fps)
                vstream.width, vstream.height = spec["width"], spec["height"]
                vstream.pix_fmt = "yuv420p"
                if codec == "libx264":
                    vstream.options = {"preset": "veryfast", "crf": str(spec["crf"])}
                total_frames = max(1, round(task.duration*fps))
            astream = container.add_stream("aac" if kind == "video" else "pcm_s16le", rate=rate)
            astream.layout = layout
            resampler = av.audio.resampler.AudioResampler(format="fltp" if kind == "video" else "s16",
                                                          layout=layout, rate=rate)
            for index in range(total_frames):
                self._poll(cancelled, deadline)
                frame = self._color_bars(spec["width"], spec["height"], index, total_frames)
                frame.pts = index
                frame.time_base = Fraction(1, fps)
                for packet in vstream.encode(frame):
                    container.mux(packet)
                progress(min(0.98, 0.5*(index+1)/total_frames))
            if vstream is not None:
                for packet in vstream.encode(None):
                    container.mux(packet)
            sent = 0
            while sent < total_samples:
                self._poll(cancelled, deadline)
                count = min(1024, total_samples-sent)
                frame = av.AudioFrame(format="s16", layout=layout, samples=count)
                frame.sample_rate = rate
                frame.planes[0].update(self._tone(sent, count, rate, channels, gain))
                frame.pts = sent
                frame.time_base = Fraction(1, rate)
                for converted in resampler.resample(frame):
                    for packet in astream.encode(converted):
                        container.mux(packet)
                sent += count
                progress(min(0.98, (0.5 if kind == "video" else 1.0)*sent/total_samples
                               + (0.49 if kind == "video" else 0)))
            for converted in resampler.resample(None):
                for packet in astream.encode(converted):
                    container.mux(packet)
            for packet in astream.encode(None):
                container.mux(packet)
        finally:
            container.close()
        return ""

    def _sheet(self, task, targets, cancelled, deadline):
        frame = None
        with av.open(str(task.source.path), mode="r", options={"protocol_whitelist": "file,pipe"}) as container:
            stream = next(item for item in container.streams if item.type == "video")
            timestamp = task.duration*(task.count-0.5)/task.count
            for offset in (max(0, int(timestamp/float(stream.time_base))), 0):
                container.seek(offset, stream=stream)
                for decoded in container.decode(stream):
                    self._poll(cancelled, deadline)
                    frame = decoded
                    break
                if frame is not None:
                    break
        require(frame is not None, "empty_sheet", "No video frames were decoded")
        width = task.cell_width
        height = max(2, (frame.height*width//frame.width) & ~1)
        image = frame.reformat(width=width, height=height, format="yuvj420p")
        with av.open(str(targets["sheet"]), mode="w") as output:
            stream = output.add_stream("mjpeg", rate=1)
            stream.width, stream.height = width, height
            stream.pix_fmt = "yuvj420p"
            for packet in stream.encode(image):
                output.mux(packet)
            for packet in stream.encode(None):
                output.mux(packet)
        return ""

    @staticmethod
    def _tone(offset, count, rate, channels, gain):
        samples = array("h")
        amplitude = 0.2*gain
        for index in range(count):
            value = int(amplitude*32767*math.sin(2*math.pi*440*(offset+index)/rate))
            for _ in range(channels):
                samples.append(value)
        return samples.tobytes()

    @staticmethod
    def _color_bars(width, height, index, total):
        bars = ((255, 255, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0),
                (255, 0, 255), (255, 0, 0), (0, 0, 255), (0, 0, 0))
        shade = 224 - (index*64//max(1, total))
        row = bytearray()
        for x in range(width):
            r, g, b = bars[x*len(bars)//width]
            row += bytes((r*shade//255, g*shade//255, b*shade//255))
        frame = av.VideoFrame(width, height, "rgb24")
        plane = frame.planes[0]
        if plane.line_size == width*3:
            plane.update(bytes(row)*height)
        else:
            padding = b"\x00"*(plane.line_size-width*3)
            plane.update(b"".join(bytes(row)+padding for _ in range(height)))
        return frame.reformat(format="yuv420p")
