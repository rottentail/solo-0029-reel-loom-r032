from fractions import Fraction
from pathlib import Path
from .errors import require
from .execution import ArtifactSpec, RenderTask, Source
from .values import digest
from .timebase import seconds


class Planner:
    def __init__(self, assets, reels, looks):
        self.assets, self.reels, self.looks = assets, reels, looks

    def freeze(self, db, reel_id, revision, look_id, version):
        reel = self.reels.get(reel_id, revision, db)
        look = self.looks.get(look_id, version, db)
        closure = {}
        for item in reel["document"]["slices"]:
            if item["asset"] not in closure:
                asset = self.assets.get(item["asset"], db, ready=True)
                closure[asset["id"]] = {"id": asset["id"], "blob": asset["blob"], "metadata": asset["metadata"]}
        if look["spec"]["kind"] == "video":
            require(all(x["metadata"]["has_video"] for x in closure.values()),
                    "video_required", "Video looks require video in every slice")
        else:
            require(all(x["metadata"]["has_audio"] for x in closure.values()),
                    "audio_required", "Audio looks require audio in every slice")
        snapshot = {"schema": 1, "reel": reel_id, "revision": reel["revision"],
                    "reel_hash": reel["hash"], "document": reel["document"],
                    "look": {"id": look["id"], "version": look["version"], "spec": look["spec"], "hash": look["hash"]},
                    "sources": closure}
        return {**snapshot, "fingerprint": digest(snapshot)}

    def compile(self, snapshot, resolve):
        """Compile a frozen snapshot into the render task an executor runs."""
        spec = snapshot["look"]["spec"]
        sources = tuple(
            Source(blob=asset["blob"], path=Path(resolve(asset["blob"])), metadata=asset["metadata"])
            for item in snapshot["document"]["slices"]
            for asset in (snapshot["sources"][item["asset"]],)
        )
        suffix = ".mp4" if spec["kind"] == "video" else ".wav"
        duration = seconds(snapshot["document"]["duration_tick"])
        tolerance = max(0.15, 2/float(Fraction(spec["frame_rate"])))
        return RenderTask(snapshot=snapshot, sources=sources,
                          product=ArtifactSpec("rendition", suffix),
                          duration=duration, tolerance=tolerance)
