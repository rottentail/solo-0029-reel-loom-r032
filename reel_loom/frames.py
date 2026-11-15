from .errors import require
from .execution import ArtifactSpec, SheetTask, Source
from .values import object_body, integer
from .timebase import seconds


class Frames:
    def __init__(self, assets, blobs, executor, artifacts):
        self.assets, self.blobs = assets, blobs
        self.executor, self.artifacts = executor, artifacts

    def contact_sheet(self, asset_id, raw):
        object_body(raw, ("count", "columns", "cell_width"))
        count = integer(raw.get("count", 9), "count", 1, 36)
        columns = integer(raw.get("columns", 3), "columns", 1, 6)
        cell_width = integer(raw.get("cell_width", 240), "cell_width", 64, 640)
        require(count >= columns, "sheet_layout", "Column count exceeds frame count")
        asset = self.assets.get(asset_id, ready=True)
        require(asset["metadata"]["has_video"], "video_required", "Contact sheets require video")
        duration = seconds(asset["metadata"]["duration_tick"])
        rows = (count+columns-1)//columns
        interval = duration/count
        timestamps = [round((i+.5)*interval, 6) for i in range(count)]
        source = Source(asset["blob"], self.blobs.resolve(asset["blob"]), asset["metadata"])
        task = SheetTask(source=source, count=count, columns=columns, rows=rows,
                         cell_width=cell_width, duration=duration,
                         product=ArtifactSpec("sheet", ".jpg", "empty_sheet", "No video frames were decoded"))
        with self.executor.perform(task) as done:
            return self.artifacts.add(asset_id, "contact_sheet", done.artifact("sheet"),
                                      {"count": count, "columns": columns, "rows": rows,
                                       "cell_width": cell_width, "sample_seconds": timestamps,
                                       "source_blob": asset["blob"], "mime": "image/jpeg"})
