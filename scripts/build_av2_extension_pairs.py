"""Build timestamp-valid latency pairs for the frozen AV2 extension."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\work\auto")
DATA = ROOT / "data" / "av2" / "sensor" / "val"
OUTPUT = ROOT / "results" / "sivp_strengthening" / "extension_latency_pairs.parquet"
DELTAS = (100, 200, 300, 400, 500)


def nearest(timestamp: int, values: np.ndarray):
    index = int(np.searchsorted(values, timestamp))
    choices = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(values)]
    if not choices:
        return None, None
    selected = min(choices, key=lambda candidate: abs(int(values[candidate]) - int(timestamp)))
    return int(values[selected]), abs(int(values[selected]) - int(timestamp))


def main() -> None:
    logs = [line.strip() for line in (ROOT / "configs" / "av2_extension_logs.txt").read_text().splitlines() if line.strip()]
    rows = []
    for log_id in logs:
        root = DATA / log_id
        annotation = pd.read_feather(root / "annotations.feather", columns=["timestamp_ns"])
        annotation_timestamps = np.sort(annotation.timestamp_ns.unique().astype(np.int64))
        image_timestamps = np.asarray(sorted(int(path.stem) for path in (root / "sensors" / "cameras" / "ring_front_center").glob("*.jpg")), dtype=np.int64)
        for delta_ms in DELTAS:
            for source_annotation_timestamp in annotation_timestamps:
                target_annotation_timestamp, target_error = nearest(int(source_annotation_timestamp + delta_ms * 1_000_000), annotation_timestamps)
                if target_annotation_timestamp is None or target_error > 10_000_000:
                    continue
                source_image_timestamp, source_error = nearest(int(source_annotation_timestamp), image_timestamps)
                target_image_timestamp, image_target_error = nearest(target_annotation_timestamp, image_timestamps)
                if source_image_timestamp is None or target_image_timestamp is None or source_error > 30_000_000 or image_target_error > 30_000_000:
                    continue
                rows.append({"split": "extension", "source_split": "val", "log_id": log_id, "delta_ms": delta_ms,
                             "source_timestamp_ns": int(source_annotation_timestamp),
                             "target_timestamp_ns": int(target_annotation_timestamp),
                             "source_image_path": str(root / "sensors" / "cameras" / "ring_front_center" / f"{source_image_timestamp}.jpg"),
                             "target_image_path": str(root / "sensors" / "cameras" / "ring_front_center" / f"{target_image_timestamp}.jpg"),
                             "source_image_error_ns": int(source_error), "target_image_error_ns": int(image_target_error),
                             "target_annotation_geometry_eval_only": True, "future_annotation_geometry_input": False})
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUTPUT, index=False)
    print(json.dumps({"logs": frame.log_id.nunique(), "pairs": len(frame), "by_delta": frame.groupby("delta_ms").size().to_dict()}, default=int))


if __name__ == "__main__":
    main()
