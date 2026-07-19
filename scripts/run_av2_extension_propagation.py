"""Apply frozen propagation definitions to extension detector checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\work\auto")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from recalculate_av2_sample import LogData
from run_av2_corrected_propagation import process_detector

AV2 = ROOT / "results" / "av2_confirmation"
OUTPUT = ROOT / "results" / "sivp_strengthening"


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["yolo11n", "yolo11s", "rtdetr_l"], required=True)
    args = parser.parse_args()
    pairs = pd.read_parquet(OUTPUT / "extension_latency_pairs.parquet")
    pairs["source_image_timestamp_ns"] = pairs.source_image_path.map(lambda value: int(Path(value).stem))
    pairs["target_image_timestamp_ns"] = pairs.target_image_path.map(lambda value: int(Path(value).stem))
    category_mapping = pd.read_csv(AV2 / "class_mapping.csv").set_index("av2_category").analysis_group.to_dict()
    logs = [line.strip() for line in (ROOT / "configs" / "av2_extension_logs.txt").read_text().splitlines() if line.strip()]
    for log_id in logs:
        source = (OUTPUT / "extension_detector_checkpoints" / args.detector / f"{log_id}.json"
                  if args.detector != "rtdetr_l" else OUTPUT / "third_detector_checkpoints" / "extension" / f"{log_id}.json")
        if not source.exists():
            print(json.dumps({"log": log_id, "status": "detector_pending"}))
            continue
        destination = OUTPUT / "extension_propagation" / args.detector / f"{log_id}.parquet"
        if destination.exists():
            print(json.dumps({"log": log_id, "status": "reused"}))
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        result = process_detector(LogData("val", log_id), payload, args.detector, "extension",
                                  pairs[pairs.log_id.astype(str) == log_id], category_mapping)
        atomic_parquet(destination, result)
        print(json.dumps({"log": log_id, "rows": len(result), "status": "completed"}))


if __name__ == "__main__":
    main()
