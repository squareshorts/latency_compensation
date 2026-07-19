"""Apply the frozen B3/B5 definitions to RT-DETR-L checkpoints."""

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
STRENGTHENING = ROOT / "results" / "sivp_strengthening"


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["development", "model_selection", "heldout"], default="heldout")
    parser.add_argument("--log-limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    filename = {"development": "av2_development_logs.txt", "model_selection": "av2_model_selection_logs.txt", "heldout": "av2_heldout_logs.txt"}[args.role]
    logs = [line.strip() for line in (ROOT / "configs" / filename).read_text().splitlines() if line.strip()]
    logs = [log_id for index, log_id in enumerate(logs) if index % args.shard_count == args.shard_index]
    if args.log_limit:
        logs = logs[: args.log_limit]
    metadata = pd.read_csv(AV2 / "cohort_source_metadata.csv")
    split_by_log = {str(row.log_id): str(row.split) for row in metadata.itertuples(index=False)}
    pairs = pd.read_parquet(AV2 / "latency_pairs.parquet")
    pairs["source_image_timestamp_ns"] = pairs.source_image_path.map(lambda value: int(Path(value).stem))
    pairs["target_image_timestamp_ns"] = pairs.target_image_path.map(lambda value: int(Path(value).stem))
    category_mapping = pd.read_csv(AV2 / "class_mapping.csv").set_index("av2_category").analysis_group.to_dict()
    for log_id in logs:
        source = STRENGTHENING / "third_detector_checkpoints" / args.role / f"{log_id}.json"
        if not source.exists():
            print(json.dumps({"log": log_id, "status": "detector_checkpoint_pending"}))
            continue
        destination = STRENGTHENING / "third_detector_propagation" / args.role / f"{log_id}.parquet"
        if destination.exists():
            print(json.dumps({"log": log_id, "status": "reused"}))
            continue
        payload = json.loads(source.read_text(encoding="utf-8"))
        log = LogData(split_by_log[log_id], log_id)
        result = process_detector(log, payload, "rtdetr_l", args.role,
                                  pairs[pairs.log_id.astype(str) == log_id], category_mapping)
        atomic_parquet(destination, result)
        print(json.dumps({"log": log_id, "rows": len(result), "status": "completed"}))


if __name__ == "__main__":
    main()
