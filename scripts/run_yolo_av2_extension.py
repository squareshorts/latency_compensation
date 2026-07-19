"""Run the two frozen YOLO checkpoints on the untouched extension logs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(r"C:\work\auto")
DATA = ROOT / "data" / "av2" / "sensor" / "val"
OUTPUT = ROOT / "results" / "sivp_strengthening" / "extension_detector_checkpoints"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["yolo11n", "yolo11s"], required=True)
    args = parser.parse_args()
    model = YOLO(str(ROOT / f"{args.detector}.pt"))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logs = [line.strip() for line in (ROOT / "configs" / "av2_extension_logs.txt").read_text().splitlines() if line.strip()]
    for log_id in logs:
        destination = OUTPUT / args.detector / f"{log_id}.json"
        if destination.exists():
            try:
                if json.loads(destination.read_text()).get("completed"):
                    print(json.dumps({"log": log_id, "reused": True}))
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        paths = sorted((DATA / log_id / "sensors" / "cameras" / "ring_front_center").glob("*.jpg"))
        rows, start = [], time.perf_counter()
        for offset in range(0, len(paths), 64):
            chunk = paths[offset: offset + 64]
            results = model.predict(source=[str(path) for path in chunk], stream=True, device=device,
                                    imgsz=640, conf=0.25, verbose=False)
            for path, result in zip(chunk, results):
                for xyxy, confidence, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                    rows.append({"timestamp_ns": int(path.stem), "image_path": str(path),
                                 "x1": float(xyxy[0]), "y1": float(xyxy[1]), "x2": float(xyxy[2]), "y2": float(xyxy[3]),
                                 "confidence": float(confidence), "class_id": int(class_id)})
        elapsed = time.perf_counter() - start
        atomic_json(destination, {"log_id": log_id, "split": "val", "role": "extension", "detector": args.detector,
                                  "frames": len(paths), "detections": rows, "inference_seconds": elapsed,
                                  "future_detection_input": False, "completed": True})
        print(json.dumps({"log": log_id, "frames": len(paths), "detections": len(rows), "seconds": elapsed}))


if __name__ == "__main__":
    main()
