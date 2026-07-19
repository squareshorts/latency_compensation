"""Checkpoint the prespecified architecture-distinct RT-DETR-L detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from ultralytics import RTDETR

ROOT = Path(r"C:\work\auto")
DATA = ROOT / "data" / "av2" / "sensor"
OUTPUT = ROOT / "results" / "sivp_strengthening"
CHECKPOINTS = OUTPUT / "third_detector_checkpoints"
WEIGHT = ROOT / "rtdetr-l.pt"
CAMERA = "ring_front_center"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["development", "model_selection", "heldout", "extension"], default="heldout")
    parser.add_argument("--log-limit", type=int)
    args = parser.parse_args()
    filename = {"development": "av2_development_logs.txt", "model_selection": "av2_model_selection_logs.txt", "heldout": "av2_heldout_logs.txt", "extension": "av2_extension_logs.txt"}[args.role]
    logs = [line.strip() for line in (ROOT / "configs" / filename).read_text().splitlines() if line.strip()]
    if args.log_limit:
        logs = logs[: args.log_limit]
    split = "val" if args.role in {"heldout", "extension"} else "train"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = RTDETR(str(WEIGHT) if WEIGHT.exists() else "rtdetr-l.pt")
    if not WEIGHT.exists():
        candidate = Path("rtdetr-l.pt")
        if candidate.exists() and candidate.resolve() != WEIGHT.resolve():
            candidate.replace(WEIGHT)
    manifest = {
        "detector": "RT-DETR-L", "implementation": "Ultralytics RTDETR official supported checkpoint",
        "checkpoint": str(WEIGHT), "checkpoint_sha256": sha256(WEIGHT), "input_resolution": 640,
        "confidence_threshold": 0.25, "nms": "none; RT-DETR end-to-end prediction",
        "class_mapping": "COCO class 0 -> pedestrian/cyclist analysis group; 2,5,7 -> vehicle; others -> other",
        "device": device, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "ultralytics": __import__("ultralytics").__version__,
        "python": platform.python_version(), "timing": "wall clock around streamed inference; excludes model load and JSON write",
        "created_utc": datetime.now(timezone.utc).isoformat(), "role_started": args.role,
        "future_detection_input": False,
    }
    manifest_path = OUTPUT / "third_detector_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        roles_started = list(dict.fromkeys([*previous.get("roles_started", [previous.get("role_started")]), args.role]))
        previous.update(manifest)
        manifest = previous
        manifest["roles_started"] = [role for role in roles_started if role]
    atomic_json(manifest_path, manifest)
    for log_id in logs:
        destination = CHECKPOINTS / args.role / f"{log_id}.json"
        if destination.exists():
            try:
                existing = json.loads(destination.read_text(encoding="utf-8"))
                if existing.get("completed"):
                    print(json.dumps({"log": log_id, "reused": True, "frames": existing["frames"]}))
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        paths = sorted((DATA / split / log_id / "sensors" / "cameras" / CAMERA).glob("*.jpg"))
        rows = []
        start = time.perf_counter()
        # Small path chunks retain streamed throughput without materializing a
        # full 319-image log in host memory.
        for offset in range(0, len(paths), 32):
            chunk = paths[offset: offset + 32]
            results = model.predict(source=[str(path) for path in chunk], stream=True, device=device, imgsz=640,
                                    conf=0.25, half=torch.cuda.is_available(), verbose=False)
            for path, result in zip(chunk, results):
                boxes = result.boxes
                for xyxy, confidence, class_id in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
                    rows.append({"timestamp_ns": int(path.stem), "image_path": str(path),
                                 "x1": float(xyxy[0]), "y1": float(xyxy[1]), "x2": float(xyxy[2]), "y2": float(xyxy[3]),
                                 "confidence": float(confidence), "class_id": int(class_id)})
        elapsed = time.perf_counter() - start
        payload = {"log_id": log_id, "split": split, "role": args.role, "detector": "rtdetr_l",
                   "frames": len(paths), "detections": rows, "inference_seconds": elapsed,
                   "future_detection_input": False, "completed": True}
        atomic_json(destination, payload)
        print(json.dumps({"log": log_id, "frames": len(paths), "detections": len(rows),
                          "inference_seconds": elapsed, "reused": False}))


if __name__ == "__main__":
    main()
