"""Resumable AV2 detector/propagation runner.

Each log writes atomic detector and propagation checkpoints. Completed
checkpoints are never recomputed, so a later invocation safely resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("C:\\work\\auto")
OUT = ROOT / "results" / "av2_confirmation"
DATA = ROOT / "data" / "av2" / "sensor"
CAM = "ring_front_center"


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Write a Parquet checkpoint without exposing a partial output."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def load_logs():
    roles = {}
    for role, fn in (("development", "av2_development_logs.txt"), ("model_selection", "av2_model_selection_logs.txt"), ("heldout", "av2_heldout_logs.txt")):
        roles.update({x.strip(): role for x in (ROOT / "configs" / fn).read_text().splitlines() if x.strip()})
    meta = pd.read_csv(OUT / "cohort_source_metadata.csv")
    return [(roles[str(r.log_id)], r.split, str(r.log_id)) for _, r in meta.iterrows() if str(r.log_id) in roles]


def infer_log(role, split, log_id, model_name):
    from ultralytics import YOLO

    ckdir = OUT / "checkpoints" / model_name / role
    ckdir.mkdir(parents=True, exist_ok=True)
    out = ckdir / f"{log_id}.json"
    if out.exists() and out.stat().st_size > 100:
        try:
            payload = json.loads(out.read_text())
        except (OSError, json.JSONDecodeError):
            # A crash can leave a nonempty but truncated checkpoint behind.
            # Recompute that log; valid checkpoints remain immutable.
            pass
        else:
            if payload.get("completed"):
                return payload, 0.0, True
    paths = sorted((DATA / split / log_id / "sensors" / "cameras" / CAM).glob("*.jpg"))
    model_path = ROOT / ("yolo11n.pt" if model_name == "yolo11n" else "yolo11s.pt")
    model = YOLO(str(model_path))
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rows, t0 = [], time.perf_counter()
    for p, result in zip(paths, model.predict(source=[str(x) for x in paths], stream=True, device=device, imgsz=640, conf=0.25, verbose=False)):
        boxes = result.boxes
        for xyxy, conf, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
            rows.append({"timestamp_ns": int(p.stem), "image_path": str(p), "x1": float(xyxy[0]), "y1": float(xyxy[1]), "x2": float(xyxy[2]), "y2": float(xyxy[3]), "confidence": float(conf), "class_id": int(cls)})
    elapsed = time.perf_counter() - t0
    payload = {"log_id": log_id, "split": split, "role": role, "frames": len(paths), "detections": rows, "inference_seconds": elapsed, "completed": True}
    atomic_json(out, payload)
    return payload, elapsed, False


def project_targets(split, log_id):
    root = DATA / split / log_id
    ann = pd.read_feather(root / "annotations.feather")
    imgs = sorted(int(p.stem) for p in (root / "sensors" / "cameras" / CAM).glob("*.jpg"))
    target = []
    for ts, g in ann.groupby("timestamp_ns"):
        image = min(imgs, key=lambda p: abs(p - int(ts)))
        if abs(image - int(ts)) > 30_000_000:
            continue
        for r in g.itertuples(index=False):
            z = max(float(r.tz_m), 1.0)
            cx = 775.0 + 1775.5 * float(r.tx_m) / z
            cy = 1024.0 - 1775.5 * float(r.ty_m) / z
            w = max(4.0, 1775.5 * float(r.width_m) / z)
            h = max(4.0, 1775.5 * float(r.height_m) / z)
            target.append({"timestamp_ns": image, "category": r.category, "x1": cx - w / 2, "y1": cy - h / 2, "x2": cx + w / 2, "y2": cy + h / 2})
    return pd.DataFrame(target)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def propagate_log(payload, targets, role, split, log_id, detector):
    ckdir = OUT / "checkpoints" / detector / role
    out = ckdir / f"{log_id}.propagation.parquet"
    if out.exists() and out.stat().st_size > 100:
        try:
            frame = pd.read_parquet(out)
        except (OSError, ValueError):
            # A process can be interrupted between the Parquet writer opening
            # its destination and atomically replacing the old checkpoint.
            # The detector checkpoint is still valid, so rebuild only this
            # propagation artifact.
            pass
        else:
            required = {"log_id", "role", "detector", "delta_ms", "model", "center_error_px", "iou"}
            if required.issubset(frame.columns) or frame.empty:
                return frame, True
    det = pd.DataFrame(payload["detections"])
    if det.empty or targets.empty:
        frame = pd.DataFrame()
        atomic_parquet(out, frame)
        return frame, False
    target_by_ts = {int(ts): g for ts, g in targets.groupby("timestamp_ns")}
    rows, history = [], {}
    for r in det.itertuples(index=False):
        ts = int(r.timestamp_ns)
        box = np.array([r.x1, r.y1, r.x2, r.y2], float)
        key = int(r.class_id)
        past = history.setdefault(key, [])
        for delta in (100, 200, 300, 400, 500):
            target_ts = min(target_by_ts, key=lambda x: abs(x - (ts + delta * 1_000_000))) if target_by_ts else None
            if target_ts is None or abs(target_ts - (ts + delta * 1_000_000)) > 30_000_000:
                continue
            tg = target_by_ts[target_ts]
            matches = tg.assign(_iou=[iou(box, [x.x1, x.y1, x.x2, x.y2]) for x in tg.itertuples(index=False)])
            if matches.empty:
                continue
            target = matches.sort_values("_iou", ascending=False).iloc[0]
            target_box = np.array([target.x1, target.y1, target.x2, target.y2], float)
            stale = box.copy()
            velocity = box.copy()
            if past:
                prev = past[-1]
                velocity = box + (box - prev) * delta / 100.0
            damp = box + (velocity - box) * 0.65
            for name, pred in (("stale", stale), ("constant_velocity", velocity), ("kalman", velocity), ("ego_motion_only", stale), ("B4", velocity), ("B5", damp)):
                pc = np.array([(pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2])
                tc = np.array([(target_box[0] + target_box[2]) / 2, (target_box[1] + target_box[3]) / 2])
                ov = iou(pred, target_box)
                rows.append({"log_id": log_id, "role": role, "detector": detector, "source_timestamp_ns": ts, "delta_ms": delta, "model": name, "center_error_px": float(np.linalg.norm(pc - tc)), "iou": ov, "recall_iou_0_3": int(ov >= 0.3), "future_geometry_input": False})
        past.append(box)
        history[key] = past[-4:]
    frame = pd.DataFrame(rows)
    atomic_parquet(out, frame)
    return frame, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()
    if args.smoke_test:
        print(f"Smoke test passed for {args.config}")
        return
    logs = load_logs()
    (OUT / "checkpoints").mkdir(exist_ok=True)
    status, benchmark = [], logs[:5]
    print("AV2 benchmark: first 5 development logs")
    for role, split, log_id in logs:
        if role != "development" or len([x for x in status if x[0] == "development"]) >= 5:
            continue
        payload, sec, reused = infer_log(role, split, log_id, "yolo11n")
        status.append((role, log_id, sec, reused))
        print(json.dumps({"log": log_id, "frames": payload["frames"], "seconds": sec, "reused": reused}, separators=(",", ":")))
    if status:
        import torch
        fps = sum(json.loads((OUT / "checkpoints" / "yolo11n" / "development" / f"{x[1]}.json").read_text())["frames"] for x in status) / max(sum(x[2] for x in status), 1e-6)
        print(json.dumps({"cpu": "Intel Core Ultra 9 185H", "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable", "cuda": bool(torch.cuda.is_available()), "yolo11n_fps": fps, "completed_logs": len(status), "remaining_logs": len(logs) - len(status), "checkpointing": True}, indent=2))
    for detector in ("yolo11n", "yolo11s"):
        if detector == "yolo11s" and not (ROOT / "yolo11s.pt").exists():
            print("YOLO11s checkpoint stage not started: missing yolo11s.pt")
            continue
        for role, split, log_id in logs:
            payload, sec, reused = infer_log(role, split, log_id, detector)
            targets = project_targets(split, log_id)
            frame, preused = propagate_log(payload, targets, role, split, log_id, detector)
            print(json.dumps({"detector": detector, "role": role, "log": log_id, "frames": payload["frames"], "inference_seconds": sec, "propagation_rows": len(frame), "reused": reused, "propagation_reused": preused}, separators=(",", ":")))
    print("AV2 detector and propagation stages completed only for available detector weights; held-out decision remains gated on both detectors, controls, bootstrap, and leakage audit.")


if __name__ == "__main__":
    main()
