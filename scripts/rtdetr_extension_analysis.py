"""Extend frozen object-level diagnostics to RT-DETR-L.

The frozen object_level_diagnostics.parquet covers only yolo11n and yolo11s, so the
class-stratified, velocity-sign, and mechanism analyses were unavailable for RT-DETR-L.
This script rebuilds the matched-object comparison from the low-level end-to-end
prediction and target tables, which do carry class fields for all three detectors.

Validation protocol: the reconstruction is first run on yolo11n and yolo11s and checked
against the published frozen values. Only an exact reproduction licenses applying the
identical code path to RT-DETR-L.
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/sessions/beautiful-eager-ramanujan/mnt/latency_compensation")
LOW = ROOT / "results/sivp_strengthening/end_to_end_low_level/heldout"
OUT = ROOT / "results/rtdetr_extension_20260721"

W, H = 1550.0, 2048.0
DIAG = float(np.hypot(W, H))
HIGH_YAW = 0.0170
PRIMARY_DELTA = 300

CLASS_MAP = {
    "REGULAR_VEHICLE": "vehicle", "TRUCK": "vehicle", "LARGE_VEHICLE": "vehicle",
    "VEHICULAR_TRAILER": "vehicle", "TRUCK_CAB": "vehicle", "BOX_TRUCK": "vehicle",
    "BUS": "vehicle", "SCHOOL_BUS": "vehicle", "ARTICULATED_BUS": "vehicle",
    "PEDESTRIAN": "pedestrian/cyclist", "BICYCLE": "pedestrian/cyclist",
    "BICYCLIST": "pedestrian/cyclist", "MOTORCYCLE": "pedestrian/cyclist",
    "MOTORCYCLIST": "pedestrian/cyclist", "WHEELED_DEVICE": "pedestrian/cyclist",
    "WHEELED_RIDER": "pedestrian/cyclist", "STROLLER": "pedestrian/cyclist",
}


def load_detector(det):
    preds, targs = [], []
    for f in sorted(glob.glob(f"{LOW}/{det}/*.predictions.parquet")):
        preds.append(pd.read_parquet(f))
    for f in sorted(glob.glob(f"{LOW}/{det}/*.targets.parquet")):
        targs.append(pd.read_parquet(f))
    return pd.concat(preds, ignore_index=True), pd.concat(targs, ignore_index=True)


def matched_frame(det):
    """Matched-object table: one row per (frame, tracked object) with B3/B5 geometry."""
    p, t = load_detector(det)
    p = p[p.model.isin(["T5", "T6"])]
    p = p[p.source_object_persistent]
    t = t[t.persistent_object]

    keys = ["log_id", "frame_id", "delta_ms"]
    t = t.rename(columns={"target_track_uuid": "source_track_uuid"})
    tcols = keys + ["source_track_uuid", "target_category", "target_x1", "target_y1",
                    "target_x2", "target_y2", "distance_m"]
    m = p.merge(t[tcols], on=keys + ["source_track_uuid"], how="inner")

    m["pred_cx"] = (m.pred_x1 + m.pred_x2) / 2
    m["pred_cy"] = (m.pred_y1 + m.pred_y2) / 2
    m["tgt_cx"] = (m.target_x1 + m.target_x2) / 2
    m["tgt_cy"] = (m.target_y1 + m.target_y2) / 2
    m["err"] = np.hypot(m.pred_cx - m.tgt_cx, m.pred_cy - m.tgt_cy) / DIAG
    m["iou"] = iou(m)
    m["object_class"] = m.target_category.map(lambda x: CLASS_MAP.get(x, "other"))
    m["high_yaw"] = m.yaw_rate_rad_s.abs() >= HIGH_YAW

    idx = keys + ["source_track_uuid", "object_class", "high_yaw", "distance_m", "confidence"]
    wide = m.pivot_table(index=idx, columns="model", values=["err", "iou"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index().rename(columns={
        "err_T5": "B3_error", "err_T6": "B5_error",
        "iou_T5": "B3_iou", "iou_T6": "B5_iou"})
    return wide.dropna(subset=["B3_error", "B5_error"])


def iou(m):
    x1 = np.maximum(m.pred_x1, m.target_x1)
    y1 = np.maximum(m.pred_y1, m.target_y1)
    x2 = np.minimum(m.pred_x2, m.target_x2)
    y2 = np.minimum(m.pred_y2, m.target_y2)
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (m.pred_x2 - m.pred_x1) * (m.pred_y2 - m.pred_y1)
    b = (m.target_x2 - m.target_x1) * (m.target_y2 - m.target_y1)
    return inter / (a + b - inter)


def cluster_bootstrap(values, replicates=10000, seed=17):
    """Paired log-cluster percentile bootstrap over per-log relative differences."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(v)
    stats = np.array([np.median(v[rng.integers(0, n, n)]) for _ in range(replicates)])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def class_breakdown(df, det, replicates=10000):
    rows = []
    for cls in ["vehicle", "pedestrian/cyclist", "other"]:
        s = df[df.object_class == cls]
        if s.empty:
            continue
        per_log = s.groupby("log_id").agg(
            b3=("B3_error", "median"), b5=("B5_error", "median"),
            b3_iou=("B3_iou", "median"), b5_iou=("B5_iou", "median"),
            objs=("B3_error", "count"))
        r_b3 = s.groupby("log_id")["B3_iou"].apply(lambda x: (x >= 0.5).mean())
        r_b5 = s.groupby("log_id")["B5_iou"].apply(lambda x: (x >= 0.5).mean())
        per_log["difference"] = per_log.b5 - per_log.b3
        per_log["relative"] = per_log.difference / per_log.b3.replace(0, np.nan)
        v = per_log.dropna(subset=["relative"])
        if v.empty:
            continue
        lo, hi = cluster_bootstrap(v.relative.values, replicates) if len(v) > 5 else (np.nan, np.nan)
        rows.append({
            "detector": det, "class": cls, "eligible_logs": len(v),
            "objects": int(v.objs.sum()),
            "B3_median_error": v.b3.median(), "B5_median_error": v.b5.median(),
            "median_paired_relative_diff": v.relative.median(),
            "ci_lower": lo, "ci_upper": hi,
            "logs_favoring_B5": int((v.difference < 0).sum()),
            "B3_median_IoU": v.b3_iou.median(), "B5_median_IoU": v.b5_iou.median(),
            "B3_recall_0.5": r_b3.median(), "B5_recall_0.5": r_b5.median()})
    return rows


PUBLISHED = {
    ("yolo11n", "vehicle"): (40, 10562, 0.006706720043404257, 0.20476150812948207),
    ("yolo11n", "pedestrian/cyclist"): (23, 527, 0.010861227067819588, 0.026005747429163426),
    ("yolo11s", "vehicle"): (40, 12668, 0.005802309046440749, 0.19007403746179297),
    ("yolo11s", "pedestrian/cyclist"): (25, 894, 0.009621862222172864, 0.1662820334373481),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, checks, frames = [], [], {}

    for det in ["yolo11n", "yolo11s", "rtdetr_l"]:
        m = matched_frame(det)
        frames[det] = m
        primary = m[(m.delta_ms == PRIMARY_DELTA) & m.high_yaw]
        rows = class_breakdown(primary, det)
        all_rows += rows
        for r in rows:
            key = (det, r["class"])
            if key in PUBLISHED:
                lg, ob, b3, rel = PUBLISHED[key]
                checks.append({
                    "detector": det, "class": r["class"],
                    "logs_pub": lg, "logs_new": r["eligible_logs"],
                    "objects_pub": ob, "objects_new": r["objects"],
                    "b3_pub": b3, "b3_new": r["B3_median_error"],
                    "b3_absdiff": abs(b3 - r["B3_median_error"]),
                    "rel_pub": rel, "rel_new": r["median_paired_relative_diff"],
                    "rel_absdiff": abs(rel - r["median_paired_relative_diff"])})

    pd.DataFrame(all_rows).to_csv(OUT / "class_breakdown_all_detectors.csv", index=False)
    chk = pd.DataFrame(checks)
    chk.to_csv(OUT / "reconstruction_validation.csv", index=False)

    for det, m in frames.items():
        m.to_parquet(OUT / f"matched_{det}.parquet", index=False)

    print(chk.to_string(index=False))
    print("\nmax |b3 diff| =", chk.b3_absdiff.max(), " max |rel diff| =", chk.rel_absdiff.max())
    print()
    print(pd.DataFrame(all_rows)[["detector", "class", "eligible_logs", "objects",
                                  "B3_median_error", "B5_median_error",
                                  "median_paired_relative_diff", "ci_lower", "ci_upper",
                                  "logs_favoring_B5"]].to_string(index=False))


if __name__ == "__main__":
    main()
