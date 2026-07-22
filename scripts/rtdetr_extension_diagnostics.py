"""RT-DETR-L diagnostics that the frozen artifacts support.

Three analyses:
  1. Image-space direction agreement between the injected object-motion correction
     and the residual that correction must remove, for all three detectors.
  2. Prospective B3-versus-B5 replication on the untouched extension cohort.
  3. Detection density and confidence distributions behind the false-positive rates.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/sessions/beautiful-eager-ramanujan/mnt/latency_compensation")
LOW = ROOT / "results/sivp_strengthening/end_to_end_low_level/heldout"
EXT = ROOT / "results/sivp_strengthening/extension_propagation"
OUT = ROOT / "results/rtdetr_extension_20260721"
W, H = 1550.0, 2048.0
DIAG = float(np.hypot(W, H))
HIGH_YAW = 0.0170
DETS = ["yolo11n", "yolo11s", "rtdetr_l"]
MIN_PX = 1.0

GATE_FEATURES = ["detector_confidence", "track_age", "number_historical_detections",
                 "association_confidence", "estimated_3d_velocity_m_s",
                 "velocity_dispersion_m_s", "acceleration_estimate_m_s2",
                 "lidar_point_count", "source_depth_m", "depth_uncertainty_m",
                 "source_box_area_px2", "ego_yaw_rate_rad_s", "delta_ms"]


def cluster_bootstrap(v, replicates=10000, seed=17):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    n = len(v)
    s = np.array([np.median(v[rng.integers(0, n, n)]) for _ in range(replicates)])
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


# ---------- 1. image-space direction agreement ----------
def direction_agreement():
    """B5 = B3 + injected object-motion correction.

    The correction that would zero the error is (target_center - B3_center), so the
    signed projection of the injected correction onto that residual states whether the
    estimate pushed the box toward or away from the truth.
    """
    rows = []
    for det in DETS:
        p = pd.concat([pd.read_parquet(f) for f in
                       sorted(glob.glob(f"{LOW}/{det}/*.predictions.parquet"))],
                      ignore_index=True)
        t = pd.concat([pd.read_parquet(f) for f in
                       sorted(glob.glob(f"{LOW}/{det}/*.targets.parquet"))],
                      ignore_index=True)
        p = p[p.model.isin(["T5", "T6"]) & p.source_object_persistent]
        t = t[t.persistent_object].rename(columns={"target_track_uuid": "source_track_uuid"})
        keys = ["log_id", "frame_id", "delta_ms", "source_track_uuid"]
        m = p.merge(t[keys + ["target_x1", "target_y1", "target_x2", "target_y2"]],
                    on=keys, how="inner")
        m["cx"] = (m.pred_x1 + m.pred_x2) / 2
        m["cy"] = (m.pred_y1 + m.pred_y2) / 2
        m["tx"] = (m.target_x1 + m.target_x2) / 2
        m["ty"] = (m.target_y1 + m.target_y2) / 2
        m["high_yaw"] = m.yaw_rate_rad_s.abs() >= HIGH_YAW
        m = m[(m.delta_ms == 300) & m.high_yaw]

        idx = keys + ["tx", "ty"]
        w = m.pivot_table(index=idx, columns="model", values=["cx", "cy"], aggfunc="first")
        w.columns = [f"{a}_{b}" for a, b in w.columns]
        w = w.reset_index().dropna()

        corr_x = w.cx_T6 - w.cx_T5           # injected correction
        corr_y = w.cy_T6 - w.cy_T5
        res_x = w.tx - w.cx_T5               # residual B3 must remove
        res_y = w.ty - w.cy_T5
        corr_n = np.hypot(corr_x, corr_y)
        res_n = np.hypot(res_x, res_y)

        eligible = (corr_n >= MIN_PX) & (res_n >= MIN_PX)
        dot = corr_x * res_x + corr_y * res_y
        same = (dot > 0) & eligible
        opp = (dot < 0) & eligible
        n_el = int(eligible.sum())
        rows.append({
            "detector": det, "objects": len(w), "eligible": n_el,
            "ambiguous_frac": float((~eligible).mean()),
            "same_halfplane": float(same.sum() / n_el),
            "opposite_halfplane": float(opp.sum() / n_el),
            "median_correction_px": float(corr_n[eligible].median()),
            "median_residual_px": float(res_n[eligible].median())})
    return pd.DataFrame(rows)


# ---------- 2. prospective extension replication ----------
def gate_readiness():
    """The gate consumes the 13 frozen features, which exist only in the object-level
    diagnostics table. Readiness is therefore a property of that table, not of the
    extension propagation outputs."""
    diag = ROOT / "results/object_motion_harm/object_level_diagnostics.parquet"
    import pyarrow.parquet as pq
    cols = set(pq.ParquetFile(diag).schema_arrow.names)
    present = [f for f in GATE_FEATURES if f in cols]
    d = pd.read_parquet(diag, columns=["detector"])
    counts = d.detector.value_counts().to_dict()
    return pd.DataFrame([{
        "detector": det,
        "diagnostic_rows": int(counts.get(det, 0)),
        "gate_features_available": len(present),
        "gate_scorable": counts.get(det, 0) > 0} for det in DETS])


def extension_replication():
    rows = []
    for det in DETS:
        files = sorted(glob.glob(f"{EXT}/{det}/*.parquet"))
        if not files:
            continue
        d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        d = d[d.model.isin(["B3", "B5", "T5", "T6"])]
        d["m"] = d.model.replace({"T5": "B3", "T6": "B5"})
        d = d[(d.delta_ms == 300) & (d.yaw_rate_rad_s.abs() >= HIGH_YAW)]
        if d.empty:
            continue
        per_log = d.pivot_table(index="log_id", columns="m",
                                values="normalized_center_error", aggfunc="median")
        iou = d.pivot_table(index="log_id", columns="m", values="iou", aggfunc="median")
        if not {"B3", "B5"}.issubset(per_log.columns):
            continue
        per_log = per_log.dropna()
        rel = ((per_log.B5 - per_log.B3) / per_log.B3).dropna()
        if len(rel) < 3:
            continue
        lo, hi = cluster_bootstrap(rel.values)
        rows.append({
            "detector": det, "extension_logs": len(rel),
            "objects": int(len(d) / 2),
            "B3_error": per_log.B3.median(), "B5_error": per_log.B5.median(),
            "median_relative_diff": rel.median(), "ci_lower": lo, "ci_upper": hi,
            "logs_favoring_B5": int((rel < 0).sum()),
            "B3_iou": iou.B3.median() if "B3" in iou else np.nan,
            "B5_iou": iou.B5.median() if "B5" in iou else np.nan})
    return pd.DataFrame(rows)


# ---------- 3. detection density behind false-positive rates ----------
def detection_density():
    rows = []
    for det in DETS:
        p = pd.concat([pd.read_parquet(f) for f in
                       sorted(glob.glob(f"{LOW}/{det}/*.predictions.parquet"))],
                      ignore_index=True)
        p = p[(p.model == "T5") & (p.delta_ms == 300)]
        per_frame = p.groupby(["log_id", "frame_id"]).size()
        rows.append({
            "detector": det, "frames": len(per_frame),
            "detections_per_frame_mean": float(per_frame.mean()),
            "detections_per_frame_median": float(per_frame.median()),
            "conf_median": float(p.confidence.median()),
            "conf_q25": float(p.confidence.quantile(0.25)),
            "frac_conf_below_0.4": float((p.confidence < 0.4).mean())})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    da = direction_agreement()
    da.to_csv(OUT / "direction_agreement_image_space.csv", index=False)
    print("== image-space direction agreement ==")
    print(da.to_string(index=False))

    ext = extension_replication()
    cover = gate_readiness()
    ext.to_csv(OUT / "extension_replication.csv", index=False)
    cover.to_csv(OUT / "extension_gate_readiness.csv", index=False)
    print("\n== extension cohort replication ==")
    print(ext.to_string(index=False))
    print("\n== gate readiness ==")
    print(cover.to_string(index=False))

    dd = detection_density()
    dd.to_csv(OUT / "detection_density.csv", index=False)
    print("\n== detection density ==")
    print(dd.to_string(index=False))


if __name__ == "__main__":
    main()
