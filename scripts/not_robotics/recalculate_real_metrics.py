import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

OUT = Path(r"C:\work\auto\results\not_robotics_real_feasibility")


def atomic_csv(df, path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def ece(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    value = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if mask.any():
            value += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(value)


def main():
    det = pd.read_parquet(OUT / "detector_outputs.parquet")
    frames = pd.read_parquet(OUT / "frame_observations.parquet")
    rel = pd.read_parquet(OUT / "reliability_scores.parquet")
    reported = json.loads((OUT / "headline_metrics.json").read_text(encoding="utf-8"))
    reported_ablation = pd.read_csv(OUT / "action_ablation.csv").set_index("model")
    y = rel["is_tp"].to_numpy(dtype=int)

    models = [c.removeprefix("score_") for c in rel if c.startswith("score_")]
    recalculated_ablation = []
    for name in models:
        p = rel[f"score_{name}"].to_numpy(float)
        recalculated_ablation.append({"model": name, "auc_roc": roc_auc_score(y, p), "brier_score": brier_score_loss(y, p), "ece_10bin": ece(y, p)})
    ab = pd.DataFrame(recalculated_ablation).set_index("model")
    best_name = ab.loc[["detector_only", "non_action"], "auc_roc"].idxmax()
    keep = rel["gate_keep_full"].to_numpy(bool)
    high = rel["high_motion_heldout"].to_numpy(bool)
    total_gt = int(frames["gt_count_evaluated"].sum())
    base_tp = int(y.sum())
    high_base_fp = int(((y == 0) & high).sum())
    high_gated_fp = int(((y == 0) & high & keep).sum())
    calc = {
        "processed_scenes": int(frames["scene_name"].nunique()),
        "processed_frames": int(len(frames)),
        "evaluated_gt_boxes": total_gt,
        "predictions": int(len(det)),
        "matched_tp": base_tp,
        "baseline_fp": int((y == 0).sum()),
        "best_non_action_model": best_name,
        "best_non_action_auc": float(ab.loc[best_name, "auc_roc"]),
        "full_action_auc": float(ab.loc["full_action", "auc_roc"]),
        "action_information_auc_benefit": float(ab.loc["full_action", "auc_roc"] - ab.loc[best_name, "auc_roc"]),
        "shifted_1s_auc": float(ab.loc["full_shifted_1s", "auc_roc"]),
        "shuffled_auc": float(ab.loc["full_shuffled", "auc_roc"]),
        "baseline_recall": base_tp / total_gt,
        "gated_recall": int((y * keep).sum()) / total_gt,
        "recall_change": (int((y * keep).sum()) - base_tp) / total_gt,
        "high_motion_baseline_fp": high_base_fp,
        "high_motion_gated_fp": high_gated_fp,
        "high_motion_fp_change_fraction": (high_gated_fp - high_base_fp) / high_base_fp,
        "best_non_action_ece": ece(y, rel[f"score_{best_name}"]),
        "full_action_ece": ece(y, rel["score_full_action"]),
    }
    calc["calibration_ece_change"] = calc["full_action_ece"] - calc["best_non_action_ece"]

    rows = []
    for key, value in calc.items():
        expected = reported[key]
        if isinstance(value, str):
            match = value == expected
            delta = ""
        else:
            delta = float(value) - float(expected)
            match = abs(delta) <= 1e-12
        rows.append({"metric": key, "reported": expected, "recalculated": value, "difference": delta, "match_within_1e-12": match})
    for model in ab.index:
        for metric in ["auc_roc", "brier_score", "ece_10bin"]:
            expected = float(reported_ablation.loc[model, metric])
            value = float(ab.loc[model, metric])
            rows.append({"metric": f"{model}.{metric}", "reported": expected, "recalculated": value, "difference": value - expected, "match_within_1e-12": abs(value - expected) <= 1e-12})
    audit = pd.DataFrame(rows)
    atomic_csv(audit, OUT / "metric_recalculation.csv")

    scene_rows = []
    merged = rel[["scene_name", "is_tp", "gate_keep_full", "high_motion_heldout"]]
    for scene, group in merged.groupby("scene_name"):
        frame_scene = frames[frames["scene_name"] == scene]
        gt = int(frame_scene["gt_count_evaluated"].sum())
        sy = group["is_tp"].to_numpy(int)
        sk = group["gate_keep_full"].to_numpy(bool)
        sh = group["high_motion_heldout"].to_numpy(bool)
        hfp = int(((sy == 0) & sh).sum())
        ghfp = int(((sy == 0) & sh & sk).sum())
        scene_rows.append({"scene_name": scene, "frames": len(frame_scene), "evaluated_gt": gt, "baseline_tp": int(sy.sum()), "baseline_fp": int((sy == 0).sum()), "baseline_recall": sy.sum() / gt if gt else np.nan, "gated_recall": (sy * sk).sum() / gt if gt else np.nan, "high_motion_baseline_fp": hfp, "high_motion_gated_fp": ghfp, "high_motion_fp_change_fraction": (ghfp - hfp) / hfp if hfp else np.nan})
    atomic_csv(pd.DataFrame(scene_rows), OUT / "heldout_scene_metrics.csv")
    trace_cols = ["scene_name", "scene_token", "sample_token", "sample_data_token", "timestamp_us", "image_path", "image_sha256", "prediction_index", "class_name", "confidence", "x1", "y1", "x2", "y2", "is_tp", "matched_iou"]
    atomic_csv(det[trace_cols].head(50), OUT / "real_sample_trace.csv")
    if not audit["match_within_1e-12"].all():
        raise SystemExit("Metric recalculation mismatch")
    print(f"Verified {len(audit)} reported metrics; maximum numeric difference {pd.to_numeric(audit['difference'], errors='coerce').abs().max():.3g}")


if __name__ == "__main__":
    main()
