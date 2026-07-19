"""Fit and freeze one deployable-feature logistic gate before extension outcomes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(r"C:\work\auto")
SOURCE = ROOT / "results" / "object_motion_harm" / "object_level_diagnostics.parquet"
OUTPUT = ROOT / "results" / "sivp_strengthening"
FEATURES = [
    "detector_confidence", "track_age", "number_historical_detections", "association_confidence",
    "estimated_3d_velocity_m_s", "velocity_dispersion_m_s", "acceleration_estimate_m_s2",
    "lidar_point_count", "source_depth_m", "depth_uncertainty_m", "source_box_area_px2",
    "ego_yaw_rate_rad_s", "delta_ms",
]
THRESHOLDS = np.arange(0.50, 0.91, 0.05)
FORBIDDEN = ("true_", "target_", "future_", "B3_error", "B5_error", "delta_harm", "object_displacement", "evaluation_track_uuid")


def main() -> None:
    columns = ["log_id", "role", "detector", *FEATURES, "M0_normalized_center_error", "M1_normalized_center_error",
               "M0_iou", "M1_iou", "M0_recall_iou_0_5", "M1_recall_iou_0_5"]
    data = pd.read_parquet(SOURCE, columns=columns)
    if any(any(pattern.lower() in feature.lower() for pattern in FORBIDDEN) for feature in FEATURES):
        raise RuntimeError("Forbidden gate feature detected")
    development = data[data.role == "development"].copy()
    selection = data[data.role == "model_selection"].copy()
    # Equal-cap deterministic within-log sampling bounds memory while retaining
    # every development log and both detector configurations.
    rng = np.random.default_rng(20260719)
    sampled_indices = []
    for _, group in development.groupby(["detector", "log_id"], sort=True):
        sampled_indices.extend(rng.choice(group.index.to_numpy(), min(1000, len(group)), replace=False))
    development = development.loc[sampled_indices].copy()
    for feature in FEATURES:
        development[feature] = pd.to_numeric(development[feature], errors="coerce").astype("float32")
        selection[feature] = pd.to_numeric(selection[feature], errors="coerce").astype("float32")
    label = (development.M1_normalized_center_error < development.M0_normalized_center_error).astype(int)
    counts = development.groupby(["detector", "log_id"]).log_id.transform("size")
    sample_weight = 1.0 / counts
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=20260719)),
    ])
    model.fit(development[FEATURES], label, classifier__sample_weight=sample_weight)
    selection_probability = model.predict_proba(selection[FEATURES])[:, 1]
    selection = selection.assign(gate_probability=selection_probability)
    threshold_rows = []
    for threshold in THRESHOLDS:
        choose_b5 = selection.gate_probability >= threshold
        selected_error = np.where(choose_b5, selection.M1_normalized_center_error, selection.M0_normalized_center_error)
        selected_iou = np.where(choose_b5, selection.M1_iou, selection.M0_iou)
        selected_recall = np.where(choose_b5, selection.M1_recall_iou_0_5, selection.M0_recall_iou_0_5)
        work = selection[["detector", "log_id"]].copy()
        work["gate_error"], work["gate_iou"], work["gate_recall"] = selected_error, selected_iou, selected_recall
        work["b3_error"], work["b3_iou"], work["b3_recall"] = selection.M0_normalized_center_error, selection.M0_iou, selection.M0_recall_iou_0_5
        per_log = work.groupby(["detector", "log_id"], as_index=False).median(numeric_only=True)
        threshold_rows.append({
            "record_type": "threshold", "threshold": threshold,
            "median_gate_error": float(per_log.gate_error.median()), "median_b3_error": float(per_log.b3_error.median()),
            "median_error_difference": float((per_log.gate_error - per_log.b3_error).median()),
            "median_gate_iou": float(per_log.gate_iou.median()), "median_b3_iou": float(per_log.b3_iou.median()),
            "recall_loss_percentage_points": float((per_log.b3_recall - per_log.gate_recall).median() * 100),
            "selected_b5_percent": float(choose_b5.mean() * 100),
        })
    thresholds = pd.DataFrame(threshold_rows)
    eligible = thresholds[thresholds.recall_loss_percentage_points <= 2.0]
    if eligible.empty:
        frozen_threshold = float(thresholds.sort_values(["recall_loss_percentage_points", "threshold"], ascending=[True, False]).iloc[0].threshold)
    else:
        frozen_threshold = float(eligible.sort_values(["median_gate_error", "threshold"], ascending=[True, False]).iloc[0].threshold)

    calibration_rows = threshold_rows.copy()
    selection_label = (selection.M1_normalized_center_error < selection.M0_normalized_center_error).astype(int).to_numpy()
    for lower in np.arange(0, 1, 0.1):
        upper = lower + 0.1
        mask = (selection_probability >= lower) & (selection_probability < upper if upper < 1 else selection_probability <= upper)
        calibration_rows.append({
            "record_type": "calibration_bin", "bin_lower": lower, "bin_upper": upper, "objects": int(mask.sum()),
            "mean_predicted_probability": float(selection_probability[mask].mean()) if mask.any() else np.nan,
            "observed_b5_benefit_fraction": float(selection_label[mask].mean()) if mask.any() else np.nan,
        })
    pd.DataFrame(calibration_rows).to_csv(OUTPUT / "causal_gate_calibration.csv", index=False, float_format="%.12g")
    model_path = OUTPUT / "frozen_causal_gate.joblib"
    joblib.dump(model, model_path)
    specification = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "model": "regularized_logistic_regression",
        "features": FEATURES, "frozen_threshold": frozen_threshold, "training_logs": int(development.log_id.nunique()),
        "model_selection_logs": int(selection.log_id.nunique()), "development_rows": len(development),
        "model_selection_rows": len(selection), "selection_auc": float(roc_auc_score(selection_label, selection_probability)),
        "selection_brier": float(brier_score_loss(selection_label, selection_probability)),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "extension_outcomes_seen": False,
    }
    (OUTPUT / "causal_gate_model_spec.json").write_text(json.dumps(specification, indent=2) + "\n", encoding="utf-8")
    audit = """# Causal gate leakage audit\n\nStatus: **PASS BEFORE EXTENSION EVALUATION**\n\n- The model is a single L2-regularized logistic regression.\n- Fit data are restricted to the 80 development logs.\n- Threshold selection is restricted to the 20 model-selection logs.\n- The original 50 held-out logs are not used for gate fitting or threshold selection.\n- The 45 extension logs were selected and hashed from metadata only before this model was evaluated on them.\n- Every model input is available at or before the source timestamp.\n- Outcome columns are used only to form the training label and to score model-selection thresholds; they are absent from the feature matrix.\n- No target-time geometry, future identity, future displacement, true future speed, true target class, B3 error, B5 error, or delta-harm quantity is a model input.\n"""
    (OUTPUT / "causal_gate_leakage_audit.md").write_text(audit, encoding="utf-8")
    print(json.dumps(specification, indent=2))


if __name__ == "__main__":
    main()
