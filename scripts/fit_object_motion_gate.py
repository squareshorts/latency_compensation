"""Fit, freeze, and evaluate one interpretable post hoc uncertainty gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(r"C:\work\auto")
sys.path.insert(0, str(ROOT / "src"))
from latency_compensation.object_motion_harm import FrozenGate, paired_cluster_bootstrap, validate_frozen_gate

OUT = ROOT / "results" / "object_motion_harm"
DATA = OUT / "object_level_diagnostics.parquet"
NUMERIC = ["delta_ms", "true_object_speed_m_s", "ego_yaw_rate_rad_s", "object_distance_m", "track_age",
           "detector_confidence", "association_confidence", "depth_uncertainty_m", "velocity_dispersion_m_s"]
CATEGORICAL = ["object_class", "detector"]
FEATURES = NUMERIC + CATEGORICAL
SEED = 20260718


def read_role(role: str) -> pd.DataFrame:
    columns = ["comparison_id", "log_id", "role", "detector", "B3_error", "B5_error", "delta_harm", "B3_iou", "B5_iou",
               "M0_recall_iou_0_3", "M0_recall_iou_0_5", "M0_recall_iou_0_7",
               "M1_recall_iou_0_3", "M1_recall_iou_0_5", "M1_recall_iou_0_7", *FEATURES]
    frame = pd.read_parquet(DATA, columns=list(dict.fromkeys(columns)), filters=[("role", "==", role)])
    frame["benefit"] = (frame.B5_error < frame.B3_error).astype(int)
    frame["abs_ego_yaw_rate_rad_s"] = frame.ego_yaw_rate_rad_s.abs()
    frame["ego_yaw_rate_rad_s"] = frame.abs_ego_yaw_rate_rad_s
    return frame


def preprocessor(spline: bool = False) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if spline:
        numeric_steps += [("spline", SplineTransformer(n_knots=4, degree=2, include_bias=False)), ("scale", StandardScaler(with_mean=False))]
    else:
        numeric_steps += [("scale", StandardScaler())]
    return ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), NUMERIC),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])


def candidate_models() -> dict[str, Pipeline]:
    return {
        "regularized_logistic": Pipeline([("features", preprocessor(False)),
                                           ("model", LogisticRegression(C=0.1, max_iter=500, class_weight="balanced", random_state=SEED))]),
        "spline_logistic_gam": Pipeline([("features", preprocessor(True)),
                                         ("model", LogisticRegression(C=0.1, max_iter=500, class_weight="balanced", random_state=SEED))]),
        "shallow_tree": Pipeline([("features", preprocessor(False)),
                                   ("model", DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.02, class_weight="balanced", random_state=SEED))]),
    }


def evaluation(model, frame: pd.DataFrame) -> dict[str, float]:
    probability = model.predict_proba(frame[FEATURES])[:, 1]
    return {"roc_auc": roc_auc_score(frame.benefit, probability), "brier_score": brier_score_loss(frame.benefit, probability),
            "log_loss": log_loss(frame.benefit, probability), "mean_predicted_benefit": probability.mean(),
            "observed_benefit": frame.benefit.mean()}


def gate_scores(frame: pd.DataFrame, probability: np.ndarray, threshold: float, age: int, depth: float, velocity: float) -> dict:
    select = (probability >= threshold) & (frame.track_age.to_numpy() >= age)
    select &= frame.depth_uncertainty_m.fillna(np.inf).to_numpy() <= depth
    select &= frame.velocity_dispersion_m_s.fillna(np.inf).to_numpy() <= velocity
    chosen = np.where(select, frame.B5_error, frame.B3_error)
    rows = []
    for detector, indices in frame.groupby("detector").groups.items():
        subset = frame.loc[indices].copy(); subset["chosen"] = chosen[frame.index.get_indexer(indices)]
        logs = subset.groupby("log_id").agg(g2=("chosen", "median"), g0=("B3_error", "median"))
        rows.append((detector, float(np.median(logs.g2 / logs.g0 - 1)), float(select[frame.index.get_indexer(indices)].mean())))
    return {"worst_detector_relative_harm": max(value[1] for value in rows), "mean_selected_fraction": np.mean([value[2] for value in rows]),
            "detector_rows": rows, "select": select, "chosen": chosen}


def main() -> None:
    development = read_role("development")
    # Deterministic cap preserves tractability while retaining every log/detector/latency stratum.
    if len(development) > 400_000:
        development_fit = development.sample(400_000, random_state=SEED).reset_index(drop=True)
    else:
        development_fit = development
    models = candidate_models()
    model_rows = []
    for name, model in models.items():
        model.fit(development_fit[FEATURES], development_fit.benefit)
        model_rows.append({"record_type": "candidate_model", "model": name, "split": "development", **evaluation(model, development)})

    # Model-selection is opened only after all candidate forms are fit.
    model_selection = read_role("model_selection")
    for name, model in models.items():
        model_rows.append({"record_type": "candidate_model", "model": name, "split": "model_selection", **evaluation(model, model_selection)})
    selection_table = pd.DataFrame(model_rows)
    selected_name = selection_table[selection_table.split == "model_selection"].sort_values(["brier_score", "log_loss", "model"]).iloc[0].model
    selected_model = models[selected_name]
    probability = selected_model.predict_proba(model_selection[FEATURES])[:, 1]

    search_rows = []
    for threshold in (0.50, 0.60, 0.70):
        for age in (1, 2, 3):
            for depth in (5.0, 10.0, np.inf):
                for velocity in (2.0, 5.0, np.inf):
                    score = gate_scores(model_selection, probability, threshold, age, depth, velocity)
                    search_rows.append({"record_type": "gate_selection", "model": selected_name, "split": "model_selection",
                                        "threshold": threshold, "minimum_track_age": age, "maximum_depth_uncertainty": depth,
                                        "maximum_velocity_uncertainty": velocity, "worst_detector_relative_harm": score["worst_detector_relative_harm"],
                                        "selected_fraction": score["mean_selected_fraction"]})
    search = pd.DataFrame(search_rows)
    eligible = search[search.selected_fraction >= 0.01]
    chosen = (eligible if not eligible.empty else search).sort_values(["worst_detector_relative_harm", "selected_fraction"]).iloc[0]
    gate = FrozenGate(selected_name, float(chosen.threshold), int(chosen.minimum_track_age),
                      float(chosen.maximum_depth_uncertainty), float(chosen.maximum_velocity_uncertainty))
    frozen = {
        "model": gate.model_name, "probability_threshold": gate.threshold, "minimum_track_age": gate.minimum_track_age,
        "maximum_depth_uncertainty_m": None if np.isinf(gate.maximum_depth_uncertainty) else gate.maximum_depth_uncertainty,
        "maximum_velocity_uncertainty_m_s": None if np.isinf(gate.maximum_velocity_uncertainty) else gate.maximum_velocity_uncertainty,
        "fit_split": "development", "selection_split": "model_selection", "heldout_status_at_freeze": "not_opened_by_this_script",
        "features": FEATURES, "seed": SEED, "future_information": False,
    }
    (OUT / "frozen_gate_specification.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (OUT / "frozen_gate_specification.md").write_text(
        "# Frozen uncertainty gate\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in frozen.items()) + "\n"
    )
    joblib.dump(selected_model, OUT / "frozen_benefit_probability_model.joblib")
    pd.concat([selection_table, search], ignore_index=True, sort=False).to_csv(OUT / "benefit_probability_model.csv", index=False, float_format="%.12g")

    # Development-only operating-envelope tables.
    envelope_rows = []
    for feature in ["delta_ms", "true_object_speed_m_s", "ego_yaw_rate_rad_s", "object_distance_m", "track_age",
                    "detector_confidence", "association_confidence", "depth_uncertainty_m", "velocity_dispersion_m_s"]:
        values = pd.to_numeric(development[feature], errors="coerce")
        bins = values if values.nunique() <= 10 else pd.qcut(values, q=5, duplicates="drop")
        temp = development.assign(bin=bins).groupby(["detector", "bin"], observed=True).agg(
            objects=("benefit", "size"), benefit_probability=("benefit", "mean"), median_delta_harm=("delta_harm", "median")
        ).reset_index()
        temp["variable"] = feature; temp["bin"] = temp.bin.astype(str)
        envelope_rows.append(temp)
    classes = development.groupby(["detector", "object_class"], observed=True).agg(objects=("benefit", "size"), benefit_probability=("benefit", "mean")).reset_index()
    classes["variable"] = "object_class"; classes["bin"] = classes.object_class; classes["median_delta_harm"] = np.nan
    pd.concat(envelope_rows + [classes], ignore_index=True, sort=False).to_csv(OUT / "operating_envelope.csv", index=False, float_format="%.12g")

    # One-shot held-out evaluation occurs only after the specification is on disk.
    validate_frozen_gate(gate, "heldout")
    heldout = read_role("heldout")
    heldout_probability = selected_model.predict_proba(heldout[FEATURES])[:, 1]
    score = gate_scores(heldout, heldout_probability, gate.threshold, gate.minimum_track_age,
                        gate.maximum_depth_uncertainty, gate.maximum_velocity_uncertainty)
    heldout["probability"] = heldout_probability; heldout["selected_B5"] = score["select"]
    heldout["G0_error"] = heldout.B3_error; heldout["G1_error"] = heldout.B5_error; heldout["G2_error"] = score["chosen"]
    heldout["G0_iou"] = heldout.B3_iou; heldout["G1_iou"] = heldout.B5_iou
    heldout["G2_iou"] = np.where(heldout.selected_B5, heldout.B5_iou, heldout.B3_iou)
    for threshold_name in ("0_3", "0_5", "0_7"):
        heldout[f"G0_recall_{threshold_name}"] = heldout[f"M0_recall_iou_{threshold_name}"]
        heldout[f"G1_recall_{threshold_name}"] = heldout[f"M1_recall_iou_{threshold_name}"]
        heldout[f"G2_recall_{threshold_name}"] = np.where(heldout.selected_B5,
                                                           heldout[f"M1_recall_iou_{threshold_name}"],
                                                           heldout[f"M0_recall_iou_{threshold_name}"])
    heldout["actual_benefit"] = heldout.B5_error < heldout.B3_error
    runtime = pd.read_csv(ROOT / "results" / "av2_confirmation" / "runtime_recovery.csv").set_index("model")
    gate_rows = []
    for detector, group in heldout.groupby("detector", observed=True):
        for gate_name in ("G0", "G1", "G2"):
            per_log = group.groupby("log_id").agg(
                error=(f"{gate_name}_error", "median"), iou=(f"{gate_name}_iou", "median"),
                recall_0_3=(f"{gate_name}_recall_0_3", "median"), recall_0_5=(f"{gate_name}_recall_0_5", "median"),
                recall_0_7=(f"{gate_name}_recall_0_7", "median"),
            )
            selected_fraction = group.selected_B5.mean() if gate_name == "G2" else (1.0 if gate_name == "G1" else 0.0)
            overhead = (1-selected_fraction) * runtime.loc["B3", "median_ms_per_object"] + selected_fraction * runtime.loc["B5", "median_ms_per_object"]
            gate_rows.append({"record_type": "endpoint", "detector": detector, "gate": gate_name, "logs": per_log.size,
                              "objects": len(group), "median_of_log_median_center_error": per_log.error.median(),
                              "median_of_log_median_iou": per_log.iou.median(), "recall_iou_0_3": per_log.recall_0_3.median(),
                              "recall_iou_0_5": per_log.recall_0_5.median(), "recall_iou_0_7": per_log.recall_0_7.median(),
                              "estimated_overhead_ms_per_object": overhead, "selected_B5_percent": selected_fraction * 100})
        log = group.groupby("log_id").agg(g2=("G2_error", "median"), g0=("G0_error", "median"))
        boot = paired_cluster_bootstrap(log.g2, log.g0, replicates=10_000, seed=SEED, statistic="median", relative=False)
        gate_rows.append({"record_type": "G2_vs_G0", "detector": detector, "gate": "G2", "logs": len(log), "objects": len(group),
                          "median_of_log_median_center_error": np.median(log.g2), "absolute_difference": boot["estimate"],
                          "ci_low": boot["ci_low"], "ci_high": boot["ci_high"], "logs_improved_percent": (log.g2 < log.g0).mean() * 100,
                          "selected_B5_percent": group.selected_B5.mean() * 100,
                          "false_positive_selection_rate": ((group.selected_B5) & (~group.actual_benefit)).mean() * 100,
                          "false_negative_selection_rate": ((~group.selected_B5) & (group.actual_benefit)).mean() * 100})
        calibration = pd.qcut(group.probability, q=10, duplicates="drop")
        for interval, cell in group.groupby(calibration, observed=True):
            gate_rows.append({"record_type": "calibration", "detector": detector, "gate": "G2", "probability_bin": str(interval),
                              "objects": len(cell), "mean_predicted_benefit": cell.probability.mean(), "observed_benefit": cell.actual_benefit.mean()})
    pd.DataFrame(gate_rows).to_csv(OUT / "gate_heldout_results.csv", index=False, float_format="%.12g")
    print(json.dumps({"selected_model": selected_name, "gate": frozen,
                      "heldout_selected_percent": float(heldout.selected_B5.mean() * 100)}))


if __name__ == "__main__":
    main()
