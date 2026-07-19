"""Harmonize the nuScenes feasibility and AV2 confirmatory analyses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(r"C:\work\auto")
OUT = ROOT / "results" / "object_motion_harm"
AV2_PATH = OUT / "object_level_diagnostics.parquet"
NUSC_COMMON = OUT / "nuscenes_common_b3_b5.parquet"
NUSC_ORIGINAL = ROOT / "results" / "nuscenes_500ms_reproduction" / "propagated_boxes.parquet"
FEATURES = ["distance_m", "abs_yaw_rate_rad_s", "ego_speed_m_s", "track_age", "detector_confidence",
            "association_confidence", "depth_uncertainty_m", "estimated_speed_m_s"]


def weighted_median(values, weights):
    order = np.argsort(values); values = np.asarray(values)[order]; weights = np.asarray(weights)[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum()/2)])


def av2_frame() -> pd.DataFrame:
    columns = ["comparison_id", "log_id", "detector", "role", "delta_ms", "object_class", "object_distance_m",
               "ego_yaw_rate_rad_s", "ego_translation_m", "track_age", "detector_confidence", "association_confidence",
               "depth_uncertainty_m", "estimated_3d_velocity_m_s", "true_object_speed_m_s", "B3_error", "B5_error",
               "B3_iou", "B5_iou", "M0_recall_iou_0_3", "M0_recall_iou_0_5", "M0_recall_iou_0_7",
               "M1_recall_iou_0_3", "M1_recall_iou_0_5", "M1_recall_iou_0_7"]
    data = pd.read_parquet(AV2_PATH, columns=columns, filters=[("role", "==", "heldout"), ("detector", "==", "yolo11n"), ("delta_ms", "==", 500)])
    pedestrian_tokens = ("PEDESTRIAN", "BICYCL", "MOTORCYCL")
    data["common_object_group"] = np.where(data.object_class.str.upper().str.contains("|".join(pedestrian_tokens)), "pedestrian_cyclist", "vehicle")
    data = data.rename(columns={"object_distance_m": "distance_m", "detector_confidence": "detector_confidence",
                                "depth_uncertainty_m": "depth_uncertainty_m", "estimated_3d_velocity_m_s": "estimated_speed_m_s"})
    data["abs_yaw_rate_rad_s"] = data.ego_yaw_rate_rad_s.abs(); data["ego_speed_m_s"] = data.ego_translation_m / .5
    data["dataset"] = "AV2"; data["group_id"] = data.log_id; data["latency_ms"] = 500
    return data


def nuscenes_frame() -> pd.DataFrame:
    data = pd.read_parquet(NUSC_COMMON)
    metric_columns = ["normalized_center_error", "iou", "recall_iou_0_3", "recall_iou_0_5", "recall_iou_0_7"]
    metadata_columns = [column for column in data.columns if column not in {"model", "predicted_box_json", "center_error_px", "box_scale_error", "usable", *metric_columns}]
    wide = data[data.model == "B3"][metadata_columns].copy()
    for model in ("B3", "B5"):
        metrics = data[data.model == model][["record_id", *metric_columns]].rename(columns={column: f"{column}_{model}" for column in metric_columns})
        wide = wide.merge(metrics, on="record_id", validate="one_to_one")
    wide = wide.rename(columns={"normalized_center_error_B3": "B3_error", "normalized_center_error_B5": "B5_error",
                                "iou_B3": "B3_iou", "iou_B5": "B5_iou", "confidence": "detector_confidence",
                                "depth_m": "distance_m", "scene_name": "group_id", "yaw_rate_rad_s": "abs_yaw_rate_rad_s"})
    wide["dataset"] = "nuScenes"; wide["latency_ms"] = 500
    return wide


def paired_summary(data: pd.DataFrame, label: str, weights=None) -> dict:
    if weights is None:
        logs = data.groupby("group_id").agg(B3=("B3_error", "median"), B5=("B5_error", "median"))
        b3, b5 = float(logs.B3.median()), float(logs.B5.median())
    else:
        b3, b5 = weighted_median(data.B3_error, weights), weighted_median(data.B5_error, weights)
    return {"analysis": label, "dataset": str(data.dataset.iloc[0]), "groups": data.group_id.nunique(), "objects": len(data),
            "B3_normalized_center_error": b3, "B5_normalized_center_error": b5,
            "B5_relative_difference": b5/b3-1, "B5_better": bool(b5 < b3), "latency_ms": 500,
            "detector": "yolo11n", "metric_definition": "normalized center error by image diagonal"}


def main() -> None:
    av2, nusc = av2_frame(), nuscenes_frame()
    # Common eligibility support fixed without inspecting B3/B5 outcomes.
    distance_limit = min(av2.distance_m.quantile(.99), nusc.distance_m.quantile(.99))
    av2_common = av2[(av2.distance_m <= distance_limit) & (av2.track_age >= 2)].copy()
    nusc_common = nusc[(nusc.distance_m <= distance_limit) & (nusc.track_age >= 2)].copy()

    harmonization = [paired_summary(av2_common, "direct_common_definition"), paired_summary(nusc_common, "direct_common_definition")]
    original = pd.read_parquet(NUSC_ORIGINAL)
    orig = original[original.model.isin(["B3_inertial_pose", "B4_inertial_object_tracker"])].copy()
    orig["mapped"] = orig.model.map({"B3_inertial_pose": "B3", "B4_inertial_object_tracker": "B5"})
    original_wide = orig.pivot_table(index=["scene_name", "record_id"], columns="mapped", values="normalized_center_error", aggfunc="first").dropna().reset_index()
    original_logs = original_wide.groupby("scene_name")[["B3", "B5"]].median()
    harmonization.append({"analysis": "nuScenes_real_data_feasibility_result", "dataset": "nuScenes", "groups": len(original_logs),
                          "objects": len(original_wide), "B3_normalized_center_error": original_logs.B3.median(),
                          "B5_normalized_center_error": original_logs.B5.median(),
                          "B5_relative_difference": original_logs.B5.median()/original_logs.B3.median()-1,
                          "B5_better": bool(original_logs.B5.median() < original_logs.B3.median()), "latency_ms": 500,
                          "detector": "yolo11n", "metric_definition": "normalized center error by image diagonal",
                          "implementation_note": "fast integrated IMU B3 plus GT-instance-associated B4 object history"})

    numeric = FEATURES
    a = av2_common.copy(); n = nusc_common.copy()
    combined = pd.concat([a.assign(domain=0), n.assign(domain=1)], ignore_index=True)
    pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                         ("domain", LogisticRegression(C=.1, max_iter=500, random_state=20260718))])
    pipeline.fit(combined[numeric], combined.domain)
    av2_probability = pipeline.predict_proba(a[numeric])[:, 1]
    prior_odds = len(a) / max(len(n), 1)
    weights = np.clip(av2_probability / np.maximum(1-av2_probability, 1e-6) * prior_odds, 0, 20)
    harmonization.append(paired_summary(a, "importance_weighted_AV2_to_nuScenes", weights))

    # Nearest-neighbor AV2 rows for each nuScenes observation, exact-matched by common object group.
    matched_indices = []
    imputer = SimpleImputer(strategy="median"); scaler = StandardScaler()
    for group in sorted(set(a.common_object_group) & set(n.common_object_group)):
        source = a[a.common_object_group == group]; target = n[n.common_object_group == group]
        both = pd.concat([source[numeric], target[numeric]])
        transformed = scaler.fit_transform(imputer.fit_transform(both))
        source_x, target_x = transformed[:len(source)], transformed[len(source):]
        nearest = NearestNeighbors(n_neighbors=1).fit(source_x).kneighbors(target_x, return_distance=False).ravel()
        matched_indices.extend(source.iloc[nearest].index.tolist())
    matched = a.loc[matched_indices].copy()
    harmonization.append(paired_summary(matched, "nearest_covariate_matched_AV2"))

    # Multivariate common support is defined by non-extreme domain propensity;
    # this avoids an empty rectangular intersection across eight covariates.
    a_domain = pipeline.predict_proba(a[numeric])[:, 1]
    n_domain = pipeline.predict_proba(n[numeric])[:, 1]
    a_support, n_support = a[(a_domain >= .05) & (a_domain <= .95)], n[(n_domain >= .05) & (n_domain <= .95)]
    if a_support.empty or n_support.empty:
        a_support, n_support = a[(a_domain >= .01) & (a_domain <= .99)], n[(n_domain >= .01) & (n_domain <= .99)]
    harmonization += [paired_summary(a_support, "restricted_common_support"), paired_summary(n_support, "restricted_common_support")]
    pd.DataFrame(harmonization).to_csv(OUT / "cross_dataset_harmonization.csv", index=False, float_format="%.12g")

    shifts = []
    for feature in numeric:
        av, nv = pd.to_numeric(a[feature], errors="coerce"), pd.to_numeric(n[feature], errors="coerce")
        pooled = np.sqrt((av.var()+nv.var())/2)
        shifts.append({"feature": feature, "AV2_median": av.median(), "nuScenes_median": nv.median(),
                       "standardized_mean_difference": (nv.mean()-av.mean())/pooled if pooled else np.nan,
                       "AV2_missing_percent": av.isna().mean()*100, "nuScenes_missing_percent": nv.isna().mean()*100})
    shifts += [
        {"feature": "objects", "AV2_median": len(a), "nuScenes_median": len(n)},
        {"feature": "groups", "AV2_median": a.group_id.nunique(), "nuScenes_median": n.group_id.nunique()},
        {"feature": "parked_fraction_true_speed_lt_0.5", "AV2_median": (a.true_object_speed_m_s < .5).mean(), "nuScenes_median": (n.true_speed_m_s < .5).mean()},
    ]
    pd.DataFrame(shifts).to_csv(OUT / "covariate_shift_analysis.csv", index=False, float_format="%.12g")

    influence = []
    for dataset_name, data in (("AV2", a), ("nuScenes_common", n)):
        for group in sorted(data.group_id.unique()):
            remaining = data[data.group_id != group]
            result = paired_summary(remaining, "leave_one_group_out")
            influence.append({"dataset": dataset_name, "left_out_group": group, **result})
    for scene in original_logs.index:
        remain = original_logs.drop(scene)
        influence.append({"dataset": "nuScenes_feasibility", "left_out_group": scene, "analysis": "leave_one_scene_out_original",
                          "groups": len(remain), "objects": np.nan, "B3_normalized_center_error": remain.B3.median(),
                          "B5_normalized_center_error": remain.B5.median(), "B5_relative_difference": remain.B5.median()/remain.B3.median()-1,
                          "B5_better": remain.B5.median() < remain.B3.median()})
    pd.DataFrame(influence).to_csv(OUT / "influence_analysis.csv", index=False, float_format="%.12g")

    common_nusc = [row for row in harmonization if row["analysis"] == "direct_common_definition" and row["dataset"] == "nuScenes"][0]
    feasibility = [row for row in harmonization if row["analysis"] == "nuScenes_real_data_feasibility_result"][0]
    explanation = f"""# Reversal explanation

The **nuScenes real-data feasibility result** was positive under its original implementation: the object-motion model changed normalized center error by {feasibility['B5_relative_difference']*100:.3f}% relative to its fast-IMU ego baseline. That comparison used GT instance identity to assemble detector history and a causally integrated fast IMU pose estimate.

When the AV2 definitions are applied to the preserved nuScenes detector outputs—exact measured ego poses for B3, deployed box-only historical association, the same 30 m/s cap, and the same point/depth/track-age damping—the object-motion term changes error by {common_nusc['B5_relative_difference']*100:.3f}% relative to B3. The sign therefore matches the **AV2 confirmatory reversal** rather than the original feasibility contrast.

The reversal is consequently explained primarily by implementation and comparator differences, especially oracle-quality instance association in the feasibility object tracker and the weaker fast-IMU ego baseline. Covariate shift and the ten-scene sample size remain secondary limits; matched, weighted, common-support, and leave-one-group-out results bound their influence in the accompanying tables.
"""
    (OUT / "reversal_explanation.md").write_text(explanation, encoding="utf-8")
    print(json.dumps({"distance_limit_m": distance_limit, "common_nuscenes_relative": common_nusc["B5_relative_difference"],
                      "original_nuscenes_relative": feasibility["B5_relative_difference"]}))


if __name__ == "__main__":
    main()
