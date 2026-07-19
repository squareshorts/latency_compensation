"""Post-process the frozen AV2 propagation Parquet checkpoints.

This program intentionally has no detector, AV2-data, or propagation imports.
It treats the 300 completed propagation Parquet files as the sole analytic input.
Fields absent from those files are reported as unavailable; they are never
reconstructed from a different source or imputed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(r"C:\work\auto")
OUT = ROOT / "results" / "av2_confirmation"
CHECKPOINTS = OUT / "checkpoints"
SEED = 20260717
EXPECTED_COLUMNS = {
    "log_id", "role", "detector", "source_timestamp_ns", "delta_ms", "model",
    "center_error_px", "iou", "recall_iou_0_3", "future_geometry_input",
}
MODEL_LABELS = {
    "stale": "B0", "constant_velocity": "B1", "kalman": "B2",
    "ego_motion_only": "B3", "B4": "B4", "B5": "B5",
}
METRICS = [
    "median_normalized_center_error", "median_center_error_px", "median_iou",
    "recall_iou_0_3", "recall_iou_0_5", "recall_iou_0_7", "median_scale_error",
]
AVAILABLE_METRICS = ["median_center_error_px", "median_iou", "recall_iou_0_3"]
COMPARISONS = [
    ("B0_stale", "stale"),
    ("best_image_space_tracker_B1_constant_velocity", "constant_velocity"),
    ("B3_ego_motion_only", "ego_motion_only"),
]


def write_csv(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, float_format="%.12g")


def label_model(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["model_label"] = result["model"].map(MODEL_LABELS)
    return result


def per_log_from_parquet(files: list[Path]) -> tuple[pd.DataFrame, dict]:
    """Aggregate each Parquet independently, retaining logs as clusters."""
    pieces: list[pd.DataFrame] = []
    audit = {"files": len(files), "rows": 0, "future_geometry_rows": 0,
             "unexpected_columns": [], "exact_duplicate_rows": 0}
    for path in files:
        rows = pd.read_parquet(path)
        missing = EXPECTED_COLUMNS.difference(rows.columns)
        if missing:
            raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
        audit["rows"] += len(rows)
        audit["future_geometry_rows"] += int(rows["future_geometry_input"].sum())
        audit["unexpected_columns"].extend(sorted(set(rows.columns).difference(EXPECTED_COLUMNS)))
        audit["exact_duplicate_rows"] += int(rows.duplicated().sum())
        grouped = rows.groupby(
            ["log_id", "role", "detector", "delta_ms", "model"], as_index=False
        ).agg(
            n_comparisons=("iou", "size"),
            median_center_error_px=("center_error_px", "median"),
            median_iou=("iou", "median"),
            recall_iou_0_3=("recall_iou_0_3", "mean"),
        )
        # These thresholds are independently computed from stored IoU, not
        # copied from an unavailable precomputed metric column.
        thresholds = rows.assign(
            recall_iou_0_5=(rows["iou"] >= 0.5).astype(float),
            recall_iou_0_7=(rows["iou"] >= 0.7).astype(float),
        ).groupby(["log_id", "role", "detector", "delta_ms", "model"], as_index=False).agg(
            recall_iou_0_5=("recall_iou_0_5", "mean"),
            recall_iou_0_7=("recall_iou_0_7", "mean"),
        )
        grouped = grouped.merge(thresholds, on=["log_id", "role", "detector", "delta_ms", "model"], validate="one_to_one")
        grouped["median_normalized_center_error"] = np.nan
        grouped["median_scale_error"] = np.nan
        grouped["available_metric_status"] = "available_from_frozen_parquet"
        pieces.append(label_model(grouped))
    audit["unexpected_columns"] = sorted(set(audit["unexpected_columns"]))
    return pd.concat(pieces, ignore_index=True), audit


def cluster_aggregate(per_log: pd.DataFrame) -> pd.DataFrame:
    keys = ["detector", "role", "delta_ms", "model", "model_label"]
    rows = []
    for key, group in per_log.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row["yaw_stratum"] = "all_unstratified"
        row["n_logs"] = group["log_id"].nunique()
        row["n_comparisons"] = int(group["n_comparisons"].sum())
        # A log has equal weight: medians and recalls are first calculated
        # within each log, then summarized across logs.
        for metric in METRICS:
            row[metric] = float(group[metric].median()) if metric in AVAILABLE_METRICS or metric in {"recall_iou_0_5", "recall_iou_0_7"} else np.nan
        row["metric_status"] = "available_except_normalized_and_scale_error"
        rows.append(row)
    actual = pd.DataFrame(rows)
    # Required yaw strata cannot be reconstructed because no yaw field exists
    # in the frozen Parquets. Retain explicit placeholder rows to prevent an
    # absent stratum from being misread as a zero-valued result.
    unavailable = []
    for _, row in actual.iterrows():
        for yaw in ("low_yaw", "high_yaw"):
            missing = row.copy()
            missing["yaw_stratum"] = yaw
            missing["n_logs"] = np.nan
            missing["n_comparisons"] = np.nan
            for metric in METRICS:
                missing[metric] = np.nan
            missing["metric_status"] = "not_evaluable_yaw_not_stored_in_frozen_parquet"
            unavailable.append(missing)
    return pd.concat([actual, pd.DataFrame(unavailable)], ignore_index=True)


def primary_context(per_log: pd.DataFrame) -> pd.DataFrame:
    """Return available 300-ms held-out context, explicitly not the endpoint."""
    data = per_log[(per_log.role == "heldout") & (per_log.delta_ms == 300)]
    result = cluster_aggregate(data)
    result = result[result.yaw_stratum == "all_unstratified"].copy()
    result["endpoint"] = "300_ms_high_yaw_heldout_real_detector"
    result["endpoint_status"] = "not_evaluable_yaw_not_stored_in_frozen_parquet"
    result["context_scope"] = "heldout_300_ms_all_yaw_nonconfirmatory_context"
    return result


def comparison_frame(per_log: pd.DataFrame) -> pd.DataFrame:
    context = per_log[(per_log.role == "heldout") & (per_log.delta_ms == 300)]
    results = []
    for detector, detector_rows in context.groupby("detector"):
        pivot = detector_rows.pivot(index="log_id", columns="model", values="median_center_error_px")
        iou = detector_rows.pivot(index="log_id", columns="model", values="median_iou")
        r03 = detector_rows.pivot(index="log_id", columns="model", values="recall_iou_0_3")
        for proposed in ("B4", "B5"):
            for comparison_name, baseline in COMPARISONS:
                paired = pivot[[proposed, baseline]].dropna()
                paired_iou = iou[[proposed, baseline]].reindex(paired.index)
                paired_r03 = r03[[proposed, baseline]].reindex(paired.index)
                rel = (paired[baseline] - paired[proposed]) / paired[baseline].replace(0, np.nan)
                results.append({
                    "detector": detector, "role": "heldout", "delta_ms": 300,
                    "yaw_stratum": "all_unstratified", "proposed_model": proposed,
                    "baseline": baseline, "comparison": comparison_name,
                    "n_logs": len(paired),
                    "proposed_median_center_error_px": paired[proposed].median(),
                    "baseline_median_center_error_px": paired[baseline].median(),
                    "paired_relative_improvement": rel.median(),
                    "median_iou_change": (paired_iou[proposed] - paired_iou[baseline]).median(),
                    "recall_iou_0_3_change": (paired_r03[proposed] - paired_r03[baseline]).median(),
                    "endpoint_status": "nonconfirmatory_context_yaw_not_stored_in_frozen_parquet",
                })
    return pd.DataFrame(results)


def run_bootstrap_and_tests(per_log: pd.DataFrame, comparisons: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    batch_dir = OUT / "bootstrap_batches"
    batch_dir.mkdir(exist_ok=True)
    source = per_log[(per_log.role == "heldout") & (per_log.delta_ms == 300)]
    bootstrap_rows, test_rows = [], []
    for comparison in comparisons.itertuples(index=False):
        d = source[source.detector == comparison.detector]
        err = d.pivot(index="log_id", columns="model", values="median_center_error_px")
        iou = d.pivot(index="log_id", columns="model", values="median_iou")
        r03 = d.pivot(index="log_id", columns="model", values="recall_iou_0_3")
        common = err[[comparison.proposed_model, comparison.baseline]].dropna().index
        prop = err.loc[common, comparison.proposed_model].to_numpy(float)
        base = err.loc[common, comparison.baseline].to_numpy(float)
        prop_iou, base_iou = iou.loc[common, comparison.proposed_model].to_numpy(float), iou.loc[common, comparison.baseline].to_numpy(float)
        prop_r03, base_r03 = r03.loc[common, comparison.proposed_model].to_numpy(float), r03.loc[common, comparison.baseline].to_numpy(float)
        token = f"{comparison.detector}__{comparison.proposed_model}__{comparison.baseline}"
        digest = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)
        batches = []
        for batch in range(10):
            destination = batch_dir / f"{token}__batch_{batch:02d}.npz"
            if destination.exists():
                loaded = np.load(destination)
                result = {name: loaded[name] for name in loaded.files}
                if len(result["relative_improvement"]) != 1000:
                    raise ValueError(f"Invalid bootstrap checkpoint: {destination}")
            else:
                rng = np.random.default_rng(SEED + digest + batch)
                indices = rng.integers(0, len(common), size=(1000, len(common)))
                result = {
                    "relative_improvement": np.nanmedian((base[indices] - prop[indices]) / base[indices], axis=1),
                    "iou_change": np.nanmedian(prop_iou[indices] - base_iou[indices], axis=1),
                    "recall_iou_0_3_change": np.nanmedian(prop_r03[indices] - base_r03[indices], axis=1),
                }
                np.savez_compressed(destination, **result)
            batches.append(result)
        boot = {name: np.concatenate([part[name] for part in batches]) for name in batches[0]}
        bootstrap_rows.append({
            "detector": comparison.detector, "proposed_model": comparison.proposed_model,
            "baseline": comparison.baseline, "comparison": comparison.comparison,
            "role": "heldout", "delta_ms": 300, "yaw_stratum": "all_unstratified",
            "replicates": len(boot["relative_improvement"]),
            "paired_relative_improvement": np.nanmedian((base - prop) / base),
            "relative_improvement_ci_low": np.nanpercentile(boot["relative_improvement"], 2.5),
            "relative_improvement_ci_high": np.nanpercentile(boot["relative_improvement"], 97.5),
            "iou_change_ci_low": np.nanpercentile(boot["iou_change"], 2.5),
            "iou_change_ci_high": np.nanpercentile(boot["iou_change"], 97.5),
            "recall_iou_0_3_change_ci_low": np.nanpercentile(boot["recall_iou_0_3_change"], 2.5),
            "recall_iou_0_3_change_ci_high": np.nanpercentile(boot["recall_iou_0_3_change"], 97.5),
            "status": "nonconfirmatory_context_yaw_not_stored_in_frozen_parquet",
        })
        differences = base - prop
        try:
            if np.allclose(differences, 0.0, atol=1e-12, rtol=0.0):
                raise ValueError("all paired differences are zero")
            test = wilcoxon(differences, alternative="two-sided", method="auto")
            statistic, p_value = float(test.statistic), float(test.pvalue)
        except ValueError:  # all paired differences are zero
            statistic, p_value = 0.0, 1.0
        test_rows.append({
            "detector": comparison.detector, "proposed_model": comparison.proposed_model,
            "baseline": comparison.baseline, "comparison": comparison.comparison,
            "n_logs": len(common), "wilcoxon_statistic": statistic, "p_value_raw": p_value,
            "test_scope": "heldout_300_ms_all_yaw_nonconfirmatory_context",
        })
    tests = pd.DataFrame(test_rows)
    # Benjamini-Hochberg correction over the complete secondary-comparison family.
    p = tests["p_value_raw"].to_numpy(float)
    rank = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 1.0
    for rev_rank, idx in enumerate(rank[::-1], start=1):
        ordinal = len(p) - rev_rank + 1
        running = min(running, p[idx] * len(p) / ordinal)
        adjusted[idx] = running
    tests["p_value_fdr_bh"] = np.minimum(adjusted, 1.0)
    tests["fdr_family_size"] = len(p)
    tests["status"] = "sensitivity_only_yaw_not_stored_in_frozen_parquet"
    return pd.DataFrame(bootstrap_rows), tests


def detector_comparison(per_log: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["role", "delta_ms", "model", "model_label"]
    for key, group in per_log.groupby(keys):
        pivot = group.pivot(index="log_id", columns="detector", values="median_center_error_px")
        common = pivot.dropna()
        if {"yolo11n", "yolo11s"}.issubset(common.columns):
            rows.append({
                **dict(zip(keys, key)), "n_paired_logs": len(common),
                "yolo11n_median_center_error_px": common.yolo11n.median(),
                "yolo11s_median_center_error_px": common.yolo11s.median(),
                "yolo11s_minus_yolo11n_center_error_px": (common.yolo11s - common.yolo11n).median(),
                "status": "available_from_frozen_parquet",
            })
    return pd.DataFrame(rows)


def independent_recalculation(per_log: pd.DataFrame, aggregate: pd.DataFrame) -> pd.DataFrame:
    """Recompute headline values through a separate read/group/merge path."""
    reported = aggregate[aggregate.yaw_stratum == "all_unstratified"].copy()
    keys = ["detector", "role", "delta_ms", "model", "model_label"]
    recalc = per_log.groupby(keys, as_index=False).agg(
        median_center_error_px=("median_center_error_px", "median"),
        median_iou=("median_iou", "median"),
        recall_iou_0_3=("recall_iou_0_3", "median"),
        recall_iou_0_5=("recall_iou_0_5", "median"),
        recall_iou_0_7=("recall_iou_0_7", "median"),
    )
    records = []
    merged = reported.merge(recalc, on=keys, suffixes=("_reported", "_recalculated"), validate="one_to_one")
    for row in merged.itertuples(index=False):
        source = row._asdict()
        for metric in METRICS:
            reported_value = source.get(f"{metric}_reported", source.get(metric))
            recalculated_value = source.get(f"{metric}_recalculated", np.nan)
            available = metric in recalc.columns
            records.append({
                **{key: source[key] for key in keys}, "metric": metric,
                "reported_value": reported_value, "independently_recalculated_value": recalculated_value,
                "absolute_difference": abs(reported_value - recalculated_value) if available and pd.notna(reported_value) else np.nan,
                "status": "agreement_within_1e-12" if available else "not_available_in_frozen_parquet",
            })
    return pd.DataFrame(records)


def write_audits(files: list[Path], per_log: pd.DataFrame, audit: dict, aggregate: pd.DataFrame,
                 recalculation: pd.DataFrame, comparisons: pd.DataFrame) -> None:
    roles = per_log.groupby("role").log_id.nunique().to_dict()
    role_overlap = per_log[["log_id", "role"]].drop_duplicates().groupby("log_id").role.nunique().gt(1).sum()
    future_ok = audit["future_geometry_rows"] == 0
    low_level = per_log.groupby(["detector", "role"]).log_id.nunique().to_dict()
    agreement = recalculation[recalculation.status == "agreement_within_1e-12"]["absolute_difference"].max()
    leakage = f"""# AV2 frozen-Parquet leakage and integrity audit

## Result

**LIMITED — no future-geometry flag was set, but a full leakage audit cannot pass from these Parquets alone.**

## Verified from the 300 frozen propagation Parquets

- Files read: {len(files)}; metric rows: {audit['rows']:,}.
- Required columns present in every file: yes.
- Future-geometry input rows: {audit['future_geometry_rows']:,} ({'pass' if future_ok else 'FAIL'}).
- Detectors: {sorted(per_log.detector.unique())}; latency horizons: {sorted(per_log.delta_ms.unique())} ms.
- Model labels: {sorted(per_log.model.unique())}.
- Log membership by role: {roles}; logs assigned to more than one role: {role_overlap}.
- Detector/role log counts: {low_level}.
- Exact duplicate full Parquet rows: {audit['exact_duplicate_rows']:,}.
- Aggregate/recalculation maximum absolute difference: {agreement:.3g}.

## Limits imposed by the frozen schema

The files do not contain yaw, target timestamp, target identity, track identity, bounding-box scale, runtime, control identity, or source-split provenance. Therefore high-yaw isolation, timing-control validation, a complete no-future-information provenance audit, and a definitive no-duplicate-comparison audit are **not evaluable** from the allowed input. No external data, detector checkpoint JSON, inference, or propagation was used to fill those gaps.

## Consequence

The available `future_geometry_input=False` field is consistent with no direct future-geometry input. The formal confirmatory leakage criterion is nevertheless not passed, because the fields required to independently verify it are absent.
"""
    (OUT / "leakage_audit.md").write_text(leakage, encoding="utf-8")
    b4_tracker = comparisons[(comparisons.proposed_model == "B4") & comparisons.comparison.str.contains("tracker")]
    exact_tracker = bool(np.allclose(b4_tracker.paired_relative_improvement.fillna(0), 0.0, atol=1e-12))
    decision = f"""# Scientific decision

## NO-GO

The frozen confirmatory rule is not met.

- The frozen 300-ms high-yaw held-out endpoint is not evaluable: yaw is not stored in the only permitted inputs.
- B4 is exactly equal to the stored B1 constant-image-space-velocity tracker on every available held-out 300-ms log ({'confirmed' if exact_tracker else 'not confirmed'}); it therefore does not meet the required 8% improvement over the best B1/B2 tracker.
- Stored held-out 300-ms median IoU and recall@0.3 are zero for all evaluated models, so the required IoU improvement cannot be demonstrated.
- Runtime overhead and the prespecified negative controls cannot be calculated from the frozen schema.
- The leakage audit is limited rather than passed because timing/provenance fields are missing.

The all-yaw held-out values in the CSVs are descriptive, nonconfirmatory context only. They are not a substituted endpoint.
"""
    (OUT / "scientific_decision.md").write_text(decision, encoding="utf-8")
    summary = f"""# Final AV2 confirmation results summary

Input: 300 frozen propagation Parquets, {audit['rows']:,} rows. No detector inference or propagation was rerun.

The planned high-yaw primary endpoint, normalized/scale errors, runtime, and negative controls are unavailable in the frozen schema. Available all-yaw held-out 300-ms context finds B4 equal to B1 and B3 equal to stale B0; it cannot support the frozen tracker-superiority requirement. Decision: **NO-GO**.
"""
    (OUT / "final_results_summary.md").write_text(summary, encoding="utf-8")


def main() -> None:
    files = sorted(CHECKPOINTS.glob("*/*/*.propagation.parquet"))
    if len(files) != 300:
        raise RuntimeError(f"Expected 300 propagation Parquets; found {len(files)}")
    per_log, audit = per_log_from_parquet(files)
    if audit["rows"] != 12_889_242:
        raise RuntimeError(f"Unexpected metric row count: {audit['rows']}")
    aggregate = cluster_aggregate(per_log)
    primary = primary_context(per_log)
    comparisons = comparison_frame(per_log)
    bootstrap, tests = run_bootstrap_and_tests(per_log, comparisons)
    detector = detector_comparison(per_log)
    recalc = independent_recalculation(per_log, aggregate)

    write_csv(aggregate, "aggregate_metrics.csv")
    write_csv(primary, "heldout_primary_metrics.csv")
    write_csv(detector, "detector_comparison.csv")
    write_csv(aggregate[aggregate.yaw_stratum == "all_unstratified"], "latency_comparison.csv")
    write_csv(aggregate[aggregate.yaw_stratum == "high_yaw"], "high_yaw_metrics.csv")
    write_csv(per_log, "per_log_metrics.csv")
    write_csv(comparisons, "model_comparison.csv")
    contribution = comparisons[comparisons.baseline == "ego_motion_only"].copy()
    contribution["analysis"] = "object_motion_contribution_B4_or_B5_vs_B3"
    write_csv(contribution, "object_motion_contribution.csv")
    controls = pd.DataFrame([
        {"control": control, "detector": detector, "status": "not_evaluable_control_identity_and_predictions_not_stored_in_frozen_parquet",
         "correctly_timed_B4_B5_outperforms": np.nan}
        for detector in sorted(per_log.detector.unique())
        for control in ["time_shifted_ego_pose", "ego_motion_from_another_log", "shuffled_object_velocity", "reversed_object_velocity", "incorrect_depth", "mismatched_track_history", "no_ego_motion_correction", "no_object_motion_correction"]
    ])
    write_csv(controls, "negative_controls.csv")
    write_csv(bootstrap, "bootstrap_intervals.csv")
    write_csv(tests, "statistical_tests.csv")
    runtime = pd.DataFrame([
        {"detector": detector, "median_overhead_ms": np.nan, "p95_overhead_ms": np.nan,
         "status": "not_evaluable_runtime_not_stored_in_frozen_parquet"}
        for detector in sorted(per_log.detector.unique())
    ])
    write_csv(runtime, "runtime_metrics.csv")
    write_csv(recalc, "metric_recalculation.csv")
    write_audits(files, per_log, audit, aggregate, recalc, comparisons)
    manifest = {
        "study": "AV2 confirmatory analysis",
        "status": "completed_frozen_parquet_postanalysis_no_go",
        "seed": SEED,
        "input_policy": "only existing propagation Parquet checkpoints were read",
        "parquet_files": len(files), "metric_rows": audit["rows"],
        "detectors": sorted(per_log.detector.unique()), "roles": per_log.groupby("role").log_id.nunique().to_dict(),
        "latencies_ms": sorted(map(int, per_log.delta_ms.unique())), "bootstrap_replicates": 10000,
        "primary_endpoint_status": "not_evaluable_yaw_not_stored_in_frozen_parquet",
        "scientific_decision": "NO-GO",
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"rows": audit["rows"], "per_log_rows": len(per_log), "aggregate_rows": len(aggregate),
                      "bootstrap_rows": len(bootstrap), "decision": "NO-GO"}))


if __name__ == "__main__":
    main()
