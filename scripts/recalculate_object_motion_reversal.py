"""Recalculate the frozen B3-versus-B5 reversal from corrected AV2 Parquets.

Detector inference is neither imported nor callable.  Every inferential row is
first collapsed within log; log is the independent unit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(r"C:\work\auto")
sys.path.insert(0, str(ROOT / "src"))

from latency_compensation.object_motion_harm import cluster_bootstrap, paired_cluster_bootstrap

SOURCE = ROOT / "results" / "av2_confirmation" / "corrected_checkpoints"
OUT = ROOT / "results" / "object_motion_harm"
METRICS = ["normalized_center_error", "iou", "recall_iou_0_3", "recall_iou_0_5", "recall_iou_0_7"]
SEED = 20260718
REPLICATES = 10_000


def safe_wilcoxon(candidate: pd.Series, reference: pd.Series) -> tuple[float, float]:
    delta = pd.to_numeric(candidate, errors="coerce") - pd.to_numeric(reference, errors="coerce")
    delta = delta[np.isfinite(delta)]
    if not len(delta) or np.allclose(delta, 0):
        return 0.0, 1.0
    result = wilcoxon(delta, zero_method="wilcox", alternative="two-sided", method="approx")
    return float(result.statistic), float(result.pvalue)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    threshold = float(json.loads((ROOT / "results" / "av2_confirmation" / "high_yaw_threshold.json").read_text())["threshold_abs_yaw_rate_rad_s"])
    per_log = []
    direct_accumulator: dict[tuple, dict[str, list[np.ndarray]]] = {}
    files = sorted(SOURCE.glob("*/*/*.propagation.parquet"))
    for path in files:
        data = pd.read_parquet(path, columns=["comparison_id", "log_id", "role", "detector", "delta_ms", "model", "yaw_rate_rad_s", *METRICS])
        data = data[data.model.isin(["B3", "B5"])].copy()
        if data.empty:
            continue
        data["high_yaw"] = data.yaw_rate_rad_s.abs() >= threshold
        role, detector, log_id = str(data.role.iloc[0]), str(data.detector.iloc[0]), str(data.log_id.iloc[0])
        for delta_ms in sorted(data.delta_ms.unique()):
            latency = data[data.delta_ms == delta_ms]
            for yaw_group, subset in (("all_yaw", latency), ("high_yaw", latency[latency.high_yaw])):
                if subset.empty:
                    continue
                counts = subset.groupby("model").comparison_id.nunique()
                if not {"B3", "B5"}.issubset(counts.index):
                    continue
                record = {"role": role, "detector": detector, "log_id": log_id, "delta_ms": int(delta_ms), "yaw_group": yaw_group,
                          "comparisons": int(min(counts["B3"], counts["B5"]))}
                for metric in METRICS:
                    medians = subset.groupby("model")[metric].median()
                    record[f"B3_{metric}"] = float(medians["B3"])
                    record[f"B5_{metric}"] = float(medians["B5"])
                    record[f"delta_{metric}"] = float(medians["B5"] - medians["B3"])
                    record[f"relative_{metric}"] = float(medians["B5"] / medians["B3"] - 1) if medians["B3"] != 0 else np.nan
                    key = (role, detector, int(delta_ms), yaw_group)
                    direct_accumulator.setdefault(key, {}).setdefault(f"{metric}_B3", []).append(subset.loc[subset.model == "B3", metric].to_numpy())
                    direct_accumulator.setdefault(key, {}).setdefault(f"{metric}_B5", []).append(subset.loc[subset.model == "B5", metric].to_numpy())
                per_log.append(record)

    per_log_frame = pd.DataFrame(per_log).sort_values(["role", "detector", "delta_ms", "yaw_group", "log_id"])
    per_log_frame.to_csv(OUT / "per_log_reversal.csv", index=False, float_format="%.12g")

    summary_rows = []
    for keys, group in per_log_frame.groupby(["role", "detector", "delta_ms", "yaw_group"], observed=True):
        role, detector, delta_ms, yaw_group = keys
        direct = direct_accumulator[keys]
        for model in ("B3", "B5"):
            row = {"role": role, "detector": detector, "delta_ms": int(delta_ms), "yaw_group": yaw_group, "model": model,
                   "logs": int(group.log_id.nunique()), "comparisons": int(group.comparisons.sum())}
            for metric in METRICS:
                row[f"median_of_log_medians_{metric}"] = float(group[f"{model}_{metric}"].median())
                row[f"pooled_object_median_{metric}"] = float(np.median(np.concatenate(direct[f"{metric}_{model}"])))
            summary_rows.append(row)
    pd.DataFrame(summary_rows).sort_values(["role", "detector", "delta_ms", "yaw_group", "model"]).to_csv(
        OUT / "reversal_recalculation.csv", index=False, float_format="%.12g"
    )

    bootstrap_rows = []
    for group_index, (keys, group) in enumerate(per_log_frame.groupby(["role", "detector", "delta_ms", "yaw_group"], observed=True)):
        role, detector, delta_ms, yaw_group = keys
        for metric_index, metric in enumerate(METRICS):
            candidate, reference = group[f"B5_{metric}"], group[f"B3_{metric}"]
            for relative in (False, True):
                if relative:
                    paired_relative = candidate.to_numpy(float) / reference.to_numpy(float) - 1.0
                    result = cluster_bootstrap(paired_relative, replicates=REPLICATES,
                                               seed=SEED + group_index * 100 + metric_index * 2 + 1,
                                               statistic="median")
                else:
                    result = paired_cluster_bootstrap(candidate, reference, replicates=REPLICATES,
                                                      seed=SEED + group_index * 100 + metric_index * 2,
                                                      statistic="median", relative=False)
                statistic, pvalue = safe_wilcoxon(candidate, reference)
                delta = candidate.to_numpy(float) - reference.to_numpy(float)
                favorable = delta < 0 if metric == "normalized_center_error" else delta > 0
                bootstrap_rows.append({
                    "analysis": "paired_log_cluster_bootstrap", "role": role, "detector": detector,
                    "delta_ms": int(delta_ms), "yaw_group": yaw_group, "metric": metric,
                    "contrast": "median_paired_log_relative_B5_minus_B3" if relative else "median_log_B5_minus_B3_absolute",
                    "logs": int(len(group)), "estimate": result["estimate"], "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"], "replicates": REPLICATES, "seed": result["seed"],
                    "wilcoxon_statistic": statistic, "wilcoxon_pvalue": pvalue,
                    "logs_B5_better": int(favorable.sum()), "logs_B5_better_percent": float(favorable.mean() * 100),
                })

    # Detector-by-model interaction: difference in paired harm on the same logs.
    for group_index, (keys, group) in enumerate(per_log_frame.groupby(["role", "delta_ms", "yaw_group"], observed=True)):
        role, delta_ms, yaw_group = keys
        for metric_index, metric in enumerate(METRICS):
            pivot = group.pivot(index="log_id", columns="detector", values=f"delta_{metric}").dropna()
            if not {"yolo11n", "yolo11s"}.issubset(pivot.columns) or pivot.empty:
                continue
            result = paired_cluster_bootstrap(pivot.yolo11s, pivot.yolo11n, replicates=REPLICATES,
                                              seed=SEED + 50_000 + group_index * 10 + metric_index,
                                              statistic="median", relative=False)
            statistic, pvalue = safe_wilcoxon(pivot.yolo11s, pivot.yolo11n)
            bootstrap_rows.append({
                "analysis": "detector_by_model_interaction", "role": role, "detector": "yolo11s_minus_yolo11n",
                "delta_ms": int(delta_ms), "yaw_group": yaw_group, "metric": metric,
                "contrast": "(B5-B3)_yolo11s_minus_(B5-B3)_yolo11n", "logs": int(len(pivot)),
                "estimate": result["estimate"], "ci_low": result["ci_low"], "ci_high": result["ci_high"],
                "replicates": REPLICATES, "seed": result["seed"], "wilcoxon_statistic": statistic,
                "wilcoxon_pvalue": pvalue, "logs_B5_better": np.nan, "logs_B5_better_percent": np.nan,
            })
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(OUT / "bootstrap_reversal.csv", index=False, float_format="%.12g")

    primary = bootstrap[(bootstrap.role == "heldout") & (bootstrap.delta_ms == 300) &
                        (bootstrap.yaw_group == "high_yaw") & (bootstrap.metric == "normalized_center_error") &
                        (bootstrap.contrast == "median_paired_log_relative_B5_minus_B3")]
    audit_lines = [
        "# Reversal recalculation audit", "", f"- Source Parquets: {len(files)} corrected propagation files.",
        f"- Independent unit: log.", f"- Bootstrap: {REPLICATES:,} paired log-cluster replicates.",
        f"- High-yaw threshold: {threshold:.15g} rad/s, frozen from development logs.",
        "- Detector inference: not run.", "- Target-time geometry: evaluation metrics only.", "",
        "## Primary 300-ms held-out high-yaw contrast", "",
    ]
    for row in primary.itertuples(index=False):
        audit_lines.append(f"- {row.detector}: B5 disadvantage {row.estimate * 100:.6f}% "
                           f"(95% log-cluster bootstrap CI {row.ci_low * 100:.6f}% to {row.ci_high * 100:.6f}%); "
                           f"B5 better in {int(row.logs_B5_better)}/{int(row.logs)} logs.")
    (OUT / "recalculation_audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    print(primary[["detector", "estimate", "ci_low", "ci_high", "logs", "logs_B5_better"]].to_string(index=False))


if __name__ == "__main__":
    main()
