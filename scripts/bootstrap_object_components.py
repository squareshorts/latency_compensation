"""Add paired log-cluster intervals for primary component substitutions."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(r"C:\work\auto")
sys.path.insert(0, str(ROOT / "src"))
from latency_compensation.object_motion_harm import cluster_bootstrap

OUT = ROOT / "results" / "object_motion_harm"
CONTRASTS = [("M2", "M1", "association"), ("M3", "M2", "detector_history_boxes"),
             ("M4", "M3", "2D_vs_3D_history"), ("M5", "M1", "source_depth"),
             ("M6", "M0", "object_motion_in_principle"), ("M7", "M0", "full_oracle_ceiling")]


def main() -> None:
    logs = pd.read_csv(OUT / "component_log_rows.csv") if (OUT / "component_log_rows.csv").exists() else pd.concat(
        [pd.read_csv(path) for path in sorted(OUT.glob("component_log_rows.shard-*.csv"))], ignore_index=True
    )
    rows = []
    for detector, group in logs[(logs.role == "heldout") & (logs.delta_ms == 300) & (logs.high_yaw.astype(str).str.lower() == "true")].groupby("detector"):
        pivot = group.pivot(index="log_id", columns="model", values="normalized_center_error")
        for index, (candidate, reference, label) in enumerate(CONTRASTS):
            relative = pivot[candidate] / pivot[reference] - 1
            boot = cluster_bootstrap(relative, replicates=10_000, seed=20260718 + index, statistic="median")
            test = wilcoxon(relative, alternative="two-sided", method="approx")
            rows.append({"analysis": "component_paired_log_cluster_bootstrap", "role": "heldout", "detector": detector,
                         "delta_ms": 300, "yaw_group": "high_yaw", "metric": "normalized_center_error",
                         "contrast": f"median_paired_log_relative_{candidate}_minus_{reference}", "component": label,
                         "logs": len(relative), "estimate": boot["estimate"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                         "replicates": 10_000, "seed": boot["seed"], "wilcoxon_statistic": test.statistic,
                         "wilcoxon_pvalue": test.pvalue, "logs_candidate_better": int((relative < 0).sum()),
                         "logs_candidate_better_percent": float((relative < 0).mean()*100)})
    existing = pd.read_csv(OUT / "bootstrap_reversal.csv")
    existing = existing[existing.analysis != "component_paired_log_cluster_bootstrap"]
    pd.concat([existing, pd.DataFrame(rows)], ignore_index=True, sort=False).to_csv(OUT / "bootstrap_reversal.csv", index=False, float_format="%.12g")


if __name__ == "__main__":
    main()
