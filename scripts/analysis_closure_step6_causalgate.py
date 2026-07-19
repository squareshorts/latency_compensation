import sys
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

ROOT = Path(r"C:\work\auto")
OUTPUT = ROOT / "results" / "sivp_strengthening"
RESULTS = ROOT / "results" / "analysis_closure"
DOCS = ROOT / "docs" / "analysis_closure"

YAW_THRESHOLD = 0.0169726458
SEED = 20260719
REPLICATES = 10_000

def bootstrap(values: np.ndarray, statistic=np.median) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    draws = np.empty(REPLICATES)
    for index in range(REPLICATES):
        draws[index] = statistic(rng.choice(values, len(values), replace=True))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))

def main():
    config = yaml.safe_load((ROOT / "configs" / "causal_gate_frozen.yaml").read_text())

    # Feature audit
    audit = []
    forbidden = ["true_", "target_", "future_", "B3_error", "B5_error", "delta_harm", "object_displacement", "evaluation_track_uuid"]
    for f in config["features"]:
        leakage = False
        for forb in forbidden:
            if forb in f: leakage = True
        audit.append({"feature": f, "status": "FAIL" if leakage else "PASS"})

    pd.DataFrame(audit).to_csv(RESULTS / "causal_gate_feature_audit.csv", index=False)

    # We will reuse causal_gate_per_log.csv if we can, but let's check it.
    per_log_path = OUTPUT / "causal_gate_per_log.csv"
    if not per_log_path.exists():
        print("Missing causal_gate_per_log.csv!")
        return

    per_log = pd.read_csv(per_log_path)

    # Cohort audit
    cohort_audit = []
    for det in ["yolo11n", "yolo11s"]:
        grp = per_log[(per_log.detector == det) & (per_log.endpoint == "300ms_high_yaw")]
        cohort_audit.append({
            "detector": det,
            "manifest_logs": 45,
            "analyzed_logs": len(grp)
        })
    pd.DataFrame(cohort_audit).to_csv(RESULTS / "causal_gate_cohort_audit.csv", index=False)

    contrasts = []
    results = []
    for det in ["yolo11n", "yolo11s"]:
        frame = per_log[(per_log.detector == det) & (per_log.endpoint == "300ms_high_yaw")].copy()
        if frame.empty: continue

        diff = frame.G2_minus_B3_error.to_numpy()
        low, high = bootstrap(diff)
        try:
            wilcoxon_p = float(wilcoxon(diff).pvalue)
        except ValueError:
            wilcoxon_p = 1.0

        b5_frac = frame.selected_B5_percent.median()

        results.append({
            "detector": det,
            "eligible_logs": len(frame),
            "eligible_object_comparisons": int(frame.comparisons.sum()),
            "fraction_assigned_to_B5": b5_frac / 100.0,
            "median_B3_error": frame.B3_error.median(),
            "median_G2_error": frame.G2_error.median(),
            "median_G2_iou": frame.G2_iou.median(),
            "median_G2_recall": frame.G2_recall.median(),
            "percent_logs_improved_over_B3": float((diff < 0).mean() * 100)
        })

        contrasts.append({
            "detector": det,
            "median_G2_minus_B3_error": np.median(diff),
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "wilcoxon_p": wilcoxon_p
        })

    pd.DataFrame(results).to_csv(RESULTS / "causal_gate_recalculated.csv", index=False)
    pd.DataFrame(contrasts).to_csv(RESULTS / "causal_gate_bootstrap_recalculated.csv", index=False)

    doc = ["# Causal Gate Audit\n"]
    for res in results:
        det = res["detector"]
        b5_frac = res["fraction_assigned_to_B5"]
        doc.append(f"## {det}")
        doc.append(f"- **Gate B5 Selection Rate:** {b5_frac:.4f}")
        if b5_frac == 0.0:
            doc.append("- **WARNING: The gate is degenerate relative to B3 (G2 equals G0).**")
            doc.append("- The gate did not identify a prospective operating region.")
            doc.append("- It must NOT be presented as successful.")

    (DOCS / "causal_gate_analysis.md").write_text("\n".join(doc), encoding="utf-8")

if __name__ == "__main__":
    main()
