import sys
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(r"C:\work\auto")
OUTPUT = ROOT / "results" / "sivp_strengthening"
RESULTS = ROOT / "results" / "analysis_closure"

def main():
    rows = []

    def add_row(claim_id, detector, cohort, endpoint, reported, recalculated, source_file, notes=""):
        try:
            r1, r2 = float(reported), float(recalculated)
            abs_diff = abs(r1 - r2)
            rel_diff = abs_diff / max(abs(r1), 1e-9)
            passed = abs_diff < 1e-4
        except:
            abs_diff = np.nan
            rel_diff = np.nan
            passed = reported == recalculated

        rows.append({
            "claim_id": claim_id, "detector": detector, "cohort": cohort, "endpoint": endpoint,
            "reported_value": reported, "recalculated_value": recalculated,
            "absolute_difference": abs_diff, "relative_difference": rel_diff,
            "tolerance": 1e-4, "pass": passed, "source_file": source_file, "source_rows": "N/A", "notes": notes
        })

    # 1. Tracker Bootstraps (B3 vs B5 decline & Tracker vs B3)
    orig_tracker = pd.read_csv(OUTPUT / "tracker_bootstrap_intervals.csv")
    new_tracker = pd.read_csv(RESULTS / "tracker_contrasts_recalculated.csv")

    for det in ["yolo11n", "yolo11s", "rtdetr_l"]:
        # B3 vs B5
        orig = orig_tracker[(orig_tracker.detector == det) & (orig_tracker.candidate == "T6") & (orig_tracker.reference == "T5") & (orig_tracker.delta_ms == 300) & (orig_tracker.yaw_stratum == "high_yaw")]
        new = new_tracker[(new_tracker.detector == det) & (new_tracker.candidate == "T6") & (new_tracker.reference == "T5") & (new_tracker.delta_ms == 300) & (new_tracker.yaw_stratum == "high_yaw")]
        if not orig.empty and not new.empty:
            add_row("B3_vs_B5_decline_relative", det, "heldout", "300ms_high_yaw", orig.iloc[0].paired_relative_difference, new.iloc[0].paired_relative_difference, "tracker_bootstrap_intervals.csv")

    # 2. Mechanisms
    orig_mech = pd.read_csv(OUTPUT / "tracker_failure_mechanisms.csv")
    new_mech = pd.read_csv(RESULTS / "failure_mechanism_by_detector.csv")
    for det in ["yolo11n", "yolo11s"]:
        try:
            # The orig mech doesn't have proportions easily accessible like this.
            # But the recalculated ones have velocity_sign_error proportion
            new_val = new_mech[new_mech.detector == det]["velocity_sign_error"].iloc[0]
            add_row("velocity_sign_error_fraction", det, "heldout", "all", "N/A", new_val, "failure_mechanism_by_detector.csv", "Original proportion not directly in CSV, just verifying recalculated")
        except:
            pass

    # 3. Causal gate
    orig_gate = pd.read_csv(OUTPUT / "causal_gate_results.csv")
    new_gate = pd.read_csv(RESULTS / "causal_gate_recalculated.csv")
    for det in ["yolo11n", "yolo11s"]:
        orig = orig_gate[(orig_gate.detector == det) & (orig_gate.endpoint == "300ms_high_yaw")]
        new = new_gate[(new_gate.detector == det)]
        if not orig.empty and not new.empty:
            add_row("gate_b5_selection_rate", det, "extension", "300ms_high_yaw", orig.iloc[0].selected_B5_percent / 100.0, new.iloc[0].fraction_assigned_to_B5, "causal_gate_results.csv")

    pd.DataFrame(rows).to_csv(RESULTS / "headline_recalculation.csv", index=False)

if __name__ == "__main__":
    main()
