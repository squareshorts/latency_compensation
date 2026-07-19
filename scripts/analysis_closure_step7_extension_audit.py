import sys
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\work\auto")
OUTPUT = ROOT / "results" / "sivp_strengthening"
RESULTS = ROOT / "results" / "analysis_closure"
DOCS = ROOT / "docs" / "analysis_closure"

def main():
    manifest_path = OUTPUT / "extension_metadata_freeze.csv"
    if not manifest_path.exists():
        print("Manifest missing!")
        return

    manifest = pd.read_csv(manifest_path)
    manifest_logs = manifest.log_id.unique().tolist()
    num_manifest = len(manifest_logs)

    flow = []
    exclusions = []

    # yolo11n, yolo11s, rtdetr_l
    for det in ["yolo11n", "yolo11s", "rtdetr_l"]:
        if det == "rtdetr_l":
            flow.append({
                "detector": det,
                "manifest_logs": num_manifest,
                "downloaded_logs": num_manifest,
                "detector_complete_logs": 0,
                "propagation_complete_logs": 0,
                "gate_eligible_logs": 0,
                "excluded_logs": num_manifest
            })
            for log in manifest_logs:
                exclusions.append({
                    "detector": det,
                    "log_id": log,
                    "reason": "Intentionally excluded; not completed; zero checkpoints; not to be rerun."
                })
            continue

        cp_dir = OUTPUT / "extension_detector_checkpoints" / det
        prop_dir = OUTPUT / "extension_propagation" / det

        cps = [p.stem for p in cp_dir.glob("*.json")] if cp_dir.exists() else []
        props = [p.stem for p in prop_dir.glob("*.parquet")] if prop_dir.exists() else []

        # Determine gate eligible from causal_gate_per_log
        per_log_path = OUTPUT / "causal_gate_per_log.csv"
        if per_log_path.exists():
            per_log = pd.read_csv(per_log_path)
            gate_logs = per_log[(per_log.detector == det) & (per_log.endpoint == "300ms_high_yaw")].log_id.unique().tolist()
        else:
            gate_logs = []

        flow.append({
            "detector": det,
            "manifest_logs": num_manifest,
            "downloaded_logs": num_manifest,
            "detector_complete_logs": len(cps),
            "propagation_complete_logs": len(props),
            "gate_eligible_logs": len(gate_logs),
            "excluded_logs": num_manifest - len(gate_logs)
        })

        for log in manifest_logs:
            if log not in gate_logs:
                if log not in cps:
                    reason = "Missing detector checkpoint."
                elif log not in props:
                    reason = "Missing propagation output."
                else:
                    reason = "Excluded during gate evaluation (e.g. no eligible comparisons)."
                exclusions.append({
                    "detector": det,
                    "log_id": log,
                    "reason": reason
                })

    pd.DataFrame(flow).to_csv(RESULTS / "extension_cohort_flow.csv", index=False)
    pd.DataFrame(exclusions).to_csv(RESULTS / "extension_exclusions.csv", index=False)

    doc = ["# Extension Cohort Audit\n"]
    for det_flow in flow:
        det = det_flow["detector"]
        doc.append(f"## {det}")
        doc.append(f"- **Manifest Logs:** {det_flow['manifest_logs']}")
        doc.append(f"- **Downloaded Logs:** {det_flow['downloaded_logs']}")
        doc.append(f"- **Detector Complete Logs:** {det_flow['detector_complete_logs']}")
        doc.append(f"- **Propagation Complete Logs:** {det_flow['propagation_complete_logs']}")
        doc.append(f"- **Gate-Eligible Logs:** {det_flow['gate_eligible_logs']}")
        doc.append(f"- **Excluded Logs:** {det_flow['excluded_logs']}")
        if det == "rtdetr_l":
            doc.append("- **Note:** RT-DETR-L extension was intentionally not completed, has zero checkpoints, and is excluded from further extension analysis.")

    (DOCS / "extension_cohort_audit.md").write_text("\n".join(doc), encoding="utf-8")

if __name__ == "__main__":
    main()
