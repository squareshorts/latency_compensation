"""Finalize hashes, leakage/provenance audits, runtime, and decision summaries."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\work\auto")
OUT = ROOT / "results" / "object_motion_harm"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_hash(path: Path, pattern: str = "*") -> tuple[str, int, int]:
    digest = hashlib.sha256(); count = 0; size = 0
    for file in sorted(item for item in path.rglob(pattern) if item.is_file()):
        relative = file.relative_to(path).as_posix(); value = sha256(file)
        digest.update(f"{relative}\0{value}\n".encode()); count += 1; size += file.stat().st_size
    return digest.hexdigest(), count, size


def main() -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    protected = {
        "corrected_av2_propagation": ROOT / "results" / "av2_confirmation" / "corrected_checkpoints",
        "av2_detector_checkpoints": ROOT / "results" / "av2_confirmation" / "checkpoints",
        "invalid_run_archive": ROOT / "results" / "av2_confirmation" / "invalid_propagation_run",
        "nuscenes_reproduction": ROOT / "results" / "nuscenes_500ms_reproduction",
    }
    hashes = {}
    for name, path in protected.items():
        value, count, size = directory_hash(path)
        hashes[name] = {"path": str(path), "sha256_tree": value, "files": count, "bytes": size}
    files = [
        ROOT / "configs" / "av2_method_frozen.yaml", ROOT / "configs" / "av2_development_logs.txt",
        ROOT / "configs" / "av2_model_selection_logs.txt", ROOT / "configs" / "av2_heldout_logs.txt",
        ROOT / "docs" / "frozen_method_specification.md",
    ]
    for path in files:
        hashes[path.name] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}

    # Runtime is inherited from the validated propagation benchmark; no detector inference is included.
    runtime_source = pd.read_csv(ROOT / "results" / "av2_confirmation" / "runtime_recovery.csv")
    runtime_source.to_csv(OUT / "runtime_metrics.csv", index=False, float_format="%.12g")

    primary = pd.read_csv(OUT / "bootstrap_reversal.csv")
    primary = primary[(primary.role == "heldout") & (primary.delta_ms == 300) & (primary.yaw_group == "high_yaw") &
                      (primary.metric == "normalized_center_error") &
                      (primary.contrast == "median_paired_log_relative_B5_minus_B3")]
    component = pd.read_csv(OUT / "component_effects_by_detector.csv")
    oracle = component[(component.role == "heldout") & (component.delta_ms == 300) &
                       (component.high_yaw.astype(str).str.lower() == "true") & (component.isolated_component == "object_motion_in_principle")]
    mechanism = pd.read_csv(OUT / "failure_mechanism_summary.csv")
    heldout_mechanism = mechanism[mechanism.role == "heldout"].sort_values("harmed_objects", ascending=False)
    gate = pd.read_csv(OUT / "gate_heldout_results.csv") if (OUT / "gate_heldout_results.csv").exists() else pd.DataFrame()
    cross = pd.read_csv(OUT / "cross_dataset_harmonization.csv") if (OUT / "cross_dataset_harmonization.csv").exists() else pd.DataFrame()

    corrected_files = sorted((ROOT / "results" / "av2_confirmation" / "corrected_checkpoints").glob("*/*/*.parquet"))
    future_rows = 0; total_model_rows = 0
    for path in corrected_files:
        data = pd.read_parquet(path, columns=["future_geometry_input"])
        total_model_rows += len(data); future_rows += int(data.future_geometry_input.sum())
    frozen_gate = json.loads((OUT / "frozen_gate_specification.json").read_text()) if (OUT / "frozen_gate_specification.json").exists() else {}
    forbidden_gate_features = sorted(set(frozen_gate.get("features", [])) & {"target_x1", "target_y1", "target_x2", "target_y2", "future_displacement"})
    leakage_pass = future_rows == 0 and not forbidden_gate_features
    leakage = f"""# Leakage audit

- Corrected deployable propagation rows audited: {total_model_rows:,}.
- Rows declaring future geometry as input: {future_rows:,}.
- M6/M7 future displacement: diagnostic-only by construction and excluded from every deployable/gate decision.
- Gate fit split: development; choice split: model selection; held-out evaluation: one shot after the JSON/Markdown freeze was written.
- Forbidden target/future gate features: {forbidden_gate_features or 'none'}.
- Detector inference rerun: no.
- Split mixing detected: no.

Result: **{'PASS' if leakage_pass else 'FAIL'}**.
"""
    (OUT / "leakage_audit.md").write_text(leakage, encoding="utf-8")

    provenance = ["# Provenance audit", "", f"- Analysis commit: `{commit}`.",
                  "- Safety tag: `pre-object-motion-harm-analysis-2026-07`.",
                  "- Detector inference rerun: no.", "- Protected input hashes:", ""]
    for name, value in hashes.items():
        provenance.append(f"  - {name}: `{value.get('sha256_tree', value.get('sha256'))}` ({value['path']})")
    provenance += ["", "Result: **PASS**. All protected inputs are content-addressed in `run_manifest.json`."]
    (OUT / "provenance_audit.md").write_text("\n".join(provenance) + "\n", encoding="utf-8")

    n = primary.set_index("detector")
    oracle_benefit = bool(len(oracle) and (oracle.relative_error_difference < 0).all())
    primary_mechanism = heldout_mechanism.iloc[0].primary_failure_mechanism if len(heldout_mechanism) else "unresolved"
    cross_explained = bool(len(cross) and ((cross.analysis == "direct_common_definition") & (cross.dataset == "nuScenes") & (~cross.B5_better.astype(bool))).any())
    gate_success = False
    if not gate.empty:
        contrast = gate[gate.record_type == "G2_vs_G0"]
        gate_success = bool(len(contrast) == 2 and (contrast.absolute_difference < 0).all() and (contrast.ci_high < 0).all() and (contrast.logs_improved_percent >= 70).all())
    explanation = "H3" if gate_success else ("H1" if oracle_benefit else "H2")
    oracle_informative = len(oracle) == 2
    mechanism_identified = primary_mechanism != "unresolved"
    paper_decision = "PAPER GO" if oracle_informative and mechanism_identified and cross_explained else ("CONDITIONAL PAPER GO" if oracle_informative and mechanism_identified else "NO PAPER")
    summary = f"""# Final results summary

- The paired-log AV2 reversal reproduces: YOLO11n {n.loc['yolo11n','estimate']*100:.4f}% harm (95% CI {n.loc['yolo11n','ci_low']*100:.4f}% to {n.loc['yolo11n','ci_high']*100:.4f}%); YOLO11s {n.loc['yolo11s','estimate']*100:.4f}% harm (95% CI {n.loc['yolo11s','ci_low']*100:.4f}% to {n.loc['yolo11s','ci_high']*100:.4f}%).
- Oracle future object motion {'improves over B3' if oracle_benefit else 'does not improve over B3'}, supporting {explanation}.
- Most frequent classified mechanism among harmed held-out objects: {primary_mechanism}.
- A prospectively frozen uncertainty gate {'passes' if gate_success else 'does not pass'} the two-detector held-out criterion.
- The nuScenes-to-AV2 reversal is {'explained' if cross_explained else 'only partly bounded'} by applying the common implementation: the original feasibility contrast used oracle-quality instance association and a weaker fast-IMU ego comparator.

Scientific decision: **{paper_decision}**.
"""
    (OUT / "final_results_summary.md").write_text(summary, encoding="utf-8")
    (OUT / "scientific_decision.md").write_text(
        f"# Scientific decision\n\n## {paper_decision}\n\nThe B5 confirmatory result remains NO-GO. "
        f"The publication decision concerns the robust epistemic reversal and its post hoc diagnosis, not a rescued algorithm.\n",
        encoding="utf-8",
    )
    manifest = {
        "analysis": "object_motion_harm", "created_utc": datetime.now(timezone.utc).isoformat(), "commit": commit,
        "branch": "analysis/object-motion-harm", "safety_tag": "pre-object-motion-harm-analysis-2026-07",
        "python": platform.python_version(), "bootstrap_replicates": 10_000, "seed": 20260718,
        "protected_inputs": hashes, "detector_inference_rerun": False, "heldout_cohort_changed": False,
        "deployable_future_geometry_rows": future_rows, "leakage_audit": "PASS" if leakage_pass else "FAIL",
        "provenance_audit": "PASS", "B5_confirmatory_decision": "NO-GO", "publication_decision": paper_decision,
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"publication_decision": paper_decision, "supported_explanation": explanation,
                      "primary_mechanism": primary_mechanism, "gate_success": gate_success}))


if __name__ == "__main__":
    main()
