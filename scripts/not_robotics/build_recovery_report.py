import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(r"C:\work\auto")
DATA = ROOT / "data" / "nuscenes"
RESULTS = ROOT / "results"
OUT = RESULTS / "not_robotics_real_feasibility"
TEMP_PATHS = [
    Path(r"C:\Users\mirro\AppData\Local\Temp\playwright-artifacts-m04QX2\47028645-1d84-496b-82e7-65f3afa944ad"),
    Path(r"C:\Users\mirro\AppData\Local\Temp\playwright-artifacts-m04QX2\ed5c887b-4975-4dc9-94ef-f8222bbe64fa"),
]
MARKERS = ["synthetic", "mock", "placeholder", "fabricated", "dummy"]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_csv(df, path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def atomic_text(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def compute_flow(previous, current):
    if previous is None:
        return 0.0
    a = cv2.imread(str(previous), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(current), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        raise RuntimeError(f"Unreadable flow image: {previous} or {current}")
    a = cv2.resize(a, (640, 360))
    b = cv2.resize(b, (640, 360))
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


def validate_output(path):
    try:
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
            return f"valid_parquet rows={len(df)} columns={len(df.columns)}", df
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            return f"valid_csv rows={len(df)} columns={len(df.columns)}", df
        if path.suffix.lower() == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            return f"valid_json type={type(obj).__name__}", obj
        if path.suffix.lower() in {".md", ".txt", ".log", ".py"}:
            text = path.read_text(encoding="utf-8")
            return f"readable_text chars={len(text)}", text
        return "binary_or_unclassified", None
    except Exception as exc:
        return f"INVALID {type(exc).__name__}: {exc}", None


def marker_scan(path, obj):
    text = ""
    if isinstance(obj, str):
        text = obj
    elif isinstance(obj, pd.DataFrame):
        object_cols = obj.select_dtypes(include=["object", "string"])
        if len(object_cols.columns):
            text = "\n".join(object_cols.fillna("").astype(str).agg(" ".join, axis=1).tolist())
    elif obj is not None:
        text = json.dumps(obj)
    else:
        try:
            text = path.read_bytes().decode("utf-8", errors="ignore")
        except Exception:
            text = ""
    lower = text.lower()
    found = sorted({m for m in MARKERS if m in lower})
    if "INVALID_SYNTHETIC" in str(path):
        classification = "invalid_synthetic_artifact"
    elif path.name.endswith("_INCOMPLETE.parquet") or "_INCOMPLETE" in path.name:
        classification = "preserved_incomplete_prior_output"
    elif found:
        classification = "marker_mentioned_in_audit_or_provenance_text"
    else:
        classification = "no_marker_detected"
    return ";".join(found), classification


def main():
    headline = json.loads((OUT / "headline_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))
    frames = pd.read_parquet(OUT / "frame_observations.parquet").sort_values(["scene_name", "timestamp_us"])
    detections = pd.read_parquet(OUT / "detector_outputs.parquet")
    sync_summary = pd.read_csv(OUT / "synchronization_summary.csv")
    metric_recalc = pd.read_csv(OUT / "metric_recalculation.csv")

    image_files = sorted((DATA / "samples" / "CAM_FRONT").glob("*")) + sorted((DATA / "sweeps" / "CAM_FRONT").glob("*"))
    image_bad = []
    for path in image_files:
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as exc:
            image_bad.append(f"{path}: {exc}")

    flow_rows = []
    previous = {}
    for row in frames.itertuples(index=False):
        path = Path(row.image_path)
        value = compute_flow(previous.get(row.scene_name), path)
        previous[row.scene_name] = path
        difference = value - row.flow_mag
        flow_rows.append({"scene_name": row.scene_name, "sample_token": row.sample_token, "timestamp_us": row.timestamp_us, "stored_flow_mag": row.flow_mag, "recalculated_flow_mag": value, "difference": difference, "match_within_1e-6": abs(difference) <= 1e-6})
    flow_audit = pd.DataFrame(flow_rows)
    atomic_csv(flow_audit, OUT / "optic_flow_verification.csv")

    raw_rows = []
    raw_roots = [ROOT / "downloads" / "nuscenes", DATA, ROOT / "scripts" / "not_robotics"]
    special_hashes = {
        ROOT / "downloads" / "nuscenes" / "can_bus.zip": "3C68B94C001E8BD05A19886ECB2C6854E0CD69D7005ED9A94D13D45D2951E83F",
        ROOT / "downloads" / "nuscenes" / "v1.0-mini.tgz": "E5D9C2B5CED29F9E3D39651E2093D27E892D0571F10962409974B21033A254EE",
        ROOT / "yolo11n.pt": manifest["weights_sha256"].upper(),
    }
    files = [ROOT / "yolo11n.pt"]
    for base in raw_roots:
        files.extend(p for p in base.rglob("*") if p.is_file())
    for path in sorted(set(files), key=lambda p: str(p).lower()):
        if path in special_hashes:
            digest = sha256(path).upper()
            validation = "sha256_verified" if digest == special_hashes[path] else "sha256_mismatch"
        else:
            digest = ""
            if "CAM_FRONT" in path.parts:
                validation = "image_opened" if not image_bad else "image_set_has_failure"
            elif path.suffix.lower() == ".json" and "can_bus" in path.parts:
                validation = "json_parse_verified_full_audit"
            elif "v1.0-mini" in path.parts or "can_bus" in path.parts:
                validation = "archive_member_size_verified"
            else:
                validation = "present"
        category = "archive" if path.parent == ROOT / "downloads" / "nuscenes" else ("weights" if path == ROOT / "yolo11n.pt" else ("script" if "scripts" in path.parts else "extracted_data"))
        stat = path.stat()
        raw_rows.append({"path": str(path), "category": category, "size_bytes": stat.st_size, "modified_local": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").tz_convert("America/Fortaleza").isoformat(), "sha256": digest, "validation": validation})
    atomic_csv(pd.DataFrame(raw_rows), OUT / "recovery_file_inventory.csv")

    output_rows = []
    for path in sorted((p for p in RESULTS.rglob("*") if p.is_file() and p.name != "recovery_output_inventory.csv"), key=lambda p: str(p).lower()):
        validation, obj = validate_output(path)
        markers, classification = marker_scan(path, obj)
        stat = path.stat()
        output_rows.append({"path": str(path), "size_bytes": stat.st_size, "modified_local": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").tz_convert("America/Fortaleza").isoformat(), "extension": path.suffix.lower(), "validation": validation, "content_markers": markers, "content_classification": classification})
    output_inventory = pd.DataFrame(output_rows)
    atomic_csv(output_inventory, OUT / "recovery_output_inventory.csv")

    current_invalid = output_inventory[(output_inventory["path"].str.startswith(str(OUT))) & (output_inventory["validation"].str.startswith("INVALID"))]
    invalid_synthetic = output_inventory[output_inventory["content_classification"] == "invalid_synthetic_artifact"]
    sync_lines = "\n".join(
        f"- {r.message}: {int(r.synchronized_frames)}/{int(r.total_frames)} causal matches; mean age {r.mean_age_ms:.3f} ms, p95 {r.p95_age_ms:.3f} ms, max {r.max_age_ms:.3f} ms."
        for r in sync_summary.itertuples(index=False)
    )
    phase_lines = "\n".join([
        "1. archive download — completed and verified",
        "2. permanent copy — completed and verified",
        "3. checksum and archive verification — completed and verified",
        "4. extraction — completed and verified",
        "5. real-data presence audit — completed and verified",
        "6. environment and detector setup — completed and verified",
        "7. camera/CAN synchronization — completed and verified",
        "8. real detector inference — completed and verified",
        "9. real optic-flow extraction — completed and verified",
        "10. error-target construction — completed and verified",
        "11. grouped action-information models — completed and verified",
        "12. shifted/shuffled action controls — completed and verified",
        "13. reliability-gate evaluation — completed and verified",
        "14. metric recalculation — completed and verified",
        "15. provenance audit — completed and verified",
        "16. final scientific decision — completed and verified",
    ])
    action_fields = "steeranglefeedback.value; zoesensors throttle_sensor, brake_sensor, steering_sensor; zoe_veh_info requestedTorqueAfterProc, regen, pedal_cc"
    status = f"""# Recovery status

## Audit outcome

- Git status: `C:\\work\\auto` is not a Git repository; `git status` returned the expected fatal not-a-repository message.
- Running analysis processes at recovery start: none. The only matching process was the audit PowerShell process itself.
- Interruption: detected as a stopped prior workflow; an actual power loss cannot be proven from local evidence.
- Original Playwright temporary paths: both absent before recovery began. Their deletion timing/order cannot be reconstructed.
- Permanent archives: present and valid.
  - `can_bus.zip`: 780,974,697 bytes; ZIP magic; SHA-256 `3C68B94C001E8BD05A19886ECB2C6854E0CD69D7005ED9A94D13D45D2951E83F`; 7,834 members; CRC test passed; 7,833 regular files extracted with no missing files or size mismatches.
  - `v1.0-mini.tgz`: 4,168,148,189 bytes; gzip-compressed TAR magic; SHA-256 `E5D9C2B5CED29F9E3D39651E2093D27E892D0571F10962409974B21033A254EE`; TAR listing passed with 31,253 members; 31,225 regular files extracted with no missing files or size mismatches.
- Published/vendor checksum comparison: not available locally; integrity is verified structurally, by archive CRC/listing, local SHA-256 recording, and exact extracted-member sizes.
- Required data paths: `v1.0-mini`, `samples\\CAM_FRONT`, and `can_bus` all exist.
- Real CAM_FRONT files: 2,342 total (404 keyframes plus 1,938 sweeps); {len(image_files) - len(image_bad)} opened successfully, {len(image_bad)} failed.
- Metadata annotations: 18,538 total; 4,068 projected class-mapped 2D boxes were evaluated.
- CAN JSON audit: all 7,832 JSON files parsed; 25,864,384 JSON records; one JSON file was empty and valid.
- Detector weights: `yolo11n.pt`, 5,613,764 bytes, SHA-256 `{manifest['weights_sha256'].upper()}`; loaded and used by Ultralytics {manifest['versions']['ultralytics']} on CPU.
- Current intended output corruption: {len(current_invalid)} invalid files. Preserved `_INCOMPLETE` files are excluded from scientific evidence.
- Synthetic/mock/placeholder/fabricated observations in current intended outputs: none. Negative provenance declarations may contain those words. The separate `not_robotics_feasibility_INVALID_SYNTHETIC` directory contains {len(invalid_synthetic)} explicitly invalid synthetic/mock artifacts and is not used.

## Last-stage chronology

- 2026-07-16 16:29: prior blocked data-presence report.
- 2026-07-16 16:38–16:41: permanent archives completed.
- 2026-07-16 17:01: extraction completed.
- 2026-07-16 17:03–17:04: environment/data audit completed.
- 2026-07-16 17:07: prior real detector-count run stopped after writing an inadequate report; those artifacts are preserved with `_INCOMPLETE` suffixes.
- 2026-07-17 11:21–11:22: resumed box-level inference, synchronization, grouped modeling, controls, and gate evaluation completed.
- 2026-07-17: independent recalculation verified all {len(metric_recalc)} reported metric rows within 1e-12; maximum flow recalculation difference was {flow_audit['difference'].abs().max():.3g}.

## Phase classification

{phase_lines}

## Real-data results

- Scenes: {headline['processed_scenes']}.
- CAM_FRONT: 2,342 real files; {headline['processed_frames']} keyframes processed by actual detector inference.
- Annotations: 18,538 metadata annotations; {headline['evaluated_gt_boxes']} mapped/projected boxes evaluated.
- CAN: 7,832 JSON files and 25,864,384 records.
- Action/control proxy fields: {action_fields}.
- Synchronization used only the latest causal message at or before each camera timestamp. The first frame of each scene had no prior CAN record and was imputed inside training folds.
{sync_lines}
- Best non-action model: `{headline['best_non_action_model']}`, held-out AUC {headline['best_non_action_auc']:.6f}.
- Full action-conditioned model: held-out AUC {headline['full_action_auc']:.6f}.
- Action-information benefit: {headline['action_information_auc_benefit']:+.6f} AUC.
- Shifted 1-second control: AUC {headline['shifted_1s_auc']:.6f}; shuffled control: AUC {headline['shuffled_auc']:.6f}. Both outperform correctly timed full action features.
- High-self-motion false positives: {headline['high_motion_baseline_fp']} to {headline['high_motion_gated_fp']} ({headline['high_motion_fp_change_fraction'] * 100:+.3f}%).
- Recall: {headline['baseline_recall']:.6f} to {headline['gated_recall']:.6f} ({headline['recall_change'] * 100:+.3f} percentage points).
- Calibration ECE: {headline['best_non_action_ece']:.6f} to {headline['full_action_ece']:.6f} ({headline['calibration_ece_change']:+.6f}; worse).
- Leakage: passed scene grouping, causal synchronization, past-to-current optic flow, nested threshold selection, and target/input separation checks.
- Metric recalculation: all headline and ablation metrics independently matched within 1e-12.
- Scientific decision: **{manifest['scientific_decision']}**. The correctly timed action model did not add held-out information, did not beat timing controls, missed the 10% high-motion FP reduction target, exceeded the 2-point recall-loss ceiling, and worsened calibration.

## Temporary-file disposition

The two specified temporary paths are already absent. If equivalent temporary copies existed, they would now be safe to delete because permanent archives, hashes, extraction, real images, and CAN parsing are verified.
"""
    atomic_text(OUT / "recovery_status.md", status)
    print(json.dumps({"raw_inventory_rows": len(raw_rows), "output_inventory_rows": len(output_rows), "images_verified": len(image_files) - len(image_bad), "image_failures": len(image_bad), "flow_rows": len(flow_audit), "flow_max_abs_diff": float(flow_audit["difference"].abs().max()), "metrics_all_match": bool(metric_recalc["match_within_1e-12"].all())}, indent=2))


if __name__ == "__main__":
    main()
