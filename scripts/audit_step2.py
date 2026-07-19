import pandas as pd
from pathlib import Path
import json

ROOT = Path(r"C:\work\auto")
OUTPUT = ROOT / "results" / "sivp_strengthening"

rtdetr_tracker_paths = list((OUTPUT / "tracker_low_level" / "rtdetr_l").glob("*.parquet"))
audit_records = []
for p in rtdetr_tracker_paths:
    try:
        df = pd.read_parquet(p)
        detector = df["detector"].iloc[0] if "detector" in df.columns else "missing"
        models = df["model"].unique().tolist()
        has_b0_b5 = all(m in models for m in ["B0", "B1", "B2", "B3", "B4", "B5"])
        dups = df.duplicated(subset=["comparison_id", "model"]).sum()
        leak = df["future_geometry_input"].astype(bool).sum() + df["future_detection_input"].astype(bool).sum()
        audit_records.append({
            "log": p.stem,
            "detector": detector,
            "valid_parquet": True,
            "has_b0_b5": has_b0_b5,
            "duplicates": dups,
            "leakage": leak,
            "rows": len(df)
        })
    except Exception as e:
        audit_records.append({
            "log": p.stem,
            "detector": "error",
            "valid_parquet": False,
            "has_b0_b5": False,
            "duplicates": 0,
            "leakage": 0,
            "rows": 0
        })

df_rtdetr = pd.DataFrame(audit_records)
df_rtdetr.to_csv(OUTPUT / "rtdetr_completion_audit.csv", index=False)
print("RT-DETR tracker audit:", len(df_rtdetr))

# RT-DETR end-to-end outputs
pred_paths = list((OUTPUT / "end_to_end_low_level" / "heldout" / "rtdetr_l").glob("*.predictions.parquet"))
targ_paths = list((OUTPUT / "end_to_end_low_level" / "heldout" / "rtdetr_l").glob("*.targets.parquet"))
print(f"RT-DETR e2e predictions: {len(pred_paths)}, targets: {len(targ_paths)}")

# YOLO11n Extension
ext_yolo11n_paths = list((OUTPUT / "extension_detector_checkpoints" / "yolo11n").glob("*.json"))
ext_audit = []
for p in ext_yolo11n_paths:
    try:
        with open(p) as fp:
            data = json.load(fp)
        ext_audit.append({
            "log": data.get("log", p.stem),
            "valid_json": True,
            "has_checkpoint": (OUTPUT / "extension_detector_checkpoints" / "yolo11n" / f"{p.stem}.pt").exists()
        })
    except Exception as e:
        ext_audit.append({
            "log": p.stem,
            "valid_json": False,
            "has_checkpoint": False
        })
df_ext = pd.DataFrame(ext_audit)
df_ext.to_csv(OUTPUT / "extension_completion_audit.csv", index=False)
print("YOLO11n Extension audit:", len(df_ext))

# Create extension data integrity markdown
integrity_md = f"""# Extension Data Integrity
- YOLO11n extension json files checked: {len(df_ext)}
- Valid json: {df_ext.valid_json.sum()}
- Corresponding checkpoints found: {df_ext.has_checkpoint.sum()}
"""
(OUTPUT / "extension_data_integrity.md").write_text(integrity_md)

# E. YOLO11s Extension
yolo11s_ckpts = list((OUTPUT / "extension_detector_checkpoints" / "yolo11s").glob("*.pt"))
print("YOLO11s Extension Checkpoints:", len(yolo11s_ckpts))

# F. RT-DETR extension
rtdetr_ext_ckpts = list((OUTPUT / "extension_detector_checkpoints" / "rtdetr_l").glob("*.pt"))
print("RT-DETR Extension Checkpoints:", len(rtdetr_ext_ckpts))
