import os
import hashlib
import pandas as pd
from pathlib import Path
import json

ROOT = Path(r"C:\work\auto")
OUTPUT = ROOT / "results" / "sivp_strengthening"

def hash_file(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "ERROR"

def get_row_count(path):
    if path.suffix == ".csv":
        try:
            return len(pd.read_csv(path))
        except:
            pass
    elif path.suffix == ".parquet":
        try:
            return len(pd.read_parquet(path, columns=[]))
        except:
            pass
    return 0

manifest = []
for p in (ROOT / "results").rglob("*"):
    if p.is_file() and p.suffix in [".parquet", ".csv", ".json", ".pt", ".yaml", ".txt"]:
        rel = p.relative_to(ROOT)
        h = hash_file(p)
        rows = get_row_count(p)
        detector = ""
        if "yolo11n" in p.parts: detector = "yolo11n"
        elif "yolo11s" in p.parts: detector = "yolo11s"
        elif "rtdetr_l" in p.parts: detector = "rtdetr_l"
        manifest.append({
            "path": str(rel),
            "sha256": h,
            "rows": rows,
            "detector": detector,
            "excluded": False
        })

for p in (ROOT / "configs").rglob("*"):
    if p.is_file():
        rel = p.relative_to(ROOT)
        manifest.append({
            "path": str(rel),
            "sha256": hash_file(p),
            "rows": 0,
            "detector": "",
            "excluded": False
        })

df = pd.DataFrame(manifest)
df.to_csv(OUTPUT / "compute_output_manifest.csv", index=False)

md = f"""# SIVP Compute Freeze

- **Commit**: {os.popen("git rev-parse HEAD").read().strip()}
- **Outputs frozen**: {len(manifest)}
- **Manifest**: `results/sivp_strengthening/compute_output_manifest.csv`
- **Exclusions**: Raw AV2 data, uncompleted extension checkpoints (YOLO11s, RT-DETR) are excluded from this analysis.
"""
(ROOT / "docs" / "sivp_compute_freeze.md").write_text(md)
print("Freeze complete.")
