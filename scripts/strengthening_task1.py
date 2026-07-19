import json
import hashlib
from pathlib import Path

def main():
    root = Path(r"c:\work\latency_compensation")
    out_dir = root / "results" / "manuscript_strengthening_20260719"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_hash = [
        "configs/av2_method_frozen.yaml",
        "configs/av2_heldout_logs.txt",
        "yolo11n.pt",
        "yolo11s.pt",
        "rtdetr-l.pt",
        "results/object_motion_harm/object_level_diagnostics.parquet"
    ]
    
    manifest = {"files": {}}
    recon_md = ["# Reconnaissance\n"]
    
    for f in files_to_hash:
        p = root / f
        if p.exists():
            h = hashlib.sha256(p.read_bytes()).hexdigest() if p.stat().st_size < 100_000_000 else "SKIPPED (TOO LARGE)"
            manifest["files"][f] = h
            recon_md.append(f"- **{f}**: `{h}`")
        else:
            manifest["files"][f] = "MISSING"
            recon_md.append(f"- **{f}**: MISSING")
            
    with open(out_dir / "input_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    with open(out_dir / "reconnaissance.md", "w") as f:
        f.write("\n".join(recon_md))

if __name__ == "__main__":
    main()
