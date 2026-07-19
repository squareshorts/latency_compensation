import sys
import hashlib
from pathlib import Path

ROOT = Path(r"C:\work\auto")
RESULTS = ROOT / "results" / "analysis_closure"
DOCS = ROOT / "docs" / "analysis_closure"

def get_hash(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()

def main():
    manifest = []

    # 1. Gather all files
    for p in RESULTS.rglob("*.*"):
        if p.is_file():
            manifest.append(f"{p.relative_to(ROOT)}, {get_hash(p)}")

    for p in DOCS.rglob("*.*"):
        if p.is_file():
            manifest.append(f"{p.relative_to(ROOT)}, {get_hash(p)}")

    manifest.sort()

    # Write manifest
    (ROOT / "analysis_manifest.txt").write_text("\n".join(manifest), encoding="utf-8")

if __name__ == "__main__":
    main()
