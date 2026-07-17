from pathlib import Path

import pandas as pd

from build_recovery_report import atomic_csv, marker_scan, validate_output

ROOT = Path(r"C:\work\auto")
RESULTS = ROOT / "results"
OUT = RESULTS / "not_robotics_real_feasibility"


def main():
    rows = []
    for path in sorted((p for p in RESULTS.rglob("*") if p.is_file() and p.name != "recovery_output_inventory.csv"), key=lambda p: str(p).lower()):
        validation, obj = validate_output(path)
        markers, classification = marker_scan(path, obj)
        stat = path.stat()
        rows.append({
            "path": str(path),
            "size_bytes": stat.st_size,
            "modified_local": pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").tz_convert("America/Fortaleza").isoformat(),
            "extension": path.suffix.lower(),
            "validation": validation,
            "content_markers": markers,
            "content_classification": classification,
        })
    atomic_csv(pd.DataFrame(rows), OUT / "recovery_output_inventory.csv")
    print(f"Refreshed {len(rows)} output records")


if __name__ == "__main__":
    main()
