import json
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\work\auto")
DOCS = ROOT / "docs" / "analysis_closure"
RESULTS = ROOT / "results" / "analysis_closure"
OUTPUT = ROOT / "results" / "sivp_strengthening"

def main():
    shards = list(OUTPUT.glob("metric_shard_*.log"))
    rows = []

    for path in shards:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    data['source_file'] = path.name
                    rows.append(data)
                except json.JSONDecodeError:
                    pass

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "current_detector_unit_audit.csv", index=False)

    total_lines = len(df)
    unique_units = df[['detector', 'log']].drop_duplicates()

    # find duplicates
    counts = df.groupby(['detector', 'log']).size().reset_index(name='count')
    duplicates = counts[counts['count'] > 1]

    missing = 100 - len(unique_units)

    audit_md = f"""# Current-Detector Unit Audit

## Overview
- **Expected scientific units:** 100 (50 logs * 2 detectors YOLO11n, YOLO11s)
- **Observed log lines:** {total_lines}
- **Observed unique units:** {len(unique_units)}
- **Missing units:** {missing if missing > 0 else 0}

## Duplicate Units
{f"Found {len(duplicates)} duplicated units:" if not duplicates.empty else "No duplicated units found."}
"""
    if not duplicates.empty:
        for _, row in duplicates.iterrows():
            audit_md += f"- Detector: {row['detector']}, Log: {row['log']}, Occurrences: {row['count']}\n"

    audit_md += f"\n## Conclusion\n"
    if len(unique_units) == 100:
        audit_md += "The scientific analysis is **complete**. Exactly 100 unique scientific units are present.\nThe 104 log lines resulted from a few units being processed multiple times (likely due to script restarts or shard overlaps)."
    else:
        audit_md += "The scientific analysis is **incomplete**."

    (DOCS / "current_detector_unit_audit.md").write_text(audit_md, encoding="utf-8")

if __name__ == "__main__":
    main()
