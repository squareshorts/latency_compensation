import pandas as pd
import glob
from pathlib import Path
import json
import shutil

ROOT = Path(r"C:\work\auto")
OUTPUT = ROOT / "results" / "sivp_strengthening"

# 1. Deduplicate metric_shard_*.log
log_files = glob.glob(str(OUTPUT / "metric_shard_*.log"))
records = []
for f in log_files:
    with open(f) as fp:
        for line in fp:
            line = line.strip()
            if not line: continue
            data = json.loads(line)
            data["source_file"] = Path(f).name
            records.append(data)

df_log = pd.DataFrame(records)
print(f"Total monitor log lines: {len(df_log)}")

# Keep the latest completion for duplicates if we wanted, or just drop duplicates based on detector and log.
# Since they are JSON log lines from a parallel process, we can drop duplicates on 'detector' and 'log'.
df_dedup = df_log.drop_duplicates(subset=["detector", "log"], keep="last")
print(f"Unique evaluation units from log: {len(df_dedup)}")

# Save audit CSV
df_dedup.to_csv(OUTPUT / "current_detector_completion_audit.csv", index=False)

# 2. Check and deduplicate the shard CSVs just in case they have duplicate outputs
csv_files = glob.glob(str(OUTPUT / "end_to_end_metric_shards" / "shard_*_of_4.csv"))
for f in csv_files:
    df_csv = pd.read_csv(f)
    n_before = len(df_csv)
    # The evaluation units are log_id and detector
    # But a single log has many rows (for different models, delta_ms, yaw, etc.)
    # If the same log_id and detector was processed twice and both appended to the shard,
    # we will have duplicate rows for the exact same configuration.
    df_csv = df_csv.drop_duplicates(keep="last")
    n_after = len(df_csv)
    if n_before != n_after:
        print(f"Deduplicated {f}: {n_before} -> {n_after}")
        df_csv.to_csv(f, index=False)

print("Metric stage deduplication and audit complete.")
