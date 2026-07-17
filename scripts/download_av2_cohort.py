import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\work\auto")
OUT = ROOT / "results" / "av2_confirmation"
MANIFEST = OUT / "download_manifest.csv"


def sync_log(row):
    source = f"s3://argoverse/datasets/av2/sensor/{row.split}/{row.log_id}/"
    destination = ROOT / "data" / "av2" / "sensor" / row.split / row.log_id
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        "aws", "s3", "sync", source, str(destination), "--no-sign-request", "--only-show-errors",
        "--exclude", "*", "--include", "annotations.feather", "--include", "city_SE3_egovehicle.feather",
        "--include", "calibration/*.feather", "--include", "sensors/cameras/ring_front_center/*", "--include", "sensors/lidar/*",
    ]
    subprocess.run(command, check=True)
    return row.log_id


def main():
    manifest = pd.read_csv(MANIFEST)
    logs = manifest[["split", "log_id", "cohort"]].drop_duplicates().sort_values(["split", "log_id"])
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(sync_log, row) for row in logs.itertuples(index=False)]
        for index, future in enumerate(as_completed(futures), start=1):
            future.result()
            if index % 5 == 0 or index == len(futures):
                print(f"downloaded logs {index}/{len(futures)}", flush=True)
    actual_sizes, present = [], []
    for row in manifest.itertuples(index=False):
        path = Path(row.local_path)
        exists = path.is_file()
        size = path.stat().st_size if exists else 0
        present.append(exists and size == row.size_bytes)
        actual_sizes.append(size)
    manifest["actual_size_bytes"] = actual_sizes
    manifest["verified_present"] = present
    temp = MANIFEST.with_name(MANIFEST.name + ".tmp")
    manifest.to_csv(temp, index=False)
    temp.replace(MANIFEST)
    missing = manifest[~manifest.verified_present]
    if len(missing):
        missing.to_csv(OUT / "download_missing.csv", index=False)
        raise SystemExit(f"{len(missing)} manifest objects missing or size-mismatched")
    required = int(manifest.size_bytes.sum())
    audit = f"""# AV2 data presence audit

- Official source: `s3://argoverse/datasets/av2/sensor/`.
- Official API: av2 0.3.6 import tested on Python 3.10.11.
- Fixed seed: 20260717.
- Fixed cohorts: 80 development train logs, 20 model-selection train logs, 50 held-out validation logs.
- Selection was frozen before sensor download using city plus pre-outcome speed, yaw, and traffic strata.
- Selective payload: ring_front_center, lidar, annotations, ego poses, and calibration only.
- Verified objects: {len(manifest)}/{len(manifest)}.
- Verified bytes: {required} ({required / 1024**3:.2f} GiB).
- Payload status: DOWNLOADED AND SIZE-VERIFIED.
- Detector inference and outcome analysis status: NOT STARTED.
"""
    (OUT / "data_presence_audit.md").write_text(audit, encoding="utf-8")
    print(json.dumps({"logs": len(logs), "objects": len(manifest), "bytes": required, "gib": required / 1024**3, "verified": True}, indent=2))


if __name__ == "__main__":
    main()
