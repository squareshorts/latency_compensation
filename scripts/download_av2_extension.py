"""Download only the frozen AV2 extension sensor subset."""

from __future__ import annotations

import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\work\auto")
LOG_LIST = ROOT / "configs" / "av2_extension_logs.txt"
DESTINATION = ROOT / "data" / "av2" / "sensor" / "val"
OUTPUT = ROOT / "results" / "sivp_strengthening" / "extension_download_manifest.csv"


def sync_log(log_id: str) -> str:
    destination = DESTINATION / log_id
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "aws", "s3", "sync", f"s3://argoverse/datasets/av2/sensor/val/{log_id}/", str(destination),
        "--no-sign-request", "--only-show-errors", "--exclude", "*",
        "--include", "annotations.feather", "--include", "city_SE3_egovehicle.feather",
        "--include", "calibration/*.feather", "--include", "sensors/cameras/ring_front_center/*",
        "--include", "sensors/lidar/*",
    ], check=True)
    return log_id


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    logs = [line.strip() for line in LOG_LIST.read_text().splitlines() if line.strip()]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(sync_log, log_id) for log_id in logs]
        for index, future in enumerate(as_completed(futures), 1):
            log_id = future.result()
            print(json.dumps({"downloaded": index, "total": len(logs), "log": log_id}), flush=True)
    rows = []
    for log_id in logs:
        root = DESTINATION / log_id
        for path in sorted(root.rglob("*")):
            if path.is_file():
                rows.append({"split": "val", "log_id": log_id, "relative_path": path.relative_to(ROOT).as_posix(),
                             "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(OUTPUT, index=False)
    summary = {"completed_utc": datetime.now(timezone.utc).isoformat(), "logs": len(logs),
               "files": len(manifest), "bytes": int(manifest.bytes.sum()),
               "manifest_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}
    (OUTPUT.parent / "extension_download_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
