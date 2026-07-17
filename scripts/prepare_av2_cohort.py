import argparse
import json
import math
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\work\auto")
METADATA = ROOT / "data" / "av2" / "metadata"
CONFIGS = ROOT / "configs"
OUT = ROOT / "results" / "av2_confirmation"
BUCKET = "argoverse"
BASE = "datasets/av2/sensor"
SEED = 20260717
PRIMARY = {"development": 80, "model_selection": 20, "heldout": 50}
MINIMUM = {"development": 50, "model_selection": 15, "heldout": 30}
RESERVE_BYTES = 10 * 1024**3


def aws_json(arguments):
    command = ["aws", *arguments, "--no-sign-request", "--output", "json"]
    return json.loads(subprocess.check_output(command, text=True))


def list_logs(split):
    result = aws_json(["s3api", "list-objects-v2", "--bucket", BUCKET, "--prefix", f"{BASE}/{split}/", "--delimiter", "/"])
    return sorted(Path(item["Prefix"].rstrip("/")).name for item in result.get("CommonPrefixes", []))


def copy_object(key, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return
    subprocess.run(["aws", "s3", "cp", f"s3://{BUCKET}/{key}", str(destination), "--no-sign-request", "--only-show-errors"], check=True)


def metadata_for_log(split, log_id):
    local = METADATA / split / log_id
    prefix = f"{BASE}/{split}/{log_id}"
    copy_object(f"{prefix}/annotations.feather", local / "annotations.feather")
    copy_object(f"{prefix}/city_SE3_egovehicle.feather", local / "city_SE3_egovehicle.feather")
    maps = aws_json(["s3api", "list-objects-v2", "--bucket", BUCKET, "--prefix", f"{prefix}/map/"])
    map_keys = [item["Key"] for item in maps.get("Contents", [])]
    city_match = next((re.search(r"____([A-Z]+)_city_", key) for key in map_keys if "log_map_archive" in key), None)
    city = city_match.group(1) if city_match else "UNKNOWN"
    pose = pd.read_feather(local / "city_SE3_egovehicle.feather")
    annotation = pd.read_feather(local / "annotations.feather")
    pose = pose.sort_values("timestamp_ns")
    translation = pose[["tx_m", "ty_m", "tz_m"]].to_numpy(float)
    dt = np.diff(pose.timestamp_ns.to_numpy(np.int64)) / 1e9
    distance = np.linalg.norm(np.diff(translation, axis=0), axis=1)
    speed = distance / np.maximum(dt, 1e-6)
    qw, qx, qy, qz = (pose[name].to_numpy(float) for name in ["qw", "qx", "qy", "qz"])
    yaw = np.unwrap(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2)))
    yaw_rate = np.abs(np.diff(yaw) / np.maximum(dt, 1e-6))
    traffic = annotation.groupby("timestamp_ns").size()
    return {
        "split": split, "log_id": log_id, "city": city,
        "mean_speed_mps": float(np.nanmean(speed)), "p75_yaw_rate_rad_s": float(np.nanquantile(yaw_rate, 0.75)),
        "mean_traffic_objects": float(traffic.mean()), "annotation_rows": len(annotation), "pose_rows": len(pose),
    }


def quantile_bin(series):
    rank = series.rank(method="first")
    return pd.qcut(rank, 4, labels=False, duplicates="drop").astype(int)


def round_robin_select(frame, count, seed):
    work = frame.copy()
    for column in ["mean_speed_mps", "p75_yaw_rate_rad_s", "mean_traffic_objects"]:
        work[column + "_q"] = work.groupby("city", group_keys=False)[column].transform(quantile_bin)
    work["stratum"] = work[["city", "mean_speed_mps_q", "p75_yaw_rate_rad_s_q", "mean_traffic_objects_q"]].astype(str).agg("|".join, axis=1)
    rng = np.random.default_rng(seed)
    groups = []
    for _, group in work.groupby("stratum"):
        order = rng.permutation(len(group))
        groups.append(group.iloc[order].to_dict("records"))
    rng.shuffle(groups)
    selected = []
    while len(selected) < count and any(groups):
        for group in groups:
            if group and len(selected) < count:
                selected.append(group.pop())
    return pd.DataFrame(selected)


def object_manifest(rows):
    def list_selected_objects(row):
        selected = []
        prefix = f"{BASE}/{row.split}/{row.log_id}/"
        response = aws_json(["s3api", "list-objects-v2", "--bucket", BUCKET, "--prefix", prefix])
        for item in response.get("Contents", []):
            key = item["Key"]
            if "/sensors/cameras/ring_front_center/" in key:
                kind = "ring_front_center"
            elif "/sensors/lidar/" in key:
                kind = "lidar"
            elif key.endswith("/annotations.feather"):
                kind = "annotations"
            elif key.endswith("/city_SE3_egovehicle.feather"):
                kind = "poses"
            elif "/calibration/" in key and key.endswith(".feather"):
                kind = "calibration"
            else:
                continue
            relative = key.removeprefix(f"{BASE}/")
            selected.append({"cohort": row.cohort, "split": row.split, "log_id": row.log_id, "city": row.city, "kind": kind, "s3_uri": f"s3://{BUCKET}/{key}", "size_bytes": int(item["Size"]), "local_path": str(ROOT / "data" / "av2" / "sensor" / relative)})
        return selected

    output = []
    records = list(rows.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(list_selected_objects, row) for row in records]
        for index, future in enumerate(as_completed(futures), start=1):
            output.extend(future.result())
            if index % 25 == 0:
                print(f"manifest logs {index}/{len(records)}", flush=True)
    return pd.DataFrame(output)


def write_lines(path, values):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text("\n".join(values) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    CONFIGS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    logs = {split: list_logs(split) for split in ["train", "val"]}
    if len(logs["train"]) != 700 or len(logs["val"]) != 150:
        raise RuntimeError({key: len(value) for key, value in logs.items()})
    tasks = [(split, log_id) for split, ids in logs.items() for log_id in ids]
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(metadata_for_log, split, log_id): (split, log_id) for split, log_id in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            records.append(future.result())
            if index % 50 == 0:
                print(f"metadata {index}/{len(tasks)}", flush=True)
    metadata = pd.DataFrame(records).sort_values(["split", "log_id"])
    metadata.to_csv(OUT / "cohort_source_metadata.csv", index=False)

    train_selected = round_robin_select(metadata[metadata.split == "train"], PRIMARY["development"] + PRIMARY["model_selection"], SEED)
    development = round_robin_select(train_selected, PRIMARY["development"], SEED + 1)
    selection = train_selected[~train_selected.log_id.isin(development.log_id)]
    heldout = round_robin_select(metadata[metadata.split == "val"], PRIMARY["heldout"], SEED + 2)
    development["cohort"], selection["cohort"], heldout["cohort"] = "development", "model_selection", "heldout"
    selected = pd.concat([development, selection, heldout], ignore_index=True)
    manifest = object_manifest(selected)
    required = int(manifest.size_bytes.sum())
    free = shutil.disk_usage(ROOT).free
    cohort_sizes = PRIMARY.copy()
    if required + RESERVE_BYTES > free:
        train_selected = round_robin_select(metadata[metadata.split == "train"], MINIMUM["development"] + MINIMUM["model_selection"], SEED)
        development = round_robin_select(train_selected, MINIMUM["development"], SEED + 1)
        selection = train_selected[~train_selected.log_id.isin(development.log_id)]
        heldout = round_robin_select(metadata[metadata.split == "val"], MINIMUM["heldout"], SEED + 2)
        development["cohort"], selection["cohort"], heldout["cohort"] = "development", "model_selection", "heldout"
        selected = pd.concat([development, selection, heldout], ignore_index=True)
        manifest = object_manifest(selected)
        required = int(manifest.size_bytes.sum())
        cohort_sizes = MINIMUM.copy()
    manifest.to_csv(OUT / "download_manifest.csv", index=False)
    write_lines(CONFIGS / "av2_development_logs.txt", sorted(development.log_id))
    write_lines(CONFIGS / "av2_model_selection_logs.txt", sorted(selection.log_id))
    write_lines(CONFIGS / "av2_heldout_logs.txt", sorted(heldout.log_id))
    audit = f"""# AV2 data presence audit

- Official source: `s3://argoverse/datasets/av2/sensor/`.
- Official API: av2 0.3.6 import tested on Python 3.10.11.
- Candidate metadata audited before outcome analysis: 700 train logs and 150 validation logs.
- Fixed seed: {SEED}.
- Selection uses city plus within-city quartiles of mean ego speed, 75th-percentile yaw magnitude, and mean annotated traffic density.
- Fixed cohorts: {cohort_sizes['development']} development, {cohort_sizes['model_selection']} model selection, {cohort_sizes['heldout']} held-out.
- Required selective objects: ring_front_center images, lidar, annotations, poses, and calibration only.
- Exact required download: {required} bytes ({required / 1024**3:.2f} GiB).
- Free disk at manifest generation: {free} bytes ({free / 1024**3:.2f} GiB); reserved headroom: {RESERVE_BYTES / 1024**3:.0f} GiB.
- Image/lidar payload status: NOT DOWNLOADED by this metadata-only command.
"""
    (OUT / "data_presence_audit.md").write_text(audit, encoding="utf-8")
    print(json.dumps({"candidate_logs": len(metadata), "selected_logs": len(selected), "cohorts": cohort_sizes, "manifest_objects": len(manifest), "required_gib": required / 1024**3, "free_gib": free / 1024**3}, indent=2))


if __name__ == "__main__":
    main()
