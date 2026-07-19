"""Recover causal yaw metadata for the original AV2 comparison rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


ROOT = Path(r"C:\work\auto")
OUT = ROOT / "results" / "av2_confirmation"
DATA = ROOT / "data" / "av2" / "sensor"


def wrap(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def main() -> None:
    pairs = pd.read_parquet(OUT / "latency_pairs.parquet")
    pairs["source_image_timestamp_ns"] = pairs["source_image_path"].map(lambda value: int(Path(value).stem))
    pairs["target_image_timestamp_ns"] = pairs["target_image_path"].map(lambda value: int(Path(value).stem))
    pair_frames = []
    for (source_split, log_id), group in pairs.groupby(["source_split", "log_id"], sort=True):
        poses = pd.read_feather(DATA / source_split / log_id / "city_SE3_egovehicle.feather").set_index("timestamp_ns")
        needed = np.unique(np.r_[group.source_image_timestamp_ns.to_numpy(), group.target_image_timestamp_ns.to_numpy()])
        selected = poses.reindex(needed)
        valid = selected[["qw", "qx", "qy", "qz"]].notna().all(axis=1)
        yaw_map = {}
        xyz_map = {}
        if valid.any():
            q = selected.loc[valid, ["qx", "qy", "qz", "qw"]].to_numpy(float)
            yaw = Rotation.from_quat(q).as_euler("xyz")[:, 2]
            exact_timestamps = [int(value) for value in selected.index[valid].to_numpy(dtype=np.int64)]
            yaw_map = dict(zip(exact_timestamps, yaw))
            xyz_map = dict(zip(exact_timestamps, selected.loc[valid, ["tx_m", "ty_m", "tz_m"]].to_numpy(float)))
        local = group[["split", "source_split", "log_id", "delta_ms", "source_image_timestamp_ns", "target_image_timestamp_ns"]].copy()
        # Series.map may coerce these 18-digit nanosecond dictionary keys to
        # float64 on some pandas versions, losing integer precision.  Explicit
        # Python-int lookup preserves the timestamp exactly.
        source_yaw = pd.Series([yaw_map.get(int(value), np.nan) for value in local.source_image_timestamp_ns], index=local.index)
        target_yaw = pd.Series([yaw_map.get(int(value), np.nan) for value in local.target_image_timestamp_ns], index=local.index)
        local["yaw_change_rad"] = wrap(target_yaw - source_yaw)
        local["yaw_rate_rad_s"] = local.yaw_change_rad / (local.delta_ms / 1000.0)
        local["abs_yaw_rate_rad_s"] = local.yaw_rate_rad_s.abs()
        local["ego_translation_m"] = [
            float(np.linalg.norm(xyz_map[t] - xyz_map[s])) if s in xyz_map and t in xyz_map else np.nan
            for s, t in zip(local.source_image_timestamp_ns, local.target_image_timestamp_ns)
        ]
        local["yaw_recovered"] = local.yaw_change_rad.notna()
        pair_frames.append(local)
    frame_yaw = pd.concat(pair_frames, ignore_index=True)
    threshold_rows = frame_yaw[(frame_yaw["split"] == "development") & (frame_yaw.delta_ms == 300) & frame_yaw.yaw_recovered]
    threshold = float(threshold_rows.abs_yaw_rate_rad_s.quantile(0.75))
    frame_yaw["high_yaw"] = frame_yaw.abs_yaw_rate_rad_s >= threshold

    joined_rows = []
    lookup = frame_yaw.set_index(["log_id", "delta_ms", "source_image_timestamp_ns"])
    for path in sorted((OUT / "checkpoints").glob("*/*/*.propagation.parquet")):
        detector, role = path.parts[-3], path.parts[-2]
        metrics = pd.read_parquet(path)
        comparisons = metrics.iloc[::6][["log_id", "source_timestamp_ns", "delta_ms"]].reset_index(drop=True)
        comparisons["comparison_ordinal_in_log"] = np.arange(len(comparisons), dtype=np.int64)
        comparisons["detector"] = detector
        comparisons["role"] = role
        comparisons = comparisons.merge(
            frame_yaw[["log_id", "delta_ms", "source_image_timestamp_ns", "target_image_timestamp_ns", "yaw_change_rad", "yaw_rate_rad_s", "abs_yaw_rate_rad_s", "ego_translation_m", "yaw_recovered", "high_yaw"]],
            left_on=["log_id", "delta_ms", "source_timestamp_ns"],
            right_on=["log_id", "delta_ms", "source_image_timestamp_ns"], how="left", validate="many_to_one",
        ).drop(columns="source_image_timestamp_ns")
        comparisons["yaw_recovered"] = comparisons.yaw_recovered.fillna(False)
        joined_rows.append(comparisons)
    recovered = pd.concat(joined_rows, ignore_index=True)
    recovered.to_parquet(OUT / "yaw_reconstruction.parquet", index=False)
    audit = recovered.groupby(["detector", "role", "delta_ms"], as_index=False).agg(
        comparison_rows=("log_id", "size"), recovered_rows=("yaw_recovered", "sum"), high_yaw_comparison_rows=("high_yaw", "sum")
    )
    audit["recovered_percent"] = audit.recovered_rows / audit.comparison_rows * 100.0
    audit["equivalent_metric_rows"] = audit.comparison_rows * 6
    audit["equivalent_recovered_metric_rows"] = audit.recovered_rows * 6
    audit.to_csv(OUT / "yaw_join_audit.csv", index=False, float_format="%.12g")
    payload = {
        "method": "absolute ego yaw rate from official AV2 city_SE3_egovehicle poses",
        "development_only": True, "latency_ms": 300,
        "quantile": 0.75, "threshold_abs_yaw_rate_rad_s": threshold,
        "development_unique_frame_pairs": int(len(threshold_rows)),
        "comparison_rows": int(len(recovered)), "comparison_rows_recovered": int(recovered.yaw_recovered.sum()),
        "equivalent_metric_rows": int(len(recovered) * 6),
        "equivalent_metric_rows_recovered": int(recovered.yaw_recovered.sum() * 6),
        "recovered_percent": float(recovered.yaw_recovered.mean() * 100.0),
    }
    (OUT / "high_yaw_threshold.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
