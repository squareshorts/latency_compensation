from pathlib import Path

import numpy as np
import pandas as pd


def load_nuscenes_reproduction_rows(path: str | Path) -> pd.DataFrame:
    rows = pd.read_parquet(path)
    required = {"scene_name", "instance_token", "source_timestamp_us", "model", "source_box_json", "target_annotation_box_json"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"nuScenes reproduction rows missing columns: {sorted(missing)}")
    return rows


def complete_scene_groups(rows: pd.DataFrame) -> list[str]:
    return sorted(rows["scene_name"].dropna().unique().tolist())


def deterministic_log_split(log_ids, development: int, selection: int, heldout: int, seed: int = 20260717):
    ordered = np.asarray(sorted(set(log_ids)), dtype=object)
    if len(ordered) < development + selection + heldout:
        raise ValueError("Insufficient unique logs for requested split")
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    chosen = ordered[permutation]
    return {
        "development": chosen[:development].tolist(),
        "model_selection": chosen[development:development + selection].tolist(),
        "heldout": chosen[development + selection:development + selection + heldout].tolist(),
    }
