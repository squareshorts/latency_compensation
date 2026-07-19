"""Freeze the untouched metadata-only AV2 extension cohort."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\work\auto")
METADATA = ROOT / "results" / "av2_confirmation" / "cohort_source_metadata.csv"
OUTPUT_LIST = ROOT / "configs" / "av2_extension_logs.txt"
OUTPUT_METADATA = ROOT / "results" / "sivp_strengthening" / "extension_metadata_freeze.csv"
OUTPUT_MANIFEST = ROOT / "results" / "sivp_strengthening" / "extension_freeze_manifest.json"
SEED = 20260719
COUNT = 45


def quantile_bin(series: pd.Series) -> pd.Series:
    return pd.qcut(series.rank(method="first"), 4, labels=False, duplicates="drop").astype(int)


def round_robin_select(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    for column in ("mean_speed_mps", "p75_yaw_rate_rad_s", "mean_traffic_objects"):
        work[column + "_q"] = work.groupby("city", group_keys=False)[column].transform(quantile_bin)
    work["metadata_stratum"] = work[["city", "mean_speed_mps_q", "p75_yaw_rate_rad_s_q", "mean_traffic_objects_q"]].astype(str).agg("|".join, axis=1)
    rng = np.random.default_rng(SEED)
    groups = []
    for _, group in work.groupby("metadata_stratum", sort=True):
        groups.append(group.iloc[rng.permutation(len(group))].to_dict("records"))
    rng.shuffle(groups)
    selected = []
    while len(selected) < COUNT and any(groups):
        for group in groups:
            if group and len(selected) < COUNT:
                selected.append(group.pop())
    return pd.DataFrame(selected).sort_values("log_id").reset_index(drop=True)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    metadata = pd.read_csv(METADATA)
    original = set()
    for filename in ("av2_development_logs.txt", "av2_model_selection_logs.txt", "av2_heldout_logs.txt"):
        original.update(line.strip() for line in (ROOT / "configs" / filename).read_text().splitlines() if line.strip())
    candidates = metadata[(metadata.split == "val") & ~metadata.log_id.astype(str).isin(original)].copy()
    selected = round_robin_select(candidates)
    if len(selected) != COUNT or selected.log_id.astype(str).isin(original).any():
        raise RuntimeError("Extension selection failed disjointness or size check")
    OUTPUT_LIST.write_text("\n".join(selected.log_id.astype(str)) + "\n", encoding="utf-8")
    OUTPUT_METADATA.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUTPUT_METADATA, index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED, "count": COUNT,
        "candidate_pool": "AV2 validation logs excluding all original 150 logs",
        "selection_inputs": ["city", "mean_speed_mps", "p75_yaw_rate_rad_s", "mean_traffic_objects"],
        "outcome_inputs": [], "log_list_sha256": sha256(OUTPUT_LIST),
        "metadata_freeze_sha256": sha256(OUTPUT_METADATA), "disjoint_from_original_150": True,
        "outcomes_inspected_before_freeze": False,
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
