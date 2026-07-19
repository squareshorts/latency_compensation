"""Consolidate independently written component-analysis shards."""

from __future__ import annotations

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(r"C:\work\auto")
OUT = ROOT / "results" / "object_motion_harm"
MODELS = ("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7")
METRICS = ("normalized_center_error", "iou", "recall_iou_0_3", "recall_iou_0_5", "recall_iou_0_7")


def main() -> None:
    parquet_shards = sorted(OUT.glob("object_level_diagnostics.shard-*.parquet"))
    if not parquet_shards:
        raise RuntimeError("No object-level shards found")
    destination = OUT / "object_level_diagnostics.parquet"
    writer = None
    for path in parquet_shards:
        source = pq.ParquetFile(path)
        for batch in source.iter_batches(batch_size=100_000):
            table = batch.to_table() if hasattr(batch, "to_table") else __import__("pyarrow").Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
    if writer:
        writer.close()

    component_logs = pd.concat([pd.read_csv(path) for path in sorted(OUT.glob("component_log_rows.shard-*.csv"))], ignore_index=True)
    component_logs = component_logs.drop_duplicates(["role", "detector", "log_id", "delta_ms", "high_yaw", "model"])
    aggregate = []
    for keys, group in component_logs.groupby(["role", "detector", "delta_ms", "high_yaw", "model"], observed=True):
        record = dict(zip(["role", "detector", "delta_ms", "high_yaw", "model"], keys))
        record["logs"] = group.log_id.nunique(); record["comparisons"] = group.comparisons.sum()
        for metric in METRICS:
            record[f"median_of_log_medians_{metric}"] = float(group[metric].median())
        aggregate.append(record)
    component = pd.DataFrame(aggregate).sort_values(["role", "detector", "delta_ms", "high_yaw", "model"])
    component.to_csv(OUT / "component_substitution.csv", index=False, float_format="%.12g")
    component_logs[component_logs.model.isin(["M0", "M1", "M4", "M6", "M7"])].to_csv(
        OUT / "oracle_diagnostics.csv", index=False, float_format="%.12g"
    )

    effects = []
    for keys, group in component.groupby(["role", "detector", "delta_ms", "high_yaw"], observed=True):
        indexed = group.set_index("model")
        for candidate, reference, label in [
            ("M2", "M1", "association"), ("M3", "M2", "detector_history_boxes"),
            ("M4", "M3", "2D_vs_3D_history"), ("M5", "M1", "source_depth"),
            ("M6", "M0", "object_motion_in_principle"), ("M7", "M0", "full_oracle_ceiling"),
        ]:
            c = indexed.loc[candidate, "median_of_log_medians_normalized_center_error"]
            r = indexed.loc[reference, "median_of_log_medians_normalized_center_error"]
            effects.append({**dict(zip(["role", "detector", "delta_ms", "high_yaw"], keys)), "candidate": candidate,
                            "reference": reference, "isolated_component": label, "absolute_error_difference": c-r,
                            "relative_error_difference": c/r-1})
    pd.DataFrame(effects).to_csv(OUT / "component_effects_by_detector.csv", index=False, float_format="%.12g")

    interactions = pd.concat([pd.read_csv(path) for path in sorted(OUT.glob("component_interactions.shard-*.csv"))], ignore_index=True)
    interactions["record_type"] = "factorial_cell"
    factor_columns = ["association_oracle", "depth_oracle", "velocity_oracle", "ego_motion_transform"]
    effects_rows = []
    for keys, group in interactions.groupby(["role", "detector"], observed=True):
        cells = group.groupby(factor_columns, observed=True).median(numeric_only=True).reset_index()
        for order in range(1, len(factor_columns) + 1):
            for selected in combinations(factor_columns, order):
                sign = np.prod([cells[column].to_numpy() * 2 - 1 for column in selected], axis=0)
                effects_rows.append({"record_type": "factorial_effect", "role": keys[0], "detector": keys[1],
                                     "effect": ":".join(selected),
                                     "effect_coefficient_normalized_center_error": float(np.mean(sign * cells.median_normalized_center_error))})
    interactions = pd.concat([interactions, pd.DataFrame(effects_rows)], ignore_index=True, sort=False)
    interactions.to_csv(OUT / "component_interactions.csv", index=False, float_format="%.12g")

    mechanism_parts = []
    columns = ["detector", "role", "delta_harm", "primary_failure_mechanism"]
    for path in parquet_shards:
        data = pd.read_parquet(path, columns=columns)
        mechanism_parts.append(data[data.delta_harm > 0])
    harmed = pd.concat(mechanism_parts, ignore_index=True)
    summary = harmed.groupby(["detector", "role", "primary_failure_mechanism"], observed=True).agg(
        harmed_objects=("delta_harm", "size"), median_delta_harm=("delta_harm", "median")
    ).reset_index()
    totals = harmed.groupby(["detector", "role"], observed=True).size().rename("total_harmed").reset_index()
    summary = summary.merge(totals, on=["detector", "role"])
    summary["percent_of_harmed"] = summary.harmed_objects / summary.total_harmed * 100
    summary.to_csv(OUT / "failure_mechanism_summary.csv", index=False, float_format="%.12g")
    print({"shards": len(parquet_shards), "object_rows": pq.ParquetFile(destination).metadata.num_rows,
           "component_log_rows": len(component_logs)})


if __name__ == "__main__":
    main()
