"""Auditable utilities for the frozen post-confirmatory harm analysis.

The functions in this module deliberately separate deployable causal inputs
from evaluation-only oracle inputs.  They contain no detector inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEPLOYABLE_MODELS = ("M0", "M1", "M2", "M3", "M4", "M5")
DIAGNOSTIC_ORACLES = ("M6", "M7")
TARGET_TIME_GEOMETRY_COLUMNS = {
    "target_x1", "target_y1", "target_x2", "target_y2",
    "target_annotation_box", "future_displacement", "future_dx",
    "future_dy", "future_dz",
}


def paired_relative_difference(candidate: float, reference: float) -> float:
    """Return candidate/reference - 1, or NaN when the reference is invalid."""
    if not np.isfinite(reference) or reference == 0 or not np.isfinite(candidate):
        return float("nan")
    return float(candidate / reference - 1.0)


def paired_log_table(frame: pd.DataFrame, *, group_columns: Sequence[str]) -> pd.DataFrame:
    """Collapse object rows to paired log-level B3/B5 endpoint summaries."""
    required = set(group_columns) | {
        "log_id", "model", "normalized_center_error", "iou",
        "recall_iou_0_3", "recall_iou_0_5", "recall_iou_0_7",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing metric columns: {sorted(missing)}")
    data = frame[frame.model.isin(["B3", "B5"])].copy()
    keys = list(group_columns) + ["log_id", "model"]
    metrics = [
        "normalized_center_error", "iou", "recall_iou_0_3",
        "recall_iou_0_5", "recall_iou_0_7",
    ]
    aggregate = data.groupby(keys, observed=True)[metrics].median().reset_index()
    wide = aggregate.pivot(index=list(group_columns) + ["log_id"], columns="model", values=metrics)
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    for metric in metrics:
        wide[f"delta_{metric}"] = wide[f"{metric}_B5"] - wide[f"{metric}_B3"]
        wide[f"relative_{metric}"] = wide.apply(
            lambda row: paired_relative_difference(row[f"{metric}_B5"], row[f"{metric}_B3"]), axis=1
        )
    return wide


def cluster_bootstrap(
    values: Iterable[float], *, replicates: int = 10_000, seed: int = 20260718,
    statistic: str = "median",
) -> dict[str, float]:
    """Deterministic paired bootstrap over already-collapsed independent logs."""
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "replicates": replicates, "seed": seed}
    rng = np.random.default_rng(seed)
    reducer = np.nanmedian if statistic == "median" else np.nanmean
    estimates = np.empty(replicates, dtype=float)
    # Chunk the index matrix so the function is stable for large replicate counts.
    cursor = 0
    while cursor < replicates:
        count = min(2_000, replicates - cursor)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        estimates[cursor:cursor + count] = reducer(array[indices], axis=1)
        cursor += count
    return {
        "estimate": float(reducer(array)),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "replicates": int(replicates),
        "seed": int(seed),
    }


def paired_cluster_bootstrap(
    candidate: Iterable[float], reference: Iterable[float], *, replicates: int = 10_000,
    seed: int = 20260718, statistic: str = "median", relative: bool = False,
) -> dict[str, float]:
    """Bootstrap a paired log-level difference or relative ratio-of-statistics."""
    candidate_array = np.asarray(list(candidate), dtype=float)
    reference_array = np.asarray(list(reference), dtype=float)
    keep = np.isfinite(candidate_array) & np.isfinite(reference_array)
    candidate_array, reference_array = candidate_array[keep], reference_array[keep]
    if not len(candidate_array):
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "replicates": replicates, "seed": seed}
    reducer = np.nanmedian if statistic == "median" else np.nanmean

    def contrast(index=None):
        c = reducer(candidate_array if index is None else candidate_array[index])
        r = reducer(reference_array if index is None else reference_array[index])
        return paired_relative_difference(c, r) if relative else float(c - r)

    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    cursor = 0
    while cursor < replicates:
        count = min(2_000, replicates - cursor)
        indices = rng.integers(0, len(candidate_array), size=(count, len(candidate_array)))
        candidate_stat = reducer(candidate_array[indices], axis=1)
        reference_stat = reducer(reference_array[indices], axis=1)
        if relative:
            estimates[cursor:cursor + count] = np.divide(
                candidate_stat, reference_stat,
                out=np.full(count, np.nan, dtype=float), where=reference_stat != 0,
            ) - 1.0
        else:
            estimates[cursor:cursor + count] = candidate_stat - reference_stat
        cursor += count
    finite = estimates[np.isfinite(estimates)]
    return {
        "estimate": contrast(),
        "ci_low": float(np.quantile(finite, 0.025)) if len(finite) else np.nan,
        "ci_high": float(np.quantile(finite, 0.975)) if len(finite) else np.nan,
        "replicates": int(replicates),
        "seed": int(seed),
    }


def assert_causal_history(history_timestamps: Sequence[int], availability_timestamp: int) -> None:
    """Reject historical inputs that occur after the deployable availability time."""
    if any(int(timestamp) > int(availability_timestamp) for timestamp in history_timestamps):
        raise ValueError("Non-causal history timestamp detected")


def assert_oracle_input_isolation(model: str, columns_used: Iterable[str]) -> None:
    """Allow target/future geometry only for the explicitly diagnostic M6/M7."""
    used = set(columns_used)
    leaked = used & TARGET_TIME_GEOMETRY_COLUMNS
    if model in DEPLOYABLE_MODELS and leaked:
        raise ValueError(f"Deployable {model} used target-time geometry: {sorted(leaked)}")
    if model not in DEPLOYABLE_MODELS + DIAGNOSTIC_ORACLES:
        raise ValueError(f"Unknown substitution model: {model}")


def validate_component_substitution(frame: pd.DataFrame) -> None:
    """Validate one and only one row per comparison/substitution model."""
    required = {"comparison_id", "model", "future_geometry_input"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing substitution columns: {sorted(required - set(frame.columns))}")
    if frame.duplicated(["comparison_id", "model"]).any():
        raise ValueError("Duplicate component-substitution row")
    invalid_future = frame.future_geometry_input & ~frame.model.isin(DIAGNOSTIC_ORACLES)
    if invalid_future.any():
        raise ValueError("Future geometry escaped into a deployable substitution")


@dataclass(frozen=True)
class FrozenGate:
    model_name: str
    threshold: float
    minimum_track_age: int
    maximum_depth_uncertainty: float
    maximum_velocity_uncertainty: float
    selected_on_split: str = "model_selection"


def validate_frozen_gate(gate: FrozenGate, evaluation_split: str) -> None:
    """Prevent held-out threshold selection or a non-frozen gate evaluation."""
    if gate.selected_on_split != "model_selection":
        raise ValueError("Gate choices must be frozen on model-selection logs")
    if evaluation_split != "heldout":
        raise ValueError("The one-shot gate evaluation is reserved for held-out logs")


def common_support_mask(source: pd.DataFrame, target: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Return target rows inside the source's robust univariate support."""
    mask = pd.Series(True, index=target.index)
    for column in columns:
        values = pd.to_numeric(source[column], errors="coerce").dropna()
        if values.empty:
            continue
        low, high = values.quantile([0.01, 0.99])
        candidate = pd.to_numeric(target[column], errors="coerce")
        mask &= candidate.between(low, high, inclusive="both")
    return mask


def validate_dataset_harmonization(frame: pd.DataFrame) -> None:
    required = {"dataset", "detector", "latency_ms", "metric_definition", "model"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing harmonization columns: {sorted(missing)}")
    if set(frame.dataset.dropna().unique()) != {"AV2", "nuScenes"}:
        raise ValueError("Harmonization must contain both AV2 and nuScenes")
    if (frame.latency_ms != 500).any():
        raise ValueError("Cross-dataset harmonization is fixed at 500 ms")
