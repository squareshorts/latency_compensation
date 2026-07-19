"""Model-output assembly guards for the corrected AV2 propagation stage."""

from __future__ import annotations

import numpy as np


MODEL_NAMES = ("B0", "B1", "B2", "B3", "B4", "B5")


def assemble_distinct_model_boxes(*, b0, b1, b2, b3, b4, b5) -> dict[str, np.ndarray]:
    """Return independently owned model coordinates after validating shape.

    Legitimate coordinates may coincide on a zero-motion observation, but no
    model is allowed to alias another model's mutable array.  The generating
    branch must explicitly supply every model rather than relabel one shared
    temporary as B2/B3/B4.
    """
    supplied = {"B0": b0, "B1": b1, "B2": b2, "B3": b3, "B4": b4, "B5": b5}
    result = {}
    for name in MODEL_NAMES:
        value = np.asarray(supplied[name], dtype=float)
        if value.shape != (4,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite four-coordinate box")
        result[name] = value.copy()
    if len({id(value) for value in result.values()}) != len(MODEL_NAMES):
        raise RuntimeError("AV2 model coordinate arrays are aliased")
    return result


def assert_nonzero_motion_branches(models: dict[str, np.ndarray], *, ego_motion_nonzero: bool,
                                   image_motion_nonzero: bool, object_motion_nonzero: bool) -> None:
    """Regression guard against the exact duplication in the invalid run."""
    if ego_motion_nonzero and np.array_equal(models["B0"], models["B3"]):
        raise RuntimeError("B3 collapsed to B0 despite nonzero ego motion")
    if image_motion_nonzero and np.array_equal(models["B1"], models["B2"]):
        # A Kalman estimate can exceptionally equal constant velocity, so this
        # guard is intended for deterministic regression fixtures, not every row.
        raise RuntimeError("B2 collapsed to B1 in the nonzero-motion regression fixture")
    if object_motion_nonzero and np.array_equal(models["B3"], models["B4"]):
        raise RuntimeError("B4 collapsed to B3 despite nonzero object motion")
