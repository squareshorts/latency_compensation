import numpy as np
import pandas as pd
import pytest

from latency_compensation.av2_models import assemble_distinct_model_boxes
from latency_compensation.causal_trackers import ByteTrackAdapter, OCSortAdapter
from latency_compensation.object_motion_harm import (
    FrozenGate,
    assert_causal_history,
    assert_oracle_input_isolation,
    cluster_bootstrap,
    common_support_mask,
    paired_log_table,
    validate_component_substitution,
    validate_dataset_harmonization,
    validate_frozen_gate,
)


def test_b3_b5_distinction():
    values = [np.asarray([index, 1, index + 2, 3], float) for index in range(6)]
    models = assemble_distinct_model_boxes(b0=values[0], b1=values[1], b2=values[2], b3=values[3], b4=values[4], b5=values[5])
    assert not np.array_equal(models["B3"], models["B5"])


def test_causal_history_enforcement():
    assert_causal_history([10, 20, 30], 30)
    with pytest.raises(ValueError):
        assert_causal_history([10, 31], 30)


def test_oracle_input_isolation_and_no_target_geometry():
    assert_oracle_input_isolation("M5", ["source_depth", "history_box"])
    assert_oracle_input_isolation("M6", ["future_displacement"])
    with pytest.raises(ValueError):
        assert_oracle_input_isolation("M5", ["future_displacement"])


def test_component_substitution_is_unique_and_future_isolated():
    frame = pd.DataFrame({"comparison_id": ["a"] * 8, "model": [f"M{x}" for x in range(8)],
                          "future_geometry_input": [False] * 6 + [True, True]})
    validate_component_substitution(frame)
    frame.loc[0, "future_geometry_input"] = True
    with pytest.raises(ValueError):
        validate_component_substitution(frame)


def test_dataset_harmonization_contract():
    frame = pd.DataFrame({"dataset": ["AV2", "nuScenes"], "detector": ["yolo11n"] * 2,
                          "latency_ms": [500, 500], "metric_definition": ["diagonal"] * 2, "model": ["B3", "B5"]})
    validate_dataset_harmonization(frame)


def test_heldout_gate_freezing():
    gate = FrozenGate("regularized_logistic", .6, 2, 10, 5)
    validate_frozen_gate(gate, "heldout")
    with pytest.raises(ValueError):
        validate_frozen_gate(FrozenGate("x", .5, 1, 5, 5, selected_on_split="heldout"), "heldout")


def test_cluster_bootstrap_reproducibility():
    one = cluster_bootstrap([1, 2, 3, 4], replicates=500, seed=17)
    two = cluster_bootstrap([1, 2, 3, 4], replicates=500, seed=17)
    assert one == two


def test_metric_recalculation_and_detector_split_isolation():
    rows = []
    for detector in ("n", "s"):
        for model, error in (("B3", .1), ("B5", .2)):
            rows.append({"detector": detector, "role": "heldout", "log_id": "log", "model": model,
                         "normalized_center_error": error, "iou": 1-error,
                         "recall_iou_0_3": 1, "recall_iou_0_5": 1, "recall_iou_0_7": 0})
    result = paired_log_table(pd.DataFrame(rows), group_columns=["detector", "role"])
    assert len(result) == 2
    assert np.allclose(result.delta_normalized_center_error, .1)


def test_common_support_is_deterministic():
    source = pd.DataFrame({"x": np.arange(100)})
    target = pd.DataFrame({"x": [-10, 50, 200]})
    assert common_support_mask(source, target, ["x"]).tolist() == [False, True, False]


def test_prediction_only_trackers_use_only_supplied_history():
    first = {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 50.0, "class_id": 2, "confidence": 0.9}
    second = {"x1": 12.0, "y1": 20.0, "x2": 32.0, "y2": 50.0, "class_id": 2, "confidence": 0.9}
    for tracker in (ByteTrackAdapter(), OCSortAdapter()):
        tracker.update(0, [first])
        assigned = tracker.update(100_000_000, [second])
        prediction = assigned[0].forecast(300_000_000)
        assert prediction.shape == (4,)
        assert np.isfinite(prediction).all()
        assert prediction[0] > first["x1"]


def test_bytetrack_low_confidence_detection_does_not_start_track():
    tracker = ByteTrackAdapter(high_threshold=0.5, low_threshold=0.25)
    low = {"x1": 10.0, "y1": 20.0, "x2": 30.0, "y2": 50.0, "class_id": 2, "confidence": 0.3}
    assert tracker.update(0, [low]) == {}
