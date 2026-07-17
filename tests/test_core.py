import numpy as np
import pandas as pd
import pytest

from latency_compensation.association import hungarian_iou
from latency_compensation.datasets import deterministic_log_split
from latency_compensation.ego_motion import integrate_planar_motion
from latency_compensation.geometry import box_iou, box_to_state, state_to_box
from latency_compensation.metrics import evaluate_box
from latency_compensation.object_motion import BoxKalmanFilter, constant_velocity
from latency_compensation.propagation import uncertainty_damping
from latency_compensation.synchronization import assert_causal, causal_index


def test_box_projection_roundtrip():
    box = np.array([10.0, 20.0, 110.0, 220.0])
    assert np.allclose(state_to_box(box_to_state(box)), box)


def test_coordinate_transform_integration():
    result = integrate_planar_motion([0, 1_000_000], [10.0, 10.0], [0.0, 0.0])
    assert np.allclose(result, [10.0, 0.0, 0.0])


def test_timestamp_causality():
    assert causal_index([10, 20, 30], 25) == 1
    assert causal_index([10, 20], 5) is None
    assert_causal([10, 20], [10, 25])
    with pytest.raises(ValueError):
        assert_causal([11], [10])


def test_hungarian_association_gate():
    matches = hungarian_iou([[0, 0, 10, 10]], [[1, 1, 11, 11], [50, 50, 60, 60]], minimum_iou=0.2)
    assert matches == [(0, 0, pytest.approx(81 / 119))]


def test_depth_uncertainty_damping():
    assert uncertainty_damping(0, 1.0, 3) == 0
    assert 0 < uncertainty_damping(5, 1.0, 2) < 1
    assert uncertainty_damping(20, 0.0, 5) == 1


def test_object_motion_extrapolation():
    predicted = constant_velocity([[0, 0, 10, 10], [10, 0, 20, 10]], [0, 1_000_000], 2_000_000)
    assert np.allclose(predicted, [20, 0, 30, 10])
    kalman = BoxKalmanFilter()
    kalman.update([0, 0, 10, 10], 0)
    kalman.update([10, 0, 20, 10], 1)
    assert box_to_state(kalman.predict_box(1))[0] > 15


def test_metric_calculation():
    result = evaluate_box([0, 0, 10, 10], [0, 0, 10, 10])
    assert result["center_error_px"] == 0
    assert result["iou"] == 1
    assert result["recall_iou_0_7"] == 1


def test_scene_split_isolation_and_determinism():
    first = deterministic_log_split([f"log-{i}" for i in range(20)], 10, 4, 6)
    second = deterministic_log_split([f"log-{i}" for i in reversed(range(20))], 10, 4, 6)
    assert first == second
    sets = [set(value) for value in first.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
