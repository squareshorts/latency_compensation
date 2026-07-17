import numpy as np

from .geometry import box_to_state, state_to_box


def apply_box_delta(box, center_delta=(0.0, 0.0), log_scale_delta=(0.0, 0.0)):
    state = box_to_state(box)
    state[:2] += np.asarray(center_delta, float)
    state[2:] *= np.exp(np.asarray(log_scale_delta, float))
    return state_to_box(state)


def uncertainty_damping(point_count: int, depth_dispersion: float, track_age: int) -> float:
    lidar_term = min(1.0, point_count / 10.0)
    depth_term = 1.0 / (1.0 + max(0.0, depth_dispersion) / 5.0)
    age_term = min(1.0, track_age / 3.0)
    return float(np.clip(lidar_term * depth_term * age_term, 0.0, 1.0))
