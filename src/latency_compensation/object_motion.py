import numpy as np

from .geometry import box_to_state, state_to_box


def constant_velocity(history_boxes, history_timestamps_us, target_timestamp_us):
    if len(history_boxes) < 2:
        return np.asarray(history_boxes[-1], float)
    previous, current = box_to_state(history_boxes[-2]), box_to_state(history_boxes[-1])
    dt = (history_timestamps_us[-1] - history_timestamps_us[-2]) / 1e6
    horizon = (target_timestamp_us - history_timestamps_us[-1]) / 1e6
    velocity = (current - previous) / max(dt, 1e-6)
    return state_to_box(current + velocity * horizon)


class BoxKalmanFilter:
    """Small constant-velocity Kalman filter over box center and scale."""

    def __init__(self, process_variance: float = 4.0, measurement_variance: float = 9.0):
        self.x = None
        self.p = np.eye(8) * 100.0
        self.q = process_variance
        self.r = np.eye(4) * measurement_variance

    def update(self, box, dt: float) -> None:
        z = box_to_state(box)
        if self.x is None:
            self.x = np.r_[z, np.zeros(4)]
            return
        f = np.eye(8)
        f[:4, 4:] = np.eye(4) * dt
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + np.eye(8) * self.q
        h = np.c_[np.eye(4), np.zeros((4, 4))]
        innovation = z - h @ self.x
        s = h @ self.p @ h.T + self.r
        gain = self.p @ h.T @ np.linalg.inv(s)
        self.x += gain @ innovation
        self.p = (np.eye(8) - gain @ h) @ self.p

    def predict_box(self, horizon_seconds: float):
        if self.x is None:
            raise RuntimeError("Kalman filter has no observation")
        f = np.eye(8)
        f[:4, 4:] = np.eye(4) * horizon_seconds
        return state_to_box((f @ self.x)[:4])


def alpha_beta_prediction(history_boxes, history_timestamps_us, target_timestamp_us, alpha: float = 0.65):
    if len(history_boxes) < 2:
        return np.asarray(history_boxes[-1], float)
    states = np.vstack([box_to_state(box) for box in history_boxes])
    velocity = np.zeros(4)
    for index in range(1, len(states)):
        dt = (history_timestamps_us[index] - history_timestamps_us[index - 1]) / 1e6
        observed = (states[index] - states[index - 1]) / max(dt, 1e-6)
        velocity = alpha * observed + (1 - alpha) * velocity
    horizon = (target_timestamp_us - history_timestamps_us[-1]) / 1e6
    return state_to_box(states[-1] + velocity * horizon)
