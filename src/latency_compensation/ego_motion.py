import math

import numpy as np


def integrate_planar_motion(timestamps_us, speeds_mps, yaw_rates_rad_s, accelerations_mps2=None):
    """Causally integrate planar ego motion from timestamped measurements."""
    if len(timestamps_us) < 2:
        return np.zeros(3)
    acceleration = np.zeros(len(timestamps_us)) if accelerations_mps2 is None else np.asarray(accelerations_mps2, float)
    speed = float(speeds_mps[0])
    x = y = yaw = 0.0
    for index in range(1, len(timestamps_us)):
        dt = (timestamps_us[index] - timestamps_us[index - 1]) / 1e6
        yaw += float(yaw_rates_rad_s[index - 1]) * dt
        speed += float(acceleration[index - 1]) * dt
        x += speed * math.cos(yaw) * dt
        y += speed * math.sin(yaw) * dt
    return np.array([x, y, yaw])
