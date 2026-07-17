import math

import numpy as np


def box_to_state(box) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, max(2.0, x2 - x1), max(2.0, y2 - y1)])


def state_to_box(state, width: int = 1600, height: int = 900) -> np.ndarray:
    cx, cy, w, h = map(float, state)
    return np.array([np.clip(cx - w / 2, 0, width - 1), np.clip(cy - h / 2, 0, height - 1), np.clip(cx + w / 2, 0, width - 1), np.clip(cy + h / 2, 0, height - 1)])


def box_iou(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ix1, iy1 = np.maximum(a[:2], b[:2])
    ix2, iy2 = np.minimum(a[2:], b[2:])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter) if area_a + area_b - inter else 0.0


def center_error(a, b) -> float:
    sa, sb = box_to_state(a), box_to_state(b)
    return float(np.linalg.norm(sa[:2] - sb[:2]))


def normalized_center_error(a, b, image_width: int = 1600, image_height: int = 900) -> float:
    return center_error(a, b) / math.hypot(image_width, image_height)


def scale_error(a, b) -> float:
    sa, sb = box_to_state(a), box_to_state(b)
    return float(abs(np.log((sa[2] * sa[3]) / (sb[2] * sb[3]))))
