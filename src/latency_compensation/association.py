import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import box_iou


def hungarian_iou(previous_boxes, current_boxes, minimum_iou: float = 0.1):
    if not previous_boxes or not current_boxes:
        return []
    cost = np.array([[1.0 - box_iou(a, b) for b in current_boxes] for a in previous_boxes])
    rows, columns = linear_sum_assignment(cost)
    return [(int(r), int(c), 1.0 - float(cost[r, c])) for r, c in zip(rows, columns) if 1.0 - cost[r, c] >= minimum_iou]
