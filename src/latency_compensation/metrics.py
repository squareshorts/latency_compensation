import numpy as np
import pandas as pd

from .geometry import box_iou, center_error, normalized_center_error, scale_error


def evaluate_box(predicted, target) -> dict:
    overlap = box_iou(predicted, target)
    return {
        "center_error_px": center_error(predicted, target),
        "normalized_center_error": normalized_center_error(predicted, target),
        "iou": overlap,
        "scale_error": scale_error(predicted, target),
        "recall_iou_0_3": int(overlap >= 0.3),
        "recall_iou_0_5": int(overlap >= 0.5),
        "recall_iou_0_7": int(overlap >= 0.7),
    }


def summarize(rows: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "objects": len(rows),
        "median_center_error_px": rows.center_error_px.median(),
        "median_normalized_center_error": rows.normalized_center_error.median(),
        "median_iou": rows.iou.median(),
        "recall_iou_0_3": rows.recall_iou_0_3.mean(),
        "recall_iou_0_5": rows.recall_iou_0_5.mean(),
        "recall_iou_0_7": rows.recall_iou_0_7.mean(),
        "median_scale_error": rows.scale_error.median(),
    })
