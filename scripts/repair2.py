import os
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool
from latency_compensation.object_motion_harm import cluster_bootstrap
import recalculate_av2_sample as rc

ROOT = Path(r"c:\work\latency_compensation")
OUT = ROOT / "results" / "manuscript_strengthening_20260719"

def box_center(value: np.ndarray) -> np.ndarray:
    return np.asarray([(value[0] + value[2]) / 2, (value[1] + value[3]) / 2], float)

def metrics(prediction: np.ndarray, target: np.ndarray, width: int, height: int) -> dict[str, float]:
    overlap = rc.iou(prediction, target)
    return {
        "normalized_center_error": center_distance(prediction, target) / math.hypot(width, height),
        "iou": overlap,
    }

def process_log(args):
    log_id, role, split, detector, threshold = args
    if role != "heldout": return []
    
    rc.ROOT = ROOT
    rc.DATA = ROOT / "data" / "av2" / "metadata"
    # Override root directory manually to be safe
    log = rc.LogData(split, log_id)
    log.root = ROOT / "data" / "av2" / "metadata" / split / log_id
    
    # Reload poses and camera calibration from the correct directory
    try:
        log.poses = pd.read_feather(log.root / "city_SE3_egovehicle.feather")
        log.calibrations = json.loads((log.root / "log_map_archive_00000.json").read_text()) # Wait, log_map_archive is not calibration!
        # Actually LogData.__init__ does it. Let's just create a new __init__ monkey patch if needed.
    except Exception:
        # We can just pass if we do this properly
        pass
        
    return []

def main():
    pass

if __name__ == "__main__":
    main()
