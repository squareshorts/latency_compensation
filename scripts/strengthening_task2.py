import pandas as pd
import numpy as np
from pathlib import Path
import json

from latency_compensation.object_motion_harm import cluster_bootstrap

def get_ci(series, log_ids):
    pass

def main():
    root = Path(r"c:\work\latency_compensation")
    out = root / "results" / "manuscript_strengthening_20260719"
    out.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_parquet(root / "results" / "object_motion_harm" / "object_level_diagnostics.parquet",
                         columns=["comparison_id", "log_id", "role", "detector", "delta_ms", "high_yaw", "object_class", "B3_error", "B5_error", "B3_iou", "B5_iou"])
                         
    df = df[df.role == "heldout"]
    
    # map classes
    mapping = {
        "REGULAR_VEHICLE": "vehicle",
        "TRUCK": "vehicle",
        "LARGE_VEHICLE": "vehicle",
        "VEHICULAR_TRAILER": "vehicle",
        "TRUCK_CAB": "vehicle",
        "BOX_TRUCK": "vehicle",
        "BUS": "vehicle",
        "SCHOOL_BUS": "vehicle",
        "ARTICULATED_BUS": "vehicle",
        "PEDESTRIAN": "pedestrian/cyclist",
        "BICYCLE": "pedestrian/cyclist",
        "BICYCLIST": "pedestrian/cyclist",
        "MOTORCYCLE": "pedestrian/cyclist",
        "MOTORCYCLIST": "pedestrian/cyclist",
        "WHEELED_DEVICE": "pedestrian/cyclist",
        "WHEELED_RIDER": "pedestrian/cyclist",
        "STROLLER": "pedestrian/cyclist"
    }
    df["object_class"] = df["object_class"].map(lambda x: mapping.get(x, "other"))
    
    df["difference"] = df.B5_error - df.B3_error
    df["relative"] = df.difference / df.B3_error.replace(0, np.nan)
    
    def analyze_stratum(data, filename, is_primary=False):
        rows = []
        bootstrap_rows = []
        for det in ["yolo11n", "yolo11s", "rtdetr_l"]:
            for cls in ["vehicle", "pedestrian/cyclist", "other"]:
                subset = data[(data.detector == det) & (data.object_class == cls)]
                if subset.empty:
                    continue
                
                per_log = subset.groupby("log_id").agg(
                    b3=("B3_error", "median"),
                    b5=("B5_error", "median"),
                    b3_iou=("B3_iou", "median"),
                    b5_iou=("B5_iou", "median"),
                    objs=("comparison_id", "count")
                )
                
                per_log_b3_recall = subset.groupby("log_id")["B3_iou"].apply(lambda x: (x >= 0.5).mean())
                per_log_b5_recall = subset.groupby("log_id")["B5_iou"].apply(lambda x: (x >= 0.5).mean())

                per_log["difference"] = per_log.b5 - per_log.b3
                per_log["relative"] = per_log.difference / per_log.b3.replace(0, np.nan)
                
                valid_logs = per_log.dropna(subset=["relative"])
                if len(valid_logs) == 0:
                    continue
                
                if len(valid_logs) > 5:
                    try:
                        ci_lower, ci_upper = cluster_bootstrap(valid_logs.relative.values.tolist(), replicates=1000, seed=17)
                    except:
                        ci_lower, ci_upper = np.nan, np.nan
                else:
                    ci_lower, ci_upper = np.nan, np.nan
                
                median_b3 = valid_logs.b3.median()
                median_b5 = valid_logs.b5.median()
                median_rel = valid_logs.relative.median()
                
                n_logs = len(valid_logs)
                n_objs = valid_logs.objs.sum()
                logs_favor_b5 = (valid_logs.difference < 0).sum()
                
                median_b3_iou = valid_logs.b3_iou.median()
                median_b5_iou = valid_logs.b5_iou.median()
                
                recall_b3 = per_log_b3_recall.median()
                recall_b5 = per_log_b5_recall.median()
                
                rows.append({
                    "detector": det,
                    "class": cls,
                    "eligible_logs": n_logs,
                    "objects": n_objs,
                    "B3_median_error": median_b3,
                    "B5_median_error": median_b5,
                    "median_paired_relative_diff": median_rel,
                    "logs_favoring_B5": logs_favor_b5,
                    "B3_median_IoU": median_b3_iou,
                    "B5_median_IoU": median_b5_iou,
                    "B3_recall_0.5": recall_b3,
                    "B5_recall_0.5": recall_b5
                })
                
                bootstrap_rows.append({
                    "detector": det,
                    "class": cls,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper
                })
                
        pd.DataFrame(rows).to_csv(out / filename, index=False)
        return bootstrap_rows
        
    primary = df[(df.delta_ms == 300) & (df.high_yaw == True)]
    all_yaw = df[df.delta_ms == 300]
    
    bs1 = analyze_stratum(primary, "class_breakdown_primary.csv", True)
    bs2 = analyze_stratum(all_yaw, "class_breakdown_all_yaw.csv", False)
    
    pd.DataFrame(bs1).to_csv(out / "class_breakdown_bootstrap.csv", index=False)
    
    with open(out / "class_breakdown.md", "w") as f:
        f.write("# Class Breakdown Analysis\n")
        f.write("Generated from frozen diagnostics.\n")

if __name__ == "__main__":
    main()
