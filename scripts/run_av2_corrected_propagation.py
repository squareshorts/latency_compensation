"""Resumable corrected AV2 propagation using preserved detector checkpoints.

Outputs are written to ``corrected_checkpoints``; original Parquets are never
deleted or overwritten.  Detector inference is not imported or callable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

ROOT = Path(r"C:\work\auto")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from latency_compensation.av2_models import assemble_distinct_model_boxes
from recalculate_av2_sample import (LogData, box_state, center_distance, damping, image_velocity_prediction,
                                    iou, kalman_prediction, state_box, track_history)

OUT = ROOT / "results" / "av2_confirmation"
CORRECTED = OUT / "corrected_checkpoints"
DELTAS = (100, 200, 300, 400, 500)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*pyarrow.feather.read_feather.*")


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def analysis_group(category: str, mapping: dict[str, str]) -> str:
    return mapping.get(category, "other")


def coco_group(class_id: int) -> str:
    if class_id in {2, 5, 7}: return "vehicle"
    if class_id == 0: return "pedestrian_cyclist"
    return "other"


def source_evaluation_association(detections, targets, category_mapping):
    if not detections or not targets: return {}
    cost = np.full((len(detections), len(targets)), 1e6)
    for i, (_, detection) in enumerate(detections):
        box = np.asarray([detection["x1"],detection["y1"],detection["x2"],detection["y2"]],float)
        for j, target in enumerate(targets):
            if coco_group(int(detection["class_id"])) != analysis_group(target[1], category_mapping): continue
            overlap = iou(box,target[2]); distance = center_distance(box,target[2])/np.hypot(1550,2048)
            if overlap >= .10 or distance <= .15: cost[i,j] = (1-overlap) + distance
    rows, columns = linear_sum_assignment(cost)
    return {detections[i][0]: targets[j][0] for i,j in zip(rows,columns) if cost[i,j] < 1e5}


def target_by_track(targets, track_uuid):
    for target in targets:
        if target[0] == track_uuid: return target
    return None


def calculate_metrics(prediction, target, width, height):
    error = center_distance(prediction,target); overlap = iou(prediction,target)
    ps, ts = box_state(prediction), box_state(target)
    return {"center_error_px":error,"normalized_center_error":error/np.hypot(width,height),"iou":overlap,
            "recall_iou_0_3":int(overlap>=.3),"recall_iou_0_5":int(overlap>=.5),"recall_iou_0_7":int(overlap>=.7),
            "scale_error":float(abs(np.log(max(ps[2]*ps[3],1e-9)/max(ts[2]*ts[3],1e-9))))}


def process_detector(log: LogData, payload: dict, detector: str, role: str, pair_rows: pd.DataFrame, category_mapping: dict[str,str]) -> pd.DataFrame:
    detections = payload["detections"]
    by_ts = {}
    for index,detection in enumerate(detections): by_ts.setdefault(int(detection["timestamp_ns"]),[]).append((index,detection))
    pair_map = {(int(r.source_image_timestamp_ns),int(r.delta_ms)):r for r in pair_rows.itertuples(index=False)}
    source_assoc_cache = {}; depth_cache = {}; rows = []
    for index,detection in enumerate(detections):
        source_ts = int(detection["timestamp_ns"])
        available = [delta for delta in DELTAS if (source_ts,delta) in pair_map]
        if not available: continue
        first_pair = pair_map[(source_ts,available[0])]
        source_key = (source_ts,int(first_pair.source_timestamp_ns))
        if source_key not in source_assoc_cache:
            source_targets = log.projected_targets(int(first_pair.source_timestamp_ns),source_ts)
            source_assoc_cache[source_key] = source_evaluation_association(by_ts[source_ts],source_targets,category_mapping)
        track_uuid = source_assoc_cache[source_key].get(index)
        if track_uuid is None: continue
        class_id = int(detection["class_id"]); source_box = np.asarray([detection["x1"],detection["y1"],detection["x2"],detection["y2"]],float)
        history = track_history(by_ts,source_ts,index,detection)
        if index not in depth_cache: depth_cache[index] = log.depth(source_box,source_ts,class_id)
        depth,dispersion,point_count,lidar_ts,depth_source = depth_cache[index]
        city_positions=[]
        for hist_ts,hist_index,hist_detection in history:
            hist_box=np.asarray([hist_detection["x1"],hist_detection["y1"],hist_detection["x2"],hist_detection["y2"]],float)
            if hist_index not in depth_cache: depth_cache[hist_index]=log.depth(hist_box,hist_ts,class_id)
            hist_depth=depth_cache[hist_index][0]; cx=(hist_box[0]+hist_box[2])/2;cy=(hist_box[1]+hist_box[3])/2
            cam=np.asarray([[(cx-log.K[0,2])/log.K[0,0]*hist_depth,(cy-log.K[1,2])/log.K[1,1]*hist_depth,hist_depth]])
            city_positions.append(log.cam_points_to_city(cam,hist_ts)[0])
        object_velocity=np.zeros(3)
        if len(city_positions)>=2:
            dt=(history[-1][0]-history[-2][0])/1e9;object_velocity=(city_positions[-1]-city_positions[-2])/max(dt,1e-6)
            speed=np.linalg.norm(object_velocity[:2]);object_velocity*=min(1,30/max(speed,1e-9))
        weight=damping(point_count,dispersion,len(history))
        for delta in available:
            pair=pair_map[(source_ts,delta)];target_tuple=target_by_track(log.projected_targets(int(pair.target_timestamp_ns),int(pair.target_image_timestamp_ns)),track_uuid)
            if target_tuple is None: continue
            _,category,target_box=target_tuple;target_ts=int(pair.target_image_timestamp_ns)
            start=time.perf_counter_ns();b0=source_box.copy();runtime_b0=(time.perf_counter_ns()-start)/1e6
            start=time.perf_counter_ns();b1,image_velocity=image_velocity_prediction(history,target_ts);runtime_b1=(time.perf_counter_ns()-start)/1e6
            start=time.perf_counter_ns();b2,kalman_velocity=kalman_prediction(history,target_ts);runtime_b2=(time.perf_counter_ns()-start)/1e6
            start=time.perf_counter_ns();b3=log.propagate_geometry(source_box,depth,source_ts,target_ts,np.zeros(3));runtime_b3=(time.perf_counter_ns()-start)/1e6
            displacement=object_velocity*(delta/1000)
            start=time.perf_counter_ns();b4=log.propagate_geometry(source_box,depth,source_ts,target_ts,displacement);runtime_b4=(time.perf_counter_ns()-start)/1e6
            start=time.perf_counter_ns();b5=log.propagate_geometry(source_box,depth,source_ts,target_ts,displacement*weight);runtime_b5=(time.perf_counter_ns()-start)/1e6
            models=assemble_distinct_model_boxes(b0=b0,b1=b1,b2=b2,b3=b3,b4=b4,b5=b5);runtimes={"B0":runtime_b0,"B1":runtime_b1,"B2":runtime_b2,"B3":runtime_b3,"B4":runtime_b4,"B5":runtime_b5}
            source_pose,target_pose=log.pose(source_ts),log.pose(target_ts);yaw_s=Rotation.from_matrix(source_pose[:3,:3]).as_euler("xyz")[2];yaw_t=Rotation.from_matrix(target_pose[:3,:3]).as_euler("xyz")[2];yaw_change=float((yaw_t-yaw_s+np.pi)%(2*np.pi)-np.pi)
            comparison_id=f"{payload['log_id']}:{detector}:{source_ts}:{index}:{delta}:{track_uuid}"
            for model,prediction in models.items():
                row={"comparison_id":comparison_id,"log_id":payload["log_id"],"role":role,"detector":detector,"source_timestamp_ns":source_ts,"target_timestamp_ns":target_ts,"source_annotation_timestamp_ns":int(pair.source_timestamp_ns),"target_annotation_timestamp_ns":int(pair.target_timestamp_ns),"delta_ms":delta,"model":model,"class_id":class_id,"target_category":category,"evaluation_track_uuid":track_uuid,
                     "source_x1":source_box[0],"source_y1":source_box[1],"source_x2":source_box[2],"source_y2":source_box[3],"target_x1":target_box[0],"target_y1":target_box[1],"target_x2":target_box[2],"target_y2":target_box[3],"pred_x1":prediction[0],"pred_y1":prediction[1],"pred_x2":prediction[2],"pred_y2":prediction[3],
                     "estimated_depth_m":depth,"depth_dispersion_m":dispersion,"lidar_point_count":point_count,"lidar_timestamp_ns":lidar_ts,"depth_source":depth_source,"track_age":len(history),"image_velocity_norm_px_s":float(np.linalg.norm(image_velocity[:2])),"kalman_velocity_norm_px_s":float(np.linalg.norm(kalman_velocity[:2])),"object_velocity_norm_m_s":float(np.linalg.norm(object_velocity[:2])),"uncertainty_damping_weight":weight,"ego_translation_m":float(np.linalg.norm(target_pose[:3,3]-source_pose[:3,3])),"yaw_change_rad":yaw_change,"yaw_rate_rad_s":yaw_change/(delta/1000),"runtime_ms":runtimes[model],"future_geometry_input":False,"future_annotation_geometry_eval_only":True,**calculate_metrics(prediction,target_box,log.width,log.height)}
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--log-limit",type=int);parser.add_argument("--detector",choices=["yolo11n","yolo11s"]);parser.add_argument("--role",choices=["development","model_selection","heldout"]);parser.add_argument("--shard-index",type=int,default=0);parser.add_argument("--shard-count",type=int,default=1);args=parser.parse_args()
    roles={}
    for role,filename in (("development","av2_development_logs.txt"),("model_selection","av2_model_selection_logs.txt"),("heldout","av2_heldout_logs.txt")):
        for value in (ROOT/"configs"/filename).read_text().splitlines():
            if value.strip():roles[value.strip()]=role
    metadata=pd.read_csv(OUT/"cohort_source_metadata.csv");logs=[(roles[str(r.log_id)],str(r.split),str(r.log_id)) for r in metadata.itertuples(index=False) if str(r.log_id) in roles]
    if args.role:logs=[value for value in logs if value[0]==args.role]
    if not (0 <= args.shard_index < args.shard_count): raise ValueError("shard-index must be in [0, shard-count)")
    logs=[value for index,value in enumerate(logs) if index % args.shard_count == args.shard_index]
    if args.log_limit:logs=logs[:args.log_limit]
    pairs=pd.read_parquet(OUT/"latency_pairs.parquet");pairs["source_image_timestamp_ns"]=pairs.source_image_path.map(lambda value:int(Path(value).stem));pairs["target_image_timestamp_ns"]=pairs.target_image_path.map(lambda value:int(Path(value).stem))
    category_mapping=pd.read_csv(OUT/"class_mapping.csv").set_index("av2_category").analysis_group.to_dict();detectors=[args.detector] if args.detector else ["yolo11n","yolo11s"]
    for role,split,log_id in logs:
        log=LogData(split,log_id);log_pairs=pairs[pairs.log_id.astype(str)==log_id]
        for detector in detectors:
            destination=CORRECTED/detector/role/f"{log_id}.propagation.parquet"
            if destination.exists():
                existing=pd.read_parquet(destination,columns=["comparison_id","pred_x1"]);print(json.dumps({"log":log_id,"detector":detector,"reused":True,"rows":len(existing)}));continue
            payload=json.loads((OUT/"checkpoints"/detector/role/f"{log_id}.json").read_text());result=process_detector(log,payload,detector,role,log_pairs,category_mapping);atomic_parquet(destination,result);print(json.dumps({"log":log_id,"detector":detector,"reused":False,"rows":len(result)}))


if __name__=="__main__":main()
