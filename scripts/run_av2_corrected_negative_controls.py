"""Prespecified controls on corrected 300-ms high-yaw held-out comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


ROOT=Path(r"C:\work\auto");OUT=ROOT/"results"/"av2_confirmation";DATA=ROOT/"data"/"av2"/"sensor";CAM="ring_front_center";SEED=20260717
COORDS=["pred_x1","pred_y1","pred_x2","pred_y2"]


def matrix(row):
    value=np.eye(4);value[:3,:3]=Rotation.from_quat([row.qx,row.qy,row.qz,row.qw]).as_matrix();value[:3,3]=[row.tx_m,row.ty_m,row.tz_m];return value


class Geometry:
    def __init__(self,split,log_id):
        root=DATA/split/log_id;poses=pd.read_feather(root/"city_SE3_egovehicle.feather");self.poses={int(r.timestamp_ns):matrix(r) for r in poses.itertuples(index=False)};self.timestamps=np.asarray(sorted(self.poses),dtype=np.int64)
        intr=pd.read_feather(root/"calibration/intrinsics.feather").set_index("sensor_name").loc[CAM];self.width=int(intr.width_px);self.height=int(intr.height_px);self.K=np.asarray([[intr.fx_px,0,intr.cx_px],[0,intr.fy_px,intr.cy_px],[0,0,1]],float)
        ext=pd.read_feather(root/"calibration/egovehicle_SE3_sensor.feather").set_index("sensor_name").loc[CAM];self.ego_from_cam=matrix(ext);self.cam_from_ego=np.linalg.inv(self.ego_from_cam)
    def nearest(self,timestamp):return int(self.timestamps[np.argmin(np.abs(self.timestamps-int(timestamp)))])
    def relative(self,source,target):return self.cam_from_ego@np.linalg.inv(self.poses[int(target)])@self.poses[int(source)]@self.ego_from_cam
    def propagate_relative(self,boxes,depths,transforms):
        output=np.empty_like(boxes)
        for i,(box,depth,transform) in enumerate(zip(boxes,depths,transforms)):
            pixels=np.asarray([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]]]);points=np.c_[(pixels[:,0]-self.K[0,2])/self.K[0,0]*depth,(pixels[:,1]-self.K[1,2])/self.K[1,1]*depth,np.full(4,depth),np.ones(4)]
            camera=(transform@points.T).T[:,:3];valid=camera[:,2]>.1
            if valid.sum()<2:output[i]=box;continue
            uvw=(self.K@camera[valid].T).T;uv=uvw[:,:2]/uvw[:,2:3];output[i]=[uv[:,0].min(),uv[:,1].min(),uv[:,0].max(),uv[:,1].max()]
        output[:,[0,2]]=np.clip(output[:,[0,2]],0,self.width-1);output[:,[1,3]]=np.clip(output[:,[1,3]],0,self.height-1);return output


def metrics(pred,target,width,height):
    pc=np.c_[(pred[:,0]+pred[:,2])/2,(pred[:,1]+pred[:,3])/2];tc=np.c_[(target[:,0]+target[:,2])/2,(target[:,1]+target[:,3])/2];error=np.linalg.norm(pc-tc,axis=1)
    inter_w=np.maximum(0,np.minimum(pred[:,2],target[:,2])-np.maximum(pred[:,0],target[:,0]));inter_h=np.maximum(0,np.minimum(pred[:,3],target[:,3])-np.maximum(pred[:,1],target[:,1]));inter=inter_w*inter_h;pa=np.maximum(0,pred[:,2]-pred[:,0])*np.maximum(0,pred[:,3]-pred[:,1]);ta=np.maximum(0,target[:,2]-target[:,0])*np.maximum(0,target[:,3]-target[:,1]);overlap=inter/np.maximum(pa+ta-inter,1e-12)
    return error,error/np.hypot(width,height),overlap


def main():
    threshold=json.loads((OUT/"high_yaw_threshold.json").read_text())["threshold_abs_yaw_rate_rad_s"]
    metadata=pd.read_csv(OUT/"cohort_source_metadata.csv");split_map=metadata.set_index("log_id").split.astype(str).to_dict();heldout=sorted(path.stem.split('.')[0] for path in (OUT/"corrected_checkpoints/yolo11n/heldout").glob("*.parquet"))
    pairs=pd.read_parquet(OUT/"latency_pairs.parquet");pairs["source_image_timestamp_ns"]=pairs.source_image_path.map(lambda value:int(Path(value).stem));pairs["target_image_timestamp_ns"]=pairs.target_image_path.map(lambda value:int(Path(value).stem));pairs=pairs[(pairs.split=="heldout")&(pairs.delta_ms==300)]
    donor_log=heldout[0];donor_geometry=Geometry(split_map[donor_log],donor_log);donor_pair=pairs[pairs.log_id.astype(str)==donor_log].iloc[len(pairs[pairs.log_id.astype(str)==donor_log])//2];donor_relative_ego=np.linalg.inv(donor_geometry.poses[int(donor_pair.target_image_timestamp_ns)])@donor_geometry.poses[int(donor_pair.source_image_timestamp_ns)]
    per_log=[];rng=np.random.default_rng(SEED)
    for detector in ("yolo11n","yolo11s"):
        for path in sorted((OUT/"corrected_checkpoints"/detector/"heldout").glob("*.parquet")):
            frame=pd.read_parquet(path);frame=frame[(frame.delta_ms==300)&(frame.yaw_rate_rad_s.abs()>=threshold)]
            if frame.empty:continue
            meta=frame[frame.model=="B0"].set_index("comparison_id");pivot=frame.pivot(index="comparison_id",columns="model",values=COORDS);keys=meta.index.intersection(pivot.index);meta=meta.loc[keys];pivot=pivot.loc[keys]
            def boxes(model):return np.column_stack([pivot[(coordinate,model)].to_numpy() for coordinate in COORDS])
            b0,b3,b4,b5=boxes("B0"),boxes("B3"),boxes("B4"),boxes("B5");effect=b4-b3;target=meta[["target_x1","target_y1","target_x2","target_y2"]].to_numpy(float);depth=meta.estimated_depth_m.to_numpy(float);geometry=Geometry(split_map[str(meta.log_id.iloc[0])],str(meta.log_id.iloc[0]))
            current=[geometry.relative(int(source),int(target_ts)) for source,target_ts in zip(meta.source_timestamp_ns,meta.target_timestamp_ns)];shifted=[geometry.relative(int(source),geometry.nearest(int(target_ts)+100_000_000)) for source,target_ts in zip(meta.source_timestamp_ns,meta.target_timestamp_ns)];donor_transform=geometry.cam_from_ego@donor_relative_ego@geometry.ego_from_cam;donor=[donor_transform]*len(meta)
            perm=rng.permutation(len(meta));mismatch=np.roll(np.arange(len(meta)),17)
            controls={"valid_B4":b4,"valid_B5":b5,"time_shifted_ego_pose":geometry.propagate_relative(b0,depth,shifted)+effect,"ego_motion_from_another_log":geometry.propagate_relative(b0,depth,donor)+effect,"shuffled_object_velocity":b3+effect[perm],"reversed_object_velocity":b3-effect,"incorrect_depth":geometry.propagate_relative(b0,depth*2,current)+effect,"mismatched_track_history":b3+effect[mismatch],"no_ego_motion_correction":b0+effect,"no_object_motion_correction":b3}
            for control,prediction in controls.items():
                prediction[:,[0,2]]=np.clip(prediction[:,[0,2]],0,geometry.width-1);prediction[:,[1,3]]=np.clip(prediction[:,[1,3]],0,geometry.height-1);error,norm,overlap=metrics(prediction,target,geometry.width,geometry.height)
                per_log.append({"detector":detector,"log_id":str(meta.log_id.iloc[0]),"delta_ms":300,"yaw_stratum":"high_yaw","control":control,"comparisons":len(meta),"median_normalized_center_error":np.median(norm),"median_center_error_px":np.median(error),"median_iou":np.median(overlap),"recall_iou_0_3":np.mean(overlap>=.3),"recall_iou_0_5":np.mean(overlap>=.5),"recall_iou_0_7":np.mean(overlap>=.7)})
    detail=pd.DataFrame(per_log);summary=detail.groupby(["detector","delta_ms","yaw_stratum","control"],as_index=False).agg(logs=("log_id","nunique"),comparisons=("comparisons","sum"),median_normalized_center_error=("median_normalized_center_error","median"),median_center_error_px=("median_center_error_px","median"),median_iou=("median_iou","median"),recall_iou_0_3=("recall_iou_0_3","median"),recall_iou_0_5=("recall_iou_0_5","median"),recall_iou_0_7=("recall_iou_0_7","median"))
    summary["same_eligible_comparisons"]=True;summary["scope"]="corrected_300ms_high_yaw_heldout_full_eligible_set";valid=summary[summary.control=="valid_B4"].set_index("detector").median_normalized_center_error.to_dict();summary["valid_B4_outperforms_control"]=summary.apply(lambda row:valid[row.detector]<row.median_normalized_center_error if row.control not in {"valid_B4","valid_B5"} else np.nan,axis=1)
    summary.to_csv(OUT/"negative_controls_recovered.csv",index=False,float_format="%.12g");detail.to_csv(OUT/"negative_controls_per_log.csv",index=False,float_format="%.12g")
    print(summary.to_string(index=False))


if __name__=="__main__":main()
