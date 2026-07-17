import os
import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def compute_global_flow(img1_path, img2_path):
    # simple magnitude of optic flow
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return 0.0
    img1 = cv2.resize(img1, (640, 360))
    img2 = cv2.resize(img2, (640, 360))
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))

def get_closest_can(nusc_can, scene_name, message_name, sample_timestamp):
    try:
        messages = nusc_can.get_messages(scene_name, message_name)
    except:
        return None
    if not messages:
        return None
    # messages is list of dicts with 'utime'
    times = np.array([m['utime'] for m in messages])
    idx = np.argmin(np.abs(times - sample_timestamp))
    return messages[idx]

def main():
    print("Loading datasets...")
    sys.path.append(r"C:\work\auto\external\nuscenes-devkit\python-sdk")
    nusc = NuScenes(version='v1.0-mini', dataroot=r'C:\work\auto\data\nuscenes', verbose=False)
    nusc_can = NuScenesCanBus(dataroot=r'C:\work\auto\data\nuscenes')
    model = YOLO(r"C:\work\auto\yolo11n.pt")

    results = []
    
    # Process just a subset of scenes if it's too slow, but let's try to process all samples
    samples = nusc.sample
    print(f"Processing {len(samples)} samples...")
    
    for sample in tqdm(samples):
        scene = nusc.get('scene', sample['scene_token'])
        scene_name = scene['name']
        
        cam_front_token = sample['data']['CAM_FRONT']
        cam_front_data = nusc.get('sample_data', cam_front_token)
        img_path = os.path.join(nusc.dataroot, cam_front_data['filename'])
        timestamp = cam_front_data['timestamp']
        
        # Get GT boxes in image plane
        _, boxes, camera_intrinsic = nusc.get_sample_data(cam_front_token)
        # We only care about car/pedestrian etc but for a rapid feasibility study just count any box
        gt_count = len(boxes)
        
        # Run YOLO
        # suppress output
        yolo_res = model(img_path, verbose=False)[0]
        pred_boxes = yolo_res.boxes
        pred_count = len(pred_boxes)
        
        # simplified matching (not exact 3D-2D overlap, just generic counts for this study)
        # In a real study we do IOU matching, but here we simulate a reliability metric:
        # F1 proxy based on counts:
        tp = min(gt_count, pred_count)
        fp = pred_count - tp
        fn = gt_count - tp
        
        # Flow (using previous sample if exists)
        flow_mag = 0.0
        if sample['prev']:
            prev_sample = nusc.get('sample', sample['prev'])
            prev_cam = nusc.get('sample_data', prev_sample['data']['CAM_FRONT'])
            prev_img = os.path.join(nusc.dataroot, prev_cam['filename'])
            flow_mag = compute_global_flow(prev_img, img_path)
            
        # CAN bus
        steer_msg = get_closest_can(nusc_can, scene_name, 'steeranglefeedback', timestamp)
        steer_angle = steer_msg['value'] if steer_msg else 0.0
        
        veh_mon_msg = get_closest_can(nusc_can, scene_name, 'vehicle_monitor', timestamp)
        speed = veh_mon_msg['vehicle_speed'] if veh_mon_msg else 0.0
        
        pose_msg = get_closest_can(nusc_can, scene_name, 'pose', timestamp)
        accel_x = pose_msg['accel'][0] if pose_msg else 0.0
        rot_rate = pose_msg['rotation_rate'][2] if pose_msg else 0.0
        
        results.append({
            'scene': scene_name,
            'timestamp': timestamp,
            'gt_count': gt_count,
            'pred_count': pred_count,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'flow_mag': flow_mag,
            'steer_angle': steer_angle,
            'speed': speed,
            'accel_x': accel_x,
            'rot_rate': rot_rate
        })
        
    df = pd.DataFrame(results)
    
    # Calculate reliability label (1 = reliable, 0 = unreliable)
    # Let's say unreliable if fn > 0
    df['reliable'] = (df['fn'] == 0).astype(int)
    
    # Drop rows without CAN data if any
    df = df.fillna(0)
    
    # Model 1: Vision + IMU
    X_vis = df[['flow_mag', 'speed', 'accel_x']]
    y = df['reliable']
    
    # Model 2: Motor Commands
    X_mot = df[['steer_angle', 'rot_rate']]
    
    # Model 3: Both
    X_both = df[['flow_mag', 'speed', 'accel_x', 'steer_angle', 'rot_rate']]
    
    if len(np.unique(y)) < 2:
        print("Not enough variance in reliability for modeling.")
        auc_vis, auc_mot, auc_both = 0, 0, 0
    else:
        # evaluate models
        m_vis = LogisticRegression().fit(X_vis, y)
        m_mot = LogisticRegression().fit(X_mot, y)
        m_both = LogisticRegression().fit(X_both, y)
        
        auc_vis = roc_auc_score(y, m_vis.predict_proba(X_vis)[:, 1])
        auc_mot = roc_auc_score(y, m_mot.predict_proba(X_mot)[:, 1])
        auc_both = roc_auc_score(y, m_both.predict_proba(X_both)[:, 1])
        
    print(f"AUC Vision+IMU: {auc_vis:.3f}")
    print(f"AUC Motor: {auc_mot:.3f}")
    print(f"AUC Both: {auc_both:.3f}")
    
    out_dir = r"C:\work\auto\results\not_robotics_real_feasibility"
    os.makedirs(out_dir, exist_ok=True)
    df.to_parquet(os.path.join(out_dir, 'detector_outputs.parquet'))
    
    with open(os.path.join(out_dir, 'feasibility_study_report.md'), 'w') as f:
        f.write("# Feasibility Study Report\n")
        f.write(f"AUC Vision: {auc_vis:.3f}\n")
        f.write(f"AUC Motor: {auc_mot:.3f}\n")
        f.write(f"AUC Both: {auc_both:.3f}\n")
        
        improvement = auc_both - auc_vis
        if improvement > 0.05:
            f.write("\nConclusion: FEASIBILITY GO. Motor commands add significant predictive value.\n")
        else:
            f.write("\nConclusion: NO-GO. Motor commands do not add sufficient value (>5% AUC improvement).\n")
            
if __name__ == "__main__":
    main()
