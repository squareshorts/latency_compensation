import os
import sys
import hashlib
from PIL import Image
import json
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("Initializing NuScenes...")
    sys.path.append(r"C:\work\auto\external\nuscenes-devkit\python-sdk")
    
    nusc = NuScenes(version='v1.0-mini', dataroot=r'C:\work\auto\data\nuscenes', verbose=True)
    nusc_can = NuScenesCanBus(dataroot=r'C:\work\auto\data\nuscenes')

    scenes = nusc.scene
    samples = nusc.sample
    sample_datas = nusc.sample_data
    annotations = nusc.sample_annotation

    cam_front_datas = [sd for sd in sample_datas if sd['channel'] == 'CAM_FRONT']
    
    # Check physical presence
    cam_front_files = []
    missing_cams = 0
    for sd in cam_front_datas:
        path = os.path.join(nusc.dataroot, sd['filename'])
        if os.path.exists(path):
            cam_front_files.append(path)
        else:
            missing_cams += 1

    can_files = list(Path(r'C:\work\auto\data\nuscenes\can_bus').glob('**/*.json'))
    
    print(f"Scenes: {len(scenes)}")
    print(f"Samples: {len(samples)}")
    print(f"Sample_data records: {len(sample_datas)}")
    print(f"Annotations: {len(annotations)}")
    print(f"CAM_FRONT keyframes: {len(cam_front_datas)}")
    print(f"CAM_FRONT image files found: {len(cam_front_files)}")
    print(f"CAM_FRONT missing: {missing_cams}")
    print(f"CAN JSON files found: {len(can_files)}")

    if missing_cams > 0 or len(cam_front_files) == 0:
        print("HARD STOP: Real images are missing.")
        sys.exit(1)

    # Hash 10 images
    print("\nHashes of 10 real images:")
    for path in cam_front_files[:10]:
        print(f"{os.path.basename(path)}: {hash_file(path)}")
        # Check if valid image
        try:
            img = Image.open(path)
            img.verify()
        except Exception as e:
            print(f"Image {path} corrupted: {e}")
            sys.exit(1)

    # Hash 10 CAN files
    print("\nHashes of 10 real CAN files:")
    for path in can_files[:10]:
        print(f"{path.name}: {hash_file(path)}")
        with open(path) as f:
            try:
                json.load(f)
            except Exception as e:
                print(f"CAN JSON {path} corrupted: {e}")
                sys.exit(1)

    # Check YOLO
    yolo_path = r"C:\work\auto\yolo11n.pt"
    print(f"\nYOLO path: {yolo_path}")
    print(f"YOLO size: {os.path.getsize(yolo_path)} bytes")

    # CSV outputs (placeholder for audit files, user didn't ask us to produce the full contents in the python script, 
    # but we can write a simple audit CSV)
    with open(r'C:\work\auto\results\not_robotics_real_feasibility\data_presence_audit.csv', 'w') as f:
        f.write("metric,count\n")
        f.write(f"scenes,{len(scenes)}\n")
        f.write(f"images,{len(cam_front_files)}\n")
        f.write(f"can_files,{len(can_files)}\n")

    print("\nFEASIBILITY GO: Data is physically present and valid.")

if __name__ == '__main__':
    main()
