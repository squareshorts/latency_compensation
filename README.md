# Latency-Compensated Detection

## Problem

Streaming detectors return boxes after the camera has moved. This project develops a detector-agnostic, retraining-free post-processing layer that transforms a detection from its acquisition frame into the camera frame current when that detection becomes available.

## nuScenes real-data feasibility result

The nuScenes real-data feasibility result contains 339 annotated frame pairs, 1,487 paired detector-object observations, and 10 complete scenes. In high-yaw frames, stale boxes had 62.950 px median center error and 0.2572 median IoU. Causal ego-motion plus historical object-motion compensation reduced error to 20.461 px and raised IoU to 0.5147. The nuScenes independent reproduction and baseline-comparison analysis exactly reproduced the result and added a 10,000-replicate paired scene bootstrap.

## Proposed method

The active method combines measured camera motion, source-time lidar depth, and object velocity estimated from historical detector boxes. It never consumes annotation geometry or object state after detector availability. Conceptually, it is inspired by rapid vestibular compensation for self-motion.

## Baselines

- N0: stale box.
- N1: constant image-space velocity.
- N2: Kalman image-space extrapolation.
- N3: tracker prediction without a new detection.
- N4: ego-motion-only geometry.
- N5: ego-motion plus historical object motion.
- N6: future-object-motion oracle, diagnostic only.

## AV2 confirmation design

The frozen plan uses official Argoverse 2 Sensor data, two real detectors, complete-log grouping, and genuine annotated endpoints at 100–500 ms. The primary endpoint is normalized center error at 300 ms in high-yaw held-out logs. Acquisition is blocked until a fixed official cohort fits available storage without violating the minimum 30 held-out-log requirement.

## Causal-data rules

Only detections, lidar, poses, vehicle state, and images timestamped no later than detector availability are model inputs. Future annotations establish identity and evaluation endpoints only. Annotation interpolation, pseudo-labeling, future velocity, and object-level significance tests are prohibited.

## Reproduction

```powershell
$env:PYTHONPATH='C:\work\auto\src'
& 'C:\work\auto\.venv-not-robotics\Scripts\python.exe' 'C:\work\auto\scripts\reproduce_nuscenes_reproduction.py'
& 'C:\work\auto\.venv-not-robotics\Scripts\python.exe' -m pytest
& 'C:\work\auto\.venv-not-robotics\Scripts\python.exe' -m latency_compensation.run_confirmation --config 'C:\work\auto\configs\av2_confirmation.yaml' --smoke-test
```

## Results

The nuScenes independent reproduction and baseline-comparison analysis is under `results/nuscenes_500ms_reproduction`. AV2 confirmatory analysis outputs will be written to `results/av2_confirmation` only after the fixed cohort is recorded.

## Limitations

The nuScenes result is a pilot only in evidential scope: it uses 10 mini scenes, one detector, source-annotation-assisted identity matching, and a 500 ms horizon. It is not a publication-level external confirmation. AV2 generalization, detector-only association, multiple detectors, and the 300 ms endpoint remain open.

## Archived routes

Closed reliability-gating, visual-residual, annotation-cadence, biological, and invalid-synthetic routes are isolated under `archive`. They are preserved as negative evidence and are not active objectives.
