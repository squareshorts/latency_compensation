# Leakage audit

- CAN/IMU lookup uses `bisect_right(timestamp) - 1`; every retained message timestamp is less than or equal to its camera timestamp.
- Detector availability uses the latest real camera frame at or before `t + delay`; no later image is used.
- Object identities are populated only on genuine annotated keyframes.
- No annotation interpolation, pseudo-labeling, synthetic detections, or future object velocity is used.
- The intended validation grouping is complete nuScenes scenes.

Status: PASSED for the completed timeline and timing audit. Propagation-model leakage cannot be audited because those models were not run.
