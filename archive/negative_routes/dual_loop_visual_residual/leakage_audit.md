# Leakage audit

- Primary inputs are actual YOLO detections at t, real lidar at t, causal IMU/CAN through the arrival frame, past detections, and real images no later than t+500 ms.
- Target annotation boxes are used only for evaluation or as training targets in other scenes for B6.
- B6 uses leave-one-scene-out training.
- Source annotation identity is used only to associate a matched YOLO detection with the persistent target instance.
- Lidar fallback is trained on other scenes only.
- Oracle ego pose is diagnostic-only and excluded from feasibility criteria.
- No annotation interpolation, pseudo-labeling, future object velocity, or future annotation geometry is used as an input.

Status: PASSED.
