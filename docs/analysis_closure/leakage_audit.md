# Leakage Audit

## Status
**PASS WITH LIMITATIONS**

## Audit Details
- **Trackers (T0-T4):** Validated. No future information used during prediction.
- **B3 (Ego-motion only):** Validated. Uses strictly rigid body equations with past ego-motion.
- **B5 (Estimated object motion):** Validated. Relies entirely on historical tracking to estimate velocities.
- **RT-DETR-L Checkpoints:** Validated. Log files are processed causally, frame-by-frame. No future detections input.
- **Causal Gate:** Validated. Feature set restricted to source-time or past features.
- **Diagnostic Mechanism Analysis:** Validated. Uses target-time ground truth solely for retrospective diagnosis, never for prospective predictions.
