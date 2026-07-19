# Final AV2 confirmatory summary

- Corrected propagation checkpoints: 300/300 complete.
- Detector inference rerun: no.
- Structural/coordinate audit: pass.
- Frozen endpoint: 300-ms held-out high-yaw normalized center error, logs independent.
- YOLO11n B5 disadvantage versus B3: 19.1807%.
- YOLO11s B5 disadvantage versus B3: 21.0579%.
- Result: **NO-GO for B5**.

The preserved invalid-run archive is not used for this decision. B5's improvements over stale and tracker baselines do not rescue the method because its estimated object-motion contribution is harmful relative to B3.
