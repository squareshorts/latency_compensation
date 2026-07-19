# Scientific decision

## NO-GO for B5

The original propagation output remains invalid and preserved under `invalid_propagation_run`. The active decision is based only on the completed corrected 300/300 propagation rerun, preserved detector checkpoints, recovered yaw stratification, and full corrected negative controls.

At the frozen 300-ms held-out high-yaw endpoint, B5 is worse than the simpler ego-motion-only B3 on both detectors when logs are the independent unit:

- YOLO11n: 19.1807% higher normalized center error.
- YOLO11s: 21.0579% higher normalized center error.

B5 improves over stale and tracker baselines, but the prespecified method criterion requires the estimated object-motion term to add value beyond ego motion. It does not. The no-object-motion B3 control is better on both detectors, so estimated object-motion augmentation fails the frozen criterion.

This NO-GO is frozen as the primary AV2 confirmatory result. Subsequent component, oracle, operating-envelope, and cross-dataset analyses are diagnostic and cannot reverse or soften it.
