# Reuse audit

- The recovered detector-error reliability study remains untouched as a negative audit under `C:\work\auto\results\not_robotics_real_feasibility`.
- Reused 2,342 verified real CAM_FRONT images, 404 annotated keyframes, 18,538 keyframe annotations, calibrated camera records, per-frame ego poses, parsed CAN streams, and 8,946 real YOLO11n keyframe predictions.
- No archive download or full archive-integrity test was repeated.
- No image, detection, pose, CAN record, object box, or annotation was simulated or interpolated.
- Algorithm latency is represented only by causal source-to-real-frame timing pairs.

## Endpoint observability

nuScenes v1.0-mini supplies object annotations and persistent instance tokens only on keyframes. Intermediate CAM_FRONT sweeps have real images, calibrated sensors, and ego poses, but no object boxes or instance identities. Consequently, a box-propagation endpoint is evaluable only when both the acquisition frame and the frame current at detector availability are annotated keyframes.

```csv
latency_ms,both_endpoints_annotated_pairs,pairs_with_persistent_visible_instance,persistent_instance_comparisons
100,0,0,0
200,0,0,0
300,0,0,0
400,8,8,449
500,339,339,14713
```

At the prespecified 300 ms endpoint, the number of real pairs with annotations and persistent visible instances at both ends is **0**. Interpolating boxes or identities into sweeps would simulate missing observations and was therefore not performed.
