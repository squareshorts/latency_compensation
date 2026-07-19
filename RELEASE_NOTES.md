# Version 1.0.0 â€” Object-Motion Harm Analysis

This release supports the study:

**When More Motion Modeling Hurts: Causal Latency Compensation for Streaming Object Detection**

## Principal result

At the frozen 300-ms high-yaw endpoint, universal ego-plus-estimated-object-
motion propagation increased normalized center error relative to ego-motion-
only propagation by:

- 19.1807% for YOLO11n;
- 21.0579% for YOLO11s;
- 16.5966% for RT-DETR-L.

Ego-motion-only propagation also outperformed stale boxes, constant image
velocity, SORT, ByteTrack, and OC-SORT prediction modules on the matched
primary comparisons.

## Included

- frozen configurations and cohort manifests;
- reusable propagation and causal-tracker code;
- analysis and audit scripts;
- compact aggregate result tables;
- paired log-cluster bootstrap outputs;
- leakage and provenance audits;
- independent headline recalculation;
- analysis-closure documentation and tests.

## Excluded

- raw AV2 and nuScenes data;
- detector weights;
- licensed benchmark payloads;
- large low-level detector and propagation checkpoints;
- local virtual environments and caches.
