# Latency-Compensated Detection

This repository evaluates causal post-processing for detector boxes that become stale while an image is being processed. It transports a source-time two-dimensional detection into the camera frame at detector availability using measured ego motion, source-time LiDAR depth, and—when enabled—object motion estimated from detector history.

## Frozen confirmatory result

The fixed Argoverse 2 analysis contains 80 development, 20 model-selection, and 50 held-out logs; preserved YOLO11n and YOLO11s outputs; and real 100–500 ms target timestamps. At the prespecified 300-ms high-yaw endpoint, universally adding uncertainty-damped estimated object motion (B5) was worse than ego-motion-only propagation (B3):

- YOLO11n: 19.1807% greater normalized center error;
- YOLO11s: 21.0579% greater normalized center error.

Both paired log-cluster bootstrap intervals exclude zero. Oracle future object motion improves over B3, so the negative result is an estimation failure—not evidence that object motion is physically irrelevant. Velocity-sign and association errors are the leading diagnosed mechanisms. The confirmatory decision is **NO-GO for universal B5** and **PAPER GO for the negative result**.

## Submission-strengthening analysis

The `revision/sivp-submission-strengthening` branch adds:

- stale, image-velocity, SORT, ByteTrack-style, and OC-SORT-style causal prediction baselines;
- persistent-object and full-frame current-time detection metrics;
- one prespecified architecture-distinct RT-DETR-L robustness analysis;
- one L2-logistic causal gate frozen before evaluation on 45 untouched AV2 validation logs;
- log-cluster bootstrap, Wilcoxon sensitivity, failure-stratum, provenance, leakage, and independent recalculation outputs.

On the exact frozen matched-comparison set, OC-SORT prediction is the strongest non-ego tracker, but B3 remains substantially better at the primary endpoint for both current detectors. The original B3/B5 tables are protected by hashes in `docs/sivp_strengthening_freeze.md` and are never rewritten by strengthening scripts.

## Information boundary

Deployable predictions may use detector outputs, LiDAR, calibration, measured ego poses, and histories timestamped no later than the source time. Target annotations define evaluation endpoints only. No tracker receives a target-time detection. Oracle diagnostics are explicitly marked and excluded from deployment claims.

## Reproduction

With licensed/selectively downloaded benchmark files and ignored detector weights available locally:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\reproduce_sivp_strengthening.ps1
```

The test-only command is:

```powershell
$env:PYTHONPATH='C:\work\auto\src'
.\.venv-not-robotics\Scripts\python.exe -m pytest -q
```

The analysis is resumable at detector/log boundaries. Raw AV2 and nuScenes files, detector weights, large low-level checkpoints, and licensed payloads are excluded from version control. Aggregate tables, configurations, manifests, hashes, and audit records are release artifacts.

## Repository map

- `configs/`: frozen cohorts, propagation method, extension cohort, and causal-gate specification;
- `src/latency_compensation/`: reusable geometry, statistics, and causal tracker code;
- `scripts/`: inference, propagation, aggregation, gate, sensitivity, and audit entry points;
- `results/av2_confirmation/`: original AV2 confirmation inputs and frozen decision;
- `results/object_motion_harm/`: component-substitution and reversal diagnosis;
- `results/sivp_strengthening/`: reviewer-facing tracker, end-to-end, architecture, gate, and audit outputs;
- `docs/`: scientific freezes and reproduction notes.

## Limitations

The tracker adapters isolate prediction under a fixed detection stream; they are not claims about full unmodified tracking systems with appearance embeddings or target-time observation updates. The architecture analysis is limited to one prespecified non-YOLO detector. Dataset licenses prevent redistribution of raw benchmark payloads. Repository URL and archival DOI remain unset until external release publication is completed.
