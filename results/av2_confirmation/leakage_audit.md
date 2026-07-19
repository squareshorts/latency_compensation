# AV2 leakage and provenance audit

## Status

**NOT PASSED FOR CONFIRMATORY USE — ORIGINAL PROPAGATION RUN INVALID.**

## Verified

- The 80 development, 20 model-selection, and 50 held-out log IDs are disjoint.
- Both detectors contain the same fixed 150-log cohort and five latency horizons.
- All 12,889,242 original metric rows set `future_geometry_input=False`.
- Source-code tracing finds no future detector box, future object state, or
  future annotation geometry in the original prediction-coordinate branches.
  Target annotations are read for evaluation only.
- Official pose-based yaw joined to 12,887,082/12,889,242 metric rows
  (99.983%). The threshold was derived only from 12,411 unique 300-ms
  development frame pairs.
- Reconstruction yields 2,148,207 unique object-comparison ordinals; all six
  model labels are present for each retained comparison.

## Invalid-run provenance defects

- The original Parquets omit comparison IDs, predicted/source/target
  coordinates, target timestamps, evaluation track identity, depth, track
  history, pose transforms, and branch runtimes.
- B0/B3 and B1/B2/B4 coordinate arrays are duplicated by the generator.
- Source detector history is keyed by class ID rather than associated track.
- Target projection omits the calibrated AV2 egovehicle-to-camera transform.

No direct future-information leak was found, but the formal confirmatory
leakage criterion cannot pass for an algorithmically invalid run with missing
row-level provenance. Corrected checkpoints retain the required provenance
fields and keep future annotation geometry evaluation-only. The corrected
300/300 rerun has zero future-geometry-input rows, disjoint roles, unique
comparison/model keys, and complete B0–B5 sets; its technical leakage audit
passes. It cannot retroactively restore the original run's confirmatory status
after held-out unblinding.
