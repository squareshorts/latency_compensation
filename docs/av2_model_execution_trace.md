# AV2 model execution trace for the invalid propagation run

## Finding

The completed propagation run did not execute the frozen B0–B5 method. The
only invoked propagation branch was
[`propagate_log`](../src/latency_compensation/run_confirmation.py#L105), called
from `main` at lines 187–190. Its six outputs are assembled at line 149.

| Model | Executed branch | Frozen component actually called | Result |
|---|---|---|---|
| B0 | `run_confirmation.py:143,149` (`stale = box.copy()`) | stale detector box | Implemented |
| B1 | `run_confirmation.py:144–147,149` | ad-hoc previous same-class box delta | Not the frozen associated image track |
| B2 | `run_confirmation.py:149` (`kalman = velocity`) | none | Exact duplicate of B1 |
| B3 | `run_confirmation.py:149` (`ego_motion_only = stale`) | none | Exact duplicate of B0 |
| B4 | `run_confirmation.py:149` (`B4 = velocity`) | none | Exact duplicate of B1 |
| B5 | `run_confirmation.py:148–149` | fixed 0.65 coordinate blend | Not uncertainty-aware damping |

## Component-by-component trace

- Ego-motion transform: no ego-pose file is loaded by `propagate_log`; the
  available causal integrator in `src/latency_compensation/ego_motion.py:6–19`
  is never imported or called. B3 is replaced by stale coordinates at
  `run_confirmation.py:149`.
- Image-space velocity: lines 144–147 subtract the immediately preceding box
  with the same class ID. The `history` key is class ID (`lines 131–132`), not
  an associated object track, so multiple objects of a class contaminate one
  another.
- Kalman prediction: `BoxKalmanFilter` is implemented in
  `src/latency_compensation/object_motion.py:16–46` but is never imported or
  called. B2 receives the B1 array at `run_confirmation.py:149`.
- Depth use: neither source-time lidar nor calibration is loaded. The frozen
  `median_points_inside_detection` depth branch has no implementation call in
  this runner.
- Object-motion extrapolation: the historical 3D constant-velocity branch is
  absent. B4 receives B1 at `run_confirmation.py:149`.
- Uncertainty damping: `uncertainty_damping` exists at
  `src/latency_compensation/propagation.py:13–17` but is never called. B5 uses
  the constant 0.65 multiplier at `run_confirmation.py:148`.
- Final box reconstruction: B0–B5 arrays are read directly at lines 149–153;
  there is no calibrated camera reconstruction. No later clipping or join
  causes the ties—the duplicated coordinates are selected before metrics.

## Invocation and input checks

- `propagate_log` was invoked for all 300 detector/log checkpoints, producing
  12,889,242 model-metric rows.
- Detector inputs were nonempty (891,266 preserved detections across 96,502
  detector frames).
- The absent ego, depth, Kalman, 3D object-motion, and damping branches cannot
  receive nonempty inputs because they are not called.
- B0/B3 and B1/B2/B4 are separate dictionary/model labels but share identical
  coordinate values. The exact ties are therefore generated duplication, not
  a bad aggregate join.
- The metric Parquets omit coordinates and comparison IDs, preventing direct
  coordinate verification without the preserved detector checkpoints and
  generator reconstruction.

## Separate target-projection defect

`project_targets` at `run_confirmation.py:77–93` treats AV2 annotation
coordinates in the egovehicle frame as though they were already in the camera
frame. It uses fixed intrinsics and does not apply the calibrated
egovehicle-to-camera transform or cuboid-corner projection. This explains the
implausible approximately 60,000-pixel errors and zero median IoU in the
premature aggregate report.

## Classification

The original propagation outputs are **INVALID RUN** artifacts. They must not
be used for a confirmatory scientific decision.
