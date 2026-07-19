# Duplicate prediction audit

## Classification

**INVALID RUN — the propagation outputs are degenerate.**

The original metric Parquets were inspected at row level. They do not contain
predicted coordinates or object-comparison IDs. To quantify the defect, this
audit reconstructed the exact coordinate arrays produced by the generating
branch from the preserved detector JSON and joined them to original Parquet row
order. This is possible because each comparison has the fixed six-model order.

## Explicit model-pair tests

- B0 versus B1: identical ordered-coordinate hashes in 0/30 detector/split/latency groups.
- B0 versus B3: identical ordered-coordinate hashes in 30/30 detector/split/latency groups.
- B1 versus B4: identical ordered-coordinate hashes in 30/30 detector/split/latency groups.
- B3 versus B4: identical ordered-coordinate hashes in 0/30 detector/split/latency groups.
- B4 versus B5: identical ordered-coordinate hashes in 0/30 detector/split/latency groups.

## Cause

`run_confirmation.py` assigns B2 and B4 directly from the B1 `velocity` array,
and assigns B3 from the B0 `stale` array. It never calls the repository's
Kalman filter, ego-motion transform, lidar-depth path, historical 3D
object-motion path, or uncertainty-damping function. B5 is a fixed 0.65 blend
rather than the frozen uncertainty-aware branch. The exact ties are duplicated
model outputs, not a scientific result.

## Schema defect

All 300 original Parquets omit `comparison_id`, target/source box coordinates,
track identity, pose/depth provenance, and motion inputs. Consequently the
stored metric values cannot independently prove which coordinates generated
them; the generator reconstruction and source trace establish the defect.
