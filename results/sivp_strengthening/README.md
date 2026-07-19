# SIVP strengthening outputs

This directory contains reviewer-facing analyses added after the confirmatory B3-versus-B5 result was frozen. The original primary result tables under `results/av2_confirmation` and `results/object_motion_harm` are inputs and are not rewritten.

Tracker labels are fixed as follows:

- T0: stale detector box;
- T1: constant image-space velocity;
- T2: SORT constant-velocity Kalman prediction;
- T3: causal ByteTrack-style two-stage association and prediction;
- T4: causal observation-centric SORT prediction;
- T5: frozen B3 ego-motion-only propagation;
- T6: frozen B5 uncertainty-damped ego-plus-estimated-object-motion propagation.

All tracker updates stop at the source timestamp. Target-time annotations are stored only in evaluation tables. Low-level Parquet shards are release-excluded because they are large derived intermediates; aggregate CSVs, manifests, hashes, audits, and scripts are the archival results.

The untouched extension contains 45 AV2 validation logs selected from metadata only with seed 20260719 and is disjoint from the original 150-log cohort. Raw AV2 data and detector weights remain excluded under dataset and checkpoint licensing terms.
