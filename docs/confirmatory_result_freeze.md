# Confirmatory result freeze

Freeze date: 2026-07-18
Safety tag: `pre-object-motion-harm-analysis-2026-07`
Tagged commit: `a8b57d6cef6844e4c1045ea950de82427f89c017`

## Models

- **B3 (ego motion only):** back-project the source detector box using source-time LiDAR depth, transform its four corners from the source camera through measured AV2 ego poses into the target camera, and reproject; object displacement is exactly zero.
- **B5 (ego motion plus damped estimated object motion):** start from B3 and add a causal city-frame object displacement estimated from the two most recent associated historical detector boxes and their source-time depth estimates. The velocity is capped at 30 m/s and multiplied by the frozen damping product of LiDAR-point support, depth dispersion, and track age.

Neither deployable model uses target-time annotation geometry. Target annotations are evaluation-only.

## Cohort and endpoints

- Dataset: the fixed 150-log AV2 cohort listed in `configs/av2_development_logs.txt`, `configs/av2_model_selection_logs.txt`, and `configs/av2_heldout_logs.txt`.
- Splits: development, model selection, and held-out exactly as listed in those files.
- Detectors: YOLO11n and YOLO11s, using preserved detector JSON checkpoints.
- Nominal latency horizons: 100, 200, 300, 400, and 500 ms.
- Primary diagnostic contrast: 300-ms held-out high-yaw normalized center error, summarized with logs as the independent unit.
- Primary endpoint: paired relative difference in normalized center error, `error(B5) / error(B3) - 1`; positive values mean harm.
- Secondary endpoints: IoU and recall at IoU 0.3, 0.5, and 0.7.

## Frozen headline result

- YOLO11n: B5 is approximately 19.2% worse than B3.
- YOLO11s: B5 is approximately 21.1% worse than B3.
- Scientific result: valid AV2 confirmatory **NO-GO for B5**.

## Output integrity

Deterministic SHA-256 tree hashes (relative path plus file hash, lexicographically ordered):

- Corrected AV2 propagation, 300 Parquets: `44a425bdab4d9925035fc6047845e7ca5487a99cf0bfef515ec5ac52ad7de921`.
- AV2 detector checkpoints, 600 files: `59a7cbac682e45da3208eb4cfb2dd4ba95483c7b077f9f6dcf2a0372622252bf`.
- Invalid-run archive, 317 files: `b24fc60551d3d3ff5ab49311bfe80fa9f809e53421fb974c492864926a6e969a`.
- nuScenes 500-ms reproduction, 22 files: `cae2110da52b2924ac6f12e3f9d28930763bdcdaf383dfbcba9b02996371ea6c`.
- Frozen method YAML: `0e3576df28fe9ca154888415dd2b045ec0f4de552a6e57843b823b79587f9e48`.
- Development log list: `44ce2a8566becb1f93f2c7ae65578245d6d15a139fcaefc8100304c28bded983`.
- Model-selection log list: `412926b5c01e2db60db9854a000278cd76f391decf60994b7e503cde7a853d35`.
- Held-out log list: `15968906597b4826142e01d69fe8c282d84d67ae0a401dd221fe92fecf0fd5c7`.

The full machine-readable inventory is in `results/object_motion_harm/run_manifest.json`.
