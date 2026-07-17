# AV2 data presence audit

- Official source: `s3://argoverse/datasets/av2/sensor/`.
- Official API: av2 0.3.6 import tested on Python 3.10.11.
- Fixed seed: 20260717.
- Fixed cohorts: 80 development train logs, 20 model-selection train logs, 50 held-out validation logs.
- Selection was frozen before sensor download using city plus pre-outcome speed, yaw, and traffic strata.
- Selective payload: ring_front_center, lidar, annotations, ego poses, and calibration only.
- Verified objects: 72557/72557.
- Verified bytes: 38058552798 (35.44 GiB).
- Payload status: DOWNLOADED AND SIZE-VERIFIED.
- Detector inference and outcome analysis status: NOT STARTED.

- Timing/pair construction status: COMPLETE; no interpolation; evaluation-only future geometry columns explicitly marked.

- Timing/pair construction status: COMPLETE; no interpolation; future geometry marked evaluation-only.
