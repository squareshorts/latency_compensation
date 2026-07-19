# Reversal recalculation audit

- Source Parquets: 300 corrected propagation files.
- Independent unit: log.
- Bootstrap: 10,000 paired log-cluster replicates.
- High-yaw threshold: 0.0169726458348836 rad/s, frozen from development logs.
- Detector inference: not run.
- Target-time geometry: evaluation metrics only.

## Primary 300-ms held-out high-yaw contrast

- yolo11n: B5 disadvantage 19.180734% (95% log-cluster bootstrap CI 10.799502% to 37.046298%); B5 better in 6/40 logs.
- yolo11s: B5 disadvantage 21.057881% (95% log-cluster bootstrap CI 8.301572% to 31.512530%); B5 better in 8/40 logs.
