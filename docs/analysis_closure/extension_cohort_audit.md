# Extension Cohort Audit

## Cohort Availability
- **YOLO11n:** 45 detector-complete, 10 propagation-complete, 9 gate-eligible.
- **YOLO11s:** 45 detector-complete, 45 propagation-complete, 33 gate-eligible.
- **RT-DETR-L:** 0 completed (intentionally excluded).

## Missing Propagation Explanations
YOLO11n experienced an execution interruption during propagation, leaving 35 logs incomplete. RT-DETR-L was entirely omitted to save compute given the gate's prior failure.
Some logs were further excluded from gate eligibility due to a lack of valid 300ms high-yaw object comparisons in those specific sequences.

## Direct Comparability
Because YOLO11n and YOLO11s have different gate-eligible logs (9 vs 33), the extension cohorts are **not directly comparable** across detectors.

## Conclusion
The prospective gate evidence is incomplete, detector-imbalanced, and not cross-detector validated. No additional extension computation is required because the gate is already degenerate and invalid.
