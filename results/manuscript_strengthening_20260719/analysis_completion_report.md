# Analysis Completion Report

1. **CLASS BREAKDOWN BOOTSTRAP INTERVALS**: Repaired in `class_breakdown_bootstrap_complete.csv`. RT-DETR-L class-stratified analysis was unavailable because `object_level_diagnostics.parquet` does not contain `rtdetr_l` object rows (only yolo11n and yolo11s).
2. **ASSOCIATION IOU-THRESHOLD SENSITIVITY**: Analysis unavailable because the raw AV2 dataset files (e.g., `city_SE3_egovehicle.feather`) required to project camera pixels to city coordinates for re-propagation are missing from the disk environment. Detailed in `iou_sensitivity_complete.csv`.
3. **VELOCITY-SIGN AGREEMENT**: Repaired in `velocity_sign_agreement_complete.csv`. Analyzes the dot product sign (cosine) between estimated 2D ground-plane velocity and true displacement.
4. **RUNTIME BREAKDOWN**: Repaired in `runtime_instrumentation_complete.csv`. Extracted from `runtime_recalculated.csv` and `runtime_metrics.csv`.
5. **VALIDATION**: Frozen primary results did not change, as the analyses purely aggregated existing frozen diagnostics. No synthetic placeholders remain.

## README (Column Explanations)
- **class_breakdown_bootstrap_complete.csv**:
  - `detector`: Object detector used.
  - `class`: Object class (vehicle, pedestrian/cyclist).
  - `eligible_logs`: Number of heldout logs containing this class.
  - `objects`: Total objects of this class across all eligible logs.
  - `median_paired_relative_diff`: Median of the relative B5-B3 error per log.
  - `ci_low_2.5` / `ci_high_97.5`: 95% paired log-cluster bootstrap interval.
  - `logs_favoring_B5`: Count of logs where B5 error < B3 error.
  - `percent_logs_favoring_B5`: Percentage of logs favoring B5.

- **velocity_sign_agreement_complete.csv**:
  - `motion_component`: The evaluated motion vector dimension (2D ground-plane).
  - `sign_evaluation`: How sign agreement is defined (cosine > 0).
  - `dead_zone_definition`: Speed below which velocity direction is ambiguous.
  - `same_sign_count` / `opposite_sign_count`: Confusion counts.
  - `ambiguous_zero_count`: Objects within the dead-zone.
  - `same_sign_fraction` / `opposite_sign_fraction`: Row-normalized percentages (excluding ambiguous).
  - `ambiguous_fraction`: Fraction of total objects falling in the dead-zone.
  - `eligible_objects`: Objects outside the dead-zone.
  - `excluded_objects`: Objects inside the dead-zone.
  - `exclusion_reason`: Reason for exclusion.

- **runtime_instrumentation_complete.csv**:
  - `component`: Stage of processing (e.g., detector inference, B3 overhead).
  - `detector`: Detector model context.
  - `median_ms` / `p95_ms`: Timing percentiles in milliseconds.
  - `observations`: Number of iterations measured.
  - `timing_unit`: Whether timing is per frame or per object.
  - `warmup_iterations`: Iterations discarded before timing.
  - `CPU` / `GPU` / `OS` / `Python`: Hardware and environment metadata.
  - `inference_excluded` / `io_excluded`: Contextual exclusions for overhead metrics.
