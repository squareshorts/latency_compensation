# Repository Status

## `git status --short`
```
 M .gitignore
 M CITATION.cff
 M README.md
 M pyproject.toml
 M results/av2_confirmation/run_manifest.json
 M results/av2_confirmation/scientific_decision.md
 M src/latency_compensation/run_confirmation.py
 M tests/test_core.py
?? analysis_manifest.txt
?? configs/av2_extension_logs.txt
?? configs/causal_gate_frozen.yaml
?? disk_cleanup_20260718_172712.txt
?? disk_usage_report.csv
?? docs/analysis_closure/
?? docs/audience_prior_map.md
?? docs/av2_model_execution_trace.md
?? docs/candidate_abstract.md
?? docs/candidate_titles.md
?? docs/claim_calibration_table.csv
?? docs/confirmatory_result_freeze.md
?? docs/epistemic_surprise_claim.md
?? docs/journal_positioning.md
?? docs/manuscript_outline.md
?? docs/object_motion_harm_analysis_plan.md
?? docs/sivp_compute_freeze.md
?? docs/sivp_reproduction.md
?? docs/sivp_strengthening_freeze.md
?? recover_src/
?? results/analysis_closure/
?? results/av2_confirmation/aggregate_metrics.csv
?? results/av2_confirmation/bootstrap_intervals.csv
?? results/av2_confirmation/corrected_propagation_integrity.csv
?? results/av2_confirmation/corrected_propagation_integrity.md
?? results/av2_confirmation/corrected_row_level_agreement.csv
?? results/av2_confirmation/detector_comparison.csv
?? results/av2_confirmation/duplicate_prediction_audit.md
?? results/av2_confirmation/final_results_summary.md
?? results/av2_confirmation/final_results_summary_PREMATURE.md
?? results/av2_confirmation/heldout_primary_metrics.csv
?? results/av2_confirmation/high_yaw_metrics.csv
?? results/av2_confirmation/high_yaw_threshold.json
?? results/av2_confirmation/latency_comparison.csv
?? results/av2_confirmation/leakage_audit.md
?? results/av2_confirmation/metric_recalculation.csv
?? results/av2_confirmation/model_comparison.csv
?? results/av2_confirmation/model_coordinate_hashes.csv
?? results/av2_confirmation/motion_input_audit.csv
?? results/av2_confirmation/motion_input_distributions.csv
?? results/av2_confirmation/negative_controls.csv
?? results/av2_confirmation/negative_controls_audit_sample.csv
?? results/av2_confirmation/negative_controls_per_log.csv
?? results/av2_confirmation/negative_controls_recovered.csv
?? results/av2_confirmation/object_motion_contribution.csv
?? results/av2_confirmation/per_log_metrics.csv
?? results/av2_confirmation/propagation_distinctness_audit.csv
?? results/av2_confirmation/row_level_recalculation_summary.csv
?? results/av2_confirmation/runtime_method.md
?? results/av2_confirmation/runtime_metrics.csv
?? results/av2_confirmation/runtime_recovery.csv
?? results/av2_confirmation/scientific_decision_PREMATURE.md
?? results/av2_confirmation/statistical_tests.csv
?? results/av2_confirmation/yaw_join_audit.csv
?? results/object_motion_harm/
?? results/sivp_strengthening/
?? scripts/analysis_closure_ff_localization.py
?? scripts/analysis_closure_finalization.py
?? scripts/analysis_closure_fix_gate_leakage.py
?? scripts/analysis_closure_step10_runtime.py
?? scripts/analysis_closure_step11_leakage.py
?? scripts/analysis_closure_step12_provenance.py
?? scripts/analysis_closure_step13_headline.py
?? scripts/analysis_closure_step15_decision.py
?? scripts/analysis_closure_step16_manifest.py
?? scripts/analysis_closure_step1_recovery.py
?? scripts/analysis_closure_step2_audit104.py
?? scripts/analysis_closure_step3_tracker.py
?? scripts/analysis_closure_step4_rtdetr.py
?? scripts/analysis_closure_step5_fullframe.py
?? scripts/analysis_closure_step6_causalgate.py
?? scripts/analysis_closure_step7_extension_audit.py
?? scripts/analysis_closure_step8_failure.py
?? scripts/analysis_closure_step9_sensitivity.py
?? scripts/analyze_cross_dataset_reversal.py
?? scripts/analyze_sivp_failure_sensitivity.py
?? scripts/audit_av2_corrected_outputs.py
?? scripts/audit_av2_invalid_propagation.py
?? scripts/audit_metric_stage.py
?? scripts/audit_sivp_strengthening.py
?? scripts/audit_step2.py
?? scripts/audit_step3.py
?? scripts/benchmark_av2_propagation.py
?? scripts/bootstrap_object_components.py
?? scripts/build_av2_extension_pairs.py
?? scripts/consolidate_object_motion_components.py
?? scripts/consolidate_sivp_end_to_end.py
?? scripts/create_object_motion_figures.py
?? scripts/download_av2_extension.py
?? scripts/evaluate_causal_gate_extension.py
?? scripts/export_benefit_model_coefficients.py
?? scripts/finalize_object_motion_harm.py
?? scripts/finalize_sivp_end_to_end.py
?? scripts/finalize_sivp_tracker_baselines.py
?? scripts/finalize_third_detector_results.py
?? scripts/finish_av2_confirmatory_analysis.py
?? scripts/fit_causal_gate.py
?? scripts/fit_object_motion_gate.py
?? scripts/freeze_av2_extension.py
?? scripts/queue_post_rtdetr.ps1
?? scripts/recalculate_av2_sample.py
?? scripts/recalculate_object_motion_reversal.py
?? scripts/recover_av2_yaw.py
?? scripts/reproduce_sivp_strengthening.ps1
?? scripts/repropagate_nuscenes_common.py
?? scripts/run_av2_corrected_negative_controls.py
?? scripts/run_av2_corrected_propagation.py
?? scripts/run_av2_extension_propagation.py
?? scripts/run_av2_negative_control_sample.py
?? scripts/run_extension_gpu_pipeline.ps1
?? scripts/run_object_motion_components.py
?? scripts/run_rtdetr_av2.py
?? scripts/run_sivp_end_to_end.py
?? scripts/run_sivp_tracker_baselines.py
?? scripts/run_third_cpu_pipeline.ps1
?? scripts/run_third_detector_propagation.py
?? scripts/run_yolo_av2_extension.py
?? scripts/temp_audit1.py
?? src/latency_compensation/av2_models.py
?? src/latency_compensation/causal_trackers.py
?? src/latency_compensation/object_motion_harm.py
?? temp_audit.csv
?? tests/test_object_motion_harm.py

```

## `git diff --check`
```
warning: in the working copy of '.gitignore', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'CITATION.cff', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'README.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'pyproject.toml', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'results/av2_confirmation/run_manifest.json', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'results/av2_confirmation/scientific_decision.md', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'src/latency_compensation/run_confirmation.py', LF will be replaced by CRLF the next time Git touches it
warning: in the working copy of 'tests/test_core.py', LF will be replaced by CRLF the next time Git touches it

```

**Note:** We will not commit, tag, push, or archive any of these files per absolute prohibitions.
