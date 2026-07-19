# SIVP strengthening freeze

Freeze date: 2026-07-18
Branch: `revision/sivp-submission-strengthening`
Safety tag: `pre-sivp-strengthening-2026-07`
Current and tagged commit: `a8b57d6cef6844e4c1045ea950de82427f89c017`

This record protects the completed AV2 confirmation while additional reviewer-facing analyses are added. The primary cohort, endpoint, propagation definitions, tables, and claims below must not be changed by the strengthening analysis.

## Detector weights

| Detector | Local checkpoint | Bytes | SHA-256 |
|---|---:|---:|---|
| YOLO11n | `yolo11n.pt` | 5,613,764 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| YOLO11s | `yolo11s.pt` | 19,313,732 | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` |

The preserved AV2 detector-checkpoint tree contains 600 files (369,356,954 bytes) with deterministic tree hash `59a7cbac682e45da3208eb4cfb2dd4ba95483c7b077f9f6dcf2a0372622252bf`.

## AV2 cohort manifests

| Split | Logs | Manifest | SHA-256 |
|---|---:|---|---|
| Development | 80 | `configs/av2_development_logs.txt` | `44ce2a8566becb1f93f2c7ae65578245d6d15a139fcaefc8100304c28bded983` |
| Model selection | 20 | `configs/av2_model_selection_logs.txt` | `412926b5c01e2db60db9854a000278cd76f391decf60994b7e503cde7a853d35` |
| Held out | 50 | `configs/av2_heldout_logs.txt` | `15968906597b4826142e01d69fe8c282d84d67ae0a401dd221fe92fecf0fd5c7` |

The frozen method configuration is `configs/av2_method_frozen.yaml`, SHA-256 `0e3576df28fe9ca154888415dd2b045ec0f4de552a6e57843b823b79587f9e48`.

## Protected model definitions

- **B3 — ego motion only:** back-project the four corners of the source detector box using source-time LiDAR depth; transform them from the source camera through the measured source and target AV2 ego poses; reproject into the target camera; set object displacement exactly to zero.
- **B5 — uncertainty-damped ego plus estimated object motion:** start from the same B3 geometry and add a city-frame object displacement estimated from the two latest causally associated detector boxes and source-time depth estimates. Cap estimated speed at 30 m/s. Multiply displacement by the frozen product of LiDAR-point support, depth-dispersion reliability, and track-age reliability.

Neither deployable model receives target-time annotations, future identity, or future object geometry. Target annotations are evaluation-only.

## Protected endpoint and result

- Dataset and split: the fixed 50-log AV2 held-out split above.
- Detectors: preserved YOLO11n and YOLO11s outputs.
- Endpoint: 300-ms latency, development-frozen high-yaw stratum (`|yaw rate| > 0.0169726458 rad/s`), normalized box-center error.
- Independent unit: complete AV2 log.
- Primary estimand: paired relative difference `error(B5) / error(B3) - 1` using within-log medians and 10,000 paired log-cluster bootstrap replicates.
- Frozen result: B5 was 19.1807% worse than B3 for YOLO11n and 21.0579% worse for YOLO11s; both 95% paired log-bootstrap intervals exclude zero.
- Scientific decision: confirmatory **NO-GO for universal B5**; publication analysis **PAPER GO**.

The corrected propagation tree contains 300 Parquet files (917,849,797 bytes), deterministic tree hash `44a425bdab4d9925035fc6047845e7ca5487a99cf0bfef515ec5ac52ad7de921`.

## Protected result-table hashes

| Path | SHA-256 |
|---|---|
| `results/av2_confirmation/heldout_primary_metrics.csv` | `d556155a9693c38f549b8daaa44bfb15522fbbf7bab57d654019afa972c25a49` |
| `results/av2_confirmation/bootstrap_intervals.csv` | `59ce2e70e9d8141cb1d03866ba5894353cc9fc4d0546dab45a6be10e9d72c1a5` |
| `results/av2_confirmation/statistical_tests.csv` | `95461aca4d806a833d27521fffcf9ed2de07d091cc43499e7219534569c02744` |
| `results/object_motion_harm/bootstrap_reversal.csv` | `83e3ad714e7a01ece0f84ffbeadab8d33fe67ebc6385223560c98fab479acd55` |
| `results/object_motion_harm/per_log_reversal.csv` | `2abc96a881cf4de018e960ae0b8575d4e386f0a1ffb19983642d6194eb8a1a1b` |
| `results/object_motion_harm/component_substitution.csv` | `f6a0bc2510712f3d2e955d6aaf70b5b5dee85f8fad18d1245290d6aa7c5525e2` |
| `results/object_motion_harm/failure_mechanism_summary.csv` | `32f69f236793637e029efdd5fd448026695047258b4d6183d00b45b751df4d4f` |
| `results/object_motion_harm/cross_dataset_harmonization.csv` | `49d014d064d23a8b9bb000d6dba5943c2198c44b2cbded7e962c1ca4e8de15d9` |

## Protected figure-source hashes

| Path | SHA-256 |
|---|---|
| `results/object_motion_harm/figures/figure_1_expected_vs_observed.csv` | `f70ac59a3cb9d5f1e4dd15c9a530d56e75e9598787b9e21ddb254780a4a92d3d` |
| `results/object_motion_harm/figures/figure_2_paired_logs.csv` | `5913e2713afc4e405f0d6987cf6bfd546778077be1148d8be5de87282c7b016d` |
| `results/object_motion_harm/figures/figure_3_component_substitution.csv` | `ea8bfe8e9ae383763d07e550b5e122323473921f3e63e0c375fa60700bda0678` |
| `results/object_motion_harm/figures/figure_4_harm_probability.csv` | `69062006d2033f8af2b1bdf613eaf60c9c6c60b64607d03ee4cb0406a2b4fa1e` |
| `results/object_motion_harm/figures/figure_5_cross_dataset_reversal.csv` | `32699b4b750f7d3a2d32ac32926d3567947d94db47bb24890e51e4eafd1fc88c` |
| `results/object_motion_harm/figures/figure_6_gate_recommendation.csv` | `43aefaa1e3befd148ab9a43f7526bced888ba3debdef9871c15efb2271cab62c` |

## Test and compute state

- Test command: `$env:PYTHONPATH='C:\work\auto\src'; .\.venv-not-robotics\Scripts\python.exe -m pytest -q`
- Result at freeze: **18 passed**.
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, 8,188 MiB; driver 591.74.
- Python environment: Python 3.10, PyTorch 2.13.0+cu126, CUDA available, CUDA runtime 12.6.
