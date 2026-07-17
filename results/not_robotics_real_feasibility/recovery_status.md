# Recovery status

## Audit outcome

- Git status: `C:\work\auto` is not a Git repository; `git status` returned the expected fatal not-a-repository message.
- Running analysis processes at recovery start: none. The only matching process was the audit PowerShell process itself.
- Interruption: detected as a stopped prior workflow; an actual power loss cannot be proven from local evidence.
- Original Playwright temporary paths: both absent before recovery began. Their deletion timing/order cannot be reconstructed.
- Permanent archives: present and valid.
  - `can_bus.zip`: 780,974,697 bytes; ZIP magic; SHA-256 `3C68B94C001E8BD05A19886ECB2C6854E0CD69D7005ED9A94D13D45D2951E83F`; 7,834 members; CRC test passed; 7,833 regular files extracted with no missing files or size mismatches.
  - `v1.0-mini.tgz`: 4,168,148,189 bytes; gzip-compressed TAR magic; SHA-256 `E5D9C2B5CED29F9E3D39651E2093D27E892D0571F10962409974B21033A254EE`; TAR listing passed with 31,253 members; 31,225 regular files extracted with no missing files or size mismatches.
- Published/vendor checksum comparison: not available locally; integrity is verified structurally, by archive CRC/listing, local SHA-256 recording, and exact extracted-member sizes.
- Required data paths: `v1.0-mini`, `samples\CAM_FRONT`, and `can_bus` all exist.
- Real CAM_FRONT files: 2,342 total (404 keyframes plus 1,938 sweeps); 2342 opened successfully, 0 failed.
- Metadata annotations: 18,538 total; 4,068 projected class-mapped 2D boxes were evaluated.
- CAN JSON audit: all 7,832 JSON files parsed; 25,864,384 JSON records; one JSON file was empty and valid.
- Detector weights: `yolo11n.pt`, 5,613,764 bytes, SHA-256 `0EBBC80D4A7680D14987A577CD21342B65ECFD94632BD9A8DA63AE6417644EE1`; loaded and used by Ultralytics 8.4.96 on CPU.
- Current intended output corruption: 0 invalid files. Preserved `_INCOMPLETE` files are excluded from scientific evidence.
- Synthetic/mock/placeholder/fabricated observations in current intended outputs: none. Negative provenance declarations may contain those words. The separate `not_robotics_feasibility_INVALID_SYNTHETIC` directory contains 13 explicitly invalid synthetic/mock artifacts and is not used.

## Last-stage chronology

- 2026-07-16 16:29: prior blocked data-presence report.
- 2026-07-16 16:38–16:41: permanent archives completed.
- 2026-07-16 17:01: extraction completed.
- 2026-07-16 17:03–17:04: environment/data audit completed.
- 2026-07-16 17:07: prior real detector-count run stopped after writing an inadequate report; those artifacts are preserved with `_INCOMPLETE` suffixes.
- 2026-07-17 11:21–11:22: resumed box-level inference, synchronization, grouped modeling, controls, and gate evaluation completed.
- 2026-07-17: independent recalculation verified all 39 reported metric rows within 1e-12; maximum flow recalculation difference was 0.

## Phase classification

1. archive download — completed and verified
2. permanent copy — completed and verified
3. checksum and archive verification — completed and verified
4. extraction — completed and verified
5. real-data presence audit — completed and verified
6. environment and detector setup — completed and verified
7. camera/CAN synchronization — completed and verified
8. real detector inference — completed and verified
9. real optic-flow extraction — completed and verified
10. error-target construction — completed and verified
11. grouped action-information models — completed and verified
12. shifted/shuffled action controls — completed and verified
13. reliability-gate evaluation — completed and verified
14. metric recalculation — completed and verified
15. provenance audit — completed and verified
16. final scientific decision — completed and verified

## Real-data results

- Scenes: 10.
- CAM_FRONT: 2,342 real files; 404 keyframes processed by actual detector inference.
- Annotations: 18,538 metadata annotations; 4068 mapped/projected boxes evaluated.
- CAN: 7,832 JSON files and 25,864,384 records.
- Action/control proxy fields: steeranglefeedback.value; zoesensors throttle_sensor, brake_sensor, steering_sensor; zoe_veh_info requestedTorqueAfterProc, regen, pedal_cc.
- Synchronization used only the latest causal message at or before each camera timestamp. The first frame of each scene had no prior CAN record and was imputed inside training folds.
- pose: 394/404 causal matches; mean age 9.253 ms, p95 17.706 ms, max 34.602 ms.
- steeranglefeedback: 394/404 causal matches; mean age 5.298 ms, p95 9.685 ms, max 13.101 ms.
- zoe_veh_info: 394/404 causal matches; mean age 4.949 ms, p95 10.428 ms, max 18.160 ms.
- zoesensors: 394/404 causal matches; mean age 0.747 ms, p95 2.049 ms, max 6.011 ms.
- Best non-action model: `detector_only`, held-out AUC 0.812494.
- Full action-conditioned model: held-out AUC 0.746909.
- Action-information benefit: -0.065585 AUC.
- Shifted 1-second control: AUC 0.780382; shuffled control: AUC 0.808896. Both outperform correctly timed full action features.
- High-self-motion false positives: 2336 to 2203 (-5.693%).
- Recall: 0.470993 to 0.449853 (-2.114 percentage points).
- Calibration ECE: 0.190454 to 0.229544 (+0.039090; worse).
- Leakage: passed scene grouping, causal synchronization, past-to-current optic flow, nested threshold selection, and target/input separation checks.
- Metric recalculation: all headline and ablation metrics independently matched within 1e-12.
- Scientific decision: **NO-GO**. The correctly timed action model did not add held-out information, did not beat timing controls, missed the 10% high-motion FP reduction target, exceeded the 2-point recall-loss ceiling, and worsened calibration.

## Temporary-file disposition

The two specified temporary paths are already absent. If equivalent temporary copies existed, they would now be safe to delete because permanent archives, hashes, extraction, real images, and CAN parsing are verified.
