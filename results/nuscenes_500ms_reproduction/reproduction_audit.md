# nuScenes 500 ms reproduction audit

- Source: nuScenes real-data feasibility result, 339 eligible pairs and 1487 paired detector-object observations across 10 complete scenes.
- N0, N1, N4 and N5 were independently recalculated from stored propagated boxes.
- N2 uses a causal constant-velocity Kalman box filter over matched historical real detections.
- N3 uses a causal alpha-beta tracker over up to four historical real detections.
- N6 copies the real future annotation box and is an oracle diagnostic excluded from every decision.
- All seven models use exactly the same 1487 objects.
- Bootstrap: 10,000 paired replicates with complete scene as the resampling unit, seed 20260717.

## Independently reproduced real-data feasibility result

- Stale high-yaw median center error: 62.949861 px.
- B4/N5 high-yaw median center error: 20.461228 px.
- Relative center-error reduction: 67.496%.
- Stale high-yaw median IoU: 0.257238.
- B4/N5 high-yaw median IoU: 0.514680.
- B4/N5 high-yaw recall: IoU 0.3 0.806452, IoU 0.5 0.527742, IoU 0.7 0.176774.
- Scenes with lower N5 center error than N0: 9/10.
- Conservative B4 frame-overhead median: 5.652500 ms; p95 14.647470 ms. This timer includes multiple per-frame propagation/control calculations and therefore upper-bounds isolated B4 cost.

Status: REPRODUCED. Configuration hash: `9f05ab31f5c60858bf4b8ec2b03b616cf65f40f014ca55236095d6f3dedea364`.
