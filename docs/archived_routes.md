# Archived routes

| Route | Location | Closure reason |
|---|---|---|
| Action-conditioned reliability gate | `archive/negative_routes/action_reliability_gate` | Correctly timed action features underperformed non-action and shuffled controls on real nuScenes data. |
| Dual-loop visual residual | `archive/negative_routes/dual_loop_visual_residual` | It beat stale boxes but lost to causal inertial plus object-motion compensation and exceeded the overhead target. |
| Annotation-cadence route | `archive/negative_routes/blocked_annotation_cadence` | nuScenes sweeps lack genuine object annotations at the original 100–300 ms endpoints. |
| SINDY and biological routes | `archive/negative_routes/sindy_and_biological_routes` | Superseded; no active reproducible artifact was found in the pre-pivot tree. |
| Invalid synthetic study | `archive/synthetic_invalid` | Fabricated/mock feasibility outputs are excluded from real-data evidence. |
