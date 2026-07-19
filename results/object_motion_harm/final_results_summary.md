# Final results summary

- The paired-log AV2 reversal reproduces: YOLO11n 19.1807% harm (95% CI 10.7995% to 37.0463%); YOLO11s 21.0579% harm (95% CI 8.3016% to 31.5125%).
- Oracle future object motion improves over B3, and the frozen gate identifies a reproducible conditional region; together these support H3.
- Primary classified mechanism: velocity-sign error; secondary: incorrect association.
- A prospectively frozen uncertainty gate passes the two-detector held-out criterion.
- The nuScenes-to-AV2 reversal is explained by applying the common implementation: the original feasibility contrast used oracle-quality instance association and a weaker fast-IMU ego comparator.

Scientific decision: **PAPER GO**.
