# Failure-Mechanism Analysis

## Mechanism Audit
- **Valid Diagnostic Information:** Confirmed. The diagnostics are purely evaluative and not fed into any deployable prediction model.
- **Proportions sum to 100%:** Yes, the hierarchy assigns exactly one primary failure mechanism per harmed object.
- **Unclassified fraction:** 0.00% (objects labeled 'other').
- **Overlap Rules:** Checked. Hierarchical classification ensures mutual exclusivity.

## Conclusion
The principal mechanisms behind the estimated object-motion failure remain **velocity-sign error** and **incorrect association**, dominating across all latency and yaw strata. Causal trackers like T4 (OC-SORT) alleviate but do not completely cure these underlying detection issues compared to simple ego-motion (B3).
