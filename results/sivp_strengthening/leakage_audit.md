# Strengthening leakage audit

Status: **PASS**

- Tracker low-level rows audited: 7,388,304.
- End-to-end prediction rows audited: 10,288,376.
- Evaluation-only target rows audited: 2,445,328.
- Future-information flags in deployable predictions: 0.
- Association histories after the source timestamp: 0.
- The causal gate feature allowlist and forbidden patterns are frozen in `configs/causal_gate_frozen.yaml`.
- Original held-out logs were excluded from gate fitting and threshold selection.
