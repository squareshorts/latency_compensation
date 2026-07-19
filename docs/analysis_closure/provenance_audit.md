# Provenance Audit

## Status
**PASS**

## Details
- **Cohort Overlap:** Validated. No intersections found between development, model-selection, held-out, or extension logs.
- **Coordinate Validity:** Validated. Bound check `pred_x2 >= pred_x1` and `pred_y2 >= pred_y1` passes on sampled parquets.
- **Row Uniqueness:** Validated. No exact duplicate rows.
- **Models:** Validated. T0 through T6 exist in all evaluation sets.
- **Missing Boxes:** Evaluated implicitly through coordinate checks and finiteness checks (which passed earlier integrity audits).
