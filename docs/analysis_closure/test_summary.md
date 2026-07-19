# Test Summary

## Execution Details
- **Command:** `python -m pytest tests\`
- **Environment:** Windows, PYTHONPATH="src"
- **Elapsed Time:** 1.73 seconds
- **Overall Status:** PASS

## Coverage and Checks
The test suite successfully evaluated:
- Metric calculations (Euclidean error, IoU bounding)
- Bootstrap reproducibility (fixed seeds)
- Tracker causality (no future information used in T0-T4)
- Gate feature allowlist adherence
- Cohort overlap rules (no intersection)
- Coordinate validity (x2 >= x1, y2 >= y1)
- Output-file readability and column presence
- Current-detector unit counting
- Headline recalculation integrity

All required tests executed correctly without triggering new inference or heavy downloads.
