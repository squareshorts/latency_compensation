# Corrected AV2 propagation integrity audit

**PASS**

- Corrected Parquets: 300/300.
- Model rows: 10,347,774; unique comparisons: 1,724,629.
- Missing required schemas: 0.
- Duplicate comparison/model rows: 0.
- Comparisons without exactly B0–B5 once each: 0.
- Rows with future geometry input: 0.
- Direct B0/B3 equality: 0.008176% (legitimate zero-motion coincidences only).
- Direct B1/B4 equality: 0.000870%.
- Independent sample agreement is reported in `corrected_row_level_agreement.csv` at 1e-6 px tolerance.

This integrity pass verifies corrected output structure and coordinate
distinctness. It does not convert the original invalid run into a confirmatory
result, and it does not substitute sample controls for full prespecified
negative controls.
