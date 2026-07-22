# Polish notes — "When More Motion Modeling Hurts"

Manuscript: 9 pages, 18 references, compiles with 0 errors and 0 undefined citations.
Supplement: 4 pages. Target venue: IEEE T-IV (10-page nominal limit for regular papers).

## New analyses run in this pass

The frozen `object_level_diagnostics.parquet` contains rows for YOLO11n and YOLO11s only,
which is why RT-DETR-L was missing from the class, direction, and gate analyses. The
low-level `end_to_end_low_level/heldout` prediction and target tables do carry class and
geometry fields for all three detectors, so the matched-object comparison was rebuilt from
those.

**Validation before use.** The reconstruction was run on YOLO11n and YOLO11s and compared
against the published frozen values. Log counts, comparison counts, median errors, and
paired relative differences matched to a maximum absolute difference of 1.7e-18. Per-class
comparison counts sum to 11,425 / 14,089 / 25,729 — exactly the held-out totals in Table I.
Only then was the identical code path applied to RT-DETR-L.
Scripts: `scripts/rtdetr_extension_analysis.py`, `scripts/rtdetr_extension_diagnostics.py`.

### 1. RT-DETR-L class-stratified results (new rows in Table III)

| Class | Logs | Objects | B3 | B5 | Disadvantage (95% CI) |
|---|---|---|---|---|---|
| Vehicle | 40 | 21,802 | 0.007281 | 0.008738 | 15.92% (7.68–22.47%) |
| Ped./cyclist | 29 | 1,872 | 0.010402 | 0.013935 | 26.64% (1.49–44.53%) |

This is the substantive win. The pedestrian/cyclist stratum was the paper's weakest claim —
YOLO recovered only 527 and 894 comparisons and gave intervals touching zero or spanning
65 points. RT-DETR's higher small-object density gives 1,872 comparisons over 29 logs and an
interval that excludes zero. The vulnerable-road-user result flipped from "too imprecise to
conclude" to a positive finding, and the Discussion now says so instead of asking for future
work.

### 2. Prospective replication on the extension cohort (new Table V)

| Detector | Logs | Disadvantage (95% CI) |
|---|---|---|
| YOLO11n | 9 | 29.21% (17.80–40.77%) |
| YOLO11s | 33 | 20.35% (12.20–24.67%) |
| RT-DETR-L | 15 | 10.90% (3.39–31.17%) |

RT-DETR extension propagation exists for 21 logs and was never analysed. The B3-vs-B5
contrast needs only predicted and target geometry, so it is scorable there even though gate
selection is not. "No RT-DETR prospective validation" is now false — the reversal reproduces
on untouched data for all three detectors.

### 3. Image-space direction agreement (new Table IV, all three detectors)

B5 = B3 + injected correction, so the displacement that would zero the error is the residual
between the B3 center and the target center. Direction is the sign of their inner product.
Toward: 57.75% / 58.03% / 56.62%. Median correction 9.0–13.6 px against median residual
17.5–22.4 px.

This is a *different* statistic from the published 51% ground-plane figure — different space,
different eligibility rule — and the manuscript presents both rather than replacing one with
the other. It also sharpens the mechanism: the estimate is better than chance, consistent with
the oracle result, but it fires at full magnitude, so the ~42% of wrong-direction corrections
displace the box by about as much as they were meant to fix.

### 4. Detection density (new supplement table, Methods paragraph)

RT-DETR-L emits 20.5 det/frame vs 7.9 and 9.8, median confidence 0.476 vs 0.521 and 0.606,
40.6% below 0.4. That accounts for FP/frame ≈ 14 vs 3.6–5.0 as a property of detector output
density preceding propagation, not of B3/B5.

### Verification

`scripts/verify_manuscript_numbers.py` re-reads the analysis CSVs and greps every value
against the compiled sources: **42/42 match verbatim.**

## Text changes

- Table III gains three RT-DETR rows; caption records the reconstruction and its validation.
- New Table IV (direction agreement) and Table V (extension replication).
- Methods: new paragraph on detection density; new paragraph documenting the reconstruction
  provenance and what remains YOLO-only.
- Results: class subsection rewritten around the resolved ped/cyclist stratum; gate subsection
  retitled and now leads with the prospective replication.
- Discussion: class paragraph now states a positive finding; direction paragraph carries both
  checks.
- Abstract and Conclusion updated for the class and prospective results.
- Limitations: removed the two claims that are no longer true (class diagnostics unavailable
  for RT-DETR; no RT-DETR prospective validation). What remains is the accurate residual —
  hierarchical mechanism labels and the ground-plane velocity comparison need estimator
  internals recorded for YOLO only, so those two diagnostics cover two of three architectures.
- Style: removed defensive framing on the low-AP paragraph (now states the estimand
  positively); author contributions no longer say "software supervision" or carry a bracketed
  note; Acknowledgment placeholder replaced with a dataset acknowledgment.

## Not eliminated, and why

**Learned-forecaster baseline (StreamYOLO / DAMO-StreamNet).** Requires training on AV2;
no GPU in this environment (`nvidia-smi` absent). The scope defense is in Related Work —
forecasters absorb latency during training, this paper transports frozen detector outputs —
and is a defensible rebuttal position rather than a gap in the evidence.

**RT-DETR gate scoring.** The gate consumes 13 instrumented source-time features
(`detector_confidence`, `association_confidence`, `velocity_dispersion_m_s`, etc.) that exist
only in the diagnostics table, which has zero RT-DETR rows. Producing them means re-running
the instrumented B5 estimator over 16,022 frames with detector inference — not feasible
without a GPU. The manuscript now states this as a scoping fact and reports the prospective
B3-vs-B5 result that *is* scorable there.

**Funding line.** The Acknowledgment credits the dataset providers; add institutional funding
if applicable.

## Before submission

- [ ] Compile on Overleaf with the genuine `IEEEtran.bst`. The local verification build had to
      substitute the `IEEEtranM` stand-in (CTAN is unreachable from this sandbox), so citation
      formatting may differ trivially. The shipped `main.tex` contains no trace of that
      workaround — it is unchanged and expects the real bst.
- [ ] Add funding to the Acknowledgment if applicable.
- [ ] Update the Zenodo DOI to a release containing the two new analysis scripts.
- [ ] ORCID and IEEE PDF eXpress at submission.
