# Final Scientific Decision

1. **Is the original reversal reproducible?** Yes.
2. **Does RT-DETR preserve the direction?** Yes.
3. **Does B3 beat the best non-B3 causal tracker?** Yes.
4. **Do persistent-object results support the claim?** Yes.
5. **Do full-frame results support, mix with, or contradict it?** Full-frame localization and recall support it; AP50:95 is mixed/tied.
6. **Is the causal gate valid?** It is structurally valid but functionally degenerate (failed).
7. **Did the causal gate improve over B3?** No.
8. **Are extension cohorts complete and comparable?** No.
9. **Are leakage and provenance acceptable for the core B3-versus-B5 analysis?** Yes.
10. **Are all headline values reproducible?** Yes, or marked explicitly as NOT_COMPARABLE.
11. **Did tests pass?** Yes.
12. **Is further scientific computation necessary before manuscript revision?** No.

### Final Classification
**ANALYSIS COMPLETE WITH LIMITATIONS**

### Limitations
- Unsuccessful/degenerate causal gate.
- Incomplete and detector-imbalanced extension propagation.
- No RT-DETR prospective extension validation.
- Mixed aggregate AP evidence.
- Unavailable RT-DETR mechanism diagnostics.
- Incompletely instrumented runtime components.
