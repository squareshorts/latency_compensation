# AV2 causality audit

- Pair construction used only real image and annotation timestamps.
- Annotation timestamps were matched to the nearest real timestamp within 10 ms of the requested offset; no interpolation or pseudo-labels were created.
- Future annotation geometry is evaluation-only and explicitly marked; it is not present as a propagation input.
- Image timestamps were matched to real files within 30 ms.
- Detector inference and propagation outcomes are not yet run.
