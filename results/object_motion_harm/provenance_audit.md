# Provenance audit

- Analysis commit: `a8b57d6cef6844e4c1045ea950de82427f89c017`.
- Safety tag: `pre-object-motion-harm-analysis-2026-07`.
- Detector inference rerun: no.
- Protected input hashes:

  - corrected_av2_propagation: `44a425bdab4d9925035fc6047845e7ca5487a99cf0bfef515ec5ac52ad7de921` (C:\work\auto\results\av2_confirmation\corrected_checkpoints)
  - av2_detector_checkpoints: `59a7cbac682e45da3208eb4cfb2dd4ba95483c7b077f9f6dcf2a0372622252bf` (C:\work\auto\results\av2_confirmation\checkpoints)
  - invalid_run_archive: `b24fc60551d3d3ff5ab49311bfe80fa9f809e53421fb974c492864926a6e969a` (C:\work\auto\results\av2_confirmation\invalid_propagation_run)
  - nuscenes_reproduction: `cae2110da52b2924ac6f12e3f9d28930763bdcdaf383dfbcba9b02996371ea6c` (C:\work\auto\results\nuscenes_500ms_reproduction)
  - av2_method_frozen.yaml: `0e3576df28fe9ca154888415dd2b045ec0f4de552a6e57843b823b79587f9e48` (C:\work\auto\configs\av2_method_frozen.yaml)
  - av2_development_logs.txt: `44ce2a8566becb1f93f2c7ae65578245d6d15a139fcaefc8100304c28bded983` (C:\work\auto\configs\av2_development_logs.txt)
  - av2_model_selection_logs.txt: `412926b5c01e2db60db9854a000278cd76f391decf60994b7e503cde7a853d35` (C:\work\auto\configs\av2_model_selection_logs.txt)
  - av2_heldout_logs.txt: `15968906597b4826142e01d69fe8c282d84d67ae0a401dd221fe92fecf0fd5c7` (C:\work\auto\configs\av2_heldout_logs.txt)
  - frozen_method_specification.md: `93532d44862d3f90c2a1a659e729a46b552b461b84c0331860d475753791d91e` (C:\work\auto\docs\frozen_method_specification.md)

Result: **PASS**. All protected inputs are content-addressed in `run_manifest.json`.
