# Leakage audit

- Scene grouping: 5-fold GroupKFold; no scene appears in both train and test in a fold.
- CAN alignment: latest message with `utime <= camera timestamp`; future messages are never used.
- Optical flow: previous keyframe to current keyframe only.
- Gate thresholds: selected with inner grouped out-of-fold predictions on outer-training scenes.
- Outcome features: projected ground truth and match labels are used only as targets/evaluation, never as model inputs.

Status: PASSED.
