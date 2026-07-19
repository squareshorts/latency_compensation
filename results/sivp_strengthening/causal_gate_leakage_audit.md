# Causal gate leakage audit

Status: **PASS BEFORE EXTENSION EVALUATION**

- The model is a single L2-regularized logistic regression.
- Fit data are restricted to the 80 development logs.
- Threshold selection is restricted to the 20 model-selection logs.
- The original 50 held-out logs are not used for gate fitting or threshold selection.
- The 45 extension logs were selected and hashed from metadata only before this model was evaluated on them.
- Every model input is available at or before the source timestamp.
- Outcome columns are used only to form the training label and to score model-selection thresholds; they are absent from the feature matrix.
- No target-time geometry, future identity, future displacement, true future speed, true target class, B3 error, B5 error, or delta-harm quantity is a model input.
