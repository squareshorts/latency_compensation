# RT-DETR-L Architectural Audit

## Checkpoint and Logs
- **Logs:** 50
- **Frames:** 16022
- **Predictions and Targets paired:** Yes
- **Future information:** None (causal evaluation)

## 300-ms High-Yaw Reversal
- **B3 Error:** 0.008511676105048913
- **B5 Error:** 0.01017603670074283
- **Absolute Difference (B5 - B3):** 0.0013911682360248198
- **Relative Difference:** 16.60%
- **IoU Difference (B5 - B3):** -0.04409615615530957
- **Wilcoxon p-value:** 0.0009936344267771346
- **Logs favoring B3:** 82.50%
- **Bootstrap 95% CI:** [0.0007911531444815896, 0.0019377150655708105]

**Conclusion:** RT-DETR-L preserves the direction of the B3 vs B5 reversal.
