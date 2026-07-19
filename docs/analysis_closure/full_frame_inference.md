# Full-Frame Statistical Inference

The full-frame evidence demonstrates that the drop in performance when using estimated object motion extends to standard full-frame object detection metrics.

Across YOLO11n, YOLO11s, and RT-DETR-L at the 300-ms high-yaw endpoint:
- **AP50:95 and AP50** are consistently reduced by B5 (T6) relative to ego-motion-only B3 (T5).
- **Persistent-object recall** is significantly reduced.

**Conclusion:** Full-frame results strongly support the localization conclusion that estimated object motion is harmful.
