# Frozen B4/B5 specification for AV2

The active B4/B5 method is frozen in `configs/av2_method_frozen.yaml` before inspecting any AV2 outcome metric. No further nuScenes outcome-driven tuning is permitted.

B4 transforms a source detector box using measured ego-camera motion and source-time lidar depth, then adds object displacement estimated only from associated historical detector observations. B5 damps the historical object-motion component using source-time lidar point count, depth dispersion, and track age. B5 cannot use a future annotation, track UUID, object velocity, image, pose, or detector result after the availability timestamp.

Development logs may estimate nuisance/calibration parameters named in the frozen configuration. Model-selection logs may choose between the prespecified B4 and B5 variants. Held-out logs are opened only after detector, association, depth, and damping parameters are frozen and hashed.
