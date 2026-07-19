# Reversal explanation

The **nuScenes real-data feasibility result** was positive under its original implementation: the object-motion model changed normalized center error by -18.154% relative to its fast-IMU ego baseline. That comparison used GT instance identity to assemble detector history and a causally integrated fast IMU pose estimate.

When the AV2 definitions are applied to the preserved nuScenes detector outputs—exact measured ego poses for B3, deployed box-only historical association, the same 30 m/s cap, and the same point/depth/track-age damping—the object-motion term changes error by 12.798% relative to B3. The sign therefore matches the **AV2 confirmatory reversal** rather than the original feasibility contrast.

The reversal is consequently explained primarily by implementation and comparator differences, especially oracle-quality instance association in the feasibility object tracker and the weaker fast-IMU ego baseline. Covariate shift and the ten-scene sample size remain secondary limits; matched, weighted, common-support, and leave-one-group-out results bound their influence in the accompanying tables.
