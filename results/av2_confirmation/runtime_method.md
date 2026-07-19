# AV2 propagation runtime recovery method

No valid per-model propagation timing was present in the original logs. The
invalid generator logged detector inference duration and propagation row count,
but not B0–B5 runtime.

A warmed microbenchmark was therefore run on deterministic inputs from the
1,000-comparison recalculation sample. Each B0–B5 branch ran 2000
timed repetitions after 200 warmups. Detector inference, disk I/O, lidar
loading, and pose/depth acquisition were excluded; detector history, depth and
poses were precomputed. The representative subset contained
9.096467 detector objects per frame. Per-frame overhead is the
measured per-object overhead multiplied by this observed object count.

Hardware: Windows-10-10.0.26200-SP0; processor=Intel64 Family 6 Model 170 Stepping 4, GenuineIntel.

These timings characterize the corrected propagation mathematics only. They
are audit measurements, not confirmatory runtime evidence for the invalid
original run.
