# Runtime Analysis

The frozen runtime data successfully isolates detector inference and bounding-box propagation/tracker prediction. However, association overhead, gate evaluation, and total pipelined overhead were **not separately instrumented in the frozen run**.

We do not infer unavailable timings by subtraction. The available timings indicate that tracker prediction and ego/object-motion propagation take <1 ms, which is a fraction of the 5-30 ms detector inference overhead.
