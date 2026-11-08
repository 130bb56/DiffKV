# evaluation

For long runs, or when performance variation across test cases must be near zero, lock the GPU clock and monitor it periodically so that thermal throttling does not occur. `lock_clock.sh` does this. The table below shows which evaluations run through it and which do not.

| Evaluation | use `lock_clock.sh` |
| --- | --- |
| offline serving (nsys, ncu 포함) | O |
| attention latency | O |
| attention profile (nsys) | O |
| correctness | X |

If another run is using the GPU, set `DIFFKV_GPU` to a free GPU before running.
