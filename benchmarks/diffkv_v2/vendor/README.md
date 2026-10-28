# Archived MATH grading code

`math_parser.py` and `math_grader.py` are unmodified copies from DiffKV commit
`c489dc1`, under `vllm/dataset/`. They are stored here so grading does not depend
on a private checkout, ignored cache directory, or the installed vLLM package.
The grader retains its original Qwen2.5-Math attribution and source references.

| File | SHA256 |
| --- | --- |
| math_parser.py | 843ec5282bacdf5b5d8901895e3893796364002883268350dd71ae69a7b635fa |
| math_grader.py | 75f0e12ac6e0d119b6a7b43290d1371d6521cbf27f155c889269457cfe5a68b6 |

The caller reports grader exceptions and caught symbolic timeout warnings as
grading errors. It also reports the explicit boxed-set, unit, ordinal corrections
from `evaluation-logs/r1-math500-greedy-20261024/grade500.py` separately from the
unmodified grader's result. Fixed-length throughput output is never graded.
