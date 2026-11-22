# Search, batch 3: B1-B8 (22 November 2026)

Harness `search_arch.py`, unchanged, source commit `ebb7f1aa`; `SEARCH_OUT=search_results_b3.tsv`; one candidate at a time, its four
tests in parallel on 4 cores; seed 1 (harness default). Base for all: `L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=2` (B1 = base alone).
Tests: moby and code held-out share (streams never used in design), keep = A after B, mem = right after the gap. Score = mean of the four.
Run time = wall time of the candidate (its slowest test, seconds). Every test gave a result (no "no result" lines).

| Rank | Candidate | Options beyond base | moby | code | keep | mem | score | run time (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | B3 | `c_th_on=0.9 c_th_match=0.75` | 0.388 | 0.378 | 1.000 | 1.000 | **0.692** | 103 |
| 2 | B5 | `c_n_grow=16` | 0.380 | 0.363 | 1.000 | 1.000 | 0.686 | 108 |
| 3 | B2 | `c_th_on=0.6 c_th_match=0.5` | 0.398 | 0.394 | 0.470 | 1.000 | 0.566 | 102 |
| 4 | B1 | (base) | 0.397 | 0.391 | 0.470 | 1.000 | 0.565 | 110 |
| 5 | B7 | `c_eta_minus=0` | 0.391 | 0.393 | 0.470 | 1.000 | 0.564 | 103 |
| 6 | B8 | `h_g_fan=8 h_g_k=0.01` | 0.394 | 0.392 | 0.470 | 1.000 | 0.564 | 109 |
| 7 | B6 | `c_eta=0.4` | 0.392 | 0.389 | 0.470 | 1.000 | 0.563 | 107 |
| 8 | B4 | `c_S=32` | 0.383 | 0.369 | 0.470 | 1.000 | 0.555 | 90 |

Memory was 1.000 for every candidate, and keeping took only two values (0.470 or 1.000), so the ranking is decided by keeping.
Only B3 (higher context-cell thresholds) and B5 (`c_n_grow=16`) kept A (1.000); both paid on the two never-used streams (moby
-0.009 / -0.017, code -0.013 / -0.028 against the base B1). The other options moved moby and code by at most 0.022 (B4 lowest) and
left keeping at 0.470; single seed, so differences of a few thousandths are not shown to be real.
