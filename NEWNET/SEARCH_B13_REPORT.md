# NEWNET search, batch 13 (22 November 2026)

Run with `search_arch.py` from source commit 6030578e (code unchanged), `SEARCH_OUT=search_results_b13.tsv`, one candidate at a time
(its six tests run in parallel on 4 cores), seed 1. Tests: moby, code (held-out share), keep (G=8), keep4, keep2 (A after B), mem (at
the gap). Score = mean of the six. Run time = wall-clock seconds of the whole candidate. All 48 tests gave a result (no "no result").
Base options K8 = `joint=1 mesh=1 mesh_base=0 mesh_dag=5 c_k_win=4 c_th_on=0.9 c_th_match=0.75`; each candidate adds the options shown.
K8's row is its seed-1 result from SEARCH_B11_REPORT.md (batch 11), not re-run here. Ranking is by the unrounded mean.

| rank | candidate | options | moby | code | keep | keep4 | keep2 | mem | score | run time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| (ref) | K8 (b11) | L=16 cC=64 + K8 | 0.399 | 0.406 | 1.000 | 1.000 | 1.000 | 1.000 | 0.801 | 203 |
| 1 | Q6 | L=16 cC=64 + K8 + h_g_fan=8 | 0.396 | 0.408 | 1.000 | 1.000 | 1.000 | 1.000 | 0.801 | 314 |
| 2 | Q8 | L=16 cC=64 + K8 + eta_c=0.3 eta_h=0.15 | 0.399 | 0.403 | 1.000 | 1.000 | 1.000 | 1.000 | 0.800 | 297 |
| 3 | Q3 | L=16 cC=64 + K8 + c_n_grow=16 | 0.394 | 0.407 | 1.000 | 1.000 | 1.000 | 1.000 | 0.800 | 283 |
| 3 | Q7 | L=16 cC=64 + K8 + nH=128 nG=1024 | 0.398 | 0.403 | 1.000 | 1.000 | 1.000 | 1.000 | 0.800 | 252 |
| 5 | Q1 | L=16 cC=32 M=32 + K8 | 0.393 | 0.402 | 1.000 | 1.000 | 1.000 | 1.000 | 0.799 | 289 |
| 6 | Q4 | L=20 cC=51 + K8 | 0.389 | 0.387 | 1.000 | 1.000 | 1.000 | 1.000 | 0.796 | 328 |
| 7 | Q2 | L=16 cC=64 joint=1 mesh=2 mesh_base=0 mesh_dag=5 c_k_win=4 c_th_on=0.9 c_th_match=0.75 | 0.385 | 0.378 | 1.000 | 1.000 | 1.000 | 1.000 | 0.794 | 301 |
| 8 | Q5 | L=12 cC=85 + K8 | 0.398 | 0.413 | 0.470 | 1.000 | 1.000 | 1.000 | 0.713 | 233 |

No candidate scored above K8's batch-11 seed-1 score (unrounded 0.8008); Q6 (0.8007), Q8, Q3 and Q7 (0.8002-0.8003) are within 0.001
of it, and Q1, Q4 and Q2 are 0.002-0.007 below. Q5 (L=12) failed keep at G=8 (0.470) while having the highest moby/code of the batch
(0.398 / 0.413), so every L>=16 candidate passed all four memory tests and the L=12 one did not. Each candidate ran once, on seed 1 only,
so these gaps are not shown to exceed seed variation; run times are longer than batch 11's (K8 203 s) for all candidates, and K8 was not
re-run on this machine, so run times are not comparable across batches.
