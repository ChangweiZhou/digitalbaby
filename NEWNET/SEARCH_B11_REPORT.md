# NEWNET search, batch 11 (22 November 2026)

Run with `search_arch.py` from source commit e166d48e (code unchanged), `SEARCH_OUT=search_results_b11.tsv`, one candidate at a time
(its six tests run in parallel on 4 cores), seed 1. Tests: moby, code (held-out share), keep (G=8), keep4, keep2 (A after B), mem (at
the gap). Score = mean of the six. Run time = wall-clock seconds of the whole candidate. All 48 tests gave a result (no "no result").

| rank | candidate | options | moby | code | keep | keep4 | keep2 | mem | score | run time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | K8 | L=16 cC=64 joint=1 mesh=1 mesh_base=0 mesh_dag=5 c_k_win=4 c_th_on=0.9 c_th_match=0.75 | 0.399 | 0.406 | 1.000 | 1.000 | 1.000 | 1.000 | **0.801** | 203 |
| 2 | K1 | L=16 cC=64 joint=1 mesh=1 mesh_base=0 mesh_dag=5 c_k_win=4 | 0.391 | 0.396 | 1.000 | 1.000 | 1.000 | 1.000 | 0.798 | 223 |
| 3 | K2 | L=16 cC=64 joint=1 mesh=2 mesh_base=0 mesh_dag=5 c_k_win=4 | 0.389 | 0.385 | 1.000 | 1.000 | 1.000 | 1.000 | 0.796 | 214 |
| 4 | K5 | L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=2 c_k_win=4 c_n_grow=16 | 0.382 | 0.389 | 1.000 | 1.000 | 1.000 | 1.000 | 0.795 | 149 |
| 5 | K7 | L=10 M=13 joint=1 mesh=2 mesh_base=0 mesh_dag=2 c_k_win=4 c_th_on=0.9 c_th_match=0.75 | 0.367 | 0.364 | 1.000 | 1.000 | 1.000 | 1.000 | 0.788 | 167 |
| 6 | K4 | L=6 M=21 joint=1 mesh=2 mesh_base=0 mesh_dag=1 c_k_win=4 c_n_grow=16 | 0.397 | 0.408 | 0.470 | 1.000 | 1.000 | 1.000 | 0.713 | 131 |
| 7 | K3 | L=6 M=21 joint=1 mesh=2 mesh_base=0 mesh_dag=1 c_k_win=4 c_th_on=0.9 c_th_match=0.75 | 0.394 | 0.402 | 0.470 | 1.000 | 1.000 | 1.000 | 0.711 | 129 |
| 8 | K6 | L=6 M=21 joint=1 mesh=2 mesh_base=0 mesh_dag=2 c_k_win=4 | 0.395 | 0.398 | 0.470 | 1.000 | 1.000 | 1.000 | 0.711 | 127 |

The five candidates with a serial path of 8 or more regions (K1, K2, K5, K7, K8) passed keep at G=8, and the three with L=6 (K3, K4, K6)
failed it at 0.470; every candidate passed keep4, keep2 and mem. K8 (two pathways, K=1 graph half, stricter thresholds) scored highest
(0.801) and is the only candidate that passed all four memory tests with moby and code both above 0.395; K4 had the highest code (0.408)
but fails keep8. Each candidate was run once, on seed 1, so the 0.003-0.006 score gaps among the top four are not shown to exceed seed
variation (batch 8: E6 moby 0.385 / 0.388 / 0.379 on seeds 1 / 2 / 3).
