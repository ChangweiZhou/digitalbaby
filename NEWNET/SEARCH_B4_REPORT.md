# Search batch 4 (22 November 2026)

Run with `search_arch.py` at source commit `ebb7f1aa`, no code changed, seed 1 (default). Each candidate's four tests were run in
parallel on 4 cores, one candidate at a time; `SEARCH_OUT=search_results_b4.tsv`. Run time is the candidate's wall time (its slowest
test). No test printed "no result". Score is the mean of the four tests.

| Rank | Candidate | Options | moby | code | keep | mem | score | run time |
|---|---|---|---|---|---|---|---|---|
| 1 | C4 | L=6 M=21 joint=1 mesh=2 mesh_base=0 mesh_dag=2 | 0.396 | 0.411 | 0.470 | 1.000 | **0.569** | 91 s |
| 2 | C7 | L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=2 nH=512 nG=4096 | 0.397 | 0.392 | 0.470 | 1.000 | 0.565 | 114 s |
| 3 | C3 | L=8 joint=1 mesh=3 mesh_base=0 mesh_dag=2 | 0.393 | 0.392 | 0.470 | 1.000 | 0.564 | 114 s |
| 4 | C6 | L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=2 eta_c=0.4 eta_h=0.2 | 0.393 | 0.385 | 0.470 | 1.000 | 0.562 | 100 s |
| 5 | C5 | L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=2 eta_c=0.1 eta_h=0.05 | 0.386 | 0.376 | 0.470 | 1.000 | 0.558 | 104 s |
| 6 | C1 | L=12 M=12 joint=1 mesh=2 mesh_base=0 mesh_dag=2 | 0.385 | 0.369 | 0.470 | 1.000 | 0.556 | 135 s |
| 7 | C2 | L=16 cC=64 joint=1 mesh=2 mesh_base=0 mesh_dag=2 | 0.372 | 0.354 | 0.470 | 1.000 | 0.549 | 144 s |
| 8 | C8 | L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=2 hold=0 M=20 | 0.397 | 0.410 | 0.470 | 0.507 | 0.446 | 79 s |

Keeping was 0.470 for every candidate, so it did not separate them; memory was 1.000 for all except C8 (no holders, 0.507). The
differences are in the two never-used streams: fewer, larger modules (C4: L=6, M=21) and the no-holder C8 scored highest on code
(0.411 / 0.410), while more, smaller modules (C1 L=12, C2 L=16) scored lowest on both. The learning-rate and holder-size changes
(C5-C7) and K=3 (C3) moved the stream scores by at most about 0.02. Each candidate was run once (seed 1).
