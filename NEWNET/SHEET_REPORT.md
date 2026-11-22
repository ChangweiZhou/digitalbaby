# Sheet report: regions wired as a cortical sheet against the random graph without loops (22 November 2026)

Plan: `SHEET_PLAN.md` (commit c8751698, written and committed before any run). Learner `net11.py` (subclass of net10; only the
graph of sources differs from the baseline). Source commit cf5ff1d1; branch claude/newnet-sheet-sei232. Development level.
Every run: `t17_general.py T=16000 NT=10000`, joint=1, K (or mesh) = 2, read 16,000 bytes from the stream's start, scored on
10,000 bytes from the middle with learning off. L=16 runs with nH=128 nG=1024. OMP_NUM_THREADS=1, 4 processes at once.
Logs: `sheet_runs/*.log`. Count tables (reference, fitted on the same bytes), last 1/2/3 bytes: Moby 0.292/0.368/0.379, code
0.255/0.349/0.386. Graph = random: `net=10 mesh=2 mesh_base=0 mesh_dag=1`; sheet: `net=11 K=2 p=...`.

| graph | L | K | p | stream | seed | held-out | run time (s) |
|---|---|---|---|---|---|---|---|
| random | 8 | 2 | – | moby | 1 | 0.394 | 114 |
| random | 8 | 2 | – | moby | 2 | 0.391 | 113 |
| sheet | 8 | 2 | 0 | moby | 1 | 0.382 | 114 |
| sheet | 8 | 2 | 0 | moby | 2 | 0.382 | 117 |
| sheet | 8 | 2 | 0.2 | moby | 1 | 0.391 | 115 |
| sheet | 8 | 2 | 0.2 | moby | 2 | 0.391 | 118 |
| random | 8 | 2 | – | code | 1 | 0.392 | 122 |
| random | 8 | 2 | – | code | 2 | 0.400 | 118 |
| sheet | 8 | 2 | 0 | code | 1 | 0.382 | 124 |
| sheet | 8 | 2 | 0 | code | 2 | 0.387 | 120 |
| sheet | 8 | 2 | 0.2 | code | 1 | 0.387 | 116 |
| sheet | 8 | 2 | 0.2 | code | 2 | 0.396 | 116 |
| random | 16 | 2 | – | moby | 1 | 0.385 | 241 |
| random | 16 | 2 | – | moby | 2 | 0.390 | 235 |
| sheet | 16 | 2 | 0 | moby | 1 | 0.379 | 234 |
| sheet | 16 | 2 | 0 | moby | 2 | 0.377 | 248 |
| sheet | 16 | 2 | 0.2 | moby | 1 | 0.380 | 238 |
| sheet | 16 | 2 | 0.2 | moby | 2 | 0.377 | 249 |
| random | 16 | 2 | – | code | 1 | 0.376 | 248 |
| random | 16 | 2 | – | code | 2 | 0.377 | 239 |
| sheet | 16 | 2 | 0 | code | 1 | 0.367 | 249 |
| sheet | 16 | 2 | 0 | code | 2 | 0.365 | 239 |
| sheet | 16 | 2 | 0.2 | code | 1 | 0.366 | 242 |
| sheet | 16 | 2 | 0.2 | code | 2 | 0.359 | 236 |

Means of the two seeds (random / sheet p=0 / sheet p=0.2): Moby L=8 0.3925 / 0.382 / 0.391; code L=8 0.396 / 0.3845 / 0.3915;
Moby L=16 0.3875 / 0.378 / 0.3785; code L=16 0.3765 / 0.366 / 0.3625. The random baseline reproduces ARCH_GENERAL at seed 1
(Moby L=8 0.394, L=16 0.385).

**Plain reading.** The sheet was not better anywhere. Sheet p=0.2 minus random: −0.0015 and −0.0045 at L=8 (Moby, code; within the
±0.005 read as no difference), −0.009 and −0.014 at L=16 (worse). By the refutation rule written in the plan (below by more than
0.005 on both streams at both L) the hypothesis is refuted at L=16 only, not at L=8; but every predicted gain (+0.005 at L=8,
+0.01 at L=16) failed, so the sheet's claim to beat the random graph is not supported at any size. Near wiring alone (p=0) is
worse than the random graph by about 0.01 everywhere, about where the chain stood (0.381 Moby L=8), as the plan predicted for p=0.
At L=8 the long-range swaps (p=0.2) recover most of the loss; at L=16 they do not. The sheet graphs are much deeper (longest path
7 against 5 regions at L=8, 11-13 against 6-7 at L=16, measured before the runs); in these runs more depth went with lower held-out,
and the gap grows with L — consistent with the earlier chain-vs-random reading that short paths of many lengths serve better than
long ones. This is an association over two seeds and two graph drawings per L, not a test of depth alone.
Not run: K other than 2, p other than 0 and 0.2, sheets ordered other than by one coordinate, longer reading, more seeds.
