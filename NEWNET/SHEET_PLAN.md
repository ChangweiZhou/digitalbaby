# Sheet plan: regions wired as a cortical sheet against the random graph without loops (22 November 2026)

Helper session of the NEWNET line; source commit cf5ff1d1 (branch claude/magical-albattani-sei232), outcome branch
claude/newnet-sheet-sei232. Development level. No existing file is edited; the learner is `net11.py` (a subclass of net10).

**Learner.** net10 joint=1 with its context regions as net7 mesh, mesh_base=0, mesh_dag=1 (same seeds, lines wiring and holders as
the baseline); only the sources of each region differ. net11: L regions at random points of the unit square, ordered by the first
coordinate; region d reads its K=2 nearest earlier regions; with probability p one source is swapped for a random earlier region.
Physical size is that of the baseline at the same L (L=8: cC=128, M=16, nH=256, nG=2048; L=16: nH=128, nG=1024).

**Baseline.** `net=10 joint=1 mesh=2 mesh_base=0 mesh_dag=1` at the same L (ARCH_GENERAL: Moby L=8 0.394, L=16 0.385, seed 1).

**Measured before any run (graph only, no learning):** the longest path / mean path depth of the graphs (regions counted) —
L=8: random 5/3.1 (both seeds), sheet p=0 7/4.1 and 7/4.0, p=0.2 7/4.1 and 6/3.6; L=16: random 6/4.2 and 7/4.4, sheet p=0
11/6.1 and 13/7.1, p=0.2 11/6.1 and 12/6.9. The sheet ordered on one axis is much deeper and more chain-like: the random graph
crowds its sources on the first regions.

**Hypothesis (one, with a falsifier).** Near wiring gives longer paths, so contexts of more lengths; with p=0.2 the long-range
edges keep short paths too. Prediction: sheet p=0.2 ≥ random graph, mean of seeds 1 and 2, by about +0.005 on Moby Dick at L=8
(about 0.399 against 0.394) and +0.01 at L=16 (where depths differ most); sheet p=0 between the chain (0.381) and the random graph,
about 0.388 at L=8. Code: the same order. **Refuted** if sheet p=0.2 is below the random graph by more than 0.005 (mean of two
seeds) on both streams at both L. A difference within ±0.005 is read as no difference (seed spread of earlier runs ~0.003-0.005).

**Runs.** t17_general.py, T=16000 read, NT=10000 scored from the middle, learning off; streams moby and code; L=8 and L=16;
graphs random / sheet p=0 / sheet p=0.2; seeds 1, 2: 24 runs. OMP_NUM_THREADS=1, at most 4 at once. Cost estimate from a
1,500-byte trial (5 s at L=8): about 1.5 min per L=8 run, 3-5 min per L=16 run; about 1.5 CPU-hours, ~25 min wall. No checkpoints.
Not run: K other than 2, p other than 0 and 0.2, other sheet orders, longer reading.
