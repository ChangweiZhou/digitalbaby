# The output end on tissue (vi) and on the tissue with no memory (19 November 2026)

This is item 3 of the reviewer's message to E57's worker after 40d93aea2.

**Stopped.** The reviewer stopped the work after three seeds: Mr. Xu withdrew the 256 input and output lines with drivers at x = 1 (WHAT_A_NEURON_HAS_20261118.md section 20 on `claude/vibrant-cerf-sje726`). Everything here is measured at those withdrawn output lines.

**Not run:**
- seeds 465973, 614332, 406364 and 545353 (three of these runs were stopped part-way, and their logs deleted);
- the split of (vi) into delays alone and slow cells alone (`q_split.sh`, never started);
- item 4.

## What is in this folder

- **`outq.py`**: the reviewer's `outq.py` (reviewer_tissue_20261119 at 6868b178a), copied. The only change is the tissue: TISSUE_LONG's `tissue.c` and `tis.py`, with delays drawn as in `TISSUE_LONG/delay.py` (`cand=rand`).
- **`deps/`**:
  - the reviewer's `score_emissions.py` and `life_stream.py` from `yesno/deps`, copied;
  - TISSUE_LONG's dependencies.
- **`q_outq.sh`**: the runs. **`outq_tables.py`**: writes `OUTQ_TABLES.md` from `outq/*.json`.

## Set-up

- **Line**: `rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/16384 make_period=1024 start_w=1/64`, 65,536 bytes, **no long axons**.
- **The two tissues**:
  - **(vi)** adds `cand=rand dmax=8 tau=f4-64`: delays 0–8, and one cell in 4 at 64 steps.
  - **No memory** adds nothing.
- **Output end**: inputs 'own' (unit v sees the neighbourhood of the v-th output-line cell; 25 to 117 cells). Rules D (eta 1/64) and Q (eta 1/16, z1). Each with the input alone and with slow copies 2,4, which carry `outq.py`'s sqrt(2k − 1) scaling.

## Results: share right in the last quarter, %

| Seed | ref_1 | ref_2 | (vi) D | (vi) Q | (vi) D copies | (vi) Q copies | none D | none Q | none D copies | none Q copies |
|---|---|---|---|---|---|---|---|---|---|---|
| 73129 | 10.55 | 12.34 | 8.36 | 8.10 | 8.93 | 8.33 | 7.69 | 7.62 | 10.88 | 8.79 |
| 93683 | 10.62 | 12.14 | 6.65 | 5.71 | 6.88 | 6.16 | 9.93 | 9.91 | 13.43 | 10.44 |
| 116089 | 10.61 | 12.20 | 8.92 | 8.27 | 9.30 | 8.56 | 10.93 | 11.08 | 14.37 | 13.18 |

- **The tissue with no memory gives the output end more than (vi) in 10 of 12 entries.** The exceptions are seed 73129 with the input alone.
- **The copies help the tissue with no memory most.**
  - Its D with copies is 10.88, 13.43 and 14.37, above ref_2 on 93683 and 116089.
  - (vi) with copies stays below ref_1 on all three seeds.
- Every figure is from one run, and nothing is confirmed independently. The quarters are in `OUTQ_TABLES.md`.
- **Time** (wall clock of one-thread processes, from the logs): a life took 1,025–1,053 s for (vi) and 599–689 s for no memory. The output end added about 2 minutes per life.
