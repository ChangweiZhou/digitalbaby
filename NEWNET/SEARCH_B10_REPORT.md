# Search batch 10: six batch-5/6/8 architectures on seeds 2 and 3

Date 22 November 2026. Helper session for NEWNET. No code was changed. Source commit `e166d48e` (branch
`claude/magical-albattani-sei232`). Every candidate was run with `SEARCH_OUT=search_results_b10.tsv python3 search_arch.py NAME "OPTS seed=S"
tests=moby,code,keep,keep4,keep2,mem`, one candidate at a time. Each candidate's six tests ran in parallel on 4 cores with OMP_NUM_THREADS=1.
Raw lines are in `search_results_b10.tsv` (72 lines: 12 runs × 6 tests). Seed 1 values come from `search_results.tsv`, the batch 8 rows for
H3–H6 and the batch 5/6 rows for E4 and F2. **E4 and F2 on seed 1 had only moby, code, keep and mem.** keep4 and keep2 were not run on that seed, so
their seed-1 score is the mean of four tests. It is not comparable with the six-test score and is left out of the per-architecture score mean.

Architectures (options as in the batch scripts; seed added):

| | options |
|---|---|
| H3 | `L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=2 c_k_win=4 c_th_on=0.9 c_th_match=0.75` (E6 with stricter thresholds) |
| H4 | `L=12 M=11 joint=1 mesh=2 mesh_base=0 mesh_dag=2 c_k_win=4` |
| H5 | `L=6 M=21 joint=1 mesh=2 mesh_base=0 mesh_dag=1 c_k_win=4` (graph without loops) |
| H6 | `L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=1` (graph without loops, baseline) |
| E4 | `L=8 joint=1 c_k_win=4` (chain, 4 winners) |
| F2 | `L=16 cC=64 joint=1 mesh=1 mesh_base=0 mesh_dag=5` (chain beside a graph) |

## Every run

Score is the mean of the six tests (for E4 and F2 on seed 1, the mean of four). Run time is the wall time of the slowest of the
candidate's parallel tests, in seconds. Seed-1 times come from earlier batches, which may have run under different machine load, so they are not
a clean comparison.

| candidate | seed | moby | code | keep (G=8) | keep4 | keep2 | mem | score | time (s) |
|---|---|---|---|---|---|---|---|---|---|
| H3 | 1 | 0.382 | 0.389 | 1.000 | 1.000 | 1.000 | 1.000 | 0.795 | 189 |
| H3 | 2 | 0.382 | 0.377 | 1.000 | 1.000 | 1.000 | 1.000 | 0.793 | 142 |
| H3 | 3 | 0.378 | 0.381 | 1.000 | 1.000 | 1.000 | 1.000 | 0.793 | 141 |
| H4 | 1 | 0.382 | 0.366 | 1.000 | 1.000 | 1.000 | 1.000 | 0.791 | 228 |
| H4 | 2 | 0.373 | 0.364 | 1.000 | 1.000 | 1.000 | 1.000 | 0.789 | 168 |
| H4 | 3 | 0.369 | 0.349 | 1.000 | 1.000 | 1.000 | 1.000 | 0.786 | 167 |
| H5 | 1 | 0.396 | 0.402 | 0.470 | 1.000 | 1.000 | 1.000 | 0.711 | 144 |
| H5 | 2 | 0.395 | 0.405 | 0.480 | 1.000 | 1.000 | 1.000 | 0.713 | 109 |
| H5 | 3 | 0.395 | 0.402 | 0.540 | 1.000 | 1.000 | 1.000 | 0.723 | 112 |
| H6 | 1 | 0.394 | 0.392 | 0.470 | 1.000 | 1.000 | 1.000 | 0.709 | 218 |
| H6 | 2 | 0.391 | 0.400 | 0.480 | 1.000 | 1.000 | 1.000 | 0.712 | 154 |
| H6 | 3 | 0.393 | 0.392 | 0.540 | 1.000 | 1.000 | 1.000 | 0.721 | 162 |
| E4 | 1 | 0.375 | 0.384 | 1.000 | not run | not run | 1.000 | (0.690, 4 tests) | 110 |
| E4 | 2 | 0.383 | 0.384 | 1.000 | 1.000 | 1.000 | 1.000 | 0.795 | 138 |
| E4 | 3 | 0.385 | 0.371 | 1.000 | 1.000 | 1.000 | 1.000 | 0.793 | 134 |
| F2 | 1 | 0.388 | 0.378 | 1.000 | not run | not run | 1.000 | (0.692, 4 tests) | 184 |
| F2 | 2 | 0.384 | 0.374 | 0.930 | 1.000 | 1.000 | 1.000 | 0.781 | 222 |
| F2 | 3 | 0.384 | 0.354 | 1.000 | 1.000 | 1.000 | 1.000 | 0.790 | 228 |

## Per architecture: mean [min–max] over seeds

Seeds 1–3, except E4/F2 keep4, keep2 and score, which use seeds 2–3 only (n=2).

| | moby | code | keep (G=8) | keep4 | keep2 | mem | score (6 tests) |
|---|---|---|---|---|---|---|---|
| H3 | 0.381 [0.378–0.382] | 0.382 [0.377–0.389] | 1.000 [1.000–1.000] | 1.000 | 1.000 | 1.000 | 0.794 [0.793–0.795] |
| H4 | 0.375 [0.369–0.382] | 0.360 [0.349–0.366] | 1.000 [1.000–1.000] | 1.000 | 1.000 | 1.000 | 0.789 [0.786–0.791] |
| H5 | 0.395 [0.395–0.396] | 0.403 [0.402–0.405] | 0.497 [0.470–0.540] | 1.000 | 1.000 | 1.000 | 0.716 [0.711–0.723] |
| H6 | 0.393 [0.391–0.394] | 0.395 [0.392–0.400] | 0.497 [0.470–0.540] | 1.000 | 1.000 | 1.000 | 0.714 [0.709–0.721] |
| E4 | 0.381 [0.375–0.385] | 0.380 [0.371–0.384] | 1.000 [1.000–1.000] | 1.000 (n=2) | 1.000 (n=2) | 1.000 | 0.794 [0.793–0.795] (n=2) |
| F2 | 0.385 [0.384–0.388] | 0.369 [0.354–0.378] | 0.977 [0.930–1.000] | 1.000 (n=2) | 1.000 (n=2) | 1.000 | 0.785 [0.781–0.790] (n=2) |

## Which differences are larger than the seed spread

The test used here: two architectures differ when their seed ranges do not overlap. With 2–3 seeds this test is weak. It sets no
confidence level, and gaps of 0.001–0.005 between ranges count as "separated" only in this narrow sense.

- **keep (G=8): the large, clear difference.** Every architecture with a serial chain (H3, H4, E4: 1.000 on every seed; F2: 0.930–1.000)
  lies far above the graphs without loops (H5, H6: 0.470–0.540 on every seed). The two groups are separated by about 0.4. This is what
  drives the score gap between the groups (0.781–0.795 against 0.709–0.723).
- **keep4, keep2, mem:** 1.000 for every architecture on every seed run, so these tests show no differences.
- **moby:** H5 (0.395–0.396) and H6 (0.391–0.394) are above every chain architecture (highest chain value 0.388, F2 seed 1), by 0.003 or
  more. H5 against H6 is separated by 0.001, which is not a meaningful gap. Among H3, H4, E4 and F2 the ranges overlap: no difference.
- **code:** H5 (0.402–0.405) is above H6 (0.392–0.400) by 0.002, again barely. H6 is above every chain architecture (highest 0.389, H3 seed 1).
  Among the chains, H4 (0.349–0.366) is below H3 (0.377–0.389) and below E4 (0.371–0.384). F2 (0.354–0.378) overlaps H3, H4 and E4.
- **score among the chain architectures:** H3 and E4 (0.793–0.795) are above H4 (0.786–0.791) by at least 0.002. E4 has only 2 six-test
  seeds. H3 and E4 do not differ. F2 (0.781–0.790) overlaps H4 and is below H3/E4 by 0.003 at the closest. That is within F2's own spread of 0.009, so it is not
  a clear difference. H5 and H6 do not differ in score (ranges overlap).

So, measured here: the graphs without loops read the two never-used streams a little better (about +0.01 to +0.04). They fail keep at G=8
on all three seeds. The chain-containing architectures keep at G=8 on all three seeds, except F2 seed 2 (0.930). Batch 8's E6 seed 3
failure (0.540) is not repeated by H3, which is E6 with stricter thresholds, on seed 3 (1.000). E6 itself was not rerun in this batch.

## Limits

Development-level search runs only: seed 1 and seeds 2–3 come from different batches and possibly different machine load (times
differ). Three seeds at most, with no confidence intervals. No architecture or code was changed and no further runs were made.
