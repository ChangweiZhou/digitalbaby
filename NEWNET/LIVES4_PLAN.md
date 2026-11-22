# Four streams in one life: plan (22 November 2026, written before the runs)

Source: branch `claude/newnet-lives4-sei232`, from `4af67ad2` (`claude/magical-albattani-sei232`). Development level; no gate, seal or
outside review is claimed. The learner is unchanged: `net10.py` with the two best structures of the search (`SEARCH_SUMMARY.md`), only
`n_values=256`:

- K8 = `L=16 cC=64 joint=1 mesh=1 mesh_base=0 mesh_dag=5 c_k_win=4 c_th_on=0.9 c_th_match=0.75`
- N1 = `L=8 cC=64 M=32 joint=1 mesh=2 mesh_base=0 mesh_dag=2 c_k_win=4`

## Streams and regions (`t20_lives4.py`)

| stream | source | life reads (bytes) | held-out test (5,001 bytes = 5,000 predictions, never read) |
|---|---|---|---|
| en | tinyshakespeare | 0 - 24,000 | 600,000 - 605,000 |
| zh | Hong Lou Meng, UTF-8 (`NEWNET/data/hongloumeng.txt`, gutenberg.pglaf.org/2/4/2/6/24264/24264-0.txt, sha256 824b0bae..., not tracked) | 100,000 - 124,000 | 1,500,000 - 1,505,000 |
| code | tracked `.py` files outside `NEWNET/` (`git ls-files`, 3,827 files, sorted path order, concatenated; 32,021,879 bytes) | 0 - 24,000 | 16,000,000 - 16,005,000 |
| num | lines `a+b=c\n`, a, b uniform in 0..999, numpy `default_rng(12345)`, 200,000+ bytes | 0 - 24,000 | 100,000 - 105,000 |

Life: blocks of 8,000 bytes, en, zh, code, num, for 3 rounds (12 blocks, 96,000 bytes), nothing reset, no task signal; each block
continues where its stream stopped. After every block, each stream's held-out test with learning off, each on a copy of the network
(so the tests do not change the life). Control: each stream alone, the same 24,000 bytes in 3 blocks, its own test after each block.
Seed 1 for both structures; seeds 2 and 3 for K8.

## Hypothesis (one, with a falsifier)

H: in a life alternating four streams, what a structure learned from a stream mostly survives the 24,000 bytes of the three others,
and the losses shrink as the life goes on; K8 keeps better than N1 (as for English/Chinese: K8 98% / 91%, N1 87% / 79%).

Predicted numbers (absolute right share; "fall" = score just after the stream's block minus the score just before its next block):

1. zh falls least (it shares almost no bytes with the others): K8 fall <= 0.015 in round 1.
2. en, code and num share ASCII (letters, digits); their K8 falls are larger, 0.02-0.06 in round 1. N1 falls about twice K8's.
3. The fall shrinks: for K8, the round-2 fall is smaller than the round-1 fall for at least 3 of the 4 streams (round 3 has no
   next block; its fall to the end of the life is reported but not part of the prediction).
4. At the end of each round the life's score on each stream is >= 0.90 of its control after the same number of its own bytes, for K8
   (N1 lower than K8 on most streams).

Refuted if: (a) the round-2 fall is not smaller than the round-1 fall for 2 or more of the 4 streams (K8, mean over seeds); (b) any
stream ends a round below 0.80 of its control for K8; (c) N1's falls are not larger than K8's on most streams. Partly refuted if only
the numeric ranges in 1-2 miss.

Seen before writing this: one timing run only (seed 99, blocks of 1,000 bytes, 1 round, 1,000-byte tests; K8: en 0.284 after its block,
0.167 after code), not part of the result.

## Cost

Timing (one core): K8 about 3.7 ms a byte, N1 about 2.6 ms (learning or not). A life is 96,000 learned + 240,000 test bytes: K8
about 21 min, N1 about 15 min. A control is 24,000 + 15,000 bytes: about 2.5 / 1.7 min. Runs: 3 K8 lives + 12 K8 controls + 1 N1 life
+ 4 N1 controls, about 2 CPU-hours, at most 4 processes at once (OMP_NUM_THREADS=1), about 40 min wall. Memory under 1 GB a process.
NOT_RUN: N1 seeds 2-3; other block lengths or orders; longer lives.
