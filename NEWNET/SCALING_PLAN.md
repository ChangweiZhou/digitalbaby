# NEWNET scaling plan (22 November 2026, written before the runs)

Source: branch `claude/magical-albattani-sei232` at `dbc8f661` (NEWNET step 8), measured on branch `claude/newnet-scaling-sei232`.
Authorization: a helper task given by the NEWNET line ("measure whether the same learner gets better when it is made larger; do not
change any learning rule"). No existing file is edited; no learning rule or default is changed.

## Learner under test

`net=9 f_mode=order eta_f=0.2 f_M=16`, all other defaults (slow stage net8: N=1024 cells, expansion gran=8192; fast stage net7:
f_C=128 columns, f_L=8 regions, 16 cells a column). Size factor s scales three numbers together and nothing else:
slow `N=1024*s`, `gran=8192*s`, fast `f_C=128*s`.

Physical size (all float arrays the learner holds: weights, eligibility traces and state; counted by instantiating net9 with
n_values=128): s=0.5 3.0 M, s=1 5.9 M, s=2 11.9 M, s=4 23.8 M numbers.

## Tests (existing scripts, unchanged)

1. Text held-out: `t13_life8.py net=9 f_mode=order eta_f=0.2 f_M=16 life=texts TB=0 TR=0 NT=10000 TA=16000` (seeds 1, 2, 3)
   and `TA=64000` (seed 1). Result: the A test after A (right share on 10,000 bytes of tinyshakespeare never read, from byte 600,000).
   Standard error of one 10,000-byte test about 0.005.
2. Keeping: `t13_life8.py net=9 f_mode=order eta_f=0.2 f_M=16 life=ctx` (A on letters G=8, 4,000 bytes; then 16,000 bytes of text).
   Result: A after B (and A after A, text test after B).
3. Memory: `t16_split.py fbm=teach G=16 T=60000 n_values=128` plus the size arguments. Result: branch right after the gap from the
   slow compartment alone, the fast alone, the sum; online at the gap.

## Hypothesis and predictions

**H-scale: text held-out rises with s.** Predicted means (TA=16,000, three seeds): s=0.5 0.33, s=1 0.35 (step 6/8 measured 0.349-0.361),
s=2 0.36. TA=64,000 (seed 1): s=0.5 0.35, s=1 0.37, s=2 0.39 (a larger learner gains more from reading longer).
Keeping and memory are expected not to depend on s at these sizes: A after B 1.00 at every s; t16 slow alone and sum 0.95 or more
at every s (possibly lower at s=0.5).

**Refuted if** the three-seed mean at s=2 is not above s=1 by more than 0.01 (two standard errors of one test), or s=0.5 is not below
s=1 by more than 0.01; and, for reading longer, if at TA=64,000 s=2 is not above s=1 by more than 0.01. A drop of A after B below 0.9
or of t16 sum below 0.9 at any size is reported as a size effect on keeping / memory.

## Cost estimate (measured on a 5,000-step probe with 4 processes at once: about 2, 3.4, 7.4, 16 ms a byte at s = 0.5, 1, 2, 4)

| test | steps | s=0.5 | s=1 | s=2 | s=4 |
|---|---|---|---|---|---|
| text TA=16,000 (16,000 + 6 tests of 10,000) | 76,000 | 2.5 min | 4.3 | 9.4 | ~20: over budget, NOT_RUN |
| text TA=64,000 | 124,000 | 4 | 7 | ~15 (borderline) | ~33: NOT_RUN |
| keeping (life=ctx) | ~31,000 | 1 | 2 | 4 | 8 |
| memory (t16, T=60,000) | ~63,000 | 2 | 3.6 | 8 | ~17 (borderline) |

Total about 2 CPU-hours, about 35-45 minutes of wall time with 4 processes (OMP_NUM_THREADS=1). Each run is stopped at 1,000 s; a run
stopped that way is recorded as too slow, not as a result.

## Honest limits

One learner, one text pair, three seeds at most; s=4 only where it fits the time budget. A size effect within 0.01 is not
distinguishable from seed noise here. The tests repeat the A test three times with no learning between (TB=0 TR=0); only the first
(after A) is reported for test 1.
