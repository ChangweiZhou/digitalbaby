# NEWNET scaling report (22 November 2026)

Plan: `SCALING_PLAN.md` (commit a6c345ce, written before the runs). Source learner: `claude/magical-albattani-sei232` at `dbc8f661`,
unchanged. Learner `net=9 f_mode=order eta_f=0.2 f_M=16`; size factor s sets slow `N=1024*s`, `gran=8192*s`, fast `f_C=128*s`.
Float arrays held: s=0.5 3.0 M, s=1 5.9 M, s=2 11.9 M, s=4 23.8 M. OMP_NUM_THREADS=1, 4 processes at once; logs (command, output,
wall time) in `scaling_runs/`. Run time is wall time with 4 runs sharing the machine.

| s | test | seed | result | run time |
|---|---|---|---|---|
| 0.5 | text held-out, TA=16,000 | 1 | 0.341 | 144 s |
| 0.5 | text held-out, TA=16,000 | 2 | 0.343 | 130 s |
| 0.5 | text held-out, TA=16,000 | 3 | 0.341 | 138 s |
| 1 | text held-out, TA=16,000 | 1 | 0.356 | 265 s |
| 1 | text held-out, TA=16,000 | 2 | 0.349 | 264 s |
| 1 | text held-out, TA=16,000 | 3 | 0.344 | 260 s |
| 2 | text held-out, TA=16,000 | 1 | 0.364 | 548 s |
| 2 | text held-out, TA=16,000 | 2 | 0.351 | 587 s |
| 2 | text held-out, TA=16,000 | 3 | 0.353 | 569 s |
| 4 | text held-out, TA=16,000 | - | NOT_RUN (estimated ~20 min, over budget) | - |
| 0.5 | text held-out, TA=64,000 | 1 | 0.364 | 277 s |
| 1 | text held-out, TA=64,000 | 1 | 0.366 | 514 s |
| 2 | text held-out, TA=64,000 | 1 | too slow: stopped at 1,000 s, no result | 1000 s |
| 4 | text held-out, TA=64,000 | - | NOT_RUN (estimated ~33 min) | - |
| 0.5 | keeping (life=ctx): A after A / A after B / after the return | 1 | 1.000 / 1.000 / 1.000 | 63 s |
| 1 | keeping (life=ctx) | 1 | 1.000 / 1.000 / 1.000 | 110 s |
| 2 | keeping (life=ctx) | 1 | 1.000 / 1.000 / 1.000 | 249 s |
| 4 | keeping (life=ctx) | 1 | 1.000 / 1.000 / 1.000 | 501 s |
| 0.5 | memory (t16 G=16): online at gap / slow alone / fast alone / sum | 1 | 0.998 / 1.00 / 0.52 / 1.00 | 130 s |
| 1 | memory (t16 G=16) | 1 | 0.998 / 1.00 / 0.52 / 1.00 | 224 s |
| 2 | memory (t16 G=16) | 1 | 0.996 / 1.00 / 0.52 / 0.99 | 502 s |
| 4 | memory (t16 G=16) | 1 | 0.996 / 1.00 / 0.52 / 1.00 | 884 s |

Text held-out = right share on 10,000 never-read bytes of tinyshakespeare (from 600,000) after reading the first TA bytes; standard
error of one test about 0.005. (The keeping runs' text test after B, for reference: 0.332 / 0.343 / 0.351 / 0.352 at s = 0.5 / 1 / 2 / 4.)

## Reading against the plan

- **Text, TA=16,000, three-seed means: s=0.5 0.342, s=1 0.350, s=2 0.356.** Larger is better in direction on every seed pair
  (s=1 over s=0.5: +0.015, +0.006, +0.003; s=2 over s=1: +0.008, +0.002, +0.009), but the gains (+0.008 and +0.006 per doubling) are
  smaller than the 0.01 the plan required. **H-scale is refuted as stated**: doubling all three sizes gives under 0.01 of text held-out
  here, for twice the run time. Predictions (0.33 / 0.35 / 0.36) were close in level, but the s=0.5 learner is better than predicted.
- **Reading longer, seed 1:** TA=64,000 gives 0.364 at s=0.5 and 0.366 at s=1 (+0.002, within noise); s=2 did not finish within
  1,000 s. Reading four times as much gains +0.023 (s=0.5) and +0.010 (s=1): more reading helps more than more size, and the prediction
  that a larger learner gains more from reading longer is not borne out (it gained less).
- **Keeping and memory do not depend on size** from s=0.5 to s=4: A after B 1.000 at every size; t16 slow alone 1.00 and sum 0.99-1.00
  at every size; the fast compartment alone stays at 0.52 (chance between the two end bytes, as in step 7).

Limits: one seed for everything but text TA=16,000; no s=4 text result; the cost per byte roughly doubles with s.
