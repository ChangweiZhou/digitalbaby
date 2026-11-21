# Plan 7: activity that neither floods nor dies: excitation and inhibition in balance (21 November 2026)

Written before any run of this plan. Development level; one seed; no gate, seal or outside review is claimed.

**Authorisation.** Owner, 21 November 2026: "继续做。如果信息无法传递，从老鼠到大象也就无法移动了，除非我们有几米长的神经细胞，显然是荒谬的。" **Program:** `TISSUE_EVENT/` at the commit of this plan. With the new switches (`--spike`, `--inh-adapt`) off it gives the same numbers as for plans 5 and 6 (checked with `==`, 2,048 bytes).

**Why.** Plan 6: at d >= 8 every cell is silent (share of active cells at the reading steps 0.000): activity, not only information, dies out with distance. Development runs of this session on seed 73129 (`checks/`: short probes of 4,096 to 6,144 bytes, outputs at the end of each 1,024 bytes; development, not evidence) found:
- raising the excitatory total to 1.5 with inhibition at a third of it: the first form floods (about 1,900 of 2,048 active), the coincidence form dies (14 active);
- all-or-none outputs at firing levels 1/8, 1/32 and 1/64: the tissue falls silent (they need a design of their own: refractoriness, the threshold's dynamics, readings over a whole byte; not in this plan);
- an inhibitory total that follows each cell's activity once a byte: still floods or dies, or swings between the two;
- **excitatory and inhibitory totals equal (1.5 and 1.5, or 1.5 and 2.25): the first form holds about 1,000 to 1,200 cells active, spread over every distance from 1 to 11; the coincidence form carries waves of activity that reach d = 9 to 11.**

In balanced networks of the literature, inhibition as strong as excitation or stronger keeps activity in a middle range. Whether the activity that now reaches far cells carries the byte is what this plan measures.

**Arms** (both forms; each on plan 6's F32 of its form: making period 32,768, trial threshold 1, trial 2,048 steps first form and 512 coincidence form, turnover and withdrawal by use; graded outputs):

| Arm | C_E, C_I |
|---|---|
| B075 | 3/4, 3/4 |
| B15 | 1.5, 1.5 |
| B15i | 1.5, 2.25 |
| B3 | 3, 3 |

References: plan 6's F32 (3/4, 1/4).

**H7.** In some arm of some form, S(d,0) in A2 is at least 0.1103 at d = 8 or beyond, with S(1,0) at least 0.1103 and at most 1,536 active cells (three quarters) at the end of every 1,024-byte stretch. **Refuted if** none.

| No. | Prediction | Chance |
|---|---|---|
| I1 | In some arm the share of active cells at the reading steps is above 0.05 at every distance 1 to 9 in window A2 | 0.6 |
| I2 | H7 holds | 0.25 |
| I3 | Some arm carries the byte farther (farthest d at level) than F32 of its form | 0.4 |
| I4 | B3 floods (above 1,536 active at some stretch end) in some form | 0.5 |

**Arms run:** 4 × 2 = 8 tissues; 16,384 bytes; counts every 1,024 bytes; E63's readings and `depth_read.py`. **Cost** up to about 5 ms a byte: under 30 minutes on 4 cores. **Not run:** all-or-none outputs; inhibition that follows activity; other seeds.

## Outcome (entered 21 November 2026; eight runs, exit 0; `p7_tables.py`)

### first

| Arm | S(d,0) for d = 1 ... 11 | Active share at reading steps, d = 1 ... 11 | Farthest d at level | Connections at the end | Active at stretch ends, min / max | ms a byte | H7 met |
|---|---|---|---|---|---|---|---|
| F32 (plan 6) | 1.000 0.971 0.800 0.499 0.250 0.179 0.022 0.022 0.022 0.022 0.022 | 0.17 0.16 0.10 0.05 0.03 0.01 0.00 0.00 0.00 0.00 0.00 | 6 | 7.5 | 2 / 190 | 0.60 | no |
| B075 | 1.000 0.949 0.762 0.508 0.121 0.124 0.022 0.022 0.022 0.022 0.022 | 0.17 0.16 0.10 0.04 0.01 0.01 0.00 0.00 0.00 0.00 0.00 | 6 | 7.3 | 10 / 267 | 0.57 | no |
| B15 | 0.755 0.429 0.263 0.151 0.103 0.081 0.067 0.062 0.059 0.047 0.033 | 0.47 0.39 0.41 0.45 0.45 0.43 0.53 0.47 0.49 0.47 0.54 | 4 | 3.7 | 527 / 1252 | 4.54 | no |
| B15i | 0.688 0.349 0.159 0.067 0.052 0.049 0.040 0.039 0.029 0.024 0.025 | 0.53 0.44 0.40 0.45 0.49 0.53 0.56 0.56 0.48 0.47 0.53 | 3 | 3.7 | 577 / 1325 | 4.66 | no |
| B3 | 0.049 0.029 0.025 0.024 0.028 0.021 0.021 0.022 0.020 0.025 0.032 | 0.35 0.35 0.37 0.37 0.37 0.37 0.36 0.35 0.36 0.35 0.34 | 0 | 5.5 | 320 / 819 | 4.76 | no |

### coincidence

| Arm | S(d,0) for d = 1 ... 11 | Active share at reading steps, d = 1 ... 11 | Farthest d at level | Connections at the end | Active at stretch ends, min / max | ms a byte | H7 met |
|---|---|---|---|---|---|---|---|
| F32 (plan 6) | 1.000 0.982 0.925 0.618 0.321 0.099 0.031 0.022 0.022 0.022 0.022 | 0.19 0.18 0.14 0.06 0.02 0.01 0.00 0.00 0.00 0.00 0.00 | 5 | 7.1 | 10 / 162 | 1.12 | no |
| B075 | 0.990 0.967 0.886 0.660 0.283 0.149 0.160 0.103 0.022 0.022 0.022 | 0.17 0.17 0.14 0.09 0.03 0.01 0.00 0.00 0.00 0.00 0.00 | 7 | 6.9 | 10 / 175 | 1.15 | no |
| B15 | 0.738 0.384 0.105 0.030 0.026 0.026 0.026 0.026 0.029 0.028 0.026 | 0.28 0.25 0.17 0.11 0.11 0.11 0.11 0.11 0.11 0.10 0.10 | 2 | 5.0 | 48 / 1376 | 2.22 | no |
| B15i | 0.784 0.385 0.064 0.023 0.023 0.024 0.020 0.019 0.019 0.019 0.023 | 0.25 0.22 0.18 0.15 0.14 0.13 0.12 0.10 0.11 0.11 0.12 | 2 | 4.9 | 14 / 1467 | 2.17 | no |
| B3 | 0.575 0.233 0.044 0.018 0.018 0.020 0.022 0.017 0.018 0.017 0.023 | 0.18 0.18 0.15 0.13 0.14 0.14 0.14 0.14 0.13 0.13 0.12 | 2 | 5.6 | 13 / 1300 | 2.20 | no |


| | Outcome |
|---|---|
| H7 | **Refuted.** No arm is at level at d = 8. Closest: coincidence form B075 (C_E = C_I = 3/4), at level to d = 7 (S(7,0) 0.160) and 0.103 at d = 8 |
| I1 | Held: B15, B15i and B3 have active cells at every distance (shares 0.10 to 0.56) |
| I2 | Failed |
| I3 | Held: coincidence B075 reaches d = 7 against F32's 5 |
| I4 | Failed: no arm exceeds 1,536 active cells at a stretch end (largest 1,467) |

**Readings (development level, one seed).**

1. **Two regimes, and neither is what is wanted.** With weak totals (3/4) activity is driven by the input and carries the byte, but it thins with distance and is gone by d = 7 to 8. With strong balanced totals (1.5 and above) activity reaches every distance (about half the cells in the first form) but it is the tissue's own activity, not the input's: S(d,0) falls to 0.06 by d = 8 in B15 and is at chance at every distance in B3. Activity that travels is not yet information that travels.
2. **Balance at the weak scale helped the coincidence form**: inhibition equal to excitation (B075) carried the byte to d = 7 and nearly to 8, the farthest measured so far, with no more activity than before.
3. Between the two regimes there may be a range where activity is both sustained and driven; the arms here (3/4, then 1.5) step over it.
