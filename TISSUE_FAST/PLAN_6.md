# Plan 6: fewer and stronger still (21 November 2026)

Written before any run of this plan. Development level; one seed; no gate, seal or outside review is claimed.

**Authorisation.** Owner, 21 November 2026: "我们至少需要传递到十层以上 … 继续做实验". **Program:** `TISSUE_EVENT/` at the commit of this plan (unchanged since plan 5).

**Why.** Plan 5: with the total incoming strength held, about 10 inputs a cell (making period 8,192 steps, trial threshold 1) carried the byte to d = 6 in the coincidence form and d = 4 in the first form, against d = 3 with 18; the number was still rising (+7 to +10% over the last 4,096 bytes).

**Arms** (each on plan 5's F of its form: C_E 3/4, C_I 1/4, turnover with trial and withdrawal by use; trial 2,048 steps in the first form, 512 in the coincidence form):

| Arm | Change from F | Bytes |
|---|---|---|
| F16 | making period 16,384 | 16,384 |
| F32 | making period 32,768 | 16,384 |
| F8t2 | trial threshold 2 (a new connection must be twice as useful as the cell's mean) | 16,384 |
| Flong | none (coincidence form only): F over 49,152 bytes, to see where the number of connections settles | 49,152 |

**H6.** Some arm of the coincidence form has S(d,0) in A2 at least 0.1103 at d = 8 or beyond, with at most 1,024 cells active at every stretch end and S(1,0) at least 0.1103. **Refuted if** none does.

| No. | Prediction | Chance |
|---|---|---|
| H1 | Some arm of either form carries the byte farther (farthest d at level) than F of that form | 0.5 |
| H2 | H6 holds | 0.2 |
| H3 | In Flong the number of connections a cell changes by less than 5% over its last 8,192 bytes | 0.5 |
| H4 | At F32, some cells end with no connection from another cell (the starting 8 lost and none kept) | 0.4 |

**Arms run:** 3 × 2 forms + Flong = 7 tissues. **Cost:** F cost 1.06 and 1.78 ms a byte; Flong about 4 minutes; all under 30 minutes on 4 cores. **Not run:** STDP; other seeds; a larger tissue.

## Outcome (entered 21 November 2026; seven runs, exit 0; `p6_tables.py`)

### first

| Arm | S(d,0) in A2 for d = 1 ... 11 | Farthest d at level | Connections a cell by 4,096 bytes | Change over the last 4,096 (Flong: 8,192) bytes | Fewest connections into a non-feedback cell at the end | Active at stretch ends, min / max | ms a byte | H6 met |
|---|---|---|---|---|---|---|---|---|
| F (plan 5) | 1.000 0.964 0.688 0.381 0.046 0.049 0.023 0.022 0.022 0.022 0.022 | 4 | 7.8 / 8.7 / 9.5 / 10.2 | +6.7% | 1 | 60 / 195 | 1.06 | no |
| F16 | 0.998 0.984 0.834 0.468 0.190 0.096 0.036 0.022 0.022 0.022 0.022 | 5 | 7.1 / 7.6 / 8.1 / 8.4 | +4.1% | 1 | 6 / 284 | 0.80 | no |
| F32 | 1.000 0.971 0.800 0.499 0.250 0.179 0.022 0.022 0.022 0.022 0.022 | 6 | 6.9 / 7.1 / 7.3 / 7.5 | +2.8% | 1 | 2 / 190 | 0.60 | no |
| F8t2 | 1.000 0.977 0.833 0.501 0.149 0.096 0.050 0.022 0.022 0.022 0.022 | 5 | 7.4 / 8.0 / 8.5 / 9.0 | +5.8% | 1 | 49 / 481 | 0.93 | no |

### coincidence

| Arm | S(d,0) in A2 for d = 1 ... 11 | Farthest d at level | Connections a cell by 4,096 bytes | Change over the last 4,096 (Flong: 8,192) bytes | Fewest connections into a non-feedback cell at the end | Active at stretch ends, min / max | ms a byte | H6 met |
|---|---|---|---|---|---|---|---|---|
| F (plan 5) | 1.000 0.987 0.795 0.558 0.268 0.162 0.070 0.022 0.022 0.022 0.022 | 6 | 7.4 / 8.2 / 9.0 / 9.9 | +9.6% | 1 | 67 / 498 | 1.78 | no |
| F16 | 1.000 0.981 0.915 0.643 0.196 0.091 0.044 0.022 0.022 0.022 0.022 | 5 | 6.9 / 7.3 / 7.7 / 8.1 | +4.0% | 1 | 10 / 363 | 1.35 | no |
| F32 | 1.000 0.982 0.925 0.618 0.321 0.099 0.031 0.022 0.022 0.022 0.022 | 5 | 6.7 / 6.8 / 7.0 / 7.1 | +1.5% | 1 | 10 / 162 | 1.12 | no |
| F8t2 | 1.000 0.996 0.867 0.659 0.413 0.162 0.093 0.022 0.022 0.022 0.022 | 6 | 7.1 / 7.6 / 8.1 / 8.5 | +5.6% | 1 | 59 / 544 | 1.60 | no |
| Flong | 1.000 0.987 0.795 0.558 0.268 0.162 0.070 0.022 0.022 0.022 0.022 | 6 | 7.4 / 8.2 / 9.0 / 9.9 / 10.7 / 11.4 / 12.1 / 12.7 / 13.3 / 13.8 / 14.3 / 14.7 | +6.3% | 1 | 67 / 695 | 2.69 | no |


Flong's readings are of the same window A2 (bytes 12,288 to 16,383) as F's, so its S(d,0) are F's; its value is the number of connections over 49,152 bytes. "Fewest connections into a non-feedback cell" is E63's count of ordinary connections, lines included, at the last count.

| | Outcome |
|---|---|
| H6 | **Refuted.** No arm reaches d = 8: S(8,0) is 0.022 (chance) in every arm; the farthest at level is 6 (first form F32, S(6,0) 0.179; coincidence form F8t2, S(6,0) 0.162, S(7,0) 0.093) |
| H1 | Held: in the first form F16, F32 and F8t2 reach 5 to 6 against F's 4 |
| H2 | Failed |
| H3 | Failed: in Flong the number keeps rising (9.9 at 16,384 bytes, 14.7 at 49,152; +6.3% over the last 8,192 bytes) |
| H4 | Not decided from the records: the fewest ordinary connections into a non-feedback cell is 1 in every arm, and the count does not tell lines from cells |

**Readings (development level, one seed).**

1. **Fewer inputs keep helping up to a point.** With about 7 inputs a cell (making period 32,768) the first form carries the byte to d = 6 (0.998, 0.971, 0.800, 0.499, 0.250, 0.179); the coincidence form gains little more beyond F (to d = 6 with 8.5 inputs at threshold 2). Beyond d = 7 every arm is at chance.
2. **The number of connections does not settle under turnover with this trial**: it rises slowly in every arm, by +1.5% to +6.3% over the last stretch, and in the long life from 9.9 to 14.7. The rate of making sets the rise; withdrawal does not keep pace. A rule that holds the number in a long life is still to be found.
3. **The cost keeps falling with fewer connections**: 0.60 ms a byte (first form F32), 1.12 (coincidence F32).
