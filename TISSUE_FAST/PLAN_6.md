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

## Outcome

(to be entered after the runs)
