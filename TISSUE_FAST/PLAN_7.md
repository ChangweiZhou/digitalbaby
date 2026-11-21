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

## Outcome

(to be entered after the runs)
