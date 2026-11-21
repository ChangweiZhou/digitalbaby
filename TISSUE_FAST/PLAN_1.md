# Plan 1: does a cell settle its own number of connections when connections that bring it nothing of use are withdrawn? (21 November 2026)

Written before any run of this plan. Development level; one seed; no gate, seal or outside review is claimed.

## Authorisation and source

- **Owner's instruction (21 November 2026, this session):** do all three items: the program's records made cheaper; the steps a byte and the tolerance made settable numbers; and the number of connections decided by the cells, "which has to be settled by experiment". Earlier ruling (definition section 19, 19 November): each cell makes connections and withdraws those over which it keeps receiving what is of no use to it; the number of connections is the balance of the two.
- **Program:** the event-driven program of E63 (`TISSUE_EVENT/` of `claude/e63-tissue-event-017dSE`, files last changed at cb554a546), with this session's changes, all checked bit for bit against that program with a fixed order of passing on (see `../SPEED_REPORT.md`). The withdrawal by use is the rule already in that program (`--withdraw-by-use`): a connection's use is the running average (1/128) of the sender's trace times the receiving cell's activity, updated only at steps where the receiving cell was active; a connection whose use stays below 1/128 of the largest use into the same cell for 128 such steps, and is at least 128 steps old, is withdrawn.
- **Learner:** E63's tissue, 2,048 cells, neighbourhood radius 1/8 (about 94 cells within reach), starting with 8 connections a cell; first form and coincidence form; the four switches on (output floor 1/128, tolerance 1/1024, one trace a sender, strengths once a byte); 32 steps a byte; no feedback.

## Why

In every run so far a cell ends connected to almost every cell within reach (93.8 a cell in the first form). That number was never chosen as a number of connections: a cell makes a connection every 128 steps while its threshold is 0 and its average activity is below the target a* = 1/8; the measured average is about 0.006, so the condition never ends, and withdrawal by strength (below 1/128) is rare. It is also the main cost of the program: a cap of 16 makes the first form five times cheaper. A cap is a number we fix; the definition asks that the cells settle it.

## The question and the hypothesis

**Question.** With withdrawal by use on, and with the target of activity lowered so that it can be reached, does the number of connections a cell stop growing by itself, and where?

**Hypothesis H1.** With withdrawal by use on and the target at 1/128 or 1/32, the mean number of connections a cell stops growing (changes by less than 5% over the last 4,096 bytes) at less than half the neighbourhood (below 47), in both forms, while the cells one connection from the input lines stay informative (score for the present byte S(1,0) in window A2 at least the level 0.1103).

**Refuted if** in every arm with withdrawal by use the mean ends at 85 or more, or is still growing by more than 5% over the last 4,096 bytes; **or** if the arms where it settles below 47 have S(1,0) in A2 below 0.1103 (the connections that remain no longer carry the byte).

**Predictions** (written now, scored after):

| No. | Prediction | Chance |
|---|---|---|
| P1 | Withdrawal by use alone (target 1/8) does not stop the growth in the first form: the mean ends at 85 or more, because what is withdrawn is made again | 0.7 |
| P2 | Lowering the target alone (1/128, no withdrawal by use) lowers the final mean below 85 in the first form | 0.4 |
| P3 | H1 holds in the first form | 0.35 |
| P4 | H1 holds in the coincidence form | 0.3 |
| P5 | Cost a byte (core) is roughly proportional to connections a cell: the arm with the fewest connections costs at most half of the reference | 0.6 |

## Arms

Seed 73129; E63's stream of stage A; 16,384 bytes; counts every 1,024 bytes (E63: 128; a choice of measurement, not of the tissue); E63's readings of section 11.2 (windows A1 and A2, stretches of 1,024 bytes).

| Arm | Withdrawal by use | Target a* |
|---|---|---|
| R | off | 1/8 (E63) |
| T128 | off | 1/128 |
| W | on | 1/8 |
| W32 | on | 1/32 |
| W128 | on | 1/128 |

Each arm in both plastic forms: 10 tissues.

## Cost

Forecast from this session's measurements: at most about 2 ms a byte with records for the first form and 5 for the coincidence form; about 10 CPU-minutes for all ten tissues. Limit 1 CPU-hour; stop and report above it.

## Not run in this plan

Other seeds; the tissue whose strengths do not change; steps a byte and tolerance other than 32 and 1/1024 (a separate cost-only sweep follows, item 2); any other making rule; feedback; lives longer than 16,384 bytes.

## Outcome (entered 21 November 2026, after all ten runs; `p1_tables.py` reads the receipts in `TISSUE_EVENT/runs/P1_*`)

All ten tissues ran to 16,384 bytes, exit 0; in each, the core run again without records gave the same emissions. About 12 CPU-minutes in all.

### first

| Arm | Connections a cell at bytes 1,024 / 4,096 / 8,192 / 12,288 / 16,384 | Change over the last 4,096 bytes | Made | Given up by strength | Withdrawn by use | ms a byte, core / with records | Emission share | S(1,0), S(2,0), S(4,0) in A2 | Informative share d=1/2/4 in A2 |
|---|---|---|---|---|---|---|---|---|---|
| R | 93.7 / 93.8 / 93.8 / 93.8 / 93.8 | +0.0% | 338,268 | 161,961 | 0 | 1.256 / 1.761 | 0.0006 | 0.6867, 0.2772, 0.0406 | 0.700 / 0.000 / 0.000 |
| T128 | 70.5 / 89.3 / 90.5 / 91.1 / 91.4 | +0.4% | 317,730 | 146,316 | 0 | 1.380 / 1.966 | 0.0003 | 0.9497, 0.5619, 0.0267 | 0.911 / 0.000 / 0.000 |
| W | 90.1 / 92.0 / 92.7 / 93.1 / 93.0 | -0.1% | 1,591,651 | 218,132 | 1,198,914 | 3.340 / 4.079 | 0.0005 | 0.6996, 0.2656, 0.0228 | 0.646 / 0.000 / 0.000 |
| W32 | 89.0 / 92.2 / 92.8 / 92.9 / 92.9 | -0.0% | 1,556,073 | 137,149 | 1,244,606 | 3.670 / 4.469 | 0.0003 | 0.7292, 0.3266, 0.0232 | 0.797 / 0.000 / 0.000 |
| W128 | 76.0 / 87.5 / 88.9 / 89.6 / 90.2 | +0.6% | 1,233,285 | 160,211 | 904,249 | 2.984 / 3.582 | 0.0004 | 0.9318, 0.4795, 0.0225 | 0.928 / 0.000 / 0.000 |

### coincidence

| Arm | Connections a cell at bytes 1,024 / 4,096 / 8,192 / 12,288 / 16,384 | Change over the last 4,096 bytes | Made | Given up by strength | Withdrawn by use | ms a byte, core / with records | Emission share | S(1,0), S(2,0), S(4,0) in A2 | Informative share d=1/2/4 in A2 |
|---|---|---|---|---|---|---|---|---|---|
| R | 29.6 / 31.3 / 32.9 / 31.0 / 33.1 | +6.9% | 2,342,611 | 2,290,745 | 0 | 3.189 / 3.639 | 0.0567 | 0.3039, 0.0287, 0.0268 | 0.574 / 0.989 / 1.000 |
| T128 | 54.1 / 69.8 / 72.8 / 65.6 / 73.0 | +11.2% | 1,542,602 | 1,408,975 | 0 | 3.955 / 4.568 | 0.0129 | 0.8888, 0.1074, 0.0259 | 0.928 / 0.966 / 0.903 |
| W | 23.3 / 22.1 / 24.5 / 25.2 / 23.1 | -8.5% | 2,790,571 | 2,043,499 | 715,714 | 3.684 / 4.213 | 0.0430 | 0.4091, 0.0236, 0.0198 | 0.477 / 0.906 / 1.000 |
| W32 | 35.7 / 47.2 / 55.8 / 55.1 / 52.6 | -4.6% | 2,145,524 | 1,465,514 | 588,191 | 6.585 / 7.191 | 0.0589 | 0.5255, 0.0265, 0.0190 | 0.785 / 0.990 / 1.000 |
| W128 | 44.0 / 67.3 / 66.6 / 70.3 / 63.0 | -10.4% | 2,144,203 | 1,282,317 | 748,733 | 5.638 / 6.248 | 0.0167 | 0.8913, 0.0569, 0.0218 | 1.000 / 0.989 / 0.957 |


Scores S(d,0) are E63's readings (`step1_read.py`, unchanged) in window A2; the level is 0.1103.

| | Outcome |
|---|---|
| H1 | **Refuted.** First form: every arm ends at 90 to 94 connections a cell, still almost the whole neighbourhood. Coincidence form: only W ends below 47 (23.1), but it still moves by 8.5% over the last 4,096 bytes, against the 5% the hypothesis required; W32 and W128 end above 47 |
| P1 | Held: W ends at 93.0 in the first form |
| P2 | Failed: T128 ends at 91.4 |
| P3 | Failed |
| P4 | Failed (W below 47 but not settled; W32 and W128 above 47) |
| P5 | Failed: withdrawal by use makes the first form 2.4 to 2.9 times dearer, not cheaper |

**What the runs show (development level, one seed).**

1. **Withdrawal by use withdraws; making puts the connections back.** First form, W: 1,198,914 connections withdrawn by use and 1,591,651 made over 16,384 bytes, about 73 and 97 a byte, with the number of connections unchanged at about 93. The making rule makes a connection every 128 steps while the cell's threshold is 0 and its average activity is below the target; neither target tried (1/32, 1/128) ends that in most cells. The number of connections is set by the making rule, not by the withdrawal rule. The cost of the churn is the rise in cost.
2. **Lowering the target changes what the near cells carry more than how many connections there are.** S(1,0) in A2 rises from 0.687 (R) to 0.950 (T128) in the first form and from 0.304 to 0.889 in the coincidence form; in the coincidence form at 1/128 the cells two connections from the lines come close to the level (S(2,0) 0.107 and 0.057 with W, against the level 0.1103). Beyond one connection no arm of the first form has informative cells.
3. **In the coincidence form withdrawal by use does bring the number down** (33.1 to 23.1 at the target 1/8) **and halves the cells sending a step** (170.8 to 71.2), but the number moves by several per cent from stretch to stretch.

**What follows for the question of the number of connections.** A cell settles its number only if making stops when the cell no longer needs another input. With the present making rule it never does, whatever the withdrawal. The next plan has to change the making rule, one change at a time: for example, a cell makes a connection only while it is under-driven compared with a level it can reach (a target below the measured activity, or one that follows the cell's own record), or only after a connection it had was withdrawn.
