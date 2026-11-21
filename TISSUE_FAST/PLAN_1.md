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

## Outcome

(to be entered after the runs)
