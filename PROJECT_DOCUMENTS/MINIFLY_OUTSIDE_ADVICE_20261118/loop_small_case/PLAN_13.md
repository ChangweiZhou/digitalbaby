# Thirteenth plan: how much of the first task is still in the cells in between (roadmap second revision, track K; a diagnostic) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026: "Keep experimenting, we are not going to break anything." **Source.** This branch at d8f62d3d; `cells2.py` with one option added (`freezeat=n`: from phase n on the cells in between do not change), compiled kernel unchanged. Written before the runs. A small case; no checker gate.

**Where it comes from.** The crosswalk of the old lists (`OLD_LISTS_AGAINST_THE_LOOP_20261120.md`, section 5, item 3) points out that the older summary's own condition for looking at hidden memory is met by the report's section 11: addition is at 1.8% after subtraction, comes back to 70% after 4,000 to 6,000 lines where it first took 60,000 to 70,000, and is at 44.6% when the answering cell's old strengths alone are put back.

**Question.** After subtraction, with the cells in between held as subtraction left them and the answering cell alone learning, how far and how fast does addition come back?

**Runs.** Addition 400,000 lines, subtraction 200,000, addition again 40,000 (`plan=add:400000,sub:200000,add:40000 fine=1`), the five reported seeds: (i) everything learning in the third phase (the report's arm, rerun); (ii) `freezeat=2`. Scored: never-shown addition 2,000 lines into the third phase, the lines needed to reach 70%, the level after 40,000 lines; never-shown subtraction at the end.

**Hypothesis.** Most of what addition needs is still in the cells in between: with them held still, addition returns to at least 60% within 40,000 lines at the median. *Refuted if* it stays below 50%; then what was lost is lost in the cells in between, and the keeping line is aimed at the right place.

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| V1 | arm (ii) reaches at least 60% of never-shown addition by 40,000 lines at the median | 0.50 |
| V2 | arm (ii) keeps subtraction above 50% at the end (arm (i) leaves 2.8%) | 0.60 |
| V3 | arm (ii) is at least 10 points below arm (i) at 40,000 lines | 0.60 |

**Cost.** Ten runs of about a minute. Not run: the same after longer subtraction; the answering cell held and the cells learning.

## Outcome (entered 20 November; `res/c2_r13_*.json`)

Never-shown addition in the third phase, seeds 73129, 93683, 116089, 406364, 545353 and the median:

| | 2,000 lines in | 40,000 lines in | lines to 70% | subtraction at the end |
|---|---|---|---|---|
| (i) everything learning | 55.0, 60.0, 44.2, 58.4, 42.0; 55.0 | 81.6, 78.8, 80.8, 79.6, 79.8; **79.8** | 4,000 to 6,000 | 2.8, 3.0, 1.6, 2.6, 3.0; 2.8 |
| (ii) cells in between held as subtraction left them | 38.8, 50.6, 31.4, 50.0, 28.2; 38.8 | 67.2, 69.8, 67.4, 70.6, 63.6; **67.4** | 8,000 on one seed; not reached on four | 2.2, 2.6, 1.8, 2.2, 3.2; 2.2 |

**The hypothesis holds**: with the cells in between untouched since subtraction, the answering cell alone brings addition back to 67.4%, where it had 84.2% before subtraction and 1.8% after it. V1 held. V3 held (12.4 points below arm (i)). **V2 failed**: with the cells held still, teaching the answering cell addition again takes subtraction down to 2.2% all the same.

Readings (mine). After the second task the cells in between still carry most of what the first task needs; what is overwritten is mainly the answering cell's strengths. And the second half says the same from the other side: the cells did not move in the third phase, so everything subtraction needs is still in them, and yet subtraction is gone, because the one answering cell cannot hold two ways of reading the same cells. So the place where keeping fails in this system is the answering cell's reading of cells that both tasks share, more than the cells. That bears on the keeping line: a candidate that makes the cells of the two tasks different cells (so that the answering cell's strengths for one task are not touched by the other) aims at the right place; one that only protects the cells in between does not.
