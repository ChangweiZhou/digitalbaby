# nano/multi, first plan: a connection of three ends on two-digit sums (roadmap C3, step 1)

20 November 2026. By the agent session that owns the line C3 and the folder `nano/multi/` (branch `claude/nano-multi-8mmhfm`, started from 260c5d7 of `claude/friendly-darwin-8mmhfm`). Written and committed before any run of this step at length; only the 300-line checks of `check_multi.py` were run before it. Development level; repeated by no one.

## The source

Mr. Xu, 20 November (from `ROADMAP_SECOND_REVISION_20261120.md`, section 1): "The connection strength (ideally can involve many cells) should be jointly decided by the cells together. For example if one cell wants to drop out, then the connection strength to it should be zero no matter how willing other cells are. Obviously we can simplify the edge to only involve two cells, and perhaps refine even further to one sided edges, with inhibitory / excite signals respectively. But whether this is justified has to be answered by experiment."

Authorisation, as relayed in this session's brief from the coordinating session: Mr. Xu asked it to "revise the road map and keep doing some experiments. Spin off some agents to speed up the experiment process", and later said "Keep experimenting, we are not going to break anything."

## The question

Everything built so far uses connections with two ends (a line and a cell). Does a connection with three ends (two lines and a cell) do better than connections with two ends, at an equal number of strengths a cell? This is one small part of whether the edge is a justified simplification; the reading that it is a part of that question is mine.

## The exact learner

The system of `loop_small_case/cells2.py` with `fan=` and `rewire=` (report sections 2 and 10), one step of relay: 10 delays of 14 byte values on 140 lines; 256 logistic cells in between; one answering cell with 14 branches fitted by its mismatch (least squares); the mismatch relayed to the cells over return strengths that the cells hold and that change by the same product as the forward strength; eta1 1/4, eta2 1/16, fading 1e-5. One stream of `aa+bb=ccc` lines, one byte at a time, 400,000 lines; 2,000 of the 10,000 pairs never shown; score: whole answer right on 500 never-shown pairs, mean of the last three readings at 50,000-line intervals.

**What is added** (`multi.py`, numpy, the reference; `kernm.c` through `fastm.py`, compiled; `check_multi.py` compares them). Beside connections from single lines a cell may hold **pair-connections**. A pair-connection names two lines at two different delays (a byte value at each) and is active at a byte when both lines are active. Its strength is added to the cell's drive when it is active, and changes, when it is active, by the cell's relayed signal times the cell's own sensitivity times eta1: the rule of a single line's strength. Which pairs a cell holds is drawn at birth, uniformly over the 8,820 pairs of lines at two different delays, with strengths uniform in (-0.5, 0.5) as for lines. **Rewiring** is the edge arm's: every 4,096 bytes each cell withdraws the weakest (smallest size of strength) of all its connections that are at least four periods old, single lines and pairs compared together, and tries another of the same kind at strength 0, so that the numbers of each kind stay as at birth and the turnover is one connection a cell a period in every arm.

With no pairs the program makes the draws and the arithmetic of `cells2.py`; `check_multi.py` compares it with `cells2.py` itself (imported, not changed).

**Arms, at equal numbers of strengths a cell** (256 cells; a cell also has its threshold; the answering cell has 3,598 numbers and the return strengths are 3,584 in every arm):

| Arm | single lines a cell | pairs a cell | numbers that learn in all |
|---|---|---|---|
| E64, the edge arm | 64 | 0 | 23,822 |
| M64 | 32 | 32 | 23,822 |
| P64 | 0 | 64 | 23,822 |
| E32, the edge arm at 32 | 32 | 0 | 15,630 |
| M32 | 16 | 16 | 15,630 |
| P32 | 0 | 32 | 15,630 |

Each arm differs from its edge arm in one thing: the kind of some or all of the connections a cell holds. A pair-connection needs two addresses where a line needs one; addresses are not counted as strengths, and I state this as a limit of "equal".

**A limit of this testbed.** The sending ends here are input lines, not cells. A line cannot decide anything, so only the receiving cell decides about a connection (by its strength, and by withdrawing it). This step therefore tests a connection that involves three ends, and does not test a strength "jointly decided by the cells together"; that is step 3's subject.

## Settings chosen on the development seed (seed 1)

Two, both about which pairs are offered to a cell, chosen on arm M64 by its score and then used in every pair arm; both outcomes are reported:
1. the two delays of a pair: any two (the default), or neighbouring delays only (`adj=1`);
2. a pair tried at a turnover: drawn uniformly (the default), or drawn among the pairs of lines active at that byte (`pdraw=active`).
A setting other than the default is taken only if it is higher by 3 points or more on the development seed. Nothing else is tuned: the period of rewiring, the steps and the fading are the edge arm's.

## Hypothesis and what would refute it

**H:** at an equal number of strengths, cells that hold pair-connections beside single lines (the M arms) answer never-shown sums at least as often as cells that hold single lines only. My reason: the units digit of a sum is a function of one pair of digits, and a pair-connection states that pair directly.
**Refuted if** the M arm's median over the five reported seeds is 3 points or more below its edge arm's at both sizes. **Supported as a gain** only if the M arm's median is 3 points or more above at one size at least and it is higher on four seeds of five there. Between the two: no difference claimed.

## Predictions (five reported seeds, medians)

- P1 (0.9): E64 reproduces the report's figures seed by seed (68.1, 83.0, 84.3, 80.8, 73.0; median 80.8) and E32 its 44.4, 40.9, 42.7, 45.1, 31.6 (median 42.7), to the digit printed.
- P2 (0.45): M64 is at or above 80.8.
- P3 (0.7): P64 is 10 points or more below E64.
- P4 (0.6): M32 is 5 points or more above E32 (42.7).
- P5 (0.6): P32 is below E32.
- P6 (0.6): in the M arms the pairs held at the end are not spread evenly over the pairs of delays: the most held pair of delays is held at least three times as often as the mean.

## Cost

A run of 400,000 lines: about one to two minutes of one core (from the 300-line checks). Development seed: at most 16 runs. Reported seeds 73129, 93683, 116089, 406364, 545353: 30 runs. Under one hour of CPU on four cores.

## What is not run

Subtraction and keeping; cell counts other than 256; connections of more than three ends; pairs added to cells that read every line (not an equal number of strengths); the numpy program at full length (it is run for 300 lines against the kernel, on every arm); sending ends that are cells (step 3, a separate plan); the text of `nano/data` (step 2, a separate plan). No checker gate, seal or review is claimed.
