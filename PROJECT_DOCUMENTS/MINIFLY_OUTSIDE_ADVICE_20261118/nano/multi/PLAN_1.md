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

## Entered on 20 November after the development runs (seed 1), before any run on the reported seeds

Never-shown sums, mean of the last three readings (`res/m1_d_*.json`, `log/d_*.txt`); every line read gives 81.5 in this program on this seed, as in `cells2.py`.

| Arm | pairs drawn uniformly (the default) | pairs tried among those active at the byte | neighbouring delays | both |
|---|---|---|---|---|
| E64 (64 lines) | 68.5 | | | |
| M64 (32 + 32) | 42.0 | 56.8 | 47.8 | 54.5 |
| P64 (64 pairs) | 49.2 | 48.8 | | |
| E32 (32 lines) | 44.9 | | | |
| M32 (16 + 16) | 16.9 | 24.1 | | |
| P32 (32 pairs) | 31.9 | 30.2 | | |

- **Setting chosen** by the rule above, on M64: a pair tried at a turnover is drawn among the pairs of lines active at that byte (56.8 against 42.0); any two delays (with the first setting taken, neighbouring delays give 54.5, which is not higher). Pairs at birth are drawn uniformly in every case. The pair arms of the reported seeds use this setting. The same arms with pairs tried uniformly are run on the reported seeds as well, as an addition, so that the cost of my choice can be seen.
- **On this seed the pair arms are below the edge arm at both sizes**, with either draw: at 64 strengths 56.8 and 48.8 against 68.5; at 32 strengths 24.1 and 30.2 against 44.9.
- The pair of delays held most often at the end is (3, 6) in every pair arm; when the units digit of the answer is named, these are the lines of the two units digits. It is held 1.4 to 2.0 times as often as the mean pair of delays.
- **A figure of the report that I do not reproduce on the development seed.** `PLAN_4.md` gives 76.5 (64 lines with rewiring) and 43.4 (32 lines) for seed 1. `cells2.py` of commit 260c5d7, copied to a scratch folder and run with its own compiled kernel, gives 68.5 for seed 1 with 64 lines, and 68.1 for seed 73129, which is the report's figure for that seed. `multi.py` gives the same readings as `cells2.py` at every 50,000 lines on both seeds, and its numpy program the same as `cells2.py`'s numpy program over 60,000 lines. So the edge arm here is the program of the report; why the plan's development figure differs I do not know (my guess: it was made by the numpy program, whose rounding differs from the kernel's, and rewiring turns a difference of rounding into another run; not tested).
- Speed: a run takes 18 to 39 CPU seconds with `kernm.c` (the edge arm 18 s, where `kern.c` takes 84 s).

## The reported seeds (entered 20 November; table in `report1.txt`, made by `report1.py` from `res/m1_r_*.json`)

Never-shown sums, whole answer right, %, mean of the last three readings; 256 cells; five reported seeds and their median. By place: hundreds, tens, units of the answer at the end of the run (median).

| Arm | 73129 | 93683 | 116089 | 406364 | 545353 | median | by place |
|---|---|---|---|---|---|---|---|
| E64: 64 single lines | 68.1 | 83.0 | 84.3 | 80.8 | 73.0 | **80.8** | 95, 87, 100 |
| M64: 32 lines + 32 pairs | 36.1 | 54.8 | 48.8 | 52.2 | 40.7 | **48.8** | 96, 59, 98 |
| P64: 64 pairs | 48.3 | 49.8 | 48.7 | 47.7 | 46.7 | **48.3** | 96, 55, 96 |
| E32: 32 single lines | 44.4 | 40.9 | 42.7 | 45.1 | 31.6 | **42.7** | 95, 54, 92 |
| M32: 16 lines + 16 pairs | 15.7 | 22.3 | 15.3 | 19.5 | 29.3 | **19.5** | 95, 36, 75 |
| P32: 32 pairs | 38.9 | 24.8 | 30.7 | 34.7 | 27.5 | **30.7** | 94, 50, 78 |

Addition (no prediction registered): the same pair arms with pairs tried uniformly at a turnover: M64 34.3, P64 46.1, M32 12.7, P32 28.5 (medians). The draw I chose on the development seed is higher in each, by 2 to 15 points; it does not change the order of the arms.

**Outcome.** At an equal number of strengths a cell, the cells that hold pair-connections answer fewer never-shown sums than the cells that hold single lines only: by 32 points at 64 strengths (both pair arms) and by 12 to 23 points at 32 strengths, on every one of the five seeds in each of the four comparisons. **H is refuted** by its own criterion (the M arm's median is 3 points or more below at both sizes).

**Predictions scored.**
- P1 held: E64 and E32 give the report's figures on every seed to the digit printed.
- P2 failed: M64 48.8, where I had said at or above 80.8.
- P3 held: P64 is 32.5 points below E64.
- P4 failed: M32 is 23 points below E32, where I had said 5 or more above.
- P5 held: P32 30.7 is below 42.7.
- P6 failed: the pair of delays held most, (3, 6), is held 1.6 to 2.0 times as often as the mean, where I had said three times.
Three of six failed, and the two that carried the hypothesis (P2, P4) failed by wide margins.

**What I read from it (my readings; none of them tested here).**
1. The loss is in the tens digit: 59 and 55 against 87 at 64 strengths. The units digit, which one pair of digits decides and for which I expected pairs to help, is as good with single lines (100) as with pairs (96 to 98). The cells with lines already combine the two units digits by adding their strengths; a pair-connection gives them nothing they lack there.
2. A pair-connection is active about fourteen times less often than a line (one byte value in fourteen at each of two delays), so at the same step it is taught that much less often, and at any byte a cell with 64 pairs has well under one active connection on average where a cell with 64 lines has four to five. The pair arms are equal to the edge arm in the number of strengths; they are not equal in how much of the input a cell reads at a byte.
3. The mixed arm is not better than the pairs alone and at 32 strengths it is worse. In the mixed arm about two thirds of the withdrawals fall on pairs (145,985 pairs against 76,935 lines made at 64 strengths), though the two kinds are equal in number: with one rule of withdrawal for both kinds, the kind that is taught less often is withdrawn more often.

**Limits.**
- The sending ends are input lines; only the receiving cell decides. Nothing here tests a strength decided jointly by several cells.
- One step size, one period of rewiring and one rule of withdrawal for both kinds, as the plan fixed them. A pair arm with a larger step for pairs, a faster turnover of pairs, or a withdrawal rule that allows for how seldom a pair is active was not run. The result says that pairs lose under the edge arm's own settings, not that no setting exists under which they gain.
- 256 cells, 400,000 lines, one task. A pair-connection needs two addresses; addresses were not counted.
- Development level; repeated by no one; no checker gate, seal or review is claimed.

**Not run:** the numpy program at full length (300 lines on each arm against the kernel, and 60,000 lines of the edge arm against `cells2.py`); cell counts other than 256; neighbouring delays on the reported seeds; subtraction; pairs beside every line.
