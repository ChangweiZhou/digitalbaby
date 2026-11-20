# Line C, fifth plan (step C4): the sign, with and without the right to withdraw (20 November 2026)

By the same separate agent session; same folder, same source commit (`fe397ca` of `claude/friendly-darwin-8mmhfm`); authorisation as recorded in `PLAN_3.md` (asked by the coordinating session after C2; no word to me from Mr. Xu directly). Written before any run of this step. Development level; repeated by no one; no checker gate, seal or review is claimed.

Mr. Xu's words that this step takes up (roadmap file, section 1): "Obviously we can simplify the edge to only involve two cells, and perhaps refine even further to one sided edges, with inhibitory / excite signals respectively. But whether this is justified has to be answered by experiment."

## 1. The question

One sign for each cell in between costs about 14 points in the earlier runs (70.7 against 84.2, `loop_small_case/REPORT.md`, section 10). Does that cost change when either end of a connection may withdraw from it?

## 2. The arms

Two things, each present or absent; nothing is tuned.

- **The sign**, copied from `cells2.py` (`dale=1`) at commit `fe397ca`: a strength from a line is never below zero (born as the size of the plain birth value; thresholds born at -2.5); each cell in between is excitatory or inhibitory, a quarter inhibitory, drawn from the same stream as in `cells2.py`; after every byte a strength of the answering cell from a cell, or a return strength of that cell, whose sign is not the cell's is set to zero.
- **The right to withdraw**, arm E of `PLAN_2.md` unchanged, at the setting chosen there (T = 720,000 bytes, theta = 0.1).

Arms: N (neither; the plain system), G (the sign alone; must reproduce `cells2.py`: 66.6 on seed 1 and 70.7, 73.9, 64.0, 62.1, 72.3 on the reported seeds, which were made with `cells2.py`'s first compiled kernel), E (withdrawal alone; must reproduce `PLAN_2.md`'s 83.8, 81.2, 87.7, 82.6, 86.5), GE (both).

What I know beforehand about GE: under the sign a connection that the mismatch pushes towards the other sign sits at exactly zero. At a moment of withdrawal such a connection is below theta times any largest strength, so both ends withdraw from it and it is gone for good, although under the sign alone it could have come back if the mismatch later pushed it the cell's way. So the two together are not two independent things: withdrawal makes the sign's zeros permanent. I count the living connections that are at zero just before each moment.

**Numbers that learn:** 43,278, less two for each withdrawn connection.

## 3. Measures

Never-shown sums (mean of the last three readings). The cost of the sign without withdrawal, N less G, and with it, E less GE, seed by seed, and their difference. Connections living at the end, from excitatory and from inhibitory cells; the share of the answering cell's strengths at zero; cells with no connection left.

## 4. Hypothesis and what would refute it

**H: the cost of the sign is the same with and without the right to withdraw: (E less GE) less (N less G), seed by seed, is within 4 points at the median.** Refuted if that difference has a median larger than 4 points in size and the same sign on at least four of five seeds. If the cost is smaller with withdrawal, withdrawing is a help to cells of one sign; if larger, making the sign's zeros permanent is a harm.

## 5. Predictions (mine), with chances

- T1 (0.8): G gives 66.6 on seed 1 and the five figures above to the digit (a different kernel from the one that made them; the sign's cut-offs are discontinuous, so a rounding could move a reading).
- T2 (0.5): H stands.
- T3 (0.3): the cost of the sign is larger with withdrawal by more than 4 points (four seeds of five).
- T4 (0.85): GE leaves fewer connections than E on five seeds of five.
- T5 (0.5): GE leaves fewer than a third of the connections (1,195) at the median.
- T6 (0.6): GE is within 4 points of G at the median of the paired differences.

## 6. Order of work, cost, not run

`joint4.py` (numpy; the sign added to the class of `joint2.py`), `kerns.c` (a copy of `kernw.c` with the sign's cut-offs) with `fasts.py`, `check_joint4.py` (compiled against numpy for the four arms; the sign against `cells2.py` with `dale=1`, numpy against numpy; without the sign against `joint2.py`). Seed 1 for the four arms as a development pass (nothing is chosen on it), then the five reported seeds: 24 compiled runs, about 3 minutes; one full-length numpy run of GE on seed 1.

Not run: other shares of inhibitory cells; withdrawal by one end only under the sign; other T and theta; the sign in the product or smaller-part forms; signs for the lines.

---

## 7. Outcome (entered 20 November, after all runs of this plan)

Tables: `report4.txt` (made by `report4.py` from `res/g_*.json`); the check: `check_joint4_output.txt` (passed: compiled against numpy 1e-14 to 4e-13 in the four arms with the same sets of living connections and every sign kept; the sign against `cells2.py` with `dale=1` 2e-15; without the sign against `joint2.py` exactly). The comparison of the full-length numpy run with the compiled one is entered in section 8.

Never-shown sums, whole answer right, %, mean of the last three readings; connections living at the end, of 3,584.

| | 73129 | 93683 | 116089 | 406364 | 545353 | median | living (median) |
|---|---|---|---|---|---|---|---|
| N, neither (the plain system) | 84.2 | 82.7 | 86.4 | 81.8 | 86.7 | **84.2** | 3584 |
| G, one sign for each cell | 70.7 | 73.9 | 64.0 | 62.1 | 72.3 | **70.7** | 3584 |
| E, either end may withdraw | 83.8 | 81.2 | 87.7 | 82.6 | 86.5 | **83.8** | 1823 |
| GE, both | 70.3 | 74.3 | 68.3 | 63.5 | 66.1 | **68.3** | 771 (743 to 793) |

| paired, same seed | 73129 | 93683 | 116089 | 406364 | 545353 | median |
|---|---|---|---|---|---|---|
| cost of the sign without withdrawal, N less G | 13.5 | 8.8 | 22.4 | 19.7 | 14.3 | **14.3** |
| cost of the sign with withdrawal, E less GE | 13.5 | 6.9 | 19.3 | 19.1 | 20.4 | **19.1** |
| the second less the first | -0.1 | -1.9 | -3.1 | -0.6 | +6.1 | **-0.6** |
| GE less G | -0.4 | +0.4 | +4.3 | +1.4 | -6.2 | **+0.4** |

**The hypothesis.** H stands: the cost of the sign with withdrawal less the cost without is -0.6 at the median (four seeds between -3.1 and -0.1, one at +6.1). The right to withdraw neither eases nor adds to the cost of one sign for each cell; the sign costs 14 to 19 points either way. (The two medians of the cost, 14.3 and 19.1, differ by 4.8, but the paired difference, which is what was registered, is -0.6: the medians fall on different seeds.)

**What else the runs show.**

1. *Under the sign, either end withdrawing leaves about a fifth of the connections at the score of the sign alone*: 771 of 3,584 (21.5%) at 68.3 against 70.7 (paired GE less G: +0.4 at the median, no common sign). Without the sign the same rule leaves 1,823 (51%). About 225 of the 771 come from the 68 inhibitory cells, which is their share of the cells. One to six cells lose every connection.
2. *Why so few remain is not what I wrote in section 2.* I expected many connections to sit at exactly zero under the sign and to be made permanent zeros by withdrawal. They do not sit at zero: with the sign alone 7.7% of the answering cell's strengths are at exactly zero at the end, of which 7.1 points are the branch that names nothing (`PLAN_2.md`, section 10), and just before the first moment of GE 261 to 327 living connections are at zero, 256 of them that branch. What the sign does is make the strengths unequal: 44% of them are below one hundredth of the largest at the end (4% in the plain system apart from the dead branch), and the largest is 0.23 against 0.09. So under the sign most connections are small beside the largest of their branch or cell, the rule withdraws them (about 2,550 at the first moment against 1,350 without the sign), and the score does not notice.
3. The sign alone reproduces the earlier figures to the digit with a different kernel (66.6 on seed 1; 70.7, 73.9, 64.0, 62.1, 72.3).

**Predictions.** T1 held. T2 held (-0.6). T3 failed. T4 held (five of five; 743 to 793 against 1,763 to 1,889). T5 held (771). T6 held (+0.4).

**Not run:** as section 6.

## 8. The numpy run (entered 20 November)

The numpy program at full length (GE, seed 1) gives the compiled program's readings to the digit: 61.0 (4, 30, 48, 58, 48, 60, 61, 60), 761 connections living.
