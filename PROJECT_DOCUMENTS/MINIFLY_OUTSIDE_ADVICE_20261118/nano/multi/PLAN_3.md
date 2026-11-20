# nano/multi, third plan: connections whose two sending ends are cells, and a rule by which each of the three cells can set one to zero (roadmap C3, step 3)

20 November 2026. Branch `claude/nano-multi-8mmhfm`. Written and committed before any run of this step at length; only the 300-line checks of `check_prod.py` were run before it (`check_prod_output.txt`). Development level; no checker gate, seal or review is claimed. Source and authorisation: as in `PLAN_1.md`.

## The questions

Steps 1 and 2 had input lines as sending ends, which cannot decide anything. Here the sending ends are cells.
1. At an equal number of strengths a cell, does a cell that holds connections from two sending cells jointly (active in proportion to the product of their outputs) do better than a cell that holds edges from single cells?
2. Mr. Xu: "if one cell wants to drop out, then the connection strength to it should be zero no matter how willing other cells are." Stated as a rule that uses only what is at the connection (below): does it change the score when each of the three cells of a connection can set it to zero, beside the receiving cell alone?

## The exact learner

The system with two steps of relay of `loop_small_case/cells2.py` (`H=128 H2=128`; report section 10: 84.4% at the median of five seeds): 140 lines; 128 logistic cells A, each reading every line; 128 logistic cells B; one answering cell with 14 branches (least squares). The answering cell's mismatch reaches cells B over return strengths; a cell B passes on what it received, times its own sensitivity, to cells A over a second set of return strengths. Two-digit sums, 400,000 lines, never-shown sums, mean of the last three readings; eta1 1/4, eta2 1/16, fading 1e-5, as there. Programs: `prod.py` (numpy, the reference), `kernq.c` through `fastq.py`, `check_prod.py`.

**What is changed.** A cell B holds K connections in place of one from every cell A:
- an **edge** names one cell A and carries that cell's output;
- a **pair-connection** names two cells A and carries the product of their two outputs.
A cell B's drive is the sum of its strengths times what its connections carry, less its threshold. A strength changes by the cell's relayed signal times its own sensitivity times what the connection carries, with the step of `cells2.py` (eta1 times the number of delays, divided by 1 + the summed squares of what the cell's own connections carry; with an edge from every cell A this is `cells2.py` exactly, and `check_prod.py` compares the two). Each connection has a return strength that receives the same change and fades alike. **What a sending cell receives over a pair-connection** is the cell B's signal times the return strength times the other sending cell's output: all three are present at the connection. Strengths at birth are uniform in (-1, 1) divided by the root of K/3, return strengths uniform in (-1, 1), for both kinds.

**Rewiring by the receiving cell** (as in step 1): every 4,096 bytes each cell B withdraws the weakest (size of strength) of all its connections at least four periods old and tries another of the same kind (an edge from a cell A it does not read; a pair of cells A drawn uniformly that it does not hold), with strength and return strength 0.

**The rule by which every cell of a connection can set it to zero** (`joint=1`; mine, one of many possible). Each sending cell keeps, for each pair-connection it sends into, a running mean (rate 1/1,024 a byte) of the size of what it receives back over that connection. At each turnover every cell A withdraws from the one of its pair-connections, at least four periods old, over which it has received least. A connection from which any of its three cells has withdrawn is at zero, whatever its strength was and whatever the other two would do; its cell B then tries another pair. The receiving cell judges by the size of its strength; a sending cell judges by what the connection has brought it; no cell uses anything that is not at the connection or in itself.

**Arms** at K strengths a cell B (a cell B also has its threshold and K return strengths lie at its connections; cells A: 18,048 numbers; the answering cell 1,806 and its return strengths 1,792 in every arm):

| Arm | edges | pairs | who can withdraw | period |
|---|---|---|---|---|
| E, the edge arm | K | 0 | cell B | 4,096 |
| M | K/2 | K/2 | cell B | 4,096 |
| P | 0 | K | cell B | 4,096 |
| PJ | 0 | K | every cell of the connection | 4,096 |
| P2, the control of PJ | 0 | K | cell B | 2,048 |

P2 is there because in PJ about twice as many connections are withdrawn in a period as in P (one by each cell B and one by each cell A); P2 has cell B withdraw twice as often, so that PJ is compared with an arm of about the same turnover in which only the receiving cell decides. Beside the arms, the reference of the report: an edge from every cell A (128 strengths a cell B, no rewiring).

## Settings chosen on the development seed (seed 1)

One: K. The edge arm E is run at K = 64, 32, 16 and 8 on seed 1. The two sizes used are the two largest K at which E is 5 points or more below the reference on that seed (so that there is room for a difference to show); if fewer than two such K exist, the two smallest. Everything else (steps, fading, the period, the rate of the running mean) is fixed as written above and not tuned.

## Hypotheses and what would refute them

**H3a:** with sending ends that are cells, pair-connections do not lose as they did from lines: P is within 5 points of E at the median of five seeds, at one size at least. **Refuted if** P is 5 points or more below E at both sizes.
**H3b:** letting every cell of the connection withdraw does not lower the score: PJ is not 3 points or more below P2 at either size. **Refuted if** it is, at either size, with the lower score on four seeds of five.

## Predictions (five reported seeds, medians)

- R1 (0.7): the reference reproduces the report's figures seed by seed (82.5, 85.9, 85.0, 82.6, 84.4).
- R2 (0.55): P is within 5 points of E at the larger K.
- R3 (0.35): M is 3 points or more above E at one size at least.
- R4 (0.6): PJ is within 3 points of P2 at both sizes.
- R5 (0.3): PJ is 3 points or more above P at one size at least.
- R6 (0.6): P2 is within 3 points of P at both sizes (the period matters little).

## Cost

A run: under one minute of one core (estimated from the checks; the dense reference took 239 s with `kern.c`). Development: at most 8 runs. Reported seeds: 5 arms at 2 sizes and the reference, 55 runs. Under one hour of CPU.

## What is not run

Edges from which the sending cell can withdraw (that is line C2's subject, `nano/joint`); the mixed arm with sending cells withdrawing; a withdrawal rule other than the one stated; connections of more than two sending cells; the text of `nano/data`; other cell counts; the numpy program at full length.

## Entered on 20 November after the development runs (seed 1), before any run on the reported seeds

Never-shown sums, mean of the last three readings (`res/p3_d_*.json`): the reference (an edge from every cell A) 68.9, which is the figure `PLAN_4.md` of `loop_small_case` gives for 128 + 128 on this seed. The edge arm E with rewiring: K = 64: 65.0; K = 32: 75.5; K = 16: 71.6; K = 8: 71.1.

At no K is the edge arm 5 points or more below the reference on this seed (it is above it at three of the four), so by the plan's rule the two sizes are **the two smallest, K = 16 and K = 8**. A cell B that reads 8 of the 128 cells A and rewires answers as often as one that reads all of them, on this seed; I had expected reading few to cost, as it does from lines (step 1), and it does not here.

A run costs 46 to 150 CPU seconds (the machine was shared with step 2's runs), more than the plan's estimate of under a minute. No arm other than E was run on the development seed; the reported seeds follow (`jobs_r3.txt`).

## An addition, entered on 20 November after the reported seeds of the five arms were read and before any run of it

Read so far (`report3.txt`): PJ, the arm in which every cell of a pair-connection can withdraw, is at 16.5 at K = 8 where P2 is at 76.7 and P at 72.3; at K = 16 it is at 71.1 beside 69.6 and 79.5. Two things looked at on the development seed after that (`res/p3_g_PJ*_1.json`; the diagnostic counters added to `prod.py` change no arithmetic, and `check_prod.py` was run again):
- A guess of mine, that the sending cells withdraw the strong connections, is wrong: the connections withdrawn by sending cells have a mean size of strength of 0.19 (K = 8) and 0.08 (K = 16), against 0.85 and 0.56 for all pairs old enough, and 0.09 and 0.05 for those withdrawn by cells B.
- At K = 8 only 27% of the pair-connections are at least four periods old at a turnover (62% at K = 16). A cell A sends into 16 pair-connections on average at K = 8, of which about four are old enough, and my rule makes it withdraw from one of them every period.

My reading (to be tested by this addition): the rule as I stated it fails at K = 8 by its rate, not by the fact that sending cells may withdraw: together the cells withdraw from connections faster than a connection can be taught.

**The addition, one thing changed in PJ:** the sending cells withdraw at every 8th turnover only (`jevery=8`: each cell A at most once in 32,768 bytes); cells B as before. Arm **PJs**, at K = 8 and K = 16, five reported seeds (`jobs_r3x.txt`). Its turnover is that of P plus about an eighth, so it is compared with P.
- Prediction R7 (0.6): PJs is within 5 points of P at both sizes (medians).
- If R7 fails at K = 8 with PJs below 40, my reading is wrong and the collapse is not a matter of rate.
The number 8 is not tuned; no other value is run.

## The reported seeds, with the addition (entered 20 November; table in `report3.txt`, made by `report3.py` from `res/p3_r_*.json`)

Never-shown sums, whole answer right, %, mean of the last three readings; 128 cells A, 128 cells B; five reported seeds and their median.

| Arm | 73129 | 93683 | 116089 | 406364 | 545353 | median |
|---|---|---|---|---|---|---|
| reference: an edge from every cell A (128 strengths a cell B) | 82.5 | 85.9 | 85.0 | 82.6 | 84.4 | **84.4** |
| E16: 16 edges | 69.3 | 87.6 | 84.1 | 78.0 | 83.6 | **83.6** |
| M16: 8 edges + 8 pairs | 66.5 | 85.9 | 78.4 | 69.1 | 79.5 | **78.4** |
| P16: 16 pairs | 68.4 | 79.5 | 85.7 | 79.1 | 82.6 | **79.5** |
| PJ16: 16 pairs, every cell can withdraw | 63.0 | 71.1 | 67.7 | 77.5 | 83.7 | **71.1** |
| P2 at 16: cell B withdraws twice as often | 69.6 | 84.1 | 68.1 | 76.3 | 60.9 | **69.6** |
| PJs16 (addition): sending cells withdraw at every 8th turnover | 67.3 | 78.6 | 71.7 | 84.2 | 81.5 | **78.6** |
| E8: 8 edges | 76.7 | 73.5 | 64.8 | 64.1 | 58.0 | **64.8** |
| M8: 4 edges + 4 pairs | 70.1 | 58.5 | 76.8 | 67.7 | 69.3 | **69.3** |
| P8: 8 pairs | 72.3 | 69.1 | 70.7 | 81.9 | 78.6 | **72.3** |
| PJ8: 8 pairs, every cell can withdraw | 12.1 | 17.4 | 16.5 | 28.6 | 16.5 | **16.5** |
| P2 at 8: cell B withdraws twice as often | 76.7 | 78.2 | 83.9 | 70.8 | 73.3 | **76.7** |
| PJs8 (addition): sending cells withdraw at every 8th turnover | 62.7 | 44.8 | 66.3 | 57.3 | 52.1 | **57.3** |

**Outcome, question 1 (pairs of sending cells against edges).** With sending ends that are cells, pair-connections do not lose as they did from lines. At 16 strengths a cell B, pairs alone are 4 points below edges at the median (higher on 2 seeds of 5) and half and half is 5 points below (lower on all 5). At 8 strengths, pairs alone are 7.5 points above edges and half and half 4.5 above, each on 3 seeds of 5. The seeds scatter by 15 to 20 points within an arm. **H3a is not refuted.** I claim no difference between pairs and edges in either direction at these sizes; what I do claim is that the large loss of steps 1 and 2 (12 to 32 points; 0.4 nats) is not there.

**Outcome, question 2 (every cell of the connection can set it to zero).** Under the rule as I stated it, at 8 strengths a cell B the system all but stops answering: 16.5 against 76.7 for the control of equal turnover, lower on all five seeds. At 16 strengths it is level with that control (71.1 against 69.6) and 8 points below P. **H3b is refuted** at K = 8.
The addition (sending cells withdraw an eighth as often): 57.3 at K = 8, which is 15 points below P and below it on all five seeds; 78.6 at K = 16, level with P's 79.5 at the median (below it on four seeds of five, by 1 to 14 points). So most of the collapse at K = 8 was the rate of withdrawal, as I had read it, and a part was not: even at an eighth of the rate, withdrawal by the sending cells under this criterion costs 15 points at 8 strengths, and I see no cost I would claim at 16.

**Predictions scored.**
- R1 held: the reference gives the report's figures on every seed to the digit printed.
- R2 held: P is 4.1 points below E at K = 16.
- R3 held by its wording (M is 4.5 above E at K = 8, at the median) and I do not claim it: 3 seeds of 5, and at K = 16 M is below E on all five.
- R4 failed: PJ is 60 points below P2 at K = 8.
- R5 failed: PJ is below P at both sizes.
- R6 failed: P2 is 9.9 below P at K = 16 and 4.4 above at K = 8.
- R7 (the addition) failed: PJs is 14.9 below P at K = 8. Its second clause (below 40, my reading wrong) is not met: 57.3.
Three held, four failed.

**What I read from it (my readings; not tested beyond what is said).**
1. A product of two cells' outputs is present at every byte, as an edge's signal is; a pair of input lines is present at one byte in some hundreds. That the pair-connections lose from lines and not from cells fits the reading of step 1 that their loss there was a matter of how seldom they are active. It does not prove it: the two testbeds differ in more than that.
2. A cell B that holds 16 connections of either kind and rewires answers as often as one that reads all 128 cells A (83.6 and 79.5 against 84.4), with 4,352 numbers in cells B and their return strengths where the reference has 32,896.
3. The right to withdraw is not free. Each withdrawal ends what a connection had learned; a rule that makes every cell use the right once a period leaves only a quarter of the connections older than four periods at K = 8. A cell that withdraws seldom costs less. Which criterion and which rate a sending cell should use is open; the one tried (the least received back, at a fixed rate) is one of many.
4. The connections withdrawn by sending cells are weak ones (mean size 0.19 against 0.85 for all pairs old enough at K = 8), so the two ends mostly agree about which connections are worth little; the receiving cell's own withdrawals are weaker still (0.09).

**Limits.**
- The rule is mine and one of many; its numbers (one withdrawal a period; the rate 1/1,024 of the running mean; every 8th turnover in the addition) were fixed and not tuned. Another criterion for the sending cell (a threshold in place of a fixed rate; what it sends in place of what it receives) was not run.
- "Set to zero" is here a withdrawal after which the receiving cell tries another pair. A connection that stays, at strength zero, or a strength made as the product of parts held by the three cells, was not run.
- K was chosen on the development seed by a rule that expected reading few cells A to cost; it did not, so the sizes are the two smallest of the four, and the edge arm at K = 8 is 64.8 on the reported seeds where it was 71.1 on the development seed.
- Five seeds with a scatter of 15 to 20 points within an arm: differences under about 10 points are not resolved.
- One task, 128 + 128 cells, 400,000 lines. Cells A read every line. Development level; repeated by no one; no checker gate, seal or review is claimed.

**Not run:** edges from which the sending cell can withdraw (line C2's subject); the mixed arm with sending cells withdrawing; K = 32 and 64 on the reported seeds; the text of `nano/data`; the numpy program at full length (300 lines a case against the kernel, 12 cases, and the reference against `cells2.py`).
