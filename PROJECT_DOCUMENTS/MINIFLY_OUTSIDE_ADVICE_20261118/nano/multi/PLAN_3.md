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
