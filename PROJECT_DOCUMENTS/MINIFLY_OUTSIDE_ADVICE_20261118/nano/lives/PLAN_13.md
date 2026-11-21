# Three returns: is what a return costs paid once, or at every return? (21 November 2026)

**Authorisation and steer.** As `PLAN_11.md`, where Mr. Xu's words are quoted in full; the one that governs this step: "这些你自己决定，不需要问我。一切从第一原理和事实出发。" (my translation: "Decide these yourself, you need not ask me. Everything from first principles and from facts."). The choice and design of this step are mine. **Source.** Branch `claude/friendly-darwin-8mmhfm` at the commit that holds this file. Development level; no checker gate, seal or review is claimed.

**What I had seen when I wrote this (said first, because it bears on the predictions).** (1) `PLAN_12.md`'s development seed and its first reported seeds (three of five for the form with the four changes). (2) One trial of this plan's program on a life that is **not** the plan's: seed 1 with a first phase of 5,000 lines in place of 400,000, run to see that the program works. It showed, for the form with the four changes, whole answers at 0.0, 0.0 and 1.0% straight after the three pieces of English, the reading among the digits kept at 0.62, 0.94 and 0.94 of its level, and whole answers after 100 lines of the three returns at 51%, 87% and 97% of the level. Before that trial I expected the cost of a return to be paid once; the chances below are set with it in view.

## 1. The facts this follows

`PLAN_12.md`, reported seeds so far (never-shown sums, whole answer; median): the network as it is goes from 96.5% to 0.0% through the English, names 0.0% after 100 lines of the return and 63.5% after 500, and its held-out English falls from 40.9% to 19.0% while the sums return. With the four changes (a strength fades only as its sending cell is active; thresholds held; cells born quiet; a silent branch not taught) it learns the sums better (98.8%), and after the English its reading is whole when only the digit branches may answer (93.2%), but its whole answers are at 0.0%: the right digit's branch has sunk from 0.95 to about 0.06, and a branch of another byte stands at 0.5 to 1.0 in the states of the sums. It is back at 70.0% after 50 lines and 84.5% after 100, and its English falls from 46.6% to 42.9%. Shown sums go and come back with the never-shown ones in both networks.

Both remaining faults sit in the answering cell and pass through cells that are active in both texts: through them the English taught the digit branches down, and through them the branches of the English speak in the states of the sums.

**Two readings, and why they must be told apart.** (i) *Paid once.* The first return teaches each text's branches to be silent in the other text's states through the cells that only the other text uses; those strengths cannot be reached from the own text, so at later returns there is nothing left to pay. (ii) *Paid every time.* The cells that both texts use carry strengths that each text pulls its own way, so each piece of English undoes what the return did. Under (i) no new mechanism is needed, only a life with more than one return, which is the life a persistent learner has anyway; the helper's reading of the second literature report says of one return a stream that it "misstates the value of memory" (the report's words, as quoted there). Under (ii) the next step is a mechanism that keeps the two texts' cells apart.

## 2. What is run (`relay13.py`)

The life of `PLAN_12.md` with three returns: 400,000 lines of sums; English 1 (the first 400,000 bytes of tiny Shakespeare); return 1 (5,000 lines); English 2 (its bytes 420,000 to 760,000); return 2 (5,000 lines); English 3 (bytes 760,000 to 1,100,000); return 3 (5,000 lines). The stream up to the end of return 1 is `PLAN_12.md`'s byte for byte, so the marks up to there must repeat that plan's. The bytes 400,000 to 420,000 are never read and are the held-out English. One byte of English 2, a '$', has a value the alphabet of the life lacks and is left out. The learner is `relay12.py`'s, unchanged, in two forms: R (as it is) and Rgtqr (the four changes, theta 4).

**Marks**, each with nothing changing: the end of the sums; the end of each piece of English; the same with the answering strengths of before that English put back; each return after 10, 20, 50, 100, 200, 500, 1,000, 2,000 and 5,000 lines. At every mark: never-shown sums, whole answer; the same with only the digit branches answering; shown sums; held-out English named right. After each piece of English: the outputs of the right digit's branch and of the loudest other branch in the states of the sums; and **how far the two texts use the same cells**: the cells that are above one half in at least one state in a hundred of the sums, of the English, of both, and the share of each text's activity that sits on cells of the other.

"Kept" at a return = whole answers straight after the English over whole answers straight before it.

## 3. Hypothesis, what refutes it, predictions (mine; see the head of this file for what I had seen)

**Hypothesis (reading ii, in the form the trial suggests).** For the network with the four changes the outvoting comes back at every return: whole answers straight after the second and the third English are under a quarter of their level. What does improve from return to return is the reading among the digits and the speed of the comeback.

**Refuted if** whole answers straight after the third English are at three quarters of their level or more (median of five seeds): then the cost is paid once, reading (i) holds, and no mechanism against shared cells is needed for this pair of texts.

| | Statement (five reported seeds; median) | Chance |
|---|---|---|
| W1 | Rgtqr keeps three quarters or more of its whole answers straight after the second English | 0.15 |
| W2 | The same after the third English (this would refute the hypothesis) | 0.20 |
| W3 | Rgtqr's reading among the digits is kept at 0.9 or more of its level at the second and at the third return | 0.75 |
| W4 | Rgtqr is back within 5 points of its level after 100 lines at the second and at the third return (at the first it is 14 points under) | 0.45 |
| W5 | Its whole answers after 100 lines of the third return are 5 points or more above those after 100 lines of the first | 0.60 |
| W6 | R keeps nothing at any return: whole answers 0 and under 10% among the digits, straight after each English | 0.85 |
| W7 | R's held-out English falls by more than 10 points over each return of the sums; Rgtqr's by less than 5 over each | 0.60 |
| W8 | The cells of the English are mostly cells of the sums as well: more than half of the English's activity sits on cells of the sums (Rgtqr, end of English 1) | 0.70 |

## 4. Runs, size, cost, what is not run

Development seed 1, both forms: to see that the marks up to the end of return 1 repeat `PLAN_12.md`'s development figures. Nothing is chosen. Reported: the seeds 73129, 93683, 116089, 406364, 545353, both forms. Size: 256 cells, about 230,000 strengths. Cost: about 2.5 CPU-minutes a run, twelve runs. **Not run:** the other fourteen forms; any mechanism against shared cells; a second text with no byte in common with the sums at all (English shares the line feed with them); more than three returns.

## 5. Outcome (entered 21 November; five reported seeds, both forms; `report_13.txt`, `res/relay13__r_*.json`, `out/relay13_*.txt`)

**The development seed** repeated `PLAN_12.md`'s development figures up to the end of the first return to within 0.1 point for the form with the four changes and exactly for the network as it is (the first phase is read in one call here and in three there, which changes the rounding of the folded fading). One run (R, seed 406364) was started twice by my job scripts; the duplicate was stopped, the result file is the surviving run's, and its log may hold lines of both.

Never-shown sums, whole answer, per cent; median of five seeds (smallest to largest).

| | Return 1 | Return 2 | Return 3 |
|---|---|---|---|
| **The four changes (Rgtqr).** Straight after the English | 0.2 (0.0 to 0.6) | 30.3 (14.0 to 44.9) | 62.4 (51.0 to 81.8) |
| kept, as a share of the level just before | 0.00 (0.00 to 0.01) | 0.31 (0.14 to 0.45) | **0.64 (0.51 to 0.83)** |
| with only the digit branches answering | 92.9 | 96.6 | 98.6 |
| after 100 lines of the return | 84.8 | 94.0 | 96.9 |
| after 5,000 lines | 99.2 | 99.0 | 98.6 |
| held-out English lost while the sums return, points | 3.9 | 2.0 | 0.9 |
| right digit's branch; loudest other branch (states of the sums, after the English) | 0.08; 0.74 | 0.45; 0.33 | 0.55; 0.14 |
| **The network as it is (R).** Straight after the English | 0.0 | 0.0 | 0.0 |
| with only the digit branches answering, as a share of the level | 0.03 | 0.35 | 0.35 |
| after 100 lines of the return | 0.0 | 0.0 | 0.0 |
| held-out English lost while the sums return, points | 21.9 | 11.9 | 5.0 |

Level at the end of the first phase: 98.8 (Rgtqr) and 96.5 (R). The cells: the sums use 253 of 256; the English 108, falling to 95 by the third piece; every cell of the English is a cell of the sums (0.99 of the English's activity sits on cells of the sums, and 0.44 of the sums' activity on cells of the English).

| | Outcome |
|---|---|
| The hypothesis | **Not supported, and not refuted by its own words.** I said the outvoting would come back in full at every return (under a quarter kept at the second and third). It kept 0.31 at the second and 0.64 at the third: more than I said, less than the three quarters that would have refuted me. The cost of a return falls steeply from return to return; it is neither paid once nor paid in full each time |
| W1 | Failed (0.31 at the second return) |
| W2 | Failed (0.64 at the third; 0.83 on the best seed, 0.51 on the worst) |
| W3 | **Held**: the reading among the digits is kept at 0.98 and 0.99 of its level at the second and third return (0.97 at the least) |
| W4 | Failed, by a tenth of a point: after 100 lines the network is 5.1 points under its level at the second return and 2.4 at the third |
| W5 | Held: 96.9 after 100 lines of the third return against 84.8 after 100 lines of the first |
| W6 | Failed in its second part: R's whole answers are nought at every return, but from the second return on a third of its reading among the digits is still there (0.35 of the level) |
| W7 | Failed for R's third return: R's English falls by 21.9, 11.9 and 5.0 points; Rgtqr's by 3.9, 2.0 and 0.9 |
| W8 | **Held**: 0.99 |

**Reading (mine; development level).** (1) A life in which the two texts keep alternating is itself a teacher: with the four changes, each return leaves the branches of the English quieter in the states of the sums (0.74, 0.33, 0.14) and the digits stronger after the next English (0.08, 0.45, 0.55), and each piece of sums costs the English less (3.9, 2.0, 0.9 points). After three alternations the network has two thirds of its whole answers the moment the sums return, where at the first return it had none. (2) The network as it is gains from alternation as well, in its English and a little in its digits, but its whole answers are nought at every return and stay nought for the first hundred lines. (3) The two texts share nearly every cell they use, so the balance between their branches can only be learned by meeting both; nothing in the first phase could have taught it. (4) One return, as in `PLAN_11.md` and `PLAN_12.md`, shows this learner at its worst; the helper's reading of the second literature report had warned of exactly that.

**Limits.** Three returns, one pair of texts, each English a new piece of one book; whether the kept share goes on rising at a fourth and fifth return was not run. The short trial named at the head of this file misled me toward reading (ii).
