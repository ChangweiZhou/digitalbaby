# Fifth plan: cells that hold bytes themselves (roadmap R3) (20 November 2026)

Same authorisation as PLAN_4.md. Written before any run of `hold.py`.

**Question.** In the plans so far a chain of delay cells, given by the program, keeps the last bytes present on lines. Here it is removed: the only lines are those of the present byte. Can cells that have a state of their own come to hold what the answer needs, learning from the same relayed mismatch, with no pass backward in time?

**System.** H cells. Each reads the 14 lines of the present byte and K other cells, drawn at birth, one byte late. Each has a gate z and a candidate c, both logistic functions of what it reads, and its state is h = z h(before) + (1 - z) c: with z near 1 it holds, with z near 0 it takes what arrives (the gated cell, which Mr. Xu has allowed to stay, in its smallest form). Thresholds of the gates are spread at birth, so that cells differ in how long they hold. One answering cell reads the cells' states, as before; the mismatch and the learned return connections are as before.

**Learning, forward in time.** Each connection keeps one running quantity, a trace: how its strength has shaped its own cell's present state. At every byte the trace is multiplied by the cell's gate z and receives what the connection carried now times the cell's own sensitivity. A strength changes by the signal relayed to its cell times its trace. A cell uses nothing of another cell but that cell's output, and nothing runs backward over the line. (This is the form with no crossing between cells that the audit of 18 November proposed for E61; credit for what a cell passes to another cell is not given.)

**Arms.** `learn`; `frozen` (the cells never change; only the answering cell learns: what a random set of holding cells gives by itself). Reference: the same text with the delay chain given (97.5% of single-digit sums shown; 88% of two-digit sums never shown). No comparator with a backward pass over the line is run; that is a gap.

**Texts and score.** Single-digit sums (60,000 lines; sums shown, since no learner answers the others); then two-digit sums (never-shown pairs). Development on seed 1 over eta1 in 1/16, 1/4, 1 and H in 64, 256 with K = 16; the best setting on the five reported seeds.

**Predictions.** H1 (0.6): on single digits `learn` has at least 80% of sums shown right at the median and `frozen` at most 50%. H2 (0.35): on two-digit numbers `learn` has at least 50% of never-shown sums right within 800,000 lines at the median. H3 (0.8): wherever at least 30% of shown two-digit sums are right, never-shown sums are within 10 points of them.

## Entered on 20 November after the development runs and before any reported seed

Development seed 1, K = 16 (`res/hold_h1_*.json`, `hold_h1long_*.json`, `hold_h2_*.json`).

- *Single digits.* At the planned 60,000 lines the sums shown were still rising (256 cells: 55% and 56% at eta1 1/4 and 1, last readings 68%; 64 cells: 28% to 34%; cells that never change: 2.5% at both sizes). **Amendment:** the run is lengthened to 240,000 lines, where the development seed gives 89% (eta1 = 1/4) and 92% (eta1 = 1) of sums shown, mean of the last three readings. The setting for the five reported seeds is **256 cells, K = 16, eta1 = 1, 240,000 lines**, for `learn` and for `frozen`. H1 is read at that length.
- *Two-digit numbers.* At 600,000 lines, 256 cells: 1.7% (eta1 = 1/4) and 1.3% (eta1 = 1) of never-shown sums; by place at the end, hundreds 86 to 87, tens 22 to 26, units 7 to 12 (a digit guessed is right one time in ten). Nothing holds on the development seed, so H2 is recorded as failed at development level and the five reported seeds are **not run** on this text. H3 has no case to be read on.
- *My reading of the two-digit failure,* which the sixth plan tests: a cell here changes only for what its own state gives the answering cell directly. One digit of an operand, held alone, says nothing about the units or the tens digit of the sum (every value is equally likely given one digit), so a cell that holds it receives no signal for holding it; it does say something about the hundreds digit, which is the one place learned. On single digits the first byte of the answer depends on one operand alone.
