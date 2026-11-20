# Line `nano/ahead2`, second plan: does a look-ahead of fixed depth fail on numbers longer than it? (20 November 2026)

**Authorisation.** As quoted to this session by the coordinating session after my final report on `PLAN_1.md`, and given there as its translation from the Chinese, Mr. Xu's word since: "Good, keep trying." The question, the arms, the hypothesis and what refutes it are the coordinating session's (its message of 20 November, 18:10 UTC); the program and the predictions are mine. **Source.** `claude/nano-ahead2-8mmhfm` at 0ff083a. Development level. No checker gate, seal or outside review is claimed. Written before the program of this plan (`ahead3.py`) exists and before any run.

## 1. The question

In `PLAN_1.md` the system guesses the bytes three and two ahead. On `aa+bb=ccc` the units digit is three bytes after the `=`, so the system can state it to itself where the first answer byte is asked. On three-digit numbers, `aaa+bbb=cccc`, the units digit is four bytes after the `=`. **Does the depth of three then fail, and does a depth of four repair it?**

## 2. The text, and what "never-shown" means here

Lines such as `047+985=1032` and a line end: 13 bytes. There are a million ordered pairs. The split is made as `cells2.py` makes it: a permutation of the million pairs from `default_rng([seed, 99])`; the first fifth (200,000) are never shown; the first 500 of those are scored; the stream draws uniformly with replacement from the other 800,000.

A run reads 80,000 lines, so about 95% of the lines read are first readings of their pair and nearly no pair is read three times (the count is reported for every run). Here, therefore, **a "never-shown" sum differs little from any sum in the stream at the moment it is read**: the system is at no point shown most lines more than once, and the score measures how far the cells of at most four delays answer a line from its parts. To put a figure on the difference, 500 pairs that were read in the stream (the first 500 different pairs of the first block) are scored beside the 500 never shown.

Scoring as before: line ends only before the line; the bytes up to `=` are the line's; the system writes its own four answer bytes; right if all four equal the line's. Nothing changes during scoring.

*Do nine delays still reach?* Yes, and nothing is changed. Where the thousands digit is asked the first number's digits are at delays 8, 7, 6 and the second's at 4, 3, 2. Where the hundreds digit is asked the first number's hundreds digit is at delay 9. Where the tens digit is asked the two tens digits are at delays 9 and 5 and the two units digits at 8 and 4; the first number's hundreds digit is out of reach there (delay 10) and is not needed. D = 9 in every arm.

## 3. The system and the arms

The system of `PLAN_1.md` with the depth made a number H (same numbers: eta 0.1 with the shrinking step, e = 2, plain trusts born at 0.02, trusts of cells keyed on a guess born at nothing; kinds as before: 255 plain, 130 keyed on each guess).

- A plain cell holds H sets of counts: of the byte that follows, and of the bytes two, ..., H places on.
- Before each byte the guesses are formed from the farthest inward: g_H from the plain cells' H-th counts; then, for d = H-1 down to 2, g_d from the d-th counts of the plain cells and of the cells keyed on each guess already made; then the answer from all first counts. The cells keyed on g_j (a set of at most three delays, or none, with the value of g_j) hold j-1 sets of counts.
- When a byte arrives, the trusts used for the answer, and for the guess of that byte made 1, ..., H-1 bytes earlier, change; then the corresponding counts count.

Arms: **H = 1** (no look-ahead), **H = 3** (as `PLAN_1.md`), **H = 4**. `ahead3.py` is a new program (reference and compiled kernel); before any run of this plan it must (a) agree with its own reference on prefixes at H = 1, 3, 4 and (b) at H = 3 on the two-digit text give the figures of `ahead.py` on seed 73129 to the digit.

Runs: development seed 1, all three arms, for the cost and to see that it runs; **nothing is chosen on it** (the numbers and the 80,000 lines are fixed here). Then the five reported seeds 73129, 93683, 116089, 406364, 545353. Readings every 10,000 lines; the figure is the mean of the last three readings, median of five seeds.

Reported: whole answer and each place; for each guess at each answer byte the share that is right; cells, counts above zero, trusts used, CPU; the share of first readings.

## 4. Hypothesis (the coordinating session's) and what refutes it

*Hypothesis.* With H = 3 the whole answer on never-shown sums is under 40% and the hundreds digit is the gap; with H = 4 it is at least 20 points above H = 3. *Refuted if* H = 3 already answers 60% or more, or if H = 4 gains under 10 points.

## 5. Predictions (mine, before the program is written)

My reading beforehand differs from the hypothesis in one respect. On this text the tens digit is fixed by four digits (two tens, two units), and all 10,000 combinations of four digits occur in 80,000 lines about eight times each, so a plain cell of four delays can hold the tens digit exactly without any guess; it could not on two digits only because there the four digits are the whole pair. The hard places are the hundreds digit (six digits fix it) and, less, the thousands digit. For the hundreds digit a cell of the two hundreds digits and the two tens digits is right except where the tens digits sum to 9; a guess of the tens digit would settle the carry, but a cell keyed on a guess has at most three delays beside it, which is not enough to hold two hundreds digits, two tens digits and the guess. So I expect the limit of three delays beside a guess to bind at least as much as the depth.

| | Statement | Chance |
|---|---|---|
| D1 | H = 3: whole answer under 40% (the hypothesis' first half) | 0.25 |
| D2 | H = 3: whole answer 60% or more (refutes) | 0.45 |
| D3 | H = 4 at least 20 points above H = 3 | 0.15 |
| D4 | H = 4 gains under 10 points over H = 3 (refutes) | 0.65 |
| D5 | the hypothesis holds as stated (D1 and D3) | 0.08 |
| D6 | H = 3 at least 5 points above H = 1 | 0.60 |
| D7 | in every arm the hundreds digit is the weakest place | 0.75 |
| D8 | H = 4: the guess four ahead at the first answer byte (the units digit) right on at least 99% | 0.85 |
| D9 | H = 4: the guess three ahead at the first answer byte (the tens digit) right on at least 90% | 0.55 |
| D10 | the 500 pairs read in the stream score within 5 points of the 500 never shown, in every arm | 0.55 |

## 6. Size, cost, what is not run

Kinds: 255 plain with H sets of counts; 130 keyed on each of H-1 guesses. Trusts: up to 10 families at H = 4. Cost by estimate: three to six CPU-minutes for a run at H = 4, about 2.3 GB addressed. Eighteen runs, about twenty minutes on four cores. Not run: a depth the system sets itself; kinds keyed on a guess with four delays, or on two guesses; subtraction; more than 80,000 lines.
