# Line `nano/ahead2`, first plan: an independent repeat of the look-ahead result of `nano/sparse` (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session and given there as its translation from the Chinese: "You can go on trying boldly; we keep to the facts; besides the ideas I have mentioned, try others as well." The task (repeat one result of the line `nano/sparse` with my own program, written from that line's description and not from its code) is the coordinating session's. **Source.** `claude/friendly-darwin-8mmhfm` at 7e2765d; this line's branch is `claude/nano-ahead2-8mmhfm`. Development level. No checker gate, seal or outside review is claimed. Written before any program of this line exists and before any run.

**What I read, and what I did not.** Read: `AGENTS.md`; the roadmap's second revision (rows C3 and K1, section 5); `loop_small_case/REPORT.md` sections 1 to 3; `loop_small_case/cells2.py` for the text, the split and the score; `loop_small_case/ERRATA_20261120.md`; `nano/table/PLAN_3.md` and `trust.py`; the prose of `nano/sparse/PLAN_1.md`, `PLAN_2.md`, `PLAN_5.md`, `PLAN_10.md`. Not opened: any `.py` or `.c` file of `nano/sparse`. One disclosure: while listing files I printed the first lines of `nano/sparse/check4_output.txt` (the output of that line's check, three lines of text). It shows that in that line's small check cases the number of kinds keyed on a guess is one more than the number of plain kinds (sets of at most m delays, or none), where `PLAN_10.md` states 130 for m = 4 (sets of at most three delays, or none). I follow the plan's prose (130) and list this under the open choices. If I open that line's programs later, the report says when.

## 1. The claim to be repeated (as that line reports it)

Two-digit addition `aa+bb=ccc`, 8,000 pairs shown, never-shown pairs scored with the system writing its own answer bytes, whole answer right; median of the five seeds 73129, 93683, 116089, 406364, 545353 of the mean of the last three readings at 40,000 lines: **91.4% with look-ahead** (88.0 to 92.7) against **77.4 without**; tens digit 91.2%. Then 20,000 lines of subtraction with nothing of addition shown: addition 91.2 before, 86.8 after (94.7% of its level); subtraction 89.4 on the clean score. At the first answer byte the guess three ahead is right on 100% of never-shown sums and the guess two ahead on about 86%. Cost: 3.27 million cells, 34.6 million counts above zero.

## 2. What I take the system to be

*Text, split, score* (from `cells2.py`): 14 byte values (`0123456789+=\n-`); a line is `aa+bb=ccc\n` (for subtraction the larger number is written first); for each operator, a permutation of the 10,000 ordered pairs drawn from `default_rng([seed, 99])`, addition first; the first 500 are scored, the first 2,000 are never shown, the other 8,000 are shown. The stream is lines drawn uniformly with replacement from the shown pairs, joined. Scoring: before the line stand line ends only; the bytes up to `=` are the line's; then the system names a byte, that byte is written in place of the line's, three times; right if all three equal the line's. Nothing changes during scoring.

*Cells.* D = 9 delays (delay d: the byte d places before the byte about to be named). A plain kind is a non-empty set of at most four delays: 255 kinds. A cell of a kind exists for each combination of byte values that has occurred at its delays. A plain cell holds three sets of 14 counts: of the byte that followed it (first), of the byte after that (second), and of the one after that (third). A cell says, for a set of counts c with total n, s(v) = log((c(v) + e) / (n + 14 e)), and speaks from a set of counts only once that set has counted (n at least 1).

*Trust.* The answering cell holds one trust for each family, kind and level of experience. Level: floor(log2 n) capped at 9, times four classes of the number of different bytes counted (1, 2, 3 to 4, 5 and more): 40 levels. Six families: plain first, plain second, plain third counts; A first, A second; B.

*Before each byte, in this order.* (1) Guess g3 of the byte two places after the next: the byte of the largest of the 14 branches, each the sum over speaking plain cells of (trust of the third family) times (what the third counts say). (2) Cells of kind A: a set of at most three delays, or none, together with the value of g3 (130 kinds); they hold first and second counts. Guess g2 of the byte after the next: largest branch of plain second counts plus A second counts, each with its trust. (3) Cells of kind B: a set of at most three delays, or none, together with g2 (130 kinds); first counts only. The answer: largest branch of plain first, A first and B counts, each with its trust.

*When the byte arrives.* Each trust used for the answer changes by step times (s(arrived) less the sum over branches of p times s), p being the branches' shares (softmax) at the time the answer was formed. The same for the trusts used for the guess g2 formed one byte earlier and for the guess g3 formed two bytes earlier (their own p and s as they were when the guess was formed). Then the cells count: the first counts of the cells active now, the second counts of the cells that were active one byte ago, the third counts of the plain cells active two bytes ago. Step = eta / sqrt(1 + uses of that trust), eta = 0.1; e = 2; plain trusts born at 0.02; trusts of A and B born at nothing. The guesses are the system's own in learning as in scoring.

*Without look-ahead* (the other arm): plain cells, first counts and their trusts only; the same numbers.

Nothing is chosen on any seed by me: the numbers are those the line states (eta 0.1, e 2, born 0.02, shrinking step).

## 3. Where the description leaves a choice, and what I take

| | Choice | Taken | Other |
|---|---|---|---|
| C1 | Level of experience of a cell's second and third counts | from that set's own total and its own number of different bytes | from the cell's first counts |
| C2 | Birth of the trusts of the plain cells' second and third counts | 0.02, as the plain first | nothing |
| C3 | Kinds keyed on a guess | at most three delays or none (130, the plan's prose) | at most four delays or none (256, as the check output's sizes suggest) |
| C4 | Order at the arrival of a byte | all trusts change (answer, then pending g2, then pending g3), then all counts | counts interleaved |
| C5 | The trust change of a guess | uses p and s stored when the guess was formed | recomputed at arrival |
| C6 | The draw of lines | `rng.integers(8000, size=5000)` for each block of 5,000 lines, from the generator after the two permutations | one draw a line; the line's own stream is not known to me |
| C7 | No cell speaks / ties | all branches 0, the first byte value wins; ties to the lower byte value | |
| C8 | Scoring: the guesses during scoring | formed afresh from the line's own bytes and the system's written bytes; nothing pending is carried | |
| C9 | A cell keyed on a guess is a cell for the guess's value whether the guess was right or wrong | yes | |
| C10 | A cell's "uses" for the shrinking step | one counter for each trust (family, kind, level) | |
| C11 | Readings | every 5,000 lines; "last three" = 30,000, 35,000, 40,000 | |
| C12 | Storage | a cell for every possible combination is addressable; "cells" reported = those whose first counts have counted | |

The two I judge most likely to matter, named now: **C1** and **C2**. For each, the other choice is run on the five reported seeds (40,000 lines) and reported beside the main figure. If the first result differs from the line's, C3 is run too.

## 4. Runs, in this order

1. A numpy program for the whole system; a compiled kernel checked against it on a prefix (probabilities, trusts, counts, guesses). Nothing else is run before the check passes.
2. Without and with look-ahead, `add:40000,sub:20000`, seeds 73129, 93683, 116089, 406364, 545353 and 2001 to 2005 (which no line has used). Readings every 5,000 lines of never-shown addition (whole and by place), and of subtraction on both scores (all 500 held-out pairs; the pairs among them whose written line no shown pair writes). The first phase is the test at 40,000 lines; the second is the keeping.
3. **Control of my choosing: guesses from another line.** Everything as with look-ahead (same kinds, same counts, same trusts, the guesses formed and their trusts taught as before), but the cells of kinds A and B are keyed on the guess the system made at the same place of the preceding line (ten bytes earlier), in scoring on the guess made at the same place of the preceding scored pair. The guess keyed on then has the same distribution and the same rate of occurring, and carries nothing about the present sum. If the gain of look-ahead is the information in the guess, this arm falls to the level without look-ahead; if the gain is from the added cells, trusts and counts as such, it stays near the look-ahead arm. Ten seeds, 40,000 lines.
4. C1 and C2 under the other choice, five reported seeds, 40,000 lines.

## 5. What counts as "repeats" and as "does not repeat" (set before any run)

Figure: never-shown additions, own bytes, whole answer, mean of the last three readings at 40,000 lines, median of five seeds; stated separately for the five reported seeds and for 2001 to 2005.

- **Repeats** (each set of five seeds judged alone; the claim "repeats" needs both sets): (a) with look-ahead at least 85%; (b) at least 8 points above my own arm without it; (c) after 20,000 lines of subtraction the last reading of addition is at least 85% of the last reading before.
- **Does not repeat:** (a) under 80%, or (b) under 4 points, or (c) under 70% of its level, on either set of seeds.
- Anything between is reported as **partly repeats**, part by part.
- Agreement to the digit is not expected (my stream of lines may differ from the line's, and C1 to C5 may differ); the figures are compared as figures.

Secondary, reported and not part of the criterion: tens digit; the two guesses at the first answer byte; subtraction on the clean score; cells and counts above zero.

## 6. Predictions (mine, before any program is written)

| | Statement | Chance |
|---|---|---|
| R1 | reported seeds: with look-ahead at least 85% | 0.70 |
| R2 | reported seeds: at least 8 points above without | 0.65 |
| R3 | reported seeds: kept at least 85% of its level through subtraction | 0.75 |
| R4 | "repeats" on both sets of seeds by all three parts | 0.50 |
| R5 | my arm without look-ahead within 3 points of the line's 77.4 | 0.70 |
| R6 | my look-ahead figure within 3 points of the line's 91.4 | 0.45 |
| R7 | the guess three ahead at the first answer byte right on at least 99% | 0.80 |
| R8 | control (guesses from another line): within 4 points of my arm without look-ahead, and at least 6 below my arm with it | 0.70 |
| R9 | C1 other choice moves the figure by less than 3 points | 0.60 |
| R10 | C2 other choice moves the figure by less than 3 points | 0.65 |
| R11 | subtraction on the clean score at least 80% | 0.60 |

## 7. Size, cost, what is not run

Numbers that learn: trusts, 6 families of up to 255 kinds by 40 levels (those used are reported); counts, 14 for each set of each cell (cells that have counted and counts above zero are reported). My estimate from the line's figures: two to four CPU-minutes for a life of 60,000 lines with look-ahead; about 60 runs in all, under one hour on four cores. Not run: the life of 400,000 lines; the answer written units first; the cap on a cell's weight; any reference taught by back-propagation; anything on prose.

## 8. Outcome (entered 20 November after the runs)

It repeats on both sets of seeds by all three parts of section 5; the control falls to within 1.2 to 2.2 points of the arm without look-ahead; C1 and C2 under the other choice give 91.3 against 91.4. Tables, predictions and readings are in `REPORT.md`.
