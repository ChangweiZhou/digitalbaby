# Line `nano/sparse`, tenth plan (the coordinating session's E7): look-ahead counts and guesses the cells can be keyed on (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas."; and, in the coordinating session's later message, given there as its translation from the Chinese: "You can go on trying boldly; we keep to the facts; besides the ideas I have mentioned, try others as well." **Source.** `claude/nano-sparse-8mmhfm` at the commit before this one. The idea, the hypothesis and what would refute it are the coordinating session's (its message after my second report); the program `sparse4.py`, its numpy reference and its check (`check4_output.txt`) are this session's. Written before any run other than the check. Development level; no checker gate, seal or outside review is claimed.

## 1. The evidence this follows

1. In the usual order of writing the tens digit is the gap (m = 4: 80.0% whole at 40,000 lines), because no byte before the answer states the carry and no cell short of the whole pair holds it (`PLAN_1.md`).
2. With the answer written units first the units digit is on the lines when the tens digit is named, and m = 4 with a shrinking step has 91.0% (`PLAN_5.md`).
3. A second layer of cells that count which of a first cell's followers arrived gave 69.3% over m = 3, at seventy times the cells (`PLAN_3.md`).

## 2. The question and the one change

**If the system states to itself the bytes it expects two and three places ahead, and cells may be keyed on those statements, does the usual order of writing come level with units first?**

One change from m = 4 with the shrinking step (eta 0.1, e 2, trusts born 0.02; nothing chosen again). Every plain cell holds three sets of counts: of the next byte, of the byte after it and of the one after that; the text supplies each when it arrives. Before each byte, in this order: (1) the plain cells' third counts, read with their own trusts, give a guess g3 for the byte two places after the next; (2) cells of kind A (a set of at most three delays, or none, together with g3) hold counts of the next byte and of the one after; the plain cells' second counts and the A cells' second counts give a guess g2 for the byte after the next; (3) cells of kind B (a set of at most three delays, or none, together with g2) hold counts of the next byte; the plain cells' first counts, the A cells' first counts and the B cells' counts give the answer. A guess is the byte of the largest branch and is the system's own, in learning as in scoring. The trusts of the cells keyed on a guess are born at nothing; a trust used for a guess changes when the guessed byte arrives. Nothing about sums, places or the '=' is laid down: every byte gets its guesses.

Runs: development seed 1 at 40,000 lines first (for the cost); then five reported seeds: (a) `plan=add:40000,sub:20000` (the first phase is the test of the hypothesis, the second the keeping); (b) the answer units first, 40,000 lines; (c) `plan=add:400000`, if the cost allows within this session.

## 3. Hypothesis (the coordinating session's), what would refute it, predictions (mine, before any run)

*Hypothesis.* Never-shown additions whole at least 85% within 40,000 lines and the tens digit at least 88% (median of five seeds). *Refuted if* the whole stays under 82%. Then: after subtraction with nothing shown again, at least 90% of its own level.

Reading (mine) before the run: at the '=' the guess three ahead is the units digit, which cells of the two units digits know exactly; the tens digit two ahead can then use it as the units-first text does, through a cell of the two tens digits, a sign and the guess, beside a plain cell of the tens digits and one units digit; the hundreds digit can use the guessed tens digit exactly. Where the tens digit itself is named, the guess two ahead is again the units digit. So I expect about the level of units first (91) if the trusts find these kinds from nothing within 40,000 lines, which is the doubt.

| | Statement | Chance |
|---|---|---|
| X1 | whole at least 85% | 0.50 |
| X2 | tens at least 88% | 0.45 |
| X3 | at the first answer byte the guess three ahead (the units digit) is right on at least 99% of never-shown pairs | 0.80 |
| X4 | at the first answer byte the guess two ahead (the tens digit) is right on at least 85% | 0.50 |
| X5 | after subtraction at least 90% of its own level | 0.50 |
| X6 | the two orders of writing within 3 points of each other | 0.40 |
| X7 | (c) 400,000 lines within 5 points of 40,000 | 0.25 (the counts' growing weight is still there; no cap is used) |

If it fails, the result files say whether the guesses were wrong, or right and unused (`guesses_right_add`, `top_trust_guess_kinds`, `trust_sum_by_family`).

## 4. Size, cost, what is not run

Kinds: 255 plain (three sets of counts each), 130 keyed on g3 (two sets), 130 keyed on g2. Trusts: 6 families x 256 x 40, those used reported. Cells and counts reported. Cost not known before the first run; my estimate is one to two CPU-minutes for 40,000 lines. Not run: the cap of `PLAN_5.md` together with this; guesses further ahead; a guess that is a spread over bytes in place of one byte; the relay system beside it.

## 5. Outcome of (a) and (b) (entered 20 November after the runs; `res/s_la_dev.json` seed 1, `res/s_la_seed*.json`, `res/s_larev_seed*.json`; tables in `report_e10.txt`, `report_e10b.txt`; check in `check4_output.txt`)

Development seed 1, 40,000 lines: 93.2 (mean of the last three readings), 122 CPU-seconds. Then the five reported seeds (73129, 93683, 116089, 406364, 545353). Never-shown additions, own bytes (the forced score is equal), whole answer, %.

| Arm | lines | each seed, last three | median, last three | median, last | hundreds, tens, units (last, median) |
|---|---|---|---|---|---|
| m = 4, shrinking step (`PLAN_5.md`) | 40,000 | | | 77.4 | 98.4, 77.4, 100 |
| **with look-ahead, usual order** | 5,000 | | | 73.0 | 98.2, 73.0, 100 |
| | 10,000 | | | 84.8 | 98.4, 84.8, 100 |
| | 40,000 | 88.0, 91.3, 91.9, 92.7, 91.4 | **91.4** | 91.2 | 99.2, 91.2, 100 |
| m = 4, shrinking step, units first (`PLAN_5.md`) | 40,000 | 89.9, 89.7, 91.6, 93.4, 91.0 | 91.0 | 91.6 | |
| with look-ahead, units first | 40,000 | 91.7, 92.7, 93.9, 94.6, 93.7 | **93.7** | 94.8 | 99.6, 94.8, 100 |
| relay system (`loop_small_case`) | 400,000 | | 84.2 | | |

Keeping, usual order, `add:40000,sub:20000`, nothing shown again, last reading of each phase:

| | before: each seed; median | after: each seed; median | after as % of before: each seed; median |
|---|---|---|---|
| with look-ahead | 88.0, 91.2, 92.6, 92.6, 91.0; **91.2** | 82.6, 89.0, 86.8, 89.0, 86.2; **86.8** | 93.9, 97.6, 93.7, 96.1, 94.7; **94.7** |
| m = 4 without (`PLAN_2.md`, constant step) | 80.0 | 79.8 | 100.2 |
| relay system (long life) | 84.2 | 1.8 | 2 |

Subtraction at the end of that life on the held-out pairs whose line no shown pair writes: 89.4 at the median (94.1, 88.5, 80.0, 89.4, 91.6), beside 79.5 for m = 4 without look-ahead; on `cells2.py`'s score 93.6.

The guesses, on the 500 never-shown additions, at the end of the life (so after subtraction): where the hundreds digit is named, the guess three ahead (the units digit) is right on 100% on every seed and the guess two ahead (the tens digit) on 85.8% at the median (83.6 to 87.0); where the tens digit is named, the guess two ahead (the units digit) on 100%. On seed 1 at the end of addition alone the tens guess was right on 92.8%.

Kinds keyed on a guess with the largest trust (seed 73129; delays counted back from the byte to be named). Keyed on the guess two ahead, for the next byte: (3, 4, 7), (4, 6, 7), (1, 4, 7): where the tens digit is named these are the two tens digits with one units digit or the hundreds digit, together with the guessed units digit. Keyed on the guess three ahead, for the byte after the next: (2, 3, 6), (3, 5, 6), (3, 4, 6): where the hundreds digit is named these are the two tens digits with a units digit or the operator, together with the guessed units digit. The sizes of all trusts used, summed by family, are in the result files; the cells keyed on a guess hold about a quarter of the total.

Size and cost (medians, after the keeping life): 3.27 million cells that have counted, 34.6 million counts above zero, 35,086 trusts used, 200 CPU-seconds for 60,000 lines (m = 4 alone: 1.36 million cells, 30 CPU-seconds).

**The hypothesis holds in all three parts**: 91.4% whole (at least 85 asked; under 82 would refute), the tens digit 91.2% (at least 88), and after subtraction 94.7% of its own level (at least 90).

| | Statement | Outcome |
|---|---|---|
| X1 | whole at least 85 | held (91.4; lowest seed 88.0) |
| X2 | tens at least 88 | held (91.2) |
| X3 | the units guess at the first answer byte right on at least 99% | held (100 on every seed) |
| X4 | the tens guess at the first answer byte right on at least 85% | held at the median as measured (85.8, after subtraction; 92.8 on seed 1 after addition alone); two seeds of five are under 85 |
| X5 | after subtraction at least 90% of its own level | held (94.7; lowest seed 93.7) |
| X6 | the two orders of writing within 3 points | held (91.4 and 93.7) |
| X7 | 400,000 lines within 5 points of 40,000 | entered with (c) below |

Readings (mine).
1. **The carry is supplied by the counting cells themselves, in the usual order of writing, with nothing laid down about sums.** The system states the units digit to itself before it writes the hundreds digit, cells keyed on that statement use it as the units-first text used the written digit, and the two orders of writing come level (91.4 and 93.7). It is 14 points above the same cells without look-ahead and 7 above the relay system's figure, within a tenth of the relay system's lines.
2. **It keeps.** 86.8% of never-shown additions after 20,000 lines of subtraction with nothing shown again, where the relay system has 1.8% after its second task; and it then answers 89.4% of subtractions whose line was never written. Keeping is a little below m = 4's 100% of its own level: the trusts are shared by the two tasks and the tens guess is right less often after subtraction (85.8 against 92.8 on the one seed where both were measured).
3. **What remains wrong is the tens digit at a tie.** Two cells of three digits each decide the carry except where their chances balance; the remaining 8 to 9 points are there, as with units first.
4. **The cost is the size**: 3.3 million cells and 35 million counts for a text of 16,000 different lines. Nearly all of them are keyed on bytes that carry nothing. This system shows that the route exists; it is not small.
5. The guess is a single byte, the largest branch. A wrong guess is used as confidently as a right one; a guess passed on as a spread over bytes was not tried.
