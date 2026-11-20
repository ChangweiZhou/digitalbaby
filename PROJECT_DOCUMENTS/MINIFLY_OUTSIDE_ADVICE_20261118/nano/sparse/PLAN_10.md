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
