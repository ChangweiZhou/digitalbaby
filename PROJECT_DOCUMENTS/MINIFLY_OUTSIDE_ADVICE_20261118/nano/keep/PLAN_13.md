# Keeping, thirteenth plan: the text with no new byte, with youth counted on learned bytes only (a finding of the repeating session, run here). 20 November 2026

Line K1 (`nano/keep`); authorisation and level as in `PLAN_1.md`. Written after reading sections 4 and 5 of `nano/keep2/REPORT.md` (branch `claude/nano-keep2-8mmhfm`, commit 2415c9e), before any run of this plan.

**What the repeat found.** In test 3 of my eighth plan I let a young cell count its lines at every byte (`openall=1`), so that the swapped text brought no new line at all. Under the rule as I describe it everywhere else (the bytes a cell learns from), the swapped text brings two lines never met, at two learned bytes of nine; the repeat reports that about thirty cells then join and that subtraction is learned worse than by the plain system (68.8 and 38.0 against 83.8 and 82.2), addition being lost either way.

**What is run.** My final candidate without `openall`, on the swapped text (`subtext=swap`), five reported seeds, no old line and one old line in 32, through `clean.py`: 10 runs. Beside the eighth and eleventh plans' figures for the plain system and for the candidate with `openall=1`.

**Hypothesis.** My program gives what the repeat's program gives. **Refuted if** my medians differ from the repeat's by more than 3 points (addition at the end, subtraction on all 500), or fewer than 20 or more than 40 cells join after the change.

**Prediction.** M1 (0.85): not refuted.

**Not run.** Why a change announced at some bytes only is worse than one announced at none; other partial announcements.

## Outcome (entered 20 November; `table13.txt`, `res/c2_c13_*`, `res/clean_c13_*`)

Seeds 73129, 93683, 116089, 406364, 545353; median. The swapped text, final candidate, youth counted on learned bytes only.

| | Addition at the end: last \| mean of three | Subtraction: all 500 \| clean | Cells joined after the change |
|---|---|---|---|
| no old line | 1.0, 0.0, 0.0, 0.0, 0.0; **0.0** \| 0.0 | 75.4, 68.8, 58.8, 71.2, 64.4; **68.8** \| 67.5 | 30 on every seed |
| one old line in 32 | 8.8, 4.2, 12.2, 4.0, 12.0; **8.8** \| 9.1 | 38.2, 24.8, 42.2, 38.0, 27.6; **38.0** \| 28.7 | 29 on every seed |
| beside: every byte counted (`PLAN_8.md`, `PLAN_11.md`) | 0.4; 13.4 | 75.2 \| 76.2; 70.6 \| 71.2 | 0 |
| beside: plain | 1.4; 16.6 | 83.8 \| 85.6; 82.2 \| 82.7 | - |

My program gives the repeat's figures to the digit (0.0 and 8.8; 68.8 and 38.0; 30 and 29 cells). The hypothesis is not refuted; M1 held.

**The repeat's finding stands, and it is a fault of my eighth plan that it was not there.** I saw that the swapped text brought two lines never met under the rule as I state it, judged that not to be the case I meant to test, and changed the rule for that test alone so that no line was new. I reported the result of the changed rule (the candidate does as the plain system does) and did not run or report the rule as it stands. As it stands the candidate does worse than the plain system: the old cells are silent at two learned bytes of nine, some thirty cells join, all cells then speak on both kinds of line at the other bytes, addition is lost, and subtraction is learned to 68.8 and 38.0 (clean 67.5 and 28.7) against the plain system's 83.8 and 82.2. **A change of task announced by a new line at some bytes only is this candidate's worst case here, worse than no announcement.** Why it is worse I have not looked into.
