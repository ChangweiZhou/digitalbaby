# nano/skip, fifth plan: more byte values to find (8, 16 and 26 letters where there were 4) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this line by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-skip-8mmhfm` at e0ea938, with `nlet` added to `skip.py` (the number of letter values; with 4 the readings are those of the earlier plans, checked on one run; `check_skip_output.txt` PASSED on 18 cases). **Status.** Development level; no checker gate, seal or outside review is claimed. Written before any run with more than four letters except the 300-line cases of the check, and before the outcomes of `PLAN_2.md`, C, and `PLAN_4.md` are known (those runs are in progress).

**Why.** The trial clock gives every entry the same single number. That is the form of learning that is known to slow down as the number of things to be decided grows, since each entry must find its own effect in a number that all the others move as well. On `g2` four letter values were found within 100,000 lines (`PLAN_2.md`, B). A text of real bytes has many more values. Before the idea is carried further, how the time to find the letters grows with their number should be measured where a run takes two minutes.

**One change.** The text `g2` with 8, 16 or 26 letter values in the gap in place of 4; the gap is still 0 to 8 letters, each drawn afresh from the letter values. Each letter value is then rarer (one gap byte in 8, 16, 26), and a line's gap is closed only if every letter value in it is.

**System.** As `PLAN_2.md`, second form, nothing retuned (alpha 2, p0 0.2, fading 1/3,000, P 256, floor 0.01). 256 cells. The numbers that learn grow with the byte values: 67,862 with 8 letters (22 byte values), 92,446 with 16 (30), 123,176 with 26 (40); at the entries 4 for each byte value.

**Runs.** A (development seed 1, 1,200,000 lines): the rule at 8, 16 and 26 letters; `clock=none` at 26; `clock=oracle` at 26 for 400,000 lines (the check that the larger system learns the aligned text). B (five reported seeds, 1,200,000 lines): the rule at 8, 16 and 26 letters; `clock=none` at 26.

**Question.** How does the time the rule needs grow with the number of byte values to be found, and does it still arrive within 1,200,000 lines at 26?

**Hypothesis H5.** With 26 letter values the rule still has at least 50% of never-shown pairs at every gap within 1,200,000 lines (median of five seeds). **Refuted if** under 50% at any gap.

*What I expect (Reading, mine, before the runs).* The share of lines that chance closing aligns at the start does not depend on the number of letter values, but each entry's own evidence is spread over as many periods as there are values, so the first closings should come later roughly in proportion, and at 26 the fading of the evidence (3,000 periods) may be faster than its arrival.

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| T1 | 8 letters: at least 50% at every gap, median of five seeds | 0.80 |
| T2 | 16 letters: the same | 0.45 |
| T3 | H5 (26 letters) | 0.30 |
| T4 | the reading from which all letters stay closed comes later with every step in the number of letters (medians of five seeds; runs that never arrive counted as later) | 0.75 |
| T5 | no digit is closed at the end of any of the fifteen reported-seed runs | 0.65 |

**Cost.** About 2.5 CPU-minutes a run at 26 letters; about 50 CPU-minutes in all. **Not run.** A retuned rule for more values (a longer memory of the evidence, a different starting chance): that would be the next plan if H5 fails; more cells; letters placed elsewhere in the line.
