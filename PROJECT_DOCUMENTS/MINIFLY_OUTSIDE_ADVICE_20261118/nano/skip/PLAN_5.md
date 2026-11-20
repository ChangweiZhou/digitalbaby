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

## Outcome (entered 20 November 2026; `res/p5_*.json`)

`g2` with more letter values, 1,200,000 lines, the trial clock (second form, nothing retuned), never-shown pairs, mean of the last three readings (one figure for the three gaps once the letters are closed). Seeds in the order 73129, 93683, 116089, 406364, 545353; the development seed 1 beside them.

| Letter values (byte values; numbers that learn) | Each seed; median | Seed 1 | The reading from which every letter is closed, each seed (thousands of lines) | Anything else closed at any reading |
|---|---|---|---|---|
| 4 (18; 55,570) (`PLAN_2.md`, C) | 96.7, 95.1, 96.9, 94.1, 97.4; **96.7** | 95.1 | 200, 200, 200, 200, 100 | no |
| 8 (22; 67,862) | 96.3, 95.4, 97.4, 93.7, 95.7; **95.7** | 94.5 | 300, 200, 200, 200, 200 | no |
| 16 (30; 92,446) | 96.3, 96.0, 96.4, 94.2, 96.6; **96.3** | 94.7 | 300, 400, 300, 400, 400 | no |
| 26 (40; 123,176) | 95.1, 94.8, 95.4, 93.9, 92.6; **94.8** | 91.1 | 500, 500, 500, 400, 500 | no |

`clock=none` with 26 letters, gaps 0 / 4 / 8: 40.9, 45.6, 48.6, 44.9, 51.1 (median 45.6) / 1.7, 1.6, 2.0, 1.3, 1.3 / 0.9, 1.0, 1.1, 0.8, 0.6; seed 1: 36.2 / 0.8 / 0.4. `clock=oracle` with 26 letters, seed 1, 400,000 lines: 94.1.

**H5 holds**: with 26 letter values the rule has 94.8% at every gap at the median (least seed 92.6), where under 50% would refute. **T1, T2, T3 held. T5 held** (in the eighteen runs nothing but letters is closed at any reading, `-` apart). **T4 failed by its letter**: from 4 to 8 letters the median reading is the same (200,000); from 8 to 16 to 26 it is later each time (200,000, 400,000, 500,000).

**Reading (mine).** My expectation before the runs was too dark, and wrong in its form. Six and a half times as many byte values to find cost about two and a half times as many lines, not six and a half, and the rule arrives with room to spare. I had reasoned that each entry's evidence thins in proportion to the number of values. What I left out is that the entries do not have to be found one after another: every letter's closing is rewarded as soon as lines are answered that happen to be aligned, the letters close in parallel (the counts closed at successive readings are, for example, 3, 12, 19, 26), and each one closed makes more lines aligned for the rest. The readings are 100,000 lines apart, so the growth is known only that coarsely. What is still untested is the regime that would matter for real text: hundreds of byte values of which only some are to be passed over, and bytes that are worth keeping in some places and not in others.

**Cost.** 117 to 211 CPU-seconds a run with the rule; about 55 CPU-minutes in all.
