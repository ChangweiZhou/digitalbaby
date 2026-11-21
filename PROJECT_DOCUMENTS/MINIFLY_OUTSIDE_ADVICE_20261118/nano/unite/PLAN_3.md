# nano/unite, third plan: three-digit numbers with letters; what the look-ahead needs (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, in Chinese, in the coordinator's translation as given to this line: "You can go on trying boldly; we keep to the facts; besides the ideas I have mentioned, try others as well." and then "Good, keep trying." The step is the coordinating session's. **Source.** `claude/nano-unite-8mmhfm` at the commit that carries this file. **Status.** Development level; no gate, seal or review is claimed. **Nothing of this plan has been run** (see the end); the first part is an argument, not a measurement.

## 1. What the look-ahead needs on `aaa+<letters>bbb=cccc` (reasoning, mine; not tested)

With the letters out of the chain the nine lines at the `=` hold `=`, the three digits of the second number, the `+` and the three digits of the first: every digit the answer needs is on a line. The answer is written thousands first. A cell is keyed on at most four lines.

- The units digit needs two lines and is exact; it arrives **four** bytes after the `=`.
- The tens digit needs four digits (two for itself, two for the carry) or two digits and the units digit of the answer.
- The hundreds digit needs six digits, or two digits, one digit of the tens and the tens digit of the answer (which settles the carry except at a tie).
- The thousands digit (0 or 1) needs the hundreds digits and that carry again, or one hundreds digit and the hundreds digit of the answer.

With a look-ahead of three, the guesses made at the `=` are of the hundreds digit (two ahead) and the tens digit (three ahead). The guess of the tens digit can only come from plain cells of four digits (10,000 keys, each met about once in 10,000 lines), which is the position of the tens digit on two-digit sums WITHOUT look-ahead (77% in `nano/ahead2`, and the place that falls with more reading here). Everything above it leans on that guess. The units digit, which would make the tens guess a matter of three lines and a guess as on two-digit sums (89 to 91%), is one byte beyond the look-ahead.

**The smallest change that meets it** (as I see it): a look-ahead of four, in the same form: every plain cell also counts the byte four ahead; the answering cell first guesses the byte four ahead; 130 more kinds are keyed on at most three lines and that guess, counting the next three bytes. `unite.py` is written for a look-ahead of any length up to four (`look=4`) and the check covers it (two cases on this text, PASSED). The cost is in size: 68 counts a plain cell in place of 51, and a third family of cells keyed on a guess: about 5.8 gigabytes addressed, against 3.6.

## 2. Question, runs, hypothesis (if run)

Text `t3` as `nano/skip/PLAN_8.md` (500 pairs never shown; nearly every line a pair not read before). Development seed 1, 200,000 lines: `clock=oracle look=3`; `clock=oracle look=4`; `clock=trial look=4`. **H3:** with a look-ahead of four the oracle arm has at least 15 points more never-shown whole answers (four digits, own bytes) than with three, at its best reading; and the united system is within 10 points of the oracle arm at that reading. Refuted if the gain is under 5 points. Predictions: the gain is at least 15 points (0.45); the trial clock closes exactly the four letters within 40,000 lines (0.7); the best reading is before 100,000 lines and the score falls after it as on two-digit sums (0.7).

## 3. Not run

None of the above was run: the three and a half hours given to this line went to steps 1 and 2 and to locating the fall found in step 1, and a run here needs 12 to 20 minutes and 5.8 gigabytes of the 16 this machine has. The reasoning of section 1 is therefore untested, and I would not build on it before the three runs are made.

## Correction (20 November 2026, after this line's runs had ended)

The coordinating session tells me that the line `nano/ahead2` ran this very question today on plain three-digit sums (its `PLAN_2.md` on `claude/nano-ahead2-8mmhfm`): a look-ahead of three answers 99.7% and a look-ahead of four gains nothing, the carry being passed along a place at a time. I have not read that file or repeated the run; I enter it as that line reports it. If it stands, **the argument of section 1 is wrong** where it says the guess of the tens digit made at the `=` can only come from cells of four digits and that everything above leans on it, and H3 as written here would be refuted. Section 1 is left as I wrote it, as a record of reasoning that was not tested and was contradicted by a measurement. Anyone running step 3 should run the look-ahead of three first. The look-ahead's depth is in any case laid down by hand in this system, not found by it.
