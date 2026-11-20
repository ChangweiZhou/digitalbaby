# Keeping, eighth plan: three tests of the seventh plan's candidate (the other order; more old lines; a change of task that brings no new line). 20 November 2026

Line K1 (`nano/keep`); authorisation and level as in `PLAN_1.md`. Written after the seventh plan's outcome was entered; before any run of this plan. The system is not changed, with one exception stated under test 3.

**The candidate** (seventh plan): `H=384 H0=1 join=hear K=128 theta=20 young=4500 meet=30 blaw=exp btau=1e4`. No setting is chosen in this plan; seed 1 is not used.

**Question.** Does what the seventh plan showed hold when the order of the tasks is reversed and when old lines are more frequent, and does it fail, as I have said it must, when the second task brings no line never met?

**Test 1, the other order.** `plan=sub:400000,add:200000`, no old line and one old line in 32 (then a subtraction line). Beside the plain system (`cells2.py`), same plans.

**Test 2, more old lines.** `plan=add:400000,sub:200000 reh=8`. One old line in 8 gives about 62 meetings of each operator line of addition in a late cell's youth, above the threshold of 30, so the late cells should open to addition lines. Beside the plain system (71.2 and 83.2 in `loop_small_case/REPORT.md`, section 9; rerun here).

**Test 3, a change of task that brings no new line.** Subtraction is written with the bytes of addition: `aa=bb+ccc` for aa - bb = ccc (larger number first), `subtext=swap`. The operator bytes are those of addition at other places. For this test only, a young cell also counts its lines at the one byte of each text line from which the system does not learn (`openall=1`); without that, two lines (`+` seven bytes back, `=` four bytes back) are active in addition only at that byte, would never be met by the old cells, and would silence them at two bytes of each new line, which is not the case I mean to test. With it, I have checked by enumeration that the swapped text activates no line that addition does not (the usual subtraction activates ten: `-` at each delay). `openall=1` is a change to the rule; one run of the candidate with it on the usual texts (seed 73129, one old line in 32) is made to see whether it changes anything there. Beside the plain system on the same text (`kage.py` with all 256 cells present and no rule, which is `cells2.py`'s arithmetic). No old line and one in 32.

**Hypothesis (one, for test 3, the one that can refute what I have claimed).** The candidate sees a change of task only through a line never met. **Refuted if** on the swapped text, with no old line, the candidate keeps addition above 20% at the median of five seeds (the plain system keeps about 2% on the usual text), or more than 8 cells join after the change.

**Measures.** Never-shown pairs of both tasks at the end, five reported seeds, median; the first task after its own phase; cells joined by the change and by the end.

**Predictions.**
- H1 (0.75): test 1, no old line: subtraction at the end within 5 points of its level after its own phase, and addition at the end 75% or more (medians).
- H2 (0.65): test 1, one old line in 32: both tasks 75% or more (medians).
- H3 (0.60): test 2: the candidate's addition at the end is 5 points or more below its 81.6 with one old line in 32 (median).
- H4 (0.55): test 2: the candidate's addition at the end is at or above the plain system's (median).
- H5 (0.85): test 3, no old line: at most 8 cells join after the change and addition at the end is 10% or less (median).
- H6 (0.70): test 3: in both conditions the candidate's addition at the end is within 10 points of the plain system's on the same text (medians).

**Cost.** 51 runs of 40 to 90 CPU seconds: about 15 minutes on the clock.

**Not run.** A rule that would see a change of task made of met lines in a new arrangement (closing by pairs of lines); a third task; other pool sizes.
