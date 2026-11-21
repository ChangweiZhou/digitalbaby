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

## Outcome (entered 20 November, after all 51 runs; `table8.txt`, `res/c2_r8_*`)

Never-shown pairs, %, seeds 73129, 93683, 116089, 406364, 545353; median. In every run of the candidate 128 cells had joined by the change of task.

**Test 1, the other order** (subtraction 400,000 lines, then addition 200,000). Subtraction after its own phase: candidate 96.4, 90.8, 95.2, 93.8, 92.8 (93.8); plain system 86.0 (`PLAN_1.md`'s figure for 200,000 lines is not comparable; here it has 400,000: not read separately).

| Arm | Subtraction (first task) at the end | Addition (second task) at the end |
|---|---|---|
| plain, no old line | 3.4, 2.8, 2.6, 3.6, 2.2; **2.8** | 75.2, 81.8, 85.4, 70.2, 78.8; 78.8 |
| candidate, no old line | 96.2, 91.0, 95.6, 93.6, 93.2; **93.6** | 81.2, 88.2, 66.0, 79.0, 75.0; **79.0** |
| plain, one old line in 32 | 13.8, 18.4, 17.8, 26.2, 17.4; **17.8** | 74.0, 77.0, 82.6, 74.4, 74.8; 74.8 |
| candidate, one old line in 32 | 96.2, 92.0, 96.0, 94.2, 93.0; **94.2** | 79.6, 90.0, 83.4, 75.0, 84.8; **83.4** |

128 more cells join at the change; at the end 128 cells may speak on subtraction lines only and 128 on addition lines only. The first task is kept at its own level in the other order too.

**Test 2, one old line in 8.**

| Arm | Addition at the end | Subtraction at the end |
|---|---|---|
| plain | 73.2, 71.2, 69.0, 72.0, 68.6; **71.2** | 83.2, 82.6, 87.8, 80.2, 83.8; 83.2 |
| candidate | 82.6, 69.8, 78.6, 66.6, 76.8; **76.8** | 85.4, 78.6, 88.6, 83.8, 85.0; 85.0 |

As expected the late cells open to addition lines (all 128 may speak on both kinds). The candidate then keeps 76.8, a little above the plain system at the median and below it on two seeds of five; and below its own 81.6 with one old line in 32 and 85.8 with none. So for this candidate more old lines are worse than fewer, until they are so many that the late cells are simply taught both tasks.

**Test 3, a change of task that brings no new line** (subtraction written `aa=bb+ccc`).

| Arm | Addition at the end | Subtraction at the end | Cells at the end |
|---|---|---|---|
| plain (256 cells), no old line | 2.2, 0.2, 0.6, 2.0, 1.4; **1.4** | 83.0, 82.6, 84.6, 84.4, 83.8; 83.8 | 256 |
| candidate, no old line | 0.2, 1.0, 0.4, 0.0, 0.4; **0.4** | 79.2, 71.0, 82.2, 73.4, 75.2; 75.2 | 128 on every seed |
| plain, one old line in 32 | 20.2, 11.2, 13.6, 29.8, 16.6; **16.6** | 85.4, 82.2, 85.8, 78.6, 78.8; 82.2 | 256 |
| candidate, one old line in 32 | 5.2, 13.4, 19.4, 4.2, 14.2; **13.4** | 66.2, 69.6, 78.0, 71.6, 70.6; 70.6 | 128 on every seed |

**The limit is real and is as stated.** When the second task brings no line never met, no cell falls silent, no cell joins, and the candidate loses addition as the plain system does (0.4 against 1.4 with no old line; 13.4 against 16.6 with one in 32), while learning subtraction less well with its 128 cells than the plain system with 256. The hypothesis of this plan (the candidate sees a change of task only through a line never met) is not refuted.
`openall=1` on the usual texts (seed 73129, one old line in 32) gives the same twelve readings as without it.

**Predictions.** H1 (0.75) held: subtraction 93.6 against 93.8, addition 79.0. H2 (0.65) held: 94.2 and 83.4. H3 (0.60) failed by a hair: 76.8, 4.8 points below 81.6 where I had said 5 or more. H4 (0.55) held at the median (76.8 against 71.2), on three seeds of five. H5 (0.85) held: no cell joins; 0.4. H6 (0.70) held: 0.4 against 1.4; 13.4 against 16.6.

**Not run**, as planned: a rule that closes by pairs of lines; a third task; other pool sizes.

*A correction, added after the outcome.* In test 1 above I gave the plain system's subtraction after its own phase as "86.0" with a muddled remark. That figure is from another plan (200,000 lines of subtraction after addition). In this test's runs the plain system's subtraction after its own 400,000 lines is 94.4, 93.4, 95.2, 94.4, 87.4; median 94.4 (the candidate: 93.8).
