# Line `nano/sparse`, second plan (E2): keeping addition through subtraction (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-sparse-8mmhfm` at the commit before this one; `sparse.py` unchanged. Written before any run with a second phase. Development level; no checker gate, seal or outside review is claimed.

## 1. The evidence this follows

1. `PLAN_1.md`, five reported seeds at 40,000 lines (entered in full there once the 400,000-line runs are back): never-shown sums whole, own bytes, m = 3: 56.8% at the median; m = 4: 80.9%; units 100% in both; the tens digit is the place that fails.
2. The relay system: 84.2% before, 1.8% after 200,000 lines of subtraction (`loop_small_case/REPORT.md`, sections 9 and 11).
3. `nano/table`, `PLAN_5.md`: counting cells with trust keep a first text through three other books (0.009 nats).

## 2. The question

**After reading subtraction, with no line of addition shown again, does the system of `PLAN_1.md` still answer never-shown additions?**

*The life.* `plan=add:40000,sub:20000` (the relay's plan add:400000,sub:200000 scaled to this line's counts); and, for the figure beside the relay's own, `plan=add:400000,sub:200000`. Nothing is rehearsed, nothing is told to the system about the change, the trusts and the counts go on changing throughout. Subtraction lines are those of `cells2.py` (the larger number first). Both arms at the numbers `PLAN_1.md` chose: m = 3 (eta 0.008, e 0.1, trust born 0.02), m = 4 (eta 0.1, e 2, trust born 0.02). Nothing is chosen again here. Five reported seeds; readings every 5,000 lines (every 50,000 in the long life).

*Scored:* never-shown addition (own bytes, whole answer) at the end of the first phase and at the end of the second; the ratio; never-shown subtraction at the end. A note on the subtraction score, which is `cells2.py`'s: a never-shown pair (a, b) with a < b is written as b - a, and that line may be among the shown ones, so the subtraction figure is not purely of never-shown lines. The addition figure is.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* Never-shown addition after subtraction is at least 80% of its own level before (m = 4, median of five seeds, short life). *Refuted if* it is under 50% of its own level.

Reading (mine) of why it might hold or fail: a kind whose delays include the operator's place holds separate counts for the two tasks, and the kinds with the most trust at the tens digit in m = 4 already include it. A kind without the operator holds both tasks in one count. The trusts are shared by the two tasks and go on changing during subtraction, so they can move away from what addition needs.

| | Statement | Chance |
|---|---|---|
| K1 | m = 4, short life: addition after is at least 80% of before | 0.65 |
| K2 | m = 3, short life: addition after is at least 80% of before | 0.45 |
| K3 | both arms: addition after is above the relay's 1.8% by at least 20 points | 0.90 |
| K4 | m = 4: subtraction at the end of the short life is at least 50% | 0.55 |
| K5 | m = 4, long life: addition after is at least 80% of before | 0.60 |
| K6 | the units digit of addition stays at 95% or more after subtraction in both arms | 0.60 |

## 4. Size, cost, what is not run

As `PLAN_1.md`, with more cells (those of subtraction's contexts). The short life is about 8 CPU-seconds at m = 3 and 35 at m = 4; the long life ten times that. Not run: rehearsal; a return to addition with learning on; a third task; the relay system through the same short life (its figures are those of the long life).

## 5. Outcome (entered 20 November after the runs; `res/s_keep3_*`, `s_keep4_*`, `s_keeplong3_*`, `s_keeplong4_*`, `s_keepc3_*`; tables in `report_e2.txt`)

Never-shown additions, own bytes, whole answer, %, at the last reading of each phase, with nothing of addition shown during subtraction. Seeds 73129, 93683, 116089, 406364, 545353. Beside them the relay system: 84.2 before, 1.8 after (long life).

| Arm, life | before: each seed; median | after: each seed; median | after as % of before: each seed; median | after, by place (median): hundreds, tens, units |
|---|---|---|---|---|
| m = 3, add 40,000 then sub 20,000 | 57.0, 57.6, 53.6, 60.2, 55.8; **57.0** | 52.4, 53.0, 45.2, 53.2, 52.0; **52.4** | 91.9, 92.0, 84.3, 88.4, 93.2; **91.9** | 89.8, 55.8, 100 |
| m = 4, add 40,000 then sub 20,000 | 80.0, 83.0, 84.0, 77.6, 79.8; **80.0** | 82.2, 83.2, 76.4, 79.8, 75.8; **79.8** | 102.7, 100.2, 91.0, 102.8, 95.0; **100.2** | 96.0, 79.8, 100 |
| m = 3, add 400,000 then sub 200,000 | 59.4, 58.4, 55.4, 61.2, 57.8; **58.4** | 32.4, 42.6, 29.8, 32.4, 38.4; **32.4** | 54.5, 72.9, 53.8, 52.9, 66.4; **54.5** | 70.2, 51.8, 98.2 |
| m = 4, add 400,000 then sub 200,000 | 66.8, 55.4, 66.0, 74.4, 67.4; **66.8** | 74.2, 66.8, 67.2, 68.8, 61.2; **67.2** | 111.1, 120.6, 101.8, 92.5, 90.8; **101.8** | 96.2, 68.2, 100 |

Mean of the last three readings of the second phase (median): m = 3 short 54.5, m = 4 short 76.1, m = 3 long 37.3, m = 4 long 68.6.

Subtraction at the end, `cells2.py`'s score (median): m = 3 short 63.8, m = 4 short 88.2, m = 3 long 63.0, m = 4 long 83.0. The coordinating session has since recorded (`loop_small_case/ERRATA_20261120.md` on `claude/friendly-darwin-8mmhfm`) that about four in five of the 500 held-out subtraction lines are lines a shown pair also writes. On the held-out subtraction pairs whose line no shown pair writes (80 to 104 pairs a seed), m = 3, short life: 62.5 at the median (65.3, 52.1, 63.7, 62.5, 59.0), beside 63.8. For m = 4 the same score is being run as this is written and is entered in `REPORT.md`; until then m = 4's subtraction figures above are mostly of lines that were shown, and I claim nothing from them.

Along the second phase of m = 4's long life the addition score is not flat: 33.4 at 5,000 lines of subtraction (units digit 52%), 47.4 at 10,000, 67.6 at 40,000, 72.4 at 150,000, 67.2 at 200,000 (medians). It falls at once and comes back with no addition shown.

**The hypothesis holds**: in the short life m = 4's never-shown addition after subtraction is 100.2% of its own level before (at least 80% asked).

| | Statement | Outcome |
|---|---|---|
| K1 | m = 4, short life: after at least 80% of before | held (100.2; lowest seed 91.0) |
| K2 | m = 3, short life: the same | held (91.9; lowest seed 84.3) |
| K3 | both arms: after at least 20 points above the relay's 1.8 | held in all four lives (32.4 at the lowest) |
| K4 | m = 4: subtraction at the end of the short life at least 50% | not scored: the score it was written for is mostly of shown lines |
| K5 | m = 4, long life: after at least 80% of before | held by its letter (101.8), but "before" had itself fallen from 80 to 67 (`PLAN_1.md`) |
| K6 | units of addition at least 95% after subtraction, both arms | held (100 and 100 short; 98.2 and 100 long) |

Readings (mine).
1. **These cells keep addition through subtraction with nothing rehearsed, where the relay system keeps 1.8%.** With four ends the level after is the level before, in the short and in the long life.
2. **With three ends keeping is partial and wears with the length of the second task**: 92% after 20,000 lines of subtraction, 55% after 200,000. The loss is in the hundreds digit (99.6 to 70.2) and the tens digit; the units digit stays. Reading: at m = 3 the cells that fix the hundreds and tens digits (two tens digits and a fixed sign) do not include the operator, so subtraction's counts are added into the same cells and after 200,000 lines weigh as much as addition's 400,000 lines did in the cells they share. A kind with the operator among three ends exists for the units digit and not, with enough digits, for the other two. At m = 4 the kinds trusted most at the tens digit include the operator's place (`PLAN_1.md`), and they keep separate counts.
3. **The trusts are shared by the two tasks, and that shows**: m = 4's addition drops to a third within 5,000 lines of subtraction and then recovers without any addition line. While the trusts move to suit subtraction the answering cell is, for a time, wrong about addition; what the cells have counted of addition is untouched throughout. The step of the trust (0.1) is large, and this transient may be much of the swinging seen elsewhere (`PLAN_5.md`).
4. The comparison with the relay system is of keeping only. At the long life's end the relay system knows subtraction better than these cells know it on lines never written (to be entered with m = 4's clean score).
