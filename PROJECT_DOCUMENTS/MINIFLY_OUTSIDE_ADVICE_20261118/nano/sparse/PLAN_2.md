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
