# A reading that knows whether its recent memory fits the present text (roadmap second revision, track K; one change from `PLAN_11.md`) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese; my translation): "You can go on trying boldly; we keep to the facts; besides the ideas I have mentioned, try others as well." **Source.** `claude/friendly-darwin-8mmhfm` at 701626250; `two.py`, the lives of `PLAN_5.md`, `PLAN_6.md` and `PLAN_11.md`. Written before any run. Development level; no checker gate.

## 1. The evidence this follows

Three times today what was lost when the task or the text changed was the answering cell's reading and not what the cells hold (`loop_small_case/PLAN_13.md`; `PLAN_6.md` here; the lines `nano/keep` and `nano/sparse` on sums). On prose the case is exact: with companions whose counts fade, after a second text the loss on the first, scored with nothing changing, rises by 0.097 nats (`PLAN_11.md`; 0.009 without companions), while every count that never fades is the same number as before. The answering cell goes on trusting companions that now speak for another text. `PLAN_6.md`, reading 5, named what a reading would need in order not to do that and did not attempt it: to know whether the present text is the recent one.

## 2. The one change

The system already holds that knowledge in a place the answering cell can see. A companion of a long run (three bytes or more) is alive only if that very run occurred within its half-life. At a byte, the number of long companions that are alive (0 to 4) says how well the recent memory fits what is being read now. The one change: **the answering cell's trust in a companion is held separately for each such number** (five trusts where there was one, for each kind of companion and level). Everything else as `PLAN_11.md`: T2, companions of half-life 10,000 bytes born with no trust, e = 0.02, the trust's step 0.00005, one reading. Nothing names a text; the system is never told that the text has changed.

Why it might work without ever meeting a return to an old text in its life: within one text a run of three to six bytes often comes back after more than 10,000 bytes, with its long companions dead and the short companions alive with the recent passage. The answering cell can learn from those moments how far the short companions are to be trusted when the long ones are silent.

*Measured,* as `PLAN_11.md`: each text's last tenth right after its own stream, nothing changing, in the life A, B and in the life of B only; A again at the end of the life A, B. The program with the change switched off must give `PLAN_11.md`'s figures.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis.* With that change the rise on A after B, scored with nothing changing, is under 0.04 nats (it is 0.097), and nothing else is worse by more than 0.005. *Refuted if* the rise stays at 0.07 or more.

| | Statement | Chance |
|---|---|---|
| F1 | the rise on A after B is under 0.04 | 0.35 |
| F2 | the rise on A after B is under 0.07 | 0.60 |
| F3 | A after A within 0.003 of 1.4869 | 0.80 |
| F4 | each book of B right after its stream, in the life A, B, within 0.005 of `PLAN_11.md`'s figure, or better | 0.75 |

## 4. Cost, what is not run

1,120 more trusts (four further sets for seven kinds of companion); about a minute of CPU. Not run: the same idea for the relay system on sums (a trust that reads whether the present line's long counting cells are alive is the line `nano/sparse`'s question in another form); other half-lives; the going-on measure.

## 5. Outcome (entered 20 November after the runs; `res/fresh_T2_0.json`, `res/fresh_T2_1.json`)

Loss in nats a byte, nothing changing during a scoring. With the change switched off the program gives `PLAN_11.md`'s figures to the digit; the compiled loop equals numpy to 2e-15 and equals `two.py`'s loop exactly with the change off.

| | A after A | Bryant, Burgess, Carroll in the life A, B | the same in the life of B only | A at the end of the life A, B | rise |
|---|---|---|---|---|---|
| trust in a companion by kind and level (`PLAN_11.md`) | 1.4869 | 1.4675, 1.2592, 1.3667 | 1.5632, 1.2735, 1.3758 | 1.5837 | 0.097 |
| also by the number of long companions alive | 1.4848 | 1.4628, 1.2536, 1.3581 | 1.5572, 1.2677, 1.3684 | 1.5601 | **0.075** |

**The hypothesis is refuted as written**: the rise on A after B falls from 0.097 to 0.075, and under 0.04 was asked, with 0.07 as the line of refutation. F1 failed. F2 failed (0.075). F3 held. F4 held: every figure is a little better with the change (by 0.002 to 0.009), on the test text and on all three books in both lives.

Readings (mine).
1. **Letting the trust in a companion depend on how many long companions are alive is worth having everywhere by a little** (0.002 to 0.009 nats on seven measures of seven, the books on which nothing was chosen included), **and takes away about a quarter of the cost on an earlier text, not most of it.**
2. Why not most: in a life that never returns to an earlier text, the moments at which the long companions are dead are moments inside one text, where the short companions still hold the same text's recent passage and deserve trust. What the system would need to have met is a return. Addition 1 gives it one.

## 6. Addition 1 (20 November; written after section 5 and before the runs it describes): a life with a return in it

The life is cut in four: the first half of A's stream, the first halves of the three books, the second half of A's stream, the second halves of the books. The system thus returns once to A (and once to the books) before the end, and ends on the books as before. Scored with nothing changing: A's last tenth right after the second half of A, and again at the end; the rise is the difference. Arms: the trust by kind and level, and the trust also by the number of long companions alive.

*Prediction (mine):* with the return in its life the system with the change has a rise under 0.04 (chance 0.40) and at least 0.02 less rise than the system without the change in the same life (chance 0.55). *If neither holds*, a reading of this kind does not learn from one return what a return calls for, and I leave it there.
