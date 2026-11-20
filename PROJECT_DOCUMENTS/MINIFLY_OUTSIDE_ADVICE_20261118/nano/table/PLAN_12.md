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
