# The reverse help: the counting system composes lines for pairs it has not met and the relay system learns them (step 4) (20 November 2026)

**Authorisation and source** as `PLAN_1.md` (Mr. Xu's words quoted there; branch `claude/nano-both-8mmhfm`, written on top of commit of `PLAN_2.md` addition 2). The question is the coordinator's. Written before any run of `teach.py`. Development level; no gate, seal or review is claimed.

## 1. The question and the arms

After 40,000 lines the counting system answers 97.4% of never-shown sums and the relay system 82.9%. **If the relay system also learns lines the counting system composes for pairs it has not met, does it reach 95% within 40,000 real lines; or does that only repeat what more reading of the shown lines would do?**

The counting system (`merge.py`'s cells with classes, the two answering cells; alone, the relay system is not a sender here) and the relay system (`cells2.py`'s, unchanged) read the same stream. After each real line the relay system learns one more line:

- **composed**: two numbers drawn by chance among the pairs not yet met (the program's record of the pairs it has shown), written with the counting system's own answer (second cell, own bytes, read after the real line). The counting system reads a chunk of 500 real lines and then composes the 500 lines, so its answers are at most 500 lines newer than the line they follow.
- **composed, scored pairs kept out** (the arm the question is judged on): the same, but the 500 pairs that are scored are never composed. Otherwise the relay system would have met the scored pairs with the counting system's answers, and its score would be in part a score of lines it was taught.
- **shown** (the control): one more line drawn by chance from the shown lines; the same number of added lines.
- none: the relay system alone (82.9 at 40,000 lines, `PLAN_1.md`); not run again.

*Stated plainly:* drawing "a pair not yet met" uses the program's knowledge of what a pair is and where it stands in a line; it is the coordinator's question as asked, and it is not a composing rule that carries to another text. The form that carries is `dream2.py`'s (the counting system draws a whole line from its own shares and the line is kept if it has not been read); it is not run in this step.

*Measured:* never-shown additions for the relay system (whole answer, own bytes, after a read line, five draws) at 5,000, 10,000, 20,000 and 40,000 real lines; how many composed lines are wrong, by stretch of the life; for the pairs whose composed line was wrong, whether the relay system at the end gives the true answer or the composed wrong one. Five reported seeds.

## 2. Hypothesis, what refutes it, predictions (mine, before any run)

*Hypothesis (the coordinator's question put as one).* With composed lines (scored pairs kept out) the relay system reaches at least 95 at 40,000 real lines (median of five seeds). *Refuted if* it is under 95; *and the help is no more than further reading* if it is not at least 3 points above the control.

| | Statement (median of five seeds, 40,000 real lines) | Chance |
|---|---|---|
| T1 | composed, scored pairs kept out: at least 95 | 0.10 |
| T2 | that arm at least 3 points above the control | 0.35 |
| T3 | the control at least 88 | 0.65 |
| T4 | composed with the scored pairs among them: at least 95 | 0.50 |
| T5 | wrong composed lines are at most 5% of all composed lines | 0.35 |
| T6 | on the pairs composed wrongly the relay system gives the true answer more often than the composed wrong one | 0.50 |

## 3. Cost, what is not run

Fifteen runs of 40,000 real lines, two to three CPU-minutes each. Not run: longer lives; the relay system as a sender at the same time; composing by drawing whole lines; both helps at once.
