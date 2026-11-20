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

## 4. Outcome (entered 20 November after all fifteen runs; `res/s4_*.json`, every seed in `report_s4.txt`, made by `report4.py`)

The relay system's never-shown additions, whole answer, own bytes, after a read line, %; median of five reported seeds (range). 65 to 79 CPU-seconds a run on one thread. The counting system's second cell in the same runs has 96.4 to 98.2 at 40,000 lines.

| Real lines | Relay alone (`PLAN_1.md`) | With one more shown line after each (the control) | **With a composed line, scored pairs kept out** | With a composed line, scored pairs among them |
|---|---|---|---|---|
| 5,000 | 3.9 | 21.3 | 26.6 | 25.6 |
| 10,000 | 19.6 | 63.2 | 67.8 | 64.6 |
| 20,000 | 66.0 | 83.6 | 81.8 | 85.9 |
| 40,000 | 82.9 (81.2 to 86.5) | 87.5 (85.7 to 92.4) | **89.7** (88.0 to 91.0) | 91.2 (88.1 to 92.1) |

Composed less control at 40,000 lines, seed by seed: +2.7, +2.9, -4.4, -1.3, +3.5.

*How many composed lines are wrong:* 4.6 to 6.7% of the 40,000 (median 5.8%); 15 to 18% in the first 5,000 lines, 4 to 9% in the next 5,000, 2.6 to 5.1% in the last 20,000. *What the relay system does with them:* on the 971 to 1,236 pairs whose composed line was wrong at least once, at the end it gives the true answer on 79.5 to 83.7% and the composed wrong answer on 13.5 to 18.9% (the counting system itself gives the true answer on 92.6 to 95.0% of them by then). So it copies about one wrong line in six and overrides the rest, at about its level on never-shown sums.

**The answer to the question: no, and it mostly repeats more reading.** With composed lines the relay system reaches 89.7 at 40,000 real lines, not 95. It is 2.2 points above the control at the median and above it on three seeds of five; 3 points were asked for it to count as more than further reading. Most of what composed lines give, a second line of any kind gives: the control is 4.6 points above the relay system alone, composed lines 6.8.

Predictions. T1 failed (89.7; I gave it 0.10). T2 failed (2.2 points). T3 failed narrowly (87.5; 88 asked). T4 failed (91.2). T5 failed (5.8% at the median). T6 held on every seed.

Readings (mine).
1. The fast learner does not make the slow one fast by handing it its answers to new pairs. The relay system's slowness at 40,000 lines is not a want of right lines; it has 40,000 more of them here and gains two points over lines it had already been shown.
2. Letting the scored pairs be composed adds 1.5 points (91.2), far less than I expected: the relay system meets each of them some fourteen times with an answer that is right 96 times in 100 and still answers them no better than 91. It does not hold lines it is shown as the counting cells do; it fits a rule and the rule is not yet fitted.
3. Early in the life both added lines matter a great deal (63 to 68 against 19.6 at 10,000 lines): at that age the relay system simply needs more steps.
4. What is not shown: the same at longer lives; composing by whole lines drawn from the system's own shares (`PLAN_2.md`); the relay system as the taught party and a sender at once.
