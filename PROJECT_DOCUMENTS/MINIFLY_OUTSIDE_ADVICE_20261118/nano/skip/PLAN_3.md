# nano/skip, third plan: where the early evidence for passing over a letter comes from (a measurement) (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this line by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-skip-8mmhfm` at 0dc3458, with `look_early.py` added (the system and kernel unchanged). **Status.** Development level, development seed 1, a measurement; no checker gate, seal or outside review is claimed. Written before its run, while the runs of `PLAN_2.md`, C, are in progress.

**Why.** In `PLAN_2.md`, B, the entries closed the four letters within the first 100,000 lines of `g2`, before the sums were learned, which my account before those runs did not foresee. The entries read the mismatch over every byte of the line. Before the result is leaned on, it should be known which bytes supply the evidence.

**The measurement.** The system of `PLAN_1.md` (256 cells, 55,570 numbers that learn) reads 10,000, 30,000 and 100,000 lines of `g2` under three clocks given by the program: none; the one letter w passed over; all four letters passed over. Then 3,000 fresh lines are read with nothing changing, and the answering cell's squared mismatch is summed for each kind of byte to be named (the second digit of the first number, `+`, the byte after `+`, the byte after a letter, the second digit of the second number, `=`, the three digits of the answer, the end mark) and divided by the number of lines. Here the program does pass over named bytes: this is a measurement of the text and the system, not a candidate rule.

**Hypothesis.** The early evidence is in the line's form and not in the answer: at 30,000 lines the chain without the letters has a lower squared mismatch a line than the chain with them, and more than half of the difference lies in bytes other than the three digits of the answer. **Refuted if** more than half lies in the answer's digits, or there is no difference.

| | Prediction (mine, before the run) | Chance |
|---|---|---|
| R1 | at 30,000 lines the squared mismatch a line is lower by at least 0.1 with the four letters passed over than with none | 0.80 |
| R2 | more than half of that difference is in bytes other than the answer's three digits | 0.60 |
| R3 | passing over w alone already lowers a line's squared mismatch at 30,000 lines | 0.70 |

**Cost.** Under 2 CPU-minutes. **Not run.** Other seeds; `m1`.
