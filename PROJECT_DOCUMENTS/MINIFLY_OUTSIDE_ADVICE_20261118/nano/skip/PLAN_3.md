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

## Outcome (entered 20 November 2026; development seed 1; `res/p3_look_early_s1.json`, printed in full by `look_early.py`)

A line's squared mismatch at the answering cell, summed over the bytes of the line, on 3,000 fresh lines with nothing changing:

| Lines read | no clock | w passed over | four letters passed over | the difference (none less four letters), of which in the answer's three digits |
|---|---|---|---|---|
| 10,000 | 8.413 | 8.389 | 7.973 | 0.440, of which 0.491 |
| 30,000 | 8.521 | 8.475 | 7.631 | 0.890, of which 0.902 |
| 100,000 | 8.244 | 8.211 | 6.907 | 1.337, of which 1.374 |

By kind of byte at 30,000 lines (no clock / four letters passed over): answer's hundreds 0.329 / 0.108, tens 0.890 / 0.609, units 0.928 / 0.528; the byte after a letter 3.548 / 3.579; every other kind differs by 0.015 or less.

**The hypothesis is refuted.** The whole of the difference is in the answer's three digits (the other bytes are, if anything, a little worse named without the letters in the chain: the sum over them is 0.012 to 0.051 higher). R1 held (0.890). **R2 failed** (none of the difference is outside the answer). R3 held in sign at each of the three lengths (0.024, 0.046, 0.033 lower with w alone passed over; single runs, and small beside the 8.4 total), mostly in the hundreds digit (0.329 to 0.300).

**Correction of my reading in `PLAN_2.md`, outcome of B.** I wrote there that a chain without the letters "must already make the line's other bytes easier to name". It does not. The evidence is in the answer, and it is early because the aligned chain begins to learn the answer at once: after 10,000 lines the units digit's squared mismatch is already 0.767 against 0.924, and the hundreds 0.118 against 0.335. My estimate before `PLAN_2.md` (that nothing rewards closing a letter until the cells answer aligned lines, and that this comes late) was wrong in its second half: in the periods in which several letters happen to be closed together many lines are aligned, the cells learn from those within a few thousand lines, and from then on closing a letter pays.

Reading (mine) of what this says about `m1`, where the same rule is slower: there the answer is one digit modulo 10, which has no part that a roughly placed first number helps with (no hundreds digit), so one letter closed alone earns nothing until the others are. Not tested here.

**Cost.** About 1 CPU-minute.
