# Line `nano/sparse`, eighth plan (J2): the relay system kept by replay, beside the counting cells (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-sparse-8mmhfm` at the commit before this one. `sparse3.py` gains the option `replay=1`: the class `Counting` and the test of a line's form are copied from `nano/replay/replay.py` on `claude/nano-joint-8mmhfm` at fab007c (unchanged; said at the place). The step and its criterion are the coordinating session's proposal. Written before any run with the option. Development level; no checker gate, seal or outside review is claimed.

## 1. The evidence this follows

1. `PLAN_6.md`, development seed 1 (five seeds running as this is written): the whole, joined by a trust that reads the relay system's confidence, has 75.2% before subtraction (relay system alone 83.2) and 13.2% after (relay system alone 1.4; m = 3 alone 32.4 at the median of five seeds). Reading (mine): during subtraction nothing can tell the answering cell that the relay system has lost addition, because no addition line is read; so the whole can keep addition only if the relay system itself keeps it.
2. `nano/replay` (as reported to this session by the coordinating session; its plan read here only in its table): with one replayed line after each real line, made from cells of whole runs of up to eight bytes, the relay system answers 77 to 79% of never-shown addition after 200,000 lines of subtraction, at a cost of about 7 points of subtraction on the clean score.

## 2. The question and the one change

**With the relay system kept by replay, does the whole answer never-shown additions before the second task and keep them after it?** One change from `PLAN_6.md`: after each real line the run cells count it and one line made from them follows; the relay system learns that line as a real one. The replayed lines reach the relay system only: the counting cells of `sparse.py` and the trusts do not see them, and nothing is counted during replay. The program shows no line again. Everything else as `PLAN_6.md` (m = 3 at its numbers, the relay system's trust by confidence, born at nothing). Life `plan=add:400000,sub:200000`, five reported seeds (nothing is chosen, so the development seed is not run first).

*Side by side on the same lives:* the counting cells alone (m = 3, `res/s_keeplong3_*`); the relay system with replay alone (scored in the same run, forced score); the whole. Subtraction on the clean score (lines no shown pair writes) beside the earlier one.

## 3. Hypothesis, what would refute it, predictions (mine, before any run)

*Hypothesis (the roadmap's question for this track, as the coordinating session put it).* The whole answers at least 84% of never-shown additions before the second task and keeps at least 80% of that after it (median of five seeds, last reading of each phase). *Refuted if* it is under 79% before or keeps under 60%.

I do not expect the first half to hold: on seed 1 the whole was 8 points under the relay system alone before subtraction, and replay does not change what happens before the second task.

| | Statement | Chance |
|---|---|---|
| R1 | the whole, before: at least 84% | 0.15 |
| R2 | the whole keeps at least 80% of its own level | 0.50 |
| R3 | the whole, after: at least 15 points above the counting cells alone (32.4) | 0.60 |
| R4 | the whole, after: below the relay system with replay alone | 0.70 |
| R5 | the relay system with replay alone, after: at least 70% | 0.75 |
| R6 | the whole's clean subtraction at the end is at least the relay system's alone | 0.35 |

## 4. Size, cost, what is not run

As `PLAN_6.md`, with the run cells (about 10,000 cells after 2,000 lines of addition on seed 1; reported at the end). About 10 CPU-minutes a run. Not run: the replayed lines shown to the counting cells as well; replay drawn from the cells of `sparse.py` in place of run cells (they do not hold a line from its start); r other than 1; the controls of `nano/replay` (a real old line in place of a replayed one; emptied cells).

## 5. Outcome (entered 20 November after the runs; `res/s_rr_seed*.json`; table in `report_e8.txt`)

Never-shown additions, whole answer, %, seeds 73129, 93683, 116089, 406364, 545353, last reading of each phase. The three systems on the same life (add 400,000 then sub 200,000):

| | before: each seed; median | after: each seed; median | after as % of before | clean subtraction at the end (`cells2.py`'s score) |
|---|---|---|---|---|
| counting cells alone, m = 3 (`PLAN_2.md`) | 59.4, 58.4, 55.4, 61.2, 57.8; 58.4 | 32.4, 42.6, 29.8, 32.4, 38.4; 32.4 | 54.5 | not run on this life (63.0) |
| relay system with replay, alone (in the same runs) | 93.8, 87.0, 86.4, 84.8, 89.8; **87.0** | 85.2, 84.0, 72.8, 82.6, 82.6; **82.6** | 94.9 | 76.9 (80.6) |
| the whole | 83.6, 79.6, 83.8, 80.0, 83.2; **83.2** | 45.0, 52.0, 45.8, 39.4, 52.6; **45.8** | 55.0 | 62.5 (71.0) |
| the whole without replay (`PLAN_6.md`) | 77.6 | 12.4 | 16 | 68.3 |
| relay system without replay, alone | 84.8 | 1.8 | 2 | 83.7 |

Along the second task the whole's addition first falls to 13.6 (5,000 lines of subtraction), returns to 65.4 (40,000) and then wears down to 45.8; the relay system with replay stays between 83 and 87 throughout. After the second task the whole's digits are 73.8, 60.4, 98.4 (hundreds, tens, units); the relay system's 97.4, 85.6, 100. Run cells at the end: 66,050 (median). Cost: about 17 CPU-minutes for the life.

**The hypothesis is refuted in its second half and does not hold in its first**: the whole has 83.2% before (84 asked; under 79 would refute) and keeps 55% of it (80% asked; under 60% refutes).

| | Statement | Outcome |
|---|---|---|
| R1 | the whole, before: at least 84 | failed by 0.8 |
| R2 | the whole keeps at least 80% | failed (55.0) |
| R3 | the whole, after: at least 15 points above the counting cells alone | failed (13.4 above) |
| R4 | the whole, after: below the relay system with replay alone | held, on every seed |
| R5 | the relay system with replay alone, after: at least 70 | held (82.6) |
| R6 | the whole's clean subtraction at least the relay system's | failed (62.5 against 76.9) |

Readings (mine).
1. **Replay keeps the relay system, as `nano/replay` reported, and here too**: 82.6% of never-shown additions after 200,000 lines of subtraction, 95% of its level, with no line shown again by the program. The replayed lines also seem to help it before the second task (87.0 against 84.8 without replay on the same seeds); I claim nothing from 2 points.
2. **The whole is below the relay system with replay at every point of the life**, before the second task by 4 points and after it by 37. With the relay system kept, what pulls the whole down after subtraction is the counting cells of three ends themselves: their hundreds and tens cells hold both tasks in one count (`PLAN_2.md`, reading 2), and they are heard beside a sender that is right.
3. So, to the question the coordinating session put: on this evidence the two kinds joined by a learned trust do not add up, with or without replay, and with replay the better system is the relay system alone. The counting cells' use to the relay system here is as the store that the replayed lines are drawn from, not as a second voice. The counting cells that would be worth joining are those of the tenth plan (91% and kept), which need no relay system for the carry; that joining was not run.
