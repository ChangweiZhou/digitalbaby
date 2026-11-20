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
