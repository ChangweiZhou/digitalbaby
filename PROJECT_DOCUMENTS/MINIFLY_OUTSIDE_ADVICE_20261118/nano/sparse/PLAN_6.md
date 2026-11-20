# Line `nano/sparse`, sixth plan (E6): the relay system's answer as one more sender with its own trust (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-sparse-8mmhfm` at the commit before this one. The relay system is `loop_small_case/cells2.py`'s plain system with its compiled kernel (`cells2.py`, `fastk.py`, `fastk3.py`, `kern.c`, `kern3.c` copied unchanged into `relay_copy/`). The way of joining follows `nano/table/relay.py` on `claude/friendly-darwin-8mmhfm` at 37f7336 (read, not copied). Written before any run of `sparse3.py` other than its check (`check3_output.txt`: the kernel with a given sender against numpy; with no sender against `sparse.py`; the relay system through this program's blocks against `cells2.py`'s own run, 3.8 and 49.8 against 3.8 and 49.8). Development level; no checker gate, seal or outside review is claimed. The step was proposed by the coordinating session.

## 1. The evidence this follows

1. The relay system answers 84.2% of never-shown sums at 400,000 lines and keeps 1.8% after 200,000 lines of subtraction.
2. `PLAN_1.md`: counting cells of up to three ends (m = 3) answer 58.2% at 400,000 lines; the tens digit (58.4) is the gap, because no cell short of the whole pair holds the carry. `PLAN_2.md`, short life: m = 3 keeps 92% of its level through subtraction (the long life is running as this is written).
3. So each has what the other lacks. `nano/table`, `PLAN_7.md` (as reported to this session): on prose the relay system as one more sender added 0.012 nats.

## 2. The question

**Read as one more sender with its own trust, does the relay system lift the counting cells' tens digit before subtraction; and after subtraction, when the relay system has lost addition, does the whole still answer additions?**

*The one change.* m = 3 of `PLAN_1.md` at its numbers (eta 0.008, e 0.1, trusts born at 0.02), and beside it the relay system (256 cells in between, its settings unchanged, seed as the run's). It reads the same stream at the same time, once. Its answers for a block of 50 lines are computed from its strengths before the block; then it learns the block. Its message at a byte is the log of its 14 branches' outputs, cut off below at 0.01 and scaled to sum to one (its answering cell gives outputs near 0 and 1, not chances; this conversion is mine). The answering cell of the whole holds a trust for it at each of ten levels of its confidence (its largest output), **born at nothing** (the coordinating session's finding on prose) and changed by the mismatch with the same step as every other trust. Nothing of the counting cells reaches the relay system. Where the whole writes its own answer bytes, the relay system reads those bytes.

*Life.* `plan=add:400000,sub:200000`, read at 5,000, 10,000, 20,000, 40,000 and every 50,000 lines. Development seed 1 first; then the five reported seeds. Scored at each reading: never-shown additions and subtractions by the whole (own bytes; forced beside), and by the relay system alone in the same run (forced, `cells2.py`'s score). Beside them m = 3 alone through the same life (`res/s_keeplong3_*`).

## 3. Hypotheses, what would refute them, predictions (mine, before any run)

*Hypothesis A (before subtraction).* At 400,000 lines the whole answers at least as many never-shown sums as the relay system alone in the same run, less 3 points. *Refuted if* it is more than 10 points below the relay system alone.

*Hypothesis B (after subtraction), which I do not expect to hold.* After 200,000 lines of subtraction the whole's never-shown addition is at least 80% of what m = 3 alone has at the same point. *Refuted if* it is under half of that.

Why I expect B to fail: the trust for the relay system is by its confidence only. During subtraction the relay system is confident and right on the lines read, so its trust stays high; on an addition line it is then confident and wrong, and nothing in a trust by confidence tells the two kinds of line apart.

| | Statement | Chance |
|---|---|---|
| G1 | the whole, before: at least 80% | 0.60 |
| G2 | the whole, before: at least the relay system alone in the same run | 0.55 |
| G3 | the whole, before: at least 90% | 0.20 |
| G4 | after: at least 80% of m = 3 alone's level at the same point | 0.30 |
| G5 | after: at least 20 points above the relay system alone | 0.65 |
| G6 | the relay system's trust at its highest level of confidence is, at the end of subtraction, at least half of what it was at the end of addition | 0.70 |

## 4. Size, cost, what is not run

Numbers that learn: m = 3's (115,000 cells, 775,000 counts, 5,160 trusts), the relay system's (about 40,000 strengths), ten trusts for the relay system. Cost: my estimate is three to five CPU-minutes for the life. Not run here: the relay system beside m = 4; a trust for the relay system that also depends on the operator or the place; anything of the counting cells reaching the relay system; the short life.
