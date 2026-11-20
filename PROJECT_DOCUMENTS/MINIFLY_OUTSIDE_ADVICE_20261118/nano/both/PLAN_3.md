# The joined system through a second task, with replay to the relay system from a store of counting cells (step 3) (20 November 2026)

**Authorisation and source** as `PLAN_1.md` (Mr. Xu's words quoted there; branch `claude/nano-both-8mmhfm`, written on top of commit 77a731d). Copied unchanged into this folder: `nano/replay/replay.py` (its class `Counting`, the store, and `kind`). Written before any run of `life.py`. Written when the runs of `PLAN_1.md` were nearly all in (their outcome is entered there after this plan): the joined system of `PLAN_1.md` falls with reading as the counting system does and hardly hears the relay system. This plan measures that system, unchanged, through two tasks; the changed trusts of `PLAN_2.md` are not used here unless an addition says so before its runs. Development level; no gate, seal or review is claimed.

## 1. The question and the life

**Does the joined system keep its addition through a second task and learn the second task, when the relay system is kept by replay as `nano/replay` found?**

Addition 400,000 lines, then subtraction 200,000 lines (`cells2.py`'s texts; the program shows no addition in the second phase); and the short life, 40,000 then 20,000. The cells that count and the two answering cells read the real lines only. After each real line the relay system learns one line drawn from the store (`replay.py`'s `Counting`, runs of at most 8 bytes inside a line, drawn exactly as `replay6.py`'s arm R: same random stream), in both phases, as that line does. The relay system's messages for a block of 50 real lines come from its strengths at the start of the block. The store is a third thing beside the two kinds of cell (118 thousand cells at most by that line's count); drawing the replayed line from the counting cells of the joined system itself is the natural next form and is not run here.

*Measured*, for the joined second cell, the joined first cell, the relay system alone (from the joined run; nothing of the counting cells reaches it) and the joined second cell with the relay system silent at the score; and for the counting system alone on two seeds (73129, 93683): never-shown additions at the end of each phase after a read addition line and after a read subtraction line; never-shown subtractions on the clean score (pairs never shown whose line no shown pair writes, of the 2,000 never shown, as `replay6.py`) after a read subtraction line. Whole answer, own bytes, five draws of the line before. Five reported seeds for the joined arm.

*Check.* The relay system with replay, scored with `replay6.py`'s own draws, against `nano/replay/res/m_z_sub_R_<seed>.json` (their field `past`).

## 2. Hypothesis, what refutes it, predictions (mine, before any run)

*Hypothesis (the coordinator's).* The joined system keeps at least 90% of its addition level through the second task and reaches at least 88 on subtraction (medians of five seeds, long life). *Refuted if* either fails.

| | Statement (joined second cell unless said; median of five seeds) | Chance |
|---|---|---|
| L1 | long life: addition after the second phase (after a subtraction line) is at least 90% of its level at the end of the first phase | 0.80 |
| L2 | long life: that is so because the level at the end of the first phase is low (under 85), and addition after the second phase is higher than before it | 0.55 |
| L3 | long life: subtraction (clean) at the end at least 88 | 0.35 |
| L4 | short life: addition kept at least 90% of its level | 0.60 |
| L5 | short life: subtraction (clean) at least 88 | 0.45 |
| L6 | the relay system from these runs repeats `nano/replay`'s figures within one point at the median (addition after the second phase 93.8, subtraction 88.0) | 0.85 |
| L7 | long life: the joined answer on addition at the end is under the relay system alone by more than 3 points | 0.65 |

## 3. Cost, what is not run

A long life is about 15 CPU-minutes for the joined arm; seven long and seven short lives, three at a time: about 40 minutes of wall time, about 0.5 GB each. Not run: no replay (the other line's 1.8% is taken as it stands); the harsher change of task; replay drawn from the joined system's own cells; the counting system alone on more than two seeds; any tuning.
