# Keeping by replay, fifth plan: the whole system's own answer (20 November 2026)

By the same separate agent session as `PLAN_1.md` to `PLAN_4.md`; same folder, same source. Written before any run of this step; the check was run first (`check_replay4_output.txt`, extended; lives of 20,000 to 25,000 lines). Development level; repeated by no one; no checker gate, seal or review is claimed.

**Authorisation and origin.** Asked by the coordinating session after `PLAN_4.md`, as the arm that plan named and did not run. The hypothesis and its bounds are the coordinating session's.

## 1. The question

`PLAN_4.md` measured the relay system and the store apart: at the end of the life the store alone has 84.8% of never-shown addition right and the relay system 77.0. **What does the system answer as a whole, when it answers from its store where the store holds a line and from the relay system where it does not: never-shown addition and never-shown subtraction, at the end of each phase?**

## 2. The system: no new mechanism

Arm D of `PLAN_4.md` unchanged (replay at one line for each real line, K = 8; one extra line in ten a dream, counted at full weight; add:400000 then sub:200000). The life, the strengths and the store are those of `PLAN_4.md`'s runs, made again with one more score; every earlier figure must come again.

**The whole answer.** After the six bytes of a line (`aa+bb=` or `aa-bb=`) the system gives the three bytes of the answer one at a time, reading its own earlier bytes. For each byte: if the longest counting cell of what has been read so far has counted, the byte is the one with the largest count in that cell (the store's answer); if no such cell has counted, the byte is the relay system's.

*The one number, stated and not tuned:* a cell "holds a line" for a byte when its run is the whole line up to that byte, back to the newline before the line, or the last eight bytes where the line so far is longer (K = 8, the cells of `PLAN_1.md`; the same cells measure (2) of `PLAN_4.md` used). No shorter cell is consulted.

## 3. Measures

At the end of each phase, on the 500 held addition pairs (and all 2,000 never-shown ones) and on the clean never-shown subtraction pairs (those, of all 2,000, whose line no shown pair writes; the mixed 500 beside them): the whole answer right, %; the relay system alone and the store alone on the same pairs; how many answers came wholly from the store, wholly from the relay system, or from both; and, where the store alone and the relay system alone both give an answer and differ, which was right.

## 4. Hypothesis and what would refute it (the coordinating session's)

**At the end of the life the whole system answers at least 80% of never-shown addition and at least 75% of never-shown subtraction (clean), with no line shown again by the program; median of the five reported seeds. Refuted if either is under 70.**

## 5. Predictions (mine), with chances

- W1 (0.85): never-shown addition, whole, at least 80% at the median.
- W2 (0.6): never-shown subtraction (clean), whole, at least 75% at the median.
- W3 (0.75): where the store and the relay system differ on a held addition at the end, the store is right more often than the relay system, on at least four seeds of five.
- W4 (0.6): on clean subtraction at the end the whole answer is below the relay system alone at the median (the store's subtraction lines are dreams made early in the second task, right 62% of the time).
- W5 (0.9): on held addition at the end the whole answer is within one point of the store alone (the store has a line for 498 of 500).

## 6. Cost, not run

Six runs of about 70 seconds (seed 1 as a development pass, nothing chosen on it; the five reported seeds). Not run: a rule that weighs the store against the relay system by the counts; the whole answer in arm L or with other q and w; on the harsh change of `PLAN_3.md`; further seeds unless a bound of the hypothesis is within five points.

---
