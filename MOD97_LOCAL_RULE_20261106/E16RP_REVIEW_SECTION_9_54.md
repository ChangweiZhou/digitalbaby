**9.54 E16-RP (cloud session, 9 November 2026): the held-panel collision did not inflate the composition figures, and sign-rule receipts do not reproduce across machines.**

`E16RP_REPORT.md`, `exp45/SUMMARY_E16RP.md`, `exp45/E16RP_SAME_WEIGHTS.json`.

**What was run.** E16-R3's arm (raw reservoir, 65,536 slots) was rerun on the standing seeds at harness 432e90fb, with the re-keyed, operand-disjoint panels.

**The parity test failed, and the cause is the machine.** Each life was identical to E16-R3 at the birth, which has the same hash. From block 1 every operation panel and the training loss differed. On this container, E16-R3's own harness (7703d6c) and the current one are bit-identical over six 64-update blocks. So the divergence comes from the machine or PyTorch build, not the code: the sign rule amplifies rounding from the first update. P16RP.1 misses as locked and P16RP.2 is void. A parity clause is valid only on the comparator's machine, and every cross-machine comparison in the program is between different runs. Across runs, the +- pair moves by up to 16 points with no panel change.

**The clean measurement is on the same weights.** I scored each life's final weights on the old colliding panels, rebuilt exactly and verified against E16-R3's stage-0 counts, and on the new ones.
- The old +- panel's triples are 600 of 600 in the taught -> set; the new panels are disjoint from every taught set.
- On the same weights, the +- difference (new − old) is +1.0 / +6.8 / −0.8, and the mean absolute difference over the four held pairs is 1.08 / 2.37 / 1.38.
- P16RP.3 holds in two seeds and misses in 93683. That miss runs against contamination: the colliding panel scored lower. It is about 2.5 SE of panel sampling.
- The store adds nothing on these panels (no-store equals with-store within 0.1).

**Conclusion.** At block 6 the collision did not inflate held-composition scores, and the E16-family composition figures need no correction on this evidence. Blocks 4–5 cannot be measured, since only final weights were saved.
