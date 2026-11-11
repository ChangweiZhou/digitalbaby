# 9.77 — E20-A: composition with an operand in memory

E20-A asks whether a learner on file composes when the first operand has scrolled out of the ten-byte window. Composition prompts carry 0–6 distractor letters (`E16_COMPGAP=6`). The test is a plus-minus twin: a pair of items that differ only in the hidden operand. A responder that sees only the recent bytes gets both members right on 0 twins by construction.

**Arms.** Four arms, on seeds 73129, 93683 and 116089, scored at end of life on whole worked answers with no store:
- the window reference (backprop, joint stream);
- the sign-rule window learner (raw reservoir);
- backprop on that learner's pipeline;
- the windowless two-bank reference.

| Arm | Twins both right (of 300) | Twin items (of 600) | Same items in view (of 600) |
|---|---|---|---|
| Window reference | 63 / 80 / 130 | 254 / 286 / 398 | 470 / 420 / 484 |
| Backprop on the learner's pipeline | 16 / 11 / 68 | 134 / 117 / 269 | 339 / 269 / 396 |
| Sign-rule learner | 0 / 0 / 0 | 6 / 7 / 8 | 121 / 83 / 236 |
| Two-bank reference | 0 / 0 / 0 | 7 / 34 / 15 | 6 / 31 / 9 |

With memory silenced, every arm gets 0 twins right.

**Verdicts.** Four of six hold:
- P20.1 (existence), P20.3 (the in-view control works), P20.4 (the two-bank network ≤ 5%) and P20.5 (the sign rule ≤ 0.60 of the reference) hold.
- P20.2 misses: hiding costs the reference 36.0, 22.3 and 14.3 points, and the mark was 15 in every seed.
- P20.6 misses: backprop on the learner's pipeline reaches 0.53, 0.41 and 0.68 of the reference's twin item accuracy, and the mark was 0.85.

**Reading.**
- **The window reference composes from memory**, at a cost of 14–36 points.
- **Backprop on the learner's pipeline does so weakly.** It reaches the 20% mark on one seed only. It also trails the reference with the operands in view.
- **P20.4 and P20.5 hold, but neither arm passes the in-view control, so neither says anything about memory.**
  - The two-bank network fails the untaught plus-minus pairs even in view. On taught pairs it still uses hidden operands: 133–166 of 300 whole answers.
  - The sign-rule learner fails the composition step with these prompts. The coordinator's caveat, filed before the result, applies: its rule credits only the memory cell through time, so the rule and that defect cannot be separated.

**Run.**
- 73,827 CPU-s over twelve completed lives, on a four-core 2.10 GHz cloud Xeon. The forecast was about 5% high.
- A container restart killed four lives partway. They were relaunched from their births, and their partial receipts are kept as `*_killed.json`.
- No finished life was rerun, and no seed was changed.

Development result on sealed panels, evidence level E2. No learning-core improvement is claimed. Decisions remain Mr. Zhou's.

Sources: `E20A_REPORT.md`, `E20A_PREDICTIONS.md`, `E20A_STAGE2_FORECAST.md`, `exp45/E20A_TABLES.md`, the twelve receipts and their logs.
