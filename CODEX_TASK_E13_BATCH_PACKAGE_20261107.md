# Task for Codex: E13, what the batch-256 package is, and whether it closes the composition gap

**Date:** 7 November 2026. **Branch:** commit on `codex/minifly-consolidated-20261023`, no new branch. **Folder:** `MOD97_LOCAL_RULE_20261106/` (your parameterized `exp8_life.py` with batch size, replay share, schedule, buffer, sampling; `exp7_decon.py`). **Purpose:** your E12 Part B found that batch 256 at fixed exposure lifts the sign rule's eight-operation retention from 76–80% to 91–92% and held composition from 48–55% to 69–73%, in every seed, and you noted correctly that the change alters the update count and the cumulative weight decay as well as the gradient average. E13 separates those, tests whether the package transfers to the de-confounded composition test, and asks whether it moves the backprop ceiling. Plus the path-integration follow-up, which has now been outstanding through two campaigns and is part of this delivery.

## Part A: decay, averaging, or update count

All blocked arm C, share 1/2, three seeds, hidden 512, exposure fixed at 393,216 items per operation stage as in E12 Part B. Decoupled AdamW decay per update is `lr × wd`; cumulative decay over `N` updates is `exp(−N lr wd)`: batch 64 (57,344 updates, wd 1.0) gives exp(−57); batch 256 (14,336 updates) gives exp(−14).

| Arm | Batch | wd | lr | Equalizes |
|---|---|---|---|---|
| A1 | 64 | 0.25 | .001 | cumulative decay of batch 256 at wd 1.0 |
| A2 | 256 | 4.0 | .001 | cumulative decay of batch 64 at wd 1.0 |
| A3 | 256 | 1.0 | .002 | learning-rate scaling with batch |
| A4 | 1024 | 1.0 | .001 | the trend (3,584 updates) |

Prediction to test: if cumulative decay drives E12-B, A1 matches batch 256 (retention ≥ 90%) and A2 matches batch 64 (≤ 80%). If gradient averaging drives it, A1 stays near batch 64 and A2 stays near batch 256. A4: retention within 3 points of batch 256 and composition not lower (the gain saturates) rather than a further rise. Report the E8 tables and one table of retention, final mean and held composition against arm.

## Part B: does the package close the local rule's composition gap?

`exp7_decon.py` arm SCR (plain scratchpad, sign rule), width 512, with batch 256 at **fixed exposure** equal to E9 Part C (16,384 × 64 = 1,048,576 items, so 4,096 updates at batch 256), three seeds. The E9 Part C result at batch 64 is 74.96 / 62.33 / 75.88% held-pair mean; the backprop ceiling (E7 arm SCRbp) is 84–97%. Add the batch parameter to `exp7_decon.py` with default 64 and verify default parity once.

Prediction to test: held-pair mean ≥ 80% in every seed (the gap to the ceiling at least halves). Report each held pair, within-type, the full-scratchpad exact score, and whether the curve is still rising.

## Part C: does the ceiling move?

E8 arm D (backprop on joint ground truth) at batch 256, fixed exposure, three seeds. Prediction to test: the ceiling's eight-operation mean changes by less than 2 points (the batch effect is specific to the noisy local rule, not to the task). If the ceiling also rises, the effect is about optimization of this task, not about the rule.

## Part D: the path-integration follow-up (outstanding)

`CODEX_TASK_CX_PATH_INTEGRATION_FOLLOWUP_20261107.md`, unchanged: decompose the 0.65 return residual into heading error and distance error; test the compass on a 180° turn; measure the clamped-heading ceiling for the return task; then refit the gains with the ceiling as the gate. Deliver it in this campaign; E13 is not complete without it. If something in that brief is infeasible, say exactly what and stop that part rather than deferring it again.

## Rules

- `E13_PREDICTIONS.md` with the predictions above and exact commands before the first run; profile one run and forecast.
- Every added parameter defaults to current behaviour; default-parity check recorded once per script, as in E9/E12.
- Same seeds, panels, checkpoints, shortcut ladder; no new panels, tasks, mechanisms or tuning. If a run fails, report and stop that part.
- Deliver `E13_REPORT.md`, receipts under `exp45/`, edited scripts. Report every seed. State the E12 controller failure's lesson in one line in the predictions file and validate only exact job receipts.

## Budget

12 lives in Part A (A4 at batch 1,024 is cheap in updates, heavier per update), 3 in Part B, 3 in Part C; roughly 8 CPU hours total plus Part D. One Colab day with six workers.
