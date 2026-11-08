# Task for Codex: E14, the decay setting confirmed on fresh seeds, its curve, and why it hurts the sign rule

**Date:** 7 November 2026. **Branch:** commit on `codex/minifly-consolidated-20261023`, no new branch. **Folder:** `MOD97_LOCAL_RULE_20261106/` (your parameterized `exp8_life.py`, `exp7_decon.py`; `exp45.py`, `exp3_state.py` for the sign-feedback map). **Purpose:** E13 found that decoupled decay 0.25 at batch 64 lifts the sign rule's eight-operation retention in E8 from 76–80% to 94% and held composition from 48–55% to 81–85%, and that the E12 batch package lifts E7 composition at width 512 to 83–88%. Three things are missing before those settings are adopted for the next phase: confirmation on seeds and splits never used, the shape of the decay curve, and a mechanism for why decay hurts the sign rule so much more than backprop (whose ceiling at decay 0.3 holds everything). No new mechanisms, tasks or panels beyond the diagnostic in Part C.

## Part A: independent confirmation on fresh seeds

Three new seeds, drawn and recorded in `E14_PREDICTIONS.md` before any run: use 200003, 300007, 400009 unless they collide with any seed in the folder (check and record). New births for those seeds (`exp8_life.py dev`, `exp7_decon.py dev` at width 512) with the splits those seeds produce. Then:

- E8 arm C, blocked, share 1/2, batch 64, decay 0.25, with the composition stage (the E13-A1 setting).
- E8 arm D (backprop ceiling) at its standard setting, same seeds.
- E7 arm SCR, width 512, batch 256 at fixed exposure 1,048,576 items (the E13 Part B setting) **and** batch 64 with decay 0.25 at 16,384 updates (the E13-A1 analogue on composition, not yet run anywhere).
- E7 arm SCRbp (backprop) at width 512, 16,384 updates, same seeds, so the comparator is matched in width and dose for once.

Prediction to test: on the fresh seeds the decay-0.25 E8 life holds the eight operations at ≥ 90% after composition and composes held pairs at ≥ 75%; E7 under either package reaches ≥ 80% held-pair mean in every fresh seed; the matched backprop comparator is within 10 points of the local rule on E7.

## Part B: the decay curve

E8 arm C and E7 arm SCR (width 512, 16,384 updates, batch 64) at decay **0.1, 0.25, 0.5** on the three standing seeds (0.25 exists for E8 from E13; 1.0 exists for both). Report retention, final mean and held composition against decay, one table per harness.

Prediction to test: 0.1 and 0.25 within 3 points of each other on every endpoint (a plateau), 0.5 at least 5 points below 0.25 on E8 composition. If 0.1 is clearly better than 0.25, the optimum has not been found and the next sweep goes lower.

## Part C: why decay hurts the sign rule more than backprop (diagnostic, no new lives)

Add an optional logging hook (default off) to `exp8_life.py` that, every 512 updates during a life, computes on the current batch, for each linear layer: (i) the cosine between the parameter update the arm actually applies (the sign-feedback gradient for arm C; the true gradient for arm D) and the **true** gradient of the same loss at the same weights (one extra backward with the standard linear map, discarded afterwards); (ii) the ratio of the Adam step norm to the weight norm; (iii) the weight norm. Run it on arm C at decay 1.0 and at decay 0.25, and on arm D at decay 0.3, three standing seeds. The logged quantities must not feed the optimizer; verify once that a logged life reproduces the unlogged receipt exactly.

Report the three quantities over the life per layer, and one summary table: mean cosine per layer per arm.

Prediction to test: the sign-rule update's cosine to the true gradient is below 0.5 in the hidden layers and above 0.8 in the output layer; at decay 1.0 the weight norm of the hidden layers falls through the life while under decay 0.25 it is stable; the Adam-step-to-weight ratio at decay 1.0 is at least twice that at 0.25 in the hidden layers. Together that would say: the sign rule's hidden-layer updates are poorly aligned, so a given decay removes more signal per unit of learning than it does for backprop. If the cosines are high everywhere, that account is wrong and the asymmetry needs another explanation.

## Rules

- `E14_PREDICTIONS.md` first: predictions, the fresh seeds, exact commands, a profile and forecast; the E12 controller lesson in one line; validate only exact job receipts.
- Added parameters default to current behaviour; one default-parity check per edited script, hashes recorded.
- Keep the shortcut ladder, panels and checkpoints; no tuning, no new mechanisms, no new panels (Part C logs, it does not change the learner). If a part fails, report and stop it.
- Deliver `E14_REPORT.md`, receipts under `exp45/`, edited scripts. Report every seed, fresh and standing, separately.

## Budget

Part A: 6 births plus 15 lives; Part B: 12 lives (E8 at 0.1 and 0.5 ×3; E7 at 0.1, 0.25, 0.5 ×3) minus what exists; Part C: 9 logged lives. About 12 CPU hours; one Colab day with six workers.
