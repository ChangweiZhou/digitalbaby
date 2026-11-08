# Task for Codex: E14-S, the missing decay control, the composition gate on nine seeds, and the curve of the failing pair

**Date:** 8 November 2026. **Branch:** commit on `codex/minifly-consolidated-20261023`, no new branch. **Folder:** `MOD97_LOCAL_RULE_20261106/` (your `exp8_life.py`; E14 receipts under `exp45/`). **Purpose:** E14 confirmed decay 0.25 on fresh seeds and showed a flat curve from 0.1 to 0.25, and it refuted the alignment account of why decay hurts the sign rule. Three things remain before the composition gate and the decay story are closed. No new mechanisms, tasks or panels; one logging knob.

## Part A: does the decay asymmetry exist at all?

The backprop ceiling (arm D) has only ever run at decay 0.3. The claim that "decay hurts the sign rule more than backprop" has therefore never been measured. Run arm D at decay 1.0 on the three standing seeds (`E8_WD=1.0`, everything else at the arm D default, with the composition stage). Report retention after the eighth stage and after composition, and held composition, beside arm D at 0.3 (existing) and arm C at 1.0 and 0.25 (existing).

Prediction to test: backprop at decay 1.0 also loses retention, with final mean after composition at or below 85% in every seed (arm D at 0.3: 94–96%). If it holds at or above 92%, the asymmetry is real and needs an explanation other than alignment; if it falls, the asymmetry was a mismatched setting and the Part C question of E14 is closed.

## Part B: the composition gate on nine seeds

Three more fresh seeds, drawn and recorded before any run: 500011, 600013, 700027 unless they collide with any seed in the folder (check and record). New E8 births for them (`exp8_life.py dev`), then arm C at the adopted settings (blocked, share 1/2, batch 64, decay 0.25, composition stage) and arm D at its standard setting. Together with E13-A1 (three standing seeds) and E14-A (three fresh seeds) this puts the composition gate on nine seeds.

Prediction to test: at most one of the three new seeds has a held pair below 50% final-answer accuracy; across all nine seeds at decay 0.25, held composition mean is at or above 75% in at least seven. Report every held pair of every seed, not only the mean.

## Part C: the curve of the failing pair

Add a logging knob to `exp8_life.py`, default off (`E8_COMPOSE_CHECKPOINTS=N`): during the composition stage, every N updates, score the four held-pair panels and the within panel and append them under a new key in the receipt. Scoring is under `no_grad` and must not touch the training RNG; verify once that a logged life reproduces the unlogged receipt exactly apart from the new key and `cpu`. Then rerun E14's `A_C_400009` life with `N = 1024` (its receipt must reproduce `e14_A_C_e8_life_C_400009.json` exactly apart from the new key and timing), and run the three Part B arm C lives with the same knob on.

Prediction to test: in seed 400009 the add-then-subtract pair never rises above 40% at any checkpoint of the composition stage (never learned, as opposed to learned and lost), while subtract-then-add rises above 60% within the first 2,048 updates.

## Rules

- `E14S_PREDICTIONS.md` first: the three predictions, the fresh seeds, exact commands, a forecast from the E14 measured pace under the actual process load; validate only exact job receipts.
- The added knob defaults to current behaviour; one default-parity check, hashes recorded.
- Keep the shortcut ladder, panels and checkpoints; no tuning; if a part fails, report and stop it.
- Deliver `E14S_REPORT.md`, receipts under `exp45/`, the edited script, and one paragraph in `MINIFLY_PAST_WORK_REVIEW_FOR_MR_ZHOU_20261106.md` as section 9.28 (9.27 is the E14 assessment). Report every seed separately.

## Budget

Three arm D lives, three births, six lives, one rerun: 13 jobs, about 7 CPU-hours; under an hour and a half of wall time with six workers at E14's measured pace. The inference acceleration you qualified may be used for every job here; record that it was.
