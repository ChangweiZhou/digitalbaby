# Task for Codex: E9, the price of persistence (compute job on the existing harness)

**Date:** 7 November 2026. **Branch:** commit on `codex/minifly-consolidated-20261023`, no new branch. **Folder:** `MOD97_LOCAL_RULE_20261106/` (the harness is there: `exp45.py`, `exp8_life.py`, `exp7_decon.py`, `summarize_exp45.py`). **Purpose:** three compute-heavy, fully specified sweeps that extend E8 and E7-b, so the design work (diagonal cell, learned store, ring) can proceed in parallel. No new mechanisms, no new tasks, no memos.

## What exists

`exp8_life.py` runs the persistent-life demonstration: development on four operations mod 89, then eight operations in sequence by the sign rule with an episodic store and self-generated replay, then a composition stage. Its result is in `E45_REPORT.md` (section E8) and `exp45/SUMMARY_E8.md`: all eight operations held at 75–98% after the eighth stage; the composition stage costs the carry-type operations 20–35 points. Two settings are hard-coded and are the subject of this job: the replay share is half of every batch, and the schedule is blocked (one operation per stage). `exp7_decon.py` runs the de-confounded composition test; its E7-b result is that the local rule reaches 52–68% held mean at width 512 and is still rising.

## Part A: retention as a function of replay share

Add a parameter to `exp8_life.py` for the replay share of each batch (currently `X.BATCH // 2`), applied in every stage including the composition stage. Run arm C with shares **1/8, 1/4, 1/2** (the existing 1/2 result may be reused if the receipts match the lock) and arm A (share 0, existing). Three seeds, hidden 512, everything else unchanged. Report, after the eighth stage and after the composition stage, every operation's sealed score and the held composition pairs, in the format of `SUMMARY_E8.md`, plus one table: retention (mean of the eight operations) and held-pair composition against replay share.

Prediction to test: a 1/4 share retains the eight operations within 5 points of the 1/2 share after the eighth stage and loses less on the carry-type operations in the composition stage (because the composition items get a larger share); a 1/8 share loses more than 15 points on the earliest operations.

## Part B: blocked versus interleaved schedule

Add a schedule option to `exp8_life.py`: **interleaved**, where the eight operations' taught items are learned in one stage of 8 × 6,144 updates with every batch drawn from all eight, followed by the composition stage as before. Run it with the sign rule, with and without replay (replay is of the same items, so the no-replay interleaved arm is the control that says whether replay matters once presentation is interleaved). Three seeds, hidden 512. Report the same tables.

Prediction to test: interleaved presentation without replay holds all eight operations at or above the blocked-with-replay result after the operation stages, and the composition stage then costs the carry-type operations less than 10 points.

## Part C: the remaining knob on E7

Run `exp7_decon.py` arm SCR (plain scratchpad, sign rule) at **width 512 and 16,384 updates**, three seeds (`EXP45_HIDDEN=512 E7_DOSE=16384`). The width-512 births already exist as `exp45/e7_dev_h512_*.pt` if you reuse the local workspace; otherwise rebuild them with `EXP45_HIDDEN=512 python3 exp7_decon.py dev SEED`. Report held-pair mean, each held pair, within-type, and whether the curve is still rising at the last checkpoint.

Prediction to test: held-pair mean at or above 70% in every seed.

## Rules

- Write `E9_PREDICTIONS.md` with the three predictions above and the exact commands before the first run. Score sealed panels at the same checkpoints the harness already uses; add no new panels.
- Keep the shortcut ladder and the backprop ceiling that the harness already produces. Do not change the tasks, the splits, the store, the loss or the optimizer settings.
- Parameters you add must default to the current behaviour, so the existing E8 and E7 receipts are reproduced bit for bit by the default path; check that once and record the hashes.
- Deliver `E9_REPORT.md` with the tables, the per-seed receipts under `exp45/`, and the two edited scripts. Three seeds everywhere; report every seed.
- Do not create a branch. Do not write a design memo. Do not add mechanisms. If a run fails, report the failure and stop that part; do not tune.

## Budget

About 30 runs at hidden 512, 30–90 minutes each on a single thread; roughly two laptop-days or one Colab day with six workers. Profile one run first and state the forecast in the predictions file.
