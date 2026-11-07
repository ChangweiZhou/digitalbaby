# Task for Codex: E12, replay mechanics (compute job on the E9 harness)

**Date:** 7 November 2026. **Branch:** commit on `codex/minifly-consolidated-20261023`, no new branch. **Folder:** `MOD97_LOCAL_RULE_20261106/` (your parameterized `exp8_life.py` from E9, with `E8_REPLAY_SHARE` and `E8_SCHEDULE`). **Purpose:** four fully specified sweeps that answer the question E9 left open and test three knobs that Raviv et al. (*Science* 394, eaeh9311, 5 November 2026; file `raviv-science-2026.pdf` in the project root) fitted per child in a model trained day by day on a lived stream with nightly replay. No new mechanisms, no new tasks, no memos.

## What E9 established and left open

Replay share buys retention roughly linearly (1/2: 89–90% mean of eight operations before composition; 1/4: 78–81; 1/8: 55–58; none: 12–13). Interleaving the eight operations holds them at 91–92% while they stay in the stream, but the composition stage without replay drives every standalone response to exactly 0%. Your report says, correctly, that the zeros "do not by themselves establish that every internal arithmetic representation was erased." Part A measures that. Raviv et al. found replay of each day's events about 25–30 times sufficient with little gain beyond, a per-child "error-accumulation window" (minutes of input per weight update, 4 min to 1 h, shorter better), and recency-decayed resampling of past input. Parts B–D test the analogues.

## Part A: do erased responses survive inside? (savings test)

Add an option to `exp8_life.py` to save the learner's weights at the end of each stage (default off). Rerun the **interleaved, no-replay** arm (three seeds, `E8_SCHEDULE=interleaved E8_REPLAY_SHARE=0 python exp8_life.py A S`) with saving on; the scientific receipt must match your E9 receipt for that arm (check and record). Then, from the post-composition weights (all eight operations at 0%), run a **reminder**: `R` updates of the sign rule on the eight operations' taught items, interleaved, no replay, with `R` ∈ {64, 256, 1024}, scoring all eight sealed panels after each reminder. Compare with the same reminder applied to the **birth** weights (`e8_dev_S.pt`, which never saw the eight operations) and to the weights saved **after the interleaved stage** (before composition, 91–92%).

Report, per seed and per `R`: the mean of the eight operations after the reminder from (i) the erased learner, (ii) the birth, (iii) the pre-composition learner. Savings = (i) minus (ii).

Prediction to test: at `R` = 256 the erased learner recovers to ≥ 60% while the birth reaches ≤ 20% (large savings: the operations survive inside and only the output mapping was overwritten). If the erased learner is within 10 points of the birth at every `R`, the erasure is real.

## Part B: error-accumulation window (batch size at fixed exposure)

Add a batch-size parameter (default 64). Run the blocked arm C at share 1/2 with batch **16, 64, 256**, keeping the **number of presented items per stage fixed** (6,144 × 64 = 393,216 items per operation stage, so 24,576 / 6,144 / 1,536 updates; the composition stage scaled the same way). Three seeds, hidden 512. Report the E8 tables (every operation after the eighth stage and after composition; held composition pairs) and one table of retention and composition against batch size, plus CPU seconds per run.

Prediction to test: batch 16 retains the eight operations within 3 points of batch 64 and composes better (more updates per item), batch 256 loses more than 10 points on the carry-type operations. If batch 16 is also within 3 points on composition, the window does not matter at this scale.

## Part C: recency-weighted replay

Add a replay-sampling option: **uniform** (current) or **recency**, where a buffered prompt from stage `j`, replayed during stage `i`, is drawn with probability proportional to `0.5^(i − j)` (half the weight per stage of age). Run blocked arm C at shares 1/2 and 1/4 with recency sampling, three seeds; the uniform results exist from E8/E9.

Prediction to test: recency sampling loses more than 10 points on the earliest four operations relative to uniform at the same share (a sequence of distinct skills is not a stationary stream) and gains less than 5 points on the latest four. If recency is within 5 points everywhere, replay order does not matter at this scale.

## Part D: replay saturation

Fix share 1/2, blocked, and vary the **buffer size per stage**: 128, 1,024 (current), 8,192 prompts (the stage's taught set is about 2,350 items; at 8,192 nearly every presented item is buffered). Three seeds. Report retention against buffer size.

Prediction to test: 128 loses more than 10 points on retention relative to 1,024; 8,192 is within 3 points of 1,024 (saturation, as Raviv et al.'s 25–30 replays).

## Rules

- Write `E12_PREDICTIONS.md` with the four predictions and the exact commands before the first run; profile one run first and state the forecast.
- Every parameter you add defaults to the current behaviour; reproduce one default C life and record the hashes, as in E9.
- Keep the shortcut ladder, the three seeds, the sealed panels and the checkpoints the harness already uses. Do not change the tasks, splits, store, loss or optimizer settings. Do not add mechanisms. If a part fails, report the failure and stop that part; do not tune.
- Deliver `E12_REPORT.md` with the tables, the per-seed receipts under `exp45/`, and the edited script. Report every seed.

## Budget

About 40 lives at hidden 512 (Part B's batch-16 runs are the long ones, roughly four times the updates), plus 27 short reminder runs. Roughly one Colab day with six workers.

## Still open from earlier

The path-integration follow-up brief (`CODEX_TASK_CX_PATH_INTEGRATION_FOLLOWUP_20261107.md`: decompose the return residual, compass on a 180° turn, clamped-heading ceiling, then refit) has not been delivered. E12 does not replace it; do whichever first.
