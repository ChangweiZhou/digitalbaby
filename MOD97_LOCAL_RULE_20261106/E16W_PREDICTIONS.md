# E16W prospective predictions and execution scope

Written before technical or scientific E16W runs, 8 November 2026 (New York).
Source: CODEX_TASK_E16W_WINDOW_NET_CROSS_FORM_RETENTION_20261108.md at a792a56.

## Predictions (unchanged from assignment)
1. C_uni held-pair final-answer mean falls at least 15 percentage points between compose and final min73, every standing seed. A fall under 10 is the stated window-net contrast; it does not by itself isolate the cell because schedule, input windows, distractors and replay differ between harnesses.
2. C_bal at least halves that loss in every seed. Report signed loss, improvement, and the literal inequality; if uniform does not lose accuracy, the intended loss-reduction interpretation is uninformative.
3. All four mod97 operations remain at least 90% through the four later stages under both replays. Also report their mean separately so a mean cannot mask an operation failure.
4. D final held-pair mean within 3 points of its post-compose value. All operation/pair scores, full scratchpad and within are reported.

Seeds: 73129, 93683, 116089; reuse standing E8 births. No new births, panels, tasks or tuning.

## Exact scientific commands (same for each SEED)
EXP45_HIDDEN=512 E8_BATCH=64 E8_WD=0.25 E8_LR=.001 E8_SCHEDULE=late_family E8_REPLAY_SHARE=1/2 E8_BUFFER_CAP=1024 E8_REPLAY_SAMPLING=uniform E8_RUN_PREFIX=e16w_C_uni_ .venv/bin/python e14_acceleration_worker.py exp8_life.py C SEED
EXP45_HIDDEN=512 E8_BATCH=64 E8_WD=0.25 E8_LR=.001 E8_SCHEDULE=late_family E8_REPLAY_SHARE=1/2 E8_BUFFER_CAP=1024 E8_REPLAY_SAMPLING=balanced E8_RUN_PREFIX=e16w_C_bal_ .venv/bin/python e14_acceleration_worker.py exp8_life.py C SEED
EXP45_HIDDEN=512 E8_BATCH=64 E8_WD=0.3 E8_LR=.001 E8_SCHEDULE=late_family E8_REPLAY_SHARE=1/2 E8_BUFFER_CAP=1024 E8_REPLAY_SAMPLING=uniform E8_RUN_PREFIX=e16w_D_ .venv/bin/python e14_acceleration_worker.py exp8_life.py D SEED
Actual interpreter: /Users/pencilbard/Documents/second_try/.venv/bin/python. Unlisted E8 diagnostic/save/checkpoint knobs explicitly off in the job lock.

## Interpretation and implementation lock points
The enumerated schedule has NINE training stages (4 mod97 + compose + 4 mod73), not thirteen: 8*6144+8192=57344 updates, equal to standard E8. Stage0 is baseline evaluation. Score existing composition panels at every stage in the new schedule only.
Balanced means choose a BUFFER STAGE uniformly, then a prompt uniformly; fresh half of a C batch stays current. All stages contribute 1024 prompts, so marginal prompt/stage probabilities match flat uniform; replacement/correlation differs. This is not a deliberate increase of composition exposure. Do not silently change it into family-level or whole-batch balancing.
Composition training examples must be admitted to the exact store and replay buffer in late_family. Self-generated composition replay must accept six-byte digit-digit-semicolon-digit-digit-EOS answers; retain the ordinary three-byte behavior unchanged. Malformed composition answers use the syntactically valid zero scratchpad 00;00+EOS, declared before fitting. No ground-truth answers substituted for generated replay. Score held composition network-only as E8 already does.
Record actual admitted prompts, stage and SHA256, presented-prompt membership, per-stage replay draw counts and malformed replay counts. No new random draws for instrumentation.

## Technical gates and forecast
Freeze old exp8 in exp8_e16w_frozen.py. Short old/default trajectory equality includes model, Adam state, Torch and Python RNG, store, losses and receipts excluding cpu only. New-mode dry loops exercise all three arms, mixed 3/6-byte replay and balanced buffer membership. Full blocked C73129 profile must reproduce E13-A1 C73129 receipt exactly excluding cpu only before launching nine scientific lives. Stop affected part on technical/parity failure, without tuning or seed substitution.
E14S comparable lives took about 1100-1434 recorded CPU seconds and 1234-1606 wall seconds under six workers. Nine lives suggest roughly 3-4 CPU hours plus one default parity life and technical overhead, with extra scoring and six-byte replay uncertain. This replaces the brief's 1.5x schedule premise. User reports nine E16-D workers through ~13:05 EST; do not touch them, and measure one full profile under actual load before refining wall forecast. Launch the profile first; cap subsequent scientific workers at three until 13:05 EST, then six. No duplicate campaigns. Exact full profile parity is the launch gate.

Final artifacts E16W_LOCK.json, E16W_CHECKS.json, E16W_REPORT.md, raw JOB/status/receipt/log files, buffer admission records and source snapshot. Append review 9.31 while preserving concurrent 9.30. Commit on existing branch, no push without a new E16W push request.
