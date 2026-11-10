# One cloud job for a Claude Opus 5.5 worker: E16-M4, the replay share on new seeds and where the curve turns (10 November 2026, 10:45 EST)

**Repository:** `ChangweiZhou/digitalbaby`. **Branch to work from:** `codex/minifly-consolidated-20261023` (run `git fetch origin codex/minifly-consolidated-20261023 && git checkout codex/minifly-consolidated-20261023`; confirm `git log -1` shows a commit of 10 November 2026, 10:45 EST or later, and that `OPUS_TASK_E16M4_CLOUD_JOB_20261110.md` is present). **Folder:** `MOD97_LOCAL_RULE_20261106/`. **Needs:** Python 3 with PyTorch on CPU and NumPy; install only those if missing. The job splits into two halves (a and b) if two workers are available; say in your first message which you are running. A half is a complete job with its own predictions file, report and review paragraph.

**Branch rule.** If your session can push to `codex/minifly-consolidated-20261023`, commit there and push after every checkpoint batch (partial receipts) and at the end; pull with `--rebase` before each push and never force. Otherwise commit to the branch your session is given, push there, and put the branch name and final commit in your last message; it will be merged by the coordinating session. Do not create any other branch. Keep partial-receipt commits few (one every hour or two is enough; E16-M3 made 61).

**Rules.** Launch every life detached from the tool command (`setsid nohup env ... python3 exp16_integrated.py ... > <log> 2>&1 < /dev/null &` from a short command that returns at once) and poll the logs with short commands; one line prints per finished checkpoint. Write logs outside the repository folder and copy them into `exp45/logs/` at the end. **Dry-run each new command shape once before launch** with `E11_STAGE_UPDATES=4` (the receipt gets a `_dry4` tag): it must run its seven stages without error; move the dry receipts to `exp45/archive_e16m4_dry/cloud_<date>/` (the coordinating session's dry runs of the two new shares are one folder up). Write and commit your predictions file before any life starts and do not change it afterwards. Add files; edit none of the harness, receipts, reports, review or audit scripts (copy a script under a new name if you need to change its seed list). No reruns of a finished life, no tuning, no change of seeds or shares; a miss is a result. Report every seed separately, holds or misses with the numbers; forecast the job from your dry run and the per-life costs below before launch, then report forecast against measured. **Run identity:** machine (`uname -a`, core count, `python3 -c "import torch; print(torch.__version__)"`), the harness commit, and the CPU-seconds of every life (the last log line prints `cpu <n>s`). Report anything that went wrong plainly.

**What the harness is.** `exp16_integrated.py` runs one life of a byte-level learner (arm `T`: a network with a diagonal memory cell and a ten-byte window, every layer trained in life by a sign-symmetric rule with an exact eligibility trace; nothing is frozen) on a stream of three families of modular-arithmetic items (mod-97 operations with distractors, twelve taught operator pairs written as scratchpads, mod-73 operations). Under `E16_STREAM=drift` each item's family is drawn from weights that move smoothly through the life, scored at six checkpoints. The replay memory is a reservoir of 65,536 raw byte strings (`E16_REPLAY=raw E16_CAP=65536`, admitted online); `E16_RSHARE=<fraction>` is the share of every batch of 64 drawn verbatim from it (default 0.5; other values tag the receipt `_rs<percent>`). The comparator `D_drift` is a backprop reference trained on the joint stream of the families admitted so far: a ceiling, not a matched control. Scores are exact answers on sealed panels; "ratio to the twin" is the learner's family mean divided by the reference's; compositions are the mean final-answer accuracy over the four held pairs (`+-`, `-+`, `><`, `<>`), operations the family mean at k = 12. The family gate as pre-registered is 0.80 of the reference on every family. "Hard pairs" is the mean of `+-` and `-+` by final answer, the two pairs whose shortcut ceilings are about 1%; the hard-pair gate is 0.80 of the reference on that mean. Report the hard pairs apart from the family means throughout.

**Standing notes.** Mr. Xu's working answer of 10 November (review 9.67): feedback relayed layer by layer is allowed, so the sign rule is inside the line for design purposes; the decision remains Mr. Zhou's. Evidence level of the E16 line is E2 (sealed held-out panels, operand-disjoint from taught triples; no reuse on never-reinforced inputs is claimed). This job is a diagnostic of a setting; it claims no learning-core improvement.

---

## Why

E16-M3 (review 9.64, `E16M3_REPORT.md`) found that the replay share acts on the rule in the direction opposite to the one predicted: more verbatim replay retains the hard pairs. On the three seeds it ran, the rule's hard pairs at shares 0.25 / 0.5 / 0.75 were 43.7 / 46.3 / 73.2 (400009), 23.3 / 44.8 / 60.2 (600013) and 59.2 / 67.7 / 75.7 (300007), and at 0.75 the failing seed 400009 passed both the family gate and the hard-pair gate. The share works through retention: the hard-pair peak at checkpoint 4 hardly moves with the share, the end of the life does. E16-F7 (review 9.62) found two seeds, 1200007 and 1300021, that pass the family gate while losing 15 and 30 points of `+-` between checkpoints 4 and 6, and only 1 of its 6 seeds passes the hard-pair gate at share 0.5.

That is one life per seed and share, on seeds the effect was found on. Two things are untested: whether share 0.75 helps seeds it was not found on, and where the curve turns, since fewer fresh items per batch must eventually cost acquisition.

## Runs: twelve sign-rule lives on the drifting stream

**Half a, share 0.75 on the six E16-F7 seeds** (births, share-0.5 lives and references are in the repository from E16-F7): **1100009, 1200007, 1300021, 1400017, 1500007, 1600033.**

```
E16_RSHARE=0.75 E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py T <seed>
```

**Half b, shares 0.875 and 0.9375 on the three E16-M3 seeds** (their lives at 0, 0.25, 0.5 and 0.75 and their references are in the repository): **400009, 600013, 300007.**

```
E16_RSHARE=0.875  E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py T <seed>
E16_RSHARE=0.9375 E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py T <seed>
```

Receipts: `exp45/e16_life_T_raw65536_drift_online_rs75_<seed>.json` (half a), `..._rs88_<seed>.json` and `..._rs94_<seed>.json` (half b), each with its `.pt` (`git add -f` the `.pt` files); logs `e16m4_T_rs<percent>_<seed>.log`. Shares 0.875 and 0.9375 are 56 and 60 of the 64 items in a batch; the tags round to `_rs88` and `_rs94`, and the receipt's `settings.rshare` holds the exact value. Before launch, check and record that none of the twelve receipts exists. Comparators in the repository: `exp45/e16_life_T_raw65536_drift_online_<seed>.json` (share 0.5), `exp45/e16_life_T_raw65536_drift_online_rs{0,25,75}_<seed>.json` for the three E16-M3 seeds, and `exp45/e16_life_D_drift_<seed>.json`. `summarize_exp16.py` lists the new arms; `e16m3_tables.py` and `e16f7_gate.py` are the scripts to copy and extend for the share curves and the hard-pair gate.

## Predictions (copy into `E16M4_PREDICTIONS.md` and commit before launch)

All at the last checkpoint unless a checkpoint is named.

Half a:
- **P16M4.1 (family gate)** all six seeds pass the family gate at share 0.75, as they do at 0.5.
- **P16M4.2 (hard pairs)** at share 0.75 the hard pairs are at least 5 points above the same seed's share-0.5 life in at least five of the six seeds, and at least three of the six reach the hard-pair gate (one does at share 0.5).
- **P16M4.3 (the late loss)** in 1200007 and 1300021 the fall of `+-` from checkpoint 4 to checkpoint 6 (15.0 and 29.7 points at share 0.5) is at most half as large at share 0.75, in both seeds.

Half b:
- **P16M4.4 (still rising at 0.875)** in at least two of the three seeds the hard pairs at share 0.875 are at or above the share-0.75 value minus 3.
- **P16M4.5 (the curve turns by 0.9375)** at share 0.9375 at least two of the three seeds fail the family gate on some family, and in at least two the mod-73 mean is at least 5 points below its share-0.75 value: with four fresh items in a batch of 64 the last family is not acquired in time.

Reading, written before the result. P16M4.1 and P16M4.2 holding make share 0.75 the candidate default for this learner, to be confirmed on all eighteen drift seeds before any gate is quoted at it; it would remain a setting found on three seeds and tested on six. P16M4.2 missing says E16-M3's gain belonged to its three seeds, two of them chosen as failures, and the share is not a general handle. P16M4.3 says whether the late loss of 9.62 is the retention the share buys. P16M4.4 and P16M4.5 together place the turn between 0.875 and 0.9375; P16M4.5 missing says replay of recently admitted items carries acquisition too and the turn is higher still, which would be worth knowing for the cost per event. State which case obtained.

## Report

Per seed and share, at every checkpoint: mod-97 add, sub, max, min at k = 12 and the four held pairs by final answer, beside the share-0.5 life and the reference. Half a: a table of the six seeds at 0.5 and 0.75 with family ratios, hard pairs, hard-pair ratio and both gates, and the checkpoint-4-to-6 change of `+-` and `-+`. Half b: the share-curve table of E16-M3 extended to 0.875 and 0.9375 (rows 0 / 0.25 / 0.5 / 0.75 / 0.875 / 0.9375; columns mod-97 mean, hard pairs, composition ratio, mod-73 mean, the hard-pair peak over checkpoints 3–4). The five verdicts; run identity and CPU-seconds per life; forecast against measured; what went wrong. Deliverables: `E16M4_PREDICTIONS.md`, `E16M4_REPORT.md`, the twelve receipts and weights, logs, `exp45/SUMMARY_E16M4.md`, and one paragraph as `E16M4_REVIEW_SECTION_9_68.md` (half b alone: `9_69`).

**Cost.** Sign-rule lives measured 5,132–5,441 CPU-s on the E16-M3 container and 5,329–5,402 on E16-F7's (4 cores, 2.10 GHz). Twelve lives are about 65,000 CPU-s: about 4.6 hours of wall on four cores if run whole, about 2.3 hours per half. Forecast from your own dry run.
