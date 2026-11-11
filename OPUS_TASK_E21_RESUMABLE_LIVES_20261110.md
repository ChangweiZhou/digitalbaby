# One cloud job for a Claude Opus 5.5 worker: E21, lives and births that survive a restart (10 November 2026, 20:30 EST)

> **Status under the process rules of 10 November (review 9.82).** Not yet audited by Codex. Send this brief to Codex first; the worker starts after that audit is answered. **Commits:** one commit at completion on the branch your session is given; nothing partial on the default branch.

**Repository:** `ChangweiZhou/digitalbaby`, default branch `codex/minifly-consolidated-20261023`. **Folder:** `MOD97_LOCAL_RULE_20261106/`. Python 3, PyTorch on CPU, NumPy. Clone shallow and sparse if you can (`--depth 1`, only this folder and the files named here): the repository is large. For one worker. This is an engineering job. It runs no experiment and makes no claim about learning.

**Branch rule, different from other jobs because this one edits the harness.** Work on the branch your session is given. Do not push to the default branch. The coordinating session will rerun your tests on the owner's machine and merge. Tell it your branch and final commit in your last message. Do not create any other branch.

## Why

On 10 November a cloud container was restarted with twelve lives thirteen minutes from finishing (review 9.68). The harness cannot continue a life, so six hours of twelve lives ended one checkpoint short and the job's predictions could not be judged. Containers restart without warning, a life takes one to two hours, and a job is nine to twelve lives. Until a life can resume, no long job is safe on a cloud container.

## What to build

In `exp16_integrated.py`, for `life()` and for `develop17()`:

1. **A checkpoint at every scored stage.** When a stage's receipt is written, also write `<receipt name>.ckpt` beside it, by temporary file and rename, keeping the previous one as `.ckpt.prev`. It holds: the model's state, the optimizer's state, the stage index, the receipt so far, a digest of the run's effective configuration (every `E16_*` and `E11_*` setting in force, the birth's hash, the panel hashes), and a digest of the data pipeline at that moment (the reservoir or buffer contents in order, the admission counters, the state of every `random.Random` the life owns, how many items each stream has yielded, and the score-time store's keys and counts where there is one). Checkpoint files are never committed; add the patterns to `.gitignore`.
2. **Resume by replaying the data with the learner switched off.** With `E16_RESUME=1`, if a checkpoint exists for this exact receipt name: check the configuration digest and stop if it differs; then run the stage loop for stages 1 to k exactly as written, building every batch, doing every admission and every stage-end bookkeeping step, and writing to the score-time store where the arm has one, but **skipping the update and the evaluation**; take those stages' records from the saved receipt; then load the model and optimizer, compare the pipeline digest with the checkpoint's, stop if it differs, and continue from stage k + 1 as an ordinary life. The streams are Python generators and cannot be saved; replaying them is exact because nothing in the data pipeline depends on the weights. Confirm that for yourself: if you find a path where it does, say so and stop.
3. **Births the same way.** `develop17` has one stream and a curve row every 4,096 updates; checkpoint at each row and resume by drawing and discarding the batches already used.
4. **A resumed receipt says so.** Add a list `resumed` to the receipt (stage resumed from, host, time, CPU spent replaying). CPU fields continue from the saved values; replay CPU is counted separately.
5. **Nothing else changes.** Without `E16_RESUME=1` a life or birth behaves exactly as it does at the default branch's head, apart from writing checkpoint files. No setting, tag, file name, score or loss may change. Do not touch `exp11_cell.py`, `exp45.py`, `arch17.py` or any receipt.

Useful facts about `life()`, to be checked against the source: the data pipeline's state lives in `rng`, `drng`, the generators `cur`, `joint`, `seen_streams` and `fstreams`, and in `buffer`, `fbuf`, `fset`, `fseen`, `presented`, `nseen` and the per-stage exposure counts; the score-time store is written inside the update call for arms that have one, under the same mask the loss uses; evaluation and the first-update census draw no random numbers.

## Acceptance tests: `e21_resume_check.py`, exit code nonzero on any failure, results in `exp45/E21_RESUME_CHECK.json`

Command shapes to cover, each at `E11_STAGE_UPDATES=16` (and births at `E11_DEV_UPDATES=8192`), seed 73129 unless a birth is missing:

| Shape | Command |
|---|---|
| window learner, drift, online reservoir, store | `E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py T 73129` |
| window reference, staged | `python3 exp16_integrated.py D 73129` |
| backprop on the pipeline, staged, block admission | `E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py M 73129` |
| two-bank learner, the Phase 4 rule | `E16_ARCH=bank2 E16_BOOKDECAY=off E16_REPLAY=raw E16_CAP=65536 E16_TRACEUP=all python3 exp16_integrated.py T 73129` |
| the same with every-byte loss | add `E16_LOSS=all` |
| window learner with letters in compositions | `E16_COMPGAP=6 E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py T 73129` |
| a birth by the rule with local heads | `E16_ARCH=bank2 E16_BOOKDECAY=off E16_HEADS=local E16_BIRTHRULE=T E16_TRACEUP=all python3 exp16_integrated.py dev 73129` |

1. **Unchanged when not resuming.** For each shape, the receipt from your branch equals the receipt from the default branch's head at the same settings: every stage's scores and losses identical, the final weights' hash identical. Run both on your machine.
2. **Exact resume.** For each shape: run A to the end uninterrupted. Run B, kill it with `SIGKILL` at a moment you record, restart with `E16_RESUME=1`, let it finish. A and B must agree in every stage's scores and losses, in the final weights' hash, and in the pipeline digest at every checkpoint. Do this for three kill points per shape: inside the first stage before any checkpoint (the run then starts from the birth, which is correct); inside a middle stage's updates; inside a middle stage's evaluation. For one shape also kill twice in one life.
3. **Fail-closed.** Resuming with one setting changed must stop with a message naming the setting. A truncated `.ckpt` must fall back to `.ckpt.prev` and say so; with both damaged the run must stop, not start over silently. A checkpoint whose pipeline digest is edited must stop the run at the comparison.
4. **A larger dose.** Tests 2 and 3 once more for two shapes at `E11_STAGE_UPDATES=512`, one kill each.

Record the cost: checkpoint size, seconds to write one, and replay seconds per stage at the full dose (estimate from the larger dose and say it is an estimate).

**If any test fails, stop and report the failure.** Do not loosen a test. A resume that is almost exact is not accepted: one differing score means the replay missed part of the state.

## Deliverables

The change to `exp16_integrated.py` in one commit; `e21_resume_check.py`; `exp45/E21_RESUME_CHECK.json`; `.gitignore` lines; `E21_RESUME_REPORT.md` (what state is saved, what is replayed and why that is exact, the test table with every kill point, the costs, anything that surprised you, and what is not covered); one paragraph as `E21_REVIEW_SECTION_9_83.md`. Report your machine, torch version and the default-branch commit you started from.

## What is not asked

No change to scoring, panels, replay or any experiment. No resume for `exp11_cell.py` births, `exp19_kbank.py` or the summarizers. No protection against two processes writing one receipt; say in the report that it is not covered.

**Cost.** No long run. The tests are a few dozen short lives; expect one to two hours of compute and most of the time in reading `life()` carefully. Keep a foreground command active while tests run, in case inactivity is what restarts the container.
