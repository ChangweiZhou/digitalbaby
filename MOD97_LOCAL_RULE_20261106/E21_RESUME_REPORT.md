# E21: lives and births that survive a restart (report, 11 November 2026)

Brief: `OPUS_TASK_E21_RESUMABLE_LIVES_20261110.md`. An engineering job: no experiment, no claim about learning.

**Provenance.** Started from `codex/minifly-consolidated-20261023` at `40cb1c208adec3f39351356e52d3f69eba4ade17` (the head when the job started; it is one commit after the brief's `5e0523d6`, adding E22 files, and does not touch `exp16_integrated.py`). Work branch: `claude/e21-resumable-lives`. Machine: cloud container, Linux 6.18, 4 vCPU Intel Xeon @ 2.10 GHz, Python 3.13.18, torch 2.15.0+cpu, one torch thread per run (as the harness sets). The owner said that the Codex audit of the brief had been done. Its text is not in the repository, so I worked from the brief as written. The audit cannot have shaped this work beyond the brief.

**Files.** `exp16_integrated.py` (the change), `e21_resume_check.py` (the acceptance tests), `exp45/E21_RESUME_CHECK.json` (their results), four lines in `.gitignore`, this report, and `E21_REVIEW_SECTION_9_83.md`. No other module, receipt or birth file was touched.

## What is saved

At the end of every scored stage of `life()`, after the stage's receipt and every stage-end admission, the harness writes `<receipt stem>.ckpt` beside the receipt (for example `exp45/e16_life_T_raw65536_drift_online_73129.ckpt`). `develop17()` does the same at every curve row, every 4,096 updates and at the end (`exp45/<birth stem>.ckpt`). Each checkpoint holds:

* the model's `state_dict`, the optimizer's `state_dict` (AdamW moments and step counts) and torch's RNG state (no code after `new_net` draws from torch's RNG, but saving it costs nothing);
* the stage index (life) or update count (birth);
* the receipt so far (every stage record, or the curve and the first-update credit census);
* the effective configuration (below), compared key by key on resume;
* the pipeline digest (below);
* the CPU counters at that point (elapsed, and scoring so far), and the list of earlier resumes.

The write goes to `.ckpt.tmp`, is flushed with fsync, and is then renamed. Before the rename the previous `.ckpt` becomes `.ckpt.prev`. The file carries a magic tag and the sha256 of its own body, so truncation or damage is detected on read rather than half-loaded. Each checkpoint also appends one line to `.ckpt.log` with the stage, the digest, the size and the seconds to write. Replayed stages append there as well, which is what the tests compare. A run started without `E16_RESUME=1` first removes any checkpoint files of the same name, so a stale checkpoint from an earlier life can never be picked up. Checkpoints are left in place when a life completes. All four patterns are in `.gitignore`.

**Configuration digest.** The effective value of every setting the harness reads: `E11_WD/REPLAY/SCHEDULE/TRACE/GATES/STAGE_UPDATES/DEV_UPDATES`, `EXP45_WINDOW/HIDDEN`, and every `E16_*` setting (`REPLAY, CAP, STREAM, ADMIT, LOSS, ARCH, STORE, FREEZE, FEEDBACK, TRACEUP, HEADS, PROJFB, FIXEDCODES, OUTTAG, COMPGAP, BIRTHRULE, BOOKDECAY, RSHARE, LEGACY_BIRTH`). Any other `E11_`/`E16_`/`EXP45_` variable present in the environment is included as given. `E16_RESUME` is excluded, since it is not a setting of the run. Also included: the batch and block sizes, the arm, the seed, the receipt name, the birth's file name and sha256, the panel hash (`panel_sha256`, as in the receipt), and the memory panels' hash. For a birth: its configuration, stem, seed and `kmax`, plus the hash of its sealed panels. Values are compared, not names, so an unset variable and the same variable set to its default are the same configuration.

**Pipeline digest** (life). One sha256 per component and one over all of them, so a mismatch names the component: the state of `rng` and `drng`; for every stream (`cur`, `joint`, each of `seen_streams`, each of `fstreams`), how many items it has yielded, plus its own `Random` state, shuffled order and place in that order; `buffer`, `fbuf` (in order), `fset`, `fseen`, `nseen`; the stage's `presented` items and its exposure counts; and the score-time store's keys in insertion order with their counts. For a birth: the stream's yield count and its `Random`, order and place.

## What is replayed, and why that is exact

With `E16_RESUME=1`, the life builds its net, birth, optimizer, store and panels as usual, reads the newest readable checkpoint, and stops if any configuration value differs. It then runs the stage loop for stages 1 to k as written: every batch built, every admission (online and at stage end), every exposure count, every store write under the same mask the loss would use (`store_write_only`: the answer mask, or every real byte with `E16_LOSS=all`), and every stage-end bookkeeping step. It skips only the update and the evaluation, and takes those stages' records from the saved receipt. At stage k it compares the pipeline digest with the checkpoint's and stops if any component differs. It also compares the recomputed stage-end bookkeeping (reservoir counts, buffer sizes, exposure) with the saved records. Only then does it load the weights, optimizer and torch RNG, set the CPU counters to the saved values, and continue from stage k + 1 as an ordinary life. A birth draws and discards the batches of the updates already done, compares the stream digest, loads, and continues.

This is exact only if nothing in the data pipeline reads the weights. I checked that in the source rather than taking it from the brief. Batches in `life()` come from `rng`, `drng`, the generators (`entry_stream`, `joint_stream`, `drift_item`: each driven by its own `Random` and the update index), and the buffers. Admissions read only the presented items and `rng`. Nothing there reads the model, the optimizer, the losses or the scores. `update16`/`E.update` write the store from the batch tensors and mask only. Evaluation (`E.respond`, argmax decoding) and the first-update census (`zero_grad_report`) draw no random numbers. No module draws from torch's RNG after `new_net` (`exp11_cell.py`, `arch17.py` and `exp3_state.lin` have no random calls on the update path, and there is no dropout). `LAST_AUX` is logging only. `develop17`'s stream is `E.batches`, driven by its own `Random`. I found no path where the data depends on the weights.

**A resumed receipt says so.** The receipt gains a list `resumed`, one entry per restart: the stage (or update) resumed from, the checkpoint file used (`.ckpt` or `.ckpt.prev`), host, UTC time, and the CPU spent replaying. A restart that found no checkpoint is recorded with `from_stage: 0` and a note. The CPU fields continue from the saved values, so `cpu`, `cpu_scoring_so_far` and `cpu_updates_so_far` exclude the replay. A receipt that was never resumed has no `resumed` key and is otherwise byte for byte what the base commit writes, apart from the CPU timings.

**Fail-closed.** A changed setting stops the run, naming it, the checkpoint's value and the new one. A damaged `.ckpt` (bad magic, wrong sha256, unreadable) falls back to `.ckpt.prev` and prints that it did. If every existing checkpoint is damaged, the run stops. A missing `.ckpt` with a good `.ckpt.prev` (a kill between the two renames) resumes from `.ckpt.prev`. A digest mismatch at the resume point stops the run, naming the differing components. With `E16_RESUME=1` and no checkpoint at all, the run starts from the birth, says so, and records it.

## Tests (`e21_resume_check.py`)

**Result: 47 checks, 47 passed, 0 failed** (`exp45/E21_RESUME_CHECK.json`). The harness under test has sha256 `133902d9…ecb0bd`, byte-identical to the committed `exp16_integrated.py`. The base is `exp16_integrated.py` from `40cb1c20`, with every other module from this folder (the script checks that none of them differs from the base).

How a kill is made. `e21_resume_check.py --child SPEC` imports the harness, wraps its update and score functions with counters, and has the process send itself `SIGKILL` at an exactly counted call. Then it runs the harness's own main. The restart is the real command with `E16_RESUME=1`. The harness has no test hook. Every run has its own directory: a copy of this folder's `*.py`, plus links to the birth files it needs.

Comparisons. *Receipt* means every field of every stage record (scores, losses, exposure, reservoir counts, settings, first-update census) or curve row, apart from the three CPU fields and `resumed`, and in the same key order. *Weights* means the sha256 of the final weights file, checked against the file itself. *Digests* means every line of the resumed run's `.ckpt.log`: checkpoints written before the kill, the replayed stages, and checkpoints after the resume, each compared with the uninterrupted run's checkpoint digest for that stage.

Doses: lives at `E11_STAGE_UPDATES=16`, the birth at `E11_DEV_UPDATES=8192` (curve rows at 4,096 and 8,192), seed 73129.

| # | Shape | Kill point (counted in the killed process) | Resumed from | Receipt | Weights | Digests (run / reference) | Result |
|---|---|---|---|---|---|---|---|
| 1 | T, drift, online reservoir, store | not resuming: this branch vs base | n/a | equal | equal `25ed64e5` | n/a | PASS |
| 2 | T, drift, online reservoir, store | update 4 (stage 1, before any checkpoint) | birth (recorded as from_stage 0) | equal | equal | 6 / 6 | PASS |
| 2 | " | update 40 (stage 3 updates) | stage 2 | equal | equal | 8 / 6, replayed 1-2 | PASS |
| 2 | " | 25th score call after update 64 (stage 4 evaluation) | stage 3 | equal | equal | 9 / 6, replayed 1-3 | PASS |
| 2 | " | twice: update 21 (stage 2), then in the resumed process the 10th score call of stage 5 | stage 1, then stage 4 | equal | equal | 11 / 6, replayed 1-4 | PASS |
| 1 | D, window reference, staged | not resuming | n/a | equal | equal `e75d9797` | n/a | PASS |
| 2 | " | update 4 / update 40 / stage 4 evaluation | birth / 2 / 3 | equal ×3 | equal ×3 | 6, 8, 9 / 6 | PASS ×3 |
| 1 | M, backprop on the pipeline, block admission | not resuming | n/a | equal | equal `f2ea189a` | n/a | PASS |
| 2 | " | update 4 / update 40 / stage 4 evaluation | birth / 2 / 3 | equal ×3 | equal ×3 | 6, 8, 9 / 6 | PASS ×3 |
| 1 | bank2, Phase 4 rule | not resuming | n/a | equal | equal `37e6fb2c` | n/a | PASS |
| 2 | " | update 4 / update 40 / stage 4 evaluation | birth / 2 / 3 | equal ×3 | equal ×3 | 6, 8, 9 / 6 | PASS ×3 |
| 1 | bank2, Phase 4 rule, `E16_LOSS=all` | not resuming | n/a | equal | equal `4dba8a76` | n/a | PASS |
| 2 | " | update 4 / update 40 / stage 4 evaluation | birth / 2 / 3 | equal ×3 | equal ×3 | 6, 8, 9 / 6 | PASS ×3 |
| 1 | T, letters in compositions (`E16_COMPGAP=6`) | not resuming | n/a | equal | equal `4c0d9c96` | n/a | PASS |
| 2 | " | update 4 / update 40 / stage 4 evaluation | birth / 2 / 3 | equal ×3 | equal ×3 | 6, 8, 9 / 6 | PASS ×3 |
| 1 | birth by the rule, local heads | not resuming | n/a | equal | equal `2e71e723` | n/a | PASS |
| 2 | " | update 100 (before any checkpoint) | beginning | equal | equal | 2 / 2 | PASS |
| 2 | " | update 5,096 (after the row at 4,096) | update 4,096 | equal | equal | 3 / 2 | PASS |
| 2 | " | 7th score call after update 8,192 (evaluation of the last row) | update 4,096 | equal | equal | 3 / 2 | PASS |

Fail-closed (test 3). Each case starts from a life killed in stage 3, which leaves `.ckpt` at stage 2 and `.ckpt.prev` at stage 1. Shape: T, drift, online reservoir, store.

| Case | What the restart did | Result |
|---|---|---|
| `E16_LEGACY_BIRTH=allow` added | exit 1 before any update: "…was written under a different configuration; changed: E16_LEGACY_BIRTH (checkpoint None, now 'allow')" | PASS |
| `E11_SCHEDULE=interleaved` | exit 1: "changed: E11_SCHEDULE (checkpoint 'blocked', now 'interleaved')" | PASS |
| `.ckpt` truncated to half | "unreadable … body does not match its sha256", "falling back to ….ckpt.prev"; resumed from stage 1; receipt, weights and digests equal to the uninterrupted life | PASS |
| `.ckpt` truncated and one byte of `.ckpt.prev` flipped | exit 1: "every checkpoint is damaged, refusing to start over" | PASS |
| pipeline digest edited in `.ckpt` (the `rng` component; file re-sealed so it reads cleanly) | exit 1 at the comparison: "the replayed data pipeline does not match … at stage 2; differing components: ['rng']" | PASS |
| killed inside the stage-3 checkpoint write, after `.ckpt` → `.ckpt.prev` and before the new file is renamed into place | resumed from `.ckpt.prev` (stage 2); exact | PASS |

A setting that is part of the receipt's name (`E16_CAP`, `E16_REPLAY`, the dose, …) leads to a different receipt and therefore no checkpoint. The run then starts from the birth, says so, and records `from_stage: 0`. So the fail-closed cases use settings that are in force but not in the name.

Larger dose (test 4), `E11_STAGE_UPDATES=512`, for the T drift/online/store life and the bank2 Phase 4 life. Each was killed once at update 1,300 (stage 3), resumed from stage 2 and came out exact: receipt, weights, 8 / 6 digests. Each also passed the four fail-closed cases: both changed settings, truncated `.ckpt` → `.ckpt.prev` (exact), both damaged, and edited digest. That is 12 checks, all passed.

Not run at the larger dose: test 1 (the brief asks for tests 2 and 3 there) and the kill inside a checkpoint write.

## Costs

Measured at `E11_STAGE_UPDATES=512` on this machine (4 vCPU Xeon @ 2.10 GHz, one torch thread per run):

| | window T life (drift, online, store) | bank2 Phase 4 life |
|---|---|---|
| checkpoint size | 9.64-9.66 MB (grows by about 4 kB per stage with the receipt) | 13.55-13.57 MB |
| seconds to write one (wall, incl. fsync) | 0.035-0.065 | 0.058-0.068 |
| replay CPU per stage at 512 updates | 1.64 s | 0.34 s |
| replay CPU per stage at the full 6,144, **estimate** (linear scaling) | ≈ 20 s | ≈ 4 s |
| for comparison, update CPU per stage at 512 | ≈ 60 s | ≈ 72 s |

The full-dose replay figure is an estimate, scaled linearly from 512 updates per stage; no full-dose life was run. Resuming a life killed late at the full dose should therefore cost about a minute and a half of replay for the window learner (five stages), against one to two hours for a rerun. The window learner's replay costs more than bank2's because it also rewrites the score-time store. The replay CPU includes computing the configuration and panel hashes. Building the panels happens before the CPU clock starts, as in every life.

## Surprises

* **The container restarted during the acceptance run, about an hour in**: the failure E21 is about. The lives inside it would now have been resumable, but my test runner was not, and its progress was lost. I made the runner keep any job that had finished, so another restart would only lose the jobs in flight, and reran. All results here come from that rerun on the same code.
* **Two bugs in the verdict code of `e21_resume_check.py`, none in the harness.** A birth's receipt has `complete: true` where a life's has a record. The setting-change verdict passed one key twice. Both raised exceptions in the final verdict step after every run had finished. Fixing them, I first changed the job cache key, which caused the four birth jobs to rerun. The script was then edited (cache key, verdict code) after the runs, so its committed hash differs from the one recorded when the runs began. The kill points, phases and comparisons did not change. The harness did not change at all: its committed hash is the one tested.
* **A digest detail.** `fset` (used by `all_balanced` replay) is a Python set of tuples that contain strings, and string hashing is randomized per process, so its iteration order differs between processes. Nothing in the pipeline depends on that order (the set is used for membership only), but a digest of the set as iterated would differ between two identical runs. The digest sorts it. The lists and dicts that carry order (`buffer`, `fbuf`, the store's insertion order) are digested in order.
* **Replay is cheap, and evaluation dominates short lives.** At 16 updates per stage a life spends about 100 s per stage scoring and under a second updating. At 512 the update work is about 60-72 s per stage, against 1.6 s or less of replay.
* No path where the data pipeline reads the weights was found, so the brief's design held without change.

## What is not covered

* **Two processes writing one receipt.** Nothing stops two runs of the same life from writing the same receipt and checkpoint. Not covered, as the brief allows.
* No resume for `exp11_cell.py` births, `exp19_kbank.py` or the summarizers (not asked). `qualify()` is unchanged.
* A kill after a life's final weights are saved but before its final receipt write is resumable by design: the stage-6 checkpoint is replayed and both files are rewritten. That window is milliseconds and was not tested. For a birth, the same window leaves weights without a receipt. With `E16_RESUME=1` and a checkpoint present, that birth is resumed rather than refused. Without them, the existing guard still stops it. Also not tested.
* The torch version, thread count and machine are not part of the configuration digest (the thread count is always one). A life started on one machine can be resumed on another, and the result will be exact with respect to the checkpoint. It need not equal what an uninterrupted run on the first machine would have given, because floating-point results can differ between machines and torch versions. The receipt's `resumed` entry names the host.
* A complete `.ckpt.tmp` left by a kill between the renames is ignored, conservatively: the run resumes from `.ckpt.prev`, one stage earlier, and the next checkpoint overwrites the temporary file.
* The tests cover the seven command shapes of the brief at seed 73129. Other arms and settings (`uniform`, `balanced`, `share`, `reservoir`, `total`, `oracle` replay, `E16_FREEZE`, `E16_HEADS=aux`, `E16_RSHARE`) use the same loop and the same pipeline digest but were not run here.
