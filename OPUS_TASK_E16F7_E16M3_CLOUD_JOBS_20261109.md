# Two cloud jobs for Claude Opus 5.5 workers: E16-F7 and E16-M3 (9 November 2026, 22:00 EST)

**Repository:** `ChangweiZhou/digitalbaby`. **Branch to work from:** `codex/minifly-consolidated-20261023` (run `git fetch origin codex/minifly-consolidated-20261023 && git checkout codex/minifly-consolidated-20261023`; confirm `git log -1` shows a commit of 9 November 2026, 22:00 EST or later: the harness commit this brief ships with adds the replay-share knob that Job 2 uses). **Folder:** `MOD97_LOCAL_RULE_20261106/`. **Needs:** Python 3 with PyTorch on CPU and NumPy; install only those if missing. One job per worker; say in your first message which job you are running. Each job splits into two halves (a and b) if four workers are available; a half is a complete job with its own predictions file, report and review paragraph.

**Branch rule.** If your session can push to `codex/minifly-consolidated-20261023`, commit there and push after every block (partial receipts) and at the end; pull with `--rebase` before each push and never force. Otherwise commit to the branch your session is given, push there, and put the branch name and final commit in your last message; it will be merged by the coordinating session. Do not create any other branch.

**Rules, the same for both jobs.** Launch every life detached from the tool command (`setsid nohup env ... python3 exp16_integrated.py ... > <log> 2>&1 < /dev/null &` from a short command that returns at once) and poll the logs with short commands; one line prints per finished block. Write logs outside the repository folder and copy them into `exp45/logs/` at the end. **Dry-run each new command once before launch** with `E11_STAGE_UPDATES=4` (four updates per block; the receipt gets a `_dry4` tag): it must run its seven stages without error; move the dry receipts to `exp45/archive_<job>_dry/`. Write and commit your predictions file before any life starts and do not change it afterwards. Add files; edit none of the harness, receipts, reports, review or the audit scripts under `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/` (copy a script under a new name if you need to change its seed list). No reruns of a finished life, no tuning, no change of seeds; a miss is a result. Report every seed separately, holds or misses with the numbers, and forecast the job from the dry run's pace before launch, then report forecast against measured. **Run identity:** record in the report the machine (`uname -a`, core count, `python3 -c "import torch; print(torch.__version__)"`), the harness commit, and the CPU-seconds of every life (the last log line prints `cpu <n>s`). Report anything that went wrong plainly.

**What the harness is.** `exp16_integrated.py` runs one life of a byte-level learner (a network with a diagonal memory cell, trained by a local sign-symmetric rule with an exact eligibility trace; arm `T`) or its backprop twin (arm `D`) on a stream of three families of modular-arithmetic items (mod-97 operations with distractors, twelve taught operator pairs written as scratchpads, mod-73 operations), with a replay memory selected by environment variables. Arm `M` is the twin's update (backprop through time, decay .3) on the learner's data pipeline (the same reservoir, admission and replay share as `T`). Under `E16_STREAM=drift` there are no blocks: each item's family is drawn from weights that move smoothly through the life, scored at six checkpoints. Every setting is an environment variable; the receipt is a JSON file under `exp45/` written after each checkpoint (plus the final weights as `.pt`); `summarize_exp16.py` builds tables for a seed list (`E16_SEEDS`) into a named file (`E16_SUMMARY`). Scores are exact answers on sealed panels; "ratio to the twin" is the learner's family mean divided by the twin's; compositions are the mean final-answer accuracy over the four held pairs (`+-`, `-+`, `><`, `<>`), operations the family mean at k = 12. The gate as pre-registered is 0.80 of the twin on every family. The 9 November audit (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/gate_by_item_type.py`) re-scores the same receipts by item type: on the `+-` and `-+` pairs alone, the pairs whose shortcut ceilings are about 1%, only 4 of 21 raw-reservoir lives reach 0.80 of the twin. Report the hard pairs apart from the family means throughout.

**Standing caveat.** Whether the sign-symmetric rule counts as "local" under the project's propagation ruling is open with Mr. Zhou (review section 9.60). Both jobs run the existing arm as a diagnostic; neither claims a learning-core improvement. Evidence level of the whole E16 line is E2 (sealed held-out panels, operand-disjoint from taught triples, no reuse on never-reinforced inputs claimed).

---

## Job 1: E16-F7, six more drift seeds and a pre-registered early signature

**Why.** Twelve seeds of the drifting stream are on file (E16-S, E16-F5, E16-F6): ten pass the 0.80 gate on family means, two fail (400009, 600013), both by the plus-minus pairs ending at 42–48 while the max-min pairs stay above 81 and the twin sits at 94–95. E16-F6 pre-registered a checkpoint-2 signature and it missed (Spearman 0.217). After seeing the data, E16-F6 noted that the lowest mod-97 family mean over checkpoints 2–4 ranks the two failing seeds lowest (47.2 and 62.0) with Spearman 0.40; that is an observation, not a finding. Six new seeds test it as a prediction, raise the drift-seed count to eighteen for the failure rate, and give six more hard-pair gate readings.

**Seeds** (none has a birth; check and record that no `exp45/e11_dev_M_<seed>.*` exists): **1100009, 1200007, 1300021, 1400017, 1500007, 1600033.** Half a: the first three. Half b: the last three.

**Steps.**
1. Dry-run one learner life and one twin life (`E11_STAGE_UPDATES=4`, seed 500011, whose birth exists), archive to `exp45/archive_e16f7_dry/`, forecast, commit predictions.
2. Launch the births together, detached: `python3 exp11_cell.py dev M <seed>` (about 2,760 CPU-s each on E16-F6's container).
3. When a birth has finished, launch its two lives, whether or not the birth met P16F7.1 (a miss is a result; run and say so):

```
E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py T <seed>
E16_STREAM=drift python3 exp16_integrated.py D <seed>
```

4. Tables: `E16_SEEDS=<your seeds> E16_SUMMARY=SUMMARY_E16F7.md python3 summarize_exp16.py`. Hard pairs: copy `gate_by_item_type.py` to `e16f7_gate.py` with the seed list extended to your seeds and `REF` set to your HEAD; its output table goes in the report.

Receipts `exp45/e16_life_T_raw65536_drift_online_<seed>.json` and `exp45/e16_life_D_drift_<seed>.json` with their `.pt`; births `exp45/e11_dev_M_<seed>.pt` and `.json` (`git add -f` the `.pt` files); logs `e16f7_<arm>_<seed>.log`.

**Predictions (copy into `E16F7_PREDICTIONS.md` and commit before the births).** Define S = the learner's lowest mod-97 family mean at k = 12 over checkpoints 2, 3 and 4.
- P16F7.1 (births) each birth reaches at least 85% at k = 12 on all four development operations (mod 89).
- P16F7.2 (twin) the drift twin ends at or above 90 / 85 / 90 on the three family means in every seed.
- P16F7.3 (rate) at least four of the six seeds end at or above 0.80 of their twin on all three families.
- P16F7.4 (the signature, pre-registered this time) every seed with S below 65 ends below 0.80 of its twin on held compositions, and every seed with S at or above 70 ends at or above 0.80; over the eighteen drift seeds (the twelve on file and these six) the Spearman rank correlation between S and the final `+-` pair by final answer is at least 0.35. Both clauses must hold.
- P16F7.5 (hard pairs) at most two of the six seeds reach 0.80 of the twin on the `+-` and `-+` pairs alone (final answer), while at least four pass the family gate: the family gate and the hard-pair gate keep disagreeing.

Reading, written before the result: if P16F7.4 holds, S is an early-warning measure worth carrying into later designs; if the correlation clause misses while the threshold clauses hold, S separates failures but does not rank the passing seeds, which is what E16-F6 saw; if a seed with S above 70 fails, the failure has no early signature of this kind. P16F7.5 is the hard-pair count on new seeds; it changes nothing about the gate as pre-registered, which stays reported as such.

**Report.** Per seed, at every checkpoint: mod-97 add, sub, max, min at k = 12 and the four held pairs by final answer, learner and twin, as tables; the eighteen-seed list of (S, final `+-`, final `-+`, composition ratio) sorted by S, with the correlation; the hard-pair table; the five verdicts; births; run identity and CPU-seconds per life; forecast against measured; what went wrong. Deliverables: `E16F7_PREDICTIONS.md`, `E16F7_REPORT.md`, receipts and weights, births, logs, `exp45/SUMMARY_E16F7.md`, `e16f7_gate.py` and its table, and one paragraph as `E16F7_REVIEW_SECTION_9_62.md` (half b: `9_63`). Cost: about 16,600 CPU-s of births and 54,000 CPU-s of lives for all six seeds (learner 4,700–4,950, twin 4,160–4,330 CPU-s each on E16-F6's container); on four cores about 5.5 hours of wall time for the whole job, about 3 hours per half. Forecast from your dry run.

---

## Job 2: E16-M3, the replay share: is the reservoir's cost to the rule a dose

**Why.** E16-M2 showed that backprop on the learner's own reservoir reaches 0.95–0.97 of the twin on compositions where the sign rule reaches 0.69–0.71 in the failing seeds, and that the twin's joint stream in place of the reservoir lifts the rule only to 0.85–0.90: the raw reservoir's sampling costs the rule 14–20 points that it costs backprop nothing. That is an interaction between rule and data, with the rule first. The simplest handle on the data side is the replay share: today half of every batch is drawn verbatim from the reservoir once it holds anything. If the cost is a dose, less replay should help the compositions and hurt mod-97 retention, more replay the reverse, and backprop should be flat across the range.

**Harness note.** The commit this brief ships with adds `E16_RSHARE=<fraction>`: the share of every batch drawn from the raw reservoir (default 0.5 reproduces every receipt made before; other values tag the receipt `_rs<percent>` and record `rshare` in the settings). It applies to the raw/total reservoirs only. The dry runs of the four new command shapes are in `exp45/archive_e16m3_dry/`; repeat them on your machine.

**Runs.** Fourteen lives on the drifting stream. Seeds: **400009** and **600013** (the two failing seeds; both fail under drift) and **300007** (a clear pass, 0.96 · 0.87 · 0.99). Half a: the nine sign-rule lives. Half b: the five backprop lives.

```
# half a: the rule at three shares, three seeds
E16_RSHARE=0    E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py T <seed>
E16_RSHARE=0.25 E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py T <seed>
E16_RSHARE=0.75 E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py T <seed>
# half b: backprop on the same pipeline at 0.25 and 0.75 for 400009 and 600013, and at the default 0.5 for 600013 (its drift M life does not exist; 400009's is E16-M2's)
E16_RSHARE=0.25 E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py M <seed>
E16_RSHARE=0.75 E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py M <seed>
E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py M 600013
```

Receipts `exp45/e16_life_T_raw65536_drift_online_rs{0,25,75}_<seed>.json`, `exp45/e16_life_M_raw65536_drift_online_rs{25,75}_<seed>.json`, `exp45/e16_life_M_raw65536_drift_online_600013.json`, each with its `.pt`; logs `e16m3_<arm>_rs<share>_<seed>.log`. Comparators in the repository: the share-0.5 learner lives `exp45/e16_life_T_raw65536_drift_online_<seed>.json`, the twins `exp45/e16_life_D_drift_<seed>.json`, and E16-M2's `exp45/e16_life_M_raw65536_drift_online_400009.json`. `summarize_exp16.py` lists all the new arms. Share 0 is the no-replay control: under drift the earlier families keep a small presence in the stream, so it is not a pure forgetting test, and it is read as such.

**Predictions (copy into `E16M3_PREDICTIONS.md` and commit before launch).** All at the last checkpoint; "hard pairs" is the mean of `+-` and `-+` by final answer.
- P16M3.1 (replay does work) share 0 ends below 0.80 of the twin on mod-97 operations in all three seeds, below the share-0.5 life by at least 10 points.
- P16M3.2 (a dose on the rule) in both failing seeds, the rule's hard pairs at share 0.25 are at least 5 points above share 0.5, and at share 0.75 at least 5 points below share 0.5; mod-97 operations move the other way (0.25 below 0.5, 0.75 above 0.5, each by at least 3 points).
- P16M3.3 (backprop flat) every `M` life ends at or above 0.90 of the twin on held compositions, with its hard pairs within 5 points of the `M` life at share 0.5 of the same seed.
- P16M3.4 (the passing seed) 300007 passes the family gate at shares 0.25 and 0.75 both, and its hard pairs at 0.25 are above those at 0.75.

Reading, written before the result: P16M3.2 and P16M3.3 holding together say the reservoir's cost to the rule is a dose of verbatim replay that backprop tolerates, and the next arm is a lower share or a replay that is not verbatim. P16M3.2 missing with the rule's hard pairs flat across shares says the share is not the handle, and the interaction is in what is replayed, not how much. P16M3.3 missing says the share matters to backprop too, and the comparison with E16-M2 was at a favourable share. State which case obtained; two seeds may split, as in E16-M2, and then both cases are stated.

**Report.** Per seed and share, at every checkpoint: mod-97 add, sub, max, min at k = 12 and the four held pairs by final answer, beside the share-0.5 life and the twin, as tables; a share-curve table (rows 0 / 0.25 / 0.5 / 0.75, columns mod-97 mean, hard pairs, composition ratio, mod-73 mean) per seed for `T` and for `M`; the four verdicts; run identity and CPU-seconds per life; forecast against measured; what went wrong. Deliverables: `E16M3_PREDICTIONS.md`, `E16M3_REPORT.md`, the fourteen receipts and weights, logs, `exp45/SUMMARY_E16M3.md`, and one paragraph as `E16M3_REVIEW_SECTION_9_64.md` (half b: `9_65`). Cost: about 4,800 CPU-s per `T` life and 4,500 per `M` life on E16-M2's container; nine `T` lives on four cores about 3.2 hours, five `M` lives about 1.7 hours, the whole job about 4.8 hours. Forecast from your dry run.

---

**Section numbers.** 9.61 is reserved for E17 Phase 2, being filed by the coordinating session. If the halves are not split, use the lower number of each pair.
