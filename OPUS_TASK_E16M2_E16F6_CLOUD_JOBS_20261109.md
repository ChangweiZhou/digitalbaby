# Two cloud jobs for Claude Opus 5.5 workers: E16-M2 and E16-F6 (9 November 2026, 12:00 EST)

**Repository:** `ChangweiZhou/digitalbaby`. **Branch to work from:** `codex/minifly-consolidated-20261023` (run `git fetch origin codex/minifly-consolidated-20261023 && git checkout codex/minifly-consolidated-20261023`; confirm `git log -1` shows a commit of 9 November 2026, 12:00 EST or later: the harness commit that this brief ships with changes arm M, see Job 1). **Folder:** `MOD97_LOCAL_RULE_20261106/`. **Needs:** Python 3 with PyTorch on CPU and NumPy; install only those if missing. One job per worker; say in your first message which job you are running. If only one worker is available, run Job 1 first, then Job 2.

**Branch rule.** If your session can push to `codex/minifly-consolidated-20261023` (E16-F5's could), commit there and push after every block (partial receipts) and at the end; pull with `--rebase` before each push and never force. Otherwise commit to the branch your session is given, push there, and put the branch name and final commit in your last message; it will be merged by the coordinating session as E16-RP's was (`claude/vibrant-gauss-tops5f` → c87a7434). Do not create any other branch.

**Rules, the same for both jobs.** Launch every life detached from the tool command (`setsid nohup env ... python3 exp16_integrated.py ... > <log> 2>&1 < /dev/null &` from a short command that returns at once) and poll the logs with short commands; one line prints per finished block. Write logs outside the repository folder and copy them into `exp45/logs/` at the end (a rebase cut off logs written in the repository on 8 November). **Dry-run each new command once before launch** with `E11_STAGE_UPDATES=4` (four updates per block; the receipt gets a `_dry4` tag): it must run its seven stages without error; move the dry receipts to `exp45/archive_<job>_dry/`. Write and commit your predictions file before any life starts and do not change it afterwards. Add files; edit none of the harness, receipts, reports or review. No reruns of a finished life, no tuning, no change of seeds; a miss is a result. Report every seed separately, holds or misses with the numbers, and forecast the job from the dry run's pace before launch, then report forecast against measured. Report anything that went wrong plainly.

**What the harness is.** `exp16_integrated.py` runs one life of a byte-level learner (a network with a diagonal memory cell, trained by a local sign-symmetric rule with an exact eligibility trace; arm `T`) or its backprop twin (arm `D`) on a stream of three families of modular-arithmetic items, with a replay memory selected by environment variables. Arm `M` is the twin's update (backprop through time, decay .3) on the learner's data pipeline (the same reservoir, admission and half-batch replay as `T`). `E16_REPLAY=oracle` is the learner's rule on the twin's joint stream (fresh items of every family seen so far at equal weight; no reservoir). Every setting is an environment variable; the receipt is a JSON file under `exp45/` written after each of six blocks (plus the final weights as `.pt`); `summarize_exp16.py` builds tables for a seed list (`E16_SEEDS`) into a named file (`E16_SUMMARY`). Scores are exact answers on sealed panels; "ratio to the twin" is the learner's family mean divided by the twin's; compositions are the mean final-answer accuracy over the four held pairs (`+-`, `-+`, `><`, `<>`), operations the family mean at k = 12. Births `exp45/e11_dev_M_<seed>.pt` exist for 73129, 93683, 116089, 200003, 300007, 400009, 500011, 600013, 700027.

---

## Job 1: E16-M2, rule or data on the two failing seeds

**Why.** Of the fifteen raw-byte reservoir lives run so far (nine on block stages, six on the drifting stream), two fall below the gate: seed 600013 under stages (E16-F4, `E16F4_REPORT.md`) and seed 400009 under the drifting stream (E16-F5, `E16F5_REPORT.md`). Both fail the same way: the mod-97 operations dip to 52–62 when the compositions arrive, the `+-` and `-+` pairs end near 45–55 while the max-min pairs stay above 80, the twin is unaffected at 93–94, and the reservoir looks the same as in passing seeds (1.05 MB, 55,000–58,000 distinct strings). Two accounts remain: the data pipeline (what the raw reservoir happened to hold and how it was sampled in these seeds under-serves the plus-minus pairs), or the sign rule's own interference between the plus-minus compositions and the addition and subtraction operations, which backprop does not suffer. Two arms on the same births separate them: `M` keeps the data pipeline and changes the rule; `oracle` keeps the rule and changes the data.

**Harness note.** Before the commit this brief ships with, arm `M` under online admission (`E16_ADMIT=online`) admitted items to the reservoir at block ends only, because admission was gated on the local arm. It now admits online, as `T` does; under block admission nothing changed. The dry run of both `M` commands is in `exp45/archive_e16m2_dry/` on the consolidated branch; repeat it on your machine.

**Runs.** Four lives, detached, launched together:

```
E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py M 400009
E16_REPLAY=oracle E16_STREAM=drift python3 exp16_integrated.py T 400009
E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py M 600013
E16_REPLAY=oracle python3 exp16_integrated.py T 600013
```

Receipts `exp45/e16_life_M_raw65536_drift_online_400009.json`, `exp45/e16_life_T_oracle_drift_400009.json`, `exp45/e16_life_M_raw65536_600013.json`, `exp45/e16_life_T_oracle_600013.json`, each with its `.pt`; logs `e16m2_<arm>_<seed>.log`. Comparators, all in the repository: the failing learner lives `exp45/e16_life_T_raw65536_drift_online_400009.json` and `exp45/e16_life_T_raw65536_600013.json`, and their twins `exp45/e16_life_D_drift_400009.json` and `exp45/e16_life_D_600013.json`. `summarize_exp16.py` lists `M` and `T_oracle` for block stages but not their drift variants; build the comparison table with a small read-only script of your own (`e16m2_tables.py`, as E16-RP's `e16rp_tables.py`) and do not edit the summarizer.

**Predictions (copy into `E16M2_PREDICTIONS.md` and commit before launch).**
- P16M2.1 (the data pipeline is not the cause) `M` ends at or above 0.90 of the twin on held compositions in both seeds, with its `+-` and `-+` pairs at or above 75.
- P16M2.2 (replay coverage is not the cause either) `T` under oracle replay ends at or above 0.80 of the twin on held compositions in both seeds, with the `+-` pair at or above 65.
- P16M2.3 (the signature belongs to the rule) the dip of the mod-97 family mean below 70 when compositions arrive (checkpoint 2 under drift, block 3 under stages) appears in `T` under oracle replay and not in `M`, which stays above 85 at that checkpoint in both seeds.

Reading, written before the result: if P16M2.1 holds and P16M2.2 misses, the failure is the sign rule's interference and not the replay data, and the next arm is on the rule. If both hold, the failure lies in what the raw reservoir held or how it was sampled in those seeds, and the next arm is on sampling. If `M` misses too, the data pipeline is the cause regardless of rule. State which case obtained.

**Report.** Per seed, at every checkpoint: the mod-97 operations (add, sub, max, min at k = 12) and the four held pairs by final answer, for the four new lives beside the failing learner and the twin, as tables; the three verdicts; machine, pace, forecast against measured; what went wrong. Deliverables: `E16M2_PREDICTIONS.md`, `E16M2_REPORT.md`, the four receipts and weights, the logs, `e16m2_tables.py` and its table `exp45/SUMMARY_E16M2.md`, and one paragraph as `E16M2_REVIEW_SECTION_9_58.md`. About 4,100–4,700 CPU-seconds per life on a four-core container; four processes on four cores, about 1.5 hours of wall time.

---

## Job 2: E16-F6, the plus-minus tail: six more seeds of the drifting stream

**Why.** The drifting stream is the form the stream keeps from here on, and the gate under it rests on six seeds, five of which pass (E16-S, E16-F5). The two misses so far (600013 under stages, 400009 under drift) share a signature that may be visible early: the learner's mod-97 operations at checkpoint 2 dipped to 52 and 62 in the failing seeds and to 72–92 in the passing ones. Six more seeds under drift give the failure rate on twelve seeds, test whether 600013 fails under drift as it did under stages, and test the early signature as a predictor across all twelve.

**Steps.**
1. Check that no `exp45/e11_dev_M_<seed>.*` exists for the new seeds **800011, 900007, 1000003**; record it. Dry-run one learner life and one twin life (`E11_STAGE_UPDATES=4`, seed 500011), archive the dry receipts, forecast, commit predictions.
2. Launch together, detached: the three births, `python3 exp11_cell.py dev M <seed>` for 800011, 900007, 1000003 (about 50 CPU-minutes each), and the six lives of the seeds whose births exist, 500011, 600013, 700027:

```
E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py T <seed>
E16_STREAM=drift python3 exp16_integrated.py D <seed>
```

3. When the births have finished, launch the same two lives for each of 800011, 900007, 1000003, whether or not the births met P16F6.1 (a miss is a result; run them anyway and say so).
4. Tables: `E16_SEEDS=500011,600013,700027,800011,900007,1000003 E16_SUMMARY=SUMMARY_E16F6.md python3 summarize_exp16.py` (arms `T_raw65536_drift_online` and `D_drift` are listed).

Receipts `exp45/e16_life_T_raw65536_drift_online_<seed>.json` and `exp45/e16_life_D_drift_<seed>.json` with their `.pt`; births `exp45/e11_dev_M_<seed>.pt` and `.json` (`git add -f` the `.pt` files as before); logs `e16f6_<arm>_<seed>.log`.

**Predictions (copy into `E16F6_PREDICTIONS.md` and commit before the births).**
- P16F6.1 (births) each new birth reaches at least 85% at k = 12 on all four development operations.
- P16F6.2 (twin) the drift twin is at or above 90 / 85 / 90 on the three family means after the last checkpoint in every seed (seed 500011's twin under stages reached only 84.0 on compositions and was still rising; a repeat is a result).
- P16F6.3 (rate) at least four of the six seeds end at or above 0.80 of their drift twin on all three families.
- P16F6.4 (seed 600013) misses the composition clause under drift as it did under stages, below 0.80 of its twin on held compositions: a seed-level fragility, not a schedule-level one.
- P16F6.5 (the early signature, over the twelve drift seeds: 73129, 93683, 116089, 200003, 300007, 400009 from E16-S and E16-F5, and these six) the Spearman rank correlation between the learner's mod-97 family mean at checkpoint 2 and its `+-` pair by final answer at checkpoint 6 is at least 0.5; every seed below 70 at checkpoint 2 ends with `+-` below 60, and every seed above 80 at checkpoint 2 ends with `+-` above 65.

**Report.** Per seed, at every checkpoint: mod-97 add, sub, max, min at k = 12 and the four held pairs by final answer, learner and twin, as tables; the twelve-seed list of (checkpoint-2 mod-97 mean, final `+-`, final `-+`, composition ratio to the twin), sorted by the first column, with the correlation; the five verdicts; births; machine, pace, forecast against measured; what went wrong. Deliverables: `E16F6_PREDICTIONS.md`, `E16F6_REPORT.md`, twelve receipts and weights, three births, logs, `exp45/SUMMARY_E16F6.md`, and one paragraph as `E16F6_REVIEW_SECTION_9_59.md`. About 2.5 CPU-hours of births and 14.7 CPU-hours of lives (4,700 CPU-s per learner, 4,100 per twin on E16-F5's machine); with the first six lives started beside the births, about 5 hours of wall time on four cores. Forecast from your dry run.
