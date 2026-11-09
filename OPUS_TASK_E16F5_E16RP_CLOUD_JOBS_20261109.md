# Two cloud jobs for Claude Opus 5.5 workers: E16-F5 and E16-RP (9 November 2026)

**Repository:** `ChangweiZhou/digitalbaby`. **Branch to work from:** `codex/minifly-consolidated-20261023` (run `git fetch origin codex/minifly-consolidated-20261023 && git checkout codex/minifly-consolidated-20261023`; confirm `git log -1` shows a commit of 9 November 2026 or later). **Folder:** `MOD97_LOCAL_RULE_20261106/`. **Needs:** Python 3 with PyTorch on CPU and NumPy; install only those if missing. One job per worker; say in your first message which job you are running.

**Branch rule.** A cloud worker cannot push to `codex/minifly-consolidated-20261023`. Commit everything to the branch your session is given and push there after every block (partial receipts) and at the end; put the branch name and final commit in your last message. Do not create any other branch. Your branch will be merged into the consolidated branch by the coordinating session, as was done for E16-RC (`claude/vibrant-gauss-tops5f` → commit c8f55e1).

**Rules, the same for both jobs.** Launch every life detached from the tool command (`setsid nohup env ... python3 exp16_integrated.py ... > exp45/logs/<name>.log 2>&1 < /dev/null &` from a short command that returns at once; the tool's two-hour limit killed foreground lives on 8 November) and poll the logs with short commands; one line prints per finished block. Write and commit your predictions file before any life starts and do not change it afterwards. Add files; edit none of the harness, receipts, reports or review. No reruns of a finished life, no tuning, no change of seeds; a miss is a result. Report every seed separately, holds or misses with the numbers. Report anything that went wrong plainly.

**What the harness is.** `exp16_integrated.py` runs one life of a byte-level learner (a network with a diagonal memory cell, trained by a local sign-symmetric rule with an exact eligibility trace; arm `T`) or its backprop twin (arm `D`) on a stream of three families of modular-arithmetic items, with a replay memory selected by environment variables. Every setting is an environment variable; the receipt is a JSON file under `exp45/` written after each of six blocks; `summarize_exp16.py` builds tables for a seed list (`E16_SEEDS`) into a named file (`E16_SUMMARY`). Scores are exact answers on sealed panels; "ratio to the twin" is the learner's family mean divided by the twin's. The standing seeds are 73129, 93683, 116089; the fresh seeds 200003, 300007, 400009 have births `exp45/e11_dev_M_<seed>.pt` in the repository.

---

## Job 1: E16-F5, the stream without stages on the fresh seeds

**Why.** E16-S (`MOD97_LOCAL_RULE_20261106/E45_REPORT.md`, section "E16-S and E16-L") ran the raw-byte reservoir of 65,536 slots on a stream whose family mixture drifts smoothly with no stage boundaries, with the reservoir admitting items as they arrive, and the learner kept 0.84–0.99 of a twin trained on the same stream, on the three standing seeds. Confirm on the three fresh seeds.

**Runs.** Six lives, detached, launched together:

```
E16_REPLAY=raw E16_CAP=65536 E16_ADMIT=online E16_STREAM=drift python3 exp16_integrated.py T <seed>
E16_STREAM=drift python3 exp16_integrated.py D <seed>
```

for seeds 200003, 300007, 400009. Receipts `exp45/e16_life_T_raw65536_drift_online_<seed>.json` and `exp45/e16_life_D_drift_<seed>.json` (each also writes a `.pt` of final weights); logs `exp45/logs/e16f5_<arm>_<seed>.log`.

**Predictions (copy into `E16F5_PREDICTIONS.md` and commit before launch).**
- P16F5.1 After the last block the learner is at or above 0.80 of the drift twin on all three families (mod-97 operations at k = 12, held compositions by final answer, mod-73 operations at k = 12) in every seed. Standing seeds: 0.97 / 0.93 / 0.84, 0.86 / 0.87 / 0.86, 0.99 / 0.96 / 0.97.
- P16F5.2 The drift twin is at or above 90 / 85 / 90 on family means after the last block.
- P16F5.3 The learner's held compositions after the last block are within 6 points of E16-F3's raw-byte result on the same seed (`exp45/e16_life_T_raw65536_<seed>.json`, in the repository); the composition panels were re-keyed on 8 November, so the band includes about 2 points of panel sampling.

**Tables.** `E16_SEEDS=200003,300007,400009 E16_SUMMARY=SUMMARY_E16F5.md python3 summarize_exp16.py` writes `exp45/SUMMARY_E16F5.md`.

**Deliverables.** `E16F5_PREDICTIONS.md`, the six receipts and weights, the logs, `exp45/SUMMARY_E16F5.md`, `E16F5_REPORT.md` (verdicts per seed with numbers; machine and pace; what went wrong), and one paragraph as `E16F5_REVIEW_SECTION_9_53.md`. About 9 CPU-hours on a four-core container, roughly 3.5 hours of wall time with six processes.

---

## Job 2: E16-RP, the effect of the held-panel seed collision, measured directly

**Why.** A static audit on 8 November found that the held +- composition panel's 600 operand triples were, by a seed collision, all among the taught -> pair's training triples (the operator pairing and the full prompts were still held out). The harness now re-keys the held and within panels by split and pair and draws them operand-disjoint from every taught triple (commit f38281c). Old receipts cannot be re-scored because their final weights were not saved. Measure the effect directly: rerun E16-R3's arm, which is unchanged except for the panels, on the standing seeds, and compare with E16-R3's receipts.

**Runs.** First rename the three existing E16-R3 receipts in your checkout, which are the comparator and must not be overwritten: `exp45/e16_life_T_raw65536_<seed>.json` → `exp45/e16_life_T_raw65536_<seed>_oldpanels.json` for 73129, 93683, 116089 (`git mv`, so the comparator stays in the repository under its new name). Then three lives, detached:

```
E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py T <seed>      # seeds 73129, 93683, 116089
```

Each writes `exp45/e16_life_T_raw65536_<seed>.json` and `.pt`; logs `exp45/logs/e16rp_<seed>.log`.

**Predictions (copy into `E16RP_PREDICTIONS.md` and commit before launch).**
- P16RP.1 (parity) Training is identical to E16-R3 (same birth, stream, reservoir, draws), so every operation panel (`add97`, `sub97`, `max97`, `min97`, `add73`, `sub73`, `max73`, `min73`, and the mod-89 ones) at every checkpoint and every k equals the old receipt's exactly, and the receipt's `settings.birth_sha256` equals the SHA-256 of `exp45/e11_dev_M_<seed>.pt`. Any difference means the run is not the same and the comparison is void.
- P16RP.2 (the effect) On the +- pair, the new panel's final-answer accuracy is within 5 points of the old at every checkpoint from block 4 on, and the mean absolute difference across all four held pairs and those checkpoints is under 3 points (panel sampling). If the +- difference exceeds 5 points in a consistent direction, the collision had an effect, and the record's E16-family composition figures carry that correction.

**Report.** Per seed and checkpoint, the four held pairs and within on the new panels beside the old, with the +- pair singled out, as a table; the parity check as a table (one row per operation panel: identical yes/no). Deliverables: `E16RP_PREDICTIONS.md`, `E16RP_REPORT.md`, the three receipts and weights, the renamed comparators, the logs, and one paragraph as `E16RP_REVIEW_SECTION_9_54.md`. About 2 CPU-hours each on a four-core container, under 2.5 hours of wall time for the three.
