# E16-F6 report: the plus-minus tail, six more seeds of the drifting stream (9 November 2026)

Brief: `OPUS_TASK_E16M2_E16F6_CLOUD_JOBS_20261109.md`, Job 2. Predictions: `E16F6_PREDICTIONS.md`, written and pushed before the births (commit c37c0530).

Seeds:
- **500011, 600013, 700027** on E16-F4's births;
- **800011, 900007, 1000003** on births made by this job.

Every receipt's settings record carries the birth's SHA-256, which equals the SHA-256 of `exp45/e11_dev_M_<seed>.pt`, and records the re-keyed panels.

Tables:
- `exp45/SUMMARY_E16F6.md`, from `summarize_exp16.py`;
- `exp45/SIGNATURE_E16F6.md`, from this job's read-only `e16f6_signature.py`. It holds the twelve-seed list, the correlation, and every checkpoint of the six seeds with all four mod-97 operations and the four pairs, learner and twin.

**Outcome:**
- **P16F6.1, P16F6.2, P16F6.3 and P16F6.4 hold.**
- **P16F6.5 misses.**

On twelve drift seeds the learner fails the 0.80 gate in two: 400009 (E16-F5) and 600013 (this job). 600013 fails under drift as it did under stages. The early signature at checkpoint 2 does not predict the final `+-` pair: the rank correlation is 0.22 against the 0.5 required.

## Verdicts

Scores after the last of six checkpoints:
- **Operations:** family mean at k = 12.
- **Compositions:** mean final-answer accuracy over the four held pairs.
- **Ratios:** to the drift twin of the same seed.

### P16F6.1 (births): each new birth at least 85% at k = 12 on all four development operations — holds

| Seed | add89 | sub89 | max89 | min89 | CPU-s | Verdict |
|---|---:|---:|---:|---:|---:|---|
| 800011 | 92.7 | 92.7 | 97.0 | 97.3 | 2,776 | holds |
| 900007 | 89.0 | 91.3 | 98.3 | 96.3 | 2,777 | holds |
| 1000003 | 90.7 | 94.0 | 95.3 | 96.7 | 2,744 | holds |

The scores are exact answers out of 300 on the sealed panel at k = 12, after the last stage of birth.

**Correction:** the commit that pushed the births (c7bb19ac) states "all >= 89.7%". That figure read the k = 20 column. At k = 12 the lowest is 89.0, for 900007's add89. The verdict is unchanged.

### P16F6.2 (twin): at or above 90 / 85 / 90 in every seed — holds

| Seed | mod-97 ops | Held comp | mod-73 ops | Verdict |
|---|---:|---:|---:|---|
| 500011 | 91.8 | 88.4 | 90.7 | holds |
| 600013 | 94.1 | 94.8 | 93.8 | holds |
| 700027 | 95.7 | 92.5 | 95.1 | holds |
| 800011 | 94.9 | 91.8 | 94.0 | holds |
| 900007 | 93.1 | 92.9 | 92.7 | holds |
| 1000003 | 94.0 | 91.5 | 93.1 | holds |

Seed 500011's twin, at 84.0 on compositions under stages, reaches 88.4 under drift. It is still the lowest twin, and it clears the mod-73 bar by only 0.7.

### P16F6.3 (rate): at least four of six seeds at or above 0.80 of the twin on all three families — holds (five of six)

| Seed | Learner mod-97 / comp / mod-73 | Ratios to the twin | Verdict |
|---|---|---|---|
| 500011 | 87.6 / 78.5 / 91.5 | 0.96 · 0.89 · 1.01 | passes |
| 600013 | 72.8 / 66.8 / 85.8 | **0.77** · **0.71** · 0.91 | **fails** |
| 700027 | 89.9 / 80.5 / 92.2 | 0.94 · 0.87 · 0.97 | passes |
| 800011 | 82.1 / 75.8 / 89.2 | 0.87 · 0.83 · 0.95 | passes |
| 900007 | 86.2 / 77.5 / 90.0 | 0.93 · 0.83 · 0.97 | passes |
| 1000003 | 76.4 / 75.9 / 87.4 | 0.81 · 0.83 · 0.94 | passes, by 0.01 on mod-97 |

### P16F6.4 (seed 600013): misses the composition clause under drift as under stages — holds

| Seed 600013 | Composition ratio | `+-` | `-+` | mod-97 ratio | mod-97 add / sub |
|---|---:|---:|---:|---:|---|
| under stages (E16-F4) | 0.69 | 49.7 | 53.8 | 0.81 | 60.5 / 59.0 |
| under drift (this job) | **0.71** | 41.7 | 47.8 | **0.77** | 52.0 / 47.5 |

Under drift the seed fails on the mod-97 operations as well, which it narrowly passed under stages.

The sequence under drift:
- **Checkpoint 3:** as the compositions enter, the mod-97 addition and subtraction collapse to 28.2 and 24.5, and max and min fall to 67–70.
- **Checkpoint 4:** the compositions peak at 80.0.
- **Checkpoint 5:** the plus-minus pairs fall back to 45 as the mod-73 operations arrive.

The max-min pairs stay at 89. The twin ends at 94.8 on compositions.

This is the signature of E16-F4's 600013 and E16-F5's 400009. It is a seed-level fragility that survives the change of schedule, as the prediction says. The reservoir looks like the passing seeds' (1,049,805 bytes, 55,231 distinct strings, against 1,047,730–1,048,884 and 55,126–55,338 in the other five).

### P16F6.5 (the early signature): misses

| Clause | Result | Verdict |
|---|---|---|
| Spearman, checkpoint-2 mod-97 mean against final `+-`, at least 0.5 | **0.217** over twelve seeds | **misses** |
| every seed below 70 at checkpoint 2 ends with `+-` below 60 | one seed, 400009: 48.0 | holds |
| every seed above 80 at checkpoint 2 ends with `+-` above 65 | 500011 64.2, 800011 63.8, 900007 63.2, 300007 65.0 (not above) | **misses** |

The twelve-seed list, sorted by the checkpoint-2 mod-97 mean:

| Seed | Source | Ckpt-2 mod-97 mean | Final `+-` | Final `-+` | Ratios to twin | All three ≥ 0.80 |
|---:|---|---:|---:|---:|---|---|
| 400009 | E16-F5 | 62.0 | 48.0 | 44.7 | 0.90 · 0.69 · 0.96 | **no** |
| 116089 | E16-S | 71.5 | 72.7 | 71.3 | 0.84 · 0.86 · 0.97 | yes |
| 1000003 | E16-F6 | 75.6 | 68.2 | 61.0 | 0.81 · 0.83 · 0.94 | yes |
| 600013 | E16-F6 | 79.9 | 41.7 | 47.8 | 0.77 · 0.71 · 0.91 | **no** |
| 500011 | E16-F6 | 83.6 | 64.2 | 70.5 | 0.96 · 0.89 · 1.01 | yes |
| 800011 | E16-F6 | 84.8 | 63.8 | 63.8 | 0.87 · 0.83 · 0.95 | yes |
| 900007 | E16-F6 | 86.1 | 63.2 | 70.7 | 0.93 · 0.83 · 0.97 | yes |
| 700027 | E16-F6 | 86.9 | 69.7 | 66.3 | 0.94 · 0.87 · 0.97 | yes |
| 200003 | E16-F5 | 88.0 | 67.2 | 72.5 | 0.93 · 0.86 · 0.96 | yes |
| 93683 | E16-S | 88.4 | 69.8 | 70.0 | 0.93 · 0.87 · 0.96 | yes |
| 73129 | E16-S | 92.1 | 65.8 | 77.0 | 0.97 · 0.86 · 0.99 | yes |
| 300007 | E16-F5 | 92.6 | 65.0 | 70.3 | 0.96 · 0.87 · 0.99 | yes |

**Why checkpoint 2 fails as a predictor.** 600013, the worst seed, stood at 79.9 at checkpoint 2. Its collapse came at checkpoint 3, to 47.2. Checkpoint 2 caught the dip in 400009 and missed it in 600013.

**Exploratory, chosen after seeing the data, not a prediction:** with the lowest mod-97 mean over checkpoints 2–4 in place of checkpoint 2, the rank correlation is 0.40. The two failing seeds then have the two lowest values (47.2 and 62.0). With checkpoint 3 alone it is 0.21.

Even the better measure sits below the pre-registered bar. Among the ten passing seeds, the final `+-` varies only between 63 and 73. In that range there is little for an early measure to rank, and seed-to-seed variation dominates.

## Across twelve drift seeds

- **Failure rate:** two of twelve seeds fail the 0.80 gate, 400009 and 600013. Both fail the same way: the plus-minus pairs end at 42–48, the max-min pairs stay above 81, and the twin is at 94–95.
- **Passing seeds:**
  - lowest composition ratio 0.83, in four seeds;
  - final `+-` from 63.2 to 72.7.
- **Raw-byte lives so far:** counting E16-F3, E16-F4, E16-S, E16-F5 and this job, three of the 21 raw-byte reservoir lives fail (nine under block stages, twelve under drift). Seed 600013 fails twice, once under each schedule.

**A caveat on pooling the twelve seeds.** E16-RP found that sign-rule receipts diverge between the owner's machine and a cloud container from the first update, with `+-` moving by up to 16 points between two runs of one seed. The receipts carry no machine field:
- **This container:** E16-F5's and this job's nine seeds ran here.
- **Elsewhere:** E16-S's three seeds (73129, 93683, 116089) ran in another session.

If those three ran on the owner's machine, a swing of that size from the machine alone is larger than the differences P16F6.5's thresholds draw on. That would weaken any single-seed reading among them, though not the rank correlation's miss on twelve. Across cloud containers the runs reproduce exactly (next section).

## Cross-check against run B: bit-identical across two cloud containers

Another cloud session ran the same job on the same seeds, and the coordinating session filed it as `E16F6_RUN_B/` (commit c390eff1). Its README says the two runs are independent because "births and sign-rule receipts do not reproduce across machines". They are not independent. Compared after this job finished:

- **All 12 receipts are identical.** Every score at every checkpoint is the same, for learner and twin, on all six seeds.
- **All 3 births are identical.** Every point of their training curves matches, and the final losses agree to all 17 printed digits (for example 0.033914950538019184 for 800011).
- **Only the CPU times differ.**

The two machines:

| | This run | Run B |
|---|---|---|
| CPU | Intel Xeon, 2.3 GHz, 4 cores | Intel Xeon, 2.1 GHz, 4 cores |
| Python | 3.11.18 | 3.13 |
| NumPy | 2.4.6 | 2.5.3 |
| PyTorch | 2.15.0 CPU | 2.15.0 CPU |

So the harness reproduces bit for bit across these two cloud containers, despite different Python and NumPy versions and a different clock speed. E16-RP's divergence was between the owner's machine (E16-R3) and a cloud container. The likely boundary is the hardware or PyTorch build class, not "any two machines".

This narrows E16-RP's finding: within the cloud containers used so far, a rerun of a seed is a repeat, not a new sample. Run B therefore adds no statistical weight to this job. It confirms the receipts.

## Machine and pace

| Item | Value |
|---|---|
| Machine | as E16-F to E16-M2: 4 cores, Python 3.11.18, torch 2.15.0+cpu, NumPy 2.4.6, one thread per process |
| Harness | births and lives at commit c37c0530, which contains 1470401b; no harness file changed by this job |
| Births | 2,744 / 2,776 / 2,777 CPU-s; 18:35:55 to 20:19:57–20:20:57 UTC, beside six lives |
| Learner lives | 4,895 / 4,935 / 4,947 (500011 / 600013 / 700027), 4,703 / 4,732 / 4,726 (800011 / 900007 / 1000003) CPU-s |
| Twin lives | 4,244 / 4,325 / 4,241, 4,160 / 4,235 / 4,179 CPU-s |
| Wall | 18:35:55 to 23:03:09 UTC, 4 h 27 min. Nine processes on four cores for 1 h 45 min, then twelve |

CPU figures are each log's last line.

**Forecast against measured:**

| Item | Forecast | Measured |
|---|---|---|
| Birth | 3,100 CPU-s | 2,760 CPU-s |
| Learner life | 4,690–4,740 CPU-s | 4,700–4,950 CPU-s |
| Twin life | 4,060–4,200 CPU-s | 4,160–4,330 CPU-s |
| Total | about 62,000 CPU-s | 62,619 CPU-s |
| Wall | 4.3 h, up to 5 h (the brief said about 5 h) | 4.45 h |
| Phase-1 lives done | about 2.5 h after launch | 3.0–3.4 h |
| Last lives done | 4.3–5 h after launch | 4.45 h |

The phase-1 lives ran longer than forecast because they shared the cores with twelve processes rather than nine once phase 2 began. The total was not affected.

## Notes and what went wrong

- **A container restart cut off the first dry run.** At 18:13 UTC the learner's dry run was killed after four blocks. The twin's dry run never started, because the second detached command in the same tool call lost its working directory. Both were run again from a script at 18:30, passed seven stages, and are archived in `exp45/archive_e16f6_dry/`. The predictions file records this, and sets the policy for a restart during the job (relaunch a killed, unfinished life from its birth and say so). No restart happened during the job. All three births and all twelve lives exited 0 on their first launch.
- **The births commit message misstates the minimum birth score** (89.7, the k = 20 figure). The k = 12 figures are above; 89.0 is the lowest.
- **Logs:** written to the session's scratch folder and copied to `exp45/logs/e16f6_*.log` at commits and at the end.
- **Branch:** pushed directly to `codex/minifly-consolidated-20261023`, as the brief allows. Partial receipts were pushed at the births, at block 5 of phase 1, and at the end of phase 1.

## Files

- **Predictions:** `E16F6_PREDICTIONS.md`.
- **Births:** `exp45/e11_dev_M_{800011,900007,1000003}.{pt,json}`.
- **Receipts:** `exp45/e16_life_T_raw65536_drift_online_<seed>.json` and `exp45/e16_life_D_drift_<seed>.json` for the six seeds.
- **Final weights:** the matching `.pt` files.
- **Dry runs:** `exp45/archive_e16f6_dry/`.
- **Logs:**
  - `exp45/logs/e16f6_birth_<seed>.log`;
  - `exp45/logs/e16f6_{T,D}_<seed>.log`;
  - `exp45/logs/e16f6_lives_wall.txt`, which records exits and phase starts.
- **Tables:** `exp45/SUMMARY_E16F6.md`; `e16f6_signature.py` and `exp45/SIGNATURE_E16F6.md`.
- **Review paragraph:** `E16F6_REVIEW_SECTION_9_59.md`.
