# E20-B report: windowless reference networks under backprop (stage 2, full dose; 11 November 2026)

**Job:** E20-B under `OPUS_TASK_E20_COMPOSITION_FROM_MEMORY_20261110.md`, Job B, with amendments 1 to 16. Stage 1 (implementation and reduced-dose qualification) is recorded in `exp45/e20b_qual/E20B_STAGE1_RECORD.md` and was accepted. Stage 2 was released on the coordinator's go, and the owner chose to run all of it on this container.

**What ran:** four networks (G0 to G3) × three seeds (73129, 93683, 116089), giving twelve births and twelve lives at full dose. Each birth is 24,576 updates and each life is 6 × 6,144 updates, all batches of 64. All 24 ran on one host (`vm`: four cores, torch 2.15.0+cpu, Python 3.13.18), in waves of three processes, one thread each.

**Sources:**
- The runner is `exp20_reference.py` and `arch20.py`, unchanged since the qualified commit `0e131a29`.
- Every run after the restart also used the resumable wrapper `exp20_resume.py` (see "Interruption and resumability").

**Every number is a whole answer** (`exact`: all six bytes, end symbol included), in percent, unless it is marked final.

## Completion and verification (amendment 16)

`summarize_e20b.py` reads the receipts only. It wrote `exp45/e20b/E20B_SUMMARY.json` and `exp45/e20b/E20B_TABLES.md`.

**All 24 completion records are verified.** For each run:
- the completion record is present;
- the optimizer step count is complete (24,576 per birth, 36,864 per life);
- every life has all 7 stage records;
- the SHA-256 of the weights file equals the recorded `weights_sha256`;
- the tensor-content hash of the loaded weights equals the recorded `tensors_sha256`.

The verification rows are in `E20B_SUMMARY.json` → `verification`, and `all_24_verified` is `true`.

Every receipt records:
- the command and environment (`provenance.command`, `environment_at_start`, `environment_set_by_runner`, `host`, `git`);
- the hashes of the eight source files.

Every life admitted its own network's birth for the same seed by path, hash and source hashes. All launch lines are in `exp45/e20b/logs/LAUNCHES.txt`.

## Parameters and cost

| Net | Core | Parameters | Birth CPU-s (mean of 3) | Life CPU-s (mean of 3, scoring included) | Stage-1 forecast, birth / life |
|---|---|---:|---:|---:|---|
| G0 | arch17 bank2 (matched reference) | 1,125,265 | 3,412 | 6,535 | 3,500 / 5,600 |
| G1 | GRU, 256 units | 1,124,113 | 2,918 | 4,587 | 2,500 / 3,900 |
| G2 | LSTM, 212 units | 1,125,873 | 2,592 | 4,045 | 2,300 / 3,600 |
| G3 | bank2 + `lb` (64×512, no bias; +32,768) | 1,158,033 | 3,389 | 5,205 | 2,800 / 4,400 |

- **Total:** 98,048 CPU-seconds for the 24 runs that produced the receipts, against a forecast of 85,000 to 90,000 (about 10% over) and the brief's estimate of about 100,000.
- **Extra cost not in the total:** the three interrupted G0 lives (below).
- **G3 against G0 (amendment 15):** G3 adds 32,768 parameters (+2.9%). It costs about the same as G0 per birth (−0.7%) and 20% less per life. Its life scoring is cheaper because G3 is answered better, and that changes the output lengths.

## Interruption and resumability (amendment 7)

**What was interrupted:** the container restarted between 06:29 and 06:38 UTC. That killed the first three full-dose G0 lives, which were run with `exp20_reference.py`.

**What was kept:**
- Their partial receipts (stages 0 to 4, through F2.2) and logs are kept as written in `exp45/e20b/interrupted_20261111/`, with a README.
- They are not used for any number here.

**What changed in response:**
- I stopped and reported.
- The owner asked for the optimizer state to be saved and authorized resuming once that was tested.
- `exp20_resume.py` saves the model, the AdamW state and the loop state atomically.
- On resume it rebuilds the deterministic data streams and fast-forwards them.
- `exp45/e20b_resume_check/RESUME_CHECK.json` shows the test: a G3 birth killed with SIGKILL twice and a G0 life killed four times ended bit-identical to the qualified uninterrupted runs.

**How the G0 lives were rerun:**
- The three G0 lives were rerun from the start, from the same verified births. A life with no checkpoint cannot be resumed.
- The G0 births had completed before the restart and were not rerun.

**Since then:** no run was interrupted. Every receipt records zero resumes.

## Results at the end of life (stage 6, after F3.2)

`within` = new operand triples of taught operator pairs (600). Taught = the taught triples themselves (600). The held pairs have 600 each. Plus-minus = `+-` and `-+` pooled (1,200). The hard mean is the birth's mean of `add89` and `sub89` at k = 12.

| Net | Seed | Birth hard mean | Taught | within | `+-` | `-+` | `><` | `<>` | Plus-minus pooled | F1 (mod 97) | F3 (mod 73) | mod 89 after life | Strict F1 / F3 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| G0 | 73129 | 93.2 | 52.3 | 47.3 | 1.0 | 2.0 | 16.0 | 26.0 | 1.5 | 91.3 | 89.4 | 37.0 | 91.0 / 88.4 |
| G0 | 93683 | 92.8 | 62.8 | 50.8 | 0.7 | 0.3 | 23.3 | 22.5 | 0.5 | 93.9 | 88.2 | 46.8 | 95.0 / 87.2 |
| G0 | 116089 | 92.8 | 64.3 | 53.3 | 11.8 | 3.2 | 23.0 | 24.2 | 7.5 | 94.0 | 90.1 | 41.2 | 92.6 / 90.1 |
| G1 | 73129 | 95.3 | 88.0 | 89.8 | 73.8 | 73.3 | 93.8 | 84.8 | 73.6 | 97.2 | 92.7 | 53.6 | 96.5 / 92.1 |
| G1 | 93683 | 99.8 | 90.5 | 87.0 | 76.3 | 81.2 | 91.2 | 92.5 | 78.8 | 95.5 | 93.6 | 51.9 | 95.9 / 94.0 |
| G1 | 116089 | 97.3 | 91.5 | 91.2 | 74.5 | 75.7 | 67.5 | 86.8 | 75.1 | 98.2 | 96.7 | 56.8 | 96.9 / 96.7 |
| G2 | 73129 | 95.2 | 89.5 | 87.8 | 77.0 | 80.2 | 79.3 | 78.8 | 78.6 | 98.2 | 93.2 | 69.2 | 96.3 / 92.8 |
| G2 | 93683 | 96.8 | 89.3 | 88.8 | 65.7 | 78.7 | 73.2 | 76.5 | 72.2 | 96.6 | 94.8 | 44.8 | 96.4 / 94.1 |
| G2 | 116089 | 93.5 | 85.0 | 85.7 | 68.0 | 70.3 | 73.8 | 73.0 | 69.2 | 96.4 | 92.6 | 62.1 | 95.3 / 91.9 |
| G3 | 73129 | 98.8 | 82.0 | 76.5 | 14.5 | 25.8 | 66.5 | 59.0 | 20.2 | 96.2 | 94.8 | 26.1 | 94.3 / 94.4 |
| G3 | 93683 | 93.2 | 75.7 | 69.3 | 0.8 | 1.3 | 63.2 | 72.5 | 1.1 | 95.1 | 93.5 | 39.8 | 94.9 / 92.0 |
| G3 | 116089 | 98.0 | 81.8 | 80.3 | 1.0 | 1.0 | 47.5 | 71.5 | 1.0 | 98.3 | 97.2 | 70.2 | 97.2 / 97.4 |

`exp45/e20b/E20B_TABLES.md` has the rest:
- every birth curve row;
- at every one of the seven life checkpoints: taught instances, `within` and each of the four held pairs, whole and final; the pooled plus-minus figure; the family means at k = 12 (F1, F3, mod 89); and the strict-panel means.

## The four predictions, judged as registered

These follow the operational reading at the end of `E20B_PREDICTIONS.md`, which the coordinator accepted before stage 2.

| Prediction | Values (seeds 73129 / 93683 / 116089) | Verdict |
|---|---|---|
| **P20.7** G1 and G2 born at a hard mean ≥ 85 in at least two seeds each | G1: 95.3 / 99.8 / 97.3. G2: 95.2 / 96.8 / 93.5 | **Holds** (all three seeds, both networks) |
| **P20.8** at least one of G1, G2 ends ≥ 85 on new instances of taught pairs in at least two seeds | G1: 89.8 / 87.0 / 91.2. G2: 87.8 / 88.8 / 85.7 | **Holds** (both networks, all three seeds; G2's 116089 is 85.7, close to the line) |
| **P20.9** at least one of G1, G2, G3 ends ≥ 40 on the plus-minus pairs in at least two seeds (held at about 50%) | G1: 73.6 / 78.8 / 75.1. G2: 78.6 / 72.2 / 69.2. G3: 20.2 / 1.1 / 1.0 | **Holds** (G1 and G2, all three seeds; G3 misses in all three) |
| **P20.10** G3 ends ≥ 15 points above G0 on new instances of taught pairs in at least two seeds (held at about 50%) | G3 − G0: +29.2 / +18.5 / +27.0 (G3 76.5 / 69.3 / 80.3; G0 47.3 / 50.8 / 53.3) | **Holds** (all three seeds) |

## Selection (amendment 13)

| Network | Seeds with `within` ≥ 85 and plus-minus ≥ 40 | Qualifies | Three-seed plus-minus mean |
|---|---|---|---:|
| G3 | none | no | 7.4 |
| G1 | 73129, 93683, 116089 | yes | 75.8 |
| G2 | 73129, 93683, 116089 | yes | 73.3 |

G3 does not qualify. G1 and G2 both qualify, so the rule takes the one with the higher three-seed mean on the plus-minus pairs. **G1 (the GRU) is the windowless candidate from this job**, at 75.8 against G2's 73.3. The margin is 2.5 points, and it is not an exact tie, so the tie rule does not apply.

## Observations outside the registered predictions (no verdict attached)

- **G3's plus-minus result is unstable across seeds.** It is 20.2 in seed 73129 and about 1 in the other two, while `><` and `<>` stay at 47 to 73. The added connection lifts `within` over G0 in every seed. It does not carry the plus-minus pairs.
- **The birth's own mod-89 families fall during the life** in every network (family mean at k = 12: 94.3 to 99.8 at the start of the life, 26.1 to 70.2 after F3.2). The life's stream carries the F1, F2 and F3 families only, so this is forgetting of the birth tasks. No prediction concerns it.
- **G1 and G2 score close to `within` on the held `><` and `<>` pairs** in most seeds: 67.5 to 93.8 against `within`'s 85.7 to 91.2. G0 does not (16 to 26).

## What the results can and cannot say (amendments 6, 14, 15)

- **P20.10 measures the added connection as a package:** 32,768 weights (a 64-by-512 matrix with no bias) and the work they cost. G3 costs about the same as G0 per birth and less per life (see Parameters and cost). The result does not show that the placement of the feedback matters; that would need a control with as many extra weights elsewhere.
- **"Windowless" describes the network's input.** The registered decoder keeps the prompt and the bytes produced so far, and reruns the network for each output byte. Nothing here measures single-pass inference or its memory cost.
- **This is a development feasibility result at evidence level E2.** It supports no claim of reuse or promotion. Backprop is a reference here, not a candidate learning rule.
- **The design limits:**
  - one host, three seeds, one dose, one teaching protocol (arm D, staged);
  - the G1/G2 design choice recorded in stage 1: arch17's `l1` input layer and readout are kept, and only the recurrent core is changed.

## Files

| What | Where |
|---|---|
| Predictions and selection rule (committed before any stage-2 run) | `E20B_PREDICTIONS.md` |
| This report; the review paragraph | `E20B_REPORT.md`; `E20B_REVIEW_SECTION_9_78.md` |
| Stage 1 record and check | `exp45/e20b_qual/E20B_STAGE1_RECORD.md`, `exp45/e20b_qual/E20B_CHECK.json` |
| Runner, networks, resumable wrapper, summary | `exp20_reference.py`, `arch20.py`, `exp20_resume.py`, `summarize_e20b.py` |
| Resume test | `exp45/e20b_resume_check/RESUME_CHECK.json` |
| Births and lives: receipts, weights, logs | `exp45/e20b/e20_dev_<net>_<seed>.{json,pt}`, `exp45/e20b/e20_life_<net>_<seed>.{json,pt}`, `exp45/e20b/logs/` |
| Verification and full tables | `exp45/e20b/E20B_SUMMARY.json`, `exp45/e20b/E20B_TABLES.md` |
| Interrupted G0 lives, kept as written | `exp45/e20b/interrupted_20261111/` |
