# E16-RP report: the held-panel seed collision, measured directly (9 November 2026)

**Sources.**
- Brief: `OPUS_TASK_E16F5_E16RP_CLOUD_JOBS_20261109.md`, Job 2 (commit 923b64a0), with the coordinating session's go-ahead and amendment (commit 1cccb4d).
- Predictions: `E16RP_PREDICTIONS.md`, committed at 807b61f1 before launch. The same-weights amendment (P16RP.3) was committed at 634fd746 before any life's weights were scored.
- Harness: `exp16_integrated.py` at 432e90fb, unchanged by this job.
- Branch: `claude/vibrant-gauss-tops5f`.

**Tables.**
- `exp45/SUMMARY_E16RP.md` (`e16rp_tables.py`): the cross-machine comparison and the parity check.
- `exp45/E16RP_SAME_WEIGHTS.json` (`e16rp_same_weights.py`): the same-weights measurement.

## Outcome

| Prediction | Verdict |
|---|---|
| P16RP.1 (parity) | **Misses as locked**, in every seed. |
| P16RP.2 (old vs new panels across runs) | **Void as locked.** |
| P16RP.3 (old vs new panels on the same final weights) | **Holds in two seeds.** Misses in 93683 on the +- clause: +6.8 points, limit 5. |

The mean-absolute-difference clause of P16RP.3 holds in every seed.

On the same weights, the colliding +- panel scores −1.0 / −6.8 / +0.8 points relative to the clean one; the old panel is *lower* in two seeds. If the collision inflated held-composition scores, the old panel would score higher. It does not, in any seed by more than 0.8 points. So there is no evidence that it inflated the E16-family composition figures at block 6. The 6.8-point gap in seed 93683 runs the other way and is about 2.5 standard errors of the difference between two independent 600-item panels (about 2.7 points at 65% accuracy).

The receipts are reproducible on one machine and not across machines; that finding has its own section below.

## Runs

| Item | Value |
|---|---|
| Lives | Three, detached with `setsid nohup`, launched together at 14:13 UTC |
| Command | `E16_REPLAY=raw E16_CAP=65536 python3 exp16_integrated.py T <seed>`, seeds 73129, 93683, 116089 |
| Finished | 15:40 UTC, all six blocks |
| CPU per life | 4,982–5,010 CPU-s |
| Wall time per block | about 14.5 minutes |
| Errors | none |
| Outputs | Receipts `exp45/e16_life_T_raw65536_<seed>.json`, final weights `.pt` and logs `exp45/logs/e16rp_<seed>.log` |
| Comparators | E16-R3's receipts, renamed with `git mv` before launch to `exp45/e16_life_T_raw65536_<seed>_oldpanels.json` |
| Machine | Four cores, Intel Xeon at 2.1 GHz, Python 3.13, PyTorch 2.15.0 (CPU), NumPy 2.5.3 |

## Finding: sign-rule receipts reproduce on one machine, not across machines

**What the lives showed.** At stage 0 (the birth, before any update) each new receipt equals E16-R3's on every operation panel. The receipt's `settings.birth_sha256` equals the SHA-256 of `exp45/e11_dev_M_<seed>.pt` in all three seeds. From block 1 every operation panel differs:

- the training loss at block 1 differs by 1–2% (for example 0.03993 against 0.04034 in seed 73129);
- the panels by up to 56 items of 400 (`min73` at k = 0 in seed 116089: 142 against 86).

**The same-machine test.** I ran E16-R3's own harness (commit 7703d6c) and the current one (432e90fb) side by side on this container, six blocks of 64 updates on seed 73129, in an isolated scratch copy.

- They are **bit-identical**: the same training loss to every printed digit at all six blocks, zero operation-panel differences, the same reservoir.
- The harness change did not alter training. That is also what the source shows: the training-triple formula and every imported module are unchanged since 7703d6c, and scoring draws no randomness and writes no state.
- The divergence comes from the machine or the PyTorch build. E16-R3 ran on the owner's machine. The sign rule amplifies rounding differences from the first update.
- Evidence: `exp45/archive_e16rp_parity/`, both 64-update receipts and both logs.

**Consequences.**
- A parity clause like P16RP.1 is valid only on the machine that produced the comparator.
- Every cross-machine comparison in this program is a comparison between different runs, including the fresh-seed confirmations. That is what those comparisons were meant to be anyway.
- For the panel question, only a same-weights measurement is clean.

How large the run-to-run drift is shows in the held compositions after block 6, held-pair mean, final answer:

| Seed | E16-R3, owner's machine, old panels | This run, new panels |
|---|---:|---:|
| 73129 | 80.2 | 79.8 |
| 93683 | 74.8 | 80.1 |
| 116089 | 80.1 | 76.2 |

Seed by seed, the +- pair moves by up to 16 points between the two runs, and that is not a panel effect. On the same weights, seed 93683's +- difference is 6.8, not 16.2.

### P16RP.1 (parity): misses in every seed

| Panel | 73129 | 93683 | 116089 |
|---|---|---|---|
| add97, sub97, max97, min97 (k = 0, 8, 12, 20) | identical at stage 0, differ from stage 1 | same | same |
| add73, sub73, max73, min73 (k = 0, 8, 12, 20) | identical at stage 0, differ from stage 1 | same | same |
| add89, sub89, max89, min89 (k = 12) | identical at stage 0, differ from stage 1 | same | same |
| silenced cell at k = 12, all operations | identical at stage 0, differ from stage 1 | same | same |
| `settings.birth_sha256` equals the file's | yes | yes | yes |

The first difference per panel is listed in `exp45/SUMMARY_E16RP.md`.

### P16RP.2 (cross-run panel comparison): void as locked

The predicted figures are reported for reference only; they mix the panel change with run-to-run drift.

| Seed | +- new − old, stages 4 / 5 / 6 | Max abs +- (limit 5) | Mean abs diff, four held pairs, stages 4–6 (limit 3) |
|---|---|---:|---:|
| 73129 | −0.2 / −1.2 / +1.5 | 1.5 | 1.79 |
| 93683 | +3.2 / +11.5 / +16.2 | 16.2 | 4.14 |
| 116089 | +2.0 / +0.8 / −6.2 | 6.2 | 2.15 |

## The same-weights measurement (P16RP.3): the collision's effect at block 6

**Method.** Read-only, no training. Each life's final weights (`exp45/e16_life_T_raw65536_<seed>.pt`) were scored on two panel sets:

- **Old panels:** rebuilt exactly as harness 7703d6c's `panels()` built them. Held pairs: `triples(pair, seed + 1, 600)`. Within: `triples(p, seed + 2, 50, exclude=taught[p])`.
  - Checked by `e16rp_check_old_panels.py`: scoring the births on the rebuilt old panels reproduces E16-R3's stage-0 composition counts exactly, final and exact, all five panels, all three seeds.
- **New panels:** the current harness's `panels()`.

**The store.** The E8 exact-key store's contents at block 6 were not saved, so both panel sets were scored on the weights alone. On the new panels, the no-store scores equal the receipts' block-6 scores made with the store, within 0.1 point on every panel in every seed. The store contributes nothing on these panels, so its absence costs nothing here.

**Disjointness counts.**

| Seed | Old +- triples in the taught -> set | New panels' triples in any taught set | Old -+ / >< / <> / within triples in any taught set | Old and new panels sharing a triple |
|---|---:|---|---|---|
| 73129 | 600 of 600 | 0 (all five panels) | 17 / 17 / 11 / 12 | none |
| 93683 | 600 of 600 | 0 | 19 / 13 / 15 / 10 | one triple in -+ |
| 116089 | 600 of 600 | 0 | 14 / 17 / 17 / 12 | none |

The old non-+- panels overlapped the taught triples only by chance, 10–19 of 600 each; the old +- panel overlapped entirely.

**Results on the same weights** (percent; diff = new − old):

| Seed | Panel | Final, old | Final, new | Final diff | Exact, old | Exact, new |
|---|---|---:|---:|---:|---:|---:|
| 73129 | **+-** | 64.0 | 65.0 | **+1.0** | 62.0 | 61.7 |
| | -+ | 71.7 | 73.0 | +1.3 | 70.3 | 71.7 |
| | >< | 89.0 | 90.2 | +1.2 | 86.2 | 86.0 |
| | <> | 91.7 | 90.8 | −0.8 | 88.2 | 87.5 |
| | within | 84.5 | 83.8 | −0.7 | 80.5 | 79.8 |
| 93683 | **+-** | 65.7 | 72.5 | **+6.8** | 63.5 | 69.7 |
| | -+ | 63.7 | 63.7 | 0.0 | 62.5 | 62.0 |
| | >< | 89.2 | 91.0 | +1.8 | 86.0 | 88.3 |
| | <> | 92.5 | 93.3 | +0.8 | 87.0 | 89.8 |
| | within | 83.3 | 81.3 | −2.0 | 80.0 | 78.8 |
| 116089 | **+-** | 63.8 | 63.0 | **−0.8** | 62.0 | 62.2 |
| | -+ | 64.5 | 62.2 | −2.3 | 63.7 | 60.7 |
| | >< | 89.7 | 89.5 | −0.2 | 83.7 | 85.0 |
| | <> | 92.5 | 90.3 | −2.2 | 89.5 | 86.3 |
| | within | 81.0 | 80.3 | −0.7 | 77.8 | 76.7 |

| Seed | +- within 5 | Mean abs diff, four held pairs, final (limit 3) | P16RP.3 |
|---|---|---:|---|
| 73129 | yes (+1.0) | 1.08 | holds |
| 93683 | **no (+6.8)** | 2.37 | **misses** |
| 116089 | yes (−0.8) | 1.38 | holds |

**Reading.**
- The collision put every operand triple of the held +- panel into the taught -> pair's training set. The pairing and the prompts stayed held out.
- On the same weights, that panel does not score higher than a clean one. The clean panel is higher by 1.0 and 6.8 points in two seeds and lower by 0.8 in the third.
- Seed 93683's +6.8 is the one excursion beyond the band.
  - Its direction is against contamination: the old, colliding panel scores lower.
  - It is not mirrored in the exact scores of the other pairs.
  - It is about 2.5 standard errors of the difference between two independent 600-item samples. With 15 panel comparisons in this table, one such excursion is not surprising, though it is not ruled out as real either.
- Averaged over the three seeds, the held-pair mean on the clean panels is 78.7, and on the old panels 78.2.
- On this evidence the record's E16-family composition figures need no correction for the collision at block 6. Blocks 4–5 cannot be measured this way, since only final weights were saved.

## What went wrong, or is limited

- **P16RP.1 failed for a reason outside the experiment.** The receipts do not reproduce across machines; see the finding above. Detected at block 1, at 14:29 UTC. The lives were kept running and the cause was isolated with the same-machine test before anything else was changed.
- **One inaccuracy in the locked amendment.** It says the lives were "in block 2–3" at 14:41 UTC; they were in block 2, finishing it at 14:44. Nothing else in the amendment depends on this, and the file is not edited.
- **The same-weights measurement covers block 6 only, and one run per seed.** Panel sampling (about 2.7 points SE on a difference at these accuracies) is the same size as the effect being bounded.
- **Twin comparisons.** The new receipts' ratios to the twin compare different machines, and the twin was scored on the old panels; they are not reported here.
- **Two extra processes.** The two 64-update test runs ran alongside the three lives for about 5 minutes, which slowed block 2 slightly. They did not touch the lives' files.
