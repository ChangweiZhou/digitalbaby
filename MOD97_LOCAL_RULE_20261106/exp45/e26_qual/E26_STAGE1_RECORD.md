# E26 stage 1: implementation and reduced-dose qualification (12 November 2026)

**Job:** E26, stage 1 only, under `OPUS_TASK_E26_GRU_UNDER_THE_SWEEP_20261111.md`, revision 2. The work started from `codex/minifly-consolidated-20261023` at `17a5fa35`, merged into the branch `claude/adoring-cori-b07aog`. Nothing has gone to the default branch, and no full-dose birth or life has been run.

**Result in one line:** every check passes except check 1. Check 1 fails in one case of 18, by a logit difference of 1.14e-5 against the 1e-5 limit. That is reported below and nothing was changed to make it pass. The decisive check 2b passes exactly: all 70 scored blocks of E20-B's three end-of-life GRU files are equal to their receipts.

## What is returned

| Item | Where |
|---|---|
| Runner and network: new files only | `arch26.py` and `exp26_candidate.py`, committed at `360e1b8d`. Every run used these sources; their hashes are in each receipt and check record. The `git.commit` field of the receipts reads `2644340b`, a later commit that adds only logs and check records, and the receipts record no modified sources. |
| Check record | `exp45/e26_qual/E26_CHECK.json`: the summary plus every part. The parts are also kept as `check_<part>.json` |
| Predictions, copied unchanged | `exp45/e26_qual/E26_PREDICTIONS.md`, committed at `c70ed430` before any run |
| The command and environment of every run | `exp45/e26_qual/run_qual.sh`, `exp45/e26_qual/logs/LAUNCHES.txt`, and the `provenance` of each receipt and check record. Provenance holds the command, the environment at start, the environment set by the runner, the host (machine, platform, Python, torch, threads, cores), git, and the source hashes |
| Reduced-dose receipts and logs | `exp45/e26_qual/e26_{dev,life}_{S,B}_*_73129.json` and `exp45/e26_qual/logs/`. The weights files are kept on the container and not committed (`.gitignore`); their hashes are in the receipts |

**What was not edited:** `exp16_integrated.py`, `arch17.py`, `arch20.py`, `exp20_reference.py`, `exp11_cell.py`, `exp3_state.py` and `e25_alignment.py`. They are imported only.

**The host:** `vm`, x86_64 Linux, four cores, Python 3.13.18, torch 2.15.0+cpu, one thread per process, at most three processes at a time. `E16_RESUME` and every other `E16_`/`E11_` variable were unset, except the two dose variables of the reduced runs. `exp20_reference` refuses any other at import.

**One error in the launch log:** its header line says "1 cores". It took `nproc`, which reads the `OMP_NUM_THREADS=1` the launcher sets just before. The receipts record `cores: 4` from Python's `os.cpu_count()`.

## How the update is built (the brief's "The update, exactly")

**The step function.** `exp26_candidate.step(m, opt, items, local)` is the only update:

```
x, y, mask = batch_tensor(items)
logits = arch26.forward(m, x, local, True)[0]
loss = cross_entropy(logits[mask], y[mask])
opt.zero_grad(); loss.backward(); opt.step()
```

- It serves both arms, in births and lives, with `local = True` for arm S and `False` for arm B.
- `E.update` is never called.

**The network.** `arch26.forward(m, x, local, through_time, keep=False, silence=False)` takes the harness's full signature.
- It runs arch20's own G1 module (`arch20.NetRNN("gru")`): the same parameter names and the same initial values from the same seed. So E20-B's G1 files load unchanged.
- The GRU is written out step by step in torch's form, with every map through `exp3_state.lin`. That covers `l1`, the stacked input map, the stacked recurrent map at every step, `l2`, `lm` and `l3`, with one scale per stacked matrix.
- Every item starts from a zero state.
- With `silence`, the readout sees a zero state.

**The optimizer.** `exp16_integrated.adamw16` at rate 0.001, betas (0.9, 0.98) and decay 0.25. The embedding table is in its own group with no decay. The groups are read back from the optimizer, asserted, and recorded in every receipt.

**One extension beyond the brief's loop.** `through_time=False`, which detaches the state at every step, exists only so that the harness's rule update can be run as the fixture that must fail. E26 never uses it.

## Results of the checks (`E26_CHECK.json` → `summary`)

| Check | Result |
|---|---|
| **1. The step** | **FAIL in 1 of 18 cases** (below). In every case the gradients agree with the library's to at most 3.7e-6 relative (limit 1e-4). The two-gates-swapped version fails in every case, by 0.03 to 33. The written-out step is bit-identical to `e25_alignment.forward_rnn` and `e25_alignment.grads` in logits and gradients in all 18 cases. |
| **2. The two paths** | **Pass** in 12 cases: three seeds, untrained and E20-B birth file, birth batch and mixed batch. The forward is bitwise equal under both paths. Under the sweep the top layer's gradient (`l3.weight`, `l3.bias`) equals the exact one (difference 0.0), and the recurrent matrix's does not (difference 6e-4 to 0.16). Under both paths every parameter and every active embedding row receives a finite, non-zero gradient, and the BOS row receives none. |
| **2a. One real step in each arm** | **Pass.** From E20-B's G1 birth file of seed 73129 and that seed's first birth batch, one production step was compared with the same AdamW given the gradient from `e25_alignment.grads` with the same flag. This was done from a fresh optimizer and again after two warm-up steps (non-zero moments). The result is equal tensor for tensor in both arms and both starts (maximum difference 0.0). After the step the two arms' weights differ in every tensor below the top layer. From a fresh optimizer the top layer is equal across the arms, as its gradient is. **The fixture fails the check as required:** `E.update(m, opt, items, "T")` stops at the trace routine ("no trace routine is defined for the GRU"). With the trace disabled, its step differs from arm S's in 7 tensors (largest difference 0.002), because the harness cuts credit at every step. `E.update(..., "D")` gives exactly arm B's step; this is reported only, and it is never called. |
| **2b. The scorer** | **Pass.** The signature is the harness's (`m, x, local, through_time, keep=False, silence=False`). `exp16_integrated.score` calls `arch26.forward`, which was observed with `silence` passed through both ways. The three stub fixtures of E20-B's stage 1 pass: the whole answer, the right ending after a wrong intermediate, and no end symbol. With `silence=True` the logits equal arch20's silenced logits and a zero-state readout exactly (difference 0.0). **E20-B's end-of-life G1 file of each seed**, loaded into the written-out network and scored by the production scorer, gives **exactly the counts in its own receipt in all 70 blocks** (`n`, `exact` and `final`). The blocks cover every operation panel at every k, the composition panels, the silenced panels, the strict panels and the taught panel. `within` gives 539 / 522 / 547 and the plus-minus pairs 443 + 440 / 458 + 487 / 447 + 454, all equal to the receipts. |
| **3. The data** | **Pass.** Seed 73129 at the reduced dose: the digest of every batch fed to the learner is equal between the arms, at every birth checkpoint and every life stage. That is 2,048 birth batches (`4f622d94…`) and 1,536 life batches (`94806656…`). |
| **4. The optimizer** | **Pass.** All four reduced receipts (birth and life, both arms) record, read back from the groups: decay 0.25 on the 12 tensors other than the embedding table, and 0.0 on `book.weight` alone. |
| **5. Causality, rows, reset** | **Pass** on a batch of 48 rows of 11 lengths (10 to 22 bytes). Causality is exact over 144 cuts. Changing other rows or reordering rows changes nothing (exact). The module keeps no state. The same batch after another batch gives the same output (exact). Each row alone matches its row in the batch to 4.2e-7 (tolerance 1e-5). The forward is equal under both paths. The initial values equal arch20's G1, and there are 1,124,113 parameters. |
| **6. Binding** | **Pass.** A correct birth is admitted. Refused: a birth of the other arm, as filed and renamed as arm S; a birth presented to another seed's life; a birth copied under another seed; a changed weights file; a receipt whose source hashes differ; a receipt with decay 0.3; an incomplete receipt; and E20-B's own G1 birth receipt. Existing output files are refused by the check and by `dev()` and `life()` themselves. |
| **7. Reduced-dose runs** | **Pass.** Seed 73129, both arms, a birth of 2,048 updates and a life of 6 × 256 updates. All four runs complete, with completion records and optimizer steps 2,048 and 1,536, and with finite losses throughout. The rates and forecast are below. |

### Check 1 in detail: the failing case, and the measurements beside it

The limits are the brief's: logits within 1e-5 absolute, gradients within 1e-4 relative, and the swapped version must exceed 1e-5. The last three columns are recorded as diagnostics only; they are not part of the criterion.

| Seed, file, batch | Largest logit difference | Largest gradient difference (relative) | Gates swapped | Logit at the largest difference | Answer positions with a different top byte | Bit-identical to E25's step | Verdict |
|---|---:|---:|---:|---:|---|---|---|
| 73129 untrained mixed | 2.98e-07 | 4.6e-07 | 0.0438 | −0.58 | 0 of 576 | yes | pass |
| 73129 untrained birth | 2.38e-07 | 3.7e-07 | 0.035 | −0.23 | 0 of 192 | yes | pass |
| **73129 birth mixed** | **1.14e-05** | 7.4e-07 | 21.3 | 5.66 | 0 of 576 | yes | **FAIL** |
| 73129 birth birth | 5.72e-06 | 2.3e-06 | 27.3 | 20.54 | 0 of 192 | yes | pass |
| 73129 life mixed | 5.72e-06 | 1.8e-06 | 29.4 | 19.65 | 0 of 576 | yes | pass |
| 73129 life birth | 5.72e-06 | 1.0e-06 | 24.8 | 19.65 | 0 of 192 | yes | pass |
| 93683 untrained mixed | 2.38e-07 | 3.1e-07 | 0.0361 | 0.52 | 0 of 576 | yes | pass |
| 93683 untrained birth | 2.68e-07 | 5.0e-07 | 0.0328 | 0.47 | 0 of 192 | yes | pass |
| 93683 birth mixed | 7.63e-06 | 5.8e-07 | 25.4 | −20.32 | 0 of 576 | yes | pass |
| 93683 birth birth | 5.72e-06 | 3.7e-06 | 22.7 | −12.38 | 0 of 192 | yes | pass |
| 93683 life mixed | 7.63e-06 | 8.7e-07 | 20.8 | 16.74 | 0 of 576 | yes | pass |
| 93683 life birth | 5.72e-06 | 5.0e-07 | 17.7 | −18.06 | 0 of 192 | yes | pass |
| 116089 untrained mixed | 2.68e-07 | 2.8e-07 | 0.039 | 0.37 | 0 of 576 | yes | pass |
| 116089 untrained birth | 2.38e-07 | 3.8e-07 | 0.0428 | 0.61 | 0 of 192 | yes | pass |
| 116089 birth mixed | 9.06e-06 | 6.3e-07 | 24 | 3.31 | 0 of 576 | yes | pass |
| 116089 birth birth | 5.72e-06 | 1.6e-06 | 25.8 | 24.36 | 0 of 192 | yes | pass |
| 116089 life mixed | 8.58e-06 | 1.5e-06 | 32.7 | 10.22 | 0 of 576 | yes | pass |
| 116089 life birth | 6.68e-06 | 1.3e-06 | 28.2 | 4.87 | 0 of 192 | yes | pass |

The batches:
- *Mixed*: 32 `add97` and 32 `sub97` items at k = 12, plus 64 items of `within`.
- *Birth*: the seed's first birth batch.

**What I read from it, for the coordinator to judge. Nothing was changed.**
- On untrained networks the written-out step and the library layer differ by at most 3e-7, which agrees with the coordinator's own E25 fixtures (4.8e-7).
- On E20-B's trained files the difference is 5.7e-6 to 1.14e-5. Those networks' logits reach 25 to 36 in magnitude, and the difference builds over up to 22 recurrent steps.
- The library GRU and the written-out step add their terms in a different order. My step matches `e25_alignment`'s bit for bit, so the difference lies between the library layer and the written-out equations, not in my code.
- At the failing logit (5.66) the difference is 24 float32 spacings of that value.
- No answer position in any case has a different top byte. Check 2b, which runs the full decoder on the same trained files, reproduces every receipt count exactly.
- The brief's 1e-5 limit on every logit was set by the fixture on untrained networks. Whether it should hold on trained files, or be read relative to the logits' size, is a decision for the coordinator. I have not changed the tolerance, and I report check 1 as failed.

**How it was found.** A development run of the same check, before the formal run, showed the same case failing, by the same number. I added the diagnostic fields shown above (logit at the largest difference, answer positions with a different top byte) without changing the criterion. The formal run then reproduced the failure. The development outputs were in a scratch directory and are not part of this record.

## Reduced-dose runs and the forecast (check 7)

These numbers come from a reduced dose. They are not results and test no prediction. They show only that the pipeline runs and that both arms learn something.
- **Birth** (k = 12, of 300): arm S `add89` 14, `sub89` 5, `max89` 292, `min89` 287; arm B 21, 14, 293, 294.
- **End of life** (of 600): arm S `within` 77, `><` 200, `<>` 184, plus-minus 0 + 0, taught 96; arm B `within` 73, `><` 188, `<>` 100, plus-minus 1 + 0, taught 82.

Measured with two to three processes running on four cores, one thread each:

| | Arm S (sweep) | Arm B (backprop) |
|---|---:|---:|
| Birth: update CPU-s per update | 0.1363 | 0.1282 |
| Birth: scoring CPU-s per checkpoint row | 9.3 (plus 0.65 for the silenced panels at the end) | 8.2 (plus 0.65) |
| Life: update CPU-s per update | 0.1348 | 0.1262 |
| Life: scoring CPU-s per checkpoint | 56.1 | 55.8 |
| **Forecast, one full birth** (24,576 updates, 6 rows) | **3,406** (3,350 updates + 56 scoring) | **3,199** (3,150 + 50) |
| **Forecast, one full life** (36,864 updates, 7 checkpoints) | **5,362** (4,969 updates + 393 scoring) | **5,043** (4,653 + 391) |

- **Stage 2, six births:** about **19,800 CPU-s**. In waves of three on four cores, that is about two waves of about 3,400 s: **roughly 2 hours of wall time**.
- **Stage 3, six lives:** about **31,200 CPU-s**, about two waves of about 5,400 s: **roughly 3 hours**.
- **Twelve runs:** about 51,000 CPU-s, against the brief's first estimate of 68,000.

Two notes on the forecast:
- The sweep's update costs 1.06 to 1.07 times the exact path's on this host. The brief's owner-machine estimate was 1.6 / 1.4, about 1.14.
- The written-out exact path, at 0.128 CPU-s per update, costs about 1.1 times E20-B's library-layer G1 birth on the same host class. That is about 0.117 per update, which includes that birth's scoring.

**Two caveats on these rates:**
- They come from short runs.
- Full-dose scoring can cost more as answers lengthen: E20-B's G1 lives spent about 61 CPU-s per checkpoint.

## Open, for the coordinator

1. **Check 1 has failed** as specified (one case, 1.14e-5 against 1e-5). Stage 2 should not be released until you rule on it. Everything else needed for the ruling is above and in `check_step.json`.
2. **Please confirm or correct the reading** at the end of `E26_PREDICTIONS.md`, which says which receipt field each predicted quantity is read from.
3. **Stage 2 needs a full-dose output folder** (default `exp45/e26`). Lives admit only births made by the same sources (hash-checked). If any source changes after the ruling, the reduced runs would be rerun first.

Stage 1 ends here. Nothing further is started.
