# E20-B stage 1: implementation and reduced-dose qualification (11 November 2026)

**Job:** E20-B, stage 1 only, under the brief `OPUS_TASK_E20_COMPOSITION_FROM_MEMORY_20261110.md`, Job B, amendments 1 to 16. No full-dose birth or life has been run, and stage 2 waits for the coordinator's go (amendment 8). All work is on the branch `claude/adoring-cori-b07aog`, which started from `codex/minifly-consolidated-20261023` at `9031effb`. Nothing has gone to the default branch.

## What is returned (amendment 8)

| Item | Where |
|---|---|
| Runner commit (sources used by every run of attempt 2) | `0e131a29`. The sources are `arch20.py` and `exp20_reference.py`; their hashes are recorded in every receipt. |
| Check record | `exp45/e20b_qual/E20B_CHECK.json` |
| Predictions and selection rule, copied unchanged | `E20B_PREDICTIONS.md`, committed at `2ff1819b` before any qualification life |
| Command and environment of every qualification run | `exp45/e20b_qual/logs/LAUNCHES.txt`, `run_qual_attempt2.sh`, and each receipt's `provenance` |

Only new files were added. `exp16_integrated.py`, `arch17.py` and `exp11_cell.py` were not edited.

## Result of the checks (`E20B_CHECK.json` → `summary`)

| Check | Result |
|---|---|
| 9a. Birth parity: my birth loop with `bank2` against `exp16_integrated.py dev`, same seed and dose | **Pass.** The curve has the same loss, the same sealed scores at every k, the same embedding shift, the same silenced scores and the same credit census. The weights' tensors are identical (tensor hash `efdf392e…`). |
| 9b. Life parity: my life loop started from the reference birth file itself (path and hash `451e7332…` checked) against `exp16_integrated.py D` | **Pass.** At all seven stages the 16 score blocks the two runs share are identical. The losses, the exposure counts, the panel hash `00c9368b…`, the strict counts, the shortcut ceilings and the first-update census are equal. There were 1,536 optimizer steps, matching the reference's 1,536 updates. The final tensors are identical (`361a36f6…`). |
| 10. Scoring fixtures (stub responder, through `exp16_integrated.score`) | **Pass** on the four held pairs, `within` and `taught`. A fully correct answer scores whole and final. A wrong intermediate with the right ending scores final only. An answer missing the end symbol scores neither. The birth scorer generates at most 4 bytes, against a 6-byte target, and is used only for births. |
| 10. Taught panel | It holds 600 items, the first 50 triples of each taught pair. It is kept in its own dictionary: `shortcut_ceilings` with a `taught` key still raises a ValueError, which was checked. All 600 are among the stream's taught (F2) entries; no `within` or held item is. During each qualification life, 600 distinct taught-panel items were presented 684 times and no `within` or held item was presented (counted on the actual batches). |
| 11. Architecture (G0 to G3, a batch of 48 rows of 11 different lengths, 10 to 22 bytes) | **Pass** for all four: shapes; causality, exact over 144 cuts; rows changed or reordered, exact; no state kept between calls; each row alone against its row in the batch, at most 6e-7 (tolerance 1e-5, fixed before running). For G1 and G2, the textbook loop with zero initial states matches to 2.4e-7, and a non-zero initial state changes the first step, so the LSTM's cell state is really reset. On a birth batch and a mixed batch every parameter has a finite, non-zero gradient, and so does every active embedding row except BOS, which is a padding row. The two optimizer groups are `book.weight` with decay 0 and everything else with decay 0.3. |
| 11. G3 | `lb` is 64×512 with no bias: 32,768 weights. G3 has 1,158,033 parameters against G0's 1,125,265. Its shared initial values equal G0's. With `lb` at zero, its outputs equal G0's **bitwise**. With `lb` enabled, the first step is bitwise equal to G0 and later steps differ. `p_t`, recomputed as `tanh(lp z1_t + lb h2_{t-1})`, matches exactly. |
| 12. Admission fixtures | **Pass.** The correct birth is admitted. All of the following are refused: a birth copied under seed 93683, a birth presented to another seed's life, another network, a changed weights file, a receipt that names another network, an incomplete receipt, an exp16-style receipt (there is no legacy admission), and an existing birth or life output path. |
| 11. Dispatch | The runner imports with `E16_ARCH=bank2 E16_BOOKDECAY=off E16_COMPGAP=0`. It refuses any other `E16_`/`E11_` variable except the two dose variables. The store is off and the k = 40 panels are present. Each receipt records the installed class and the parameter count. |

## Qualification runs (seed 73129, birth 2,048 updates, life 6 × 256 updates, one thread, three processes at a time on four cores)

**Reference:** the coordinator's exact commands, with `<d>` = 2048 and `<s>` = 256. A `find` beforehand showed that no file of those names existed (`NAME_CHECK_BEFORE_RUNS.txt`). The reference's birth, receipt, life and logs are in `reference/`.

These numbers come from a reduced dose. They are not results, they test no prediction, and they are given only to show that the pipeline produces non-zero scores. Each is whole answers (`exact`), with its panel named.

| Net | Params | Birth CPU-s | Birth k=12 add89 / sub89 (of 300) | Life `within` (of 600) | `+-` / `-+` / `><` / `<>` (of 600) | Taught (of 600) | Life update CPU-s per update | Scoring CPU-s per checkpoint |
|---|---|---|---|---|---|---|---|---|
| G0 | 1,125,265 | 304 | 27 / 5 | 26 | 0 / 0 / 67 / 80 | 35 | 0.136 | 80 |
| G1 | 1,124,113 | 210 | 18 / 16 | 81 | 0 / 0 / 202 / 151 | 83 | 0.097 | 51 |
| G2 | 1,125,873 | 194 | 18 / 17 | 102 | 1 / 0 / 207 / 207 | 113 | 0.090 | 46 |
| G3 | 1,158,033 | 238 | 12 / 9 | 66 | 2 / 0 / 149 / 149 | 80 | 0.109 | 59 |

The G0 life from my own E20 birth gives the same scores as the parity life, because the two births are tensor-identical.

**Forecast for stage 2**, from these rates, which were measured with three processes running:

| Net | One birth (24,576 updates) | One life (36,864 updates and 7 scorings) |
|---|---|---|
| G0 | about 3,500 CPU-s | about 5,600 CPU-s |
| G1 | about 2,500 | about 3,900 |
| G2 | about 2,300 | about 3,600 |
| G3 | about 2,800 | about 4,400 |

That is about 29,000 CPU-s per seed, or about 85,000 to 90,000 for all twelve cells. On the G0/G3 host, run in waves of three, it is roughly 4.5 to 5 hours of wall time.

## Decisions made in implementing, for the coordinator and Codex to check

1. **G1 and G2 keep arch17's front end and readout.** The path is `book` (258×8) → `l1` (8→512, tanh) → GRU or LSTM → `z2 = tanh(l2 z1 + lm h)` → `l3`. Only the recurrent core differs from arch17. The widths are GRU 256 and LSTM 212, which put the counts within 0.1% of 1,125,265. The brief's "byte embedding, one GRU layer, the two-layer readout" could also be read as having no `l1`; I chose the reading that changes one component.
2. **Weights hashes.** `torch.save` writes the file's own name into the archive. Two files holding identical tensors under different names (`e17_dev_…` and `e20_dev_…`) therefore have different SHA-256 hashes. Parity compares the tensors, by a hash over names, dtypes, shapes and bytes, and also re-saves both under one name and compares those file hashes. Both agree in 9a and 9b. The file hashes are recorded beside them.
3. **The gradient check's exemption.** The exempt set is BOS only. A row counts as active if its byte occurs at or before a row's last supervised position. EOS as an input occurs only after that position, so it can receive no gradient, in G0 as in the reference census.
4. **The environment.** The container's torch is 2.15.0+cpu (the coordinator's receipts say 2.14.0), on Python 3.13.18 with four cores. Parity holds on this one host, which is what amendment 9 asks.

## A defect in attempt 1, kept as written (`attempt1_defective/`)

The first runs used commit `7a7a60a6`. In that version the life wrote stages 1 to 6 without their `scores` field, so those receipts could not show parity panel by panel. I found it when the parity comparison failed with a KeyError. I stopped the G3 life, which was still running, and the fix was one line. Every E20 birth binds the runner's source hashes, so I redid all four births and all the lives with the fixed runner. The reference runs were not repeated. Every printed line of the attempt-1 logs is identical to attempt 2 apart from CPU time: all four birth curves, the G0, G0-parity, G1 and G2 lives, and the stopped G3 life up to where it stopped. That attempt is not used for any check.

## Open, for the coordinator

- Stage 2 is closed until your go and Codex's reading of this record.
- Please confirm or correct the operational reading at the end of `E20B_PREDICTIONS.md` (which receipt field each predicted quantity is read from) before stage 2.
- The G0/G3 host must run with the runner at `0e131a29` or a later commit with the same sources. A life refuses a birth whose recorded source hashes differ.
