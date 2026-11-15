# E37 report: fresh networks on one life family alone (15 November 2026)

Brief: `OPUS_TASK_E37_FRESH_REFERENCE_FOR_A_LIFE_FAMILY_20261115.md`, revision 2, at 18751457 (`origin/codex/minifly-consolidated-20261023`). Evidence level E2.
Runner `exp37_fresh_family.py`, tables `e37_report.py` → `exp45/e37/E37_TABLE.{md,json}`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu, three processes, one thread each, each under `timeout`. Nothing existing was edited; `exp26_candidate.step` and `make_opt` were called, not patched.
Life receipts read: `exp45/e26/e26_life_{S,B}_<seed>.json` and `exp45/e31/e31_life_K_<seed>.json`. Model: `arch26.new_net(seed)` (E20-B's G1 GRU, written out), the same in every run.

## Checks (all passed before the full runs)

| Check | Result | Receipt |
|---|---|---|
| 1. Part 2's 6,144-batch digest = `stages[1].data_digest` | equal for S, B and K lives, all three seeds; also re-checked inside each of the six F1 runs (equal) | `E37_CHECK_DATA.json`, `e37_F1_*.json` |
| 2. F3 items in the life | block 5: 131,079 / 131,131 / 130,950 beside 131,072; blocks 5+6: 262,120 / 261,736 / 261,821 beside 262,144 (identical across S, B, K). Beyond the brief: a rebuild of each life's joint stream gives these counts, and the F3 items it drew are, in order, a prefix of part 1's stream | `E37_CHECK_DATA.json` |
| 3. Update path vs E26 dry births (2,048 updates, seed 73129) | sweep `79618ab29390e6ec…` = reference; backprop `adf933d7f2ab1a8f…` = reference; batch digests equal too | `E37_CHECK_PATH_S.json`, `E37_CHECK_PATH_B.json` |
| 4. Dry runs of 256 updates (seed 73129) | CPU 42 s (F3 S), 37 s (F3 B), 39 s (F1 S), 37 s (F1 B); forecast 0.17 CPU-h per F3 run, 0.25 per F1 run (reviewer's estimate: 0.9 and 1.4) | `dry/e37_*_dry256_73129.json` |

## Runs

Twelve runs, all exit 0, all complete. Actual CPU: 0.16 h per F3 run, 0.24 h per F1 run, 2.39 CPU-hours in all. Receipts `exp45/e37/e37_<F3|F1>_<S|B>_<seed>.json`. Final weights committed (the 12 `_row4096.pt` / `_row6144.pt` files); the other rows' weights sha256 and tensor sha256 are in each receipt's `curve` (files not committed).

## Table (score: mean of add and sub at k = 12, % exact of 400, `X16.panels(seed)[0]`, `X16.score`; max · min tabled, not read)

| Comparison | Life | Fresh rule | Seed | Life items of family | Fresh items | Life | Fresh | Life − fresh | Life max · min | Fresh max · min |
|---|---|---|---|---|---|---|---|---|---|---|
| F3 at ~131,000 items | S | S | 73129 | 131,079 | 131,072 | 84.1 | 7.2 | **+76.9** | 98.8 · 96.2 | 95.2 · 93.0 |
| F3 at ~131,000 items | S | S | 93683 | 131,131 | 131,072 | 76.9 | 10.9 | **+66.0** | 99.0 · 97.8 | 96.5 · 94.5 |
| F3 at ~131,000 items | S | S | 116089 | 130,950 | 131,072 | 88.1 | 16.9 | **+71.2** | 98.2 · 99.5 | 93.8 · 97.0 |
| F3 at ~131,000 items | B | B | 73129 | 131,079 | 131,072 | 82.4 | 15.2 | **+67.1** | 100.0 · 99.5 | 97.2 · 95.0 |
| F3 at ~131,000 items | B | B | 93683 | 131,131 | 131,072 | 93.8 | 13.4 | **+80.4** | 99.0 · 98.5 | 97.2 · 95.2 |
| F3 at ~131,000 items | B | B | 116089 | 130,950 | 131,072 | 89.8 | 18.5 | **+71.2** | 99.5 · 100.0 | 93.5 · 94.2 |
| F3 at ~131,000 items | K | S | 73129 | 131,079 | 131,072 | 86.9 | 7.2 | **+79.6** | 98.0 · 98.8 | 95.2 · 93.0 |
| F3 at ~131,000 items | K | S | 93683 | 131,131 | 131,072 | 88.6 | 10.9 | **+77.8** | 96.5 · 99.0 | 96.5 · 94.5 |
| F3 at ~131,000 items | K | S | 116089 | 130,950 | 131,072 | 90.0 | 16.9 | **+73.1** | 99.8 · 99.5 | 93.8 · 97.0 |
| F3 at ~262,000 items | S | S | 73129 | 262,120 | 262,144 | 80.4 | 36.5 | **+43.9** | 98.0 · 98.0 | 94.2 · 93.8 |
| F3 at ~262,000 items | S | S | 93683 | 261,736 | 262,144 | 85.5 | 47.1 | **+38.4** | 99.2 · 97.2 | 93.0 · 90.0 |
| F3 at ~262,000 items | S | S | 116089 | 261,821 | 262,144 | 91.9 | 53.5 | **+38.4** | 97.8 · 98.8 | 91.5 · 95.0 |
| F3 at ~262,000 items | B | B | 73129 | 262,120 | 262,144 | 85.6 | 64.2 | **+21.4** | 99.2 · 99.5 | 96.2 · 95.0 |
| F3 at ~262,000 items | B | B | 93683 | 261,736 | 262,144 | 95.1 | 59.6 | **+35.5** | 97.5 · 97.2 | 96.2 · 93.2 |
| F3 at ~262,000 items | B | B | 116089 | 261,821 | 262,144 | 90.2 | 73.5 | **+16.8** | 99.2 · 100.0 | 94.5 · 96.2 |
| F3 at ~262,000 items | K | S | 73129 | 262,120 | 262,144 | 83.6 | 36.5 | **+47.1** | 99.2 · 99.2 | 94.2 · 93.8 |
| F3 at ~262,000 items | K | S | 93683 | 261,736 | 262,144 | 89.4 | 47.1 | **+42.2** | 99.8 · 98.8 | 93.0 · 90.0 |
| F3 at ~262,000 items | K | S | 116089 | 261,821 | 262,144 | 90.4 | 53.5 | **+36.9** | 98.8 · 99.0 | 91.5 · 95.0 |
| F1 at 393,216 items | S | S | 73129 | 393,216 | 393,216 | 92.5 | 67.0 | **+25.5** | 99.0 · 99.2 | 94.2 · 95.5 |
| F1 at 393,216 items | S | S | 93683 | 393,216 | 393,216 | 98.0 | 81.4 | **+16.6** | 99.0 · 99.2 | 96.5 · 97.2 |
| F1 at 393,216 items | S | S | 116089 | 393,216 | 393,216 | 95.5 | 67.6 | **+27.9** | 98.5 · 98.2 | 93.8 · 96.0 |
| F1 at 393,216 items | B | B | 73129 | 393,216 | 393,216 | 91.6 | 80.4 | **+11.2** | 99.2 · 99.5 | 98.5 · 97.2 |
| F1 at 393,216 items | B | B | 93683 | 393,216 | 393,216 | 91.4 | 83.5 | **+7.9** | 97.8 · 99.8 | 95.2 · 97.2 |
| F1 at 393,216 items | B | B | 116089 | 393,216 | 393,216 | 94.5 | 78.9 | **+15.6** | 98.5 · 98.8 | 97.8 · 99.5 |
| F1 at 393,216 items | K | S | 73129 | 393,216 | 393,216 | 90.6 | 67.0 | **+23.6** | 98.8 · 98.8 | 94.2 · 95.5 |
| F1 at 393,216 items | K | S | 93683 | 393,216 | 393,216 | 96.8 | 81.4 | **+15.4** | 97.8 · 98.8 | 96.5 · 97.2 |
| F1 at 393,216 items | K | S | 116089 | 393,216 | 393,216 | 95.5 | 67.6 | **+27.9** | 98.0 · 98.5 | 93.8 · 96.0 |

| Comparison and rule | Differences (73129, 93683, 116089) | Verdict |
|---|---|---|
| F3 at ~131,000 items | life S vs fresh S | +76.9, +66.0, +71.2 | **saving** |
| F3 at ~131,000 items | life B vs fresh B | +67.1, +80.4, +71.2 | **saving** |
| F3 at ~131,000 items | life K vs fresh S | +79.6, +77.8, +73.1 | **saving** |
| F3 at ~262,000 items | life S vs fresh S | +43.9, +38.4, +38.4 | **saving** |
| F3 at ~262,000 items | life B vs fresh B | +21.4, +35.5, +16.8 | **saving** |
| F3 at ~262,000 items | life K vs fresh S | +47.1, +42.2, +36.9 | **saving** |
| F1 at 393,216 items | life S vs fresh S | +25.5, +16.6, +27.9 | **saving** |
| F1 at 393,216 items | life B vs fresh B | +11.2, +7.9, +15.6 | **none** |
| F1 at 393,216 items | life K vs fresh S | +23.6, +15.4, +27.9 | **saving** |

Fresh curves (mean of add and sub at k = 12, %):

| Run | row | row | row |
|---|---|---|---|
| F3_S_73129 | 1024: 2.5 | 2048: 7.2 | 4096: 36.5 |
| F3_S_93683 | 1024: 1.6 | 2048: 10.9 | 4096: 47.1 |
| F3_S_116089 | 1024: 3.2 | 2048: 16.9 | 4096: 53.5 |
| F3_B_73129 | 1024: 1.6 | 2048: 15.2 | 4096: 64.2 |
| F3_B_93683 | 1024: 1.4 | 2048: 13.4 | 4096: 59.6 |
| F3_B_116089 | 1024: 3.5 | 2048: 18.5 | 4096: 73.5 |
| F1_S_73129 | 2048: 4.2 | 4096: 47.1 | 6144: 67.0 |
| F1_S_93683 | 2048: 7.1 | 4096: 72.5 | 6144: 81.4 |
| F1_S_116089 | 2048: 5.5 | 4096: 42.8 | 6144: 67.6 |
| F1_B_73129 | 2048: 3.8 | 4096: 65.5 | 6144: 80.4 |
| F1_B_93683 | 2048: 9.1 | 4096: 73.2 | 6144: 83.5 |
| F1_B_116089 | 2048: 7.4 | 4096: 65.5 | 6144: 78.9 |

## Verdicts (life minus fresh, at least two of three seeds; ±15 points)

| Comparison | Sweep (life S vs fresh S) | Backprop (life B vs fresh B) | K (life K vs fresh S) |
|---|---|---|---|
| F3 at ~131,000 items | saving | saving | saving |
| F3 at ~262,000 items | saving | saving | saving |
| F1 at 393,216 items (read separately: born vs unborn on identical batches) | saving | **none** (+11.2, +7.9, +15.6) | saving |

Reported beside it, not read: E35-A arm `G` at f = 0.25 (fresh GRU, a quarter of the pairs of mod 89, backprop; `origin/claude/nice-rubin-y5de8k` at d5b89442, `exp45/e35a/e35a_dev_G_f25_73129.json`, seed 73129 only on that branch), at its 4,096-update row: held-out add89 77.0, sub89 76.3 (of 300, k = 12; its own panels, so not on our panels). Beside the fresh backprop F3 row 4,096 (64.2 / 59.6 / 73.5) and F1 row 4,096 (65.5 / 73.2 / 65.5).

## Predictions

| | Prediction | Chance | Outcome |
|---|---|---|---|
| P37.1 | F3 at 131,000 items: a saving under the sweep | 80% | **right** (+76.9, +66.0, +71.2) |
| P37.2 | F3 at 131,000 items: a saving under backpropagation | 80% | **right** (+67.1, +80.4, +71.2) |
| P37.3 | F3 at 262,000 items: a saving under each rule | 55% | **right** (S +43.9, +38.4, +38.4; B +21.4, +35.5, +16.8) |
| P37.4 | F1 at 393,216 items: a saving under each rule | 65% | **wrong**: saving under the sweep, none under backprop |
| P37.5 | A reading that differs between the sweep and backpropagation | 15% | **right**: F1 (saving vs none) |

## Limits (from the brief, unchanged)

- The lived learner had also seen ~1.6 million birth items and ~1.8 million items of other families; the saving is in items of the new family, not in total experience.
- In part 1 the fresh network gives every update to F3, the life a third; this favours the fresh network per update and is neutral per item.
- It does not say what in the lived network gives the saving. Three seeds; no formal checker gate or external review is claimed.
