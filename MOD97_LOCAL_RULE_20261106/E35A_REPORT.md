# E35-A report: innate circuits, inventory 甲 — stage 0 and stage 1a (seed 73129)

Brief: `OPUS_TASK_E35A_INNATE_CIRCUITS_INVENTORY_JIA_20261115.md`, revision 2 (dc05107d on `codex/minifly-consolidated-20261023`). Worker branch `claude/nice-rubin-y5de8k`. Host: Linux x86-64, 4 cores, Python 3.13.18, torch 2.15.0+cpu, one thread per process, at most three processes, all long runs detached (`nohup setsid`, no wrapper limit). Evidence level E2, reference stage (backpropagation); nothing here is a candidate rule. No formal checker gate or external review is claimed.

Runners (new files; nothing existing edited): `exp35a_parsed.py` (stage 0), `exp35a_stream.py` (stage 1a and checks 2-5), `exp45/e35a/summarize.py` (tables). Receipts: `exp45/e35a/`.

## Stage 0: the upper bound with parsing supplied

E10's RingNet (copied with MOD = 89; exp10_ring fixes 97 at import), AdamW(.003, .01), batch 64, 32,768 updates, held-out accuracy on every held-out pair, exact two-digit answer. Receipts `exp45/e35a/stage0/e35a_parsed_<arm>_f<10|25>_<seed>.json`. CPU 5,188 s for the twelve runs.

| fraction | seed | arm | final held mean | add | sub | max | min | curve of held mean (every 4,096) |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 73129 | ring | 92.7 | 94.7 | 94.4 | 90.3 | 91.3 | 14.3, 40.0, 71.5, 83.7, 88.4, 90.7, 92.7, 92.7 |
| 0.25 | 93683 | ring | 91.4 | 88.3 | 88.4 | 94.5 | 94.5 | 11.5, 18.2, 37.3, 57.8, 74.6, 83.4, 88.3, 91.4 |
| 0.25 | 116089 | ring | 49.1 | 7.7 | 5.7 | 91.2 | 91.8 | 12.0, 20.0, 40.0, 45.6, 47.5, 48.0, 48.8, 49.1 |
| 0.25 | 73129 | table | 3.5 | 2.0 | 2.5 | 4.6 | 5.1 | 3.4, 3.5, 3.5, 3.6, 3.6, 3.6, 3.5, 3.5 |
| 0.25 | 93683 | table | 11.1 | 5.8 | 5.8 | 16.5 | 16.3 | 5.6, 7.3, 7.3, 8.7, 9.5, 10.2, 10.5, 11.1 |
| 0.25 | 116089 | table | 7.7 | 5.8 | 6.2 | 9.7 | 9.2 | 3.9, 5.7, 6.7, 7.3, 7.4, 7.7, 7.5, 7.7 |
| 0.10 | 73129 | ring | 97.3 | 97.1 | 97.1 | 97.7 | 97.1 | 7.3, 47.3, 56.0, 79.3, 91.8, 95.5, 96.4, 97.3 |
| 0.10 | 93683 | ring | 86.8 | 77.4 | 76.6 | 96.8 | 96.3 | 19.4, 40.4, 52.1, 61.2, 73.8, 79.7, 84.4, 86.8 |
| 0.10 | 116089 | ring | 92.1 | 90.2 | 90.2 | 94.4 | 93.8 | 19.0, 39.5, 46.4, 55.8, 72.6, 83.4, 89.2, 92.1 |
| 0.10 | 73129 | table | 7.9 | 3.7 | 5.0 | 12.4 | 10.5 | 5.2, 6.3, 6.3, 6.8, 6.8, 7.6, 8.0, 7.9 |
| 0.10 | 93683 | table | 13.6 | 4.3 | 4.2 | 23.9 | 21.9 | 3.9, 4.5, 8.3, 10.5, 11.7, 12.6, 13.7, 13.6 |
| 0.10 | 116089 | table | 5.0 | 2.5 | 2.7 | 5.2 | 9.5 | 3.1, 3.2, 3.3, 3.2, 3.3, 3.4, 3.4, 5.0 |

**Gate** (ring at f = 0.25, final row, at least 90 in at least two of three seeds): 92.7, 91.4, 49.1 → **passes**. The stream stage ran.

## Checks before any full birth (`exp45/e35a/checks/`)

| # | Check | Result |
|---|---|---|
| 1 | Dry birth of G, f = 0.5, E26 panels, 2,048 updates: `tensors_sha256` equals `exp45/e26_qual/e26_dev_B_dry2048_73129.json` | **pass**: adf933d7f2ab1a8f…3a0b both; data digest and sealed panels also identical |
| 2 | Four operations exact on all 7,921 one-hot pairs, all mass on the answer; `table_op`(true add) = `op_add` (one-hot max abs 0.0; random distributions 6.5e-9); tables from `manual_seed(35)` | **pass**; tables sha256 a39d3f86edd20d2565e74823d83e5ad136dd333b2e59d215c373619fd50896d7; agreement with the true operation 1.36 / 1.12 / 1.12 / 0.98 % |
| 2b | `module_path` with ideal interfaces, all 7,921 pairs × 4 operations × gaps 0, 12, 40; `dec` with `wd` set by hand | **pass**: 7,921/7,921 argmax, min mass on the answer 1.0, two digit bytes 7,921/7,921 in all twelve cases |
| 3 | M, X, F equal G in every G parameter before the first update (13 tensors); M and X new parameters equal | **pass** |
| 4 | `module_off=True` logits of M and X equal a G network holding the same shared parameters (after 8 updates) | **pass**, max abs 0.0; module-on logits differ |
| 5 | Forward signature is the harness's `(m, x, local, through_time, keep, silence)`; no mask, role, position, item or prompt name in `forward` or `module_path` | **pass** |
| 6 | Dry births M and X, f = 0.25, 2,048 updates | finite losses; M 954 CPU-s (0.428 s/update, 3.5 × G), X 1,334 CPU-s (0.614 s/update, 5.0 × G). Forecast for six births 16.4 CPU-h (`check6_forecast.json`) |

On check 6: the six-birth forecast exceeded the brief's 10 CPU-hours. Under the order the reviewer recommended, only one fraction had room (below), so four births plus arm F were run: 11.8 CPU-h in all for stage 1a. The excess over 10 h is arm F, which the brief adds after row 3.

## Stage 1a: births (seed 73129, 24,576 updates)

| Birth | Fraction | complete | CPU s | tensors_sha256 | source commit |
|---|---|---|---|---|---|
| G | 0.10 | yes | 2,726 | f65a4e96bb78393f… | b5ab19ac |
| G | 0.25 | yes | 2,726 | dc160da2ea653828… | b5ab19ac |
| M | 0.10 | yes | 10,994 | 6aaedbb8122bf2d8… | 990828a8 |
| X | 0.10 | yes | 15,079 | 85c5c91678bce60e… | 990828a8 |
| F | 0.10 | yes | 11,117 | 24b0bc69b22617b8… | ad0bdcd2 |
| M, X | 0.25 | NOT RUN: no family has room at f = 0.25 | | | |
| F | 0.25 | NOT RUN: row 3 did not occur at f = 0.25 | | | |

Every loss finite. Weights at rows 16,384, 20,480 and 24,576 saved and committed (`git add -f`) for each birth; their hashes are in each receipt's `saved_rows`. Last-block losses (final row): G 0.0167, M 0.0156, X 0.0144, F 0.0334 (f = 0.10); G 0.25: see receipt.

### Tables (percent exact; each cell is the mean of rows 16,384, 20,480 and 24,576 / the final row)

"off" is the held-out panel with `module_off=True`. Families are the means over their two operations.

**mean of four** (last-three mean / final row)

| arm | fraction | held12 | held40 | taught12 | off held12 | off held40 |
|---|---|---|---|---|---|---|
| G | 0.10 | 55.3 / 55.0 | 53.9 / 53.3 | 98.3 / 97.9 | — / — | — / — |
| M | 0.10 | 56.0 / 57.9 | 55.2 / 56.7 | 98.5 / 98.8 | 55.9 / 58.2 | 54.6 / 56.2 |
| X | 0.10 | 55.3 / 55.7 | 54.2 / 54.2 | 98.4 / 99.0 | 55.1 / 55.6 | 54.4 / 54.4 |
| F | 0.10 | 48.5 / 49.1 | 47.0 / 47.2 | 97.4 / 98.9 | 0.9 / 0.9 | 0.9 / 0.9 |
| G | 0.25 | 93.2 / 93.7 | 91.0 / 92.8 | 97.1 / 97.8 | — / — | — / — |

**add+sub** (last-three mean / final row)

| arm | fraction | held12 | held40 | taught12 | off held12 | off held40 |
|---|---|---|---|---|---|---|
| G | 0.10 | 19.6 / 20.5 | 19.8 / 21.5 | 97.5 / 97.7 | — / — | — / — |
| M | 0.10 | 21.1 / 25.5 | 21.3 / 25.8 | 97.6 / 98.7 | 21.2 / 26.3 | 20.7 / 25.2 |
| X | 0.10 | 19.4 / 19.7 | 18.9 / 19.7 | 97.8 / 98.3 | 19.3 / 19.7 | 19.4 / 20.0 |
| F | 0.10 | 6.6 / 7.3 | 5.9 / 6.3 | 94.8 / 98.0 | 1.2 / 1.2 | 1.2 / 1.2 |
| G | 0.25 | 89.2 / 90.3 | 87.1 / 89.3 | 95.0 / 96.0 | — / — | — / — |

**max+min** (last-three mean / final row)

| arm | fraction | held12 | held40 | taught12 | off held12 | off held40 |
|---|---|---|---|---|---|---|
| G | 0.10 | 91.0 / 89.5 | 87.9 / 85.2 | 99.2 / 98.2 | — / — | — / — |
| M | 0.10 | 90.9 / 90.3 | 89.2 / 87.5 | 99.4 / 98.8 | 90.6 / 90.0 | 88.4 / 87.3 |
| X | 0.10 | 91.1 / 91.7 | 89.4 / 88.8 | 98.9 / 99.7 | 91.0 / 91.5 | 89.4 / 88.8 |
| F | 0.10 | 90.3 / 90.8 | 88.1 / 88.0 | 99.9 / 99.8 | 0.7 / 0.7 | 0.7 / 0.7 |
| G | 0.25 | 97.2 / 97.0 | 95.0 / 96.2 | 99.2 / 99.5 | — / — | — / — |

**add89** (last-three mean / final row)

| arm | fraction | held12 | held40 | taught12 | off held12 | off held40 |
|---|---|---|---|---|---|---|
| G | 0.10 | 21.8 / 22.0 | 21.7 / 23.7 | 97.8 / 98.0 | — / — | — / — |
| M | 0.10 | 22.3 / 25.7 | 22.9 / 26.7 | 97.1 / 98.3 | 22.4 / 26.7 | 22.7 / 26.7 |
| X | 0.10 | 20.6 / 23.7 | 19.9 / 23.0 | 97.2 / 97.7 | 20.4 / 23.3 | 20.9 / 24.3 |
| F | 0.10 | 8.1 / 10.0 | 7.3 / 8.3 | 93.7 / 97.3 | 1.0 / 1.0 | 1.0 / 1.0 |
| G | 0.25 | 90.2 / 91.0 | 88.2 / 89.3 | 95.7 / 96.3 | — / — | — / — |

**sub89** (last-three mean / final row)

| arm | fraction | held12 | held40 | taught12 | off held12 | off held40 |
|---|---|---|---|---|---|---|
| G | 0.10 | 17.4 / 19.0 | 18.0 / 19.3 | 97.2 / 97.3 | — / — | — / — |
| M | 0.10 | 19.8 / 25.3 | 19.7 / 25.0 | 98.0 / 99.0 | 19.9 / 26.0 | 18.8 / 23.7 |
| X | 0.10 | 18.3 / 15.7 | 18.0 / 16.3 | 98.4 / 99.0 | 18.1 / 16.0 | 17.9 / 15.7 |
| F | 0.10 | 5.1 / 4.7 | 4.6 / 4.3 | 95.9 / 98.7 | 1.3 / 1.3 | 1.3 / 1.3 |
| G | 0.25 | 88.1 / 89.7 | 85.9 / 89.3 | 94.3 / 95.7 | — / — | — / — |

**max89** (last-three mean / final row)

| arm | fraction | held12 | held40 | taught12 | off held12 | off held40 |
|---|---|---|---|---|---|---|
| G | 0.10 | 91.0 / 90.0 | 88.1 / 85.3 | 99.1 / 98.0 | — / — | — / — |
| M | 0.10 | 90.9 / 91.0 | 90.0 / 88.7 | 99.8 / 100.0 | 90.6 / 90.3 | 89.1 / 88.7 |
| X | 0.10 | 92.0 / 92.0 | 91.3 / 91.3 | 99.8 / 99.7 | 91.9 / 92.0 | 91.1 / 91.7 |
| F | 0.10 | 88.8 / 89.7 | 85.4 / 85.3 | 99.9 / 99.7 | 0.0 / 0.0 | 0.0 / 0.0 |
| G | 0.25 | 96.1 / 96.3 | 93.1 / 95.3 | 99.0 / 99.3 | — / — | — / — |

**min89** (last-three mean / final row)

| arm | fraction | held12 | held40 | taught12 | off held12 | off held40 |
|---|---|---|---|---|---|---|
| G | 0.10 | 91.0 / 89.0 | 87.7 / 85.0 | 99.2 / 98.3 | — / — | — / — |
| M | 0.10 | 90.9 / 89.7 | 88.3 / 86.3 | 99.0 / 97.7 | 90.7 / 89.7 | 87.8 / 86.0 |
| X | 0.10 | 90.2 / 91.3 | 87.6 / 86.3 | 98.0 / 99.7 | 90.1 / 91.0 | 87.7 / 86.0 |
| F | 0.10 | 91.9 / 92.0 | 90.7 / 90.7 | 100.0 / 100.0 | 1.3 / 1.3 | 1.3 / 1.3 |
| G | 0.25 | 98.3 / 97.7 | 96.9 / 97.0 | 99.4 / 99.7 | — / — | — / — |

### Gate diagnostics at the end of each module birth (training batch at update 24,576, by role of the byte just read)

| arm | quantity | code | operand | operator | distractor | equals | answer digit | EOS |
|---|---|---|---|---|---|---|---|---|
| M | gA | 0.656 | 0.974 | 0.956 | 0.890 | 0.606 | 0.964 | 0.913 |
| M | gB | 0.656 | 0.974 | 0.956 | 0.892 | 0.607 | 0.964 | 0.912 |
| M | q_tens | 0.919 | 0.991 | 1.000 | 0.999 | 1.000 | 0.999 | 0.999 |
| M | q_units | 0.893 | 0.981 | 0.992 | 0.982 | 0.999 | 0.998 | 0.996 |
| X | gA | 0.262 | 0.525 | 0.892 | 0.837 | 0.985 | 0.489 | 0.677 |
| X | gB | 0.971 | 0.489 | 0.079 | 0.058 | 0.037 | 0.501 | 0.294 |
| X | q_tens | 0.416 | 0.930 | 0.994 | 0.986 | 0.999 | 0.960 | 0.935 |
| X | q_units | 0.452 | 0.951 | 0.997 | 0.993 | 1.000 | 0.976 | 0.958 |
| F | gA | 0.816 | 0.780 | 0.958 | 0.998 | 0.854 | 0.868 | 0.820 |
| F | gB | 0.836 | 0.996 | 0.999 | 1.000 | 0.829 | 0.997 | 0.993 |
| F | q_tens | 0.905 | 0.581 | 0.923 | 0.983 | 1.000 | 0.836 | 0.903 |
| F | q_units | 0.850 | 0.555 | 0.582 | 0.647 | 0.997 | 0.728 | 0.738 |

| arm | mean largest operation weight at the step that reads '=' | argmax counts (add, sub, max, min) over the 64 items |
|---|---|---|
| M | 0.986 | [0, 0, 64, 0] |
| X | 0.868 | [0, 0, 52, 12] |
| F | 0.912 | [0, 4, 34, 26] |

## Outcome rows (each family at each fraction; the first row that applies)

| Fraction | Family | G held12 | M held12 | X held12 | M off held12 | Row |
|---|---|---|---|---|---|---|
| 0.10 | add+sub | 19.6 | 21.1 | 19.4 | 21.2 | **3**: room; M within 15 of G (+1.5) |
| 0.10 | max+min | 91.0 | 90.9 | 91.1 | 90.6 | **0**: no room |
| 0.25 | add+sub | 89.2 | NOT RUN | NOT RUN | — | **0**: no room |
| 0.25 | max+min | 97.2 | NOT RUN | NOT RUN | — | **0**: no room |

**After row 3: arm F, f = 0.10, seed 73129.** add+sub held12: F 6.6, G 19.6, M 21.1. F is within 15 of G (−13.0) → the brief's reading: *"Forced through the module, the controller still did not learn the interface on this seed. Stage 1b applies the stop rule with arm F."*

Marks (add+sub, f = 0.10): G, M, X and F each meet **memorises** (taught12 97.5 / 97.6 / 97.8 / 94.8, held12 ≤ 30). **Uses the module:** M's add+sub held12 changes by +0.1 with the module off (21.1 → 21.2): not met.

Stage 1b is not run: the coordinator releases it after reading stage 1a. Under the brief, row 3 with F within 15 of G names arm F at f = 0.10 on seeds 93683 and 116089.

## Predictions

| | Prediction | Outcome |
|---|---|---|
| P35A.1 | Stage 0 gate passes | **Yes** (92.7, 91.4, 49.1) |
| P35A.2 | Ring reaches 90 at f = 0.10 in two of three seeds | **Yes** (97.3, 86.8, 92.1) |
| P35A.3 | Random-table arm ≤ 30 at both fractions in all seeds | **Yes** (highest 13.6) |
| P35A.4 | G has room at f = 0.10 for add and sub | **Yes** (19.6) |
| P35A.5 | G has room at f = 0.25 for add and sub | **No** (89.2) |
| P35A.6 | Row 1 for add and sub at one fraction or both | **No** (row 3 at f = 0.10; row 0 at f = 0.25) |
| P35A.7 | Row 3 for add and sub at every fraction with room | **Yes** (f = 0.10, the only fraction with room) |
| P35A.9 | If F runs, it is at least 15 above both G and M | **No** (F 6.6; G 19.6; M 21.1) |
| P35A.8 | Where M generalises at 12 letters, its held40 is at least 60 | M generalises only in max+min at f = 0.10 (held12 90.9; a row-0 family): held40 89.2, **yes** there. M does not generalise in add+sub. |

## Limits

One seed in stage 1a. Single births; differences under 15 points are not read. The gate diagnostics describe; they do not explain. All the brief's "cannot show" items apply (no rule inside the line; no composition, few-shot or retention; breadth not tested; pass-through; thin coverage at f = 0.10; inventory 甲 wires the answer's structure).

## Addendum 1: stage 1b (arm F, f = 0.10, seeds 93683 and 116089)

Two births of arm `F` were launched at 02:39 UTC on 16 November 2026 from HEAD cdbe5daa. Each ran `python3 exp35a_stream.py dev F 10 <seed>` as its own detached process with one thread. Both exited with code 0: seed 93683 at 05:39 and seed 116089 at 05:43. Each receipt is complete and has three saved rows (16,384, 20,480 and 24,576). CPU: 10,693 s for seed 93683 and 10,951 s for seed 116089.

- Receipts: `exp45/e35a/e35a_dev_F_f10_{93683,116089}.json`.
- Weights: `_row{16384,20480,24576}.pt`.
- Log: `exp45/e35a/logs/LAUNCHES.txt`.

Arm `G` was not rerun on these seeds. Its figures come from the E35-B worker's receipts, `exp45/e35b/e35b_dev_G_f10_{93683,116089}.json` (complete). The data digest of `F` equals E35-B `G`'s at all six scored rows on both seeds.

Each figure is the mean of rows 16,384, 20,480 and 24,576 for add and sub.

| Seed | G held12 | F held12 | F − G | G held40 | F held40 | G taught12 | F taught12 | F within 15 of G |
|---|---|---|---|---|---|---|---|---|
| 73129 | 19.6 | 6.6 | −13.0 | 19.8 | 5.9 | 97.5 | 94.8 | yes |
| 93683 | 32.1 | 11.8 | −20.3 | 31.6 | 11.8 | 96.7 | 97.2 | no (below by more) |
| 116089 | 27.8 | 21.3 | −6.5 | 26.2 | 20.8 | 98.2 | 97.7 | yes |

**Stop rule:** it applies. `F` is within 15 points of `G` (held12) on two of the three seeds, 73129 and 116089. Under the brief, the obstacle is the interface and inventory 甲's line stops here. Nothing is briefed after stage 1b.

**P35A.10** ("`F` is within 15 of `G`, or below it by more, on both new seeds, so that the stop rule applies"): **yes**.

- Seed 93683: 20.3 below, so "below it by more".
- Seed 116089: 6.5 below, so within 15.
- The stop rule applies.

Limits: single births per seed, and the stage 1a limits apply.
