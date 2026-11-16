# E43 report: is a family lost when the learner then learns something unrelated?

Brief: `OPUS_TASK_E43_IS_A_FAMILY_LOST_TO_UNRELATED_LEARNING_20261116.md`, revision 2, at 517a05a8 on `codex/minifly-consolidated-20261023`. Work branch: `claude/peaceful-hopper-8ofq4i`. Runner `exp43_unrelated.py`, checks `e43_checks.py`, reader `e43_read.py`, unrelated task `e43_task.py` (a byte-identical copy of `PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e43_unrelated_task_spec.py`). These were committed at df443e49; the check script got one later fix at a2b0a863, a clean environment for the spec self-check's subprocess. Every run receipt records its sources unmodified against the commit it ran from. Nothing existing was edited. Evidence level E2: three seeds, two rules, one dose.

**Host.** Linux x86-64, 4 cores, Python 3.13.18, torch 2.15.0+cpu. Each process ran one thread, at most three at once, launched with `setsid nohup` outside any time-limited shell.

**Admission.** I used E38's recorded exception (`exp38_erase.admit`, as `exp31_switch.admit_s`). The births were made with `exp16_integrated.py` at a301ae8e; the branch now has deea398c, which adds only E21b's resume machinery. Check 1 shows the result is bit-identical to E38.

## Checks

| # | Check | Result | Receipt (`checks/`) |
|---|---|---|---|
| 1 | `conflict-decay` = E38 `zero`, `conflict` = E38 `none`: `tensors_sha256` at 64, 256, 1,024, 4,096, seed 73129, both rules | **pass**: all 16 hashes equal E38's and the brief's prefixes | `e43_check_1_conflict_is_e38.json` |
| 2 | Row 0 has the birth file's `tensors_sha256` (all 24 runs; also equal to the E26 birth receipt's); first-family scores at row 0 equal E38's row 0 (seed 73129: add and sub on all pairs, split by reduction, and the four held-out panels) | **pass** | `e43_check_2_row0.json` |
| 3 | Specification file byte-identical; its self-checks on this host print the Mac's three seed lines, shared-bytes line, batch-tensor line and score line exactly; first-batch loss 18.155 against 18.155; taught-panel and 4,096-update stream digests equal the audit's for all three seeds | **pass**. The byte-table line reads **0.0e+00** here (1.4e-06 on the Mac) | `e43_check_3_spec.json` |
| 4 | Every prompt byte of every `unrelated` batch is 128 or above. On every first-family panel item, every prompt byte and every answer byte except the end symbol is below 128 | **pass**: asserted in every run, none failed | runner assertions |
| 5 | Same batches: the batch digest is equal at every row across the four runs of a seed that share a stream (both arms × both rules) | **pass**: 6 groups × 6 rows | `e43_check_5_same_batches.json` |
| 6 | `unrelated` groups read back 0.0, 0.0 at start and end; each arm's decay is in its own `decay` field (the decay arms read 0.25, 0.0) | **pass** | `e43_check_6_decay_groups.json` |
| 7 | One dry run of 64 updates per arm under the sweep | 52–54 CPU-s each, of which scoring was 45 (`dry/`). Forecast about 0.13 s per update plus about 22 s per scored row, so about 660 s per run and about 4.4 CPU-h for 24. Actual: 13,956 CPU-s (3.9 CPU-h) | `dry/*.json` |

**Score 5, decay alone** (no run; all 7,921 pairs per operation, against the audit's 1,500). The reduction score as a share of its start agrees with the audit's table within 3 points at every update count for all six births (`e43_check_score5_decay_<rule>_<seed>.json`). The largest gap is 2.9 points: B 93683 at 1,024, 51.9 here against 49.0. The sweep's birth of seed 73129 gives 99.4, 95.2, 17.9, 0.5 and 0.0, as the coordinator's rerun did.

## The runs

24 runs of 4,096 updates, scored at 0, 64, 256, 1,024, 2,048 and 4,096. Committed weights: row 4,096 of every run (24 files), and rows 64 and 1,024 of seed 73129's `unrelated` arm under each rule (4 files). The other 128 row files are kept locally in `rows_uncommitted/` and not pushed; their sha256 are in the receipts. Every number is in `E43_READ_TABLES.md` and `E43_READ.json`.

### Curves: reduction score / pairs without reduction / max / min (score 1) and the new task (score 3)

The new task is taught/held-out for `unrelated`, and add97/sub97 for `conflict`.

| run | 0 | 64 | 256 | 1,024 | 2,048 | 4,096 |
|---|---|---|---|---|---|---|
| unrelated S 73129 | 96.6/96.0/99.1/99.1 · 0/0 | 0.1/0.1/0.0/0.0 · 0/0 | 0/0/0/0 · 6/8 | 0/0/0/0 · 100/100 | 0/0/0/0 · 100/100 | 0/0/0/0 · 100/100 |
| unrelated S 93683 | 98.7/98.6/98.7/98.5 · 0/0 | 0.1/0.3/0.1/3.7 · 0/0 | 0/0/0/0 · 32/44 | 0/0/0/0 · 100/100 | 0/0/0/0 · 100/100 | 0/0/0/0 · 100/100 |
| unrelated S 116089 | 99.6/99.1/99.4/97.8 · 0/0 | 0.0/0.0/1.3/0.1 · 0/0 | 0/0/0/0 · 31/40 | 0/0/0/0 · 100/100 | 0/0/0/0 · 100/100 | 0/0/0/0 · 100/100 |
| unrelated B 73129 | 98.1/98.8/98.5/97.1 · 0/0 | 0.1/0.0/0.0/0.4 · 0/0 | 0/0/0/0 · 98/98 | 0/0/0/0 · 100/100 | 0/0/0/0 · 100/100 | 0/0/0/0 · 100/100 |
| unrelated B 93683 | 97.8/98.7/98.9/98.5 · 0/0 | 4.9/7.3/0.1/2.9 · 0/0 | 0/0/0/0 · 92/96 | 0/0/0/0 · 100/100 | 0/0/0/0 · 100/100 | 0/0/0/0 · 100/100 |
| unrelated B 116089 | 94.0/94.6/98.6/99.1 · 0/0 | 0.0/0.0/0.1/0.0 · 0/2 | 0/0/0/0 · 100/100 | 0/0/0/0 · 100/100 | 0/0/0/0 · 100/100 | 0/0/0/0 · 100/100 |
| unrelated-decay (all six) | as `unrelated` | 0.0–4.3 on every score 1 column | 0 | 0 | 0 | 0 · 100/100 |
| conflict S 73129 | 96.6/96.0/99.1/99.1 · 16/30 | 17.6/79.6/89.6/98.2 · 62/64 | 3.0/78.9/94.6/98.6 · 88/84 | 0.6/96.3/93.8/99.6 · 93/92 | 0.3/95.3/92.9/99.1 · 91/90 | 0.1/96.2/96.1/96.9 · 94/94 |
| conflict S 93683 | 98.7/98.6/98.7/98.5 · 2/3 | 37.4/70.8/95.4/97.5 · 33/37 | 18.4/71.4/95.7/95.8 · 84/78 | 7.4/60.9/95.5/98.7 · 95/93 | 4.5/65.0/84.3/79.5 · 97/97 | 3.0/55.8/63.6/21.4 · 94/96 |
| conflict S 116089 | 99.6/99.1/99.4/97.8 · 19/4 | 39.1/67.9/97.3/94.8 · 55/40 | 13.8/79.2/95.9/96.4 · 82/81 | 2.7/88.8/99.3/99.4 · 96/96 | 0.6/90.0/96.9/99.4 · 98/92 | 0.7/85.5/98.8/97.2 · 98/97 |
| conflict B 73129 | 98.1/98.8/98.5/97.1 · 5/7 | 10.1/55.8/95.8/95.3 · 52/54 | 0.2/83.3/99.1/91.2 · 78/82 | 0.1/91.0/98.9/76.8 · 92/89 | 0.0/90.1/99.2/69.1 · 95/92 | 0.0/93.1/98.4/49.5 · 96/93 |
| conflict B 93683 | 97.8/98.7/98.9/98.5 · 2/5 | 24.6/48.5/93.9/95.2 · 17/18 | 4.3/51.8/70.6/97.2 · 76/75 | 0.7/47.9/52.3/73.2 · 86/89 | 0.8/35.7/29.3/24.1 · 92/95 | 1.2/39.8/29.4/31.7 · 94/94 |
| conflict B 116089 | 94.0/94.6/98.6/99.1 · 4/5 | 19.4/52.9/95.4/95.9 · 42/30 | 1.2/69.9/95.3/95.4 · 77/76 | 0.1/89.7/97.4/98.1 · 90/94 | 0.1/85.6/97.6/98.9 · 89/92 | 0.2/86.8/98.3/99.1 · 94/94 |

`conflict-decay` is in the tables file. Its reduction score is 0.0 to 5.7 from row 1,024 on.

**The first family's held-out panels at k = 12 (score 2)** are at 0.0% in every `unrelated` and `unrelated-decay` run from row 256 on. In the `conflict` runs add and sub are between 7.7 and 60.0 from row 256 on.

**State change at `=` (score 6, relative to row 0)** at 4,096: `unrelated` 1.04 to 1.41 and `conflict` 0.36 to 0.61. At row 64 `unrelated` is already at 0.39 to 0.86, against 0.21 to 0.30 for `conflict`.

**Observed, outside the brief's scores.** I ran the committed weights of seed 73129's `unrelated` arm on 600 of the first family's held-out prompts. At row 64 the network answered with the end symbol alone on 497 (S) and 511 (B) of them, and with some of the new task's symbols on 82 and 64. At 4,096 the sweep's network answers the end symbol alone on all 600, and backpropagation's on 478, with the new task's symbols on 122.

## The condition

It holds at row 4,096 under both rules. The `unrelated` arm's taught panel is 100, 100, 100 under the sweep and 100, 100, 100 under backpropagation; the held-out panel is 100 in all six. **No extension to 16,384 was needed**, and none was run. Taught panel at row 256: S 6, 32, 31 and B 98, 92, 100.

## Readings (the brief's table only)

`unrelated` at row 4,096, reduction score as a share of its own at row 0:

| Rule | Shares (73129, 93683, 116089) | Reading |
|---|---|---|
| Sweep | 0.0, 0.0, 0.0 | **Lost.** Updating every weight removes even unrelated knowledge; separation of what is learned becomes the first thing to test. |
| Backpropagation | 0.0, 0.0, 0.0 | **Lost** (the same) |

- **`conflict`, the reference:** reduction score 0.1 to 7.4 at row 1,024 (0.1 to 2.7 except S 93683 at 7.4). It is not exactly zero by 1,024 in every run; it is at most 3.0 by 4,096.
- **`unrelated-decay` against decay alone, under the standing decay.** The brief's table gives two readings, and neither applies: it is not 15 or more points above decay alone at 1,024, and it is not within 15 points at all three rows. So the differences are given (points, `unrelated-decay` minus decay alone):

  | Rule, seed | 256 | 1,024 | 2,048 |
  |---|---|---|---|
  | S 73129 | −91.9 | −17.3 | −0.5 |
  | S 93683 | −96.5 | −33.8 | −1.3 |
  | S 116089 | −98.2 | −27.8 | −0.6 |
  | B 73129 | −95.6 | −38.8 | −0.5 |
  | B 93683 | −95.3 | −50.8 | −4.1 |
  | B 116089 | −90.6 | −43.1 | −1.4 |

  Every difference is below decay alone. In the brief's words, below means the new learning removes more than decay does.
- **Next step.** The brief names one only for *kept* or *partly kept*, so none applies.

### Facts beside the reading

- **Share at rows 64, 256 and 1,024 (`unrelated`, reduction score).** At row 64: S 0.1, 0.1, 0.0 and B 0.1, 5.0, 0.0. At rows 256 and 1,024: 0.0 in all six.
- **Same share at row 4,096** for pairs without reduction, for max and for min: 0.0 in all six `unrelated` runs and all six `unrelated-decay` runs.
- **Score 4 at row 4,096**, recurrent matrix (weight_hh and bias_hh), relative norm of the difference from birth / cosine:

  | Rule, seed | conflict | conflict-decay | unrelated | unrelated-decay |
  |---|---|---|---|---|
  | S 73129 | 0.765 / 0.882 | 0.552 / 0.847 | 0.842 / 0.761 | 0.803 / 0.597 |
  | S 93683 | 0.839 / 0.857 | 0.636 / 0.805 | 0.780 / 0.789 | 0.753 / 0.662 |
  | S 116089 | 0.841 / 0.861 | 0.627 / 0.810 | 0.798 / 0.780 | 0.792 / 0.611 |
  | B 73129 | 0.770 / 0.896 | 0.525 / 0.864 | 0.608 / 0.859 | 0.739 / 0.676 |
  | B 93683 | 0.840 / 0.861 | 0.622 / 0.811 | 0.634 / 0.846 | 0.735 / 0.682 |
  | B 116089 | 0.811 / 0.879 | 0.586 / 0.832 | 0.585 / 0.870 | 0.715 / 0.707 |

  Every group and every tensor is in `E43_READ.json` (`facts.<rule>.<arm>_<seed>.movement_4096`). The groups `l1` and `l3` move more under `unrelated` than under `conflict` in all six (for example `l1` relative difference 0.62 to 1.06 against 0.25 to 0.38).

## Predictions

| | Prediction | Chance | Outcome |
|---|---|---|---|
| P43.1 | Check 1 passes under both rules | 85% | **yes** |
| P43.2 | The unrelated task generalises (held-out ≥ 80 at 4,096, two of three) under each rule | 60% | **yes** (100 in all six) |
| P43.3 | `unrelated`, sweep: kept 25%, partly kept 45%, lost 30% | | **lost** |
| P43.4 | Under `unrelated` the reduction score is ≥ 30 points above `conflict`'s at 1,024, two of three, each rule | 75% | no (unrelated 0.0; conflict 0.1 to 7.4) |
| P43.5 | `unrelated-decay` ≥ 15 points above decay alone at 1,024, two of three, sweep | 30% | no (17 to 34 points below) |
| P43.9 | The condition holds at 4,096 on the taught panel under each rule | 75% | **yes** |
| P43.6 | The reading is the same under the sweep and under backpropagation | 75% | **yes** (lost, lost) |
| P43.7 | Under `unrelated`, max and min keep a larger share than the reduction pairs, two of three, sweep | 60% | no (all 0.0) |
| P43.8 | The recurrent matrix moves less under `unrelated` than under `conflict` at 4,096 (score 4), two of three, each rule | 60% | **yes**, read on the relative norm of the difference: S 2 of 3 (93683, 116089), B 3 of 3. On the cosine the sweep's three go the other way (`unrelated` 0.76 to 0.79 against 0.86 to 0.88). |

P43.8 is read on the relative norm of the difference because neither arm has decay. The brief points to the cosine only for the arms under decay.

## Limits

- **The brief's own limits apply.** These are a dose of 4,096 updates, one simple unrelated task, a rule computed by a shared circuit rather than a stored fact, and a fresh AdamW whose first steps move every weight by about the learning rate.
- **Speed of the loss.** It was complete by row 64 in every `unrelated` run, before the new task scored anything (0 to 2 at row 64). The brief lists the optimizer under what it cannot show. Nothing here separates the new task's content from that first-step effect.
- **No independent review** has taken place. Nothing is confirmed, sealed or adopted.
