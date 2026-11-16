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

---

# Addendum 1 (16 November 2026): the weight files, a reminder, a plain-step arm

Brief: addendum 1 of `OPUS_TASK_E43_IS_A_FAMILY_LOST_TO_UNRELATED_LEARNING_20261116.md`, at 436d9677 on `codex/minifly-consolidated-20261023`. New code:
- `e43_addendum.py` (runner), committed at ca524807. The run-to-16,384 option was added at 09064135, before those runs started.
- `e43_addendum_read.py` (reader).

Outputs are in `exp45/e43/addendum1/`: the receipts, `E43_ADDENDUM1_TABLES.md` and `E43_ADDENDUM1_READ.json`. Nothing existing was edited. The CPU for the addendum was 2.8 CPU-h (receipts summed); the brief estimated about 2. The excess is the three reruns to 16,384 below, which the condition required.

## 1. The weight files (an erratum to both reports)

The report above says the weights at update 4,096 of every run, and at 64 and 1,024 of seed 73129's `unrelated` runs, were committed. They were not: the repository's root `.gitignore` excludes `*.pt`, and I did not check what reached the remote. The same is true of E38's 20 files at rows 6,144 and 12,288.

I checked each of the 48 files (28 for E43, 20 for E38) against the `weights_sha256` in its receipt: all 48 match. I added them with `git add -f` in commit 6bde76ad and pushed. On the remote branch, `git ls-tree` lists all 48, and each remote blob's sha256 equals the local file's.

The addendum's own committed rows are added the same way: 37 files, each checked against its receipt. They are the reminders at 128 and 1,024, and the plain runs at 64 and 4,096, plus 16,384 for the reruns.

## 2. The reminder

From the six `unrelated` networks at update 4,096 (no decay), 1,024 updates on the first family. The start files were checked against their receipts. Each run uses the rule that made its file, a fresh `exp26_candidate.make_opt(m)`, E36's batches and E36's scorer (`e36_reminder.Scorer`, `e36_reminder.batches`, imported and not edited).

**Checks: pass** (`addendum1/e43_addendum_reminder_checks.json`):
- row 0 has the start file's `tensors_sha256`;
- row 0's held-out panels and the add/sub wrap and no-wrap counts equal E43's row-4,096 scores;
- the batch digest at row 1,024 equals E36's fresh reference of the same rule and seed, in all six.

### Curves (held-out add/sub mean at k = 12 / wrap score / no-wrap score), with E36's fresh reference

| rule, seed | run | 0 | 16 | 32 | 64 | 128 | 256 | 1,024 |
|---|---|---|---|---|---|---|---|---|
| S 73129 | reminder | 0.0/0.0/0.0 | 1.0/0.5/0.3 | 0.5/0.5/0.7 | 1.8/1.2/0.9 | 0.8/1.7/1.0 | 6.7/5.8/7.3 | **95.2/91.3/94.9** |
| | reference | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0.8/1.1/1.1 | 1.2/1.0/1.2 | 0.2/1.2/0.9 | 0.8/1.1/1.8 |
| S 93683 | reminder | 0.0/0.0/0.0 | 0.0/0.2/0.3 | 1.3/1.0/0.9 | 1.8/1.0/1.3 | 0.7/0.9/0.8 | 1.3/1.1/1.3 | **9.7/10.7/10.8** |
| | reference | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 1.2/1.1/1.1 | 0.5/1.1/1.5 | 1.3/1.1/1.4 | 0.8/1.3/1.2 |
| S 116089 | reminder | 0.0/0.0/0.0 | 0.3/0.1/0.2 | 0.0/0.0/0.0 | 2.0/0.9/0.7 | 0.8/0.8/0.7 | 1.2/1.2/1.3 | **74.2/69.3/72.2** |
| | reference | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 1.7/1.2/1.0 | 1.5/1.1/1.0 | 0.7/0.7/1.2 | 2.0/1.1/4.4 |
| B 73129 | reminder | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0.8/1.0/0.8 | 0.8/1.4/0.8 | 1.2/0.9/1.2 | 1.0/1.7/0.8 | **88.3/91.3/88.9** |
| | reference | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0.8/1.2/1.1 | 0.8/0.9/1.3 | 1.2/1.6/1.1 | 1.0/1.1/1.5 |
| B 93683 | reminder | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0.5/1.5/0.7 | 1.0/1.0/1.3 | **14.2/14.4/15.4** |
| | reference | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 1.2/1.1/1.1 | 0.3/1.0/1.3 | 0.7/1.2/1.0 | 0.8/1.4/1.5 |
| B 116089 | reminder | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0.2/0.0/0.0 | 1.2/1.4/0.8 | 2.0/1.5/2.1 | **45.8/42.0/49.1** |
| | reference | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 1.5/1.1/1.0 | 1.7/1.1/1.0 | 1.5/0.9/1.6 | 3.7/2.5/4.5 |

### Reading (the addendum's table only)

| Rule | Held-out mean at 128 (≥ 85?) | Wrap at 1,024 minus the reference's | Reading |
|---|---|---|---|
| Sweep | 0.8, 0.7, 0.8 (no) | +90.2, +9.4, +68.2 | **Partly back** (73129 and 116089 at +30 or more; 93683 within 15) |
| Backpropagation | 1.2, 0.5, 1.2 (no) | +90.2, +12.9, +39.6 | **Partly back** (73129 and 116089; 93683 within 15) |

No network is back within a hundred updates: every one is at 0 to 7 on every score up to row 256. After that they part, and two of three per rule end above the reference by 40 to 90 points.

## 3. The plain-step arm, `unrelated-plain`

`unrelated` (no decay) with `torch.optim.SGD(m.parameters(), lr)`, no momentum, in place of AdamW. The step is `exp26_candidate.step`'s text. The stream, rows and scores are the `unrelated` arm's.

**The rate.** Three runs on seed 73129 under backpropagation, to 4,096 updates:

| lr | Taught at 4,096 | Held-out | Losses finite |
|---|---|---|---|
| 0.01 | 68.0 | 63.3 | yes |
| 0.03 | **100.0** | 100.0 | yes |
| 0.1 | 100.0 | 100.0 | yes |

**Rate chosen: 0.03**, the smallest at which the taught panel is at 80 or more. The run of seed 73129 under backpropagation at 0.03 is the rate run itself (same code, birth, stream and rate), so it was not repeated. The 0.01 and 0.1 runs are kept as receipts.

**The condition.**
- **Backpropagation:** holds at 4,096 (taught 100, 100, 100).
- **Sweep:** fails at 4,096 (taught 0.3, 36.3, 95.3). As the brief's condition requires, the three sweep runs were continued to 16,384. My runner keeps no checkpoint, so each was rerun from its birth to 16,384. Its rows 0 to 4,096 have the same `tensors_sha256` and batch digests as the 4,096 runs (`addendum1/e43_addendum_plain16384_checks.json`, pass). The condition holds at **8,192** (taught 68.0, 99.3, 100), and the reading is taken there.

### Curves (score 1: reduction / pairs without reduction / max / min · new task taught / held-out · state change at `=`)

| rule, seed | 0 | 64 | 256 | 1,024 | 4,096 | 8,192 | 16,384 |
|---|---|---|---|---|---|---|---|
| S 73129 | 96.6/96.0/99.1/99.1 · 0/0 | 16.7/3.5/18.3/16.0 · 0/0 · 0.35 | 6.6/1.3/3.5/0.5 · 0/0 | 3.8/0.8/0.7/0.0 · 0/0 | 0.7/0.4/0.8/0.0 · 0.3/0.3 | 0/0/0/0 · 68.0/55.3 | 0/0/0/0 · 99.3/93.0 |
| S 93683 | 98.7/98.6/98.7/98.5 · 0/0 | 45.2/49.7/60.4/73.7 · 0/0 · 0.32 | 25.7/27.7/10.1/33.2 · 0/0 | 4.9/6.6/0.1/5.6 · 0/0 | 0/0/0/0 · 36.3/20.7 | 0/0/0/0 · 99.3/97.3 | 0/0/0/0 · 100/100 |
| S 116089 | 99.6/99.1/99.4/97.8 · 0/0 | 19.7/11.9/76.0/66.2 · 0/0 · 0.33 | 6.4/4.4/27.0/8.0 · 0/0 | 0.1/0.0/2.8/0.0 · 0.3/0.7 | 0/0/0/0 · 95.3/95.0 | 0/0/0/0 · 100/100 | 0/0/0/0 · 100/100 |
| B 73129 | 98.1/98.8/98.5/97.1 · 0/0 | 0.4/0.5/0.7/0.9 · 0/0 · 0.66 | 0/0/0/0 · 0/0 | 0/0/0/0 · 35.7/21.7 | 0/0/0/0 · 100/100 | – | – |
| B 93683 | 97.8/98.7/98.9/98.5 · 0/0 | 2.8/2.7/0.1/0.2 · 0/0 · 0.50 | 1.1/0.8/0.0/0.0 · 0/0 | 0/0/0/0 · 21.0/10.3 | 0/0/0/0 · 100/100 | – | – |
| B 116089 | 94.0/94.6/98.6/99.1 · 0/0 | 2.1/0.3/0.4/0.8 · 0/0 · 0.52 | 0.1/0/0/0 · 1.0/0.7 | 0/0/0/0 · 70.3/68.0 | 0/0/0/0 · 100/100 | – | – |

Row 2,048 is in the tables file.

### Reading (the brief's table, unchanged)

| Rule | Row | Share of the reduction score (73129, 93683, 116089) | Reading |
|---|---|---|---|
| Sweep | 8,192 | 0.0, 0.0, 0.0 | **Lost** |
| Backpropagation | 4,096 | 0.0, 0.0, 0.0 | **Lost** |

The addendum states the consequence in advance: if the family is lost here too, the equal-sized step on every weight is not the cause.

**Share at rows 64, 256 and 1,024** (reduction score; reported beside the reading):
- Sweep: 17.3 / 6.8 / 4.0; 45.8 / 26.1 / 5.0; 19.8 / 6.5 / 0.1.
- Backpropagation: 0.4 / 0.0 / 0.0; 2.8 / 1.1 / 0.0; 2.3 / 0.1 / 0.0.
- For comparison, under AdamW at row 64: sweep 0.1, 0.1, 0.0; backpropagation 0.1, 5.0, 0.0.

## Predictions

| | Prediction | Chance | Outcome |
|---|---|---|---|
| P43.10 | Reminder: back at row 128 (held-out mean ≥ 85), two of three seeds, sweep | 50% | no (0.7 to 0.8) |
| P43.11 | Reminder: at row 1,024 the wrap score is ≥ 30 above the fresh reference's, two of three seeds, under each rule | 85% | **yes** (73129 and 116089 under each rule) |
| P43.12 | Some rate of 0.01, 0.03, 0.1 meets the condition on seed 73129 under backpropagation | 70% | **yes** (0.03 and 0.1) |
| P43.13 | `unrelated-plain`, backpropagation: kept 15%, partly kept 30%, lost 55% | | **lost** |
| P43.14 | Under `unrelated-plain` the share at row 64 is above 50% in two of three seeds, backpropagation | 55% | no (0.4, 2.8, 2.3) |

## Limits

- **Running the sweep's plain arm to 16,384** follows the brief's condition, which the addendum keeps unchanged. Read at 8,192, that arm had twice the dose of the backpropagation arm read at 4,096.
- **Not independently reviewed.** No result here has been independently reviewed.

---

# Addendum 2 (16 November 2026): a rehearsed unrelated arm, and E42's networks

Brief: addendum 2 of `OPUS_TASK_E43_IS_A_FAMILY_LOST_TO_UNRELATED_LEARNING_20261116.md`, at bd9f18fe on `codex/minifly-consolidated-20261023` (merged into this branch first).

New code:
- `e43_addendum2.py` (runner) and `e43_addendum2_read.py` (reader).
- `exp42_cells.py` and `e42_cells.py`: byte-identical copies from `claude/gallant-dijkstra-i9zgsr` at 534df8f0. Their sha256 (340f9cc5…, 985814d0…) are the ones recorded in E42's four receipts of seed 73129.

Nothing existing was edited. Outputs are in `exp45/e43/addendum2/`: receipts, `E43_ADDENDUM2_TABLES.md` and `E43_ADDENDUM2_READ.json`. CPU was 2.4 CPU-h for the sixteen runs (receipts summed); the brief estimated 2.5.

## 4. `unrelated-rehearse` and `unrelated-rehearse-decay`

**Set-up.** The `unrelated` arm, on the six births, without decay and with 0.25.
- **Stream:** each item comes from one of two streams with equal chance, chosen by `Random(f"e43|{seed}|joint").randrange(2)`. Value 0 takes the next item of `e43_task.stream(seed)`; value 1 takes the next of E38's rehearsal stream `X16.entry_stream(first family's taught half, seed + 100)`. The order (0 = unrelated, 1 = first family) is my choice; the brief did not fix it.
- **Batches:** 64 consecutive items.
- **Item counts at 4,096:** 131,138 to 131,384 unrelated items and 130,760 to 131,006 first-family items, recorded per row.
- **Assertion on every item:** unrelated prompts are all at byte 128 or above; first-family prompts all below 128.

### Curves (reduction score · unrelated task taught; the full score-1 columns and item counts are in the tables file)

| run | 0 | 64 | 256 | 1,024 | 2,048 | 4,096 |
|---|---|---|---|---|---|---|
| rehearse S 73129 | 96.6 · 0 | 79.6 · 0 | 78.1 · 0 | 92.6 · 100 | 90.7 · 100 | 97.1 · 100 |
| rehearse S 93683 | 98.7 · 0 | 97.9 · 0 | 89.0 · 3.3 | 96.5 · 99.3 | 98.7 · 100 | 98.9 · 100 |
| rehearse S 116089 | 99.6 · 0 | 92.6 · 0 | 82.7 · 5.0 | 95.6 · 99.7 | 95.0 · 100 | 96.1 · 100 |
| rehearse B 73129 | 98.1 · 0 | 94.2 · 0 | 83.9 · 74.3 | 91.9 · 100 | 90.6 · 99.7 | 96.7 · 100 |
| rehearse B 93683 | 97.8 · 0 | 95.8 · 0 | 91.9 · 50.0 | 96.8 · 100 | 95.4 · 100 | 98.3 · 100 |
| rehearse B 116089 | 94.0 · 0 | 94.4 · 0 | 90.4 · 88.0 | 92.6 · 100 | 95.1 · 100 | 98.7 · 100 |
| rehearse-decay S 73129 | 96.6 · 0 | 78.1 · 0 | 78.0 · 0 | 88.0 · 98.3 | 90.5 · 99.7 | 93.5 · 100 |
| rehearse-decay S 93683 | 98.7 · 0 | 97.7 · 0 | 87.6 · 2.3 | 95.2 · 100 | 97.7 · 100 | 98.3 · 100 |
| rehearse-decay S 116089 | 99.6 · 0 | 91.4 · 0 | 85.4 · 5.0 | 95.7 · 99.7 | 96.5 · 100 | 97.4 · 100 |
| rehearse-decay B 73129 | 98.1 · 0 | 94.1 · 0 | 81.4 · 75.3 | 87.8 · 100 | 89.1 · 100 | 93.1 · 100 |
| rehearse-decay B 93683 | 97.8 · 0 | 95.7 · 0 | 93.6 · 50.7 | 95.4 · 100 | 96.2 · 100 | 96.9 · 100 |
| rehearse-decay B 116089 | 94.0 · 0 | 94.2 · 0 | 87.8 · 88.7 | 93.2 · 99.3 | 95.4 · 100 | 91.3 · 100 |

### Reading (the addendum's table only)

| Arm, rule | Share of the reduction score at 4,096 (73129, 93683, 116089) | Taught panel at 4,096 | Reading |
|---|---|---|---|
| `unrelated-rehearse`, sweep | 100.5, 100.3, 96.5 | 100, 100, 100 | **kept at no cost** |
| `unrelated-rehearse`, backpropagation | 98.6, 100.5, 105.0 | 100, 100, 100 | **kept at no cost** |
| `unrelated-rehearse-decay`, sweep | 96.8, 99.6, 97.8 | 100, 100, 100 | **kept at no cost** |
| `unrelated-rehearse-decay`, backpropagation | 94.8, 99.0, 97.2 | 100, 100, 100 | **kept at no cost** |

**Beside the reading: the first row at which the taught panel reaches 80%** (rows are 0, 64, 256, 1,024, …). Each cell gives the update, then the unrelated items seen by then.
- Comparison arms: `unrelated` for the rehearse arm and `unrelated-decay` for the rehearse-decay arm.
- Every comparison arm has 64 unrelated items per update.

| Arm, rule | 73129 | 93683 | 116089 | Without rehearsal (all three) |
|---|---|---|---|---|
| rehearse, sweep | 1,024 / 32,755 | 1,024 / 32,833 | 1,024 / 32,911 | 1,024 / 65,536 |
| rehearse, backpropagation | 1,024 / 32,755 | 1,024 / 32,833 | 256 / 8,333 | 256 / 16,384 |
| rehearse-decay, sweep | 1,024 / 32,755 | 1,024 / 32,833 | 1,024 / 32,911 | 1,024 / 65,536 |
| rehearse-decay, backpropagation | 1,024 / 32,755 | 1,024 / 32,833 | 256 / 8,333 | 256 / 16,384 |

- **Under the sweep** both arms reach 80% at the same row with and without rehearsal, on half the unrelated items.
- **Under backpropagation** two seeds reach it one row later with rehearsal: 1,024 against 256, and 32,755 to 32,833 unrelated items against 16,384. Their taught panels at row 256 were 74.3 and 50.0.
- The rows are coarse: nothing is measured between 256 and 1,024.

## 5. E42's networks: the `unrelated` arm from `F10` and `P8`, seed 73129

**Set-up.**
- **Start:** E42's final weights and structure of each run: the mask, the periods and phases, `made`, and for P8 the learned period coefficients `a`. Each start file's sha256 equals its E42 receipt's `weights_sha256`. The loaded network's `tensors_sha256`, mask, `made` and `a` digests equal the receipt's, and E42's limit check passes.
- **Training:** `exp42_cells.step` with E42's forward; a fresh `exp42_cells.make_opt` with the decayed group's `weight_decay` set to 0; no replacement event; 4,096 updates.
- **Row-0 check: pass.** E42's own scorer (`exp42_cells.score_all`) at row 0 gives the A, R, taught and length counts of the receipt's final row, in all four.

| Run | 0 | 64 | 256 | 1,024 | 2,048 | 4,096 | Reading (share at 4,096) |
|---|---|---|---|---|---|---|---|
| F10 S | 91.2/93.6/98.7/96.7 · 0 | 1.0/0.1/9.5/2.7 · 0 | 0.3/0.0/3.2/0.1 · 0 | 0/0/0/0 · 88.7 | 0/0/0/0 · 97.3 | 0/0/0/0 · 99.3 | **Lost** (0.0) |
| F10 B | 95.1/94.3/97.6/96.8 · 0 | 0/0/0/0 · 0 | 0/0/0.1/0 · 7.7 | 0/0/0/0 · 100 | 0/0/0/0 · 100 | 0/0/0/0 · 100 | **Lost** (0.0) |
| P8 S | 94.3/95.7/98.5/98.1 · 0 | 0.3/0.3/28.8/11.1 · 0 | 0/0/0.9/0.7 · 3.3 | 0/0/0/0 · 100 | 0/0/0/0 · 100 | 0/0/0/0 · 100 | **Lost** (0.0) |
| P8 B | 97.7/97.7/98.1/98.4 · 0 | 0/0/6.9/3.2 · 0 | 0/0/0/0.1 · 27.7 | 0/0/0/0 · 100 | 0/0/0/0 · 100 | 0/0/0/0 · 100 | **Lost** (0.0) |

Cells: reduction / pairs without reduction / max / min · unrelated task taught. The condition (taught panel at 80% or more at 4,096) holds in all four runs. Held-out panel of the unrelated task at 4,096: 96.3, 97.7, 100 and 100.

## Predictions

| | Prediction | Chance | Outcome |
|---|---|---|---|
| P43.15 | `unrelated-rehearse`, sweep: *kept at no cost* | 60% | **yes** |
| P43.16 | `unrelated-rehearse-decay`, sweep: *kept at no cost* | 65% | **yes** |
| P43.17 | With rehearsal the taught panel reaches 80% no more than twice as many updates later than without, two of three seeds, under each rule | 60% | **no**: the sweep holds (3 of 3: same row), but under backpropagation only 116089 does (73129 and 93683: 1,024 against 256, four times) |
| P43.18 | E42's `F10` and `P8`: *lost* in all four runs | 85% | **yes** |

## Committed weights (from `git ls-tree -r -l origin/claude/peaceful-hopper-8ofq4i`, after the push)

Each file was checked against its receipt's `weights_sha256` before `git add -f`. Columns: blob, bytes, path.

```
    6ac273d785e9e008be870ea12d11e0b5f1fefda1  4502044  MOD97_LOCAL_RULE_20261106/exp45/e43/addendum2/e43_unrelated-e42F10_B_73129_row4096.pt
    f7dcabf1b7e9e717c701642799066fd7d6b9f426  4502044  MOD97_LOCAL_RULE_20261106/exp45/e43/addendum2/e43_unrelated-e42F10_S_73129_row4096.pt
    9bf01f3ec6d17d4036bdf4f5f7d73ac4f458b781  4502025  MOD97_LOCAL_RULE_20261106/exp45/e43/addendum2/e43_unrelated-e42P8_B_73129_row4096.pt
    c4406da5ba66f94140039e3d4a46e019b0dd8cb6  4502025  MOD97_LOCAL_RULE_20261106/exp45/e43/addendum2/e43_unrelated-e42P8_S_73129_row4096.pt
    722a2f1a0deaa1420ae23c80d59e02d5903afc95  4502215  MOD97_LOCAL_RULE_20261106/exp45/e43/addendum2/e43_unrelated-rehearse-decay_B_116089_row4096.pt
    6774cced0ee48ca365bfaba988fb8b352fbcb8c5  4502196  MOD97_LOCAL_RULE_20261106/exp45/e43/addendum2/e43_unrelated-rehearse-decay_B_73129_row4096.pt
    578955e920d2279a5ce6b1a0028c6beb42c13634  4502196  MOD97_LOCAL_RULE_20261106/exp45/e43/addendum2/e43_unrelated-rehearse-decay_B_93683_row4096.pt
    a435d12e3bd8b90a586ecb0ad28a0da7272e304d  4502215  MOD97_LOCAL_RULE_20261106/exp45/e43/addendum2/e43_unrelated-rehearse-decay_S_116089_row4096.pt
    6d0504b573dc6e83033992940be86e24a1a4e24c  4502196  MOD97_LOCAL_RULE_20261106/exp45/e43/addendum2/e43_unrelated-rehearse-decay_S_73129_row4096.pt
    5390097a78eab10560c0ca4a01d1ec624f8db543  4502196  MOD97_LOCAL_RULE_20261106/exp45/e43/addendum2/e43_unrelated-rehearse-decay_S_93683_row4096.pt
    fdc1d54a3043af705d2b356373f192c1d81efd19  4502101  MOD97_LOCAL_RULE_20261106/exp45/e43/addendum2/e43_unrelated-rehearse_B_116089_row4096.pt
    9b9c9d04c15f516c3045e0cc732f822d59716cb4  4502082  MOD97_LOCAL_RULE_20261106/exp45/e43/addendum2/e43_unrelated-rehearse_B_73129_row4096.pt
    f70f411e388fae3cf71f3040e0242c02f9d26644  4502082  MOD97_LOCAL_RULE_20261106/exp45/e43/addendum2/e43_unrelated-rehearse_B_93683_row4096.pt
    014b726b8a7ead1c055247d347580b7700467b96  4502101  MOD97_LOCAL_RULE_20261106/exp45/e43/addendum2/e43_unrelated-rehearse_S_116089_row4096.pt
    0d62543ec1dd7bbc01c96b78f0eeed408c7aa065  4502082  MOD97_LOCAL_RULE_20261106/exp45/e43/addendum2/e43_unrelated-rehearse_S_73129_row4096.pt
    0573a68be43d51d64878185352b87b7ab9327558  4502082  MOD97_LOCAL_RULE_20261106/exp45/e43/addendum2/e43_unrelated-rehearse_S_93683_row4096.pt
```

Including the earlier additions, the remote now holds 101 weight files under `exp45/e38/` and `exp45/e43/`: 20 for E38, 28 for E43, 37 for addendum 1 and 16 here.

## Limits

- **The P8 row files hold the network's `state_dict` only.** They omit the learned period coefficients `a` (8 × 256), which change in training. The mask, periods and phases are E42's, unchanged, since no replacement event was run. So the P8 networks at 4,096 cannot be rebuilt from the committed files alone; the receipts record only `a`'s relative change, not its values. F10 has no added parameters and is complete. Rerunning the two P8 runs with `a` saved would take about 1.3 CPU-hours.
- **Not independently reviewed.** No result here has been independently reviewed.
