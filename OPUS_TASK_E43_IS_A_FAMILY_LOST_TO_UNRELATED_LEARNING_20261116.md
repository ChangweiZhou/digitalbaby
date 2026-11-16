# E43: is a family lost when the learner then learns something unrelated? (16 November 2026, revision 1)

Evidence level E2. The question, the arms' outline and the reading are the outside reviewer's (its memo, revision 14, section 24.5, on `claude/vibrant-cerf-sje726` at 57497920), written after the user's objection of 16 November. This specification, the unrelated task and the predictions are mine. **Not released: it waits for the reviewer's audit.** One worker, best the E38 worker, whose runner already does the `conflict` arms.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; K=PROJECT_DOCUMENTS/COORD_CHECKS_20261116`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit. Edit nothing existing.

## The objection this answers

The user, 16 November: by the three findings so far, Einstein would have to review world geography at intervals in order to recall his life in Germany. Long-term memory exists as a matter of fact; what have we left out?

What has been measured is only the most adverse case. Every life on file consists of families that contradict the earlier ones: the same operands and the same operation symbol, another modulus. There the old family's modulus is gone within 64 to 256 updates (E38), under every rule, with or without decay. The user's example is the other case: later learning that neither repeats nor contradicts the memory. It has never been measured here.

## What is on file

- **E38, seed 73129, the first family's reduction pairs (whole answers, no letters) while the second family is learned:** 97% at the start, 18% after 64 updates, 3% after 256, 0% after 1,024, alike with decay 0.25 (`zero`) and with none (`none`), under the sweep and under backpropagation.
- **At the end of the nine lives** (the reviewer's read of the saved files; I did not rerun it): reduction pairs 0 to 1.7%; pairs without reduction 2 to 93%; max or min at 92% or more in 8 of 18 cases and at 66% or less in the other 10, the surviving operation going with the seed and not with the rule.

## Arms: four, from each of the six E26 births

Starts: `exp45/e26/e26_dev_S_<seed>.pt` continued under the sweep and `e26_dev_B_<seed>.pt` under backpropagation, three seeds, each admitted by `exp26_candidate.admit_e26`. Update: `exp26_candidate.step`'s text; a fresh `exp26_candidate.make_opt(m)`; 4,096 updates. 24 runs.

| Arm | Stream | Decay |
|---|---|---|
| `conflict` | the second family alone (mod 97): `X16.entry_stream(X16.family_entries("F1", seed), seed + 101)`, as the first block of E26's life | none (`weight_decay` 0 in group 0, as E38's arm `none`) |
| `conflict-decay` | the same | 0.25, as E26's life and E38's arm `zero` |
| `unrelated` | the task below | none |
| `unrelated-decay` | the same | 0.25 |

The memo names the two arms without decay. I add the two with decay because 0.25 is the standing setting of every life, and because in the unrelated case decay is the one thing that could act on weights the new task does not use: at rate 0.001 and decay 0.25 a weight nothing pushes on keeps 36% of its size after 4,096 updates. In E38 decay made no difference, but there the conflict removed the family first.

## The unrelated task, as code

**`$K/e43_unrelated_task_spec.py` on the default branch is the specification.** Copy it to `$M/e43_task.py` and import it; do not rewrite it. In words:

- **Bytes.** 64 symbols (bytes 128 to 191), 32 fillers (192 to 223), a code byte (224) and a query byte (225). No family's prompt or answer uses any of them; the families use digits, `+ - > < = ;`, upper-case letters and some lower-case codes, all below 128.
- **Item.** `code s1 s2 f_1 … f_k query`, then the answer `s1 s2` and the end symbol. k is 0 to 12, as in the families. The answer is two bytes and the end symbol, so the harness, the answer mask and the scorer are unchanged.
- **Pairs.** The 4,096 ordered pairs of symbols are shuffled by `Random(f"e43|{seed}|split")`; the first 2,048 are taught, the rest held out.
- **Stream.** `Random(f"e43|{seed}|stream")`: the taught pairs in a fresh shuffled order each pass; for each item `randint(0, 12)` for k, then k fillers by `choice`. Batches of 64 consecutive items.
- **Panels.** `Random(f"e43|{seed}|panel|held")` samples 300 held-out pairs and renders each at k = 12; the same with `taught`.

**What it shares with the first family, stated plainly.** No input byte. It does share: the whole network, since every weight but the embedding rows of other bytes takes part; the start symbol; the end symbol as the third answer byte; the length of items and the format of two answer bytes. It asks for memory across fillers, as the families do across letters. So it is unrelated in content and not in machinery, which is the user's case.

**What I checked without training** (`$K/OUTPUT_E43_SPEC_MAC.txt`, on E26's sweep birth of seed 73129): the taught and held-out pairs are disjoint; the items pass through `exp45.batch_tensor` and `exp11_cell.score` unchanged; the born network scores 0 of 300 on the held-out panel before any update, so there is no starting level to subtract for the new task; the first-batch loss is 18.2 with finite gradients. The embedding rows of the task's bytes differ from the initial network's by at most 1.4e-6 in the birth files, and are bit-equal between a birth and its life: no training has shaped them.

## Rows and scores

Score at updates 0, 64, 256, 1,024 and 4,096. Save weights at each; commit those at 4,096 (24 files) and record the sha256 of the rest.

1. **First family, all pairs without letters, split by reduction** (as E36's score 2): add and sub, pairs with a + b ≥ 89 or a − b < 0 against the rest; and max and min on all pairs. **The reduction score** is the mean of the add and sub reduction-pair shares.
2. **First family, held out at 12 letters:** `exp16_integrated.panels(seed)[0]` entries `add89`, `sub89`, `max89`, `min89`.
3. **The new task:** for the `unrelated` arms, the held-out and taught panels above; for the `conflict` arms, `add97` and `sub97` at 12 letters from the same `panels` call.
4. **How much the weights moved:** for each of the groups (embedding, `l1`, input map, recurrent matrix, `lm`, `l2`, `l3`), the norm of the difference from the birth divided by the norm at birth.

Score 4 is there because "kept" would mean little if the unrelated task were learned with a small fraction of the change that the second family causes. It is reported beside the reading for both pairs of arms.

## Checks before the full runs

1. **The `conflict` arms are E38's.** On seed 73129, `conflict-decay` has E38's `zero` row hashes and `conflict` has E38's `none` row hashes at updates 64, 256, 1,024 and 4,096, under each rule (`tensors_sha256` in `exp45/e38/e38_<zero|none>_<S|B>_73129.json`; for the sweep, `zero`: f3819049…, 95f5de74…, e2d170be… at 64, 256 and 1,024).
2. **Row 0** of every run has the birth file's `tensors_sha256`, and its first-family scores equal E38's row 0 for seed 73129.
3. **The specification file.** `$M/e43_task.py` is byte-identical to `$K/e43_unrelated_task_spec.py`; run on `$K`, its self-checks print what `$K/OUTPUT_E43_SPEC_MAC.txt` has, the stream and panel digests of the three seeds included.
4. **No shared input.** Asserted on every batch of the `unrelated` arms: every prompt byte is 128 or above. And on every first-family panel item: every byte is below 128.
5. **The same batches** for the four runs of a seed that share a stream: the batch digest is equal at every row.
6. **`unrelated` has no decay:** the optimizer's groups read back with `weight_decay` 0.0 in both; each arm's decay is recorded in a field of its own.
7. One dry run of 64 updates per arm under the sweep, with CPU time and a forecast (the reviewer's estimate: about 2 CPU-hours with scoring).

Outputs go to `exp45/e43/`, named `e43_<arm>_<S|B>_<seed>` with `_row<updates>`; the runner calls `arch26.install()` first and refuses an existing path.

## Reading (fixed beforehand; the reviewer's, with one condition added)

**The condition.** The reading is taken for a rule only if the unrelated task was learned: in at least two of three seeds the `unrelated` arm's held-out panel is at 80% or more at row 4,096. If not, the report says that the new task was not learned in the dose, gives the curves and score 4, and draws nothing about retention.

**Per rule, on the `unrelated` arm**, with each run's reduction score at row 4,096 as a share of its own at row 0, in at least two of three seeds:

| Share kept | Reading |
|---|---|
| 80% or more | **Kept.** The fast loss measured so far is loss by contradiction; the learner holds what later learning does not contradict, over this dose. |
| 20% or less | **Lost.** Updating every weight removes even unrelated knowledge; separation of what is learned becomes the first thing to test. |
| Otherwise, or the seeds disagree | **Partly kept.** The shares are given per seed. |

- `conflict` is the reference and is expected at zero by row 1,024.
- `unrelated-decay` is read by the same table and reported beside it as "under the standing decay". If it differs from `unrelated`, decay alone erodes what nothing contradicts, at the life's standing setting.
- Reported beside the reading, as facts: the same share for pairs without reduction, for max and for min; the share at rows 64, 256 and 1,024; score 4 for all four arms.

## What it cannot show

- **A dose of 4,096 updates.** A life is 36,864. "Kept" here is kept over a ninth of a life.
- **One unrelated task**, a simple one. A harder unrelated task would move the weights more.
- **Long-term memory as the user means it.** Nothing here is a fact stored once and recalled by a cue; the first family is a rule computed by a shared circuit. This job separates loss by contradiction from loss by any further learning, and no more.
- **Why.** If the family is kept, the job does not say what protects it; if lost, not which weights carried the loss.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P43.1 | Check 1 passes under both rules | 85% |
| P43.2 | The unrelated task is learned (held-out panel at 80 or more at row 4,096, two of three seeds) under each rule | 70% |
| P43.3 | `unrelated`, sweep: kept 25%, partly kept 45%, lost 30% | |
| P43.4 | Under `unrelated` the reduction score is above `conflict`'s by at least 30 points at row 1,024, two of three seeds, under each rule | 75% |
| P43.5 | `unrelated-decay`'s share kept is at least 15 points below `unrelated`'s at row 4,096, two of three seeds, under the sweep | 45% |
| P43.6 | The reading is the same under the sweep and under backpropagation | 75% |
| P43.7 | Under `unrelated`, max and min keep a larger share than the reduction pairs, two of three seeds, under the sweep | 60% |
| P43.8 | The recurrent matrix moves less under `unrelated` than under `conflict` at row 4,096 (score 4), two of three seeds, under each rule | 60% |

## Done

Runner, check receipts, 24 runs with receipts and the committed weights, and `E43_REPORT.md` with the curves (run × row: the four scores), whether the condition holds, the reading per rule, the facts listed, and the outcome of each prediction. No readings beyond the table above.
