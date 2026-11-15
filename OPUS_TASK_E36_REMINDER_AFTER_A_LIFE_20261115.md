# E36 (job A): after a life, how many updates bring the unrehearsed birth family back? (15 November 2026, revision 2)

Evidence level E2. The question, the design and the proposed marks are the outside reviewer's (its memo `MEMO_TO_COORDINATOR_LEARNED_STRUCTURE_20261115.md` on `claude/vibrant-cerf-sje726` at a8b5ace2, section 5); this specification and the predictions are mine. Revision 1 (4dd0b9f7) was audited by the reviewer (`AUDIT_E36_E37_E38_BRIEFS_20261115.md` on `claude/vibrant-cerf-sje726` at 823a3884), who stated that with its required changes made as written no re-audit is needed. Revision 2 makes them and the recommendations; the last section lists them. **Released.** One worker.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106`. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each; launch outside any wrapper with a time limit. Edit nothing existing; do not call or patch existing `dev`, `life` or `step` functions except where a line below names one to call.

## Why

The E26 and E31 lives teach families F1 (mod 97), F2 (compositions) and F3 (mod 73). The birth family (the four mod-89 operations) is never in a life's stream: `exp26_candidate.life` builds `entries` for F1, F2 and F3 only (`git show $D:$M/exp26_candidate.py | sed -n '214p;242,244p'`). The life receipts score it anyway, and it is lost in the first block. Held-out add and sub mod 89 at 12 letters, mean of the two, by stage 0 to 6, from the receipts (my read of `exp45/e26/e26_life_<arm>_<seed>.json`, `stages[i].scores.add89["12"]` and `sub89`):

| Life | Stages 0 to 6 |
|---|---|
| S 73129 | 98.0, 49.0, 47.8, 44.2, 33.2, 31.0, 33.0 |
| S 93683 | 98.2, 6.8, 2.0, 1.0, 0.2, 11.7, 17.7 |
| S 116089 | 99.8, 34.0, 29.3, 9.2, 3.2, 14.0, 13.5 |
| B 73129 | 98.5, 44.2, 47.5, 35.5, 34.5, 40.2, 37.3 |
| B 93683 | 98.7, 1.2, 1.0, 0.7, 0.0, 23.2, 26.7 |
| B 116089 | 93.8, 32.7, 22.7, 34.0, 26.3, 38.0, 36.3 |
| K 73129 | 98.0, 45.5, 47.7, 43.0, 38.0, 41.5, 31.2 |
| K 93683 | 98.2, 4.8, 8.8, 1.7, 0.8, 23.2, 36.5 |
| K 116089 | 99.8, 28.2, 30.8, 5.5, 1.0, 27.0, 29.5 |

The reviewer's reading of the saved files is that what is lost is mainly the old modulus: pairs that need the wrap at 89 are right 0.0 to 1.3% of the time, pairs that do not 2.5 to 93% (whole answers with the end symbol, as `exp16_integrated.score` counts; the K rows of the table above are the audit's read). A network that has lost the wrap and nothing else still scores about 50 on the held-out panel, which is why the marks below read the wrap pairs. E36 asks whether the family's rule is still in the network, by measuring how fast a reminder restores it.

## Runs: 9 reminders and 6 references, 1,024 updates each

| Kind | Start | Rule | Seeds |
|---|---|---|---|
| Reminder S | `exp45/e26/e26_life_S_<seed>.pt` | the sweep: `arch26.forward(m, x, True, True)`, answer-masked loss, as `exp26_candidate.step` (lines 84-91) | 3 |
| Reminder B | `exp45/e26/e26_life_B_<seed>.pt` | backpropagation: the same with `local=False` | 3 |
| Reminder K | `exp45/e31/e31_life_K_<seed>.pt` and `e31_life_K_<seed>_Bhh.pt` | the sweep with Kolen–Pollack feedback on the recurrent map, continued from the saved matrix | 3 |
| Reference S, B | `arch26.new_net(seed)`, untrained | sweep; backpropagation | 3 each |

- **Arm K's code.** The K lives were made by worker B's files, now `exp31_switch_workerB.py` and `arch31_workerB.py` (byte-identical to that worker's originals; the receipts still say `exp31_switch.py` and `arch31.py`). The runner cannot be imported as it stands: its line 43 is `import arch31`, which on the default branch is worker A's file with a different `forward`. Write the K step out in the new runner: `exp26_candidate.step`'s text with `arch31_workerB.forward(m, x, True, True, switch="K", Bhh=B)` in place of `arch26.forward(...)`; before `opt.step()`, `W0 = m.rnn.weight_hh_l0.detach().clone()`; after it, `B = (1 - 0.00025) * B + (W1 - (1 - 0.00025) * W0)` with `W1` the recurrent matrix after the step (as `exp31_switch_workerB.py` lines 187 and 196 to 202, and lines 74 to 77). `B` starts as `T.load(<the _Bhh.pt file>)["Bhh"]`.
- **Optimizer.** A fresh `exp26_candidate.make_opt(m)` in every run. The lives saved no optimizer state.
- **Data, the same batches for every run of a seed.** `exp11_cell.batches([(t, exp11_cell.split(t, .5, seed)[0]) for t in exp11_cell.DEV], seed + 36, 12)`: the birth's taught half of the pairs, 0 to 12 letters, with a stream seed that is not the birth's.
- **Rows.** Score and save weights at 0, 16, 32, 64, 128, 256, 512 and 1,024 updates. At 256 updates each taught pair has been shown about once. Commit (`git add -f`) the weights at 256 and 1,024 (30 files); record the sha256 of the others.

## Scores at every row

1. **Birth family, held out:** `exp16_integrated.panels(seed)[0]` entries `add89`, `sub89`, `max89`, `min89` at k = 12 (300 held-out pairs each; these are the panels the life receipts use), by `exp16_integrated.score`.
2. **Birth family, all pairs, no letters, split by wrap:** for `add89` and `sub89`, every one of the 7,921 pairs rendered with k = 0; "wrap" is a + b ≥ 89 for add and a − b < 0 for sub. Report exact answers for wrap and no-wrap pairs separately.
3. **The life's families:** `add97`, `sub97`, `add73`, `sub73` at k = 12 and `compose["within"]`, from the same `panels` call. These show what the reminder costs.

## Checks before the full runs

1. Row 0 of each reminder has the life-end file's `tensors_sha256` and reproduces the last stage's `add89`, `sub89`, `max89`, `min89` scores in the life receipt exactly.
2. The batch digest (`exp26_candidate.Digest`) is equal across all runs of a seed at every row.
3. Arm K: `exp20_reference.tensors_sha({"Bhh": B})` of the loaded matrix equals `e31.Bhh_final_tensors_sha256` in the life receipt. On the first batch, with `Bhh` set to the current `weight_hh_l0`, the K step's parameter gradients equal those of `switch="H"` to 1e-6 relative (worker B's own check, lines 383 to 386), and with the loaded matrix they differ from those of `switch=None`. The difference `Bhh − W` shrinks by 1 − 0.00025 per step at updates 1, 10 and 100.
4. **The update path against E26's dry births.** From `arch26.new_net(73129)` with `make_opt`, 2,048 updates on the birth's own stream (`exp11_cell.batches(tp, 73129, 12)`, `tp` as in the Data bullet) give the `tensors_sha256` of `exp45/e26_qual/e26_dev_S_dry2048_73129.json` (79618ab2…) under the sweep and of `e26_dev_B_dry2048_73129.json` (adf933d7…) under backpropagation.
5. One dry run of 64 updates of each kind, with CPU time and a forecast (the reviewer's estimate for all fifteen, with scoring, is about 1.5 CPU-hours).

## Marks and readings (fixed; the reviewer's proposal, adopted)

"Restored" at a row: the mean of held-out `add89` and `sub89` at 12 letters is at least 85 at that row. "Wrap score" at a row: the mean of the add and sub wrap-pair scores of score 2. Arm K is read against the sweep reference. Per arm, a pattern must hold in at least two of three seeds; the rows are tried in order and the first that holds is the reading.

| # | Pattern | Reading |
|---|---|---|
| 1 | Restored at row 256 | The family's rule was still in the network. A life costs the old family about one showing of each taught pair. |
| 2 | At row 1,024 the wrap score is at least 30 points above the reference's | Partial savings. |
| 3 | At row 1,024 the wrap score is less than 15 points from the reference's | No savings: what the life removed is relearned no faster than a fresh network learns it. |
| 4 | Anything else | Mixed; the curves are reported. |

Reported beside it, as facts: the wrap and no-wrap scores at each row; the fall of the life's families at rows 256 and 1,024; the paired difference between the sweep and backpropagation reminders at each row.


## Common to every run (the audit's recommendations)

- Each runner calls `arch26.install()` first, as `dev` and `life` do; the scorer depends on it.
- Each runner refuses an existing output path (`exp20_reference.refuse_existing`).
- Outputs go to `exp45/e36/`, one receipt and its weight files per run, named `e36_<kind or arm>_<rule>_<seed>` with `_row<updates>` for saved rows.

## What it cannot show

- The optimizer state is fresh in reminder and reference alike.
- The reminder is given alone, not inside a life's stream, so it says nothing about how to keep a family during a life. E38 does.
- Nine saved lives designed for another purpose; three seeds.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P36.1 | Sweep reminders: restored by row 256 in two of three seeds | 30% |
| P36.2 | Sweep reminders: restored by row 1,024 in two of three seeds | 60% |
| P36.3 | Every arm: wrap score at least 30 above its reference's at row 1,024 in two of three seeds (restated on the wrap score after the audit) | 70% |
| P36.4 | The table row differs between the sweep and backpropagation | 15% |
| P36.5 | At row 1,024 the life's own add and sub mod 97 have fallen by at least 30 points in two of three seeds (sweep) | 55% |

## Done

Runner, check receipts, fifteen runs with receipts, and `E36_REPORT.md` with the curves (run × row: the three score groups), the table row per arm, the facts listed, and the outcome of each prediction. No readings beyond the table above.

## What changed from revision 1 (the audit)

Made as the audit wrote them. I confirmed two of its facts myself: line 43 of `exp31_switch_workerB.py` is `import arch31` and the two `forward` signatures differ; the two dry hashes begin 79618ab2 and adf933d7.

| | Change | Required? |
|---|---|---|
| E36-1 | Arm K's step written out against `arch31_workerB.forward` with `switch="K"`; check 3 replaced | yes |
| E36-2 | Check 4 replaced by the update path against E26's dry births (a first-batch loss is the same under both rules and could not fail) | yes |
| E36-3 | Rows 2 and 3 read the wrap score: the reminders start at 13 to 37 on the held-out panel, so "30 above the reference" could be met with nothing relearned | yes |
| — | K lives added to the table; whole-answer figures; output paths, overwrite refusal, `arch26.install()`; forecast with scoring | recommended |
