# E36 (job A): after a life, how many updates bring the unrehearsed birth family back? (15 November 2026, revision 1)

Evidence level E2. The question, the design and the proposed marks are the outside reviewer's (its memo `MEMO_TO_COORDINATOR_LEARNED_STRUCTURE_20261115.md` on `claude/vibrant-cerf-sje726` at a8b5ace2, section 5); this specification and the predictions are mine. **Not released: it waits for the reviewer's audit.** One worker.

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

The reviewer's reading of the saved files is that what is lost is mainly the old modulus: pairs that need the wrap at 89 are right 0 to 2% of the time, pairs that do not 10 to 93%. E36 asks whether the family's rule is still in the network, by measuring how fast a reminder restores it.

## Runs: 9 reminders and 6 references, 1,024 updates each

| Kind | Start | Rule | Seeds |
|---|---|---|---|
| Reminder S | `exp45/e26/e26_life_S_<seed>.pt` | the sweep: `arch26.forward(m, x, True, True)`, answer-masked loss, as `exp26_candidate.step` (lines 84-91) | 3 |
| Reminder B | `exp45/e26/e26_life_B_<seed>.pt` | backpropagation: the same with `local=False` | 3 |
| Reminder K | `exp45/e31/e31_life_K_<seed>.pt` and `e31_life_K_<seed>_Bhh.pt` | the sweep with Kolen–Pollack feedback on the recurrent map, continued from the saved matrix | 3 |
| Reference S, B | `arch26.new_net(seed)`, untrained | sweep; backpropagation | 3 each |

- **Arm K's code.** The K lives were made by worker B's runner, now `exp31_switch_workerB.py` with `arch31_workerB.py` (renamed at the merge; the receipts still say `exp31_switch.py`). It saves the matrix as `T.save({"Bhh": a.fb.B}, ...)` (line 280). It builds the starting matrix from the seed, so loading a saved one is new code: construct the same feedback object, set its matrix from the file, and continue its after-step update `Bhh ← (1 − 0.00025)·Bhh + (W1 − (1 − 0.00025)·W0)`.
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
3. Arm K: the loaded matrix has the saved file's sha256; the identity check of the K lives (the difference `Bhh − W` shrinks by 1 − 0.00025 per step) holds at updates 1, 10 and 100.
4. Reference S and B: the first-batch loss equals that of `exp26_candidate.step` on a fresh net of the seed.
5. One dry run of 64 updates of each kind, with CPU time and a forecast (the reviewer's estimate for all fifteen is 0.6 CPU-hours).

## Marks and readings (fixed; the reviewer's proposal, adopted)

"Restored" at a row: the mean of held-out `add89` and `sub89` at 12 letters is at least 85. Arm K is read against the sweep reference. Per arm, a pattern must hold in at least two of three seeds; the first row of the table that applies is the reading.

| # | Pattern | Reading |
|---|---|---|
| 1 | Restored by row 256 | The family's rule was still in the network. A life costs the old family about one showing of each taught pair. |
| 2 | Not 1, and at row 1,024 at least 30 points above the reference | Partial savings. |
| 3 | Within 15 points of the reference at row 1,024 | No savings: the life removed what the birth built for this family. |
| 4 | Anything else | Mixed; the curves are reported. |

Reported beside it, as facts: the wrap and no-wrap scores at each row; the fall of the life's families at rows 256 and 1,024; the paired difference between the sweep and backpropagation reminders at each row.

## What it cannot show

- The optimizer state is fresh in reminder and reference alike.
- The reminder is given alone, not inside a life's stream, so it says nothing about how to keep a family during a life. E38 does.
- Nine saved lives designed for another purpose; three seeds.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P36.1 | Sweep reminders: restored by row 256 in two of three seeds | 30% |
| P36.2 | Sweep reminders: restored by row 1,024 in two of three seeds | 60% |
| P36.3 | Every arm: at least 30 above its reference at row 1,024 in two of three seeds | 80% |
| P36.4 | The table row differs between the sweep and backpropagation | 15% |
| P36.5 | At row 1,024 the life's own add and sub mod 97 have fallen by at least 30 points in two of three seeds (sweep) | 55% |

## Done

Runner, check receipts, fifteen runs with receipts, and `E36_REPORT.md` with the curves (run × row: the three score groups), the table row per arm, the facts listed, and the outcome of each prediction. No readings beyond the table above.
