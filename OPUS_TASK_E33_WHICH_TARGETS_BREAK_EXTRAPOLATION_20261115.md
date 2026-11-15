# E33: which non-answer targets stop the sweep from extrapolating (15 November 2026)

Evidence level E2; stage 1 is a one-seed screen and supports no claim by itself. The design is the outside reviewer's (its memo of 15 November); this specification is mine. **Not yet audited: no worker starts until the reviewer has read it or Mr. Xu waives that.** One worker. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each.

Shell names used below: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106`.

## Why

A GRU born under the sweep extrapolates past its trained range of 0 to 12 letters when only answer bytes are targets, and not when the other bytes are targets too. Seed 73129, mod-89 addition, final files (`git show $D:$M/exp45/e30/E30_UNSEEN.json`):

| Birth | Targets | At 12 letters | At 40 letters |
|---|---|---|---|
| E26 arm S | answer bytes only | 98.3 | 96.0 |
| E28 S-bal | answer bytes at half the loss, every other byte at the other half | 91.3 | 6.0 |
| E28 S-all | every byte equally | 75.3 | 2.0 |

E32 changed the backward pass and the gates and did not move this: the best arm reached 12% at 40 letters (review 9.128). E33 changes only which non-answer bytes are targets, to find which class does it.

## The three classes

`exp28_everybyte.roles_of` labels every target position (`git show $D:$M/exp28_everybyte.py | sed -n '97,110p'`). An item is `code a1 a2 op d_1..d_k b1 b2 = c1 c2 EOS`.

| Class | Fine roles | Bytes per item | Predictable? |
|---|---|---|---|
| **D** distractor | `distractor` | k (0 to 12) | no: random letters |
| **O** operand | `operand` | 4 | no: random digits |
| **S** structural | `code`, `operator`, `equals` | 3 | the operator and `=` are; the task code is one of four at random |

The fine roles are recorded, so S can be split later if it matters.

## Arms: all eight subsets, seed 73129

`A` (no non-answer target), `A+D`, `A+O`, `A+S`, `A+DO`, `A+DS`, `A+OS`, `A+DOS`. Every arm is a birth under the sweep (`local = True`), 24,576 updates, same data, optimizer and network as E28. Eight births, about 6.5 CPU-hours (S-bal took 2,867 CPU-seconds).

The reviewer proposed the six middle arms. I add the two ends so that both references are made by this runner: `A+DOS` is S-bal's loss, and `A` is the answer-only loss at half weight.

## The loss, as code

```
x, y, amask, real, fine = exp28_everybyte.masks(items)          # imported, unchanged
logits = arch26.forward(m, x, True, True)[0]
other  = real & ~amask                                            # every non-answer target in the batch
F      = exp28_everybyte.FINE.index
cls    = {"D": fine == F("distractor"),
          "O": fine == F("operand"),
          "S": (fine == F("code")) | (fine == F("operator")) | (fine == F("equals"))}
sel    = torch.zeros_like(other)
for c in arm_classes: sel |= cls[c]                               # arm_classes: "", "D", "O", "S", "DO", "DS", "OS", "DOS"
assert not (sel & ~other).any() and T.equal(cls["D"] | cls["O"] | cls["S"], other)
loss   = 0.5 * cross_entropy(logits[amask], y[amask])
if arm_classes:
    loss = loss + 0.5 * cross_entropy(logits[sel], y[sel], reduction="sum") / other.sum()
```

**The weighting is a choice, and it is this one:** each selected position keeps exactly the weight it has in S-bal, `0.5 / other.sum()`; unselected positions are dropped and the rest are **not** scaled up. So `A+DOS` is S-bal, and every other arm is S-bal with some terms removed. The consequence is stated under "What stage 1 cannot show".

## Code

One new runner `exp33_targets.py`: E28's birth loop (`exp28_everybyte.dev`) with the loss above. It imports `masks`, `FINE`, `roles_of` and the diagnostics from `exp28_everybyte`; it does not call or patch `exp28_everybyte.dev`, `.step` or `.loss_of`. Edit nothing existing.

- Receipts say `experiment="E33"`, `arm`, `arm_classes`, and go to `exp45/e33/e33_dev_<arm>_<seed>.json`.
- The loss by fine role is recorded per 512 updates for **all** roles, selected or not, as E28 does.
- Rows are saved every 4,096 updates. Commit (`git add -f`) the eight final files only; row files are measured, their hashes recorded, and not committed.

## Checks before any full birth

Tolerance, wherever "agrees" is said: the largest absolute difference is at most 1e-5 of the tensor's largest absolute entry.

1. **`A+DOS` is S-bal.** On the first three batches its loss agrees with `exp28_everybyte.loss_of(logits, y, amask, real, "balanced")` to 1e-6 relative. A dry birth at 2,048 updates has the `tensors_sha256` of `exp45/e28_qual/dry/e28_dev_S-bal_dry2048_73129.json`; if only the last bits differ (a sum divided by a count against a mean), say so and show that the first batch's gradients agree.
2. **Only the selected positions carry error.** For each arm, on the first batch, with `logits.retain_grad()`: the gradient at the logits is exactly zero at every position outside `amask | sel`, and non-zero somewhere in each selected class. Record the count of selected positions by fine role.
3. **Same data.** Each arm's `data_digest` equals E28's at every saved row (`git show $D:$M/exp45/e28/e28_dev_S-bal_73129.json`, field `curve[*].data_digest`).
4. **Cost.** One dry birth of each arm at 2,048 updates, with CPU time and a forecast.

## Reading the files

A new script `e33_read.py` imports `panels`, `measure`, `load_net` and `row_files` from `e32_read.py` unchanged (`git show $D:$M/e32_read.py | sed -n '54,116p'`).

- **First**, on files already on the default branch, it reproduces 91.33 and 6.0 (12 and 40 letters) for `exp45/e28/e28_dev_S-bal_73129.pt` and 98.3 and 96.0 for `exp45/e26/e26_dev_S_73129.pt`.
- Then, for every saved row of every arm: accuracy at 12, 20 and 40 letters, keepers and keeps-and-writes, as `measure` returns them.

## Endpoint and marks (fixed)

Every figure is the mean of the last three saved rows (16,384, 20,480, 24,576); the final row is given beside it.

- **Learned in range:** accuracy at 12 letters at least 80.
- **Extrapolates:** accuracy at 40 letters at least 60.
- **Fails:** accuracy at 40 letters at most 20.
- **Intermediate:** between 20 and 60.
- An arm that did not learn in range is not read for extrapolation.

Accuracy at 20 letters is reported beside each and not used in the marks.

## Outcomes and readings (fixed; written before the predictions)

Checked in this order; the first that applies is the reading.

| # | Pattern over the eight arms | Reading |
|---|---|---|
| 0 | `A` does not extrapolate, or `A+DOS` does not fail | The runner does not reproduce the references. Stop; no reading. |
| 1 | For one class c: all four arms containing c fail and all four without c extrapolate | **c is decisive** on this seed: targets of that class, at S-bal's weight, are what stops extrapolation. |
| 2 | `A` extrapolates and the other seven fail | **Any class is enough.** No single class accounts for it. |
| 3 | All three single-class arms extrapolate and `A+DOS` fails | **Only combinations break it.** The pairs' marks are reported; no single class accounts for it. |
| 4 | Anything else, including any intermediate arm or any arm that did not learn in range | **Mixed.** The eight-arm table is reported with each class's mean over its four arms beside the mean of the four without it. No single class is named. |

## What stage 1 cannot show

- **Kind against amount.** Dropping terms lowers the total weight on non-answer targets, and the classes differ in size: per item D has 6 bytes on average, O has 4 and S has 3. If D alone is decisive, that may be because it is the largest class. Stage 2 carries the control.
- **One seed.** Births of one seed swing by tens of points between rows; the marks are wide for that reason, and nothing is claimed until stage 2.
- **Why.** A decisive class says where the interference enters, not how.

## Stage 2 (released by me after stage 1 is read; seeds 93683 and 116089)

- **Outcome 1:** `A+c` and the arm with the other two classes, on both seeds (four births), plus `A+c-light` on all three seeds: class c's terms multiplied by one constant so that their summed weight over the first 128 batches equals class S's (or class O's, if c is S). Three more births.
- **Outcome 2:** `A+D`, `A+O`, `A+S` on both seeds (six births).
- **Outcome 3 or 4:** I name the arms from the table before any run.

## What follows from it

- A decisive class: the next candidate is a local loss weighting that needs no knowledge of where the answer is, for example weighting each position by how predictable it has proved to be.
- No single class: the remedy is sought in the feedback pathway beyond the recurrent map, with exact feedback at the readout-side maps as a bound and Kolen–Pollack feedback on all maps as the candidate. That needs new code and two independent implementations, and its own brief.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P33.1 | Outcome 0 does not occur: `A` extrapolates and `A+DOS` fails | 85% |
| P33.2 | `A+OS` (no distractor targets) extrapolates | 40% |
| P33.3 | `A+D` (distractor targets only) fails | 55% |
| P33.4 | Outcome 1: exactly one class is decisive | 20% |
| P33.5 | Outcome 2: any class is enough | 20% |
| P33.6 | All eight arms learn in range | 70% |

## Done

Runner, reader, check receipts, eight complete births, and `E33_REPORT.md` with the eight-arm table (arm; accuracy at 12, 20, 40 as last-three mean and final row; keepers; loss by fine role at the end), which outcome row applies, and the outcome of each prediction. No readings beyond the table above.
