# E33: which non-answer targets stop the sweep from extrapolating (15 November 2026, revision 2)

Evidence level E2; stage 1 is a one-seed screen and supports no claim by itself. The design is the outside reviewer's (its memo of 15 November); this specification is mine. Revision 1 (00fad472) was audited by the reviewer, who named three required text changes and three recommendations and stated that with the required ones made it need not see the brief again. Revision 2 makes all six; the last section lists them. One worker. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each.

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

**Class S is in effect the task code alone.** At the end of S-bal the loss per position is 1.40 nats for the code, 0.004 for the operator and 0.000 for `=`. The code is also an unusual target: it is unpredictable but has no history, since it sits at position 0 where the state is zero.

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
sel    = T.zeros_like(other)
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
- Rows are saved every 4,096 updates. Commit (`git add -f`) each arm's final file and its rows at 16,384 and 20,480, so the endpoint can be recomputed from the branch (24 files, about 108 MB). Earlier rows are measured, their hashes recorded, and not committed.

## Checks before any full birth

Tolerance, wherever "agrees" is said: the largest absolute difference is at most 1e-5 of the tensor's largest absolute entry.

1. **`A+DOS` is S-bal.** On the first three batches its loss agrees with `exp28_everybyte.loss_of(logits, y, amask, real, "balanced")` to 1e-6 relative. A dry birth at 2,048 updates has the `tensors_sha256` of `exp45/e28_qual/dry/e28_dev_S-bal_dry2048_73129.json`; if only the last bits differ (a sum divided by a count against a mean), say so and show that the first batch's gradients agree.
   - If the dry hash is equal, the full `A+DOS` birth must then have, at all six saved rows, the `tensors_sha256` in the `curve` of `exp45/e28/e28_dev_S-bal_73129.json`. Compare as each row is written; on a mismatch, stop.
2. **Only the selected positions carry error.** For each arm, on the first batch, with `logits.retain_grad()`: the gradient at the logits is exactly zero at every position outside `amask | sel`, and non-zero somewhere in each selected class. Record the count of selected positions by fine role.
3. **Same data.** Each arm's `data_digest` equals E28's at every saved row (`git show $D:$M/exp45/e28/e28_dev_S-bal_73129.json`, field `curve[*].data_digest`).
4. **Cost.** One dry birth of each arm at 2,048 updates, with CPU time and a forecast.

## Reading the files

A new script `e33_read.py` imports `panels`, `measure`, `load_net` and `row_files` from `e32_read.py` unchanged (`git show $D:$M/e32_read.py | sed -n '54,116p'`).

- **First**, on files already on the default branch, it reproduces 91.33 and 6.0 (12 and 40 letters) for `exp45/e28/e28_dev_S-bal_73129.pt` and 98.3 and 96.0 for `exp45/e26/e26_dev_S_73129.pt`.
- Then, for every saved row of every arm: accuracy at 12, 20 and 40 letters, keepers and keeps-and-writes, as `measure` returns them.

## Endpoint and marks (fixed)

Every figure is the mean of the last three saved rows (16,384, 20,480, 24,576); the final row is given beside it.

- **Learned in range:** accuracy at 12 letters at least 70. (S-bal's own last-three-row mean is 85 to 90 and its single rows swing by up to 12 points.)
- **Extrapolates:** accuracy at 40 letters at least 60.
- **Fails:** accuracy at 40 letters at most 20.
- **Intermediate:** between 20 and 60.
- An arm that did not learn in range is not read for extrapolation.

Accuracy at 20 letters is reported beside each and not used in the marks.

## Outcomes and readings (fixed; written before the predictions)

Checked in this order; the first that applies is the reading.

| # | Pattern over the eight arms | Reading |
|---|---|---|
| 0a | `A` does not extrapolate | The answer-only loss at half weight does not behave as E26's answer-only birth. Stop; no reading. |
| 0b | `A+DOS` does not fail | If its six row hashes matched S-bal's, the birth is S-bal and the "fails" mark is mis-set for a last-three-row mean (only S-bal's final row, 6.0, was known when it was set). Stop, report the rows; I reset the mark before anything is read. If the hashes did not match, the runner is at fault. Stop; no reading. |
| 1 | There is a non-empty set C of classes such that the failing arms are exactly those containing at least one class of C, and every other arm extrapolates | Targets of any class in C, at S-bal's weight, stop extrapolation on this seed; classes outside C are harmless at their weight. **C of one class:** that class is decisive. If it is D, the sentence ends "pending the control", because this is also what a threshold on error between 12.2 and 20.6 nats per item would give. **C of two:** either of the two suffices. **C of all three:** any class is enough. |
| 2 | For one class c, the failing arms are exactly those containing c or both of the other two, and every other arm extrapolates | c suffices alone; the other two suffice only together. |
| 3 | All three single-class arms extrapolate and `A+DOS` fails | Only combinations break it. The pairs' marks are reported. If all three pairs fail, the sentence adds that this is what a threshold on non-answer weight between 0.231 and 0.269 would give. |
| 4 | Anything else, including any intermediate arm or any arm that did not learn in range | **Mixed.** The eight-arm table is reported with each class's mean over its four arms beside the mean of the four without it. No class is named. |

Whenever S is outside C, the reading says only that one history-free unpredictable target (the task code) is harmless. It says nothing about predictable targets, which carry no error once learned in any arm.

## What stage 1 cannot show

- **Kind against amount, in two forms.** Dropping terms lowers both the weight on non-answer targets and the error they carry (the reviewer's figures, from S-bal's end-of-birth losses):

  | Arm | Non-answer weight | Error per item (nats) |
  |---|---|---|
  | A+S | 0.115 | 1.4 |
  | A+O | 0.154 | 10.8 |
  | A+D | 0.231 | 20.6 |
  | A+OS | 0.269 | 12.2 |
  | A+DS | 0.346 | 22.0 |
  | A+DO | 0.385 | 31.4 |
  | A+DOS | 0.500 | 32.8 |

  - D decisive is exactly what a threshold on error between 12.2 and 20.6 nats would give.
  - All pairs failing with every single class passing is exactly what a threshold on weight between 0.231 and 0.269 would give.
  - If O or S is decisive, a class carrying less error fails while D passes, which already rules amount out.
- **One seed.** Births of one seed swing by tens of points between rows; the marks are wide for that reason, and nothing is claimed until stage 2.
- **Why.** A decisive class says where the interference enters, not how.

## Stage 2 (released by me after stage 1 is read; seeds 93683 and 116089)

- **Outcome 1 with one class c:** `A+c` and the arm with the other two classes, on both seeds (four births).
- **The control, only if c is D:** `A+D-light` on all three seeds, the D terms multiplied by 0.5. That leaves about 10.3 nats per item, about what O carries (10.8). **Its reading is fixed now:** if `A+D-light` fails while `A+O` extrapolates, it is the kind of target and not the amount of error; if `A+D-light` extrapolates, it is the amount. An intermediate result settles neither.
- **Outcome 1 with two or three classes:** the single-class arms on both seeds (six births).
- **Outcome 2, 3 or 4:** I name the arms from the table before any run.
- The "fails" mark of 20 is close for seed 93683, whose S-bal final row is 16.3 at 40 letters. Stage 2 states its marks per seed before it runs.

## What follows from it

- A decisive class: the next candidate is a local loss weighting that needs no knowledge of where the answer is, for example weighting each position by how predictable it has proved to be.
- No single class: the remedy is sought in the feedback pathway beyond the recurrent map, with exact feedback at the readout-side maps as a bound and Kolen–Pollack feedback on all maps as the candidate. That needs new code and two independent implementations, and its own brief.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P33.1 | Neither 0a nor 0b occurs: `A` extrapolates and `A+DOS` fails | 80% |
| P33.2 | `A+OS` (no distractor targets) extrapolates | 40% |
| P33.3 | `A+D` (distractor targets only) fails | 55% |
| P33.4 | Outcome 1 with C of exactly one class | 20% |
| P33.5 | Outcome 1 with C of all three classes (any class is enough) | 20% |
| P33.6 | Outcome 1 with C = {D, O}: either unpredictable class with history suffices, the task code is harmless | 20% |
| P33.7 | All eight arms learn in range | 80% |

## Done

Runner, reader, check receipts, eight complete births, and `E33_REPORT.md` with the eight-arm table (arm; accuracy at 12, 20, 40 as last-three mean and final row; keepers; loss by fine role at the end), which outcome row applies, and the outcome of each prediction. No readings beyond the table above.

## What changed from revision 1 (the reviewer's audit; its script is `e33_brief_checks.py` on `claude/vibrant-cerf-sje726` at 7df89744)

I made these as the audit wrote them and reran none of its figures.

| | Change | Required? |
|---|---|---|
| 1 | Outcome rows generalised: failing arms are exactly those containing a class of C, for C of one, two or three; a row for "c, or the other two together"; qualifiers on D decisive and on all pairs failing | yes |
| 2 | Outcome 0 split and reworded; six-row hash check for `A+DOS` against S-bal | yes |
| 3 | Both amount confounds stated with the weight and error table; the control fixed as D at half weight, with its reading, and limited to the case c = D | yes |
| 4 | "Learned in range" at 70 | recommended |
| 5 | Rows 16,384 and 20,480 committed for every arm | recommended |
| 6 | Class S described as the task code in effect, and what "S is harmless" can show | recommended |

P33.5 and P33.6 are restated in the new outcome rows, and P33.6 is new: it is the pattern the reviewer's own hypothesis predicts.

---

# Addendum 1 (15 November 2026): stage 1 read; a reduced stage 2 is released

## Stage 1 as I read it

The worker's branch `claude/vibrant-curie-ivmrt1` (9349d0b9) is merged. I recomputed the eight endpoints from `exp45/e33/E33_READ.json` and they equal the report's; the outside reviewer states that its own reader reproduces them from the weight files. I did not re-measure the weight files.

Seed 73129, mean of the last three rows, percent at 40 letters:

| Distractor targets | Arms | At 40 letters |
|---|---|---|
| absent | `A`, `A+O`, `A+S`, `A+OS` | 94.8, 90.2, 88.1, 94.2 |
| present | `A+D`, `A+DS`, `A+DO`, `A+DOS` | 23.8, 38.1, 2.3, 4.1 |

- **Outcome row 4 (mixed) applies**, as the worker reported: `A+D` and `A+DS` are intermediate, so no class is named by the registered rule. That stands.
- What the table shows beside the rule: no arm with distractor targets extrapolates, and every arm without them does (means 17.1 against 91.8).
- A threshold on non-answer weight is excluded on this seed: `A+OS` carries more of it than `A+D` (0.269 against 0.231) and extrapolates. A threshold on the error the extra targets carry is not excluded.
- P33.1, P33.2 and P33.7 hold; P33.3 to P33.6 miss.

## Stage 2, reduced: four births

Arms `A+D` and `A+OS` on seeds 93683 and 116089. These two arms carry the contrast: distractor targets alone, and both other classes without them. About 3.2 CPU-hours (stage 1 took 2,852 to 2,882 CPU-seconds per birth). The control `A+D-light` is not part of this stage.

**Code.** `exp33_targets.py` unchanged; it already takes the seed. Its reference lookups need `exp45/e28/e28_dev_S-bal_<seed>.json`, which exist for both seeds (`git show $D:$M/exp45/e28/ | grep S-bal`). `e33_read.py` has `SEED = 73129` fixed at line 24: give it an optional seed argument whose default is 73129, and show that `python3 e33_read.py tables` still writes a byte-identical `E33_READ_TABLES.md` for seed 73129.

**Checks before the births.**
1. The reader, with the new seed argument, reproduces on files already on the default branch the 12- and 40-letter figures of `E30_UNSEEN.json` for `e28_dev_S-all_93683.pt` (79.67 and 2.33), and reports its figures for `e28_dev_S-bal_93683.pt` and `e28_dev_S-bal_116089.pt` (no recorded figure of mine to match; these are the two seeds' `A+DOS` final rows and are reported as references).
2. Each birth's `data_digest` equals S-bal's of the same seed at every saved row (the runner asserts it).

**Wall time.** Stage 1 lost two births to a 2-hour limit on the launching shell. Launch each birth so that no wrapper limit under 90 minutes applies, at most three at a time.

## Marks and the replication rule (fixed before the run)

Marks as in stage 1, on the mean of the last three rows: learned in range at least 70 at 12 letters; extrapolates at least 60 at 40 letters; fails at most 20; intermediate between. The same marks for both seeds. The note in the brief that 20 is close for seed 93683 concerns `A+DOS`, which is not run here.

| # | On a new seed | Reading for that seed |
|---|---|---|
| R1 | `A+OS` extrapolates, and `A+D` does not extrapolate and is at least 40 points below `A+OS` at 40 letters | The contrast replicates on this seed |
| R2 | `A+OS` does not extrapolate | The contrast does not replicate: extrapolation without distractor targets failed |
| R3 | `A+D` extrapolates, or is within 40 points of `A+OS` | The contrast does not replicate: distractor targets did not stop it |
| R4 | Either arm did not learn in range | Not read |

**The sentence, if R1 holds on both new seeds:** "On three seeds, a sweep birth with answer and distractor targets does not extrapolate to 40 letters, and one with answer, operand and task-code targets does." Nothing about why, and nothing about amount against kind. If R1 holds on one seed only, the result is reported as two of three. Otherwise the stage 1 contrast is reported as not replicated.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P33.8 | `A+OS` extrapolates on both new seeds | 75% |
| P33.9 | `A+D` does not extrapolate on either new seed | 85% |
| P33.10 | `A+D` fails (at most 20) on both new seeds | 35% |
| P33.11 | R1 holds on both new seeds | 65% |

## Done

Four complete births with receipts, finals and rows 16,384 and 20,480 committed; the reader's rows for both seeds; an addendum to `E33_REPORT.md` with the per-seed table (both arms; accuracy at 12, 20, 40 as last-three mean and final row; keepers), which of R1 to R4 applies on each seed, and the outcome of P33.8 to P33.11. No readings beyond the table above.
