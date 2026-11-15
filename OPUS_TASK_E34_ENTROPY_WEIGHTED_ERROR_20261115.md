# E34: sweep births with each byte's error scaled by the network's own confidence (15 November 2026, revision 2)

Evidence level E2; stage 1 is a one-seed screen and supports no claim by itself. The remedy and the outline of the design are the outside reviewer's (its memo of 15 November); this specification, the normalisation, the temperatures, the outcome table and the predictions are mine. Revision 1 (9291d23b) was audited by the reviewer, who named one required change (the stage 2 rule) and smaller recommendations, and stated that with the required one made it need not see the brief again. Revision 2 makes them; the last section lists them. The user has ruled the rule inside the line. **Released.** One worker. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each.

Shell names used below: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106`.

## Why

A GRU born under the sweep with every byte a target does not extrapolate past its trained range of 0 to 12 letters; with only answer bytes as targets it does. E33 stage 1 (seed 73129) put the difference at the distractor targets: every arm with them reached 2 to 38% at 40 letters and every arm without them 88 to 95% (addendum 1 of the E33 brief). But E33's arms choose targets with the answer mask and the role labels, which a learner does not have. E34 tests a rule that uses neither: each position's output error is multiplied by a gain computed from the network's own prediction at that position, small where the prediction is spread out.

References, seed 73129, same panels (`git show $D:$M/exp45/e30/E30_UNSEEN.json`, `$M/exp45/e33/E33_REPORT.md`):

| Birth | Targets | At 12 letters | At 40 letters |
|---|---|---|---|
| E28 S-all (the baseline here) | every byte equally | 59.6 (final row 75.3) | 1.0 (final row 2.0) |
| E28 B-all (backpropagation; the like-for-like reference, no mask) | every byte equally | 92.8 (final row 93.7) | 82.0 (final row 84.7) |
| E33 `A` | answer bytes only, with the mask | 99.0 | 94.8 |
| E33 `A+OS` | answer, operand and task code, with the mask | 97.7 | 94.2 |

Figures are means of the last three rows; S-all's and B-all's are the `floor` and `ceiling` rows of seed 73129 in `$M/exp45/e32/E32_READ.json`. S-all's rows at 12 letters are 66.3, 37.0 and 75.3, so **the baseline itself does not meet the "learned in range" mark of 70.**

## The rule, as code

```
x, y, amask, real, fine = exp28_everybyte.masks(items)        # imported, unchanged; amask and fine are used by the diagnostics only
logits = arch26.forward(m, x, True, True)[0]                  # the sweep, as in E28's S arms
ce = cross_entropy(logits[real], y[real], reduction="none")   # one value per real position
with T.no_grad():
    lp = log_softmax(logits[real], -1)
    Hh = -(lp.exp() * lp).sum(-1)                             # predictive entropy in nats, per real position
    w  = T.exp(-Hh / tau)                                     # tau = float("inf") gives w == 1.0 exactly
    assert T.isfinite(w).all() and w.sum() > 0
loss = (w * ce).sum() / w.sum()
```

- **The gain carries no gradient.** `w` is computed under `no_grad`; the error at a position's logits is `(softmax - onehot) * w / w.sum()`. The network is not trained to lower its entropy.
- **No label and no mask enter the loss.** `y` enters only through `ce`, as in S-all. `amask` and `fine` are not used by the update.
- **Division by `w.sum()` is my addition, and it is needed.** Without it the loss is `(w * ce).sum() / real.sum()`, and on the first batch of an untrained network every weight is `exp(-5.54 / tau)`: 1.6e-5 at tau 0.5. The median parameter gradient is then 1.7e-9, below AdamW's `eps` of 1e-8, so the first updates would be damped by the optimizer and not by the rule. With the division the median gradient is 1.07e-4 at every temperature, the same as S-all's (`PROJECT_DOCUMENTS/COORD_CHECKS_20261115/first_batch_scale.py`, output in `OUTPUT_MAC.txt` beside it). The division is by one number per batch, common to all positions; it changes no position's weight relative to another's.
- With `tau = float("inf")` the loss is bit-equal to `cross_entropy(logits[real], y[real])` on the first batch (same script), so the unit-weight arm should be S-all.

## What the gain does to a network trained without it (my measurement; a projection, not a result)

`PROJECT_DOCUMENTS/COORD_CHECKS_20261115/entropy_by_class.py 73129 S-all`, first eight training batches, on the six saved rows of `e28_dev_S-all_73129`. Run on the Mac (Python 3.11.5, torch 2.9.1), so last digits may differ on the cloud host. Seed 93683 gives the same entropies to within 0.05 nats.

Entropy by class is stable from row 4,096 to the end: distractor 3.45 to 3.48 nats, operand 2.52 to 2.56, task code 1.40 to 1.41, answer digit 0.98 falling to 0.42, operator, `=` and EOS below 0.09.

| | tau 0.5 | tau 1 | tau 2 | unit weights |
|---|---|---|---|---|
| Weight of a distractor position over an answer position (digits and EOS) | 0.0014 to 0.0019 | 0.039 to 0.049 | 0.20 to 0.23 | 1 |
| Distractors' share of the weighted error, row 24,576 | 6% | 30% | 47% | 61% |
| Operands' share | 26% | 42% | 40% | 34% |
| Task code's share | 28% | 17% | 9% | 4% |
| Answer digits' share | 39% | 11% | 4% | 1% |

For comparison, in E33's `A+D` a distractor position has 0.245 of an answer position's weight (192 answer positions against 782 others on the first batch), and that arm reached 23.8% at 40 letters. **So tau 2 puts about `A+D`'s weight on distractors, tau 1 about a fifth of it, and tau 0.5 about a hundredth.** These ratios come from a network trained with unit weights; a network trained under the gain may settle elsewhere, which is why the runner records them (below).

**The projections begin at row 4,096, when the answer is partly learned. Earlier the gain is weaker.** The reviewer's estimate from the baseline's loss record, if entropy tracks loss: over the first 512 updates the ratio is about 0.08, 0.29 and 0.54 at tau 0.5, 1 and 2. So at tau 1 the rule starts above `A+D`'s dose and falls below it within about 1,000 updates. I did not rerun this estimate. The division by `w.sum()` also does not hold the gradient scale constant later: as confident positions with near-zero error come to dominate the sum, the reviewer estimates the scale at tau 0.5 falls to roughly a twentieth of the baseline's, still several hundred times the optimizer's floor.

## Arms, seed 73129

| Arm | tau | Run |
|---|---|---|
| `U` | infinity | dry birth only (2,048 updates), as the check that the runner is S-all |
| `T050` | 0.5 | full birth |
| `T100` | 1.0 | full birth |
| `T200` | 2.0 | full birth |

Every arm is a birth under the sweep (`local = True`), 24,576 updates, same data, optimizer and network as E28. Three full births, about 2.5 CPU-hours (S-all took about 2,870 CPU-seconds; the entropy adds one `log_softmax` per update).

## Code

One new runner `exp34_entropy.py`: E33's birth loop (`exp33_targets.dev`, itself E28's) with the loss above. Import `masks`, `FINE` and the role diagnostics from `exp28_everybyte`; do not call or patch any existing `dev`, `step` or loss function. Edit nothing existing.

- Receipts say `experiment="E34"`, `arm`, `tau`, and go to `exp45/e34/e34_dev_<arm>_<seed>.json`, in E33's format. The reference file for digests and hashes is `exp45/e28/e28_dev_S-all_<seed>.json` (dry: `exp45/e28_qual/dry/e28_dev_S-all_dry2048_73129.json`).
- **Weight diagnostics, per 512 updates and per fine role**, on the same detached logits, beside E28's loss by role: the sum of `Hh`, the sum of `w`, the sum of `w * ce`, and the count. From these the report gives, per block, each role's mean entropy, mean weight, and share of the weighted error.
- Rows are saved every 4,096 updates. Commit (`git add -f`) each full arm's final file and its rows at 16,384 and 20,480 (9 files). Earlier rows are measured, their hashes recorded, and not committed.
- Launch each birth so that no wrapper limit under 90 minutes applies.

## Checks before any full birth

1. **`U` is S-all.** On the first three batches its loss equals `exp28_everybyte.loss_of(logits, y, amask, real, "all")` to 1e-6 relative, and the first batch's gradients agree to 1e-5 of each tensor's largest entry. The dry birth at 2,048 updates has the `tensors_sha256` of `exp45/e28_qual/dry/e28_dev_S-all_dry2048_73129.json`. If the hash differs, report the first-batch comparison and stop; I decide before any full birth.
2. **The gain is what the code says.** For each of the three temperatures, on the first batch, with `logits.retain_grad()`: the gradient at the logits equals `(softmax - onehot) * w / w.sum()` at real positions to 1e-5 of its largest entry, with `w` recomputed independently from the logits, and is exactly zero at every position that is not real.
3. **No gradient through the gain.** On the first batch at tau 1, the parameter gradients equal those of the same loss with `w` replaced by a constant tensor holding the same values.
4. **Same data.** Each arm's `data_digest` equals S-all's at every saved row.
5. **Dry births.** One dry birth of each of `T050`, `T100`, `T200` at 2,048 updates, with CPU time, a forecast, all losses finite, and the weight diagnostics of the four blocks. Report them before the full births start; do not wait for me unless a loss is not finite.

## Reading the files

A new script `e34_read.py` imports `panels`, `measure`, `load_net` and `row_files` from `e32_read.py` unchanged.

- **First**, on files already on the default branch, it reproduces 75.33 and 2.0 (12 and 40 letters) for `exp45/e28/e28_dev_S-all_73129.pt` and 98.33 and 96.0 for `exp45/e26/e26_dev_S_73129.pt`.
- Then, for every saved row of every full arm: accuracy at 12, 20 and 40 letters, keepers and keeps-and-writes, as `measure` returns them.

## Endpoint and marks (fixed; E33's)

Every figure is the mean of the last three saved rows (16,384, 20,480, 24,576); the final row is given beside it.

- **Learned in range:** at least 70 at 12 letters.
- **Extrapolates:** at least 60 at 40 letters.
- **Fails:** at most 20 at 40 letters.
- **Intermediate:** between 20 and 60.
- An arm that did not learn in range is not read for extrapolation.

## Outcomes and readings (fixed; written before the predictions)

Checked in this order; the first that applies is the reading.

| # | Pattern over `T050`, `T100`, `T200` | Reading |
|---|---|---|
| 0 | Check 1 fails | The runner is not S-all. Stop; no reading. |
| 1 | At least one arm learned in range and extrapolates | The gain rescues extrapolation on this seed at that temperature. A candidate on one seed; nothing is claimed. Stage 2 runs. |
| 2 | No arm extrapolates; at least one learned in range and is intermediate | Partial on this seed; not a pass. The table is reported. No stage 2. |
| 3 | No arm extrapolates or is intermediate; at least one learned in range (and fails) | At these temperatures the gain does not rescue extrapolation on this seed. |
| 4 | No arm learned in range | The gain did not give in-range learning on this seed. The report states, from the weight diagnostics, the answer digits' mean weight and share over the birth, and nothing more. |

Reported beside the row, as facts and not readings:

- **Dose order.** Whether accuracy at 40 letters does not rise with tau across the three arms.
- **Realised weights.** For each arm, the measured ratio of a distractor position's weight to an answer position's, and each role's share of the weighted error, for each of the first eight blocks (updates 1 to 4,096) and at the end of the birth, beside the projections above.
- **Range.** Each arm's accuracy at 12 letters beside the baseline's 59.6 and B-all's 92.8.
- **Distance to the reference.** For each arm that extrapolates, whether its accuracy at 40 letters is within 15 points of B-all's 82.0. The "extrapolates" mark of 60 is 22 points below that reference.

## What stage 1 cannot show

- **Amount against kind.** The gain lowers the weight on distractors, on operands and on the task code together, each by its own entropy. A pass does not say which reduction mattered, nor whether a fixed weight of the same size on distractor targets would do as well.
- **Why.** E33 says where the interference enters; neither experiment says how.
- **A like-for-like bound.** E33's `A` and `A+OS` use the answer mask and give the answer half the loss. They are references, not a ceiling this rule must reach.
- **One seed.** Rows of one birth swing by tens of points (S-all's 12-letter rows: 66, 37, 75).
- **Later use.** Whether a network born this way then lives and composes under the sweep is not tested here.

## Stage 2 (released by me after stage 1 is read)

Only under outcome 1: **every** arm that learned in range and extrapolates runs on seeds 93683 and 116089, about 1.6 CPU-hours per arm. No arm is chosen among those that pass, because two passing arms will usually differ by less than the 15 points that single births are read to. Same marks.

The sentence is stated per temperature. If an arm learns in range and extrapolates on all three seeds: "At temperature tau, a sweep birth with every byte a target and each byte's error scaled by the network's own confidence learns in range and extrapolates to 40 letters on three seeds, with no answer mask. One of the three seeds, 73129, was the screen that selected this temperature." A confirmation on new seeds, and the combination with Kolen–Pollack feedback during a life, need their own brief.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P34.1 | Check 1 passes: `U`'s dry hash equals S-all's | 90% |
| P34.2 | `T200` does not extrapolate | 75% |
| P34.3 | `T100` learns in range and extrapolates | 35% |
| P34.4 | `T050` learns in range and extrapolates | 45% |
| P34.5 | Outcome 1 | 55% |
| P34.6 | Accuracy at 40 letters does not rise with tau across the three arms | 60% |
| P34.7 | All three arms learn in range | 50% |
| P34.8 | At the end of `T100`, a distractor position's measured weight is between 0.02 and 0.10 of an answer position's | 65% |

## Done

Runner, reader, check receipts, four dry births, three complete births, and `E34_REPORT.md` with: the three-arm table (accuracy at 12, 20, 40 as last-three mean and final row; keepers; loss, mean entropy, mean weight and share of weighted error by fine role at the end), the weight ratio distractor to answer for each of the first eight blocks and per saved row, which outcome row applies, the four facts listed under the outcome table, and the outcome of each prediction. No readings beyond the table above.

## Admissibility (ruled by the user, 15 November)

Both parts are inside the line: each output position scaling its own error by a gain computed from its own predicted distribution, and one common gain shared by the batch (the division by `w.sum()`). The user's words: "Both 1 and 2 are fine", and, as the standing rule, "as long as we do not use full scale backpropagation, anything is fine." An arm that passes stage 2 is therefore a candidate rule. The sweep itself is unchanged here.

## What changed from revision 1 (the reviewer's audit; its script is `e33_brief_checks.py e34` on `claude/vibrant-cerf-sje726` at 12eb5a9f)

I made these as the audit wrote them. I checked one figure myself: B-all's last-three-row mean of 92.8 and 82.0 is the `ceiling` row of seed 73129 in `E32_READ.json`.

| | Change | Required? |
|---|---|---|
| 1 | Stage 2 runs every passing arm on the two new seeds; the sentence is per temperature and says that seed 73129 selected it | yes |
| 2 | `assert w.sum() > 0` | recommended |
| 3 | The early-training qualification on the projected ratios; the report gives the realised ratio for the first eight blocks | recommended |
| 4 | The like-for-like reference (B-all) cited as a last-three-row mean, 92.8 and 82.0 | recommended |
| 5 | The report says whether a passing arm is within 15 points of 82.0 | recommended |

The reviewer agreed with the loss code, the division by `w.sum()`, the entropy and weight figures, the three temperatures, the outcome table and the five checks.

---

# Addendum 1 (15 November 2026): stage 1 read; stage 2 released with two additions

## Stage 1 as I read it

The worker's branch `claude/vibrant-curie-ivmrt1` (ba6f5d52) is merged. I recomputed the three endpoints from `exp45/e34/E34_READ.json` and they equal the report's; the reviewer's own reader reproduces them from the committed weight files.

| Arm | tau | At 12 letters | At 40 letters | The three rows at 40 letters | Mark |
|---|---|---|---|---|---|
| `T050` | 0.5 | 83.7 | 79.8 | 78.0, 62.0, 99.3 | learned in range; extrapolates |
| `T100` | 1 | 95.3 | 57.6 | 72.7, 53.0, 47.0 | learned in range; intermediate |
| `T200` | 2 | 86.1 | 2.0 | 2.3, 1.3, 2.3 | learned in range; fails |

- **Outcome row 1.** `T050` is a candidate on one seed; nothing is claimed. It is 2.2 points below backpropagation's 82.0.
- P34.1, P34.2 and P34.4 to P34.8 hold; P34.3 misses.
- **The endpoint hides a swing**, as the table's fourth column shows.
- **The price of tau 0.5 is slow acquisition** (the reviewer's observation, which the rows confirm). Accuracy at 12 letters by row:

  | Arm | 4,096 | 8,192 | 12,288 | 16,384 | 20,480 | 24,576 |
  |---|---|---|---|---|---|---|
  | `T050` | 2.3 | 6.3 | 17.0 | 78.3 | 73.3 | 99.3 |
  | `T100` | 17.7 | 85.0 | 91.7 | 95.0 | 91.7 | 99.3 |
  | `T200` | 30.7 | 73.3 | 83.3 | 84.3 | 79.0 | 95.0 |

  An answer the network cannot yet predict has high entropy and so almost no weight: the gain holds back what is new. **The gain therefore does not go into a life until this cost has been measured in stage 2**, and any brief for a life without the answer mask must carry an acquisition score row by row for each new family.

## Stage 2

**As the brief says:** `T050` on seeds 93683 and 116089, two births.

**Addition 1 (the reviewer's): one backpropagation birth with the same gain, arm `B050`, seed 73129.** `exp34_entropy.dev` sets `local = True` at line 138; add an arm whose only difference is `local = False` in the forward, tau 0.5, in a way that leaves the existing arms' code path unchanged. It says whether the gain's effect belongs to the sweep or to the objective.

- **Check before it runs:** with tau infinite and `local = False`, a dry birth at 2,048 updates has the `tensors_sha256` of `exp45/e28_qual/dry/e28_dev_B-all_dry2048_73129.json`.
- **Its reading, fixed now**, against plain backpropagation's 82.0 at 40 letters (last-three mean) and `T050`'s 79.8, on seed 73129 only:

  | `B050` | Reading |
  |---|---|
  | Does not learn in range | Under backpropagation the gain at this temperature prevents in-range learning on this seed. |
  | Learns in range; at 40 letters within 15 of 82.0 | The gain neither helps nor hurts backpropagation's extrapolation here. What `T050` shows is the sweep being brought to backpropagation's level by the objective. |
  | Learns in range; at least 15 above 82.0 | The gain helps backpropagation too: the benefit belongs to the objective, not to the rule. |
  | Learns in range; at least 15 below 82.0 | The gain hurts backpropagation's extrapolation on this seed and not the sweep's. |

**Addition 2 (the reviewer's): two reported facts for every arm of stage 2 and for `T050` on 73129.** The first saved row at which accuracy at 12 letters reaches 70, beside plain backpropagation's (row 4,096 on 73129) and the plain sweep's; and the three rows at 40 letters beside their mean.

**The sentence**, if `T050` learns in range and extrapolates on both new seeds, is the brief's, followed by: "It first learns in range at row … (plain backpropagation: row …)."

Three births, about 2.5 CPU-hours. Marks as in stage 1.

## Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| P34.9 | `T050` learns in range and extrapolates on both new seeds | 40% |
| P34.10 | `T050` does so on at least one new seed | 75% |
| P34.11 | `T050` first reaches 70 at 12 letters at row 12,288 or later on both new seeds | 65% |
| P34.12 | `B050` is below 70 at 12 letters at row 8,192 | 60% |
| P34.13 | `B050` learns in range and is within 15 of 82.0 at 40 letters | 50% |

## Done

The check receipt for `B050`; three births with receipts, finals and rows 16,384 and 20,480 committed; an addendum to `E34_REPORT.md` with the per-seed table, the two facts, which reading applies to `B050`, and P34.9 to P34.13. No readings beyond the tables above.

---

# Addendum 2 (15 November 2026): two long births (the reviewer's memo, revision 3, section 18)

**Why.** The user's point of 15 November: a learner that starts slower may overtake later, and neither a short run nor an average over a run can show it. `T050` on seed 73129 is the one case on file with that shape: behind for 12,000 updates (17 against 95 at 12 letters at row 12,288), ahead at the last row (99.3 against 84.7 at 40 letters), with the last three rows still swinging. The test of that shape is a longer run, not more runs. Evidence level E2; one seed.

**Runs.** Seed 73129, 98,304 updates each (four times the dose), rows every 4,096:

- `T050`, the sweep with the gain at tau 0.5 (`exp34_entropy.py`);
- `B-all`, plain backpropagation with every byte a target (`exp28_everybyte.py`).

About 6.6 CPU-hours by the reviewer's estimate. They may run beside stage 2 within three processes.

**Code. The memo says none is needed; one change is.** `exp28_everybyte.dev` checks the data digest only at rows its reference has (lines 231 and 260), so `E11_DEV_UPDATES=98304` runs as it is. `exp34_entropy.dev` asserts at line 181 that every saved row is in the reference, and S-all's reference ends at 24,576; and `e28_reference` (lines 126 to 131) must return the full-dose reference `exp45/e28/e28_dev_S-all_<seed>.json` when the dose is above 24,576, since `DEVTAG` is set for any non-standard dose and the dry path it builds does not exist. (Added after the reviewer's audit of 15 November, which found this second obstacle; the worker's branch already carries the fix at be01b07c.) Make the smallest change that lets a longer dose run: the digest and hash comparisons apply at rows the reference has and are skipped, and recorded as skipped, beyond it. Show that a dry birth of `T050` at 2,048 updates still gives the `tensors_sha256` in `exp45/e34/dry/` after the change.

**Check.** At row 24,576 each long run has the `tensors_sha256` of its existing final receipt (`exp45/e34/e34_dev_T050_73129.json` and `exp45/e28/e28_dev_B-all_73129.json`), since stream and update are unchanged. If either differs, stop and report.

**Files.** Commit the weights at rows 49,152, 73,728 and 98,304 of each run; record the sha256 of the others. Read every row with the reader (12, 20 and 40 letters).

**Reading (fixed; the reviewer's), at 40 letters over the 13 rows from 49,152 to 98,304.**

| Result | Reading |
|---|---|
| `T050`'s mean is at least 10 points above `B-all`'s and it is above in at least 10 of the 13 rows | Overtakes and stays, on this seed up to 98,304 updates. The same two runs then go to seeds 93683 and 116089. |
| `T050`'s mean is not above `B-all`'s | Does not overtake up to 98,304 updates on this seed. |
| Otherwise | Mixed; the rows are reported. |

Rows at 12 and 20 letters are reported beside them. `B050` (addendum 1) is run at this length only if its endpoint at the present length differs from plain backpropagation's by 15 points or more at 40 letters.

**What it cannot show.** The stream is stationary, so this bears on the shape "slow start, later lead" and not on continual learning. Every statement carries its horizon, 98,304 updates.

**Predictions (mine).**

| | Prediction | Chance |
|---|---|---|
| P34.14 | The row-24,576 hash check passes for both runs | 85% |
| P34.15 | Overtakes and stays | 30% |
| P34.16 | Does not overtake | 35% |
| P34.17 | `B-all`'s mean at 40 letters over the 13 rows is below its present endpoint of 82.0 by at least 10 points (backpropagation itself degrades with length of training) | 20% |
