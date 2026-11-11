# Draft for audit: E24, is the every-byte collapse the loss or the rule? (11 November 2026)

> **Status.** Draft for Codex's audit. **No worker starts from this file.** Coordinator's design; evidence level aimed at E2, development panels, births only. Codex's note of 11 November on the outline (answer-only and every-byte backprop beside the matched sign conditions at one dose; bounded readings) is taken into the text below.

## Why this job

Three results on file say the working rule fails when every byte is a target instead of the answer bytes only:

- a life from a good birth collapses (review 9.88: the backprop twin keeps 89 to 91, the sign sweep falls to 5 to 10);
- a birth with local heads collapses (E18b: 0.3 / 0.3 / 0.7 against 74.8 / 69.7 / 66.3);
- every-byte targets on the first bank's head alone are enough (E23's base arm: 7.0 / 5.5 / 10.0).

The target test has no answer mask. A learner there predicts every next byte. So this failure stands between the working rule and the test the project is heading for.

What is not on file is the smallest cell: **a backprop birth of the same network under every-byte targets.** Backprop survives every-byte targets in a life that starts from a trained birth. Nobody has run it from nothing. Without that cell the collapse cannot be laid at the rule's door.

## The question, and what each answer means

| Backprop birth under every-byte targets | Reading, fixed now |
|---|---|
| ends near its answer-target level | In this network at this dose, the objective harms the sign sweep and not backprop: an interaction between objective and update. Work goes to the rule. |
| collapses too | In this network at this dose, the objective defeats both updates at birth. The rule is not shown to be the cause, and the remedy is looked for in the loss or the stream. |

Either reading is about the configuration tested: one network, one dose, three seeds. It names where to look next and is not a statement about every-byte targets in general (Codex's audit of 11 November).

A second, cheaper question rides along: is the answer signal simply outnumbered? In an item of about twenty bytes, three are answer bytes and most of the rest are distractor letters that cannot be predicted. One arm gives the answer bytes half the loss and all other bytes the other half.

## The learner

The learner of record, the one E22 took through a life: `E16_ARCH=bank2 E16_BOOKDECAY=off`, one head at the top, no local heads. Births are `exp16_integrated.py dev <seed>`: 24,576 updates of 64 items on the four mod-89 operations with 0 to 12 distractor letters; scored on sealed panels, whole own answers with the end symbol.

The local-heads line is closed (review 9.97) and no arm here uses it.

## Arms. Seeds 73129, 93683, 116089. One host for all of them

| Arm | Update | Targets | Extra environment | Status |
|---|---|---|---|---|
| D-ans | backprop | answer bytes | (none) | on file: 87.5 / 90.7 / 84.7. Rerun seed 73129 only, as a host check |
| T-ans | sign sweep | answer bytes | `E16_BIRTHRULE=T E16_TRACEUP=all` | on file: 80.7 / 82.2 / 77.2. Rerun seed 73129 only, as a host check |
| **D-all** | backprop | every byte | `E16_LOSS=all` | new, three seeds |
| **T-all** | sign sweep | every byte | `E16_BIRTHRULE=T E16_TRACEUP=all E16_LOSS=all` | new, three seeds |
| **T-bal** | sign sweep | every byte, answer bytes carrying half the loss | the same plus `E16_LOSSW=balanced` | new, three seeds |

- **Host check.** The two reruns must reproduce the sealed scores on file at every checkpoint. E18b's same-host partners did so exactly on a different host and Python. If either differs, stop and report before any new arm.
- **`E16_LOSSW=balanced`** is a new option, default off, added by the coordinator with its own check (see Admission). With it the loss is one half the mean cross-entropy over answer positions plus one half the mean over all other real positions. Nothing else changes.
- **One difference per comparison.** D-all against D-ans: targets. T-all against T-ans: targets. T-bal against T-all: the weighting. D-all against T-all: the update.
- **Not run here:** a balanced backprop arm (wanted only if D-all collapses, and then it is its own decision); lives; any predictable-filler stream; any local head.

## Endpoints

At the final checkpoint, k = 12, the top head, whole answers: hard mean (add89 and sub89) and max/min mean. Also: the six-checkpoint curve of the hard mean; the embedding movement by symbol role and the credit census the receipts already record; CPU-seconds.

Max and min are read against 50, the level that copying one operand reaches (erratum in review 9.87). The hard mean carries the verdicts.

## Predictions (copy unchanged into `E24_PREDICTIONS.md` before any new arm starts)

- **P24.1** D-all ends at a hard mean of at least 60 in at least two seeds. Confidence about 60%.
- **P24.2** T-all ends at or below 10 in all three seeds. Confidence about 80%.
- **P24.3** T-bal ends at or above 40 in at least two seeds. Confidence about 35%.

Readings, fixed now:

- P24.1 and P24.2 hold: here backprop learns from every byte from nothing and the sign sweep does not. That is an interaction between the objective and the update in the tested configuration, and the next step is to look at what the sign feedback does with targets that cannot be predicted.
- P24.1 misses (D-all under 60 in two seeds): the loss on this stream defeats backprop at birth too. The rule is not shown to be the cause. The next step is the stream or the loss, for both updates.
- P24.3 holds: in this configuration the weighting restores learning. It establishes the effect of the weighting here and nothing about where such a weighting could come from without an answer mask, which becomes the next question.
- P24.3 misses with P24.2 holding: the harm is not a matter of proportion.
- P24.2 misses: the top-head learner survives every-byte targets at birth where the local-head learner did not, and the earlier collapses have to be reread as belonging to the local heads and to the life.

## Admission, before any run

1. `python3 e24_check.py` exits 0 on your machine. It shows, on one batch: the balanced loss equals one half the answer-position loss plus one half the other-position loss computed separately; its gradient equals that combination of the two gradients; with the option unset, every-byte and answer-target losses and gradients are what they were at the commit before the option was added; under `E16_LOSS=all` every operator, digit, letter and task code present in the batch receives a gradient under both updates. The coordinator's record is `exp45/E24_CHECK.json`.
2. A dry birth of each of the five arms at 512 updates completes with a completion record, the right `loss` and `lossw` fields, and its own file name (`_lossall`, `_lossall_bal`). The coordinator's are in `exp45/archive_e24_dry/`.
3. Births are bound by their completion receipts as before. No file on record may be overwritten: the launcher refuses if a product exists.

## Running it

- One host, at most three processes on four cores, one thread each.
- Order: the two host-check births and D-all seed 73129 first. Then the remaining D-all, then T-all, then T-bal.
- Resume is not qualified on the merged harness yet (Codex's audit of 11 November) and the harness refuses `E16_RESUME=1`. Until `exp45/RESUME_QUALIFIED.json` exists for the sources you run, an interrupted birth is kept as written, stopped and reported, and it is restarted from the beginning only on the coordinator's word. A birth is about an hour, so a restart costs at most that.
- Work in progress goes to your own branch. The default branch receives one commit when all eleven completion records and weight hashes are verified.
- Forecast: nine new births and two reruns at about 3,500 to 4,500 CPU-seconds each, about 45,000 in all, about four and a half hours on three cores. Replace this with the pace you measure on the first wave.

## Report

Per arm and seed: the sealed scores at every checkpoint; the hard-mean curve; embedding movement by role; the three verdicts; CPU-seconds; the host check. `E24_PREDICTIONS.md`, `E24_REPORT.md`, receipts, weights, logs, and one paragraph as `E24_REVIEW_SECTION.md`. State what was not run. Claim nothing about lives, composition or the target test.

## Questions for the audit

1. Is one backprop arm under every-byte targets enough to separate "the loss" from "the rule", or does it need the backprop balanced arm beside it from the start?
2. The balanced weighting uses the answer mask to set weights. It is a diagnostic and not a candidate mechanism. Is that stated strongly enough?
3. Should the host check rerun all three seeds of the two partners rather than one?
