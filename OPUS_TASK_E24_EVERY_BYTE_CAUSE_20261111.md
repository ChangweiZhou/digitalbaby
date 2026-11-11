# Draft for audit: E24, is the every-byte collapse the loss or the rule? (11 November 2026)

> **Status.** Draft for Codex's audit. **No worker starts from this file.** Coordinator's design; evidence level aimed at E2, development panels, births only. Codex's note of 11 November on the outline (answer-only and every-byte backprop beside the matched sign conditions at one dose; bounded readings) is taken into the text below.

> **Revision 2, 11 November 2026, 16:19 EST (clock), after Codex's audit of this brief** (`PROJECT_DOCUMENTS/MINIFLY_E24_STEP0_V3_AUDIT_20261111.md`). Every point was checked and holds. Changed:
> 1. The two reference baselines were mislabelled. The backprop birth is 91.2 / 94.5 / 90.3 and the sign-sweep birth is 87.5 / 90.7 / 84.7. The first draft gave the sign-sweep scores to backprop and a local-head arm's scores to the sign sweep.
> 2. The host checks are a wave of their own, in a fresh folder, before any new birth. The first draft ran a new birth beside them, against its own stop condition.
> 3. The two updates also differ in weight decay (0.30 and 0.25). The units compared are the two training configurations, and the result is read as an interaction between objective and configuration.
> 4. The readings are narrower: a miss is reported with its level, not as a collapse.
> 5. The admission check now really compares with the harness before the option existed (a fixture added to `e24_check.py`).
> The roster is unchanged: nine new births and two host checks. Still a draft; no worker starts before Codex has read this revision.

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
| ends near its answer-target level | In this network at this dose, the objective costs the sign-sweep configuration far more than the backprop one: an interaction between objective and configuration. Work goes to the rule first, with decay as the one known confound. |
| ends well below its answer-target level | The objective costs both configurations at birth. The size of each penalty is reported; the rule is not singled out, and the remedy is looked for in the loss or the stream. |

Either reading is about the configuration tested: one network, one dose, three seeds. It names where to look next and is not a statement about every-byte targets in general (Codex's audit of 11 November).

A second, cheaper question rides along: is the answer signal simply outnumbered? In an item of about twenty bytes, three are answer bytes and most of the rest are distractor letters that cannot be predicted. One arm gives the answer bytes half the loss and all other bytes the other half.

## The learner

The learner of record, the one E22 took through a life: `E16_ARCH=bank2 E16_BOOKDECAY=off`, one head at the top, no local heads. Births are `exp16_integrated.py dev <seed>`: 24,576 updates of 64 items on the four mod-89 operations with 0 to 12 distractor letters; scored on sealed panels, whole own answers with the end symbol.

The local-heads line is closed (review 9.97) and no arm here uses it.

## Arms. Seeds 73129, 93683, 116089. One host for all of them

| Arm | Update | Targets | Extra environment | Status |
|---|---|---|---|---|
| D-ans | backprop, decay 0.30 | answer bytes | (none) | on file: 91.2 / 94.5 / 90.3. Rerun seed 73129 only, as a host check |
| T-ans | sign sweep, decay 0.25 | answer bytes | `E16_BIRTHRULE=T E16_TRACEUP=all` | on file: 87.5 / 90.7 / 84.7. Rerun seed 73129 only, as a host check |
| **D-all** | backprop, decay 0.30 | every byte | `E16_LOSS=all` | new, three seeds |
| **T-all** | sign sweep, decay 0.25 | every byte | `E16_BIRTHRULE=T E16_TRACEUP=all E16_LOSS=all` | new, three seeds |
| **T-bal** | sign sweep, decay 0.25 | every byte, answer bytes carrying half the loss | the same plus `E16_LOSSW=balanced` | new, three seeds |

- **Host check, a wave of its own.** Run the two reruns alone, before any new birth, in a work folder whose `exp45` is empty except for `BIRTH_MANIFEST.json`. The harness returns without running if it finds a birth of the same name and configuration, so confirm that each check wrote a new receipt (its time and its `cpu` fields) and did not find an old one. Each must reproduce the sealed scores of its pinned original at every checkpoint. If either differs, stop and report; no new birth starts.

  The six originals, pinned (receipt hash, weights hash from the manifest):

  | Receipt | sha256, first 16 | Weights sha256, first 16 |
  |---|---|---|
  | `e17_dev_bank2_nbd_73129.json` | `e1e0826e4856e03e` | `8ee5e4e673b70b28` |
  | `e17_dev_bank2_nbd_93683.json` | `f9927292629f13fa` | `5d37943da6dbe1d7` |
  | `e17_dev_bank2_nbd_116089.json` | `b5b447a38815fa6d` | `aac1f6dfe55db03c` |
  | `e17_dev_bank2_nbd_bT_tall_73129.json` | `54f14b0227b059af` | `e679f51601ee4128` |
  | `e17_dev_bank2_nbd_bT_tall_93683.json` | `e7caa017b886d787` | `03f1ed34257bd6e1` |
  | `e17_dev_bank2_nbd_bT_tall_116089.json` | `a0b394c449b334d1` | `77d8e9c722dc3cf3` |

- **`E16_LOSSW=balanced`** is a new option, default off, added by the coordinator with its own check (see Admission). With it the loss is one half the mean cross-entropy over answer positions plus one half the mean over all other real positions. Nothing else changes.
- **What differs in each comparison.** D-all against D-ans: the targets. T-all against T-ans: the targets. T-bal against T-all: the weighting. D-all against T-all: the update **and the weight decay** (0.30 against 0.25, as every birth of each kind on file). The two updates are therefore compared as two training configurations. Isolating the update would need matched decay, and that is not run here.
- **Not run here:** a balanced backprop arm (wanted only if D-all collapses, and then it is its own decision); lives; any predictable-filler stream; any local head.

## Endpoints

At the final checkpoint, k = 12, the top head, whole answers: hard mean (add89 and sub89) and max/min mean. For each seed, the penalty of every-byte targets within each configuration (D-ans minus D-all, T-ans minus T-all) and the difference between the two penalties, which is the interaction this job measures. Also: the six-checkpoint curve of the hard mean; the embedding movement by symbol role and the credit census the receipts already record; CPU-seconds.

Max and min are read against 50, the level that copying one operand reaches (erratum in review 9.87). The hard mean carries the verdicts.

## Predictions (copy unchanged into `E24_PREDICTIONS.md` before any new arm starts)

- **P24.1** D-all ends at a hard mean of at least 60 in at least two seeds. Confidence about 60%.
- **P24.2** T-all ends at or below 10 in all three seeds. Confidence about 80%.
- **P24.3** T-bal ends at or above 40 in at least two seeds. Confidence about 35%.

Readings, fixed now:

- P24.1 and P24.2 hold: here backprop learns from every byte from nothing and the sign sweep does not. That is an interaction between the objective and the update in the tested configuration, and the next step is to look at what the sign feedback does with targets that cannot be predicted.
- P24.1 misses (D-all under 60 in two seeds): report the level. A birth under 60 may still have learned a good deal, so a miss is read with the penalty beside it and is not called a collapse unless the scores are near the floor. The rule is then not singled out, and the next step is the stream or the loss, for both configurations.
- P24.3 holds: in this configuration the weighting restores learning. It establishes the effect of the weighting here and nothing about where such a weighting could come from without an answer mask, which becomes the next question.
- P24.3 misses with P24.2 holding: the weighting did not restore learning to 40. T-bal is still judged against T-all seed by seed, and a partial improvement below 40 is reported as such.
- P24.2 misses: it takes one T-all seed above 10 to miss, and that leaves the question of survival unresolved, not answered. The three scores are reported, and only if all three are well above the floor are the earlier collapses reread as belonging to the local heads and to the life.

## Admission, before any run

1. `python3 e24_check.py` exits 0 on your machine. It shows, on one batch: the balanced loss equals one half the answer-position loss plus one half the other-position loss computed separately; its gradient equals that combination of the two gradients; with the option unset, for both updates and both objectives, the loss and every gradient equal bit for bit those of the harness at the commit before the option was added (read from the repository's history; in a clone without that commit this part reports itself as not run and the coordinator's record stands); under `E16_LOSS=all` every operator, digit, letter and task code present in the batch receives a gradient under both updates. The coordinator's record is `exp45/E24_CHECK.json`.
2. A dry birth of each of the five arms at 512 updates completes with a completion record, the right `loss` and `lossw` fields, and its own file name (`_lossall`, `_lossall_bal`). The coordinator's are in `exp45/archive_e24_dry/`.
3. Births are bound by their completion receipts as before. No file on record may be overwritten: the launcher refuses if a product exists.

## Running it

- One host, at most three processes on four cores, one thread each.
- Order: the two host-check births alone. If both reproduce, D-all, then T-all, then T-bal, three seeds at a time.
- Resume is not qualified on the merged harness yet (Codex's audit of 11 November) and the harness refuses `E16_RESUME=1`. Until `exp45/RESUME_QUALIFIED.json` exists for the sources you run, an interrupted birth is kept as written, stopped and reported, and it is restarted from the beginning only on the coordinator's word. A birth is about an hour, so a restart costs at most that.
- Work in progress goes to your own branch. The default branch receives one commit when all eleven completion records and weight hashes are verified.
- Forecast: nine new births and two reruns at about 3,500 to 4,500 CPU-seconds each, about 45,000 in all, about four and a half hours on three cores. Replace this with the pace you measure on the first wave.

## Report

Per arm and seed: the sealed scores at every checkpoint; the hard-mean curve; embedding movement by role; the three verdicts; CPU-seconds; the host check. `E24_PREDICTIONS.md`, `E24_REPORT.md`, receipts, weights, logs, and one paragraph as `E24_REVIEW_SECTION.md`. State what was not run. Claim nothing about lives, composition or the target test.

## Questions for the audit

1. Is one backprop arm under every-byte targets enough to separate "the loss" from "the rule", or does it need the backprop balanced arm beside it from the start?
2. The balanced weighting uses the answer mask to set weights. It is a diagnostic and not a candidate mechanism. Is that stated strongly enough?
3. Should the host check rerun all three seeds of the two partners rather than one?
