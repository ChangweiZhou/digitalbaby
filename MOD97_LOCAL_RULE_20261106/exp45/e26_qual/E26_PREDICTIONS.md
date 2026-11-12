# E26 predictions (written in stage 1, before any full-dose birth or life)

**Brief:** `OPUS_TASK_E26_GRU_UNDER_THE_SWEEP_20261111.md`, revision 2.

**Source of the text below.** Copied unchanged from the brief at branch `codex/minifly-consolidated-20261023`, commit `17a5fa35`: the predictions (lines 88 to 96) and the readings (lines 98 to 105). Nothing has been reworded. This file is committed before any run, of any dose, and is not to be changed afterwards.

**Status.** No full-dose birth or life has been run. Stage 2 (six births) and stage 3 (six lives) are each released by the coordinator.

## Predictions (copy unchanged into `E26_PREDICTIONS.md` before any full-dose run). "Holds" means in at least two of three seeds

- **P26.1** A sweep-born GRU is born at a hard mean of at least 70. Confidence about 50%.
- **P26.2** It is born at 0.85 or more of its backprop partner's hard mean. Confidence about 40%.
- **P26.3** Arm S ends its life at or above 85 on new instances of taught pairs. Confidence about 25%.
- **P26.4** Arm S ends its life at or above 40 on the untaught plus-minus pairs. Confidence about 25%.
- **P26.5** Arm B meets both marks (85 and 40) in the same two seeds. Confidence about 80%.

The marks 85 and 40 are E20-B's, fixed before E20-B ran.

## Readings, fixed now

- Two things are read, and kept apart. *Feasibility:* does arm S meet the absolute marks. *The gap:* S minus B, seed by seed.
- **A candidate pair exists** if arm S meets both marks in the same two or more seeds. Then the sweep and the GRU were born and lived together and compose from memory on untaught pairs, under answer targets and a joint stream. That is what the close may name, with those two conditions attached.
- **P26.5 misses:** the reference of this job did not reach the marks. Arm S's own result against the marks is still reported, and so are the paired differences, beside the fact that the reference fell short. E20-B's G1 differs from arm B in decay, in the recurrent layer's code and in host, so it cannot say which of the three is responsible; a backprop arm at 0.30 on this host would, and is then the follow-up.
- **P26.1 holds and P26.3 or P26.4 misses:** the sweep learns single operations in a GRU and does not carry them to composition. The scores are reported beside arm B's, and the gap is located by checkpoint and by operator pair.
- **P26.1 misses:** report the level. A birth under 70 may still have learned a good deal, and it still gets its life. Only if the three births are near the floor is it said that the sweep does not bring a GRU to birth at this dose, although its update is well aligned on backprop-trained weights. Then the remedies of E25's predictions file are considered in their order, with learned feedback weights now permitted (Mr. Xu's ruling of 11 November). The alignment measurement is repeated on the sweep-born file to see where the sweep took the network.
- Under every outcome the paired difference S minus B is reported for each seed at birth and at the end of the life.

## How each quantity will be read from the receipts (the worker's reading, stated before any result; for the coordinator to confirm or correct)

- **Hard mean (P26.1, P26.2):** in the birth receipt `e26_dev_<arm>_<seed>.json`, the last curve row (step 24,576): the mean of `sealed.add89.12.exact` and `sealed.sub89.12.exact`, each of 300 items, in percent. P26.2 is the ratio of arm S's hard mean to arm B's, seed by seed.
- **New instances of taught pairs (P26.3, P26.5):** in the life receipt `e26_life_<arm>_<seed>.json`, stage 6: `scores.compose.within.exact` of 600, in percent.
- **Untaught plus-minus pairs (P26.4, P26.5):** stage 6, `compose["+-"].exact` and `compose["-+"].exact` pooled over 1,200, in percent. Each of the four untaught pairs is also reported separately at every checkpoint.
- **Whole answer** = `exact` (all six bytes, end symbol included), scored by `exp16_integrated.score`. **Holds** = in at least two of three seeds; for P26.5 the same two seeds for both marks.


## Verdicts, 12 November 2026, 09:53 EST (clock; review 9.110)

**E26, births** (recomputed from `exp45/e26/e26_dev_<arm>_<seed>.json`; seeds 73129 / 93683 / 116089):
- P26.1 holds: the sweep-born GRU's hard mean is 98.0 / 98.2 / 99.8.
- P26.2 holds: it is 0.995 / 0.995 / 1.065 of its backprop partner's (98.5 / 98.7 / 93.7).
- P26.3 to P26.5 wait for the lives.

