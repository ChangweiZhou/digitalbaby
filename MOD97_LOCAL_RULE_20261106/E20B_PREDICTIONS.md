# E20-B predictions: can a windowless network of this size compose from memory under backprop? (written in stage 1, before any full-dose birth or life)

**Brief:** `OPUS_TASK_E20_COMPOSITION_FROM_MEMORY_20261110.md`, Job B, with amendments 1 to 16 (they govern where they differ from the text of Job B). This session runs Job B, stage 1 only. Job A is not this session's.

**Source of the text below.** Copied unchanged from the brief at branch `codex/minifly-consolidated-20261023`, commit `9031effb`: the four predictions (lines 76-81) and the selection rule of amendment 13 (lines 138-143). Nothing has been reworded. This file is committed before any stage-2 run and is not to be changed afterwards.

**Status.** No full-dose birth or life has been run. Stage 2 (twelve births, twelve lives) starts only on the coordinator's go, after Codex has read the stage-1 record (amendment 8).

## Predictions (copied unchanged)


**Predictions (copy into `E20B_PREDICTIONS.md`).** Whole answers, end of life.
- **P20.7** G1 and G2 are born at a hard mean (add89, sub89 at k = 12) of at least 85 in at least two seeds each.
- **P20.8** At least one of G1 and G2 ends at or above 85 on new instances of taught operator pairs in at least two seeds.
- **P20.9** At least one of G1, G2 and G3 ends at or above 40 on the plus-minus pairs in at least two seeds. Held at about 50%.
- **P20.10** G3 ends at least 15 points above G0 on new instances of taught operator pairs in at least two seeds. Held at about 50%.

## Selection rule, amendment 13 (copied unchanged)

13. **How the predictions feed the decision. Fixed now, before any result.** P20.7 to P20.10 are judged exactly as registered. They do not by themselves choose a network: P20.8 and P20.9 can hold for different networks, or for one network in different seeds. The selection rule is separate:
    - A network **qualifies** if it ends at or above 85 on new instances of taught operator pairs and at or above 40 on the plus-minus pairs, whole answers, in the same two or more seeds.
    - If G3 qualifies, G3 is the windowless candidate, whether or not a gated network also qualifies. Reason: it is one matrix away from the network on which the working rule already trains and is born (review 9.70, 9.85).
    - Otherwise, if exactly one of G1 and G2 qualifies, that one is the candidate. If both do, the one with the higher three-seed mean on the plus-minus pairs; on an exact tie, the GRU.
    - If none qualifies, there is no windowless candidate from this job, whatever the four verdicts are.
    - Report each of the four held operator pairs separately at every checkpoint, so that the pooled plus-minus figure can be read.

## Limits that apply to every verdict (amendments 6, 14, 15; restated, not new)

- If all three new networks miss P20.9, that bounds these three packages at this dose; it does not show that the teaching is the cause.
- P20.10 measures the added connection as a package (32,768 weights, a 64-by-512 matrix with no bias, and the work they cost), not the placement of the feedback.
- "Windowless" describes the network's input. The registered decoder keeps the prompt and the bytes produced so far and reruns the network for each output byte; nothing here measures single-pass inference.
- Development feasibility result at evidence level E2; no claim of reuse or promotion. Backprop is a reference, not a candidate rule.

## How each quantity will be read from the receipts (the worker's operational reading, stated before any result; for the coordinator to confirm or correct before stage 2)

- **Hard mean (P20.7):** in the birth receipt `e20_dev_<net>_<seed>.json`, the last curve row (step 24,576): the mean of `sealed.add89.12.exact` and `sealed.sub89.12.exact`, each over its 300 items, in percent.
- **New instances of taught operator pairs (P20.8, P20.10, selection):** in the life receipt `e20_life_<net>_<seed>.json`, stage 6: `scores.compose.within.exact` over its 600 items, in percent.
- **Plus-minus pairs (P20.9, selection):** stage 6, `compose["+-"].exact` and `compose["-+"].exact` pooled over their 1,200 items, in percent (the brief's Job A defines "plus-minus" as the `+-` and `-+` pairs pooled). Each of the four held pairs is also reported separately at every checkpoint.
- **Whole answer** = `exact` (all six bytes, end symbol included), scored by `exp16_integrated.score`. `final` (last three bytes) is diagnostic only (amendment 10).
- **Taught instances** = `scores.taught` (600 items: the first 50 of `triples(pair, seed, 2000)` for each taught pair), kept outside `compose`.
