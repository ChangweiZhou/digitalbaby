# E27: does the lack of magnitudes in the sign path explain the every-byte collapse and the weaker composition? Predictions, written before any saved file is read

13 November 2026, 11:39 EST (clock). Coordinator. Two measurements on saved weights; nothing is trained. Diagnostic, E2. Suggested by the audit session's note of 13 November and adopted in review 9.112 E. No outside auditor has read this; the audit session is asked to, as second reader.

## The hypothesis

The working rule sends error back through weight signs with one scale per matrix. It carries no magnitudes. That could explain two open failures at once:

1. **The every-byte collapse (E24).** Error from bytes that cannot be predicted would flood back undamped and swamp the answer's error.
2. **The weaker composition (E26, E26-X).** Credit would not favour the pathways that already do the work, and the sweep's update would follow the true gradient less well on composition items than on single operations.

## What is measured (`e27_measure.py`)

**A. Flood.** On 256 birth items (mod-89 operations at k = 12), under every-byte targets, the loss splits exactly into an answer part and a part on all other bytes. Both gradients are computed under the exact path and under the sign path. For every weight matrix and for all parameters together:
- *flood*: the size of the other bytes' gradient over the size of the answer's gradient, each weighted by its share of positions;
- *amplification*: flood under signs over flood under the exact path. The hypothesis needs this well above 1;
- *usable*: the cosine between the every-byte gradient under a path and the exact answer-only gradient. It says how much of the answer's direction survives.

Files: the two-bank network untrained, born by backprop and by the sweep under answer targets, and E24's three every-byte births; the GRU untrained and E26's two births. Three seeds.

**B. Alignment by kind of item.** On the end-of-life GRU files of E26 and its crossover, under answer targets, the cosine between the sweep's gradient and the true one on single operations, on new instances of taught pairs and on untaught plus-minus pairs.

## What I knew when writing this

- E24, E26 and E26-X in full (reviews 9.110, 9.112).
- E25 (9.103): on backprop-lived GRU files the all-parameter cosine was 0.67 / 0.64 / 0.74 on single operations and 0.62 / 0.64 / 0.67 on new instances of taught pairs; under every-byte targets alignment was not worse than under answer targets.
- This script's fixtures on untrained networks only (`exp45/E27_FIXTURES_<seed>.json`, all pass): the decomposition is exact to 5e-7 under both paths and fails with the shares swapped. On untrained networks the amplification is 0.98 to 1.05, and usable is 0.65 to 0.71 exact against 0.56 to 0.57 under signs in the two-bank network.
- No saved file has been read by this script. Of the files for B, the six crossover lives are in the repository; the six E26 lives' weights are not yet, and B is run on what is there and completed when they arrive.

## Predictions. "Holds" means in at least two of three seeds

- **P27.1** On the two-bank network born by the sweep under answer targets, the amplification over all parameters is at least 1.5. Confidence about 35%.
- **P27.2** On the same files, usable under signs is at most half of usable under the exact path. Confidence about 35%.
- **P27.3** On the collapsed file (born by the sweep under every-byte targets), flood under signs is at least twice what it is on the file born by the sweep under balanced targets. Confidence about 50%. This one describes the collapsed state and cannot by itself say what caused it.
- **P27.4** On GRU files that lived under the sweep (either birth), the all-parameter cosine on untaught plus-minus pairs is at least 0.10 below the cosine on single operations. Confidence about 40%.
- **P27.5** On the untaught plus-minus pairs, the cosine is lower on files that lived under the sweep than on files that lived under backprop, by at least 0.05, comparing within a birth. Confidence about 40%.

My confidences are below one half on purpose. E25 already showed the sweep well aligned through time and not worse under every-byte targets, which speaks against the hypothesis in its simple form.

## Readings, fixed now

- **P27.1 and P27.2 hold:** the flood is real at the point where a sweep-born network would meet every-byte targets. Feedback that carries magnitudes is the remedy to try, and learned feedback weights are inside the line by Mr. Xu's ruling of 11 November. A source check comes first.
- **P27.1 and P27.2 miss:** at a single update the sign path does not let filler error through any more than the exact path does. The collapse is then not explained by flooding at one step; it is something that builds over training, and the next look is at the trajectory (the receipts show the every-byte births peaking early and then dying), not at feedback magnitudes. Learned feedback weights lose their main motive for E24.
- **P27.4 or P27.5 holds:** the sweep's update is less faithful exactly where composition is learned, which fits the crossover. The same remedy applies, tested as lives from the existing sweep-born births, with the bounds on file (37 to 54 under the sweep, 72 to 83 under backprop).
- **P27.4 and P27.5 miss:** the sweep follows the gradient as well on compositions as on single operations, and the weaker carry-over is not a matter of alignment at these weights. The hypothesis is withdrawn for composition, and I say that no explanation is on file.
- **A mixed result** is reported as mixed. One hypothesis for both failures is the claim under test; half of it holding is not that claim.

## What this cannot say

Gradients at saved weights are not training. Files trained by one update are measured where that update took them. Three seeds, one size, 256 items a panel.


---

## Amendment before any saved file is read (13 November 2026, 11:49 EST, clock), after a hostile audit of this file, the brief and the script

The audit's disposition was to hold. The coordinator ran its main findings and they hold (review 9.114). **The five predictions above stay as registered and are judged as written.** What changes is what is recorded beside them and how much each reading may claim. No saved file had been run through the script when this was written.

**Where each prediction is read from** (`exp45/E27_MEASURE_<seed>.json`):
- P27.1: `A["two-bank born by the sweep, answer targets"].all_parameters.amplification`.
- P27.2: the same entry, `all_parameters.sign.usable` against `all_parameters.exact.usable`. If either is zero or negative the prediction is void for that seed.
- P27.3: `A[... (collapsed)].all_parameters.sign.flood` against `A[... balanced].all_parameters.sign.flood`.
- P27.4: `B[arm].untaught_plus_minus.all_parameters` against `B[arm].single_operations.all_parameters`, for the two arms that lived under the sweep. A seed counts only if both arms show it.
- P27.5: `B[...].untaught_plus_minus.all_parameters`, sweep-lived against backprop-lived within a birth: two comparisons a seed. A seed counts only if both show it.

**Known weaknesses of the registered predictions, stated before the result.**
- *P27.1 can miss with the hypothesis true.* The top layer's own gradient is the same under both paths, so its amplification is 1 by construction, and it carries a third to a half of the gradient's squared norm. In a control where every lower tensor is amplified about threefold, the all-parameter figure is only 1.55 in the two-bank network.
- *P27.1 and P27.2 are measured at weights the every-byte births never visited.* Those births started from random weights, where the amplification is already known to be about 1, and no checkpoint of them exists. And "other bytes" includes operands, the operator and `=`, which are learnable, beside the letters, which are not.
- *P27.2 has no baseline.* The sign path is misaligned with the exact one in general, so "usable under signs" is low for a reason that has nothing to do with flooding.
- *P27.3 is not informative.* It compares the sign path with itself, and its denominator follows the answer loss, which differs between the two files for reasons outside the hypothesis. It is judged and reported, and nothing is read from it.
- *P27.4 mixes composition with format.* Single operations have a 19-byte prompt with letters and three target bytes; composition items have a 10-byte prompt without letters and six. E25 already shows a gap of 0.01 to 0.07 between them on backprop-lived files.
- *P27.5 cannot be judged until the six E26 life weight files exist.* Each of its comparisons needs one of them.

**Supplementary quantities, registered now, each with its reading.**
- **S1. Amplification over all parameters except the top layer** (`l3.*`), and per tensor including the GRU's two recurrent matrices. *Reading:* this, not P27.1's figure, says whether filler error is amplified below the top layer. At least 1.5 in two seeds on the sweep-born answer-target file is read as flood present at those weights.
- **S2. The split of "other" into distractor letters and structural bytes** (operands, operator, task code, `=`), with amplification for each. *Reading:* the hypothesis is about the letters.
- **S3. Amplification on the file born by backprop under every-byte targets,** where what is left of the other bytes' error is the unpredictable part. *Reading:* the cleanest single test of "unpredictable error floods back under signs" that saved weights allow.
- **S4. A null:** amplification for random splits of positions at the same shares, twenty draws. *Reading:* a figure inside the null's range is not a flood.
- **S5. Two baselines for "usable":** the cosine of the sign answer gradient with the exact answer gradient, and of the sign every-byte gradient with the sign answer gradient.
- **S6. For part B, the same-format contrast:** untaught pairs minus taught pairs, on sweep-lived and on backprop-lived files, with the taught panel drawn evenly from all twelve taught pairs. *Reading:* only this difference, larger on sweep-lived files than on backprop-lived ones, speaks to composition.

**Readings narrowed.**
- A miss of P27.1 and P27.2 with S1 and S3 also near 1 is read as: *no amplification at these saved weights.* It does not say learned feedback weights have lost their motive, which the first version claimed. The weights where the collapse happens were never saved; that needs a birth with checkpoints, which is a different job.
- "P27.4 or P27.5 holds" no longer triggers the confirming reading. Composition is read from S6.
- P27.1 and P27.2 disagreeing, or the registered figures and the supplements disagreeing, is reported as that, with both.
- No life ever updates on untaught items, so nothing here can show the sweep "less faithful where composition is learned". Part B measures the sweep's direction on items of a kind, at the end of a life.
