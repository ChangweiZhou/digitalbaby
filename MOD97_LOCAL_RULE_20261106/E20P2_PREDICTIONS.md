# E20-P2: what the wrong intermediates are, and whether the first operator is kept. Predictions, written before this script touches a trained checkpoint

10 November 2026, 23:27 EST (clock). Coordinator. Exploratory and diagnostic, evidence level E2. **Designed after E20-P's results were seen** (review 9.93), on the same three checkpoints, so it cannot confirm that result independently; it can only add detail to it.

## What is asked

E20-P found that the window reference keeps the hidden operands and still gets the first step wrong more often from memory than from view. Two things it did not look at:

- **A. What the wrong intermediates are.** Each readable wrong intermediate goes into the first category it fits, in this fixed order: the other operator of the same family; the operands swapped; another modulus (89 or 73); a copied operand; the first operand with one digit wrong; the second operand with one digit wrong; off by one; one digit of the result right; other. The shares are set beside what a uniformly random wrong two-digit value would give for the same items.
- **B. Whether the first operator is still in the state at `=`.** It is out of view in most twins. A four-way ridge readout (penalty 1.0, as before), fitted on 6,400 fresh items of all sixteen operator pairs and tested on 1,600 more and on the twin items.

## Why sixteen pairs

Among the four held pairs the first operator determines the second. A readout fitted on held pairs alone could report the first operator while reading the second, which is often in view. The fixtures show it: with features that contain only the second operator, a readout fitted on the held four "reads" the first operator on 40 of 40 twin items; fitted on all sixteen it gets 8 to 11 of 40, which is chance.

## What I knew when writing this

- All of E20-P (review 9.93): operands read out on 100% of twin items; intermediates right on 47 / 57 / 76% of plus-minus twin items; the second step sound; losing the second operator from view costs 0.3 to 7 points.
- This script's fixtures, on untrained nets and stub answers only (`exp45/E20P2_FIXTURES_<seed>.json`). On an untrained net with a small fit the first operator reads out on 35 to 42% of items against a chance of 25%, so the untrained control is not at chance here and is reported beside the result.
- This script has not been run on a trained checkpoint.

## Predictions. "Holds" means in at least two of the three seeds

- **P-P2.1** The first operator reads out on at least 90% of twin items in which it is out of view, and on at least 90% of the sixteen-pair test. Confidence about 65%. Void if the shuffled-label control exceeds 35%.
- **P-P2.2** Among the wrong readable intermediates of plus-minus twin items, "the other operator" holds at least 25%. The uniform baseline is about 1%. Confidence about 45%.
- **P-P2.3** At least 50% of them fall in the four near-miss categories together (either operand one digit wrong, off by one, one digit of the result right). The uniform baseline is roughly a quarter. Confidence about 55%.
- **P-P2.4** The largest category is the same for the twins and for the leading control (the same items with everything in view). Confidence about 50%.

## Readings, fixed now

- **P-P2.1 holds:** operands and operator are all in the state at `=`. Everything the first step needs is kept, and the failure is in computing with it.
- **P-P2.1 misses with its control in order:** the operator is the piece not shown to be kept. The reading of 9.93, "kept and not used", then holds for the operands only, and the next look is at where the operator goes.
- **P-P2.2 holds:** from memory the learner often applies the other operation of the pair in the first step. That is a confusion of roles between the two operators, specific to an untaught pair.
- **P-P2.3 holds without P-P2.2:** the first step is the right operation done imprecisely. The loss is in precision of use, not in the choice of operation.
- **P-P2.4 misses:** hiding the operand does not just make the same errors more frequent; it brings in a different kind.
- **Most errors in "other" with no category above baseline:** this breakdown does not explain them, and I say so.

## Limits

Three seeds, one network, backprop, end-of-life weights. The categories are mine, their order decides ties, and the baseline assumes uniform wrong values, which real errors are not. Nothing here is about the working rule.
