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
