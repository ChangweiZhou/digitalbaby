# E20-P: probes on the saved E20-A checkpoints. Predictions, written before any trained checkpoint is probed

10 November 2026, 22:17 EST (clock). Coordinator. Evidence level E2, development panels; diagnostic only. Nothing here changes a registered E20-A verdict.

## What is asked

E20-A stage 1 showed that the window reference under backprop composes with the first operand out of view, partly (review 9.77). Where does a twin fail?

1. **At the intermediate, or after it?** (Codex's first probe.)
2. **Does the score really come from earlier state?** (Codex's second probe.)
3. **Is the hidden operand still in the state at `=`, or already gone?** (My addition. It separates "did not keep it" from "kept it and could not use it".)

## What is run

`e20_probe.py`, scoring only, on the final weights of the three window references `e16_life_D_cm6_{73129, 93683, 116089}.pt`. One process, about 20 CPU-seconds per checkpoint by the self-test's pace. The four stage 2 checkpoints already complete are not probed until stage 2 is complete and filed.

- **Probe 1.** For every twin item (4 held operator pairs, 150 twins each), the leading-control item, and the two hidden-operand panels of taught pairs: the answer written alone; the ending written after the true intermediate is supplied; and whether the written ending is the correct second step of the learner's own written intermediate.
- **Probe 2.** The same twins with the state set to zero (a) just before `=`, after which everything the learner reads is shared by the two members, and (b) just before the first of the last ten bytes, which is Codex's wording.
- **Probe 3.** A ridge readout (penalty 1.0, fixed) of two digits from the 512 state values after `=`, fitted on 6,000 fresh items of the held operator pairs that share no operand triple with the twin panel or with teaching, tested on the 1,200 twin items. Read out: the first operand (hidden), the second, the third (in view), and the intermediate. Controls: shuffled labels; an untrained net of the same seed.

## What I knew when writing this

- The receipts of the seven complete window lives of E20-A, including the three probed here. For those three: twins both correct, pooled over the four pairs, 42.3 / 42.7 / 49.7%; plus-minus pairs 21.0 / 26.7 / 43.3%; comparison pairs 63.7 / 58.7 / 56.0%; twin items 60.6 / 62.2 / 70.9%; leading control both correct 73.2 / 66.2 / 72.0%.
- The fixtures of the script, on untrained nets and stub answers only (`exp45/E20P_FIXTURES_<seed>.json`). On an untrained net the hidden first operand reads out on 3 of 160 twin items (chance) and the in-view third operand on 118 to 134 of 160.
- No probe has been run on any trained checkpoint.

## A point about probe 2, found while building it

Reset (a) gives zero twins by construction: after it the two members are the same input to the same state, so they get the same answer and cannot both be right. The script asserts this. It is a check that the assay has no leak, not a measurement. Reset (b) is a measurement for this learner, because its window still reaches back to the first operand for up to nine steps after the reset.

## Predictions. Each is judged on the three seeds; "holds" means in at least two

- **P-PR.1** Of the twins that fail, pooled over the four pairs, at least 70% have a wrong intermediate in at least one member. Confidence about 70%.
- **P-PR.2** With the true intermediate supplied, the ending is right on at least 85% of twin items in which the second operator is still in the window at that point (three letters or fewer after it). Confidence about 65%.
- **P-PR.3** The same rate is at least 10 points lower where the second operator has left the window (four letters or more). Held at about 50%.
- **P-PR.4** Where the written intermediate is wrong and readable, the ending is the correct second step of that wrong intermediate in at least 60% of twin items. Confidence about 55%.
- **P-PR.5** The first operand reads out (both digits) on at least 50% of twin items. Confidence about 60%. The shuffled-label control stays under 5% and the untrained net under 10%; if either does not, the readout says nothing and P-PR.5 and P-PR.6 are void.
- **P-PR.6** Among twin items, the intermediate is right at least 20 points more often where the first operand reads out than where it does not. Confidence about 60%.
- **P-PR.7** After reset (b), twins both correct fall to at most half of the ordinary score, pooled over the four pairs. Confidence about 55%.

## Readings, fixed now

- **P-PR.1 and P-PR.2 hold:** the second step is sound and the loss is at the first step, which is the one that needs the hidden operand. The useful intervention point is the intermediate.
- **P-PR.1 misses:** a large share of failures happen after a correct intermediate. Then the hidden operand is not the main obstacle and the second step of an untaught operator pair is.
- **P-PR.5 holds while plus-minus twins stay low:** the operand is kept and not used. The limit is computation on an untaught pair from memory, not retention.
- **P-PR.5 misses with its controls in order:** the operand is not linearly present at `=`. A linear readout is a lower bound, so this says "not shown to be kept", which points at retention without proving it.
- **P-PR.7 holds:** the learner relies on state built before the last ten bytes, not on re-reading the operand through the window.

## What these probes cannot say

- They are run on end-of-life weights of one architecture under backprop. They say nothing about the working rule.
- Max and min pairs have a copy shortcut near 50% per item (erratum in review 9.87). Results are reported per operator pair and the plus-minus pairs carry the reading.
- A supplied intermediate is not the registered autonomous score. It is reported separately and never added to it.
