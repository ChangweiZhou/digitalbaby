# Step 2: other gates, each as one more family of parts born at nothing (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and "继续大胆尝试。" (the coordinator's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." and "Go on trying boldly."). **Source.** `claude/nano-layers-8mmhfm` at 248dac7 (`layers.py` as checked there, with the arms of this step added and nothing else changed). Written before any run of this step. Development level; no checker gate, seal or review is claimed.

## 1. The evidence this follows

`PLAN_1.md`, section 6: the second cell's parts gain more when the level of counts is left out of their index (97.4 against 96.7 on sums seed 1; 0.0292 against 0.0239 nats on three worn books), with 2.5% to 5% of the parts. On sums the gain is the place in the line, which one byte tells; on prose it is spread over the six delays.

## 2. The one question, and the arms

**Does any other fact of the moment, used as a gate in the same form, add beside the delays' parts on both texts?** Four gates, each one family of parts born at nothing and taught by the cell's own mismatch with the system's own step:

- *pair*: the two bytes at delays 1 and 2 together (one value among S times S);
- *own*: no gate at all, but a part that the sending cell owns: one number for each cell, added to the trust of its kind and level;
- *sure*: the first cell's largest probability, in ten equal bins;
- *class* (systems with classes: sums here): the class, at the class kind whose mean trust is largest at the last finding, of the bytes that kind reads (no class: its own value). Its parts begin again at nothing at every finding of classes.

Each gate is run alone and beside the delays' parts, in two forms: with the level of counts in the index (beside the coordinator's second cell) and without it ("-nl", beside the no-level cell of step 1). Also: the no-level parts of delays 1 to 3 only (owed from step 1); all gates together beside the no-level cell; and the two forms of the delays' parts in one cell (coarse and fine together). Twenty further cells on sums, sixteen on prose, in one run each. Systems, material and measures as in `PLAN_1.md`, section 3 (sums seed 1; the life with the three worn books).

## 3. Hypothesis, what refutes it, predictions (mine, before the runs)

*Hypothesis.* At least one of the four gates adds beside the no-level delays' parts on both texts: at least 1 point on sums (seed 1; under that is not read) and at least 0.003 nats on the mean of the three books. *Refuted if* none does on both. If it is refuted on sums only because the single seed cannot show a point, that is said.

| | Statement | Chance |
|---|---|---|
| G1 | pair alone (no-level) gives at least 0.015 nats on prose | 0.50 |
| G2 | pair beside the no-level cell adds at least 0.003 nats on prose | 0.60 |
| G3 | pair beside the no-level cell changes sums by under 1 point | 0.70 |
| G4 | own beside the no-level cell adds at least 0.003 nats on prose | 0.40 |
| G5 | own beside the no-level cell adds at least 1 point on sums | 0.20 |
| G6 | sure alone gives at least 0.003 nats on prose | 0.50 |
| G7 | sure beside the no-level cell adds at least 0.002 nats on prose | 0.40 |
| G8 | sure changes sums by under 1 point, alone and beside | 0.70 |
| G9 | class beside the no-level cell adds at least 1 point on sums | 0.15 |
| G10 | the two forms of the delays' parts in one cell are within 0.001 nats of the no-level cell on prose and within 1 point on sums | 0.55 |
| G11 | the no-level parts of delays 1-3 give at least three quarters of the no-level cell's gain on prose (the hypothesis of step 1) | 0.50 |
| G12 | the hypothesis of this step holds | 0.20 |

## 4. Size, cost, what is not run

Parts held, sums / prose: pair 2.26 million / 1.69 million with the level, 56,448 / 42,250 without; own: one for each cell (1.2 million / 2.4 million); sure 115,200 / 4,000 with the level, 2,880 / 100 without; class 2.27 million with, 56,736 without. Numbers used are reported. About five CPU-minutes on sums, one on prose. *Not run:* the reported seeds, the unopened book, cells on cells, search, other pairs of delays, a gate from the levels of other senders (it is among the gates drawn in step 4).

## 5. Outcome (entered 20 November after the runs; `res/sums_armsstep2_seed1.json`, `res/life_armsstep2_booksworn.json`, `report_2.txt`)

Sums: seed 1, 40,000 lines, with classes, 1,231,004 cells, % of never-shown sums after a read line, mean of the last three readings; 464 CPU-seconds for twenty further cells. Prose: T2, 2,419,727 cells, mean gain over the first cell on the three worn books, nats; 44 CPU-seconds. First cell and the coordinator's second cell in the same runs: 92.7 and 96.7 on sums; 1.4802, 1.3162, 1.4109 and 1.4621, 1.2932, 1.3802 on prose.

| Further cell | Sums, % (gain over first) | parts used | Prose, gain over first (and over the single second cell) | parts used |
|---|---|---|---|---|
| second (coordinator's) | 96.7 (+4.1) | 702,110 | 0.0239 | 83,709 |
| second-nl (delays, no level) | 97.4 (+4.8) | 33,547 | 0.0292 (+0.0053) | 3,857 |
| no-level, delays 1-3 only | 97.6 (+4.9) | 11,193 | 0.0231 | 1,922 |
| pair alone, with level / without | 96.5 / 97.3 | 557,478 / 43,625 | 0.0072 / 0.0125 | 105,768 / 13,749 |
| own alone | 93.5 (+0.8) | 1,207,365 | 0.0065 | 564,482 |
| sure alone, with level / without | 95.5 / 96.3 | 47,470 / 2,824 | 0.0071 / 0.0056 | 2,636 / 100 |
| class alone, with / without (sums) | 92.6 / 93.3 | 27,787 / 3,726 | | |
| second + pair, + own, + sure, + class | 97.2, 96.4, 96.6, 97.1 | | 0.0260, 0.0261, 0.0299 | |
| nl + pair-nl | 97.7 | 77,172 | 0.0329 | 17,606 |
| nl + own | 97.6 | 1,240,912 | 0.0321 | 568,339 |
| nl + sure-nl | 97.6 | 36,371 | 0.0329 | 3,957 |
| nl + class-nl (sums) | 96.9 | 37,273 | | |
| **nl + all gates** | **97.8 (+5.1)** | 1,291,087 | **0.0390 (+0.0151)** | 582,188 |
| second + nl (fine and coarse parts of the delays) | 96.8 | 735,657 | 0.0364 (+0.0125) | 87,566 |

By place on sums (hundreds, tens, units; last reading, after a read line): first 98.3, 92.3, 100; second 98.0, 97.2, 100; second-nl 97.9, 98.0, 100; nl + all 98.1, 98.1, 100. What is kept on prose (test text right after its stream, then after the three books): nl + all 1.4693 and 1.5019; first 1.4973 and 1.5067.

*Predictions.* G1 failed (pair alone without level 0.0125, not 0.015). G2 held (+0.0037 beside the no-level cell). G3 held (+0.3 on sums). G4 failed narrowly (own adds 0.0029, not 0.003). G5 failed (+0.2). G6 held (0.0071 and 0.0056). G7 held (+0.0037). G8 failed: alone the first cell's sureness gives 2.8 and 3.7 points on sums (beside the delays' parts it changes nothing). G9 failed (the class gate takes 0.5 off). G10 failed in the direction of a gain: fine and coarse parts together are 0.0072 nats better than the coarse alone on prose (on sums 0.6 worse, within the noise). G11 held (delays 1-3 without the level give 0.79 of the no-level cell's gain on prose, and all of it on sums: the hypothesis of step 1 holds in the direct arm). G12, **the hypothesis of this step, is refuted as stated**: no gate adds a point on sums. On prose all three add (0.0029 to 0.0037 each, and 0.0099 together: nearly their sum).

Readings (mine).
1. **On prose the gates add and add nearly independently**: pair, own and sure each add about 0.003 to 0.004 nats beside the delays' parts, 0.0099 together, on each of the three books. The cell with all of them is 0.015 nats below the single second cell (0.0146, 0.0129, 0.0179 on the three books).
2. **On sums every arm that knows the place stands at 97.3 to 97.8, and the errors left are shared between the hundreds and the tens digit at about 98% each**, whatever the gate. The hundreds digit is not moved by any further cell (98.3 for the first cell, 97.9 to 98.1 above). My reading is that what is left on sums at 40,000 lines is in what the cells hold, not in the reading; one seed cannot show differences of half a point, and none of these is read.
3. **How sure the first cell is tells the place on sums** (2.8 to 3.7 points alone, with 2,824 parts): a fact about the answer stands in for a fact about the text. It is the one gate here that reads nothing of the input.
4. **Coarse and fine parts together beat either on prose**: the form "a part for all levels, plus a part for each level" is the same backing-off as the kind's own trust plus a situation's part (`nano/table/PLAN_14.md`, addition 1), inside the gates.
5. The class as a gate fails here. It begins again at every finding of classes, and the last finding leaves it 8,000 lines.
