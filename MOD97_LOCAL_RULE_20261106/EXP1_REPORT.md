# Experiment 1: development-versus-life transfer on modular arithmetic

**6 November 2026 · 171 runs · three seeds · pre-stated predictions and decision rule in the chat record before launch**

**Question.** Does a shared computation fitted during a development phase on one task, then handed to a local rule, let the learner acquire a related new task from fewer examples and reuse it on never-reinforced inputs, with the gain coming from what was learned rather than from pretraining as such?

**Answer.** Yes for the same operation with a new modulus, by the pre-stated rule: an aligned birth beats both controls by 13–39 points in every seed at 25% taught, and at 10% taught it beats backpropagation from scratch. Partially for a different operation: an aligned warm start beats the random birth by 4–20 points at 25% (below the 10-point rule in two seeds) and beats backprop from scratch at 10% in two of three seeds. The gain is from the learned representation: a frozen random trunk gives 7–28%, a shuffled birth is unstable and worse than no birth. The three-factor scalar-reward rule learned nothing at this budget.

## Setup

Same network and harness as the earlier rounds (`mod97_local.py`), new driver `exp1_transfer.py`.

- **Development.** Backprop on mod-97 addition, Codex's taught split (4,704 prompts), AdamW lr 0.001, decoupled decay 0.3, 12,288 updates. A matched **shuffled** birth gets the same budget with answer bytes permuted across taught prompts (98.9% of rows changed, answer marginals preserved). Aligned births reached 95.2–99.7% sealed held on addition; shuffled births 0.8–1.0%.
- **Life tasks.** `sub97`: `a-b=c` mod 97 with a new operator byte. `add89`: `a+b=c` mod 89. Fresh splits: a fraction of unordered pairs taught (10%, 25%, 50%), both orders together, the rest halved into monitoring and sealed panels. 12,288 life updates, batch 64, scored every 512.
- **Life rule.** Sign-symmetric feedback (each layer receives the error through the sign of its own forward weights), AdamW lr 0.001, decay 1.0, unless stated.

| Arm | Birth | Plastic in life | Rule |
|---|---|---|---|
| A1 | aligned | second layer + readout | sign feedback |
| A1bp | aligned | second layer + readout | backprop (diagnostic: features or rule?) |
| A2 | aligned | all layers | sign feedback |
| B | shuffled | second layer + readout | sign feedback |
| B2 | shuffled | all layers | sign feedback (fair "any pretraining" control) |
| C | random | all layers | sign feedback |
| C1 | random | second layer + readout | sign feedback (reservoir baseline for A1) |
| D | random | all layers | backprop, decay 0.3 (ceiling) |
| E | aligned | second layer + readout | three-factor: scalar reward per record, node perturbation + REINFORCE readout, decay 1.0 |
| E0 | aligned | second layer + readout | three-factor, no decay |

Qualification (`exp1/QUALIFY.json`): splits partition all pairs, swapped forms of sealed pairs are sealed, every rendered answer passes the oracle, frozen trunks are bit-identical after updates in every frozen arm, three-factor updates touch only the plastic layers.

## Results: sealed held exact, final, three seeds (percent)

Shortcut ceilings on the sealed panels: lookup 0, frequency ≤1.1, nearest taught pair 10.3 / 7.4 / 1.4 (sub97 at 10 / 25 / 50%) and 8.9 / 6.1 / 0.7 (add89). Two structural half-ceilings: for sub97 the plain difference `|a-b|` is right on 50.4% (the no-wrap half); for add89 the mod-97 answer is right on about 50% (sums below 89). The **wrap subset** (a<b for subtraction, a+b≥89 for mod-89) is where those shortcuts fail, and is reported beside the total.

### Subtraction mod 97

| Arm | 10% taught | 25% taught | 50% taught | Wrap subset at 25% |
|---|---|---|---|---|
| A1 aligned, frozen trunk, sign | 25.0 · 48.3 · 13.5 | 26.9 · 52.2 · 13.1 | 30.1 · 53.3 · 13.9 | 14.7 · 40.3 · 5.3 |
| A1bp aligned, frozen trunk, backprop top | 60.6 · 67.9 · 38.7 | 71.5 · 82.1 · 53.4 | 75.5 · 83.4 · 62.9 | 67.5 · 79.0 · 43.1 |
| **A2 aligned, warm start, sign** | **84.1 · 91.3 · 70.7** | **82.9 · 94.0 · 80.2** | 83.4 · 94.1 · 79.8 | 78.5 · 93.2 · 75.1 |
| B shuffled, frozen trunk, sign | 4.5 · 1.1 · 17.0 | 6.0 · 0.8 · 24.2 | 5.6 · 0.8 · 25.1 | 3.9 · 0.6 · 14.6 |
| B2 shuffled, warm start, sign | 27.6 · 1.1 · 51.7 | 0.9 · 0.8 · 75.5 | 0.8 · 1.4 · 81.2 | 0.7 · 0.8 · 71.7 |
| C random, sign | 63.0 · 43.1 · 77.1 | 78.0 · 74.0 · 76.3 | 74.3 · 69.0 · 73.0 | 71.6 · 66.1 · 72.4 |
| C1 random, frozen trunk, sign | 17.6 · 7.3 · 11.1 | 27.7 · 9.9 · 15.8 | 27.2 · 12.3 · 15.7 | 22.8 · 5.2 · 10.6 |
| D random, backprop (ceiling) | 70.7 · 61.9 · 59.5 (peak 82 · 74 · 70) | 94.8 · 95.8 · 93.8 | 95.4 · 94.8 · 97.8 | 94.7 · 95.2 · 93.0 |
| E three-factor, decay 1.0 | 1.6 · 2.1 · 2.4 | 2.0 · 2.6 · 2.7 | 1.3 · 2.4 · 2.9 | 1.4 · 1.9 · 1.4 |
| E0 three-factor, no decay | 3.7 · 4.1 · 3.3 | 5.1 · 5.4 · 3.4 | 4.0 · 5.1 · 4.1 | 2.7 · 3.6 · 2.3 |

### Addition mod 89 (aligned births start at 47–51% from inherited mod-97 answers; shuffled and random births at 0–1%)

| Arm | 10% taught | 25% taught | 50% taught | Wrap subset at 25% |
|---|---|---|---|---|
| **A1 aligned, frozen trunk, sign** | **71.4 · 87.5 · 72.6** | **77.3 · 91.5 · 79.7** | 80.5 · 89.2 · 80.6 | 73.0 · 92.3 · 76.1 |
| A1bp aligned, frozen trunk, backprop top | 88.9 · 92.7 · 89.7 | 89.6 · 95.9 · 93.4 | 93.1 · 97.5 · 96.3 | 88.0 · 95.2 · 90.8 |
| **A2 aligned, warm start, sign** | **69.8 · 87.7 · 71.1** | **82.8 · 89.7 · 80.2** | 85.1 · 89.3 · 82.7 | 81.0 · 89.0 · 81.1 |
| B shuffled, frozen trunk, sign | 4.4 · 0.8 · 13.5 | 5.9 · 1.1 · 18.5 | 8.0 · 1.0 · 20.9 | 5.5 · 1.5 · 16.8 |
| B2 shuffled, warm start, sign | 29.8 · 0.9 · 60.6 | 49.7 · 0.9 · 63.0 | 52.5 · 1.1 · 80.7 | 46.6 · 1.0 · 68.3 |
| C random, sign | 41.9 · 47.3 · 54.0 | 64.6 · 52.6 · 59.4 | 61.1 · 11.9 · 79.3 | 59.6 · 52.8 · 67.8 |
| C1 random, frozen trunk, sign | 6.7 · 7.8 · 14.1 | 9.7 · 11.1 · 18.6 | 12.2 · 13.7 · 20.3 | 9.8 · 11.8 · 22.5 |
| D random, backprop (ceiling) | 29.2 · 40.7 · 30.9 (peak 44 · 59 · 49) | 92.0 · 94.9 · 94.4 | 94.9 · 97.1 · 98.6 | 90.2 · 95.5 · 93.9 |
| E three-factor, decay 1.0 | 6.2 · 3.3 · 7.5 | 6.0 · 5.1 · 6.6 | 5.0 · 4.0 · 5.6 | 3.6 · 4.7 · 5.0 |
| E0 three-factor, no decay | 34.2 · 41.6 · 38.3 | 32.9 · 37.6 · 37.2 | 36.8 · 33.7 · 38.9 | 8.3 · 13.5 · 8.2 |

Taught accuracy, both-order scores, peak and monitoring-best values are in `exp1/SUMMARY.md`; every checkpoint is in `exp1/life_*.json`.

## Scorecard against the pre-stated predictions

| Prediction | Outcome |
|---|---|
| A reaches ≥80% at 25% taught and ≥60% at 10% | **Met** by A2 on both tasks and by A1 on add89. Not met by A1 on subtraction (13–53%). |
| A beats C by ≥15 points at every fraction | **Met on add89** at 10% and 25% (13–39 points), mostly at 50%. **Not met on subtraction**: A2 over C is +4, +20, +4 at 25% and +21, +48, −6 at 10%. |
| B matches C (pretraining as such gives nothing) | **Direction confirmed, magnitude wrong.** Shuffled births are *worse* than no birth: B 1–25%, and the fair warm-start control B2 is unstable (one seed dead at 1% in every cell, the others 28–81%). Pretraining as such does not help; a wrong birth can hurt. |
| D within 5 points of A at 25–50% | **Not met on subtraction** (D leads A2 by 11–15 points). **Met on add89** at 50% within 10. **Reversed at 10% on both tasks**: backprop from scratch peaks at 44–82% and decays to 29–71% as it memorizes, while an aligned birth with the local rule holds 70–91%. |
| E (scalar reward) lands at 20–50% at 50% taught | **Not met.** E learned nothing on subtraction (1–5%, taught 2–6%). On add89 E0 ended at 33–42%, below its 47–51% inherited start, with the wrap subset at 7–15%: it retained part of the inherited answers and learned no new ones. With decay 1.0 it also destroyed the inherited knowledge. |
| A1 ≈ A2 with a small edge for A2 | **Met on add89**, **failed on subtraction** where the frozen trunk collapses A1 to 13–53%. |

## What the diagnostics say about the subtraction failure

- **Not swap-invariant features.** The hypothesis that the addition trunk collapsed operand order was tested directly (`exp1/SWAP_INVARIANCE.json`): cosine between trunk features for (a,b) and (b,a) is 0.27–0.44 in aligned births and 0.31–0.40 in random ones, both well below the 0.57–0.69 for (a,b) versus (a,c). The trunk keeps operand order.
- **Mostly the rule, partly the features.** With backprop on top of the same frozen aligned trunk (A1bp), subtraction reaches 72–83% in two seeds and 53–63% in the third, against 13–53% with the sign rule and 94–98% for full backprop. So the frozen features support subtraction reasonably; the local rule cannot fit a two-layer head onto fixed features for a harder mapping, and the warm start (A2) is what makes the local rule work.
- **Development builds reusable features.** A frozen random trunk with the same local head (C1) gives 7–28%. A frozen aligned trunk gives 71–92% on add89 and 13–53% on subtraction with the same rule. That gap is the representation.

## What this settles

1. **The development-versus-life split works, for the local rule we have, when life is a warm start.** An aligned birth plus sign-symmetric feedback reaches 80–94% on a new task at a quarter of the pairs, 70–91% at a tenth, and at a tenth it beats backprop trained from scratch on both tasks. Sample efficiency is where development pays.
2. **The gain is from what was learned.** Shuffled births are worse than no birth; random frozen trunks are near the floor. There is no "pretraining as such" effect to explain the result away.
3. **Freezing the trunk is safe for the same operation and costly for a different one under the local rule.** Keep the lower layers plastic in life, at a low rate, rather than frozen.
4. **A scalar reward is not enough to learn these tasks at this budget.** Node perturbation on 128 hidden units plus a REINFORCE readout, 12,288 updates, learned nothing. The life rule needs a vector error signal at the output, as the sign rule has. In biological terms, dopamine alone would not do this; the target or error must reach the output population.
5. **Decision by the pre-stated rule:** advance on add89 (A1 and A2 beat B and C by ≥10 points in all seeds at 25%); not met on subtraction for A2 (margins +4, +20, +4 over C). The rule said: if A matches C, pretrain on a broader family. A2 does not match C, it beats it modestly and robustly in 8 of 9 cells, but below the margin. The next development phase should be a mixed family (several moduli, addition and subtraction) before rung 3.

## Evidence level and limits

Development result, three seeds, one harness, no independent confirmation. Both life tasks share the two-digit byte vocabulary with the development task; transfer to an unrelated vocabulary was not tested. The add89 task inherits half its answers from the development task, so its wrap subset is the honest score there (73–92% for aligned births at 25%). The three-factor result is one implementation at one setting and bounds that implementation, not every scalar-reward rule.

## Reproduction

```sh
python3 exp1_transfer.py qualify
for s in 73129 93683 116089; do for b in aligned shuffled; do python3 exp1_transfer.py develop $s $b; done; done
for task in sub97 add89; do for f in 0.1 0.25 0.5; do for arm in A1 A1bp A2 B B2 C C1 D E E0; do for s in 73129 93683 116089; do python3 exp1_transfer.py life $arm $s $task $f; done; done; done; done
python3 summarize_exp1.py
```
