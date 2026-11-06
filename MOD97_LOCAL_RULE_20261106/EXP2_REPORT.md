# Experiment 2: mixed-family development, then transfer to a held-out modulus and to decimal addition

**6 November 2026 · 99 runs · three seeds · predictions P1–P5 stated in the chat record before launch**

**Question.** Does breadth of development (six tasks: addition and subtraction mod 83, 89, 97) buy transfer to a held-out modulus and to decimal addition, beyond what single-task development (addition mod 97) buys?

**Answer.** No, at this capacity. The single-task birth beats the mixed birth on both mod-73 tasks by 2–21 points, and the two are within a few points on decimal addition. The mixed birth is capacity-limited: it reaches only 80–97% on its own six development tasks where the single birth reaches 96–100% on its one. Breadth cost representation quality, and that cost showed up in life. The decision rule (P2 and P3 in all seeds) is not met. Two predictions held: shuffled births are useless or worse, and at 10% taught every developed birth beats backprop from scratch on the modular tasks by 20–40 points.

## Setup

Network: the harness network with a 10-byte window (62,097 parameters; the 8-byte window of earlier rounds cannot see the task code or the first operand's units digit when emitting a third answer digit). Every record carries a one-byte task code before the operands: `p`, `q`, `r` for mod 83, 89, 97; `s` for the held-out mod 73; `d` for decimal. Decimal answers are three zero-padded digits (`d24+81=105`), the modular answers two.

| Birth | Development stream | Budget |
|---|---|---|
| mixed | add and sub × {83, 89, 97}, half of each task's unordered pairs | 24,576 updates, backprop, AdamW lr .001, decay 0.3 |
| single | add97 only, same code scheme | same |
| shuffled | the mixed stream with answers permuted within each task | same |

Life: 12,288 updates, sign-symmetric feedback, AdamW lr .001, decay 1.0, all layers plastic (the winning setting of Experiment 1). Arms M (mixed birth), S (single birth), H (shuffled birth), C (random birth), D (random birth, backprop, decay 0.3). Life tasks: `add73`, `sub73`, `dec`, at 10% and 25% of unordered pairs taught, both orders together, rest halved into monitoring and sealed panels.

Qualification (`exp2/QUALIFY.json`): splits partition all pairs; swapped forms of sealed pairs are sealed; every rendering passes the oracle; the task codes never collide; mod 73 and decimal never appear in any development stream; the shuffled stream preserves per-task answer marginals and changes 98.9% of rows; the 10-byte window's local gradients match autograd at the output layer.

Because the held-out tasks carry an unseen code byte, every birth starts life at 0% (Experiment 1's aligned birth started mod-89 addition at 50% from inherited answers). The inherited-answer shortcut is therefore absent here; the "mod-97 answer" ceiling in the shortcut ladder is what a learner would get by ignoring the code and treating the task as its development task, about 50% on the modular tasks.

## Development births: sealed held on each development task

| Birth | add83 | sub83 | add89 | sub89 | add97 | sub97 |
|---|---|---|---|---|---|---|
| mixed (3 seeds) | 80 · 91 · 84 | 86 · 93 · 86 | 95 · 96 · 96 | 95 · 97 · 97 | 84 · 91 · 92 | 90 · 95 · 93 |
| single | 0 | 0 | 0 | 0 | 99 · 100 · 96 | 1 |
| shuffled | ≈1 | ≈1 | ≈1 | ≈1 | ≈1 | ≈1 |

The mixed birth at 62,000 parameters does not master six tasks. Its mod-83 scores are the lowest and its mod-97 scores are 5–8 points below the single birth's.

## Life results: sealed held exact, final, three seeds (percent)

| Arm | add73, 10% | add73, 25% | sub73, 10% | sub73, 25% | dec, 10% | dec, 25% |
|---|---|---|---|---|---|---|
| M mixed birth, sign rule | 61 · 60 · 38 | 78 · 69 · 63 | 58 · 65 · 53 | 73 · 81 · 68 | 84 · 69 · 75 | 87 · 84 · 79 |
| **S single birth, sign rule** | **80 · 81 · 44** | **86 · 85 · 65** | **79 · 79 · 48** | **85 · 90 · 70** | 81 · 90 · 78 | 87 · 85 · 85 |
| H shuffled birth, sign rule | 1 · 1 · 1 | 1 · 1 · 1 | 1 · 1 · 1 | 1 · 1 · 1 | 50 · 57 · 58 | 77 · 72 · 72 |
| C random birth, sign rule | 72 · 50 · 64 | 19 · 32 · 50 | 61 · 59 · 61 | 55 · 1 · 62 | 77 · 86 · 89 | 77 · 80 · 87 |
| D random birth, backprop | 20 · 25 · 19 | 91 · 88 · 86 | 23 · 29 · 24 | 91 · 90 · 90 | 79 · 91 · 83 | 98 · 100 · 97 |

Wrap subsets at 25% (the half a code-ignoring learner cannot solve): add73 M 74 · 65 · 59, S 88 · 89 · 66, C 16 · 39 · 49; sub73 M 66 · 75 · 62, S 82 · 83 · 61, C 52 · 2 · 56. Shortcut ceilings: lookup 0, frequency ≤1.4%, nearest taught pair 6–10%, code-ignoring (development-task) answer ≈50% on modular tasks, digit-wise no-carry 29–30% on decimal. Peak, monitoring-best, carry-subset, taught and both-order values are in `exp2/SUMMARY.md`.

## Scorecard

| Prediction | Outcome |
|---|---|
| P1 Mixed beats single on sub73 by ≥10; matches on add73 | **Refuted in direction.** Single beats mixed on sub73 by 2–20 points at 10% (one seed −5) and 2–12 at 25%, and on add73 by 6–21 and 2–16. |
| P2 Mixed beats random by ≥15 on both mod-73 tasks, every seed | **Not met at 10%** (margins −12 to +9). Met at 25% (14–79) only because the random birth collapsed in some cells (18%, 1%), which is instability of the random-birth local rule, not strength of the mixed birth. |
| P3 Mixed beats random by ≥10 on decimal at 25%; ≥60% absolute | **Half met.** Absolute 79–87%. Margin over random +10, +4, −8: the random birth with the local rule already does 77–89% on decimal addition. |
| P4 Shuffled birth ≤ random birth | **Held.** Shuffled births are dead on the modular tasks (1%) and 15–31 points below random on decimal. |
| P5 Backprop from scratch beats mixed by ≤10 at 25%, loses at 10% | **Half met.** At 10% backprop from scratch gets 19–29% on the modular tasks against 38–81% for developed births. At 25% it leads the mixed birth by 10–23 points, more than predicted. |

## What this settles

1. **Breadth of development does not pay at this capacity.** A 62,000-parameter network trained on six modular tasks reaches 80–97% on them and transfers worse than a network trained on one task to 96–100%. The quality of the development representation matters more than its coverage. The obvious next knob is capacity: repeat the mixed birth at 2–4× the hidden width and see whether breadth pays once the development tasks are mastered.
2. **The single-task birth remains the live positive.** 80–90% on a held-out modulus and on a different operation at a quarter of the pairs, with a local life rule, in three of three seeds; 44–81% at a tenth.
3. **Decimal addition with carries is learnable by the local rule from scratch** (77–89% at a tenth of the pairs) and development adds little to it. The carry is not the hard part at this scale.
4. **Developed births beat backprop from scratch at low data, again.** At a tenth of the pairs, backprop memorizes the modular tasks (taught 100%, held 19–29%) while any developed birth transfers.
5. **The random-birth local rule is unstable.** Two of six modular cells at 25% collapsed to 1–19%, with the monitoring-best checkpoint no better. Early stopping does not help a run that never rose. A warm start from any sensible birth removes this failure mode.
6. **The unseen task code starts every birth at zero.** This made the comparison cleaner than Experiment 1's (no inherited half) and is the right design going forward.

## Evidence level and limits

Development result, three seeds, one harness, no independent confirmation. The 10-byte window differs from the earlier rounds' 8-byte window, so absolute scores are not comparable to Experiment 1 cell by cell; all arms within this experiment share it. The mixed birth's capacity limit is inferred from its development scores, not tested with a wider network. Life tasks share the two-digit byte vocabulary with development.

## Reproduction

```sh
python3 exp2_mixed.py qualify
for s in 73129 93683 116089; do for b in mixed single shuffled; do python3 exp2_mixed.py develop $s $b; done; done
for t in add73 sub73 dec; do for f in 0.1 0.25; do for arm in M S H C D; do for s in 73129 93683 116089; do python3 exp2_mixed.py life $arm $s $t $f; done; done; done; done
python3 summarize_exp2.py
```
