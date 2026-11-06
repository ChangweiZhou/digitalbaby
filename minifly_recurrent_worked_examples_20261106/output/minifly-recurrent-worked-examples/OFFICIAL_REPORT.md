# MiniFly recurrent state and worked examples results

**The campaign stopped at the presentation-transfer gate.** No model and teaching condition met both thresholds in both blocks under both opcode laws.

All 24 histories completed the locked A dose, totaling **196,608 updates and 24,182,784 delivered symbols**. This is a two-block development comparison of gradient-trained references. It isolates the added effects of carried recurrent state and worked-example supervision on the same raw-byte task.

## Acquisition and presentation transfer

Taught accuracy scores literal training prompts. A held item counts only when both fresh edge-row orders give the exact location byte followed by native EOS. The thresholds are 52/64 taught and 48/64 held pairs per history. All histories continue to 8,192 pairs regardless of their early scores.

| Model | Teaching | Taught at 2,048 | Held at 2,048 | Taught at 8,192 | Held at 8,192 | Final taught by history | Final held by history |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| window | answer | 188/256 | 124/256 | 250/256 | 126/256 | 64 · 63 · 60 · 63 | 29 · 35 · 31 · 31 |
| window | shuffled | 163/256 | 110/256 | 247/256 | 126/256 | 61 · 63 · 61 · 62 | 28 · 30 · 34 · 34 |
| window | worked | 174/256 | 115/256 | 247/256 | 121/256 | 59 · 63 · 64 · 61 | 27 · 29 · 30 · 35 |
| recurrent | answer | 134/256 | 83/256 | 243/256 | 113/256 | 62 · 61 · 60 · 60 | 27 · 26 · 28 · 32 |
| recurrent | shuffled | 116/256 | 73/256 | 234/256 | 125/256 | 62 · 62 · 54 · 56 | 35 · 29 · 32 · 29 |
| recurrent | worked | 112/256 | 73/256 | 235/256 | 108/256 | 58 · 61 · 57 · 59 | 26 · 32 · 24 · 26 |

The per-history columns use block 93157 laws 0 and 1, then block 104729 laws 0 and 1. At the final checkpoint, **24/24** histories cleared the taught floor and **0/24** cleared the held floor. The best held score was **35/64**. The final probes emitted EOS on 4,608/4,608 outputs and hit the cap without EOS on 0. First-byte counts and full native sequences are in the receipts.

## Matched contrasts

These differences count both-correct held pairs at 8,192, with each condition and control compared under the same birth and law. A practical gain requires eight additional pairs in every history in a contrast.

| Comparison | Difference in held pairs by history |
| --- | --- |
| Recurrent minus ordered window with answer teaching | -2 · -9 · -3 · +1 |
| Recurrent minus ordered window with shuffled teaching | +7 · -1 · -2 · -5 |
| Recurrent minus ordered window with worked teaching | -1 · +3 · -6 · -9 |
| Worked minus answer with window model | -2 · -6 · -1 · +4 |
| Worked minus shuffled with window model | -1 · -1 · -4 · +1 |
| Worked minus answer with recurrent model | -1 · +6 · -4 · -6 |
| Worked minus shuffled with recurrent model | -9 · +3 · -8 · -3 |

The locked stopping rule kept model evaluation on unseen graph families, the inconsistent-law control and B continuation out of this run. The exhaustive graph oracle and adaptive baseline were qualified independently of training. The earlier B rehearsal remains an engineering receipt.

The highest evidence available from the presentation endpoint is E2 on the prescribed row-order change. A canonical lookup can recover the taught semantic key. Reliable rule reuse would require a separate positive E3 result with the prior-teaching control.

## Reproduction and cost

The main workers used 20.1 aggregate CPU minutes. The highest recorded single-worker RSS was 234 MiB. These measurements cover the 12 main workers and their native A probes. The run lock records the feedback-stream and probe hashes, sources and thresholds. The 48 dose receipts and matching checkpoints are listed by SHA-256.

[Locked design](../minifly-whole-repo-audit/MINIFLY_RECURRENT_WORKED_EXAMPLES_EXPERIMENT_DESIGN_20261106.md) · [Three-cycle qualification](CYCLES.md) · [Run lock](official/LOCK.json) · [Main summary](official/SUMMARY.json) · [Receipt and checkpoint hashes](official/ARTIFACT_HASHES.json) · [Main runner](official_worked.py) · [Conditional gates](conditional_worked.py)
