# MiniFly re-entry continuation — causal memory and training dose

**2026-11-05 · Completed paired development study.**

The R8 learner and the longer-credit R32-BP diagnostic trained on the same one-world stream, through all five A doses. Each received 65,536 distinct addition records and 4,096 fact presentations, totaling 967,507 symbols. The earlier core, data generator, uniform next-symbol objective, and engineering receipts were preserved. BPTT32 is a diagnostic with a different credit horizon.

Both probe modes freeze the slow parameters and optimizer and discard all query state afterward. Frozen probes retain the original fixed matrix. Causal probes apply the normal matrix writer to observed prompt bytes and the model’s own emitted symbols. The true-prefix diagnostic has a separate disposable state. The probe checks confirm frozen-output parity with the original code, unchanged training digests, and identical autonomous output when the evaluator’s gold answer is altered.

At the 65,536-record endpoint, all four arm/probe combinations scored zero exact answers on the taught and held panels. The longer-credit causal probe produced one exact held answer at 32,768 records. Causal memory greatly improved EOS under the correct prefix, while later digits remained weak. The tested longer credit horizon did not restore acquisition or held arithmetic on this world.

## Learning curve

Cells give exact autonomous answers including EOS. The two numbers in each cell are frozen / causal. The same 128 taught and 512 never-taught prompts were scored at each dose.

| Addition records | R8 taught | R8 held | R32-BP taught | R32-BP held |
|---:|---:|---:|---:|---:|
| 4,096 | 0/128 / 0/128 | 0/512 / 0/512 | 0/128 / 0/128 | 0/512 / 0/512 |
| 8,192 | 0/128 / 0/128 | 0/512 / 0/512 | 0/128 / 0/128 | 0/512 / 0/512 |
| 16,384 | 0/128 / 0/128 | 0/512 / 0/512 | 0/128 / 0/128 | 0/512 / 0/512 |
| 32,768 | 0/128 / 0/128 | 0/512 / 0/512 | 0/128 / 0/128 | 0/512 / 1/512 |
| 65,536 | 0/128 / 0/128 | 0/512 / 0/512 | 0/128 / 0/128 | 0/512 / 0/512 |

## Where held answers break at the final dose

| Arm and probe | Exact | First byte correct | Mean first-byte rank | Later answer bits/byte | EOS top-1 at true boundary | Correct payload, wrong termination |
|---|---:|---:|---:|---:|---:|---:|
| R8 frozen | 0/512 | 117/512 | 4.56 | 3.531 | 4/512 | 0 |
| R8 causal | 0/512 | 127/512 | 4.33 | 3.691 | 450/512 | 0 |
| R32-BP frozen | 0/512 | 117/512 | 4.62 | 3.521 | 4/512 | 0 |
| R32-BP causal | 0/512 | 128/512 | 4.48 | 3.721 | 398/512 | 0 |

The evaluator’s width-conditioned first-byte frequency baseline gets **128/512** at the final dose. This baseline measures how far answer priors can go without tracking the specific operands.

True-prefix losses by operand bytes, delimiters, answer bytes, and EOS are stored separately in SUMMARY.json. A correct EOS under the true prefix uses the correct prior bytes. It is distinct from stopping at the correct place during free generation.

## Repeated-record acquisition

The independent acquisition fixture cycles through four addition and four fact records, with a seeded reshuffle each round. It uses a new learner with the same birth and no state shared with the main history.

| Presentations per record | R8 exact frozen / causal | R32-BP exact frozen / causal |
|---:|---:|---:|
| 1 | 0/8 / 0/8 | 0/8 / 0/8 |
| 4 | 0/8 / 0/8 | 0/8 / 0/8 |
| 16 | 0/8 / 0/8 | 0/8 / 0/8 |
| 64 | 0/8 / 0/8 | 0/8 / 0/8 |
| 256 | 1/8 / 0/8 | 1/8 / 0/8 |

## Complete input path

Across 128 held one-digit carry pairs, the audit compares sampled sensory inputs, the byte-dependent writer target, workspace, matrix, read, and output logits. It records both-correct answer pairs on the final checkpoints.

| Model and probe | Distinct sampled KCs at any prompt step | Distinct workspace at answer onset | Distinct memory read at onset | Both answers exact |
|---|---:|---:|---:|---:|
| birth frozen | 128/128 | 128/128 | 0/128 | — |
| birth causal | 128/128 | 128/128 | 128/128 | — |
| R8 frozen | 128/128 | 128/128 | 128/128 | 0/128 |
| R8 causal | 128/128 | 128/128 | 128/128 | 0/128 |
| R32-BP frozen | 128/128 | 128/128 | 128/128 | 0/128 |
| R32-BP causal | 128/128 | 128/128 | 128/128 | 0/128 |

Detectable state differences establish that the edits reach this learner on these probes. The audit leaves open how accessible the required arithmetic transformation is to its output head.

## Execution scope and reproduction

The paired curves share one development world and birth. The many held questions give item-level behavior within that world. Independent replication across worlds remains open.

BPTT checkpoints use exact record boundaries for scoring. If a dose stops inside a 32-symbol block, its uncommitted prefix is replayed on a copy for evaluation and later replayed with the complete block for training. Both arms update only on the locked block boundaries. Save files include the restart state and exact-dose reference copy when these differ.

The original learner directory sits alongside this continuation. Use the environment from its engineering report. Run `run.py check`, then `run.py acquisition --arm R8` and `run.py curve --arm R8`. Repeat with `R32-BP`. Run `response_weight.py` for the independent objective pilot. Run `input_audit.py` and `report.py` to build the tables from the receipts. [Dose curve](DOSE_CURVE.png)

The experiment covers acquisition and held behavior during A. The B/A′ retention campaign, full-dose response weighting, and remaining route/credit arms are separate follow-ups.

## Response-weight intervention

A separate R8 pilot gives four times the loss weight to response bytes and EOS on the same repeated-record history. The observed equals sign opens the weighted span. Each 32-symbol block is normalized to total weight 32 before the unchanged optimizer update. All observations still advance the sensor and fast memory. Multiplier one was bit-identical to the core on a 64-symbol test.

| R8 at 256 presentations per record | Exact | First byte | Later answer bits/byte | EOS top-1 at true boundary |
|---|---:|---:|---:|---:|
| Uniform, frozen | 1/8 | 6/8 | 2.518 | 5/8 |
| Uniform, causal | 0/8 | 7/8 | 2.243 | 4/8 |
| 4× response, frozen | 1/8 | 6/8 | 1.738 | 7/8 |
| 4× response, causal | 1/8 | 7/8 | 1.477 | 7/8 |

The response and other-site gradients had negative cosine in 689/856 mixed update blocks along the weighted trajectory (mean cosine -0.165). This diagnostic concerns that small repeated history. The objective pilot has no held generalization test.

Total recorded worker CPU was 30.3 minutes across the two curves and two acquisition histories, including their probe evaluations.
