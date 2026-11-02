# V178 three-cycle result

The original dense V164 reference reproduced. The first failure occurred in the sparse package, before introducing a trainable gate. The prescribed extension improved it but did not pass; downstream stages were not executed.

| Checkpoint | Training bits/target | Calibration | Held test |
|---|---:|---:|---:|
| A epoch 10 | 0.942382 | 1.145648 | 1.1516293436841034 |
| B epoch 10 | 2.440248 | 2.478026 | unopened |
| B_EXTENSION epoch 20 | 1.775671 | 1.839156 | 1.9094815084094903 |

Lower is better; an uninformed eight-symbol predictor costs 3 bits/target. These are delayed-copy scores, not addition accuracy or complete own-byte/EOS performance.

The dense test differs from the historical 1.151632095 by −0.000002751 bits. This is strict numerical reproduction under the predeclared tolerance, with recovered source hashes and forward/gradient/update qualification. It is not bitwise identity across historical GPU and current CPU environments.

The sparse epoch-10 calibration loss 2.478026 fails the 1.50 gate. Its same-state, same-optimizer extension reached 1.839156 at epoch 20 (test 1.909482), still above the gate. Its original ten-epoch verdict remains **SPARSE_PACKAGE_FAIL_10**; the extension is **UNRESOLVED_OR_ARCHITECTURE_LIMITED_AT_20**. There was no reset, replacement initialization, or rescue of the ladder. C and the fresh guard remain untested.

## What the diagnostics support

The frozen-state linear probe converged in both executed endpoint diagnoses. Dense probe test loss was 0.871584 (78.25% token accuracy); sparse-at-20 probe test loss was 1.662473 (57.84%). A better fitted head recovers some information from sparse states, but it does not erase the representation gap. Probe answers are evaluator outputs, not model outputs, and probe fitting cost is separate.

Both endpoints have a slightly negative measured train-only Lyapunov estimate (dense −0.02439; sparse −0.02905). The sampled eight-step maximum singular values are 7.79 and 8.11. This argues against diagnosing the result from a single global “vanishing” or “exploding” number. These summaries do not establish causation or describe every direction/sequence. Sparse recurrence-off and cross-edge-off lesions raise calibration loss to 3.042 and 3.068; the trained model uses its recurrent and inter-module paths. Lesions also shift the state distribution.

Sparsification changes topology and reduces effective recurrent weights from 65,536 to 4,096. Matching the initial operator norm does not match rank, spectral shape, optimization geometry, or trainable capacity. Thus this isolates the first failing **package**, not the unique causal feature. A single historical data/initialization setting cannot establish population reliability.

## Three cycles actually performed

1. **Design → audit → technical test → revise execution:** recovered literal V164 source and data semantics, checked source hashes, independent forward calculation, gradients and optimizer behavior, then benchmarked execution. Kept the faster plain recurrence and cached indices/masks only. No learned states were cached. Compiled derivative timing was 0.06336 CPU seconds versus 0.04049 plain; no GPU conversion was justified for this tiny CPU workload.
2. **Design → audit → reference test → retain:** executed the original dense ten-epoch schedule. Same-checkpoint train/calibration and held test reproduced the reference; retained the frozen equations and gates.
3. **Design → audit → contrast test → revise interpretation:** sparse ten-epoch gate failed; executed only its predeclared ten-epoch continuation with the saved model, Adam state and ordering generator. Stopped at the first failed package and revised the next-step document. This was a gated three-cycle exercise, not three independent successful replications; no new architecture or post-score tuning was run.

## Cost and scope

Main learner training: 3,000 record applications / 336,000 scored targets. Technical optimizer qualification: 64 / 7,168. Total: **3,064 / 343,168**. Repeated records and positions are not independent worlds. Additional derivative-only checks: 32 record exposures / 3,584 targets, no optimizer updates. Frozen readout fitting: **14,200 repeated record exposures / 1,590,400 target applications** (83 + 59 objective evaluations), not part of the learner's training count. Training/calibration/held evaluations and dynamics diagnostics add read-only work and are included in worker CPU, rather than being hidden as free compute.

The sequential science worker used 10.23 CPU seconds and peaked at 714 MB RSS. A conservative 120-second preflight allowance gives 130.23 accounted CPU seconds; it is an allowance, not an exact per-process preflight measurement. Resource admission's largest-route forecast was intentionally conservative. This does not establish a measured end-to-end speedup against an unoptimized full campaign. No trajectories were repeated for this report.

This reference uses BPTT and the literal V164 schedule, with no added gradual warmup. It is explicitly an instrumental reference, not a local-rule or warmup candidate. No native integration, arithmetic result, deployment, or adoption is claimed.
