# MiniFly experiment design — memory re-entry and temporal credit in one learner

**2026-11-05 · Design only.** The architecture, thresholds, and training schedule below are proposals. Implementation, profiling, and execution are pending. This design has no version assignment.

## The experiment

Can a fixed byte learner use acquired memory inside recurrent computation to solve **unseen addition problems**, then retain what it learned while acquiring and correcting facts? Train one architecture under a crossed comparison of memory routing and temporal credit. Keep every arm on the same chronological stream and compare complete, self-generated answers at increasing doses.

The primary contrast is **re-entry versus output-only retrieval at the same eight-symbol credit horizon**. A second contrast keeps re-entry and changes credit from eight symbols to one. All arms get three silent computation steps per symbol. This distinguishes the effect of where a memory read goes from the effects of more processing steps or longer credit. Memory continues through the entire lifetime.

V166 motivates the route intervention. Its BPTT instrument reached 99.75% on a synthetic two-hop task, versus 13% with output-only retrieval. Its structured study writes and task differ from this proposal. V170 reached 54.7% on held-out instruction pairs despite 100% on trained pairs. V173's reference was still improving at its budget edge and its planned learner comparison stopped. These results motivate a held-orbit behavioral endpoint and a measured learning curve. [V166 report](https://github.com/ChangweiZhou/digitalbaby/blob/f4146856c3241d94f7fa948e0b7d30fa0b869bc1/minifly_v166_memloop_dev_v1_20261026/REPORT.md) · [V170 report](https://github.com/ChangweiZhou/digitalbaby/blob/f4146856c3241d94f7fa948e0b7d30fa0b869bc1/minifly_v170_reentry_dev_v1_20261027/REPORT.md) · [V173 report](https://github.com/ChangweiZhou/digitalbaby/blob/f4146856c3241d94f7fa948e0b7d30fa0b869bc1/minifly_v173_microcircuit_dev_v1_20261028/REPORT.md)

## Fixed learner

Reuse the immutable Full151-derived byte sensor and causal FE update used by V219, including the same EOS receptor construction. Lock its source, arrays, precision, and birth wiring. Each of eight independent circuits receives 16 fixed randomly selected KC coordinates per symbol and maintains eight workspace activations and an 8 × 8 acquired matrix. The circuits share the observed stream and the output error. Their workspace states and memory matrices have separate recurrent paths. Reuse the sensor, byte/EOS interface, and checkpoint conventions. The workspace, memory writer, and credit rule specified here constitute the experimental change. [V219 learner source](https://github.com/ChangweiZhou/digitalbaby/blob/bc900efe3d79524c33005b199133ecf65ff1d248/minifly_v219_evolved_local_plasticity_design_20261105/learner.py)

The proposed memory contract allows a bounded matrix of acquired weights to act on a key derived from the circuit's current state. Each read is one matrix-vector product. The model stores fixed-size synaptic state, with no per-example rows, history buffer, similarity search over stored items, or growing key/value cache. The trainer supplies ordinary bytes and genuine record EOS, including during arithmetic and fact learning. The learner gets no digit values, carry state, field boundaries beyond observable symbols, entity identifiers, task flags, or worked steps. The contract supplies a generic recurrent loop, a three-step clock, and a generic writer. Their use in computation is the experimental question.

### One prediction and update

For circuit `i`, let `h` be its workspace, `M` its acquired matrix, and `x` its sensory input from the already observed prefix. Keep `x` and `M` fixed over the three silent steps at each symbol. Use the same weights at all three steps.

```text
q(h) = h / sqrt(1 + dot(h, h))
r(h) = M q(h)
h_next = 0.75 h + 0.25 tanh(U x + R h + b + route * r(h))
```

Set `route = 1` for re-entry and `route = 0` for output-only. After step three, compute `f_i = concat(h, r(h))`. Both routes pass the final read and workspace to the same form of linear head. The logits are `bias + sum_i O_i f_i` over 257 outputs. Stable log-softmax defines a normalized distribution before the actual byte or EOS `y` is revealed. Route-off receives the same read in the head, has the same trainable coordinates, and advances the same number of workspace steps.

After reveal, update the memory with a gated error-correcting write. `v(y)` is a fixed projection of the revealed symbol's existing receptor vector, normalized to norm at most one. Draw each 8 × 88 projection independently at birth with Gaussian entries of standard deviation `1/sqrt(88)`. The same raw symbol and the same generic update apply at every site, including EOS.

```text
q = q(h_after_step_three)
alpha = 0.10 sigmoid(g_h · h_after_step_three + g_v · v(y) + b_g)
M_proposed = M + alpha (v(y) - M q) q^T
M_next = M_proposed / max(1, ||M_proposed||_F / 4)
```

Then deliver `y` to the sensor for the next site. The write uses the predicted site's workspace and the newly observed symbol. Its gate learns from subsequent prediction errors through the credit rule below. A small gate preserves the current matrix except for the bounded write it permits. The writer receives the same observations at syntax and answer sites. Log write magnitudes, gate activity, memory norm and clipping, and read magnitude. This will show whether the trained model uses the memory or learns to suppress it, as happened in the Round 29 search. [Round 29 memo](https://github.com/ChangweiZhou/digitalbaby/blob/2fc9da7b1b19b1d9c84d99778c0ce94b130bdc18/minifly_research_memos/MINIFLY_MEMO_ROUND29_20261105.md)

At birth, start `h`, `M`, and sensory state at zero. Initialize `U` with zero-mean Gaussian entries of standard deviation `1/sqrt(16)`, `R` with a Gaussian draw scaled to spectral norm 0.9, and head weights with standard deviation 0.02. Set biases and gate vectors to zero, apart from `b_g = logit(0.2)` for an initial write rate of 0.02. Use separate seed streams for wiring, initialization, sensory projection, data, and evaluation. Use the same birth within each paired world.

### Forward credit through the assembled circuit

For each circuit, let `s_i = (h_i, vec(M_i))` and `theta_i = (U_i, R_i, b_i, g_h_i, g_v_i, b_g_i)`. Propagate the forward sensitivity `D_i = ∂s_i/∂theta_i` through each silent step and through the post-reveal memory update using the product and chain rules. For a map `F` use `D_next = F_s D + F_theta`. Compute the read derivative using both terms `dM q + M dq`. Include the derivative of the gate, error-correcting write, and active norm cap. At the cap boundary use the uncapped branch's derivative by convention. Compare finite differences away from the kink. Each circuit's next state depends only on its own state, shared fixed sensory input, and the revealed symbol. This keeps the recurrent Jacobian block diagonal even with re-entry.

Broadcast `e = p - one_hot(y)` from the common head. The circuit accumulates `e^T O_i (∂f_i/∂theta_i)` with its own outgoing head weights. The head accumulates its direct softmax gradient. This uses forward sensitivities and a circuit-level readout error. It uses no reverse-time pass or hidden-layer feedback chain. The use of current outgoing head weights in the circuit error is explicit.

- **FWD8** starts each eight-symbol credit window with zero sensitivities, conditional on the actual incoming workspace and memory. Both states continue across the boundary. The derivative is exact within that window, including writes. Effects of earlier parameter-dependent state are detached at the window boundary.
- **FWD1** starts each symbol with zero sensitivities. Credit still passes through the three internal transitions for the current prediction. Writes influence future states, but their delayed loss has no gradient into the gate in this arm. The contrast tests credit through earlier state **and** earlier writes.
- Use the same per-coordinate Adam update in all arms, every 32 revealed symbols, on mean gradients over that block. Start with rate 0.003, betas 0.9 and 0.999, epsilon `1e-8`, and gradient norm cap 1 separately for each circuit and the head. Hold parameters fixed inside each update block. Keep the optimizer and block counter across records and phases. Partial blocks at diagnostic checkpoints stay intact.

Numerically compare FWD8 to a matched stop-gradient reverse-mode calculation and to central directional differences with the incoming window state held fixed. Check memory-mediated derivatives with nonzero memory and actual writes. An external BPTT32 diagnostic uses the same forward graph, loss, update cadence, and optimizer, with all 32 symbols differentiated. This tests a longer credit horizon. A FWD8 versus BPTT32 gap is evidence about the specified horizon and regimen. FWD8 versus an equivalent eight-symbol BPTT calculation is an implementation parity check.

## One continuous lifetime

Use ordinary decimal addition, rendered as `a+b=sum` followed by true EOS. Operands are positive, equal-width decimal integers with no leading zero. Keep arithmetic feedback to final answer records. Evaluator-side code generates and audits data. It never gives the candidate digit columns, carries, correct intermediate states, or operand positions. Use facts as a separate type of observed record within the same stream, rendered as `f/<five-lowercase-letter-key>=<three-digit-value>` plus EOS. Assign random distinct keys and values independently of the arithmetic. Their explicit syntax is observable text.

| Phase | Stream | Read-only checks |
|---|---|---|
| A · acquire | 65,536 distinct addition orbits, with one 2-digit, fifteen 3-digit, and sixteen 4-digit records in each seeded group of 32. Interleave one of 64 initial facts per 16 addition records, rotating through seeded permutations of the keys. | Dose checkpoints at 4,096, 8,192, 16,384, 32,768, and 65,536 addition records. Measure arithmetic and original fact acquisition. |
| B · interfere | 8,192 new 4-digit addition orbits and 64 new facts, one fact record per four addition records. Keep old facts out of this phase. | Measure new acquisition, still-valid original facts, and retained arithmetic at B end. |
| A′ · correct | 4,096 more new 4-digit addition orbits. Replace the values of 32 of the original keys, one correction record per four addition records. The other 32 original values and all B values remain valid. | Measure exact new and obsolete values after 1, 4, 16, and 32 correction exposures per key. Test unchanged facts, B facts, and retained arithmetic at the final checkpoint. |

The model sees the chronological byte stream and learns with the same rule throughout. A, B, and A′ are evaluator labels. They supply neither a phase flag nor a state reset. The changing fact values provide legitimate new evidence. Checkpoints and probe rollouts use separate copies, with all held answer bytes kept outside feedback-bearing histories.

### Held panels

- **Primary** uses 512 never-taught unordered operand pairs, with 64 per cell across 3- and 4-digit inputs and four mutually exclusive carry strata. Strata are no carry, one non-final carry, multiple carries without final overflow, and final overflow. Split multiple carries by whether they propagate in the report. Hold each orbit out of every phase and reserve both operand orders from training. Select one orientation per primary orbit before scoring.
- **Counterfactual** uses 128 additional matched pairs. Change one low-order digit so a carry changes a higher output column while leaving the rest of the prompt fixed. Allocate 64 pairs per operand width. Reserve both members' entire commutative orbits and keep them separate from the primary panel. Score the pair as correct only if both complete answers are right. For example, `268+431=` gives `699` and `269+431=` gives `700`.
- **Length extrapolation** uses 256 5-digit operand pairs, 64 per carry stratum. Report this separately from same-length generalization. The training roster includes only 2- to 4-digit operands.
- **Taught and factual panels** use 128 fixed taught addition items from the first dose and the full original, new, unchanged, and corrected fact sets. Their taught-answer status differs from held arithmetic. For retention, also report the subset already answered correctly before interference, alongside unconditional rates and its denominator.

Generate all held panels and the exact exposure manifest before training. Use separate seed roles for engineering and final worlds. Check that the minimum-dose stream witnesses each required digit-pair/carry-in transition in the relevant units, interior, or leading column role for the primary and counterfactual panels. Use this as a determinacy check under the declared decimal column-rule class. Keep the carry trace on the evaluator side. Preserve the operand-orbit exclusions through the entire lifetime. Publish actual distinct orbits, all repeats, each stratum count, and witness coverage. The task-aware oracle verifies every target.

## Comparisons and causal probes

| Arm | Route | Temporal credit | Scope |
|---|---|---|---|
| R8 | Memory returns to workspace and reaches the head | FWD8 | Main candidate through A′ |
| O8 | Memory reaches the head | FWD8 | Tests re-entry at matched credit through A′ |
| R1 | Memory returns to workspace and reaches the head | FWD1 | Tests inter-symbol credit at matched routing through A′ |
| O1 | Memory reaches the head | FWD1 | Completes the factorial comparison through A′ |
| R32-BP and O32-BP | Matched re-entry and output-only | BPTT32 | External diagnostics at all A doses |

All arms share sensor, tensor shapes, births, stream, next-symbol objective, workspace-step count, and optimizer schedule. For the main 2 × 2, the interventions change only the re-entry edge and the derivative boundary. Record elapsed CPU for each arm, since equal exposure and architecture do not imply equal compute.

Run two read-only probes on R8. First, cut the return edge on a checkpoint copy while retaining the direct memory-to-head read. This tests whether the trained learner uses the edge during generation, alongside the matched from-birth O8 comparison. Second, fork the B-end checkpoint twice with slow parameters fixed. Run the fast writer on the first four correction rounds with true values in one copy and a fixed derangement of the same new values across keys in the other. The correction block keeps its byte counts and identical intervening addition records. Install only each acquired matrix state in a common B-end workspace and sensor snapshot. Measure whether the exact outputs track the assigned values, including both-correct pairs and still-valid old facts. A drop under a generic lesion alone would leave memory content unresolved.

On every scored question, consume the visible prompt on a disposable checkpoint copy. Freeze the head, circuit parameters, and acquired matrices through the probe. Let the sensor and workspace evolve with the prompt and then the learner's own greedy outputs. The evaluator checks the full byte sequence and the learner's own EOS within 12 generated symbols, including EOS. Early EOS, extra bytes, and a missing EOS are errors. Run the same read-only procedure for every arm. Report teacher-forced loss on a separate copy and keep the score mask out of the learner.

Use the same exposure stream to fit strong shortcut baselines. Include canonical exact-key retrieval with frequency fallback, nearest taught operand pair with its stored answer, and per-column digit prediction without carry. Report each, including its exact byte/EOS result. For nearest-copy retrieval use L1 distance on the sorted operand pair within the same width, with ties to first exposure and frequency fallback for absent widths. The no-carry model fits modal output digits from right-aligned taught answers by operand width, column role, and unordered input digit pair. It emits an operand-width answer with leading zeros stripped and backs off to width-and-column digit frequencies. Report the shortcuts separately in each carry stratum. Use an arithmetic solver as the evaluator-only ceiling. Use a latest-value table as the separate factual reference. Its direct key access makes it a useful task oracle for taught facts.

## Decision rule

Use three separate development worlds for technical checks and early learning curves. The intended final roster is eight independent stream/birth pairs, with all arms paired within each world. Fix source, seeds, panels, comparators, dose, and thresholds before the final roster. Score saved checkpoints on the locked held panels after training so their outcomes cannot steer the dose. Report all worlds. Use world-level paired 95% intervals. Each independent stream/birth pair is a statistical unit. Show exact numerator and denominator by arm, dose, world, and carry stratum.

**Primary endpoint.** At the largest A dose, compare balanced held 3-/4-digit exact answer plus EOS for R8 versus O8. A meaningful re-entry result for this pilot is a mean gain of at least 10 percentage points with the paired interval's lower bound above zero. Report R8−R1, R1−O1, and the difference of re-entry effects across credit horizons. A credit gain worth following is at least five points for R8−R1 with its lower bound above zero.

**Integrated behavior screen.** Require R8 to reach at least 70% held accuracy, at least 50% both-correct on carry counterfactuals, and a mean held gain of at least 10 points over the per-world best declared shortcut with the paired lower bound above zero. Report the causal cut and memory swap alongside those results. Report any smaller or single-factor improvement as a bounded component result. These thresholds are proposed development decisions. A later sealed study would be needed for formal E3 confirmation.

**Persistence screen.** At A end require at least 80% of initial facts. At B end require at least 70% of new facts. At A′ end require at least 70% of corrected values with at most 20% obsolete-value outputs. Limit mean losses to ten points. Compare unchanged original facts and held arithmetic at B end and A′ end to A end. Compare B facts at A′ end to B end. Show each denominator, absolute level and paired interval. Score persistence jointly with acquisition so a model that never learned the original answers cannot pass by retaining a low score.

| Result | Decision for this learner |
|---|---|
| R8 beats O8 and R1 and meets the behavioral and persistence screens | Retain the fixed implementation as the candidate reference for a separately locked confirmation. Report content use only to the extent supported by the checkpoint probes. |
| Re-entry helps and the upper bound on R8−R1 is at most five points | Keep re-entry. Prefer the one-symbol rule for a follow-up if it meets the behavioral screens and has lower measured cost. |
| Longer credit helps with either route and the upper bound on R8−O8 is below five points | Keep the useful credit horizon. Defer added memory routing. |
| BPTT32 learns while matched FWD8 remains weak | Investigate credit horizon or optimization on this graph. Matched-horizon derivative parity isolates implementation errors first. |
| Both longer-credit references and local arms remain weak | Record the observed capability and learning curves for this architecture and dose. A curve still improving at the cap leaves the asymptote unresolved. |
| Held arithmetic improves while revision or retention fails | Keep the bounded arithmetic finding and diagnose persistence on the same learner. |

Run through the locked maximum while the implementation and resource bounds hold. A weak BPTT reference leaves the local arms on the same schedule. For reporting the locked-roster curves, call a plateau only when the paired intervals for the final two dose doublings are within ±2 points on held exact accuracy and ±0.02 bits per answer symbol on normalized diagnostic loss. If evidence is too noisy to meet those conditions, report the slope as unresolved. Use the final planned checkpoint for the primary result.

## Compute and engineering checks

| Quantity | Planned count or bound |
|---|---:|
| Gradient-trained scalars | 34,889, including the 257-way head and eight write gates |
| Additional acquired memory scalars | 512 across the eight matrices |
| Fixed receptor projection scalars | 5,632, in addition to the reused sensor assets |
| Forward sensitivity state | 124,992 scalars for eight 72 × 217 blocks, about 0.48 MiB in float32 |
| A training symbols per life | At most 993,280 including EOS and fact records |
| B and A′ symbols per life | At most 155,648 and 77,824 respectively |
| 32 local lives through A′ | At most 39,256,064 training symbols |
| 16 BPTT diagnostic lives through A | At most 15,892,480 training symbols |
| Combined main training | At most 55,148,544 training symbols, with three workspace transitions per symbol |

These upper bounds follow from the registered record mix and the largest valid record lengths. The main total excludes engineering, donor histories, and scored rollouts. Three development worlds through the first A dose add at most 1,117,440 training symbols across the six arms. Two four-round donor histories in each of eight worlds add at most 155,648 symbol applications with the head and circuit weights frozen. Count evaluation prompt, generated, and teacher-forced symbols separately. Record actual unique records and repeated presentations, sensory operations, derivative and optimizer work, peak memory, and CPU by arm. Report the full cost of fixed sensory assets, optimizer moments, checkpoints, copied states, and BPTT tapes as well as trainable weights.

Before locking the main roster, profile the assembled implementation on held-independent engineering streams with native sensing, derivatives, writes, and snapshot generation enabled. Set CPU and memory limits from measured per-arm throughput and complete roster costs with 25% reserve. If the proposed dose is unaffordable, revise the scope before final worlds are opened. Distinguish a resource stop from a completed scientific negative.

Keep engineering to three concrete checks. A two-hop wiring fixture checks that an acquired read changes the next state and next read with the route on, and leaves the workspace unchanged relative to the corresponding cut control when the route is off. A derivative fixture checks forward credit and finite differences through nonzero reads and writes, including exact zero cross-circuit sensitivities and the intended truncation boundaries. A short end-to-end rehearsal verifies byte/EOS generation, exposure exclusions, realized writes, scorer parity, and deterministic checkpoint resume through a phase transition. Retain raw predictions, source and split hashes, checkpoints, timing receipts, and any incomplete attempts. This package leaves the experiment in **design-only** status.

The proposal uses the two attached reviews as input. V216 provides a measured dose precedent and a caution about low absolute counterfactual performance. V217 provides a worked distinction between learned behavior and supplied state machinery. The consolidated and parallel source heads inspected were `bc900efe` and `2fc9da7b` respectively. Fable review (supplied in chat) · Cross-branch review and response (supplied in chat) · [V216–V218 assessment](https://github.com/ChangweiZhou/digitalbaby/blob/2750e76f2632d45c1faa22b372c23f047a5ba7a6/PROJECT_DOCUMENTS/MINIFLY_V216_V218_CROSS_BRANCH_ASSESSMENT_20261105.md)
