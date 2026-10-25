# V150 — three-cycle output-learning diagnostic

Completed 2026-10-25. **Output-only learning passed; native development did not improve. No new core is adopted and no addition efficacy is established.** These are three engineering cycles, not three independent scientific replications.

## Cycle 1: design, mathematical audit, interface test, revision

Separated output representational capacity, local-rule learning, and native integration. All 256 bytes plus EOS remain legal. No vocabulary mask, forced stop, arithmetic rules, attention, Transformer, ELM, or gradient through the encoder/native dynamics/time was introduced. Output-local gradients are explicit. The prior design snapshot is preserved in `minifly_output_learning_design_v1_20261025`; the named V150 design and pre-fit clarifications record authorization and accounting before training.

The centered C16 output has rank 15. Finite differences agreed with the analytical gradient to 4.09e-10; reachability, projection, gradual warmup, cloning, serialization, and read-only evaluation checks passed. Sixty-four isolated signed native compartment writes, each paired with a passive timing control, qualified the sign. Those partial technical writes are not full trained histories.

## Cycle 2: bounded local tests and gated native test

The fixed roster ran once: 66 output-only fits, each with 4,096 observed targets and 1,024 write-free validation probes; three fixtures, seeds 11–13, and rates 0.03/0.1. One common rate, 0.1, was selected by the predeclared validation aggregate. No extension or retry occurred.

A is the old independent-bit residual; B uses the coherent categorical gradient with the fixed code; C also learns the output code, with row centering and norm at most 4. C0 ignores context; BIAS uses the fixed code with a learned shared class bias. The per-context output-state reference has perfect context keys: it is an output diagnostic, not native memory.

Mean expected bits per symbol at the selected rate (lower is better):

| Fixture | A | B | C | C0 | BIAS |
|---|---:|---:|---:|---:|---:|
| Uniform 257 | 8.0403 | 8.0463 | 8.0463 | — | — |
| Skewed, full support | 5.9407 | 5.8071 | 5.4881 | — | — |
| Four contexts, full support | 5.9479 | 5.3729 | 3.5614 | 5.2655 | 3.4998 |

B was closer than A to the numerical fixed-code reference on both nonuniform fixtures. C beat the context-ignored and add-one frequency controls in every conditional seed, satisfying the declared readiness gate. However, **BIAS slightly outperformed C**: learned code geometry is not shown to be necessary. B did not dominate A on the uniform finite-sample fixture. Numerical reference optimization is not a certified capacity bound.

The qualified native sign and output gate admitted six fresh native histories: two previously used development texts × A/B/C, one pass each. All arms used the same normalized permutation frontend, 16 native stores, fresh adaptation, disabled homeostasis, common rate and gradual warmup. Native synaptic equations were unchanged; supplied teaching coefficients differed. C learned its code from its own native values; toy weights were not transplanted.

Development teacher-forced bits per symbol:

| Text | A: old bit | B: coherent fixed | C: learned code | Unigram | Bigram |
|---|---:|---:|---:|---:|---:|
| Bryant | **7.5631** | 7.6746 | 7.6489 | 4.7796 | 4.4109 |
| Buster | **7.5394** | 7.6634 | 7.6356 | 4.6263 | 3.8189 |

**Neither new rule improved native development loss at this fixed dose.** All native arms remained worse than the frequency references. All 96 own continuations reached the 64-decision cap; none emitted EOS or exactly matched its expected continuation. Teacher-forced loss must not be represented as own-output task success. These are two development texts, not independent confirmation or an arithmetic test. Earlier versions with different homeostasis/warmup are not matched baselines.

## Cycle 3: artifact audit, resume test, revision

All locked sources matched, the roster was exact, saved-state and compact scores agreed, five corrupted artifact fixtures were rejected, and cross-process resume preserved state/RNG and the warmup counter. Resume qualification used 32 separately charged technical targets. An additional independent manual 257-logit check on one first calibration record from each saved native state matched exactly, without learned writes or state changes (429 passive target probes).

The revised design is `REVISED_DESIGN.md`. The completed test narrows the problem to transferring a mathematically coherent output signal into useful native learning under a measured budget; it does not prove local learning impossible or identify one sufficient repair.

## Counts, resources, and limits

- Output-only: 66 fits; 270,336 observed targets; 67,584 validation probes.
- Native: six histories; 240 full training records; 12,504 byte/EOS target applications; 200,064 compartment teaching calls. These are correlated events, not independent lives.
- Technical: 32 synthetic observed targets and 64 partial native compartment writes, separately classified. Total observed targets including synthetic resume checks: 282,872, below the 283,096 cap.
- Native passive evaluation: 306 line queries / 18,777 target probes, plus 429 audit probes. Generation: 96 capped continuations / 6,144 decisions.
- Measured main execution: 393.919 summed child CPU seconds, 397.489 wall seconds, peak RSS 685 MiB. Cycle-3 audit plus child: 0.966 CPU seconds; native response audit: 4.466 CPU seconds. Measured total: 399.351 CPU seconds, below the 7,200-second budget. Authoring, preparation/imports outside the supervisor, and small inspection commands were not fully timed; this is not a complete project-energy estimate. No claim that all stage-boundary interruption cases were qualified.

No scientific fit failed or was retried. One inspection command used a Python without NumPy and failed before any learning; it was corrected for inspection only. No Alice, addition holdback, prior trained trajectories, or new task campaign was opened. Arithmetic track and its pointer remain unchanged. Raw arrays, per-position predictions, corpus text and checkpoints remain local and are excluded from the compact review bundle. No GitHub push was requested in this turn.
