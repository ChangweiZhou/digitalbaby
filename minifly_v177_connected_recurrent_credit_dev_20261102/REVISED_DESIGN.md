# V177-R1 — revision after the bounded three-cycle exercise

**Documentation/protocol revision only.** Preserve D1 at `ba2632dc6da39693458b9b51544051e100ff3c8a`, both failed BP references, thresholds and the pilot choice. No new global version, new training, extended epochs, changed gate or adopted core is introduced here.

The graph learned its eight-symbol marginal distribution and some useful temporal information, but did not qualify for testing local learning. W1's final accuracy21.13% and gain0.1724 bits/target, and W2's14.54% and0.0573, fall below35%/0.75. Both exact-eight rates were zero. This is an expressivity/optimization feasibility failure at the registered regimen; it does not adjudicate BR.

Corrections incorporated in implementation and audit:

- State every parameter layout explicitly: each neuron has two33-element affine rows; clipping uses the combined66-element neuron row, and output clipping includes its129-element row/bias. No global norm communicates gradients across modules.
- A recurrence-off lesion clears the identity/leak carry as well as learned recurrent edges. Merely zeroing R/S would leave recurrent carry active.
- Keep the mathematical BS check distinct from whole-graph exactness: isolated-block gradients match BPTT; connected derivatives deliberately omit intermodule temporal paths.
- Distinguish full fitting, gradient-only diagnostics, read-only rollouts and reference-countermodel costs. The latter do not create successful neural training or independent worlds.
- Apply a prospective pilot source/data lock in future runs. This exercise's main lock preceded all main training, but its pilot snapshot was late. Use only the technical world for engineering cost measurements; the initial timing-only main-held forward is disclosed, and its scores were not used for selection.
- Persist OS process-tree CPU and exact RSS peaks per stage in future execution rather than relying on checks and startup reserves. No current result is upgraded to independent confirmation.

Next scientific question, **for a separate prospectively authorized version**: can a generic recurrence find this dependence with a justified feasibility budget and a reproducible reference-learning curve? Current W1 curves were still improving at the fixed endpoint; W2 was near the marginal baseline. That supports neither an impossibility claim nor an automatic extension. Before allocating local-credit treatments, define the reference optimization/dose study and its stopping rule independently of any new held outcomes. Keep byte projection generic, full257 outputs, genuine EOS, no lag/position/arithmetic features, no attention, and fresh matched births.

Only a reference that crosses its declared payload and own-action gate can justify the expensive causal comparison among BS and BR. Gradient alignment alone cannot rescue a failed behavioral reference. The high BR applied-step cosines here share BP optimizer history in a frozen diagnostic; they are not the trajectory of a BR learner with its own moments.

Preserve the long-term aim of reusable learning, retention and never-taught arithmetic ability. V177 remains a bounded temporal diagnostic, with no Full151 modification or arithmetic/persistence claim.
