# KERNEL-SEARCH V2 — revised design brief

**Architecture-search proposal,13 October2026. Not a frozen runnable science protocol.** This brief replaces the incremental emphasis of the previous design while preserving [E2E-RULE V1](../arithmetic_end_to_end_design_v1_20261013/EXPERIMENT.md) for comparison. No new training is included in this delivery.

## Fixed objective

Learn four-operation and bounded mixed-expression arithmetic from actual feedback, producing its own exact byte answers/EOS on unseen cases with better sample and computation efficiency and retained ability after later learning. All semantic representations are learned end-to-end. Model computation and learning structure may change.

Keep the current V1 data/split/exposure specification for model comparisons: byte relabeling, whole binary-family exclusions, mixed-group exclusions, balanced observations, causal prior-learning histories and later mixed learning. Intermediate values and grouping remain evaluator-only. Do not substitute a new task or loosen exact scoring to make a candidate look useful.

## Main structural candidates

| Candidate | Fundamental change | Direct question |
|---|---|---|
| Recurrent work kernel | A shared learned transition repeatedly reads the available prefix and edits an unlabeled workspace | Can repeated reusable computation learn precise composition rather than only fit question/answer correlations? |
| Local cellular kernel | A generic learned local rule operates on a distributed byte-position field, shared over space and time | Does distributed intermediate state and local propagation discover an efficient arithmetic computation? |
| Feedback-plastic kernel | A trainable feedback-conditioned update writes persistent fast state, with credit from later observed predictions | Can the kernel learn how to incorporate experience while preserving previously useful knowledge? |

These are different organizations of computation/learning, not three loop counts of one model. Their concrete prototypes and limitations are described in [the brainstorm](BRAINSTORM.md). A combined work+plasticity kernel is an eventual hypothesis, not a default compound to which every possible mechanism is added.

## Comparators and interpretation

Preserve ordinary G0 and T0 from V1 as conventional references. They are specified but still unqualified implementations. Replay-only G1/T1 remain documented; replay is not the main structural search.

For the recurrent work kernel, include a one-work-step twin and an ordinary untied-depth twin in the same study. The first distinguishes editable computation from just adding an interface; the second distinguishes weight reuse from buying ordinary depth and parameters. Both useful-parameter state and real compute are charged. A loop-count effect alone is not a biological or algorithm-discovery claim.

For a feedback-plastic candidate, include a matched policy/credit control, not just a smaller plain network. Preserve actual causal restrictions on all persistent fast writes and on the outer gradient window. If the candidate changes update cadence, declare it as a complete kernel-learning recipe and match cadence in its mechanism control. No private meta-training dataset may be omitted from sample cost.

Hierarchical recurrence and learned generic module composition remain serious alternatives. Do not automatically add them to every primary. Fixed-rate recursion and source-conditioned read/write access are a cleaner first test than an unqualified learned halting controller. Adaptive internal work remains an explicitly charged follow-up option, not an oracle 'think until correct'.

## Learning and inference boundaries

Only real observed reply bytes/EOS create supervised targets. Repeated supervision of internal refinements may reuse these targets; all target applications, internal forward/backward work and optimizer steps are counted. There is no arithmetic trace, hidden answer length, correctness oracle at inference, numeric module or generated self-label.

Workspaces/fields may evolve during solving in transient state. Persisted parameters/fast feedback memory change only under allowed observation-learning events. Own output generation is pure with respect to learning state. Internal work stop and answer EOS are different controls: a model may finish its calculation yet still need to emit more bytes.

End-to-end differentiability does not mean that a task-specific representation can be hidden in an initializer, matrix layout, feature loss, prototype or output renderer. All256 payload bytes remain ordinary categories; generic topology/order is permissible, arithmetic meaning must be learned.

## Evidence and allocation

Use the same acquisition, exact held-case, prior-learning-control, final retained and sample/compute endpoints. A more sophisticated kernel must outperform useful conventional references under paid computation; high taught recall or better proximity alone does not promote it.

Run any qualifying technical checks inside the eventual complete comparison, not as a new series of unrelated problem-localization studies. First verify actual interface/gradient/state contracts and measure complete-job cost. Then prospectively freeze a bounded screen and fresh confirmation roster. The changed candidate/control roster needs its own declared inference family; V1's17 tests and88-job maximum cannot be blindly reused.

No scientific allocation, runtime guarantee or model implementation is claimed by this brief. Future sessions remain subject to the existing8-hour admission/10-hour hard boundary, whole-job checkpoints, detached supervisor/caffeinate, and no automatic restarts/extensions. Parameter-sharing, fast-state memory and repeated feedback are charged, not treated as free capacity or free samples.
