# Structural kernel search — brainstorm and revised direction

13 October 2026. Research/design work only. No new training, model implementation, technical cycles, science launch or core adoption. The [E2E-RULE V1 specification](../arithmetic_end_to_end_design_v1_20261013/EXPERIMENT.md) remains intact as a comparison specification; its models have not yet been implemented or qualified. Completed RAW/STREAM results remain frozen.

## 1. What we should change

The target is a better learning/computing kernel: more retained exact ability on unseen arithmetic per external sample and total useful computation. Generic bytes in, own bytes/EOS out, all semantic representations learned from observed feedback. No numeral features, operand registers, arithmetic-specific modules, symbolic interpreter, target-length input, restricted output alphabet or supplied calculation traces.

The important hypothesis is **learn a reusable state transition**, not merely a larger context-to-answer fit. Weight reuse across internal steps, learned access to an editable workspace, and learned feedback-dependent adaptation are generic inductive biases. They can be used unchanged on a different byte task. None asserts that a compact arithmetic procedure will necessarily emerge.

Three kinds of recurrence must be kept distinct:

1. Recurrence while reading successive external bytes: the GRU reference already does this.
2. Recurrence while solving one problem: repeatedly re-read the available prefix and revise temporary state before emitting a byte.
3. Recurrence across experiences: observed feedback changes a persistent fast state through a trainable update rule, influencing later examples.

The recent failures do not exclude all three. RAW V1 did test extra zero-input GRU transitions with SQ/RLS. Repeating that recipe is not a new kernel. The next version must alter what is accessible, editable and learned during the computation.

## 2. Twelve ideas, grouped rather than counted as twelve new mechanisms

| Idea | Concrete generic change | Expected value and main failure risk | Decision |
|---|---|---|---|
| K01 — recurrent workspace | Shared learned read/write transition repeatedly operates on unlabeled latent slots and the actual byte prefix | Reusable computation and temporary binding; may settle into constant/EOS states or merely buy extra depth | Primary structural candidate |
| K02 — hierarchical recurrence | Coupled fast and slow work states updated at different internal frequencies | Maintains a contextual plan while local computation changes; more states/modules may be unnecessary | Serious alternative within the recurrence family, not automatically superior |
| K03 — local cellular recurrence | One learned gated local rule applied across a chronological byte field and repeated in time | Distributed intermediate state and spatial/time parameter reuse; communication distance and training stability can fail | Primary structural alternative |
| K04 — feedback-plastic memory | Train keys, values, write/forget gates and a fast-state update using later observed prediction errors | Learn how to adapt without forcing every new fact into all slow weights; may remain an associative lookup or suffer coordinate drift | Primary learning-kernel candidate |
| K05 — learned module composition | A controller selects generic learned transformations in a recurrent computation | Reusable subroutines; routing can collapse or specialize into whole-question memorization | Second tier; never pre-name arithmetic modules |
| K06 — differentiable external working memory | A learned controller reads/writes a bounded memory with generic addressing | Binding and variable manipulation; complex allocation may repair only taught recall | Alternative workspace implementation, not an independent law-discovery guarantee |
| K07 — learned energy/equilibrium computation | A latent state relaxes under a learned objective before readout | Iterative consistency; convergence and expensive higher-order training may dominate | Conditional; no fixed-point assumptions without evidence |
| K08 — learn the update algorithm | A small learned policy chooses how feedback modifies persistent state/weights | Sample-efficient adaptation; meta-training and credit assignment may be expensive | Combine with K04, not count twice |
| K09 — adaptive internal computation | A learned controller decides how many work steps to spend, under a hard cap | Unit-compute gains on easy/hard inputs; extra halting optimization can obscure whether the core works | Later control mechanism for a working recurrent kernel |
| K10 — repeated supervision of refinement | Reuse the actually observed next-byte target at several latent refinement stages | Gives credit to reusable work steps; gains may come from extra target applications rather than structure | Matched learning-policy comparison, not a new kernel family |
| K11 — stable/flexible learning coordinates | Separate or constrain state that should persist while new details adapt | Retention; cannot create missing arithmetic capability by itself | Conditional after acquisition; retain historical variant boundaries |
| K12 — bounded real-example replay | Re-encode actual stored observations during later learning | Retention and exposure reuse; may reinforce memorization | Preserve in V1 comparison; no longer the main architectural idea |

This is a design-space map, not a new survivor count. The108 mechanisms,136 catalogue rows,254 version records and71-row older view overlap. Relevant lineages are M001/M007/M010, M016/M094, M025/M026, M027/M012, M028/M102, M031/M033, M022/M106 and M050. The full ledger is [here](../arithmetic_feedback_rule_discovery_dev_v1_20261013/ops/read_only_review_20261013/inventory/FULL_INVENTORY_REVIEW.json). Historical failures retain their exact substrate, rule and endpoint scope.

## 3. Candidate A: a reusable recurrent work cell

This is the strongest immediate proposal. Separate three kinds of state without assigning any arithmetic meaning:

- Learned parameters theta: accumulated persistent knowledge and the reusable transition.
- Current prefix X: embeddings of the actual available byte identities and learned generic positions, including the model's earlier emitted bytes.
- Temporary workspace Z: eight32-dimensional learned-seed slots, editable within one framed problem and reset at public BEGIN. No slot means operand, carry, operator or result.

Before each next-byte decision:

    X = learned embedding/position encoding of available prefix only
    repeat K times:
        Z = Phi_theta(Z, X)
    next-byte logits = learned_readout_theta(Z)

Phi performs learned cross-attention from work slots to X, slot-to-slot interaction and a generic nonlinear update. Its weights are shared across all work steps, output positions and tasks. The next byte is selected only from its ordinary257-way logits. Its own byte is appended to the temporary prefix; the new workspace is retained for the next output decision. In teaching, the actual arriving byte replaces the generated alternative as context, and only the actual feedback produces a loss.

Unlike the old R4, this cell can repeatedly **retrieve input and modify multiple interacting work states**. The knowledge is not a fixed random code or an RLS table. Unlike ordinary untied depth, one learned operation can be reused many times.

Proposed starting capacity: width32, eight unlabeled slots, four attention heads, FF64, context128, fixed K=8. A K=1 twin has the same parameters. An eight-block untied twin has the same work schedule but independent transition weights. The structural comparison must pay each model's actual arithmetic, movement, gradients and parameter-update cost.

A possible block parameterization gives34,113 shared parameters and123,937 for its untied-eight counterpart. These are specification arithmetic, not measured instantiated models; independent construction tests must verify them before a lock. No padding parameters are added to claim equal size.

The initial version uses full gradients through all eight short-task work steps. It does not assume that eight iterations reached an equilibrium and does not silently substitute a one-step gradient approximation.

Learning-policy comparison: final-stage CE versus the same actual-byte CE averaged at work steps2,4,6,8. Intermediate supervision reuses only the already arriving real target; there are no intermediate arithmetic labels. Every target application, gradient path and optimizer step is counted. The visible answer still comes from the final declared work stage, not an oracle-selected draft.

## 4. Candidate B: a local recurrent field

Keep a latent vector at each active chronological byte position. A small gated convolution/message-passing rule updates neighboring vectors; the same rule is reused at every position and work step. A learned query pools the current field into next-byte logits. The pool and recurrence are part of the end-to-end model.

The field is a generic sequence topology, not a digit grid: do not separate numbers into rows, align decimal places, create carry lanes, supply operand channels or teach a numerical renderer. It contains the single raw byte sequence with learned embedding and generic order information. This adaptation cannot inherit results from algorithms trained with a different task representation.

The hypothesis is that a local reusable transformation can propagate and compose intermediate information with less global communication than repeated all-to-all attention. A full field still costs memory and active-position work. Pooling, initialization, field movement, backward passes and the communication radius must be billed; sparse connectivity alone is not computational sparsity.

Use learned global readout so a fixed work budget does not categorically hide distant input. Then compare fixed-step local dynamics to recurrent attention at common compute. Long-distance handling and exact EOS remain part of the same arithmetic test, not a new benchmark escape route.

## 5. Candidate C: learn how feedback writes memory

A persistent fast matrix A complements slow learned parameters theta. Queries, keys, values and update gates are neural functions of the current raw-byte context and actually observed feedback. One possible generic differentiable rule is:

    k, v, alpha, beta = learned functions(context, arriving byte, prediction error)
    A_next = alpha*A + beta*outer(v - A*k, k)

This is a proposed associative update form, not an arithmetic representation. Learned k/v contain no supplied numbers or operator roles. The fixed outer-product operation is generic model structure. Reads from A affect the operative neural logits; no external exact-key answer branch is added.

Crucially, train the write policy on **later real feedback**, not only on reconstruction of the byte it just saw. A bounded multi-record gradient window, e.g.four actual observed records, lets later prediction losses assign credit to earlier writes. Every used record already belongs to the allowed teaching stream. No private meta-training worlds, extra examples, hidden correct held-out answers or uncharged pretraining are permitted.

This changes the learning algorithm and optimizer timing as well as state structure. It needs an explicit complete-recipe comparison, not an isolated-mechanism claim. A same-state stop-gradient-through-fast-write control can test the value of learning the policy; a fast-state-reset control tests persistence. Fair comparisons require recording slow updates, fast writes, gradient-window work, feedback reuse and memory independently.

No-write freezes A, theta, moments, age, write-policy learning state and any pending forbidden gradients. Generated guesses do not commit fast writes. Queries may use temporary working-state copies but leave all persistent learning state unchanged. Otherwise a seemingly successful meta-learner can leak teacher answers or learn during evaluation.

This candidate could reduce destructive shared-weight updates; it does not itself promise arithmetic laws. If it improves only taught lookup, it remains E1 and does not meet the core goal.

## 6. What the research supports, and what it does not

- [Universal Transformers](https://arxiv.org/abs/1807.03819) motivate recurrent learned attention. Our slot/read-write byte kernel is an adaptation, not a replication or an inherited result.
- [HRM](https://arxiv.org/html/2506.21734v1) and [TRM](https://arxiv.org/html/2510.04871v1) motivate latent iterative computation and repeated supervision. Their puzzle protocols differ from variable-length own-byte generation. HRM's ARC evaluation also uses task embeddings and augmentation/voting; those components cannot be silently imported into this no-task-ID comparison. TRM questions the earlier fixed-point justification. Neither paper makes our arithmetic feasibility established.
- [Neural GPU](https://arxiv.org/abs/1511.08228) and [its later improvement](https://arxiv.org/abs/1702.08727) give a relevant precedent for learned recurrent algorithmic computation, including arithmetic. We may use generic parameter sharing/local dynamics, while rejecting task-shaped numeric input layouts or representations.
- [Fast Weight Memory](https://arxiv.org/abs/2011.07831), [fast-weight programmers](https://arxiv.org/abs/2102.11174) and [differentiable plasticity](https://arxiv.org/abs/1804.02464) support trainable association/update mechanisms. They do not demonstrate our four-operation sample/compute or retention targets.
- [DNC](https://www.nature.com/articles/nature20101), [DEQ](https://arxiv.org/abs/1909.01377) and [TTT](https://arxiv.org/abs/2407.04620) broaden the space of editable memory, iterative computation and expressive adaptive state. Their complexity and training contracts are reasons to select carefully, not reasons to add all of them to one hybrid.

No source's headline accuracy, parameter count or 'small dataset' claim is treated as our evidence. Repeated supervision, augmentation, test-time voting, meta-training and extra work must enter our own cost ledger if used. No learned intermediate supervision scheme may quietly become a supplied calculation trace.

## 7. Recommended convergence after the brainstorm

The first priority is **Candidate A**, with the K=1 and untied-depth controls inside the same end-to-end experiment. It most directly tests whether a reusable computation with editable state helps exact unseen arithmetic, without requiring a new external learning signal.

Candidate B is the genuinely different computational alternative: local shared dynamics instead of global workspace access. Candidate C is the more ambitious learning-kernel alternative: learn how experience changes persistent state. They are not three parameter settings of one recurrence. K02 hierarchical recurrence and K05 module composition remain serious alternatives, but should not be mixed into every new candidate by default.

Keep ordinary G0 and T0 from V1 as references. Keep the replay configurations documented rather than erase them, but do not make replay the principal kernel search. A future formal roster and resource allocation must follow measured implementation costs and independently audited state contracts; this brainstorming document does not create another runnable88-job contract.

Freeze the existing V1 arithmetic data, opaque-byte relabeling, whole-family/group exclusions, ownEOS score, acquisition/retention measures and sample/compute accounting for the structural comparison. There is no new easier task, numeric-error success criterion, noise objective, NLP side task or manufactured arithmetic relation. The exact interpretation of additional fresh-reference comparisons and controls must be predeclared before science, rather than inherited from V1's17 tests despite a changed roster.

A new kernel earns a bounded improvement only through the same held-case causal and cost endpoints. More thought steps, a more biological name, more state, a sophisticated write policy or high taught accuracy are not the success condition.
