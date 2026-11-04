# MiniFly V213 · Learning persistent event effects from observations

**Status** DESIGN_ONLY  
**Date** November 4, 2026  
**Decision** Can a bounded local learner use earlier observation feedback to predict how an opaque event changes a different entity's future state, on entity pairs never encountered together during training?

## Why this experiment

V211C showed a specific failure in the recovered MiniWorld direct path. After a pickup or drop about the queried person, the person's still-valid location disappeared from 497 of 512 candidate lists, despite a 94% reduction in shared entity hash rows. Round 18's larger table likewise did not rescue its tested object learner. Neither result tests whether a learner can acquire an observation-driven state transition. V213 holds language aside and makes that question measurable. The action–consequence confirmation in V208 remains separate. V213's movements are externally scheduled observations, with no policy optimization.

Use a compact **new mechanism instrument**, with no claim of executable Full151 inheritance. The methodological precedent is to test internal state through predictions of what will actually be observed. Dynamic relational inference supplies a precedent for testing changing dependencies, with different architecture and training assumptions. Earlier MiniFly held-pair experiments establish why no-prior-write controls and never-reinforced combinations matter.

[V211C report](https://github.com/ChangweiZhou/digitalbaby/blob/ea404d4b6e37dfe82a869a6006bb98596e9ceaf3/minifly_v211c_object_interference_diagnostic_20261104/REPORT.md) · [Round 18 memo](/Users/pencilbard/Downloads/MINIFLY_MEMO_ROUND18_FAILURE_MODES_20261104.md) · [Predictive representations of state](https://proceedings.neurips.cc/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html) · [Dynamic relational inference](https://openaccess.thecvf.com/content_CVPR_2020/html/Graber_Dynamic_Neural_Relational_Inference_CVPR_2020_paper.html)

## Simulator and observable interface

Use eight opaque entity IDs and four opaque position IDs per world. The generator uses four IDs in the first participant field and four in the second. All IDs, opcode meanings, and position symbols are independently permuted per world. IDs are visible. Hidden simulator relations and their labels are evaluator-only.

Three generic framed record types are supplied:

| Record | Observable information |
|---|---|
| State observation | Entity ID and its observed position |
| Pair event | Opaque opcode and two ordered entity IDs |
| Movement observation | ID of the first participant and its new position |

Each bounded packet concerns one pair. A pair starts independent in the simulator. One of two opaque opcodes makes the second entity move with the first on subsequent moves. The other restores independence. The event does not itself change either observed position. When independent, the second entity stays at its last position during a move of the first. Apply each opcode repeatedly in both previously linked and previously independent states. Expose two ordinary observations of other entities between each pair event and the revealing movement. Their states and positions are independently chosen, and the same observations appear on both sides of a matched probe.

The learner sees the pair event, the unrelated observations, and the first entity's movement. At that point it predicts the next position observation of the second entity. Only after prediction does that actual observation arrive. In training, its position supplies the only loss target. The simulator never supplies link status, opcode meaning, a correct action, a reasoning trace, or a counterfactual outcome. Do not include a second unresolved pair event between the training event and its outcome. Timing and destination positions are sampled without consulting the learner. During teaching and evaluation, sample the first entity's new position uniformly from the positions that differ from both entities' last observed positions. This ensures each revealing move can distinguish the two effects. Initial observed positions are drawn independently and uniformly. Report the actual position-pair coverage per opcode.

**Scaffolds supplied.** Record framing, stable IDs, ordered participant fields, the identity of the entity whose observation is next in this interface, one active pair per packet, and fixed packet reset at the observable start. The candidate also has a generic pair-state slot and two predictive branches. These are deliberate structural priors. This experiment asks whether the *effect of a token on that state* can be learned. It leaves open discovery of entities, routing among competing pairs, and learning a state representation from raw language. No part of the learner hard-codes the token-to-effect mapping or copies one participant's position into the other participant's observed-state slot.

## One local learner

Keep last observed position per entity and a scalar g per ordered pair, born at 0.5. These are transient state. Reset them at the same packet start in every arm. Keep learned weights across packets. Position slots change only on actual observations. At an event with token o, transform the active pair state using two learned parameters for that token:

    u_o = sigmoid(a_o)
    v_o = sigmoid(b_o)
    g_new = (1 − g_old) u_o + g_old v_o

Both parameters start at zero. The family can express a state-dependent transformation, approximate persistence, or push either prior state toward the same endpoint. Which effect each token has is learned. The gate has no passive decay. Holding state across the unrelated records is an architectural prior. Learning a robust duration or assigning delayed credit among multiple causes is outside this assay.

Use two shared learned 4 × 4 logits tables. One takes the second entity's last observed position as its index. The other takes the first entity's newly observed position. Convert each row to a probability distribution with softmax and denote the results p_self and p_other. Use seeded independent normal initial logits with mean zero and standard deviation 0.01, with the same birth across arms. Emit the model's own four-position prediction:

    p = (1 − g_new) p_self + g_new p_other

Neither table contains a built-in copy or persistence mapping. Sharing both tables and opcode parameters across entity pairs is explicit. The experiment tests whether a rule learned with that sharing transfers. Interpret g through changes in predictions under intervention. Its numeric value has no supervised meaning.

On the first revealing observation y, update only the two selected table rows and the active opcode's two parameters. The row updates are the local log-loss error vectors, each weighted by its mixture responsibility:

    w_self  = (1 − g_new) p_self[y] / p[y]
    w_other = g_new p_other[y] / p[y]
    Δlogits_branch = eta × w_branch × (onehot(y) − p_branch)

The scalar gate credit is (p_other[y] − p_self[y]) / p[y]. Multiply it by (1 − g_old) u_o (1 − u_o) for a_o and by g_old v_o (1 − v_o) for b_o. Use eta × that increment, clipping the scalar gate credit to [−4, 4] before multiplication. Use max(p[y], 1e−8) in all update denominators. The probability floor does not alter emitted scores. Calculate all increments from the prediction-time values before any write. Update the currently selected parameters only. The stored g_new is fixed for that event and is not retroactively recomputed after the update. Clear the temporary record of g_old, token and active rows after consuming the observation. This is a local update with one bounded pending record and no gradient through the event history. Preserve gradual warmup with eta 0.02 and 0.05 in the first two eight-packet blocks and 0.10 afterward. Freeze the schedule before opening science.

Use float32 arrays with declared numerical tolerances. The learned state has 36 scalars, or 144 bytes excluding fixed IDs and metadata. Working state is eight observed-position slots, at most 64 scalar pair slots, and one bounded pending record. Each event touches a bounded number of entries. Track per-record work, actual byte movement and overhead of the evaluator separately. Use no attention, Transformers, ELM, task labels, or backpropagation through time.

## Causal contrasts and shortcut checks

| Arm | Update from the observed position |
|---|---|
| ALIGNED | Real local prediction error updates the emission rows and the opcode that occurred |
| SHUFFLED-OP | Same records and emission updates. Route only the two gate increments to a prelocked, independently permuted opcode assignment. |
| FROZEN-GATE | Emission rows learn from the same observations. Gate parameters stay at birth. |
| NO-PRIOR-WRITE | Same observations, clock, and transient states. All learned parameters stay at birth throughout teaching. |

In SHUFFLED-OP compute the increment from the same active-event snapshot within that arm, then redirect it to the assigned slot. Prelock permutations within block and packet step to preserve the exact opcode write counts without changing delivery time. Preserve all physical records. This isolates alignment of gate credit, with matched update opportunities and comparable work. Realized weight motion can differ and is reported. All arms freeze learned weights during probes. Only normal causal transient-state updates are allowed in a probe.

Score evaluator-only references: last observed position, copying the observed mover every time, a recent-suffix predictor trained on the last three visible records, and an exact pair/outcome lookup that abstains on held pairs. Also fit a pooled count table on (most recent pair opcode, last observed second position, new first position) with add-one smoothing. It uses the same training observations and has a supplied one-event memory. It can legitimately learn the simple transition and is a strong reference for sample and compute comparisons. Report its cost and accuracy. A tie with it may support rule reuse, with no demonstrated efficiency advantage for the local mechanism. The evaluator-only known-rule oracle must solve every paired case from the visible history and the true opcode map. Its map is never exported to learner code.

## Roster and decisive probe

Use eight independent worlds from the first 64 bits of SHA-256 of MINIFLY|V213|WORLD|i for i = 0 through 7. In each world, exclude one perfect matching of four of the 16 ordered pairs from every training record containing a pair. Every entity still appears with other partners. Use independent engineering seeds. For each of the 12 taught pairs, schedule eight teaching packets, each with two event–movement–reveal steps. Within each pair use each ordered opcode combination twice. Shuffle packets in blocks of eight with both opcodes balanced at each step. Each packet starts with two ordinary position observations, then five records per step (event, two unrelated observations, movement, reveal).

**Counterfactual twins.** Start two read-only branches from the same trained weights and the same observed positions. Apply the same preceding opcode to both branches. In one branch the challenge event links the pair, and in the other it releases it. The learner sees only the respective opaque tokens. Both then receive identical unrelated observations and identical first-participant movement. At prediction time their last three records are byte-identical and the second entity's last observed position is the same. Choose the mover's new position to differ from both previous positions. The two ground-truth outcomes must differ. Give no revealing outcome between the preceding opcode and the scored movement. Before scoring, verify that a position oracle using the complete observable prefix and known opcode map distinguishes the twins, and that an oracle restricted to the last three records cannot distinguish their outcomes.

Use four twin pairs per held entity pair, with both prior opcode states represented twice. This includes release after link and relink after release. Use the same probe template on four predeclared taught pairs, for a separate acquisition check. Each branch gets one prediction and starts with a fresh transient packet state. Score the branch's own argmax from its four-position distribution. A tie fails strict accuracy. Score a twin only if **both** branches predict their respective observations correctly. Do not learn from any probe. Freeze the exact scenes and audit the absence of held pairs, positions at reveal, and role-balanced exposure before launch.

## Decision and limits

The primary endpoint is strict both-correct accuracy on held-pair twins. The world is the unit. Report individual world scores, paired means and intervals, and proper log loss on all branch predictions. Show release and relink strata, taught pairs, early acquisition curves, and outcomes where persistence is correct separately. A flat rule that always stays or always follows cannot pass a twin.

Advance to a later retention and transfer study only if:

- ALIGNED reaches at least 75% strict held-twin accuracy, at least 65% in each preceding-state stratum, and at least 80% on taught-pair twins.
- ALIGNED beats SHUFFLED-OP, FROZEN-GATE and NO-PRIOR-WRITE by at least 15 percentage points in each paired mean. For the three positive-effect tests use one-sided paired Student-t lower bounds with Bonferroni correction at family alpha 0.05. Also report descriptive two-sided 95% intervals. Require superiority over the recent-suffix and persistence witnesses on strict twins. The floors are prospective development criteria.
- Oracle solvability, suffix identity, source and exposure audits, strict scoring, finite-state updates and resource ceilings all pass. Report the pooled-count comparison without moving the progression gate after outcomes.

A full pass supports a bounded **E3 held-combination diagnostic** within this supplied pair-event interface, subject to the no-prior-write and shortcut controls. The diagnostic leaves general persistent learning unmeasured. Transient state survives the short observation gap by construction. Sustained retention across new learning, transfer to a fresh ecology, and rule revision require a later locked assay. If ALIGNED ties shuffled credit, the learned-token interpretation is unqualified. If the oracle fails or matched suffixes differ, the test itself is invalid. If the pooled table solves the problem at lower cost, keep it as the reference and require a measurable reason before expanding the learned mechanism.

## Census and qualification

Each world uses 96 training packets and 192 revealing observations. Across eight worlds and four arms, this gives **3,072 charged training packets**, **36,864 consumed training records** and **6,144 prediction–outcome applications**. Of these, 4,608 apply emission updates and 3,072 apply gate updates. The matched evaluation uses 16 held and 16 taught twins per world per arm, **2,048 branch predictions** in total. Each branch consumes seven visible records before prediction, with the target held by the evaluator. The counts above exclude evaluator-only baselines and technical work. If each reference is charged separately, the three fitted references add at most 27,648 training record applications and 4,608 training outcome applications. Six reference scorers add at most 3,072 branch predictions and 21,504 visible record applications. Thus even fully separate accounting gives at most 100,352 record applications, 10,752 training outcome applications and 5,120 read-only branch predictions before qualification. Record actual reuse and both physical and charged counts. Lock emitted probability records, row visits and actual runtime.

Proposed cumulative caps including qualification and charged comparator work: 115,000 charged consumed records, 12,000 training outcome applications, 8,000 read-only branch predictions, 300 CPU seconds, 600 active wall seconds, 512 MiB peak RSS and 64 MiB retained outputs. Measure the full-path technical cost and stop at admission if it cannot fit.

1. **Refine → audit → test the environment.** Use a reference simulator and independent reducer to verify links, releases, relinks, identical suffixes and observability. Exercise both opcode permutations. Validate the held-pair and budget census.
2. **Refine → audit → test the local rule.** Check the one-event numerical derivatives, pre-observation emission, bounded writes, no retroactive state rewrite, matched shuffled-opcode counts and no-write birth. Run a small technical learner fit and exercise positive and negative decision fixtures.
3. **Refine → audit → test the entry point.** Run all arms on disjoint engineering worlds, verify countermodel scores, immutable receipts and archive readback, and dry-hash the eight-world roster. Lock sources and budget before separately authorized official execution.

**Follow-up only after a pass.** First keep the mechanism fixed and compare experienced and fresh learners on the same new stream with matched target feedback. Score old still-valid relations and late new-rule acquisition separately. Vary appearance or token renaming separately from a genuine change in transition rules. Add overlapping pending events and eligibility traces only after a test establishes what observable evidence identifies the relevant cause. Return to language only when a new interface can deliver these learned state updates without supplying the rule.

**Delivery** Design only. No implementation or experiment has run.
