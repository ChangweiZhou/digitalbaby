# MiniFly v5.1-R1 — revision after the three-cycle exercise

20 October 2026. **Proposed revision, not implemented or tested.** The parent [v5.1 design](DESIGN.md) and its received/source snapshots remain unchanged. The [technical report](../minifly_v5_1_dev_20261020/REPORT.md) records what was actually executed.

The goal remains retained exact own-output unseen-addition generalization per all samples and actual compute. Gradual warmup, raw-byte IO, native value memory and the exclusions of attention, Transformers, ELM, backpropagation and arithmetic-specific structure remain in force. This revision does not turn byte discrimination or taught recall into the research goal.

## What the exercise resolved

The matched zero-value native reference passed numerical checks. With isolated, target-independent activity frames, the port learned eight arbitrary variable-length byte replies, including its own EOS, in both engineering births. Exact recall was 100% at each registered checkpoint and after one simulated hour; matched no-write recall was 0%. The selected fast-write schedule ramps gradually from 1/256 to 1/16 over 1,024 external records. This qualifies **only the isolated port's E1 acquisition**, not the complete sequence learner or arithmetic generalization.

The learned byte stage then failed its own diagnostic. All four predeclared rate/bias settings, in both births, stably mapped 12 observed byte types into only eight winning units after 200,000 heard bytes per setting. Fifty-six units never won for an observed type. No context-class learning, fitted-head search, integrated native arithmetic training or scientific confirmation followed. The raw-bit bypass remains distinct, so this is a failure of the intended identity path, not an information-theoretic proof that the entire architecture cannot learn.

The existing competitive rule updates only the winner. Its bias is `gain * usage`, bounded by 0.01 in this grid. An aliased winner can converge to an average of several input patterns while dormant prototypes remain far from the observed stream. Faster updates and more of the same input did not resolve that problem. Repeating the same grid or increasing the KC population is therefore not the next repair.

## R1 change: error-triggered splitting of sensory prototypes

Change M2 only. Keep 64 generic sensory units and the eight raw input afferents; do not assign units to digits, operators or any other byte role. No expected answer, arithmetic value, task ID, pair identity or correctness signal is available to this mechanism.

For each winning unit, maintain local running input mean, per-afferent variance, squared quantization error and observation count. These are statistics of the actually arriving raw bits. Use the same gradual prototype-rate menu and the same 200,000-byte listening cap as v5.1. Do not require all units to be used equally.

At fixed 10,000-byte checkpoints, a unit is eligible for a split only if it has at least 128 observations, its mean squared quantization error exceeds 0.0001, and at least one input variance exceeds 0.0001. These are engineering novelty tolerances on normalized inputs, not arithmetic success thresholds. Select the largest accumulated-error eligible unit. Select a dormant unit that has not won in the preceding 10,000 actual observations; prefer a never-used unit. If none exists, report capacity exhaustion rather than allocating more neurons.

Initialize the dormant prototype from the selected unit's current prototype. Separate the two prototypes along the input coordinate with largest **observed variance**, using a signed displacement that ramps from zero to 0.05 over the next 1,024 arriving bytes. This is a generic statistical split, not a supplied byte codebook: neither prototype is set to a named byte or initialized from a target answer. Continue competitive Hebbian updates at the existing gradual rate. Admit at most one split per checkpoint and at most 16 splits within the listening cap. Log the triggering statistics, identities, displacement, update norms and subsequent assignments.

The purpose is to let unused capacity approach an unresolved sensory mixture. A pure, already selective prototype does not split merely because it wins often. This avoids replacing the old underpowered bias with forced uniform occupancy that would rotate stable identities indefinitely. It also avoids adding a full answer cache, a new output model or a semantic parser.

All splits occur before any native arithmetic-value learning. M2 must still pass zero observed-type aliasing and stability at the same fixed checkpoints before freezing. Context counts start only after that freeze. A split is not evidence of arithmetic abstraction and cannot justify skipping the complete-encoder acquisition test. If the bounded revision still aliases types, stop and report it; do not enlarge the search automatically.

## Controls and next qualification

Compare the unchanged v5.1 M2 against the single R1 splitting intervention on paired births and identical addition-only listening streams. Keep byte/answer exposure and actual CPU visible. Reuse of the earlier listening corpus is additional exposure, not a fresh scientific sample. No held-out answer is added to listening. The R1 choice must be locked before its outcomes; it cannot be selected using unseen-addition accuracy.

Only if the repaired identity path qualifies may the unchanged context-class diagnostic run. Then qualify the **complete learned encoder plus native port** on the distinct generic cue/reply fixtures. Only after those steps can the native addition dose and causal E3 comparison be attempted. The isolated activity-frame test cannot stand in for those integrations.

Before that integration, also finish the bookkeeping identified in this audit: carry actual listening payload and EOS intervals into the initial native clock; account separately for permitted optimization of the still-zero native value state; calibrate M4 on a declared broad target-free listening prefix rather than relying on a few arbitrary port fixtures; and lock those exact bytes/thresholds before any native writes. These paths were not exercised because M2 failed, so they have no qualification certificate.

The original exact own-output E3 endpoint, global held-orbit exclusion, OLD/rest/NEW/rest chronology, causal and recitation controls, retention guard and sample/compute comparisons are unchanged. No arithmetic rule or hand-built representation is introduced. R1 has **no measured performance yet**, no scientific launch authorization and no adoption claim.
