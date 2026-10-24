# Review of Claude's NUMBER-CODE V1

24 October 2026. Source: `0f41f569b15827bc6259301563bf6b11c9600a9b`, preserved verbatim in [CLAUDE_NUMBER_CODE_V1.md](CLAUDE_NUMBER_CODE_V1.md). The revision is [DESIGN.md](DESIGN.md). This review uses saved scripts/results and algebra; **no training or three-cycle exercise was run**.

## Judgment

Keep the central experiment: **can generic predictive experience improve the representation used for unseen addition?** It is a better discriminating question than assuming another memory mechanism will solve the problem. Counting is legal proposed training data; it must be declared as additional information and charged. It is neither a built-in arithmetic rule nor free experience.

The original does not yet justify “we have explained all failures,” “the sum is impossible in symbol codes,” or “a fixed affine successor creates an addition-ready number concept.” Those are hypotheses. It also changes the endpoint to classes and changes receptors only for digits. Both weaken alignment with the owner's requirements.

## Changes and reasons

| Source issue | Evidence / consequence | Revision |
| --- | --- | --- |
| Rank 19 treated as an exact row-plus-column explanation | `analysis/a1.log` also reports affine fit residual 0.22599. Low rank alone does not identify that specific factorization. | Keep rank/smoothing as a bounded diagnostic; remove universal causal claims. |
| Oracle line/circle comparison described as changing only code | `a3.py` concatenates parsed operand vectors, changes input dimension and uses class-softmax/Adam; it is not receptor substitution through FE-PERM or the V144 byte/EOS learner. | Require a same-interface oracle check in development; no inheritance of the reported 0.64 ceiling. |
| Digit-only receptor substitution | Learner would contain a task-specific byte-category gate. | One learned code/projection rule for all 256 payload bytes; data alone determines the visited rows. |
| Unspecified “three-factor” code learning | No reproducible update order, target detachment, initialization, decoder loss or feedback matrix. | State one exact fixed-feedback proposal and its one-step gradient reference, both using old-state updates and legal observed-byte targets. Its effectiveness remains unproven. |
| Affine fit/composition treated as arithmetic homomorphism | A parabola or exponential affine orbit predicts successors without being a linear additive code; equal codes can fit transitions. | Transition/geometry meters are descriptive. Autonomous held addition decides readiness. |
| Norm constraint claimed to prevent collapse | Equal unit vectors satisfy it. | Growth clipping is only numerical protection; report discrimination, variance/rank and degeneracies separately. |
| Random-start counts overrepresent later digits | A fixed label permutation changes which literal digit is frequent. NO-COUNT cannot isolate order. | Full runs with equal token counts, a fixed-permutation control, and an independently shuffled exact-histogram control. |
| SHUF-COUNT expected not to help | Some permutations or generic pretraining may still help. | Measure its effect; equal gains limit the numeric-order attribution rather than invalidate the data. |
| BP class accuracy is primary; own output deferred | Nineteen-class T-SUM cannot establish own byte/EOS addition. | Keep BP as reference; make complete own-generation the shared endpoint and report the DFA candidate route separately. |
| Parity >=0.90 required for every arm | Source reports line parity 0.41, circle 0.77 and low frozen parity. The gate rejects useful representations by construction. | No universal parity gate. Byte protocol is tested with synthetic fixtures; parity remains historical context. |
| Four-world 0.20/4-of-4 rule resembles confirmation | Neither effect threshold nor small world count establishes population significance. | Report paired relation-level effects and a directional screen, explicitly not confirmation/adoption. |
| Added counting cost excluded from downstream improvement language | Extra data can improve a task without improving total sample/CPU efficiency. | Count all observations/targets; include birth-plus-extra-addition controls and cost-only checkpoint matching. |
| Roadmap installs “+1 b times” or digit-column procedures | Such software would supply the arithmetic structure the learner is meant to discover. | No such algorithm is authorized. Any later recurrence/carry representation must be generic and learned. |

## Scope and versioning

**V145 is the proposed next design, not an executed version.** V144/R2 fixtures, incomplete capacity-screen tooling and V143 parity findings retain their original identities. The reference learner does not become native Full151 by using its input trace. This revision neither runs the pending capacity screen nor modifies a frozen experiment.

The proposed two-CPU-hour ceiling is retained, but the original “seconds to minutes” assurance is removed. The exact development/control roster and its maximum exposures must be materialized and cost-qualified before a main lock. A design document is not a completed dispatcher or qualified experiment. No GitHub push was requested in this turn.

## What should happen next

If this route is selected for implementation, its first useful test is the **actual receptor -> FE-PERM -> own-byte/EOS interface**, including an oracle diagnostic outside candidate code. Only then test the generic code learner and matched counting controls. A positive reference result would justify a separate native integration/retention design. A valid negative result would narrow this particular representation/training hypothesis without reviving arithmetic shortcuts or declaring local learning impossible.
