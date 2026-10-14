# Independent RAW-RULE architecture and next-design critique

13 October 2026. Read-only review of the frozen preparation source, its declared design, preparation report, and historical mechanism records. No training, replay, formal trajectories or source changes were performed.

## What is established

The failure already exists before the new cohort. In the final reserved world, R1-FULL recalls 20.6% of taught cues immediately and 17.6% retained; R4-FULL recalls 20.1% immediately and retained. Their matched frozen encoders recall 24.1% and 18.1% immediately. The unchanged legacy reference recalls 100% immediately and retained. These are preparation fixtures, not a statistical mechanism exclusion.

The new architecture removed the exact associative bank and now asks a single 65-feature linear categorical readout to distinguish the whole cue/answer-prefix population. That is a substantive learning architecture change, not simply a better byte bridge. A 64-dimensional recurrent state has not been shown to be a useful linearly decodable representation of these distinctions at the actual training dose. Conversely, a 65-feature linear readout can classify more than 65 contexts, so its dimension alone does not prove a hard capacity limit.

The implemented encoder derivative is of a squared one-hot regression objective. The categorical predictor's probabilities are computed with softmax only for reporting; softmax cross-entropy is not the training objective. V and P change after every observed answer event, while the encoder changes once per actual observed EOS. FULL confidence is explicitly an approximation once its feature basis moves. These facts are in core.py and were disclosed in the design; they are not newly discovered coding violations.

The frozen twins' poor recall excludes encoder movement as a sufficient explanation of the entire failure. The depth-1 arm's poor recall excludes extra zero-input depth as a sufficient explanation. More tests of confidence or processing depth alone would not directly answer why feedback fails to create a useful representation.

The actual exposure is 512 arithmetic records at the terminal checkpoint, distributed across four operations and many unique cues. The final taught set contains about 200 cues. This is only a few appearances per cue on average, not 512 repeats of each association. Acquisition must therefore be judged at the stated sample budget, rather than assuming that mathematical gradient qualification implies reliable memorization.

## Mechanism hypotheses, not settled causal diagnoses

1. **Representation/head coordination:** the readout's covariance describes older feature coordinates, but the encoder moves those coordinates. The current head has no transport, forgetting or reset operation. This can reduce the benefit of representation learning; the frozen failures require an additional explanation.
2. **Categorical objective:** squared fitting to a one-hot vector differs from optimizing the observed byte's categorical likelihood. Updating a linear least-squares head immediately can also alter the encoder's later learning signal. This remains an empirical question, not a theorem that categorical RLS cannot work.
3. **Cue geometry:** shared REPLY and suffix events may leave small distinctions between prompt states. More processing through a tied zero-input GRU need not amplify relevant distinctions; it can contract them. Saved-state feature and teacher-forced diagnostics can test this without another life.
4. **Generation:** a small first-byte error or a wrong EOS turns a mostly correct sequence into an exact-output failure. Teacher-forced per-position errors versus autonomous rollout must be reported before calling every exact failure a memory failure. Teacher forcing is diagnostic and cannot replace the scored output.

## Revised experiment recommendation

Keep the task families, held relation split, raw bytes, existing six sample budgets, intervening cohort, own complete output plus EOS, and paired sample/CPU measures. Do not add an arithmetic parser, digit values, operation IDs, supplied relations, a numeric auxiliary target or a renderer. Do not restore an episodic lookup merely to make the acquisition table look good: the legacy result already shows that taught recall alone is insufficient for the central goal.

The most direct next mechanism is **synchronous feedback learning of representation and categorical head**, with an objective comparison, on the same depth-1 GRU rather than another unrelated encoder search. A useful fixed roster is:

| Arm | Operation isolated | Interpretation |
| --- | --- | --- |
| Current R1-FULL | Frozen published RAW-RULE implementation | Failure reference |
| JOINT-SQ | Encoder and linear head optimized together at actual EOS using the same squared target loss | Coordination/optimizer recipe change, without changing the objective |
| JOINT-CE | Same representation and update timing, categorical cross-entropy | Categorical objective versus JOINT-SQ |
| FROZEN-CE | Identical initial encoder, encoder fixed, same CE head update | Whether learned representation contributes |
| LEGACY-B | Unchanged historical source | Memory/performance anchor, with disclosed numerical features and episodic bank |

All JOINT losses are computed from the actual feedback byte sequence under teacher-forced previous observations. Weights remain fixed throughout the record; one generic optimizer step occurs at actual EOS, never generated EOS. No-write branches block all changes to head, encoder, moments and counters. All gradient calculations and optimizer steps count as computation; no repeated exposure or replay is free. Head and encoder initialization, parameter count and output alphabet are matched. Use a common predeclared optimizer and scale convention for JOINT-SQ/CE; loss scale must be documented because these losses have different gradient magnitudes. Comparisons with current RLS diagnose a composite learning recipe, not a single isolated loss change.

M001/M007 support the generic recurrent architecture and M024 supports feedback adaptation, but its old locality disqualification must remain recorded. The new joint-backprop operation is an engineered feasibility test, not inherited biological evidence. M050/U047 bounded replay should not be the default remedy: reservoir/recent/fast-head and several newer replay recipes already have mixed or negative findings. A distinct replay operation would need explicit evidence, an actual bounded cache and charged internal updates.

## Acceptance and audit without goal shifting

Taught exact recall, teacher-forced byte/EOS errors, own-output failures, feature geometry and confidence traces are secondary acquisition diagnostics. They are useful only for explaining whether the mechanism is operative. The promotion endpoint remains retained unseen four-operation exact output at the same sample budgets and measured teaching computation, with the same prior-teaching causal and answer-shuffle controls. Better memorization or better conditional token likelihood does not earn the E3 claim.

Do not invent an 80% acquisition cutoff or lower an existing generalization gate after seeing fixtures. Carry the existing comparative DEV and confirmation criteria forward prospectively, and report every operation separately. A partial addition/subtraction gain is a scoped partial result; it is not four-operation mastery. Fully mixed expressions remain a subsequent capability and are not silently claimed from interleaved binary records.

Before science, qualify independent finite-difference derivatives of the complete record objective, no-write/evaluator isolation, generator parity, pure generation, arbitrary byte/EOS behavior, parent/fork independence and actual-budget behavior. The read-only saved-state review should determine which outputs are wrong before finalizing this roster. A technical pass alone cannot admit a large run as though useful learning were established.

The previous measured jobs are seconds rather than hours, but a new joint recipe needs its own complete-job timing. Future cost must be projected from that measured job, including probes/audits/serialization and a margin, inside the 8-hour admission/10-hour hard session boundary. Do not change the sample budgets merely to rescue behavior.
