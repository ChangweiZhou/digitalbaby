# MiniFly V216–V218 cross-branch assessment

Date 2026-11-05. Research interpretation based on pushed reports and source inspection. This note records the current evidence and proposed priorities.

The review checked the consolidated branch at 9864bc75b8fd2d67bf76536511238bbb8a06da07 and the parallel codex/minifly-v39-v40-source branch at d30c15cd269f64b9c6a74828a20affdc12e400f7. V217 is the latest result added by that parallel session. We have working learning mechanisms with different strengths. Their capabilities have yet to be demonstrated in one continuous learner.

## Evidence across three routes

| Route | Supported result | Remaining limitation |
| --- | --- | --- |
| V216, raw-byte consequence prediction | More experience and more environment instances improve complete autonomous consequence prediction. At 100,000 events, MANY answers 25/256 versus FEW 6/256. | MANY has 0/128 counterfactual pairs correct on both sides. Its warm-start advantage largely precedes changed-rule updates. |
| V217, state-operation learning | With disjoint content vocabulary, prior experience raises object top-1 accuracy from 60.4% to 98.1%, and person accuracy from 76.3% to 91.0%. All 150 reproduction jobs completed. | Vocabulary reuse exposes role-detection weaknesses. The entity-addressed memory and operation repertoire are supplied. |
| V218, consequence-driven control | Final greedy closed-loop goal-reaching is 99.41%, against 42.58% for shuffled credit and 25.78% for no writing. Complementary goals both succeed in 98.83% of paired tests. | The four-state task is learnable by small tables. The observed-transition reference reaches 100%, and goal-to-sequence recency reaches 98.44%. |

[V216 result receipts](https://github.com/ChangweiZhou/digitalbaby/tree/74d1f1544ae1d0c540cfa44f89bd0f865f6f2a1e/minifly_v216_experience_pretraining_design_20261104/campaign_20261104) · [V217 report](https://github.com/ChangweiZhou/digitalbaby/blob/d30c15cd269f64b9c6a74828a20affdc12e400f7/minifly_v217_statelaw_reproduction/REPORT.md) · [V218 result package](https://github.com/ChangweiZhou/digitalbaby/blob/9864bc75b8fd2d67bf76536511238bbb8a06da07/minifly_v218_goal_conditioned_control_design_20261105/official/RESULT_BUNDLE.zip)

V218's 99.41% figure is frozen greedy evaluation using each learner's own actions and intermediate observations. Actual exploring training success was lower, averaging 68.84% during acquisition, 57.91% during reversal and 63.70% during interference. The package documents two audit/reference interruptions, a replayed first world and incomplete cumulative CPU measurement. Those limits matter for compute-efficiency claims.

## What V217 adds

V217 demonstrates feedback-driven selection of useful state operations. The implementation supplies CO, OWN, SNAP and PTR, an entity-keyed PLACE/LAST store, event boundaries, question parsing and per-attribute updates. It learns slot evidence and reliability among those available operations. That supports conditional learning within the supplied state architecture.

The vocabulary-reuse result is weaker. Under the later span detector, object accuracy changes from 60.4% to 54.6%, with an uncertainty interval crossing zero; person accuracy changes from 76.7% to 65.0%. The evidence shows sensitivity to lexical-role detection. It does not establish uniform negative transfer across detectors.

The original answer-bit calculation uses a gold-dependent fallback and lacks a normalized answer distribution. Those scores cannot establish compression. The audited top-1 choice precedes current-answer feedback, so that scoring defect does not directly supply the current answer to top-1 selection. The output remains token selection, with autonomous byte/EOS generation outside that reproduction.

[V217 report, source and audit boundaries](https://github.com/ChangweiZhou/digitalbaby/blob/d30c15cd269f64b9c6a74828a20affdc12e400f7/minifly_v217_statelaw_reproduction/REPORT.md)

## Overall interpretation

The strongest remaining bottleneck is acquiring and maintaining useful state representations from generic experience. Local memory writes and action–consequence learning have concrete positive evidence. Raw-byte structural reuse remains much weaker.

The operative routes remain distinct. V216 uses Full151-derived fixed sensing and the V181 NORMAL learner. V218 uses the V201/V208 associative action-value route. R_center is absent from both operative runs. V217 has its own token-level state machinery. Successes from these routes cannot be added together as the capabilities of one integrated core.

## Proposed priorities

1. Audit V217 on matched streams by freezing slot typing and operation reliability separately. Determine whether transfer comes from reusable operation choice, improved role detection, or both.
2. Continue fixed-learner experience scaling for V216 while retaining the both-correct counterfactual measure. Quantity and diversity gains motivate the test; low absolute performance constrains its interpretation.
3. Before claiming integration with V218, make its acting policy use the state learned from experience, then intervene on that shared state and measure the effect on actual goal-reaching.

The parallel structural-plasticity memo offers a candidate way to improve shared computation. Its benefit requires comparison with fixed connectivity and budget-matched random rewiring. The proposal itself is not an executed result.

[Structural-plasticity and U-turn reassessment](https://github.com/ChangweiZhou/digitalbaby/blob/b063b26/PROJECT_DOCUMENTS/MINIFLY_END_TO_END_STRUCTURAL_PLASTICITY_AND_UTURN_REASSESSMENT_20261105.md)
