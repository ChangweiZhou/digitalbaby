# MiniFly gap sweep

Profile **pilot**. Completed **36** jobs with verified matched panel hashes.

| Block | Seed | Arm | Metric | Score | Interpretation |
| --- | ---: | --- | --- | ---: | --- |
| sparse_fast_slow | 73129 | coupled_sparse | old_rule_accuracy | 0.4757 | Development |
| sparse_fast_slow | 73129 | no_replay | old_rule_accuracy | 0.5069 | Development |
| sparse_fast_slow | 73129 | shuffled_teaching | old_rule_accuracy | 0.4653 | Development |
| separate_wm_gates | 73129 | separate | held_long_gap_accuracy | 0.8542 | Development |
| separate_wm_gates | 73129 | tied | held_long_gap_accuracy | 0.6875 | Development |
| separate_wm_gates | 73129 | always_open | held_long_gap_accuracy | 0.4375 | Development |
| apical_feedback | 73129 | adaptive_apical | held_accuracy | 0.9167 | Development |
| apical_feedback | 73129 | fixed_apical | held_accuracy | 0.9167 | Development |
| apical_feedback | 73129 | shuffled_error | held_accuracy | 0.8021 | Development |
| cerebellar_timing | 73129 | localized_ltd | held_timing_rmse | 0.1979 | Development |
| cerebellar_timing | 73129 | scrambled_ltd | held_timing_rmse | 0.7687 | Development |
| cerebellar_timing | 73129 | linear_ltd | held_timing_rmse | 0.7520 | Development |
| multi_action_credit | 73129 | trace | held_trajectory_success | 1.0000 | Development |
| multi_action_credit | 73129 | immediate | held_trajectory_success | 0.1667 | Development |
| multi_action_credit | 73129 | shifted | held_trajectory_success | 0.0312 | Development |
| grid_graph_transfer | 73129 | structured_grid | held_two_edge_destination_accuracy | 1.0000 | Development |
| grid_graph_transfer | 73129 | shuffled_grid | held_two_edge_destination_accuracy | 0.5521 | Development |
| grid_graph_transfer | 73129 | reset | held_two_edge_destination_accuracy | 0.0000 | Development |
| fb_goal_writes | 73129 | aligned | hidden_goal_choice_accuracy | 0.6875 | Development |
| fb_goal_writes | 73129 | permuted | hidden_goal_choice_accuracy | 0.1042 | Development |
| fb_goal_writes | 73129 | frozen | hidden_goal_choice_accuracy | 0.1042 | Development |
| mb_apl_reversal | 73129 | apl_addressed | held_reversal_and_preservation_accuracy | 0.7604 | Development |
| mb_apl_reversal | 73129 | apl_off | held_reversal_and_preservation_accuracy | 0.7500 | Development |
| mb_apl_reversal | 73129 | dan_permuted | held_reversal_and_preservation_accuracy | 0.5208 | Development |
| sleep_order_scaling | 73129 | ordered_scaled | held_old_rule_accuracy | 0.4792 | Development |
| sleep_order_scaling | 73129 | shuffled_scaled | held_old_rule_accuracy | 0.4896 | Development |
| sleep_order_scaling | 73129 | ordered_unscaled | held_old_rule_accuracy | 0.5312 | Development |
| thalamic_feedback_gate | 73129 | learned | held_cross_order_accuracy | 0.0000 | Development |
| thalamic_feedback_gate | 73129 | constant | held_cross_order_accuracy | 0.0000 | Development |
| thalamic_feedback_gate | 73129 | open_loop | held_cross_order_accuracy | 0.0000 | Development |
| operator_binding | 73129 | attention | held_composed_final_accuracy | 0.0799 | Development |
| operator_binding | 73129 | register | held_composed_final_accuracy | 0.0694 | Development |
| operator_binding | 73129 | static | held_composed_final_accuracy | 0.0764 | Development |
| diagonal_trace | 73129 | trace | held_long_gap_accuracy | 1.0000 | Development |
| diagonal_trace | 73129 | no_trace | held_long_gap_accuracy | 0.4688 | Development |
| diagonal_trace | 73129 | bptt | held_long_gap_accuracy | 1.0000 | Development |

## Scope

- **sparse_fast_slow**. Sparse parametric fast path teaches a shared slow path on retained prompts. Synthetic sequential binary rules.
- **separate_wm_gates**. Learned separate input and read gates with a distractor gap. Supervised development.
- **apical_feedback**. Local hidden mismatch and a trained feedback projection. Small classification proxy.
- **cerebellar_timing**. Climbing-error-inspired depression/potentiation on temporally localized expansion features.
- **multi_action_credit**. Scalar outcome after four decisions. Success requires the whole action sequence.
- **grid_graph_transfer**. Fourier grid transport of learned edge displacement. Synthetic toroidal graph and unseen edge pairs.
- **fb_goal_writes**. Fan-shaped-body-style columnar goal trace with local writes. Engineered proxy without a FlyWire plasticity mask.
- **mb_apl_reversal**. Sparse KC-style features, activity-dependent gain, and compartment-specific reversal. Synthetic proxy.
- **sleep_order_scaling**. Same experiences and replay count. Compare one order ablation and one scaling ablation.
- **thalamic_feedback_gate**. Three shared computation passes with learned, constant, or disabled feedback. Supervised development.
- **operator_binding**. Raw-byte rewrite and isolated learned execution, with held directed operator pairs. The harness supplies stage indicators.
- **diagonal_trace**. Online eligibility on a diagonal state, against detached state and exact BPTT; delayed cue extrapolation.

All scores except timing RMSE are fractions correct. Timing RMSE is lower-is-better.
A successful receipt establishes execution. A mechanism claim requires a capable reference and a paired advantage.
