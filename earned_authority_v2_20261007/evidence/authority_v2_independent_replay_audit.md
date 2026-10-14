# Earned-authority V2 independent replay audit

**Verdict: prototype replay passes this independent spot audit. Scientific adoption has not passed.**

I implemented FIFO support, help/hurt counters, readout arithmetic and causal update flags independently. The audit imports the frozen input factory, but does not import or execute `ExperienceAuthority`, construct a native core, teach native memories, or rerun a trajectory. Reproduction: `/private/tmp/minifly_next_core_20261006_env/bin/python /private/tmp/authority_v2_independent_replay_audit.py`. The script and detailed result are retained at `/private/tmp/authority_v2_independent_replay_audit.py` and `.json`.

## Checked scope

Three selected completed archived jobs: binary and lifetime world61008101, and the previous V1 Latin technical development world62010003. Across them, the independent calculation checked:

- 5,040 ordinary teacher records, including actual per-store causal write decisions;
- 3,648 allowed observer updates and1,392 disabled updates;
- 41 checkpoint authority states, counters, FIFO order and state digests;
- 1,624 scored query rows, including supported flags, weights and correctness computed only after emission.

The saved replay states and outputs agree exactly. ERROR/Q_HALF arithmetic matches the archived pre-answer outputs. Disabled branches do not increment counters or insert keys; their native calls show false write flags, zero applied plastic writes and equality to the no-write reference. Training uses the recorded actual arriving teacher, not probe answers. Probe queries do not mutate the continuing authority. Checkpoints occur after the proper number of ordinary records; simulated rest changes no observer state in this time-independent rule.

This is a targeted prototype check, not a repeat of the full historical science validation. It cannot establish file authenticity beyond the archived provenance already supplied, nor independently prove that the rule was frozen before replay; the source lock and immutable-rule audit cover that separate matter.

## What the exploratory replay says

The producer replay covers96 previously completed native jobs from48 historical binary/lifetime worlds plus3 prior full DEV jobs. It performs new observer learning on saved forecasts and feedback, but executes **zero new native learning trajectories**. Those old worlds were available during design, so the results are exploratory, not fresh confirmation.

Across the full99-job replay, the candidate half-weights492 final queries and changes49 final answers. That total includes causal controls and does not mean49 improved writing-arm answers.

In historical binary W held-out final responses, only **one of288 answers changes**, correctly. Accuracy is57.986% versus ERROR57.639%; the causal W−N effect is4.167pp versus ERROR3.819pp. Q_HALF remains8.681pp. The learned rule therefore behaves mostly like ERROR and recovers little of the previously observed half-weight benefit.

Historical W lifetime old/new/revision accuracy remains exactly ERROR:97.743%,99.740%,96.354%. This is useful preservation, not evidence of a new memory capability.

The only Latin evidence in this V2 replay is **one previous technical development world**. Its W held-out response improves from0/4 to1/4, while N_old is1/4; causal E3 remains zero. Old/new W remain9/12 and12/16, avoiding V1's additional old-memory loss but not demonstrating Latin reuse. Complete raw64-world Latin score vectors are not available here.

## Interpretation limits

The cache is real additional engineered, fixed-capacity memory:64 four-byte sensory addresses without answers. In these assays it fits all64 lifetime keys,28 Latin keys or18 binary keys, so no cache eviction occurs in the audited lives. It should not be described as Full151's native familiarity signal, unlimited capacity or cue-independent reasoning.

The learned help/hurt statistics use single-context four-output byte forecasts during teaching, then choose a weight for two-option L/R queries. That deployment assumes transfer of readout reliability across decision protocols. It is legal under the sensory interface and labels are not leaked, but has not been empirically established. It helps explain why mere technical correctness does not justify promotion.

The present evidence supports: implementation is auditable, the hypothesis is bounded, E1 is preserved in this exploratory replay, and its reuse gain is too small/unproven to warrant a positive result claim. It does **not** show that learned authority as a mechanism family is impossible, nor that no future structural core improvement can work. No fresh science was launched by this audit.
