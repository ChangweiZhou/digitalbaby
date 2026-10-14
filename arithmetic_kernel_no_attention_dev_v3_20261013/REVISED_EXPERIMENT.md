# KERNEL-SEARCH V3 — revision after three preparation cycles

13 October 2026. Three technical cycles completed. Implemented kernels and causal/gradient checks are available; no formal DEV/confirmation or core adoption has occurred. [Preparation report](PREPARATION_REPORT.md) and [original V3 brief](../arithmetic_kernel_no_attention_design_v3_20261013/EXPERIMENT.md).

## Objective and constraints stay fixed

Learn retained exact own-output generalization to unseen four-operation and bounded mixed-expression arithmetic from feedback, with better ability per external sample and actual computation. Raw categorical bytes enter; the model generates payload bytes and its own EOS. All semantic representations are learned. No Transformers, attention, key/value retrieval, content pooling, numeral features, operand roles, arithmetic modules, supplied laws, intermediate traces, numerical auxiliary loss or semantic output renderer.

The implemented forward path uses only learned embeddings, ordinary gated recurrence, fixed local convolution, affine readout and elementwise modulation. Vocabulary softmax is classification, not sequence/memory attention. Correct answers, arithmetic ASTs and group labels exist only in the evaluator. Source/module inspection and 295 contract checks per cycle cover the actual implementation, not only renamed architecture descriptions.

## Implemented comparison and attribution

| Arm | Structure | Parameters | Role |
|---|---|---:|---|
| G0 | Ordinary 64-unit GRU |36,593|Reference|
| G1 | Same GRU, bounded actual-record replay |36,593|Learning-policy comparison|
| G4 | Same GRU, four-record optimizer cadence |36,593|Cadence comparator|
| R1 | One chronological scan + learned work-state update |36,113|R8 loop control|
| R8 | Eight shared scans/work updates |36,113|Reusable-computation candidate|
| RU8 | Eight initially identical, independently trainable transitions |142,961|Parameter-sharing control|
| C |32-channel local recurrent field, radius 1, terminal readout |19,377|Local-computation candidate|
| P |32-unit recurrent reader with bounded feedback modulation |30,897|Learning-kernel candidate|
| P_STOP |Same forward update, policy credit stopped |30,897|Learned-update control|
| P_RESET |Modulation reset at every public frame |30,897|Cross-experience persistence control|

G uses embedding 16 and standard two-bias GRU. R has a32-dimensional fast state and32-dimensional work state, with embedding 16; it rereads every available position, without selecting positions. C repeatedly applies one generic local gated convolution and reads the terminal position after at least L−1 steps. P reads a fixed chronological recurrent path with neuronwise modulation; its learned update is explicitly tanh-bounded. P/G4 update slow parameters at a four-record boundary; the P controls match that cadence. G4 has a different width from P, so it is a recipe comparator, not a perfectly matched plasticity lesion. P_STOP/P_RESET carry the mechanism-specific comparisons.

All use actual reply/EOS mean CE, AdamW 0.001 with gradient norm 1 and whole-displacement RMS cap 0.001 including decay. All parameters are trainable except credit intentionally stopped in the control. No-write blocks decay, moments, modulation, replay/RNG and pending gradients; only instrumentation counters may change. Inference changes transient computation only. Replay stores at most 64 raw permitted records / 4,096 payload bytes and samples before current insertion. No generated guesses or held probes enter the reservoir.

Last 128 raw events are the reencoding context; overflow decisions are counted. R's work-state graph additionally spans the current response, whose admitted training payload is bounded at 64 bytes. Oversized training replies are rejected before mutation. Own generation remains 16 payload bytes plus an EOS decision, with TRUNCATED distinct from EOS. These are generic resource limits, not arithmetic representations or unlimited context/output claims.

Relevant inventory lineages remain recurrence/work-state M001/M007/M010/M102, local topology M016/M094, feedback plasticity/update policy M012/M027/M031 and real replay M050. The new modulation vector is not the previous associative key/value fast matrix. The [full overlapping inventory](../arithmetic_feedback_rule_discovery_dev_v1_20261013/ops/read_only_review_20261013/inventory/FULL_INVENTORY_REVIEW.json) is preserved; this preparation adds no family-wide exclusion or survivor count.

## Data, evidence and cost

Retain V1's 0–15 seed range, opaque permutation of all 256 payload IDs, whole SUM/PRODUCT-family holdouts and canonical mixed-group holdouts. Mixed feedback contains final answer bytes only. Balance all four primitive operations and all 16 ordered inner/root operation pairs; no operator/phase/group side channel reaches the core.

Keep W/N_BINARY/SH_BINARY binary prefixes and W/N_BINARY/SH_BINARY/N_MIXED continuations, with independently owned mutable branches and a reused W prefix for N_MIXED. Generated answers precede feedback. A constant or trained-key table cannot earn a law-learning claim: publish exact/canonical lookup and feedback-only global/length/suffix/bag countermodels alongside the causal comparisons.

The revised technical probes choose distinct groups by a fixed hash order, without using answers or outcomes, then one fixed form/group. Full-dose preparation scores one representative of every held group. A future formal protocol must explicitly decide whether to score all presentations within each group and average by group; this distinction cannot be silently changed after results. Exposure exclusions remain unchanged.

The final runner records fixed binary checkpoints at 256/512/1,024/2,048/4,096/8,192 total records and mixed checkpoints at 64/128/256/512/1,024/2,048. Query CPU is included in each checkpoint's paid model CPU. Exact taught recall, held-case ability, later retained ability, constant/EOS degeneracy and response-length errors are separate quantities. Keep samples, internal replay, target applications, optimizer updates and correlated histories separate.

## Consequence of the preparation

The kernels are executable and the final implementation passed its technical contracts. The short equal-dose worlds do not demonstrate a useful unseen-arithmetic improvement; a longer G0 job shows weak mixed-case behavior but very low primitive held-case ability. These results are not a statistical rejection of R/C/P and do not justify promotion of a new core.

Do not advance a recipe based on lower CE, constant-answer hits, taught recall, one operation, shorter training than its comparator or numerical proximity. No noise objective, easier task or predefined arithmetic structure is introduced to rescue performance.

The next authorized scientific study would compare R8/C/P with G0 and their declared controls on **the same full exposure schedule and fresh paired worlds**, using retained held-case ability/sample and paid-compute endpoints. Selection must occur only in a bounded prospective screen and at most one selected recipe may enter separate fresh confirmation. The statistical comparison family, probe coverage and allocation must be fixed before that screen. This document does not quietly reuse V1's old Transformer roster or 88-job count.

Before such a launch, measure candidate full-job costs, complete FLOP/movement/peak-state accounting and validate the sample/compute reducer against fixtures. Current CPU and circuit-visit counters do not substitute for that complete efficiency implementation. Final source/environment verification must pass; the external data-generator dependency is now explicitly included in the final lock. Earlier technical receipts identified four local source files, not a pre-run all-dependency lock; their narrower provenance is disclosed rather than retroactively upgraded.

No automatic science launch is authorized. Eight-hour admission, ten-hour hard session, detached supervision/caffeinate and whole-job checkpoints remain mandatory for a later authorized run. No retries or extensions. Intended evidence remains bounded retained E3; this technical preparation establishes interface/numerical/causal checks only, with descriptive E1/held-case measurements and no efficiency adoption.
