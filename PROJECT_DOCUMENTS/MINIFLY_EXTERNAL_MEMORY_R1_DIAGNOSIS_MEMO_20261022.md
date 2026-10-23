# MiniFly external memory R1: diagnosis and next experiment

Date: 22 October 2026. Status: discussion memo, published at the user's request. This records the diagnosis discussed after the R1 results and the subsequent external audit. It does not authorize implementation, training, another three-cycle exercise or a formal campaign. Frozen results, source locks and the separate addition track are unchanged.

## Current conclusion

**The native-memory design still fails complete recall.** Completing three diagnostic cycles means the tests ran and their artifacts were checked; it does not mean the design works. The actual result is a noisy associative signal, not a reliable arbitrary-byte memory.

| Component | Evidence from the completed R1 diagnostics |
|---|---|
| Item-specific content storage | Final payload-bit accuracy was 91.4% with correct writes, against approximately 49% with no-write or shuffled-answer controls. |
| Exact payload recovery | Even supplying the correct four-byte length would recover only 31.25% of values in that test. This is a diagnostic, not a permitted output shortcut. |
| Stopping | Own EOS was unreliable; complete value plus own EOS accuracy remained 0%. |
| Transformer hybrid | No Transformer host, learned byte bridge or H0/H1 training was implemented or qualified by these tests. |

The evidence is E1 component diagnostics: random target-independent surrogate addresses were explicitly taught arbitrary values, and the same associations were probed. No unseen arithmetic relation was learned or tested here. Exact-key lookup can solve this task after one observation, so even successful E1 recall would not establish arithmetic-rule learning or a general persistent core.

Sources: [R1 report](../minifly_transformer_external_memory_v1_r1_20261022/REPORT.md), [derived metrics](../minifly_transformer_external_memory_v1_r1_20261022/SUMMARY.json), [artifact audit](../minifly_transformer_external_memory_v1_r1_20261022/FINAL_AUDIT.json), and [postcycle design](../minifly_transformer_external_memory_v1_r1_20261022/REVISED_DESIGN.md). These are small technical tests with correlated checkpoints/controls, not confirmation over independently trained hosts.

## What the new audit helps explain

### Interference is a leading explanation, not an isolated causal finding

R1's random-input surrogate produced roughly 10% between-key support overlap at the same slot and clear item-specific content storage. The earlier R2 learned encoder reportedly had substantially greater overlap and very poor exact addition acquisition. This supports interference as an important hypothesis. It does not isolate its contribution because the task, representation and other execution details differ between those experiments.

The approximately 3.6% chance-overlap reference assumes approximately uniform independent cell recruitment at the native activity fraction. Random 88-channel inputs through heterogeneous native PN/KC wiring need not select cells uniformly. Some cells may repeatedly win. Our mean-centering calibration removes average drive offsets; it does not equalize activation variance or winning frequency. Uneven recruitment and correlated drives are hypotheses to measure, not established defects to fix blindly. Unweighted support overlap also need not equal effective write-to-read interference: native gains, signs and routing matter.

### Weak early correction is plausible; pure summation is not yet established

The [implemented residual](../minifly_v5_1_dev_20261020/engine.py) contains `strength * (tanh(margin/2) - target) / 2`. With tiny margins the target dominates, so writes approximately accumulate target-associated patterns. That explains why growing response magnitude need not immediately improve its sign.

Saved checkpoint margins become larger later. Therefore the blanket claim that correction is negligible throughout the exercise is too strong. Checkpoint margins are also not a complete log of margins at each actual teaching event. The improvement at 64 passes confounds presentations, rising strength, consolidation and elapsed native time. A matched target-only accumulation control is needed to identify the prediction-dependent correction's contribution. The evidence does not establish that exact recall will stay at 0% for arbitrarily many passes or that the native lineage cannot support arithmetic with another learned representation.

### A single EOS threshold is too narrow as the main repair

A threshold may offset a response bias, but cannot restore corrupted payload bits. Its behavior across different loads, lengths and delays is untested. There is one positive EOS target versus four negative payload-position EOS targets in the fixed-length assay; overlapping addresses can create a common negative bias. Balancing the teaching classes could address the expected bias without eliminating particular-record interference. Neither calibration nor balanced teaching is already a demonstrated solution.

At 64 passes, Cycle2's EOS-at-slot4 rate was 25%; at its earlier checkpoints it was 0–6.25%. Complete recall was still0%. The distinction matters when interpreting summaries that describe EOS as almost never firing.

## Recommended next experiment

Keep gradual warmup. Postpone expensive Transformer training until these native bottlenecks are better localized. The following is a proposed small factorial diagnosis, not an executable or authorized run plan:

1. **Uneven recruitment:** measure per-cell winning frequencies, drive distributions, same-slot cross-key overlap, cross-slot overlap and signed one-write cross-read influence. Compare the unchanged encoder with one generic training-only activity-normalization arm. Freeze calibration before evaluation; never select addresses using answer correctness or allocate private cells per key.
2. **Summation versus correction:** compare the current residual with a control removing only its prediction-dependent term. Match inputs, targets, schedule and clocks. Record actual write-time residual components. Keep the control explicitly separate from the MiniFly candidate; it diagnoses the rule rather than replacing it silently.
3. **Payload versus EOS interference:** measure acquisition and retention across several loads and variable-length values. Include balanced EOS teaching as a separate proposed arm, retaining the unmodified reference. Any load comparison must expose the fact that record-count warmup changes strength with load; use matched comparisons when attributing a change to one factor. Fit any weights on training data and lock them before evaluation. Never force the answer length or consult its true length during recall.

All observed bytes, repeat presentations, calibration data, CPU and memory costs count. Keep exact own-byte output plus own EOS as the behavioral endpoint; payload-bit gains and forced-length diagnostics cannot substitute for it. Preserve no-write/shuffle controls and include simple accumulation and exact-table references with honest cost accounting. These are generic memory/address/feedback mechanisms, with no arithmetic features or predefined arithmetic rules.

A later bounded learned-bridge/host prototype may legitimately denoise imperfect native signals; raw thresholding failure is not proof that it cannot. But it would need its own causal memory controls, leakage checks and exact-output test. Adding a Transformer now could obscure a storage weakness before we have explained or repaired it.

## Decision boundary

Publish this memo and retain the existing comparison version. Do not launch another run, change frozen experiments, raise write strength, adopt a new core or claim an arithmetic improvement from this discussion. The next request should turn the proposed diagnosis into a locked experiment with explicit arms, exposure accounting and resource bounds.
