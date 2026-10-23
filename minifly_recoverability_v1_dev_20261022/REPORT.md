# RECOVERABILITY V1 — three-cycle report

22 October 2026. **Complete bounded pilot; negative joint recovery result.** All four RAW/STD × MEAN/SEP cells failed the declared OLD-and-NEW criterion. The separate arithmetic track and gradual native warmup were unchanged. No new core, host integration, or arithmetic improvement is qualified.

The experiment separates two problems. Standardized native responses often preserve a recently taught payload, but the learned joint head damages many correct payload bits while changing stopping decisions. After substantial NEW teaching, OLD payload recall also collapses relative to a matched continuation with NEW writes disabled. Increasing decoder capacity is not the supported next action.

## Executed scope and evidence boundary

The [original design](../minifly_recoverability_pilot_design_v1_20261022/DESIGN.md) remained frozen. This request authorized its implementation and bounded pilot as the test payload of three development cycles. Execution covered all 12 training episodes, four development episodes, and four assessment pairs, each with two independently sampled value lives. None was dropped. All 12 readout fits were retained; no seed was selected.

This is **E1: taught-association recovery at known keys**. Evaluation values were fresh to the frozen head, but were explicitly taught to native memory. Keys were mapped to target-independent surrogate receptor vectors, not read by a learned byte-key interface. Exact tables solve this task after observation. These results do not establish E3 unseen-addition generalization, a complete memory service, or useful LLM memory. The original objective remains better unseen-addition generalization per training sample and compute.

## The three cycles

| Cycle | Design and audit | Test result | Revision |
|---|---|---|---|
| 1 | Isolate native equations, read-only observations, codec and external head; check analytic gradients | PASS: four partial synthetic native events; event/state parity within 1e-12, averaged/component error 0; gradient max error 8.87e-11; 5,801 parameters | Resolve an import namespace collision that occurred before any native event; reserve final full-training-loss terms inside the fitting ceiling |
| 2 | Generate the complete training roster; fit four cells × three paired seeds; freeze before evaluation values | 12,096 native record applications, 60,480 events; 12 heads frozen; all fit budgets respected | Add first/fourth-pass acquisition logging to evaluation. Missing early training probes remain missing; no training trajectory was repeated |
| 3 | Execute the complete development/assessment roster and all controls; independently reduce saved outputs | 42,336 applications, 211,680 events; all heads/scales unchanged; four cells fail joint criterion | Preserve native payload signs in a future stopping-only arm; keep native interference diagnosis separate. See [revised design](REVISED_DESIGN.md) |

The bootstrap failure and prelaunch revisions are preserved under [cycles](cycles/). It was a zero-native-event import failure, not a scientific trajectory failure. There was no training restart after exposure. Audits are self-audits plus a separate artifact-reduction process, **not an independent human or agent review**.

## Actual implementation

Canonical native N=5,177, nine compartments, float64, `gated=False`, original native time constants and side top-k competition, replay off. RAW centers native drive; STD also divides by population standard deviation from 2,048 disjoint unlabeled calibration vectors. Actual PN/B/KC circuitry supplies the addresses. MEAN/SEP share exactly the same native states and responses.

Gradual native feedback remained `a(n)*(tanh(m/2)-target)/2`, with `a(n)=(1/16)*(1/16+(15/16)*min(n/1024,1))`, native averaged margin `m`, original EOS payload mask, and `DT=30/14`. The offline readout never supplied native feedback. Each life taught 18 OLD and 18 NEW keys, two values at each length 0–8, fourteen passes per cohort, with one-hour simulated rests before and after NEW.

The external head sees only the saved native response tensor and a slot one-hot: 171 inputs → tanh(32) → nine residual logits, added to a fixed direct-decoder skip. It has 5,801 parameters and no recurrent state, attention, Transformer, ELM, key/value directory, true length, cohort label, or arithmetic features. Training uses masked payload/EOS BCE and NumPy AdamW. Only this external head receives backpropagation.

## Primary assessment results

Percentages below are **complete nonempty value plus the head's own EOS**, averaged within life, then across both lives of each pair and three head seeds, then across four pairs. Each life/cohort has 16 nonempty facts. They are correlated queries, not independent trials.

| Storage / observation | Final OLD | Final NEW | Joint criterion |
|---|---:|---:|---|
| RAW / MEAN | 0.26% | 0.52% | Fail |
| RAW / SEP | 0.26% | 1.56% | Fail |
| STD / MEAN | 0.26% | 3.91% | Fail |
| STD / SEP | 0.00% | 3.91% | Fail |

The criterion required both OLD and NEW to beat direct decoding and N/SH/ZERO/FOREIGN in every assessment pair after averaging the three seeds. No cell met it. Direct final nonempty complete recall was zero in both storage configurations. Unpooling has no demonstrated joint benefit; STD improves some NEW recovery without establishing retained memory. Four assessment pairs do not justify a confirmatory significance or practical-utility claim.

[SUMMARY.json](SUMMARY.json) includes every pair/control result. [DETAIL_METRICS.json](DETAIL_METRICS.json) supplies all development and assessment checkpoint, seed, length, bit/byte, early/late EOS, truncation, actually-taught SH, absent-key and paired factorial rows. The comparison is not restricted to the strongest combined cell.

## Diagnosis: payload damage and stopping are different

The following **posthoc diagnostic** ignores stopping and scores payload signs over the true length. It uses evaluator length only for diagnosis and is not the primary own-output result.

| STD checkpoint | Direct payload exact | MEAN head payload exact | Direct bit accuracy | MEAN head bit accuracy |
|---|---:|---:|---:|---:|
| OLD, immediately after teaching | 54.69% | 14.58% | 97.92% | 87.85% |
| OLD, final after NEW writes | 0.00% | 0.78% | 75.89% | 69.76% |
| NEW, final | 61.72% | 20.57% | 98.20% | 91.53% |
| OLD, matched final with NEW writes off | 53.91% | 9.90% | 97.81% | 82.23% |

Thus a negative whole-value score does not mean the native store never learned payloads. The joint head frequently changes correct payload signs. Its stopping scores improve in rank: on STD final NEW, 99.22% of nonempty queries have a true-EOS score greater than every earlier score, versus 88.28% for direct responses. But that uses an evaluator-chosen separating threshold per query; the actual fixed zero-threshold decisions are valid for only 21.88%. Rank separation is not usable stopping accuracy or permission to tune on assessment labels.

An additional artifact-only recombination keeps direct native payload signs and takes EOS from the already-frozen head. STD final NEW complete nonempty exact rises to 13.28% (MEAN) / 13.80% (SEP), while OLD remains zero. It uses no gold length during generation and no fitting, but **was not a preregistered candidate**. It cannot replace the negative primary result. [PAYLOAD_BYPASS_POSTHOC.json](PAYLOAD_BYPASS_POSTHOC.json) records the full rows.

The matched S_NEW comparison supports write-dependent OLD interference under these observations: time, inputs and passive NEW exposure continue, but OLD payload exact stays near 54% when NEW value writes are disabled. It does not prove information is irrecoverably erased everywhere in native state. Decoder payload regression and native write interference both matter.

## Attribution and countermodels

W/N/within-length SH controls, ZERO/FOREIGN read lesions, and S_NEW continuations all completed. [PAYLOAD_EOS_DIAGNOSIS.json](PAYLOAD_EOS_DIAGNOSIS.json) additionally checks the same key across counterfactual lives. No changed nonempty key was decoded correctly in both lives by any head seed/cell at either final cohort. This is a stricter binding diagnostic than two similar aggregate scores; observed isolated matches do not qualify reliable binding.

Of 144 paired key assignments, 128 changed length and one had an identical value; collisions are disclosed, not resampled. The evaluator-only exact table correctly retrieved all 864 W final known values checked by the reducer. A pooled training byte/length predictor emits empty because balanced length counts tie and the fixed tie-break chooses zero. It scores 11.11% over all values and 0% over nonempty ones. Hence empty/default answers are not recovery. These references are never candidate backends. [REFERENCE_COUNTERMODELS.json](REFERENCE_COUNTERMODELS.json) records their accounting. Eight absent keys per life are diagnostics only; no absence-recognition capability is claimed.

## Exposure, compute and limitations

| Item | Executed |
|---|---:|
| Training native record applications | 12,096 |
| Development applications | 14,112 |
| Assessment applications | 28,224 |
| **Physical total / main native events** | **54,432 / 272,160** |
| E0 partial synthetic events, outside main total | 4 |
| Base histories / correlated S_NEW continuations | 96 / 24 |
| Shared OLD logical exposure equivalents | 6,048 |
| Total logical record equivalents | 60,480 |
| Fitted heads / optimizer updates | 12 / 19,740 |
| Binary supervised terms, including final training-loss diagnostics | 23,992,440 |

Each fit used 1,644–1,647 updates and fewer than 2,000,000 binary terms. There were 2,592 physical training query traces shared between head variants; repeated offline presentations are additional supervision and compute, not new native experiences. Head seeds and readouts do not multiply native histories. Calibration and training exposure must be charged to each compared learner even when computed once.

The controller ledger records 279.24 CPU seconds and 284.28 seconds summed active wall time, including a conservative five-second bootstrap debit. Sampled controller/caffeinate RSS stayed below 0.5 GiB. [DELIVERY_RESOURCE_LEDGER.json](DELIVERY_RESOURCE_LEDGER.json) separately meters reduction/packaging and adds a conservative reserve for previously unmetered artifact work. The ledger does not measure interactive reasoning/waiting time or establish continuous OS-enforced peak memory. All observed work fits the numerical process budgets; no throughput or sparse-compute advantage is established. Encoding uses global scans/sorts and readonly probes use full native clones. Fine-grained stage timers requested in the design were not implemented; do not infer detailed cost shares.

Other limitations: first/fourth-pass training probes were omitted; full dependency closure was not prospectively hashed beyond the listed native sources and wrapper snapshots; four pairs share one canonical anatomy/calibration. [FINAL_AUDIT.json](FINAL_AUDIT.json) accepts artifact consistency, not complete prospective protocol compliance, efficacy, or adoption. Sources listed in the authority/design locks, head hashes and the arithmetic pointer are checked; saved emissions, bit/length scores, counts and pair decisions are independently reconstructed without native calls or fitting.

## Delivery and next decision

[REVISED_DESIGN.md](REVISED_DESIGN.md) proposes an EOS-only correction with immutable native payload signs, retaining the same native warmup and storage protocol. This directly tests the readout failure without adding decoder capacity or pretending to solve OLD interference. A native protection intervention remains a separate, subsequently justified experiment. No further training or larger host was launched.

[RESULT_BUNDLE.zip](RESULT_BUNDLE.zip) is a compact reproducibility bundle: raw response tensors, frozen heads/scales, outputs, metrics, code, reports and source snapshots. Bulky native checkpoints and redundant derived-logit arrays are retained as individual repository files and indexed by [FULL_ARTIFACT_MANIFEST.json](FULL_ARTIFACT_MANIFEST.json). The independent reducer reconstructs omitted logits from frozen weights and raw responses. See [REPRODUCE.md](REPRODUCE.md) for artifact-only commands; they do not repeat learning.
