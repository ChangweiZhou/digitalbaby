# MiniFly V176 — Native deployment of locally learned predictive features

**Identity:** `V176-NATIVE-FEATURE-BRIDGE-V1`  
**Date:** 2026-11-01  
**State:** DESIGN ONLY. No implementation, new model query, pretraining, native fitting, replay, confirmation or adoption has occurred or is authorized by this document.  
**Primary question:** Does a generic feature state learned from genuine byte feedback improve the native memory's own byte/EOS answers on a different text, compared with the same native path using matched-birth frozen features?

## 1. Decision after the involved review

The long-term target remains a bounded learner that forms reusable knowledge from feedback, retains useful knowledge while accepting corrections, and uses that knowledge on never-taught situations per total observations and actual compute. Memory, structure formation and use are three evidence questions, not a requirement to build three independent modules or finish memory before studying learning.

V176 deliberately tests one missing link: **locally formed predictive features → operative native cue encoding → native value learning → native own-byte/EOS output**. It is a staged component-integration diagnostic. It is NOT an online end-to-end persistent learner, a reasoning experiment, a new arithmetic success, or an adopted kernel. Arithmetic and later action remain separate prospective assessments.

Review findings that determine this choice:

- WCFI-V1 restored substantial native acquisition but exposed write-associated OLD loss. Memory is useful, not solved.
- Native V9.9A/B already demonstrated narrow learned-category reuse on unreinforced symbol combinations. Do not say native learning has never generalized.
- V167 and Claude V169 demonstrated local predictive-feature carry-over on two text pairs. Their transfer probes were fitted instruments; their success did not establish native deployment. V169's own predictor remained weaker than matched counting, and continuing B improved A: forgetting is not every experiment's bottleneck.
- V171/SHARED-POLICY's constrained gate on existing native output values produced no consistent complete-answer improvement. V176 does not repeat that output-gate port.
- V175's uncentered coactivity inhibition changed codes but did not yield the registered separation or capability benefit. Its random lateral graph and hypothetical pre-inhibition activity are not a justified structure-learning rule. V176 does not add inhibition, protection or replay.
- A fly-hash retrieval result concerns sensory encoding, not a complete theory of learning/reasoning. Better distinction between memories need not produce better shared predictive structure.

## 2. Claim, exposure and birth boundaries

**Evidence:** E0 actuation/local-update correctness; E1 taught B-context acquisition; a bounded **cross-text predictive-feature deployment** result on exact-context-held-out B examples. The natural-text target is not a deterministic held relation with a unique rule witness: this design does not assign it E3 arithmetic/reasoning status. Retention of taught B examples under rest is E1-R, not E2. No changed-presentation E2 claim is made.

**Exposure:** source A supplies actual arriving bytes and genuine record-end EOS. H and its source prediction machinery may learn from these observations. B-train supplies the expected next byte and EOS for 128 selected contexts. B-held targets are read only by evaluators after the model's first complete response; none reaches any learner, adapter, schedule or checkpoint selector.

**Recitation controls:** an exact taught-key table can solve B-train. It has no matching key for quarantined B-held contexts; global/position frequencies and bounded context counting may still predict them. This is distributional predictive reuse, not proof of an abstract relational program. Exact A/B context overlap is audited before fitting. Other shared substrings and generic language statistics are legitimate prior experience, not secretly claimed absent.

**Fresh native B birth:** A trains an encoder template and source-only predictor; no operative native value bank accumulates A associations. B creates fresh canonical native fast/slow/adaptation banks with inherited frozen H. Its native target counter starts at zero because this is a genuine new bank birth. H's A exposure and all source/probe compute remain inherited costs. This reset/staging is an evaluator intervention, NOT one continuously learning fly or evidence that native A memories survived.

If implementation instead allows native banks to observe A, their target counters must advance and may not be reset at B. That would be a different protocol requiring prospective revision.

## 3. Explicitly allowed and excluded mechanisms

Keep generic bytes0–255, distinct EOS256, gradual applied-update warmup, local/circuit-local candidate learning, native Full151 value memory and fixed C16 output. No attention/Transformer, learned query into a key/value table, ELM candidate, arithmetic feature, task parser, latent-class teacher, task/phase ID input, additional memory-protection replay, new native value rule, homeostasis, extra stopping policy or output-gate controller.

**Training-only prediction head disclosure:** H is formed using a generic local next-observed-byte predictor, inspired by V167. That predictor is real learning machinery with charged labels/parameters/cost, not a free evaluator oracle. Its T/readout are discarded before native B fitting and cannot supply answers, logits, EOS, gains or codes in B. This is the sole scoped auxiliary mechanism in the design. It does not relax the prohibition on an independent operative answer decoder. If the owner intends to prohibit every training-only prediction head as well, this proposal cannot honestly reuse V167's learning signal and must be reframed before execution; do not disguise or silently substitute that head.

## 4. Native source and the exact feature-to-memory port

Native anchor: WCFI-V1 at `52e73caae1259ecd4aa7c95e7ad4cd02d1a9881c`, including `minifly_cls_dev_20261017/fastmem.py`, codebook and its complete dependency/assets lock. Keep:

- 16 canonical compartments, original B/wiring and native fast/slow/adaptation equations;
- FE-PERM fade.95, fresh adaptation convention, theta0, homeostasis OFF, q0;
- DELTA gain1, temperature2 and unchanged signed punishment convention;
- C16 over all257 classes and at most4 own decisions. CAP remains failure; never force EOS.

For the ordinary FE trace p at each actual or prospective native encoding instant:

    x0 = original native per-side TopK(Bᵀ p), with original positive-drive eligibility/tie rule
    h  = ReLU(W_Hᵀ x0 + b_H)                         # 512 dimensions
    pH = native_norm(A h)                            # 88 receptor channels
    xH = original native per-side TopK(Bᵀ pH).

The H arms use xH in the existing native read/write activity route. RAW uses x0. H consumes x0, **never xH**; there is no recursive encoder or learned retrieval operation. The native `values()` prospective clock, teaching code, adaptations and already observed generated/teacher-prefix bytes run unchanged. Own-generated continuations must recompute H and xH from their actual prefix, not use a cached correct-prefix answer.

`native_norm` is the existing function: divide by the maximum when positive; otherwise return the vector unchanged. If Ah is zero, return zero and report it. No raw-input fallback, epsilon trick selected by labels, learned rotation, mixture coefficient or adapter sweep is permitted.

**One fixed sparse adapter:** A has88 rows and512 columns. Every H unit sends weight1/2 to two distinct receptors. Construct from independent SHA256(`V176|bridge`) PCG64 permutations of H indices and receptor indices: for H at permuted position j∈[0,511], connect it to receptors at j mod88 and (j+512) mod88 in the receptor permutation. All H columns have degree2; receptor degrees are11 or12. A is identical across pairs, feature arms and seeds and fixed before any text/label inspection. No adaptive normalization or online bridge plasticity. This intentionally tests one fixed bridge, not an outcome-selected adapter family.

This fixed hash-fold adapter has only88 distinct column patterns; several H coordinates are summed identically. The 512→88 compression and nonnegative tonic drive may destroy the useful feature geometry. It is not512 independent random directions or a geometry-preserving guarantee. That is an explicit bridge hypothesis, not a guaranteed transfer. It is an augmented native cue encoder, not unchanged Full151.

## 5. Source feature learner: source-inspired, not a literal V167 replication

Source reference at Claude `f4146856c3241d94f7fa948e0b7d30fa0b869bc1`:
`minifly_v167_local_dev_v1_20261026/v167_local.py`,
`minifly_v163_sslcola_dev_v1_20261026/v163_lib.py`.

Use the same functional graph: original sensory KC code x0 → H512 ReLU → T2048 ReLU/dropout.5 →257-class softmax. W_H has one row per actual native KC; no learned PAD/task-marker row. H weights start N(0,variance1/184), H biases0. T/readout use the pinned parent PyTorch Linear constructor, fixed generator seed and shared birth arrays across source arms. Record exact runtime/version/initial-array hashes before learning.

For batch mean prediction error e=(p−onehot(y))/batch_size:

    delta_O = e;                      grad_O = presynaptic_Tᵀ e
    delta_T = (e F_T) * dropmask * 1[a_T>0]
    delta_H = (e F_H)            * 1[a_H>0].

Every layer's update uses its presynaptic activity and its own fixed-broadcast-error signal. No gradient crosses a layer; no BPTT, global derivative through native TopK, desired-code target or derivative of a future write is supplied. F_H and F_T are independent fixed Gaussian matrices with variance1/257, pinned by named seeds; biases receive their unit's signal. Locality here means dependence of the update signal, not a claim that flies implement dense broadcast feedback or Adam.

Use batch256; Adam moments beta1=.9,beta2=.999,epsilon1e-8; nominal rate.001 and decoupled decay.0001. Source n_H counts every arriving byte/EOS target, including repeated presentations. Define w(n)=1/16+(15/16)min(n/1024,1). Form moments from the **unscaled** local gradients; then apply the batch-mean warmup to the actual Adam step and decay:

    theta ← (1 − eta*wbar*wd)*theta − eta*wbar*mhat/(sqrt(vhat)+epsilon).

No gradient clipping or coordinate clipping is added. Log proposed/applied norms, warmup, moments, finite values and actual visits. H/T/readout all follow this schedule; FROZEN-H excludes H and its moments/decay from updates. Train exactly3 passes over the source dataset (these are charged repeated optimization/data presentations, not free new unique information or a memory-protection replay treatment), use the final checkpoint, no best-epoch or B-selected checkpoint.

**Declared adaptations:** historical V167 had no gradual warmup, used observed-symbol+OTHER classes and a padded exporter, and inherited a different export configuration. V176 uses actual-step warmup, fixed257 outputs and current FE-PERM x0 with no semantic extra channels. It is a new V167-inspired pretraining variant. Historical3.134/3.262 feature results motivate it but are not promised or inherited.

Source A is minibatch pretraining on authenticated observed-prefix/target examples, with all three passes charged as repeated data/optimization use. It is not one chronological, continuously learning fly. Distinguish unique observed bytes from repeated target applications. Every source prediction precedes use of its expected byte/EOS in the learning rule. FE starts/reset exactly by the declared native record convention. Cache x0 only if it equals the literal live sensory prefix code; do not cache learned h across H updates. A sensory template has no learning value stores; no native acquisition is silently performed during A.

## 6. Six native arms and equal information

| Arm | H during A | Native B cue | Purpose |
|---|---|---|---|
| TRUE-H | Correct, input-matched local feedback | Frozen learned H, fixed A bridge | Main treatment |
| FROZEN-H | H at shared birth; T/readout still train on same A observations | Birth H, same bridge | Primary nonlearning control |
| SHUFFLED-H | Same rule; error rows delivered to H/T permuted within batch; readout keeps true error | Frozen resulting H, same bridge | Secondary feedback-specificity control |
| RAW | No source-H learner | Original native x0 | Historical architecture/cost anchor |
| TRUE-H-NOWRITE | Share frozen TRUE H asset | Same TRUE native cue, identical B observations with value writes inhibited | Time/observation-matched native-learning control |
| FROZEN-H-NOWRITE | Share frozen birth H asset | Same FROZEN native cue, identical B observations with value writes inhibited | Matched control for the feature×value-write interaction |

TRUE/FROZEN/SHUFFLED share source examples, target arrivals, batch order, dropout masks, initial arrays, bridge and native B byte schedule. Use PCG64 seeded from the first16 SHA256 bytes for data/bridge permutations; PyTorch generators use the first8 bytes masked to63bits. Named strings are V176|pair|H-init, T-init, O-init, F-H, F-T, batch, dropout, shuffle; B orders use V176|pair|B-pass|k and probe births V176|pair|probe|space. Use separate explicit RNG streams for batch order, dropout, feedback and shuffling; never allow arm divergence to shift common random draws. A singleton batch has identity permutation and is disclosed. Birth H is identical within a pair; use independently named birth seeds across pairs.

Beating a destructive shuffle is insufficient: TRUE−FROZEN is primary. RAW differs in frontend and has no H pretraining cost; it is not a pure H-learning causal contrast. Report actual RAW cost without fabricating physical A training that did not occur, and compare complete pipeline costs including A for H arms.

All twelve native B banks are distinct matched canonical births, with native target n_N=0. NOWRITE arms execute the identical prediction calls/procedures, target-arrival teacher calls, token feeds and native clocks using write=False. Passive decay and genuine target counters remain active. There is no extra H training for these controls; frozen H is shared as an immutable asset with its corresponding write arm. H, F and A are frozen during B. Native DELTA coefficients are

    w(n_N) * [tanh((v−mean(v))/2) − C_y]/2.

Only native value state learns during B. No A-head output or external probe is reachable in the native generation code path. Remove head objects before B, and qualify assessment parity when all source heads/probes are unavailable.

## 7. Data and the two independent text pairs

Reuse exact corpus editions and split algorithms from the source family, not opened learned checkpoints:

| Pair | A feature learning | B native deployment |
|---|---|---|
| P1 | chesterton-thursday.txt | edgeworth-parents.txt |
| P2 | austen-emma.txt | chesterton-ball.txt |

Corpus SHA256s and source paths are in SPEC.json. Missing files/hash mismatch is `MISSING_SOURCE_ASSET`: stop; do not substitute a different Gutenberg edition, download an unverified corpus, cut a new task or reconstruct old outcomes.

Preserve the authenticated preprocessing: nonempty raw lines, remove original bracketed title under the parent rule, strip trailing CR, split lines longer than128 bytes; bytes plus EOS define a source record. Apply the V167 exporter correction at the first record starting `End of Project Gutenberg` before splitting. Keep A40/5/10KiB and B10/5/10KiB train/calibration/held splits in len(record)+1 symbols, using the parent's exact boundary algorithm. The held block remains the last block. Record exact indices/hashes before any fit. These two text pairs were previously opened for research; they are development fixtures, not fresh independent confirmation texts.

No arbitrary32-byte truncation is added. A cache prefix and B prompt are the actual raw record prefix before a byte, up to128 bytes, with live/native reset and timing parity checked. Payload bytes such as NUL/newline/space are ordinary inputs; no arithmetic protocol is encoded.

**B examples:** from each original B-train and B-held block, enumerate actual next-payload-byte positions (not EOS positions). Group by exact prompt bytes across ALL B splits before selecting. Retain only groups confined to their original split. For held prediction additionally quarantine a prompt if that exact prompt occurred at any scored target in A train/calibration/probe exposure. Never move a quarantined item to training. Existing shared shorter substrings are disclosed.

Within each group use its first chronological occurrence as the scored source target, not its majority label or easiest occurrence. Choose128 train groups and128 held groups by ascending SHA256(`V176|pair|split|prompt_hex`) independently of targets. If insufficient groups survive, stop before fitting. No balancing by answer class, entropy, decoder success or code separability. Publish alternate-label frequencies of repeated contexts as ambiguity diagnostics; natural text is not a deterministic relation task.

Each B record is that prompt → one real payload byte plus EOS. Train128 records for8 fixed, seeded permutations:1024 records/2048 native targets per life, six arms ×two pairs =12,288 native records. The second half includes full-warmup learning. No additional prewarming/replay. This one-payload diagnostic does not establish general variable-length stopping or compositional reasoning.

## 8. Qualification separates feature formation, bridge loss and native deployment

Do not repeat a free-decoder witness and label it native learnability. Do not claim that a soft/autograd TopK relaxation certifies the actual hard native graph.

**C0 design/source/data feasibility, before any main fit:**

- Verify all assets, editions, branch/version allocation, raw vocabulary, data splits/overlap quarantine, topology and declared RNGs; freeze source/runtime/corpus/data manifests.
- Validate auxiliary-head scope, no prohibited retrieval/semantic features and a sole native B answer route.
- Use label-independent engineering contexts to check nonzero code actuation, zero-H handling, native eligibility/populations and latency/byte caps. Finite lookup separability is metadata, not a native reachability gate.

**C1 technical qualification, at most32 complete technical records:**

- RAW reproduces literal WCFI DELTA1; H input x0 uses original p, not xH; same p/time gives identical H/code in prediction and writing.
- Independent explicit local-gradient formulas and Adam actual-step warmup; zero feedback freezes feature error-driven changes (decay handled separately); no cross-layer autograd or hidden global gradient.
- Cached/live code and h parity on exact prefixes, all256 bytes/EOS, no target leakage, head removal, actual own-prefix continuation, writer sign/bounds, clone/frozen-asset independence, full-state roundtrip and read-only generation.
- Write-inhibited native events include passive decay. Partial failed state is saved before assertion; failed observations and costs remain charged.
- Charge a complete source mini-batch, native full record, whole own/TF panel, code export and probe fit before forecasting the complete plan at1.5×. If it cannot fit, `RESOURCE_NOT_ADMITTED` before main learning; no reduction after opening scores.

**C2A fixed source training and bridge qualification:**

Train the six source states (three arms ×two pairs) at the registered dose. This work is actual learner training, not zero-record evaluator preparation. Freeze final H states and optimizer/source receipts.

On A-calibration only, use two prespecified evaluator feature spaces: direct H and bridged xH, for TRUE/FROZEN/SHUFFLED. Use the same source-family2048-ReLU/dropout.5 probe architecture within each feature space, fixed257 output, same initial arrays/seed within each space, three fixed passes, AdamW nominal.001/decay.0001 and actual-step warmup. Backprop is permitted **only in this evaluator probe**, never to H or native state. Fit on the first2048 target positions of A-calibration in original order and score the remaining positions; preserve EOS at its original position, not at an artificial cut. No best epoch, rate grid or refit. Charge every position, source-record touch and probe pass.

Qualification is directional, not a new arbitrary item threshold: for EACH pair, TRUE must have lower aggregate evaluation next-byte NLL than FROZEN both in direct H and in bridged xH. Publish per-record paired differences, zero/tie cases and uncertainty. SHUFFLED is secondary and cannot rescue a TRUE−FROZEN tie/failure. A directional pass is engineering headroom, not confirmation.

- Direct H not better: `SOURCE_FEATURE_SIGNAL_NOT_QUALIFIED_WITH_THIS_PROBE`.
- Direct H better, bridged code not better: `BRIDGED_SIGNAL_NOT_QUALIFIED_WITH_THIS_PROBE`. This cannot distinguish information loss from optimization/readout limits of the fixed probe; the two feature spaces have different dimensions and conditioning.
- A probe numerically fails/does not complete: `PROBE_UNQUALIFIED`, not evidence of missing structure.

Stop before ANY B fitting in these cases. Do not tune A, normalization, H size, passes, source head, probe or data on the failed result. The original historical feature-transfer results remain intact; this adapted recipe and probe may fail to reproduce them.

**C2B native deployment, only if both pairs qualify:**

Execute all twelve native lives to their fixed final B8 checkpoint. Probe the identical128 taught+128 held contexts at B0, B4 and B8; at an additional final128-step rest probe the128 taught contexts only. These are10,752 physical own/TF pairs (9,216+1,536). No labels are supplied during probes; all target losses are evaluator-only. Delay is not intervening learning or continuous persistence.

No checkpoint, bridge, arm, direction or exposure is selected from B-held scores. Complete all scheduled lives unless a technical/resource stop occurs.

## 9. Primary outcomes, controls and permitted decisions

**Co-primary descriptive contrasts at fixed B8:** TRUE−FROZEN in held-context complete own-byte/EOS accuracy, and TRUE−FROZEN in held next-payload-byte NLL. Also preregister the feature×native-write interaction: for accuracy, (TRUE-WRITE−TRUE-NOWRITE)−(FROZEN-WRITE−FROZEN-NOWRITE); for NLL benefit, (L_TRUE-NOWRITE−L_TRUE-WRITE)−(L_FROZEN-NOWRITE−L_FROZEN-WRITE). Publish B0→B8 changes as well. This distinguishes a prior-feature effect on newborn native expression from added usefulness of actual native value writes. Report both pairs independently, all absolute scores, exact paired gains/losses and full learning curves. TRUE−SHUFFLED and comparisons to RAW are secondary. Report taught acquisition, payload accuracy, EOS accuracy, empty EOS/CAP and their errors separately.

No +8-item or Jaccard efficacy gate is introduced. To label a development pattern `NATIVE_FEATURE_DEPLOYMENT_SIGNAL`, complete accuracy must be higher AND payload NLL lower for TRUE than FROZEN in EACH pair, with no technical failure. A tie or opposed endpoints is mixed/unresolved, not a positive result. The stronger label `WRITE_MEDIATED_FEATURE_SIGNAL` additionally requires both interaction benefits to be positive in each pair; otherwise the result cannot be attributed to improved native learning merely because B8 is better. This is a sign-consistency development reading, not a significance test, power claim, confirmation or adoption. One changed answer is1/128 and is shown explicitly.

Always report a target-frequency and exact-key backoff predictor fitted to B-train; a frozen-A counting predictor and A-then-B counting predictor are secondary context-use yardsticks. Use Claude V169's interpolated Kneser–Ney implementation, fixed order5 and discount.9, chosen here from historical results without new fitting. Pin its full code/config before execution; do not run another parameter grid on these B scores. If that exact source/config cannot be authenticated, publish `COUNTING_REFERENCE_UNAVAILABLE` rather than invent one or withhold native results. Counting may exploit every observed prompt byte, whereas native B value teaching is answer-site-only. This is an information-matched predictive yardstick, not a matched update-site mechanism control; report its actual update/target counts. Beating these is not required to demonstrate a relative H effect; efficiency/general structure superiority cannot be claimed while a cheaper comparator remains better.

Causal checks:

- TRUE/FROZEN source exposure and fresh B births isolate prior feature-learning treatment rather than carried native fact values.
- Main matched NOWRITE controls, rather than B0 alone or technical parity, identify total effects of enabling native value writes under identical observations/time. No B0 improvement alone is called improved native learning. Interactions do not prove which synapses encode a semantic rule.
- At final checkpoints, an optional **design-excluded** H-reset/content-scramble lesion would change addresses as well as prior knowledge. It is not silently run here. This primary design uses the stronger matched FROZEN-H training control, not a destructive last-minute reset.
- External probe success supplies evidence of available information only. If B fails despite qualification, report failure of this fixed bridge/native writer/reader package under its dose, not that H contains no structure or local learning is impossible.
- A gain on taught contexts only is E1 acquisition. Held natural-text gains are bounded predictive transfer with recitation/shortcut caveats; no arithmetic-rule/E3 reasoning claim.

Two text pairs with one source birth per pair are development cases, not a population sample. Correlated positions, probes, repeated presentations and native forks are not independent worlds. Do not bootstrap positions into a world-level confidence interval.

## 10. Counts, locality and proposed resources

These are proposed caps for a later explicitly authorized exercise, not execution permission.

| Work | Planned/accounted amount |
|---|---:|
| Source feature states |2 pairs ×3 arms =6 |
| Source A passes |3, fixed final checkpoint |
| Expected source complete-record applications |(705+657)×3×3 =12,258; verify exact authenticated lists before admission |
| Fresh native B lives |2 pairs ×6 arms =12 |
| Native B records |128×8×12 =12,288 |
| Native B observed targets |24,576 |
| Diagnostic probe complete source-record touches |≤2,048, separately charged; no zero-record evaluator exemption |
| Probe fitting target applications |2 pairs×3 arms×2 spaces×3 passes×2048 =73,728 |
| Technical complete records |≤32 |
| Complete-record/touch ceiling |28,000; planned12,258+12,288+2,048+32 =26,626 |
| Total training-target application cap |1,200,000, including all source/native/probe/technical uses |
| Native own/TF query pairs |10,752 main +≤64 technical; cap10,816 |
| Source/probe evaluation labels |count separately; probability-evaluation cap1,600,000 across all phases |
| Native vector reads |cap100,000 |

A/source record counts once per actual full presentation per arm/pass. Probe minibatching does not make its source-record touches or repeated target uses free. Before admission, compute the exact byte/target/record counts from the authenticated manifests; if counts exceed a cap, stop rather than truncate. H arms inherit all A costs; RAW does not get a fictitious free pretraining result.

Propose one worker/one numerical thread;7,200 total process-tree CPU seconds,14,400 wall seconds,1GiB peak RSS,2GiB local output. Carry elapsed time, failed attempts, metadata/probes/native queries/serialization across stages; no resets at a continuation. Caffeinate follows the actual active worker only.

At5177 KCs, deployed H alone has≈2.65M weights (~10.6MB float32); the source H/T/257 head has≈4.23M parameters before gradients and two Adam moments. Forming H from≈184 active inputs costs≈94k additions per position, plus1024 bridge contributions and a second native projection/top-k. Encoding is not a task parser but is also not a tiny free addition or a demonstrated sparse-work implementation. Report global scans/sorts, actual bytes moved and production versus evaluator overhead.

Share frozen H/A/fixed connectivity as immutable assets across query clones; preserve independent native mutable banks. Do not copy10MB of frozen H on every probe. Cache sparse x0 and only needed frozen features, not a history-sized learned lookup disguised as neural state. Hash assets once, qualify mutable roundtrip, and preserve partial failures without trajectory replay.

## 11. What follows, and what does not

- Feature signal missing: resolve the explicit pretraining adaptation/probe before calling it a native deployment failure.
- Bridged signal not qualified: stop this probe/bridge combination. Information loss is not established. Any redesign of the probe or adapter is prospective; no automatic new seed, raw mixture, inverse or larger adapter.
- Signal survives bridge but native output does not benefit: native reachability/learning remains the unresolved link. A different writer or output route is a new design, not a repair hidden in V176.
- Signal improves native held prediction: propose a separately authorized integrated online experiment with matched exposure and representation-drift controls. Do not adopt this staged reset system as the persistent core.

Only AFTER an operative integration signal should a small compliant memory-content-to-next-state computation be proposed. Use ordinary fixed-connectivity recurrent neurons and disconnected/content-shuffled feedback controls; do not resurrect V166's learned query/key-value retrieval. Later held-relation and action tests must expose genuine learned composition, not a planner supplied with the answer or hidden arithmetic structure.

Memory protection and the V175 retrospective value-write×encoder-update factorial remain separate work. They may be worthwhile, but are not arms or uncharged preprocessing in this experiment.

## 12. Delivery and provenance

This design came from three parallel reviews: evidence/constraints, architecture/source feasibility, and protocol/accounting. The reviews agreed on staged deployment, a frozen-H primary control, auxiliary-head disclosure, the hard-TopK learnability caveat, genuine native births and corrected warmup/resource arithmetic. REVIEW.md records the synthesis and unresolved limitations.

Relevant frozen reports/code: WCFI publication52e73caa; V175 results1af8a497 and hostile audit783a5ea; Claude V167/V169 source/report at f4146856. The source, corpus and runtime closure for V176 does not yet exist; do not invent its implementation hashes. SPEC/STATUS distinguish reservations and proposed caps from executed results. No current adopted-core pointer is changed.
