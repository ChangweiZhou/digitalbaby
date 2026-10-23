# RECOVERABILITY V1: bounded native-memory decoding pilot

**22 October 2026 · Design only · No implementation or execution**

Question: can a small readout, frozen before fresh associations arrive, recover complete values from native memory—and retain relevant OLD values after substantial NEW learning? Compare RAW/STD storage and averaged/separate native outputs. Preserve gradual native warmup and the separate arithmetic track.

## 1. Scope, evidence and rationale

[MEMORY-CHANNEL V2](../minifly_memory_channel_v2_dev_20261022/REPORT.md) found genuine content storage, a favorable payload effect from standardized addressing, and damaging NEW-write interference. Normalized/current-residual final OLD/NEW bit scores were 77.3%/97.3% in one technical world, but whole-value recall with own EOS was 0%/0%. These observations justify a recoverability probe, not a working memory service or a normalization adoption.

The current [native read interface](../minifly_v5_1_dev_20261020/engine.py) averages two output responses per compartment. This pilot tests whether preserving that distinction improves a learned readout. It does not assume the responses contain independent information. The previous snapshot/sequential/time comparisons did not rescue fixed-length recall; keep SNAP and do not repeat that search.

**Evidence ceiling: E1 taught-association recall.** Test labels are arbitrary independent bytes explicitly written to native memory. Fresh keys/values prevent readout memorization of evaluation facts, but do not turn this into E3 arithmetic transfer. Exact-key and canonicalized-key tables can solve the task after one observation; common byte/length priors can explain some empty/default answers. Report those countermodels. The original goal of improved unseen-addition generalization per samples/compute is unchanged and untested here.

This pilot has one readout family and one bounded training budget. No Transformer, attention, ELM, trained key encoder, new native learning rule, protection gate or larger brain is introduced. A negative result ends this pilot; it does not authorize progressively larger decoders.

## 2. Four paired cells

| Cell | Native address selection | Readout observation |
|---|---|---|
| RAW–MEAN | Mean-centered native drive | Repeated average of the two responses |
| RAW–SEP | Same RAW native states | Both responses separately |
| STD–MEAN | Mean/standard-deviation normalized drive | Repeated average |
| STD–SEP | Same STD native states | Both responses separately |

There are **two native storage configurations, not four separately trained stores**. MEAN/SEP read the exact same states and native query outputs. The two readouts for a storage configuration differ only in which information is preserved in their inputs. Each cell has its own offline-trained head, with matched data/budget and three paired initialization seeds. Three fits are readout replications, not three native worlds.

Primary comparisons: STD versus RAW at each observation type, and SEP versus MEAN at each storage type; report their interaction. Do not compare only the strongest combined cell with the weakest reference. Include the inherited direct bit-sign/zero-EOS codec, without offline fitting, on all states.

## 3. Storage and address contract

Reuse the canonical 5,177-cell, 9-compartment v5.1 native engine in float64, `gated=False`, native time constants/side-specific competition, `DT=30/14` seconds, BEGIN per observed record, original prediction-dependent residual and EOS byte-channel mask. Replay remains off.

The native write coefficient remains

`a(n)*(tanh(m/2)-target)/2`,

with `a(n)=(1/16)*(1/16+(15/16)*min(n/1024,1))` and n the completed external records in the current life. **Native teaching continues to use the original averaged native margin m**, even in SEP cells. Readout predictions never enter native feedback or confirmations. No fixed-strength writes, rate search, altered prediction scale or write reweighting.

Inputs are evaluator-owned random 16-byte keys and target-independent tanh-normal 88-channel surrogate vectors for each key/slot0–8. Actual PN/B/KC circuitry produces the memory addresses. These are not learned byte-key representations; success does not qualify the later learned interface. Derive vectors deterministically from key bytes and a frozen public protocol salt, independent of values, cohort, seed labels or memory life. Use the same vectors across RAW/STD and both counterfactual value lives. No key-specific private neurons, answer-dependent address search, raw-value cache or record directory in the candidate.

Calibrate mu_i and population sigma_i on exactly 2,048 unlabeled vectors from a disjoint training-only calibration stream. RAW ranks positive `drive_i-mu_i`; STD ranks positive `(drive_i-mu_i)/max(sigma_i,epsilon)`. Let epsilon be `sqrt(float64_epsilon)*median(positive sigma)`; an all-zero/invalid calibration is an integrity failure. Keep inherited side top-k counts and stable tie-breaking. Freeze statistics before creating episodic values. Persist mu/sigma, receptor vectors, addresses, occupancy and recruitment statistics rather than relying only on regeneration. Calibration work is shared operationally but charged to each compared learner's information/cost account.

## 4. Read once, expose information without pooling

At checkpoint time t, privately clone the native state and BEGIN once. Evaluate all 9 slot addresses at that common horizon, with the same native query equations and neutral reference, no inter-slot ticks and no teaching. Obtain:

`D[j,c,o] = actual[j,c,o] - neutral[j,c,o]`,

for slots j=0…8, compartments c=0…8 and native output index o=0,1. Do not call these two outputs hemispheres without separate anatomical verification. The direct averaged margin must match `mean_o(D)` within a prospectively locked float64 tolerance before training; the inherited reference predictor cancels algebraically, but numerical parity must be checked.

Define SEP input as D. Define MEAN input by replacing **both** output components with their average, retaining the same 9×9×2 tensor shape. MEAN is a fixed projection of SEP; no extra information is inserted by repetition.

Fit one 9-element RMS scale vector per storage configuration from its training-life D tensors, pooling both outputs and all slots/known-query traces without targets. Floor each scale at sqrt(float64_epsilon). Use that same vector for MEAN and SEP; do not fit independent per-output normalizations that change the pooling comparison. Divide by scales without subtracting a learned mean, so the ZERO lesion remains exactly zero. No evaluation normalization, clipping adaptation or running statistics.

Flatten the normalized 162 responses and append a 9-element output-position one-hot at each generation step. The head sees **no raw key, address vector, true length, membership bit, OLD/NEW identity, clock, life index or record-history state**. All9 responses are a transient per-query tensor, cleared afterward. Persistent native state, counters, parameters and caches must hash identically before/after probes. Nine reads are charged even when EOS is emitted immediately.

Fast/slow component views and reference/background probes are deferred: adding them would create another information factor. No per-item choice among read protocols or heads.

## 5. One small learned readout

Use a shared-per-position MLP: **171 inputs → 32 tanh units → 9 logits**, with biases: 5,801 trainable parameters. Eight logits predict payload bits; one predicts EOS. Use a residual output: `logits_j = mean_o(D_normalized[j,:,:]) + W2*tanh(W1*input_j+b1)+b2`. Initialize W2 and b2 to zero; W1 uses the locked paired random initialization and b1=0. The fixed skip has no parameters and reproduces the inherited sign/EOS decoder at initialization because all scales are positive. The learned branch estimates corrections rather than having to rediscover slot-to-bit routing. This retains the existing generic byte-bit codec, not arithmetic structure or semantic features. Both observation types and storage types use exactly the same architecture, parameter count, initialization distribution and optimization budget. Backpropagation trains only this external readout on cached training traces; it never enters the native engine or address selection.

For a training value of length L:

- Payload BCE applies only to the 8 bits at positions j<L, averaged over those bits/positions.
- EOS BCE applies only to actual decisions j≤L, with the sole positive at j=L. Ignore all positions after true EOS in the loss.
- For nonempty values, weight the two averaged losses equally. For L=0 use the EOS loss alone. Average query loss with equal cohort/checkpoint weights as specified below.

Length is used only by the offline training mask/evaluator. At generation it is unknown. Evaluate the head at j=0…8: sigmoid(EOS)>0.5 emits EOS immediately; otherwise use payload-logit signs, with ties decoded0. After 8 emitted bytes, a further payload proposal at j=8 is truncation and wrong; never force EOS or emit a ninth byte. There is no teacher prefix or actual-answer feedback during evaluation. The head has no recurrent state, so no hidden cross-query memory.

Train with AdamW, peak learning rate 0.001, first 100 optimizer updates linearly warmed up, then constant rate; weight decay 0.0001, gradient-norm cap 1, batch 32 whole query traces. Stop at the earlier of 2,000 updates or 2,000,000 supervised binary loss terms per fit, checked before each whole batch. Terms count only unmasked positions; log expected-byte/EOS labels, query reuses and optimizer steps. These are bounded provisional settings, not quality guarantees. No hyperparameter search, early-stopping selection or validation-driven retry. Use the final permitted iterate for each of three initialization seeds, paired across all four cells. All fits receive the same episode/query order and exposure ceilings. Do not ensemble or choose the best seed at evaluation.

## 6. Streams, snapshots and frozen boundaries

All native lives start with empty learned values. Each has 18 OLD and 18 disjoint NEW keys. Each cohort contains exactly 2 values of each length 0–8, with independent uniform payload bytes and randomly permuted length assignment to keys. Fourteen seeded passes per cohort give 252 OLD records and 252 NEW records. Balanced lengths yield 1,260 native events per cohort. Both cohorts are deliberately relevant at final query time.

Chronology: OLD 14 passes →OLD-end probe → 1h simulated rest →OLD rest probe/fork → NEW 14 passes →OLD+NEW-end probes → 1h rest →final OLD+NEW probes. Acquisition checkpoints at 1 and 4 passes are diagnostic; training uses only the six phase pools below. No query teaches. Substantial NEW exposure is retained; the matched no-NEW-write continuation below identifies write-induced damage.

Three disjoint phases:

1. **Training:12 fresh episodes.** Generate W native traces for RAW/STD, with shared facts and input vectors. Cache known-key read responses from six equal-weight pools: OLD at OLD-end; OLD after rest1; OLD/NEW separately at NEW-end; OLD/NEW separately at final. Each episode/storage supplies 108 query traces. Draw batches uniformly over episode, six pools and key within pool. Train/read-normalize only here.
2. **Development:4 fresh episodes.** Freeze all head weights/scales, normalization and address assets first. Execute every cell/control; report all outcomes but do not refit, change seeds, architecture, schedule or roster. Development is an integrity/feasibility report, not model selection.
3. **Assessment:4 fresh counterfactual pairs.** Each pair has the **same 36 keys and surrogate vectors in two separate empty memory lives**, but independently sampled values and independently permuted balanced lengths. The whole head/bridge is frozen before either association set is generated. Report accidental identical targets explicitly; do not manufacture deterministic transformations between values. Both lives use identical record-order schedules. A competent readout must follow its current native state when a key's value/length changes between lives.

Train/development/assessment key sets are disjoint, except the intentional two-life reuse within an assessment pair. No seed or split identifier is a model input. Assessment pairs are the analysis unit; their two value lives, native controls, readout seeds and queries are correlated. Claims are conditional on one anatomy and one frozen calibration asset, not biological or host-training diversity.

## 7. Causal controls and positive references

In each development/assessment episode and storage configuration, run:

- W: correct OLD and NEW native teaching.
- N: same inputs, timing and targets supplied operationally; native value writes/confirmations disabled for both cohorts.
- SH: OLD payloads cyclically deranged within length strata; NEW correct. This preserves per-key event counts/clock. Empty or identical payloads remain coincident and are reported. Score original and actually taught assignments.
- S_NEW: fork W after rest1, receive identical NEW inputs/time with NEW writes/confirmations off. OLD is learned once and shared, not replayed. Compare final W−S_NEW OLD loss alongside NEW acquisition.

The frozen W-trained head is applied unchanged to N/SH/S_NEW. At every W evaluation checkpoint also decode ZERO responses and FOREIGN responses, using a fixed nonidentity cyclic key permutation within the same cohort/checkpoint. Permutations are not chosen by true length or success. These are readonly lesions, not additional native histories. Native masks/timing may use observed values during teaching; neither true length nor assignment identity enters a probe.

Direct bit-sign/zero-EOS decoding is the zero-training baseline. An evaluator-only exact table stores observed key/value records and renders bytes/EOS directly; it is a task-solvability/cost reference, never the candidate's backend. Include a pooled byte/length predictor fitted from training observations. An oracle-fed finite codec check verifies all 256 payload bytes, EOS, empty strings and the 8-byte cap; optional oracle-input tests of the trained head remain evaluator-only diagnostics, not native-memory success.

Eight never-written keys per life receive the same fixed query API. No presence classifier or ABSTAIN action is added here. Report false retrieval/defaults and native-response distributions; empty output is not evidence of absence recognition. The pilot can establish known-key recoverability, not a complete memory service with reliable rejection.

## 8. Joint endpoints and the decision rule

**Co-primary descriptive endpoints:** final OLD and final NEW exact complete value plus own EOS, separately for nonempty values and all values. Report per-length rows, empty defaults, forced-length payload-only diagnostics, bit/byte accuracy, early/late EOS, truncation and margins. Nonempty denominators are 16 facts per cohort per life. Do not trade away OLD to claim NEW success, or hide poor long strings behind empty-value averages.

Reduce queries within life, then the two lives within assessment pair. Show every pair for each of the three frozen head seeds; no best-seed reporting. Compute paired SEP−MEAN and STD−RAW effects under matched inputs and seeds. Report W−N/SH/ZERO/FOREIGN attribution and W−S_NEW retention/plasticity differences. Direct-decoder and readout costs are separate. No confirmatory p-value, universal accuracy percentage or deployment claim is justified by four pairs.

The decision is about **tested recoverability**, not a proof that information exists nowhere else in native state:

| Result | Decision |
|---|---|
| Fresh OLD and NEW nonempty exact both improve over direct decoding and matched causal controls | Report partial causal E1 recovery; consider a fresh learned-key-interface experiment. |
| SEP improves both cohorts over MEAN at fixed storage | Preserving separate outputs is useful for this readout; do not infer independent biological channels. |
| SEP and MEAN tie | No demonstrated value from unpooling; retain simpler observation unless other evidence justifies cost. |
| STD improves one cohort while harming the other | Report a trade-off, not a storage improvement. |
| NEW improves but OLD remains poor against S_NEW | Stop enlarging the decoder; prioritize one prospective native protection/interference intervention. |
| Payload improves while EOS remains poor | A distinct stopping-learning experiment is justified; no extra scalar tuning on these labels. |
| Gains arise only from empty defaults or survive ZERO/FOREIGN/N | Fail native attribution; investigate leakage/priors. |
| Neither trained observation recovers complete values | Stop this readout family; do not escalate automatically to a Transformer. |

For a positive pilot recommendation, require the same direction of OLD and NEW nonempty exact advantage over direct decoding and all four causal controls in **each assessment pair after averaging the three seeds**; disclose seed reversals and absolute accuracy. Zero-effect is the attribution boundary, not a practical-accuracy standard or statistical confirmation. If mixed, report inconclusive; no extra pairs are added. Practical utility requires a separate cost/quality advantage against simpler stores.

## 9. Prospective exposure and resource accounting

One W/N/SH life costs 504 records and 2,520 events. S_NEW adds 252 passive NEW records and 1,260 events, with shared OLD 252 charged to each descendant's information budget.

| Phase | Physical native applications |
|---|---:|
|12 training episodes ×2 storage configurations ×504|12,096|
|4 development episodes ×2 configurations ×(3×504+252)|14,112|
|4 assessment pairs ×2 value lives ×2 configurations ×(3×504+252)|28,224|
|**Maximum total**|**54,432**|

Total main native events 272,160. There are 96 base histories and 24 S_NEW continuations; cloned OLD adds 6,048 logical record-equivalents, for 60,480 information-accounting equivalents. These are not 120 independent worlds. MEAN/SEP/three head seeds reuse the same native traces and do not multiply physical native records. Each fitted cell is charged its complete training exposure, even where computation is shared. The 2,592 physical known-key training traces produce separate repeated supervised exposures during fitting; 12 fits yield at most 24,000 optimizer updates / 24,000,000 binary loss terms. All are prospective maxima, not executed work.

Operational ceiling: one worker/one numerical thread, 1,800 total process-tree CPU seconds, 2,700 wall seconds, 2GiB peak process-tree RSS and 2GiB measured admission headroom. Charge imports, both storage configurations, generation, control probes, twelve head fits, serialization, reduction and packaging. Profile one complete training episode and one head fit plus complete development episode; profile artifacts count once inside the declared roster. Stop before assessment if a conservative remaining-cost projection cannot fit. Do not silently drop cells, seeds, facts or endpoints to finish. Preserve completed artifacts and report RESOURCE_LIMIT; no automatic retry or budget reset.

Nine read evaluations per query serve both input representations. Separate encoding/top-k, native writes/reads, copies/hashes, transfer/normalization, head training/inference and audit costs. Shared transient response caches are cleared per query. Record persistent/fixed/temporary bytes and full scans/sorts; this bounded canonical implementation does not demonstrate sparse computation or large-scale efficiency. Native snapshots persist once per checkpoint for readonly comparisons, not once per head/seed. Caffeinate may follow a future authorized controller while active; this document launches none.

## 10. Locks, roster and delivery boundary

Reserve calibration202610229001; training202610229101–112; development202610229201–204; assessment pairs202610229301–304 with two separately derived value RNG substreams each; head seeds202610229401–403; E0 fixtures202610229501. These are namespaces only: collision-check historical rosters, freeze actual stream manifests and hash all assets before executing. No new facts or associations are generated during this design delivery.

Before a future run, lock dependency versions, exact head initialization/AdamW implementation, query sampling/term-counter semantics, deterministic key-to-vector mapping, native source/dependency closure and complete resource enforcement. E0 checks must cover averaged/separate parity, exact initial residual-head/direct-codec parity, identical parameter counts, no gold-bearing features, 256-byte/EOS/cap round trips, native/head hash isolation, shared-prestate controls and snapshot restore. If an integrity check fails, preserve artifacts and stop; do not fill missing behavior with assumptions.

Deliver raw native tensors, scales/addresses/recruitment statistics, emitted sequences, masked training counters, all frozen head checkpoints, per-pair/cohort/length/control results, independent artifact reduction and a report naming the supported next decision. Integrity acceptance and efficacy remain separate. Historical V2 results and arithmetic sources/pointer remain unchanged.

**Current status: design/specification/static review/source hashes only. Implementations 0, training runs 0, native applications 0, scientific jobs 0.** Implementation, the three-cycle exercise, pilot execution, GitHub publication and any later host campaign require their own explicit requests. This document does not authorize them.
