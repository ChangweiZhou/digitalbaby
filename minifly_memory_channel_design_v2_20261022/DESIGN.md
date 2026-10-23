# MEMORY-CHANNEL V2: determine what native memory can deliver

**22 October 2026 · Design only · Not implemented or executed**

One bounded native memory-channel experiment, followed by a decision about a small learned readout. It precedes substantial Transformer training. The separate arithmetic objective remains better retained unseen-addition generalization per all samples and compute; this experiment diagnoses an E1 prerequisite and does not claim progress on addition-rule learning.

## 1. Decision and scope

The [R1 tests](../minifly_transformer_external_memory_v1_r1_20261022/REPORT.md) stored item-specific information but recovered no complete arbitrary-byte values with their own EOS. Final payload-bit accuracy was 91.4% against approximately 49% for controls; supplying length would recover only 5/16 payloads. Byte recognition in the tested R2 alphabet was repaired; this is not a reason to restart byte formation, change segmentation or enlarge the brain.

The [new review](RECEIVED_REVIEW.md) adds two important omissions: NEW acquisition was not scored alongside OLD retention, and writing/reading used different adaptation trajectories. The previous proposal for a learned bridge was therefore premature. A single EOS threshold is a diagnostic arm, not an assumed solution.

This design asks three separable questions: **Are addresses functionally interfering? Does the prediction-dependent write term help? Does the read protocol expose the stored state compatibly?** It then tests generic EOS calibration on fresh variable-length facts. No new host, arithmetic mechanism, cell count, time constant or stronger write schedule is included. The existing Transformer proposal remains a separate later application.

## 2. Evidence, exposure and attribution contract

- **Claim ceiling: E1.** The same key/value associations are explicitly taught and later recalled after delay and new learning. No scored held-out relation, semantic abstraction or arithmetic transfer is asserted.
- **Inputs and outputs:** evaluator-owned random 16-byte keys and independent uniform 0–255 payload bytes. Keys name facts, not arithmetic features. Primary fixed-length assay uses 4-byte values; the length assay uses 0–8. Output uses its own payload decisions and distinct EOS, maximum 8 payload bytes. At slot 8, EOS is allowed; another payload proposal is truncation and wrong, never a ninth emitted byte. Empty value plus EOS is a taught value, not abstention.
- **Surrogate address boundary:** independent seeded tanh-normal88-channel vectors for each key/slot0–8, projected through actual native B/PN/KC wiring. The evaluator can deterministically regenerate them, but the memory has no key directory or value table. These vectors are not a trained byte encoder or proof that a learned bridge will work. There is no arithmetic parser, target-dependent routing or private allocation per key.
- **Information audit:** count all OLD and NEW presentations, observed bytes/EOS, passive applications, calibration labels and extra perturbation targets. Queries never teach or reveal true length/presence. Generated guesses are not observations. Repeated checkpoints are correlated, not independent lives.
- **Countermodels:** exact raw-key and canonicalized-key tables can recall this task after one observation; no presentation change is tested, so canonicalization adds nothing. A pooled byte/length-frequency baseline, fitted only from observed writes, tests common-format shortcuts. An evaluator-only table is the positive control and cost reference, never part of the candidate.
- **Persistent state:** only native fast/slow values, adaptation and native bookkeeping change during a life. Calibration arrays are frozen before evaluation. All probes operate on private copies, with persistent state/counters/hash invariance checked. Postwrite values exist only in evaluator data and native learned state, not a backup cache.

## 3. What stays fixed

Reuse the [published v5.1 engine](../minifly_v5_1_dev_20261020/engine.py), canonical 5,177-cell anatomy, 9 native compartments, actual activity fraction and side competition, float64, `gated=False`, native time constants and equations. Keep `DT=30/14` seconds, BEGIN adaptation reset once per external write record, EOS masks byte-channel teaching, replay off and the native reader-reference subtraction. This is the tested native baseline, not a newly inherited arithmetic capability.

Gradual native strength remains

`a(n)=(1/16)*(1/16+(15/16)*min(n/1024,1))`,

where n is completed external records in the current life, starting at0. Write coefficient is `a(n)*(tanh(m/2)-y)/2`, with existing masks. No rate search, fixed-strength arm, accelerated ramp or prediction-scale change. Probes and synthetic perturbations on disposable forks do not advance the main life's n. All perturbation information/work is nevertheless charged separately.

All stages use one fixed engine/anatomy and fresh mutable state per declared branch. Seed variation samples facts and surrogate inputs, not independent biological anatomies. Never describe these seeds as independently trained hosts.

## 4. Read protocol: compare three views of the same memory

At each checkpoint t, hash/store the native state and query a private clone. Query addresses depend only on the key and slot, never on expected output or actual length. Read all 9 slots even when an earlier proposal is EOS; generation consumes only its own prefix and the first EOS. Record fixed nine-read cost.

| View | Explicit operation at slot j | What the comparison identifies |
|---|---|---|
| SNAP, inherited baseline | Clone state at t; BEGIN once; call `margins(x_j, DT)` for every j with no ticks between slots. | Reproduces R1's common reset snapshot. |
| SEQ, passive sequence | Clone at t; BEGIN once. At each slot read `margins(x_j, DT)`, then `tick(x_j, coeff=None, dt=DT)` before the next slot. No end-confirmation call. | Read history uses the same passive adaptation transition as native teaching events, without targets or value writes. |
| TIME, time-matched reset | For slot j independently clone at t, passive-rest j*DT, BEGIN, then read `margins(x_j, DT)`. | Matches SEQ's accumulated passive value decay/clock while resetting adaptation. SEQ−TIME isolates sensory/adaptation history under verified passive dynamics; TIME−SNAP measures the short timing change. |

Read functions retain the existing internal observation horizon; do not add a second read-induced tick or replace it with a new formula. Before use, verify passive `tick` and rest have matching fast/slow decay, and that predictions do not mutate query state. SEQ's private clock advances, but the main memory remains at t. No outputs/targets enter its passive transitions. Report if nonlinear/reference behavior prevents the intended isolation rather than labeling every SEQ difference “adaptation.”

Do not select the best view separately for each key, stage or seed. SNAP remains the primary baseline; all three views are preregistered paired diagnostics throughout. A future API change requires an explicit new specification. R1 saved margins/results, not the full native checkpoints needed for this experiment: do not pretend the new probes can be obtained from those files. Future execution uses fresh declared lives and saves checkpoints once, without retraining old R1 trajectories.

## 5. Stage A: finish functional diagnosis with no intervention

Use 2 fresh engineering worlds at M=16, 4-byte values. Stage A retains the R1 baseline calibration procedure with256 target-free frames; these are fresh calibration inputs, not old R1 checkpoints. Three paired histories per world:

- W: correct OLD and NEW writes.
- N: same observations/addresses/timing, all value writes and confirmations off.
- SH: fixed nonzero cyclic permutation of OLD values; correct NEW writes. Preserve the value multiset and score both original and actually taught assignments.

Teach OLD 16 passes; probe after 1/4/16; rest 1h; probe; teach disjoint NEW 16 passes; probe NEW after 1/4/16 and OLD after NEW 1/16; rest 1h; probe both. Increasing NEW from R1's single pass is an explicit change to the diagnostic, shared by controls, not retrospective evidence of a stronger R1 interference test. Each base life has 512 record applications and 2,560 teaching/passive events. After the first rest, also fork W into S_NEW, which receives the identical 256-record NEW stream with value writes/confirmations off. OLD is already learned and is not rerun. W−S_NEW at the matched final clock isolates NEW writes; S_NEW tracks loss with the same NEW sensory exposure and elapsed time. Score NEW on both branches as well. Stage A totals 3,584 physical records/17,920 events, including 512 extra passive continuation records. Charge the shared OLD 256 records to each descendant when comparing information budgets. No new time constant is proposed merely because margins decay.

### Measure functional influence, not just shared support

At empty/OLD-end/final checkpoints in each world, use a prospectively fixed panel: OLD keys 0/1 and source slots 0/3/4, payload compartment 0 and EOS compartment 8, both target signs. These are 6 source addresses ×2 channels ×2 polarities, not an exhaustive matrix over every possible source/channel. Destinations cover every OLD key and all 9 slots. Publish every sampled entry, plus support matrices; never call this the full9-compartment influence tensor.

For each source/polarity/channel, prepare a private clone with BEGIN and a passive prefix through earlier slots of that same source key; then fork into one actual masked residual event and one passive event. Use the current life's a(n). Both forks advance one DT. Reset their probe adaptation before destination SNAP reads. Measure

`I[j,i,c,y] = margin_j(active fork) - margin_j(matched passive fork)`.

Log the coefficient actually applied, self target-aligned response, cross-target signed change, sign flips on previously correct bits, and EOS crossings. Separate within-key/cross-slot from cross-key damage. Normalize off-diagonal effects by diagonal gain only when the gain is nonzero; otherwise report uncontrollable/undefined. Never average away opposite signs using only a norm. Empty-state influence has no destination truth; damage-to-truth is evaluated only at taught checkpoints.

The fixed panel adds 288 masked perturbation events (including passive twins) and 336 passive prefix events across 2 worlds ×3 checkpoints:624 synthetic native events total, separate from full records. It supplies144 synthetic scalar target labels. Include destination-read, cloning and hashing overhead in profiling.

### Instrument actual writes

Before every main event, log margins, target/mask, `u_target=-a*y/2`, `u_pred=a*tanh(m/2)/2`, resulting coefficient and saturation. Record applied fast/slow masses and event-specific read/write adaptation summaries. This distinguishes an accumulation-like early regime from effective correction; checkpoint averages alone cannot identify it. Error maps must include item, slot, bit, EOS, load, age and frame occupancy. Persist raw margins and own emissions.

## 6. Stage B: one storage intervention and one mechanistic control

Proceed only after Stage A's integrity/read isolation checks pass. Poor efficacy does not fail an integrity check. This is a single 2×2 comparison, with only two declared factors; no EOS training change, encoder training or host fitting occurs here.

| Address factor | Write factor |
|---|---|
| RAW: existing per-KC mean-centered drive | RES: current prediction-dependent residual |
| STD: per-KC mean/standard-deviation normalized drive | ADD: remove only the prediction term, coefficient `-a*y/2` with the same masks |

For STD, use 2,048 target-free calibration frames to compute mean mu_i and population standard deviation sigma_i. Score `(drive_i-mu_i)/max(sigma_i, sqrt(float64_epsilon)*median(positive sigma))`; if the median is zero, abort calibration as degenerate. RAW uses the same mu_i and frames but no variance scaling. This Stage B calibration uses more unlabeled frames than Stage A; A-to-B changes are not a causal intervention contrast. Stage C reuses the frozen Stage B RAW calibration. Both keep positive-score eligibility and the same native side-specific top-k counts/tie rule. Freeze statistics before data generation; no target or recall score enters calibration. Save occupancy, recruitment-frequency dispersion, correlations and effective signed influence. STD tests a generic variance/hub hypothesis, not a guarantee of uniform addresses or an overlap-minimization objective.

ADD retains all native nonlinear event equations, decay, adaptation, allocation and timing. It is an ablation of prediction-dependent feedback, **not** a claim that the remaining native system is mathematically a linear Hebbian store. It must not be renamed the new candidate.

Primary block: 4 fresh M=16 worlds, 4-byte values. Each world runs all 4 factor cells under W and SH, plus one N history per address factor (N is identical across write rules), 10 physical histories total. OLD 256 records/rest 1h/NEW 256 records/rest 1h, with the Stage A probe grid. Each world has 5,120 base records. Each of its 4 W factor cells also forks an S_NEW continuation after the first rest, adding 4×256 passive NEW records. Per-world physical total is 6,144; all 4 worlds cost 24,576. Parent OLD exposure is charged to both descendants, but its computation is performed once. References share streams where the intervention permits; charge shared calibration to every compared learner's information budget.

Report single-factor effects at the other factor's baseline, and their interaction, not just the best combined result. The primary diagnostic contrasts are STD-RES versus RAW-RES and RAW-RES versus RAW-ADD at both OLD and NEW endpoints. W−N and W−SH test target-specific content. Overlap reduction alone cannot qualify STD; it must reduce destructive influence or improve usable retrieval. No “winner” is promoted from four engineering worlds.

### Load curve without silently changing the ramp

An auxiliary RAW-RES block uses 1 fresh world at each M=4/8/32, W/N/SH. Each OLD phase and NEW phase has 256 total records: passes 64/32/8 respectively (M=16 reference has16 passes). Thus all loads share the same a(n) history, total observed bytes and total checkpoint clocks. This is explicitly a **fixed-total-budget load/repetition trade-off**, not a pure capacity curve at equal repetitions. Report unique facts and presentations per fact alongside accuracy. Fixed-pass checkpoints are descriptive and have differing dose/time; never attribute those differences only to load. Each auxiliary W also has an S_NEW continuation. Auxiliary cost is 5,376 records. Stage B maximum: 29,952 records/149,760 events. Report final W−S_NEW OLD and NEW contrasts alongside the other paired contrasts; do not label an OLD decline as write interference from the before/after difference alone.

## 7. Stage C: variable length and stopping, without stacking storage changes

Use RAW-RES only, regardless of which Stage B cell looks best. This isolates stopping/calibration from a selected address/learning repair. Keep all three read views. Future combination testing requires a fresh design; no post-hoc best-cell pipeline.

Roster: 4 calibration worlds, 4 development worlds, 8 untouched assessment worlds. Loads are 18 or 36, so every length 0–8 occurs equally often (2 or 4 keys of each length). Training/development each use loads 18/36/18/36; assessment uses 4 of each. Choose lengths randomly balanced within cohort; no key/slot signal receives the true length. OLD and NEW each receive 252 records: 14 passes at M=18 or 7 at M=36. Each phase has 1,260 native events because the average length plus EOS is 5 exactly. Each base life has 504 records / 2,520 events. Each W life also forks an S_NEW continuation after the first rest, adding 252 passive NEW records and 1,260 events with no repeated OLD computation. W/N/SH histories are matched at phase/checkpoint clocks. SH deranges OLD values **within length strata** to preserve per-key event counts; empty/identical values cannot be made distinct and must be counted separately in attribution. No length strata are disclosed at recall.

Probe OLD and NEW acquisition before and after rests/interference; use the first full pass and both phase ends as common checkpoints. Eight distinct never-written keys per life are probed throughout. They are evaluator-labeled only; the memory receives no membership bit.

### One calibrated readout comparison

Compare inherited zero-threshold EOS with a single scalar EOS threshold learned from the 4 W calibration worlds, separately for each preregistered read protocol. Payload sign decoding is unchanged. Fit by minimizing balanced positive/negative classification error over **all 9 slots** in stored-query traces, where the sole positive is the actual EOS slot; tie break by smallest absolute threshold, then smaller numeric threshold. Candidate thresholds are0, distinct sorted margin midpoints and the two exterior thresholds. Freeze thresholds before development; never refit from development/assessment, actual query length or current memory load. Calibration uses exactly these query pools from W: OLD at OLD-end, OLD+NEW at NEW-end, and OLD+NEW at final. Give each world/checkpoint equal total weight. Across the 4 calibration worlds, sum(M)=108: 540 known-key traces and 4,860 EOS-slot labels per protocol, 14,580 labeled slots across all 3 protocols. Count these readout labels separately from native feedback. No absent-key labels are used to fit this EOS threshold. This is an error-minimizing readout fit, not a native update or hidden length rule.

Slots after true EOS are negative classifier examples; no additional native writes occur there. This use of observed lengths is allowed only in calibration labels and exposure accounting. A different threshold per length, load, key, age or stage is forbidden. Test drift across both loads, lengths and delays. Report false early EOS and late/missing EOS separately, not only balanced classification error. A calibration gain must improve whole-value+EOS accuracy on fresh values, not just classifier loss.

No new ABSTAIN/presence channel is implemented in this experiment. Never-written queries measure false retrieval, emissions, confidence distributions and contrast with known-key responses. An empty output is not evidence of recognizing absence. A memory-service or host-readiness claim that requires reliable abstention remains unqualified until a separately trained and controlled presence/readout mechanism is tested. This boundary prevents a length calibration result from masquerading as a complete storage API.

**Balanced native EOS teaching is deferred.** If positives/negatives remain inseparable rather than offset, propose it prospectively as a distinct update-rule arm. Do not silently add reweighting, larger rates or payload-specific changes to this run. Stage C maximum: 28,224 physical record applications/141,120 events, including calibration/control data. Readout fitting exposes additional classifier targets, fully logged, with zero native teaching from those labels.

## 8. Endpoints and the decision this experiment must produce

Primary behavior remains **whole-value exact with own EOS**. Report it for OLD and NEW, immediately and after final interference/rest, under each read view. Include payload-byte/bit accuracy, forced-length payload-only diagnostic, early/late EOS, truncation, false retrieval on absent keys, signed margins and complete emitted traces. Aggregate within each life, then show every paired world row. Read views and forks are not extra independent worlds. There is no confirmatory significance gate or arbitrary90/95% success threshold in this small diagnostic.

| Observed pattern | Decision justified |
|---|---|
| SEQ differs from TIME while passive decay parity holds | Treat adaptation/history compatibility as an interface issue; revise the API prospectively before encoder/host work. |
| TIME differs from SNAP but SEQ≈TIME | Short passive timing explains that read-view effect; do not attribute it to sensory adaptation alone. |
| STD reduces overlap but not destructive influence/behavior | Do not adopt STD just for prettier geometry. |
| STD reduces damage and improves recall across paired worlds | Candidate generic addressing repair; fresh replication required before adoption. |
| RES helps versus ADD at matched dose/clocks | Prediction-dependent term contributes under the tested regime. |
| RES ties/worsens versus ADD | Do not credit correction by name; inspect measured terms before changing gain or scale. |
| OLD stays good while NEW is not acquired above controls | Retention–plasticity remains unresolved; do not claim protected learning. |
| EOS calibration improves stopping but payload remains wrong | Report partial readout repair; no reliable-memory claim. |
| No target-specific signal survives N/SH comparison | Stop before learned decoder/host training; a host cannot reconstruct independent random labels from no measured signal. |
| Target-specific signal survives but direct exact remains imperfect | A small frozen readout probe can be justified; perfect raw thresholding is not required. |

For this boundary, target-aligned payload margin means the within-world mean of m*y over the 4 payload slots and8 payload compartments; EOS is reported separately. The next-readout boundary is descriptive: W target-aligned margin advantage over both N and SH must be positive at OLD-end and final NEW, in each of the 4 primary worlds under one fixed protocol; report absolute values and exceptions. This zero-effect boundary is not proof of reliable recall or a statistical confirmation. If results are mixed, report inconclusive rather than expanding seeds or tuning protocol. The full Transformer campaign is never automatically authorized by any row.

For a later learned readout, train only on separate episodes, freeze before new facts arrive and test identical keys with independently drawn values in two separate lives. Outputs must follow the current native state. Retain no-write/shuffle/zero/foreign controls and the table cost baseline. Repeating this lesson on a learned key encoder requires actual address/influence/read-protocol qualification, not inheritance from random inputs. Neither a diagnostic pass nor E1 recall is E3 arithmetic transfer.

## 9. Integrity, cost and stopping

Prospective main maximum: Stage A 3,584 + B 29,952 + C 28,224 = **61,760 physical full-record applications**, including passive controls and calibration. With the exact balanced-length rosters: **308,800 main native events**, plus **624 synthetic prefix/perturbation events**. Scalar perturbation targets, classifier training targets, queries, table operations and offline calibration frames count separately. There are 103 base histories and 37 additional S_NEW continuations from shared OLD states; these are correlated branches, not 140 independent worlds. Their cloned OLD exposure adds 9,408 logical record-equivalents for per-descendant information accounting: 71,168 total logical record-equivalents versus 61,760 physical applications. Do not charge cloned OLD exposures as newly computed trajectories. Conditional skipped stages are never included as executed counts.

Before execution, freeze the implementation/environment, actual streams, splits, source hashes, checkpoint schema, reader convention and scalar-event parity receipts. Reserve new namespaces: A202610227101–102; Bprimary202610227201–204 and Bload202610227211–213; Ccalibration202610227301–304, Cdevelopment202610227311–314, Cassessment202610227321–328; unlabeled input calibration202610227001. These are reservations only: collision-check historical rosters and freeze generated streams before running. No test associations or numerical results have been generated for this delivery.

Runtime budget: one worker, one numerical thread; 1,800 total process-tree CPU seconds, 2,700 wall seconds, 2GiB peak process-tree RSS and at least 2GiB live system headroom at admission. These are operational ceilings, not scientific thresholds. Profile an entire Stage A world including all controls, 9-slot read views, matrix probes, checkpoint serialization, reduction and audit. Charge the profile's world once as actual execution; do not repeat it for production. Stop if conservative remaining-cost projection cannot fit the unchanged plan. Do not shrink the roster/task, restart a failed life or reset accumulated budgets. Resource exhaustion produces an incomplete report with actual committed counts.

Save native checkpoints at the declared stage ends for readonly view comparisons; record hashes and readonly receipts. Main-state copies are an audit/diagnostic cost, not claimed scalable production behavior. Separate encoding/projection/top-k, native writes, private passive reads, copies/hashes, calibration, reduction and packaging. Report full sweeps/global sorts, fixed/native mutable bytes, temporary bytes and bytes moved where measurable; sparse activity is not sparse computation. Caffeinate may follow an authorized future controller while active; none is launched by this document.

Failure/integrity checks include all 256 bytes plus distinct EOS, cap/truncation, passive zero writes/confirmations, source/clock/read-only invariants, exact stage counts, no cache/target leakage and recomputation from raw rows. A table and oracle-fed finite codec certify task/serialization solvability without supplying candidate responses. If any invariant fails, preserve the failed artifacts, stop and report; do not fill missing endpoints with estimates or call an integrity pass a capability pass.

## 10. Deliverables and current status

A future authorized run must produce: full behavior and error maps for both cohorts; signed sampled influence tensors; actual write-term logs; occupancy/normalization statistics; paired read-view and factor contrasts; variable-length EOS calibration assessment; absent-key diagnostics; exposure/cost ledger; source/checkpoint locks; independent artifact reduction; and a report naming the supported next decision. No automatic extension, host training, core adoption or GitHub upload follows.

**This delivery contains the design, SPEC.json, static document review and source/evidence hashes only. Implemented modules 0, technical runs 0, native training records 0, scientific jobs 0.** It preserves the [R1 diagnosis memo](../PROJECT_DOCUMENTS/MINIFLY_EXTERNAL_MEMORY_R1_DIAGNOSIS_MEMO_20261022.md), frozen prior results and the addition planning pointer. The new proposal is a bounded memory-channel comparison, not another claim that the fly has learned arithmetic.
