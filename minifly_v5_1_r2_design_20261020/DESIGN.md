# MiniFly v5.1-R2 — integrated repair and an end-to-end addition harness

20 October 2026. **Design only. No new implementation, training, three-cycle exercise, scientific launch or GitHub push is performed by this delivery.** This replaces the proposed M2-only [R1](../minifly_v5_1_design_20261020/REVISED_DESIGN.md). Published v3/v4/v5.1 results and source locks remain unchanged.

**The goal remains better retained exact own-output generalization to unseen addition problems per all training presentations and actual compute.** Byte discrimination, taught recall, plausible syntax and isolated port accuracy are diagnostics. They cannot earn the goal by themselves.

## 1. What changes, and why

| Evidence from the [reconciled audit](../PROJECT_DOCUMENTS/V51_CLAUDE_AUDIT_RECONCILIATION_20261020/AUDIT.json) | R2 decision |
| --- | --- |
| Random, winner-only M2 training kept 12 observed byte types on eight winners; unused prototypes stayed distant | Initialize competitive prototypes from actual observations; exclude dormant units from competition and give both competitive layers the same bounded dead-prototype repair |
| M3 collapsed with perfect byte identities; normalizing its initialization alone did not cure it | Repair M3 initialization and occupancy as well as its scale; test both layers together |
| The port fixture used 2% activity while the candidate uses approximately 3.54% | Read density and side masks from the pinned native birth; measure actual encoder frames before accepting any dose calibration |
| M4 calibration omitted REPLY and its prediction frames; corrected-frame diagnostics greatly reduced cutoff ties without rewiring | Fix frame calibration first; retain KC wiring initially and measure any remaining problem |
| The isolated, disjoint port assay removes interference | Retain it as a wiring check; assess acquisition through the actual learned encoder with overlapping activity |
| V3 achieved taught recall but zero held-out exact accuracy | Keep taught acquisition and generalization separate; run the full addition chain early and score both |

R2 is a native-memory-derived learner with a changed generic encoder, not an unchanged Full151/R_center runtime. The operative output still comes exclusively from native value memory. There is no independent answer model or answer cache.

The next development unit is **one small complete harness**, with component controls alongside it. A component warning must not create another long series of isolated qualification rounds before anyone observes the real learner's answers.

## 2. Constraints and evidence boundary

Keep raw 256-byte payload IO, a distinct EOS, ordinary written answers, gradual warmup and local competitive/native learning. Exclude attention, Transformers, ELM, backpropagation, numeric features, supplied arithmetic relations, designated digit/operand/carry units, arithmetic delimiters in the architecture, numerical error rewards and answer-length hints. No solution traces or reversal of the answer order are introduced.

Learned prototypes and classes must depend only on actually observed training bytes and their contexts. The implementation must not enumerate the addition alphabet, install the audit's perfect identities/classes, or label a prototype by its arithmetic role. Evaluator-only parsing may generate labels, establish held-out orbits and annotate errors; none of it enters the core.

The harness supplies E0 interface checks, E1 acquisition/retention diagnostics and an **exploratory E3 measurement**. Two engineering births do not constitute scientific confirmation. The intended causal E3 claim is incremental reuse of the OLD native teaching episode, conditional on shared listening. A final scientific claim still requires fresh frozen tests, exact own-output improvement, retention, recitation controls and sample/compute accounting.

The [v3 report](../minifly_v3_dev_20261020/REPORT.md) is the historical reference: taught OLD-end accuracy was 100% while held-out final accuracy was 0% in all four arms. Those are earlier engineering results, not performance inherited by R2. Historical COV-R/COV-SUPPORT adoption and other frozen experiments retain their original status.

## 3. Repair both competitive layers with one generic procedure

### Observation-derived initialization

Retain capacities K=64 byte units and at most six context classes. Remove the ineffective `gain * usage` competition bias; retain usage statistics for diagnosis. Do not require uniform occupancy.

Both layers use deterministic, seed-paired **farthest-first initialization from their own observed input vectors**:

1. Keep a bounded reservoir of actual vectors. Pick the first seed by the birth RNG from that reservoir. Pick subsequent seeds by maximum distance to the nearest already chosen seed, with observation order resolving exact ties.
2. Stop at capacity or when no distinct vector remains within the numerical tolerance established by the precision test. Never add duplicate zero placeholders to the competing set. Unused capacity stays inactive.
3. After initialization, retain winner-only local updates `w <- w + eta * (v - w)` with the gradual rate below. No gradient is propagated from the answer loss through these layers.
4. At fixed 4,096-payload-byte checkpoints during listening only, a prototype that received no wins in a complete block **since its initialization** may be reseeded from the worst-reconstructed actually observed vector. Reseed at most one unit per layer per checkpoint, and only if at least 1,024 observations remain before its fixed freeze. A unit is not reseeded simply because its usage is low. Record the old/new state and triggering observations.

Use mean squared distance within each layer. The byte input is its eight signed raw bits. The context input is the normalized profile defined below. Maximum reservoirs are 256 observed byte vectors and 64 current context profiles, respectively. These are bounded sensory statistics, not stored cue–answer relations.

Initializing from observations is an explicit change from random birth prototypes. It supplies no arithmetic grouping or procedure. It is expected to make deterministic sensory identity easier when observed diversity fits K; that is an interface repair, not arithmetic learning. All 256 payload values remain valid through the raw-bit bypass; 64 learned units cannot promise an injective winner for an arbitrary 256-symbol stream.

### Gradual formation and freeze

A newly seeded/reseeded unit is introduced through an output-activation gate that rises from zero to one over the next 1,024 actually observed payload bytes. Its local prototype learning can begin immediately; the gate controls its downstream contribution. Initialize M2 after the first 1,024 D0-A payload bytes so its activation ramp finishes before the 4,096-byte freeze. Initialize M3 after its first 4,096 D0-B bytes. Late reseeding cannot extend a phase beyond its cap. Native value teaching is disabled throughout representation formation. Prototype learning intensity rises smoothly from `eta_max/16` to `eta_max`; use eta_max=0.001 in both layers initially, with a 4,096-observation ramp. No rate grid is selected by addition accuracy in this harness.

M2 identities freeze before M3 context accumulation. M3 and M4 thresholds freeze before any native value writes. There is no post-freeze low-rate representation drift. Newly appearing bytes after freeze use the frozen sensory computation and raw-bit bypass, without allocating new units during a query.

Record observed-type confusion, reconstruction error, prototype movement, usage, reseeding and assignment stability. Exact identity separation is informative, but raw-bit access means an alias warning alone does not prove total information loss. Do not force a desired number of winners or a semantic partition to pass a check.

### Context profiles and class count

Use immediate predecessor/successor counts over frozen M2 winners, without crossing BEGIN/EOS. Increment a successor count only when the successor actually arrives. For units with both contexts observed, use:

```text
p_u = 0.5 * concat(left_u / sum(left_u), right_u / sum(right_u))
```

Profiles and class prototypes therefore use the same unit-sum scale. An insufficiently observed context has an explicit unknown class; no future byte fills the missing context.

Treat C=1…6 as a finite generic model-selection set, not six arithmetic hypotheses. Seed each case from actual profiles after the first 4,096 context payload bytes, then update it locally over the remaining context stream. All cases use the same profiles and rate. Select C using a separate listening-validation prefix: next-winner log loss plus the already declared approximate two-part/BIC penalty, counting the candidate's retained class prototypes and assignments even when a prototype receives no wins. Use smoothing 0.5, class degrees of freedom `3*C*(A-1)`, identity-comparator degrees of freedom `A*(A-1)`, parameter penalty `0.5*dof*log2(N)` and class assignment cost `(A+C)*log2(6)` bits. Here A is observed byte-unit count and N is context-training transition count. This is an engineering model-selection criterion, not a theorem about description length or arithmetic capacity.

Choose the smallest C on a numerical tie. Do not force C≥2, tune on semantic digit/operator annotations, or inspect held addition accuracy to choose C. If one class wins, report it and assess the full learner's behavior with that result. Internal repeated profile/validation views and their compute count as learning/model-selection work. Perfect identity/class fixtures remain separate diagnostic controls, never candidate state.

## 4. Preserve one causal sequence and native output path

Retain the v5.1 generic open segment, four closed registers, right-relative buffer coordinates, raw-bit bypass, learned byte-unit inputs, boundary-age cells, generic reply-step cells and previous-byte inputs. Input limit is 256 payload bytes, open/closed segment limit 16 bytes each, and generation limit 16 payload bytes plus its own EOS decision. Overflow/truncation is explicit and incorrect.

A class-change boundary closes the preceding open segment and starts a new open segment **with the arriving boundary byte retained**. REPLY resets the generic reply-step counter and does not close a segment. At the first answer frame, the final prompt byte remains in the open segment. The fly's actually emitted first byte may cause the next closure; the expected byte cannot cause an earlier one. Reply step and boundary age remain distinct.

At every answer event:

```text
prepare from available bytes/state -> compute native value/reference ->
reveal actual training target OR emit own token -> apply permitted native event ->
feed the actual observed/emitted byte -> advance the reply step
```

Use one immutable prefeedback frame for readout and teaching. The target must not change the current address. Generated guesses never become observations. Byte targets teach eight bits and EOS-off; terminal EOS teaches only EOS-on. Generation has no digit mask, expected length or analytic correction.

Retain the matched zero-value physiological reference, native temperature T=2 and signed residual:

```text
r = native_read(x, zero learned fast/slow deviations, current adaptation)
s = a(n) * (tanh((z-r)/2) - target) / 2
```

The reference has no moving mean or trainable answer offset. It uses the same native clipping, activity, time and adaptation as the real read. Queries use a private read view and cannot commit changes to any learned state, q, thresholds, prototype statistics or replay bookkeeping. Retain numerical bridge and readonly-state tests at 5,177 and 65,536 cells where applicable; an unchanged primitive certificate does not qualify the whole encoder.

## 5. Calibrate the frames that will actually be used

Retain N=65,536 and existing generic KC family/degree/weight distributions initially. Use the pinned native `active_fraction` and actual side masks. The present birth value is approximately 0.035355339; read it from the source and abort on an unexpected source mismatch. Never substitute the previous fixture's hard-coded 0.02 and call it density-matched.

One shared frame iterator must serve training, generation and threshold calibration:

| Event | Temporary state used for calibration and the native event |
| --- | --- |
| Observed prompt byte | Feed it, then encode the same passive prompt frame used by the actual learner |
| REPLY | Set reply step to zero; no payload or hidden native interval |
| Observed answer byte | Encode the prefeedback frame, calibrate its threshold statistics, then feed the observed byte and increment reply step |
| EOS | Encode the terminal prediction frame; count the actual EOS interval; do not invent a payload byte |

Calibrate on an actual, broad training-stream prefix after M2/M3 freeze, with native value writes disabled. Include all output steps and terminal frames. Record payload bytes and calibration-event counts separately. Carry listening bytes and EOS intervals into the native clock using dt=30/14 simulated seconds per event. Skipping still-zero native-state work requires an explicit equivalence check under the inherited BEGIN adaptation reset; otherwise execute and charge it.

Measure same-prompt stability, pairwise shared activity, common-to-all activity, input-family contributions, active counts, cutoff values and selected/available cutoff ties, before and after calibration. Shared activity is not intrinsically wrong: useful reuse requires selective sharing. Do not demand orthogonal whole-problem keys.

Keep stable deterministic tie resolution initially. The matched diagnostic already reduced huge ties through corrected-frame calibration. Fresh randomness on each query is forbidden because it moves addresses. Any additional wiring rebalance, neuron heterogeneity or tie-breaking intervention needs residual evidence and a separate declared comparison; none is silently added to R2.

Before interpreting a dose, measure native directional/response curves on actual encoder frames at observed low/median/high active counts, using passive controls and both target signs. The disjoint fixture remains a unit test only. The provisional native schedule is unchanged: `a(n)=(1/16)*[1/16+(15/16)*min(n/1024,1)]`. Its ceiling is not declared effective or safe for the complete learner until these real-frame measurements are available. Do not replace gradual warmup with full-strength teaching after a failed test.

## 6. One bounded development harness

The next implementation should exercise components and the real chain in the same harness. This is a prospective engineering assay, not authorization to execute it now.

### Data and exposure

Retain ordinary canonical decimal `a+b=` and normal-order answer bytes, operands 0–9999, equal and unequal lengths. No new task or arithmetic-controlled curriculum is introduced. Before any listening, calibration or teaching, partition unordered operand pairs globally; both orientations share a partition. Held pairs are excluded from every answer-bearing learning source. The core sees payload and typed controls only.

Use two paired engineering births reserved as 202610205301 and 202610205302; audit prior roster collisions before generating data. For each birth, draw one immutable training stream. The representation prefixes are bounded by whole-record payload budgets:

| Prefix | Payload-byte ceiling | Purpose |
| --- | ---: | --- |
| D0-A | 4,096 | M2 observation-derived initialization and gradual formation; then freeze |
| D0-B | 16,384 | M3 profiles, initialization after its first 4,096 bytes, local class learning |
| D0-C | 4,096 | Generic class-count selection; no arithmetic-score selection |
| D0-D | 4,096 | M4 calibration on the actual passive and prefeedback frames; then freeze |

Do not cut a source record to hit the ceiling. Report actual bytes, observed answer bytes/EOS, records, repeats and internal class-model views. All correct answers heard during D0 are training information. Charge the complete representation/model-selection cost to each branch, even when a paired immutable prestate is computed once operationally.

For the integration assay, choose 16 OLD training orbits and eight disjoint NEW training orbits. OLD has 512 presentations, 32 seeded shuffled passes over its 16 fixed examples. NEW has 128 presentations, 16 passes over eight examples. Repetition is deliberate for acquisition diagnosis and every presentation counts. It is not 512 independent facts. Select 64 held orbits, with length-stratified sampling, excluded globally from D0 and both teaching cohorts. Probe both orientations where distinct, averaging within orbit first.

This small dose is an end-to-end diagnostic, not a powered test of arithmetic rule discovery over four-digit inputs. A null held result cannot establish impossibility or justify an unplanned dose increase. The final research endpoint remains exact unseen addition; demonstrating taught recall only identifies that the chain can acquire observed feedback.

### Arms and histories

Use four correlated histories per birth, processed sequentially on one numerical thread:

| History | Encoder | Native allocation | OLD feedback | NEW feedback |
| --- | --- | --- | --- | --- |
| R-W | Repaired, learned | Strict v5.1 confirmation gate | Correct, writes on | Correct, writes on |
| A-W | Same repaired prestate | Pinned native allocation | Correct, writes on | Correct, writes on |
| R-N_OLD | Same repaired prestate | Strict gate | Same bytes/clocks, native writes and confirmations off | Correct, writes on |
| R-SH_OLD | Same repaired prestate | Strict gate | Locked within-length shuffled answers | Correct, writes on |

R-W versus A-W tests the allocation hypothesis only after the shared encoder repair. R-W versus R-N_OLD tests incremental use of prior native teaching conditional on D0. The shuffled branch preserves answer-length/target multisets and clocks, with no hidden correct-label loss. Do not count the four histories as independent n.

Retain the strict gate's signed q, q0=3, 86,400-second decay, 60-second spacing, eight-identity recent ring, pre-event slow allocation and at-most-one completed-record confirmation. Mixed-sign suppression and the fact that frequent format bits may consolidate are hypotheses to measure. Do not change those rules to rescue this integration test. **Disable replay in this first harness** so it cannot confound acquisition or manufacture confirmations; report this deliberate difference from the larger parent protocol. Native rest/decay is retained.

Chronology: shared D0 → OLD → one simulated hour → NEW → one simulated hour → final probes. Log exact own-output taught and held curves before teaching and at OLD presentations 16, 64, 256 and 512; after the first rest; at NEW 64 and 128; and after the final rest. Use a fixed eight-orbit held subset and eight taught probes at intermediate checkpoints; score all 64 held orbits and all 16 OLD taught items at OLD-end and final. Probes never supply feedback.

Keep the known v3 path as a historical diagnostic reference. Any new execution of it must use an explicitly shared supported range/IO contract and count its costs; no historical recall number is pasted into an R2 arm. No perfect-input or analytic-output helper may supply the scored candidate answer.

## 7. Read the whole learning curve, not another isolated pass

Measure concurrently:

- observed-byte confusion, occupied prototypes/classes, context reconstruction and boundary/run statistics;
- actual prediction-frame coverage, stability, activity/family contributions and cutoff ties;
- teacher-forced residual/byte curves **and** own-prefix generated bytes, predicted EOS/length, truncation and exact whole answers;
- early/recent taught retention, held exact accuracy and the prior-teaching/shuffled contrasts;
- q/sign/spacing statistics, fast/slow write mass, common-format versus input-dependent predictive utility;
- all exposure, production/evaluation/model-selection CPU, peak RSS, simulated time and failures.

Missing/extra bytes, wrong EOS, invalid decimal rendering or truncation make an exact answer incorrect. Partial diagnostics do not replace the endpoint. Countermodels are exact prompt lookup, evaluator-canonicalized pair lookup, output frequency/length and fixed suffix backoff 0–4, trained only on allowed exposure. Learned reusable composition is legitimate E3; a supplied arithmetic decoder is not.

| Observed pattern | Next diagnosis supported |
| --- | --- |
| Isolated port works; real encoder taught recall remains poor | Addressing, interference, write magnitude or frame integration remains unresolved |
| Taught exact rises; held exact remains at countermodel level | Tested learner memorizes without demonstrated reusable addition structure |
| Teacher-forced output improves; own-prefix output/EOS does not | Sequential generation/exposure mismatch needs investigation |
| OLD-end improves; rest or NEW removes the gain | Time dynamics or interference needs investigation |
| Held exact exceeds causal/recitation controls and survives NEW/rest | Promising exploratory E3; proceed to an independently frozen scientific design |

These interpretations localize failure. None is a family-wide theorem about local learning or biological brains.

## 8. Stop rules, resource bounds and scientific continuation

Before long work, run one cheap cross-layer diagnostic suite, capped at 60 CPU seconds: inspect birth assignments, exercise both competitive layers, inspect actual prediction/calibration frames and test the reference/sign/readonly invariants. Perfect fixtures may localize a defect, but their results are kept separate from the real learner.

Hard failures—target leakage, forbidden writes, nonfinite state, IO corruption or a source mismatch—stop the affected run. Preserve its actual exposure and error. Do not claim model efficacy from it. Prototype collapse, excessive shared activity or weak acquisition are diagnostic warnings; they do not prohibit a bounded look at the complete unqualified learner. In particular, the raw bypass means a learned-unit alias is not automatically an information-access failure. Formal promotion requires valid behavioral evidence and attribution, not an arbitrary cluster-accuracy percentage.

Cap the entire two-birth harness, including component diagnostics and evaluation, at **900 process CPU seconds** and 30 wall minutes. Initial concurrency is one worker, one numerical thread; require ≤2 GiB worker RSS and ≥2 GiB measured system headroom. Profile a 16-record prefix before projecting the full cost. If the plan cannot fit, report the measured partial curves and `RESOURCE_LIMIT`; do not silently reduce the task, increase hardware/model size, or reset elapsed time. Caffeinate follows the controller during an actually authorized run.

No automatic stronger-rate grid, extra listening, larger KC population, new mechanism roster or additional worlds follows a disappointing curve. The harness writes one source lock per attempt, a compact event/exposure ledger, probe outputs and final states needed for audit. Preserve changed source versions on an actual revision; avoid duplicating identical full snapshots for every check. Global projection/sorting, native all-cell sweeps and readonly clone bandwidth remain explicit prototype costs. No sparse-compute or billion-neuron claim is made.

This harness cannot authorize adoption. A promising result leads to a fresh scientific design with one frozen candidate, independent worlds, paired causal controls, exact retained E3, taught-retention protection and sample/CPU curves charged for D0, internal learning views and all repeated teaching. Two engineering births are insufficient for a superiority claim. The existing v5.1 confirmation contract is not silently repurposed for the changed encoder/data schedule.

## 9. Delivery and implementation boundary

This revision changes competitive-layer initialization/occupancy, context normalization and selection, calibration-frame semantics and the next development workflow. It retains the native memory equations, raw-byte output goal, generic sequence state and architecture exclusions. It supersedes R1 as the next planning document; R1 remains archived as an unexecuted proposal.

Before a future run, lock exact native assets, class-model implementation, deterministic streams/roster, event iterator, fixture namespaces, source hashes and telemetry. Verify coupled train/calibrate/query behavior, not just separate functions. [SPEC.json](SPEC.json) records fixed limits and unresolved executable locks; [STATIC_REVIEW.json](STATIC_REVIEW.json) is a document check, not a model qualification certificate.

No model performance is claimed for R2. No training or push accompanies this design delivery.
