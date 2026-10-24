# A10 design review and proposed revision

**Status: design only; not implemented or launched.**

Reviewed proposal: `/Users/pencilbard/Downloads/A10_forensic_closed_loop_B_mechanism_20260920.md`.
Parent: frozen FULL_151 and exact A8/A9 B. Review date: 2026-09-20, New York.

The attached document is a proposal, not execution authorization. This revision changes no learner, experiment, source lock, or archived result. The companion audit reads existing A9 records only; it performs no learning, clamp, probe, or trajectory replay.

## 1. Recommendation

A10 is worth doing as a bounded investigation, but its present causal claims exceed its design. Narrow the question to:

> How much of B's behavior comes from applying reference-trajectory channel corrections, and how much changes when the local allocation controller and alpha teaching feedback are allowed to respond?

The frozen implementation rules out an activity/adaptation mediator for this contrast. It also sharply limits the alpha teaching pathway that B can change. Do not launch a five-way hunt through mechanisms absent from the current code.

The B/K/R tournament has completed its 1,536 original trajectories. BK had no confirmed incremental benefits and 17 confirmed harms versus B; BR and BKR produced tradeoffs. B remains the incumbent candidate, while FULL_151 remains unchanged. The diagnostic annotation correction was still replaying at review time. A10 should use the completed behavioral findings and corrected event definitions without treating uncompleted diagnostic summaries as final evidence. “Runs in parallel with the tournament” and “shift next to the tournament” are now outdated descriptions.

## 2. Evidence checked before revising the proposal

The read-only audit compared all 72 A9 E3 REF/B primitive-ledger pairs. These are 64 canonical histories and eight K8 histories built from those canonical materials; they are separate conditions, not two independent replication sets. Counts below are repeated coordinate-event records, not independent samples.

| Archived quantity | Canonical, 64 histories | K8, 8 histories |
|---|---:|---:|
| Recorded active plastic events | 221,184 | 221,184 |
| Recorded alpha-support coordinate-events | 11,603,220 | 11,603,220 |
| Eligibility h, REF versus B | Bitwise identical everywhere | Bitwise identical everywhere |
| Maximum absolute allocation difference | 0.501437 | 0.504967 |
| h-weighted mean absolute allocation difference | 0.010631 | 0.016978 |
| Maximum absolute raw alpha update difference | 1.26905e−7 | 3.49787e−7 |
| Maximum inferred amplitude difference, Δu/h | 5.87111e−7 | 1.62113e−6 |
| Raw alpha update sign changes | 0 | 0 |

Across an additional check of 136 paired final-state receipts—64 E2 and 72 E3—gamma-fast, adaptation, elapsed time, event count, and presentation count hashes/values all matched REF/B. Final-state agreement alone is not an all-event proof; the source dependency argument below supplies that structural reason, and instrumentation must verify it at every transition.

The effective alpha-MBON → gamma-DAN submatrix `F[2:,:2]` is exactly zero. Alpha-MBON → alpha-DAN coefficients are approximately 0.868e−9 to 1.694e−9. Given the alpha output clipping range of width 31.16 and `fw0 = −21.1055045362`, the largest possible paired difference in each alpha payload, when the common gamma/activity inputs are fixed, is 1.684925e−6. The maximum coordinate teaching-amplitude bound after T is 1.895718e−6. The archived differences respect this bound.

**Interpretation:** allocation responds substantially in some coordinates; the changed teaching drive is numerically very small. This is not a comparison of commensurate effect sizes, and it is not yet a behavioral mediation result. Small changes can matter near decision thresholds or accumulate. Keep the teaching clamp, preserve nonzero F entries, and do not declare feedback behaviorally irrelevant from coefficient size alone.

Reproducible evidence: [audit results](/Users/pencilbard/Documents/second_try/A10_ARCHIVED_SOURCE_AUDIT_20260920.json), [read-only audit script](/Users/pencilbard/Documents/second_try/audit_a10_existing_records.py).

## 3. Required corrections to the original proposal

| Issue | Why it matters | Revision |
|---|---|---|
| Increasing a Euclidean residual norm is treated as establishing closed-loop mediation | A fixed trajectory can improve a task-specific nonlinear readout while its norm increases | Treat direct temporal/readout effects and online responses as unresolved alternatives |
| D uses c from B, while C1 uses c from REF | They are different interventions; c from B already contains allocation's response to B | Use REF corrections for the primary open-loop reference; keep realized-B corrections as a second accounting convention |
| I is defined by subtraction, then its closure is used as validation | This equality cannot independently check event alignment, decay, or ledger completeness | Construct independent native-write and correction-response ledgers, then compare their sum with observed state differences |
| Activity, adaptation, and gamma are offered as generic moving mediators | For this frozen source and shared event stream they are invariant | Prove and check the invariants; do not spend factorial arms on them |
| Clamp names say FREE while definitions clamp that variable | Easy implementation and interpretation inversion | Use explicit PFREE/ACLAMP names and a complete 2×2 table |
| “All begin from REF pre-event states” is ambiguous | Resetting to REF before every event would destroy the trajectory response being tested | Same initial state only; subsequent states belong to each trajectory |
| Clamp insertion and B's p are unspecified | Clamping allocation in native writes but leaving another p in c tests a different mechanism | One effective p must feed both channel splitting and B correction; recompute allocation from the effective teaching drive |
| A clamp loss is called proof of a main/necessary mediator | Clamps interact and interrupt coupled dynamics | Report conditional effects and their interaction; interpret within this controlled intervention family |
| Nested Shapley results use inconsistent backgrounds | Fast/slow attribution on REF alone does not sum to the outer direct attribution in the presence of I | Use a specified hierarchical attribution, or label the quantities as different contrasts |
| Fractions are primary and stabilized only by a numerical floor | Signed cancellation yields negative shares, shares over 100%, and unstable ratios | Primary results are absolute margins and percentage points with intervals |
| Separate lag-bin surgeries are treated as additive contributions | Nonlinear readout effects of separate bins do not generally sum | Make lag analysis secondary; use a complete declared allocation if an additive total is claimed |
| “Irreducible nonlinear interaction” is an allowed conclusion | These few controls cannot prove irreducibility | Allow “not resolved by the prespecified controls” |

## 4. Source-specific causal graph and invariants

For a fixed anatomy, genes, initial state, and exogenous event stream:

1. PN encoding and the sparse KC mask x depend on the cue and fixed encoding parameters, not fast/slow memory.
2. Adaptation updates depend on its own previous value, x, duration, and its fixed time constant. Thus REF and B have identical adaptation and effective activity dx.
3. `h = dx × dt/90 × alpha_scale`, hence h is identical.
4. The current source has no alpha-output → gamma-teaching route. Gamma states, gamma outputs, and gamma payloads are therefore identical by induction.
5. The remaining dynamic variables that can differ are alpha-fast f and alpha-slow s. They influence allocation p and alpha MBON output. Only the small native alpha-MBON feedback submatrix can transmit B-induced output changes back into alpha teaching.
6. The source path passes `rep=0, feedback=0` to `advance73`. The latter disables the additional experimental feedback block, not the native F matrix.

This graph permits two returning paths:

`B correction → alpha state → local allocation p → later writes and later B corrections`

`B correction → alpha output → native F → alpha payload A → later writes and allocation`

Clipping sits in the expression/feedback path; it is not an independent memory state. Both paths share states and can interact.

Qualification must verify x/h/adaptation/gamma identities and exact matrix zero structure. A failure stops interpretation and triggers a source audit; it does not automatically authorize additional mediator arms. These statements apply to this frozen model, not to biological learning generally.

## 5. Exact variables and clamp semantics

For active plastic event t and coordinate i:

\[
h_{it}=dx_{it}\,\Delta t/90\,\alpha_{scale},\qquad
A_{it}=T_{i2}\,payload_{2t}+T_{i3}\,payload_{3t},\qquad
u_{it}=h_{it}A_{it}.
\]

Let P be the exact native allocation function using pre-event adaptation, f, s, and `sign(u)`. With effective p and A:

\[
q_f=(1-p)hA,\quad q_s=phA,\quad c=phb,
\]
\[
f^+=d_f(f^-+q_f+c),\qquad s^+=d_s(s^-+q_s-c).
\]

These formulas specify dependencies, not permission to reorder floating-point operations in the archived REF/B path. Reproduction uses the source patch order and the inherited tolerance.

All arms start at the same life-initial state. No per-event reset or checkpoint restart is permitted. Event timing, punishment, cues, and plastic flags are shared. Each arm owns its subsequent state.

An amplitude clamp replaces the **two alpha payload values** with the same-event paired REF values before the T projection and before allocation is evaluated. A free allocation function uses the effective u after this clamp, not an unclamped u retained from a different computation. An allocation clamp substitutes p_REF everywhere p is used, including c. Baseline b is frozen in every B condition.

Do not compute A by dividing by h in the implementation: h may be zero. Read payload and T at their actual source locations. The archival Δu/h calculation above is a descriptive check on stored eligible coordinates only.

Apply B on the source's active/plastic gate. In particular, u=0 is not sufficient to suppress B if h and b are nonzero. On rest and nonplastic transitions c=0 and both correction ledgers still decay.

## 6. Five conditions; one complete controlled comparison

First generate the exact same-world REF mediator ledger. Every clamp uses REF at the matching subevent, never a future value or a B-selected event.

| Condition | Allocation p | Alpha payload | Correction |
|---|---|---|---|
| REF | Native | Native | None |
| B00 / PCLAMP_ACLAMP | REF | REF | p_REF h b |
| B10 / PFREE_ACLAMP | Own prestate, effective u | REF | p_own h b |
| B01 / PCLAMP_AFREE | REF | Own state | p_REF h b |
| B11 / PFREE_AFREE | Own prestate, effective u | Own state | Exact original B |

This retains the original five-condition scope. It makes the open-loop arm the double clamp, eliminating the incompatible direct-effect definitions.

With the verified invariants, B00 can be reconstructed from the REF state plus a two-vector correction recurrence. It need not rerun the full neural computation. Verify equivalence against a literal double-clamped path on qualification fixtures before using that shortcut.

For endpoint Y on one world, write Y_R and Y_00,Y_10,Y_01,Y_11. Define:

\[
E_0=Y_{00}-Y_R,
\]
\[
E_P=\tfrac12[(Y_{10}-Y_{00})+(Y_{11}-Y_{01})],
\]
\[
E_A=\tfrac12[(Y_{01}-Y_{00})+(Y_{11}-Y_{10})],
\]
\[
J=Y_{11}-Y_{10}-Y_{01}+Y_{00}.
\]

Then `E0 + EP + EA = B − REF` exactly. J is an interaction diagnostic, **not an extra term to add to that sum**. Also publish the four conditional differences used in EP/EA; averaging them must not hide sign reversals.

These are controlled source-level intervention effects. They are neither unique natural mediation fractions nor biological necessity claims. For example, `B11−B10` measures the loss from clamping teaching when allocation is free; it does not independently identify a universally necessary teaching mechanism.

## 7. Independent state accounting

Maintain source-matched predecay ledgers at every transition, including idle decay:

\[
D^R_{f,t+1}=d_f(D^R_{f,t}+c^R_t),\quad
D^R_{s,t+1}=d_s(D^R_{s,t}-c^R_t).
\]

Separately accumulate correction response C from `c_B−c_R`, with positive fast and negative slow signs, and native-write response N from `q_f,B−q_f,R` and `q_s,B−q_s,R`. Require:

\[
z_B-z_R=D^R+C+N
\]

on the alpha blocks, with zero differences on all invariant blocks. This checks observed states against independently accumulated event effects. Keep `D^B=D^R+C` as an explicitly named realized-correction view, so readers can see how attributing endogenous corrections to “direct” changes the result.

Because h is common, native write differences need only a two-factor decomposition. Put bars for paired arithmetic means and deltas for B minus REF:

\[
\Delta q_s=h\,\Delta p\,\bar A+h\,\bar p\,\Delta A,
\]
\[
\Delta q_f=-h\,\Delta p\,\bar A+h(1-\bar p)\,\Delta A.
\]

Propagate each component under its own fast/slow decay. Check their sum against the native-write ledger and check `c_B−c_R=h b Δp`. A nonzero “eligibility contribution” is a qualification error for the frozen model, not a discovery.

Alpha payload provenance also simplifies: sensory drive, punishment, and gamma feedback are common, so the mathematical paired payload difference is the native alpha feedback term. Reconcile that expression numerically with the actual source payload subtraction, including floating-point evaluation error. Do not search for invented “other” terms to absorb unexplained discrepancies.

## 8. Reader attribution is separate from intervention effects

Use the native evaluator, including its fixed reader and both correctness gates. Freeze identical populations, truths, query codes, and weights for all state surgeries. Signed-margin attribution uses the inherited actual signed margin; classification attribution uses the full native correctness indicator, not a substituted single margin threshold.

Set `D=D^R` and `I=C+N`. Evaluate R, R+D, R+I, and B. The two-player endpoint Shapley contributions sum to B−REF but need not equal E0 and EP+EA. One describes endpoint state substitutions; the other describes trajectories under clamps. Report them in separate tables.

If splitting D into fast/slow contributions, use the hierarchical convention:

\[
\phi_{D_f}=\tfrac12\left[Sh_f(R;D_f,D_s)+Sh_f(R+I;D_f,D_s)\right],
\]

with the analogous slow expression. Their sum then equals the outer phi_D. Splitting only on the R background answers a different question. The same rule applies to a subdivision of I.

Synthetic endpoint states may not be naturally reachable. Label them diagnostic substitutions; do not present a large attribution as independent evidence of an online causal path. A lack of clamp significance does not turn an endpoint attribution into causal proof.

At an immediate fixed query, alpha expression reads f+s. Fast and slow labels become distinguishable through their histories and unequal decay, not separate immediate reader channels.

The proposed analysis does **not** identify the full causal source of the historical B-transfer versus B-drop difference: B-drop follows its own trajectory and correction sequence. Without a paired drop trajectory analysis, describe fast attribution as the contribution of the specified B correction component, not a decomposition of the transfer-versus-drop treatment effect. This revision adds no drop arm.

## 9. Horizons, event labels, and populations

Preserve current usefulness, acquisition, revision, anomaly recovery, first-use return at exposure 0, reactivation at exposure 4, and the four inherited +24h/+7d endpoints. First return is not interchangeable with retraining-assisted recovery.

For delayed clone probes, apply the full native passive transition to every relevant reconstructed state. Do not carry an immediate D unchanged into a 24-hour reader evaluation. A predecay event increment receives that event's decay as well as subsequent decay. Store start, end, duration, and event index; use end time only after multiplying the contribution by its within-event decay.

State-only lag bins, if retained, are `[0,5 min)`, `[5 min,1 h)`, `[1 h,24 h)`, `[24 h,7 d)`, and `[7 d,∞)`, measured from event end. Preserve the within-event multiplier. Empty bins are reported as empty. Main results do not depend on a lag-bin decomposition.

If additive behavioral lag attribution is desired, use a prespecified full five-bin Shapley calculation within the chosen parent component and on both hierarchical backgrounds. Independent leave-one-bin-out effects are sensitivity contrasts and cannot be summed as “fractions explained.” Keep this secondary and do not open additional replay arms for it.

Use the corrected challenge-map annotation semantics from V5: scheduled revision begins with its first scheduled revised teaching, independently of the later scorer-truth flip. Anomaly schedules may have one or two wrong bouts. Compare epoch identities only within the same association. Such labels annotate evaluator records; they never select a clamp or enter learner inputs. Populations and labels need explicit fixtures before the main replay, preventing another full annotation-repair replay.

## 10. Panels and bounded workload

**Phase A, archive reconstruction:** all A9 E3 histories: 64 canonical and eight K8. Existing primitive ledgers are reused first. Replay REF/B only where missing source fields or checkpoint states make it necessary; any replay must reproduce archived records. All five conditions are evaluated, without choosing favorable histories. Paired canonical and K8 results remain separate because their base materials overlap.

**Phase B, fresh confirmation:** retain the proposal's fixed 32 canonical and 16 independent K8 histories. Each K8 history uses eight unique donor worlds; none is reused across K8 histories or among the 32 canonical histories. Thus the fresh material is 32+128=160 distinct base worlds. Exclude A8/A9, all BKR worlds and reserved namespaces, and every previously used cue key.

This is 240 fresh condition-history evaluations. If B00 reconstruction qualifies, only 192 are full learner trajectories and 48 are derived reference-ledger paths. Across both phases the maximum is 600 condition-history evaluations, or 480 full trajectories plus 120 derived paths with the shortcut. Qualification fixtures are counted separately and use known worlds; no hidden independent science samples.

Freeze clamp definitions, populations, accounting conventions, and endpoints before new Phase A interventions. This review's existing-data audit is exploratory provenance. Before Phase B, seal code, all world manifests, extraction rules, statistical code, and the complete list of hypotheses; no preferred mediator or endpoint is selected from Phase B. Technical amendments are versioned with affected outputs invalidated, never silently pooled.

Do not gate release of completed behavioral findings on optional diagnostics. Deliver the controlled-effect table when valid, and clearly mark any unfinished mechanistic supplement.

## 11. Statistical claims and decision rules

Primary estimands are world-level effects in native signed-margin units and current-usefulness percentage points. Within-world records use the frozen inherited P1 population and weighting; the accompanying margin extractor uses exactly those same records and weights. Cohorts, checkpoints, events, and coordinates are not independent replicates.

For fresh confirmation, report six contrasts—E0, EP, EA, J, B−REF, and B−B00—on two scales and two panels: **24 prespecified simultaneous two-sided intervals**, using the declared paired-t working model and Bonferroni alpha=.05 over the 24 intervals. Canonical n=32 and K8 n=16. Report the normal-approximation assumption explicitly, especially for sparse/discrete accuracy differences. Zero sample variance is not an exact equality certificate without a proved invariant.

Phase A is descriptive. Use its paired dispersions to publish a precision forecast for the fixed Phase B sizes, with small-sample uncertainty; do not resize the sample or redefine contrasts in response. Delayed, first-return, tail, factor, and lag analyses are prespecified secondary diagnostics, not additional uncorrected primary discoveries.

Absolute effects are primary. Signed ratios can be negative or exceed 100%; do not clip them into a pie chart. A numerical denominator floor is insufficient statistical protection. This revision makes no confirmatory percentage-mediated claim. If exploratory ratios are shown, accompany them with numerator/denominator intervals and label near-zero or sign-uncertain denominators unresolved.

Allowed conclusions include:

- a positive controlled reference-correction contribution;
- a positive, negative, or unresolved allocation-response contribution;
- a positive, negative, or unresolved teaching-response contribution;
- measurable interaction or contrasting conditional effects;
- failure of attribution stability on fresh worlds;
- insufficient precision or failure of the prespecified controls to distinguish explanations.

Do not call B≈B00 from a nonsignificant difference. Without an independently justified, predeclared equivalence margin, report the interval and retain uncertainty. Do not label a negative mediated contribution as a “fraction of benefit.” Do not equate average endpoint improvement with protection of every memory. Do not conclude biological necessity, general optimality, or irreducibility.

## 12. Qualification and economical logging

Before any new scientific interpretation, require:

1. Frozen source/anatomy/gene/reader/baseline hashes, including A9's documented source amendments; no re-created or recalibrated baseline.
2. REF/B nonzero qualification reproducing complete known-life state/checkpoint/score fixtures, denominators, and exact event order.
3. All-event x/h/adaptation/gamma invariants and shared event counters; proof plus observed checks, not just final hashes.
4. Independent D_R, correction-response, native-write, and factor recurrences closing to observed states.
5. B00 matching literal double-clamp fixtures; B11 reproducing B; the no-B reference values giving reference transitions. A zero-B test alone is not a test of the nonlinear B path.
6. Explicit clamp wiring checks: exactly two alpha payloads; allocation evaluated after the effective drive; the same effective p used for writes and c; active zero-u B events retained.
7. Clone/read-only state invariance; native correctness and denominator equality for reconstructed endpoints.
8. Correct decay through rest, active-nonplastic events, long dormancy, and delayed probes; exact timestamps and no compressed teaching schedule.
9. Endpoint attribution and controlled-effect sums; both in floating precision for margins and on exact decision counts where possible.
10. Annotation fixtures separating scheduled revision and anomalous feedback before long runs.

Retain the inherited `1024 × eps × max(1, relevant magnitudes)` numerical rule for matching like computations, with operation-specific scales frozen on fixtures. For long independently accumulated sums, either use a documented accumulation-error bound or a predeclared stable reference summation; do not enlarge tolerances after scientific failures. Logical invariants such as identical h and unchanged denominators require exact checks. Near-zero alpha feedback must not be zeroed by a generic tolerance.

Log the exogenous stream once; fixed encoded masks once; sparse active alpha-support u,p,h and source payload components for each required path; passive durations; full neural states at checkpoints; and final state/write digests. Corrections and native split vectors are reconstructible from primitives and the frozen b. Keep a small explicit pre/post-state qualification fixture, rather than dense copies of every 5,177-vector before and after every event across all worlds.

Use known double-precision primitives and preserve source operation order. Provide streaming reconstruction and a small independent numerical checker in the eventual implementation. Estimate disk/RAM/time from known fixture sizes before launch and record a fixed resource budget. The current design review does not create this runner.

## 13. Deliverables and stopping boundary

Recommended eventual deliverables are a protocol/source lock, world and event manifests, source-invariant certificate, sparse primitive ledgers and checkpoint states, state-accounting table, controlled-effect table with simultaneous intervals, separate endpoint-Shapley table, horizon/tail supplement, qualification report, and final interpretation.

Stop after the fixed fresh panel and these prespecified summaries. No extra gains, baseline variants, clamps, sample expansion, compression search, or additional mechanism tournament is part of this proposed scope. “Not resolved” is a valid outcome; it does not authorize a new forensic branch.

A10 cannot determine neuron-capacity sufficiency, compress or learn b online, establish longer-lifetime generalization, or identify a biological implementation. Those remain separate questions. The scientific result sought here is a reproducible decomposition of B's effect under explicit source-level interventions, with a clear account of what remains unidentified.

## Source references

- [Native event kernel](/Users/pencilbard/Downloads/second_try_copy/minifly/iteration18/src/model73.py:9)
- [Allocation dependency](/Users/pencilbard/Downloads/second_try_copy/minifly/V82E/src/model_evo.py:193)
- [Exact A8 B event path](/Users/pencilbard/Downloads/second_try_copy/WorkstreamA8_baseline_transfer_20260920/engine.py:63)
- [A9 sparse ledger and transfer/drop paths](/Users/pencilbard/Downloads/second_try_copy/WorkstreamA9_coordinate_controls_20260920/engine9.py:4)
- [A9 interpretation and results](/Users/pencilbard/Downloads/second_try_copy/CORE_RECOVERY_workstreamA9/coordinate_controls_v1/REPORT.md)
- [Completed V5 behavioral report](/Users/pencilbard/Downloads/second_try_copy/CORE_RECOVERY_BKR/v5/REPORT.md)
- [V5 annotation correction](/Users/pencilbard/Downloads/second_try_copy/CORE_RECOVERY_BKR/v5/annotation_replay/AMENDMENT.json)
