# RECOVERABILITY V1-R2: separate stopping from native interference protection

23 October 2026 · **Design only; unimplemented and untested**

This replaces the sequencing in the [V1-R1 proposal](../minifly_recoverability_v1_dev_20261022/REVISED_DESIGN.md), not the frozen V1 experiment or its negative verdict. Run two bounded component tests: **A, learn stopping without changing payload bits; B, test one native protection gate with a fixed readout.** B does not wait for A to recover OLD. The eventual system must recover both relevant OLD and NEW values completely, including its own EOS. That requirement is unchanged.

## 1. Evidence, objective and what changes

[RECOVERABILITY V1](../minifly_recoverability_v1_dev_20261022/REPORT.md) completed its roster. All four RAW/STD × MEAN/SEP cells failed the joint OLD/NEW criterion. On STD final NEW, direct payload-only exact was 61.72%, versus 20.57% with the learned MEAN head. On STD final OLD it was 0%, versus 53.91% at the matched time with NEW writes off. This supports two different bottlenecks, not a general proof of erased information or insufficient population capacity.

For a payload-preserving decoder, complete exact recall cannot exceed direct payload exact. An EOS-only learner therefore cannot recover OLD answers on saved states whose payload exact is zero. The mistake in R1 was making that impossible component outcome a prerequisite for native protection research. Here, stopping and storage have separate component decisions; neither decision alone promotes the combined system.

The project goal remains better unseen-addition generalization per all samples and actual compute, with retention. **This side experiment is E1, known-key taught random-association acquisition/retention.** Surrogate key addresses are not a learned byte-key interface. Neither component tests arithmetic rules, E3 transfer, NLP, or an LLM host. The arithmetic experiment pointer, code, data and historical adoption records remain unchanged.

No larger decoder, native population, attention, Transformer, ELM, replay, new routing, task-labelled bank, arithmetic representation or rule is introduced. Offline gradient training is restricted to the external EOS head. Native learning remains its original local update plus the single proposed multiplier in B.

## 2. Common contract

Reuse the V1 canonical N=5,177 anatomy, nine compartments, float64, `gated=False`, original fast/slow allocation and decay, `DT=30/14`, replay off, BEGIN/adaptation reset, side-specific positive top-k, and target-independent 16-byte-key-to-receptor mapping. Preserve the frozen 2,048-vector calibration. RAW is centered drive; STD additionally divides by the original fixed standard deviations/floor. Use MEAN responses as the primary interface; SEP had no joint advantage and is not another search axis here.

The nominal feedback remains

`c_c(n) = a(n) * (tanh(m_c/2) - t_c) / 2`

`a(n) = (1/16) * (1/16 + (15/16)*min(n/1024, 1))`.

Native averaged margins supply `m`, expected observed byte/EOS supplies the original signed target `t`, and EOS masks payload writes. The nominal schedule and residual are not retuned. **B adds a selective multiplier to plastic increments; it does not claim that effective writes remain identical.** Passive decay, adaptation, clocks and native fast/slow allocation are unchanged.

Each life: eighteen OLD keys, one-hour simulated rest, eighteen disjoint NEW keys, another one-hour rest. Each cohort has two uniform random payloads of every length 0–8 and fourteen seeded passes. All arms receive identical keys, values, order, targets and elapsed time. Both cohorts are relevant at final recall. Queries are private common-horizon SNAP reads: BEGIN once, all nine slots without teaching or inter-slot ticks, actual minus neutral output. Teacher length is unavailable to candidate inference.

The continuous warmup is asymmetric: mean nominal `a(n)` is 0.0110874 over OLD records 0–251 and 0.0255070 over NEW records 252–503, about 2.30 times larger. Keep this schedule. Pass counts do not match write strength, residual correction, or useful learning; B instruments all four. Do not infer a capacity limit from this protocol.

## 3. A — small stopping-only test

**A changes only the external EOS residual.** Input is the original 162 normalized native-response components plus a nine-dimensional slot one-hot, with MEAN repeated across the two output positions. Use tanh(32) and one output: **5,537 trainable parameters**, no recurrent state. The output layer and bias start at zero; the fixed skip is the original normalized native EOS margin. Payload bytes always come from the signs of the eight unmodified native averaged payload margins. The head has no path to alter them.

Generate EOS when its logit is strictly positive. Ties continue. At slot 8, failure to emit EOS is truncation, not forced success. Empty strings use the same slot-0 decision. No per-query threshold, true length, cohort identity, keys, directory, probe labels or seed is an input. Frozen per-compartment RMS scales are reused from V1's designated training traces; positive normalization does not change payload signs.

Fit on the **existing twelve V1 training episodes**, six equal-weight end/rest phase pools, separately for RAW/STD and three paired seeds: six fits, not twelve. These are 1,296 saved queries per storage condition, with repeated supervised presentations counted. Only EOS targets at `j<=L` are exposed to the head; no payload loss or post-EOS target is supplied. Use the original NumPy AdamW settings: lr .001, 100-update linear warmup, beta .9/.999, eps 1e-8, decay .0001, norm clip 1, batch 32, final iterate. Each fit stops at 2,000 updates or 2,000,000 binary terms, whichever comes first, reserving final training-loss terms inside that ceiling. No best-seed selection or later calibration.

Previously inspected V1 development/assessment traces may illustrate errors but are now exploratory. Do not select an architecture, threshold, fit length or seed on them. Freeze all six heads before fresh evaluation associations are generated. Do not generate another native training roster just to refit the head.

### A's component decision

For each readout on the **same native state**, report:

- Complete nonempty exact bytes plus own EOS, unconditionally.
- Stopping validity conditional on the direct native payload being completely correct: no EOS before its actual end and EOS at the end. The evaluator defines this subset after generation; the candidate never sees its membership or length.
- The denominator, payload exact, empty/default behavior, per-length results, early/late EOS and truncation. No-payload-correct subsets are undefined, never scored as zero or perfect.

Use OLD-end, final NEW, and final passive-OLD S_NEW as the three prespecified usable-state checkpoints; still publish final written-OLD. Reduce queries within life, seeds within life, then the two lives within pair. A favorable component result requires positive average conditional stopping and complete-recall differences over the direct codec across the three usable checkpoints in every pair, with the individual checkpoint differences and denominators disclosed. No result can conceal a checkpoint regression or earn a reliability claim from a tiny subset. Failure stops this head family; success retains only a candidate stopping component.

ZERO/FOREIGN native-response lesions, native no-write and within-length shuffled-OLD controls remain. Compare the frozen V1 joint heads and its expressly posthoc payload/EOS recombination as historical references, not fresh selected candidates. **A need not rescue final OLD and does not block B.** Its conditional metric does not replace the combined-system endpoint.

## 4. B — one native contribution-credit availability gate

**Candidate name: CREDIT-AVAIL V1.** This is a new native operation version motivated by historical importance/availability mechanisms. It is not an ELM port, an established biological rule, or a claim that an untested family is new. [MECHANISM_DELTA.md](MECHANISM_DELTA.md) records the exact historical distinctions and failures.

The hypothesis is that active KC/output coordinates which currently support a correctly predicted observed bit should become less available for subsequent plastic changes, while retaining nonzero plasticity. A use count alone need not identify useful contributions. We therefore specify a bounded, target-signed *local proxy*, not an undefined “importance” oracle.

### B.1 State and credit, defined before outcomes

For each compartment c and KC i, maintain `h_ci in [0,1]` and a last-materialized timestamp, initially zero. On access, decay the trace lazily by `exp(-elapsed/86400)`; this one-day scale is inherited from native confirmation bookkeeping, not fitted on V1 outcomes. BEGIN does not clear it. At queries, use a private snapshot and never update credit.

Before a teaching event, retain the same pre-event state/activity used by the original native query. Let `dx_ci` be the query activity factor already calculated by the native equations. Define

`s_ci = dx_ci * (Q_i2 + Q_i3)/2 * (fast_ci1*exp(-DT/tf) + slow_ci*exp(-DT/ts))`.

This is the coordinate's direct alpha-path contribution before clipping; it omits gamma-mediated contributions and is **not exact causal attribution**. Fixed Q is native connectivity, not a learned attention map. The candidate uses no gradient, leave-one-cell-out probe or stored old example.

After the expected target becomes externally available, and only on channels eligible under the original teaching mask, define from the **pre-write** values

`v_ci = 1[t_c*m_c > 0] * tanh(max(0, t_c*s_ci)/2)`.

The scale 2 is the existing native residual scale. A wrong/tied bit gives no credit. No credit is based on a corrected post-teaching answer. A novel zero-valued store cannot protect itself merely by being active. This is per-bit warmup supervision, not a claim of learning with scalar-only feedback.

Use the old trace for the current write. After the event, update active eligible coordinates by

`h_ci <- h_ci + (1-h_ci)*v_ci`.

No-write, passive, replay-disabled or masked channels acquire no new credit. Idle time causes decay only. Apply the same rule to OLD and NEW throughout the life; the learner receives no cohort or phase switch. Credit does not require task IDs or a prompt identity cache. Empty/EOS records follow the original channel mask.

### B.2 Single change to native plasticity

Separate each original native event into passive evolution and its plastic increment `Delta0_ci,k`, where k covers fast0, fast1 and slow. Compute every proposal from the original event prestate. Then apply

`g_ci = 1/(1+h_ci)`

`state_next = passive_next + g_ci*Delta0_ci,k`.

The same g multiplies all three plastic increments at that coordinate. `g in [1/2,1]`: no hard freezing, amplification, new fast/slow split or renormalization restores the removed correction. This reciprocal unit-strength choice deliberately limits the first test to at most halving a coordinate's plastic increment; it is a fixed engineering hypothesis, not an optimal biological constant. No strength/time-constant scan is authorized by this design. It can fail by saturating credit everywhere, crediting the wrong coordinates, or weakening useful acquisition.

Do not multiply passive decay or adaptation by g. At g=1 the event must match the frozen native primitive; at g=0 in a synthetic fixture only, it must match passive evolution. Preserve confirmation bookkeeping, but keep the inherited slow-allocation gate off in every arm. Credit is separate from that historical signed record-confirmation gate.

## 5. B's four arms and exact control semantics

All native arms use **STD**; RAW stays only in A's baseline comparison. Use one candidate and these controls, without a factorial architecture search:

| Arm | Plastic increment rule | Purpose |
|---|---|---|
| BASE | Original native increment | Retention/acquisition anchor |
| CREDIT | Above coordinate-specific g | Sole candidate |
| UNIFORM | One scalar attenuation per compartment/event, yoked to CREDIT's applied increment L1 | Tests reduced writing as an explanation |
| SHUFFLED | CREDIT's gate values permuted across eligible active coordinates, within KC side and compartment | Tests location specificity with the same gate-value multiset |

Process paired arms in lockstep on the same event; diagnostic controls may use only CREDIT's **current** gates/write totals, never future events. They are experiment references, not proposed deployable learners. Gate permutations are frozen deterministic nonidentity cyclic permutations of sorted active indices within each side; singleton sets remain unchanged and are counted. They are independent of targets, length and success. Preserve the gate multiset exactly; report when equal gate values make a shuffle inert.

For UNIFORM let `T_c=sum_i,k |g_ci*Delta0_CREDIT_ci,k|` and `U_c=sum_i,k |Delta0_UNIFORM_ci,k|`. If `0<=T_c<=U_c`, use `rho_c=T_c/U_c`, or zero if both are zero. This matches **applied plastic-increment L1**, not net state change after passive decay. If `T_c>U_c`, use rho=1 and record the unmatched excess; amplification is forbidden. Closed-loop states diverge, so feasibility cannot be assumed. Exact matching tolerance is `1e-12*max(1,T_c,U_c)`.

SHUFFLED matches the distribution of multiplier strengths, not automatically the weighted update L1 once states diverge. Publish its actual increment budgets. **Do not call both controls perfectly write-budget matched.** If UNIFORM matching fails materially beyond numerical tolerance, the clean matched-dose selectivity conclusion is unavailable; keep the storage observations and mark the mechanism comparison inconclusive. No retry, new dose or extra control is silently added. A shuffled advantage/disadvantage that can be explained by its different actual write budget is likewise not conclusive location attribution.

Each arm also forks at its own rest1 checkpoint into S_NEW: identical NEW inputs/time, no NEW value writes or credit acquisition, followed by the same rest. This controls passive dynamics without replaying OLD. BASE's N and within-length SH controls are shared with A. These are not substitutes for a fresh full-system attribution test if a candidate survives.

Use the same external stopping head for every STD arm: the final A STD-MEAN head for each of the three seeds, frozen regardless of A's success. Direct native decoding is always reported too. If A fails integrity qualification and no head can be frozen, B can still test native payloads using the fixed direct codec; no calibrated whole-output benefit is then claimed. No protection-specific readout fitting is allowed.

## 6. B's endpoints, update ledger and decisions

Co-primary component quantities: final OLD and final NEW **direct native payload-only nonempty exact**, with evaluator length used solely to measure stored content. Also measure acquisition at passes 1/4/14, bit/byte accuracy, signed margins, complete generated bytes plus own EOS, and passive-branch retention.

For every cohort and arm, record nominal a(n), original residuals, active support/overlap, proposed and applied plastic-increment L1/L2, signed readout-projected change where available, fast/slow totals, gate and credit distributions, saturation, uniform-match errors, and useful NEW acquisition. Log the pre-event values before any correction. Pass counts alone cannot distinguish retention from reduced learning.

A **favorable protection pilot** requires CREDIT's final OLD payload exact to improve over BASE, UNIFORM and SHUFFLED in every pair, while final NEW payload exact is no worse than each comparator in every pair. Publish both absolute accuracies and acquisition curves; zero difference is the conservative trade-off boundary, not a practical reliability threshold. Apply the control-matching qualification above. Ties, reversals or gains paid for by worse NEW learning do not pass. Four pairs support a development decision, not confirmatory significance or a family-wide exclusion.

Do not discard the result if payload storage improves but stopping does not: that supports a storage component only. Conversely, more complete NEW recall with unchanged OLD content is a stopping result, not protection. If both components help independently, propose a **fresh combined evaluation**, retaining all original OLD/NEW own-output and attribution requirements. Neither component result alone permits host integration or core adoption.

If both fail their assigned functions, reconsider native write/read representation using these measurements; do not automatically enlarge a decoder, population or mechanism roster. Inability to revise an association is a plausible cost of protection; this particular two-cohort assay does not establish revision competence and cannot promote a general persistent core.

## 7. One fresh bounded roster, shared computation

Reserve four fresh assessment-pair namespaces `202610239301–202610239304`, two independent value lives per pair, same keys/addresses but independently sampled balanced values/lengths. Reserve head seeds `202610239401–202610239403` and E0 namespace `202610239501`. Collision-check all actual keys against prior rosters and freeze manifests before generating values. No best-world or best-seed selection. No separate development screen or conditional increase in n.

For A, run RAW and STD BASE-W/N/within-length-SH lives, with a BASE-W S_NEW continuation. B reuses **the identical fresh STD BASE states**, then adds CREDIT/UNIFORM/SHUFFLED lives and each arm's S_NEW continuation. Computational sharing is not extra evidence.

| Prospective native applications | Count |
|---|---:|
| A: 4 pairs ×2 lives ×2 storage configs ×(3×504+252) | 28,224 |
| B additions: 4 pairs ×2 lives ×3 arms ×(504+252) | 18,144 |
| **New main total / events** | **46,368 / 231,840** |
| Additional E0 maximum, separately itemized | 128 full records +16 partial events |

There are 72 new base histories and 40 S_NEW continuations, with 10,080 shared-OLD logical equivalents; 56,448 main logical record equivalents. They are not 112 independent worlds. The assessment unit is the pair, conditional on one anatomy/calibration. Three head seeds and lesions do not multiply native histories. Historical V1 data stays immutable.

Reusing A's native training traces costs zero new native applications, but each learner's information ledger must include their original twelve-episode exposure, calibration, revealed EOS labels and repeated offline presentations. Six fits have ceilings of 12,000 optimizer updates and 12,000,000 binary terms, including final losses. Explain operational reuse separately from learning-budget savings. The surrogate-key exact table and canonicalized-key table can recover taught values; their success keeps this E1. Include the training-only byte/length prior, empty defaults and absent-key emissions. None is a candidate backend.

## 8. Qualification, cost and delivery boundary

Before the future test: source-lock the full runtime dependency closure and birth assets; verify native g=1/g=0 equivalence, original gradient-free native semantics, credit update ordering/bounds/lazy decay, correct-mask/no-write behavior, gate/permutation invariants, paired prestates/clocks, snapshot restore, EOS-head finite differences, immutable payload signs, and zero label/length/key leakage at inference. Native fixtures and synthetic head tests count in the stated caps. Preserve failures; do not restart exposed trajectories.

Use one worker/one numerical thread. Ceiling: 1,800 cumulative process-tree CPU seconds, 2,700 active wall seconds, sampled 2 GiB process-tree RSS, and measured 2 GiB admission headroom. Meter imports, preprocessing, existing-artifact loads, fitting, all controls/probes, serialization, audit and packaging. A complete head fit and the first complete native pair are part of the roster/profile, not discarded trials. Stop with a partial report if a conservative projection cannot fit; do not drop endpoints/arms or reset the budget. Caffeinate may follow a future authorized active controller.

New persistent candidate state is two float64 arrays of shape 9×5,177: **745,488 bytes**, apart from logs. Controls allocate matched shadow storage and charge any shadow computation separately. Active credit updates use accessed coordinates, O(A); the canonical native event and encoding still sweep/project/sort, and readonly audits clone full states. Permuting active gates and L1 reductions add measured control overhead. This design establishes no computational sparsity or scaling advantage.

Deliver head weights/scales, all stream and address assets, full native checkpoints including credit/timestamps, raw native responses, emitted outputs, update ledgers, per-pair/seed/length/conditional denominators, independently reduced endpoints, source hashes and a compact bundle. Distinguish integrity acceptance, component efficacy, whole-system capability and E1/E3 evidence in the report.

**This delivery creates documents, machine-readable specification, evidence links and static consistency checks only. New implementation: 0. Fits: 0. Native applications: 0. Three-cycle exercises: 0.** The earlier V1 run remains complete and negative. No run, background process, arithmetic change, host campaign or GitHub push is started by shipping this revision.
