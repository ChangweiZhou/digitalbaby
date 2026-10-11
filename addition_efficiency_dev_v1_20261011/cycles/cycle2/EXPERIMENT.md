# MiniFly addition learning efficiency — design V1

**Date:** 2026-10-11  
**Status:** DESIGN_DELIVERED; implementation, qualification and execution not authorized by this document.  
**Configurations:** STABLE, U042-WG1, U021-SF1. No parameter search or additional mechanism is included.

本轮继续发展已通过确认的有限加法核心。问题是：在相同原始字节、数字／角色表示和输出规则下，两个明确的新内容更新规则能否用较少示例提高未教组合的正确率，同时保住已有的保持能力并控制实际计算成本？本轮不重启编码器轮换，不增加运算、精度、NLP 或开放输出。

## 1. Starting evidence and permitted claim

The accepted predecessor is `addition_value_retention_confirm_v1_20261011`, not the earlier exact-key content core. Its 64 fresh worlds completed 128 world-arm jobs and 384 histories. STABLE reached 100% final accuracy on old, subsequent and held-out expressions; the independent audit accepted the limited-addition V2 instance. See [the accepted report](../addition_value_retention_confirm_v1_20261011/results/REPORT.md).

That result used operands 0–4, one answer digit 0–8, engineered magnitude/role features, ordinal rendering, 40 observations per trained expression, 80% correct / 10% wrong / 10% missing answers, intervening training and two 24-hour model-clock gaps. It does not establish unrestricted arithmetic, an unmodified Full151 arithmetic circuit, or learning without external answer information. A portable copy of the report is [BASELINE_REPORT.md](BASELINE_REPORT.md).

This design targets **E3 learning efficiency within the same addition family**, with E1 acquisition/retention as protection measures. Exact emitted digits before feedback are the endpoint. Code distance, reconstruction accuracy, storage relocation and fewer writes are not substitute endpoints.

The full-dose predecessor is already at a ceiling. Therefore:

- The scientific comparison is at 10 and 20 examples per expression, with 40 as a protection condition.
- All three configurations receive the same information and fixed representation.
- A data-efficiency signal alone does not demonstrate a compute-efficiency improvement.
- A negative result retains STABLE. It does not reopen the byte bridge or eliminate either entire mechanism family.

The 71-item inventory remains a provenance ledger, not 71 runnable candidates. This round instantiates only U042's write-gate branch and U021's provisional/committed-content branch. U043 topology change, U047 replay, U055 recurrence, U057 and U058 are not silently added.

## 2. Fixed task, input and output

Use the predecessor's `ByteRoles`, `receptors` and fixed `SharedEncoder` without changing feature construction, projection, sampling seed, neuron mask or cache policy.

- Expressions: `a+b=`, with `a,b ∈ {0,1,2,3,4}`.
- Output: one model-emitted digit in `0`–`8`, at `=`, before any answer byte arrives.
- Visible input: the expression, an actually delivered answer digit when present, and newline. No correct-answer bit, noise category, pair identifier, phase identifier or held-out indicator enters the learner.
- The parser supplies independently decoded operand identities; it does not supply their sum.
- Fixed ordinal rendering: `value = z · (F + S)`; logits `−0.5 × (k−value)^2`, `k=0..8`; lowest digit wins an exact tie. The inherited scale cancels in the softmax and is retained identically.
- No formula for addition, fitted arithmetic coefficient, exact-key answer table, external choice between alternatives or auxiliary trained readout is allowed inside any learner.

There are 5,177 KC coordinates. The existing representation uses 28 fixed role/magnitude features, 15 dedicated KCs per feature, and 180 active KCs per expression. These engineering priors are disclosed and shared; they are not candidate mechanisms.

### Feedback split

Five complete exchange orbits are withheld:

```text
HELD_ORBITS = (1,1), (1,2), (0,4), (2,3), (3,3)
HELD = (0,4), (1,1), (1,2), (2,1), (2,3), (3,2), (3,3), (4,0)
NEW  = (1,3), (3,1), (2,4), (3,0)
OLD  = every other trained pair; 13 ordered expressions
```

There are 25 scored ordered expressions, 8 held out and 17 taught. NEW means subsequent teaching, not four universally novel semantic exchange orbits: some reversals already occur in OLD. This limitation is inherited and explicitly retained.

## 3. Nested dose construction and matched clocks

This is a prospective schedule refinement of the existing task. The old experiment shuffled all 40 repeats together. Here each phase has **four independently randomized ten-repeat blocks**. The new 40-dose stream is not claimed to reproduce the predecessor's stream exactly. A fresh STABLE comparator is run throughout.

For each world, construct one immutable master object using `PCG64(SeedSequence([20261011, world, phase_code, block]))`, with phase codes 0=OLD, 1=NEW and block numbers 0–3. Iterate pairs in lexicographic order before permutation.

Within each block, every phase's expression has exactly:

- eight correct answer observations;
- one wrong digit drawn uniformly from the eight other digits;
- one missing answer, represented by the immediate newline after `=`.

Shuffle the ten observation kinds per expression, concatenate expressions, then shuffle the entire phase-block. All configurations and histories derive from this same master object. Dose 10 observes block 0; dose 20 observes blocks 0–1; dose 40 observes all four. Incorrect digits and record order in shared blocks are bit-identical across doses. No resampling after outcomes are seen.

### Equal model time

Each master record occupies a six-byte slot of length `6 × DT`, with `DT=30/14` seconds. A missing-answer record has five real bytes followed by `DT` of silent rest. An omitted dose block supplies only rest for its complete slot duration: its bytes and labels must never be passed to the learner.

Chronology for every dose and configuration:

```text
OLD master slots → OLD_END probe → rest 86,400 s → DAY1 probe
NEW master slots → NEW_END probe → rest 86,400 s → FINAL probe
```

Thus all parent states reach identical checkpoint clocks:

- OLD phase: `13 × 40 × 6 × DT = 6,685.714285714… s`;
- NEW phase: `4 × 40 × 6 × DT = 2,057.142857142… s`;
- final parent clock: `181,542.857142857… s`.

Rest advances model time, not wall time, and provides no teaching. Use integer byte-slot ticks plus explicit day increments to avoid accumulating rounding differences. Probes run on read-only copies and do not advance the parent. Each dose starts from a fresh birth; it is not a fork of a 40-dose history after a different chronology.

## 4. Histories and prohibition on hidden teaching

Each configuration × dose × world runs three independent histories:

| History | OLD delivered bytes | OLD learned state | NEW delivered bytes / learning |
|---|---|---|---|
| W | Original master stream | Updates allowed | Original stream; updates allowed |
| N_OLD | Exactly W's bytes | All target-dependent updates forbidden | Exactly W's stream; updates allowed |
| SHUFFLED | OLD answer digits permuted within each master block | Updates allowed from the actually delivered digit | Exactly W's stream; updates allowed |

SHUFFLED preserves missing-answer positions and the multiset of present answer digits in each block. Its permutation uses a separate namespace `SeedSequence([20261011, world, phase_code, block, 99])`, applied to present-target positions after record ordering. The NEW stream is never shuffled. An accidental unchanged answer is retained; do not force a derangement or redraw a block.

For N_OLD, forbidden OLD state includes F, S, visits, gate error summaries, sign/commitment traces and any other target-derived learned auxiliary. The sensor may process the same bytes and maintain the same clock, but no answer-dependent content may survive OLD learning. The inherited `last` metadata may record elapsed-time/materialization bookkeeping; it must not encode the answer value, change a readout, or serve as a substitute learning trace. A permitted read-only computation of a loss is not permission to accumulate it.

Both counterfactual histories receive subsequent NEW teaching. Therefore the contrast concerns reuse of OLD learning after delay and intervening learning, rather than simply comparing a trained model with an entirely untrained model.

The environment may know the correct sum to generate feedback. The learner may see only the delivered digit, including wrong ones. Missing answers produce no update and no visit increment.

## 5. Shared state and update notation

All learner arrays start at zero. Eligible active coordinates are `I`, and `z_i=1/sqrt(|I|)`. Every configuration predicts using the same `v=z·(F+S)` cached at `=` before arrival of the target.

On an actually arriving answer digit `y`:

```text
r = y − cached_v                     # raw observed residual; not truth residual
e = clip(r, −1, +1)
eta_i = 1 / sqrt(1 + visits_i)        # visits before this observation
delta_i = eta_i × z_i × e
```

Every permitted present-target observation increments active `visits_i`, capped at 65,535, even when a candidate's content gate closes. This keeps the exposure-based learning-rate clock aligned with STABLE; gate rejection cannot secretly preserve a larger learning rate. No increment occurs in forbidden OLD learning or on missing targets.

F and S remain float64 and individually bounded in `[−2,+2]`. Inactive coordinates are untouched. None of the three configurations has unconditional time decay. All parameters below are fixed before DEV; there is no gain, threshold or decay sweep in this design.

## 6. Exact three configurations

### STABLE — accepted baseline update

```text
F_i ← clip(F_i + 0.2 × delta_i, −2, +2)
S_i ← clip(S_i + 0.8 × delta_i, −2, +2)
```

Rest leaves content unchanged. This is the predecessor's STABLE numerical update, with only the common prospective schedule of §3 changed.

### U042-WG1 — local residual write gate; forget gate closed

Add one float64 error-summary coordinate `a_i ∈ [0,8]`, initially zero. Compute the gate from the **pre-observation** summary and visits:

```text
if abs(r) <= 0.10:                    g_i = 0
elif visits_i < 8:                    g_i = 1
elif abs(r) <= max(1, 2 × a_i):        g_i = 1
else:                                g_i = 0.25

F_i ← clip(F_i + 0.2 × g_i × delta_i, −2, +2)
S_i ← clip(S_i + 0.8 × g_i × delta_i, −2, +2)
a_i ← (7 × a_i + min(abs(r), 8)) / 8
```

The summary updates only on a permitted delivered target, including observations with `g=0`. Visits then increment as in §5. The gate is independent of retention: there is no erase or forget operation. Closing the F/S gate does not mean that all memory writes stop: error-summary, visit and bookkeeping writes are counted too.

Hypothesis: suppressing near-zero corrections and attenuating unusually large observed residuals may save redundant writes and reduce the impact of isolated wrong examples. The warmup and nonzero outlier gain avoid permanently refusing to learn an initially large answer. Risks: a surprising correct example may be attenuated; shared features can contaminate error summaries; gate arithmetic may cost more than the writes it saves.

This is a new bounded engineering instance of U042's write branch. It is not a test of all independent write/forget gates or a claim of identifying noisy examples correctly.

### U021-SF1 — provisional content with agreement-triggered commitment

Add one int8 signed run `c_i ∈ {−2,−1,0,1,2}`, initially zero. F is provisional content; S is committed content. On a permitted delivered target:

```text
F_i ← clip(0.5 × F_i + delta_i, −2, +2)

q = sign(r) if abs(r) > 0.10 else 0
if q == 0:                            c_i ← 0
elif sign(c_i) == q:                  c_i ← q × min(abs(c_i) + 1, 2)
else:                                c_i ← q

if abs(c_i) == 2 and c_i × F_i > 0:
    S_new = clip(S_i + F_i, −2, +2)
    transferred = S_new − S_i
    F_i ← F_i − transferred
    S_i ← S_new
    c_i ← 0
```

Visits then increment as in §5. Actual clipping and actual transferred mass must be logged. The alignment condition prevents commitment of a provisional value whose sign still contradicts the current run. No phase-end flush or label-dependent commitment is permitted.

Transfer itself conserves `F+S`. The variant is **not** the previously equivalent TRANSFER arm: its `0.5 × F` attenuation acts at subsequent permitted local feedback events, and persistent content is written only on the specified agreement rule. Those operations change the prediction trajectory. Rest alone changes neither F nor S; there is no hidden wall-clock consolidation.

Hypothesis: provisional updates may absorb inconsistent observations while repeated same-direction corrections commit to stable content. Risks: agreement on shared features is not independent evidence of truth; legitimate opposing gradients may prevent commitment; attenuation can erase useful provisional content or slow learning.

This is a compound but fully specified U021 instance, not a pure estimate of one scalar parameter or a biological replication. Its test concerns the whole update rule. It does not certify adaptation to a changed addition rule, which is outside this experiment.

## 7. Readout, checkpoints and causal audits

At each of OLD_END, DAY1, NEW_END and FINAL, score all 25 expressions on a read-only copy. No scored expression receives an answer byte. Save per-expression emitted digit, scalar value, probabilities and checkpoint identity, not only a mean.

Also score an erased-content copy at FINAL: erase F/S and reset learned summaries, signed runs and visits. Its output must follow the disclosed zero-content ordinal prior. Erasure must not change the parent or fixed representation.

Required evidence checks, applied to every completed job:

1. Complete held-out orbits absent from all OLD/NEW teaching; no disguised reverse-orbit exposure.
2. W and N_OLD have identical delivered bytes, birth state, fixed assets and parent clocks.
3. N_OLD's forbidden OLD content and target-dependent auxiliaries remain exactly at birth values.
4. SHUFFLED learns from its delivered digit rather than the original target or an audit label.
5. Predictions precede answer arrival; newline-only missing targets produce no teaching.
6. Query copies cannot modify parent content, visits, error summaries, signed runs, caches with learned labels or clocks.
7. Every active-coordinate update and checkpoint value matches an independent numerical reducer that does not call the learner's update functions.
8. Readout uses actual F+S. No countermodel, fitted coefficient or evaluator-selected candidate supplies the answer.

An E3 effect is `W−N_OLD` and `W−SHUFFLED` on held-out exact answers. Engineering representation priors remain present in all controls, so those priors alone cannot be credited as learning.

## 8. Endpoint and decision rules

Let `A[a,d,h,s]` be FINAL exact-answer accuracy on the eight held-out expressions for configuration a, dose d, history h and world s. The primary quality score is:

```text
H[a,h,s] = (A[a,10,h,s] + A[a,20,h,s]) / 2
D[s] = H[candidate,W,s] − H[STABLE,W,s]
```

This is a mean of two prespecified behavioral accuracies, not an AUC over event rows. Each world is the independent unit. Dose 40, old/new accuracy, earlier checkpoints, per-expression failures and probabilities are secondary/protection results.

### Full-dose protection — point guards, not statistical noninferiority claims

At dose 40, the candidate must satisfy all of the following on the confirmation world means:

- W OLD, NEW and HELD accuracy each ≥75%, and each no more than 5 percentage points below the contemporaneous STABLE result.
- Held-out W−N_OLD ≥20 pp and W−SHUFFLED ≥20 pp.
- W held-out accuracy exceeds the maximum declared shortcut score by ≥10 pp.
- All exposure, branch, numerical, finite-value and read-only audits pass.

These preserve the previous task's absolute/causal floors while allowing at most the stated measured loss against the stronger current comparator. They are **point guards**; passing is not permission to report a 5-pp statistical noninferiority guarantee.

### Quality superiority and learning attribution

For a candidate quality improvement, require jointly:

1. `mean(D) ≥0.05` and its one-sided 95% lower bound >0.
2. One-sided 95% lower bounds for `H[candidate,W]−H[candidate,N_OLD]` and `H[candidate,W]−H[candidate,SHUFFLED]` both >0.
3. The full-dose protection guards above pass.

Lower-bound method is fixed: paired Student-t over worlds if sample variance >`1e−24`; otherwise the conservative bounded-difference bound `mean − sqrt(2 × ln(20)/n)`. No zero-width bootstrap interval, post-hoc method switch or extension of n is allowed. Report the method used. Zero-variance uncertainty may yield UNRESOLVED rather than adoption.

### Resource decision

For each world, measure CPU seconds for the W histories at doses 10 and 20, including byte ingestion, prediction, target handling, gating/commitment and model-clock advancement. Sum those two costs. Common imports/JIT, receipt construction, hashing, independent audit, packaging and process startup are separately timed; they are also included in total experiment cost.

Core adoption additionally requires:

- candidate aggregate measured W CPU at doses 10+20 ≤ STABLE's aggregate measured W CPU at doses 10+20;
- candidate array-state payload ≤1.5× STABLE's array-state payload;
- no history-sized learned cache, replay, global decay sweep, full-state copy or digest on the normal byte/update path.

These are measured resource guards on the declared machine, not a confidence claim that performance will be cheaper on every future host. Report per-world paired CPU ratios, the aggregate ratio and uncertainty descriptively. Do not substitute fewer nonzero writes for measured CPU savings.

Final verdicts:

| Condition | Verdict |
|---|---|
| Quality, attribution, protection and resource gates all pass | ADOPT_SELECTED_ADDITION_EFFICIENCY_INSTANCE |
| Quality/attribution/protection pass, resource guard fails | RETAIN_STABLE_DATA_EFFICIENCY_SIGNAL_ONLY |
| Confirmation completed but any scientific guard is unresolved or fails | RETAIN_STABLE_NO_CONFIRMED_IMPROVEMENT |
| No candidate passes frozen DEV admission | RETAIN_STABLE_NO_ADVANCE |
| Integrity/technical failure | INVALID_OR_TECHNICAL_FAILURE; no scientific adoption |

No result claims that STABLE or a candidate maximizes ability across all 71 mechanisms. The narrower comparison is sufficient to decide whether this candidate replaces the current finite-addition baseline.

## 9. Shortcut and solvability checks

Recompute the following on each actual world × dose, using the same delivered training bytes:

- raw exact-expression lookup, majority delivered target; missing key falls back to overall majority target;
- canonical commutative lookup, with sorted operand key and the same fallback;
- frequency-only output, with lowest-digit tie break;
- left-operand-only and right-operand-only majority predictors, with the same fallback;
- zero-content ordinal model, fixed to digit 0.

Complete held-out orbits prevent exact and commutative tables from receiving those answers. Their fallback scores must still be reported; do not assume a universal 25% chance level in a nine-digit task.

An evaluator-only linear fit on the disclosed role/magnitude features may be recorded as a solvability witness. It is not a learner arm or an exposure-independent shortcut. It must never provide coefficients, targets or output to the MiniFly core. A witness failure is a fixture/qualification issue, not permission to alter held-out pairs after science starts.

## 10. Finite DEV and confirmation roster

Reserve fresh world namespaces:

- technical: 834001–834006, no science inference;
- fixed DEV: 834101–834108, 8 worlds, all 3 configurations × all 3 doses;
- confirmation: 834201–834264, 64 fresh worlds, STABLE and **at most one** selected candidate × all 3 doses.

Before locking, an exposure census must confirm these IDs have not previously been used. A collision blocks implementation admission; do not silently substitute IDs.

DEV is explicitly exploratory and never pooled with confirmation. Both variants and all numeric constants are fixed before DEV. It is not an optimization loop.

DEV candidate admission requires:

1. All technical and causal audits pass.
2. Full-dose point protection guards in §8 pass.
3. Mean primary difference ≥5 pp; both low-dose causal contrasts have positive point estimates.
4. Measured low-dose W CPU ≤1.25× STABLE and state ≤1.5× STABLE. This looser CPU threshold admits a feasibility candidate; it does not replace the final resource gate.

If both qualify, select the larger DEV primary difference; exact ties within `1e−12` select U042-WG1. Lock selection, source identity, manifest and all gates before examining any confirmation result. If neither qualifies, analyze/audit/package the negative screen and stop.

No held-out-dependent parameter refinement, replacement candidate, extra seed, second winner or confirmation retry belongs to this protocol. A later design, if wanted, needs a separately justified question.

### Multiplicity and precision

There are two exploratory candidate comparisons. Only one selected candidate is tested against STABLE on independent confirmation worlds. Adoption requires **all** confirmatory quality/attribution tests to pass, an intersection-union decision; they are not multiple opportunities to win. No separate claims of superiority for individual doses, checkpoints or the unselected candidate are authorized. Thus selection is not reused as confirmation and no program-wide historical m=26 family is imported into this new design.

The fixed n=64 is a bounded precision choice, not a one-hour quota or a claimed power guarantee. If the world-level SD of D is 0.15, its approximate one-sided t margin is 3.1 pp; if SD is 0.30, the margin is about 6.3 pp. An observed 5-pp improvement may remain unresolved. There is no postrun enlargement. This design accepts that uncertainty instead of inventing a minimal detectable effect from runtime.

## 11. Counts and resource envelope

A full job is **one world × configuration × dose**, with all three histories and its independent audit committed together. A history or partial checkpoint does not count as a full job. Technical tests are separate from science/DEV counts.

| Dose per expression | Records/history | Training bytes/history | Present answer opportunities/history |
|---|---:|---:|---:|
| 10 | 170 | 1,003 | 153 |
| 20 | 340 | 2,006 | 306 |
| 40 | 680 | 4,012 | 612 |
| All doses | 1,190 | 7,021 | 1,071 |

Missing targets contribute five bytes and silent padding, not a target byte. W/N_OLD/SHUFFLED have equal training-byte counts. N_OLD has fewer permitted updates by design.

| Stage | Full jobs | Histories | Training records | Training bytes |
|---|---:|---:|---:|---:|
| DEV: 8 worlds × 3 configurations × 3 doses | 72 | 216 | 85,680 | 505,512 |
| Maximum confirmation: 64 worlds × 2 configurations × 3 doses | 384 | 1,152 | 456,960 | 2,696,064 |
| Maximum total | 456 | 1,368 | 542,640 | 3,201,576 |

The maximum is conditional on one candidate advancing. No advancement means only the DEV roster is executed. No native histories are reused between configurations, doses or worlds. Immutable source assets and master input objects may be shared.

Array-state payload, excluding common fixed assets, pending query buffers and receipt data:

- STABLE: `(8+8+8+4) × 5,177 = 144,956` bytes;
- U042-WG1: add one float64 array = 186,372 bytes;
- U021-SF1: add one int8 array = 150,133 bytes.

Measure actual total persistent and peak temporary storage as well. Independent mutable births must not alias. Share only immutable assets whose sharing is disclosed and qualified. Extra audit snapshots are experiment storage, not production neuron state.

The predecessor's 384 histories took approximately 143 seconds on its recorded two-worker host. That does not establish a guaranteed duration here: dose, candidate updates and auditing differ. This roster should remain modest; quote runtime only after timing the final logging/audit policy. Do not grow n because it is fast.

Default execution plan, if later authorized: **two detached workers**, caffeinate attached to the supervisor, whole-job commits, no dependence on an interactive terminal. No concurrency expansion is part of this design.

Set a 10-hour session limit. Stop admitting jobs at 9 hours. Qualification must support a 30-minute per-job operational timeout; timeout is a terminal operational failure, not a negative mechanism result. With no late admissions, either whole jobs drain by 9.5 hours or failure is recorded; save a safe paused status within the remaining window. A paused session requires explicit resume authorization. Do not replay a partially executed trajectory to pretend it was never run.

Before launch, measure the worst qualified 40-dose job with the final audit/receipt policy, include startup and postprocessing, and project total wall time and disk. If the conservative projection exceeds 8 hours or 2 GB of new output, do not launch this roster under the present budget. Report the measured bound and seek a separately agreed schedule; scientific n and gates do not change automatically.

## 12. Qualification before any launch

The shipped artifact is a design. The following are acceptance requirements for the later implementation, not tests already completed.

1. **Baseline parity:** STABLE with the predecessor's original stream reproduces saved numeric/probe results from a technical fixture. Use saved records and the independent equation reducer, without rerunning completed science. The common new slot schedule is separately qualified.
2. **Mechanism identity:** a hand-calculated local fixture demonstrates U042's open/closed/outlier gates and U021's provisional attenuation, sign reset, aligned commitment and conserved transfer. U021 must produce a different trajectory from a time-only TRANSFER rule.
3. **No-write truth test:** vary every OLD delivered answer while N_OLD is disabled; all target-dependent content arrays/auxiliaries remain at birth. Wrong metadata must have no effect with delivered bytes fixed.
4. **Dose nesting / clock parity:** all shared real records match their master bytes; unobserved slots contain only rest; checkpoint parent clocks equal across doses/configurations/histories. Dropped padding and double day advancement must be rejected.
5. **Birth and asset parity:** each mutable learner is independently constructed; each receives the same canonical immutable birth assets and projection. Mutating one content state must not change any other.
6. **Independent equation audit:** validate full technical 40-dose histories for all configurations and all controls, then a technical end-to-end job for each dose. Do not use the learner's update function inside its auditor.
7. **Tamper rejection:** altered answer/update, forbidden OLD auxiliary update, wrong gate prestate, forged commitment, changed projection, held-out exposure, missing training row, reversed prediction/feedback order and query-parent mutation must each fail a named check.
8. **Operations qualification:** atomic receipt commits, bounded logs, stale-process diagnosis, safe pause, source-lock enforcement and packaging after a negative/no-advance result. Only qualification fixtures may test interruption behavior.

All technical witnesses and failures remain visible. A technical failure must be repaired and the actual revised implementation locked before DEV. The authorized scientific scope cannot be extended by a failed qualification.

## 13. Source lock, reporting and stopping

Do not edit predecessor sources, manifests or results. Implement in a separate version directory. Source-lock the exact runner, candidate equations, auditor, data generator, settings, runtime and referenced assets before DEV. Operational adapters may not alter science; changes require explicit identity/parity records.

Content updates touch only active coordinates. Any inherited full-support scan or full-sized cached representation is still counted and disclosed; sparse updates do not prove wholly sparse computation. Audit serialization, full-state hashing and read-only clone costs may be performed at record/checkpoint boundaries and counted separately, but must not be disguised as sparse neural work. Save compressed training traces, checkpoint content certificates and per-expression outputs sufficient for an independent audit. No full PN/KC CSV per byte is required.

Each receipt distinguishes `answer_observed`, `update_allowed`, actual nonzero F/S change, auxiliary writes, provisional attenuation, commitment transfer, clipping and elapsed-time bookkeeping. A closed content gate or zero residual is not recorded as a forbidden observation, and an allowed teaching call is not automatically counted as a nonzero content write.

The final report must include:

- verdict, actual E1/E3 evidence and external-feedback provenance;
- every dose × configuration × history result, including the unselected DEV candidate;
- per-expression errors, all four checkpoints and OLD/NEW protection;
- causal controls and all shortcut scores;
- predicted and actual CPU, wall time, writes, state bytes, peak RAM and disk;
- gate-closure/attenuation, provisional/committed mass and clipping diagnostics;
- fixed rosters, selection record, source identity, failures and exact committed/partial counts;
- analysis audit, report and complete result bundle, including valid negative results.

An execution failure stops the affected run, preserves data and requires user direction. No automatic retry, trajectory replay, changed parameter, expanded n or substituted variant is allowed. Publication to GitHub and new monitor creation require the user's separate request.

## 14. Decision after this experiment

If a candidate meets all gates, it becomes the **next tested instance of the same limited-addition core**, with measured learning-efficiency gains and explicitly bounded resource claims. It does not solve other arithmetic or authorize scaling.

If there is a data-efficiency gain with added CPU, preserve the finding and retain STABLE under the present compute objective. If neither candidate improves the fixed low-dose range, record that both specific variants failed to earn promotion. Do not automatically lower dose, increase noise, add recurrence or open a topology search to manufacture headroom.

The purpose is to make one bounded core-selection decision using already working byte input, persistent content and held-out answers. The experiment is not another requirement to prove that those modules can connect.
