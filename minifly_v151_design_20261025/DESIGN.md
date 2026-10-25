# V151 — native teaching-response operator

25 October 2026. **Design only. No implementation, native interventions, three-cycle exercise, training, or push is authorized by this delivery.**

The goal remains **better complete own-byte/EOS generalization to unseen addition per total training samples and actual compute**. V151 is one bounded E0/E1 diagnostic of the existing native output-learning path. It does not replace the arithmetic track, start an NLP campaign, establish a new core, or reopen addition holdbacks.

## 1. Why this experiment

V150 established that a coherent categorical teaching signal can learn conditional output distributions when context values are directly writable. It did not establish that the native stores can execute the same optimization. Native B/C both underperformed A at the tested dose. The B/A coefficient-energy ratios were about 0.259 and 0.258, consistent with roughly half the coefficient amplitude. That is a real difference in delivered signal, not proof that signal gain caused the loss difference.

The attached audit sensibly prioritizes actual native response over another capacity increase. Three corrections are necessary:

- V150 saved six final native checkpoints, **no native midpoint or birth checkpoints**. Use the final states and a canonical newborn reconstructed without training. Do not replay V150 to manufacture intermediate states.
- A codebook's net drift is not its accumulated learning. Small drift could reflect small increments, cancellation, constraints, or the data. Measure an actual one-step code update separately.
- A typical low logit span limits concentration at that position. A mean span/RMS cannot establish a bound for every position, or prove that low amplitude uniquely caused failure.

V151 asks: **from the same frozen native state, what does a delivered teaching coefficient change, and does B improve the taught target more efficiently after a separately calibrated displacement match?** Any answer is useful; passing an interface check never becomes an arithmetic success.

## 2. Lineage, state and exposure

Executable parent: V150 commit `94cbf4e6811e9e960657e0a970dbca3d5efd638f`; native parent source remains pinned by V150's `SOURCE_LOCK.json`. Preserve FE-PERM normalization, canonical connectivity, 16 stores, native fast/slow/adaptation equations, event timing, `fresh_adapt=True`, `homeo=False`, temperature 1 and all 257 legal classes. No attention, Transformer, ELM, arithmetic parser, numerical features, external decoder, learned inverse operator or backpropagation through native dynamics/time is added.

Primary anchors are Bryant/Buster × reconstructed newborn, saved A-final, saved B-final: six anchor cases, only four of them trained histories. The repeated canonical newborn is not an independent replication. Both A and B teaching conditions are applied to clones of **each identical anchor**; A-final versus B-final scores are not used as an estimate of the effect of the new intervention.

The two saved C-final states are reserved for the separate factorial in §8. Their learned codebook is frozen for their native-only condition. ASSETS.json lists exact checkpoint hashes. Missing/mismatched assets block the affected assay; no replacement by replay or by a different state is allowed.

Use only the already taught V150 training partitions: Bryant records [0,47), Buster [0,33). Source sites, next-byte/EOS labels and peer queries are fixed in SITES.json without running a native model:

| Per text / anchor | Selection |
|---|---|
| Two gain-calibration sources | Records 0 and 1, position min(8, length−1) |
| Three scored byte sources | Records 2/3/4, positions min(8/16/4, length−1) respectively |
| One scored EOS source | Shortest record of length ≤16 at index ≥5, ties by record index; target at record end |
| Two peer queries per source | Different exact prefixes, one same next-symbol label and one different label, selected by fixed SHA-256 rank from the training pool |

All source/query prefixes are at most 16 bytes; this is a declared assay sampling limit, not a truncation of an existing longer query. Peer candidates exclude all six source-prefix byte forms, with positions 0 through min(record length,16). Selecting peers by next-symbol label makes them evaluator-chosen contrasts, not semantic neighbors or evidence of inferred arithmetic relations. Duplicate peer reuse is disclosed and is not an independent trial.

Calibration and scored source prefixes are distinct. Buster calibration source 0 is an empty prefix because its first record has one byte. Keep this prespecified structural-null case: FE-PERM starts at zero and the frozen encoder has no positive KC activation for that empty input. Record active KC count and response; do not interpret a zero response as a wrong gradient sign. Report the effective non-null calibration count; the aggregate match may rely on the other calibration source. Do not replace the null case after inspection. No V150 calibration/development-evaluation block, Alice, or addition assessment is accessed. Prefix bytes are sensory input with writes off; labels are used only for the explicit one-event counterfactual feedback or read-only scoring. All repeated label applications count again. First pre-feedback class choice is recorded before the intervention; post-feedback same-cue changes are E1, not unseen-relation learning.

## 3. Quantities and units

Let P=I−11ᵀ/16, u=Pv/4, z=Cu, and p=softmax(z). Every byte and EOS=256 remains legal. Natural-log loss is L=−log p(y); bit loss is L/ln(2).

For the common frozen code C:

- `g = P Cᵀ(p−onehot(y))/4`, the correct output-value gradient.
- `e_A = [tanh(Pv/2)−C_y]/2`, the old bit residual, not generally that categorical gradient.
- `a_A = 0.1 s(n) e_A`; `a_B = 0.1 s(n) g`.
- `s(n)=1/16+(15/16)min(n/1024,1)` with n the anchor's previous observed targets. Newborn n=0; saved finals inherit their original counters. No reset, acceleration or extra pass.

The physical signed-coefficient response is

`K_qx = ∂v_post(q) / ∂a_x | a=0`, and `G_qx = −K_qx`.

For any delivered coefficient vector a, the fixed-code first-order prediction in bits is

`ΔBPS(q) ≈ g_qᵀ K_qx a / ln(2)`.

Here g_q is evaluated at the **matched zero-coefficient post-event query**, not the temporally earlier pre-event output. Eta and warmup are already inside a and must not be multiplied twice. The query gradient can use a different y_q from the source target y_x. A negative prediction means loss reduction; sign is never repaired after seeing results.

Each physical store is separate, so uncentered K should be diagonal in compartment coordinates in this configuration. Centering and the categorical readout still couple outputs. Report off-diagonal residuals, raw Δv and PΔv; common-mode changes are invisible to the decoder. A nonnegative self gain alone says nothing sufficient about cross-context usefulness.

## 4. A counterfactual event must hold timing and input fixed

Each source starts from a fresh clone of its frozen anchor. Recreate its prefix through `_prompt(b'')` and passive `_event(byte)` calls, exactly as in the parent evaluation path. No prefix teaching occurs. Read the pre-event values once and compute the declared coefficients before any updates.

Then clone that prepared state independently for each branch. Advance exactly one native DELTA; obtain the same current KC code; call each store's unchanged `teach_signed(b=0, s=a_j, write=True)` at the pending time. **Do not feed the target byte.** V150's normal `_event(y,y)` would feed it and change context. This assay separates write effects from consuming the next byte.

The principal zero control performs the identical signed-event path with a=0. It is paired with an ordinary passive event and a second zero branch. These checks measure passive-path equivalence and the numerical/repeatability floor. Do not compare a taught post-state only with its pre-state: native decay, adaptation and elapsed time otherwise confound the write.

For operator-pulse/control branches, scored target branches and C-factorial branches, query four paths on independent read-only clones. The 24 gain-calibration A/B target branches are the exception: query only the immediate source displacement norm, with no loss calculation before the gain lock. The four paths are:

1. Immediate source context, without another prompt/reset.
2. Reset/re-prompt the exact same source prefix.
3. Reset/re-prompt the fixed same-target peer.
4. Reset/re-prompt the fixed different-target peer.

Each query has the identical byte, time and adaptation path in its paired zero control. `values()` itself looks one virtual DELTA ahead on clones; preserve that behavior and count its cost. Differences between immediate and reset self response are findings, not reasons to change the protocol. No branch becomes the starting state of the next case.

Artificial coefficient pulses are technical interventions without an observed target and do not advance the warmup observation counter. Each real target branch advances its own n once; no branch's count is propagated to another clone. Changes in internal technical teach counters are logged separately from observation counts. State hashes/checkpoint hashes guard immutable anchors; clone/hash operations are evaluator overhead, not a proposed scalable learning algorithm.

## 5. Identify the operator, including trained states

At all six source sites in all six primary anchors (36 sites), use ±h e_j for every j=0…15, at h=1e−3 and h/2=5e−4. Each pulse starts from the identical prepared source state. Deliver the coefficient directly; the denominator is its **actual value**, not the nominal rate or warmup.

For each query path:

`K[:,j](h) = [v(+h e_j)−v(−h e_j)]/(2h)`.

Save K(h), K(h/2), one-sided deviations from the zero baseline, scale disagreement, raw/centered displacement and off-diagonal residuals. The two h values measure sensitivity to scale; they are not an outcome-selected rate menu. Report zero-repeat error and passive-versus-zero differences before interpreting a small response.

Use the h/2 estimate, without fitting, to predict the actual full-vector interventions in §7. Report prediction residuals in Δv, centered Δv, logits and bit loss. Larger errors at the full teaching amplitude may reveal nonlinearity or loss curvature; label the observed error rather than forcing a linear explanation. Native finite differences remain an **evaluator measurement** and are never supplied to the learner as a Jacobian, inverse, preconditioner or representation.

Per source site: 64 pulse branches plus zero, repeated zero and ordinary passive =67 branches. Total: 2,412 branch events, 2,304 single-compartment nonzero increments. These are repeated interventions on 36 prepared contexts, not 2,304 independent learned lives.

## 6. Match displacement without optimizing scored loss

For each primary anchor separately, apply A_COMMON and B_COMMON to the two calibration sources. Only their measured immediate centered displacement norms relative to zero are used:

`λ = sqrt[ Σ_cal ||PΔv_A||² / Σ_cal ||PΔv_B||² ]`.

Execution order is fixed: preflight; the twelve primary calibration sites and their A/B interventions; lock all six multipliers; then the twenty-four scored operator sites and their three conditions; finally the C factorial. Freeze one positive scalar λ per anchor in GAIN_LOCK.json **before scored-site interventions, vectors or loss deltas are produced or opened**. Calibration loss is not a fitting objective. Do not use scored norms, class probabilities, query losses or interference to choose λ. All sites are already locked, and no alternative calibration contexts are tried.

The multiplier must be finite, within [1/4,4], and representable by the signed interface for every delivered component. This interval is a finite intervention/resource guard, not a scientific success threshold. No clipping, sign changes, diagonal inverse or fallback rate search is allowed. If calibration response does not exceed the repeated-zero numerical floor, the ratio is not identifiable. An out-of-range, nonfinite or inadmissible match is `MATCH_UNIDENTIFIABLE`; report it and run only the fixed-rate branches already valid.

The pulse operator is used to explain/predict the comparison, not to tune λ. Using only a scalar prevents a response-matrix optimizer from being smuggled into MiniFly. This is still evaluator-assisted calibration; its cost is paid, and it is not evidence for an autonomous production gain controller.

## 7. Score three one-event conditions

On four scored sources per primary anchor, compare:

| Condition | Actual teaching vector | Purpose |
|---|---|---|
| A_COMMON | a_A | Unchanged old teaching direction and nominal rate |
| B_COMMON | a_B | Correct categorical gradient at the same nominal rate |
| B_MATCHED | λ a_B | Same B direction after independent displacement calibration |

C is fixed throughout. No temperature change, learned bias or code-learning arm enters this primary comparison. Shared-code A/B directions always come from the identical pre-state, x and y.

Primary measurements are paired immediate-source ΔBPS, own next-class choice (including EOS), actual ||PΔv||, and the directional first-order prediction. Also report reset-self response, same/different-target peer loss changes, KC overlap, gradient-response alignment, logit span and per-position probability concentration. A same-label peer is not necessarily semantically related; overlap and byte labels describe the assay contrasts only.

Report displacement ratio on scored sources. Matching in calibration does **not** imply matching on scored contexts. If it fails to transport, do not retune or call it an equal-effective-dose comparison. Mean coefficient energy, raw plastic fast/slow changes and native elapsed time remain available to distinguish nominal rate from realized effect.

Use paired raw per-site values and equal-source aggregates within each anchor. Show newborn and each final state separately, including the EOS source. Cases share assets, prefixes and histories; do not manufacture an n=24 independent-seed significance test. No full-sequence exact-output or stopping competence is inferred from a one-symbol assay. Missing/unrepresentable branches remain explicit rather than being dropped from favorable means.

## 8. Small ancillary factorial: is code learning actually dormant?

Use only the two saved C-final anchors and each text's four scored source sites: eight cases. Independently clone the same prepared state for four branches:

- ZERO: zero native event; frozen saved C.
- V_ONLY: native B coefficient at the common 0.1 rate; frozen saved C.
- C_ONLY: zero native event plus the declared V150 code update.
- JOINT: same native B coefficient and same code update, both computed from the pre-state.

Code update: `ΔC = project[C−(0.1/16)s(n)(p−onehot(y))uᵀ]−C`, with row centering and row norm ≤4. Every real-target branch counts once even if it updates both objects. No output values/code are imported from the ideal fixture, and no primary λ is extrapolated to a C-final anchor.

Measure ||ΔC||, its loss-direction inner product, projection effect and the full one-step outcome on the same four query paths. The first-order joint prediction includes both `g_v,qᵀΔv` and `<g_C,q,ΔC>`, evaluated at the zero post-query state, divided by ln(2). Report the joint residual and `ΔL_joint−ΔL_v_only−ΔL_C_only`. This separates native response, direct code effect and interaction; net historical code drift cannot do that.

Also log per-query r=RMS(Pv), ||u||=r, span(z), p_max, and the bound `p_max≤1/[1+256 exp(−span(z))]`. With code row norms ≤4, span≤8r is a looser consistency check. Apply bounds **per position**, never to an average RMS/span as if it bounded every position. This factorial is not a test that learned code is necessary; V150's better BIAS control remains part of the interpretation. No necessity claim or larger decoder is proposed.

## 9. Evidence, countermodels and decision rules

**Claim and level:** physical coefficient response and derivative checks are E0. A real-target one-write change on an already taught context is an E1 mechanistic diagnostic. Cross-context results here have no E2/E3 promotion claim: peers were selected from previously taught data, and there is no causal never-taught arithmetic task.

**Exposure/countermodel:** exact-key and canonicalized-key tables can improve taught responses after feedback. A frequency update can improve same-target peers without learning any relation. Therefore self improvement or a same-label peer gain is not an abstraction criterion. Report these alternatives explicitly; no lookup table is inserted into a candidate.

**State/control:** immutable anchors, paired zero/passive paths and frozen code identify a one-event effect conditional on the selected state and query path. The algebraic direct-output update `v←v−a_B` is a numerical interface reference only, not native performance or an answer source. The inherited native A condition is the behavioral comparison. All clones and calibration costs are charged.

**Interpretation:**

- If B_MATCHED improves over B_COMMON and A on the same tested native states while scored displacement matching actually holds, this supports a dose/coupling explanation **for those one-step responses**. It does not show that V150's whole-run failure was mainly caused by gain, or that multi-event learning will improve.
- If B_MATCHED improves B_COMMON but remains behind A, gain is a partial explanation at most.
- If B does not improve the self target after matching, examine the measured direction, response magnitude, passive dynamics, loss curvature and interaction. Do not automatically add another mechanism or retry longer.
- If matching does not transport, is numerically unidentifiable, or exceeds the guard, the matched comparison is inconclusive. That is a result.
- Cross-query losses distinguish immediate response from address interference; they do not establish useful semantic generalization. Mixed signs, adverse EOS response and trained-state failures must remain visible.

Use the natural zero-loss-change null and paired signs/effect sizes; no chosen accuracy cutoff turns this assay into a success. Numerical tolerances qualify reproducible arithmetic, not biological efficacy. No result authorizes adoption or another training campaign. End V151 with a compact mechanism report and one concrete next-decision recommendation. If coupling is adequate, a future separately authorized fixed-budget native addition comparison must test the original goal rather than extending this text diagnostic indefinitely.

## 10. Counts, limits and planned qualification

| Work | Fixed maximum |
|---|---:|
| Primary operator sites / pulse branches | 36 / 2,304 |
| Operator event branches including controls | 2,412 |
| Gain-calibration real-target branches | 24 |
| Scored A/B/B_MATCHED real-target branches | 72 |
| Ancillary C branches, including zero | 32 |
| Ancillary real-target branches | 24 |
| Technical preflight real-target branches | ≤8 |
| **Total observed-target applications** | **≤128** |
| **Nonzero compartment write increments** | **≤4,224** |
| Signed API calls including zero | ≤40,320 |
| Branch event paths including passive/preflight | ≤2,556 |
| Native vector readouts including pre-event/preflight | ≤10,388 |
| Full sequential training histories / addition records | **0 / 0** |

Each full target event uses 16 compartment calls; coordinate pulses use one nonzero compartment. The nonzero bound is 2,304 pulses +1,536 primary target increments +256 C-factorial native increments +128 preflight increments. The signed API upper bound includes all zero components and controls. Preflight has at most sixteen event branches total, at most eight of which receive a target; its paired zero/passive branches must fit within those sixteen. The 256 preflight readouts include its pre-event reads. Any additional fixture would exceed this roster and is not authorized. Calls with nonzero coefficients internally evaluate native 0/1 writes and a passive evolution; count those evaluations too, without calling them external observed targets. Query-prefix/passive byte steps, duplicates and record resets are extra sensory/compute work, explicitly logged rather than treated as free training samples.

Native readout maximum comprises 9,648 operator query vectors, 24 calibration norm vectors, 288 scored vectors, 128 C-factorial vectors, 44 source pre-vectors and ≤256 preflight vectors. Query replay is sensory-only on discardable clones; it never replays a training trajectory. Report actual bytes, prefix events, all 16-store readouts, native inner evaluations, clone/hash/I/O time and bytes separately. Avoid retaining all branch states: process serially, keep small arrays and hashes, discard each clone.

Limits: one worker, one numerical thread, 1,800 summed process CPU seconds, 2,700 wall seconds, measured resident-memory target 1 GiB. Check remaining allowance before starting a source site; stop at a whole-site boundary where possible. If a hard limit interrupts a site, mark it partial and exclude it from committed site aggregates. Never extend budget, reset the session clock, replay completed interventions, retry a numerical/scientific failure or reconstruct missing assets by training. A resource pause needs explicit continuation authorization. Caffeinate may follow an authorized future supervisor while active; none is launched for this design delivery.

A later authorized three-cycle exercise should: (1) verify formulas, passive/zero equivalence, no target feeding, clone independence and exposure counters; (2) run the exact bounded site roster with the gain lock preceding scored outcomes; (3) audit source/checkpoint hashes, saved vectors, counter totals and a resume fixture within the eight-target preflight allowance, then revise the interpretation. Numerical qualification must disclose float64 tolerances and repeatability floors before opening scored deltas. More targets are not a fallback if qualification fails.

## 11. Deliverables and current authorization

This directory contains DESIGN.md, AUDIT_REVIEW.md, SPEC.json, SITES.json, ASSETS.json, PROVENANCE.json, STATIC_CHECKS.json and STATUS.json. The original V150 design/results/checkpoints and the arithmetic pointer remain unchanged. The diagnostic pointer may link to V151 as a **design-only next proposal**, preserving V150 as its completed parent.

Future execution deliverables, if authorized: source lock, site ledger, preflight certificate, GAIN_LOCK.json, compact operator matrices/predictions, per-site response and code-factorial results, immutable source-asset checks, counts/resources, report and audit. Raw checkpoints, corpus text and per-branch states remain local. There is no new fit, efficacy result, or implemented kernel in this design.
