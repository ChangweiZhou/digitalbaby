# V146-R1 — identity-preserving map learning and a controlled shrinkage test

24 October 2026. **Prospective design only. No V146 learner, three-cycle exercise or main experiment has run.** This refines [Claude's V146](CLAUDE_V146_ORIGINAL.md) at `c243b5ae8b0e3c0d944e014cf76a44d4e32c477b`. Original designs, analysis and V143–V145 results remain unchanged. [REVIEW.md](REVIEW.md) records the reasons for the revisions.

**Goal:** improve complete own-byte/own-EOS unseen-addition generalization per **all training experience and actual compute**, while developing MiniFly. Keep gradual warmup, generic byte processing, no attention, Transformers, ELM or arithmetic-specific candidate structure. V146 also retains Claude's proposed **local-only learning restriction**, including reference learners; it does not introduce a new exact-gradient or BPTT learner.

## 1. Question and evidence boundary

Two hypotheses must be separated:

1. Does an identity-preserving, generically learned context map improve held addition over an otherwise identical identity-only route?
2. Does scheduled weight shrinkage improve that result over equal continued training without shrinkage?

V145 supports neither hypothesis yet. Its original-symbol reference acquired all taught answers but scored 13.33% held; its line diagnostic scored 38.75% taught / 26.67% held, and ordered learned codes scored 6.67% BP / 10% DFA held. The line's weak acquisition motivates preserving identity, but does not establish that loss of separability was its sole failure. The ratio held/taught is not an acquisition-controlled generalization measure.

Claude's saved size analysis is **modular addition with ordered-pair withholding, class outputs, one seed and exact-gradient AdamW**. It changes the alphabet, number of labels, training examples, parameter count and total presentations. At 20,000 steps the 50% p=10 fit has 1,000,000 example presentations; the 75% p=47 fit has 33,120,000. Some held pairs have their reversed pair in training. These results do not identify a critical data size for ordinary addition or prove that a rule cannot be cheaper at ten symbols. No new size-scaling task is added here.

A SOM may organize contextual traces, but repeated `*` under a permuting, decaying and normalized trace is **not guaranteed to form a monotone length coordinate**. That is an outcome to measure. Likewise, shrinkage before hard top-k does not necessarily simplify behavior: positive scaling can leave every selected KC unchanged.

## 2. What is being built, and what is not inherited

| Component | Status in V146 |
| --- | --- |
| FE-PERM identity receptor, timing, canonical PN/KC assets and deterministic top-k convention | Source-pinned parent dependencies |
| Generic topographic map and byte-to-map association | New proposed local learner |
| Augmented combination layer and nine-bit fixed-feedback teaching | V143-inspired adaptation, not unchanged V143 |
| Opponent delta naming store | Explicit evaluator-side surrogate; not native Full151 memory |
| Native banks, combined native system and persistence | Not tested by this design |

A naming-store rate resembling native acquisition does not establish native equivalence. Any calibration gets a separately logged development fixture and cannot confer native efficacy. The map and naming store are not allowed to access an operand table, numeric byte index, carry bit, answer length, arithmetic decoder or target-derived feature.

Evidence level: interface checks E0; taught acquisition E1; withheld-relation gains can support **bounded E3 reference evidence** with the causal controls below. This experiment has no intervening-learning phase and cannot establish OLD retention, a persistent core or successful native integration. No V147/native port is automatically authorized by a favorable screen.

## 3. Preserve identity through the entire interface

For every payload byte x, produce the same two routes:

- `R(x)`: the unchanged 88-dimensional identity receptor;
- `m(x)`: the output of one generic learned association from R(x) to M map units.

There is no `isdigit`, numeric-rank branch, operand position, special treatment of `+`, `=`, `*`, spaces or newline in the learner. All 256 payload values are accepted. EOS 256 and API boundaries remain distinct controls; EOS is never fed as a payload byte.

**Use separate trace blocks.** The 88-channel identity trace must run the parent's actual FE-PERM event API bit for bit. The M-channel map trace uses the same event times/decay and a separately seeded fixed permutation. Do not permute identity and map coordinates together: that would change the supposedly preserved identity computation. Readout input is `[p_ID ; g p_MAP]`; do not globally renormalize it in a way that rescales p_ID according to learned-map activity. A zeroed map block must reproduce the declared identity-only input exactly.

The association returns a positive unit-mass q(x); its receptor contribution is `m(x)=q(x)/||q(x)||₂`. Set g once to the RMS L2 norm of all 256 identity birth receptors. This bounds per-byte map amplitude independently of tuning width. Freeze g and normalizers before data. Unit mass alone would not provide this amplitude control. Map dimensionality and all fixed arrays remain present in map-disabled controls, with zero map output.

Prompt bytes, observed answer-prefix bytes and generated prefix bytes all take the same route. Generation uses only its own prefix and own EOS, with eight output decisions. Invalid codes and cap exhaustion are failures, not implicit EOS. Generation and probes must not modify weights, warmup counters, map prototypes, adaptation, clocks used for future training, or RNG.

## 4. Specify the generic map, rather than assume a number line

Proposed fixed starting specification: M=32 units on an **open one-dimensional sheet**, not a ring; no unit has an assigned byte or quantity. This is a declared generic topographic prior, not a proven biological requirement. Only data may associate a byte with a location.

Let prototypes `P_i` be 88-vectors and `A` an M-by-88 association matrix with bias a. Initialize prototypes from a seeded zero-mean Gaussian with identity-receptor birth RMS; initialize A/a to zero. Break distance ties by lower unit index. For every observed payload event x, regardless of byte identity:

1. Read the **pre-byte identity context** u, before feeding x. The first context of a reset record is the defined zero vector. Do not read the subsequent digit or EOS to build u.
2. Compute `j=argmin_i ||u-P_i||²` and `w_i=exp(-(i-j)²/(2σ²))`. The normalized old-state w is the association target t.
3. Compute `q=softmax(A R(x)+a)` using only the current byte's identity. This is a fixed feedforward population classifier, not query/key retrieval or attention over stored episodes.
4. Apply local updates from the same pre-update state:
   `P_i += η_map w_i (u-P_i)`;
   `A += η_assoc (t-q) R(x)ᵀ`;
   `a += η_assoc (t-q)`.
5. Feed x into the identity trace. Do not perform a separate label-specific write on the final byte.

The teacher for A is the map's own pre-update context response, not the evaluator's k or digit identity. Every `*`, digit and other payload event uses this rule. EOS ends the record without a payload-map write. Readout competition is generic and dense over 32 units; it is not evidence of computational sparsity or scalable memory access.

Both rates start at 0.01 times `s(n)=1/16+(15/16)min(n/1024,1)`, where n is the number of prior observed payload events. Shrink σ geometrically from 4 to 1 over the predeclared Q+C payload-event count. These are fixed development starting values, not claims of optimality. No additional map-rate/topology sweep is admitted silently. The same schedule and input order apply to causal arms.

Persist P/A/a and measure recall from **R(x) alone** after Q and again after C. C can overwrite quantity-associated contexts: record that possibility rather than assuming Q and C are automatically compatible. Report map variance, code separation, tuning width, run-length/order associations and association errors descriptively. Degenerate or zero-denominator metrics are undefined, not successful grounding. No geometry score can replace behavioral addition.

## 5. Experience and lawful controls

Proposed bounded pretraining dose:

- **Q:** 64 epochs; each epoch presents all ten records `*` repeated k times followed by the byte for k, then EOS, in seeded random order. The k=0 record is `0` then EOS. This is 640 records, 3,520 payload bytes and 640 EOS observations.
- **C:** 64 full records `0123456789` then EOS, with no cross-record 9-to-0 edge: 64 records, 640 payload bytes and 64 EOS observations.
- **Total Q+C:** 704 records, **4,864 observed tokens including EOS**; 4,160 map/association update opportunities. Every digit appears 128 times; the repeated marker appears 2,880 times. The marker dominates exposure and may dominate the SOM; equal digit counts do not mean a uniform input distribution.

These quantities are starting doses to qualify, not previously authorized training. Q is explicit quantity-to-symbol instruction **in the data**, not absence of semantic help. No claim of learning from addition facts alone is allowed. A different marker/renamed vocabulary can be a technical equivariance fixture, not an added main task.

For **MAP-PERM**, draw one fixed nontrivial byte permutation π of the ten digit symbols, independent of addition outcomes. Apply it to *every digit occurrence* in both Q and C: k stars are followed by π(byte(k)); C is π applied to `0123456789`. Keep addition in its original byte vocabulary. Do not instead change the star count by π in one phase and relabel digits in the other: that would give inconsistent controls. Token counts, lengths, EOS and warmup opportunities are exactly preserved. A permutation may still help addition; equal performance limits numeric-order attribution rather than disproving all structured representations.

**SYMBOL arms** also observe the same Q/C stream, with map writes disabled and map readout zero; their addition counters start at zero like every other arm. **MAP-NOEXP** sees the identical Q/C events and schedule with P/A/a writes disabled; it is not a zero-exposure comparator. Its map is an untrained uniform code. A separate birth/extra-addition control tests spending the pretraining budget differently.

Map plasticity stops before addition for all main arms. All trainable downstream modules begin from the same birth state/substreams, with the same taught-example order. No persistent Q/C context is carried into the first addition prompt.

## 6. Local combination, expansion and naming

Proposed starting H=128; no simultaneous H/M/top-k growth sweep. Initialize the identity columns of W with the pinned V143 identity initialization, and map columns from an independent `N(0,1/M)` stream. All controls have the same full W shape. Fixed V maps H units to 88 virtual receptors; import the exact canonical PN/KC connectivity and side-wise top-k counts from the locked parent asset, including its deterministic tie convention and V143 learned-layer `eligible_positive=False` rule. Do not claim a new random expansion is identical to native wiring.

For an answer-token event, using pre-update state:

```
a_h = W p + b; h = relu(a_h)
x = binary_sidewise_topk(canonical_drive(V h))
v_j = (theta_plus[j]-theta_minus[j]) dot x / sqrt(max(1,sum(x)))
e_j = (tanh(v_j/2)-signed_bit_j(observed_token))/2
```

Nine opponent pairs emit all 256 payload codes and EOS; invalid nine-bit codes are not masked to digits. Opponent arrays start at zero. Their delta rule is:

`theta_plus[j] -= alpha s(n) e_j x / sqrt(max(1,sum(x)))`;
`theta_minus[j] += alpha s(n) e_j x / sqrt(max(1,sum(x)))`.

The feature update uses one fixed seeded `G_ij ~ N(0,1/9)`:

`d=(G e)*(a_h>0)`;
`W -= 0.003 s(n) outer(d,p)`; `b -= 0.003 s(n) d`.

Here n is completed addition records, not answer-token count; targets within a record share its warmup factor. All residuals and gates use old state. This is a direct fixed-feedback teaching proposal, **not the gradient through top-k**, and no claim of credit alignment is assumed. FROZEN-LAYER disables W/b writes. No ordinary weight decay is stacked beneath the scheduled-shrink intervention. Alpha is the sole naming-rate choice in the finite development menu below. Failed finite/numerical checks invalidate the fit; they do not license hidden resets.

Training uses observed answer-prefix bytes and EOS during this declared supervised phase. Evaluation uses generated prefixes. Scalar-only learning, blurry feedback, memory arbitration, replay and new tasks are out of scope.

## 7. Shrinkage: define a causal intervention

Use equal training doses with a **fixed anchor at pass 256**, final pass 1,024. Every arm sees the same taught pool and record order. At the anchor, a paired on/off comparison may fork one checkpoint into identical states; record the shared logical exposure and physical compute once, not as independent evidence.

In ON arms, after each fifth complete pass **following the anchor**, multiply W, b, theta_plus and theta_minus by `1-rho`; rho is 0.01 or 0.03 selected on development. OFF arms receive the same boundary event/counter advancement, without weight changes. No online taught-score trigger changes the dose or rest onset. If acquisition at the anchor is below 95%, label it shrinkage during acquisition, **not post-fit grokking**.

Uniform positive scaling of W/b preserves ReLU gates and, for the declared homogeneous expansion, top-k membership. Scaling opponent arrays preserves immediate margin signs. Qualify this on no-training rest fixtures; record gates, KC IDs, margins, ties, clipping and output behavior before/after. If any other normalization, bias, eligibility state or threshold breaks that prediction, explain it from the implementation before interpreting results.

Consequently, lower norms are not proof of simpler representations. Shrinkage may change subsequent residuals, learning dynamics or margin magnitude. Report those effects; only the ON/OFF intervention can support a shrinkage-benefit claim. Do not equate scheduled decay with a demonstrated sleep/homeostasis mechanism.

## 8. Roster: map and shrinkage form a factorial comparison

| Arm | Map experience/readout | Shrink after anchor |
| --- | --- | --- |
| FULL | Q+C learned map | ON |
| MAP-NO-SHRINK | Same Q+C map | OFF |
| SYMBOL-SHRINK | Map output zero; identity unchanged | ON |
| SYMBOL-NO-SHRINK | Same identity-only route | OFF |
| MAP-PERM | Consistently relabeled Q+C | ON |
| MAP-NOEXP | Same Q+C exposure, map writes disabled | ON |
| FROZEN-LAYER | Learned Q+C map, W/b frozen | ON for naming arrays only |
| SHUF-ADD | Learned Q+C; scrambled taught answer content | ON |
| BIRTH-MORE-ADD-SHRINK | No Q/C exposure, identity-only; extra legal addition | ON |
| BIRTH-MORE-ADD-NO-SHRINK | Same birth/extra-addition control | OFF |

Thus the prospective main roster is **10 arms × 4 worlds = 40 histories**, not the original 36. This revision adds the missing identity-only OFF and matched budget controls so a map-by-shrink interaction is interpretable. No new experiment has run and no cap is increased by this roster change.

SHUF-ADD permutes observed answers among taught orbits, consistently across orientations. Shuffle separately within target-length and orientation-multiplicity strata so total target tokens and byte/EOS timing match. Lock the shuffle before fitting; report realized chance agreements and limited available permutations. The shuffled content never reaches a held relation. It is a content-learning control, not guaranteed failure.

For a no-addition-write probe, preserve the input/target stream but disable all downstream writes; report the first held answers before and after exposure. It is a read-only companion, not an extra learning history or an independent replicate. Do not confuse disabling only later shrinkage with disabling prior addition learning.

Every arm gets evaluator-only exact/unordered lookup, frequency, COPY_A/B and nearest-taught-answer scores using its **actual frozen augmented trace**. Lock distances/ties/fallbacks. These controls and arithmetic evaluation cannot enter learner code. FIXED-k truncation may diagnose stopping from saved generated bytes but is not own EOS and never replaces the primary endpoint.

## 9. Readiness before costly map learning

The candidate/reference code remains arithmetic-free. **ORACLE-MAP is explicitly an evaluator-only arithmetic prior**, not a permitted learner or a demonstrated upper bound. Keep its construction in a separate fixture that cannot be loaded by a main candidate. This is the limited diagnostic convention used in V145, not an exception for candidate structure.

Run it through the exact identity+map trace, local combination, canonical expansion and own-byte/EOS naming path. Put bumps at the true digit positions only in the fixture; report how the remaining bytes are handled. Evaluate SYMBOL through the same pipeline. Freeze oracle width and scale in advance; an oracle chosen on held outcomes is not a ceiling.

Finite development starting menu on D1/D2: alpha in {0.003,0.03}; rho in {0.01,0.03}; H/M/k and map schedules above fixed. For each alpha run ORACLE and SYMBOL OFF once, and each ON rho once: **12 unique fits per world, 24 total**, through the declared checkpoints. OFF runs are not duplicated for rho. There is no open-ended decoder/expansion search.

Retain Claude's operational oracle readiness hurdle: at least 90% taught complete accuracy and held complete accuracy >= strongest shortcut + 0.20 in both D1/D2. The 0.20 is a chosen screen filter, not a biological constant or proof of an arithmetic algorithm. A 0.20 difference is three *orbit-equivalents*; report fully solved orbits separately because averaging orientations permits half-orbit scores.

If the finite menu cannot clear readiness, report `REFERENCE_PIPELINE_NOT_READY`; do not run Q/C or call map learning disproven. A negative local oracle does not exclude every pipeline/representation. Do not redesign and repeatedly retest the same worlds as untouched evidence.

If readiness passes, choose alpha/rho by mean final development ORACLE held score, then lower total measured CPU, then lexicographic settings. This is deliberately reference-selected, not optimized separately for the candidate. Use D3 for the finite learned-map/causal-arm development comparison, with no further rate/topology tuning. Save map state after Q and C. D4 rehearses the selected **complete ten-arm protocol**, not a shortened fixture advertised as a full rehearsal. Any scientific change after inspecting D3/D4 is dated as a new revision before main.

## 10. Main endpoint, mechanism contrasts and claims

Names remain `V146-M1`–`M4`; generate them only after source/settings/baselines/roster lock. Development identities are D1–D3, rehearsal D4. The finite table has 55 unordered single-digit orbits; split 40 taught / 15 held with the inherited coverage and one-/two-byte constraints, keeping both orientations together. Main worlds vary split and initialization, not arithmetic domain. No Q/C/addition feedback contains a held addition relation. First scored responses precede any held feedback, and held feedback is never supplied.

Primary: final **complete own-payload plus own EOS** accuracy, orientations averaged within orbit then across 15 orbits. Also report fully correct orbits, payload-only on the same denominator, correct-length EOS, invalid/truncated outputs, taught accuracy and one-/two-byte strata. Class labels or teacher-forced loss cannot replace it.

Report paired map effects at both shrink settings, shrink effects with and without the map, and the interaction:

`(FULL - MAP-NO-SHRINK) - (SYMBOL-SHRINK - SYMBOL-NO-SHRINK)`.

Keep Claude's 0.20 conservative readiness filter for FULL over SYMBOL-SHRINK, MAP-PERM, MAP-NOEXP, SHUF-ADD, no-addition-write and the strongest shortcut in each main world, plus taught >=90% and at least one **fully solved two-byte held orbit**. All four passing is a promising *map-assisted reference* screen; three is mixed; otherwise no screen-positive result. It is not population confirmation, statistical significance or native adoption. Four positive signs alone give 1/16 under a simple sign null; the more complex gate is not itself that test.

A delayed-generalization diagnostic is fixed at the anchor/final checkpoints: if taught accuracy at pass 256 is >=95%, report the held change to pass 1,024, including Claude's two-orbit-equivalent (2/15) indicator. **Claim a shrink-associated delayed benefit only if ON improves more than its OFF counterpart**, with positive final ON-minus-OFF held difference; show all worlds and uncertainty descriptively. The same rule applies to SYMBOL. An early pass selected after looking at held curves is forbidden. Failure to see that signature does not establish shrinkage is unnecessary; the direct factorial contrasts decide what is supported.

A successful FROZEN-LAYER means trained combination features are not needed at this recipe, not that all feature learning is unnecessary. Equal MAP-PERM performance means numeric grounding was not isolated. Good map meters plus poor behavior do not prove that the downstream pipeline alone is at fault; meters are incomplete. Even a successful reference screen does not establish a human-like counting procedure, carry algorithm, persistence or a native MiniFly gain.

## 11. Total samples, cost and execution integrity

Per-arm ledgers distinguish Q/C raw payload, EOS and update opportunities; addition records and revealed target tokens; repeated passes; probes; failed attempts; initialization/import CPU; summed child CPU, wall time, peak RSS, model/cache bytes and audit overhead. Q/C experience counts toward efficiency. Do not call 704 pretraining records equivalent to 704 addition records; compare total observed tokens and measured CPU as well as downstream-only curves.

BIRTH-MORE-ADD uses only the legal taught pool, through at most 2,048 passes. Compare its latest whole-pass state whose total observed tokens and CPU are both <= FULL's budget, using cost counters only, never held accuracy to choose the point. It skips Q/C rather than merely disabling its writes. If its cap or absent snapshots prevent a valid budget match, withhold the corresponding total-efficiency claim. Pretraining is not amortized across hypothetical future tasks.

Proposed overall ceiling remains **six summed process CPU-hours** including preparation, development, failures, rehearsal, main and reduction, with at most two one-thread workers. It is a cap, not an estimate of runtime. Before execution, enumerate every fit/continuation/probe, exact Q/C/addition exposure maxima, shared prefixes and remaining CPU allowance in a machine manifest. Cost the inherited dense projection, sorting and active-set selection; sparse KCs alone are not computational sparsity. Admit whole work items conservatively. If the roster cannot finish, return `INCOMPLETE_V146` with missing cells; do not publish a completed best-so-far verdict.

Shared immutable Q/C states and common pre-shrink prefixes may be computed once and forked. Keep per-arm logical exposure separate from physical executed presentations/CPU; forks are correlated histories, not independent additional evidence. Never omit earlier learning from the inherited branch budget or sum the same physical prefix repeatedly in campaign cost.

An outer supervisor starts the append-and-flush attempt/cost ledger **before imports/training** and records exceptional exits. Persist immutable Q/C map tables and all source/asset hashes before addition. Closure includes imported and late-loaded source, graph/birth assets, fixed arrays and configuration. Qualified restoration must fail closed on missing dependencies rather than substitute old data.

Checkpoints include the full map, association, both trace definitions, W/b, naming arrays, fixed matrices, RNG, phase/pass/record cursor, warmup/rest counters and accumulated cost. Resume starts at the **next uncommitted unit**, never replays a committed training record. A disposable uninterrupted-vs-resumed fixture may process the same suffix twice solely to test equality; charge both paths and do not count them as independent science. Validate full combined objects, as V145's small-code pickle check missed its real wrapper failure.

Reducer requirements: exact roster and checkpoint identity; immutable receipt/source hashes; finite counts/metrics; exact token/record totals; generation scored from saved bytes/status; missing/duplicate/extra/partial rejection; no held write or probe mutation; retained records for failed attempts. A validity audit is not independent scientific confirmation.

## 12. Next authorized boundary

This delivery refines and saves the design. It does **not** implement the map, run the oracle, launch training, change an adopted core or push GitHub. If the owner requests the three-cycle exercise, first test local equations and generic byte/trace invariants; then the bounded actual-interface readiness menu; then, only if ready, the learned-map test and full qualified rehearsal. A negative readiness result is useful and should stop that path without another mechanism stack.

Future native integration and retention need a separate design and explicit run instruction. Four-operation arithmetic, noise optimization and new language tasks are not added. The fixed objective remains complete unseen addition learned from feedback, at lower total sample/compute cost.
