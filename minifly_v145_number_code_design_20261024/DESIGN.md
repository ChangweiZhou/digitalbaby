# V145 / NUMBER-CODE V1-R1 — does generic sequence pretraining help unseen addition?

24 October 2026. **Design only. No implementation, new training, three-cycle exercise, main run or adoption is claimed by this delivery.** V145 identifies this prospective design; V144 remains the latest completed exercise. This refines [Claude's NUMBER-CODE V1](CLAUDE_NUMBER_CODE_V1.md), pinned at `0f41f569b15827bc6259301563bf6b11c9600a9b`. It leaves that memo, V143/V144 results and the unexecuted V144-R2 capacity-screen contract unchanged. [REVIEW.md](REVIEW.md) explains the revisions.

The goal remains **better complete, self-generated unseen-addition answers per total training experience and actual compute**, ultimately through MiniFly. Counting is proposed additional experience, not a number encoder installed in the learner. This pilot asks whether that experience improves a generic receptor code; it does not supply addition, successor or carry operations to the answering model.

## 1. What the evidence does and does not say

Claude's analysis is a useful representation hypothesis: some input geometries make the tested addition classifier generalize much better than others. It does not establish a single cause of all previous failures.

- The FE-PERM trace has centered rank 19 in the tested finite prompt set. That does not prove an exact row-plus-column representation: the same analysis reports a relative affine-fit residual of **0.22599**.
- Its line/circle comparisons concatenate explicit operand vectors into an MLP. They do not test learned receptor substitution through the actual FE-PERM event API. They also use class outputs and different optimization from V144's own-byte/EOS reference.
- Low performance after the tested practice doses is a finite negative result, not proof that symbolic input makes addition impossible. Native storage, credit assignment, optimization and stopping have not all been eliminated as bottlenecks.
- A shared affine successor can fit `z(a)=(a,a²)` exactly with `T=[[1,0],[2,1]]`, `c=(1,1)`. Good transition fitting therefore does not identify a number line or show that addition is simple in the learned code. Constant codes can also have zero transition error.

V144's 64-pass reference fixture obtained taught complete accuracy 77.5%/81.25%/15% and held complete accuracy 6.67%/3.33%/6.67% for BP/DFA/FROZEN. It is neither the full capacity screen nor an adequate negative theorem. V143's positive parity result is a different, partial-output task.

## 2. Claim, lineage and exposure boundary

| Route | What it is | What success could establish |
| --- | --- | --- |
| Ordered-stream code + DFA downstream | Proposed generic fixed-feedback reference, inspired by V143 | A permitted local-credit candidate can reuse pretraining for this bounded byte-output task |
| Ordered-stream code + BP downstream | Evaluator-side exact-gradient reference | Representation readiness under that reference; not local-rule success |
| Exact-gradient code learner | Same pretraining forward model/loss, different credit | A comparison of tested credit rules; not native efficacy |
| V143 native banks and summed output | Historical parent; not executed by this pilot | No new claim from reference results |

The pretraining predictor, shallow addition reference and native bank are distinct executable routes. A fixed-feedback update is not automatically the native V143 update. The code table is mutable learned state, not a relation-indexed answer cache. No native source is replaced or silently adopted.

Evidence level: byte-interface checks are E0; taught addition recall is E1; complete answers on never-reinforced operand orbits can provide bounded **E3 reference evidence** if the prior-experience controls support it. This design has no intervening-learning phase and cannot establish persistent-core success or OLD retention. Retention and native integration require a later, separate design; they remain part of the ultimate goal.

All 55 unordered single-digit operand orbits are split at the relation level: 40 taught, 15 held, with both orientations together. The evaluator alone renders `a+b=` and `str(a+b)`. It uses the inherited split constraints, including held one-/two-byte coverage and digit coverage in the taught pool. Held probes receive no feedback, never enter code learning, and never update weights, RNG or normalization state. All legal payload bytes remain accepted; the task is small, the interface is generic.

Counting exposes the ten digit bytes and their observed order; it does **not** expose addition pairs, sums, arithmetic losses, operand tags or a target length. Repetition and knowledge in this curriculum are additional information and must be charged. This is not a claim of discovery from 40 addition facts alone.

## 3. One generic learned-receptor path

Replace digit-only substitution with the same mechanism for **every payload byte**:

`r(x) = s U z_x`, for every `x` in the 256-byte vocabulary.

Here `z_x` is an independently initialized learned d-vector; U is one fixed seeded 88-by-d projection. Token IDs are table addresses only: no numeric sorting, ASCII-distance feature, digit membership test, parsed position or operand slot is available to the learner. EOS is the separate inherited control 256, never a payload terminator byte. The predictor code table has **257 rows**, including z_256 initialized by the same rule. The EOS row supplies the detached target on a final observed edge; it stays fixed because EOS is never a source token in these reset records. Only the 256 payload rows are projected into receptor patterns. Include the EOS row in storage, source/state locks and checkpointing; do not feed it back as a payload byte.

The same replacement covers prompt bytes, observed-answer prefix bytes and generated prefix bytes. It uses the parent's actual FE-PERM factory/timing; a literal-decay approximation is not admitted. Non-digit bytes have no alternative route. Byte/EOS boundaries are generic API metadata. The experimenter chooses which bytes occur in the stream, just as in any data curriculum.

Initialize all code rows from `N(0,I/d)` and use seeded orthonormal columns for U. Set the scalar s once from the RMS of **all 256 birth rows**, matching the historical receptor RMS; keep it fixed thereafter. Never normalize using held prompts, expected outputs or a digit subset. For a given d, all causal arms start from identical z/U and use identical downstream initialization. Original SYMBOL uses a different historical receptor and is a descriptive reference, not the matched no-pretraining ablation.

## 4. Explicit proposed predictive update

This is one implementable hypothesis, not an established human/fly mechanism. Three-cycle qualification may repair implementation errors; any change to its learning equations must be dated and tested before a main lock.

For one actually observed adjacent token `x -> y`, use the **pre-update** state:

```
q = T z_x + c
u = (q - stop_gradient(z_y)) / d
l = D q + b                         # nine logits for byte/EOS bits
e = (sigmoid(l) - bit01(y)) / 9
L = mean_bit_BCE(l,y) + ||q-stop_gradient(z_y)||² / (2d)
```

D has shape 9-by-d. Targets are the existing nine-bit byte/EOS serialization, not arithmetic classes. No prediction or generated guess is a training target. Stop-gradient means the target row is not changed on this edge; it can change later when it is an observed source token.

For **LOCAL**, B is a fixed seeded d-by-9 feedback matrix with independent entries `N(0,1/9)` and C a fixed seeded orthogonal d-by-d matrix. Sample C and the orthonormal columns of U by Gaussian QR with positive-diagonal-R sign convention. Initialization substreams are fixed by world/d and shared across ORDER/PERM/BAG; their scale is not an extra tuned parameter:

```
v = B e + u
D <- D - eta e qᵀ;     b <- b - eta e
T <- T - eta v z_xᵀ;   c <- c - eta v
z_x <- z_x - eta C v
```

For **BP-CODE**, replace `v` by `Dᵀe+u` and `C v` by `Tᵀv`, using old D/T throughout. This is the exact one-step gradient with a detached target, not a gradient through the whole counting history. LOCAL's B/C feedback is an approximation; no alignment or biological-locality success is assumed.

Initialize T to identity, c/b to zero and D from `N(0,I/d)`. Clip each proposed parameter-block update's Euclidean/Frobenius norm at 1, then limit any updated code row's norm to 4. These are uniform numerical bounds, not an anti-collapse guarantee or learned number line. Report every clip. Reject nonfinite state; do not substitute fresh codes or retry a failed main fit.

Warmup is gradual: `eta = eta_max * min(1, prior_observed_transition_count/1024)`. All target counts, including the zero-rate initial edge, are retained. No unlogged pretraining jumps to full-rate learning. Each edge update is atomic; the x==y case uses a single old row and the same detached-target convention. There are no across-record transitions, replay, latent arithmetic labels, noise curriculum or scalar-only phase in this pilot.

## 5. Data and matched pretraining arms

Use repeated full records `0123456789`, followed by observed EOS. Every payload byte occurs once in every record. There is no 9-to-0 edge; the start of a new record is an API reset, not evidence of a cyclic number system. Full runs remove the original random-start/endpoint frequency confound. Stream generation is evaluator-side; the learner contains no counting generator.

| Code condition | Experience and plasticity | Question |
| --- | --- | --- |
| ORDER-LOCAL | Ordered full records; LOCAL update | Candidate |
| PERM-LOCAL | One locked nontrivial fixed digit permutation per world, repeated full records; LOCAL | Does the particular observed order matter? |
| BAG-LOCAL | Independently shuffled full records; LOCAL | Does consistent order help beyond identical marginal bytes/boundaries/length? |
| NO-WRITE | Identical ordered records/timing; all pretraining updates disabled | Is learned pretraining state useful? |
| ORDER-BP | Identical ordered records; BP-CODE update | Does this alternative credit rule help? |

PERM and BAG have exactly the same ten payload counts, record lengths, EOS targets and observed-target count as ORDER. A fixed permutation is generated from the world seed and rejected only if it is natural or reverse order; it is not selected using addition outcomes. Other permutations may still help addition. Equal gains mean numeric order was not isolated, not that all representation learning is disproven.

Every record presents ten payload bytes plus EOS and supplies ten next-token targets (nine within-record edges and the final EOS edge). The first payload byte is observed, not predicted from a hidden digit-specific start. NO-WRITE is information/timing matched but intentionally computes fewer learning operations; both actual costs and exposed targets are reported. No claim of exact CPU matching is made.

The ordered pretraining model is frozen **before** addition. All code conditions then receive identical addition examples and answer/EOS feedback in the same order. Adding pretraining targets may help at fixed addition exposure while worsening total efficiency; report both findings if so.

## 6. Addition readout and controls

Use the V144 task-independent shallow reference: FE-PERM trace -> ReLU layer -> nine-bit output, with BP, DFA or feature-frozen hidden weights. Train on observed answer prefix bytes and EOS; test on the model's own prefix. The inherited record-based 1,024-record learning-rate ramp is retained. No expected prefix, known output length, legal-digit mask or 19-sum-class head enters scored generation. Invalid codes, cap exhaustion and wrong EOS remain failures. Cap is eight output decisions, unchanged.

For every code condition run BP and DFA downstream. Add FROZEN-DOWN for ORDER-LOCAL and NO-WRITE only, to test whether learned downstream features are necessary. These are **12 core fits per main world**. Add the two BIRTH-MORE-ADD budget controls specified in section 9: **14 learning fits per world, 56 across four main worlds**, plus the declared no-addition-write probes. There is no full new mechanism sweep. The DFA path remains a reference approximation, not a native-bank port.

Each world also has a no-addition-write control: expose the same addition inputs/observed answers with downstream learning disabled. It cannot count as a trained core, but it tests whether the claimed gain depends on addition teaching rather than the pretrained code alone. Generate its predictions once per code condition; never train on a held answer.

Score evaluator-only recitation countermodels on the same held orbits: exact-byte and unordered-pair lookup, train-only answer frequency, literal COPY_A/COPY_B, and nearest taught answer under the **actual frozen receptor/FE-PERM code**. Lock distance, tie break and fallback on development data. Their parsing or expected-answer calculation never enters the learner. Comparing only against a lookup guaranteed to miss held keys is insufficient.

## 7. Development, interface check and selection

**Do not run a large grid before checking the actual proposed path.** First use development worlds to send diagnostic line/circle receptor rows through U and the unchanged FE-PERM API, followed by the same own-byte/EOS readout. Such oracle rows are evaluator-only arithmetic priors and must live outside candidate source. Claude's concatenated-operand result is not this check. Compare against SYMBOL and initialized generic codes with a common scale convention. A negative oracle result limits this interface/recipe; it does not prove that all learned codes fail.

Proposed finite development menu:

- Three new development worlds, namespaced `V145-NCODE-D1` to `D3`; do not reuse V143/V144 world identities or their outputs as untouched evidence.
- Code dimension d in {2,4,8}; LOCAL peak rate in {0.001,0.01}; prediction checkpoints after 128/512/2,048 counting records. These are six continued histories per development world, not 18 fresh starts. The 128/512 code checkpoints are predictive diagnostics only; **only the 2,048-record code snapshot** feeds downstream selection. BP-CODE uses the same selected settings.
- Downstream recipes are the V144 H128/rate0.03/decay0.001 fixture and H256/rate0.01/decay0.001. Addition checkpoints after 16/64/256/1,024 passes. For the six code settings and two recipes this is 12 selection addition fits per development world, 36 over D1–D3; intermediate code checkpoints do not multiply that grid. Other oracle/control/development/rehearsal cells remain separately enumerated in the required manifest. Continue a fit; never restart it at a later checkpoint and call it the same history.
- Select **one common configuration** using ORDER-LOCAL with BP downstream, mean final development complete held-orbit accuracy; ties: lower measured total CPU, smaller d/H, then lexicographic settings. This favors reference readiness deliberately; DFA is not claimed to have received its own optimal search. Code meters, main outcomes and the strongest arm-specific results cannot alter this selection rule.
- Fix the counting dose at 2,048 records and the final addition dose at 1,024 passes for the prospective main roster. Earlier checkpoints show efficiency; they cannot become the primary endpoint after looking at main results.
- D4 is a single technical dress rehearsal of the selected full roster. It is not a fourth tuning world. A scientifically meaningful change after D4 requires a new revision and fresh rehearsal identity.

The menu is a **proposed maximum**, not an authorization or a runtime guarantee. Before training, a machine manifest must enumerate the exact interface-check/tuning/control/rehearsal fits, not merely main fits. Its worst-case exposure must be calculated and admitted against section 10. If it cannot fit, narrow and date the menu before results; do not drop unfavorable cells during execution. Interface check failure stops the pretraining pilot with a scoped result rather than prompting an unbounded decoder search.

V144-R2's capacity screen remains a separate unexecuted proposal. This pilot neither silently runs it nor pretends to replace its result. There is no need to complete both before this small representation experiment, provided the owner selects this route and the interface check is valid.

## 8. Untouched main assessment and decisions

Use four new complete identities `V145-NCODE-M1` through `M4`, including separate seeds for split, birth, order and permutation. Main code/settings/control definitions are locked before any main split or output is inspected. Four worlds sample different initializations/splits of the **same finite 55-orbit table**; they are not four independent arithmetic domains. Both orientations stay in the same split and count as one relation.

Primary behavioral endpoint: **final complete held own-byte/own-EOS accuracy**, averaged over orientations within orbit, then over the 15 orbits of each world. Publish ORDER-LOCAL with DFA and BP separately; the former answers the local-route question. Never substitute a class-prediction or parity score.

For each downstream mode report paired ORDER-LOCAL differences against NO-WRITE, BAG, PERM and the strongest shortcut. Show each world and the pooled descriptive mean, all earlier checkpoints, taught accuracy, one-/two-byte strata, EOS at the correct length, invalid codes and truncation. Payload-only and complete scores must use the same orbit weights.

A **promising local reference signal** requires final ORDER-LOCAL+DFA to beat NO-WRITE, BAG, PERM, the corresponding no-addition-write control and the strongest shortcut in all four worlds, with at least one complete held two-byte-answer relation in each. Apply the same rule to BP separately and label it reference-only. Otherwise report mixed/null results without selecting favorable worlds. This is a directional screening decision, **not a significance threshold, confirmation or adoption**; four positive world signs alone would have one-sided sign-test probability 1/16, not <0.05. No fixed 0.20 effect is presented as biologically meaningful.

No geometry threshold, mandatory parity score or expected SHUF failure is a gate. Transition error, multi-step prediction, code variance/rank, minimum pair distance, clipping and byte-prediction accuracy are diagnostics. Report zero denominators as undefined, not as perfect structure. Measure multi-step propagation without inserting the actual intermediate token/code; otherwise it is a replayed edge-fit check. Neither a good distance correlation nor zero affine error is called arithmetic homomorphism.

| Observed outcome | Next interpretation/action |
| --- | --- |
| ORDER helps held full answers over matched controls at the same addition exposure | Pretraining is useful at that downstream dose; assess its total cost before an efficiency claim |
| BP helps, DFA does not | Tested reference/local credit gap; not proof local learning is impossible |
| ORDER-BP helps but ORDER-LOCAL does not | Code-learning rule is a candidate bottleneck at matched settings |
| BAG/PERM helps equally | General exposure/regularization may help; numeric successor order not isolated |
| Transition prediction succeeds but addition does not | This predictive representation is insufficient with this readout; no guaranteed line/addition geometry |
| Taught answers fail or oracle-through-FE fails | Diagnose acquisition/interface/optimization; do not attribute everything to missing number concepts |
| Complete behavior improves only after much extra pretraining | Capability/extra-data result; no fewer-total-samples claim |

The next native experiment, if justified, must separately test the V143-derived operative banks, the combined native output, own EOS, no-prior-teaching and content-shuffled controls, and retention after actual later learning. This memo does not authorize that port. No “apply +1 b times” routine, digit-column decomposition or carry state is supplied in this design or adopted as a roadmap implementation.

## 9. Efficiency accounting

Report separate per-fit and all-attempt ledgers for: counting records; raw payload/control tokens presented; next-token targets; addition records; revealed answer bytes/EOS; repeated passes; probes; process CPU; wall time; peak RSS; model/trace/cache memory and audit overhead. Synthetic fixtures and failed attempts appear separately and in total cost. Unknown historical-analysis CPU remains unknown; it is not silently treated as zero or charged as new main execution.

A counting record and an addition record are not equal information units. Show record counts **and token/feedback counts**, and learning curves against total exposed tokens and total measured CPU. Also show addition-only curves to isolate pretraining's effect. For total-efficiency comparison add **BIRTH-MORE-ADD**: the same initialized generic receptor, no counting exposure or code updates, and BP/DFA addition training from birth through at most 2,048 passes on the same taught pool. This is distinct from NO-WRITE, which actually observes counting records. At each ORDER final budget compare the latest whole-pass BIRTH-MORE-ADD state whose cumulative exposed-token count and measured CPU are both no greater; select that point by cost counters alone, never held scores. Predeclare this rule and save the required state/probe before opening its held result. If the baseline hits its 2,048-pass cap before matching the available budget, report that limitation; do not assert an optimal matched-budget comparator. The 56-fit main roster includes these eight additional fits. Main budget ceilings include their training and probes. If any cost-matched comparison is absent, withhold the corresponding total-efficiency claim.

Do not amortize pretraining across hypothetical future tasks. Frozen code reuse may be costed per-task and one-time separately; both must be visible. Full256 code storage, dense projections, shared transition/decoder and V144 ReLU costs are bounded but not scalable sparse native computation. No raw-history-sized cache is part of the learner.

## 10. Execution ceiling, qualification and delivery

Propose **two summed process CPU-hours total for all new preparation, tuning, rehearsal, main fits and reductions**, at most two workers with one numerical thread each. This is a ceiling, not the original “seconds to minutes” prediction. Exposures are bounded by the locked finite roster; compute the exact upper bound before implementation can be called ready. Aggregate child CPU and carry prior attempts forward. If setup cost cannot be measured, disclose it and do not claim a complete all-in figure.

Before a run, qualify the finite dispatcher, whole-record checkpoint/resume, source/asset restoration, output protocol, typed validator and exact-roster reducer. A checkpoint contains codes, projection/normalization, all weights, RNG states, stream/pass/record cursor, warmup counters and accumulated cost. Interrupted prefixes are neither new completed records nor discarded cost. No automatic scientific retries or silent restarts.

Three cycles, **if subsequently requested**, should cover: (1) explicit-equation/finite-difference checks, arbitrary-byte path and same-interface oracle; (2) bounded development comparison with exposure-matched streams and cost measurement; (3) frozen-protocol dress rehearsal, uninterrupted/resumed equality and invalid-artifact rejection. Technical success does not mean addition success. Test the combined module as implemented, not disconnected synthetic equations alone.

On budget exhaustion, unavailable assets, incomplete roster or unqualified restoration return `INCOMPLETE_NUMBER_CODE_PILOT` and name the gap. Source mismatch, leakage, illegal target information or generation that mutates learning state invalidates the affected claim. Negative behavior under a valid complete run is a result, not an execution failure to restart.

Deliver source/asset lock, explicit manifest, compact per-world/checkpoint metrics, per-attempt cost ledger, validity status and REPORT. Raw logs/checkpoints stay local unless separately requested. **This document starts no experiment and makes no GitHub publication or core-adoption claim.**
