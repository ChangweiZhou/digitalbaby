# V144 — Addition readiness and learning-signal diagnosis

24 October 2026. **Design memo only: not implemented, qualified or run.**

V143 is the consolidated serial alias for Claude's LEARNED-BANK V1. V144 is the next experiment package under the numbering agreed with the user; it is not a claim that 144 distinct brain architectures were built. Its purpose is to decide the next native addition experiment from evidence. This delivery does not adopt a new core, alter V143, or authorize an experiment launch.

## 1. Fixed goal and the immediate question

The goal remains **better retained generalization to never-taught addition problems, measured by complete own-byte answers and own EOS, per all training presentations and actual compute**. Expected output bytes are allowed as real feedback during gradual warmup. No arithmetic parser, numeral/operand features, carry circuit, supplied laws, intermediate solutions, attention, Transformer or ELM is added. Other operations, NLP, external-memory applications and noise robustness are outside this experiment.

The next question is narrower than a new mechanism campaign:

> What limits the current learned-address route: the learning signal, the attainable mapping from its byte state, or complete answer generation?

V144 has three sequential stages: integrity preparation, a paired native learning-signal diagnostic, and a bounded evaluator-side addition capacity screen. A later native efficacy comparison is specified as the next decision, not silently included in this execution budget. More mechanisms are not added simply because a catalog names them.

## 2. What V143 established, and what remains a hypothesis

The saved four-world parity results give mean balanced held accuracy of 91.56% for the learned bank, 46.72% for frozen features and 56.30% for shuffled labels. The actual combined system gives 69.49%. The locked verdict remains partial support, 3/4 worlds. LEARN used the same exposure and about 0.52% more total measured CPU than FROZEN on parity. This is a useful fixed-budget component improvement on parity, not ordinary-addition or complete-generation success.

V143 used a single externally delimited answer byte, no own EOS, and rest without intervening learning. Its ordinary last-digit addition development runs failed. The broader project is not being redefined around parity.

Three explanations are open:

| Hypothesis | Current evidence | What would discriminate it |
|---|---|---|
| The native memoriser weakens useful upstream teaching | Continuous informative-bit residual remained approximately 0.22 at pass 20 in a native diagnostic. Correct answer signs do not imply zero residual. | Residual, actual feature-update magnitude and direction, native-margin growth, and held progress measured together. A later causal change in relative learning speed would still be needed. |
| Common answer bits cause harmful representation collapse | Mean KC overlap increased. Centring residuals reduced overlap without improving held accuracy in the tested surrogate. | Separate within-class grouping from between-class overlap, effective feature variation and actual loss of held performance. Overlap alone is not a failure gate. |
| The present representation/training recipe cannot efficiently extract addition | Some small BP/DFA settings failed mod-10; parity worked. | A finite broader screen on the exact byte state, complete answers and own EOS, with data-dose and exposure/CPU curves. Failure remains scoped to the screen. |

Neither a failed BP search nor success of this learned-feature intervention proves a family-wide theorem about fixed kernels, local learning or arithmetic feasibility. Jaccard alignment is an address-geometry meter, not the complete native response kernel or a mediation proof.

## 3. Executable lineage and allowed changes

V143 source authority is Claude commit `4c12621d0804d2495c2eacaebf4e17aaeb994771`, with its original lock at `1ce1eadb`. Relevant files are `minifly_learned_bank_v1_20261024/{lb1.py,data.py,run.py,LOCK.json}` and imported `minifly_cls_dev_20261017/fastmem.py` plus the Full151 native dependencies and canonical birth assets.

The native diagnostic retains V143's FE-PERM byte trace, 128-unit learned feature layer, fixed projection and feedback matrices, canonical PN→KC connectivity, native fast/slow/adaptation equations, 16 opponent compartments per bank, shared-bank homeostasis, adaptation reset, independent bank teaching and summed system readout. It does not insert EXPAND, COV-SUPPORT or a recurrent replacement. Source inheritance is not evidence that earlier addition capabilities are integrated here.

The only native diagnostic change is instrumentation in a new V144 runner. Historical source, locks, outputs and verdicts remain immutable. The evaluator reference in Stage C is explicitly a separate feasibility model, not an adopted MiniFly and not a native result.

## 4. Stage A — integrity and accounting before any new training

Implement these requirements in the new runner/checker before qualification; this memo does not claim they are already fixed:

1. Source-lock every imported repository dependency and required birth/graph asset, as well as the generator, settings, roster and reducer, before execution. Record runtime versions and numerical-thread settings. An unavailable asset stops admission.
2. Missing/extra histories, duplicate IDs, source or asset mismatch, runtime-closure failure, invalid settings, missing checkpoints or missing required metrics invalidate the verdict. All such failures must reach the same validity decision and a nonzero checker exit. Do not merely print separate false flags beside a valid scientific verdict.
3. Mutation fixtures must demonstrate rejection for each failure above, including a missing control and a one-world substitute for the full required roster. Fixtures are synthetic and excluded from learning evidence.
4. Save actual KC-code hashes and the arrays needed to recompute their meters at declared checkpoints. Rounded Jaccards do not certify byte identity. Hash persistent state before and after probes; separately test every array/counter that may change, including homeostasis frequency and pending event state.
5. Prove instrumentation parity against the unmodified V143 implementation on a short disposable development stream: equal outputs and learning state after every event, with telemetry enabled and disabled. A failed parity check blocks the native diagnostic.
6. Maintain an append-only attempt ledger from process start. Count interrupted records, incomplete byte sequences, restarts, failed fixtures, deterministic rechecks, all tuning fits and probes. Reused stored observations are replay, not new external examples. Report retained-history costs and total incurred campaign costs separately; unknown costs stay explicitly unknown.

Use one numerical thread per process, at most two concurrent processes, and no automatic restart. Wall time, summed process CPU, training/probing/serialization CPU, peak RSS, target events, visited edges, clone count and bytes copied are separate quantities. Global top-k and full native state copying remain real inherited costs; sparse activity is not a sparse-compute claim.

## 5. Stage B — measure the native learning signal first

Run exactly one paired development comparison after authorization: V143's `D2`, T-PAR, LEARN and FROZEN, the original 150-pass schedule and one-hour simulated final rest. This is an explicitly new development diagnostic, with two new histories, not an extension of M1 or an independent confirmation. V143 M1–M4 are not rerun, retuned or pooled with it.

Keep V143's native strength `min(1, 1/16 + (15/16)n/1024)`, feature rate `0.03 min(1,n/1024)`, weight decay `0.001`, temperature 2 and all other scientific settings unchanged. Here n is completed external record presentations; it does not reset after a probe or restart. Feedback is the observed answer byte for that development task. No new homeostasis, residual centring, replay, selector or hidden loss is introduced.

For every teaching event record per bank/bit the pre-write margin, target provenance, residual and applied signed coefficient; feature-path telemetry includes `G e`, active ReLU gates, update norm before decay, decay norm, bias update, clipping/nonfinite events, and their aggregate contribution by output bit. Compute telemetry from quantities already used in the update; do not perturb state to obtain it.

At birth and passes 1, 5, 10, 20, 25, 50, 75, 100, 125, 150 and after rest save read-only per-prompt outputs, component and system balanced accuracies, held transfer gap, code hashes, within-answer and between-answer Jaccard, hidden covariance spectrum/effective rank and representation norm. The extra probes are charged and verified nonmutating. Residual distributions must distinguish constant payload bits from the varying parity bit.

Report whether update magnitude falls, whether its informative component persists, and whether geometry/held performance changes afterward. These are temporal diagnostics, not causal identification of starvation or collapse. Do not add an intervention merely to make a preferred explanation win. If the diagnostic contradicts either explanation, retire that explanation for this recipe.

Budget: at most 24,000 complete training-record applications across the two histories and 1.5 process CPU-hours, including probes and failed attempts in this stage. These are engineering ceilings, not behavioral thresholds. A limit yields `INCOMPLETE_DIAGNOSTIC`; it does not turn missing evidence into a negative scientific result or automatically authorize another run.

## 6. Stage C — bounded addition capacity screen

### 6.1 Data and withholding

Use ordinary prompts `a+b=` for a,b in 0…9 and canonical expected ASCII strings `str(a+b)`, followed by a separate EOS control 256. Thus outputs include both one- and two-byte answers. Only the evaluator computes sums and constructs splits. The learner receives raw bytes, generic record boundaries and observed output tokens; it receives no digit values, position/operand labels, answer length, carry labels or arithmetic traces.

There are 55 unordered operand-pair orbits. In each world hold 15 entire orbits out of every feedback stage; both orientations remain withheld. The other 40 form a training pool. Generate nested training doses of 20, 30 and 40 orbits without looking at model outcomes, with every digit present in at least two training orbits at the smallest dose and four in the full pool. Held sets must include at least five one-byte-answer and five two-byte-answer orbits so both output lengths can be assessed. Split constraints use evaluator metadata only and are recorded in the manifest.

No tens-digit padding, new rendering, counting stream, multiplication or extra curriculum is added. Unique training relations, oriented prompts, record presentations and revealed answer/EOS tokens are reported separately. A novel rendering of a seen relation would be E2, not another independent relation.

### 6.2 Reference architecture and own generation

Use the same FE-PERM raw-byte receptor state as V143. At each output event an H-unit ReLU layer feeds nine unconstrained output margins. The token is the integer defined by their signs: 0…255 are payload bytes, 256 is EOS, and 257…511 are invalid outputs. The code is generic byte serialization, not decimal or arithmetic coding. There is no answer-class codebook, vocabulary restriction to digits, numeric decoder or target-dependent mask.

Each observed or generated payload byte advances the same generic trace before the next prediction. A generic prompt-end boundary starts answering; the model is not told the answer length. During training use the actual received answer prefix, predict before each observed byte/EOS, then update. At evaluation use only the model's own prefix. Stop only at its own EOS; allow at most eight output decisions, including EOS. Missing EOS, invalid output, wrong payload, extra bytes or cap exhaustion makes complete exact accuracy zero. No whitespace stripping, numeric parsing, forced EOS or beam rescue is permitted.

The prompt and output share one byte interface. In addition the maximum correct payload length is two, but the model gets a fixed generic resource cap rather than that per-example fact. Qualification includes synthetic payloads containing NUL, `0xff`, spaces, `=` and lengths 0, 1, 3 and 6; those fixtures are separate from addition training. Arithmetic generality beyond the bounded lengths is not claimed.

Modes have identical initialization, stream, interface and nine-bit loss:

| Mode | Learned state and purpose |
|---|---|
| BP reference | Hidden layer and output weights trained by exact gradients. Evaluator-side feasibility reference only. |
| DFA reference | Hidden layer receives a fixed random nine-channel error projection; output weights use their direct gradient. Tests a local-credit reference, not the Full151 bank. |
| Feature-frozen lesion | Hidden layer held at birth; the same output weights still learn. Matched intervention on the reference, not a proposed ELM kernel or recycled architecture. |

For normalized receptor state p, use `h=ReLU(Wp+b)` and `m=Uh`, with no output bias. Initialize W with independent normal entries of SD `1/sqrt(88)`, b and U to zero, and the independent fixed DFA matrix G with SD `1/sqrt(9)` and shape H×9. Use a separate deterministic RNG substream for G so changing the learning mode does not change its other initial arrays or presentation order. All output and feature updates use the same pre-update state.

Use the sum of nine binary cross entropies with logits `2m/T`, T=2. Its derivative with respect to m is `g=(tanh(m/T)-y)/T`, equal to V143's residual at T=2. Update U from `g hᵀ`; the hidden signal is `(Uᵀg)⊙1(Wp+b>0)` in BP, `(Gg)⊙1(Wp+b>0)` in DFA, and zero in the feature-frozen lesion. This differs from the old diagnostic's squared-tanh derivative and must be recorded as a new evaluator reference. There is no derivative through the native store because there is no native store here. No reference output is substituted for a native answer.

Gradual warmup remains explicit: both modes' learning rates multiply `min(1,n/1024)`; the feature-frozen lesion uses the same output-rate ramp. Output and hidden rates otherwise use the same configured learning rate. Apply weight decay once per observed target token, log that frequency and all target counts, and exclude biases from decay. Supervised expected bytes continue throughout this bounded screen. Scalar-only or blurry-feedback learning is not assessed, and no claim about that later stage follows.

### 6.3 Finite search, data scaling and holdback

Derive seeds from SHA256 of the following fixed names, following V143's little-endian first-four-byte convention. `V144-CAP-D1` and `V144-CAP-D2` are development worlds; `V144-CAP-H1` and `V144-CAP-H2` are untouched holdback worlds. These are not V143's M worlds. Save exact split and seed manifests before training; do not change a world because it looks unfavorable.

The grid is H∈{128,256,512}, learning rate∈{0.003,0.01,0.03}, decay∈{0,0.001,0.01}: 27 configurations. No optimizer search, additional seeds or model family is added.

1. Run all 27 BP configurations on D1/D2 at the full 40-orbit dose for 64 passes. Shortlist three configurations by mean development held **complete own-generation orbit accuracy**, breaking ties by lower measured CPU, then lower H, then lexicographic parameter order. These held labels are development-selection information and are not independent confirmation evidence.
2. Run the three shortlisted configurations with all three modes on D1/D2 through 4,096 passes, recording birth and passes 1, 4, 16, 64, 256, 1,024 and 4,096. Continue the six existing BP histories from pass 64; do not replay them from birth or omit the previous exposure. The twelve other histories start once from birth. Select one common configuration by final mean development BP complete orbit accuracy, with the same tie rule. Report all configurations and modes, including losses.
3. Freeze that configuration and evaluate the three modes on H1/H2 at each nested training dose, through the same fixed checkpoints, without further selection or adjustment. This creates 18 holdback fits. DFA failure at a BP-selected setting is explicitly not an exhaustive test of all local rules.

This means at most 66 development fit starts plus 18 holdback fit starts. The conservative training ceiling is 10,567,680 record presentations and 31,703,040 observed target tokens, derived using at most two orientations per orbit and at most two payload bytes plus EOS. Actual counts will be lower when diagonals/one-byte answers occur. All warmup/repeats are charged; unique examples are not presented as the whole sample budget.

Stage C has a six-process-CPU-hour ceiling across all fits, probes and failed attempts. Use the first short fits to estimate cost without expanding this ceiling. Admit each next work item conservatively; checkpoint at whole-record boundaries. If the roster does not fit, publish `INCOMPLETE_CAPACITY_SCREEN` and the exact remaining cells. Missing cells are not failures or successes, and best-so-far is not called a completed holdback result. No automatic continuation or higher ceiling.

The 64/256/1,024/4,096 checkpoints span early and prolonged practice, including a maximum beyond the old 3,000-epoch diagnostic. These are finite search budgets, not a claim that prolonged repetition is sample efficient. More unique training orbits and more presentations are distinct axes.

## 7. Scoring, evidence level and next decision

Primary screen output is a table/curve of complete own-generation accuracy against **all presentations, revealed target tokens and actual CPU** at every dose and checkpoint. Average the two orientations within each held orbit, then average orbits within each world. Report world-level scores separately; two holdback worlds support a bounded development decision, not a confirmatory population claim. No p-value is computed by treating output tokens or orientations as independent lives.

Also report oriented-prompt exact accuracy, taught exact recall, payload-only accuracy, own-EOS accuracy, truncation/invalid rates, and teacher-forced next-token accuracy. The one-/two-byte strata are descriptive diagnostics. Teacher-forced success cannot substitute for complete generation.

Score explicit countermodels on the same splits: raw-key lookup; unordered-pair lookup; most frequent complete training answer; most frequent complete answer conditional on prompt length; and first-operand-only / second-operand-only predictors fitted from training labels with no arithmetic operation. All abstentions/ties follow fixed manifest rules. An evaluator next-token oracle checks plumbing only and is never a candidate or an efficacy score.

Record birth probes, feature-frozen and DFA controls. If proposing E3 promotion later, include a same-input/time **no-prior-teaching** branch disabling all relevant prior learned updates, plus a structure-destroyed, orbit-consistent whole-answer shuffle and an actual intervening-learning phase. A no-write flag only during later interference is insufficient. V144 itself does not provide that native persistent-E3 confirmation.

No arbitrary absolute accuracy such as 90% determines whether the research goal is solved. The immediate decision uses the following evidence categories:

| Observation | Consequence |
|---|---|
| Integrity/IO qualification fails, or required fits are missing | Fix the localized interface/accounting problem or report incomplete. No claim about arithmetic learnability. |
| Complete BP output is consistently above all declared countermodels in both untouched worlds | A bounded feasibility signal exists for this reference. Show the paired margins and budgets; do not call it a native success or statistical confirmation. |
| BP works but DFA does not, with comparable taught acquisition | Credit assignment becomes a candidate bottleneck; do not explain it as inadequate data alone. |
| Both learn taught examples but not held ones | This finite recipe overfits. The screen has not established impossibility or justified moving the endpoint back to parity. |
| Payload works but own-EOS/continuation fails | Localize the generic generation interface; do not add arithmetic-specific output rules. |
| Frozen features also work | Keep that useful result. Learned features must justify themselves on exposure/CPU curves; a frozen success does not invalidate an addition task. |

The capacity reference has no time-decaying learned state. Waiting one simulated hour would not constitute a retention test for it. Stage B's rest result concerns the parity diagnostic; neither stage establishes addition retention during later learning.

Even with a feasibility signal, the next native comparison needs a separate revised design/authorization. It should compare V143-derived learned and frozen addressing on the same ordinary-addition stream and own-byte/EOS interface; report both banks and the combined system, taught acquisition and retained held performance. It must expose the cost and effect of extending the byte port from eight bits to nine (18 compartments per bank) before a full native run. It must not import the BP decoder, select the best bank using held labels, or silently add replay/arbitration. A final native source lock and fresh-world roster come after that interface rehearsal, not from this memo alone.

## 8. Mechanism catalogs: reviewed inputs, not an automatic roster

Claude's [150-entry human-learning catalog](https://github.com/ChangweiZhou/digitalbaby/blob/4c12621d0804d2495c2eacaebf4e17aaeb994771/human_learning_mechanisms_20261023/CATALOG.md) is a separate literature inventory from the [254 historical MiniFly recipes](../PROJECT_DOCUMENTS/FULL151_VERSION_CENSUS_20261023/MECHANISMS_254.md). They overlap; the counts are not added. The human catalog says 50 entries rely on background knowledge, grades largely use abstracts/snippets, and mapping to MiniFly was by family names. This memo does not upgrade those grades into verified neuroscience or algorithm efficacy.

All 150 original rows and their source/grade fields are retained in [MECHANISM_CROSSWALK.csv](MECHANISM_CROSSWALK.csv), with explicit V144 dispositions. This is a complete inventory triage, not a full-text literature re-review or a fresh execution of 254 recipes. Exact history is linked for the operational families below; unmapped entries are marked as such rather than called new or disproven.

| Catalog entry | Actual project history | V144 disposition / requirement before a new arm |
|---|---|---|
| CELL-09 three-factor rule; PERC-01 reweighting; RL-08 error-driven learning | V143 already learns W1 from projected own-bank residual; older ERROR/COV routes use other substrates. | Retain/measure the actual V143 rule. Do not rename a residual a newly validated biological mechanism. |
| CELL-04/05/14 scaling, excitability and competition; SYS-03 separation | Shared-bank homeostasis exists; centring reduced overlap without improving V143 surrogate held performance. V25/context_stable is a different historical implementation. | Diagnose common vs discriminative components first; no new normalizer in V144. |
| SYS-01 CLS; SYS-15 replay; SLP-02 reactivation; RL-11 replay priority | CLS v0 implemented replay and improved addition retention to 70–85%, without reliable unseen-sum success. Exact prior recipes include V23/replay_reservoir8/32, V24/budget4_replay/budget8_replay, SAMPLE_GEN_V2/REPLAY-P/U and FEEDBACK_EFFICIENCY_V1/COV-REPLAY. | Deferred. A learned-address recipient would differ from the earlier slow recurrent core, but true records, bounded cache, equal-budget online-repetition control and all replay costs remain mandatory. |
| RL-03 arbitration; SYS-22 novelty | CLS v0 already used confidence-normalized arbitration and sometimes scored below its stronger subsystem. V143 sums margins and also loses accuracy. | Deferred. A prospectively error-trained selector would be a changed implementation, not a never-tried family; no held-label choice of bank. |
| CELL-07 tagging; CELL-08 BTSP; CELL-10/11 dendritic mechanisms | The catalog motivates possible generic credit/conjunction changes; no exact V144 operation has been implemented or validated. | Parked until a measured credit/interaction failure identifies an intervention. No hard-coded XOR or arithmetic conjunction cell. |
| CELL-16 cascade; CTRL-04 silent state | V23/fast_slow2/3, V23/cascade4/8, V23/feedback_consolidation and native fast/slow stores already supply related operations. | Preserve attribution; no extra memory or state hierarchy before acquisition/generalization is localized. |
| CTRL-10/11 order/start-stop; LANG-12 staged learning; EDU-07 feedback | Generic order and EOS are existing project requirements; gradual warmup is already retained. | Implement only the specified task-independent interface and schedule. Do not smuggle answer length or arithmetic roles into it. |
| NUM-01…12 | Arithmetic-learning phenomena do not license an innate numerical code, counting routine or carry circuit. | No numerical mechanism added to the architecture or extra teaching curriculum in V144. |

The earlier [CLS report](https://github.com/ChangweiZhou/digitalbaby/blob/4c12621d0804d2495c2eacaebf4e17aaeb994771/minifly_cls_dev_20261017/REPORT.md) is required history for any replay/arbitration proposal. Its retention success does not prove reuse, and its failure does not mathematically exclude every future replay implementation.

## 9. Three-cycle preparation and deliverables

These cycles are proposed, not executed by writing this memo:

1. **Design/audit:** source/runtime locks, fail-closed mutation fixtures, independent token decoder, split/withholding audit, attempt ledger, and equality of instrumented/uninstrumented native state. Synthetic fixtures only.
2. **Test:** run Stage B once and a disposable short Stage C interface fixture; inspect telemetry for actual learning-signal availability, generation using own prefixes, EOS and cost. Any scientific change receives a new revision and its consumed budget stays in the ledger.
3. **Revise/rehearse:** freeze the Stage C finite grid, manifests, byte protocol, reducer and budget; exercise resume/checker failure paths using synthetic fixtures. Run the bounded development/holdback screen only after explicit execution authorization. Do not use H1/H2 for rehearsal or tuning.

Required future outputs are `SOURCE_LOCK.json`, `MANIFEST.json`, `ATTEMPT_LEDGER.jsonl`, per-history compact metrics plus local recomputation artifacts, `SCREEN_TABLE.csv`, a separately source-locked reducer, `REPORT.md` and a final audit. Reports repeat actual E0/E1/E3 evidence boundaries and incomplete cells. Raw traces/checkpoints stay local; publication is design, source, compact results and audit metadata, not raw data.

This delivery contains the design, machine-readable proposal, all-row mechanism crosswalk and the user-provided Claude response. No checker patch, new training, three-cycle result, formal science, background worker or successful addition learner is claimed.

## 10. Sources and planning authority

- [User-supplied Claude response](CLAUDE_FEEDBACK.md), received 24 October 2026; distinguishes accepted corrections from proposed work.
- [V143 report](https://github.com/ChangweiZhou/digitalbaby/blob/4c12621d0804d2495c2eacaebf4e17aaeb994771/minifly_learned_bank_v1_20261024/REPORT.md), [cycle 2](https://github.com/ChangweiZhou/digitalbaby/blob/4c12621d0804d2495c2eacaebf4e17aaeb994771/minifly_learned_bank_v1_20261024/cycles/CYCLE2.md), [post-hoc checker](https://github.com/ChangweiZhou/digitalbaby/blob/4c12621d0804d2495c2eacaebf4e17aaeb994771/minifly_learned_bank_v1_20261024/posthoc_check.py).
- [Complete historical version census](../PROJECT_DOCUMENTS/FULL151_VERSION_CENSUS_20261023/MEMO.md), [254-row recipe review](../PROJECT_DOCUMENTS/FULL151_VERSION_CENSUS_20261023/MECHANISMS_254.md), [V142 bounded NATIVE-NEXT result](../native_next_v1_dev_20261023/REPORT.md).
- Standing [objective contract](../../MINIFLY_OBJECTIVE_CONTRACT.md), [persistent-core requirements](../../PERSISTENT_CORE_REQUIREMENTS.md), [generic byte IO](../PROJECT_DOCUMENTS/GENERIC_BYTE_IO_CONTRACT_20261012.md) and [no-attention requirement](../PROJECT_DOCUMENTS/NO_ATTENTION_KERNEL_REQUIREMENT_20261013.md).

`SPEC.json` records the same design/execution boundary. Updating the central next-design pointer does not replace its historical adoption, execution or result records. V144 is the active proposed design, not the name of an already executed or adopted brain.
