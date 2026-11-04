# MiniFly V200 · Stable query state for complete intermediate answers

**Status** DESIGN_ONLY · **Version** V200 · **Scope** Bounded byte-memory architecture diagnostic in the V199 lineage.

## Question and source evidence

Can a bounded query state held across answer generation improve complete intermediate answers under an unseen prompt wrapper, while preserving trained-wrapper recall? The new comparison changes how the address carries query bytes into generation. It retains local importance-protected writes and the same external two-query controller.

V199's modified Round 8 evaluator generated exact intermediate keys with its own period and completed the chain on 93.8–96.1% of trained-template queries. On the held *remember* template, 35.2–50.8% of generations began with the correct key, but only 0–0.8% completed the strict chain. The held *tell me* template produced zero correct key prefixes. The permissive evaluator removes the final emitted byte even if it is a payload byte and can credit a wrong intermediate that shares the correct final digit. V200 uses strict scoring. It treats the modified evaluator, which adds one- and two-byte suffix features, as the executable historical reference.

[V199 source](https://github.com/ChangweiZhou/digitalbaby/blob/6a96c087a568ad884e4981a4543d93f96cb4b891/minifly_v199_round8_reproduction/hop_eval.py) · [V199 results](https://github.com/ChangweiZhou/digitalbaby/blob/6a96c087a568ad884e4981a4543d93f96cb4b891/minifly_v199_round8_reproduction/RESULTS.json)

V187 already tested changing the output head and missed its progression gates. V200 uses a common 257-way local readout and changes the input state of that readout. The mixed-stream integration remains a separate reference for cued coexistence. V198's matched-address retention comparison and V196's routing question are independent of this test. This is a port of the V199 associative writer, with no claim of Full151 inheritance, arithmetic improvement, or autonomous routing.

[V187 campaign](https://github.com/ChangweiZhou/digitalbaby/blob/dae9822047dd00c06e915819f1a3d8e87c3ceddd/minifly_v187_eos_credit_dev_20261103/campaign/REPORT.md) · [Mixed-stream runner](https://github.com/ChangweiZhou/digitalbaby/blob/6a96c087a568ad884e4981a4543d93f96cb4b891/minifly_mixed_stream_integration/run.py)

## Stage A · Localize failure in the V199 parent

Use one historical world, seed 1 and Python hash seed 0. First reproduce the locked strict, prefix, termination and oracle-hop-two counters from the pinned V199 artifact. Preserve its six-byte cap and generated period. Instrument the same trained instances, with fixed scoring wrappers and read-only probes. Every instrument starts from a fresh transient state with the same weight hash.

| Probe | What runs | What it distinguishes |
|---|---|---|
| Free generation | Model emits every byte including the period | First wrong payload, premature stop, missing stop after correct payload, complete reply |
| Gold-prefix next-byte trace | Evaluator advances the transient state with each correct prior response byte | Earliest payload error under correct history, and whether the next prediction after the full gold key is the period |
| Gold intermediate | Evaluator gives the genuine intermediate key to the fixed second-query controller | Whether the second query can return the final byte plus period given a correct key |
| Training-wrapper replay | Same item, trained framing, same frozen weights | Whether the failure is specific to the wrapper |

Report conditional EOS accuracy both on the subset with an exact free-running payload prefix and on the full gold-prefix panel. Report the first wrong-byte position and free-versus-gold-history accuracy at that position. A gold-prefix success may indicate error propagation. A gold-prefix failure after an otherwise recoverable key points to the boundary prediction. A first-byte error occurs before any generated-byte state can diverge and points to the query code or its learned readout.

Stage A passes source admission when counters reproduce, probes leave learned state unchanged, and at least one held wrapper has both nonzero free key-prefix recall and a gap of at least 10 percentage points between prefix recall and complete hop-one recall. The trained wrapper must exceed 85% strict chaining. The gap is an engineering signal for testing query retention through generation. All Stage A outputs remain historical diagnostics. If the source bridge or this localization condition fails, stop before fresh-world training and report the failure.

## Stage B · Isolate the effect of retaining the query state

A query begins with raw observed bytes and a legitimate end-of-query signal. Every arm receives that same signal during teaching and recall. It identifies when answer generation starts. The learner receives the entire byte prompt, including wrapper and key, through the same bounded byte interface. The signal carries no semantic key, answer, hop label, response length, or stop bit.

Let q be the last 16 prompt bytes at query end. Let r be the generated payload since query end, up to the shared eight-decision cap. At teacher sites, actual observed answer bytes advance r after the prediction and local update. At test sites, only the model's emitted bytes advance r. Terminal EOS is a learned symbol, index 256. Emitting EOS immediately ends the response, even if it is wrong. If the cap is reached without EOS the response fails. Seven payload bytes plus EOS fit the cap.

| Arm | Rows used for the next-byte read and write |
|---|---|
| ROLL | V199-style single rolling window over q followed by r. This is the Stage B baseline with the shared explicit EOS and training schedule. |
| SPLIT-ROLL | Two disjoint row namespaces. Query rows encode the prompt portion remaining inside the rolling 16-byte window, q's last max(0, 16 − length(r)) bytes. Response rows encode r. |
| SPLIT-HOLD | Same namespaces and response encoding. The query rows encode all of q at every answer site. |

In both split arms, rows 0–32,767 are query rows and 32,768–65,535 are response rows. Use the same n-gram recipe as ROLL inside each band. For the empty response use one fixed start-of-response address with its own domain-separated hash. Within each band, deduplicate rows and choose up to 32 query rows and 16 response rows by a frozen deterministic keyed order. ROLL chooses up to 48 rows from its single band using the same ordering method. No learned or target-dependent selection. Freeze the salts, band sizes, rank method, and byte-hash implementation in qualification. The Stage A reference retains the historical Python hash. Stage B uses stable domain-separated byte hashing across arms to make the address definition portable. SPLIT-HOLD may cache the selected query rows until EOS or cap. SPLIT-ROLL can cache all bounded suffix selections or compute the current one on demand. Use a matched cache byte ceiling. There is no persistent cache of complete answers.

For every arm, set s to the sum of the accessed rows' 257 scores. For each selected row i, apply the V199-style update

    W[i] ← W[i] + eta × (onehot(target) − s) / (K × (1 + I[i]/5))
    I[i] ← I[i] + 1/K

K is the number of unique selected rows. Use zero logits and skip the write for K = 0, and count such events. Use float32 weights and float64 importance, with the same accumulation and cast order in all arms. Initial weights and importance are zero. Preserve the inherited exposure-conditioned local protection. The two split arms have identical state and codes at the first response decision for a given query before any training. Their trained first-byte scores may differ because later training writes use different addresses. Report the first-byte effect separately. The comparison matches capacity and the access ceiling. Realized distinct rows, write motion, and CPU may differ and must be reported.

The engine has 65,536 × 257 weights and 65,536 importance entries, **64.75 MiB** of arrays per learner. Each decision reads at most 48 rows, or 49,344 weight bytes, and a teaching decision writes at most the same number. Measure actual sparse accesses, feature/hash work, logits reduction, importance traffic, cache use and CPU. Work follows the bounded prefix and row set. Hashing whole state is restricted to checkpoint audits.

## Stream, exposure and comparator rules

Use eight fresh paired worlds from the first 64 bits of SHA-256 of MINIFLY|V200|WORLD|i for i = 0…7. Freeze separate random streams for byte keys, link assignment, digit assignment, order and filler. Each world contains 128 original chains A → B → digit and 32 subsequent distractor chains of the same form. Assign distinct byte-string keys and disjoint A and B sets across both cohorts. Draw lowercase ASCII keys from the V199 alphabet. Keep A keys five bytes and distribute B keys across three through seven bytes with counts differing by at most one in each cohort. Randomize the link assignment independently of key content and length. Balance final digits within each cohort and randomize their assignment independently of keys. The teacher presents only the two constituent links. The byte stream never presents A → digit. Filler uses shared inter-record prose, with zero associative writes. Start a fresh transient prompt buffer at each query. Filler and delay therefore match exposure and schedule, while the later distractor writes test memory interference.

Each link gets 12 marked presentations, alternating evenly between `the code of KEY is ` and `recall the code for KEY => `. Shuffle link order once per pass. Use step sizes 0.1 and 0.3 for passes 1 and 2 of initial acquisition, then 0.6 from pass 3 through intervening learning. The training signal is the observed target byte or EOS at each marked output site after the model predicts. EOS receives the same update as any payload symbol. No feedback taper or reward-only claim is tested. The separate distractor cohort follows an equal, predeclared 256-record unmarked prose delay and uses 12 passes at 0.6. Probe originals immediately after acquisition, after the delay, and after distractor learning. Probes are read-only and occur before any further feedback.

Hold out `remember KEY: ` and `tell me, what is KEY? ` from every write. Audit actual marked prompt bytes and key-to-target histories, including the evaluation controller's second-query prompts. All arms see the same observations and feedback in the same order. Report trained-versus-held address overlap, key-collision rates, and all output lengths. The evaluator knows the canonical keys to score behavior. The learner receives raw bytes and the legitimate teaching/query boundary only. No supplied intermediate key, answer length, arithmetic rule, attention, Transformer, ELM, learned router, or learned action policy enters any primary arm.

### Controller and evidence level

The fixed controller makes one query for A. It may pass the emitted payload unchanged to a second query only if the first response emitted EOS within the cap. In that case it wraps those bytes in the same selected template. It does not check the gold key, strip payload, pad, retry or force the model to continue. The strict scorer separately requires exact B plus own EOS and exact digit plus own EOS, in that order. Gold-prefix and gold-key interventions run on separate read-only diagnostic paths and cannot enter primary scores. Report correct final digits after wrong intermediates separately.

An exact-prompt table built only from taught prompts abstains on held templates. A canonical-key table built from observed constituent links can solve both hops given the same controller. A global modal-digit baseline is a weak shortcut, and a key-length/byte-position baseline must be fitted only on observed labels. Run these as evaluator controls and report their actual scores. The target is **E2 changed-presentation retrieval** of taught associations, plus engineered chaining of two such recalls. The fixed controller can compose a taught lookup table. The absent direct A → digit teaching therefore does not establish autonomous reasoning or a general persistent-core result. E1 trained recall and delayed retention remain separately reported.

## Decision rule

Primary endpoint is exact first-hop intermediate bytes **and** own EOS on both held wrappers after distractor learning. The paired world is the unit. Average the two wrappers within each world and show each wrapper and answer length separately. Count every preassigned query, including premature stops and capped generations. Secondary endpoints include strict two-hop completion, per-position payload accuracy, exact payload prefix without EOS, and EOS prediction after a gold prefix. Report checkpoint transitions for the same items. Score a correct key followed by extra bytes as a failure.

Advance SPLIT-HOLD to a later confirmation design only if all of these development gates pass.

- Final held-wrapper first-hop completion gains at least **10 percentage points** over ROLL in the paired mean and exceeds SPLIT-ROLL. Both paired two-sided 95% Student-t lower bounds must be above zero. Report all world-level differences.
- Neither held wrapper loses on the paired mean versus ROLL. Final trained-wrapper exact hop-one and strict chain scores each lose at most **5 points** versus ROLL, and ROLL achieves at least **80%** final trained-wrapper strict chaining. An unqualified trained reference leaves the architecture comparison unresolved.
- Final held-wrapper strict chain accuracy reaches at least **20%** and improves by at least **5 points** over ROLL. This prevents a small recall gain near zero downstream use from triggering advancement. The absolute thresholds are design choices for a useful development effect.
- Exposure, EOS, bounded address, frozen probe and resource audits all pass. Report the cost of any benefit. Matched access ceilings alone do not demonstrate compute efficiency.

If SPLIT-ROLL improves and the SPLIT-HOLD contrast is inconclusive, record a result for the split layout with query retention unresolved. If only gold-history diagnostics improve, report a conditional readout effect. If held first bytes remain poor, prioritize wrapper-stable addressing before revising the stop learner again. If first-hop gains fail to reach hop two, use the gold-intermediate result to localize the downstream constraint. A pass is grounds for further development within the task family.

## Census, resource admission and three cycles

The Stage B marked roster is eight worlds × three arms × 320 links × 12 presentations = **92,160 marked presentations**. The matched unmarked delay adds **6,144 scheduled records**, before filler. Each presentation carries one EOS target and either three to seven payload bytes or one digit. The conservative science upper bound is **460,800 supervised output targets** using eight targets for each A → B and two for each B → digit. Stage A contributes **3,072 historical marked presentations and 12,288 output targets** for one 128-chain reproduction. Count the actual Stage A run once only if instrumentation and parity share that trained state. Charge any additional attempt separately.

Proposed cumulative ceilings, including qualification, are **105,000 marked presentations**, **600,000 supervised output targets**, **4 million read-only decision calls**, **900 CPU seconds**, **1,800 active wall seconds**, **1 GiB peak RSS**, **72 MiB deployed state per learner including cache**, and **256 MiB retained outputs**. Charge unmarked bytes, all scheduled arm applications and shared physical executions separately. Lock the exact filler stream and byte census, number of read-only branches and retained outputs before admission. A small qualification roster provides cost estimates. The full launch requires a projection on the exact locked roster with headroom and explicit authorization. No silent cap change, extra candidate, target exposure or retry.

1. **Refine → audit → test the historical bridge.** Pin source and scoring, reproduce the single historical world, and qualify the three read-only failure-localization probes with the original period output.
2. **Refine → audit → test the state separation.** Lock stage-B salts and warmup, verify causal state timing, first-decision parity on identical state, post-output query-row persistence, natural EOS and bounded traffic. Test cap, extra payload, premature EOS, missing EOS, same-digit wrong intermediate and variable lengths on engineering fixtures.
3. **Refine → audit → test the complete runner.** Use fresh technical seeds and a reduced roster through all arms, the delay, new learning, exposure audits, strict reducer and receipt archive readback. Independently compute the exact full roster and projected costs. Stop at admission if any launch requirement fails.

**Delivery** Design only. Implementation and execution remain pending.
