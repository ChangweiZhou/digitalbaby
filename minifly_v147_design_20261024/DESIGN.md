# V147 — bounded local relational-learning diagnostic

24 October 2026. **Design only. No implementation, three-cycle exercise, training, main assessment or adoption is reported here.** The user authorized this modular task as a bounded diagnostic. **The project goal remains better complete own-byte/own-EOS generalization to unseen ordinary addition per total training sample and compute.** Modular success does not replace that goal.

Keep gradual warmup, local rules only, generic raw-byte input and autonomous byte/EOS output. No backpropagation/BPTT, attention, Transformers, ELM, pretrained models, numeric features, supplied arithmetic rules, answer-length hints or vocabulary masks enter any learner. Arithmetic exists only inside the evaluator that creates observations and scores answers.

## 1. Question and narrow claim

Can a pinned MiniFly-derived local learner use feedback to answer genuinely untaught relations in a randomly relabelled composition table? Does local feature plasticity help beyond a fixed representation with a learned readout?

V146-R1's full 24-fit local readiness menu failed. Its higher-rate oracle-map/no-shrink conditions reached 88.75%/85% taught accuracy but 0%/13.33% held accuracy. Learned quantity/counting curricula were therefore not run. That result does not identify a universal local-learning failure or prove that quantity grounding is required. It motivates isolating acquisition and relational transfer without another map/expansion/shrinkage bundle.

V147 changes the diagnostic **data**, not the mathematical machinery midway through a trajectory. It does not import the gradient-trained modular reference as a learner or a proven local feasibility ceiling. The published larger-alphabet log used different exposures, ordered-pair splits and class outputs.

Evidence levels: interface qualification E0; taught acquisition E1; successful first answers on wholly unreinforced relations can support bounded E3 within this task. This is **not** a sequential-operation/composition test, ordinary-addition success, retention result, native Full151 efficacy result or persistent-core promotion.

## 2. Task: an evaluator-hidden relation, not a numeric input feature

Fix p=23 before any trial. In each world the evaluator samples one bijection sigma from hidden indices 0–22 to 23 distinct payload bytes drawn without replacement from all 256 byte values. NUL, whitespace and punctuation are valid symbols; no byte has an implicit terminator role.

The evaluator defines:

`sigma(a) star sigma(b) -> sigma((a+b) mod 23)`.

The learner receives only the **two raw prompt bytes**, a generic end-of-prompt API event, then observed response byte and EOS during teaching. There is no literal operator/parser, operand-specific encoder, index a/b, sorted alphabet, group identity, number line, angle, Fourier feature, transition table or modular arithmetic routine in learner code. The alphabet list and sigma inverse stay in the evaluator. Model initialization is independent of sigma.

The task asks for one payload byte, but the model is not told that length: it may emit zero or more bytes and must choose EOS 256 itself. Score exact payload plus EOS within eight output decisions. Premature EOS, extra bytes, invalid nine-bit codes and cap exhaustion are failures. Do not turn a class prediction into a forced `byte,EOS` answer.

The same generic byte interface must accept variable-length prompts and outputs. A two-byte diagnostic prompt does not authorize a two-operand parsing branch. The only phase information is the inherited generic observed-input/observed-response boundary.

## 3. Strict split and exposure audit

There are 276 unordered pairs including diagonals. Each result symbol has 12 unordered relations in the complete evaluator table. Select **three held orbits per result symbol**, leaving nine taught orbits per symbol: **207 taught / 69 held**, with a balanced orbit-weighted target distribution.

Both orientations of every off-diagonal orbit belong to the same split. Present both during teaching; present a diagonal once per pass. Shuffle the complete oriented record list each pass with the shared order substream; do not supply orbit-grouping metadata to the learner. Require every symbol to occur in both taught input positions and as an observed taught output. Draw a new split under fixed seeded rejection if coverage fails; cap attempts at 10,000 and fail rather than change constraints. This outcome-stratification is an evaluator sampling policy, not information passed to the learner.

A pass has `N=414-D_train` records, where D_train is the number of taught diagonal orbits; thus 391 <= N <= 414. Each record has two external prompt bytes and two observed response/control tokens: one response byte and EOS. Record N and exact token counts per world. Generated guesses and probes are not external feedback.

Held prompts are never reinforced in either orientation. Their first scored response precedes feedback, and **no held feedback is ever given**. No held labels, losses, normalizers or predictions select the final configuration. In calibration, score taught behavior only; held behavior is reserved for assessment after settings are frozen.

## 4. Pin the existing local route

Executable parent: the V146-R1 local surrogate at commit `03cf7cd7`, particularly `minifly_v146_r1_dev_20261024/core.py`, together with its source-pinned FE-PERM and canonical PN/KC dependencies. V143 fixed-feedback feature learning is its historical inspiration, not proof of inherited full-addition capability.

| Component | V147 disposition |
| --- | --- |
| 88-channel byte identity FE-PERM, actual event timing | Retain unchanged |
| Extra 32 map channels in the parent's 120-input layout | Set identically to zero in every arm; no map learning |
| H=128 ReLU combination layer and fixed 88-by-H projection | Retain initialization family and forward computation |
| Canonical PN/KC sparse expansion, side-wise top-k and ties | Retain exact source/asset definitions |
| Nine opponent delta output pairs | Retain local surrogate and full byte/EOS serialization |
| Feature teaching | Retain fixed-feedback three-factor update where enabled |
| Shrinkage, ordinary weight decay, replay, SOM and quantity/counting experience | Absent |

The nine opponent pairs are not a 23-class answer head. All payload bytes remain legal emissions; the model receives no allowed-symbol mask. The surrogate naming store is **not** the native Full151 value-memory update, even though connectivity/selection assets are inherited. No native output or retention claim follows without integration.

Before implementation, verify parent hashes and the actual dependency/asset closure. Restore only missing source and canonical assets, not historical trajectories. Do not silently replace the canonical expansion with a convenient random matrix.

## 5. Local updates and gradual warmup

For input trace p, use the parent's forward path: `a=W p+b`, `h=relu(a)`, fixed projection, canonical drive and binary side-wise top-k code x. Let K=sum(x). Opponent margins are:

`v_j=(theta_plus[j]-theta_minus[j]) dot x / sqrt(max(1,K))`.

For an actually observed token y, `t_j(y)` is its signed nine-bit serialization and

`e_j=(tanh(v_j/2)-t_j(y))/2`.

All updates use pre-update margins, gates and arrays. The local naming update is:

`theta_plus[j] -= alpha s(n) e_j x/sqrt(max(1,K))`;
`theta_minus[j] += alpha s(n) e_j x/sqrt(max(1,K))`.

The parent's single-array representation of exactly opposite opponent weights is permitted; verify the algebraic equivalence. For feature-plastic arms:

`d=(G e)*(a>0)`;
`W -= 0.003 s(n) outer(d,p)`; `b -= 0.003 s(n) d`.

G is fixed seeded random feedback, not the derivative through hard top-k. Its distribution and all initialization substreams remain pinned to the parent specification. No claim that this is an exact gradient or a biologically validated circuit is made.

**Every learning channel uses** `s(n)=1/16+(15/16)min(n/1024,1)`, with n completed prior teaching records. Both response and EOS targets within a record use that factor; increment n only after the full record. Count low-rate warmup exposures. Do not jump directly to a fully trained-rate state.

During training the response prefix consists of actually observed bytes. During testing it consists exclusively of generated bytes. Expected bytes are legal supervised feedback in this pilot; scalar-only learning is not tested. Probes leave learning state, adaptation and future-training RNG unchanged. A bounded input-only trace cache may change operationally, but must contain no targets/answers, yield identical cold-cache results, and stay within 4,096 entries.

## 6. Four causal arms

All arms share their world, birth arrays, taught record order, interface, token budget and checkpoint schedule. Mutable learning states are independent.

| Arm | Feature W/b writes | Naming writes | Observed teaching targets |
| --- | --- | --- | --- |
| FIXED | Disabled | Enabled | True relation |
| LOCAL | Enabled | Enabled | True relation |
| SHUFFLED | Enabled | Enabled | Orbit-consistent shuffled targets |
| NO-WRITE | Disabled | Disabled | True relation, identical input/timing |

NO-WRITE still receives the observations and advances exposure counters; parameter arrays stay fixed. Disabling all prior relation learning is distinct from disabling only a later regularizer. Compare against the initial birth probe as well.

SHUFFLED permutes the observed target assignments **among taught orbits**, separately for diagonal/off-diagonal multiplicity so oriented target frequencies and token counts match. Keep each orbit's two orientations consistent. Lock the shuffle before fitting and report realized agreement with true targets. Reject a draw that is merely one global output-label permutation across all taught orbits, using at most 10,000 seeded attempts and then failing closed. Such a global renaming is not sufficient: it preserves structured composition. Shuffled teaching is not assumed to score exactly zero.

Primary mechanistic contrast: LOCAL minus FIXED at matched presentations. Generalization can also occur through FIXED's locally learned readout; that is legitimate limited relation learning, not evidence that feature plasticity helped. LOCAL-minus-SHUFFLED and trained-minus-NO-WRITE test teaching/structure dependence; they do not fully factor every architecture-by-label interaction.

## 7. Recitation and interface controls

For each assessment world compute evaluator-only:

- exact ordered-key and unordered-key lookup;
- most frequent taught output;
- copying the first or second prompt byte;
- nearest taught answer under the arm's actual checkpoint representation, with squared Euclidean distance and lexicographic raw-byte tie breaking.

Each comparator emits its chosen byte plus EOS or explicitly abstains. Its evaluator-side output routine is not inherited by the learner. Since held relations exclude both orientations, lookup abstention is expected and is not a meaningful positive hurdle. With the balanced held split, a constant single-symbol answer has orbit accuracy 1/23; do not confuse that with the actual birth model's complete-byte/EOS baseline.

Check fixed-representation collisions and any conflicting labels before interpreting acquisition. Unique codes alone do not prove that the local readout can fit all labels. A taught-key table and scripted synthetic byte/EOS fixtures verify data/scoring plumbing; neither is a learned local positive control for unseen relations. There is **no already verified solver for this exact strict-split local assay**, so negative behavior cannot establish task impossibility.

Log same-input KC stability, feature-update norms and residuals separately for informative versus nearly constant bit channels. These diagnostics test the address/credit hypotheses suggested by V146; they do not become substitute success metrics.

## 8. Finite development and assessment plan

Use SHA256-derived independent substreams for byte assignment, relation split, model birth, presentation order and shuffle. Use a pinned NumPy PCG64 generator for each substream and record its runtime version. Seed rule: first eight SHA256 bytes of `V147|world|substream`, interpreted as unsigned big-endian integer. Do not choose a successful seed, change p, or alter symbol coding after inspecting assessment results.

**Calibration:** worlds `V147-D1/D2`; alpha in {0.003,0.03}; FIXED and LOCAL only; 256 passes each. This is **eight fits**. Inspect taught complete accuracy, loss, stability and cost, not held scores.

An alpha is acquisition-ready if at least one of the two routes reaches >=95% taught complete accuracy in **both** calibration worlds at pass256. This is an engineering readiness threshold, not a definition of abstraction or a universal scientific constant; report exact errors. Among eligible alphas choose highest mean taught complete accuracy over both routes/worlds; tie by lower summed CPU then lower alpha. If neither is eligible, stop as `ACQUISITION_NOT_READY`; do not widen the menu or conclude that modular structure is unlearnable.

**Technical rehearsal:** fresh `V147-TECH1`, the selected four arms for 16 passes, plus small paid interruption/serialization fixtures. Test the complete model, graph assets, traces, RNG, counters and cursor in a separate process. This shortened rehearsal is technical, not a long-learning or grokking result. No novel intervention turns on after warmup, so it need not repeat thousands of practice passes merely to qualify identical late-step plumbing.

**Assessment:** `V147-H1` through `H4`, generated only after source/settings/metrics/baselines are locked. Four arms per world: **16 continued histories**, each from birth through at most 4,096 passes. Checkpoints at birth and passes 1/4/16/64/256/1,024/4,096. No restart at each checkpoint. The four worlds are relabellings/splits/initializations of one finite group, not independent task families.

Total planned substantive fits: 8 calibration + 4 rehearsal + 16 assessment = **28**, excluding separately enumerated tiny technical fixtures. All are developmental/diagnostic; H denotes an untouched assessment holdback, not population confirmation.

## 9. Endpoints and interpretations

Primary outcome is final **complete held payload-plus-own-EOS accuracy**, averaging orientations within orbit and then all 69 held orbits. Publish every world/arm/checkpoint, fully correct orbit counts and paired LOCAL−FIXED, LOCAL−SHUFFLED and trained−NO-WRITE effects. Use the same orbit-weighting convention for taught and held complete scores. Report payload-only on the same denominator, correct stopping, invalid/truncation rates, taught accuracy and bitwise predictive loss. Teacher-forced loss is a diagnostic, not generated-answer success.

A **screen-positive relation signal** means at least one true-trained route beats NO-WRITE and the strongest recitation comparator in all four assessment worlds; for LOCAL also require superiority to SHUFFLED. This is a directional trigger for the addition return test, not statistical confirmation or a claim of a recovered algorithm. Show effect magnitude and every disagreement; one-orbit gains must be described as weak. Do not manufacture a universal effect threshold. Feature-plasticity benefit requires positive LOCAL−FIXED differences and its extra cost to be disclosed; FIXED success alone does not meet that claim.

Report paired world-level uncertainty descriptively. Orbits/orientations/tokens are not independent training replicates, and four world signs are not a p<0.05 confirmation. Any later confirmation needs a fresh, separately designed assessment.

**Grokking is secondary.** For descriptive timing, define fit time as the first of two successive scheduled checkpoints with taught complete accuracy >=95%; define generalization time as the first of two successive checkpoints with held complete accuracy >=50% and above controls. Report the exposure/CPU gap if both exist. These reporting thresholds are conventions, not learned-mechanism evidence. A final-checkpoint-only rise is not sustained by this definition. Earlier generalization passes the capability question without a delay, and the entire curve remains visible.

The 4,096-pass dose is a bounded diagnostic limit, not an established sufficient duration for grokking. Failure within it is finite negative evidence, not a universal nonlearnability conclusion.

No regularization arm is added here. No architecture, data curriculum or learning rule changes midway through a history. This avoids repeating V146's confounded attempt to infer rule extraction from smaller norms.

## 10. Total sample/compute accounting and ceiling

Per world/arm record unique taught orbits, all repeated records, raw prompt bytes, observed response bytes/EOS, probes, failed attempts, CPU, wall time, peak RSS, mutable/fixed storage and cache size. Each completed task record exposes four external tokens and two target tokens. Report both logical branch exposure and physical executed work if immutable assets or prefixes are shared; sharing does not create independent evidence.

Conservative maxima are:

- Calibration: 8 × 256 × 414 = 847,872 records.
- Rehearsal: 4 × 16 × 414 = 26,496 records.
- Assessment: 16 × 4,096 × 414 = 27,131,904 records.
- Total substantive runs: **28,006,272 records; 56,012,544 observed target tokens; 112,025,088 external tokens**.

Allow at most **4,096 additional technical record applications**, each with at most four external tokens/two target tokens, on separately named fixture identities. Thus the combined design maximum is **28,010,368 record applications, 56,020,736 observed target tokens and 112,041,472 external tokens**. Non-learning byte/probe operations have separate event/CPU counters and cannot conceal teaching outside this allowance.

These are limits, not promised exposure. Replace the substantive bounds with exact per-world counts in the locked manifest; enumerate each technical fixture within its separate allowance before execution. Repetition is charged. A long successful run is capability evidence, not automatically sample efficiency.

Proposed all-new execution ceiling: **six summed process CPU-hours**, at most two one-thread workers, including imports, development, fixtures, failed attempts and reduction. Qualify measured per-work-item cost before assessment admission. If the finite roster cannot fit, stop as `INCOMPLETE_V147`; changing the budget/dose requires a dated prospective revision, not an extension after seeing outcomes. This document does not authorize that run.

At identical record counts, report the extra feature-learning CPU. For compute comparisons use only overlapping measured budgets and latest whole-pass/checkpoint states selected by counters rather than held accuracy. Report both presentation and CPU curves; a favorable final accuracy alone proves neither lower-sample nor lower-compute learning. Do not compare epochs or optimizer steps directly with Claude's full-batch gradient reference.

## 11. Integrity without another infrastructure research program

Reuse existing qualified byte/scoring/roster machinery where possible. Before launching, save the finite machine manifest, source/asset/runtime locks and expected roster. An outer supervisor writes a durable attempt ledger before imports. Save all predictions and complete checkpoints locally; publish compact tables/source/reports rather than raw arrays by default.

Checkpoints contain all learned/fixed arrays, graph identity, RNG, warmup counts, phase/pass/record cursor and accumulated cost. Graceful pause occurs at whole-record boundaries; resume begins at the next uncommitted record. No automatic scientific retries or replay of committed trajectories. Charge every failed attempt and paid equality-test continuation.

Reject missing/duplicate/extra/partial histories, nonfinite metrics, source/asset mismatch, invalid exposure counts, held writes, and mutable probe behavior. Do not silently summarize whatever files exist. A validity check is not scientific confirmation. Historical V143–V146 designs/results remain unchanged.

## 12. The required return to unseen addition

If acquisition fails, stop and diagnose it. If acquisition succeeds but held relations do not, investigate one generic interaction/credit change on fresh development data rather than add maps, replay and recurrence together.

If either true-trained route shows a credible held-relation signal, the next proposal must return to **ordinary decimal addition with the same kernel and byte/EOS interface**, using fresh whole-orbit holdouts and matched controls. Only data/feedback streams change; no number parser, arithmetic representation, carry procedure or new output head may be installed. Starting a fresh addition learner tests the architecture; it must not be called transfer of the learned modular operation. Transferring the modular checkpoint would be a distinct, controlled pretraining question.

Until that return test succeeds, the original unseen-addition objective remains unmet. Sequential reuse, native-memory integration and retention after genuine intervening learning are later tests, not capabilities inherited from this diagnostic. Four-operation arithmetic, action-world tasks and language benchmarks are not added to V147.

**Delivery boundary:** this Markdown ships a prospective design. It starts no training, performs no three-cycle exercise, adopts no core and does not push GitHub. Implementation and execution require subsequent explicit instructions.
