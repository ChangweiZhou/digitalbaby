NOTE (discussion record; no experiment, no claim promotion).

# Outstanding issues in MiniFly after V170, classified as architectural or technical (27 October 2026)

## 1. Purpose and classification criterion
This note lists the issues that currently stand between the MiniFly programme and its stated objective. The objective is a persistent learning core that transfers never-taught relations (criterion E3) using its own byte and end-of-line outputs. The note draws on:
* experiments V153–V170;
* the memo `minifly_memo_credit_horizon_20261027/MEMO.md`, with its erratum;
* the external synthesis on the Codex branch (`PROJECT_DOCUMENTS/MINIFLY_MECHANISM_SYNTHESIS_20261026/`, commit fb6d9b5b).

Each issue is classified by the criterion used in earlier discussions:
* **Architectural:** the present structure cannot resolve it without an additional component or pathway.
* **Technical:** refinement, repair, tuning, integration or improved methodology can resolve it within the present structure.

Issues whose classification depends on evidence not yet available, or on a decision about constraints, are listed separately, with the observation that would decide them.

## 2. Architectural issues
**A1. Retrieved memory cannot influence subsequent computation.**
* Native MiniFly reads its KC → MBON memory only at the output stage. No retrieved content re-enters a working state.
* In V166, a matched network without re-entry performed single retrieval but not two-step composition (3.09 against 0.009 bits/position).
* The sufficiency of the added loop has been shown only with back-propagation through time, on synthetic episodes, outside the native system.

**A2. The context horizon is bounded by a fixed superposition trace.**
* The order-keeping front end represents the past as an 88-dimensional superposition of attenuated, permuted input patterns.
* The memory capacity of such a linear system is bounded by its dimension (Jaeger, 2002); the attenuation constant redistributes that capacity across lags but cannot increase it.
* Empirically, the trace retains about four symbols (headroom check), and a slower attenuation degrades prediction by 0.87 bits/symbol (V153).
* Exact long-range information must therefore be written to and retrieved from a content-addressed store during processing, which depends on A1 and A3.

**A3. Memory operations are not under learned control.**
* In V166, V168 and V170, writes and queries are imposed by input markers and fixed schedules. Native writes occur only at teaching events.
* No component maps internal state to plasticity or to retrieval timing. In the insect, the plausible analogue is state-dependent neuromodulatory gating.

**A4. Order is represented as lag rather than as role.**
* The front end tags each symbol by its lag, so identity and position are entangled. This supports discrimination of orders (100% in the CLS round), but provides no representation shared across positions.
* Order-dependent operations therefore cannot generalise. Consistent with this, the addition diagnosis found that a+b and b+a received the same answer only about half the time.
* A position-invariant role representation would require a recurrent state with a shared update rule, or explicit role binding. Neither exists natively.

**A5. The learning signal for the memory loop must depend on the memory itself.**
* The exact error at the query is Mᵀ(∂L/∂r), where the episodic memory M changes in every episode.
* A fixed random feedback matrix cannot reproduce this (synthesis, memo 02 §8), and diagonal eligibility traces cannot carry it. In V168, the e-prop update for the query weights is identical to one-step truncation.
* Online approximations tested so far fail: COLA in V164, and KF-RTRL, which recovers 3% of the available gain, in V168.
* The issue is architectural if learning must be strictly synapse-local, because a feedback pathway through the memory would then be required. It is technical otherwise (C1).

**A6. Learned composition is highly sample-inefficient without structural support.**
* Back-propagation through time required more than 9,600 line presentations for V166's loop, and did not succeed within 9,600 on V170's task.
* A fixed re-entry pathway (V170, strategy A) made composition learnable from 500 presentations with one-step credit.
* This suggests the required efficiency derives from structure rather than from the learning rule. Because strategy A's pre-registered guard failed, the observation is not a verdict.

## 3. Technical issues
* **T1. The locally trained top predicts poorly.**
  * Direct feedback alignment trains the second hidden layer slowly: its alignment with the true gradient is 0.43, against 0.74 for the first layer.
  * The learner's own predictions on its training text are about 0.9 bits/symbol worse than counting, with the best epochs at the budget limit (V167, V169).
  * Remedies: a larger training budget, learned feedback alignment, and learning-rate selection.
* **T2. Output port and native writer.** V153 attributed 2.38 bits/symbol to the output port (2.13 recoverable by a class bias), 0.78 to native structure, and 0.15 to online learning. The repairs are identified but not integrated.
* **T3. Birth wiring.** Fan-out across input channels is highly uneven (coefficient of variation 0.91), and 323 KCs receive no input. Even fan-out improved prediction by 0.096 bits (V159). Not integrated.
* **T4. Variance and optimiser interaction of approximate online credit.**
  * In V168 the KF-RTRL estimate is unbiased, but a single draw has cosine 0.23–0.44 with the exact gradient, and post-optimiser updates were not measured.
  * Diagnosis (synthesis E1) and an exact three-step forward-sensitivity reference (synthesis E2) are the appropriate next steps, if non-local online rules are admissible.
* **T5. Absence of native integration.** The learned layer, the memory loop and the re-entry pathway exist only as external simulations. Own-byte generation with end-of-line termination has not been tested with any of them.
* **T6. Methodological defects:**
  * a guard that measured confidence rather than retrieval (V170);
  * an arithmetic error in an exploratory script (memo erratum);
  * a single learner seed and at most two text pairs;
  * back-propagated evaluator probes used as instruments;
  * an unverified numerical scaling in COLA's statistics (V164);
  * the absence of a local readout in the transfer evaluation (synthesis E0).
* **T7. Training budgets.** Record caps have repeatedly placed results at the budget limit (V167, V169, V170), and reference models often fail to qualify within them.

## 4. Issues not yet classifiable
* **C1. Temporal credit through memory** (A5 without the locality requirement).
  * Classification depends on whether learning rules must be synapse-local or only online.
  * Decided by an owner decision on that constraint, followed by synthesis E1/E2: whether an exact three-step trace learns the loop at the target budget.
* **C2. Generalisation of order.**
  * V170's held-out test required transitive inference over arbitrary tokens, an ordinal code that no component supplies. It did not test role binding.
  * Decided by a role-binding task, in which the same operation is applied to operands in held-out orders or positions, with token identity separated from token ordinality.
* **C3. Value semantics of symbols.**
  * Digits are arbitrary bytes. With tally input, the same learner solved 98–100% of unseen sums (addition diagnosis).
  * Value structure may be inducible from prediction (technical), or may require an ordinal or magnitude representation (architectural).
  * Decided by whether a learner trained only on byte prediction develops ordered digit representations that support unseen sums.
* **C4. The deficit relative to counting in prediction.**
  * All learners, including back-propagated ones, fall short of interpolated Kneser–Ney in every protocol tested. Under the matched protocol of V169 the deficit is 0.59 bits/symbol.
  * The cause may be budget and regularisation (T1, T7), or the absence of an effective combination of the slow learner with the native fast memory, which is the natural store of exact n-gram statistics. V162's combination added nothing.
  * Decided by measuring a converged local learner under the matched protocol, and then testing whether its combination with the fast memory outperforms either component.
* **C5. Persistence of the slow learner.**
  * No forgetting was observed across two related authors (V169), and replay infrastructure exists from the CLS round.
  * Not yet an observed problem; it requires longer and more dissimilar sequences of training texts.

## 5. Issues settled in development
* **Spatial credit for a feedforward layer.**
  * A broadcast output error delivered through fixed random feedback trains a transferable layer above the KC code without back-propagation in any component.
  * Replicated on two text pairs (V165, V167, V169).
  * This is a technical resolution: a learning rule.
* **Locus of representation learning.**
  * Learning must occur above the KC layer, because the native PN → KC wiring offers negligible headroom (V155).
  * The added learned layer is an architectural change, adopted in development but not in the native system.

## 6. Issues binding the objective, and open decisions
For criterion E3, the binding issues are:
* A1, A3 and A4: the memory–workspace loop, memory control and role representation;
* A5 / C1: temporal credit through memory;
* C3: value semantics;
* followed by T2 and T5: the output port and native integration.

T1 and C4 bind text prediction more directly than they bind E3.

Two decisions rest with the owner, because they determine how A5/C1 and A6 are classified:
1. Whether learning rules must be strictly synapse-local, or only online.
2. Whether fixed, task-independent structure, such as the re-entry pathway of V170, is admissible as part of the architecture.

## References
* Jaeger, H. (2002). Short term memory in echo state networks. GMD Report 152.
* Repository: the V153, V155, V159 and V162–V170 reports; `minifly_memo_credit_horizon_20261027/MEMO.md`; `minifly_cls_dev_20261017/REPORT.md`; `addition_difficulty_diagnosis_dev_20261017/MEMO.md`; `PROJECT_DOCUMENTS/MINIFLY_MECHANISM_SYNTHESIS_20261026/` (Codex branch, commit fb6d9b5b).
