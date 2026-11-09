# MiniFly roadmap after Fable's reviews: a streaming, locally plastic learner with episodic recall

**Date:** 2026-11-08 EST (updated against repository evidence at commit `9d260d60d`).  
**Status:** **ADVISORY ROADMAP / DESIGN NOTE ONLY.** Not an experiment, a new Phase-A gate, a change to running jobs, or authorization to modify the core.  
**Target branch:** `codex/minifly-consolidated-20261023`.  
**Owner decision required before changing the adopted program roadmap.**

## Executive decision

The goal is **one learner, one continuous life** that (i) acquires reusable rules through permitted lifetime-local updates, (ii) preserves earlier capabilities while acquiring new ones, (iii) keeps short-lived working state, (iv) retrieves and revises one-off past episodes on demand, and (v) composes the retrieved information with learned rules. Neither "arithmetic benchmark success" nor "an ever-larger replay buffer" is the goal.

**Recommended dependency chain:**

```text
A. Freeze and audit the E16 reference
   -> B. Genuine streaming learner, including the every-byte-loss bottleneck
   -> C. Phase-B episodic binding and selective historical read
   -> D. Consolidation / revision / transfer under a fixed total budget
   -> E. Small natural-text pilot against matched recurrent and Transformer baselines
```

Do **not** add a catalogue of biological mechanisms or a new feedback architecture in parallel with this chain. Any proposed mechanism must correct a measured failure against a matched control. Prioritize learning architecture or learning mechanism improvements; routine hyperparameter tuning is diagnostic, not an end goal.

## 1. Current evidence: what works, and what does not

### Supported at synthetic task scale

- **E16-R2:** one 65,536-entry Algorithm-R reservoir, no family labels in the replay path, retains three separately learned synthetic families at >=0.80 of matched backprop twin **on family-average endpoints** in all three standing seeds and all three fresh E16-F2 seeds. On the fresh seeds: F1 operations 89.0–91.9% exact, F2 held compositions 77.0–80.5% final-answer, F3 operations 91.0–92.0%; final composition loss from the end of its teaching block 8.0–11.8 points. These are not per-operation all-pass claims or independent-code replication.
- **E16-R3:** holding prompt and observed answer **verbatim** in a single raw-byte reservoir rather than parsing and recomputing them recovers the Phase-A gate in the three standing seeds; raw and parsed family averages are within about 3 points; the reported sum of raw payload lengths is ~1.05 MB, **not total process RAM**. Fresh-seed raw confirmations E16-F3/F4 have not been certified at the time of this note.
- **E11:** a 512-dimensional diagonal-gated recurrent cell and eligible time traces support holding information across distractor bytes within a sample. The high-speed `scan` gradient is algebraically checked against forward traces but is **not time-local as an implementation**.
- **E16-WR:** self-generated targets in an E8-type replay buffer can become an error-amplification mechanism; at large capacity late-stage wrong composition targets reach 81–91%. Recorded observed targets are safer in that specific experiment. This does **not** prove that all model-generated replay or dream consolidation is intrinsically invalid.
- **Fable's review §9.47:** on-demand read head is a plausible Phase-B problem, not a demonstrated new mechanism. Retrieve **past episodes** versus replay to **retain rules**. Compare fly-style sparse indexing to a dense embedding index at a matched memory and teaching budget.

### Newest correction: the environment-scaffold ablation splits

- **E16-S (9d260d60):** smoothly drifting family mixture with online reservoir admission retains all three synthetic families at 0.84–0.99 of the corresponding backprop twin, in three standing seeds. This supports removing explicit stage boundaries in this synthetic stream; it does not yet provide cross-call persistent recurrent state or natural-language learning.
- **E16-L (9d260d60):** removing the answer-only loss mask and applying undifferentiated next-byte loss across the entire synthetic prompt **fails decisively**. At k=12 the add/sub operations only reach ~3–5%; final F1 is about 30–33% and compositions lose 30–35 points. The synthetic prompt contains many random/unpredictable bytes, so unweighted loss can swamp the rule-bearing answer signal. A **backprop twin under the identical every-byte objective** remains to be measured; do not claim that the local rule is uniquely at fault until that comparison is complete.

### Interpretation limits not to hide

1. E16 development uses ordinary backprop before lifetime learning and observed correct answers during life. It is not autonomous learning from unlabelled lived experience.
2. Some F1 operation pairs reappear as supervised subcomputations in later F2 examples. Distinguish "held out of direct F1 teaching" from "never reinforced during the entire history." Build history-wide contamination audits for next-phase tests.
3. The E16 replay reservoir is not an inference-time addressable episodic memory. The exact-window `exp45.Store` is neither a PN–KC–MBON neural network nor a learned semantic retriever.
4. In `exp11_cell.py::forward`, `h=zeros(B,C)` resets each call. A lifelong sequence of weight updates is not one unbroken internal recurrent state.
5. The reported bytes for the raw reservoir exclude Python container overhead, model/optimizer weights, eligibility state, exact-key store and temporary batch arrays.
6. A scalar teaching reward, a learned READ decision, and genuine local online credit assignment beyond the qualified cell gradients are **not yet jointly demonstrated**.

**Phase-A scientific status:** a credible and improving **toy-scale continual-learning** demonstration, not generalized persistent intelligence. Do not redefine "done" retroactively to conceal E16-L's failure.

## 2. Milestone A — freeze and audit one reference implementation

Finish or explicitly report the already queued tests before adding new models:

- E16-F3/F4: raw-byte replay on fresh seeds; preserve preregistered misses.
- E16-A: turn `exp45.Store` off, holding other training and evaluation conditions fixed. If it adds no measurable value, retire it from this harness; that is **not** a refutation of fly-like sparse associative memory on a purpose-built episodic task.
- E16-RC: capacity vs. diverse, retained raw observations, reported in **realized distinct examples and bytes** rather than entry count alone.
- E16-WR2: observed-target replay in the window-net matched to the cell learner's design assumptions.
- E16-L same-loss backprop comparator and loss attribution: assess raw unpredictability / signal weighting, exposure, token count and compute. Do not silently restore a task-specific answer mask while claiming "natural text" success.

Deliver a pinned code revision, actual model checkpoint hashes, predictions, complete held panels, source/input hashes, measured CPU/RAM, and an **authoritative table** that separates *done / failed / in progress / design only*. Do not launch additional E16 suffixes merely to force all endpoints above threshold. Keep the arithmetic task as a regression suite after it has served its causal purpose.

**Decision gate A:** an independently repeatable baseline with disclosed limitations. It need not pass every new scaffold-removal test.

## 3. Milestone B — genuine stateful streaming and a viable predictive objective

### B1. Streaming API

Implement `step(x_t,h_t,learning_state,memory) -> prediction,new_state,actions`, with explicit persistence of recurrent state and eligible traces **across calls**. Require *split-call invariance*: a given ordered history produces the same predictions, addresses and model updates when chunked in different ways. Include an explicit reset and provenance policy. Replay episodes run in an isolated workspace; they may update slow weights but must not overwrite the active live state.

### B2. E17 vs. existing fixed window

Compare the existing 10-byte window + cell to a bank of diagonal recurrent features at multiple retention timescales, fed one byte at a time, with matched trainable weights, state size, total FLOPs and retained bytes. Removing the window is **not** a success by itself. Check delayed binding, order sensitivity, new task acquisition, forgetting and length extrapolation.

### B3. Resolve E16-L before calling the learner a language predictor

E16-L established that full next-byte loss on highly random arithmetic prompts can overwhelm the answer signal. First run the **exact same loss** under backprop to separate the objective from lifetime credit assignment. Then perform a *predeclared*, one-factor study: predictability-aware weights, event/error normalization, separate predictive vs behavioral learning signals, or a locally learnable salience/teaching gate. Avoid a rule-specific parser or hidden correctness mask masquerading as natural learning. Measure every-byte predictive cross-entropy **and** retained task/answer performance; an answer-only gate can remain as an explicit supervised ceiling but not the intended natural-text learner.

**Decision gate B:** an indefinitely chunkable stateful implementation and a learning objective that remains viable on an unsegmented stream without privileged answer locations. The gate is not met by E16-S alone.

## 4. Milestone C — Phase-B episodic memory and learned read head

Fable §9.47 accepts the temporal read-head concept but correctly places it **after defining a Phase-B stream and task battery**. Incorporate `DESIGN_TEMPORAL_READ_HEAD_20261108.md` into the battery rather than spawning an unrelated branch.

### C1. Preregister the environment and splits

One continuous stream of new referents, names, events and revisions. Tests include:
- one-shot novel word/referent binding after an unseen number of distractor bytes;
- recall of a fact stated once and later revised, querying latest/earliest/as-of-time;
- two-hop compositions requiring **two separately retrieved episodes**;
- controls where no read is useful, for evaluating a learned SKIP decision.

Use random unseen entities, lexical templates, whole-timeline split audits, and newly generated facts. A test fact may be *observed and stored* during its episode: this is essential to episodic recall; what must be forbidden is hidden target access or oracle address at decision time.

### C2. Three genuine competing retrieval architectures

Keep the same learner and observation stream. Compare:
- explicit random/time-based lookup (diagnostic);
- **fly-inspired PN→KC sparse expansion / KC→MBON-like local associations** mapping cue to episode addresses, carefully distinguishing an authentic connectome-derived implementation from a generic random sparse index;
- learned dense content-key index of matched capacity.

All read **the same underlying bounded raw episode log**. Report three budgets separately: raw log bytes, index bytes, training-replay bytes. A replaceable Algorithm-R reservoir slot is not a stable historical address; either create a versioned stable address namespace or make eviction explicit.

### C3. Learned READ/SKIP control

Start with fixed or oracle candidate-address baselines, random-address and scrambled-evidence controls. Only after oracle-read+merge works train a **lifetime-local** read/skip decision (scalar broadcast reward with eligible action trace is one candidate, not an established result). A BP-trained reader is a ceiling, never the decisive local-learning evidence. Trigger reading by expected *value of information*, not entropy alone; log confident-wrong skipped reads and needless reads.

Read a **bounded number K** of **bounded byte slices**; re-encode old raw bytes and merge into a temporary workspace while preserving the live `h_t`. A purely final-logit bonus is not evidence that retrieved data can support later reasoning.

**Decision gate C:** learned retrieval and selective reading beat random/scrambled controls at matched budgets, approach an oracle retrieval ceiling on new facts over long gaps, and support revision and two-hop use. Require evidence addresses as well as exact answers, plus actual read counts, latency and storage costs.

## 5. Milestone D — consolidation and transferable knowledge

The key distinction is between **retrieving an arbitrary one-off fact** and **learning a reusable relation**. At a fixed total memory budget test:

1. **Rule durability:** after adequate examples, remove selected specific teaching records and test unseen instances of the same regularity. Do not require perfect recall of unrelated arbitrary facts that have no surviving record.
2. **Positive transfer:** quantify additional examples/updates needed to learn a new related relation, compared with a fresh matched learner and a learner with retrieval-only access.
3. **Fact revision:** new observations update currently valid conclusions while preserving unrelated skills and, where requested, historical versions.
4. **2×2 mechanism attribution:** retrieval on/off crossed with lifetime weight plasticity on/off. Distinguish remembering a record from consolidating a rule.

Only a failed and localized test justifies opposing synaptic traces, additional sleep schedules, learned feedback, or another new plasticity mechanism. Fable's mechanism catalogue and recent Nested Learning review are **hypothesis libraries**, not unbounded implementation queues.

**Decision gate D:** measured retention or sample-efficiency transfer attributable to plasticity, not only available old evidence; preserved task competencies and explicit resource growth.

## 6. Milestone E — small natural-text pilot and matched baselines

Do not begin with a large corpus or claim that the toy optimizer settings scale. Use one chronological, modest text stream with source provenance, fixed token/byte budget, held-out future text and ongoing novel fact tests. Start with equal-budget recurrent/GRU and small Transformer baselines **with comparable memory and continual-update opportunities** where the claim is architectural. Include both:

- predictive pretraining metrics: next-byte NLL, bits/byte, corpus and compute accounting;
- lifelong metrics: one-shot factual recall, correction, delayed relation reuse, forgetting, transfer sample efficiency, read rate and total training/inference+storage cost.

Stop if objective learning collapses, if test-answer leakage is detected, if error-amplified self-replay returns, or if the measured improvements disappear under matched memory/teaching budgets. "No fixed context window" is a design aspiration, **not** infinite finite-memory information capacity or free retrieval.

## 7. Relationship to original fly circuit and prior literature

The E11/E16 core uses byte embedding, shared MLP, diagonal gated cell, local sign-feedback and reservoir replay; it is **not** the original 302-PN/5,177-KC/MBON connectivity. `exp45.Store` is a degenerate exact-key logit cache. If E16-A shows that cache is inert, say so; it does not settle whether the fly's associative memory helps selective episode addressing.

External read heads and content-addressed memories have substantial prior art: Memory Networks, Neural Turing Machines, Differentiable Neural Computers, neural episodic control, kNN-LMs, retrieval-augmented models, and hippocampal indexing theory. Do **not** claim invention of retrieval. The potentially interesting result would be **local read-head training + auditable episodic recall + reusable knowledge consolidation** within a small continually learning system, with sparse-vs-dense and causal controls.

## 8. Guardrails against another back-and-forth loop

- Freeze a baseline per milestone and keep one primary scientific question per experimental change.
- Write predictions and falsifiers before results; retain every failed literal threshold and all non-promoted variants.
- Stop optimizing the mod-97 toy benchmark when it ceases to discriminate architectural or learning mechanisms.
- Never confound update rule, network capacity, replay quality, memory size, objective and training exposure in one causal inference.
- Quantify **effective teaching bandwidth**, including labels, correct observed outcomes, feedback channels, oracle addresses and structured renderers.
- Publish one compact capability ledger: **demonstrated; development-only; failed; untested**. Do not treat prior optimistic reports as current without later corrections.
- Separate engineering progress (new harness, new documentation, faster execution) from scientific gains.
- Count model, optimizer, eligibility, working state, raw log, index, replay and temporary RAM/disk individually; no memory-budget fiction.

### Immediate priority order (as of this review)

**First:** record E16-S as positive and E16-L as a genuine negative; complete E16-L's same-loss BP control. **In parallel, but without changing their gates:** finish E16-F3/F4, E16-A, E16-RC and E16-WR2. **Next:** freeze the E16 reference, implement split-call-stable streaming and run E17. **Then:** publish the Phase-B episodic battery and investigate sparse vs dense locally trained read heads. **Finally:** attempt consolidation/transfer and natural text in that order.

**Disposition of earlier design note:** retain `MOD97_LOCAL_RULE_20261106/DESIGN_TEMPORAL_READ_HEAD_20261108.md` as the detailed retrieval specification; this roadmap sets sequencing and scientific gates. The two notes are complementary.

**Bottom line:** the next important advance is not another arithmetic accuracy gain. It is one bounded, genuine streaming learner that remembers *rules* through plasticity and replay, can recover *episodes* by learned content/time addressing, and can use recalled evidence to reason and learn without privileged hidden teacher signals.
