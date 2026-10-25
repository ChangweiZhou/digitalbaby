# Literature map: Local, biologically informed learning toward MiniFly

**Date:** 2026-10-25. **Status:** literature research and triage only; no experiment or new core. **Scope:** peer-reviewed papers plus explicitly labeled preprints; 44 distinct entries. The paper list was checked against primary publication/author pages and source abstracts where accessible, not every result rerun or every supplement read. **Do not convert this bibliography into a list of 44 untested MiniFly mechanisms.**

## Purpose

This is a broadened follow-up to the recent search on biologically plausible local credit assignment, online recurrent learning, transferable representation learning, selective retention, and language-oriented memory. It is grounded in the current project failures: **V149 input/context sufficiency**, **V150 categorical output gradient succeeds in an idealized state table but fails to improve native Full151**, **V143 limited parity transfer**, and **OLD/NEW interference**.

The list includes papers demonstrating limitations and assumptions, not just positive results. Readings from full-primary publications are distinguished from preprints and engineering comparators. The focus is mechanisms, not ranking published accuracy numbers across incomparable tasks.

## Priority and evidence tags

- **P0 — read first for an immediate causal/mechanistic question** (not an instruction to implement it).
- **P1 — solid theory, boundary/alternative, or longer-term learning relevance**.
- **P2 — contextual theory or engineering comparison**.
- **peer-reviewed** identifies an accepted article/conference paper, **preprint** signals more preliminary status. No item's simulated success is direct evidence of human synaptic implementation.

**A note on the meaning of local:** A method may be local across layers, local in time, local only in the forward *update*, or require nonlocal targets/outer-loop gradients. These are different claims. A method using local objectives with external labels, a learned feedback controller, BP during a meta-optimization stage, or an attention/BPTT-trained sequence model is not a fully locally trained MiniFly analogue. Read each entry's caveat.

## First reading sequence — mechanism-first

| Step | Papers | Why this sequence matters |
| --- | --- | --- |
| 1 | **P15 RTRL**, **P16 e-prop**, **P17 RFLO** | What is the true causal teaching effect, and which terms are approximated away? |
| 2 | **P10 DFC**, **P11 PC-ALM**, **P13 manifold noise** | How can intended output error yield the right *physical state displacement* in V150? |
| 3 | **P05 scaling failures**, **P01 random feedback**, **P12 DLL** | What failure modes survive correct gradients, and what circuits are actually required? |
| 4 | **P26 CLAPP**, **P27 local/global SSL theory** | Why some local rules form useful, reusable representation geometry while others memorize? |
| 5 | **P32 context selectivity**, **P33 two-factor consolidation**, **P38 TEM** | How to preserve *relevant* facts and transfer common structure across contexts? |

## Specific questions for the existing MiniFly experiments

**V150 / native coupling:** Measure the signed local descent effect `-grad_loss(q) dot delta_native(q)`, and the cross-context response under matched teaching amplitude before new training. DFC/PC-ALM/NMNC/RFLO papers provide competing hypotheses; do **not** infer that a local softmax gradient is automatically implemented by native signed plasticity.

**V149 / input code:** Establish whether frozen KC activity supports *context-dependent predictions* with the same information budget. CLAPP/SoftHebb/local-loss papers suggest representation criteria; their deep vision benchmarks do not prove general language representation.

**V143 and beyond / systematic transfer:** Test whether a learned representation carries reusable transformations, not only answer-class equivalence. TEM structurally separates world-specific content and reusable relations, but its relation/action inputs are supplied: this is a mechanistic comparison, not permission to hand-code a parser.

**OLD/NEW / value of memory:** Measure relevant-future-query risk and protected versus unprotected influence rather than universal raw OLD retention. Compare GATE, synaptic-intelligence and CLS/cascade papers against the already-tried MiniFly variants before proposing repetitions.

**Computational baselines:** RTRL, KF-RTRL, SnAp, Titans and fast-weight programs are valuable mathematical and engineering references; some explicitly depend on global derivatives, attention or BP outer loops. They are not thereby approved native architecture proposals.

## No-bypass rules

1. The published method's *training information* must be recorded: labels, generated negatives, future signals, replay buffer, global targets, outer-loop BP, attention, and context/task hints.
2. Classifier accuracy, correlation with brain recordings, or bits-per-byte scores on a different corpus are **not** evidence for MiniFly E3 complete byte/EOS transfer.
3. A paper on retention does not become evidence of learned rules; a paper on supervised image representations does not become evidence for autonomous NLP training.
4. Include negative scaling evidence (P05) and ideal-vs-native mediation tests in any later synthesis.
5. Mechanisms already in Claude's 150 human-learning catalog and the 254 historical configurations are not counted as newly missing merely because they appear here.
6. These priority labels express **relevance to the unresolved project questions**, not scientific merit, biological likelihood, or a prediction of success in MiniFly.

## Paper list

[All 44 entries, sources, short mechanistic contributions and explicit scope limitations](PAPERS.md). Structured machine-readable index: [PAPERS.json](PAPERS.json).

## Other projects (not counted as peer-reviewed papers)

- **[FLM / Fly Language Model](https://github.com/Kuberwastaken/flm):** from-scratch connectome-derived recurrent language architecture, but core language training uses standard through-time gradient optimization. Its WikiText and BabyLM claims are project-reported engineering results, not peer-reviewed demonstrations of local-from-scratch NLP.
- **[Thousand Brains Project / Monty](https://github.com/thousandbrainsproject/tbp.monty):** lifelong sensorimotor/object learning research and open architecture, not a proven solution to MiniFly's local byte/EOS training problem.

## Provenance

- MiniFly context: `minifly_v150_dev_20261025/REPORT.md`, `minifly_v149_input_ceiling_design_20261025/DESIGN.md` (design status at cataloging), `PROJECT_DOCUMENTS/MINIFLY_PREDICTIVE_DIAGNOSIS_20261025/REPORT.md`, and `PROJECT_DOCUMENTS/HUMAN_LEARNING_AUDIT_20261024/REVISED_AUDIT.md`.
- Primary literature: URLs in the structured and Markdown lists. Paper titles/venues/years are source-checked where available; direct code URLs are linked only for works with identified repositories.
- As of October 25, 2026. Newer papers or changes in project state are outside this snapshot.
