---
name: minifly-research-loop
description: "Use for any MiniFly/DigitalBaby research iteration, experiment design, result audit, architecture proposal, status claim, version allocation, skill update or cross-agent handoff. Applies before proposing another version and at every design/implementation/development/confirmation/promotion transition. Reconciles Claude MRK-1 recursive-improvement governance with current V164–V181 evidence. Requires source-grounded verdicts, lifetime-oriented learning, quantitative mini-scale accounting, falsifiable controls, owner authorization and honest stops. A retired MRK-1 checker is NOT assumed executable."
---

# MiniFly recursive research loop — integrated skill, 2026-11-02

**Read first:** [CURRENT_EVIDENCE_20261102.md](references/CURRENT_EVIDENCE_20261102.md) and [SKILL_READINESS_20261102.md](references/SKILL_READINESS_20261102.md). Use the **latest branch HEAD and actual source files**, never this dated snapshot alone, for a current answer. The [original Claude skill](references/ORIGINAL_CLAUDE_SKILL_20261102.md) and its [STAGES](references/STAGES.md) / [AUTHORIZATION](references/AUTHORIZATION.md) are preserved; they are **historical MRK-1 instructions, not evidence that its checker can run here**. The other four skill entries cover governance, claims, independent review and E3 design.

## If you only read one screen

1. **Locate the active learning lineage and frozen baseline** before inventing an architecture. Record repo, branch, exact commit, candidate model, parameter/storage cost, actual last result and pending/NOT_RUN status. Read `PROJECT_DOCUMENTS/MINIFLY_VERSION_INDEX.json`, not just README headings or version number.
2. **State the limiting observable and two mutually exclusive explanations.** Say exactly what result would distinguish them and which next action either result would justify. If both results merely motivate "another diagnostic," **STOP and redesign the question**.
3. **Keep the actual learner's life together.** Its own raw-byte/257-way next-byte and EOS behavior, E1 own complete answers, preservation of still-valid knowledge, revision, reuse and eventually held never-taught arithmetic are the endpoints. Fitted probes, BPTT, connectivity spectra and teacher-forced NLL are distinct diagnostics.
4. **Mini is a physical resource contract, not a marketing adjective.** Count trainable parameters including output head, immutable native sensory arrays, active neurons, optimizer moments, all clones/checkpoints, RAM, CPU, unique examples, repeated presentations, and output bytes **before training**. V180's 4.23M-parameter/16-life program failed admission; V181 is a 37,313-learnable-parameter **unexecuted design**, not a verified deployed core.
5. **Make controls falsifiable.** Use exact same-source, same-history forks; match the intervention's effective *post-optimizer* update when evaluating plasticity. Do not serially block a possible remedy on failure of an intermediate architecture that is not logically necessary (V178 sparse-ungated → gated failure).
6. **Only publish what was observed.** `DESIGN_ONLY`, `NOT_RUN`, `CANNOT_TEST_CLAIM_AS_SPECIFIED`, `RESOURCE_NOT_ADMITTED`, `FAIL`, `INVALID` and bounded development positives are distinct. No adopted core, independent confirmation or E3 from an evaluator probe, late-checkpoint improvement or a soft claim.
7. **No phantom checking or authorization.** The historical `tools/minifly_check/minifly_check.py` dispatcher and several specified modules are **absent** from Claude's archived tree. Until restored and independently qualified, NEVER report `MFC gate/launch/score/claim` as run or pass, and do not invent its records. A read-only GitHub audit is not a runtime replication. Explicit permission to edit skills **is not** automatic permission to train, change experiment caps, promote, merge model branches or modify active Codex runs.

## A. Readiness: tools before protocols

The 2026-11-02 Claude snapshot has five skills, rich reference files and a historical KB, **but NOT a functioning MRK-1 CLI**. The checker specification on Claude's branch describes a proposed program, not an executable present on the consolidated branch. Older skill sections calling commands such as `MFC gate --stage lock` are **dormant**, even if you can read their check names.

For a local clone:

```bash
git rev-parse --show-toplevel
git branch --show-current
git rev-parse HEAD
git status --short
test -f tools/minifly_check/minifly_check.py && echo CLI_FILE_PRESENT || echo MRK1_CLI_MISSING
```

Even when the CLI file appears, **do not assume certification**. Inspect its dependency closure, expected tests/templates and installed schemas, run a selftest on a disposable copy, and obtain cross-agent review for any promotion claim. A missing file, missing module or failed selftest means `CHECKER_NOT_ADMITTED`, not PASS. Do not create synthetic output/ledger files to satisfy a spec. GitHub-connector-only agents report `STATIC_SOURCE_INSPECTION_ONLY`, with no pretend shell execution.

The historical MRK-1 technical specification remains at Claude's [original checker source directory](https://github.com/ChangweiZhou/digitalbaby/tree/claude/vibrant-cerf-sje726/tools/minifly_check). **Only the usable skill and historical KB are integrated here**, not a fictitious runnable checker. The missing dispatcher/worker/templates and pre-UD status are recorded in [SKILL_READINESS](references/SKILL_READINESS_20261102.md).

## B. Immutable scientific objective and evidence types

**Goal:** A small, generic, locally learning model that learns from observed raw bytes and legitimate environmental feedback, retains useful acquired knowledge despite subsequent learning, corrects obsolete knowledge, uses retained learning on never-taught relations and emits its own *complete* byte sequence and EOS. Evaluate sample efficiency (distinct experience and actual repeats) plus real compute.

- **E0:** wiring/update/interface/source admissibility, representational witness or reference. No learned competence implied.
- **E1:** complete *own*-generated taught answers with authentic EOS. Next-byte teacher-forced improvement alone is not E1.
- **E2:** still-useful knowledge retained after subsequent learning; make genuine corrections distinguishable from preserving obsolete labels.
- **E3:** well-defined, never-taught addition/transfer, with whole-relation/orbit holdouts, no-prior-teaching, frequency/lookup/shortcut controls, complete *own* bytes and EOS. Development synthetic transfer is not independently confirmed E3.
- **E4:** broader cognition is not established by MiniFly's current evidence. Do not infer it from E3.

Original E0–E4 contract: `minifly_b/package/DOCS/MINIFLY_OBJECTIVE_CONTRACT.md` when source is available. E3 skills specify determinacy, causal units, strata, the shortcut ladder, OLD protection and sealed guard methodology. The historical KB treats only sealed G as **certified program progress**; other outcomes still contribute **honest engineering knowledge**, not proof of end-goal progress.

**Architectural boundaries:** no Transformer/attention or renaming an attention-equivalent content-addressed read, no supplied digit/operand/carry features or target-dependent representation, no pretrained ELM/model providing the answer, no forced EOS, no future-label leakage. Output-local and fixed-error-broadcast candidate learning are distinct from cross-layer BPTT/BP fitted evaluator probes. Circuit-local is not automatically sparse topology. Warmup must follow the **actual locked experiment**; V179–V181 explicitly prescribed no added warmup, while native Full151 lineages often used the owner's 1/16 schedule. Never silently impose either across unrelated branches.

## C. Recursive iteration protocol: the unit is a decision, not a version number

### S0 — evidence reconstruction (read-only)

Collect:
- `branch@commit` and the matching version registry, the most recent REPORT, STATUS, SOURCE_LOCK and underlying code/metrics. Check all active branches; **Claude V169 ≠ Codex V169**, and a named design is not evidence of an executed run.
- One *operative* learner and comparator. Full151 associative memory, V167/V169 local predictor and V164/V178 dense BPTT are **different lineages**; only the local predictor has bounded own-lifetime evidence relevant to V179. Do not force them into one sequential integration pipeline by default.
- A capability vector `(acquisition, E1, retention, correction, reuse, E3)` and its status at actual final checkpoint; distinguish probe, teacher-forced model-head scoring, and autonomous native generation.
- Exact experimental cost (distinct records, repeated record presentations, byte/EOS sites, optimizer steps, CPU, RSS, checkpoint/storage, control branches). Technical and read-only probes count in CPU but are not called trained lives.

**Decision test:** Name the best falsifier using source math or saved artifacts. If the proposal cannot say what a negative answer would retire, stop instead of versioning another speculative diagnostic. Prefer source inspection/finite-difference checks over 5,000-record experiments when these settle the question.

### S1 — lock the minimal experiment

Write a dated design (and machine spec if appropriate): one mechanistic hypothesis, one baseline, a matched causal control, immutable data/world split, no target in state before prediction, exact own-output endpoints, possible outcomes **including failure**, resource admission, cross-arm fairness and no adaptive rescue. Preserve original design/results on correction; revisions get separate dated records or a new version if the scientific question/architecture changes.

Do **not** make a chain of unrelated capacity tasks a prerequisite. Require a BPTT reference only to test a local credit rule *on the same forward architecture*, not to veto a different architecture because the ungated intermediate failed. Do not demand Full151 head bridging before testing an independent local learner.

Record the user's approval for whatever they actually requested. Design-writing is not permission to run, to exceed the historic default limits, to adopt or to alter authorizations. Separate “proposed” and “authorized.”

### C0 — capability and data falsifiers before learner training

Test source/data hashes, authentic EOS and token-site definitions, target conditioning, task determinacy, complete-response feasibility, numerical sign, expected active-neuron/fan-in, spectral/initialization pathologies and baseline competitor. A ceiling that fails on an undertrained architecture does **not** prove the task unlearnable; a synthetically easy task does **not** establish general lifelong learning. Fix the model-vs-task question first.

**History to avoid repeating:** V177 257-class pilot mostly learned output alphabet, not lag-eight structure; V178 dense→sparse transition changed 65,536 to 4,096 recurrent edges; V179 full-prose exact matching with only 32 generated tokens was impossible for many records. Use those observations to prevent concrete bad designs, not as an excuse to add endless gates.

### C1 — technical and mini-scale admission

Require a measured model+experiment ledger, not just the trainable parameter count:

| Scope | Must count |
|---|---|
| Single deployable candidate | sensory birth/wiring, learned weights, fixed feedback, recurrent/transient states, output classes, mutable optimizers if required to continue learning, byte/EOS interface |
| Experimental training | all arms/worlds/forks, distinct records, repeated target sites, batch forward/backward/eligibility work, actual CPU and peak RSS |
| Reproducibility | all checkpoint states (weights, moments/clocks, RNG), read-only prediction panels, manifests, hashes, log outputs, reserve margins |

For **V181 specifically**, inspect `minifly_v181_mini_local_plasticity_design_20261102/PARAMETER_AND_RESOURCE_CENSUS.json`: 37,313 trainable scalars planned, ≤40,000 declared; one 5,177-KC sensory circuit, ≤8 MiB single-deployed inference state, ≤1 MiB full mutable checkpoint, ≤256 MiB outputs, ≤5,000 physical records, ≤600k target apps, ≤900 CPU seconds. **These are proposal-level V181 gates, not a global theorem about fly size, and sensor memory/parity were NOT MEASURED in the design.** Do not change caps, omit saved states or hide pretrained features to pass admission. If fail, publish `RESOURCE_NOT_ADMITTED` without training or automatic version escalation.

Numerically check local synaptic gradients against explicit formulas, output error broadcast, real H/T applied-step movement *after Adam preconditioning* and optimizer/RNG continuation; the sham must match the relevant update energy at a common state, not claim cumulative equal energy after paths diverge. When a skill's rule relies on several records, verify true independent record assignment and tail-group fallback. Do not confuse NumPy/array or serialization technical fixtures with scientific outcomes.

### C2 — execute the complete real-life development history

One learner's `A→B→corrected A` state includes weights and optimizer state; no phase-conditioned reset, chosen validation-best restoration, external answerer, or model knowledge of old/new item labels. Measure all four **even when acquisition is weak**; only invalid technical/data/resource execution may stop the roster. Compare against B-only/no-prior-teaching and same-state/same-dose controls when the claim requires them.

**Efficacy readings:**
- same-checkpoint train versus held (avoid comparing online train during updates with final held);
- own 257-way payload NLL/top1 and **true EOS** separately;
- A_OLD, corrected A' with unchanged label **and exact same full FE/KC code**, corrected A' with changed KC code, and corrected target sites;
- B loss before and after corrected A; old/new log-odds versus NOEDIT; both retention and plasticity;
- teacher-forced prediction **versus** honest free-generation, and matched unigram/strong counting competitors.

Unseen literary sentence **exact reproduction is not a valid generic success criterion**, especially if the generation cap is shorter than the required line. Use length-feasible denominators and EOS calibration, reserving exact whole-output gates for deterministic tasks. V179's edited common two-byte next-symbol statistics show adaptation, **not semantic factual correction**.

### C3 — prospective guard, independent criticism, and no post-hoc rescue

Once the source/algorithm is frozen, evaluate on a new predeclared world. Previously used texts, repeated seeded worlds, prompt/held data used for engineer decisions and hidden test exposure **cannot** be relabeled sealed confirmation. Keep two-world evidence as development unless a genuinely sealed protocol is established. Cross-agent review must execute checks and attempt to break controls; a same-agent summary is not independent review. A missing checker cannot issue formal seal/attestations.

If development is poor, still publish the negative. No raising dataset size, changing output policy, adding warmup or choosing a better intermediate pass after viewing held scores. A scientifically necessary fix after outcomes gets its own prospectively locked comparison and all earlier negative receipts remain.

### S7 — close the loop (the recursive improvement step)

Every iteration emits exactly one **decision record**:

```text
Anchor: repo@commit + candidate/baseline lineage.
Question: one specific hypothesized limiting mechanism.
Falsifier: comparison whose positive and negative result change a decision.
What ran: source/technical/dev/guard, physical costs and status.
Result: full signed outcome vector, not best checkpoint only.
Lesson: what is learned, what is NOT demonstrated, alternative explanations.
Disposition: RETAIN / REJECT EXACT PACKAGE / INVESTIGATE ONE CAUSE / STOP.
Next action: one bounded test with explicit dependencies, OR no test.
```

Add new durable findings only with source links and caveats; use append-only entries (or a **proposed** entry if the historical checker/activation mechanism is unavailable). Never rewrite old findings, versions, status pointers, adjudications or the owner's verbatim grant. Version increments, number of memos, cycles and check flags are not progress metrics. Review the decision itself for Goodhart drift: *did we merely optimize the previous proxy?*

## D. Current evidence shortcuts and known traps

Read the detailed [dated evidence ledger](references/CURRENT_EVIDENCE_20261102.md). Key anchors:

- **V164:** dense BPTT can solve a synthetic lag-eight delayed-copy reference (1.152 bits), COLA did not recover the BPTT–TBPTT temporal gain; do not claim BPTT works in every new graph.
- **V167/V169:** locally trained representation is informative; native own top still loses to tuned counting, and prior A may help B even without observed forgetting. A fitted downstream probe is not an own learned head.
- **V172:** changing error-corrective writer substantially improved native taught recall, but NEW learning still erased many OLD items. Writer improvement ≠ proven reusable cognition.
- **V173/V177:** required positive references failed; **local candidate learning was not executed**. Do not count these as negative efficacy trials of the local rule.
- **V175/V176:** local inhibition screens and native feature bridge did not establish robust operative gains; acknowledge positive diagnostics but no core adoption.
- **V178:** the historic dense reference reproduced; one sparse ungated architecture was poorer; its gated continuation wasn't tested. Sparse ≠ local.
- **V179:** two opened natural-text lifetimes; acquisition and retention inconsistent, own predictions sometimes reuse A and adapt to synthetic local edits. All six B arms fit their training observations better while held B worsened; H/T freeze improves late B yet impairs subsequent correction retention.
- **V180:** 4.23M trainable weights and 16 lives; technical CGLP checks PASS, **scientific training NOT_RUN**, resource admission **failed**. Do not imply CGLP efficacy.
- **V181:** 37,313-param mini learner is **DESIGN_ONLY**, proposed sensor footprint/sparse kernel not measured, no efficacy or adoption. Check branch for later Codex work before using this as current status.

## E. Companion skills and original work

- [minifly-governance](../minifly-governance/SKILL.md): protected edits, owner decisions, reviewer independence and skill meta-learning.
- [minifly-claims-knowledge](../minifly-claims-knowledge/SKILL.md): claim evidence, status taxonomy, provenance, proposed findings, historical KB.
- [minifly-independent-review](../minifly-independent-review/SKILL.md): executed hostile checks, cheap shortcuts, source/score reproduction and review debt.
- [minifly-e3-design](../minifly-e3-design/SKILL.md): deterministic unit reuse, strict own byte/EOS E3 tests, causal controls and calibration.
- [Claude's original MRK-1 research-loop](references/ORIGINAL_CLAUDE_SKILL_20261102.md): preserves detailed proposed checker/typed-cycle behavior, **dormant until runnable and independently verified**.
- [Claude historical KB](../../../PROJECT_DOCUMENTS/KB/README.md): historical/provisional entries; this merge does **not** activate findings or manufacture owner decisions.
