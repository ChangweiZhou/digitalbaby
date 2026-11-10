---
name: causal-experiment-loop
description: Design, execute, audit, and interpret falsifiable ML or computational-science experiments using preregistered predictions, matched causal controls, representation and gradient diagnostics, leakage checks, compute budgets, and explicit stop rules. Use when reviewing a research repo, debugging failed learning, choosing the next ablation, reconciling conflicting reports, or converting experiment outcomes into an evidence ledger. Do not use for literature summaries without experiments or unconstrained architecture brainstorming.
---

# Causal experiment loop

Use a **diagnosis-first, one-question-per-experiment** workflow. The objective is not to generate more model versions or celebrate every result. The objective is to reduce uncertainty about a fixed target capability at the lowest justified experimental cost.

This is a reusable synthesis of practices observed in the MiniFly / Fable research loop, **not** an official skill or a claim that Fable authored these instructions. Read `references/minifly-evidence.md` only when a worked example is useful. For a new project, translate the methods, not the MiniFly model or thresholds.

## Deliverable contract

For each request, produce the smallest useful set of:

1. **Evidence snapshot:** exact repository ref, date, artifact links, what has run vs merely been proposed, known invalidated results.
2. **One causal question:** target failure, competing hypotheses, and the next discriminating observation.
3. **Preregistered experiment contract:** metric, seeds, arms, controlled differences, data provenance, acceptance and failure thresholds, costs, stop conditions.
4. **Execution/audit receipts:** code/input/checkpoint hashes, per-seed metrics and traces, test contamination audits, resource log.
5. **Verdict:** `SUPPORTED`, `REFUTED`, `MIXED`, `INCONCLUSIVE`, or `INVALIDATED`, followed by the next *necessary* decision.

Do not assert an experiment is complete because a commit says "prepared," "launched," or "partial". Do not treat an instruction file or dry-run receipt as a scientific result.

## Procedure

### 0. Lock the user's scientific contract

- Record the target capability, allowed learning/information routes, learning stages (development versus life), resource ceiling, and evaluation population.
- Separate **allowed** from **demonstrated**. E.g., a signed multi-layer feedback relay may not meet a locality constraint that disallows full-circuit signal propagation; a BP ceiling does not establish locality.
- Do not silently strengthen or relax goals, replace an architecture, reinterpret a metric, or adopt an agent-authored roadmap as a user decision.
- Ask about truly essential unresolved normative constraints. Otherwise continue with conditional interpretations and mark the disputed constraint in the ledger.

### 1. Freeze a source-of-truth snapshot

- Read newest commits and full relevant reports, predictions, receipts, code paths, comparator settings and errata; never rely on a review summary alone.
- Name the **exact commit** used. Distinguish completed experiments, unfinished jobs, pending approvals, and design-only branches.
- Maintain a capability ledger: `SUPPORTED | PARTIAL | FAILED | UNTESTED | INVALIDATED`, with a provenance link and the *exact* population tested.
- Never average success from different architectures, curricula, machines, or held panels into the "same learner" without clear justification.

### 2. Locate one bottleneck before proposing new mechanisms

For a failure, make a short causal map spanning:

`input representation -> objective/target provenance -> model computational expressivity -> gradient/learning-signal reachability -> optimizer/update -> data/replay distribution -> evaluation`.

Ask in order:

1. Is there enough information and a nonleaking test? Audit complete examples **and partial steps/intermediates** across the entire learning history.
2. Can a strong reference (e.g. BPTT/BP) solve the **same task and inputs**? If not, investigate representation, task construction, or missing computation first.
3. Can the candidate solve it under oracle/full-coverage observations? If yes, isolate the data/sampling issue; if not, examine update rule and architecture.
4. Can reference updates solve it on the candidate's **exact data pipeline** (matched-control or M arm)? If yes while candidate fails, suspect a rule-by-data interaction, not merely inadequate storage.
5. Is a recently introduced failure a software bug? Inspect zero/nonfinite gradients, parameter groups, optimizer decay on unreachable weights, logits and hidden-state magnitudes, detach paths, eligibility trace parity, and frozen-parameter behavior **before** inventing a new rule.

A reference with a different stream, supervision, replay ratio, decay, optimizer or compute is an **upper bound/diagnostic**, not a matched causal control. Name which.

### 3. Create and lock a minimal experiment contract

Write predictions **before launch**, in a versioned file/commit. Use `assets/experiment_contract.example.json` as a starting point and optionally run `scripts/lint_contract.py`.

Required:
- One primary hypothesis and a plausible competing hypothesis, each with a falsifiable observation.
- Exactly specified arms, baseline, allowed differences, seeds, metrics, target populations, comparisons, sample budget and cost cap.
- A matched arm (candidate versus reference on identical data pipeline), an attainable ceiling, and a negative/shortcut control as appropriate. Not all three are needed if a smaller experiment is decisive.
- Named hard subsets and tests of shortcut leakage; report subgroup gates **alongside**, never substituted for, the original preregistered primary gate.
- Explicit `halt_if` for violated invariants; schedule machine-wide mutual exclusion when several agents share one computer.
- Predicted effect size, direction, tolerance, and how to count multiple seeds. A near miss stays a miss.

For architecture-vs-rule questions, use a factorial or matched design rather than changing feedback, trace, memory, initialization and schedule together. A multi-change rescue is a **candidate**, not a causal attribution: follow with one-change dismantling.

### 4. Qualification before science

Use a **cheap dry run** to test mechanics only; label it `TECHNICAL`, not `SCIENTIFIC`:
- Check shapes, finite losses, versioned input/seed/checkpoint hashes, determinism within the target machine.
- Compare local/trace derivatives against a defined exact computation when meaningful (absolute/relative error and tested tensors).
- Inspect **every used input-token class and each plastic layer** for gradient reachability and update magnitude; detect a zero-gradient row being decayed, stale embeddings, or frozen weights changing.
- Check train/held split, task-generator seeds, operand/intermediate collisions, answer labels, and evaluation joins.
- Measure CPU-seconds, peak RAM, replay bytes/index bytes, effective unique examples and repeated presentations.

Do not promote qualification accuracy to a scientific finding. A severe invariant breach stops the run. Preserve logs and annotate the receipt as invalid; fix the defect, preregister corrected design, and run fresh.

### 5. Run the minimum decisive science

- Use locked predictions and immutable data/checkpoint lineage. One shared machine -> one explicit lock owner; never overlap expensive jobs unintentionally.
- Keep per-seed and per-arm receipts, intermediate checkpoint scores, worst subgroup accuracy, failure trajectories, data/replay exposure, code and data hashes, cost.
- Protect active live recurrent state from replay workspace; explicitly verify across-call state continuity when claimed.
- No selective reruns, seed exclusions, post-hoc threshold widening, or modifying a running arm without a logged amendment.
- Cross-machine parity is not assumed; record hardware and library versions. A byte-identical rerun is replication of a computation, **not a new independent seed**.

### 6. Audit the result adversarially

1. Recompute pass/fail from raw receipts, not prose. Preserve original clauses separately.
2. Split family averages by causal difficulty: hard pairs, shortcut-bearing pairs, intermediate correctness, true unseen operators and history-wide exposure.
3. Compare acquisition, retention and later recovery. "Never learned" differs from "learned then forgot".
4. Distinguish **stored bytes** from true memory usage (optimizer, model, eligibility, metadata, transient pending arrays, index and replay).
5. Check parameter/gradient traces before interpreting collapse as an architectural law.
6. State which inferences are causal, which merely correlate, and which have confounded interventions.
7. Keep unsuccessful experiments and failed predictions in the public/internal record. If a critical confound emerges, mark impacted causal readings `INVALIDATED`; do not erase source receipts.

### 7. Choose the next experiment by information gain

Prefer the cheapest experiment that separates the remaining top two causes. Prioritize a **mechanism repair** or **architecture repair** only if the evidence identifies that bottleneck. Do not spawn many unrelated suffixes or introduce a mechanism catalogue.

Common high-value discriminators:
- Freeze only embeddings vs freeze entire early module; otherwise identical run.
- Exact versus signed feedback at fixed eligibility, then extended versus baseline trace at fixed feedback.
- BP on learner's exact replay pipeline vs BP on a richer joint stream.
- Same network, same total memory and presentations, reservoir sampling vs oracle data.
- Windowed vs windowless with matched compute, sufficient development accuracy, and two-step hard-pair scores.

If none can discriminate, report `INCONCLUSIVE` and the missing measurement rather than rebranding failure as a success.

## Standard final response format

Use short connected prose with: **Current verdict; what changed; strongest causal evidence; invalidated/uncertain claims; next single discriminating test; cost/approval status; links to source receipts**. Include a compact per-arm table when it materially helps. Do not declare "breakthrough" solely from a new file, partial metrics, or a better descriptive summary.

## Safety and authority boundaries

- Do not run expensive experiments, push code, launch overnight jobs, or rewrite benchmark gates without the user's authorization and available execution tools.
- Do not invent GitHub links, results, clock times, or runtimes. Cite actual sources.
- If the user's requested locality / learning-rule constraints are unsettled, obtain a ruling before certifying a mechanism as compliant. Continue descriptive audits without claiming compliance.
- Keep the skill reusable across ML, robotics, causal inference and formal-science experiments; task-specific thresholds belong in a contract, not in this file.