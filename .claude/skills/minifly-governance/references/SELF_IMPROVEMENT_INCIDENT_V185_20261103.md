# Recursive self-improvement incident: wrong skill routing and unsafe V185 admission

**Date:** 2026-11-03  
**Status:** \`PROPOSED_NOT_VERIFIED\` — governance *proposal*, not an activated historical KB incident, an MFC checker result or a scientific experiment.  
**Origin:** user correction, “I asked you to activate the recursive self improvement skill, not the research skill. Your V185 does not work - a waste of tokens.”  
**Base:** \`ChangweiZhou/digitalbaby@129694faf16b8bc662d5b6cdcd8361355eb8fd8c\` (\`codex/minifly-consolidated-20261023\`).

## Incident evidence

1. **Wrong activity.** After V184's negative result, the assistant invoked \`.claude/skills/minifly-research-loop/SKILL.md\`, designed a new 33,153-learnable-scalar architecture and committed V185 design at [617bc8605d8d656d5c5a22e98388b5c039e026a8](https://github.com/ChangweiZhou/digitalbaby/commit/617bc8605d8d656d5c5a22e98388b5c039e026a8). That was the wrong class of action: the owner's intended skill was the recursive *process self-improvement* procedure documented in \`PROJECT_DOCUMENTS/KB/RESEARCH_LOOP_OVERVIEW.md\` section 4 and \`.claude/skills/minifly-governance/SKILL.md\`. Merely “doing another version” is not a process improvement.
2. **Failed C0 admission, not mechanism efficacy.** V185 qualification commit [129694faf16b8bc662d5b6cdcd8361355eb8fd8c](https://github.com/ChangweiZhou/digitalbaby/commit/129694faf16b8bc662d5b6cdcd8361355eb8fd8c), \`minifly_v185_dual_timescale_local_memory_dev_20261103/REPORT.md\`, \`STATUS.json\`, and \`exercise_first_attempt.py\`. The proposed 80/80 fixed-feature readout witness used a multiclass linear-programming formulation; the first attempt reached **3,798,089,728 bytes RSS** against **1,073,741,824 bytes cap**, returned no feasible witness before timeout, and consumed 69.016 CPU seconds. A second attempt was interrupted when the first attempt's memory receipt was reviewed. Its CPU/RSS are not certified. \`REPRESENTATION_CEILING_UNRESOLVED\` and \`RESOURCE_NOT_ADMITTED\`; **zero scientific model training**. The learner's conflict-aware plasticity was not experimentally falsified.
3. **Cheap falsifier that was missed.** The C0 source forms one inequality per labeled site and alternative class, expands per-class design blocks, and calls HiGHS without a measured/pre-admitted workspace or enforced live memory guard. There are 240 labeled sites and up to 257 output classes. A worst-case dense 257-class construction at 240×256 inequalities and 16,705 output variables would require approximately **8.21 decimal GB for float64 entries alone**, excluding solver workspace (the actual code reduces to observed classes and uses sparse matrices, so 8.21 GB is a *naive worst-case warning*, NOT a claimed measured allocation). This should have triggered an early resource-bound redesign rather than an expensive attempt.
4. **Self-improvement skill readiness.** The original MRK-1 kit is not fully executable in the integrated branch: \`tools/minifly_check/minifly_check.py\` and multiple workers/templates are missing; \`PROJECT_DOCUMENTS/KB/INTEGRATION_STATUS_20261102.md\` records missing approval/cross-family verification. A proposed prose rule is not a functioning blocking check.

## Root causes

- **Intent routing failure:** the user referred to a particular *meta* skill, but the assistant defaulted to the model-research workflow and allocated a scientific version as if “iteration” always meant “new architecture.”
- **Admission blind spot:** model mini-scale accounting was written down, but the evaluator's own solver work/RSS was not bounded before its first expensive allocation. A numeric diagnostic was turned into an unqualified feasibility check.
- **Optimization against activity:** new design files and a version number displaced improvement in delivered capability or research-process reliability.

## One bounded governance repair

Propose adding a **two-stage pre-action gate** in \`minifly-governance/SKILL.md\`:

1. Route by **purpose**, not by the word “iterate.” A recursive self-improvement request enters the incident/check/review process; it does not allocate a model version, write a model design or run learning. Model-design intent must be separately explicit.
2. For any subsequent computational test or solver, require **pre-allocation resource-admission arithmetic and a runtime enforcement plan**. Reject an unsafe or unbounded C0 plan *before* launching it; preserve a design-only pathway for valid proposals.

This is prospective, text-only and classified **tighten/neutral proposal** pending exact diff audit. It does not retroactively relabel V185, change frozen V184 results or loosen owner resource limits. No live checker patch, hidden canary or formal ledger record is claimed.

## Should-fail and should-pass regressions

\`ROUTING_REGRESSIONS_V185_20261103.json\` contains cases with distinct phrasings to expose overfitting. The **failure fixtures** require: (a) an explicit recursive self-improvement request to produce *zero* new model versions/experiments, and (b) a solver proposal whose lower-bound/worst-case budget exceeds the configured cap to be stopped before expensive allocation. The **legitimate fixtures** permit: (a) a specifically requested *model design* to remain design-only and (b) a memory-bounded streamed C0 witness with measured enforcement to proceed through normal review. These are **proposed evaluation cases**, not hidden cases and not executed automated checks.

## Stop and independent-review boundary

- No V186 allocated, no model run, no change to existing data, status, original V185 designs or active registry on the science branch from this proposal.
- The historical MFC CLI cannot certify \`MFC governance\`, \`MFC heldout\`, \`MFC selftest\` or official incident closure.
- This proposal must be challenged by a **different agent family** on a new, non-isomorphic task before integration; specifically examine false positives (valid designs wrongly blocked) and missed renamed meta requests. No self-issued \`PASS\` or approval.
- Until then the practical disposition for V185 is **STOP_AT_RESOURCE_ADMISSION** and \`REPRESENTATION_CEILING_UNRESOLVED\`; do not create V186 as a substitute for repairing the loop.
