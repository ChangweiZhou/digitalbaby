---
name: minifly-claims-knowledge
description: "Use whenever reporting or updating MiniFly results, verdicts, design/NOT_RUN status, version registers, lessons, findings, research memos, experiment comparisons or adoption pointers. Requires branch@commit provenance, own-output evidence types, source-first historical corrections, append-only findings and explicit limits on independent confirmation."
---

# MiniFly claims and knowledge — current entry point, 2026-11-02

**Start with:** [current evidence and source pointers](../minifly-research-loop/references/CURRENT_EVIDENCE_20261102.md), the *live* `PROJECT_DOCUMENTS/MINIFLY_VERSION_INDEX.json` and [research loop](../minifly-research-loop/SKILL.md). The [original Claude MRK-1 claims skill](references/ORIGINAL_CLAUDE_SKILL_20261102.md) and [claim/reference grammar](references/CLAIM_AND_POINTERS.md) / [knowledge ledger rules](references/KNOWLEDGE_LEDGER.md) remain preserved; their automated `MFC claim` command is **unavailable**, not an implied PASS.

## 1. Evidence vocabulary

Distinguish **DESIGN_ONLY**, **NOT_RUN**, **TECHNICAL_PASS**, **RESOURCE_NOT_ADMITTED**, **CANNOT_TEST_CLAIM_AS_SPECIFIED**, **INVALID**, **DEV_SCREEN_FAILED**, **BOUNDED_DEV_POSITIVE**, **CONFIRMATION_NOT_OPENED**, and **ADOPTED**. A whole version may have a positive technical subtest and a failed scientific primary simultaneously (V180). Preserve both.

An experiment is identified by **repo + branch + immutable commit + version + arm/world/checkpoint**, not by number alone. Claude and Codex both used overlapping numbers in separate histories. When a report references another branch, cite that branch explicitly. Prefer locked original `REPORT.md`, result JSON/source, then independent audit. A generated memo, README or human summary is **not primary data**. Absence of public raw checkpoint/outputs is explicitly `NOT_EXTERNALLY_RAW_REPRODUCED`.

Evidence hierarchy:
- `E0` source/math/numerical feasibility and data admission. Fits, source hashes and finite differences cannot support behavior claims.
- `PREDICTIVE_DEV` the model's own teacher-forced 257-class probabilities; distinguish probe, native and BPTT instrument.
- `E1` complete own-generated taught byte sequence plus *real own EOS* on deterministic tasks.
- `E2` retained **still-useful** learned knowledge while acquiring newer information; obsolete corrections excluded from protected set.
- `E3` never-taught causal relation transfer beyond matched lookup/frequency/replay/shortcut controls on genuine held items; own exact bytes+EOS, no targeted teaching.
- `CONFIRMED` only with a separately qualified independent/sealed procedure; `ADOPTED` only when the owner actually authorizes the core change.

**Do not** label improved probe NLL, raw-byte input reconstruction, cloned evaluator, a BPTT reference or the full-prose matching of a stochastic text as E1/E2/E3.

## 2. Current anchored outcomes (not the live latest without rescan)

V164: dense BPTT delayed copy 1.152 bits, COLA 0% temporal improvement. V167: learned local features transferred, mostly measured with an evaluator probe. V169: A-pretrained own B predictions 3.443 versus 3.917 B-only; tuned counting 2.819, no forgetting seen in that two-author setting. V172: error-corrective native writer improved taught recall, but NEW learning still overwrote OLD. V173: positive reference unqualified and local candidate NOT_RUN.

V175 failed its inhibitory screen; V176 gained probe signals without confirmed native complete output; V177 BP reference failed and BS/BR NOT_RUN; V178 dense BPTT reproduced but sparse ungated underperformed (gated not tested); V179 complete-life development had mixed acquisition/retention and positive statistical edit adaptation, B train improved while held worsened; V180 technical CGLP PASS but scientific **NOT_RUN** and resource **NOT_ADMITTED**; V181 **DESIGN_ONLY**, 37,313 proposed learned scalars, mini memory still unmeasured. Read exact commits/qualifiers in the linked dated evidence file.

The latest design/allocation can change during Codex's active work. Rescan `branches`, `commits`, version index and the exact latest delivery before producing a “current” claim. Do not infer that a version was run because its DESIGN or a technical report exists.

## 3. Append-only knowledge integration

Claude's `PROJECT_DOCUMENTS/KB` is imported as a **historical/provisional** ledger, including old FM failure classes. Read its integration status and the original findings' latest narrowing/correction rows. Do not edit or delete original lines to make a former claim current.

When the formal MRK-1 ledger writer is unavailable, author a **proposed dated** memo/row with exact evidence references and `activation_status=PROPOSED_NOT_ACTIVATED`. Do not imitate checker-generated IDs or provenance, approve your own finding, write a fabricated USER_DECISIONS entry or silently change `CONSTRAINTS.json`. A subsequent independent reviewer can review activation/errata once the process is actually runnable.

Every published new version records: source SHA, status, primary signed outcomes, record and target counts (unique vs repeated), CPU, peak memory, model learned-parameter census, actual output EOS, common-source comparator, changed controls, data exposure, accepted caveats, and adoption (`false` unless demonstrably changed).

## 4. Recursion and stopping

The knowledge loop only generates another experiment after answering the question **“what decision did this result change?”** If none, close the finding with an unresolved limitation and **stop**. For negative results distinguish architecture package failure, improper task design, source parity failure, insufficient training at a declared budget, failed evidence gate and not-run. A source-frozen negative result is useful scientific information without becoming an automatic new version.

When editing documentation, record both before/after SHA and a narrow erratum instead of rewriting the historical report. Pointer changes and adoption require the owner's explicit authorization, not the editing agent's enthusiasm.

[Legacy detailed claim machinery](references/ORIGINAL_CLAUDE_SKILL_20261102.md) remains for future checker restoration, not an executable guarantee.
