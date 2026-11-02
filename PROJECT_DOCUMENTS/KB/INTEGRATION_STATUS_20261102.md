# Claude MRK-1 historical knowledge base — integration notice

**Imported without editing existing historical rows** from `claude/vibrant-cerf-sje726@66925b04b92768e0f80272b5d44dfd69d5aeb206` (source tree `PROJECT_DOCUMENTS/KB` sha `81855ef32d19defebc1bba3eb5a572c03b533723`). This preserves failure classes, legacy evidence/incident/decision ledgers and 2026-10-24 source analyses.

## Status

- Historical `FINDINGS.jsonl` seed rows are **provisional**, not newly verified or automatically activated in the current consolidated project. A stale first-row claim cannot override a later narrowing or original experiment report.
- Historical `CONSTRAINTS.json` contains draft MRK-1 checker settings; it is **not** newly approved by the user merely because this directory is imported. Do not treat old UD-0001 as recorded owner approval; `USER_DECISIONS.jsonl` is preserved as originally committed.
- The original MRK-1 dispatcher and several mandatory checker modules/templates are missing from the Claude committed source. **No automated KB fold/claim/gate/ledger write was run** as part of this integration.
- The updated skills read the KB as **historical evidence**. New source-grounded findings should be written as **PROPOSED / REVIEW_PENDING** in a separate dated note until real ledger tooling and independent review are qualified; do not fabricate formal checker-generated IDs.
- The integrated [current evidence snapshot](../../.claude/skills/minifly-research-loop/references/CURRENT_EVIDENCE_20261102.md) points to V164–V181 evidence without rewriting old rows.

**Review status:** source path/tree integrity inspectable; cross-agent skill-governance review **pending**; no model training, official/confirmation, version allocation or adopted-core change performed by this KB import.
