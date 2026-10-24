# History analyses of 2026-10-24 (inputs, not verified facts)

These files are agent-generated analyses written on 2026-10-24 by Claude Code workflows. The MRK-1 research-loop kit
(`tools/minifly_check/SPEC.md`) and the seed rows of the knowledge-base ledgers in `PROJECT_DOCUMENTS/KB/` were built
from them. They are **inputs, in effect training data: they are not verified facts.** Do not cite them as evidence for a
result. Cite the primary file they point to, after checking it yourself.

| File | What it is |
|---|---|
| `UNDERSTAND_DIGEST.md` | Digest of workflow 1: 8 lens readers plus 1 synthesis critic over the post-Full151 MiniFly history. Defines the failure modes FM-01..FM-30 and durable unit facts UF-01..UF-29. |
| `understand1.json` | Full structured output of workflow 1 (`lenses[]` with findings and paths; `synthesis`). |
| `ARCHIVE_DIGEST.md` | Digest of workflow 2: 4 readers plus 1 merge agent over the pre-Full151 archive in `PROJECT_DOCUMENTS/PRE_FULL151_DESIGNS_RESULTS_20261023` (cited as `ARCH/`). Defines the archive failure classes FC01..FC19 and the archive unit facts. |
| `understand_archive.json` | Full structured output of workflow 2 (`readers[]`, `merged`). |
| `design_panel.json` | The design judge panel: three candidate kit designs (minimal enforcement, separation of duties, loop as learner), two judges (practicality, adversarial) and the synthesized final spec with its work packages. |

How far to trust them:
- They were only partly spot-checked. Claims marked SPOT-CHECKED were checked against the repository or git by the
  synthesizing agent. Everything else rests on reader-level reading and was not re-verified.
- In the archive merge, about one fifth of the reader claims that were spot-checked needed correction (see
  `ARCHIVE_DIGEST.md`, 'Review practices that failed', corrections 1-7). Expect a similar error rate in claims nobody
  checked.
- Some numbers depend on when they were read: commit counts, and branch positions such as `origin/charming-head` at
  `f33d8dee`. Pin a commit when you re-check them.
- Line numbers can drift. While seeding the ledgers, the coarse-binary caveat that the understand digest cites as
  `FULL151 MEMO.md:23` was found at line 21. UF-02 is labelled UNVERIFIABLE_IN_REPO, yet report-level copies of its
  numbers exist under `minifly_b/package/DOCS/BYTE_CORE_V9_9A/` and `BYTE_CORE_V9_9B_RANK/`. Both points are recorded
  in `FINDINGS.jsonl` for reconciliation.
- For the checker's held-out evaluation, the archive digest counts as training data (spec 12.3, item 8). Fixtures
  derived from it never count as held-out.

These files are frozen. Corrections go into the KB ledgers as appended rows, for example a narrowing or retraction of
a seed finding, never as edits here.
