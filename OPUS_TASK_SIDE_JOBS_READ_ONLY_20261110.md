# Three read-only side jobs for Claude Opus 5.5 workers (10 November 2026, 21:00 EST)

> **Status under the process rules of 10 November (review 9.82).** Read-only jobs: they may start without an audit of this brief. Each deliverable is audited like any other. **Commits:** one commit at completion; nothing partial on the default branch.

**Repository:** `ChangweiZhou/digitalbaby`, default branch `codex/minifly-consolidated-20261023`. One job per worker; say which in your first message. **None of these jobs trains anything, launches a life or a birth, or edits an existing file.** Each survives a container restart: work in steps and keep your notes in the deliverable as you go. Commit to the default branch if your session can push there (pull with `--rebase` first, never force), otherwise to the branch your session is given, named in your last message. No other branch.

A fourth read-only job is already briefed and takes priority if no one has started it: `OPUS_TASK_LITERATURE_CHECK_MULTI_TIMESCALE_20261110.md`.

---

## Job S1: an evidence ledger, with every number recomputed

**Why.** The review (`MINIFLY_PAST_WORK_REVIEW_FOR_MR_ZHOU_20261106.md`, sections 9.59 to 9.82) was written fast, and three rounds of errata now sit beside the statements they correct (9.75, 9.79, 9.81, 9.82). Two outside audits read source and receipts but ran nothing. Nobody has gone through the standing claims one by one against the receipts. The coordinating session has repeatedly mis-stated a figure by a point or a count by one.

**What to do.**
1. List every claim that currently stands in 9.59 to 9.82 and in `PROJECT_DOCUMENTS/COORDINATOR_NOTE_WHAT_WAS_LEARNED_20261110.md` section 2: a claim is a sentence with a number, a count of seeds, or a verdict on a prediction. Where a later section corrects an earlier one, the later one stands and the earlier is listed as superseded.
2. For each, find the receipts it rests on and **recompute the number from them** with a script you write (`MOD97_LOCAL_RULE_20261106/ledger_recompute.py`, reading JSON only). Check the receipt's final stage, its settings against what the claim says was run, and where present its completion record and panel hash.
3. Give each claim one status, in the form of `.agents/skills/causal-experiment-loop/assets/evidence_ledger.template.md`: supported, partial, failed, untested or invalidated, with the exact population it was measured on (which learner, which panel, which seeds, which machine).

**Deliver** `PROJECT_DOCUMENTS/EVIDENCE_LEDGER_20261111.md`: the ledger; then a list of every disagreement between the review and the receipts, however small, with both values; then statements that are superseded but not marked so where they stand; then claims you could not trace to a receipt. Do not edit the review. The first screen of the file should let Mr. Zhou see what holds.

**Rules.** A number you did not recompute is marked "not recomputed" and says why. Do not judge whether an experiment was well designed; the audits did that. Do not resolve a disagreement by choosing a side: show both. Expect a day of reading and a few CPU-minutes.

---

## Job S2: a source check for the delayed-observation memo and three remarks

**Why.** Three things now steer the target-test design (`PROJECT_DOCUMENTS/TARGET_TEST_DESIGN_DRAFT_20261110.md`) and none has had its sources opened by anyone on this project: the precedents section of Codex's memo (`PROJECT_DOCUMENTS/MINIFLY_E18_DELAYED_OBSERVATION_MEMO_20261110.md`); the audit session's remark, marked by it as from memory, that rodent hippocampus replays recent sequences in reverse and that this is the nearest biological analogue of a backward sweep over recent experience (`MINIFLY_REVIEW_20261109/NEXT_STEPS_NOTE_20261110.md`, 3.1); and the earlier finding that no study applies sign-symmetric feedback through time (`LIT_CHECK_20261110/LITERATURE_CHECK_LOCAL_CREDIT_IN_TIME.md`), which is now the project's working life rule and deserves a second, wider search.

**Needs web access.** If your session cannot open web pages, say so and stop.

**Questions.**
1. For each precedent the memo cites: does the source say what the memo says it says? Give the passage, table or figure.
2. Reverse replay: what is established about its timing, its content and its relation to reward or to learning; what is contested; and whether any source treats it as credit assignment. Report without arguing that it supports the project's rule.
3. Sign-symmetric or sign-concordant feedback in recurrent networks or through time: search again, including 2025 and 2026, and including work that uses other names (weight-sign feedback, sign-only transport, signed feedback alignment). Report what exists, at what scale, with what result against backprop through time; or that nothing was found and what was searched.
4. Prediction of later observations as a training target (multi-step prediction, successor-style targets, latent prediction of future states) under local or online rules: the two or three results most relevant to a learner that predicts every next byte, with their scale and their controls. In particular whether any uses a shuffled-target control like the one the memo proposes.

**Rules of evidence,** as in the two earlier literature briefs: every claim cites a source opened in this session, with authors, year, title, venue, link and location; unverified items go under "From memory, not verified" or are left out; never cite what you did not open; numbers as reported, with scale and seeds; say where sources disagree.

**Deliver** `PROJECT_DOCUMENTS/LIT_CHECK_20261110/SOURCE_CHECK_DELAYED_OBSERVATION.md`, at most about 2,000 words plus references, verdicts first; and one paragraph as `REVIEW_SECTION_9_84.md` in the same folder. It decides nothing.

---

## Job S3: where the repository's size comes from, and what could be done about it

**Why.** A cloud session was stopped on 10 November, possibly for storage. The repository holds model weights, logs and several hundred commits of partial receipts. Each new container clones it. Nobody has measured it.

**What to do, changing nothing.** With a full clone, measure and report:
1. Total size of the object store and of a checkout of the default branch; the same by top-level folder and by file type (`.pt`, `.json`, `.log`, other).
2. The fifty largest blobs in history, with path and the commit that added them; how much of the history is blobs no longer present at the head of any branch.
3. Growth by day since 1 November, and the share of commits that are partial receipts.
4. What a new worker actually needs: the size of a shallow clone, and of a shallow sparse clone of `MOD97_LOCAL_RULE_20261106/` without weights, and the exact commands for each. Test them.
5. Duplicates: identical blobs under different paths, and weights that no later job has read (say how you decided).

**Then set out options without choosing:** leave as is with shallow sparse clones for workers; stop committing weights except at completion and never for dry runs; move weights to release assets or large-file storage from now on; rewrite history. For each: what it saves, what it breaks, who has to act, and whether it can be undone. History rewriting and anything that changes what Mr. Zhou's other tools rely on are to be described, not recommended.

**Deliver** `PROJECT_DOCUMENTS/REPO_STORAGE_AUDIT_20261111.md`, numbers first. **Do not** delete, move, rewrite or force-push anything, on any branch.
