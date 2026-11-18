# Proposed update of the repository's entry points (18 November 2026, 15:39 UTC)

**What this is.** A draft by the outside reviewer, written at Mr. Xu's request to implement what it can of the outside advice memo (`PROJECT_DOCUMENTS/MINIFLY_OUTSIDE_ADVICE_20261118/ADVICE_MEMO_20261118.md` on the branch whose name ends in `friendly-darwin-8mmhfm`, section 5.5, last point). It changes no file. `README.md` and `AGENTS.md` are on the default branch, to which the reviewer does not push; `AGENTS.md` is a protected file, whose change the owner authorises. The coordinator applies the text below, or another, when the owner agrees.

**The fault, checked by the reviewer on the default branch at 392b3eca.** `README.md` opens with "MiniFly current research navigation — 2 November 2026" and names V181 as the latest allocated experiment; its last change of content is of 2 November. `AGENTS.md` (15 lines, last changed on 2 November) sends a session to `.claude/skills/minifly-research-loop/references/CURRENT_EVIDENCE_20261102.md` and to `PROJECT_DOCUMENTS/MINIFLY_VERSION_INDEX.json`. A session that starts from either is sent to the state of two weeks ago; the outside session reports that this happened to it. The work since 6 November (the experiments E11 to E60) is recorded elsewhere.

## 1. Proposed text to stand at the head of `README.md`, above the present first heading

> ## Where the project stands (updated 18 November 2026)
>
> The text below this section describes the state of 2 November 2026 (V181 and earlier) and is kept as history. The live state is in four places, in this order:
>
> 1. **The review for Mr. Zhou**, `MINIFLY_PAST_WORK_REVIEW_FOR_MR_ZHOU_20261106.md`: sections 1 to 8 give the programme, and section 9 is its running record (9.161 on 18 November).
> 2. **The coordinator's decisions**, `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/COORDINATOR_DECISIONS_20261118.md`: every decision is complete in that file and is the channel by which it reaches the reviewer and the workers.
> 3. **The reviewer's memo to the coordinator**, `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/MEMO_TO_COORDINATOR_LEARNED_STRUCTURE_20261115.md`: section 1 lists what is asked of the coordinator; section 33 holds the results verified from receipts and weights on 17 and 18 November, each with what it does and does not show.
> 4. **The plan and the log**, `PLAN.md` and `LOG.md` in `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/`: the ordered list of work, and every message, decision and correction with its time.
>
> The requirements that the present experiments serve are Mr. Xu's (15 to 18 November): long-term memory that forms in the course of learning and not by an act of the program; a learner that may be slower at first and is to reach at least the generalisation of back-propagation at equal computation, perhaps from fewer samples; and learning that is local, each unit receiving only over its own sparse connections. The experiments of this line are in `MOD97_LOCAL_RULE_20261106/`; each has a brief, receipts, a report and, where it is verified, a verification document with a script in `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/`. Results are at the level of a first measurement on one to three seeds unless a document says otherwise; nothing has been confirmed on tasks sealed in advance.

## 2. Proposed addition to `AGENTS.md` (protected; the owner's authorisation is needed), as its first item

> **Start here (18 November 2026):** the evidence snapshot of 2 November linked below is historical. Before any work read the section "Where the project stands" at the head of `README.md` and the four documents it names; the newest of them wins where they differ from this file's links.

## 3. What is not proposed

No change to `PROJECT_DOCUMENTS/MINIFLY_VERSION_INDEX.json`: the experiments since 6 November are numbered E11 to E60 and are not in that index's scheme; whether they are entered there is the coordinator's decision. No change to any skill.
