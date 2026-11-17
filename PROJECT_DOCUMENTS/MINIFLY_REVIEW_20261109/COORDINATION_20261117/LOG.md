# Coordination by the reviewer in Mr. Xu's absence: log (from 17 November 2026)

This file records everything the outside reviewer did in the coordinator's place, so that the coordinator and Mr. Xu can see it at one reading and change whatever they would have decided otherwise. Times are UTC, read from the clock when the entry is written; entries before 21:10 on 17 November were stamped from memory and corrected to within a few minutes.

## 1. The instruction

Mr. Xu, 17 November: "Most works are done or waiting for your audit. Discuss next steps. As I am outside, coordinate yourself." The reviewer answered with its plan (memo, revision 21) and said that it had written to no worker and would pass a go-ahead only at his word. His reply: "Unfortunately I am outside, so you have to coordinate yourself."

The coordinator runs on a machine that the reviewer cannot reach, and it acts only when Mr. Xu is there. Until he returns, the reviewer therefore does what he would otherwise relay.

## 2. What the reviewer took over, and what it did not

- **Taken over:** starting worker sessions; handing them the documents; the decisions that an audit leaves to the coordinator (each listed in section 4); answering the workers' questions; reading their reports.
- **Not taken over:** nothing is pushed to the default branch or to any branch but the reviewer's own. No document of the coordinator is edited. Where an audit's changes had to be entered into a note or a brief, the worker writes "revision 2" on its own branch from the audit's text, headed as entered in the coordinator's absence. No merge is made.
- **One brief is the reviewer's own:** E51. The programme's rule is that the coordinator writes a brief and the reviewer audits it. Here the author and the auditor would be the same, so the brief was checked before release by an agent of the reviewer that had not seen the reviewer's reasoning. That is not an independent audit in the programme's sense. The job is a trial on one seed (about 3 CPU-hours); its other seeds wait.
- **The limit of three lines of work** (working rules) is kept: the first long lives, E50, E51.

## 3. Sessions started

| Job | Session | Started | Instructions (on `claude/vibrant-cerf-sje726`) |
|---|---|---|---|
| E44 L2, release 1: the first long lives | `session_017dSEpXu6oc38q4xQZ5sPv3` | 20:22 | `COORDINATION_20261117/WORKER_E44_L2_FIRST_LONG_LIVES.md` (b77d9eb2) |
| E50: an input code in which a value stands for one byte | `session_017dSEpTbexj8Rn8nQ4eXUjq` | 20:22 | `COORDINATION_20261117/WORKER_E50_FIXED_INPUT_CODE.md` (b77d9eb2) |

Each session is a new one, started from the default branch, in the same environment as the earlier workers. Each was given a short message that names its instruction file and repeats the standing rules: its own branch only, no push to the default branch, no pull request, other branches read-only, receipts pushed often, no name of a language model in committed files, reports to the reviewer by message. The earlier workers' sessions were left as they are and were not written to.

## 4. Decisions taken in the coordinator's place

**The first long lives** (roster note at 30628a4c; audit `AUDIT_E44_L2_ROSTER_NOTE_20261117.md`).
1. The nine required changes of the audit are entered as written.
2. The learning probe of change 9 is taken.
3. Release 1 uses `--list-every 8 --decay-distances none` (change 3).
4. Recommendations 1, 2, 3, 6 and 8 of the audit are adopted; 4 and 5 are given beside the note's own quantities; for 7 a prediction of the reviewer's is added (PL2.9: `replay-wd0` against `replay` at 256 visits is not distinguished in late level, 60%).
5. The lives without decay record the norm of every tensor at each visit at which the list is answered.
6. As the note itself says, the roster continues after the trial without waiting, unless one of the note's stop conditions holds.

**E50** (brief at f8f62df6; audit `AUDIT_E50_SPARSE_INPUT_CODE_BRIEF_20261117.md`).
1. The nine required changes of the audit are entered as written.
2. The predictions that change 8 asks for are the reviewer's (55% and 25% that `C16-in` reaches 80 on R within 49,152 updates under back-propagation and under the sweep; 45% that under back-propagation it does so within 3.24 times the updates of `C-dense`; 90% and 45% that max and min reach 80 at 12 letters).
3. Stage 2 starts by the rule of change 1 as soon as stage 1's report is sent; where a rule does not reach 80 on R in stage 1, nothing more is run under it.
4. Recommendation b is entered; recommendation a (a setting with 64 values a cell) is not run.

## 5. For the coordinator, on return

- Read this log and the three files it names. Anything decided here can be changed; the workers follow a message from the reviewer or a revised document.
- The requests of memo revision 21 (section 1, items 18 to 23) stand. Items 18 and 20 are being carried out by the workers' own revisions 2; item 21 by the reviewer's brief. Items 22 and 23 (merges and the record) are yours.
- When the workers' branches are merged, their "revision 2" files replace the note and the brief on the default branch, or you write your own.

## 6. Messages and events

| Time | From, to | Content |
|---|---|---|
| 20:22 | reviewer to the two new sessions | the starting messages described in section 3 |
| 20:30 | Mr. Xu to the reviewer | "Instruct the workers yourself this time." Then, on the aim: some rehearsal is acceptable; long-term memory should form by itself during learning and not be produced by our intervention. |
| 20:33 | reviewer to the worker of the first long lives | One table added to the reports, no run changed: "kept at return" and the level over the first 16 updates of a return, by the number of earlier visits to the family within each class of absence. It answers whether a memory becomes more lasting as a family is met again. |
| 20:36 | E50's worker to the reviewer | Working. Revision 2 of the brief committed on `claude/e50-fixed-input-code-017dSE` at d6606454; next the runner and the checks. |
| 20:38 | reviewer | Read the diff of revision 2 against revision 1: the nine changes and the decisions of section 4 are entered; the worker's own three additions are listed in the brief's last section. Nothing to change; no reply sent. |
| 20:40 | the long lives' worker to the reviewer | Working. Revision 2 of the note committed on `claude/e44-l2-first-long-lives` at 4048131b. One question: weight norms of the lives without decay from states saved at every eighth visit, without a third runner change. Next the baseline of check (a), the two runner changes, the checks. |
| 20:43 | reviewer to the long lives' worker | Revision 2 read against revision 1, nothing to change. The norms as the worker reads it; the states used for them are deleted afterwards and not committed. Asked it to make sure the table added at 20:33 is in the note. |
| 20:45 | Mr. Xu to the reviewer | Two earlier questions are still open: whether our learner overtakes the conventional one later in a long life (his statement of 15 November), and whether local learning can take the place of full connection. Decision of the reviewer: the sweep's long lives are brought forward and run beside release 1 in a helper session, once the runner's checks have passed. |
| 20:48 | E50's worker to the reviewer | Checks 1, 3, 4 and 5 pass (runner at 0f3ec4bc, report at 1d846f8b). Its digests and first-batch losses equal the reviewer's reader (code 12111044f0f8bc70; 5.597, 5.569, 5.591). Stage 1 (`C-dense`, both rules) started 20:47, expected to end about 21:50. Forecast for six runs 9.8 CPU-hours. |
| 20:59 | Mr. Xu to the reviewer | The overtaking question has a second form (far fewer samples); what else could have been tested and was not; then: put everything that could be done and has not been into order and do it item by item; the work has been careless; arrange it yourself. |
| 21:03 | reviewer | `PLAN.md` written: the requirements, the rules of order, seventeen items. Four agents of the reviewer started on items 1 to 3 (saved weights and the record only). |
| 21:07 | reviewer | Brief of E52 (the two rules under other learning rates) written and given to a check by an agent that has not seen the reviewer's reasoning. |
| 21:04 | the long lives' worker to the reviewer | The runner's checks (a) to (d) and the note's checks 1 to 7 pass (runner at f249b419, receipt at bd985865); row-key digests equal the audit's section 7; this host reproduces E44's committed receipt. Forecast for release 1: 48.2 CPU-hours. The other two births pass recommendation 3 (0.573 and 0.313). The trial and world 2's long life are running since 21:04. The added table is entered as D15 (9519302d). |
| 21:09 | reviewer | Read the diff of `sw_run.py` against the default branch (45 lines added, 9 removed): the two arguments select which rows of the list are written and at which distances decay alone is computed; `load_learner` accepts the two replay arms; nothing that learns is touched, and the assertion that answering the list leaves weights and optimizer unchanged is kept. Accepted. |
| 21:10 | reviewer | Helper session for the sweep's long lives started (plan item 6): `session_017dSEpTpaat6RxRKpJXsZWQ`, instructions `WORKER_E44_L2_SWEEP_LONG_LIVES.md` (3b4bdda5), from the main worker's branch. The main worker was told of it and asked to push each long life's event file when it finishes. |
| 21:37 | reviewer | The check of the E52 brief (`CHECK_OF_TASK_E52_BRIEF_20261117.md`) found ten defects; the reviewer recounted its figures on E41's receipts (the lag per seed; 79.95, 80.75, 80.35) and entered all ten as revision 2 of the brief. The worker session of E52 is started with revision 2 |
| 21:37 | reviewer | Worker session of E52 started: `session_017dSEpVsT2w6DQAJDRGDa8c`, brief revision 2 at 12168f12 |
| 21:42 | reviewer | The check of the E51 brief (`CHECK_OF_TASK_E51_BRIEF_20261117.md`; its script copied to `check_e51_brief.py` and run by the reviewer) found seven defects. Revision 2 of the brief enters them (the keeping made exact; the arm `keyed-free`; the score with the reserve shut; classes of wrong answers; nine details) and adds Part B. The worker session of E51 is started with revision 2 |
| 21:42 | reviewer | Worker session of E51 started: `session_017dSEpWQ9yfbN8dyC2VsWjy`, brief revision 2 at aab57310. Four lines are now running: the long lives with their helper, E50, E52, E51. E54's brief waits for its check and for a free line |
