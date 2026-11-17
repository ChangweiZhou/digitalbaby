# Instructions to the worker of E50: an input code in which a value stands for one byte (17 November 2026)

From the outside reviewer, acting for the coordinator. Mr. Xu, through whom the programme's owner works, is away. His instruction to the reviewer on 17 November, twice: "As I am outside, coordinate yourself" and "Unfortunately I am outside, so you have to coordinate yourself." The coordinator, which normally enters an audit's changes into a brief and releases it, runs on a machine that cannot be reached until he returns. These instructions therefore hand you the brief and its audit together. The coordinator has not seen this file; it will read it, and your work, on its return.

## 1. Documents

Shell names: `D=origin/codex/minifly-consolidated-20261023` (the default branch, your checkout); `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch); `W49=origin/claude/peaceful-noether-vcvrmf` (E49's worker, at a36de661); `M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. Fetch the two other branches with `git fetch origin '+refs/heads/<name>:refs/remotes/origin/<name>'` and read their files with `git show`; do not check them out and do not merge them. Where the brief builds on a file of E49's runner, copy that file byte for byte into your branch, record its SHA-256 and the commit it came from, and import it unchanged.

Read in this order.

1. `OPUS_TASK_E50_AN_INPUT_CODE_IN_WHICH_A_VALUE_STANDS_FOR_ONE_BYTE_20261116.md` on the default branch (revision 1, f8f62df6) and its specification file `PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e50_sparse_code_spec.py` with `OUTPUT_E50_SPEC_MAC.txt`: the brief. It is your task, as changed by the audit.
2. `$R/AUDIT_E50_SPARSE_INPUT_CODE_BRIEF_20261117.md` on the reviewer's branch: the audit. Its section 9 gives nine required changes as text to apply. The scripts `$R/e50_brief_checks.py` and `$R/e50_code_reader.py` (numpy, no torch) reproduce the audit's figures; the digests and first-batch losses they print are what your runner must give.
3. `OPUS_TASK_E49_A_LIMITED_INTERFACE_20261116.md` (revision 2) on the default branch, and E49's runner, receipts and report on `$W49`: the job whose masks and runner E50 uses.
4. `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md`: the repository's rules for agents, the commit trailer among them.

## 2. What to do, in this order

1. **Write revision 2 of the brief** as a file in your own branch, at the brief's path: the brief with the audit's nine changes entered exactly as section 9 gives them, and with the decisions of section 3 below. Head it "revision 2, entered by the worker from the reviewer's audit at ead8dcd8, in the coordinator's absence". Commit and push it before any run, and send the reviewer the commit (section 5). Do not wait for an answer.
2. **The runner and the checks** of the brief as changed (the audit's change 7 among them). Compare the code's digest, the masks' digests and the first-batch losses with the reviewer's figures in the audit before any run.
3. **Stage 1:** `C-dense` under the sweep and under back-propagation, seed 73129, 24,576 updates. Write the stage's report (E49's readings; the four operations apart, as change 3 asks; `C-dense` against E42's `F10`) and send it.
4. **Stage 2:** `C16-in`, 49,152 updates, under each rule under which `C-dense` reached 80 on R within its dose. It starts as soon as stage 1's report is sent; you do not wait for an answer. Under a rule under which `C-dense` did not reach 80 on R, nothing more is run, and the report says what the audit's change 1 words for that case.
5. **Stage 3:** `C46-one`, 49,152 updates, under each rule under which `C16-in` did not reach 80 on R within its dose.
6. **Write `E50_REPORT.md`** as the brief's last section asks, with the committed files listed from `git ls-tree`.

## 3. Decisions the audit leaves to the coordinator, taken here

- **The predictions that change 8 asks to be stated before the runs** (the reviewer's chances, entered into revision 2 in the place of P50.4 to P50.7).

  | | Statement | Chance |
  |---|---|---|
  | P50.4' | `C16-in` reaches 80 on R within 49,152 updates under back-propagation | 55% |
  | P50.5' | `C16-in` reaches 80 on R within 49,152 updates under the sweep | 25% |
  | P50.6' | Under back-propagation `C16-in` reaches 80 on R in no more than 3.24 times the updates of `C-dense` (1.5 times the factor 2.16 that E49's `in16` needed against `F10`) | 45% |
  | P50.7' | In `C16-in`, max and min at 12 letters reach 80 under back-propagation | 90% |
  | P50.8' | In `C16-in`, max and min at 12 letters reach 80 under the sweep | 45% |

- **Stage 1 is read by the rule of the audit's change 1** and stage 2 starts by it; where that text says "what follows is my decision", the decision is: nothing more is run under that rule, and the reviewer is told.
- **Recommendations a and b** of the audit's section 10: b is entered; a (a setting with 64 values a cell) is not run.
- **For the reviewer's later use:** commit the final weights of every run and the starting network of every setting, with the code table and the list of owners under the names that change 7 asks for. The births of `C-dense` will be used by a later job, so each `C-dense` receipt records the SHA-256 of its final weight file.

## 4. Rules

- At most three processes at once, one thread each (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`), launched detached and outside any wrapper with a time limit. The host of the brief: Python 3.13, torch 2.15.0+cpu from `https://download.pytorch.org/whl/cpu`, in a virtual environment. Record the versions you run in every receipt.
- The container can restart without warning and loses what is not pushed. Write a receipt with the scored rows as a run proceeds, commit and push receipts and logs at least every hour, and after a restart start an interrupted run again from its beginning (or from a checkpoint, if you have shown that a resumed run gives the uninterrupted run's hashes). Say in the report which runs were restarted.
- Push only to your own branch. Never push to the default branch or to any other branch, and open no pull request. Other branches are read-only for you.
- Every existing source file stays as it is; new code goes in new files.
- A receipt for every run; no path overwritten; the report lists committed files from `git ls-tree`.
- Reports in standard academic English: full sentences, every term defined at first use, no figure without the command or the receipt that gives it. No name of any language model in any committed file or commit message.
- If something in the brief and the audit cannot both be followed, follow the audit, say so in the report, and tell the reviewer.

## 5. Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: revision 2 committed; the checks passed (or which failed); each stage's report committed; the end. Each message gives your branch, the commit, and the main figures in at most ten lines. If you need a decision, ask in the same way and go on with whatever does not depend on the answer. The reviewer reads the runner, the check receipts and the reports as they arrive, and writes to you only if something must change.
