# Instructions to the worker of E44, job L2, release 1: the first long lives (17 November 2026)

From the outside reviewer, acting for the coordinator. Mr. Xu, through whom the programme's owner works, is away. His instruction to the reviewer on 17 November, twice: "As I am outside, coordinate yourself" and "Unfortunately I am outside, so you have to coordinate yourself." The coordinator, which normally enters an audit's changes into a note and releases it, runs on a machine that cannot be reached until he returns. These instructions therefore hand you the released note and its audit together. The coordinator has not seen this file; it will read it, and your work, on its return.

## 1. Documents

Shell names: `D=origin/codex/minifly-consolidated-20261023` (the default branch, your checkout); `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch); `M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. Fetch the reviewer's branch with `git fetch origin '+refs/heads/claude/vibrant-cerf-sje726:refs/remotes/origin/claude/vibrant-cerf-sje726'` and read its files with `git show $V:<path>`; do not check it out and do not merge it.

Read in this order.

1. `OPUS_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md` on the default branch (revision 1, entered at 30628a4c): the roster note. It is your task.
2. `$R/AUDIT_E44_L2_ROSTER_NOTE_20261117.md` on the reviewer's branch: the audit of that note. Section 8 gives nine required changes as text to apply; section 5 the definitions D1 to D14 that change 5 refers to; section 7 figures for you to compare with; section 9 recommendations; section 10 the order of work. The script `$R/e44_l2_checks.py` (numpy, no torch) reproduces every figure of the audit; its parts J and K compute the quantities and the digests named below.
3. `OPUS_TASK_E44_LONG_LIFE_WORLD_IN_THE_FULL_SCALE_FORMAT_20261116.md` (revision 2): the brief of the world, whose protocol the note runs inside.
4. `$M/exp45/e44/E44_REPORT.md` and the runner `$M/smallworld/sw_run.py` with `sw_score.py`, `sw_net.py`, `lw_world.py`, `lw_checks.py`: the calibration and the code you extend.
5. `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md`: the repository's rules for agents, the commit trailer among them.

## 2. What to do, in this order

1. **Write revision 2 of the note** as a file in your own branch, at the note's path: the note with the audit's nine changes entered exactly as section 8 gives them, and with the decisions of section 3 below. Head it "revision 2, entered by the worker from the reviewer's audit at ef8b512b, in the coordinator's absence". Commit and push it before any run, and send the reviewer the commit (section 5). Do not wait for an answer.
2. **Make the two changes to the runner** (the two arguments of the note, with the birth family following both arguments as change 2 says, and `none` as a value of `--decay-distances` as change 3 says; and the line of `load_learner` that change 1 names). Change nothing else in the runner.
3. **Run the checks of the change**, (a) to (d), as change 4 restates them, and compare your row keys with the digests of the audit's section 7. Then the note's checks 1 to 7.
4. **Run the two 8-visit lives of recommendation 3** (worlds 2 and 3 with the births of seeds 93683 and 116089, in turn, `naive`, 256 updates a visit) and report their median first-visit level. If either is below 0.20, stop and report: the gate would not hold for that birth.
5. **Run the trial**: world 1's two lives of 1,024 visits, `naive` and `replay`. A third process may start world 2's long life beside them. Write the trial's report (the note's diagnostics; the quantities D2 to D10 and D14 of the audit's section 5; `replay`'s median first-visit level beside `naive`'s, as change 8 asks) and send it.
6. **Continue with the roster without waiting**, unless one of the note's stop conditions holds, in the order of recommendation 1: the six long lives, then the eight short ones, then the sixteen lives without decay. Then the learning probe of change 9.
7. **Write `E44_L2_REPORT.md`** as the note's last section asks, with the committed files listed from `git ls-tree`.

## 3. Decisions the audit leaves to the coordinator, taken here

- Change 9: the learning probe is taken, as the change words it.
- Change 3: release 1 uses `--list-every 8 --decay-distances none`.
- Recommendations 1, 2, 3, 6 and 8 are adopted as written. Recommendations 4 and 5: give the stated quantities beside the note's own. Recommendation 7: add prediction PL2.9, "`replay-wd0` against `replay` at 256 visits is *not distinguished* in late level", chance 60%; it is the reviewer's.
- The lives without decay record the sizes of the weights (the norm of each tensor) at the end of every visit at which the list is answered. E46 found the recurrent matrix still growing at the end of a birth without decay.

## 4. Rules

- At most three processes at once, one thread each (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`), launched detached and outside any wrapper with a time limit. The host of the note: Python 3.13, torch 2.15.0+cpu from `https://download.pytorch.org/whl/cpu`, in a virtual environment. Record the versions you run in every receipt.
- The container can restart without warning and loses what is not pushed. Save states as recommendation 2 says, commit and push receipts, event files and the latest state of every running life at least every two hours, and after a restart continue each life with `life --resume` from its latest state. Say in the report which lives were resumed.
- Push only to your own branch. Never push to the default branch or to any other branch, and open no pull request. Other branches are read-only for you.
- Every existing source file stays as it is, except `sw_run.py` for the two changes; additions to the scorer go in a new file or in clearly marked additions to `sw_score.py`, as change 5 says.
- A receipt for every run; no path overwritten; the report lists committed files from `git ls-tree`.
- Reports in standard academic English: full sentences, every term defined at first use, no figure without the command or the receipt that gives it. No name of any language model in any committed file or commit message.
- If something in the note and the audit cannot both be followed, follow the audit, say so in the report, and tell the reviewer.

## 5. Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: revision 2 committed; the runner's checks passed (or which failed); the trial started; the trial's report committed; every stop condition; the end. Each message gives your branch, the commit, and the main figures in at most ten lines. If you need a decision, ask in the same way and go on with whatever does not depend on the answer. The reviewer reads the changed runner, the check receipts and the trial's report as they arrive, and writes to you only if something must change.
