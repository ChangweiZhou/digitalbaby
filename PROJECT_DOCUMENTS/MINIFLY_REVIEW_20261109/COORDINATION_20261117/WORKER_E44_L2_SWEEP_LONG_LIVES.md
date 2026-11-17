# Instructions to the helper of E44, job L2: the sweep's long lives, run beside release 1 (17 November 2026)

From the outside reviewer, acting for the coordinator, who cannot be reached while Mr. Xu is away (his instruction of 17 November: arrange it yourself). The coordinator has not seen this file. It is item 6 of `PLAN.md` beside this file.

## 1. Why

Mr. Xu's question of 15 November: our learner may be slower than the conventional one at first and overtake it in the course of continued learning; while it is still learning and the conventional learner has stopped, that cannot be ruled out. The roster note of the first long lives put the sweep into a second release, to start after the first was read. Mr. Xu pointed out on 17 November that this question is older than the others and is still open. The sweep's lives are therefore run now, at the same lengths and in the same worlds as release 1's lives under back-propagation, so that the two rules can be compared at 256 and at 1,024 visits.

## 2. Terms

As in the roster note and the audit: *the sweep* is the learning rule that passes the error backward by the signs of the weights (`local=True`; the runner's arms `sweep` and `sweep+replay`); *back-propagation* is the exact gradient (arms `naive` and `replay`). A *visit* is 256 updates on one family; a *life* is a schedule of visits in a *world*. The *store* is the replay buffer of the size of the network; `sweep+replay` learns half of each update from it (ρ = 0.5).

## 3. Documents

Your checkout is the main worker's branch `claude/e44-l2-first-long-lives` as it stood when this session started; it holds everything below. Shell names: `M=MOD97_LOCAL_RULE_20261106`; `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch; fetch it with `git fetch origin '+refs/heads/claude/vibrant-cerf-sje726:refs/remotes/origin/claude/vibrant-cerf-sje726'` and read with `git show`); `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`.

1. `OPUS_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md`, revision 2 (in your checkout): the roster note. Its definitions D1 to D15, its settings, its checks and its rules hold for you unless this file says otherwise.
2. `$R/AUDIT_E44_L2_ROSTER_NOTE_20261117.md` on the reviewer's branch, sections 5 and 7, and `$R/e44_l2_checks.py`.
3. The changed runner `$M/smallworld/sw_run.py` (the main worker's commit f249b419) and its check receipts under `$M/exp45/e44/L2/checks/`. **You do not change the runner or the scorer.** If you find you need to, stop and write to the reviewer.
4. `$M/exp45/e44/E44_REPORT.md`: the calibration, which ran the arm `sweep` in eight-visit lives.
5. `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md`: the repository's rules for agents.

## 4. Runs

| Arm | Worlds | Visits | Learner |
|---|---|---|---|
| `sweep` | 1, 2, 3, 4 | 1,024 | E41's half-width birth under the sweep, `e41_S_h256_f50_<seed>.pt`, with the note's seed for each world (worlds 1 and 4: 73129; 2: 93683; 3: 116089) |
| `sweep+replay` (ρ = 0.5) | 1, 2, 3, 4 | 1,024 | the same |

Eight lives. Every setting is release 1's: `--world lw --H 256 --U 128`, visits of 256 updates, weight decay 0.25, learning rate 0.001, `--list-every 8 --decay-distances none`, `--save-states 64,128,256,384,512,640,768,896,1024`. Order: world 1's two lives and world 2's `sweep` first; then the rest. At the main worker's measured 3.45 CPU-hours a life and the sweep's 7 to 11% more a update (E41), about 30 CPU-hours.

The learning rate 0.001 is the one every experiment has used and was chosen for back-propagation. Job E52 is testing other rates for both rules. If it finds that the sweep gains from another rate at this width, the reviewer will ask for these lives again at that rate; say so in your report.

## 5. Checks before the lives (a receipt for each)

1. `sw_run.py` has the SHA-256 that the main worker's check receipt records under `sources_sha256` (`$M/exp45/e44/L2/checks/E44_L2_CHECKS.json`; it begins 41714c5b). The scorer was extended by the main worker after that receipt; you use its file as it stands in the commit you start from and record its hash and that commit.
2. On this host, world 1's eight-visit calibration life under `sweep` at 256 updates a visit (run with `--in-turn`; E44's receipt is `$M/exp45/e44/F1/lw_sweep_w1_v8_turn.json`), with the defaults of the two new arguments, gives the hashes of that receipt (list, events, final weights and optimizer). If this host does not reproduce them, say which differ and by how much the scores differ, and go on only if the event file's scores agree to the last item.
3. `sweep+replay` with ρ = 0 equals `sweep` on world 1 for one visit: the same weights hash and event file.
4. Resume: a `sweep+replay` life of world 1 stopped at visit 4 of 8 and resumed gives the uninterrupted hashes and event file.
5. The births: the three files' hashes equal their E41 receipts, and their scores through the runner equal the receipts' final rows.
6. Cost: CPU-seconds a visit for each arm, and a forecast.

## 6. What is recorded and stated

Everything the note records for a life (D1 to D15), computed with the main worker's scorer for your eight lives.

**Between the rules**, which is the purpose of these lives. For each world and each of the pairs `sweep` against `naive` and `sweep+replay` against `replay` (the main worker's lives of the same world, read from its branch when their event files are committed), at 256 and at 1,024 visits: the difference in late level (D4), in kept at return at medium and long absences, in the score trend (D5) and in the trend of learning (D7); the mean over the four worlds with its 90% interval (D11); and whether the comparison is closed by D12. Beside it the curves of score by age of the two rules in each world, so that a crossing, if there is one, is seen. Note in the table that the two rules start from different births (each from the birth made under its own rule), whose scores at birth you give.

One sentence is fixed beforehand for each pair: over 1,024 visits the sweep is *behind*, *level* or *ahead* in late level (by D11's interval), and its trend over the second half is *steeper*, *the same* or *flatter* than back-propagation's (the difference of the score trends, by its interval). "Overtaking in time" would show as a steeper trend with a late level that is behind or level. Nothing is said beyond 1,024 visits.

## 7. Predictions (the reviewer's, on file before any life)

| | Statement | Chance |
|---|---|---|
| PS.1 | `sweep` at 1,024 visits is *stuck* (the note's definition) | 50% |
| PS.2 | `sweep` is *behind* `naive` in late level at 1,024 visits | 45% (level 40%, ahead 15%) |
| PS.3 | `sweep+replay` is *higher* than `sweep` in late level at 1,024 visits | 70% |
| PS.4 | The sweep's score trend over the second half is *steeper* than back-propagation's, without the store | 15% |
| PS.5 | The answer of PS.2 at 256 visits is the same as at 1,024 visits | 70% |

## 8. Rules

- At most three processes at once, one thread each (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`), launched detached and outside any wrapper with a time limit. Python 3.13, torch 2.15.0+cpu from `https://download.pytorch.org/whl/cpu`, numpy as the main worker's receipts record, in a virtual environment. Record the versions in every receipt.
- The container can restart without warning and loses what is not pushed. The main worker's restart script (`d564c219`) shows one way to resume; commit and push receipts, event files and the latest state of every running life at least every two hours, and continue an interrupted life with `life --resume`.
- Push only to the branch this session was given. Never push to the default branch, to the main worker's branch or to any other branch, and open no pull request. Other branches are read-only for you.
- Outputs go under `$M/exp45/e44/L2/sweep/`; no path of the main worker is overwritten.
- The report, `E44_L2_SWEEP_REPORT.md`, in standard academic English: full sentences, every term defined at first use, every figure with the receipt or the command that gives it; the committed files listed from `git ls-tree`. No name of any language model in any committed file or commit message; commit trailers as `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` says.

## 9. Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: the checks passed (or which failed); world 1's two lives finished, with their figures at 256 and 1,024 visits beside the main worker's if those are committed; the eight lives finished; the report committed. Each message gives your branch, the commit and the main figures in at most ten lines. If you need a decision, ask in the same way and go on with whatever does not depend on the answer. The main worker is session `session_017dSEpXu6oc38q4xQZ5sPv3`; you may ask it where a file is, and you do not give it tasks.
