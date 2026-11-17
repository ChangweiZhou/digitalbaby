# E59: a task of another kind learned by a fresh network, by a born network and by a lived network, from three numbers of taught pairs (17 November 2026, revision 1)

Evidence level E2. **Status.** Written by the outside reviewer, acting for the coordinator, who cannot be reached while Mr. Xu is away (his instruction of 17 November: arrange it yourself). It is item 15 of `PLAN.md` beside this file. It is given to a check by an agent of the reviewer that has not seen the reviewer's reasoning before it is released; that check is not the independent audit the programme's rules ask for, and the coordinator has not seen the brief. Three seeds, about 16 CPU-hours.

## Shell names and host

`D=origin/codex/minifly-consolidated-20261023` (the default branch, at 7cabb13a); `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch); `M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. Host: Linux x86-64, Python 3.13, torch 2.15.0+cpu; at most three processes, one thread each.

## Terms

- *Back-propagation* (rule B) is the exact gradient; *the sweep* (rule S) passes the error backward through each linear map by the signs of its weights times their mean size.
- *A fresh network* is `arch26.new_net(seed)` before any update. *A born network* is E26's birth: the fresh network after 24,576 updates on the first family (add, sub, max and min under modulus 89), `$D:$M/exp45/e26/e26_dev_<S|B>_<seed>.pt`. *A lived network* is the born network after the six blocks of E26's life (36,864 further updates on three other families), `$D:$M/exp45/e26/e26_life_<S|B>_<seed>.pt`. All three have the present size (1,124,113 parameters).
- *The task of another kind* is E43's unrelated task (`$D:$M/e43_task.py`): an item is a code byte, two symbols out of 64, then 0 to 12 filler bytes and a query byte; the answer is the two symbols and the end symbol. None of its bytes occurs in any family's items. Of the 4,096 ordered pairs of symbols, E43 teaches 2,048 and holds 2,048 out.
- *The held panel* is E43's: 300 held-out pairs at 12 fillers (`e43_task.panel(seed, "held")`). *The taught panel* is 300 taught pairs at 12 fillers, or every taught pair where fewer than 300 are taught.
- Rows are scored every 32 updates to update 1,024 and every 128 updates from there to the end. *Updates to 80* is the first of three consecutive scored rows at which the held panel is at or above 80% (240 of 300). A *level* at the end is the mean over the last three rows.
- A curve *has flattened* if the mean of the held panel over the eight rows from update 7,296 to 8,192 is less than 3 points above its mean over the eight rows from update 5,248 to 6,144.

## The question

Mr. Xu, 17 November: the end result of our learner may be about the same as the conventional one's while it needs far fewer samples. The record shows such a saving in one place. A network that has lived learns a new family of the same kind from a tenth of its pairs, which a fresh network does not (E40); the saving comes from having lived and is found under both rules (`$R/E40_SAVING_IN_WHOLE_EXPERIENCE_20261117.md`). The new family there has the format and the operations of the families already learned. **Does having learned the families also lower what a network needs for a task of another kind: the number of presentations, and the number of distinct taught pairs?**

## What the record says (working rule 1)

- **E43, the unrelated task from the six born networks, 2,048 taught pairs** (`$D:$M/exp45/e43/e43_unrelated-decay_<rule>_<seed>.json`, weight decay 0.25; right of 300 on the taught and on the held panel; seeds 73129, 93683, 116089). At update 64 nothing is learned (0 to 4 of 300). At update 256: under back-propagation 294 and 294, 289 and 276, 300 and 300; under the sweep 22 and 14, 135 and 91, 124 and 92. At update 1,024 and later every panel is at 300 of 300 under both rules. The rows are 0, 64, 256, 1,024, 2,048 and 4,096, so the update at which 80% is reached is not located more finely than "by 256" under back-propagation and "between 256 and 1,024" under the sweep. The runs without decay give the same figures to within 4 items.
- No run of the unrelated task from a fresh network or from a lived network exists, and none with another number of taught pairs.
- **The related case, for comparison** (E40): on a new family of the same kind, from a tenth of its pairs, the lived networks reach 77 to 95% on untaught pairs that need the reduction, and the fresh networks 6 to 12%, after 8,192 updates.

## Runs

| Start | Taught pairs | Rules | Seeds | Updates | Runs |
|---|---|---|---|---|---|
| `fresh`, `born`, `lived` | 2,048 (E43's); 512; 256 | S, B | 73129, 93683, 116089 | 8,192 | 54 |

Each run continues its start under the rule that made it (`fresh` under the rule named), with a new optimizer `exp26_candidate.make_opt(m)` (AdamW, rate 0.001, weight decay 0.25 on every tensor but the byte table), in batches of 64, as E43's arm `unrelated-decay`.

**The taught sets are nested.** With `taught, held = e43_task.split(seed)`, the taught set of size n is `taught[:n]`. The held panel is the same for every n. The stream for n pairs is the text of `e43_task.stream` with `taught[:n]` in place of `taught`: a generator `Random(f"e43|{seed}|stream")`, the pairs in a new shuffled order each pass, for each item `randint(0, 12)` fillers. With n = 2,048 it is E43's stream. (With 256 pairs a given symbol is absent from a given place of the taught pairs with a chance of 1.8%, so about one symbol in each place is never shown there; with 512 the chance is 0.03%.)

Order: the checks; the eighteen runs of seed 73129; the other two seeds. Nothing waits for a reply from the reviewer unless a check fails.

Cost: E40's runs took 0.12 CPU-seconds an update at this size (974 CPU-seconds for 8,192 updates); a row here scores 600 items, and a run has 89 rows. About 0.3 CPU-hours a run, 16 CPU-hours in all.

## The runner

A new file `exp59_another_kind.py` in `$M`; no existing source is edited. It takes from `exp43_unrelated.py` and `e43_task.py`, unchanged, the update (`exp26_candidate.step`), the panels and the assertion that every prompt byte of a batch is 128 or above; from `e40_new_family.start` the loading of a lived network with the check of its file against its receipt; and E43's admission of the born networks (`exp26_candidate.admit_e26`). It does not score the first family or the life's families at every row; it scores them once, at row 0 and at the last row, with E43's `score2` panel for the first family and E40's six earlier panels, so that the report can say what the new learning cost. Receipts: `$M/exp45/e59/e59_<start>_n<2048|512|256>_<rule>_<seed>.json`, with the start's file and hashes, the optimizer's groups as read back, and at every row the batch digest, the counts of the two panels and the mean loss since the last row; the `tensors_sha256` at rows 64, 256, 1,024, 2,048, 4,096 and 8,192. Final weights are committed for the runs of seed 73129 only (4.5 MB each).

## Checks before the runs (a receipt for each)

1. **`born` with 2,048 pairs is E43's run.** Read from the two such runs of seed 73129, which are launched first. At rows 64, 256, 1,024, 2,048 and 4,096 the batch digest, the counts of the two panels and the `tensors_sha256` equal those of `$D:$M/exp45/e43/e43_unrelated-decay_<rule>_73129.json`. The first comparison is at row 64, and a difference stops every run.
2. **The stream.** For each seed, `e43_task.digest(seed, 64)` equals the digest of the first 64 batches of the runner's stream with n = 2,048, computed in the same way. For n = 512 and n = 256 the runner asserts that every pair presented is in `taught[:n]`, that no pair of the held panel is, and records the digest of the first 64 batches.
3. **The starts.** `fresh`: the `tensors_sha256` of the start equals `start.tensors_sha256` in `$D:$M/exp45/e40/e40_fresh_<rule>_<seed>.json`. `born`: admitted by `exp26_candidate.admit_e26`, as in E43. `lived`: `e40_new_family.start("lived", rule, seed)` passes its own check of the file against the life's receipt.
4. **Cost:** CPU-seconds an update and a scored row, and a forecast.

## How the result is stated (fixed beforehand)

For each rule and each number of taught pairs, three comparisons: `born` against `fresh`, `lived` against `fresh`, `lived` against `born`.

- **Presentations.** Updates to 80 on the held panel, a seed at a time, and the ratio of the two starts. The first start needs *fewer* if the ratio is below 2/3 on at least two seeds and not above 1.5 on the third; *more* is the mirror; otherwise *about the same*. Where one of the two does not reach 80 within the run, E42's rule applies (`describe` in `e42_tables.py` on `origin/claude/gallant-dijkstra-i9zgsr`, lines 63 to 72, with a run of 8,192 updates on both sides).
- **Pairs.** A start *generalises from n pairs* under a rule if the held panel reaches 80 within the run on at least two of the three seeds. "Having learned the families lowers the number of pairs needed for a task of another kind" is licensed under a rule if, for some n, `born` or `lived` generalises from n pairs and `fresh` does not, and the curves of `fresh` have flattened on the seeds on which it does not reach 80. If they have not flattened, the report says "reaches 80 from n pairs within the run, and the fresh network does not; whether the fresh network would arrive is not decided".
- **Levels** of the held and of the taught panel at the end, for every run.
- Beside these, without a reading: the first family's and the life's families' panels at row 0 and at the last row.

## Predictions (the reviewer's, on file before any run)

| | Statement | Chance |
|---|---|---|
| P59.1 | Checks 1 to 3 pass | 90% |
| P59.2 | With 2,048 pairs, `born` needs *fewer* updates than `fresh`, under both rules | 70% |
| P59.3 | With 2,048 pairs, `lived` against `born` is *about the same*, under both rules | 55% |
| P59.4 | With 256 pairs, `fresh` does not generalise, under either rule | 55% |
| P59.5 | The sentence on pairs is licensed under at least one rule | 35% |
| P59.6 | With 2,048 pairs, `fresh` under the sweep needs more than 1.5 times the updates of `fresh` under back-propagation on at least two seeds | 60% |

The explanation under test: what a network gains from the families is in part an ability that does not depend on their content, namely to hold two items across bytes that do not matter and to give them back, and a task of another kind that asks for the same ability is learned from fewer presentations, and perhaps from fewer pairs, by a network that has it. It is refuted, for this task, if with every number of pairs `born` and `lived` are *about the same* as `fresh` or need *more*, under both rules.

## What it cannot show

- A task that shares nothing with the families. This one shares the format of an item, the length of the answer and the need to carry two items across other bytes (E43's brief says so). A saving here is a saving on a task of another content and not of another kind in every respect.
- What a learner keeps of the families while it learns the new task; the two extra scores are given without a reading, and E43 and E51 measure that question.
- More than one new task; other numbers of taught pairs than three; other sizes of network.
- Whether the sweep and back-propagation differ in what they gain from a life, beyond what the three comparisons show under each.

## Rules for the worker

- At most three processes, one thread each (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`). torch 2.15.0+cpu from `https://download.pytorch.org/whl/cpu` in a virtual environment; versions recorded in every receipt.
- **This host stops a machine about five minutes after its session's turn ends, with the disk kept, and detached processes die with it.** While runs proceed, keep your turn open with a waiting command (a command that waits on the processes, repeated in pieces within the command's time limit), and commit and push finished receipts between the pieces. A run that was interrupted is started again from its beginning; the runs are deterministic and about 0.3 CPU-hours long. The report says which runs were restarted.
- Push only to your own branch. Never push to the default branch or to any other branch, and open no pull request. Other branches are read-only: fetch with `git fetch origin '+refs/heads/<name>:refs/remotes/origin/<name>'` and read with `git show`.
- Commit trailers as `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` says. No name of any language model in any committed file or commit message.
- The report, `E59_REPORT.md`, in standard academic English: full sentences, every term defined at first use, every figure with the receipt or the command that gives it; the committed files listed from `git ls-tree`.
- If the brief cannot be followed at some point, stop that part, say what you found, and ask.

## Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: checks 2 and 3 passed (or which failed); check 1 passed; seed 73129 finished, with its table; all three seeds finished, with the readings; the report committed. Each message gives your branch, the commit and the main figures in at most ten lines. If you need a decision, ask in the same way and go on with whatever does not depend on the answer.

## Done

The four checks; the 54 runs; `E59_REPORT.md` with the table of every run, the readings in the words fixed above, the outcome of each prediction, and the list of committed files.
