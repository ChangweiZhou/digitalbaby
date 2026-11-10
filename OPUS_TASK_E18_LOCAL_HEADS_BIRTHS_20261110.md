# One cloud job for a Claude Opus 5.5 worker: E18, births with a next-byte target at every bank (10 November 2026, 12:00 EST)

**Repository:** `ChangweiZhou/digitalbaby`. **Branch to work from:** `codex/minifly-consolidated-20261023` (run `git fetch origin codex/minifly-consolidated-20261023 && git checkout codex/minifly-consolidated-20261023`; confirm that `OPUS_TASK_E18_LOCAL_HEADS_BIRTHS_20261110.md` and `MOD97_LOCAL_RULE_20261106/e18_check.py` are present). **Folder:** `MOD97_LOCAL_RULE_20261106/`. **Needs:** Python 3 with PyTorch on CPU and NumPy; install only those if missing. The job splits into two halves if two workers are available (half a: arms B0 and B1; half b: arms B2 and B3); say in your first message which you are running. A half is a complete job with its own predictions file, report and review paragraph.

**Branch rule.** If your session can push to `codex/minifly-consolidated-20261023`, commit there and push at most once an hour (partial receipts) and at the end; pull with `--rebase` before each push and never force. Otherwise commit to the branch your session is given, push there, and put the branch name and final commit in your last message; it will be merged by the coordinating session. Do not create any other branch.

**Rules.** Launch every birth detached from the tool command (`setsid nohup env ... python3 exp16_integrated.py dev <seed> > <log> 2>&1 < /dev/null &` from a short command that returns at once) and poll the logs with short commands; one line prints every 4,096 updates, six in all. Write logs outside the repository folder and copy them into `exp45/logs/` at the end as `e18_birth_<arm>_<seed>.log`. **First run `python3 e18_check.py`** and record its output; every number in it must be 0.0 and `head1_out_switch` true, as in `exp45/E18_CHECK.json`. **Dry-run each of the four commands once** with `E11_DEV_UPDATES=512` (the files get a `_dry512` tag) and move the dry files to `exp45/archive_e18_dry/cloud_<date>/`. Write and commit your predictions file before any birth starts and do not change it afterwards. Add files; edit none of the harness, receipts, reports, review or audit scripts. No reruns of a finished birth, no tuning, no change of seeds; a miss is a result, and a birth that learns nothing is a result. Report every seed separately; forecast the job from your dry run before launch, then report forecast against measured. **Run identity:** machine (`uname -a`, core count, `python3 -c "import torch; print(torch.__version__)"`), the harness commit, the environment of each process read from `/proc/<pid>/environ` while it runs, and the CPU-seconds of every birth (the last log line prints `cpu <n>s`). Report anything that went wrong plainly.

## Why

Mr. Xu's question of 10 November: local rules have not matched backprop head to head, but a learner in a stream also receives delayed feedback from the environment, in the purest form the next byte. Does that open a different design? One answer is that the next byte can be a target for every stage of the learner, so that no error has to be relayed from one stage to another. This job is the first measurement of what that costs and what it keeps.

The learner is the windowless two-bank network of E17 (`E16_ARCH=bank2`, `arch17.py`): it reads one byte at a time; a first bank of memory units with fixed retention rates is written from the byte; a second bank is written from the first bank's state; a readout predicts the next byte from both. Trained end to end by backprop through time it is born at 85–98% on the four development operations (mod 89, twelve distractor letters). A first bank alone (`E16_ARCH=bank`) is born at 22–30% on addition and 9–13% on subtraction.

The coordinating session's measurement of this morning (review 9.67, `exp45/E17_ALIGN_AT_BIRTH.md`): under the sign rule as run, the layers above the first bank's write get an update almost uncorrelated with the true gradient, because their credit is cut off in time; with credit carried through time at every layer and weight-sign feedback kept, every layer is at cosine 0.45–0.81. E17 Phase 4, running on the owner's machine today, tests that full relay in lives started from backprop births. This job tests births from nothing, with and without the relay between banks.

## What the harness adds (commit this brief ships with)

- `E16_HEADS=local` (tag `_lh`): the first bank gets its own next-byte head (three new layers `l2a`, `lma`, `l3a`), and every gradient path between the banks is cut: the second bank and the main readout read the first bank's state and first layer detached. Each bank learns from its own prediction error only. The learner's answers come from the main head, as before; the first bank's head is scored too (`head1` in the birth file).
- `E16_BIRTHRULE=T` (tag `_bT`): `dev` uses the life's own rule instead of backprop: weight-sign feedback, with the trace setting in force (`E16_TRACEUP`), decay 0.25. No backprop anywhere.
- `e18_check.py` verifies: with the options unset the gradients equal the previous commit's to the last bit; with local heads no gradient passes between the banks under either rule, and the shared layers start from the same weights as the plain network of the same seed.

## Runs: twelve births, seeds 73129, 93683, 116089

All with `E16_ARCH=bank2 E16_BOOKDECAY=off`, 24,576 updates, the standard development stream.

| Arm | What it is | Extra environment | Birth file `exp45/e17_dev_bank2_nbd<tag>_<seed>` |
|---|---|---|---|
| B0 | error relayed between banks and through time, weight signs in place of weights | `E16_BIRTHRULE=T E16_TRACEUP=all` | `_bT_tall` |
| B1 | no relay between banks; exact backprop inside each bank | `E16_HEADS=local` | `_lh` |
| B2 | no relay between banks; weight signs, credit through time inside each bank | `E16_HEADS=local E16_BIRTHRULE=T E16_TRACEUP=all` | `_lh_bT_tall` |
| B3 | no relay between banks; weight signs, trace on the write synapses only (the cheapest, forward-only for those synapses) | `E16_HEADS=local E16_BIRTHRULE=T` | `_lh_bT` |

```
E16_ARCH=bank2 E16_BOOKDECAY=off <extra environment> python3 exp16_integrated.py dev <seed>
```

Reference on file, not rerun: the end-to-end backprop births `exp45/e17_dev_bank2_nbd_<seed>.json` (made on the owner's machine; cloud and owner machines do not reproduce bit for bit, so compare levels, not digits) and the one-bank births `exp45/e17_dev_bank_<seed>.json`. Before launch, check and record that none of the twelve birth files exists. Commit each birth's `.json` and `.pt` (`git add -f` the `.pt`).

Reference levels at k = 12, final checkpoint, seeds in the order above:

| | add89 | sub89 | max89 | min89 |
|---|---|---|---|---|
| Two banks, end-to-end backprop | 93.0 / 94.7 / 95.7 | 89.3 / 94.3 / 85.0 | 97.7 / 94.7 / 97.0 | 94.3 / 95.7 / 94.0 |
| One bank, backprop | 30.3 / 29.0 / 21.7 | 13.0 / 9.7 / 8.7 | 53.0 / 61.0 / 58.0 | 61.3 / 61.0 / 60.7 |

## Predictions (copy into `E18_PREDICTIONS.md` and commit before any birth)

"Hard mean" is the mean of add89 and sub89 at k = 12 at the final checkpoint, from the main head unless the first bank's head is named. The reference hard means are 91.2 / 94.5 / 90.3. These were written by the coordinating session after its own dry runs of 512 updates, in which B0, B1 and B2 stood level with the default birth and B3 behind; nothing longer has been run.

- **P18.1 (the first bank alone stays a one-bank learner)** In B1, the first bank's own head ends with add89 below 45 at k = 12 in all three seeds.
- **P18.2 (the second bank can build on it)** In B1, the main head's hard mean is at or above 70 in at least two of the three seeds, and below the reference in all three. Held at about 55%: the first bank was never asked to serve the second.
- **P18.3 (weight signs with the relay give birth)** In B0, the hard mean is at or above 0.80 of the reference in at least two of the three seeds.
- **P18.4 (the relay is worth more than exact weights)** B0's hard mean is at least 5 points above B1's in at least two of the three seeds.
- **P18.5 (signs cost little once the relay is gone)** B2's hard mean is within 10 points of B1's in at least two of the three seeds.
- **P18.6 (the cheapest version fails)** In B3 the hard mean is below 40 in all three seeds, and its max89/min89 mean is at least 20 points below B2's: the layers above the first bank's write get credit from the current step only.

Reading, written before the result. If B1 comes within a few points of the reference, the relay between banks buys little on this task, and a learner with a target at every bank becomes the base design; the next step is to make each bank shallow enough that its credit is forward-only. If B1 falls far below while B0 holds, what backprop buys here is the relay itself, a next-byte target at each bank is not enough, and the first bank has to be told what the second needs by some other pathway (for example a top-down prediction compared locally with what arrives). P18.5 and P18.6 price the two cheaper versions of the within-bank rule. State which case obtained, seed by seed.

## Report

Per arm and seed: the six checkpoints of the birth curve (the four operations at k = 0, 4, 8, 12, 20), the first bank's head at the final checkpoint for the local-heads arms, the silenced score, and the hard mean beside the two references. The six verdicts; `e18_check.py` output; run identity and CPU-seconds per birth; forecast against measured; what went wrong. Deliverables: `E18_PREDICTIONS.md`, `E18_REPORT.md`, the twelve births, logs, and one paragraph as `E18_REVIEW_SECTION_9_71.md`.

**Standing notes.** Mr. Xu's working answers of 10 November (review 9.67): feedback relayed layer by layer or by a sub-network is allowed; a learner with a frozen part does not meet the aim; decisions remain Mr. Zhou's. Births are development results on sealed panels (evidence level E2); no life, no retention and no reuse on never-reinforced inputs is tested here, and no learning-core improvement is claimed. Lives from these births are a later job, briefed after the births are assessed.

**Cost.** On the owner's machine an end-to-end backprop birth took about 1,730 CPU-s; cloud containers have run this harness at roughly half that speed. The local-heads arms add a head (about 15–25% in the dry runs) and the sign-rule arms about as much again. Expect 3,500–5,500 CPU-s per birth, about 55,000 CPU-s for twelve: about 3.8 hours on four cores if run whole, about 1.9 hours per half. Forecast from your own dry run.
