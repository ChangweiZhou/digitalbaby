# E58: a new family learned in a gated term against free learning: the decay of the lived weights, the place of the change, and the number of taught pairs each needs (17 November 2026, revision 1)

Evidence level E2. **Status.** Written by the outside reviewer, acting for the coordinator, who cannot be reached while Mr. Xu is away (his instruction of 17 November: arrange it yourself). It is item 19 of `PLAN.md` beside this file, with a second part that the item did not have. It is given to a check by an agent of the reviewer that has not seen the reviewer's reasoning before it is released; that check is not the independent audit the programme's rules ask for, and the coordinator has not seen the brief. Three seeds, about 23 CPU-hours.

## Shell names and host

`D=origin/codex/minifly-consolidated-20261023` (the default branch, at 7cabb13a); `W8=origin/claude/sweet-turing-lwi3no` (E48's worker's branch, at 2d9d1726: the runner `exp48_gated_run.py`, `e48_gated.py`, and E48's receipts and term files under `$M/exp45/e48/`; the default branch is not an ancestor of it, so files are taken from it with `git show` and it is not checked out); `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch); `M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. Host: Linux x86-64, Python 3.13, torch 2.15.0+cpu; at most three processes, one thread each.

## Terms

- *The lived networks* are the six networks at the end of E26's lives (three seeds, each under two rules): `$D:$M/exp45/e26/e26_life_<S|B>_<seed>.pt`. Each has learned the first family and the families of its life.
- *Back-propagation* (rule B) is the exact gradient; *the sweep* (rule S) passes the error backward through each linear map by the signs of its weights times their mean size.
- *The new family* is E40's: add, sub, max and min under modulus 83, with the code bytes `e f g h`. *The taught fraction* f is the share of its pairs of operands that the stream presents; the others are *untaught*. In E40 and E48 f = 0.10.
- *Free learning* (E40, `$D:$M/e40_new_family.py`, runs `lived`): every weight of the lived network may change; the optimizer is the standing one (AdamW, rate 0.001, weight decay 0.25 on every tensor but the byte table).
- *The gated term* (E48, `$W8:$M/exp48_gated_run.py`, specification `e48_gated.py`): every weight of the lived network is frozen and is not in the optimizer. The change is made in a product of two matrices of rank r added to each of the two matrices of the recurrent layer, and in the four rows of the byte table that belong to the new codes. The term is added only for items whose code byte is one of the new family's; the harness reads this from the item. The term's factors decay at 0.25.
- *The reduction score* is E40's: the share of whole answers right on every untaught pair of add and of sub whose answer needs the reduction by the modulus, without letters (about 3,070 pairs each at f = 0.10). A *row* is a number of updates at which the network is scored.
- *The earlier families' panels* are E40's six panels (`earlier_k12`: add and sub under 89, 97 and 73).

## The question

Mr. Xu, 17 November: the end result of our learner may be about the same as the conventional one's while it needs far fewer samples. The record holds one hint in that direction. In E48 a new family learned from a tenth of its pairs in a gated term ended above the same family learned freely in E40. E48 differs from E40 in two things at once: where the change is made, and whether the lived weights decay (E48's brief says so; `$R/E48_RESULT_VERIFICATION_20261117.md`, section 10). E48 also stopped at 1,024 updates while E40 ran 8,192.

**Part A.** Does the gated term still end above free learning at 8,192 updates, and if so, is the reason the absence of decay on the lived weights or the place of the change? **Part B.** How does the level reached on untaught pairs depend on the taught fraction for each way of learning: does the gated term reach with half the pairs what free learning reaches?

## What the record says (working rule 1)

Seeds in the order 73129, 93683, 116089. The reviewer recomputed every figure of this section from the receipts (`$D:$M/exp45/e40/e40_lived_<rule>_<seed>.json`; `$W8:$M/exp45/e48/e48_<rule>_<seed>_r<rank>.json`).

- **E40, free learning, f = 0.10.** Reduction score at rows 64, 128, 256, 512, 1,024, 2,048, 4,096, 8,192. Sweep: 76.3, 85.3, 81.0, 81.7, 84.5, 83.6, 81.4, 77.0; 69.7, 70.1, 83.2, 83.5, 83.9, 82.8, 85.5, 82.1; 76.1, 80.8, 83.8, 79.7, 88.4, 88.3, 82.4, 83.7. Back-propagation: 83.5, 87.2, 68.1, 80.4, 86.1, 83.8, 82.3, 85.0; 92.0, 94.7, 92.1, 88.8, 93.8, 92.8, 90.5, 94.5; 95.4, 92.8, 89.8, 90.6, 90.8, 89.6, 91.4, 89.3. The score is reached within 64 to 256 updates and does not rise afterwards; single rows move by up to 19 points between neighbouring rows before row 1,024 and by up to 6 points from row 1,024 on.
- **E48, the gated term of rank 16, f = 0.10.** Reduction score at rows 512 and 1,024. Sweep: 89.6 and 92.2; 87.7 and 92.5; 86.0 and 89.5. Back-propagation: 90.7 and 93.1; 96.9 and 97.9; 98.5 and 98.7. The term is slower at first (below E40 at row 64 in every run) and was still rising at its last row by 0.2 to 4.8 points.
- **The hint.** At row 1,024 the term of rank 16 is above E40's run of the same seed and rule by 7.7, 8.6 and 1.1 points under the sweep and by 7.0, 4.1 and 7.9 under back-propagation. It is also above the highest row of that E40 run in all six cases, by 1.1 to 7.0 points. This was not a reading fixed beforehand.
- **Rank.** At row 1,024 the terms of rank 4, 16 and 64 reach 90.7, 92.2, 90.8; 83.4, 92.5, 93.5; 82.1, 89.5, 89.0 under the sweep and 91.1, 93.1, 94.0; 97.6, 97.9, 98.3; 98.1, 98.7, 98.8 under back-propagation. From rank 16 the level does not depend on the rank, so a narrow term is not what holds the level up.
- **The earlier families.** In E40 the lived network loses 38 to 43 points on the panels of its life's families within 256 updates. In E48 their answers are unchanged item for item, by construction.
- **A network without a life** (E40's `fresh` runs) reaches 6.0 to 11.8 on the reduction score at 8,192 updates. The new family is generalised from a tenth of its pairs only by a network that has learned the earlier families.
- No run exists at any other taught fraction of the new family, without decay in free learning, or of the gated term beyond 1,024 updates.

## Runs

Every run starts from the lived network of its seed and rule, uses E40's item format and batches of 64, and is scored at rows 0, 16, 32, 64, 128, 256, 512, 1,024 and then every 1,024 updates to its last update.

**Part A, f = 0.10 (E40's stream), three seeds, both rules.**

| Arm | What may change | Decay | Updates | Runs |
|---|---|---|---|---|
| `free` | every weight (E40's `lived` run again, with more rows) | 0.25, byte table 0 | 8,192 | 6 |
| `free-wd0` | every weight | 0 on every tensor | 8,192 | 6 |
| `term` | the gated term of rank 16 and the four code rows (E48's run again, continued) | 0.25 on the term's factors | 8,192 | 6 |
| `term-wd0` | as `term` | 0 | 8,192 | 6 |
| `term-shrink` | as `term`; in addition every tensor of the lived network outside the byte table is multiplied by (1 − 0.00025) after each update, which is the standing decay acting alone | 0.25 on the term's factors | 1,024 | 6 |

`term-shrink` stops at 1,024 updates because the factor is 0.774 there and 0.129 at 8,192, where nothing of the network would be left. In this arm the earlier families are not kept; that is its purpose.

**Part B, f = 0.05 and f = 0.20, three seeds, both rules.** Arms `free`, `free-wd0` and `term`, 8,192 updates: 36 runs. The stream at a fraction is E40's with that fraction: batches of 64 consecutive items of `X16.entry_stream([("op", t, a, b) for t in F4 for a, b in E.split(t, f, seed)[0]], seed + 140)`. The panels are E40's, built at the run's own fraction.

Order: the checks; Part A on seed 73129 (ten runs); Part A on the other two seeds; Part B. Nothing waits for a reply from the reviewer unless a check fails.

Cost, from the receipts of seed 73129 (E40: 974 CPU-seconds for 8,192 updates and 130 for nine rows; E48: 135 for 1,024 updates and 202 for eight rows): about 0.33 CPU-hours for a free run, 0.40 for a term run of 8,192 updates and 0.1 for `term-shrink`; 9.4 CPU-hours for Part A, 12.7 for Part B, about 0.5 for the checks.

## The runner

A new file `exp58_new_family.py` in `$M`; no existing source is edited. `exp48_gated_run.py` and `e48_gated.py` are copied byte for byte from `$W8` (they are not on the default branch; `e48_gated.py` equals `$D:PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e48_gated_spec.py`), with their SHA-256 and source recorded in a text file beside them. The new runner imports `e40_new_family` and `exp48_gated_run` and takes from them, unchanged, the start from a lived network, the stream, the scorer, the record of presented items, the gated forward and its update.

- **The fraction.** `e40_new_family.FRAC` is read by `e40_new_family.split` each time it is called and nowhere else, and the stream, the scorer and the record of presented items all go through `split`. The runner sets `e40_new_family.FRAC` to the run's fraction once, before any of them is built, records the fraction in the receipt, and asserts the number of taught pairs of each operation: 344 at f = 0.05, 689 at f = 0.10, 1,378 at f = 0.20.
- **The free arms** make E40's update (`exp26_candidate.step`) with the optimizer `X16.adamw16(m, C.LR, wd, ())`, wd = 0.25 for `free` (which is `exp26_candidate.make_opt`) and wd = 0 for `free-wd0`. The receipt records the optimizer's groups as read back from the optimizer, not as constants.
- **The term arms** make E48's update with `e48_gated.make_opt(m, term, wd=0.25)` or `wd=0`. `term-shrink` multiplies, after `opt.step()`, every parameter of the lived network except `book.weight` by `1 − 0.00025` in place.
- **Receipts** in E40's format with the arm, the fraction and E48's fields where the arm has a term: `$M/exp45/e58/e58_<arm>_<rule>_f<05|10|20>_<seed>.json`. The earlier families' six panels are scored at every row in every arm. In the term arms the digest of the earlier families' answers is recorded at every row, as in E48.
- **Files committed:** the receipts; the final weights of every free run (4.5 MB each); the term files at rows 1,024 and 8,192 of every term run; for `term-shrink` the final weights of the network as well. No weights at other rows.

## Checks before the runs (a receipt for each)

1. **`free` is E40's run.** Read from the two `free` runs of seed 73129, which are launched first; no other free run is launched before row 1,024 has passed. At each of E40's rows (64, 128, 256, 512, 1,024, 2,048, 4,096, 8,192) the batch digest, every count of every score and the `tensors_sha256` equal those of `$D:$M/exp45/e40/e40_lived_<rule>_73129.json`. The first comparison is at row 64, and a difference stops every run. (The mean loss since the last row is not compared at rows 4,096 and 8,192, since the new runs have rows between.)
2. **`term` is E48's run.** Read from the two `term` runs of seed 73129 in the same way: at each of E48's rows (0, 16, 32, 64, 128, 256, 512, 1,024) every count of every score, the batch digest and the digests of the earlier families' answers equal those of `$W8:$M/exp45/e48/e48_<rule>_73129_r16.json`, and the five tensors of the term at row 1,024 equal those of `$W8:$M/exp45/e48/e48_<rule>_73129_r16_row1024.pt` bit for bit.
3. **`free-wd0` differs from `free` in the decay only.** After the first update from the same start, for every tensor outside the byte table the difference between the two arms' tensors equals 0.00025 times the starting tensor, with a largest absolute deviation below 1e-6; the byte table is bit-equal in the two arms; the optimizer read back shows a weight decay of 0 in every group.
4. **`term-shrink`.** After the first update every tensor of the lived network outside the byte table equals 0.99975 times its start, with a largest absolute deviation below 1e-6; the rows of the byte table other than the four code rows are bit-equal to the start. At row 1,024 the ratio of the norm of each such tensor to its start is 0.774 to three digits.
5. **The other fractions.** For f = 0.05 and f = 0.20 on seed 73129: the asserted numbers of taught pairs; no pair of any panel of untaught pairs is among the taught pairs of its operation, and every pair of the taught panel is; a dry run of 64 updates of `free` under each rule ends with no untaught pair presented (E40's record of presented items); the batch digest at row 64 is recorded for the later runs to equal.
6. **Keeping in `term` and `term-wd0`:** at every row the digests of the earlier families' answers equal row 0's (E48's check 3), in every run. Read from the runs.
7. **Cost:** CPU-seconds an update and a scored row for each arm from the first runs, and a forecast.

## How the result is stated (fixed beforehand)

- *R-late* of a run is the mean of its reduction score over the five rows 4,096, 5,120, 6,144, 7,168 and 8,192. *R-mid* is the mean over rows 512 and 1,024.
- For two sets of three runs X and Y under one rule (one run a seed), X is *above* Y if X's figure exceeds Y's by 5 points or more on at least two seeds and X is not below Y by 5 points or more on the third; *below* is the mirror; otherwise the two are *not distinguished*. The figure is R-late unless R-mid is named. (From row 1,024 on, the single rows of E40's runs have a standard deviation of 1.0 to 3.3 points within a run; a difference of two means of five rows then has a scatter of about 0.6 to 2.1 points if the rows were independent, which is not known.)

**Part A, for each rule.**

1. If `term` is above `free`, the hint holds at 8,192 updates. If not, the report says that the gated term's lead at 1,024 updates does not hold at 8,192, gives both curves, and makes none of the statements of point 2.
2. If the hint holds: "the decay of the lived weights accounts for it" if `free-wd0` is above `free` and `term` is not above `free-wd0`; "the place of the change accounts for it" if `free-wd0` is not distinguished from `free` and `term` is above `free-wd0`; "both contribute" if `free-wd0` is above `free` and `term` is above `free-wd0`. Any other pattern is reported as found, without one of these sentences.
3. `term-wd0` against `term`: whether the decay of the term's own factors changes its level.
4. `term-shrink` against `term` on R-mid: whether shrinking the lived weights, with nothing else changed, lowers the level of the new family; beside it, what the shrinking does to the earlier families' panels at rows 512 and 1,024.
5. For `free` and `free-wd0`: the mean of the six earlier panels at every row, since it shows whether the loss of the earlier families in free learning depends on the decay.

**Part B, for each rule.** A table of R-late by arm, fraction and seed, with the levels of the taught panel beside it. At each fraction, `term` against `free` and against `free-wd0` in the words above. "The gated term needs fewer pairs" is licensed, for the fractions tested, by `term` at f = 0.05 not being below `free` at f = 0.10, or by `term` at f = 0.10 not being below `free` at f = 0.20; the report says which of the two holds under which rule. Where both levels of a comparison are above 97, the report says that the comparison is at the ceiling of the score and licenses nothing.

## Predictions (the reviewer's, on file before any run)

| | Statement | Chance |
|---|---|---|
| P58.1 | Checks 1 to 5 pass | 85% |
| P58.2 | `term` is above `free` at f = 0.10 under the sweep | 65% |
| P58.3 | `term` is above `free` at f = 0.10 under back-propagation | 70% |
| P58.4 | `free-wd0` is above `free` at f = 0.10 under at least one rule | 25% |
| P58.5 | `term-wd0` is not distinguished from `term` under both rules | 60% |
| P58.6 | `term-shrink` is below `term` on R-mid under both rules | 50% |
| P58.7 | The mean of the six earlier panels at row 1,024 is higher in `free-wd0` than in `free` by 10 points or more on two of three seeds, under at least one rule | 20% |
| P58.8 | At f = 0.05, `term` is above `free` under both rules | 50% |
| P58.9 | `term` at f = 0.05 is not below `free` at f = 0.10, under at least one rule | 45% |
| P58.10 | At f = 0.20 under back-propagation, `term` and `free` are not distinguished | 60% |

The explanation under test: a new family is generalised from few pairs by means of what the network learned from the earlier families (a network without a life does not generalise it at all), and free learning damages exactly that while it learns, so that a change confined to connections that only the new family uses reaches a higher level from the same pairs, or the same level from fewer. It is refuted, for this setting, if `term` is not above `free` at 8,192 updates under either rule, or if `free-wd0` is above `free` and `term` is not above `free-wd0` under both rules (the decay then accounts for the hint). `term-shrink` tests one part of it directly: whether damage to the lived weights alone lowers the level.

## What it cannot show

- How a learner would arrive at the separation by itself. The gate is read by the harness from the item's code byte, and the term is supplied from outside. The job measures what such a separation is worth in taught pairs; it says nothing on Mr. Xu's requirement that long-term memory form in the course of learning without our intervention.
- More than one new family, or new families in sequence; fixed memory (each family adds a term).
- Ranks other than 16; a term on other matrices; another rate.
- Whether a level that is still moving at 8,192 updates has reached its end. The curves are given.

## Rules for the worker

- At most three processes, one thread each (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`). torch 2.15.0+cpu from `https://download.pytorch.org/whl/cpu` in a virtual environment; versions recorded in every receipt.
- **This host stops a machine about five minutes after its session's turn ends, with the disk kept, and detached processes die with it.** While runs proceed, keep your turn open with a waiting command (a command that waits on the processes, repeated in pieces within the command's time limit), and commit and push finished receipts between the pieces. A run that was interrupted is started again from its beginning; the runs are deterministic and at most 0.4 CPU-hours long. The report says which runs were restarted.
- Push only to your own branch. Never push to the default branch or to any other branch, and open no pull request. Other branches are read-only: fetch with `git fetch origin '+refs/heads/<name>:refs/remotes/origin/<name>'` and read with `git show`.
- Commit trailers as `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` says. No name of any language model in any committed file or commit message.
- The report, `E58_REPORT.md`, in standard academic English: full sentences, every term defined at first use, every figure with the receipt or the command that gives it; the committed files listed from `git ls-tree`.
- If the brief cannot be followed at some point, stop that part, say what you found, and ask.

## Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: checks 3, 4 and 5 passed (or which failed); checks 1 and 2 passed at row 1,024; Part A finished on seed 73129, with its table; Part A finished on three seeds, with the readings; Part B finished; the report committed. Each message gives your branch, the commit and the main figures in at most ten lines. If you need a decision, ask in the same way and go on with whatever does not depend on the answer.

## Done

The seven checks; the 66 runs; `E58_REPORT.md` with the table of every run, the readings of Parts A and B in the words fixed above, the outcome of each prediction, and the list of committed files.
