# E58: a new family learned in a gated term against free learning: the decay of the lived weights, the place of the change, and the number of taught pairs each needs (17 November 2026, revision 2)

**Status.** The brief was written by the outside reviewer, acting for the coordinator, who cannot be reached while Mr. Xu is away (his instruction of 17 November: arrange it yourself). Revision 1 was checked by an agent of the reviewer that had not seen the reviewer's reasoning (`CHECK_OF_TASK_E58_BRIEF_20261117.md` beside this file). The check found twelve defects and gave a text for each; revision 2 enters all of them, and the last section lists them. The check is not the independent audit that the programme's rules ask for, and the coordinator has not seen the brief. The job is at the evidence level that the coordinator's briefs call E2: a first measurement, which confirms nothing. It is item 19 of `PLAN.md` beside this file, with a second part that the item did not have. The job uses three seeds and about 27 CPU-hours.

## Shell names and host

`D=origin/codex/minifly-consolidated-20261023` (the default branch, at 7cabb13a); `W8=origin/claude/sweet-turing-lwi3no` (E48's worker's branch, at 2d9d1726: the runner `exp48_gated_run.py`, `e48_gated.py`, and E48's receipts and term files under `$M/exp45/e48/`; the default branch is not an ancestor of it, so files are taken from it with `git show` and it is not checked out); `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch); `M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. E48's report is `$W8:$M/exp45/e48/E48_REPORT.md`. The host is Linux on x86-64 with Python 3.13 and torch 2.15.0+cpu.

## Terms

- A *family* is a set of four operations on two numbers (add, sub, max, min) under one modulus; a code byte at the start of an item names the operation and the family. An *item* is the code byte, the first operand in two digits, the operator, 0 to 12 capital letters that carry no information, the second operand and '='; its answer is two digits. In E26 each network was trained on the family of modulus 89 (*the first family*) and then, one after the other, on the family of modulus 97, on compositions of two operations and on the family of modulus 73. The project calls this sequence the network's *life* and a network at its end a *lived network*.
- *The lived networks* are the six networks at the end of E26's lives (three seeds, each under two rules): `$D:$M/exp45/e26/e26_life_<S|B>_<seed>.pt`. At the end of its life each answers 87 to 97% of the four panels of its life's families and 13 to 37% of the two panels of the first family. A *panel* is a fixed set of test items.
- *Back-propagation* (rule B) is the exact gradient; *the sweep* (rule S) passes the error backward through each linear map by the signs of its weights times their mean size.
- *The new family* is E40's: add, sub, max and min under modulus 83, with the code bytes `e f g h`. *The taught fraction* f is the share of its pairs of operands that are taught; the others are *untaught*. In E40 and E48 f = 0.10.
- *The byte table* is the embedding table `book.weight`, with one row of eight numbers for each byte. *The standing optimizer* is the optimizer used in every experiment since E26: AdamW, rate 0.001, weight decay 0.25 on every tensor but the byte table.
- *Free learning* (E40, `$D:$M/e40_new_family.py`, runs `lived`): every weight of the lived network may change, under the standing optimizer.
- *The gated term* (E48, `$W8:$M/exp48_gated_run.py`, specification `e48_gated.py`): every weight of the lived network is frozen and is not in the optimizer. The change is made in a product of two matrices of rank r added to each of the two matrices of the recurrent layer, and in the four rows of the byte table that belong to the new codes. The term is added only for items whose code byte is one of the new family's; the program that runs the experiment reads this from the item, and the network does not decide it. The term's factors decay at 0.25.
- *Decay toward the start* (`free-anchor`): after each update every tensor outside the byte table is moved toward its value in the lived network by 0.00025 of its distance from it. The standing decay moves a weight w with lived value w0 by −0.00025·w = −0.00025·(w − w0) − 0.00025·w0; `free-anchor` keeps the first part, which pulls the change toward zero, and drops the second, which pulls the lived weight toward zero.
- *The reduction score* is E40's: the percentage of answers that are right in every byte, over all untaught pairs of add and of sub whose answer requires the reduction by the modulus (a + b ≥ 83, or a − b < 0), each presented with no letters; the score is the mean of the two percentages (about 3,070 pairs each at f = 0.10). A *row* is a number of updates at which the network is scored.
- *The earlier families' panels* are E40's six panels (`earlier_k12`: add and sub under 89, 97 and 73).

## The question

Mr. Xu, 17 November: the end result of our learner may be about the same as the conventional one's while it needs far fewer samples. The record holds one observation that points in that direction. In E48 a new family learned from a tenth of its pairs in a gated term ended above the same family learned freely in E40. E48 differs from E40 in two things at once: where the change is made, and whether the lived weights decay (E48's brief says so; `$V:$R/E48_RESULT_VERIFICATION_20261117.md`, section 10). E48 also stopped at 1,024 updates while E40 ran 8,192.

**Part A.** Does the gated term still end above free learning at 8,192 updates, and if so, is the reason the absence of decay on the lived weights or the place of the change? **Part B.** How does the level reached on untaught pairs depend on the number of taught pairs for each way of learning: does the gated term reach from fewer pairs the level that free learning reaches from more?

## What the record says

The seeds are given in the order 73129, 93683, 116089. The reviewer recomputed the figures of this section from the receipts (`$D:$M/exp45/e40/e40_lived_<rule>_<seed>.json`; `$W8:$M/exp45/e48/e48_<rule>_<seed>_r<rank>.json`), and the check recomputed them again.

- **E40, free learning, f = 0.10.** Reduction score at rows 64, 128, 256, 512, 1,024, 2,048, 4,096, 8,192. Sweep: 76.3, 85.3, 81.0, 81.7, 84.5, 83.6, 81.4, 77.0; 69.7, 70.1, 83.2, 83.5, 83.9, 82.8, 85.5, 82.1; 76.1, 80.8, 83.8, 79.7, 88.4, 88.3, 82.4, 83.7. Back-propagation: 83.5, 87.2, 68.1, 80.4, 86.1, 83.8, 82.3, 85.0; 92.0, 94.7, 92.1, 88.8, 93.8, 92.8, 90.5, 94.5; 95.4, 92.8, 89.8, 90.6, 90.8, 89.6, 91.4, 89.3. The score is reached within 64 to 256 updates, and no later row exceeds the highest of rows 64 to 256 by more than 4.6 points; single rows move by up to 19 points between neighbouring rows before row 1,024 and by up to 6 points from row 1,024 on.
- **E48, the gated term of rank 16, f = 0.10.** Reduction score at rows 512 and 1,024. Sweep: 89.6 and 92.2; 87.7 and 92.5; 86.0 and 89.5. Back-propagation: 90.7 and 93.1; 96.9 and 97.9; 98.5 and 98.7. The term is slower at first (below E40 at row 64 in every run; it reaches 80% at 256, 512 and 512 updates under the sweep, where free learning reaches it at 128, 256 and 128) and was still rising at its last row by 0.1 to 4.8 points.
- **The lead at 1,024 updates.** At row 1,024 the term of rank 16 is above E40's run of the same seed and rule by 7.6, 8.6 and 1.1 points under the sweep and by 6.9, 4.1 and 7.9 under back-propagation. It is also above the highest row of that E40 run in all six cases, by 1.1 to 7.0 points. This was not a reading fixed beforehand.
- **Rank.** At row 1,024 the terms of rank 4, 16 and 64 reach 90.7, 92.2, 90.8; 83.4, 92.5, 93.5; 82.1, 89.5, 89.0 under the sweep and 91.1, 93.1, 94.0; 97.6, 97.9, 98.3; 98.1, 98.7, 98.8 under back-propagation. From rank 16 the level does not depend on the rank, so the lead over free learning is not an effect of the small rank.
- **The earlier families.** In E40 the lived network has lost 38 to 43 points on the four panels of its life's families (add and sub under 97 and 73) at row 256. In E48 their answers are unchanged item for item, by construction.
- **Shrinking without learning.** If every tensor of a lived network outside the byte table is multiplied by (1 − 0.00025) r times and nothing is learned, the mean of the four panels of the life's families falls by 0.1 to 2.2 points at r = 64, by 1.2 to 13.9 at r = 256, by 7.6 to 42.8 at r = 512 and by 64 to 86 at r = 1,024 (`CHECK_OF_TASK_E58_BRIEF_20261117.md` beside this file, defect 2). In E40's free learning the same mean has fallen by 17 to 46 points at row 64, where the norm of the weights outside the byte table is 0.996 to 1.001 of its starting value; at row 8,192 the norm is 0.84 to 0.97, where decay acting alone would give 0.129. The loss of the earlier families in the first 64 updates of free learning is therefore not a loss by shrinking.
- **A network that was not trained before** (E40's `fresh` runs) reaches 6.0 to 11.8 on the reduction score at 8,192 updates. The new family is generalised from a tenth of its pairs only by a network that has learned the earlier families.
- No run exists at any other taught fraction of the new family, without decay in free learning, or of the gated term beyond 1,024 updates (checked in the folders of E40 and E48).

## Runs

Every run starts from the lived network of its seed and rule, uses E40's item format and batches of 64, and is scored at rows 0, 16, 32, 64, 128, 256, 512, 1,024 and then every 1,024 updates to its last update.

**Part A, f = 0.10 (E40's taught pairs), three seeds, both rules.**

| Arm | What may change | Decay | Updates | Runs |
|---|---|---|---|---|
| `free` | every weight (E40's `lived` run again, with more rows) | 0.25, byte table 0 | 8,192 | 6 |
| `free-again` | as `free`, with the items in another order: the stream is `X16.entry_stream(e40_new_family.entries(seed), seed + 141)` in batches of 64, so that its batch digests are its own | as `free` | 8,192 | 6 |
| `free-wd0` | every weight | 0 on every tensor | 8,192 | 6 |
| `free-anchor` | every weight | none in the optimizer; decay toward the start at 0.00025 an update, byte table none | 8,192 | 6 |
| `term` | the gated term of rank 16 and the four code rows (E48's run again, continued) | 0.25 on the term's factors | 8,192 | 6 |
| `term-wd0` | as `term` | 0 | 8,192 | 6 |

**Part B, f = 0.05 and f = 0.20, three seeds, both rules.** Arms `free`, `free-wd0` and `term`, 8,192 updates: 36 runs. The stream at a fraction is E40's on the nested taught pairs of that fraction ("The runner", the fraction): batches of 64 consecutive items of `X16.entry_stream([("op", t, a, b) for t in F4 for a, b in nested(t, f, seed)[0]], seed + 140)`. The panels are E40's, built at the run's own fraction.

The order is: checks 3, 4, 5 and 8; Part A on seed 73129 (twelve runs; checks 1 and 2 are read from its `free` and `term` runs, check 6 from its term runs, check 7 from all twelve); Part A on the other two seeds; Part B on seed 73129 (twelve runs), with its table reported before the other two seeds are launched; Part B on the other two seeds. Nothing waits for a reply from the reviewer unless a check fails.

The cost is estimated from the receipts of seed 73129 under the sweep (E40: 974 CPU-seconds for 8,192 updates and 130 for nine rows; E48: 135 for 1,024 updates and 202 for eight rows, two of which are scored twice): about 0.35 CPU-hours for a free run and 0.41 for a term run, each with fifteen rows and the common score; 13.3 CPU-hours for Part A (24 free runs and 12 term runs), 13.3 for Part B (the same numbers), about 0.5 for the checks; about 27 in all, or nine hours with three processes.

## The runner

The runner is a new file, `exp58_new_family.py`, in `$M`; no existing source file is edited. `exp48_gated_run.py` and `e48_gated.py` are copied byte for byte from `$W8` (they are not on the default branch; `e48_gated.py` equals `$D:PROJECT_DOCUMENTS/COORD_CHECKS_20261116/e48_gated_spec.py`), with their SHA-256 and source recorded in a text file beside them. The new runner imports `e40_new_family` and `exp48_gated_run` and takes from them, unchanged, the start from a lived network, the stream, the scorer, the record of presented items, the gated forward and its update.

- **The fraction.** `e40_new_family.split` is the only place where the taught pairs are chosen: the stream (`entries`, `batches`), the scorer (`Scorer`, `held_plain`) and the record of presented items (`Presented`) all call it by its module-level name when they are built (`batches` at its first `next`). Before any of them is built the runner replaces it once, `e40_new_family.split = lambda t, seed: nested(t, f, seed)`, with

      def nested(t, f, seed):
          taught, rest = E.split(t, 0.10, seed)     # E40's shuffled list of the 6,889 pairs, cut at 689
          pairs = taught + rest
          k = round(f * len(pairs))
          return pairs[:k], pairs[k:]

  At f = 0.10 this is E40's split itself. At f = 0.05 the taught pairs are the first 344 of E40's 689, and at f = 0.20 they are E40's 689 and the next 689 of the same list. (`exp11_cell.split(t, f, seed)` is not used for the other fractions: the fraction enters its shuffle's seed, and it draws unrelated pairs.) The runner records the fraction in the receipt and asserts in every run, after the stream's entries, the scorer and the record of presented items are built: (a) the number of taught pairs of each operation, 344, 689 or 1,378; (b) that the stream's entries are exactly the taught pairs; (c) that for add and for sub the numbers of items of the scorer's two panels of untaught pairs without letters add to 6,889 minus the number of taught pairs; (d) at f = 0.10, that the taught pairs equal `exp11_cell.split(t, 0.10, seed)[0]`.
- **The common pairs.** For add and for sub of a seed, the common pairs are the pairs that need the reduction, are untaught at f = 0.20 (and so at every fraction), and whose two operand values each occur in the same position in at least one pair taught at f = 0.05. The runner renders them as `held_plain` renders its pairs (`E.render(t, a, b, 0, Random(0))`) and scores them with `X16.score` at every row of every run, after the scorer; the receipt records the two counts and the mean of the two percentages, the *common score*, and the numbers of common pairs (2,491 to 2,718 for an operation and seed).
- **The free arms** make E40's update (`exp26_candidate.step`) with the optimizer `X16.adamw16(m, C.LR, wd, ())`, wd = 0.25 for `free` and `free-again` (which is `exp26_candidate.make_opt`) and wd = 0 for `free-wd0` and `free-anchor`.
- **`free-anchor`** uses the optimizer of `free-wd0`. Before the first update the runner keeps a copy p0 of every parameter except `book.weight`. After each call of `exp26_candidate.step` it executes, under `torch.no_grad()`, `p.sub_(p - p0, alpha=0.00025)` for each of them.
- **Norms.** In every arm the receipt records at every row, for every tensor outside the byte table, its norm and the norm of its difference from the lived network's tensor.
- **The term arms** make E48's update with `e48_gated.make_opt(m, term, wd=0.25)` or `wd=0`.
- **One run, one process.** A free arm calls `arch26.install()` and never `exp48_gated_run.install()`; a term arm calls `exp48_gated_run.install()`. Checks that need both kinds use two processes.
- **E48's fields.** The term arms record E48's row fields `earlier_answers_sha256`, `check3`, `code_rows_tensors_sha256`, `term_norms`, `sweep_cosines` and `term_file`. They do not record E48's `check2` and `check4`, which compare with E40's receipts and are false by construction at another fraction. No file of generated answers is written, and no score is taken with the gate forced off.
- **The optimizer in the receipt** is read back from the optimizer in every arm: with `exp20_reference.opt_groups` in the free arms, and as the `weight_decay`, `lr` and `betas` of each group with the names of its tensors in the term arms. Neither `exp26_candidate.opt_record`, which writes the constant 0.25 beside the groups, nor the constant record of `exp48_gated_run.run` is used.
- **Provenance.** Every receipt records the SHA-256 of `exp58_new_family.py`, `exp48_gated_run.py`, `e48_gated.py` and `e40_new_family.py` beside E40's provenance. Expected: `exp48_gated_run.py` f4f03b01ad5dba650f1d8304ad8547baeacf8279126332cc466e2c66d8d131ac, `e48_gated.py` ab324e9ce4f55b01fa8d8f802b4a1622990c73a70d974727a1f19916e2ccb1a0, `e40_new_family.py` aa11c27f1588abbac6659afc124980b88994f6fbae375ffc742f0d31dbdda9cf.
- **Receipts** are in E40's format with the arm and the fraction: `$M/exp45/e58/e58_<arm>_<rule>_f<05|10|20>_<seed>.json`. The earlier families' six panels are scored at every row in every arm.
- **Files committed:** the receipts; the final weights of every run of `free`, `free-wd0` and `free-anchor` (42 files of 4.5 MB); the term files at rows 1,024 and 8,192 of every term run (48 files of 0.15 MB). `free-again` commits its receipt only. No weights at other rows. The total is about 200 MB; it is pushed with the receipts as the runs finish, not in one push at the end. Weight files are added with `git add -f`, since `.gitignore` excludes `*.pt`.

## Checks before the runs

Each check has a receipt; for a check that is read from the runs, the receipt is a short file under `$M/exp45/e58/qual/` that names the run, the rows compared and the outcome.

1. **`free` is E40's run.** Read from the two `free` runs of seed 73129, which are launched first; no other free run is launched before row 1,024 has passed. At each of E40's rows (64, 128, 256, 512, 1,024, 2,048, 4,096, 8,192) the batch digest, every count of every score and the `tensors_sha256` equal those of `$D:$M/exp45/e40/e40_lived_<rule>_73129.json`. The first comparison is at row 64, and a difference stops every run. (The mean loss since the last row is not compared at rows 64, 4,096 and 8,192, where the new runs have a row between E40's row and the row before it.)
2. **`term` is E48's run.** Read from the two `term` runs of seed 73129 in the same way: at each of E48's rows (0, 16, 32, 64, 128, 256, 512, 1,024) every count of every score, the batch digest and the digests of the earlier families' answers equal those of `$W8:$M/exp45/e48/e48_<rule>_73129_r16.json`, and the five tensors of the term at row 1,024 equal those of `$W8:$M/exp45/e48/e48_<rule>_73129_r16_row1024.pt` bit for bit. Scores with the gate forced off are not taken in E58, and E48's are not compared.
3. **`free-wd0` differs from `free` in the decay only.** For seed 73129 under each rule, after the first update from the same start: for every tensor outside the byte table, the tensor of `free-wd0` minus the tensor of `free` equals 0.00025 times the starting tensor, with a largest absolute deviation below 1e-6 (float32 arithmetic gives about 1e-7; without the decay the deviation would be 6e-6 or more in every tensor); the byte table is bit-equal in the two arms; the optimizer read back shows a weight decay of 0 in every group of `free-wd0`.
4. **`free-anchor`.** For seed 73129 under each rule: (a) applied to a copy of the lived network in which every tensor outside the byte table has been doubled, the decay toward the start leaves each such tensor at 1.99975 times its lived value, with a largest absolute deviation below 1e-6 (the standing decay would leave 1.9995 times, which differs by 6e-6 or more in every tensor), and leaves the byte table bit-equal; (b) applied to the lived network itself it changes no tensor; (c) the optimizer read back shows a weight decay of 0 in every group.
5. **The fractions.** For f = 0.05, 0.10 and 0.20 and each of the three seeds, without a network: the assertions (a) to (d) of "The fraction"; the taught pairs are nested (344 within 689 within 1,378); no item of a panel of untaught pairs has a pair among the taught pairs of its operation, and every item of the taught panel has, the taught pairs being computed here by `nested` directly and not through `e40_new_family.split`; the whole stream of 8,192 batches is walked through `e40_new_family.Presented`, as in E40's check 5, and ends with no untaught pair presented and with every taught pair of every operation presented at least once; the batch digest at every row is recorded, and every run except those of `free-again` asserts its own digests against it; at f = 0.10 the digests equal E40's at E40's rows; the numbers of common pairs are recorded.
6. **The earlier families are unchanged in `term` and `term-wd0`.** At every row of every such run, in Part A and in Part B, the digest of the earlier families' answers equals row 0's, and the hash of the lived network's tensors with the four code rows reset to their starting values equals `tensors_sha256` of the life's receipt (the first and third parts of E48's check 3). The runner asserts both at every row. A failure stops that run and is reported before another term run is launched.
7. **Cost.** CPU-seconds an update and a scored row for each arm from the runs of seed 73129, and a forecast for the whole job. If the forecast exceeds 35 CPU-hours, the worker reports it and waits for a reply before Part B.
8. **`term-wd0` differs from `term` in the decay only.** For seed 73129 under each rule, after the first update: the two `down` factors of `term-wd0` are bit-equal to their starting values, and those of `term` equal 0.99975 times theirs with a largest absolute deviation below 1e-7 (at the first update the gradient of a `down` factor is zero, because its `up` factor starts at zero; E48's receipts record it); the two `up` factors and the four code rows are bit-equal in the two arms; the optimizer read back shows a weight decay of 0 in both groups of `term-wd0`, and of 0.25 and 0 in the groups of `term`.

## How the result is stated (fixed beforehand)

- *R-late* of a run is the mean of its reduction score over the five rows 4,096, 5,120, 6,144, 7,168 and 8,192. *R-mid* is the mean over rows 512 and 1,024. For every run the table gives R-mid, R-late, the highest single row from row 64 on with its row number, and the score at row 8,192 minus the score at row 4,096.
- *The scatter of one arm.* `free-again` differs from `free` only in the order of the items. N, for a seed and a rule, is the absolute difference between R-late of `free-again` and R-late of `free`. If N is 5 points or more on any seed under a rule, the margin below does not separate two runs of one arm under that rule; the report then gives the tables for that rule and states no verdict under it, and a prediction that depends on a verdict under that rule is recorded as not decided. The report gives N for every seed and rule.
- For two sets of three runs X and Y under one rule (one run a seed), X is *above* Y if X's figure exceeds Y's by 5 points or more on at least two seeds and X is not below Y by 5 points or more on the third; *below* is the mirror; otherwise the two are *not distinguished*. The figure is R-late unless another is named. "Not distinguished" is the absence of a verdict, and no sentence of the report is licensed by it alone. The rule is not transitive (differences of 5, 5, −4 and of −4, 5, 5 add to 1, 10, 1), so no verdict is derived from two others; each is read from the per-seed differences of the two sets it names.
- *The ceiling.* On a seed on which the lower of the two figures is above 95, a lead of 5 points is not possible. A comparison in which this holds on two or three seeds is reported as at the ceiling of the score, with the figures and 100 minus each figure, and no verdict is stated.
- From row 1,024 on, the single rows of E40's runs have a standard deviation of 1.0 to 3.3 points within a run. A difference of two means of five rows has a scatter of about 0.6 to 2.1 points if the rows are independent, and of up to 1.4 to 4.7 if neighbouring rows move together; `free-again` measures which holds.

**Part A, for each rule.**

1. If `term` is above `free`, the lead of the gated term persists to 8,192 updates. The report says, for each rule, whether the lead on R-late is larger or smaller than the lead on R-mid and whether `free` fell, `term` rose, or both, between R-mid and R-late. If `term` is not above `free`, the report says that the gated term's lead at 1,024 updates does not persist to 8,192, gives both curves, and makes none of the statements of point 2.
2. If the lead persists, it is split on each seed. L = R-late(`term`) − R-late(`free`); A = R-late(`free-anchor`) − R-late(`free`); P = R-late(`term`) − R-late(`free-anchor`); so L = A + P. A is what the absence of the decay of the lived weights adds when the change is made in every weight and decays in both arms. P is what the place of the change adds when the lived weights decay in neither arm and the change decays in both. The report gives L, A and P for every seed. On each seed on which L is 5 points or more (at least two seeds), the share a = A / L is formed. The report says "the decay of the lived weights accounts for the lead" if a is 0.75 or more on each of these seeds; "the place of the change accounts for the lead" if a is 0.25 or less on each; "both contribute" if a lies between 0.25 and 0.75 on each; otherwise it says that the seeds disagree and gives the table without one of the three sentences.
3. The decay of the change: `term-wd0` against `term`, and `free-wd0` against `free-anchor`, show whether the level changes when the change no longer decays, in either place; `term-wd0` against `free-wd0` compares the two places when nothing decays. These are reported in the words above and enter no sentence of point 2.
4. `free-wd0` against `free` is reported as the effect of removing the weight decay altogether. It is not read as the effect of the decay of the lived weights.
5. For `free`, `free-wd0` and `free-anchor` the report gives, at every row, the mean of the four panels of the life's families (add and sub under 97 and 73), the mean of the two panels of the first family beside it, and the norms recorded in the receipts. These show whether the loss of the earlier families in free learning depends on the decay, and whether, across the three arms and the seeds, the run that keeps more of the life's families also has the higher R-late, as the explanation under test predicts.

**Part B, for each rule.** A table by arm, fraction and seed of R-late, of R-late on the common pairs, and of the taught panel at row 8,192; beside it, for every run, the first row at which the reduction score is 80 or more and the first at which it is 90 or more, with the number of presentations of each taught pair at that row. At each fraction, `term` against `free` and against `free-wd0` in the words above.

Every comparison between two fractions is made on R-late of the score on the common pairs, so that both sides are scored on the same pairs. For a rule and a pair of fractions (f, g) out of (0.05, 0.10), (0.10, 0.20) and (0.05, 0.20), the report says "the gated term reaches from f the level that free learning reaches from g" only if all three of the following hold: (a) `free` at g is above `free` at f, so that free learning gains from the additional pairs; (b) on at least two seeds `term` at f is at most 2 points below the higher of `free` and `free-wd0` at g, and on the third seed it is less than 5 points below; (c) the comparison is not at the ceiling. If (a) fails, the report says that between f and g the level of free learning does not depend on the number of taught pairs, and makes no statement on pairs. If (b) fails, the report gives the deficit by seed.

The same comparisons are tabled across the rules, `term` under the sweep at f against `free` and `free-wd0` under back-propagation at g, because requirement B2 compares our learner with the conventional one; they are reported in the same words.

These statements concern the number of different pairs at an equal number of presented items (524,288 at row 8,192: each taught pair 381 times at f = 0.05, 190 times at 0.10, 95 times at 0.20). They are not statements on the number of items a learner needs; the table of first rows shows that.

## Predictions (the reviewer's, recorded before any run)

| | Statement | Chance |
|---|---|---|
| P58.1 | Checks 1 to 5 and 8 pass | 85% |
| P58.2 | `term` is above `free` at f = 0.10 under the sweep | 65% |
| P58.3 | `term` is above `free` at f = 0.10 under back-propagation | 70% |
| P58.4 | `free-wd0` is above `free` at f = 0.10 under at least one rule | 25% |
| P58.5 | `term-wd0` is not distinguished from `term` under both rules | 60% |
| P58.6 | `free-anchor` is above `free` at f = 0.10 under at least one rule | 25% |
| P58.7 | The mean of the four panels of the life's families at row 1,024 is higher in `free-anchor` than in `free` by 10 points or more on two of three seeds, under at least one rule | 15% |
| P58.8 | At f = 0.05, `term` is above `free` under both rules | 50% |
| P58.9 | For the pair (0.05, 0.10), the report says "the gated term reaches from f the level that free learning reaches from g" under at least one rule | 30% |
| P58.10 | At f = 0.20 under back-propagation, the comparison of `term` with `free` is at the ceiling, or the two are not distinguished | 60% |
| P58.11 | N is below 5 points on every seed under both rules | 75% |

The explanation under test has two parts. The first: a new family is generalised from few pairs by means of what the network learned from the earlier families (a network that was not trained before does not generalise it), and free learning lowers the level it reaches by changing the weights that hold this knowledge. The second: a change confined to connections that only the new family uses therefore reaches a higher level from the same pairs, or the same level from fewer. For this setting, both parts are refuted if `term` is above `free` under neither rule. The second part is refuted, and the first is not, if under both rules the decay of the lived weights accounts for the lead (point 2): free learning then reaches the term's level as soon as the lived weights do not decay. No arm of this job damages the lived weights without learning in them; the reason is given under "What it cannot show".

## What it cannot show

- How a learner would arrive at the separation by itself. The gate is read from the item's code byte by the program that runs the experiment, and the term is supplied from outside. The job measures what such a separation is worth in taught pairs; it says nothing on Mr. Xu's requirement that long-term memory form in the course of learning without our intervention.
- What the decay of the lived weights does when the change is confined to the term. This fourth cell of the design cannot be realised: in free learning the gradient opposes the decay in every weight, and a term on two matrices cannot (a lived network shrunk by the decay of 1,024 updates answers 5 to 31% of its life's families).
- That the gated term needs fewer presented items. Part B compares numbers of different pairs at an equal number of presented items. In E48 the term reached a given level later than free learning (`$V:$R/E48_RESULT_VERIFICATION_20261117.md`, section 6.2).
- More than a factor of four in the number of pairs. A fraction below 0.0436 (300 of 6,889 pairs) cannot be scored by E40's scorer unchanged, because its taught panel samples 300 taught pairs.
- More than one new family, or new families in sequence; fixed memory (each family adds a term).
- Ranks other than 16; a term on other matrices; another rate.
- Whether a level that is still moving at 8,192 updates has reached its end. The curves are given.

## Rules for the worker

- Create your branch from `$D` (7cabb13a). The lived networks, their receipts and E40's receipts are in it; the two files of E48 and E48's receipts are read from `$W8`, this brief and the verification of E48 from `$V`.
- Run at most three processes at a time, each with one thread (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`). Install torch 2.15.0+cpu from `https://download.pytorch.org/whl/cpu` and numpy in a virtual environment; every receipt records the versions.
- **This host stops a machine about five minutes after its session's turn ends, with the disk kept, and detached processes die with it.** While runs proceed, keep your turn open by running a command that waits for the processes; because a command has a time limit, repeat it until the processes have ended, and commit and push the finished receipts between two such commands. A run that was interrupted is started again from its beginning, after its incomplete receipt and any file it wrote have been moved to `$M/exp45/e58/interrupted/` (the runner refuses an output path that exists); the runs are deterministic and at most 0.4 CPU-hours long. The report says which runs were restarted.
- Push only to your own branch. Never push to the default branch or to any other branch, and open no pull request. Other branches are read-only: fetch with `git fetch origin '+refs/heads/<name>:refs/remotes/origin/<name>'` and read with `git show`.
- Write the commit trailers as `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` prescribes. Do not write the name of any language model in a committed file or a commit message.
- Write the report, `E58_REPORT.md`, in standard academic English: full sentences, every term defined at first use, every figure with the receipt or the command that gives it; the committed files are listed from `git ls-tree`.
- If the brief cannot be followed at some point, stop that part, say what you found, and ask.

## Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: checks 3, 4, 5 and 8 passed (or which failed); checks 1 and 2 passed at row 1,024; Part A finished on seed 73129, with its table; Part A finished on three seeds, with the readings; Part B finished on seed 73129, with its table; Part B finished; the report committed. Each message gives your branch, the commit and the main figures in at most ten lines. If you need a decision, ask in the same way and go on with whatever does not depend on the answer.

## Done

The job is done when the eight checks, the 72 runs and `E58_REPORT.md` exist; the report holds the table of every run, the readings of Parts A and B in the words fixed above, the outcome of each prediction, and the list of committed files.

## History of the revisions

Revision 2 enters the twelve findings of the check of revision 1.

| | Finding of the check | What was done |
|---|---|---|
| 1 | The licence for "the gated term needs fewer pairs" held when the number of pairs played no part, and on the absence of a difference | The sentence now needs free learning to gain from more pairs, a deficit of at most 2 points, and no ceiling; comparisons between fractions are made on common pairs; the same comparisons are tabled across the rules |
| 2 | The arm `term-shrink` could not be read: a lived network shrunk as by 1,024 updates of decay answers 5 to 31% of its life's families | The arm is removed; what shrinking does without learning is stated from the check's computation |
| 3 | `free-wd0` removes two pulls at once, and `term` was compared with an arm that differs from it in two things | The arm `free-anchor`, in which the change decays and the lived weights do not; the lead is split on each seed |
| 4 | The margin of 5 points was not measured; the rule is not transitive; the ceiling rule failed where needed | The arm `free-again` measures the scatter of one arm; no verdict is derived from two others; a new ceiling rule |
| 5 | The refutation clause counted as a refutation an outcome that supports half of the explanation | The explanation is stated in two parts with what refutes each; three predictions replaced |
| 6 | The taught pairs of the three fractions were unrelated draws; scores were taken on different pairs; 0.05 leaves operand values untaught; no trial on one seed in Part B | Nested taught pairs; the common pairs; Part B on seed 73129 first |
| 7 | Several checks were incomplete or could not fail | Checks 1, 2, 3, 5, 6 and 7 amended; check 8 added |
| 8 | The runner's description left five things open | Five bullets added to "The runner"; the rule on interrupted runs |
| 9 | Six figures or statements about the record were inexact by 0.1 or in wording | Corrected |
| 10 | References without a branch; the base of the worker's branch; numpy | Entered |
| 11 | The total size of the committed files was not stated | Stated |
| 12 | Language (fragments, undefined terms, metaphors, coined labels) | Entered |
