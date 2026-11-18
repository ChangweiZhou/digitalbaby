# E59: the unrelated task learned by an untrained network, by a network trained on one family and by a network trained on four, from three numbers of taught pairs (17 November 2026, revision 2)

**Status.** The brief was written by the outside reviewer, acting for the coordinator, who cannot be reached while Mr. Xu is away (his instruction of 17 November: arrange it yourself). Revision 1 was checked by an agent of the reviewer that had not seen the reviewer's reasoning (`CHECK_OF_TASK_E59_BRIEF_20261117.md` beside this file). The check found fourteen defects and gave a text for each; revision 2 enters all of them, and the last section lists them. The check is not the independent audit that the programme's rules ask for, and the coordinator has not seen the brief. The job is at the evidence level that the coordinator's briefs call E2: a first measurement, which confirms nothing. It is item 15 of `PLAN.md` beside this file. The job uses three seeds and about 21 CPU-hours. (Revision 1 had the file name `TASK_E59_A_TASK_OF_ANOTHER_KIND_FROM_A_FRESH_A_BORN_AND_A_LIVED_NETWORK_20261117.md`.)

## Shell names and host

`D=origin/codex/minifly-consolidated-20261023` (the default branch, at 7cabb13a); `V=origin/claude/vibrant-cerf-sje726` (the reviewer's branch; this brief and the reviewer's documents are on it); `M=MOD97_LOCAL_RULE_20261106`; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`. The host is Linux on x86-64 with Python 3.13 and torch 2.15.0+cpu.

## Terms

- *Back-propagation* (rule B) is the exact gradient; *the sweep* (rule S) passes the error backward through each linear map by the signs of its weights times their mean size. *Our learner* is the network trained by the sweep, and *the conventional learner* the network trained by back-propagation.
- A *family* is a set of operations under one modulus, or of their compositions, whose items begin with code bytes of their own. *A fresh network* (`fresh`) is `arch26.new_net(seed)` before any update. *A born network* (`born`) is the fresh network after the first stage of E26's training, which the programme calls a birth: 24,576 updates on the first family (add, sub, max and min under modulus 89), `$D:$M/exp45/e26/e26_dev_<S|B>_<seed>.pt`. *A lived network* (`lived`) is the born network after the second stage, which the programme calls a life: six blocks of 6,144 updates each, 36,864 in all, on three further families, `$D:$M/exp45/e26/e26_life_<S|B>_<seed>.pt`. All three have the present size (1,124,113 parameters). *The byte table* is the embedding of the input bytes, `book.weight`.
- *A shuffled network* (`born-shuffled`) is the born network with the values of each tensor other than the byte table put into a random order within that tensor: for each `name, t` of `named_parameters()` other than `book.weight`, `perm = list(range(t.numel())); Random(f"e59|{seed}|shuffle|{name}").shuffle(perm)`, and under `torch.no_grad()` `t.copy_(t.flatten()[torch.tensor(perm)].view_as(t))`. It has the born network's values in every tensor, hence its sizes of weights, and nothing of what the born network computes. It is trained under the rule of the born network from which it is made. The runner asserts that the sorted values of every tensor equal the born network's and that the byte table is unchanged, and records the shuffled network's `tensors_sha256`.
- *The unrelated task* is E43's (`$D:$M/e43_task.py`): an item is a code byte, two symbols out of 64, then 0 to 12 filler bytes and a query byte; the answer is the two symbols and the end symbol. None of its bytes occurs in any family's items. It is of other content than the families and of the same form: it shares with them the start symbol, the end symbol as the third answer byte, the length of an item and the need to carry two items across bytes that do not matter (E43's brief, line 45). Of the 4,096 ordered pairs of symbols, E43 teaches 2,048 and holds 2,048 out.
- A *panel* is a fixed set of items on which a network is scored and never trained. *The held panel* is E43's: 300 held-out pairs at 12 fillers (`e43_task.panel(seed, "held")`). *The held panel at 0 fillers* is the same 300 pairs at 0 fillers (`e43_task.panel(seed, "held", k=0)`). *The taught panel* for n taught pairs is built by `rng = Random(f"e43|{seed}|panel|taught")` and `[e43_task.render(a, b, 12, rng) for a, b in rng.sample(taught[:n], min(300, n))]`. For n = 2,048 it is E43's taught panel (the same call on the same list); for n = 512 it is 300 of the 512 taught pairs; for n = 256 it is all 256 taught pairs. Its counts are given with its number of items.
- The network is scored at update 0, after every 16 updates up to update 512, and after every 32 updates from there to update 8,192. Each scoring is called a row; a run has 273 rows. *Updates to 80* is the first row r at which the share of right answers on the held panel is at or above 80% (240 of 300) and is so at every row up to r + 64. A run with no such row up to update 8,128 *does not reach 80*. A *level* at the end is the mean over the nine rows from update 7,936 to 8,192.
- A curve *has flattened* if the mean of the held panel over the rows from update 7,296 to 8,192 is less than 3 points above its mean over the rows from update 5,248 to 6,144 (29 rows each).
- A *receipt* is the JSON file that a run writes, with its settings, its rows and its hashes. *Admission* is the check of a weights file against its receipt before the file is used. A sentence is *licensed* if the report may state it.

## The question

Mr. Xu stated on 17 November that the end result of our learner may be about the same as that of the conventional learner while ours needs far fewer samples (requirement B2 of `PLAN.md`). The record shows such a saving in one place. A lived network learns a new family of the same form from a tenth of its pairs, which a fresh network does not (E40); the saving is found under both rules against fresh networks; E40 has no network with only the first family behind it, so the record does not say how much of the saving the life adds (`$V:$R/E40_SAVING_IN_WHOLE_EXPERIENCE_20261117.md`). The new family there has the format and the operations of the families already learned. **Does having learned the families also lower what a network needs for a task of other content: the number of presentations, and the number of distinct taught pairs?**

## What the record says

- **E43, the unrelated task from the six born networks, 2,048 taught pairs** (`$D:$M/exp45/e43/e43_unrelated-decay_<rule>_<seed>.json`, weight decay 0.25; right of 300 on the taught and on the held panel; seeds 73129, 93683, 116089). At update 64 nothing is learned (0 to 4 of 300). At update 256: under back-propagation 294 and 294, 289 and 276, 300 and 300; under the sweep 22 and 14, 135 and 91, 124 and 92. At update 1,024 and later every panel is at 300 of 300 under both rules. The rows are 0, 64, 256, 1,024, 2,048 and 4,096, so the update at which 80% is reached is not located more finely than "between 64 and 256" under back-propagation and "between 256 and 1,024" under the sweep. The runs without decay give the same figures to within 4 items.
- **What a born network learns first.** After 64 updates E43's networks of seed 73129 give the form "symbol, symbol, end symbol" on 300 (sweep) and 283 (back-propagation) of 300 held items while the first symbol is right on 3 and 15 (the check's numpy forward on the weights at row 64). How long a fresh network needs for the form is not on record.
- **The sizes of the weights differ between the starts.** The byte table's rows of the task's bytes are bit-equal in the three starts under both rules, for each seed. Every other map differs in size: over the three seeds and both rules, against the initial values, `l1.weight` is 0.21 to 0.48 of its initial size in the born and lived networks, the recurrent matrix 1.64 to 1.98 times, the output matrix 2.4 to 4.6 times and the output bias 8 to 41 times.
- No run of the unrelated task from a fresh network or from a lived network exists, and none with another number of taught pairs.
- **The related case, for comparison** (E40): on a new family of the same form, from a tenth of its pairs, the lived networks reach 77 to 94% on untaught pairs of add and sub whose answer needs the reduction by the modulus, and the fresh networks 6 to 12%, after 8,192 updates. On a family, fresh networks of this size stay below 10% on the taught panel for the first 1,024 updates under both rules.

## Runs

| Start | Taught pairs | Rules | Seeds | Updates | Runs |
|---|---|---|---|---|---|
| `fresh`, `born`, `lived` | 2,048 (E43's); 512; 256 | S, B | 73129, 93683, 116089 | 8,192 | 54 |
| `born-shuffled` | 2,048 | S, B | 73129, 93683, 116089 | 8,192 | 6 |

Each run continues its start under the rule that made it (`fresh` under the rule named), with a new optimizer `exp26_candidate.make_opt(m)` (AdamW, rate 0.001, weight decay 0.25 on every tensor but the byte table), in batches of 64, as E43's arm `unrelated-decay`.

**The taught sets are nested.** With `taught, held = e43_task.split(seed)`, the taught set of size n is `taught[:n]`. The held panel is the same for every n. The stream for n pairs is the text of `e43_task.stream` with `taught[:n]` in place of `taught`: a generator `Random(f"e43|{seed}|stream")`, the pairs in a new shuffled order each pass, for each item `randint(0, 12)` fillers. With n = 2,048 it is E43's stream. (With 512 pairs every symbol occurs at least twice in each place of the taught pairs, for the three seeds. With 256 pairs a symbol occurs in a place 0 to 10 times. Symbols 131 and 139 never stand in the first place and 167 never in the second for seed 73129; 171 and 152 for seed 93683; none and 186 for seed 116089. The held panel has 11, 6 and 2 items with a symbol in a place in which no taught pair has it. For n = 256 the receipts give the held panel's counts in two parts, those items and the remaining 289, 294 and 298; updates to 80 and the levels are taken on the whole panel of 300.)

The order is: checks 1 to 3; the runs of seed 73129, of which the first three are those of check 4; the other two seeds. After seed 73129 the worker sends its table and goes on, unless a loss was not finite, a check failed, or `fresh` is at 0 on the taught panel at the end of each of its six runs; in those cases the worker stops and asks.

Cost: E43's own runs took 0.100 to 0.110 CPU-seconds an update on this task and E40's 0.104 to 0.119 at this size; a row scores 900 items (the two held panels and the taught panel) and a run has 273 rows. About 0.35 CPU-hours a run, 21 CPU-hours for the 60 runs. The check of cost measures both before the runs.

## The runner

The runner is a new file, `$M/exp59_unrelated_starts.py`; no existing source is edited. The runner calls `arch26.install()` first, as E43's `run` does. Training draws no random number, so no call of `T.manual_seed` is needed beyond the one inside `arch26.new_net`. It takes from `exp43_unrelated.py` and `e43_task.py`, unchanged, the update (`exp26_candidate.step`), the held panel and the assertion that every prompt byte of a batch is 128 or above; from `e40_new_family.start` the loading of a fresh and of a lived network; and E43's admission of the born networks, `exp38_erase.admit(bfile, rule, seed)` (line 125 of `exp43_unrelated.py`); it calls `exp26_candidate.admit_e26` and carries the one recorded exception for `exp16_integrated.py`, without which `admit_e26` fails on the default branch.

It does not score the first family or the life's families at every row; it scores them once, at row 0 and at the last row, so that the report can give, without a reading, what remained of the earlier families. Both scores come from one call, `X16.panels(seed)`: E43's score 2 is `P[n][12]` for the four operations of the first family, and E40's six earlier panels are `P[n][12]` for `add89`, `sub89`, `add97`, `sub97`, `add73` and `sub73`; `add89` and `sub89` are in both and are held once.

Each run writes a receipt, a JSON file `$M/exp45/e59/e59_<start>_n<2048|512|256>_<rule>_<seed>.json`, which holds the start's file and hashes, the optimizer's groups as read back, the SHA-256 of the new runner and of the files it takes from (`exp43_unrelated.py`, `e43_task.py`, `e40_new_family.py`, `exp38_erase.py`, `arch31_workerB.py`), and at every row: the `data_digest` (`exp26_candidate.Digest`), the `tensors_sha256`, the mean loss since the last row, the counts of the held panel at 12 fillers, of the held panel at 0 fillers and of the taught panel, and, on the held panel at 12 fillers, the number of items on which the first answer byte is right, on which the second is right, and on which the third byte is the end symbol. Final weights are committed for the runs of seed 73129 (4.5 MB each); those of the other runs are kept on the disk as the rules for the worker say.

## Checks (a receipt for each)

1. **The stream and the panels** (before any run; nothing is trained). For each seed: (a) the digest of the first 64 batches of the runner's stream with n = 2,048, computed by the lines of `e43_task.digest`, equals `e43_task.digest(seed, 64)`, which begins `34eefb4442b48399`, `153cafdcda7379d0` and `821bcb2a9ca6dd54` for the three seeds (`$D:$M/exp45/e43/checks/e43_check_3_spec.json`); (b) the runner's held panel, and its taught panel with n = 2,048, equal `e43_task.panel(seed, "held")` and `e43_task.panel(seed, "taught")` item for item; (c) for each n, the `sha256` of the runner's `data_digest` (`exp26_candidate.Digest`) after 64 and after 8,192 batches equals the value listed below; (d) over the 8,192 batches the set of pairs presented is exactly `taught[:n]`. A difference stops the job.
2. **The starts** (before any run). `fresh`: the `tensors_sha256` of the start equals `start.tensors_sha256` in `$D:$M/exp45/e40/e40_fresh_<rule>_<seed>.json`. `born`: admitted by `exp38_erase.admit`, and the `tensors_sha256` of the loaded network equals `settings.birth_tensors_sha256` in `$D:$M/exp45/e43/e43_unrelated-decay_<rule>_<seed>.json`. `lived`: `e40_new_family.start("lived", rule, seed)` passes its assertion on the file's hash, and the runner asserts `info["tensors_sha_matches_receipt"]`, which `start` records and does not assert. `born-shuffled`: the assertions of its definition. For each seed, the rows 128 to 225 of the byte table are equal in every start under both rules. The counts of every start on the two held panels and on the taught panel at row 0 are recorded; for `born` and `lived` they are 0 (computed in the check of this brief). A difference stops the job.
3. **Cost.** A run of 64 updates of `fresh` with 256 pairs under the sweep, written to `$M/exp45/e59/dry/`, gives the CPU-seconds of an update and of a row and a forecast for the 60 runs. If the forecast is above 28 CPU-hours, stop and ask.
4. **`born` with 2,048 pairs is E43's run.** The two such runs of seed 73129 are launched first, with one run of `fresh` beside them; no other run is launched until their rows 64 have been compared. At rows 64, 256, 1,024, 2,048 and 4,096 the `data_digest`, the counts of the held panel at 12 fillers and of the taught panel, and the `tensors_sha256` equal those of `$D:$M/exp45/e43/e43_unrelated-decay_<rule>_73129.json`. The runner makes the same comparison in the `born` runs with 2,048 pairs of the other two seeds. A difference at any row stops every run. If the `tensors_sha256` alone differs while the digest and the counts agree, say so in the message, since the cause may then be the host.
5. **The same batches.** At every row the `data_digest` is equal across the runs of a seed and a number of pairs. It is read from the receipts when a seed is finished.

The values for check 1 (c), in the order of the seeds 73129, 93683, 116089 (from the check's numpy copy of `exp26_candidate.Digest`, which reproduces E43's `data_digest` at five rows on three seeds):

```
n = 2,048, row 8,192   bbea8212a5c8f1d57cc462f4c5a9992e77a8013ea6692cb06ac801a6f78a5b67
                       9c4e76b18ad899a8a463adb2d4b2b9a06ab1210ee54010c8606134afa333c71f
                       8760e2485a81c98136b9f06f98a331ceb8581fee2b2f791be70aa3ba3495b70f
n = 2,048, row 64      the data_digest.sha256 at row 64 of E43's receipt of the seed
n = 512, row 64        4dd6f47484f72cfefc5138cc4298e402a234cb745eac606cb379541c63e762ac
                       05d69b502bca7e4dc268ba38599609bbf3605e15cff549a7324b28eec2e58e98
                       8312268753e6452949ab5dae9929de0ba6d3149440e56ba99b58e5e173da16c5
n = 512, row 8,192     db66d453988b1810e56d00b3195a2166ba6f6d19b9e8b0c5b069abcfbd570d7e
                       d18a695db39003851631b5fdc23054e893072663d0dae5422a0fe0fab35963a1
                       4c5baac78565f0ee022730a119a3227246a1a750110f8a043060cfd17e22fb25
n = 256, row 64        1ad9aaaf93b0ea7d10bce78230ce810a563dcfe6b456dc205d15afaa32dc6694
                       f9782a97b7e5f72e105ff95d7e57ac8306743f514e8936be9762735ee68a899b
                       a49e561f0d6f5a8f143e063b72982d4bd7667e19c6c01a64951d193088f8cf0a
n = 256, row 8,192     e61f65514d935cf788d3382b430a3e2a637eb1eab6534b9388fee599a2edc49e
                       a717d2cb247d57fad3c6bc4e511d7d1a096182795826ca4b2ad6dab2fca6d543
                       cee2a5968c59c4b2af3a9850735b2181cd32dd81df1f6b97f734008bcd43e31e
```

## How the result is stated (fixed beforehand)

For each rule and each number of taught pairs there are three comparisons: `born` against `fresh`, `lived` against `fresh`, `lived` against `born`. In "X against Y" the first start is X.

- **Presentations.** Updates to 80 on the held panel, a seed at a time. In a comparison of a first start against a second, let a and b be the updates to 80 of the first and of the second, and let L = 8,128 be the last row at which updates to 80 can lie. The seed's word is: where both reach 80, *fewer* if a/b is below 2/3 and b − a is at least 48, *more* if a/b is above 1.5 and a − b is at least 48, otherwise *about the same*; where only the first reaches 80, *fewer* if a/L is at most 2/3, otherwise *not decided*; where only the second reaches 80, *more* if L/b is at least 1.5, otherwise *not decided*; where neither reaches 80, *neither reaches*. These are the four cases of `describe` in `e42_tables.py` (lines 63 to 72, on `origin/claude/gallant-dijkstra-i9zgsr`) with the second start as its reference, the first as its candidate and L as the length of the run on both sides, as `thr_desc` in that file shortens it. The comparison's word is *fewer* if at least two seeds say *fewer* and the third does not say *more*; *more* if at least two say *more* and the third does not say *fewer*; *about the same* if at least two say *about the same*; otherwise *not decided*. The report gives the three seeds' values and words beside every comparison's word. The three comparisons of a rule and a number of pairs need not form an order: where *fewer* holds for `lived` against `born` and for `born` against `fresh` and not for `lived` against `fresh`, the report says so and draws no order of the three starts.
- **Pairs.** For a rule, a number n of taught pairs and a seed, a trained start (`born` or `lived`) *saves pairs against `fresh`* if all of these hold: (i) the trained start reaches 80 within the run and its level at the end on the held panel is at or above 80; (ii) `fresh` does not reach 80 within the run; (iii) the curve of `fresh` has flattened; (iv) the level of `fresh` on the taught panel at the end is at or above 95, so that it has learned the pairs it was shown; (v) the level of `fresh` on the held panel at the end is at least 10 points below that of the trained start. The sentence "a network that has learned the first family needs fewer taught pairs for the unrelated task than a fresh network" is licensed under a rule if, for some n, `born` saves pairs against `fresh` on at least two seeds and, on the third, the level of `fresh` on the held panel at the end is not above that of `born`. For `lived` the sentence reads "the four families" and the condition is the same. The sentence is stated for the start for which it holds and with its n; it is not stated of "the families" when it holds for one start only. The other outcomes are stated in these words. Where (i) and (ii) hold on at least two seeds and (iii) fails on one of them: "reaches 80 from n pairs within the run, and the fresh network does not; whether the fresh network would reach 80 with more updates is not decided". Where (i) and (ii) hold and (iv) fails: "the fresh network has not learned its taught pairs within the run; the difference is one of presentations, and the number of pairs is not compared". Where every start reaches 80 from 256 pairs on at least two seeds: "every start generalises from the smallest number of pairs tested; no difference in the number of pairs is shown, and none is excluded below 256". Where `fresh` meets (i) and a trained start meets (ii) to (v) in its place: the same sentence with "more" for "fewer".
- **The source of a saving** (2,048 pairs, for each rule). If `born` against `fresh` is *fewer*, the saving is ascribed to what the first family taught if `born` against `born-shuffled` is also *fewer*; it is ascribed to the sizes of the weights that training leaves if `born-shuffled` against `fresh` is *fewer* and `born` against `born-shuffled` is *about the same*; otherwise the three words are given and no source is named. Beside it, for each start and seed: the first row at which the third answer byte is the end symbol on 240 of the 300 held items; updates to 80 on the held panel at 0 fillers; updates to 80 at 12 fillers. A saving that is as large at 0 fillers as at 12 does not come from holding items across fillers, and the report says so.
- **The two rules.** For each start and each number of taught pairs, the sweep against back-propagation (the first is the sweep), in the words of "Presentations" and of "Pairs", with the two levels at the end. And, as the case that requirement B2 describes, `born` and `lived` under the sweep against `fresh` under back-propagation, in the same words.
- **The account.** Every count of presentations and of pairs in this job is of the unrelated task alone. Beside each reading for `born` and `lived` the report gives what the start was given before: 24,576 updates (1,572,864 presented items) for `born`, and 61,440 updates (3,932,160 items) for `lived`.
- **Levels** of the two held panels and of the taught panel at the end, for every run.
- The report also gives, and does not interpret, the counts at row 0 and at the last row on the panels of the first family and of add and sub of the life's two families of single operations.

## Predictions (the reviewer's, recorded before any run)

| | Statement | Chance |
|---|---|---|
| P59.1 | Checks 1 to 4 pass | 85% |
| P59.2 | With 2,048 pairs, `born` against `fresh` is *fewer*, under both rules | 70% |
| P59.3 | With 2,048 pairs, `lived` against `born` is *about the same*, under both rules | 50% |
| P59.4 | With 256 pairs, `fresh` reaches 80 on at most one seed under the sweep and on at most one seed under back-propagation | 45% |
| P59.5 | The sentence on pairs is licensed for `born` or for `lived`, at some n, under at least one rule | 25% |
| P59.6 | With 2,048 pairs, the seed's word for `fresh` under the sweep against `fresh` under back-propagation is *more* on at least two seeds. (The record points both ways: fresh networks on the first family at a half of its pairs give ratios of 1.10, 1.19 and 1.24, and E43's born networks reach 80% by 256 updates under back-propagation and between 256 and 1,024 under the sweep) | 40% |
| P59.7 | With 2,048 pairs, `born` against `born-shuffled` is *fewer*, under both rules | 70% |
| P59.8 | With 2,048 pairs, `born-shuffled` against `fresh` is *about the same*, under both rules | 35% |
| P59.9 | With 2,048 pairs, `lived` under the sweep against `fresh` under back-propagation is *fewer* | 55% |

The explanation under test: what a network gains from the families is in part an ability that does not depend on their content, namely to hold two items across bytes that do not matter and to reproduce them; a task of other content that asks for the same ability is then learned from fewer presentations, and perhaps from fewer pairs, by a network that has it. A saving of `born` or `lived` over `fresh` does not by itself support it, because the starts also differ in the sizes of their weights and in whether the answer's form is known. It is supported, for this task and no further, if at 2,048 pairs `born` needs *fewer* than `fresh` and *fewer* than `born-shuffled`, and the saving against `fresh` (the updates to 80 of `fresh` less those of `born`) is larger at 12 fillers than at 0 fillers on at least two seeds, under both rules; where `fresh` reaches 80 at 0 fillers and not at 12, the saving at 12 fillers counts as the larger. It is refuted, for this task, if under both rules no comparison of `born` or of `lived` against `fresh` says *fewer*, at any number of pairs, and the two comparisons at 2,048 pairs say *about the same* or *more*. Where a comparison at 2,048 pairs is *not decided* or *neither reaches*, the explanation is not tested under that rule.

## What it cannot show

The job cannot show the following.

- A task that shares nothing with the families. This one shares the format of an item, the length of the answer and the need to carry two items across other bytes (E43's brief says so). A saving here is a saving on a task of other content and not of another kind in every respect.
- What the optimizer's new start costs each start. Every run begins with a new AdamW, whose first steps move every weight by about 0.001 whatever the gradient; that is a larger share of the weights of a born or lived network in some maps and a smaller share in others than of a fresh network's (`l1.weight`: root mean square 0.05 against 0.20; the output matrix: 0.06 to 0.08 against 0.03).
- Whether a difference between `lived` and `born` comes from the further families or from the further updates.
- What a learner keeps of the families while it learns the new task; the two extra scores are given without a reading, and E43 and E51 measure that question.
- More than one new task; other numbers of taught pairs than three; other sizes of network.
- Why the sweep and back-propagation differ, where the comparison of the two rules shows that they do.
- A graft of the born network's recurrent weights into a fresh network, or the reverse. The parts were trained together, and a graft that fails to save could not be read; it may follow as a job of its own if this one finds a saving.

## Rules for the worker

- Start a branch of your own from `$D` at 7cabb13a. This brief and the reviewer's documents are on `$V`; read them with `git show`.
- Run at most three processes at a time, each with one thread (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`). Install torch 2.15.0+cpu from `https://download.pytorch.org/whl/cpu` in a virtual environment. The virtual environment needs numpy beside torch; its version is recorded in every receipt.
- **This host stops a machine about five minutes after its session's turn ends, with the disk kept, and detached processes die with it.** Do not end your turn while runs proceed, since the host counts a session as idle once the agent's turn has ended: run a command that waits for the processes and, because a command may run for a limited time only, repeat it until they have finished; commit and push the finished receipts between two such commands. At three processes the runs take about seven hours by the clock; the turn stays open for all of that time.
- The runner refuses an output path that exists. Before a run is started again, move its partial receipt to `$M/exp45/e59/interrupted/` and commit it; do not delete it. The run started again must give, at every row that the partial receipt holds, the same `data_digest` and `tensors_sha256`, and the report says for each such run that it did.
- `.gitignore` at the root excludes `*.pt`. Add the weights that this brief asks for with `git add -f` and confirm with `git ls-tree` that the commit holds them. The final weights of the other runs go to `$M/exp45/e59/rows_uncommitted/`, with their SHA-256 in the receipt, and stay on the disk until the reviewer has verified the result; the reviewer may name runs whose weights are then committed.
- Push only to your own branch. Never push to the default branch or to any other branch, and open no pull request. Other branches are read-only: fetch with `git fetch origin '+refs/heads/<name>:refs/remotes/origin/<name>'` and read with `git show`.
- Write the commit trailers as `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` prescribes. Apart from the trailers, do not write the name of any language model in a committed file or a commit message.
- The report is `$M/exp45/e59/E59_REPORT.md`, written in standard academic English: full sentences, every term defined at first use, every figure with the receipt or the command that gives it; the committed files are listed from `git ls-tree`.
- If the brief cannot be followed at some point, stop that part, say what you found, and ask.

## Reporting

Send the reviewer a short message with the tool `send_message` of the claude-code-remote server, to session `session_01YBMaMvtGtQvDFcgppgThrZ`, at each of these points: checks 1 to 3 passed (or which failed); check 4 passed at row 4,096; seed 73129 finished, with its table; all three seeds finished, with the readings; the report committed. Each message gives your branch, the commit and the main figures in at most ten lines. If you need a decision, ask in the same way and go on with whatever does not depend on the answer.

## Done

The job is done when the five checks and the 60 runs have their receipts and `E59_REPORT.md` holds the table of every run, the readings in the words fixed above, the outcome of each prediction and the list of committed files.

## History of the revisions

Revision 2 enters the fourteen findings of the check of revision 1.

| | Finding of the check | What was done |
|---|---|---|
| 1 | The licence for the sentence on pairs passed a fresh network that had learned nothing, and turned on one seed and on one point | Five conditions for "saves pairs"; fixed sentences for the other outcomes |
| 2 | The reading of presentations called unmeasured comparisons "about the same", cited E42's rule with the wrong length of run, and could contradict itself | A word for each seed and a rule for the comparison; the refutation rewritten |
| 3 | The explanation could not be told from two others (the sizes of the weights; the form of the answer); the title claimed more than the task gives | The start `born-shuffled`; counts by place of the answer; a held panel at 0 fillers; "the unrelated task" |
| 4 | Two spacings of the rows made two criteria; under back-propagation the born network's rise lay within six rows | Rows every 16 updates to 512 and every 32 after; updates to 80 defined by a span of 64 updates |
| 5 | The function named for the admission of the born networks fails on the default branch | `exp38_erase.admit` |
| 6 | The taught panel could not be built for 512 and 256 pairs | One construction for the three sizes |
| 7 | Weights would have been lost and interrupted runs could not be started again; the starting commit was not named | Rules for the worker |
| 8 | Several checks could not fail or were not gated | Five checks in the order in which they are carried out, with reference digests |
| 9 | The comparison of the two rules and the account of what came before were left out | Two bullets in "How the result is stated" |
| 10 | The chances given for 256 pairs were wrong; the realised sets are known | The realised sets; counts in two parts |
| 11 | Four statements about the record were inexact | Corrected |
| 12 | Three predictions were ambiguous or forced; predictions for the control and the two rules were missing | Reworded; P59.7 to P59.9 added |
| 13 | Smaller points | Entered |
| 14 | Language | Entered |
