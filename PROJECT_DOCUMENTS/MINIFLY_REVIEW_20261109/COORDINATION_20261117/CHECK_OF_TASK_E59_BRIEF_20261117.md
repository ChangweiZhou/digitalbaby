# Check of the brief of E59 before its release (17 November 2026)

Written for the outside reviewer by an agent that has not seen the reviewer's reasoning. The brief is `TASK_E59_A_TASK_OF_ANOTHER_KIND_FROM_A_FRESH_A_BORN_AND_A_LIVED_NETWORK_20261117.md` beside this file, revision 1, as committed at 1ed3d41c (its SHA-256 begins `0111e02f`; it was unchanged when this check ended). The work was read-only. Nothing was trained, torch is not installed, no code of another branch was executed, and the files of other branches were read with `git show`. The check began at 22:22 UTC and ended at @@END@@ UTC. It is the check that rule 4 of section 4 of `PLAN.md` asks for. It is not the independent audit that the programme's rules ask for.

**Shell names.** `D=origin/codex/minifly-consolidated-20261023` (at 7cabb13a); `M=MOD97_LOCAL_RULE_20261106`. A line of the brief is cited as "brief, line n". `APP <part>` is the script of the appendix, run as `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I <the script> <part>`; it reads every file with `git show`, and uses the standard library and, for three parts, numpy.

**Terms.** As in the brief: `fresh`, `born` and `lived` are its three starts; the two rules are the sweep (S) and back-propagation (B); the held panel is 300 held-out pairs at 12 fillers. The seeds are given in the order 73129, 93683, 116089. A *receipt* is the JSON file that a run writes.

## Verdict

**Release after the listed changes; do not release as it stands.**

- The figures of the record are right apart from three (defects 10 and 11), and two statements about the record go beyond it (defect 11). The runner can be built from the sources named, and a run of `born` with 2,048 pairs would be E43's run bit for bit in principle. The three starts and their receipts are on the default branch and agree with their hashes.
- Defects 1 to 4 would lead to a wrong or an empty conclusion. The licence for the sentence on pairs passes a fresh network that has learned nothing. The reading of presentations labels unmeasured comparisons "about the same", and the explanation can then be refuted by runs that measured nothing. The explanation under test cannot be told from two others with the runs as specified. The two spacings of the rows make "three consecutive rows" two different criteria on the two sides of the main comparison, and under back-propagation the rise of the born network lies within six rows.
- Defects 5 to 7 stop the job or lose its output. The function named for the admission of the born networks fails on the default branch. The taught panel cannot be built as described for 512 and for 256 pairs. Weights are excluded from commits by `.gitignore`, and an interrupted run cannot be started again while its partial receipt exists.
- Defects 8 to 14 are the checks, what requirement B2 needs, the figures, the predictions, smaller points and the language.

If the texts below are entered, I see no need for a second check of the whole brief. The reviewer should read the new section "How the result is stated" once as a whole, since defects 1, 2, 3 and 9 all change it.

## Defects, the most serious first

### 1. The licence for the sentence on pairs passes a fresh network that has learned nothing, and turns on one seed and on one point

**What is wrong.** The brief (line 58) licenses "Having learned the families lowers the number of pairs needed for a task of another kind" if, for some n, `born` or `lived` reaches 80 on two of three seeds, `fresh` does not, and the curves of `fresh` "have flattened" (line 16: the mean of the held panel over the rows 7,296 to 8,192 is less than 3 points above the mean over the rows 5,248 to 6,144).

(a) *A curve that stays at chance has flattened.* If the held panel of `fresh` is at 0 throughout, the difference of the two means is 0, the curve "has flattened", and the sentence is licensed. Nothing then shows that pairs are what the fresh network lacks. If its taught panel is low as well, it has not learned the task within the run, and the difference between the starts is one of presentations. The record makes this outcome plausible: fresh networks on a family have the taught panel at 5.7 to 9.3% after 1,024 updates and at 57.7 to 87.0% after 2,048, under both rules (E40, `APP flat`). The rule is sound where the fresh network has learned its taught pairs. I applied it at update 8,192 to the 23 curves of fresh networks in E41 that are below 80 there and run on to 12,288 or 40,960 updates: it calls 9 flattened, all 9 have the taught panel at 96.8% or more, one of the 9 is 6 points higher at the end of its run and none is 10 points higher (`APP flat`). The brief does not ask for that condition.

(b) *One seed decides.* "Generalises" is two of three seeds and "does not" is one or none, so the sentence can rest on one seed of one start.

(c) *One point decides.* A fresh curve that is flat at 79% beside a born curve at 81% licenses the sentence.

(d) *"`born` or `lived`".* The sentence speaks of "having learned the families" whichever start earns it. If `born` generalises from n pairs and `lived` does not, the sentence would be stated although more learning removed the saving.

(e) *Outcomes without words.* If every start generalises from 256 pairs, "not licensed" will be read as "no saving", although the number of pairs needed was then below the smallest number tested for every start. The brief's own prediction P59.4 gives 45% to `fresh` generalising from 256 pairs under at least one rule. The reverse outcome (`fresh` generalises, a trained start does not) has no sentence either. A run that reaches 80 and then falls back counts as generalising although its level at the end is below 80.

**Repair.** Replace the bullet "Pairs" (brief, line 58) by:

> - **Pairs.** For a rule, a number n of taught pairs and a seed, a trained start (`born` or `lived`) *saves pairs against `fresh`* if all of these hold: (i) the trained start reaches 80 within the run and its level at the end on the held panel is at or above 80; (ii) `fresh` does not reach 80 within the run; (iii) the curve of `fresh` has flattened; (iv) the level of `fresh` on the taught panel at the end is at or above 95, so that it has learned the pairs it was shown; (v) the level of `fresh` on the held panel at the end is at least 10 points below that of the trained start. The sentence "a network that has learned the first family needs fewer taught pairs for the unrelated task than a fresh network" is licensed under a rule if, for some n, `born` saves pairs against `fresh` on at least two seeds and, on the third, the level of `fresh` on the held panel at the end is not above that of `born`. For `lived` the sentence reads "the four families" and the condition is the same. The sentence is stated for the start for which it holds and with its n; it is not stated of "the families" when it holds for one start only. The other outcomes are stated in these words. Where (i) and (ii) hold on at least two seeds and (iii) fails on one of them: "reaches 80 from n pairs within the run, and the fresh network does not; whether the fresh network would arrive is not decided". Where (i) and (ii) hold and (iv) fails: "the fresh network has not learned its taught pairs within the run; the difference is one of presentations, and the number of pairs is not compared". Where every start reaches 80 from 256 pairs on at least two seeds: "every start generalises from the smallest number of pairs tested; no difference in the number of pairs is shown, and none is excluded below 256". Where `fresh` meets (i) and a trained start meets (ii) to (v) in its place: the same sentence with "more" for "fewer".

### 2. The reading of presentations labels unmeasured comparisons "about the same", cites E42's rule with the wrong run length, and can contradict itself

**What is wrong.** The brief (line 57) reads the ratio of two "updates to 80" as *fewer*, *more*, "otherwise *about the same*", and sends the case in which one start does not reach 80 to `describe` in `e42_tables.py`.

(a) *"Otherwise" takes in what was not measured.* `describe` (lines 63 to 72, as cited; verified) returns one of five words for a seed: "fewer", "more", "about the same", "not decided", "neither reaches". The brief's rule of two seeds in three is written on ratios and says nothing of a seed without a ratio. Read as written, a comparison in which neither start reaches 80 on any seed is *about the same*. The refutation (brief, line 73) is "with every number of pairs `born` and `lived` are *about the same* as `fresh` or need *more*": if no start reaches 80 from 256 or from 512 pairs, those comparisons count toward the refutation although nothing was measured.

(b) *The run length.* The brief says "with a run of 8,192 updates on both sides". With three consecutive rows the last row at which "updates to 80" can lie is 7,936 (rows 7,936, 8,064, 8,192; `APP task`). A run that does not reach 80 is known to need more than 7,936 updates, not more than 8,192. E42's own table function shortens the run by two rows for this criterion (`thr_desc`, lines 79 and 80 of the same file: `dose - 512`); the brief cites `describe` without it.

(c) *The two sides.* `describe(xr, xc, Dr, Dc)` divides the candidate by the reference (`q = xc / xr`). The brief does not say which start is which.

(d) *The three comparisons can contradict each other.* With updates to 80 of (`fresh`, `born`, `lived`) = (1,000, 600, 840), (1,000, 600, 360), (1,000, 1,400, 840) on the three seeds, the brief's rule gives: `born` needs fewer than `fresh` (ratios 0.60, 0.60, 1.40); `lived` needs fewer than `born` (1.40, 0.60, 0.60); `lived` needs about the same as `fresh` (0.84, 0.36, 0.84) (`APP votes`). A vote of two seeds in three is not transitive.

(e) *A difference of two rows is called "fewer".* With rows 32 updates apart, 96 against 160 is a ratio of 0.60 (see defect 4).

**Repair.** Replace the bullet "Presentations" (brief, line 57) by the text below. It is written for the rows of defect 4; if the rows of the brief are kept, L is 7,936 and the least difference is 96.

> - **Presentations.** Updates to 80 on the held panel, a seed at a time. In a comparison of a first start against a second (in "`born` against `fresh`" the first is `born`), let a and b be the updates to 80 of the first and of the second, and let L = 8,128 be the last row at which updates to 80 can lie. The seed's word is: where both reach 80, *fewer* if a/b is below 2/3 and b − a is at least 48, *more* if a/b is above 1.5 and a − b is at least 48, otherwise *about the same*; where only the first reaches 80, *fewer* if a/L is at most 2/3, otherwise *not decided*; where only the second reaches 80, *more* if L/b is at least 1.5, otherwise *not decided*; where neither reaches 80, *neither reaches*. These are the four cases of `describe` in `e42_tables.py` (lines 63 to 72, on `origin/claude/gallant-dijkstra-i9zgsr`) with the second start as its reference, the first as its candidate and L as the length of the run on both sides, as `thr_desc` in that file shortens it. The comparison's word is *fewer* if at least two seeds say *fewer* and the third does not say *more*; *more* if at least two say *more* and the third does not say *fewer*; *about the same* if at least two say *about the same*; otherwise *not decided*. The report gives the three seeds' values and words beside every comparison's word. The three comparisons of a rule and a number of pairs need not form an order: where *fewer* holds for `lived` against `born` and for `born` against `fresh` and not for `lived` against `fresh`, the report says so and draws no order of the three starts.

Replace the last sentence of the paragraph on the explanation (brief, line 73) by:

> It is refuted, for this task, if under both rules no comparison of `born` or of `lived` against `fresh` says *fewer*, at any number of pairs, and the two comparisons at 2,048 pairs say *about the same* or *more*. Where a comparison at 2,048 pairs is *not decided* or *neither reaches*, the explanation is not tested under that rule.

### 3. The explanation under test cannot be told from two others with the runs as specified, and the title claims more than the task gives

**What is wrong.** The explanation (brief, line 73) is that a network gains from the families "an ability that does not depend on their content, namely to hold two items across bytes that do not matter and to give them back". The only outcome named is the refutation. If `born` needs fewer updates than `fresh`, which the brief expects at 70% (P59.2), the result will be read as support. Three things besides that ability differ between the starts, and the runs do not separate them.

(a) *The size of the weights.* The byte table's rows of the task's bytes (128 to 225) are bit-equal in the three starts and under both rules, for each seed (`APP weights`; the fresh rows are read from E40's fresh runs, in which they receive no gradient and no decay). So no start has an advantage in the code of the new bytes, and that part of the comparison is clean. Every other map differs in size. For seed 73129 the root mean square of the entries is:

| Tensor | `fresh` (after 64 updates of E40) | `born` S, B | `lived` S, B |
|---|---|---|---|
| `l1.weight` (reads the byte table) | 0.201 | 0.051, 0.054 | 0.098, 0.050 |
| `rnn.weight_hh_l0` (recurrent) | 0.038 | 0.064, 0.059 | 0.071, 0.065 |
| `lm.weight` (state to readout) | 0.038 | 0.072, 0.066 | 0.078, 0.069 |
| `l3.weight` (output) | 0.029 | 0.072, 0.061 | 0.080, 0.069 |
| `l3.bias` | 0.029 | 0.346, 0.237 | 0.415, 0.309 |

Over the three seeds and both rules, against torch's initial values, `l1.weight` is 0.21 to 0.48 of its initial size, the recurrent matrix 1.64 to 1.98 times, the output matrix 2.4 to 4.6 times and the output bias 8 to 41 times. A fresh network is therefore another network in the scale of every map, and under AdamW at a fixed rate a step of 0.001 is a different share of a weight in each start (2% of the root mean square of `l1.weight` in `born`, 0.5% in `fresh`). All three starts get a new optimizer, whose first steps move every weight by about the rate whatever the gradient (E43's report says so of its own runs, `$D:$M/exp45/e43/E43_REPORT.md`, line 118): the same optimizer is not the same disturbance for the three.

(b) *The slow start of any fresh network.* On a family, fresh networks of this size stay below 10% on the taught panel for the first 1,024 updates under both rules (E40; in E41 at a half of the pairs the rise begins between 1,500 and 2,300 updates; `APP flat`). A trained network may pass this stage at once whatever it was trained on. A ratio far below 2/3 would then say little about holding two items.

(c) *The form of the answer.* Before any update the born and the lived networks do not answer in the task's form: on the held panel they mostly give the end symbol alone, or digits and other bytes of the families before it, and they give three bytes with the end symbol third on 0 to 36 of 300 items (`APP format`, a numpy forward pass that reproduces E43's count of 0 of 300 at row 64 and its first-batch loss of 18.155). After 64 updates E43's networks of seed 73129 (the arm without decay, whose weights at row 64 are on file) give the form "symbol, symbol, end symbol" on 300 (S) and 283 (B) of 300 held items while the first symbol is right on 3 and 15. The form is learned first and fast by a born network. How long a fresh network needs for it is not on record, and the brief's scores would not show it.

(d) *The name.* The brief calls the task "a task of another kind" (title; line 13; line 20). E43's brief, which defined it, says that it shares with the first family "the whole network", "the start symbol; the end symbol as the third answer byte; the length of items and the format of two answer bytes", that "it asks for memory across fillers, as the families do across letters", and concludes "unrelated in content and not in machinery" (`$D:OPUS_TASK_E43_IS_A_FAMILY_LOST_TO_UNRELATED_LEARNING_20261116.md`, line 45). The brief concedes it under "What it cannot show" (line 77) and keeps the name.

A cheap control exists for (a), and two scores that cost almost nothing address (b) and (c). I do not recommend the graft that the question suggests (the recurrent weights of the born network in a fresh network, or the reverse) as part of this job: the parts were trained together, the fresh `l1` gives the born recurrent map inputs about four times larger than those it was trained on (0.20 against 0.05), and a graft that fails to save can therefore not be read. A graft that saves could be read, so it may follow as a job of its own.

**Repair.** (1) In the title, in line 13 and throughout, write "the unrelated task" for "the task of another kind" and "a task of other content" for "a task of another kind". Replace line 13 by:

> - *The unrelated task* is E43's (`$D:$M/e43_task.py`): an item is a code byte, two symbols out of 64, then 0 to 12 filler bytes and a query byte; the answer is the two symbols and the end symbol. None of its bytes occurs in any family's items. It is of other content than the families and of the same form: it shares with them the start symbol, the end symbol as the third answer byte, the length of an item and the need to carry two items across bytes that do not matter (E43's brief, line 45). Of the 4,096 ordered pairs of symbols, E43 teaches 2,048 and holds 2,048 out.

(2) Add a fourth start for 2,048 pairs only (six runs). In the table of runs add the row `| born-shuffled | 2,048 | S, B | 73129, 93683, 116089 | 8,192 | 6 |` and write 60 for 54. Add to the terms:

> - *A shuffled network* (`born-shuffled`) is the born network with the values of each tensor other than the byte table put into a random order within that tensor: for each `name, t` of `named_parameters()` other than `book.weight`, `perm = list(range(t.numel())); Random(f"e59|{seed}|shuffle|{name}").shuffle(perm)`, and under `torch.no_grad()` `t.copy_(t.flatten()[torch.tensor(perm)].view_as(t))`. It has the born network's values in every tensor, hence its sizes of weights, and nothing of what the born network computes. It is trained under the rule of the born network from which it is made. The runner asserts that the sorted values of every tensor equal the born network's and that the byte table is unchanged, and records the shuffled network's `tensors_sha256`.

(3) Add to the receipts, at every row: the number of held items on which the first answer byte is right, on which the second is right, and on which the third byte is the end symbol; and the count on a second held panel, the same 300 pairs at 0 fillers (`e43_task.panel(seed, "held", k=0)`; it draws the same pairs as the panel at 12 fillers, `APP task`).

(4) Add to "How the result is stated":

> - **The source of a saving** (2,048 pairs, for each rule). If `born` against `fresh` is *fewer*, the saving is ascribed to what the first family taught if `born` against `born-shuffled` is also *fewer*; it is ascribed to the sizes of the weights that training leaves if `born-shuffled` against `fresh` is *fewer* and `born` against `born-shuffled` is *about the same*; otherwise the three words are given and no source is named. Beside it, for each start and seed: the first row at which the third answer byte is the end symbol on 240 of the 300 held items; updates to 80 on the held panel at 0 fillers; updates to 80 at 12 fillers. A saving that is as large at 0 fillers as at 12 does not come from holding items across fillers, and the report says so.

(5) Replace the first sentence of the paragraph on the explanation (brief, line 73) by:

> The explanation under test: what a network gains from the families is in part an ability that does not depend on their content, namely to hold two items across bytes that do not matter and to reproduce them; a task of other content that asks for the same ability is then learned from fewer presentations, and perhaps from fewer pairs, by a network that has it. A saving of `born` or `lived` over `fresh` does not by itself support it, because the starts also differ in the sizes of their weights and in whether the answer's form is known. It is supported, for this task and no further, if at 2,048 pairs `born` needs *fewer* than `fresh` and *fewer* than `born-shuffled`, and the saving against `fresh` (the updates to 80 of `fresh` less those of `born`) is larger at 12 fillers than at 0 fillers on at least two seeds, under both rules; where `fresh` reaches 80 at 0 fillers and not at 12, the saving at 12 fillers counts as the larger.

(6) Add to "What it cannot show":

> - What the optimizer's fresh start costs each start. Every run begins with a new AdamW, whose first steps move every weight by about 0.001 whatever the gradient; that is a larger share of the weights of a born or lived network in some maps and a smaller share in others than of a fresh network's (`l1.weight`: root mean square 0.05 against 0.20; the output matrix: 0.06 to 0.08 against 0.03).

### 4. The rows: two spacings make two criteria, and under back-propagation the rise of the born network lies within six rows

**What is wrong.** The brief (line 15) scores every 32 updates to update 1,024 and every 128 after, and defines "updates to 80" as the first of three consecutive rows at or above 80%.

(a) *Two criteria.* Three consecutive rows span 64 updates when the first is at 960 or earlier, 160 updates when it is at 992, and 256 updates from 1,024 on (`APP task`). A curve must therefore hold 80% four times as long to be counted after update 1,024 as before it, and its crossing is located to 128 updates in place of 32. The two sides of the main comparison are likely to fall on different sides of that line: E43's born networks reach 80% between updates 64 and 256 under back-propagation and between 256 and 1,024 under the sweep, while fresh networks on a family have not left 10% on the taught panel by update 1,024 (defect 3 b). A fresh run that first reaches 80% at row 992 must hold it to row 1,152; one that reaches it at row 960 only to row 1,024.

(b) *Six rows.* Under back-propagation with 2,048 pairs the born network is at 0 of 300 on the held panel at update 64 and at 294, 276 and 300 at update 256 (the brief's own figures; verified). Its updates to 80 can take only the values 96, 128, …, 256. For `lived` against `born` to be *fewer* there, the lived network must reach 80% at row 96 when the born one does at 160 or later; with the born network at 128, no row of the lived network gives a ratio below 2/3 except 64. The word *about the same*, which P59.3 predicts, is then nearly forced by the rows.

(c) *The last rows.* A run that first reaches 80% at row 8,064 or 8,192 cannot show three rows and "does not reach 80"; the brief does not say so.

**Repair.** Replace line 15 by the text below. By E40's and E43's scoring costs a row of 600 items at 12 fillers costs about 1.2 CPU-seconds (an estimate from two receipts with different mixes of items; check 4 measures it), so 273 rows cost about 5.5 CPU-minutes a run against about 1.8 for the brief's 89.

> - The network is scored at update 0, after every 16 updates up to update 512, and after every 32 updates from there to update 8,192. Each scoring is called a row; a run has 273 rows. *Updates to 80* is the first row r at which the share of right answers on the held panel is at or above 80% (240 of 300) and is so at every row up to r + 64. A run with no such row up to update 8,128 *does not reach 80*. A *level* at the end is the mean over the nine rows from update 7,936 to 8,192.

In line 16 write "over the rows from update 7,296 to 8,192" and "over the rows from update 5,248 to 6,144" (29 rows each) for "the eight rows". Replace the paragraph on cost (line 40) by:

> Cost: E43's own runs took 0.100 to 0.110 CPU-seconds an update on this task and E40's 0.104 to 0.119 at this size; a row scores 900 items (the two held panels and the taught panel) and a run has 273 rows. About 0.35 CPU-hours a run, 21 CPU-hours for the 60 runs. Check 4 measures both costs before the runs.

If the reviewer keeps the brief's rows to save about 4 CPU-hours, the least that is needed is: the definition "at r and at every row up to r + 256", the sentence of (c) with 7,936, and a statement in the report that under back-propagation with 2,048 pairs the rows do not resolve `lived` against `born`.

### 5. The function named for the admission of the born networks fails on the default branch

**What is wrong.** The brief names `exp26_candidate.admit_e26` as "E43's admission of the born networks" (line 44) and again in check 3 (line 50). `admit_e26` asserts that the source hashes in the birth's receipt equal those of the files now present (`$D:$M/exp26_candidate.py`, lines 187 to 189). All six birth receipts hold `exp16_integrated.py` with a SHA-256 that begins `f4559971`; on the default branch that file's begins `c9fcedcf`, and it is the only source that differs (`APP figures`). Called directly, `admit_e26` raises "made by other sources: ['exp16_integrated.py']". E43 does not call it directly: `exp43_unrelated.run` calls `E38.admit(bfile, rule, seed)` (line 125), and `exp38_erase.admit` (lines 51 to 69) catches exactly this assertion for exactly this pair of hashes. E43's receipts record the exception in `settings.birth_admission`.

**Repair.** In line 44 replace "and E43's admission of the born networks (`exp26_candidate.admit_e26`)" by:

> and E43's admission of the born networks, `exp38_erase.admit(bfile, rule, seed)` (line 125 of `exp43_unrelated.py`); it calls `exp26_candidate.admit_e26` and carries the one recorded exception for `exp16_integrated.py`, without which `admit_e26` fails on the default branch

Check 3 is rewritten in defect 8.

### 6. The taught panel cannot be built as described for 512 and for 256 pairs

**What is wrong.** The brief takes "the panels" from `e43_task.py` "unchanged" (line 44) and defines the taught panel as "300 taught pairs at 12 fillers, or every taught pair where fewer than 300 are taught" (line 14). `e43_task.panel(seed, which, n=300, k=12)` (lines 40 to 42) calls `split(seed)` itself and samples from the whole taught half; it cannot be given `taught[:n]`. For 512 pairs it would return 300 pairs of which most were never taught in that run. For 256 pairs its sampling cannot be reused at all: `Random.sample` of 300 from 256 raises `ValueError` (`APP task`), and the brief gives neither a generator nor an order for "every taught pair". Check 1 needs the taught panel with 2,048 pairs to be E43's, item for item.

**Repair.** One construction serves the three sizes and is E43's panel at 2,048 (verified for the three seeds, `APP task`). Replace the second sentence of line 14 by:

> *The taught panel* for n taught pairs is built by `rng = Random(f"e43|{seed}|panel|taught")` and `[e43_task.render(a, b, 12, rng) for a, b in rng.sample(taught[:n], min(300, n))]`. For n = 2,048 it is E43's taught panel (the same call on the same list); for n = 512 it is 300 of the 512 taught pairs; for n = 256 it is all 256 taught pairs. Its counts are given with its number of items.

In line 44 write "the held panel" for "the panels".

### 7. Weights are lost and interrupted runs cannot be started again; the starting commit is not named

**What is wrong.**

(a) `.gitignore` at the root of the default branch excludes `*.pt` (its line 2). The final weights of seed 73129 are not added by a plain `git add`. E43's worker reports that it had to use `git add -f` (`$D:$M/exp45/e43/E43_REPORT.md`, line 136).

(b) The brief says that an interrupted run "is started again from its beginning" (line 85). E43's `run` and E40's `run` begin with `R.refuse_existing(path)` on the receipt, and E43's writes the receipt at every row (lines 123 and 211). A runner built from them leaves a partial receipt when the host stops the machine and then refuses to start. The brief does not say what becomes of the partial receipt.

(c) The brief does not name the commit from which the worker starts, nor that the brief itself and the document cited in line 20 are on the reviewer's branch and not on the default branch. `V` is defined (line 7) and used nowhere.

(d) numpy is needed (`exp26_candidate.Digest` and `exp20_reference.tensors_sha` call `.numpy()`); the rules name torch only.

(e) The report has a name and no path.

(f) Rule 4 of the plan asks that a result be "verified from receipts and weights before it is read". With final weights committed for seed 73129 only (line 44), two of the three seeds can be verified from receipts alone.

**Repair.** Add to "Rules for the worker":

> - Start a branch of your own from `$D` at 7cabb13a. This brief and the reviewer's documents are on `$V`; read them with `git show`.
> - The virtual environment needs numpy beside torch; its version is recorded in every receipt.
> - `.gitignore` at the root excludes `*.pt`. Add the weights that this brief asks for with `git add -f` and confirm with `git ls-tree` that the commit holds them. The final weights of the other runs go to `$M/exp45/e59/rows_uncommitted/`, with their SHA-256 in the receipt, and stay on the disk until the reviewer has verified the result; the reviewer may name runs whose weights are then committed.
> - The runner refuses an output path that exists. Before a run is started again, move its partial receipt to `$M/exp45/e59/interrupted/` and commit it; do not delete it. The run started again must give, at every row that the partial receipt holds, the same `data_digest` and `tensors_sha256`, and the report says for each such run that it did.
> - The report is `$M/exp45/e59/E59_REPORT.md`.

In line 20 write `$V:$R/E40_SAVING_IN_WHOLE_EXPERIENCE_20261117.md`. In line 44 write "the `tensors_sha256` at every row" for "the `tensors_sha256` at rows 64, 256, 1,024, 2,048, 4,096 and 8,192" (the hash costs a few hundredths of a second, and the rule on runs started again needs it).

### 8. The checks

**What is wrong.**

- *Check 1* stands under "Checks before the runs" and is read from runs. The brief does not say what may run beside the two runs, nor whether the other runs of seed 73129 wait for row 64 or for row 4,096. It is limited to seed 73129, although E43's receipts of the other two seeds are on file and the comparison costs nothing. It can fail when the thing it is named after is wrong.
- *Check 2*, for 512 and 256 pairs, asserts only that every pair presented is in `taught[:n]`. Any stream drawn from those pairs passes, whatever its order and its numbers of fillers, so the check cannot fail when "the text of `e43_task.stream`" was not reproduced. The second assertion ("no pair of the held panel is") follows from the first and cannot fail alone. The brief uses "the batch digest" for two different digests: that of `e43_task.digest` (over the bytes of prompts and answers) and the receipts' `data_digest` (`exp26_candidate.Digest`, over the tensors of a batch). I wrote the second in numpy from the text of `exp26_candidate.Digest` and `exp45.batch_tensor`; it reproduces E43's `data_digest` at rows 64, 256, 1,024, 2,048 and 4,096 for the three seeds (`APP digest`), so its values for 512 and 256 pairs can serve as references.
- *Check 3.* For `born` the function named fails (defect 5). For `lived`, `e40_new_family.start` asserts only that the file's SHA-256 is the receipt's; the comparison of the loaded tensors with the receipt is recorded as `tensors_sha_matches_receipt` and not asserted (`$D:$M/e40_new_family.py`, lines 113 to 119). The path and the field for `fresh` are right (`start.tensors_sha256` exists in the six receipts and is one value for both rules of a seed).
- *Check 4* cannot fail: it names no run on which the cost is measured and no forecast at which the job stops.
- No check covers a run that was started again.

**Repair.** Replace the section "Checks before the runs" by:

> ## Checks (a receipt for each)
>
> 1. **The stream and the panels** (before any run; nothing is trained). For each seed: (a) the digest of the first 64 batches of the runner's stream with n = 2,048, computed by the lines of `e43_task.digest`, equals `e43_task.digest(seed, 64)`, which begins `34eefb4442b48399`, `153cafdcda7379d0` and `821bcb2a9ca6dd54` for the three seeds (`$D:$M/exp45/e43/checks/e43_check_3_spec.json`); (b) the runner's held panel, and its taught panel with n = 2,048, equal `e43_task.panel(seed, "held")` and `e43_task.panel(seed, "taught")` item for item; (c) for each n, the `sha256` of the runner's `data_digest` (`exp26_candidate.Digest`) after 64 and after 8,192 batches equals the value listed below; (d) over the 8,192 batches the set of pairs presented is exactly `taught[:n]`. A difference stops the job.
> 2. **The starts** (before any run). `fresh`: the `tensors_sha256` of the start equals `start.tensors_sha256` in `$D:$M/exp45/e40/e40_fresh_<rule>_<seed>.json`. `born`: admitted by `exp38_erase.admit`, and the `tensors_sha256` of the loaded network equals `settings.birth_tensors_sha256` in `$D:$M/exp45/e43/e43_unrelated-decay_<rule>_<seed>.json`. `lived`: `e40_new_family.start("lived", rule, seed)` passes its assertion on the file's hash, and the runner asserts `info["tensors_sha_matches_receipt"]`, which `start` records and does not assert. `born-shuffled`: the assertions of its definition. For each seed, the rows 128 to 225 of the byte table are equal in every start under both rules. The counts of every start on the two held panels and on the taught panel at row 0 are recorded; for `born` and `lived` they are 0 (computed in the check of this brief). A difference stops the job.
> 3. **Cost.** A run of 64 updates of `fresh` with 256 pairs under the sweep, written to `$M/exp45/e59/dry/`, gives the CPU-seconds of an update and of a row and a forecast for the 60 runs. If the forecast is above 28 CPU-hours, stop and ask.
> 4. **`born` with 2,048 pairs is E43's run.** The two such runs of seed 73129 are launched first, with one run of `fresh` beside them; no other run is launched until their rows 64 have been compared. At rows 64, 256, 1,024, 2,048 and 4,096 the `data_digest`, the counts of the held panel at 12 fillers and of the taught panel, and the `tensors_sha256` equal those of `$D:$M/exp45/e43/e43_unrelated-decay_<rule>_73129.json`. The runner makes the same comparison in the `born` runs with 2,048 pairs of the other two seeds. A difference at any row stops every run. If the `tensors_sha256` alone differs while the digest and the counts agree, say so in the message, since the cause may then be the host.
> 5. **The same batches.** At every row the `data_digest` is equal across the runs of a seed and a number of pairs. It is read from the receipts when a seed is finished.

The values for check 1 (c), in the order of the seeds 73129, 93683, 116089 (`APP digest`):

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

In "Reporting" write "checks 1 to 3 passed (or which failed); check 4 passed at row 4,096" for "checks 2 and 3 passed (or which failed); check 1 passed". In line 38 write: "Order: checks 1 to 3; the runs of seed 73129, of which the first three are those of check 4; the other two seeds. After seed 73129 the worker sends its table and goes on, unless a loss was not finite, a check failed, or `fresh` is at 0 on the taught panel at the end of each of its six runs; in those cases the worker stops and asks."

### 9. What requirement B2 needs and the brief leaves out, at no cost

**What is wrong.** Requirement B2 (`PLAN.md`, section 2) compares our learner with the conventional one: "the end result may be about the same while our learner needs far fewer samples". The brief's three comparisons are all within a rule. The two rules are set against each other only in P59.6, for `fresh` with 2,048 pairs, and "What it cannot show" (line 80) declines the comparison. The runs hold it at no cost: every start, number of pairs and seed is run under both rules on the same batches. The case that B2 describes, our learner with experience against the conventional learner without, is the pair `lived` under the sweep against `fresh` under back-propagation, and it too is in the runs.

The brief also does not say in which account a saving is counted. The reviewer's document on E40 found that E40's saving exists when the new family alone is counted and not when the experience before it is counted (`E40_SAVING_IN_WHOLE_EXPERIENCE_20261117.md`, section 8). A born network has 24,576 updates behind it and a lived network 61,440, against the tens or hundreds of updates that a saving here can amount to.

**Repair.** Add to "How the result is stated":

> - **The two rules.** For each start and each number of taught pairs, the sweep against back-propagation (the first is the sweep), in the words of "Presentations" and of "Pairs", with the two levels at the end. And, as the case that requirement B2 describes, `born` and `lived` under the sweep against `fresh` under back-propagation, in the same words.
> - **The account.** Every count of presentations and of pairs in this job is of the unrelated task alone. Beside each reading for `born` and `lived` the report gives what the start was given before: 24,576 updates (1,572,864 presented items) for `born`, and 61,440 updates (3,932,160 items) for `lived`.

Replace line 80 by: "- Why the sweep and back-propagation differ, where the comparison of the two rules shows that they do."

### 10. With 256 pairs: the two chances are wrong, the realised sets are known, and the held panel has two kinds of item

**What is wrong.** The brief (line 36) gives 1.8% as the chance that a given symbol is absent from a given place of 256 taught pairs and 0.03% for 512. The taught pairs are drawn without replacement, and the chances are 1.56% and 0.018% (`APP task`); the brief's figures are exp(−4) and exp(−8). "About one symbol in each place" is right (0.995 expected). The realised sets are fixed by the seeds and need no chance:

| Seed | Absent from the first place | Absent from the second place | Held-panel items with such a symbol |
|---|---|---|---|
| 73129 | 131, 139 | 167 | 11 of 300 |
| 93683 | 171 | 152 | 6 of 300 |
| 116089 | none | 186 | 2 of 300 |

With 512 pairs no symbol is absent from a place (each occurs at least twice). With 256 pairs a symbol occurs in a place 0 to 10 times, and 24 to 28 of the 128 symbol-places are shown only once or twice. An item whose symbol was never shown in that place cannot be answered by a network that has learned each symbol in each place separately; it can be answered by one that copies whatever stands there. If such items cannot be answered, the held panel's ceiling is 289, 294 and 298 of 300, which is above the bar of 240 for every seed. The panel need not be restricted. The two kinds of item should be counted apart, since the second kind bears on the explanation under test, although 2 to 11 items are too few to read alone.

**Repair.** Replace the parenthesis at the end of line 36 by:

> (With 512 pairs every symbol occurs at least twice in each place of the taught pairs, for the three seeds. With 256 pairs a symbol occurs in a place 0 to 10 times. Symbols 131 and 139 never stand in the first place and 167 never in the second for seed 73129; 171 and 152 for seed 93683; none and 186 for seed 116089. The held panel has 11, 6 and 2 items with a symbol in a place in which no taught pair has it. For n = 256 the receipts give the held panel's counts in two parts, those items and the remaining 289, 294 and 298; updates to 80 and the levels are taken on the whole panel of 300.)

### 11. Statements about the record

| Brief | Statement | The record | Replacement |
|---|---|---|---|
| line 26 | "the lived networks reach 77 to 95%" | 77.01 to 94.48 over the six lived runs under the two rules; 6.02 to 11.83 for the fresh runs (`APP figures`). 95 is 94.48 rounded twice. | "77 to 94%" |
| line 20 | "the saving comes from having lived and is found under both rules" | E40 has the starts `lived` and `fresh` only. No network with the first family alone behind it was given that family, so the record does not say what the life adds to the first family. The reviewer's document says the same (section 8: "no lived network with a shorter history exists"). E59 is the first job with both starts. | "the saving is found under both rules against fresh networks; E40 has no network with only the first family behind it, so the record does not say how much of the saving the life adds" |
| line 24 | "not located more finely than 'by 256' under back-propagation" | At update 64 the held panel is at 0 of 300 in all six runs. | "than 'between 64 and 256' under back-propagation" |
| line 40 | "E40's runs took 0.12 CPU-seconds an update at this size (974 CPU-seconds for 8,192 updates)" | 974 is the largest of E40's twelve (851 to 974, 0.104 to 0.119). E43's runs on this very task took 0.100 to 0.110. | see defect 4 |
| line 36 | "1.8%", "0.03%" | 1.56%, 0.018% | see defect 10 |

### 12. The predictions

- **P59.3** ("with 2,048 pairs, `lived` against `born` is *about the same*, under both rules") is nearly forced under back-propagation by the rows of the brief (defect 4 b). With the rows of the repair it can fail.
- **P59.4** ("`fresh` does not generalise, under either rule") can be read as "under neither" or as "under at least one". Write: "With 256 pairs, `fresh` does not generalise under the sweep and does not under back-propagation".
- **P59.5** depends on the licence. With the text of defect 1 it is decidable and names its start. Write: "The sentence on pairs is licensed for `born` or for `lived`, at some n, under at least one rule".
- **P59.6** ("`fresh` under the sweep needs more than 1.5 times the updates of `fresh` under back-propagation on at least two seeds") has no rule for a seed on which one of the two does not reach 80. Write: "With 2,048 pairs, the seed's word for `fresh` under the sweep against `fresh` under back-propagation is *more* on at least two seeds". It does not contradict the record, but the nearest record points the other way: for fresh networks of this size on the first family at a half of its pairs, the ratio of the sweep to back-propagation in updates to 80 is 1.10, 1.19 and 1.24 (E41; at a quarter it is 2.00, 2.79 and not reached; `APP flat`). E43's born networks give the larger difference (by 256 against by 1,024). The reviewer may wish to state which of the two the 60% rests on.
- No prediction covers the control `born-shuffled` or the comparison of the two rules of defect 9. Each needs one before the runs, since they are the reviewer's.
- P59.1, P59.2 and P59.5 are decidable from the runs; none of the six contradicts another.

### 13. Smaller points

1. The runner paragraph (line 44) does not say that `arch26.install()` is called first. E43's `run` calls it at line 119, before the admission, which creates the network through `exp11_cell.new_net`, and before any score. Add: "The runner calls `arch26.install()` first, as E43's `run` does. Training draws no random number, so no call of `T.manual_seed` is needed beyond the one inside `arch26.new_net`."
2. Line 44 says the two extra scores are taken "so that the report can say what the new learning cost"; line 60 gives them "without a reading" and line 78 says the job cannot show what a learner keeps. Write in line 44: "so that the report can give, without a reading, what remained of the earlier families".
3. The two extra scores come from one call, `X16.panels(seed)`: E43's score 2 is `P[n][12]` for the four operations of the first family (4 × 300 items) and E40's six earlier panels are `P[n][12]` for `add89`, `sub89`, `add97`, `sub97`, `add73`, `sub73` (2 × 300 and 4 × 400). `add89` and `sub89` are in both; the receipt should hold them once. The six panels leave out max and min of the life's families and the compositions; the brief's "the life's families' panels" (line 60) should read "add and sub of the life's two families of single operations".
4. The receipts should record the SHA-256 of the new runner and of the files it takes from (`exp43_unrelated.py`, `e43_task.py`, `e40_new_family.py`, `exp38_erase.py`, `arch31_workerB.py`), as E43's `provenance` does.
5. "No name of any language model in any committed file or commit message" (line 87) contradicts the trailers that the same line prescribes: `AGENTS_POINTER.md` asks for `Acting-Agent: claude:<session>`, and the commits of the workers on file carry it. Add "apart from the trailers".
6. The brief compares `lived` against `born` as "having learned more families". The lived network has also had 36,864 more updates under decay and has lost most of the first family. Add to "What it cannot show": "Whether a difference between `lived` and `born` comes from the further families or from the further updates."

### 14. Language

The brief asks the worker for standard academic English with every term defined at first use (line 88). Measured by that, the following fail. Where a defect above already gives the replacement, it is named.

| Brief | Text | Fault | Replacement |
|---|---|---|---|
| title | "a task of another kind learned by a fresh network, by a born network and by a lived network" | "born" and "lived" are the programme's labels, a figure of birth and life, and stand undefined in a title; "another kind" (defect 3) | "E59: the unrelated task learned by an untrained network, by a network trained on one family and by a network trained on four, from three numbers of taught pairs" |
| line 3 | "Evidence level E2." | not a sentence; the level is not defined | "The evidence is of the programme's level E2, an untaught presentation of a taught relation." |
| line 3 | "Three seeds, about 16 CPU-hours." | not a sentence | "The job uses three seeds and about 21 CPU-hours." |
| line 12 | "*A born network* is E26's birth"; "the six blocks of E26's life" | "birth", "life", "block" and "family" are used before they are explained | "A *family* is a set of operations under one modulus, or of their compositions, whose items begin with code bytes of their own. *A born network* (`born`) is the fresh network after the first stage of E26's training, which the programme calls a birth: 24,576 updates on the first family (add, sub, max and min under modulus 89). *A lived network* (`lived`) is the born network after the second stage, which the programme calls a life: six blocks of 6,144 updates each, 36,864 in all, on three further families." |
| line 14 | "*The held panel* is E43's" | "panel" is not defined | begin with "A *panel* is a fixed set of items on which a network is scored and never trained." |
| line 15 | "Rows are scored every 32 updates"; "the held panel is at or above 80%" | "row" is not defined, and a row is not what is scored; a panel is not a percentage | defect 4 |
| line 20 | "Mr. Xu, 17 November: the end result of our learner may be about the same as the conventional one's while it needs far fewer samples." | no verb of saying; "our learner" and "the conventional one" are not defined | "Mr. Xu stated on 17 November that the end result of our learner (the network trained by the sweep) may be about the same as that of the conventional learner (the network trained by back-propagation) while ours needs far fewer samples (requirement B2 of `PLAN.md`)." |
| line 20 | "A network that has lived learns a new family of the same kind" | the figure carries the meaning | "A lived network learns a new family of the same form" |
| line 22 | "(working rule 1)" | the reference is not given; the check of the E54 brief, revision 2, raised the same point | "(the first of the programme's six working rules for briefs: a comparison with the record in every brief)" |
| line 26 | "untaught pairs that need the reduction" | "the reduction" is not defined | "untaught pairs of add and sub whose answer needs the reduction by the modulus" |
| line 34 | "every tensor but the byte table" | "the byte table" is not defined | "every tensor but the byte table (the embedding of the input bytes, `book.weight`)" |
| line 38 | "Order: the checks; the eighteen runs of seed 73129; the other two seeds." | not a sentence | defect 8 |
| line 40 | "About 0.3 CPU-hours a run, 16 CPU-hours in all." | not a sentence | defect 4 |
| line 44 | "A new file `exp59_another_kind.py` in `$M`; no existing source is edited." | not a sentence; the name repeats "another kind" | "The runner is a new file, `$M/exp59_unrelated_starts.py`; no existing source is edited." |
| line 44 | "E43's admission of the born networks" | "admission" is not defined | add "(the check of a weights file against its receipt before the file is used)" |
| line 44 | "Receipts: `$M/exp45/e59/…json`, with the start's file and hashes, …" | not a sentence; "receipt" is not defined | "Each run writes a receipt, a JSON file `$M/exp45/e59/e59_<start>_n<2048|512|256>_<rule>_<seed>.json`, which holds the start's file and hashes, …" |
| line 48 | "Read from the two such runs of seed 73129, which are launched first." | no subject | defect 8 |
| line 51 | "**Cost:** CPU-seconds an update and a scored row, and a forecast." | not a sentence | defect 8 |
| line 57 | "*more* is the mirror" | a figure in place of the rule | defect 2 |
| line 58 | "is licensed" | the term is not defined | add at its first use: "A sentence is *licensed* if the report may state it." |
| line 60 | "Beside these, without a reading: the first family's and the life's families' panels at row 0 and at the last row." | not a sentence | "The report also gives, and does not interpret, the counts at row 0 and at the last row on the panels of the first family and of add and sub of the life's two families of single operations." |
| line 73 | "to give them back" | colloquial | "to reproduce them" (defect 3) |
| line 76 | the heading "What it cannot show" followed by noun phrases | the items are not sentences | add the line "The job cannot show the following." before the list |
| line 85 | "keep your turn open with a waiting command (a command that waits on the processes, repeated in pieces within the command's time limit)" | "turn" and "in pieces" are the host's terms and are not explained | "do not end your turn, since the host counts a session as idle once the agent's turn has ended: run a command that waits for the processes and, because a command may run for a limited time only, repeat it until they have finished" |
| line 97 | "The four checks; the 54 runs; `E59_REPORT.md` with …" | not a sentence | "The job is done when the five checks and the 60 runs have their receipts and `E59_REPORT.md` holds the table of every run, the readings in the words fixed above, the outcome of each prediction and the list of committed files." |

## Verified and found sound

**Figures (A).**

- E43, the arm with decay: at update 64 the taught panel is at 0, 0, 4 and 0, 0, 0 of 300 under back-propagation and under the sweep, and the held panel at 0 in all six; at update 256 the twelve counts are those of the brief; from update 1,024 on every panel is at 300 of 300. The rows on file are 0, 64, 256, 1,024, 2,048 and 4,096, and no receipt of an extension is on file. The arm without decay differs by at most 4 items at any row and panel (`APP figures`).
- E40: the fresh networks end at 6.02 to 11.83% ("6 to 12").
- A run of the brief has 89 rows; each window of "has flattened" has 8; the last three rows are 7,936, 8,064 and 8,192; 54 runs, 18 a seed (`APP task`).
- The cost of about 0.3 CPU-hours a run and 16 in all follows from E43's and E40's costs of an update (820 to 975 CPU-seconds for 8,192 updates) and about 110 CPU-seconds for 89 rows of 600 items, the second being an estimate.
- All three starts have 1,124,113 parameters; the birth has 24,576 updates and the life 36,864.
- "About one symbol in each place is never shown" with 256 pairs: 0.995 expected.

**The runner (B).**

- *Check 1 holds in principle.* I read `exp43_unrelated.run` in full. Between updates it computes scores 1, 2 and 3 (`exp11_cell.respond`, under `no_grad`), score 4 (reads the parameters), score 6 (`eq_states`, under `no_grad`), `tensors_sha`, `save_weights` (which saves the `state_dict`) and writes the receipt. None of these touches the optimizer or a parameter. The update is `exp38_erase.step` with no pull, which is the text of `exp26_candidate.step`; with `arch26.forward`, `exp3_state.lin`, `exp45.batch_tensor` and AdamW it draws no random number (no dropout, no sampling). The stream has its own generator, `Random(f"e43|{seed}|stream")`. The order in E43 is `arch26.install()`, the admission (which creates the network; `arch20.new_net` calls `T.manual_seed(seed)` itself), `load_state_dict`, `exp26_candidate.make_opt(m)`, and the weight-decay line, which sets group 0 to 0.25, the value `make_opt` gave it. So `make_opt(m)` alone is E43's optimizer for this arm. E43's `run` has no `T.manual_seed` of its own and E40's has one before `start`; since training draws no random number, the place cannot matter. A run scored at other rows gives the same tensors. Two precedents exist: E51's check 6 reproduced E43's `tensors_sha256` at row 64 under both rules for seed 73129 from another runner in another container (`origin/claude/e51-reserve-017dSEpW:$M/exp45/e51/checks/e51_check_6_open_is_e43_73129.json`), and E43's first addendum reran three runs and found their rows 0 to 4,096 equal in tensors and digests (`E43_REPORT.md`, line 191).
- *The nested sets.* `taught[:n]` is the first n pairs of the shuffled list, so the three sets are nested and each is a uniform draw. The brief's stream text with `taught[:n]` is well defined, and with n = 2,048 it is E43's stream: my copy gives `e43_task.digest(seed, 64)` as on file, and the receipts' `data_digest` at five rows for three seeds (`APP task`, `APP digest`). The held panel has its own generator and draws from the held half only; it does not depend on n, its 300 pairs are distinct, and none of them is taught at any n.
- *`e40_new_family.start`.* For `fresh` it returns `(m, None, info)` with `info["tensors_sha256"]` and asserts that the rule is S or B. For `lived` it returns `(m, None, info)` after asserting that the file's SHA-256 is the receipt's `complete.weights_sha256`. `e40_new_family` imports `exp26_candidate`, `arch26` and `arch31_workerB`; the last defines functions and installs nothing, and the module's own top level runs three assertions on the codes of E40's family. `exp43_unrelated` imports `exp38_erase` and `e43_task`. Both reach the harness through the same module objects, and I found no assignment at import in either on which the other depends. This was read, not executed.
- *The starts.* The six `e26_dev_<S|B>_<seed>.pt` and the six `e26_life_<S|B>_<seed>.pt` with their receipts are on the default branch; the SHA-256 of each file equals its receipt's; each life names its birth by hash; E43's `birth_tensors_sha256` equals the birth receipt's `tensors_sha256`, and E40's lived start equals the life receipt's (`APP figures`). `fresh` is the network from which the birth of the same seed was made (`exp26_candidate.dev`, line 126), and it is one network for both rules.
- *The two extra scores* can be taken at row 0 and at the last row without E43's heavy scoring: both come from `X16.panels(seed)` and amount to 2,800 distinct items (defect 13, point 3).
- *Row 0.* The born and the lived networks have 0 right on the held panel at 12 and at 0 fillers and on the three taught panels (`APP row0`).

**The design (C).**

- The byte table's rows of the task's bytes are bit-equal in the three starts and under both rules for each seed (defect 3 a), so the code of the new bytes favours no start.
- The rule "has flattened", applied at update 8,192 to the 23 curves of fresh networks on record that go on beyond it and are below 80 there, is wrong by 5 points or more once in the 9 cases that it calls flattened, and by 10 points never (defect 1 a). It is sound where the taught pairs have been learned.
- 256 is a sensible smallest number: the held panel's ceiling stays above the bar (289 to 298 of 300).
- "No run of the unrelated task from a fresh network or from a lived network exists, and none with another number of taught pairs": I found none. On the default branch `e43_task` is used by E43's runner and its two addenda (a reminder, plain gradient steps, rehearsal, and E42's networks), all from births or from networks made from them; on `origin/claude/e51-reserve-017dSEpW` E51 uses it from births.
- The brief's own limits (lines 77 to 80) are stated fairly, apart from the points of defects 3, 9 and 13.

**The checks (D).** `start.tensors_sha256` exists in E40's six fresh receipts; `tensors_sha256`, `data_digest` and `score3_new_task` exist in every row of E43's receipts; `describe` stands at lines 63 to 72 of `e42_tables.py` on `origin/claude/gallant-dijkstra-i9zgsr`; every path of the brief that I looked up exists.

## Not checked

- Nothing was run with torch. That the runner as built gives E43's hashes, and that the worker's host repeats them beyond row 64, rest on the reading above and on the two precedents.
- The fresh network at update 0. No file of it is on record. Its sizes of weights are taken from E40's fresh runs after 64 updates and from torch's default initial values, which I did not verify against torch's source. E43's check 3 records that the initial values of the byte table differ between a Mac and the Linux host by 1.4e-06; check 2 of the repair would catch such a difference.
- The cost of a scored row. It is an estimate from E40's and E43's scoring times, whose mixes of items differ.
- Whether a fresh network learns the unrelated task within 8,192 updates under either rule, and whether `born-shuffled` trains at all. No record bears on either.
- The import of `e40_new_family` beside `exp43_unrelated` in a running process.
- The tool `send_message` and the session named in "Reporting"; the trailers beyond the commits on file.
- The reviewer's document on E40 beyond its sections 5 and 8, E43's report beyond the lines cited, and E43's brief beyond its line 45.
- Branches other than the default branch, E51's worker's, E43's worker's (`origin/claude/peaceful-hopper-8ofq4i`) and E42's worker's were not searched for runs of the unrelated task.
- The definition of evidence level E2 beyond one sentence in older documents of the programme.

## Appendix: the script

Saved as `check_e59.py` anywhere and run with the working directory inside the repository; every part ran to its end at 22:54 UTC. The output of the parts is quoted in the defects above.

```python
"""Check of the E59 brief (17 November 2026). Read-only: every file is read with `git show`; no code of another branch is executed; nothing
is trained; torch is not used. Standard library, and numpy for the parts digest, weights and format.

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I check_e59.py <part> ...      (with the working directory inside the repository)
    parts: figures task votes digest weights format row0 flat

Lines marked "copied" are copied from the named file at origin/codex/minifly-consolidated-20261023 (7cabb13a)."""
import collections, hashlib, io, json, pickle, subprocess, sys, zipfile
from fractions import Fraction
from math import comb
from pathlib import Path
from random import Random

D = "origin/codex/minifly-consolidated-20261023"; M = "MOD97_LOCAL_RULE_20261106"
TOP = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip() or "."      # run it from inside the repository
SEEDS = (73129, 93683, 116089)
mean = lambda v: sum(v) / len(v)


def blob(path, ref=D):
    r = subprocess.run(["git", "-C", TOP, "show", f"{ref}:{M}/{path}"], capture_output=True)
    assert r.returncode == 0, (ref, path)
    return r.stdout


def js(path, ref=D):
    return json.loads(blob(path, ref))


# ---------------------------------------------------------------- the task (copied from e43_task.py)
SYM = list(range(128, 192)); FILL = list(range(192, 224)); CODE, ASK = 224, 225; EOS, BOS = 256, 257; KMAX, TAUGHT = 12, 2048


def split(seed):
    pairs = [(a, b) for a in SYM for b in SYM]
    Random(f"e43|{seed}|split").shuffle(pairs)
    return pairs[:TAUGHT], pairs[TAUGHT:]


def render(a, b, k, rng):
    return [CODE, a, b] + [rng.choice(FILL) for _ in range(k)] + [ASK], [a, b, EOS]


def panel(seed, which, n=300, k=12):
    taught, held = split(seed); rng = Random(f"e43|{seed}|panel|{which}")
    return [render(a, b, k, rng) for a, b in rng.sample(taught if which == "taught" else held, n)]


def stream_n(seed, n):                               # e43_task.stream's text with taught[:n]; n = 2,048 is e43_task.stream
    taught = split(seed)[0][:n]; rng = Random(f"e43|{seed}|stream")
    while True:
        order = list(range(len(taught))); rng.shuffle(order)
        for i in order:
            yield render(*taught[i], rng.randint(0, KMAX), rng)


def spec_digest(st, updates, batch=64):              # e43_task.digest's text, on any stream
    h = hashlib.sha256()
    for _ in range(updates):
        for p, y in [next(st) for _ in range(batch)]:
            h.update(bytes([len(p)]) + bytes(p) + bytes(v % 256 for v in y))
    return h.hexdigest()


def taught_panel_n(seed, n, k=12):                   # the construction proposed in the repair of defect 6
    pool = split(seed)[0][:n]; rng = Random(f"e43|{seed}|panel|taught")
    return [render(a, b, k, rng) for a, b in rng.sample(pool, min(300, n))]


# ---------------------------------------------------------------- weights without torch (reader copied from compare_pt_tensors.py)
class _Unpickler(pickle.Unpickler):                  # refuses every class but those that describe a tensor: nothing from a file is executed
    def find_class(self, mod, name):
        if mod == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if mod == "torch._utils" and name == "_rebuild_tensor_v2":
            return lambda storage, offset, size, stride, *a, **k: ("T", storage, offset, tuple(size))
        if mod == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, *a, **k: data
        if mod == "torch" and name.endswith("Storage"):
            return name
        raise pickle.UnpicklingError(f"refused {mod}.{name}")

    def persistent_load(self, pid):
        return (pid[1], pid[2])


def weights(path):
    import numpy as np
    z = zipfile.ZipFile(io.BytesIO(blob(path))); pk = [n for n in z.namelist() if n.endswith("data.pkl")][0]; root = pk[:-len("data.pkl")]; out = {}
    for name, o in _Unpickler(z.open(pk)).load().items():
        _, (stype, key), offset, shape = o
        assert stype == "FloatStorage"
        n = 1
        for s in shape:
            n *= s
        out[name] = np.frombuffer(z.read(root + "data/" + key)[offset * 4:(offset + n) * 4], dtype="<f4").reshape(shape).copy()
    return out


def gru(w, x, all_steps=False):                      # torch's GRU as arch26.forward writes it out; returns the logits
    import numpy as np
    sg = lambda v: 1.0 / (1.0 + np.exp(-v))
    z1 = np.tanh(w["book.weight"][x] @ w["l1.weight"].T + w["l1.bias"])
    gi = z1 @ w["rnn.weight_ih_l0"].T + w["rnn.bias_ih_l0"]; Whh, bhh = w["rnn.weight_hh_l0"], w["rnn.bias_hh_l0"]; U = Whh.shape[1]
    h = np.zeros((x.shape[0], U), dtype=np.float32); Hs = []
    for t in range(x.shape[1]):
        gh = h @ Whh.T + bhh
        r = sg(gi[:, t, :U] + gh[:, :U]); z = sg(gi[:, t, U:2 * U] + gh[:, U:2 * U]); n = np.tanh(gi[:, t, 2 * U:] + r * gh[:, 2 * U:])
        h = (1 - z) * n + z * h; Hs.append(h)
    if all_steps:
        Hs = np.stack(Hs, 1)
        return np.tanh(z1 @ w["l2.weight"].T + w["l2.bias"] + Hs @ w["lm.weight"].T + w["lm.bias"]) @ w["l3.weight"].T + w["l3.bias"]
    return np.tanh(z1[:, -1] @ w["l2.weight"].T + w["l2.bias"] + h @ w["lm.weight"].T + w["lm.bias"]) @ w["l3.weight"].T + w["l3.bias"]


def respond(w, items, cap=6):                        # greedy decoding as exp11_cell.respond
    import numpy as np
    x = np.array([[BOS] + p for p, _ in items]); gen = [[] for _ in items]; done = [False] * len(items)
    for _ in range(cap):
        y = gru(w, x).argmax(1)
        for q in range(len(items)):
            if not done[q]:
                gen[q].append(int(y[q])); done[q] = int(y[q]) == EOS
        x = np.concatenate([x, y[:, None]], 1)
        if all(done):
            break
    return gen


# ---------------------------------------------------------------- parts
def figures():
    print("E43: right of 300 on the taught and on the held panel, by row")
    R = {}
    for arm in ("unrelated-decay", "unrelated"):
        for rule in ("B", "S"):
            for seed in SEEDS:
                j = js(f"exp45/e43/e43_{arm}_{rule}_{seed}.json")
                R[(arm, rule, seed)] = {r["updates"]: (r["score3_new_task"]["taught"]["exact"], r["score3_new_task"]["held"]["exact"]) for r in j["rows"]}
                c = j["complete"]
                print(f"  {arm:15s} {rule} {seed}: decay {j['settings']['decay']} rows {sorted(R[(arm, rule, seed)])} counts {[R[(arm, rule, seed)][u] for u in sorted(R[(arm, rule, seed)])]} "
                      f"cpu per update {c['cpu_updates'] / 4096:.4f}")
    dmax = max(abs(a - b) for (arm, rule, seed), v in R.items() if arm == "unrelated" for u in v for a, b in zip(v[u], R[("unrelated-decay", rule, seed)][u]))
    print("  largest difference between the arm with decay and the arm without, any row and panel:", dmax, "items")
    ext = subprocess.run(["git", "-C", TOP, "ls-tree", "-r", "--name-only", D, "--", f"{M}/exp45/e43/"], capture_output=True, text=True).stdout
    print("  receipts of an extension beyond 4,096 updates on file:", [l for l in ext.splitlines() if "_ext" in l])
    print("E40: reduction score at update 8,192, and cost")
    for kind in ("lived", "fresh"):
        vals, cost = [], []
        for rule in ("S", "B"):
            for seed in SEEDS:
                j = js(f"exp45/e40/e40_{kind}_{rule}_{seed}.json")
                assert j["rows"][-1]["updates"] == 8192
                vals.append(j["rows"][-1]["scores"]["summary"]["reduction_score"]); cost.append(j["cpu_updates"])
                if kind == "fresh":
                    assert "tensors_sha256" in j["start"]
        print(f"  {kind}: {min(vals):.2f} to {max(vals):.2f}; CPU-seconds of the updates {min(cost):.0f} to {max(cost):.0f} ({min(cost) / 8192:.4f} to {max(cost) / 8192:.4f} an update)")
    print("  fresh start: one tensors_sha256 for both rules of a seed:", all(js(f"exp45/e40/e40_fresh_S_{s}.json")["start"]["tensors_sha256"] == js(f"exp45/e40/e40_fresh_B_{s}.json")["start"]["tensors_sha256"] for s in SEEDS))
    print("The starts on the default branch")
    for rule in ("S", "B"):
        for seed in SEEDS:
            b = js(f"exp45/e26/e26_dev_{rule}_{seed}.json"); l = js(f"exp45/e26/e26_life_{rule}_{seed}.json")
            srcs = b["provenance"]["sources_sha256"]
            diff = sorted(k for k in srcs if hashlib.sha256(blob(k)).hexdigest() != srcs[k])
            okb = hashlib.sha256(blob(f"exp45/e26/e26_dev_{rule}_{seed}.pt")).hexdigest() == b["weights_sha256"]
            okl = hashlib.sha256(blob(f"exp45/e26/e26_life_{rule}_{seed}.pt")).hexdigest() == l["complete"]["weights_sha256"]
            e43 = js(f"exp45/e43/e43_unrelated-decay_{rule}_{seed}.json"); e40 = js(f"exp45/e40/e40_lived_{rule}_{seed}.json")
            print(f"  {rule} {seed}: birth file = receipt {okb}; life file = receipt {okl}; birth updates {b['optimizer_steps']}, life updates {l['complete']['optimizer_steps']}; parameters {b['model']['params']}; "
                  f"sources of the birth that differ on the default branch {diff}; E43's birth_tensors_sha256 = birth receipt {e43['settings']['birth_tensors_sha256'] == b['tensors_sha256']}; "
                  f"E40's lived start = life receipt {e40['start']['tensors_sha256'] == l['complete']['tensors_sha256']}")
    print("  E43's admission:", js("exp45/e43/e43_unrelated-decay_S_73129.json")["settings"]["birth_admission"])


def task():
    print("chance that a given symbol is absent from a given place of n pairs drawn without replacement from 4,096")
    for n in (256, 512):
        p = float(Fraction(comb(4096 - 64, n), comb(4096, n)))
        print(f"  n = {n}: {100 * p:.4f}% (expected absent symbols in a place {64 * p:.3f}); exp(-64 n / 4096) = {100 * 2.718281828459045 ** (-64 * n / 4096):.4f}%")
    for seed in SEEDS:
        taught, held = split(seed); hp = panel(seed, "held"); hpairs = [(p[1], p[2]) for p, _ in hp]
        assert set(taught).isdisjoint(held) and all(pr in set(held) for pr in hpairs) and len(set(hpairs)) == 300
        assert taught_panel_n(seed, 2048) == panel(seed, "taught")
        print(f"seed {seed}: e43_task.digest(seed, 64) {spec_digest(stream_n(seed, 2048), 64)}; proposed taught panel at 2,048 is E43's: True")
        for n in (2048, 512, 256):
            sub = taught[:n]; c1 = collections.Counter(a for a, _ in sub); c2 = collections.Counter(b for _, b in sub)
            m1 = sorted(set(SYM) - set(c1)); m2 = sorted(set(SYM) - set(c2))
            bad = sum((a in m1) or (b in m2) for a, b in hpairs)
            try:
                Random(f"e43|{seed}|panel|taught").sample(sub, 300); samp = "possible"
            except ValueError as e:
                samp = f"ValueError ({e})"
            tp = taught_panel_n(seed, n)
            print(f"  n = {n}: symbols absent from the first place {m1}, from the second {m2}; a symbol occurs in a place {min(min(c1[s], c2[s]) for s in SYM)} to {max(max(c1.values()), max(c2.values()))} times; "
                  f"shown once or twice in a place: {sum(0 < c1[s] <= 2 for s in SYM)} + {sum(0 < c2[s] <= 2 for s in SYM)}")
            print(f"           held-panel items with a symbol in a place where no taught pair has it: {bad} of 300; sampling 300 as e43_task.panel does: {samp}; proposed taught panel: {len(tp)} items, "
                  f"{len({(p[1], p[2]) for p, _ in tp})} distinct pairs, all taught: {all((p[1], p[2]) in set(sub) for p, _ in tp)}; e43_task.digest form, first 64 batches: {spec_digest(stream_n(seed, n), 64)}")
    rows = list(range(0, 1025, 32)) + list(range(1152, 8193, 128))
    print("rows of the brief:", len(rows), "| in 7,296..8,192:", sum(7296 <= r <= 8192 for r in rows), "| in 5,248..6,144:", sum(5248 <= r <= 6144 for r in rows), "| last three:", rows[-3:])
    for first in (960, 992, 1024, 7936, 8064):
        tri = rows[rows.index(first):rows.index(first) + 3]
        print(f"  three consecutive rows from {first}: {tri}" + (f", a span of {tri[-1] - tri[0]} updates" if len(tri) == 3 else ": fewer than three rows remain"))
    new = list(range(0, 513, 16)) + list(range(544, 8193, 32))
    print("rows of the repair (every 16 to 512, every 32 after):", len(new))


def votes():
    """The brief's rule on two of three seeds, applied to one invented set of updates to 80 (fresh, born, lived)."""
    data = {1: (1000, 600, 840), 2: (1000, 600, 360), 3: (1000, 1400, 840)}

    def word(first, second):
        q = [data[s][first] / data[s][second] for s in data]
        lo, hi = sum(v < 2 / 3 for v in q), sum(v > 1.5 for v in q)
        w = "fewer" if lo >= 2 and hi == 0 else "more" if hi >= 2 and lo == 0 else "about the same"
        return w, [round(v, 2) for v in q]
    print("updates to 80 (fresh, born, lived) by seed:", data)
    print("  born against fresh:", word(1, 0)); print("  lived against born:", word(2, 1)); print("  lived against fresh:", word(2, 0))


def digest():
    """exp26_candidate.Digest over exp45.batch_tensor's tensors, written in numpy from their text; tested against E43's receipts."""
    import numpy as np

    def run(seed, n, rows):
        h = hashlib.sha256(); st = stream_n(seed, n); out = {}
        for u in range(1, max(rows) + 1):
            items = [next(st) for _ in range(64)]
            L = max(len(p) + len(t) for p, t in items) + 1
            data = np.full((64, L), BOS, dtype=np.int64)
            for i, (p, t) in enumerate(items):
                data[i, 1:len(p) + len(t) + 1] = p + t
            x, y = data[:, :-1], data[:, 1:]; mask = np.zeros(y.shape, dtype=np.bool_)
            for i, (p, t) in enumerate(items):
                mask[i, len(p):len(p) + len(t)] = True
            for t_ in (x, y, mask):
                h.update(str(tuple(int(s) for s in t_.shape)).encode()); h.update(np.ascontiguousarray(t_).tobytes())
            if u in rows:
                out[u] = h.copy().hexdigest()
        return out
    for seed in SEEDS:
        want = {r["updates"]: r["data_digest"]["sha256"] for r in js(f"exp45/e43/e43_unrelated-decay_S_{seed}.json")["rows"] if r["updates"]}
        got = run(seed, 2048, set(want) | {8192})
        print(f"seed {seed}: n = 2,048 equals E43's receipt at rows {sorted(want)}: {all(got[u] == want[u] for u in want)}; row 8,192: {got[8192]}")
        for n in (512, 256):
            g = run(seed, n, {64, 8192}); print(f"   n = {n}: row 64 {g[64]}; row 8,192 {g[8192]}")


def weights_part():
    import numpy as np
    rms = lambda a: float(np.sqrt(np.mean(np.square(a.astype(np.float64)))))
    TASKB = list(range(128, 226)); FAM1 = sorted(set(b"0123456789+-><=pqvwABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    W = {}
    print("the byte table's rows 128 to 225; 'fresh' is read from E40's fresh runs (after 8,192 updates on another task; after 64 for seed 73129)")
    for seed in SEEDS:
        files = [(k, r, f) for r in ("S", "B") for k, f in (("fresh8192", f"exp45/e40/e40_fresh_{r}_{seed}_row8192.pt"), ("born", f"exp45/e26/e26_dev_{r}_{seed}.pt"), ("lived", f"exp45/e26/e26_life_{r}_{seed}.pt"))]
        if seed == 73129:
            files += [("fresh64", r, f"exp45/e40/e40_fresh_{r}_{seed}_row64.pt") for r in ("S", "B")]
        ref = None; eq = []
        for k, r, f in files:
            W[(k, r, seed)] = weights(f); rows = W[(k, r, seed)]["book.weight"][TASKB]
            ref = rows if ref is None else ref; eq.append(bool(np.array_equal(rows, ref)))
        print(f"  seed {seed}: {len(files)} files, rows bit-equal in all: {all(eq)}; rms of these rows {rms(ref):.3f}")
        for r in ("S", "B"):
            print(f"     rule {r}: rms of the rows of the first family's bytes: born {rms(W[('born', r, seed)]['book.weight'][FAM1]):.3f}, lived {rms(W[('lived', r, seed)]['book.weight'][FAM1]):.3f}")
    cols = [("fresh64", "S"), ("fresh64", "B"), ("born", "S"), ("born", "B"), ("lived", "S"), ("lived", "B")]
    print("root mean square of each tensor, seed 73129; columns:", cols)
    for n in W[("born", "S", 73129)]:
        print(f"  {n:18s} " + " ".join(f"{rms(W[(k, r, 73129)][n]):8.4f}" for k, r in cols))
    print("  parameters:", sum(v.size for v in W[("born", "S", 73129)].values()))
    print("range over the three seeds of the ratio born/initial and lived/initial, the initial rms taken as torch's default (uniform on +-1/sqrt(fan-in))")
    init = {"l1.weight": 8 ** -0.5 / 3 ** 0.5, "rnn.weight_ih_l0": (1 / 16) / 3 ** 0.5, "rnn.weight_hh_l0": (1 / 16) / 3 ** 0.5, "lm.weight": (1 / 16) / 3 ** 0.5,
            "l2.weight": 512 ** -0.5 / 3 ** 0.5, "l3.weight": 512 ** -0.5 / 3 ** 0.5, "l3.bias": 512 ** -0.5 / 3 ** 0.5}
    for n, v0 in init.items():
        q = [rms(W[(k, r, s)][n]) / v0 for k in ("born", "lived") for r in ("S", "B") for s in SEEDS]
        print(f"  {n:18s} initial {v0:.4f}; ratio {min(q):.2f} to {max(q):.2f}")
    print("loss on the answer bytes of the first 64 items of E43's stream, before any update of the task (ln 257 = 5.55)")
    for seed in SEEDS:
        st = stream_n(seed, 2048); items = [next(st) for _ in range(64)]; line = []
        for k, r in (("born", "S"), ("born", "B"), ("lived", "S"), ("lived", "B")):
            ls = []
            for p, t in items:
                lg = gru(W[(k, r, seed)], np.array([[BOS] + p + t[:-1]]), all_steps=True)[0, -3:].astype(np.float64)
                lg -= lg.max(1, keepdims=True); lp = lg - np.log(np.exp(lg).sum(1, keepdims=True)); ls.append(-mean([lp[j, t[j]] for j in range(3)]))
            line.append(f"{k} {r} {mean(ls):.2f}")
        print(f"  seed {seed}: " + "; ".join(line))


def format_part():
    kind = lambda b: "digit" if 48 <= b <= 57 else "end" if b == EOS else "symbol" if 128 <= b <= 191 else "other"
    for seed in SEEDS:
        items = panel(seed, "held")
        files = [("born S", f"exp45/e26/e26_dev_S_{seed}.pt"), ("born B", f"exp45/e26/e26_dev_B_{seed}.pt"), ("lived S", f"exp45/e26/e26_life_S_{seed}.pt"), ("lived B", f"exp45/e26/e26_life_B_{seed}.pt")]
        if seed == 73129:
            files += [("E43 unrelated S, row 64", "exp45/e43/e43_unrelated_S_73129_row64.pt"), ("E43 unrelated B, row 64", "exp45/e43/e43_unrelated_B_73129_row64.pt")]
        for name, f in files:
            outs = respond(weights(f), items)
            print(f"seed {seed} {name:24s}: whole answer right {sum(o == t for o, (_, t) in zip(outs, items)):3d}; three bytes with the end symbol third {sum(len(o) == 3 and o[2] == EOS for o in outs):3d}; "
                  f"right by place {[sum(len(o) > j and o[j] == t[j] for o, (_, t) in zip(outs, items)) for j in range(3)]}; commonest forms {collections.Counter(tuple(kind(b) for b in o) for o in outs).most_common(2)}")


def row0():
    """Right answers of the born and the lived networks before any update of the task, on every panel of the repaired brief."""
    tot = 0
    for seed in SEEDS:
        pans = {"held 12": panel(seed, "held"), "held 0": panel(seed, "held", k=0), "taught 2,048": taught_panel_n(seed, 2048), "taught 512": taught_panel_n(seed, 512),
                "taught 256": taught_panel_n(seed, 256)}
        for name, f in (("born S", f"exp45/e26/e26_dev_S_{seed}.pt"), ("born B", f"exp45/e26/e26_dev_B_{seed}.pt"), ("lived S", f"exp45/e26/e26_life_S_{seed}.pt"), ("lived B", f"exp45/e26/e26_life_B_{seed}.pt")):
            w = weights(f); r = {k: sum(o == t for o, (_, t) in zip(respond(w, it), it)) for k, it in pans.items()}; tot += sum(r.values())
            print(f"seed {seed} {name}: {r}")
    print("right answers in all:", tot)


def flat():
    """The brief's rule 'has flattened' at update 8,192, on E41's curves of fresh networks on the first family (held-out pairs at 12 letters)."""
    n = fl = w5 = w10 = 0
    for h in (512, 256):
        for f in (10, 25, 50):
            for rule in ("S", "B"):
                for seed in SEEDS:
                    c = js(f"exp45/e41/e41_{rule}_h{h}_f{f}_{seed}.json")["curve"]; v = {r["step"]: r["A"] for r in c}; t = {r["step"]: r["taught"] for r in c}
                    if any(all(v[s + 256 * i] >= 80 for i in range(3)) for s in sorted(v) if s <= 8192 - 512):
                        continue
                    late = mean([v[s] for s in v if 7296 <= s <= 8192]); early = mean([v[s] for s in v if 5248 <= s <= 6144])
                    l8 = mean([v[s] for s in (7680, 7936, 8192)]); le = mean([v[s] for s in sorted(v)[-3:]]); isf = late - early < 3
                    n += 1; fl += isf; w5 += isf and le - l8 >= 5; w10 += isf and le - l8 >= 10
                    print(f"  width {h} fraction 0.{f} {rule} {seed}: windows {early:5.1f} -> {late:5.1f}; flattened {str(isf):5s}; level at 8,192 {l8:5.1f}, taught panel {t[8192]:5.1f}; level at {max(v)} {le:5.1f}")
    print(f"curves below 80 at 8,192: {n}; called flattened: {fl}; of these higher later by 5 points or more: {w5}; by 10 or more: {w10}")
    print("fresh networks, sweep against back-propagation, updates to 80 on held-out pairs at 12 letters (first of three rows), E41 at the present width")
    for f in (50, 25):
        for seed in SEEDS:
            x = {}
            for rule in ("S", "B"):
                v = [(r["step"], r["A"]) for r in js(f"exp45/e41/e41_{rule}_h512_f{f}_{seed}.json")["curve"]]
                x[rule] = next((v[i][0] for i in range(len(v) - 2) if all(a >= 80 for _, a in v[i:i + 3])), None)
            print(f"  fraction 0.{f} seed {seed}: sweep {x['S']}, back-propagation {x['B']}, ratio {None if None in x.values() else round(x['S'] / x['B'], 2)}")
    print("fresh networks on a family: the taught panel in the first rows")
    for rule in ("S", "B"):
        for seed in SEEDS:
            j = js(f"exp45/e40/e40_fresh_{rule}_{seed}.json")
            print(f"  E40 fresh {rule} {seed}: (update, taught panel) {[(r['updates'], round(r['scores']['summary']['taught_add_sub_mean'], 1)) for r in j['rows']]}")


if __name__ == "__main__":
    P = dict(figures=figures, task=task, votes=votes, digest=digest, weights=weights_part, format=format_part, row0=row0, flat=flat)
    for a in sys.argv[1:] or list(P):
        print(f"==== {a}"); P[a]()
```
