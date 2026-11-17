# Check of the brief of E58 before its release (17 November 2026)

**Brief checked:** `TASK_E58_A_GATED_TERM_AGAINST_FREE_LEARNING_DECAY_AND_NUMBER_OF_PAIRS_20261117.md`, revision 1, beside this file, as committed at 54901576.

**How.** The check was made on 17 November 2026 from 22:16 UTC (the closing time is at the end of the document) by an agent of the reviewer that had not seen the reviewer's reasoning. Nothing was trained and torch was not installed. Every figure below was computed with the Python standard library and numpy from the receipts and weight files that the brief names, read with `git show` from the default branch (`D`, at 7cabb13a) and from the branch of E48's worker (`W8`, at 2d9d1726). No code of another branch was executed; where its logic was needed, the lines were copied into the scripts of the appendix. "Point 2", "check 5" and the like are the brief's numbers.

## Verdict

**Release after the listed changes.**

The runner can be built as the brief describes, checks 1 to 3 can be carried out and can fail when they should, and all figures of the record but three are right (the three are wrong by 0.1). The design, however, licenses three sentences that its data would not support, and one arm cannot be read:

- "The gated term needs fewer pairs" would be stated when the number of pairs plays no part at all, and it is licensed by the absence of a difference (defect 1).
- `term-shrink` imposes on the lived network a shrinking that free learning does not show, stops the network from working by row 1,024, and gives, for the earlier families, figures that are already known without a run (defect 2).
- `free-wd0` removes two things at once, and `term` is compared with an arm that differs from it in two things, so that "the place of the change accounts for it" can be stated when the decay of the lived weights is a cause (defect 3).

Twelve defects follow, the most serious first. Each has an exact replacement.

## Defects

### 1. The licence for "the gated term needs fewer pairs" licenses the sentence when the number of pairs plays no part, and on the absence of a difference

**What is wrong.**

(a) *No condition on free learning.* The brief (Part B, last paragraph) licenses the sentence "by `term` at f = 0.05 not being below `free` at f = 0.10, or by `term` at f = 0.10 not being below `free` at f = 0.20". Nothing requires that free learning gain from the doubled pairs. If the level does not depend on the fraction in this range (for example `free` at 83 and `term` at 90 at every fraction), `term` at 0.05 is not below `free` at 0.10 and the sentence is licensed, although free learning also reaches with half the pairs what free learning reaches. The record does not exclude this case: no run exists at another fraction.

(b) *"Not below" is the absence of a verdict.* By the brief's own rule, X is below Y only if Y leads by 5 points or more on two seeds and is not behind by 5 or more on the third. `term` at 0.05 lower than `free` at 0.10 by 4.9 points on every seed is "not below"; so are differences of −10, −10 and +6. With three seeds, "not below" is what scatter alone produces.

(c) *The ceiling rule misses the common case.* "Where both levels of a comparison are above 97" nothing is licensed. With `free` at 98.5 and `term` at 96 (a deficit of 2.5 points and an error rate 2.7 times as large) one level is below 97, the comparison is "not below", and the sentence is licensed. Under back-propagation the term of rank 16 is already at 93.1, 97.9 and 98.7 at f = 0.10 (row 1,024).

(d) *Only one of the two free arms is the comparator.* If `free-wd0` is above `free` at the larger fraction, the better free learner is `free-wd0`, and the sentence is still judged against `free`.

(e) *The pair 0.05 against 0.20, a factor of four, is in the runs and is not read.* Requirement B2 speaks of "far fewer samples"; the brief reads two factors of two.

(f) *"Fewer pairs" is not "fewer samples" in every sense.* At an equal row all arms have been presented the same number of items (524,288 at row 8,192); the fractions differ in the number of different pairs, so that each taught pair has been presented 381 times at f = 0.05, 190 times at 0.10 and 95 times at 0.20 (appendix, `splits.py`). In presented items needed to reach a level the record points the other way: the term of rank 16 reaches 80% on the reduction pairs at 256, 512 and 512 updates under the sweep, E40's free learning at 128, 256 and 128 (`E48_RESULT_VERIFICATION_20261117.md`, section 6.2). The brief does not say which sense it measures.

(g) *The comparison that B2 names is across the rules.* B2 compares "our learner" with "the conventional one". Part B compares arms within each rule only. `term` under the sweep against `free` under back-propagation is in the same runs.

**Repair.** Replace the paragraph "**Part B, for each rule.** …" by the following (the *common pairs* and the nested taught sets are introduced in defect 6):

> **Part B, for each rule.** A table by arm, fraction and seed of R-late, of R-late on the common pairs, and of the taught panel at row 8,192; beside it, for every run, the first row at which the reduction score is 80 or more and the first at which it is 90 or more, with the number of presentations of each taught pair at that row. At each fraction, `term` against `free` and against `free-wd0` in the words above.
>
> Every comparison between two fractions is made on R-late of the score on the common pairs, so that both sides are scored on the same pairs. For a rule and a pair of fractions (f, g) out of (0.05, 0.10), (0.10, 0.20) and (0.05, 0.20), the report says "the gated term reaches from f the level that free learning reaches from g" only if all three of the following hold: (a) `free` at g is above `free` at f, so that free learning gains from the additional pairs; (b) on at least two seeds `term` at f is at most 2 points below the higher of `free` and `free-wd0` at g, and on the third seed it is less than 5 points below; (c) the comparison is not at the ceiling. If (a) fails, the report says that between f and g the level of free learning does not depend on the number of taught pairs, and makes no statement on pairs. If (b) fails, the report gives the deficit by seed.
>
> The same comparisons are tabled across the rules, `term` under the sweep at f against `free` and `free-wd0` under back-propagation at g, because requirement B2 compares our learner with the conventional one; they are reported in the same words.
>
> These statements concern the number of different pairs at an equal number of presented items (524,288 at row 8,192: each taught pair 381 times at f = 0.05, 190 times at 0.10, 95 times at 0.20). They are not statements on the number of items a learner needs; the table of first rows shows that.

Add to "What it cannot show":

> - That the gated term needs fewer presented items. Part B compares numbers of different pairs at an equal number of presented items. In E48 the term reached a given level later than free learning (`$V:$R/E48_RESULT_VERIFICATION_20261117.md`, section 6.2).
> - More than a factor of four in the number of pairs. A fraction below 0.0436 (300 of 6,889 pairs) cannot be scored by E40's scorer unchanged, because its taught panel samples 300 taught pairs.

### 2. `term-shrink` cannot be read, and the half of it that concerns the earlier families is known without a run

**What is wrong.**

(a) *It is not what the standing decay does in free learning.* The brief calls the arm "the standing decay acting alone" and uses it (point 4; last sentence of "Predictions") to test "whether damage to the lived weights alone lowers the level". In free learning the decay never acts alone. In E40's six lived runs the norm of all tensors outside the byte table is 0.996 to 1.001 of its starting value at row 64 and 0.836 to 0.970 at row 8,192 (single tensors: 0.54 to 1.40); decay acting alone would give 0.984 and 0.129 (appendix, `norms.py`, from `$D:$M/exp45/e26/e26_life_<rule>_<seed>.pt` and `$D:$M/exp45/e40/e40_lived_<rule>_<seed>_row{64,8192}.pt`). `term-shrink` imposes 0.774 at row 1,024, more shrinking than free learning shows after 8,192 updates.

(b) *The shrinking stops the network from working.* The earlier families' items never open the gate, so their answers in `term-shrink` depend on the shrunk lived weights only and can be computed now. A numpy forward of the network, which reproduces every count of the receipts on all six networks (appendix, `validate.py`: the six earlier panels and the reduction pairs at E40's row 0, and the reduction pairs at E48's row 1,024 with the saved term; 0 of 6 networks differ), gives for the mean of the four panels of the life's families (appendix, `shrink.py`):

| Row | Factor | Loss against row 0, sweep (three seeds) | Loss, back-propagation | Level, six networks | E40's `free` at that row |
|---|---|---|---|---|---|
| 64 | 0.984 | 0.9, 2.2, 0.8 | 1.1, 0.1, 0.1 | 85.7 to 96.7 | 42.4 to 69.4 |
| 256 | 0.938 | 9.7, 13.9, 7.4 | 4.4, 1.2, 1.8 | 74.4 to 95.6 | 45.2 to 58.4 |
| 512 | 0.880 | 35.4, 42.8, 32.8 | 20.7, 7.6, 9.2 | 45.5 to 89.2 | 43.8 to 51.8 |
| 1,024 | 0.774 | 79.2, 83.1, 86.0 | 70.1, 65.3, 64.1 | 5.2 to 31.5 | 38.0 to 49.2 |

R-mid is the mean over rows 512 and 1,024. At row 1,024 the frozen network answers 5 to 31% of the life's families, less than free learning leaves (38 to 49%). The arm does not damage "what the network learned from the earlier families" in particular; it stops the network from working.

(c) *The new family's items are moved as much, and the term cannot repair it.* E48's saved term of rank 16 (row 1,024), placed on the shrunk network, loses 0.8 to 10.0 points of reduction score at the factor of row 256, 7.6 to 34.0 at row 512 and 46 to 81 at row 1,024 (same script). In the arm the term is trained while the network shrinks, so it will recover a part; but it is two products of rank 16 on the recurrent layer and four rows of the byte table, and the shrinking acts on every layer (`l1`, the recurrent layer, `lm`, `l2`), whereas in free learning the gradient can restore every weight. A lower R-mid would show that such a term cannot follow a network that shrinks in all its layers. It would not show that the decay lowers the level in free learning.

(d) *What a reader would wrongly conclude.* The outcome "`term-shrink` is below `term`" is close to certain from (b) and (c) (P58.6 gives it 50%), and the brief would read it as support for the explanation: damage to the lived weights lowers the level, so the damage that free learning does is the cause of its lower level. Neither step follows. Only the opposite outcome would carry information.

(e) *The earlier-family half needs no worker.* "What the shrinking does to the earlier families' panels at rows 512 and 1,024" (point 4) is the table above.

(f) "where nothing of the network would be left" (sentence after the table of Part A) is said of the factor 0.129. The earlier families are at 5 to 31% at 0.774 already.

**Repair.** Remove the arm, and say why the fourth cell of the design is absent.

- In the table of Part A delete the row `term-shrink`. Delete the paragraph "`term-shrink` stops at 1,024 updates … that is its purpose."
- In "The runner" delete the sentence "`term-shrink` multiplies, after `opt.step()`, …"; in "Files committed" delete "for `term-shrink` the final weights of the network as well".
- Check 4 is replaced by the check of `free-anchor` (defect 3). Point 4 of Part A is replaced as in defect 3. P58.6 is replaced as in defect 5.
- Add to "What the record says":

> - **Shrinking without learning.** If every tensor of a lived network outside the byte table is multiplied by (1 − 0.00025) r times and nothing is learned, the mean of the four panels of the life's families falls by 0.1 to 2.2 points at r = 64, by 1.2 to 13.9 at r = 256, by 7.6 to 42.8 at r = 512 and by 64 to 86 at r = 1,024 (`$V:$R/COORDINATION_20261117/CHECK_OF_TASK_E58_BRIEF_20261117.md`, defect 2). In E40's free learning the same mean has fallen by 17 to 46 points at row 64, where the norm of the weights outside the byte table is 0.996 to 1.001 of its starting value; at row 8,192 the norm is 0.84 to 0.97, where decay acting alone would give 0.129. The loss of the earlier families in the first 64 updates of free learning is therefore not a loss by shrinking.

- Add to "What it cannot show":

> - What the decay of the lived weights does when the change is confined to the term. This fourth cell of the design cannot be realised: in free learning the gradient opposes the decay in every weight, and a term on two matrices cannot (a lived network shrunk by the decay of 1,024 updates answers 5 to 31% of its life's families).

### 3. `free-wd0` against `free` does not isolate the decay of the lived weights, and `term` is compared with an arm that differs from it in two things

**What is wrong.**

(a) *The standing decay is two pulls, and `free-wd0` removes both.* AdamW replaces a weight w by w − 0.00025·w before its step. With w0 the lived value, this is w − 0.00025·(w − w0) − 0.00025·w0: a pull of the change toward zero and a pull of the lived weight toward zero. The first is the only regularisation of what is newly learned; `term` has its counterpart (the decay of the factors pulls the added product toward zero). `free-wd0` removes both pulls. Its difference from `free` is therefore "weight decay or none", not "the decay of the lived weights". In the literature on modular arithmetic learned from a part of the pairs, weight decay is the setting reported to favour generalisation most (Power et al., 2022), and the record has no run without decay (the brief's last bullet of "What the record says"), so that an effect of the regularisation, of either sign, cannot be excluded.

(b) *The size of the weights.* Under `free` the norm outside the byte table is held between 0.84 and 0.97 of its start over 8,192 updates while single tensors change by 12 to 125% of their norm (appendix, `norms.py`). Without decay nothing bounds the norm, so that rows 4,096 to 8,192 of `free-wd0` and of `free` differ in the size of the weights and thereby in the size of a step of 0.001 relative to them.

(c) *An unmatched pair.* Point 2 compares `term`, whose change decays, with `free-wd0`, in which nothing decays: the two differ in the place of the change and in the regularisation of the change. The matched pairs are `term` with a free arm whose change decays and whose lived weights do not, and `term-wd0` with `free-wd0`. The brief uses `term-wd0` in point 3 only.

(d) *A sentence that the data would not support.* If keeping the lived weights raises the level and losing the regularisation lowers it, `free-wd0` is not distinguished from `free`, `term` is above `free-wd0`, and point 2 says "the place of the change accounts for it", although the decay of the lived weights is a cause. The same sentence is licensed when `free-wd0` leads `free` by 4 points and `term` leads `free-wd0` by 5 on every seed: the decay then gives 4 of the 9 points. "Not distinguished" is used as if it were "no effect".

**Repair.** Add one free arm in which the change decays and the lived weights do not; it separates the two pulls exactly. Split the lead by seed instead of chaining verdicts. (The arm has the form of the pull toward earlier weights that the plan sets aside as a means of protection, section 6, entry D25. Here it is not a means of protection; it is the control that removes one of the two pulls of the standing decay and keeps the other.)

In "Terms" add:

> - *Decay toward the start* (`free-anchor`): after each update every tensor outside the byte table is moved toward its value in the lived network by 0.00025 of its distance from it. The standing decay moves a weight w with lived value w0 by −0.00025·w = −0.00025·(w − w0) − 0.00025·w0; `free-anchor` keeps the first part, which pulls the change toward zero, and drops the second, which pulls the lived weight toward zero.

In the table of Part A add (the row `free-again` is in defect 4):

> | `free-anchor` | every weight | none in the optimizer; decay toward the start at 0.00025 an update, byte table none | 8,192 | 6 |

In "The runner", after the bullet on the free arms:

> - **`free-anchor`** uses the optimizer of `free-wd0`. Before the first update the runner keeps a copy p0 of every parameter except `book.weight`. After each call of `exp26_candidate.step` it executes, under `torch.no_grad()`, `p.sub_(p - p0, alpha=0.00025)` for each of them.
> - **Norms.** In every arm the receipt records at every row, for every tensor outside the byte table, its norm and the norm of its difference from the lived network's tensor.

Replace check 4 by:

> 4. **`free-anchor`.** For seed 73129 under each rule: (a) applied to a copy of the lived network in which every tensor outside the byte table has been doubled, the decay toward the start leaves each such tensor at 1.99975 times its lived value, with a largest absolute deviation below 1e-6 (the standing decay would leave 1.9995 times, which differs by 6e-6 or more in every tensor), and leaves the byte table bit-equal; (b) applied to the lived network itself it changes no tensor; (c) the optimizer read back shows a weight decay of 0 in every group.

Replace points 2 to 5 of Part A by:

> 2. If the lead persists, it is split on each seed. L = R-late(`term`) − R-late(`free`); A = R-late(`free-anchor`) − R-late(`free`); P = R-late(`term`) − R-late(`free-anchor`); so L = A + P. A is what the absence of the decay of the lived weights adds when the change is made in every weight and decays in both arms. P is what the place of the change adds when the lived weights decay in neither arm and the change decays in both. The report gives L, A and P for every seed. On each seed on which L is 5 points or more (at least two seeds), the share a = A / L is formed. The report says "the decay of the lived weights accounts for the lead" if a is 0.75 or more on each of these seeds; "the place of the change accounts for the lead" if a is 0.25 or less on each; "both contribute" if a lies between 0.25 and 0.75 on each; otherwise it says that the seeds disagree and gives the table without one of the three sentences.
> 3. The decay of the change: `term-wd0` against `term`, and `free-wd0` against `free-anchor`, show whether the level changes when the change no longer decays, in either place; `term-wd0` against `free-wd0` compares the two places when nothing decays. These are reported in the words above and enter no sentence of point 2.
> 4. `free-wd0` against `free` is reported as the effect of removing the weight decay altogether. It is not read as the effect of the decay of the lived weights.
> 5. For `free`, `free-wd0` and `free-anchor` the report gives, at every row, the mean of the four panels of the life's families (add and sub under 97 and 73), the mean of the two panels of the first family beside it, and the norms recorded in the receipts. These show whether the loss of the earlier families in free learning depends on the decay, and whether, across the three arms and the seeds, the run that keeps more of the life's families also has the higher R-late, as the explanation under test predicts.

### 4. The rule "above / below / not distinguished": the margin is not measured, the rule is not transitive, and the ceiling rule fails where it is needed

**What is wrong.**

(a) *Nothing in the record measures the scatter of the difference between two runs.* The brief supports the margin of 5 points with the standard deviation of single rows within a run (1.0 to 3.3; recomputed: 1.01 to 3.35) and adds "if the rows were independent, which is not known". With only four rows from 1,024 on in each of E40's runs the dependence cannot be estimated, and under the sweep the four rows are not a stationary scatter: all three runs are lower at row 8,192 than at row 1,024 (by 7.5, 1.8 and 4.7 points). If neighbouring rows move together, the scatter of a difference of two five-row means is up to 1.4 to 4.7 points, and 5 points is then 1.1 to 3.5 standard deviations. No two runs of one arm on one network exist in the record, so the margin cannot be defended from it. Two runs that differ only in the order of the items would measure it, for 0.33 CPU-hours each.

(b) *The rule is not transitive, in both directions.* Per-seed differences of X − Y = (5, 5, −4) and Y − Z = (−4, 5, 5) give "X above Y" and "Y above Z", and X − Z = (1, 10, 1), "not distinguished". Differences of (4, 4, 4) and (4, 4, 4) give two "not distinguished" and X − Z = (8, 8, 8), "above". Point 2 of the brief chains two verdicts (defect 3 (d)).

(c) *The ceiling.* Where the lower figure is above 95, a lead of 5 points is impossible, so "not distinguished" is forced; the brief's rule ("both above 97") does not catch a pair such as 95.5 and 99.5. P58.10 predicts exactly such a forced verdict.

(d) *R-late hides the direction of movement.* Between rows 4,096 and 8,192 E40's runs move by −4.4, −3.4, +1.3 (sweep) and +2.7, +4.0, −2.1 (back-propagation): there is no common drift inside the window, so a mean over it is reasonable. From row 1,024 to row 8,192, however, the sweep's three runs fall, and E48's term was still rising at row 1,024 (by 0.1 to 4.8 points from row 512). A lead on R-late can therefore come from a fall of `free`, a rise of `term`, or both, and the table of R-late alone does not say which.

**Repair.** In the table of Part A add:

> | `free-again` | as `free`, with the items in another order: the stream is `X16.entry_stream(e40_new_family.entries(seed), seed + 141)` in batches of 64, so that its batch digests are its own | as `free` | 8,192 | 6 |

Replace the two bullets under "How the result is stated (fixed beforehand)" by:

> - *R-late* of a run is the mean of its reduction score over the five rows 4,096, 5,120, 6,144, 7,168 and 8,192. *R-mid* is the mean over rows 512 and 1,024. For every run the table gives R-mid, R-late, the highest single row from row 64 on with its row number, and the score at row 8,192 minus the score at row 4,096.
> - *The scatter of one arm.* `free-again` differs from `free` only in the order of the items. N, for a seed and a rule, is the absolute difference between R-late of `free-again` and R-late of `free`. If N is 5 points or more on any seed under a rule, the margin below does not separate two runs of one arm under that rule; the report then gives the tables for that rule and states no verdict under it, and a prediction that depends on a verdict under that rule is recorded as not decided. The report gives N for every seed and rule.
> - For two sets of three runs X and Y under one rule (one run a seed), X is *above* Y if X's figure exceeds Y's by 5 points or more on at least two seeds and X is not below Y by 5 points or more on the third; *below* is the mirror; otherwise the two are *not distinguished*. The figure is R-late unless another is named. "Not distinguished" is the absence of a verdict, and no sentence of the report is licensed by it alone. The rule is not transitive (differences of 5, 5, −4 and of −4, 5, 5 add to 1, 10, 1), so no verdict is derived from two others; each is read from the per-seed differences of the two sets it names.
> - *The ceiling.* On a seed on which the lower of the two figures is above 95, a lead of 5 points is not possible. A comparison in which this holds on two or three seeds is reported as at the ceiling of the score, with the figures and 100 minus each figure, and no verdict is stated.
> - From row 1,024 on, the single rows of E40's runs have a standard deviation of 1.0 to 3.3 points within a run. A difference of two means of five rows has a scatter of about 0.6 to 2.1 points if the rows are independent, and of up to 1.4 to 4.7 if neighbouring rows move together; `free-again` measures which holds.

Replace point 1 of Part A by:

> 1. If `term` is above `free`, the lead of the gated term persists to 8,192 updates. The report says, for each rule, whether the lead on R-late is larger or smaller than the lead on R-mid and whether `free` fell, `term` rose, or both, between R-mid and R-late. If `term` is not above `free`, the report says that the gated term's lead at 1,024 updates does not persist to 8,192, gives both curves, and makes none of the statements of point 2.

### 5. The refutation clause counts as a refutation an outcome that supports half of the explanation; three predictions are affected

**What is wrong.** The explanation has two parts: free learning lowers the level by damaging what the network learned from the earlier families, and the remedy is to confine the change. The brief calls the explanation refuted "if `free-wd0` is above `free` and `term` is not above `free-wd0` under both rules (the decay then accounts for the hint)". In that outcome free learning does lose level through what it does to the lived weights (by their decay): the first part is supported and only the second is refuted. "Under either rule" in the first clause can be read as "under at least one rule" or as "under neither rule". The last sentence ("`term-shrink` tests one part of it directly") falls with defect 2. P58.6 concerns the removed arm and, as written, gives 50% to an outcome that the weights make close to certain. P58.7 takes the mean of six panels, of which the two of the first family stand at 13.5 to 37.3% at row 0 and rise in some of E40's runs; the record's own figure (38 to 43 points) is on the four panels of the life's families. P58.10 predicts a verdict that the ceiling forces.

**Repair.** Replace the paragraph "The explanation under test: …" by:

> The explanation under test has two parts. The first: a new family is generalised from few pairs by means of what the network learned from the earlier families (a network that was not trained before does not generalise it), and free learning lowers the level it reaches by changing the weights that hold this knowledge. The second: a change confined to connections that only the new family uses therefore reaches a higher level from the same pairs, or the same level from fewer. For this setting, both parts are refuted if `term` is above `free` under neither rule. The second part is refuted, and the first is not, if under both rules the decay of the lived weights accounts for the lead (point 2): free learning then reaches the term's level as soon as the lived weights do not decay. No arm of this job damages the lived weights without learning in them; the reason is given under "What it cannot show".

In the table of predictions replace three rows (the chances are the reviewer's to set):

> | P58.6 | `free-anchor` is above `free` at f = 0.10 under at least one rule | …% |
> | P58.7 | The mean of the four panels of the life's families at row 1,024 is higher in `free-anchor` than in `free` by 10 points or more on two of three seeds, under at least one rule | …% |
> | P58.10 | At f = 0.20 under back-propagation, the comparison of `term` with `free` is at the ceiling, or the two are not distinguished | 60% |

and the statement of P58.1 by "Checks 1 to 5 and 8 pass".

### 6. Part B: the taught pairs of the three fractions are unrelated draws, the scores are taken on different pairs, 0.05 leaves operand values untaught, and there is no trial on one seed

**What is wrong.**

(a) *Unrelated draws.* `exp11_cell.split` seeds its shuffle with `seed * 1000 + round(frac * 100) + m * 7 + code[0]`. The fraction enters the seed, so each fraction has its own shuffle: of the 344 pairs taught at 0.05, 27 to 40 are among the 689 taught at 0.10 (chance: 34), and of these 689, 131 to 164 are among the 1,378 taught at 0.20 (chance: 138) (appendix, `splits.py`). "Fewer pairs" is then confounded with "other pairs". The twelve shuffle seeds of a network seed are distinct, so the mechanism works as the brief says; it does not give subsets.

(b) *Different test pairs.* The reduction score is taken on the untaught pairs of the run's own fraction: 3,227 to 3,251 pairs of an operation at 0.05, 3,055 to 3,078 at 0.10, 2,716 to 2,739 at 0.20. A comparison between fractions compares scores on different sets. The effect of the identity of the pairs is small in expectation, but it is avoidable.

(c) *Values never taught in a position.* E40's brief counted that at 0.10 every value from 0 to 82 is taught in each operand position of each operation. At 0.05 this fails on every seed: one to four values of add or sub are absent from a position, and 0 to 170 of an operation's untaught reduction pairs (up to 5.2%) contain such a value; at 0.10 and 0.20 none does (appendix, `coverage.py`). On the mean of add and sub this can cost up to 2.9 points at 0.05, more than half the margin, in a comparison with another fraction.

(d) *No trial on one seed.* Rule 4 of the plan's section 4 asks for "a trial on one seed before more". Part A has one; Part B launches 36 runs at two new fractions after a check of 64 updates.

(e) *The stated question is weaker than the requirement.* "Half the pairs" is a factor of two; B2 says "far fewer". No added fraction is needed for the stated question, and a fraction below 0.0436 would break E40's scorer (defect 1, repair).

**Repair.** Replace the bullet "**The fraction.**" of "The runner" by:

> - **The fraction.** `e40_new_family.split` is the only place where the taught pairs are chosen: the stream (`entries`, `batches`), the scorer (`Scorer`, `held_plain`) and the record of presented items (`Presented`) all call it by its module-level name when they are built (`batches` at its first `next`). Before any of them is built the runner replaces it once, `e40_new_family.split = lambda t, seed: nested(t, f, seed)`, with
>
>       def nested(t, f, seed):
>           taught, rest = E.split(t, 0.10, seed)     # E40's shuffled list of the 6,889 pairs, cut at 689
>           pairs = taught + rest
>           k = round(f * len(pairs))
>           return pairs[:k], pairs[k:]
>
>   At f = 0.10 this is E40's split itself. At f = 0.05 the taught pairs are the first 344 of E40's 689, and at f = 0.20 they are E40's 689 and the next 689 of the same list. (`exp11_cell.split(t, f, seed)` is not used for the other fractions: the fraction enters its shuffle's seed, and it draws unrelated pairs.) The runner records the fraction in the receipt and asserts in every run, after the stream's entries, the scorer and the record of presented items are built: (a) the number of taught pairs of each operation, 344, 689 or 1,378; (b) that the stream's entries are exactly the taught pairs; (c) that for add and for sub the numbers of items of the scorer's two panels of untaught pairs without letters add to 6,889 minus the number of taught pairs; (d) at f = 0.10, that the taught pairs equal `exp11_cell.split(t, 0.10, seed)[0]`.
> - **The common pairs.** For add and for sub of a seed, the common pairs are the pairs that need the reduction, are untaught at f = 0.20 (and so at every fraction), and whose two operand values each occur in the same position in at least one pair taught at f = 0.05. The runner renders them as `held_plain` renders its pairs (`E.render(t, a, b, 0, Random(0))`) and scores them with `X16.score` at every row of every run, after the scorer; the receipt records the two counts and the mean of the two percentages, the *common score*, and the numbers of common pairs.

Replace in "Runs" the sentence "The stream at a fraction is E40's with that fraction: …" by:

> The stream at a fraction is E40's on the nested taught pairs of that fraction ("The runner", the fraction): batches of 64 consecutive items of `X16.entry_stream([("op", t, a, b) for t in F4 for a, b in nested(t, f, seed)[0]], seed + 140)`.

Replace the sentence "Order: …" by:

> The order is: checks 3, 4, 5 and 8; Part A on seed 73129 (twelve runs; checks 1 and 2 are read from its `free` and `term` runs, check 6 from its term runs, check 7 from all twelve); Part A on the other two seeds; Part B on seed 73129 (twelve runs), with its table reported before the other two seeds are launched; Part B on the other two seeds. Nothing waits for a reply from the reviewer unless a check fails.

and add "Part B finished on seed 73129, with its table" to the points of "Reporting".

(A nested split cut from the shuffle of the tenth was verified: it equals `exp11_cell.split` at 0.10 for every seed and operation and gives 344 within 689 within 1,378; the common pairs number 2,491 to 2,718 for an operation and seed; appendix, `nested.py`.)

### 7. The checks

**What is wrong, check by check.**

- **Check 1.** It can fail when `free` is not E40's run, and what stops is clear. Its parenthesis is incomplete: with rows 16 and 32 added, the mean loss since the last row at row 64 covers updates 33 to 64, where E40's covers 1 to 64. A worker who compares it at row 64, the first comparison, finds a difference, and "a difference stops every run".
- **Check 2.** Sound. "Every count of every score" leaves open whether E48's scores with the gate forced off (rows 256 and 1,024) are taken and compared.
- **Check 3.** The statement is exact for AdamW's decoupled decay (the parameter is multiplied by 1 − 0.00025 and the step is then added; the step depends on the gradient only). In float32 the deviation is about 1e-7 (simulated: 8e-8; the largest weight outside the byte table is 0.78 to 1.90, where float32 values are 6e-8 to 1.2e-7 apart), and without the decay it would be 6.4e-6 or more in every tensor (the smallest per-tensor maximum of |w| is 0.0258, in `l1.bias`). The tolerance 1e-6 is safe and the check can fail. It names neither the seed, nor the rule, nor the sign of the difference.
- **Check 4.** Falls with defect 2.
- **Check 5.** (i) A run of 64 updates presents 4,096 items; at f = 0.20 the stream has 5,512 taught entries, so the run does not pass once through the taught pairs. E40's own check 5 walks the whole stream of 8,192 batches through the record of presented items without a network. The brief's form is weaker and costs training. (ii) "Under each rule" adds nothing: the stream does not depend on the rule. (iii) If the taught pairs against which the panels are checked are obtained through `e40_new_family.split`, the check compares the runner's setting with itself and cannot fail when that setting is wrong. (iv) It covers seed 73129 only, although the splits of the other seeds are as new.
- **Check 6.** It names one of the three parts of E48's check 3 (the digest of the answers). The stronger part, the hash of the lived network's tensors with the four code rows reset, costs nothing. It does not say that anything stops.
- **Check 7.** It has no threshold and cannot fail.
- **Missing.** No check shows that `term-wd0` differs from `term` in the decay only; the sentence on reading the optimizer back stands under the free arms.

**Repair.** Replace the parenthesis of check 1 by:

> (The mean loss since the last row is not compared at rows 64, 4,096 and 8,192: before each of these the new runs have a row that E40 does not have, so that their mean covers fewer updates.)

Add to check 2: "Scores with the gate forced off are not taken in E58, and E48's are not compared."

Replace check 3 by:

> 3. **`free-wd0` differs from `free` in the decay only.** For seed 73129 under each rule, after the first update from the same start: for every tensor outside the byte table, the tensor of `free-wd0` minus the tensor of `free` equals 0.00025 times the starting tensor, with a largest absolute deviation below 1e-6 (float32 arithmetic gives about 1e-7; without the decay the deviation would be 6e-6 or more in every tensor); the byte table is bit-equal in the two arms; the optimizer read back shows a weight decay of 0 in every group of `free-wd0`.

Replace check 5 by:

> 5. **The fractions.** For f = 0.05, 0.10 and 0.20 and each of the three seeds, without a network: the assertions (a) to (d) of "The fraction"; the taught pairs are nested (344 within 689 within 1,378); no item of a panel of untaught pairs has a pair among the taught pairs of its operation, and every item of the taught panel has, the taught pairs being computed here by `nested` directly and not through `e40_new_family.split`; the whole stream of 8,192 batches is walked through `e40_new_family.Presented`, as in E40's check 5, and ends with no untaught pair presented and with every taught pair of every operation presented at least once; the batch digest at every row is recorded, and every run except those of `free-again` asserts its own digests against it; at f = 0.10 the digests equal E40's at E40's rows; the numbers of common pairs are recorded.

Replace check 6 by:

> 6. **The earlier families are unchanged in `term` and `term-wd0`.** At every row of every such run, in Part A and in Part B, the digest of the earlier families' answers equals row 0's, and the hash of the lived network's tensors with the four code rows reset to their starting values equals `tensors_sha256` of the life's receipt (the first and third parts of E48's check 3). The runner asserts both at every row. A failure stops that run and is reported before another term run is launched.

Replace check 7 by:

> 7. **Cost.** CPU-seconds an update and a scored row for each arm from the runs of seed 73129, and a forecast for the whole job. If the forecast exceeds 35 CPU-hours, the worker reports it and waits for a reply before Part B.

Add:

> 8. **`term-wd0` differs from `term` in the decay only.** For seed 73129 under each rule, after the first update: the two `down` factors of `term-wd0` are bit-equal to their starting values, and those of `term` equal 0.99975 times theirs with a largest absolute deviation below 1e-7 (at the first update the gradient of a `down` factor is zero, because its `up` factor starts at zero; E48's receipts record it); the two `up` factors and the four code rows are bit-equal in the two arms; the optimizer read back shows a weight decay of 0 in both groups of `term-wd0`, and of 0.25 and 0 in the groups of `term`.

In "Reporting" replace "checks 3, 4 and 5 passed (or which failed)" by "checks 3, 4, 5 and 8 passed (or which failed)".

### 8. What the runner's description leaves open

**What is wrong.** The description is correct in what it says (see "Verified and found sound"). Five things that a worker would meet are not said.

(a) *Which forward scores.* `exp26_candidate.step` calls `arch26.forward` itself, but the scorer answers through `exp11_cell.forward`, which `exp48_gated_run.install()` replaces by the gated forward. A free arm scored after `exp48_gated_run.install()` fails on the first item of the new family if no term is set (`STATE["term"]` is `None`), and is scored with a term added, without any failure, if one was set earlier in the process. A program that makes checks 3 and 8 in one process meets this.

(b) *"E48's fields where the arm has a term."* E48's row 0 carries `check2` (scores equal E40's row 0) and every row carries `check4` (batch digest equals E40's), which are false by construction at another fraction, and `check3.equals_life_end`. `exp48_gated_run.run` records them and asserts none, so nothing breaks; a worker or a later reader may still take a `false` for a failure. Whether the file of generated answers (0.7 MB a run in E48) is written is not said.

(c) *Constants beside the optimizer read back.* `exp26_candidate.opt_record` writes `weight_decay=WD`, the constant 0.25, beside the groups; in `free-wd0` the field would be false. `exp48_gated_run.run` writes its optimizer record as a constant (`weight_decay: 0.25` for the term); copied into `term-wd0` it would be false. The brief's sentence on reading the groups back stands under the free arms only.

(d) *A run started again.* E40's and E48's runners refuse an output path that exists (`R.refuse_existing`). An interrupted run leaves an incomplete receipt.

(e) *The new runner's own hash* is in no receipt: E40's and E48's provenance functions hash their own files only.

**Repair.** Add to "The runner":

> - **One run, one process.** A free arm calls `arch26.install()` and never `exp48_gated_run.install()`; a term arm calls `exp48_gated_run.install()`. Checks that need both kinds use two processes.
> - **E48's fields.** The term arms record E48's row fields `earlier_answers_sha256`, `check3`, `code_rows_tensors_sha256`, `term_norms`, `sweep_cosines` and `term_file`. They do not record E48's `check2` and `check4`, which compare with E40's receipts and are false by construction at another fraction. No file of generated answers is written, and no score is taken with the gate forced off.
> - **The optimizer in the receipt** is read back from the optimizer in every arm: with `exp20_reference.opt_groups` in the free arms, and as the `weight_decay`, `lr` and `betas` of each group with the names of its tensors in the term arms. Neither `exp26_candidate.opt_record`, which writes the constant 0.25 beside the groups, nor the constant record of `exp48_gated_run.run` is used.
> - **Provenance.** Every receipt records the SHA-256 of `exp58_new_family.py`, `exp48_gated_run.py`, `e48_gated.py` and `e40_new_family.py` beside E40's provenance. Expected: `exp48_gated_run.py` f4f03b01ad5dba650f1d8304ad8547baeacf8279126332cc466e2c66d8d131ac, `e48_gated.py` ab324e9ce4f55b01fa8d8f802b4a1622990c73a70d974727a1f19916e2ccb1a0, `e40_new_family.py` aa11c27f1588abbac6659afc124980b88994f6fbae375ffc742f0d31dbdda9cf.

and replace, in "Rules for the worker", the sentence "A run that was interrupted is started again from its beginning; …" by:

> A run that was interrupted is started again from its beginning, after its incomplete receipt and any file it wrote have been moved to `$M/exp45/e58/interrupted/` (the runner refuses an output path that exists); the runs are deterministic and at most 0.4 CPU-hours long.

### 9. Figures and statements about the record

**What is wrong.**

| Place in the brief | The brief | From the receipts |
|---|---|---|
| "The hint", sweep, seed 73129 | 7.7 | 7.62 (92.15 − 84.53); the brief subtracted figures already rounded |
| "The hint", back-propagation, seed 73129 | 7.0 | 6.93 |
| E48, "still rising at its last row by 0.2 to 4.8" | 0.2 | 0.15 (back-propagation, seed 116089) |
| "loses 38 to 43 points … within 256 updates" | 38 to 43 | 38.4 to 43.1 at row 256, on the four panels of the life's families; at row 64 one run (sweep, 93683) has lost 45.9. The six panels lose 23.6 to 28.6 at row 256 |
| Terms: "Each has learned the first family and the families of its life" | | at the end of the life the two panels of the first family stand at 13.5 to 37.3%, the four of the life's families at 86.6 to 96.8% |
| "The score is reached within 64 to 256 updates and does not rise afterwards" | | in two runs the highest later row exceeds the highest of rows 64 to 256: by 4.6 points (sweep, 116089: 83.8, then 88.4 at row 1,024) and by 2.3 (sweep, 93683) |

All other figures of "What the record says", of the cost paragraph and of "How the result is stated" were recomputed and agree (see "Verified and found sound").

**Repair.** In "The hint" write "by 7.6, 8.6 and 1.1 points under the sweep and by 6.9, 4.1 and 7.9 under back-propagation". In the bullet on E48 write "by 0.1 to 4.8 points". Replace the first sentence of "The earlier families" by "In E40 the lived network has lost 38 to 43 points on the four panels of its life's families (add and sub under 97 and 73) at row 256." In "Terms" replace "Each has learned the first family and the families of its life." by "At the end of its life each answers 87 to 97% of the four panels of its life's families and 13 to 37% of the two panels of the first family." In the bullet on E40 replace "and does not rise afterwards" by "and no later row exceeds the highest of rows 64 to 256 by more than 4.6 points".

With the changes of defects 2 to 4 and 6 the numbers of runs and the cost change. Replace the status line's "Three seeds, about 23 CPU-hours." by "The job uses three seeds and about 27 CPU-hours.", "the 66 runs" under "Done" by "the 72 runs", "(ten runs)" by "(twelve runs)", and the cost paragraph by:

> The cost is estimated from the receipts of seed 73129 under the sweep (E40: 974 CPU-seconds for 8,192 updates and 130 for nine rows; E48: 135 for 1,024 updates and 202 for eight rows, two of which are scored twice): about 0.35 CPU-hours for a free run and 0.41 for a term run, each with fifteen rows and the common score; 13.3 CPU-hours for Part A (24 free runs and 12 term runs), 13.3 for Part B (the same numbers), about 0.5 for the checks; about 27 in all, or nine hours with three processes.

### 10. References, and what a worker in a fresh container needs

**What is wrong.**

- `$R/E48_RESULT_VERIFICATION_20261117.md` is cited without a branch. It is on the reviewer's branch only (`git cat-file -e $D:$R/E48_RESULT_VERIFICATION_20261117.md` fails).
- E48's report is at `$W8:$M/exp45/e48/E48_REPORT.md`; the brief gives no path for it, and a reader may look for it in `$M`.
- The brief does not say from which commit the worker's branch starts.
- torch's wheel does not install numpy; `exp26_candidate.Digest` and `exp20_reference.tensors_sha` call `Tensor.numpy()`.

The lived networks and their receipts are on the default branch (six `.pt` and six `.json` under `$M/exp45/e26/`), as are E40's receipts, `e48_gated_spec.py` with its `OUTPUT_E48_SPEC_MAC.txt`, and `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md`. Nothing else is missing.

**Repair.** In "The question" write "(E48's brief says so; `$V:$R/E48_RESULT_VERIFICATION_20261117.md`, section 10)". In "Shell names and host" add after the definition of `W8`: "E48's report is `$W8:$M/exp45/e48/E48_REPORT.md`." In "Rules for the worker" add as the first rule:

> - Create your branch from `$D` (7cabb13a). The lived networks, their receipts and E40's receipts are in it; the two files of E48 and E48's receipts are read from `$W8`, this brief and the verification of E48 from `$V`.

and replace "torch 2.15.0+cpu from `https://download.pytorch.org/whl/cpu` in a virtual environment; versions recorded in every receipt." by "Install torch 2.15.0+cpu from `https://download.pytorch.org/whl/cpu` and numpy in a virtual environment; every receipt records the versions."

### 11. The files to be committed: the sizes are right, the total is not stated

**What is wrong.** A network file is 4,501,600 to 4,501,619 bytes (the lived networks) and 4,501,797 to 4,501,854 (E40's rows): "4.5 MB each" is right. A term file of rank 16 is 150,484 to 150,517 bytes; the brief does not give it. The total is not stated. As revision 1 stands it is 42 network files (36 free runs and 6 of `term-shrink`), 189 MB, 54 term files, 8 MB, and the receipts, about 3 MB: about 200 MB. E40 committed 24 network files (108 MB), so the size has a precedent, and no file approaches a limit of 100 MB for one file; it is acceptable. It should be stated, because the reviewer's container had 2.9 GB free at the time of this check and the worker's branch will be fetched into it.

**Repair.** Replace the bullet "**Files committed:**" by:

> - **Files committed:** the receipts; the final weights of every run of `free`, `free-wd0` and `free-anchor` (42 files of 4.5 MB); the term files at rows 1,024 and 8,192 of every term run (48 files of 0.15 MB). `free-again` commits its receipt only. No weights at other rows. The total is about 200 MB; it is pushed with the receipts as the runs finish, not in one push at the end.

### 12. Language

The brief is to be in standard academic English: complete sentences, every term defined at first use, no metaphor, and no coined label that a reader outside the project could not understand. The following fail. (Lists of figures and the rows of tables are not listed.)

| No. | The brief | Why it fails | Replacement |
|---|---|---|---|
| 1 | "Evidence level E2." | not a sentence; the level is not defined | "The evidence level is E2 in the programme's scale." The reviewer adds the file that defines the scale. |
| 2 | "Three seeds, about 23 CPU-hours." | not a sentence | see defect 9 |
| 3 | "It is given to a check by an agent of the reviewer that has not seen the reviewer's reasoning before it is released" | "before it is released" can be attached to "has not seen" | "Before its release it is checked by an agent of the reviewer that has not seen the reviewer's reasoning" |
| 4 | "*The lived networks* are the six networks at the end of E26's lives" | "life", "family" and "the first family" are used and never defined; "lived" and "life" are metaphors | Put before it: "A *family* is a set of four operations on two numbers (add, sub, max, min) under one modulus; a code byte at the start of an item names the operation and the family. An *item* is the code byte, the first operand in two digits, the operator, 0 to 12 capital letters that carry no information, the second operand and '='; its answer is two digits. In E26 each network was trained on the family of modulus 89 (*the first family*) and then, one after the other, on the family of modulus 97, on compositions of two operations and on the family of modulus 73. The project calls this sequence the network's *life* and a network at its end a *lived network*." |
| 5 | "the share of whole answers right on every untaught pair of add and of sub whose answer needs the reduction by the modulus, without letters" | "whole answers right" and "without letters" are not standard and not defined | "the percentage of answers that are right in every byte, over all untaught pairs of add and of sub whose answer requires the reduction by the modulus (a + b ≥ 83, or a − b < 0), each presented with no letters; the score is the mean of the two percentages" |
| 6 | "*The earlier families' panels* are E40's six panels" | "panel" is not defined | add "A *panel* is a fixed set of test items." |
| 7 | "the byte table" (first in "Terms", free learning) | not defined | "the byte table (the embedding table `book.weight`, one row of eight numbers for each byte)" |
| 8 | "the optimizer is the standing one"; "the standing decay" | "standing" is a project label | "the optimizer used in every experiment since E26 (AdamW, rate 0.001, weight decay 0.25 on every tensor but the byte table), called *the standing optimizer* below" |
| 9 | "the harness reads this from the item" | "harness" is not defined | "the program that runs the experiment reads this from the item; the network does not decide it" |
| 10 | "What the record says (working rule 1)" | the rule is cited and not stated | give the rule's sentence in a footnote, or delete the parenthesis |
| 11 | "The record holds one hint in that direction."; "**The hint.**"; "the hint holds at 8,192 updates" | a coined label for one observation | "The record holds one observation that points in that direction."; "**The lead at 1,024 updates.**"; "the lead persists to 8,192 updates" |
| 12 | "so a narrow term is not what holds the level up" | metaphor, and ambiguous as to which level | "so the lead over free learning is not an effect of the small rank" |
| 13 | "**A network without a life** (E40's `fresh` runs)" | metaphor | "**A network that was not trained before** (E40's `fresh` runs)" |
| 14 | "Seeds in the order 73129, 93683, 116089." | not a sentence | "The seeds are given in the order 73129, 93683, 116089." |
| 15 | "Order: the checks; Part A on seed 73129 (ten runs); …" | not a sentence | see defect 6 |
| 16 | "Cost, from the receipts of seed 73129 (…): about 0.33 CPU-hours …" | not a sentence | see defect 9 |
| 17 | "A new file `exp58_new_family.py` in `$M`; no existing source is edited." | not a sentence | "The runner is a new file, `exp58_new_family.py`, in `$M`; no existing source file is edited." |
| 18 | "3. `term-wd0` against `term`: whether the decay …"; points 4 and 5 likewise | not sentences | see defect 3 |
| 19 | "**Keeping in `term` and `term-wd0`:**" | "keeping" is a project label | see defect 7, check 6 |
| 20 | "a dry run of 64 updates" | project label | removed by defect 7; elsewhere "a trial run" |
| 21 | "At most three processes, one thread each (…)." | not a sentence | "Run at most three processes at a time, each with one thread (…)." |
| 22 | "keep your turn open with a waiting command (a command that waits on the processes, repeated in pieces within the command's time limit), and commit and push finished receipts between the pieces" | "waiting command" and "pieces" are coined | "keep your turn open by running a command that waits for the processes; because a command has a time limit, repeat it until the processes have ended, and commit and push the finished receipts between two such commands" |
| 23 | "Commit trailers as `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` says. No name of any language model in any committed file or commit message." | not sentences | "Write the commit trailers as `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` prescribes. Do not write the name of any language model in a committed file or a commit message." |
| 24 | "The report, `E58_REPORT.md`, in standard academic English: …" | not a sentence | "Write the report, `E58_REPORT.md`, in standard academic English: …" |
| 25 | "The seven checks; the 66 runs; `E58_REPORT.md` with …" (under "Done") | not a sentence | "The job is done when the eight checks, the 72 runs and `E58_REPORT.md` exist; the report holds the table of every run, the readings of Parts A and B in the words fixed above, the outcome of each prediction, and the list of committed files." |

## Verified and found sound

**Figures (A).** Recomputed from the 12 receipts of E40 and the 18 of E48 (appendix, `figs.py`, `more.py`):

- E40's reduction scores at the eight rows, all six runs: every figure of the brief.
- "Up to 19 points between neighbouring rows before row 1,024 and up to 6 from row 1,024 on": 19.1 and 6.0.
- E48, rank 16, rows 512 and 1,024: every figure; the term is below E40 at row 64 in all six runs.
- The lead over E40's highest row: 1.1 to 7.0 (1.14, 3.26, 3.28, 5.84, 6.82, 7.04).
- The ranks at row 1,024: all eighteen figures.
- A network that was not trained before: 6.0 to 11.8 at row 8,192.
- About 3,070 reduction pairs for an operation at f = 0.10: 3,055 to 3,078.
- The standard deviation of 1.0 to 3.3 and the scatter of 0.6 to 2.1: 1.01 to 3.35, and 0.64 to 2.12.
- Cost: 974 and 130 CPU-seconds (E40), 135 and 202 (E48), sweep, seed 73129; with fifteen rows 0.33 CPU-hours for a free run, 0.40 for a term run of 8,192 updates, 0.09 for one of 1,024 updates; 9.4 for Part A, 12.7 for Part B, 22.6 in all. The numbers of runs (30, 36, 66; ten on seed 73129) are right.
- The factors 0.774 and 0.129: 0.7741 (also by 1,024 multiplications in float32) and 0.1290.

**The runner (B).**

1. `e40_new_family.FRAC` is read in `split` only, at the time of the call (`e40_new_family.py`, lines 40 and 56 to 57). `entries` and `batches` (60 to 67), `held_plain` (74 to 79), `Scorer.__init__` (84 to 86) and `Presented.__init__` (179 to 180) all go through `split`. Nothing else in either file is tied to a tenth except text (the `data` strings of the two `run` functions, which the new runner does not call) and, in `exp48_gated_run.run`, the comparisons with E40's receipts, which are recorded and not asserted. No object that depends on the fraction is built at import, and no default argument holds it. The only trap is that `batches` is a generator and reads the taught pairs at its first `next`.
2. `exp11_cell.split` gives 344, 689 and 1,378 taught pairs at 0.05, 0.10 and 0.20 on all three seeds; `round(f * 100)` is 5, 10 and 20; the twelve shuffle seeds of a network seed are distinct.
3. Nothing in the scoring or in the update depends on when rows are scored. The stream has its own generator (`Random(seed + 140 + 77)`), the panels are built once with their own generators, `respond` runs without gradient and without random numbers, the batch digest and the record of presented items are cumulative, the optimizer is not touched by scoring, and torch's global generator is used only when the network is created. `free` with the new rows reproduces E40's `tensors_sha256` at E40's rows, and `term` continued beyond 1,024 updates reproduces E48's rows, provided the host computes as E40's and E48's did; the only row-dependent field is the mean loss since the last row (defect 7). A precedent for bit reproduction across hosts exists: E40's check 3 reproduced E26's trial births of 2,048 updates bit for bit on another virtual machine (kernel build v470 against v378, torch 2.15.0+cpu on both).
4. `X16.adamw16(m, C.LR, 0.0, ())` gives two groups with weight decay 0 (`exp16_integrated.py`, lines 176 to 187, with `E16_BOOKDECAY=off` set by `exp20_reference`). Check 3 is exact and its tolerance is safe (defect 7).
5. The in-place product of `term-shrink` is, for a tensor without a gradient step, what AdamW's decay does (the same float32 factor, 0.999750018), and it interacts neither with `keep_code_rows`, nor with the frozen state, nor with any assertion of `exp48_gated_run` (the hashes of the lived network are recorded in `check3` and asserted nowhere). The arm fails in its reading, not in its mechanics (defect 2).
6. Only `exp48_gated_run.py` and `e48_gated.py` are missing on the default branch. Every other file that they and `e40_new_family.py` import or hash has the same blob on both branches (`arch26.py`, `exp26_candidate.py`, `arch20.py`, `exp20_reference.py`, `exp16_integrated.py`, `arch17.py`, `exp11_cell.py`, `exp45.py`, `exp3_state.py`, `mod97_local.py`, `e40_new_family.py`, `e36_reminder.py`, `arch31_workerB.py`, `exp31_switch_workerB.py`), and `e48_gated.py` on `W8` has the SHA-256 of `e48_gated_spec.py` on the default branch. `exp48_gated_run.py` records hashes and asserts none of them.
7. The sizes are right (defect 11).

**The receipts and the weights agree.** The numpy forward gives, for all six lived networks, exactly the receipts' counts on the six earlier panels and on the reduction pairs at E40's row 0, and exactly E48's counts on the reduction pairs at row 1,024 when E48's saved term of rank 16 is added. This confirms independently that E48's term files are the terms that its receipts score. At every row of the six receipts of rank 16, E48 records the earlier families' digest as equal to row 0's and the lived tensors as equal to the life's.

**The design (C), where it is sound.** The comparison of `term` with `free` at 8,192 updates is like for like in data, rows and scorer, and checks 1 and 2 tie both to the record. `term-wd0` is the right control for the decay of the factors. R-late over rows 4,096 to 8,192 is a reasonable figure (defect 4 (d) adds what it hides). The lead of E48 over E40 passes the brief's rule on R-mid under both rules (differences of 7.8, 6.4, 3.7 and 8.6, 6.1, 7.9), so the rule is not in conflict with the record.

**The predictions (E).** P58.1 to P58.5, P58.8 and P58.9 are decidable from the runs as specified without a further decision. None contradicts another. P58.6, P58.7 and P58.10 are treated in defect 5.

**The checks (D), paths.** The files that checks 1 and 2 cite exist: `$D:$M/exp45/e40/e40_lived_{S,B}_73129.json`, `$W8:$M/exp45/e48/e48_{S,B}_73129_r16.json` and `$W8:$M/exp45/e48/e48_{S,B}_73129_r16_row1024.pt`. E48's receipts hold every field that check 2 compares (`data_digest`, `earlier_answers_sha256`, the scores, and the term file's `tensors_sha256`).

**The worker's rules (F).** The stop of the machine five minutes after a turn ends, the installation of torch, the limit of three processes, the reading of other branches and the session to report to are all stated. Defects 8 and 10 add what is missing.

## Not checked

- Nothing was run with torch. What is said of AdamW (the order of decay and step; the conversion of the factor to float32) rests on the algorithm as torch documents it and on a numpy simulation, not on execution. Whether the worker's host reproduces E40's and E48's bits is known from one precedent only.
- `free-anchor`, `free-again`, the nested split and the common score are proposals of this check. Their arithmetic was verified (the nested split and the counts of common pairs; the algebra of the decay toward the start). No run was made, and the effect of the weight decay on generalisation in this task is argued, not measured.
- The statement "No run exists at any other taught fraction of the new family, without decay in free learning, or of the gated term beyond 1,024 updates" was checked against the folders of E40 and E48 only.
- The cost at other fractions and the cost of the common score are estimates (a few seconds a row).
- The chances given to the predictions.
- The thresholds proposed in the repairs (the shares 0.25 and 0.75 of defect 3, the 2 points of defect 1, 95 for the ceiling, 35 CPU-hours for the forecast) are this check's choices. The reviewer may set others, before the release and not after the runs.
- E38, in which a pull toward earlier weights was tested, was not read; the plan's sentence on it was.
- The briefs of E40 and E48 were read in part (E40: to its checks; E48: its passage on the decay and its headings). E48's verification was read in sections 6 and 10. E48's report was not read.
- Whether the worker's environment may reach `download.pytorch.org`.
- The receipts of E40's runs with learned feedback (rule K) and of E40's and E48's trial runs.

## Appendix: scripts and their outputs

**Inputs.** They were placed in a scratch folder `B`. `B/rec/` holds E40's twelve receipts (`git show $D:$M/exp45/e40/e40_{lived,fresh}_{S,B}_<seed>.json`) and E48's eighteen (`git show $W8:$M/exp45/e48/e48_{S,B}_<seed>_r{4,16,64}.json`), under the same file names. `B/w/` holds, for each rule and seed, `life_<rule>_<seed>.pt` (`git cat-file blob $D:$M/exp45/e26/e26_life_<rule>_<seed>.pt`), `e40_<rule>_<seed>_row64.pt` and `e40_<rule>_<seed>_row8192.pt` (`$D:$M/exp45/e40/e40_lived_<rule>_<seed>_row{64,8192}.pt`) and `e48_<rule>_<seed>_r16_row1024.pt` (`$W8:$M/exp45/e48/e48_<rule>_<seed>_r16_row1024.pt`). Every script was run as `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -I <script> B` (`figs.py` takes `B/rec/`). The scripts `ptread.py` and `npnet.py` are modules; the others import them from `B`.

### `ptread.py`

```python
"""Read a torch.save file (zip archive) without torch: standard library and numpy only."""
import io, pickle, zipfile
from collections import OrderedDict
import numpy as np

DT = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_}

class _Storage:
    def __init__(self, name): self.name = name

def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False, hooks=None, metadata=None):
    arr = storage
    n = 1
    for s in size: n *= s
    if len(size) == 0:
        return arr[offset:offset + 1].reshape(())
    # contiguous tensors only
    exp = []
    acc = 1
    for s in reversed(size):
        exp.append(acc); acc *= s
    assert tuple(reversed(exp)) == tuple(stride), (size, stride)
    return arr[offset:offset + n].reshape(size).copy()

def _rebuild_parameter(data, requires_grad, hooks):
    return data

class U(pickle.Unpickler):
    def __init__(self, f, zf, prefix):
        super().__init__(f); self.zf, self.prefix = zf, prefix
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict": return OrderedDict
        if module == "torch._utils" and name == "_rebuild_tensor_v2": return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_parameter": return _rebuild_parameter
        if module == "torch" and name.endswith("Storage"): return _Storage(name)
        raise pickle.UnpicklingError(f"refused {module}.{name}")
    def persistent_load(self, pid):
        kind, st, key, loc, numel = pid
        assert kind == "storage"
        raw = self.zf.read(f"{self.prefix}/data/{key}")
        return np.frombuffer(raw, dtype=DT[st.name])

def load(path):
    zf = zipfile.ZipFile(path)
    pk = [n for n in zf.namelist() if n.endswith("/data.pkl")][0]
    prefix = pk[:-len("/data.pkl")]
    return U(io.BytesIO(zf.read(pk)), zf, prefix).load()
```

### `npnet.py`

```python
"""The lived network's forward in numpy (arch26.forward, with e48_gated.forward's added terms), and the harness's panels.
The task lines (split, render, panel) are copied from exp11_cell.py; the forward follows arch26.py and e48_gated.py line by line.
Standard library and numpy only. Nothing is trained."""
from random import Random
import numpy as np

EOS, BOS = 256, 257
SYM = {"add": "+", "sub": "-", "max": ">", "min": "<"}
DISTRACT = list(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DEV = [("add", 89, b"p"), ("sub", 89, b"q"), ("max", 89, b"v"), ("min", 89, b"w")]
LIFE = [("add", 97, b"r"), ("sub", 97, b"t"), ("max", 97, b"x"), ("min", 97, b"n")]
F3 = [("add", 73, b"s"), ("sub", 73, b"u"), ("max", 73, b"y"), ("min", 73, b"z")]
F4 = [("add", 83, b"e"), ("sub", 83, b"f"), ("max", 83, b"g"), ("min", 83, b"h")]
CAP = 6


def apply(op, m, a, b):
    return {"add": (a + b) % m, "sub": (a - b) % m, "max": max(a, b), "min": min(a, b)}[op]


def render(task, a, b, k, rng):
    op, m, code = task
    d = [rng.choice(DISTRACT) for _ in range(k)]
    prompt = list(code) + list(f"{a:02d}".encode()) + [ord(SYM[op])] + d + list(f"{b:02d}".encode()) + [ord("=")]
    return prompt, list(f"{apply(op, m, a, b):02d}".encode()) + [EOS]


def split(task, frac, seed):
    op, m, code = task
    pairs = [(a, b) for a in range(m) for b in range(m)]
    rng = Random(seed * 1000 + round(frac * 100) + m * 7 + code[0])
    rng.shuffle(pairs)
    k = round(frac * len(pairs))
    return pairs[:k], pairs[k:]


def panel(task, pairs, seed, ks=(0, 8, 12, 20), n=400):
    rng = Random(seed + 3)
    sub = rng.sample(pairs, n)
    return {k: [render(task, a, b, k, rng) for a, b in sub] for k in ks}


def earlier_panels(seed):
    """exp16_integrated.panels with E16_ARCH=bank2: k = 0, 8, 12, 20, 40 for the life's families; k = 12, n = 300 for the first family."""
    P = {f"{t[0]}{t[1]}": panel(t, split(t, .25, seed)[1], seed, ks=(0, 8, 12, 20, 40))[12] for t in LIFE + F3}
    P.update({f"{t[0]}{t[1]}": panel(t, split(t, .5, seed)[1], seed, ks=(12,), n=300)[12] for t in DEV})
    return P


def needs_reduction(op, a, b):
    return a + b >= 83 if op == "add" else a - b < 0


def reduction_panel(t, frac, seed):
    return [render(t, a, b, 0, Random(0)) for a, b in split(t, frac, seed)[1] if needs_reduction(t[0], a, b)]


sig = lambda v: 1.0 / (1.0 + np.exp(-v))


class Net:
    def __init__(self, sd, term=None, dtype=np.float32):
        self.w = {k: np.asarray(v, dtype=dtype) for k, v in sd.items()}
        self.term = None if term is None else {k: np.asarray(v, dtype=dtype) for k, v in term.items()}
        self.dtype = dtype

    def scaled(self, s):
        """Every tensor outside the byte table multiplied by s."""
        n = Net({k: (v if k == "book.weight" else v * self.dtype(s)) for k, v in self.w.items()}, self.term, self.dtype)
        return n

    def embed(self, x, gate):
        w = self.w
        z1 = np.tanh(w["book.weight"][x] @ w["l1.weight"].T + w["l1.bias"])
        gi = z1 @ w["rnn.weight_ih_l0"].T + w["rnn.bias_ih_l0"]
        if gate:
            gi = gi + (z1 @ self.term["down_i"].T) @ self.term["up_i"].T
        return z1, gi

    def step(self, h, z1, gi, gate):
        w = self.w
        gh = h @ w["rnn.weight_hh_l0"].T + w["rnn.bias_hh_l0"]
        if gate:
            gh = gh + (h @ self.term["down_h"].T) @ self.term["up_h"].T
        n = h.shape[1]
        r = sig(gi[:, :n] + gh[:, :n]); z = sig(gi[:, n:2 * n] + gh[:, n:2 * n]); nn_ = np.tanh(gi[:, 2 * n:] + r * gh[:, 2 * n:])
        h = (1 - z) * nn_ + z * h
        z2 = np.tanh(z1 @ w["l2.weight"].T + w["l2.bias"] + h @ w["lm.weight"].T + w["lm.bias"])
        return h, z2 @ w["l3.weight"].T + w["l3.bias"]

    def respond(self, prompts, gate):
        out = [None] * len(prompts)
        groups = {}
        for i, p in enumerate(prompts):
            groups.setdefault(len(p), []).append(i)
        for L, idx in groups.items():
            for j in range(0, len(idx), 1024):
                ids = idx[j:j + 1024]
                x = np.array([[BOS, *prompts[i]] for i in ids])
                z1, gi = self.embed(x, gate)
                h = np.zeros((len(ids), 256), dtype=self.dtype)
                for t in range(x.shape[1]):
                    h, lg = self.step(h, z1[:, t], gi[:, t], gate)
                gen = [[] for _ in ids]; done = [False] * len(ids)
                for _ in range(CAP):
                    yy = lg.argmax(1)
                    for q in range(len(ids)):
                        if not done[q]:
                            gen[q].append(int(yy[q]))
                            if int(yy[q]) == EOS:
                                done[q] = True
                    if all(done):
                        break
                    z1n, gin = self.embed(yy[:, None], gate)
                    h, lg = self.step(h, z1n[:, 0], gin[:, 0], gate)
                for q, i in enumerate(ids):
                    out[i] = gen[q]
        return out

    def score(self, items, gate=False):
        outs = self.respond([p for p, _ in items], gate)
        return sum(o == t for o, (_, t) in zip(outs, items)), len(items)
```

### `validate.py`

```python
import sys, json, time
B = sys.argv[1]; sys.path.insert(0, B)
import ptread, npnet as N
SEEDS = (73129, 93683, 116089); E6 = ("add89", "sub89", "add97", "sub97", "add73", "sub73")
t0 = time.time(); bad = 0
for seed in SEEDS:
    P = N.earlier_panels(seed)
    red = {f"{t[0]}83": N.reduction_panel(t, .10, seed) for t in N.F4[:2]}
    for rule in "SB":
        e40 = json.load(open(f"{B}/rec/e40_lived_{rule}_{seed}.json")); e48 = json.load(open(f"{B}/rec/e48_{rule}_{seed}_r16.json"))
        want0 = e40["rows"][0]["scores"]
        net = N.Net(ptread.load(f"{B}/w/life_{rule}_{seed}.pt"))
        got = {n: net.score(P[n])[0] for n in E6}; w = {n: want0["earlier_k12"][n]["exact"] for n in E6}
        gr = {n: net.score(red[n], False) for n in red}; wr = {n: (want0["held_plain_k0"][n]["reduction"]["exact"], want0["held_plain_k0"][n]["reduction"]["n"]) for n in red}
        # the E48 term at row 1024 on the lived network (code rows from the term file)
        term = ptread.load(f"{B}/w/e48_{rule}_{seed}_r16_row1024.pt")
        sd = dict(net.w); bk = sd["book.weight"].copy(); bk[[101, 102, 103, 104]] = term["code_rows"]; sd["book.weight"] = bk
        tn = N.Net(sd, term)
        gt = {n: tn.score(red[n], True) for n in red}; row = e48["rows"][-1]["scores"]["held_plain_k0"]; wt = {n: (row[n]["reduction"]["exact"], row[n]["reduction"]["n"]) for n in red}
        ok = got == w and gr == wr and gt == wt; bad += not ok
        print(seed, rule, "earlier row 0:", got, "receipt:", w, "| reduction row 0:", gr, "receipt:", wr, "| E48 term row 1024:", gt, "receipt:", wt, "EQUAL" if ok else "DIFFER", f"{time.time()-t0:.0f}s", flush=True)
print("runs that differ from the receipts:", bad)
```

Output of `validate.py`:

```text
73129 S earlier row 0: {'add89': 65, 'sub89': 133, 'add97': 380, 'sub97': 362, 'add73': 341, 'sub73': 302} receipt: {'add89': 65, 'sub89': 133, 'add97': 380, 'sub97': 362, 'add73': 341, 'sub73': 302} | reduction row 0: {'add83': (374, 3069), 'sub83': (0, 3074)} receipt: {'add83': (374, 3069), 'sub83': (0, 3074)} | E48 term row 1024: {'add83': (2835, 3069), 'sub83': (2826, 3074)} receipt: {'add83': (2835, 3069), 'sub83': (2826, 3074)} EQUAL 13s
73129 B earlier row 0: {'add89': 74, 'sub89': 150, 'add97': 384, 'sub97': 379, 'add73': 333, 'sub73': 352} receipt: {'add89': 74, 'sub89': 150, 'add97': 384, 'sub97': 379, 'add73': 333, 'sub73': 352} | reduction row 0: {'add83': (0, 3069), 'sub83': (269, 3074)} receipt: {'add83': (0, 3069), 'sub83': (269, 3074)} | E48 term row 1024: {'add83': (2802, 3069), 'sub83': (2915, 3074)} receipt: {'add83': (2802, 3069), 'sub83': (2915, 3074)} EQUAL 24s
93683 S earlier row 0: {'add89': 78, 'sub89': 28, 'add97': 354, 'sub97': 374, 'add73': 328, 'sub73': 356} receipt: {'add89': 78, 'sub89': 28, 'add97': 354, 'sub97': 374, 'add73': 328, 'sub73': 356} | reduction row 0: {'add83': (121, 3078), 'sub83': (13, 3061)} receipt: {'add83': (121, 3078), 'sub83': (13, 3061)} | E48 term row 1024: {'add83': (2783, 3078), 'sub83': (2897, 3061)} receipt: {'add83': (2783, 3078), 'sub83': (2897, 3061)} EQUAL 38s
93683 B earlier row 0: {'add89': 61, 'sub89': 99, 'add97': 397, 'sub97': 391, 'add73': 378, 'sub73': 383} receipt: {'add89': 61, 'sub89': 99, 'add97': 397, 'sub97': 391, 'add73': 378, 'sub73': 383} | reduction row 0: {'add83': (26, 3078), 'sub83': (3, 3061)} receipt: {'add83': (26, 3078), 'sub83': (3, 3061)} | E48 term row 1024: {'add83': (3051, 3078), 'sub83': (2960, 3061)} receipt: {'add83': (3051, 3078), 'sub83': (2960, 3061)} EQUAL 55s
116089 S earlier row 0: {'add89': 46, 'sub89': 35, 'add97': 384, 'sub97': 366, 'add73': 365, 'sub73': 370} receipt: {'add89': 46, 'sub89': 35, 'add97': 384, 'sub97': 366, 'add73': 365, 'sub73': 370} | reduction row 0: {'add83': (1, 3070), 'sub83': (8, 3055)} receipt: {'add83': (1, 3070), 'sub83': (8, 3055)} | E48 term row 1024: {'add83': (2793, 3070), 'sub83': (2689, 3055)} receipt: {'add83': (2793, 3070), 'sub83': (2689, 3055)} EQUAL 69s
116089 B earlier row 0: {'add89': 83, 'sub89': 135, 'add97': 382, 'sub97': 349, 'add73': 366, 'sub73': 356} receipt: {'add89': 83, 'sub89': 135, 'add97': 382, 'sub97': 349, 'add73': 366, 'sub73': 356} | reduction row 0: {'add83': (3, 3070), 'sub83': (707, 3055)} receipt: {'add83': (3, 3070), 'sub83': (707, 3055)} | E48 term row 1024: {'add83': (3029, 3070), 'sub83': (3016, 3055)} receipt: {'add83': (3029, 3070), 'sub83': (3016, 3055)} EQUAL 82s
runs that differ from the receipts: 0
```

### `shrink.py`

```python
"""What `term-shrink` does to the lived network, computed without any training: every tensor outside the byte table multiplied r times by
float32(0.99975). (a) The six earlier panels, whose items never open the gate, so that the term plays no part: these are the figures the arm
itself would give at row r. (b) The reduction score of E48's term of rank 16 as saved at row 1,024 (trained on the unshrunk network), on the
shrunk network: not what the arm gives (there the term is trained while the network shrinks), but how far the shrinking alone moves the
function on the new family's items."""
import sys, json, time
import numpy as np
B = sys.argv[1]; sys.path.insert(0, B)
import ptread, npnet as N
SEEDS = (73129, 93683, 116089); E6 = ("add89", "sub89", "add97", "sub97", "add73", "sub73"); LIFE4 = E6[2:]
ROWS = (0, 16, 32, 64, 128, 256, 512, 1024)
f = np.float32(0.99975)
res = {}
for seed in SEEDS:
    P = N.earlier_panels(seed)
    red = {f"{t[0]}83": N.reduction_panel(t, .10, seed) for t in N.F4[:2]}
    for rule in "SB":
        sd = ptread.load(f"{B}/w/life_{rule}_{seed}.pt"); term = ptread.load(f"{B}/w/e48_{rule}_{seed}_r16_row1024.pt")
        e40 = {r["updates"]: r for r in json.load(open(f"{B}/rec/e40_lived_{rule}_{seed}.json"))["rows"]}
        cur = {k: v.copy() for k, v in sd.items()}; u = 0; line = []
        for r in ROWS:
            while u < r:
                for k in cur:
                    if k != "book.weight": cur[k] = cur[k] * f          # float32 times float32, rounded to nearest, as the in-place product
                u += 1
            net = N.Net(cur)
            c = {n: net.score(P[n])[0] / len(P[n]) * 100 for n in E6}
            life4 = sum(c[n] for n in LIFE4) / 4; six = sum(c[n] for n in E6) / 6
            sdt = dict(cur); bk = sdt["book.weight"].copy(); bk[[101, 102, 103, 104]] = term["code_rows"]; sdt["book.weight"] = bk
            tn = N.Net(sdt, term); rs = sum(100.0 * tn.score(red[n], True)[0] / len(red[n]) for n in red) / 2
            pct = lambda d: 100.0 * d["exact"] / d["n"]
            fr = (sum(pct(e40[r]["scores"]["earlier_k12"][n]) for n in LIFE4) / 4) if r in e40 else None
            res[(rule, seed, r)] = (life4, six, rs, fr)
            line.append(f"{r}: life4 {life4:.1f} six {six:.1f} E48term {rs:.1f}" + (f" [free {fr:.1f}]" if fr is not None else ""))
        ratio = float(np.linalg.norm(cur['l2.weight'].astype(np.float64)) / np.linalg.norm(sd['l2.weight'].astype(np.float64)))
        print(rule, seed, f"(norm ratio at 1024: {ratio:.4f})", " | ".join(line), flush=True)
print("summary over the six networks: range of the mean of the four life-family panels (life4), of the six panels, and of E48's term's reduction score")
for r in ROWS:
    v = [res[(rule, s, r)] for rule in "SB" for s in SEEDS]
    fr = [x[3] for x in v if x[3] is not None]
    print(r, f"factor {0.99975**r:.3f}", f"life4 {min(x[0] for x in v):.1f}..{max(x[0] for x in v):.1f}", f"six {min(x[1] for x in v):.1f}..{max(x[1] for x in v):.1f}",
          f"E48 term on shrunk net {min(x[2] for x in v):.1f}..{max(x[2] for x in v):.1f}", (f"free(E40) life4 {min(fr):.1f}..{max(fr):.1f}" if fr else ""))
    v0 = [res[(rule, s, 0)] for rule in "SB" for s in SEEDS]
    print("    loss against row 0: life4", " ".join(f"{a[0]-b[0]:.1f}" for a, b in zip(v0, v)), "| E48 term", " ".join(f"{a[2]-b[2]:.1f}" for a, b in zip(v0, v)))
```

Output of `shrink.py`:

```text
S 73129 (norm ratio at 1024: 0.7741) 0: life4 86.6 six 68.7 E48term 92.2 [free 86.6] | 16: life4 86.2 six 68.3 E48term 92.1 | 32: life4 85.9 six 67.9 E48term 92.1 | 64: life4 85.7 six 67.6 E48term 91.9 [free 69.4] | 128: life4 84.4 six 66.0 E48term 90.8 [free 66.1] | 256: life4 76.9 six 59.8 E48term 86.6 [free 45.2] | 512: life4 51.2 six 39.5 E48term 64.1 [free 46.3] | 1024: life4 7.3 six 6.5 E48term 13.0 [free 45.7]
B 73129 (norm ratio at 1024: 0.7741) 0: life4 90.5 six 72.8 E48term 93.1 [free 90.5] | 16: life4 90.4 six 72.7 E48term 93.1 | 32: life4 90.0 six 72.4 E48term 93.1 | 64: life4 89.4 six 72.0 E48term 93.0 [free 61.4] | 128: life4 88.3 six 70.8 E48term 92.6 [free 60.6] | 256: life4 86.1 six 68.3 E48term 90.5 [free 48.9] | 512: life4 69.8 six 55.3 E48term 76.8 [free 44.4] | 1024: life4 20.4 six 17.6 E48term 24.9 [free 47.1]
S 93683 (norm ratio at 1024: 0.7741) 0: life4 88.2 six 64.7 E48term 92.5 [free 88.2] | 16: life4 87.8 six 64.5 E48term 92.3 | 32: life4 87.3 six 64.1 E48term 92.2 | 64: life4 86.0 six 63.3 E48term 91.6 [free 42.4] | 128: life4 83.0 six 60.9 E48term 89.5 [free 42.4] | 256: life4 74.4 six 54.7 E48term 82.6 [free 49.0] | 512: life4 45.5 six 34.5 E48term 58.5 [free 43.8] | 1024: life4 5.2 six 4.8 E48term 11.9 [free 38.0]
B 93683 (norm ratio at 1024: 0.7741) 0: life4 96.8 six 73.4 E48term 97.9 [free 96.8] | 16: life4 96.8 six 73.3 E48term 97.9 | 32: life4 96.8 six 73.2 E48term 98.0 | 64: life4 96.7 six 73.1 E48term 98.0 [free 68.9] | 128: life4 96.3 six 72.4 E48term 97.8 [free 68.3] | 256: life4 95.6 six 71.7 E48term 97.1 [free 58.4] | 512: life4 89.2 six 66.5 E48term 90.2 [free 51.8] | 1024: life4 31.5 six 24.4 E48term 51.6 [free 49.2]
S 116089 (norm ratio at 1024: 0.7741) 0: life4 92.8 six 66.4 E48term 89.5 [free 92.8] | 16: life4 92.7 six 66.3 E48term 89.6 | 32: life4 92.3 six 66.0 E48term 89.6 | 64: life4 92.0 six 65.8 E48term 89.4 [free 55.9] | 128: life4 90.9 six 65.0 E48term 88.4 [free 57.1] | 256: life4 85.4 six 61.0 E48term 83.9 [free 49.7] | 512: life4 60.0 six 42.9 E48term 65.3 [free 49.3] | 1024: life4 6.8 six 5.2 E48term 18.6 [free 43.5]
B 116089 (norm ratio at 1024: 0.7741) 0: life4 90.8 six 72.7 E48term 98.7 [free 90.8] | 16: life4 90.8 six 72.7 E48term 98.7 | 32: life4 90.7 six 72.5 E48term 98.7 | 64: life4 90.8 six 72.4 E48term 98.6 [free 57.7] | 128: life4 90.1 six 71.7 E48term 98.5 [free 57.7] | 256: life4 89.1 six 70.4 E48term 97.4 [free 51.3] | 512: life4 81.6 six 64.3 E48term 91.0 [free 48.6] | 1024: life4 26.7 six 22.2 E48term 44.8 [free 47.8]
summary over the six networks: range of the mean of the four life-family panels (life4), of the six panels, and of E48's term's reduction score
0 factor 1.000 life4 86.6..96.8 six 64.7..73.4 E48 term on shrunk net 89.5..98.7 free(E40) life4 86.6..96.8
    loss against row 0: life4 0.0 0.0 0.0 0.0 0.0 0.0 | E48 term 0.0 0.0 0.0 0.0 0.0 0.0
16 factor 0.996 life4 86.2..96.8 six 64.5..73.3 E48 term on shrunk net 89.6..98.7 
    loss against row 0: life4 0.4 0.4 0.1 0.1 0.1 0.0 | E48 term 0.1 0.2 -0.1 -0.0 -0.0 -0.0
32 factor 0.992 life4 85.9..96.8 six 64.1..73.2 E48 term on shrunk net 89.6..98.7 
    loss against row 0: life4 0.7 0.9 0.5 0.5 0.1 0.1 | E48 term 0.0 0.3 -0.1 -0.0 -0.0 0.0
64 factor 0.984 life4 85.7..96.7 six 63.3..73.1 E48 term on shrunk net 89.4..98.6 free(E40) life4 42.4..69.4
    loss against row 0: life4 0.9 2.2 0.8 1.1 0.1 0.1 | E48 term 0.3 0.9 0.1 0.1 -0.0 0.1
128 factor 0.969 life4 83.0..96.3 six 60.9..72.4 E48 term on shrunk net 88.4..98.5 free(E40) life4 42.4..68.3
    loss against row 0: life4 2.2 5.2 1.9 2.2 0.5 0.8 | E48 term 1.4 3.0 1.1 0.5 0.1 0.2
256 factor 0.938 life4 74.4..95.6 six 54.7..71.7 E48 term on shrunk net 82.6..97.4 free(E40) life4 45.2..58.4
    loss against row 0: life4 9.7 13.9 7.4 4.4 1.2 1.8 | E48 term 5.5 10.0 5.6 2.6 0.8 1.3
512 factor 0.880 life4 45.5..89.2 six 34.5..66.5 E48 term on shrunk net 58.5..91.0 free(E40) life4 43.8..51.8
    loss against row 0: life4 35.4 42.8 32.8 20.7 7.6 9.2 | E48 term 28.0 34.0 24.2 16.3 7.7 7.6
1024 factor 0.774 life4 5.2..31.5 six 4.8..24.4 E48 term on shrunk net 11.9..51.6 free(E40) life4 38.0..49.2
    loss against row 0: life4 79.2 83.1 86.0 70.1 65.3 64.1 | E48 term 79.1 80.7 70.9 68.2 46.4 53.8
```

### `norms.py`

```python
import sys; sys.path.insert(0, sys.argv[1])
import ptread, numpy as np
W = sys.argv[1] + "/w/"
SEEDS = (73129, 93683, 116089)
names = None
print("ratio of each tensor's norm in E40's free run to its norm at the start; pure decay would give 0.9841 at row 64 and 0.1290 at row 8192")
print("then: relative size of the change, |w_row - w_start| / |w_start|; and the cosine of w_row with w_start")
for rule in "SB":
    for s in SEEDS:
        a = ptread.load(W + f"life_{rule}_{s}.pt"); b = ptread.load(W + f"e40_{rule}_{s}_row64.pt"); c = ptread.load(W + f"e40_{rule}_{s}_row8192.pt")
        out = []
        tot = lambda d: np.sqrt(sum(float((v.astype(np.float64) ** 2).sum()) for k, v in d.items() if k != "book.weight"))
        for k in a:
            if k == "book.weight": continue
            n0 = np.linalg.norm(a[k].astype(np.float64))
            r64 = np.linalg.norm(b[k].astype(np.float64)) / n0; r8 = np.linalg.norm(c[k].astype(np.float64)) / n0
            d8 = np.linalg.norm((c[k] - a[k]).astype(np.float64)) / n0
            cos8 = float((c[k].astype(np.float64) * a[k]).sum() / (np.linalg.norm(c[k].astype(np.float64)) * n0))
            out.append(f"{k.replace('rnn.','').replace('_l0','')}: {r64:.3f}/{r8:.3f} d{d8:.2f} c{cos8:.2f}")
        print(rule, s, f"all non-table: {tot(b)/tot(a):.4f} at 64, {tot(c)/tot(a):.4f} at 8192;", "max|w| start", f"{max(np.abs(v).max() for k, v in a.items() if k != 'book.weight'):.3f}")
        print("    " + "; ".join(out))
print(0.99975 ** 64, 0.99975 ** 256, 0.99975 ** 512, 0.99975 ** 1024, 0.99975 ** 8192)
f = np.float32(0.99975); print("float32(0.99975) =", repr(float(f)), " product of 1024 in float32:", float(np.prod(np.full(1024, f, dtype=np.float64))))
x = np.float32(0.5)
for _ in range(1024): x = np.float32(x * f)
print("a float32 weight of 0.5 after 1,024 multiplications:", float(x) / 0.5)
```

Output of `norms.py`:

```text
ratio of each tensor's norm in E40's free run to its norm at the start; pure decay would give 0.9841 at row 64 and 0.1290 at row 8192
then: relative size of the change, |w_row - w_start| / |w_start|; and the cosine of w_row with w_start
S 73129 all non-table: 0.9997 at 64, 0.9501 at 8192; max|w| start 0.775
    l1.weight: 0.992/0.916 d0.31 c0.95; l1.bias: 1.147/0.977 d0.78 c0.69; weight_ih: 0.997/0.955 d0.69 c0.75; weight_hh: 0.995/0.875 d0.75 c0.69; bias_ih: 0.996/0.887 d0.47 c0.88; bias_hh: 0.994/0.891 d0.46 c0.89; lm.weight: 0.991/0.906 d0.59 c0.81; lm.bias: 1.017/1.143 d0.55 c0.87; l2.weight: 1.030/1.076 d1.25 c0.28; l2.bias: 1.017/1.143 d0.55 c0.87; l3.weight: 1.016/1.042 d0.48 c0.89; l3.bias: 1.005/1.120 d0.16 c0.99
S 93683 all non-table: 0.9982 at 64, 0.8891 at 8192; max|w| start 1.000
    l1.weight: 0.986/0.938 d0.32 c0.95; l1.bias: 1.057/0.541 d0.68 c0.77; weight_ih: 0.994/0.979 d0.70 c0.75; weight_hh: 0.994/0.865 d0.79 c0.65; bias_ih: 0.993/0.834 d0.53 c0.85; bias_hh: 0.990/0.843 d0.54 c0.84; lm.weight: 0.988/0.877 d0.65 c0.77; lm.bias: 1.019/1.017 d0.62 c0.81; l2.weight: 1.012/0.988 d1.14 c0.34; l2.bias: 1.019/1.017 d0.62 c0.81; l3.weight: 1.010/0.815 d0.84 c0.59; l3.bias: 1.001/0.805 d0.21 c1.00
S 116089 all non-table: 0.9987 at 64, 0.8361 at 8192; max|w| start 1.066
    l1.weight: 0.991/0.914 d0.34 c0.94; l1.bias: 0.940/1.024 d0.66 c0.79; weight_ih: 0.996/0.948 d0.68 c0.75; weight_hh: 0.995/0.866 d0.79 c0.65; bias_ih: 0.993/0.841 d0.53 c0.85; bias_hh: 0.993/0.844 d0.54 c0.84; lm.weight: 0.987/0.872 d0.63 c0.78; lm.bias: 0.968/1.033 d1.17 c0.34; l2.weight: 1.002/1.044 d1.15 c0.37; l2.bias: 0.968/1.033 d1.17 c0.34; l3.weight: 1.008/0.707 d0.73 c0.68; l3.bias: 0.998/0.613 d0.39 c1.00
B 73129 all non-table: 1.0006 at 64, 0.9701 at 8192; max|w| start 1.630
    l1.weight: 0.994/1.151 d0.32 c0.97; l1.bias: 1.013/0.982 d0.34 c0.94; weight_ih: 0.999/0.962 d0.65 c0.78; weight_hh: 0.996/0.890 d0.79 c0.65; bias_ih: 1.000/0.938 d0.44 c0.90; bias_hh: 0.996/0.939 d0.45 c0.90; lm.weight: 0.986/0.876 d0.66 c0.76; lm.bias: 1.055/1.261 d0.91 c0.70; l2.weight: 1.010/0.995 d1.15 c0.33; l2.bias: 1.055/1.261 d0.91 c0.70; l3.weight: 1.023/1.137 d0.62 c0.84; l3.bias: 1.014/1.399 d0.41 c1.00
B 93683 all non-table: 0.9961 at 64, 0.9320 at 8192; max|w| start 1.895
    l1.weight: 0.981/1.035 d0.26 c0.97; l1.bias: 1.063/0.913 d0.30 c0.95; weight_ih: 0.995/0.945 d0.62 c0.80; weight_hh: 0.995/0.850 d0.76 c0.67; bias_ih: 0.992/0.925 d0.45 c0.90; bias_hh: 0.986/0.918 d0.46 c0.89; lm.weight: 0.990/0.891 d0.57 c0.82; lm.bias: 1.038/1.076 d0.42 c0.92; l2.weight: 0.999/0.987 d1.06 c0.44; l2.bias: 1.038/1.076 d0.42 c0.92; l3.weight: 1.004/1.003 d0.33 c0.94; l3.bias: 1.003/1.096 d0.12 c1.00
B 116089 all non-table: 0.9988 at 64, 0.9145 at 8192; max|w| start 1.488
    l1.weight: 1.000/1.076 d0.29 c0.96; l1.bias: 1.006/1.044 d0.19 c0.98; weight_ih: 0.999/0.927 d0.58 c0.82; weight_hh: 0.996/0.840 d0.75 c0.68; bias_ih: 1.008/0.962 d0.40 c0.92; bias_hh: 1.007/0.969 d0.42 c0.91; lm.weight: 0.988/0.867 d0.60 c0.81; lm.bias: 1.028/1.058 d0.49 c0.89; l2.weight: 1.032/0.938 d0.99 c0.48; l2.bias: 1.028/1.058 d0.49 c0.89; l3.weight: 1.010/1.019 d0.46 c0.90; l3.bias: 1.014/1.292 d0.31 c1.00
0.9841253514745116 0.937997494269858 0.8798392992565324 0.774117192516226 0.12895960764616443
float32(0.99975) = 0.999750018119812  product of 1024 in float32: 0.7741315597437654
a float32 weight of 0.5 after 1,024 multiplications: 0.7741315960884094
```

### `figs.py`

```python
import json, statistics as st, sys
S = sys.argv[1]
SEEDS = (73129, 93683, 116089)
def L(n): return json.load(open(S + n + ".json"))
def red(row): return row["scores"]["summary"]["reduction_score"]
def rows(rec): return {r["updates"]: r for r in rec["rows"]}
f1 = lambda v: f"{v:.1f}"
E40 = {(k, r, s): L(f"e40_{k}_{r}_{s}") for k in ("lived", "fresh") for r in "SB" for s in SEEDS}
E48 = {(r, s, k): L(f"e48_{r}_{s}_r{k}") for r in "SB" for s in SEEDS for k in (4, 16, 64)}
print("== E40 lived reduction score by row")
R40 = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
for r in "SB":
    for s in SEEDS:
        rw = rows(E40[("lived", r, s)])
        print(r, s, "row0", f1(red(rw[0])), "|", ", ".join(f1(red(rw[u])) for u in R40))
print("== E40: largest move between neighbouring rows, before row 1024 (rows 64..1024) and from 1024 on")
mb = ma = 0
for r in "SB":
    for s in SEEDS:
        rw = rows(E40[("lived", r, s)])
        v = [red(rw[u]) for u in R40]
        b = max(abs(v[i + 1] - v[i]) for i in range(0, 4))     # 64-128,128-256,256-512,512-1024
        a = max(abs(v[i + 1] - v[i]) for i in range(4, 7))     # 1024-2048, 2048-4096, 4096-8192
        print(r, s, "before", f1(b), "after", f1(a), "| max row", max(R40, key=lambda u: red(rw[u])), f1(max(v)),
              "| sd rows>=1024", f"{st.pstdev(v[4:]):.2f} (pop) {st.stdev(v[4:]):.2f} (sample)", "| rise 256->max later", )
        mb = max(mb, b); ma = max(ma, a)
print("max before", f1(mb), "max after", f1(ma))
print("== E48 rank 16: rows")
R48 = (0, 16, 32, 64, 128, 256, 512, 1024)
for r in "SB":
    for s in SEEDS:
        rw = rows(E48[(r, s, 16)]); e = rows(E40[("lived", r, s)])
        print(r, s, ", ".join(f"{u}:{f1(red(rw[u]))}" for u in R48), "| rise 512->1024", f"{red(rw[1024]) - red(rw[512]):.2f}",
              "| at 64: term", f1(red(rw[64])), "E40", f1(red(e[64])), "below" if red(rw[64]) < red(e[64]) else "NOT BELOW")
print("== the hint: term r16 at 1024 minus E40 at 1024; minus E40's highest row")
for r in "SB":
    for s in SEEDS:
        rw = rows(E48[(r, s, 16)]); e = rows(E40[("lived", r, s)])
        hi = max(red(e[u]) for u in e)
        hi_pos = max(e, key=lambda u: red(e[u]))
        print(r, s, f"{red(rw[1024]) - red(e[1024]):.2f}", "| vs highest", f"{red(rw[1024]) - hi:.2f}", "(E40 highest", f1(hi), "at row", hi_pos, ")",
              "| vs E40 8192:", f"{red(rw[1024]) - red(e[8192]):.2f}")
print("== rank at row 1024")
for r in "SB":
    for s in SEEDS:
        print(r, s, ", ".join(f1(red(rows(E48[(r, s, k)])[1024])) for k in (4, 16, 64)))
print("== earlier panels: E40 lived, mean of the four life-family panels (add97 sub97 add73 sub73) and six; loss within 256")
pct = lambda d: 100.0 * d["exact"] / d["n"]
for r in "SB":
    for s in SEEDS:
        rw = rows(E40[("lived", r, s)])
        def mean(u, names): return sum(pct(rw[u]["scores"]["earlier_k12"][n]) for n in names) / len(names)
        life = ("add97", "sub97", "add73", "sub73"); six = ("add89", "sub89") + life; first = ("add89", "sub89")
        print(r, s, "life4:", " ".join(f"{u}:{mean(u, life):.1f}" for u in rw), "| loss by 256:", f"{mean(0, life) - mean(256, life):.1f}",
              "min over rows<=256:", f"{mean(0, life) - min(mean(u, life) for u in rw if u <= 256):.1f}")
        print("     six:", " ".join(f"{u}:{mean(u, six):.1f}" for u in rw), "| loss by 256:", f"{mean(0, six) - mean(256, six):.1f}")
        print("     first2:", " ".join(f"{u}:{mean(u, first):.1f}" for u in rw), "| loss by 256:", f"{mean(0, first) - mean(256, first):.1f}")
print("== fresh at 8192")
for r in "SB":
    for s in SEEDS:
        rw = rows(E40[("fresh", r, s)])
        print(r, s, f"{red(rw[8192]):.2f}", "max over rows", f"{max(red(v) for v in rw.values()):.2f}")
print("== n of reduction pairs")
for s in SEEDS:
    rw = rows(E40[("lived", "S", s)])
    print(s, {n: rw[0]["scores"]["held_plain_k0"][n]["reduction"]["n"] for n in ("add83", "sub83")})
print("== cost, seed 73129")
for key in (("lived", "S", 73129), ("lived", "B", 73129)):
    r = E40[key]; print("E40", key, "cpu_updates", f"{r['cpu_updates']:.0f}", "cpu_scoring", f"{r['cpu_scoring']:.0f}", "rows", len(r["rows"]), "per row", f"{r['cpu_scoring']/len(r['rows']):.1f}", "per update", f"{r['cpu_updates']/8192:.4f}")
for key in (("S", 73129, 16), ("B", 73129, 16)):
    r = E48[key]; print("E48", key, "cpu_updates", f"{r['cpu_updates']:.0f}", "cpu_scoring", f"{r['cpu_scoring']:.0f}", "rows", len(r["rows"]), "per update", f"{r['cpu_updates']/1024:.4f}")
    rr = r["rows"]; prev = 0
    print("   scoring per row:", [round(x["cpu_scoring_so_far"] - (rr[i-1]["cpu_scoring_so_far"] if i else 0), 1) for i, x in enumerate(rr)])
for s in SEEDS:
    for rl in "SB":
        a = E40[("lived", rl, s)]; b = E48[(rl, s, 16)]
        print(s, rl, "E40 upd", f"{a['cpu_updates']:.0f}", "score", f"{a['cpu_scoring']:.0f}", "| E48 upd", f"{b['cpu_updates']:.0f}", "score", f"{b['cpu_scoring']:.0f}")
```

Output of `figs.py`:

```text
== E40 lived reduction score by row
S 73129 row0 6.1 | 76.3, 85.3, 81.0, 81.7, 84.5, 83.6, 81.4, 77.0
S 93683 row0 2.2 | 69.7, 70.1, 83.2, 83.5, 83.9, 82.8, 85.5, 82.1
S 116089 row0 0.1 | 76.1, 80.8, 83.8, 79.7, 88.4, 88.3, 82.4, 83.7
B 73129 row0 4.4 | 83.5, 87.2, 68.1, 80.4, 86.1, 83.8, 82.3, 85.0
B 93683 row0 0.5 | 92.0, 94.7, 92.1, 88.8, 93.8, 92.8, 90.5, 94.5
B 116089 row0 11.6 | 95.4, 92.8, 89.8, 90.6, 90.8, 89.6, 91.4, 89.3
== E40: largest move between neighbouring rows, before row 1024 (rows 64..1024) and from 1024 on
S 73129 before 9.0 after 4.4 | max row 128 85.3 | sd rows>=1024 2.90 (pop) 3.35 (sample) | rise 256->max later
S 93683 before 13.0 after 3.4 | max row 4096 85.5 | sd rows>=1024 1.29 (pop) 1.49 (sample) | rise 256->max later
S 116089 before 8.6 after 6.0 | max row 1024 88.4 | sd rows>=1024 2.71 (pop) 3.13 (sample) | rise 256->max later
B 73129 before 19.1 after 2.7 | max row 128 87.2 | sd rows>=1024 1.42 (pop) 1.64 (sample) | rise 256->max later
B 93683 before 5.0 after 4.0 | max row 128 94.7 | sd rows>=1024 1.52 (pop) 1.75 (sample) | rise 256->max later
B 116089 before 3.0 after 2.2 | max row 64 95.4 | sd rows>=1024 0.88 (pop) 1.01 (sample) | rise 256->max later
max before 19.1 max after 6.0
== E48 rank 16: rows
S 73129 0:6.1, 16:8.7, 32:19.1, 64:50.1, 128:67.8, 256:83.4, 512:89.6, 1024:92.2 | rise 512->1024 2.55 | at 64: term 50.1 E40 76.3 below
S 93683 0:2.2, 16:5.0, 32:13.3, 64:29.4, 128:47.1, 256:75.8, 512:87.7, 1024:92.5 | rise 512->1024 4.82 | at 64: term 29.4 E40 69.7 below
S 116089 0:0.1, 16:1.6, 32:19.9, 64:38.3, 128:57.6, 256:76.3, 512:86.0, 1024:89.5 | rise 512->1024 3.54 | at 64: term 38.3 E40 76.1 below
B 73129 0:4.4, 16:15.4, 32:44.4, 64:59.4, 128:71.3, 256:85.3, 512:90.7, 1024:93.1 | rise 512->1024 2.34 | at 64: term 59.4 E40 83.5 below
B 93683 0:0.5, 16:14.2, 32:52.9, 64:68.9, 128:80.6, 256:92.3, 512:96.9, 1024:97.9 | rise 512->1024 0.98 | at 64: term 68.9 E40 92.0 below
B 116089 0:11.6, 16:29.3, 32:72.4, 64:84.8, 128:91.9, 256:96.5, 512:98.5, 1024:98.7 | rise 512->1024 0.15 | at 64: term 84.8 E40 95.4 below
== the hint: term r16 at 1024 minus E40 at 1024; minus E40's highest row
S 73129 7.62 | vs highest 6.82 (E40 highest 85.3 at row 128 ) | vs E40 8192: 15.14
S 93683 8.60 | vs highest 7.04 (E40 highest 85.5 at row 4096 ) | vs E40 8192: 10.45
S 116089 1.14 | vs highest 1.14 (E40 highest 88.4 at row 1024 ) | vs E40 8192: 5.82
B 73129 6.93 | vs highest 5.84 (E40 highest 87.2 at row 128 ) | vs E40 8192: 8.11
B 93683 4.12 | vs highest 3.26 (E40 highest 94.7 at row 128 ) | vs E40 8192: 3.44
B 116089 7.89 | vs highest 3.28 (E40 highest 95.4 at row 64 ) | vs E40 8192: 9.43
== rank at row 1024
S 73129 90.7, 92.2, 90.8
S 93683 83.4, 92.5, 93.5
S 116089 82.1, 89.5, 89.0
B 73129 91.1, 93.1, 94.0
B 93683 97.6, 97.9, 98.3
B 116089 98.1, 98.7, 98.8
== earlier panels: E40 lived, mean of the four life-family panels (add97 sub97 add73 sub73) and six; loss within 256
S 73129 life4: 0:86.6 64:69.4 128:66.1 256:45.2 512:46.3 1024:45.7 2048:43.6 4096:37.7 8192:32.5 | loss by 256: 41.4 min over rows<=256: 41.4
     six: 0:68.7 64:58.7 128:57.6 256:40.1 512:42.8 1024:43.5 2048:41.9 4096:36.0 8192:30.2 | loss by 256: 28.6
     first2: 0:33.0 64:37.3 128:40.5 256:30.0 512:35.8 1024:39.2 2048:38.3 4096:32.5 8192:25.5 | loss by 256: 3.0
S 93683 life4: 0:88.2 64:42.4 128:42.4 256:49.0 512:43.8 1024:38.0 2048:29.2 4096:21.4 8192:17.5 | loss by 256: 39.2 min over rows<=256: 45.9
     six: 0:64.7 64:34.2 128:32.1 256:37.6 512:34.6 1024:28.6 2048:24.2 4096:17.2 8192:15.3 | loss by 256: 27.2
     first2: 0:17.7 64:17.7 128:11.7 256:14.7 512:16.2 1024:9.7 2048:14.2 4096:9.0 8192:10.8 | loss by 256: 3.0
S 116089 life4: 0:92.8 64:55.9 128:57.1 256:49.7 512:49.3 1024:43.5 2048:40.6 4096:36.4 8192:37.8 | loss by 256: 43.1 min over rows<=256: 43.1
     six: 0:66.4 64:41.4 128:44.3 256:37.8 512:40.1 1024:37.7 2048:35.9 4096:32.9 8192:35.9 | loss by 256: 28.5
     first2: 0:13.5 64:12.3 128:18.8 256:14.2 512:21.7 1024:26.0 2048:26.3 4096:25.8 8192:32.2 | loss by 256: -0.7
B 73129 life4: 0:90.5 64:61.4 128:60.6 256:48.9 512:44.4 1024:47.1 2048:40.8 4096:39.7 8192:33.4 | loss by 256: 41.6 min over rows<=256: 41.6
     six: 0:72.8 64:54.8 128:54.7 256:45.8 512:42.5 1024:42.7 2048:39.6 4096:37.6 8192:32.6 | loss by 256: 26.9
     first2: 0:37.3 64:41.5 128:43.0 256:39.7 512:38.5 1024:33.8 2048:37.2 4096:33.5 8192:31.0 | loss by 256: -2.3
B 93683 life4: 0:96.8 64:68.9 128:68.3 256:58.4 512:51.8 1024:49.2 2048:44.1 4096:41.2 8192:38.8 | loss by 256: 38.4 min over rows<=256: 38.4
     six: 0:73.4 64:58.0 128:55.1 256:48.4 512:44.0 1024:43.3 2048:36.3 4096:31.6 8192:30.8 | loss by 256: 25.0
     first2: 0:26.7 64:36.3 128:28.7 256:28.3 512:28.5 1024:31.5 2048:20.7 4096:12.3 8192:15.0 | loss by 256: -1.7
B 116089 life4: 0:90.8 64:57.7 128:57.7 256:51.3 512:48.6 1024:47.8 2048:45.0 4096:40.7 8192:41.6 | loss by 256: 39.5 min over rows<=256: 39.5
     six: 0:72.7 64:53.7 128:53.8 256:49.0 512:46.6 1024:46.2 2048:43.6 4096:40.6 8192:40.5 | loss by 256: 23.6
     first2: 0:36.3 64:45.8 128:46.0 256:44.5 512:42.7 1024:43.2 2048:40.7 4096:40.3 8192:38.2 | loss by 256: -8.2
== fresh at 8192
S 73129 6.02 max over rows 6.02
S 93683 7.57 max over rows 7.57
S 116089 6.37 max over rows 6.37
B 73129 7.65 max over rows 7.65
B 93683 11.83 max over rows 11.83
B 116089 8.29 max over rows 8.29
== n of reduction pairs
73129 {'add83': 3069, 'sub83': 3074}
93683 {'add83': 3078, 'sub83': 3061}
116089 {'add83': 3070, 'sub83': 3055}
== cost, seed 73129
E40 ('lived', 'S', 73129) cpu_updates 974 cpu_scoring 130 rows 9 per row 14.4 per update 0.1189
E40 ('lived', 'B', 73129) cpu_updates 938 cpu_scoring 125 rows 9 per row 13.8 per update 0.1145
E48 ('S', 73129, 16) cpu_updates 135 cpu_scoring 202 rows 8 per update 0.1316
   scoring per row: [26.0, 24.3, 23.3, 17.6, 17.0, 39.3, 16.2, 38.1]
E48 ('B', 73129, 16) cpu_updates 94 cpu_scoring 168 rows 8 per update 0.0919
   scoring per row: [16.9, 21.7, 18.0, 16.9, 15.3, 31.0, 16.2, 31.8]
73129 S E40 upd 974 score 130 | E48 upd 135 score 202
73129 B E40 upd 938 score 125 | E48 upd 94 score 168
93683 S E40 upd 963 score 126 | E48 upd 134 score 225
93683 B E40 upd 931 score 125 | E48 upd 92 score 178
116089 S E40 upd 936 score 125 | E48 upd 140 score 190
116089 B E40 upd 931 score 124 | E48 upd 96 score 172
```

### `more.py`

```python
import sys, json, statistics as st
B = sys.argv[1]; sys.path.insert(0, B)
import ptread, numpy as np
SEEDS = (73129, 93683, 116089)
red = lambda row: row["scores"]["summary"]["reduction_score"]
print("== sample standard deviation of E40's rows 1,024, 2,048, 4,096, 8,192; change 1,024 -> 8,192 and 4,096 -> 8,192")
for rule in "SB":
    for s in SEEDS:
        r = {x["updates"]: x for x in json.load(open(f"{B}/rec/e40_lived_{rule}_{s}.json"))["rows"]}
        v = [red(r[u]) for u in (1024, 2048, 4096, 8192)]
        e = {x["updates"]: x for x in json.load(open(f"{B}/rec/e48_{rule}_{s}_r16.json"))["rows"]}
        rmid_t = (red(e[512]) + red(e[1024])) / 2; rmid_f = (red(r[512]) + red(r[1024])) / 2
        print(rule, s, f"sd {st.stdev(v):.3f}  x sqrt(2/5) = {st.stdev(v) * (2/5) ** .5:.2f}; 1024->8192 {v[3]-v[0]:+.2f}; 4096->8192 {v[3]-v[2]:+.2f}; hint at 1024 {red(e[1024]) - red(r[1024]):.3f}; rise 512->1024 {red(e[1024]) - red(e[512]):.3f}; R-mid term {rmid_t:.2f} free {rmid_f:.2f} diff {rmid_t - rmid_f:+.2f}")
print("== smallest and largest max|w| per tensor over the six lived networks (tolerance of checks 3 and 4)")
mx = {}
for rule in "SB":
    for s in SEEDS:
        sd = ptread.load(f"{B}/w/life_{rule}_{s}.pt")
        for k, v in sd.items():
            if k != "book.weight": mx.setdefault(k, []).append(float(np.abs(v).max()))
for k, v in mx.items():
    print(f"{k}: {min(v):.4f} .. {max(v):.4f}; 0.00025 x smallest max = {0.00025 * min(v):.2e}")
print("float32 spacing below 1: 5.96e-8; in [1, 2): 1.19e-7")
# simulate check 3's arithmetic in float32 on one tensor: p0*(c) + d  against p0 + d
rng = np.random.default_rng(0)
sd = ptread.load(f"{B}/w/life_B_93683.pt")
worst = 0
for k, p0 in sd.items():
    if k == "book.weight": continue
    d = (rng.choice([-1.0, 1.0], size=p0.shape) * 0.001 * rng.uniform(0.2, 1.0, size=p0.shape)).astype(np.float32)   # a first Adam step: about the rate in size
    free = (p0 * np.float32(0.99975)).astype(np.float32) + d
    wd0 = p0 + d
    dev = np.abs((wd0 - free) - np.float32(0.00025) * p0).max()
    worst = max(worst, float(dev))
print("simulated largest deviation of (free-wd0 - free) from 0.00025 x start, float32, network B 93683:", f"{worst:.2e}")
```

Output of `more.py`:

```text
== sample standard deviation of E40's rows 1,024, 2,048, 4,096, 8,192; change 1,024 -> 8,192 and 4,096 -> 8,192
S 73129 sd 3.347  x sqrt(2/5) = 2.12; 1024->8192 -7.52; 4096->8192 -4.36; hint at 1024 7.619; rise 512->1024 2.554; R-mid term 90.88 free 83.09 diff +7.78
S 93683 sd 1.488  x sqrt(2/5) = 0.94; 1024->8192 -1.84; 4096->8192 -3.40; hint at 1024 8.603; rise 512->1024 4.825; R-mid term 90.12 free 83.70 diff +6.41
S 116089 sd 3.127  x sqrt(2/5) = 1.98; 1024->8192 -4.68; 4096->8192 +1.32; hint at 1024 1.137; rise 512->1024 3.542; R-mid term 87.73 free 84.05 diff +3.68
B 73129 sd 1.645  x sqrt(2/5) = 1.04; 1024->8192 -1.17; 4096->8192 +2.67; hint at 1024 6.935; rise 512->1024 2.343; R-mid term 91.89 free 83.27 diff +8.62
B 93683 sd 1.752  x sqrt(2/5) = 1.11; 1024->8192 +0.68; 4096->8192 +4.01; hint at 1024 4.120; rise 512->1024 0.977; R-mid term 97.42 free 91.31 diff +6.11
B 116089 sd 1.012  x sqrt(2/5) = 0.64; 1024->8192 -1.54; 4096->8192 -2.16; hint at 1024 7.890; rise 512->1024 0.148; R-mid term 98.62 free 90.72 diff +7.90
== smallest and largest max|w| per tensor over the six lived networks (tolerance of checks 3 and 4)
l1.weight: 0.1749 .. 0.4121; 0.00025 x smallest max = 4.37e-05
l1.bias: 0.0258 .. 0.1874; 0.00025 x smallest max = 6.44e-06
rnn.weight_ih_l0: 0.5727 .. 0.9146; 0.00025 x smallest max = 1.43e-04
rnn.weight_hh_l0: 0.6905 .. 1.4232; 0.00025 x smallest max = 1.73e-04
rnn.bias_ih_l0: 0.1757 .. 0.2324; 0.00025 x smallest max = 4.39e-05
rnn.bias_hh_l0: 0.1579 .. 0.2324; 0.00025 x smallest max = 3.95e-05
lm.weight: 0.6166 .. 1.1894; 0.00025 x smallest max = 1.54e-04
lm.bias: 0.0793 .. 0.2049; 0.00025 x smallest max = 1.98e-05
l2.weight: 0.1242 .. 0.1745; 0.00025 x smallest max = 3.10e-05
l2.bias: 0.0793 .. 0.2049; 0.00025 x smallest max = 1.98e-05
l3.weight: 0.6299 .. 1.8948; 0.00025 x smallest max = 1.57e-04
l3.bias: 0.2162 .. 1.0657; 0.00025 x smallest max = 5.41e-05
float32 spacing below 1: 5.96e-8; in [1, 2): 1.19e-7
simulated largest deviation of (free-wd0 - free) from 0.00025 x start, float32, network B 93683: 7.98e-08
```

### `splits.py`

```python
import sys
sys.path.insert(0, sys.argv[1])
import npnet as N
from random import Random
SEEDS = (73129, 93683, 116089)
print("round(f*100):", {f: round(f * 100) for f in (0.025, 0.03, 0.05, 0.10, 0.15, 0.20)}, " 0.05*100 =", repr(0.05 * 100), " round(0.05*6889) =", round(0.05 * 6889), round(0.10 * 6889), round(0.20 * 6889), round(0.025 * 6889))
for seed in SEEDS:
    T = {f: {t[0]: set(N.split(t, f, seed)[0]) for t in N.F4} for f in (0.05, 0.10, 0.20)}
    print(seed, "taught pairs per operation:", {f: sorted({len(v) for v in T[f].values()}) for f in T})
    print("   shuffle seeds:", {f: [seed * 1000 + round(f * 100) + 83 * 7 + t[2][0] for t in N.F4] for f in T})
    for op in ("add", "sub"):
        a, b, c = T[0.05][op], T[0.10][op], T[0.20][op]
        print(f"   {op}: taught at 0.05 also taught at 0.10: {len(a & b)} of {len(a)} (a nested split would give {len(a)}; chance gives {len(a)*len(b)/6889:.0f});",
              f"taught at 0.10 also taught at 0.20: {len(b & c)} of {len(b)} (chance {len(b)*len(c)/6889:.0f})")
    for t in N.F4[:2]:
        red = {f: {(a, b) for a, b in N.split(t, f, seed)[1] if N.needs_reduction(t[0], a, b)} for f in (0.05, 0.10, 0.20)}
        allr = sum(1 for a in range(83) for b in range(83) if N.needs_reduction(t[0], a, b))
        common = red[0.05] & red[0.10] & red[0.20]
        print(f"   {t[0]}: untaught pairs needing the reduction: {len(red[0.05])} at 0.05, {len(red[0.10])} at 0.10, {len(red[0.20])} at 0.20 (of {allr}); untaught at all three: {len(common)};",
              f"0.05 and 0.10: {len(red[0.05] & red[0.10])}; 0.10 and 0.20: {len(red[0.10] & red[0.20])}")
print("items presented in 8,192 updates:", 8192 * 64, "; presentations of each taught pair at 8,192 updates:", {f: round(8192 * 64 / (4 * round(f * 6889)), 1) for f in (0.05, 0.10, 0.20)},
      "; at 1,024:", {f: round(1024 * 64 / (4 * round(f * 6889)), 1) for f in (0.05, 0.10, 0.20)}, "; at 64:", {f: round(64 * 64 / (4 * round(f * 6889)), 2) for f in (0.05, 0.10, 0.20)})
print("taught entries of the stream (4 operations):", {f: 4 * round(f * 6889) for f in (0.05, 0.10, 0.20)}, "; items in 64 updates:", 64 * 64)
# the taught panel samples 300 of the taught pairs
for f in (0.025, 0.04, 0.05):
    try:
        Random(1).sample(list(range(round(f * 6889))), 300); print(f, "taught panel of 300: possible")
    except ValueError as e:
        print(f, "taught panel of 300:", e)
```

Output of `splits.py`:

```text
round(f*100): {0.025: 2, 0.03: 3, 0.05: 5, 0.1: 10, 0.15: 15, 0.2: 20}  0.05*100 = 5.0  round(0.05*6889) = 344 689 1378 172
73129 taught pairs per operation: {0.05: [344], 0.1: [689], 0.2: [1378]}
   shuffle seeds: {0.05: [73129687, 73129688, 73129689, 73129690], 0.1: [73129692, 73129693, 73129694, 73129695], 0.2: [73129702, 73129703, 73129704, 73129705]}
   add: taught at 0.05 also taught at 0.10: 40 of 344 (a nested split would give 344; chance gives 34); taught at 0.10 also taught at 0.20: 149 of 689 (chance 138)
   sub: taught at 0.05 also taught at 0.10: 27 of 344 (a nested split would give 344; chance gives 34); taught at 0.10 also taught at 0.20: 139 of 689 (chance 138)
   add: untaught pairs needing the reduction: 3241 at 0.05, 3069 at 0.10, 2739 at 0.20 (of 3403); untaught at all three: 2357; 0.05 and 0.10: 2923; 0.10 and 0.20: 2475
   sub: untaught pairs needing the reduction: 3227 at 0.05, 3074 at 0.10, 2739 at 0.20 (of 3403); untaught at all three: 2348; 0.05 and 0.10: 2915; 0.10 and 0.20: 2478
93683 taught pairs per operation: {0.05: [344], 0.1: [689], 0.2: [1378]}
   shuffle seeds: {0.05: [93683687, 93683688, 93683689, 93683690], 0.1: [93683692, 93683693, 93683694, 93683695], 0.2: [93683702, 93683703, 93683704, 93683705]}
   add: taught at 0.05 also taught at 0.10: 35 of 344 (a nested split would give 344; chance gives 34); taught at 0.10 also taught at 0.20: 164 of 689 (chance 138)
   sub: taught at 0.05 also taught at 0.10: 30 of 344 (a nested split would give 344; chance gives 34); taught at 0.10 also taught at 0.20: 136 of 689 (chance 138)
   add: untaught pairs needing the reduction: 3251 at 0.05, 3078 at 0.10, 2733 at 0.20 (of 3403); untaught at all three: 2368; 0.05 and 0.10: 2939; 0.10 and 0.20: 2486
   sub: untaught pairs needing the reduction: 3227 at 0.05, 3061 at 0.10, 2726 at 0.20 (of 3403); untaught at all three: 2318; 0.05 and 0.10: 2902; 0.10 and 0.20: 2451
116089 taught pairs per operation: {0.05: [344], 0.1: [689], 0.2: [1378]}
   shuffle seeds: {0.05: [116089687, 116089688, 116089689, 116089690], 0.1: [116089692, 116089693, 116089694, 116089695], 0.2: [116089702, 116089703, 116089704, 116089705]}
   add: taught at 0.05 also taught at 0.10: 36 of 344 (a nested split would give 344; chance gives 34); taught at 0.10 also taught at 0.20: 131 of 689 (chance 138)
   sub: taught at 0.05 also taught at 0.10: 39 of 344 (a nested split would give 344; chance gives 34); taught at 0.10 also taught at 0.20: 139 of 689 (chance 138)
   add: untaught pairs needing the reduction: 3238 at 0.05, 3070 at 0.10, 2721 at 0.20 (of 3403); untaught at all three: 2336; 0.05 and 0.10: 2919; 0.10 and 0.20: 2455
   sub: untaught pairs needing the reduction: 3237 at 0.05, 3055 at 0.10, 2716 at 0.20 (of 3403); untaught at all three: 2327; 0.05 and 0.10: 2908; 0.10 and 0.20: 2442
items presented in 8,192 updates: 524288 ; presentations of each taught pair at 8,192 updates: {0.05: 381.0, 0.1: 190.2, 0.2: 95.1} ; at 1,024: {0.05: 47.6, 0.1: 23.8, 0.2: 11.9} ; at 64: {0.05: 2.98, 0.1: 1.49, 0.2: 0.74}
taught entries of the stream (4 operations): {0.05: 1376, 0.1: 2756, 0.2: 5512} ; items in 64 updates: 4096
0.025 taught panel of 300: Sample larger than population or is negative
0.04 taught panel of 300: Sample larger than population or is negative
0.05 taught panel of 300: possible
```

### `coverage.py`

```python
import sys
sys.path.insert(0, sys.argv[1])
import npnet as N
from collections import Counter
SEEDS = (73129, 93683, 116089)
for f in (0.05, 0.10, 0.20):
    for seed in SEEDS:
        out = []
        for t in N.F4:
            tr, he = N.split(t, f, seed)
            ca = Counter(a for a, b in tr); cb = Counter(b for a, b in tr)
            ma = [v for v in range(83) if ca[v] == 0]; mb = [v for v in range(83) if cb[v] == 0]
            n_unc = sum(1 for a, b in he if a in ma or b in mb)
            n_unc_red = sum(1 for a, b in he if (a in ma or b in mb) and t[0] in ("add", "sub") and N.needs_reduction(t[0], a, b))
            n_red = sum(1 for a, b in he if t[0] in ("add", "sub") and N.needs_reduction(t[0], a, b))
            out.append(f"{t[0]}: absent first {len(ma)}, second {len(mb)}, fewest {min(min(ca[v] for v in range(83)), min(cb[v] for v in range(83)))}; untaught pairs with an absent value {n_unc}" + (f" ({n_unc_red} of {n_red} reduction pairs)" if n_red else ""))
        print(f, seed, " | ".join(out))
```

Output of `coverage.py`:

```text
0.05 73129 add: absent first 1, second 2, fewest 0; untaught pairs with an absent value 247 (97 of 3241 reduction pairs) | sub: absent first 2, second 0, fewest 0; untaught pairs with an absent value 166 (34 of 3227 reduction pairs) | max: absent first 0, second 0, fewest 1; untaught pairs with an absent value 0 | min: absent first 3, second 0, fewest 0; untaught pairs with an absent value 249
0.05 93683 add: absent first 0, second 3, fewest 0; untaught pairs with an absent value 249 (170 of 3251 reduction pairs) | sub: absent first 1, second 0, fewest 0; untaught pairs with an absent value 83 (21 of 3227 reduction pairs) | max: absent first 2, second 2, fewest 0; untaught pairs with an absent value 328 | min: absent first 3, second 4, fewest 0; untaught pairs with an absent value 569
0.05 116089 add: absent first 4, second 1, fewest 0; untaught pairs with an absent value 411 (164 of 3238 reduction pairs) | sub: absent first 0, second 0, fewest 1; untaught pairs with an absent value 0 (0 of 3237 reduction pairs) | max: absent first 1, second 0, fewest 0; untaught pairs with an absent value 83 | min: absent first 3, second 2, fewest 0; untaught pairs with an absent value 409
0.1 73129 add: absent first 0, second 0, fewest 2; untaught pairs with an absent value 0 (0 of 3069 reduction pairs) | sub: absent first 0, second 0, fewest 2; untaught pairs with an absent value 0 (0 of 3074 reduction pairs) | max: absent first 0, second 0, fewest 2; untaught pairs with an absent value 0 | min: absent first 0, second 0, fewest 3; untaught pairs with an absent value 0
0.1 93683 add: absent first 0, second 0, fewest 1; untaught pairs with an absent value 0 (0 of 3078 reduction pairs) | sub: absent first 0, second 0, fewest 1; untaught pairs with an absent value 0 (0 of 3061 reduction pairs) | max: absent first 0, second 0, fewest 2; untaught pairs with an absent value 0 | min: absent first 0, second 0, fewest 1; untaught pairs with an absent value 0
0.1 116089 add: absent first 0, second 0, fewest 2; untaught pairs with an absent value 0 (0 of 3070 reduction pairs) | sub: absent first 0, second 0, fewest 2; untaught pairs with an absent value 0 (0 of 3055 reduction pairs) | max: absent first 0, second 0, fewest 2; untaught pairs with an absent value 0 | min: absent first 0, second 0, fewest 2; untaught pairs with an absent value 0
0.2 73129 add: absent first 0, second 0, fewest 9; untaught pairs with an absent value 0 (0 of 2739 reduction pairs) | sub: absent first 0, second 0, fewest 4; untaught pairs with an absent value 0 (0 of 2739 reduction pairs) | max: absent first 0, second 0, fewest 9; untaught pairs with an absent value 0 | min: absent first 0, second 0, fewest 8; untaught pairs with an absent value 0
0.2 93683 add: absent first 0, second 0, fewest 7; untaught pairs with an absent value 0 (0 of 2733 reduction pairs) | sub: absent first 0, second 0, fewest 8; untaught pairs with an absent value 0 (0 of 2726 reduction pairs) | max: absent first 0, second 0, fewest 5; untaught pairs with an absent value 0 | min: absent first 0, second 0, fewest 7; untaught pairs with an absent value 0
0.2 116089 add: absent first 0, second 0, fewest 8; untaught pairs with an absent value 0 (0 of 2721 reduction pairs) | sub: absent first 0, second 0, fewest 7; untaught pairs with an absent value 0 (0 of 2716 reduction pairs) | max: absent first 0, second 0, fewest 8; untaught pairs with an absent value 0 | min: absent first 0, second 0, fewest 8; untaught pairs with an absent value 0
```

### `nested.py`

```python
import sys
sys.path.insert(0, sys.argv[1])
import npnet as N
from collections import Counter
SEEDS = (73129, 93683, 116089)
def split_nested(t, seed, f):
    taught, rest = N.split(t, 0.10, seed)
    pairs = taught + rest
    k = round(f * len(pairs))
    return pairs[:k], pairs[k:]
for seed in SEEDS:
    for t in N.F4:
        assert split_nested(t, seed, 0.10) == N.split(t, 0.10, seed)
        a, b, c = (set(split_nested(t, seed, f)[0]) for f in (0.05, 0.10, 0.20))
        assert a < b < c and (len(a), len(b), len(c)) == (344, 689, 1378)
    out = []
    for t in N.F4[:2]:
        tr, he = split_nested(t, seed, 0.05)
        ca = Counter(x for x, y in tr); cb = Counter(y for x, y in tr)
        ma = {v for v in range(83) if ca[v] == 0}; mb = {v for v in range(83) if cb[v] == 0}
        red = [(x, y) for x, y in he if N.needs_reduction(t[0], x, y)]
        unc = sum(1 for x, y in red if x in ma or y in mb)
        common = [(x, y) for x, y in split_nested(t, seed, 0.20)[1] if N.needs_reduction(t[0], x, y)]
        unc_c = sum(1 for x, y in common if x in ma or y in mb)
        out.append(f"{t[0]}: values absent at 0.05 first {len(ma)} second {len(mb)}; reduction pairs untaught at 0.05 with an absent value {unc} of {len(red)}; reduction pairs untaught at 0.20 (common to the three fractions) {len(common)}, with a value absent at 0.05: {unc_c}")
    print(seed, " | ".join(out))
print("nested split equals exp11_cell.split at 0.10 on all seeds and operations; taught sets are nested 344 < 689 < 1378")
```

Output of `nested.py`:

```text
73129 add: values absent at 0.05 first 3 second 2; reduction pairs untaught at 0.05 with an absent value 255 of 3236; reduction pairs untaught at 0.20 (common to the three fractions) 2714, with a value absent at 0.05: 223 | sub: values absent at 0.05 first 0 second 1; reduction pairs untaught at 0.05 with an absent value 68 of 3234; reduction pairs untaught at 0.20 (common to the three fractions) 2727, with a value absent at 0.05: 56
93683 add: values absent at 0.05 first 2 second 1; reduction pairs untaught at 0.05 with an absent value 188 of 3238; reduction pairs untaught at 0.20 (common to the three fractions) 2720, with a value absent at 0.05: 164 | sub: values absent at 0.05 first 2 second 2; reduction pairs untaught at 0.05 with an absent value 175 of 3228; reduction pairs untaught at 0.20 (common to the three fractions) 2704, with a value absent at 0.05: 144
116089 add: values absent at 0.05 first 1 second 0; reduction pairs untaught at 0.05 with an absent value 14 of 3235; reduction pairs untaught at 0.20 (common to the three fractions) 2729, with a value absent at 0.05: 11 | sub: values absent at 0.05 first 1 second 1; reduction pairs untaught at 0.05 with an absent value 79 of 3241; reduction pairs untaught at 0.20 (common to the three fractions) 2700, with a value absent at 0.05: 71
nested split equals exp11_cell.split at 0.10 on all seeds and operations; taught sets are nested 344 < 689 < 1378
```

## Closing

The check was closed at 22:54 UTC on 17 November 2026. One file was written, this one. Nothing was committed or pushed, and no other file was edited.
