# Audit of the roster note for the first long lives of E44 (job L2), of the change to the runner that it describes, and a reading of the calibration (17 November 2026)

From the outside reviewer. Read-only: nothing was trained, torch is not installed in the reviewer's container, and no code of another branch was executed. The note is `OPUS_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md`, revision 1, on the default branch at 30628a4c (review 9.157). It releases a roster of 32 lives "for everything except" one change to the runner, "which waits for the reviewer's check". It rests on the brief `OPUS_TASK_E44_LONG_LIFE_WORLD_IN_THE_FULL_SCALE_FORMAT_20261116.md`, revision 2 (cc00760d), which `AUDIT_E44_LONG_LIFE_WORLD_BRIEF_20261116.md` audited, and on the E44 worker's calibration (report a11702e4, runs 4b203807, merged at 08ef17e2; review 9.156). Four questions were put to this audit: the change to the runner; the calibration; the note as a brief; and whether the nine changes of the earlier audit are present in what the roster will run.

**Verdict.**

1. **The change to the runner is sound, with four conditions.** It does not exist as code yet: the runner on the default branch is byte-identical to the one the calibration ran, and the change is a paragraph of the note. As that paragraph describes it, the change only selects which rows of a list are written. The list is answered without learning and the runner asserts so at every call, so the change cannot alter what a life learns, and with its defaults it leaves every figure of the calibration as it is. The conditions: (a) a second change is needed that the note does not name, because the present runner refuses the roster's `replay` arm; (b) the note does not say what its two arguments do to the birth family's items, and the two readings differ by a factor of four in the list's cost; (c) the rows of "decay alone" that the note asks for cannot be read beside what is kept in a life in which families return; (d) the note's check (b) cannot fail on the first of the two arguments, because the life it names visits one family only.
2. **The calibration stands.** From the 48 event files, with every one of 917,504 answers marked again against the reviewer's own stream, the gate's medians equal the report's to the third decimal, and 256 updates a visit is the shortest length that passes. The second lap's table and the outcome of the eleven predictions are reproduced. The numpy reader gives the 3,600 list answers at the end of world 1's calibration life byte for byte, and the rows of decay alone of the born weights item for item.
3. **The note as a brief needs five further changes, none of which touches the runner.** The quantities it says it records are not those the scorer computes, and "kept" is not defined for a life with returns; the score's trend at 256 visits partly reads the schedule; the note does not say what the wrong answers are, which the working rules ask of a retention experiment; the `replay` arm teaches half as many new items a visit, the dose at which the calibration's gate failed in four of eight worlds; and the learning probe of the agreed design is absent without a word.

Nine changes are required in all (section 8). After them no second audit of the note is needed. The lives may start when the changed runner's checks pass against the figures of section 7; the reviewer reads the changed code and the check receipts when they are committed, and that reading need not hold the lives, for the reason given in section 10.

**Terms.**

- A *family* is the four operations add, sub, max and min at one modulus. Its *reduction pairs* are the pairs of add and sub whose answer needs the modulus; a *reduction item* is an asked item on such a pair. An *asked* pair is never taught.
- A *visit* is 256 updates on one family; each update answers 16 asked items first and then learns from 48 items. A visit is a *first visit* (the family's first), a *continuation* (the family is also the family of the visit before) or a *return* (any other visit to a family met before). The *absence* of a visit is the number of visits since the family's last visit; a continuation has absence 1.
- The *born network* is one of E41's births on the project's first family (mod 89), at the *half width*: 256 feedforward units and 128 GRU units, 317,457 parameters. The *birth family* is that first family.
- The *list* is the fixed set of items that the runner answers without learning at the end of visits: for each family taken so far, 200 reduction items (100 of add, 100 of sub, no letters), and 2,000 items of the birth family. A *row* is one such set answered at one visit; its *key* names the visit, the source, the family, the number of decay updates and the visit whose weights answered.
- *Decay alone* is not a run: saved weights with every parameter but the byte table multiplied by 0.99975 once for each update, then answering the list.
- `naive` learns from the 48 new taught items of each update. `replay` learns from the first 24 of them and from 24 items drawn from a *store* of past taught items. `naive-wd0` and `replay-wd0` are the same with weight decay 0 in the life.
- The *reader* is `e44_world_reader.py` (the world, written from the brief's words) and `np_gru.py` (the network, in numpy), beside this audit.

Shell names: `D=origin/codex/minifly-consolidated-20261023; M=MOD97_LOCAL_RULE_20261106; R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109; C=$R/e44_l2_checks.py` on `claude/vibrant-cerf-sje726`. `python3 -I $C <letter>` runs one part (A to J); the whole takes between two and five minutes, most of it in parts C to G.

## 1. The change to the runner (question 1)

### 1.1 What exists, and where the code would change

`git diff --stat 39e4e594 $D -- $M/smallworld/sw_run.py` is empty: the runner on the default branch at 7cabb13a is the file every calibration receipt records (sha256 5d643e2d…, `python3 -I $C B`). So the audit is of the note's section "The runner change", not of code. The lines that the two arguments would touch are these.

| Place in `sw_run.py` at 39e4e594 | What it does now | What the note changes |
|---|---|---|
| `Life.answer_list`, lines 499 to 501 | at the end of every visit the network answers every family taken so far, and the birth family's 2,000 items | `--list-every K`: every family taken so far at the end of every K-th visit and of the last; the family just visited at every visit |
| `Life.answer_list`, lines 502 to 507 | on a family's first visit, its weights answer its 200 items under decay alone for every later visit of the life | `--decay-distances`: only at the given distances in visits from that first visit |
| `Life.answer_list_before`, lines 479 to 483 | before visit 0 the born network answers the birth family's items, and under decay alone for every visit of the life | not mentioned |
| `main`, `Life.__init__` | — | the two arguments, with defaults that keep the present behaviour |

A model of which rows the present runner writes (`rows_present` in `$C`) gives the keys of 13 committed list files exactly and in order, and the entry count of all 48 gate receipts (`python3 -I $C B`). The figures below come from that model and the reader's schedules.

### 1.2 Why a change is needed, and one figure of the note to correct

In a life of 1,024 visits the present runner would answer 14.6 to 16.8 million list items (worlds 1 to 4). At the calibration's measured 0.269 ms an item that is 28 to 33% of the life's computation, and about 370 MB of rows held in memory. Of world 1's 16.8 million, 6.4 million are decay alone of its 52 families, 4.1 million are the birth family, and 6.4 million are the families answered by the network.

The note says "about 10 million answered items per family for decay alone". It is at most 204,600 a family (the first family: 1,023 later visits of 200 items), and 5.3 to 6.4 million a life over all families. The conclusion, that the present rule is too costly, is unchanged.

### 1.3 Does the described change do what the note says, and nothing else?

By reading, yes for what it names. Both arguments only reduce the set of rows. Every row is answered under `no_grad`, and `answer_list` asserts that the hashes of the weights and of the optimizer state are the same before and after (lines 498 and 508). The copies used for decay alone are new networks (`decayed`, lines 293 to 298), and making one draws from torch's random generator. Learning does not depend on that generator: E44's check 6 resumed a life at visit 4, after fewer such copies than the uninterrupted life had made, and ended with the same hashes of weights, events and list (`checks/E44_CHECKS.json`, `check6.resume`). So the number of copies cannot change a life either; the note's check (c) tests exactly this. With the defaults the set of rows is the present one, so the calibration's files are reproduced; check (a) tests that.

Four conditions follow from what the paragraph leaves out or gets wrong.

**(a) A second change is needed.** The roster's arms `replay` and `replay-wd0` cannot run with the present runner. `load_learner` maps the arm to the rule of the birth it may load:

```
git show $D:$M/smallworld/sw_run.py | sed -n '233p'
        want = {"naive": "B", "sweep": "S", "sweep-after-naive": "B"}[arm]
```

`life replay … --learner F` raises `KeyError: 'replay'` on this line. The brief said that "the replay arms of E39 remain in the runner and are not run here"; they remain, but not together with a loaded learner. The note says the roster needs "one small runner change". It needs two.

**(b) The birth family.** The note's two arguments speak of "every family taken so far" and of "a family's first visit". They do not say what becomes of the birth family's 2,000 items, which are answered after every visit (2.05 million answers a long life) and under decay alone of the born weights for every visit (another 2.05 million). The two readings (`python3 -I $C B`, `H`):

| Reading | Answered items, a life of 1,024 visits (worlds 1 to 4) | Share of the life's computation | Over the 32 lives |
|---|---|---|---|
| birth family as now | 5.03 to 5.18 million | 9.8 to 10.0% | 5.2 CPU-hours (8.7%) |
| birth family by the same two arguments | 1.21 to 1.36 million | 2.3 to 2.6% | 1.2 CPU-hours (2.0%) |

Under the first reading four fifths of the list's cost goes to a family that no arm rehearses (the store holds only items offered in the life) and whose score the calibration found at 0.03 after eight visits. It also uses up the 10% that the forecast adds for the store.

**(c) Decay alone, as described, cannot be read.** The note computes decay alone "at these distances in visits from a family's first visit only" and promises "the list at the stated visits with decay alone beside it". In the calibration each family was visited once, so its first visit was its last. In a life by the schedule nineteen visits in twenty are to a family met before. A decay-alone row from a family's first visit says something about what is kept only if the family has not been visited again within the distance, and it can be put beside a figure only if the network answers that family at that same visit. The count over (family, distance) pairs (`python3 -I $C B`):

| Distance in visits | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1,024 visits, worlds 1 to 4: pairs | 203 | 203 | 203 | 203 | 203 | 201 | 198 | 193 | 170 | 110 |
| … not visited again within the distance | 172 | 153 | 113 | 105 | 78 | 58 | 24 | 9 | 2 | 0 |
| … and answered by the network at that visit | 23 | 24 | 14 | 11 | 9 | 7 | 4 | 1 | 0 | 0 |
| 256 visits, worlds 1 to 8: pairs | 105 | 105 | 105 | 104 | 100 | 93 | 82 | 63 | — | — |
| … not visited again within the distance | 79 | 61 | 38 | 24 | 17 | 7 | 0 | 0 | — | — |
| … and answered by the network at that visit | 11 | 8 | 3 | 1 | 1 | 1 | 0 | 0 | — | — |

So a figure of the network can stand beside decay alone for about a tenth of the pairs at short distances and for none at the long distances that the note gives as its reason ("decay can matter here even though it did not within one visit"). At those distances the rows are also known beforehand: decay alone leaves 0.014 of a family after seven visits in the calibration's table, and the born weights score 0.0000 after 2,048 updates (section 2.4). The two decay-0 arms answer the note's question directly, and with identical batches: the store's draws depend on the stream and not on the weights, so `replay` and `replay-wd0` learn from the same items in the same order.

**(d) Check (b) cannot fail on `--list-every`.** The note's check (b) is "a 16-visit life of world 1 with `--list-every 4 --decay-distances 1,2,4,8`". World 1's second family arrives at visit 31:

```
python3 -I $C A      # … families of the 16 visits: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

Family 0 is "the family just visited" at every one of the 16 visits, so it is answered at every visit whatever K is. An implementation that ignored `--list-every`, and one that never answered any other family, would both pass. In worlds 1, 3, 4, 6 and 8 a 16-visit life has one family. The check also asks only that the rows present equal the default run's; a missing row or an extra row passes.

### 1.4 Does the change alter any figure of the calibration that the roster relies on?

No. The roster takes from the calibration the visit length (the gate, read from event files), the cost of an update (measured with the list timed apart) and, as part of the record, the figures of what is kept, which it does not compute again. With the defaults the changed runner writes what the present one writes, and the roster's lives are new lives. One consequence for the forecast is in 1.3 (b): the calibration's 0.053 CPU-seconds an update leaves the list out, and the list adds 2 or 9% depending on the reading.

### 1.5 Would the check receipts catch a wrong implementation?

Check (a) catches a change of the default behaviour, with one reservation: it asks for E44's committed hashes "exactly", and those were made on another container. If the new host's arithmetic differs in the last bit, check (a) fails for a reason that is not the change. Check (c) catches any effect on learning. Check (b) catches a wrong decay distance and nothing else, by 1.3 (d). Section 8, change 4, restates the three checks so that each can fail on what it is meant to test, and section 7 gives the reviewer's digests of the row keys for the lives they name.

## 2. The calibration (question 2)

### 2.1 The gate

The brief's words: "In at least six of eight worlds, the median over the second to the eighth family taken (families 1 to 7 in the file's numbering) of the first-visit level on reduction pairs is between 0.20 and 0.90; and a life of 1,024 visits at that length is forecast within 6 CPU-hours. Use the shortest visit length that passes." The first-visit level is the mean of add's and sub's share right on asked reduction items over the last quarter of the family's first visit.

`python3 -I $C C` regenerates the 24 streams with the reader, marks every asked item of the 48 lives again from the generated bytes (917,504 answers), finds the task, the reduction mark and right or wrong equal to the event files' in every row, and recomputes the medians:

| Arm | Updates a visit | Medians, worlds 1 to 8 | Inside 0.20 to 0.90 | Forecast, 1,024 visits | Passes |
|---|---|---|---|---|---|
| naive | 64 | 0.059 0.112 0.068 0.065 0.068 0.053 0.084 0.133 | 0 (8 below) | 0.97 h | no |
| naive | 128 | 0.169 0.434 0.169 0.322 0.213 0.096 0.195 0.301 | 4 (4 below) | 1.94 h | no |
| naive | 256 | 0.368 0.566 0.301 0.457 0.491 0.368 0.436 0.636 | 8 | 3.85 h | yes |
| sweep | 64 | 0.186 0.355 0.122 0.182 0.240 0.112 0.152 0.233 | 3 | 1.01 h | (not gated) |
| sweep | 128 | 0.452 0.458 0.396 0.416 0.420 0.345 0.422 0.441 | 8 | 2.07 h | |
| sweep | 256 | 0.669 0.656 0.538 0.561 0.737 0.537 0.622 0.595 | 8 | 4.23 h | |

The naive rows equal the report's table to the third decimal. 256 is the shortest length that passes, and it is the adopted one.

### 2.2 The second lap

`python3 -I $C D`, from the 16 branch files (each life continued from its state after 8 visits over families 0 to 7 again), means over the eight worlds:

| Arm | First update of the return | First 16 updates | Last quarter of the return | Last quarter of the first visit | Whole return | Whole first visit |
|---|---|---|---|---|---|---|
| naive | 0.011 (254 items) | 0.085 | 0.583 | 0.489 | 0.458 | 0.340 |
| sweep | 0.014 (254 items) | 0.082 | 0.666 | 0.596 | 0.528 | 0.445 |

The first four columns are the means over the eight families of the report's table, three of whose rows were checked to the third decimal; 254 is the number of reduction items in the 64 first updates. The last two columns are the reviewer's: the share right on reduction items over a whole visit, which is what a score per visit averages. A return eight visits later starts from nothing and ends higher than the first visit did.

### 2.3 The eleven predictions

`python3 -I $C E`. Holding: P44.1 (the reader's digests at the first attempt), P44.2 (row 0: largest difference 0.045 points), P44.3 (the gate passes at 256), P44.6 (at 64 below 0.20 in 8 of 8 worlds), P44.9 (kept one visit later at most 0.20: 8 of 8 worlds, medians 0.007 to 0.076). Failing: P44.4 (64 is not the shortest), P44.5 and P44.10 (no world above 0.90), P44.11 (no world below 0.20 at 256), P44.7 (later families above earlier ones in 2 of 8 worlds), P44.8 (the sweep within 0.15 of `naive` in 3 of 8 worlds; it is above `naive` in 7). This is the outcome that review 9.156 records.

### 2.4 Saved weights through the numpy reader

`python3 -I $C F`, on world 1's `naive` life at 256 updates a visit:

- The state after 8 visits has the file hash and the tensors hash of its receipt.
- The reader answers the list at the end of the life: 1,600 items of families 0 to 7 and the birth family's 2,000. Every generated byte equals the list file's (3,600 of 3,600 items). The shares right are 0.095, 0.010, 0.020, 0.000, 0.000, 0.000 and 0.030 for families 0 to 6; 0.095 for family 7, the family just taught, whose first-visit level in this world is 0.162; and 0.032 for the birth family.
- The born network (E41's backpropagation birth, seed 73129) scores 0.9465 on the 2,000 items, the receipt's final row.
- Decay alone of the born weights, by the reader: 0.8755, 0.7015, 0.4275, 0.2120, 0.0720, 0.0195, 0.0025 and 0.0000 after 256, 512, …, 2,048 updates. These are the list file's shares, with no item different at any of the eight steps.

### 2.5 Two things the calibration's files say that the roster should use

**The first-visit level differs widely between families.** In world 1 under `naive` the eight levels are 0.886, 0.885, 0.368, 0.216, 0.181, 0.747, 0.910 and 0.162 (`python3 -I $C J`). The report's joint fit gives 0.110 per unit of the logarithm of the number of taught pairs, beside −0.105 per unit of ln n. A slope fitted on ln n alone carries that spread as noise (recommendation 5).

**What the wrong answers are** (`python3 -I $C I`). On the calibration's list, a family's 200 reduction items answered one visit after its own visit, eight worlds, 11,200 answers an arm. Almost every answer is two digits and the end symbol. The classes, tried in this order:

| Class of the answer | naive | sweep |
|---|---|---|
| right | 0.028 | 0.023 |
| the sum or difference without the reduction | 0.040 | 0.097 |
| the right answer under the modulus of the family taught last | 0.151 | 0.209 |
| the right answer under the modulus of an earlier family of the life | 0.053 | 0.034 |
| the right answer under 89, the birth family's modulus | 0.019 | 0.009 |
| one of the operands | 0.037 | 0.024 |
| another two-digit number | 0.664 | 0.590 |
| chance, for comparison: a modulus that the life never meets, tried in the place of the family taught last | 0.009 | 0.013 |

A sixth of the answers under backpropagation and a fifth under the sweep are the right answer of the family taught last, applied to the earlier family's code; chance is one in a hundred. Three visits later the shares are 0.138 and 0.253. The family taught last is itself only half learned (0.49 and 0.60 over the last quarter of its visit), so this class is a large part of what the network can then do at all. The memo's section 24.8 found the first family's function overwritten throughout the network; these answers show in part with what. The table is made from files that already exist, and it shows that a reading of the wrong answers carries information in this world. Change 7 asks for it.

## 3. The learners and the width rule

`python3 -I $C G`. The six half-width births at a taught fraction of 0.50 (rules B and S, seeds 73129, 93683, 116089) are committed on the default branch with their receipts (`git ls-tree -r --name-only $D -- $M/exp45/e41 | grep h256_f50`). For each: the file hash and the tensors hash equal the receipt's; the receipt is complete, at H = 256 and U = 128, after 40,960 updates; and the numpy reader's count on E41's 2,000 accuracy-R items equals the receipt's final row exactly (for example seed 93683, rule B: 945 and 940 of 1,000). So the note's check 2 will pass on the files as they are.

The width rule of the brief asks for the half width if it reaches 80% on accuracy R at a taught fraction of 0.50 with the fewest weight-items in at least two of three seeds under backpropagation. From the receipts, by E41's rule of three rows:

| Seed | Half width | Present width | Double width | Cheapest |
|---|---|---|---|---|
| 73129 | 13,056 updates, 2.65 × 10¹¹ | 6,400, 4.60 × 10¹¹ | 3,328, 8.97 × 10¹¹ | half |
| 93683 | 8,704, 1.77 × 10¹¹ | 4,096, 2.95 × 10¹¹ | 2,816, 7.59 × 10¹¹ | half |
| 116089 | 8,960, 1.82 × 10¹¹ | 5,120, 3.68 × 10¹¹ | 3,840, 1.04 × 10¹² | half |

The half width is the cheapest on all three seeds, as the note says. The calibration need not be repeated at another width.

## 4. The note as a brief (question 3)

### 4.1 Arms, and "equal memory"

Backpropagation with a plain replay store is among the arms, as review 9.148 holds to. The store is defined in bytes and checked: 4 × 317,457 = 1,269,828 bytes, the size of one copy of the weights in 32-bit numbers; at most 23 bytes an item gives 55,209 items (`Life.capacity`, lines 515 to 518; `python3 -I $C H`); the runner asserts at every update that neither the count nor the bytes exceed the capacity (line 660), the ledger records the largest of each, and the note's check 4 reads them. "Equal memory" therefore means that the store may hold as many bytes as one further copy of the weights, which is what a candidate with an anchor would hold. It does not mean that `replay` and `naive` have equal memory: `replay` has that copy more. The note's table says so in its last column.

Two facts about this store that the note should state, because they decide how its result is read (`python3 -I $C H`):

- **In a life of 256 visits the store's size hardly limits it.** The families met have 16,896 to 36,672 distinct taught pairs and the store has 55,209 places: 1.5 to 3.3 places for each pair. At 1,024 visits the families met have 80,364 to 108,224 distinct taught pairs: 0.5 to 0.7 places a pair. A stored item has 17 bytes on average, so the store uses about three quarters of its bytes.
- **The share ρ = 0.5 is not tuned** (the note says so), where the memo's section 17.6 tunes it on worlds set apart.

All four arms make the same number of updates, each on 48 items, and consume the same stream. The decay arms differ from their partners in the decay only.

### 4.2 `replay` teaches half as many new items a visit

Under `replay` an update learns from 24 new items and 24 stored ones (lines 643 to 647). A visit therefore teaches 6,144 new items. That is the number the calibration taught in a visit of 128 updates, where the gate failed: four of eight worlds had a median first-visit level below 0.20 (section 2.1). During a later family's first visit the 24 stored items of an update are almost all of other families. So the first-visit level of `replay` may lie at the floor that the gate was set to exclude, and the note's "trend of learning" for `replay` would then be read where the brief said it cannot be read. The gate was asked of `naive` only. This is the cost of replay at equal computation and is a legitimate part of the comparison; but the note does not mention it, the trial does not report it, and nothing says what follows. Change 8.

### 4.3 Lengths and the closing rule

The agreed rule (memo 17.5): lengths grow by a factor of four; a comparison is closed only when the sign of the late-level difference is the same at two successive lengths and the trend difference does not point the other way. The note applies it to 256 and 1,024 visits and drops 64, for the reason the brief gives (a life of 64 visits meets a median of four families). That is in order. Two words of the rule need a definition before the lives, because the note also defines `higher`, `lower` and `not distinguished` by an interval: the "sign" of a difference whose interval contains zero, and which of the two trends "the trend difference" means. As written, a late-level difference of +0.001 at both lengths closes the comparison. Change 5, definition D12.

The note has four worlds at 1,024 visits and eight at 256, as the memo's section 17.7 has. With four worlds the 90% interval is the mean ± 1.18 standard deviations; "stuck" is then the statement that no trend was found, at that precision. Recommendation 6.

### 4.4 The quantities the note names are not the quantities the scorer computes

The note's section "What is recorded and computed (the E44 scorer; read on reduction pairs)" names eight quantities. The ledger is in every receipt already. For the others, `git show $D:$M/smallworld/sw_score.py` has these:

| The note's quantity | In the scorer |
|---|---|
| score by age on asked reduction items, with the blind predictor's | `score` (line 99) and `lw_by_visit` (lines 285 to 288) take the share over all asked items of a visit, max and min and pairs without reduction included |
| late level, score trend | from that score (lines 152 to 156) |
| trend of learning over every family met, n from 1 | `lw_trend` (lines 279 to 282) fits n = 2 to 8; `score` fits per task, not per family |
| kept at return on the return's first update, by absence class, absences of one visit apart | `score` (lines 125 to 144) pools per task over all asked items, in three classes with 1 to 4 visits together |
| relearning at return | `score` counts taught items until 26 of a task's last 32 asked items are right; on reduction items of a family it is not defined, and for add and sub the calibration's levels seldom reach that criterion |
| the list, with decay alone beside it | `lw_kept` (lines 291 to 314) divides by the share at the end of the family's **first** visit (line 301) and counts visits since that first visit |

The last row matters most. In a life with returns, what is kept of a family at some visit must be measured against the end of its most recent visit. Measured against the first visit, a family that has returned five times shows "kept" above 1.

None of this needs the runner: the scorer reads files and can be corrected after the lives. But the definitions decide what the predictions PL2.1 to PL2.8 mean, so they must be fixed before the lives are read. Section 5 proposes them, and part J of `$C` computes them from an event file and a list file alone, so that the worker's scorer can be compared with it on the trial, as E44's check 7 did.

### 4.5 At 256 visits the score's trend partly reads the schedule

The definition of the score's trend is the memo's (17.4) and the defect is the reviewer's. A continuation starts from the level at which the visit before ended (0.49 to 0.58 in the calibration); a return starts from 0.01. So the score of a visit depends on which kind it is, and the share of continuations changes through a life (`python3 -I $C A`):

| Life | Share of continuations by quarter of the life |
|---|---|
| 256 visits, world 1 | 0.69, 0.11, 0.12, 0.14 |
| 256 visits, world 3 | 0.73, 0.30, 0.28, 0.14 |
| 256 visits, world 6 | 0.59, 0.33, 0.22, 0.16 |
| 256 visits, world 7 | 0.41, 0.23, 0.09, 0.27 |
| 1,024 visits, worlds 1 to 4 | 0.27 to 0.36, then 0.07 to 0.12 in every later quarter |

Over the second half of a 256-visit life the share moves by up to 0.18 in either direction. The calibration has no continuation, so the difference in score between the two kinds of visit is not measured; if it is about 0.3 (a continuation starts near 0.5, where a return starts near 0 and averages 0.46 over the visit), the change of share alone moves the mean score by about 0.05 between the third and the fourth quarter. That is a slope of about 0.15 per unit of ln(v + 1), where the trends of learning in the calibration lay between −0.03 and −0.13. "Stuck" and "still learning" for every arm at 256 visits would then partly be statements about the schedule. Differences between arms are not affected, because the arms of a world share its schedule. At 1,024 visits the share is steady over the second half. Change 6 reads the score's trend on returns only and reports the late level by kind of visit.

### 4.6 What the wrong answers are

The third working rule adopted in review 9.148 is that retention experiments score what wrong answers are. L2 is one, the event file and the list carry the generated bytes for this purpose (change 6 of the earlier audit), and the note asks for no reading of them. Section 2.5 shows on the calibration's own files that the reading carries information. Change 7.

### 4.7 The learning probe and the saved states

The agreed design (memo 17.7 and 17.10) makes two probes part of L2, taken as branches from states saved at ages that double: the learning probe, which teaches the same reserved families at every age, and the keeping probe. The world reserves its last four families for the first ("the last four families are never scheduled: the probe tasks"). The keeping probe is covered by the list. The learning probe is absent from the note without a word, and the note commits states of world 1 only. The trend of learning over first visits measures the same thing inside a life, but on a different family at each age, and section 2.5 shows how much families differ. The probe measures the same four families at every age and in every arm, and costs about 22 CPU-minutes for the eight long lives (three ages, four visits each). The worker's container does not outlast its session, so states that are neither committed nor used at once are gone, and the long lives would have to be run again (34 CPU-hours) to take the probe later. Change 9.

### 4.8 Cost, the trial, the order

- **Forecast.** `python3 -I $C H` gives 59.3 CPU-hours for the 32 lives at the calibration's 0.0529 CPU-seconds an update with the note's 10%; the note says about 59. The list adds 1.2 CPU-hours under change 2 and 5.2 without it.
- **A cross-reference.** The note says of the store's cost "check 6 measures it". Check 6 reads the decay back; the cost is check 7.
- **The trial** (working rule 2) is world 1's two lives of 1,024 visits, and the worker "continues without waiting" unless one of four mechanical conditions holds. That is a full-length trial on one seed. It uses the birth of seed 73129, the only one the calibration used; five of the eight worlds use the other two births (recommendation 3).
- **The order** puts the sixteen lives of the decay arms before the six remaining long lives. If the session is cut short, the main question is the one that loses lives (recommendation 1).
- **Restarts.** A long life takes four hours, and a container restart killed three runs of E41 in flight on 16 November (commit 3daf3234). By default the runner saves a state after 1, 2, 4, …, 512 and 1,024 visits; a restart at visit 1,000 loses about two hours (recommendation 2).
- **Lines at once.** The roster is one line of work with three processes. Whether the programme stays within three lines with it (the fifth working rule) is for the coordinator to state; the note does not.

### 4.9 Predictions

PL2.1 to PL2.8 are yes-or-no statements in the terms of the note's section "How the result is stated", so each names its own refuting result, and they are consistent with one another (the chance of PL2.7 lies below the complement of PL2.1). Three remarks. PL2.3 and PL2.5 are stated on absences of 33 visits or more at 256 visits, where a world has 3 to 26 such returns and 13 to 112 asked reduction items in their first updates (world 3: 3 returns, 13 items; `python3 -I $C A`); the note's own rule for comparing arms pools absences of five visits or more (recommendation 4). The comparison of `replay-wd0` with `replay` is named and has no prediction. No number is on file for a level; section 9 records the reviewer's.

## 5. Definitions proposed for the recorded quantities (text for change 5)

All from the event file unless the list is named. V is the number of visits read.

- **D1. Reduction item:** an asked item of add or sub whose pair needs the reduction (`wrap` = 1; task number 4g or 4g + 1).
- **D2. Score by age on reduction items**, visit v: the share right over the reduction items of visit v, add and sub pooled. The blind predictor's: the same share of the `blind` column.
- **D3. Kinds of visit:** first visit; continuation (absence 1); return (absence 2 or more).
- **D4. Late level:** the mean of D2 over the last quarter of the visits (the last max(1, V // 4)). Reported also for each kind of visit apart.
- **D5. Score trend:** the least-squares slope of D2 on ln(v + 1) over the returns among the visits V // 2 to V − 1. The slope over all visits of that half (the memo's definition) is reported beside it and not used.
- **D6. First-visit level of a family:** the mean of add's and sub's share right on reduction items over the last 64 updates of its first visit (the E44 scorer's `lw_first_visits`).
- **D7. Trend of learning:** the least-squares slope of D6 on ln n over all families met in the V visits, n = 1 for the first met.
- **D8. Kept at return, from the stream:** for a continuation or a return, the reduction items of the visit's first update, which are answered before any update of the visit; right and count pooled over the visits of an absence class. Classes: 1 visit; 2 to 4; 5 to 32; 33 or more.
- **D9. Kept on the list:** for a family f answered by the network at the end of visit v, with ℓ ≤ v its most recent visit: its share right on its 200 items at v divided by its share at the end of ℓ. Tabled by v − ℓ in the classes of D8, with mean and median; a family whose share at ℓ is 0 is left out and counted.
- **D10. Relearning at return,** for the visits of D8 by absence class: the family's figure (as D6) over the first 16 updates and over the last 64 updates of the visit, beside the last 64 updates of the family's previous visit.
- **D11. Between arms:** per world the difference of a quantity; the mean and its 90% interval (t over worlds); `higher` or `lower` where the interval excludes zero, otherwise `not distinguished`.
- **D12. Closed:** the comparison of `naive` and `replay` is closed at 1,024 visits if the late-level difference is `higher` at both 256 and 1,024 visits, or `lower` at both, and at 1,024 visits neither the difference in D5 nor the difference in D7 is distinguished on the opposite side. Otherwise it is open at 1,024.
- **D13. Stuck, still learning:** as the note has them, with D5 and D7, and with the two intervals printed.
- **D14. What the wrong answers are:** for D8's items and for the list's rows by D9's classes, the share of answers in each class of section 2.5, tried in that order, with the chance line.

`python3 -I $C J <ref>:<events.npz> <visits> <ref>:<list.npz> <world>` prints D2 to D10 for a committed life; without arguments it runs on world 1's calibration life followed by its second lap.

## 6. The nine changes of the earlier audit (question 4)

| | Change of the audit of the brief | In what the roster will run |
|---|---|---|
| 1 | the level over the last quarter of a first visit | present: D6 is the scorer's `lw_first_visits` |
| 2 | the gate's floor of 0.20, its families and the trend defined | applied in the calibration. The roster's trend runs over all families from n = 1, where the calibration's ran over n = 2 to 8; the note says so |
| 3 | every way the gate can fail has a named end | applied in the calibration; no further step was called |
| 4 | what is kept, on a fixed list answered at the end of every visit, with decay alone beside it | **not carried over in a usable form.** The list is kept every eighth visit, which is enough (it gives 72 to 182 points of 200 items a world at absences of 5 to 32 visits in a life of 256 visits, and 1,377 to 2,021 at 33 or more in a life of 1,024; `python3 -I $C B`). But "kept" is not defined for a family that returns (4.4), and decay alone cannot be read (1.3 c). Changes 3 and 5 |
| 5 | one second lap at the adopted length | run; section 2.2 |
| 6 | the event file records the bytes of the answer given | present in the runner; **not used by the note** (4.6). Change 7 |
| 7 | checks against the reviewer's digests and its hand-made event file | passed in the calibration; the note's check 1 repeats the digests. Section 7 adds the figures for the roster's lives |
| 8 | statements about the schedule, the optimizer and the split corrected | present; the note's schedule figures are the reader's (`python3 -I $C A`: 41 to 52 families at 1,024 visits, 9 to 17 at 256, 440 to 497 of 972 to 983 returns after more than 8 visits) |
| 9 | seed 73129's births in the calibration; what follows if the width rule gives another width | present; the rule gives the half width on all three seeds (section 3) |

## 7. Figures for the worker to compare

**Row keys of a list file.** A key is `at_visit,source,family,decay_updates,weights_from_visit,items`, one line each, sorted, hashed with SHA-256 (`keys_digest` in `$C`; `at_visit` 65535 is "before visit 0", family −1 the birth family). `--list-every K` is read as: at the end of the visits after which K, 2K, … visits are done, and of the last. The birth family follows both arguments (change 2). `python3 -I $C K <list.npz> <world> <visits> <K> <all|none|d1,d2,…> [in-turn]` compares a list file with these keys and its items with the reader's.

| Life | Keys | Items | Digest |
|---|---|---|---|
| the calibration life: world 1, 8 visits in turn, defaults | 81 | 46,800 | 138031253dfc3cb6fa3149a21285e008209a2d619da5122d28d4d012de9dfd02 |
| world 5, 32 visits by the schedule, defaults | 241 | 165,200 | 7808f56e968445359a2acf6e16ee209999f947b60427ca96f467b4c0b4459476 |
| world 5, 32 visits, `--list-every 4 --decay-distances 1,2,4,8` | 77 | 38,800 | 58490377353a33d8d7c818cba2335eaba4f0cb2738574ef86981a7bbd9ff0303 |
| world 5, 32 visits, `--list-every 4 --decay-distances none` | 57 | 27,600 | b53f105db373e2030e1579154e5b4e2b3fc3e12a0c7dafb197a5c859af07f13d |
| the trial: world 1, 1,024 visits, `--list-every 8 --decay-distances none` | 5,031 | 1,238,400 | e7cdf44f0f9485ff9a64824d3544e11361d5b451309651ba3ad2c927b089bfe7 |
| world 1, 256 visits, `--list-every 8 --decay-distances none` | 548 | 169,000 | f8de2869adcdd1ed6df3c1fae39e54907e3a7d2826bdc291e5bd048f929970c0 |

The first digest is that of the committed file `F1/lw_naive_w1_v8_turn_list.npz` (part K on it prints "keys equal as sets: True"). World 5's first 32 visits take the families 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 2, 1, 2, 1, 2, 1, 2, 3, 2, 1, 0, 3, 0, 3, 3, 2, 0: four families, five continuations, 23 returns. With K = 4 the network answers 42 family rows fewer than with the defaults, so the argument is exercised.

**The list's items** (the receipt's `list.items_sha256`, families in the order first met; the reader's digest equals the calibration receipt's for world 1):

| World | 256 visits | 1,024 visits |
|---|---|---|
| 1 | 17 families, 964db73fd4743388c1560010c78de52b1300eb98d7c4ea8b0c29254b54bc45eb | 52, 48e6d710e93cbe6777cc2c0cb51c611949fbf1309e17c66c911c04487b7282fc |
| 2 | 13, 6543da12b3bf1aea6f395eaa58b986c4357706260900d51dcbc25ce456f2eb7d | 52, af05e03d09dd9dfc856a53d02482a318ac909254407074e70a1932de78648001 |
| 3 | 9, 1119b530753f8c58f3a4ae1196bd50218707f1b0f24f9c53ad19f98783b3b038 | 52, 83ce5344ae3802969a9a4cc5b866bff19fc90b27eea38e81d216feebc926c60a |
| 4 | 17, 22efa38c62ff0b0a3c6a5450cba73e75814b97b496f86d068081f85318259907 | 47, dc1954983fdba82be32cc497c9691c8e0833cf26c974cd3aef404eafd3b53719 |
| 5 | 13, 8014b5612fedd3025334a44606692193af32f58e05c8cd8a3fa71305ca4dc1e6 | |
| 6 | 13, 54ebf072c9fd9392c3eb67dcadf39c6caca1e132332dbe9db7947f0df5518e2f | |
| 7 | 13, 5c72f286d83b9c9e76e07b213c4977700c5622615771115a27c60d177b22341e | |
| 8 | 10, 1d09bf5b4a63007f0ceb6d6ee0b45b2c0b60fa60a86fd3753d0e4f1a6a8955e5 | |

**The schedules** of worlds 1 to 8 and their digests are in section 7 of the earlier audit; the calibration's check 1 already compared them.

## 8. The nine required changes, as text to apply

Changes 1 to 4 concern the runner and its checks; 5 to 9 the note.

1. **Name the second change.** In "The runner change": "`load_learner` also accepts the replay arms: `replay` loads backpropagation's birth and `sweep+replay` the sweep's (the line `want = {…}[arm]`). Nothing else in the replay path changes." In the first paragraph "one small runner change" becomes "two small runner changes". Check 3 (ρ = 0 equals `naive`) exercises it.
2. **Say what the two arguments do to the birth family.** "The birth family's 2,000 items follow both arguments: the network answers them before visit 0 and at the end of every K-th visit and of the last; decay alone of the born weights is computed at the given distances, counted in visits from the start of the life." Correct the count: "in a life of 1,024 visits the present rule answers 15 to 17 million items, 28 to 33% of the life's computation: 5 to 6 million for decay alone of the families, 4 million for the birth family, 5 to 6 million for the families answered by the network".
3. **Decay alone.** "`--decay-distances` also takes `none`. L2 uses `--list-every 8 --decay-distances none`: decay alone cannot be read in a life with returns (audit, 1.3 c), it leaves nothing beyond eight visits, and the two decay-0 arms carry the question." If decay alone is wanted nevertheless, the readable form is: from the weights at the end of each visit of a family, at the next K-th visit, and only if the family is not visited again before it; that form would need its own check.
4. **The three checks of the change, restated, and a fourth.**
   - (a) "With the defaults, world 1's 8-visit calibration life at 256 updates gives, under the changed runner and under the runner at 39e4e594 on this host, the same hashes: the list's arrays, the events' arrays, the final weights and optimizer state, and the learned batches. Where this host reproduces E44's committed receipt, they are the committed hashes."
   - (b) "World 5, 32 visits by the schedule under `naive`, with the birth of its world (seed 93683), three times: with the defaults; with `--list-every 4 --decay-distances 1,2,4,8`; with `--list-every 4 --decay-distances none`. The row keys of each list file have the reviewer's digest (audit, section 7), and every row present in two of the runs is equal in them item by item, generated bytes included." About 25 CPU-minutes in all.
   - (c) as the note has it, on the three runs of (b), with the optimizer's hash added.
   - (d) "The two arguments are entries of the life's settings; the receipt and every state carry them, and a resumed life keeps them. Check 5 compares the list file's arrays hash as well."
5. **Define the recorded quantities.** Replace the paragraph "What is recorded and computed" by D1 to D14 of the audit's section 5, or refer to them, and replace "(the E44 scorer; read on reduction pairs)" by: "the scorer's additions for these are made with the trial's report and compared on the trial's two lives with the reviewer's part J". In "How the result is stated", replace the sentence on closing by D12.
6. **Read the score's trend on returns.** D5 and D4 as in section 5. In "Stuck": "the score trend is the slope over the returns of the second half".
7. **Say what the wrong answers are** (working rule 3): D14, for `naive` and `replay`, at 256 and 1,024 visits.
8. **State what `replay` costs in new items, report it in the trial, and name what follows.** In "What the record says": "Under `replay` a visit teaches 6,144 new items, the number the calibration taught at 128 updates a visit, where four of eight worlds were below the gate's floor." In the trial's report: "the median first-visit level (D6) of `replay` over the families met, beside `naive`'s." And: "If it is below 0.20, the trend of learning is not read for `replay`, and release 2 adds `replay` at ρ = 0.25."
9. **Take the learning probe, or say why not.** "From the states after 64, 256 and 1,024 visits of the eight long lives (`--save-states`), a branch of four visits over the probe families 52 to 55 (`sw_run.py branch <state> --groups 52,53,54,55`). Reported by age and arm: each probe family's first-visit level (D6). About 22 CPU-minutes." If the probe is left out, the note says so with its reason, and says which states are kept and where. In "Done", add: "the report lists the committed files from `git ls-tree`" (working rule 4).

Also correct "check 6 measures it" to "check 7".

## 9. Recommendations (none holds the lives)

1. **Order:** the trial, then the six long lives, then the eight short ones, then the sixteen decay lives; the third process can start world 2's long life beside the trial.
2. **Restarts:** `--save-states 64,128,256,384,512,640,768,896,1024` for the long lives and one sentence that an interrupted life is continued with `life --resume <latest state>`; older states may be deleted except those of change 9.
3. **The other two births:** before the lives, one 8-visit life in turn at 256 updates under `naive` in worlds 2 and 3 with the births of seeds 93683 and 116089 (4 CPU-minutes). The calibration's medians with seed 73129's birth were 0.566 and 0.301 there; a median below 0.20 would say that the gate does not hold for that birth.
4. **PL2.3 and PL2.5** on absences of five visits or more, as the note's own rule for arms has it, or with the count of returns beside them.
5. **The trend of learning** with the coefficient of ln n in the joint fit with the logarithm of the taught pairs beside it, as in the calibration's tables.
6. **"Stuck":** print the two intervals, and say `declining` where an interval lies wholly below zero, so that a trend not found is told apart from a trend downward.
7. **A prediction for `replay-wd0` against `replay`.**
8. **Size in the repository:** at the calibration's 3.0 bytes an entry the event files of the 32 lives are about 180 MB. That is acceptable; states beyond those of change 9 should stay out.

**The reviewer's expectations, on record before any life.** (i) `naive`'s late level on reduction items at 1,024 visits lies between 0.40 and 0.55 (55%): this is what the calibration's whole-visit levels give if nothing changes with age (0.34 at a first visit, 0.46 at a return, about a tenth of the visits continuations). A level outside the band is an effect of the long life. (ii) `replay`'s median first-visit level is below 0.20 in at least three of the eight worlds at 256 visits (50%). (iii) Under `naive`, at returns after two or more visits, more than a tenth of the answers are the right answer under the modulus of the family taught last (75%).

## 10. Can the lives start?

Yes, after the nine changes, with no second audit of the note. The order of work: the worker makes the two changes to the runner, runs checks (a) to (d) and the note's checks 1 to 7, compares its row keys with section 7, and starts the trial. The reviewer reads the diff of `sw_run.py` and the check receipts when they are committed, with parts K and J. That reading need not hold the trial: if check (c) passes, a defect in the list's code cannot change the weights or the event file of any life, and D2 to D8 and D10 come from the event file. Only the list's rows (D9, and the list's part of D14) would be lost for the lives already run.

## 11. What I did not check

- The changed runner, which does not exist yet. Only its description was audited.
- Anything that needs torch: that `replay` runs after change 1, the store's cost, resume with a store, a host's arithmetic.
- The note's statements about E43, E38 and E46 in "What the record says"; other parts of the reviewer's reading of 17 November cover E43 and E46.
- The decay-0 and sweep-after-naive sets of the calibration (48 lives); the gate does not read them.
- The scorer's driver `lw_f1.py`; the tables were recomputed from the event files instead.
- The streams of the 32 lives. Each receipt records a digest for every visit, and the reader can regenerate any visit; the reviewer will sample the trial's when it is committed.
