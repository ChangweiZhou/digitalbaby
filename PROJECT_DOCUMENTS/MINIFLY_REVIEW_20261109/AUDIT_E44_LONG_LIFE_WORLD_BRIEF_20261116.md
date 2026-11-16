# Audit of the E44 brief and its specification files at 436d9677 (16 November 2026)

From the outside reviewer. Read-only: nothing was trained, torch is not installed in the reviewer's container, and no code from another branch was executed. The brief is `OPUS_TASK_E44_LONG_LIFE_WORLD_IN_THE_FULL_SCALE_FORMAT_20261116.md`, revision 1. The generator is `PROJECT_DOCUMENTS/COORD_CHECKS_20261116/lw_world_spec.py`, with `lw_row0_check.py` and the coordinator's outputs `OUTPUT_LW_WORLD_MAC.txt` and `OUTPUT_LW_ROW0_MAC.txt` beside it. They implement section 25.4 of the reviewer's memo (revision 16, at d0426a74). No list of points came with the brief; the six questions below are the reviewer's own.

**Verdict.** The world is the one the memo proposed. A second implementation written from the brief's words gives the coordinator's committed output line for line, the six digests included, at the first attempt; and the numpy reader gives the nine lines of the row-0 check exactly from the two half-width births. Nine changes are required before release. None alters a rule of the generator. They concern what the calibration reads and records: the level the gate reads means different things at different visit lengths; the brief reports what is kept but lists nothing that measures it; nineteen visits in twenty of a life are returns and the calibration has none; and three statements about the schedule and the optimizer are not exact. After the nine changes no re-audit is needed.

**Terms.**

- A *family* is the four operations at one modulus; its *reduction pairs* are the pairs of add and sub whose answer needs the modulus. The *family's figure* is the mean of add and sub on asked reduction items.
- An *asked* pair is never taught. A *visit* is a run of updates on one family; each update learns from 48 taught items and answers 16 asked ones first.
- A *born* network is one of E41's births on the first family of the project (mod 89). *Half width* is 256 feedforward units and 128 GRU units, 317,457 parameters.
- *Decay* is the optimizer's weight decay: each update multiplies every parameter but the byte table by 0.99975. *Decay alone* is not a run: saved weights so multiplied t times and scored.
- The *reader* is `e44_world_reader.py` beside this audit: the world written from the brief's words, standard library only.

Shell names: `D`, `M` and `K` as in the brief; `R=PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109`; `C=$R/e44_brief_checks.py` on `claude/vibrant-cerf-sje726`. `python3 -I $C <letter>` runs one part (A to I).

## 1. The world (question 1)

**The reader reproduces the file.** `python3 -I $R/e44_world_reader.py 1 2 3` prints the twelve lines of `$K/OUTPUT_LW_WORLD_MAC.txt` exactly: the schedule figures, the first moduli, the taught-pair counts and the six digests (`python3 -I $C A`). The reader ran under Python 3.13.17; the committed output was made under 3.11. P44.1 holds.

**Three places where the words alone would not have sufficed.** None is a defect of the file.

1. *The number of taught pairs.* The brief writes `round(0.10 · p²)`. Python's `round` sends a half to the even number, so the six moduli ending in 5 get 202, 302, 422, 562, 722 and 902 taught pairs. A reader that rounds a half upwards gives each of their 24 tasks one pair more, and another stream: the in-turn digests of worlds 1 and 3 differ and world 2's does not, because none of its first eight moduli ends in 5 (`python3 -I $C A`).
2. *The rules taken from the small world.* The pair is `pool[randrange(len(pool))]`; the letters are drawn by `choice`; one `random()` is drawn at every visit after the first; an empty absence class falls back to the first non-empty of short, medium, long; the family is `sorted(class)[randrange(len(class))]`; the digest hashes `[id // 256, id % 256, a, b, len(prompt)]`, the prompt and each answer value mod 256. The brief says "as in the small world" for all of these. They are lines 33, 34 and 36 of the E39 brief, and the reader took them from there.
3. *Family numbers.* The file and the row-0 output number families from 0. The gate and predictions P44.7 and P44.9 say "families 2 to 8", "6 to 8" and "2 to 4", which count from 1. A calibration life has families 0 to 7.

**What the committed digests cover.** In worlds 1, 2 and 3 the first four visits of a life are all on family 0, so that digest covers 4 of the 224 tasks. The in-turn digest covers 32. No committed figure fixes the schedule exactly, and the calibration runs worlds 1 to 8. The reader's digests for the 24 calibration streams, the schedules and the tasks of worlds 1 to 8 are in section 7.

**Moduli and bytes, from the harness's source** (`python3 -I $C B`; 285 `.py` files at 436d9677):

- The moduli in every task tuple of the harness are 73, 83, 89 and 97. The small world's are 8 to 37. The world's 56 moduli from 41 to 100 share none.
- The families use 59 bytes, the largest 122; the small world adds lower-case codes and symbols up to 126. The world's code bytes run from 128 to 255 and share none of them. All 224 codes are distinct in worlds 1 to 3.
- E43's unrelated task used bytes 128 to 225. About 342 of a world's 448 code bytes fall there. It does not matter for this learner, which never met that task; it would for a network that had.
- In the two half-width births the byte table's rows from 128 upward are bit-equal between the sweep's birth and backpropagation's, so neither birth moved them. A code's two bytes are read as two untouched initial rows.

**Sizes differ sixfold** (`python3 -I $C C`).

| Modulus | Taught pairs per operation | Showings of each taught pair in a visit of 64 / 128 / 256 updates |
|---|---|---|
| 41 | 168 | 4.6 / 9.1 / 18.3 |
| 50 | 250 | 3.1 / 6.1 / 12.3 |
| 70 | 490 | 1.6 / 3.1 / 6.3 |
| 83 (E40, 16 taught items a task an update) | 689 | 1.5 / – / – |
| 100 | 1,000 | 0.8 / 1.5 / 3.1 |

- Of the 56 moduli, 41 give fewer taught pairs than E40's 689, and 11 give 264 or fewer, the range in which L1a's fresh networks stayed at 0.36 or below.
- Whether a born network learns a family of 168 to 264 taught pairs is not on file. L1a does not decide it: its networks were fresh. E40 does not either: one modulus, 689 pairs, three families behind the learner. The calibration will show it, if its levels are tabled against the number of taught pairs. In worlds 4, 5 and 8, three or four of the eight families taken in turn have 264 pairs or fewer.
- So the measures must be given by modulus, as the brief does for the first-visit level. The trend over eight families within one world is confounded with their sizes (recommendation 3).

## 2. The learner (question 2)

**The births are E41's and the row-0 check reproduces.** The two files on the default branch have the sha256 their receipts record (da14a18be302 for the sweep, 4c6d4302b51f for backpropagation), 317,457 parameters and 128 GRU units; the births end at accuracy A 95.8 and R 92.0 (sweep) and 94.5 and 94.7 (backpropagation). With the brief's sample, the numpy reader gives all nine lines of `$K/OUTPUT_LW_ROW0_MAC.txt` to the last digit (`python3 -I $C E`): asked reduction pairs 0.0 to 5.2%, pairs without reduction 1.3 to 53.0%, max and min 2.3 to 49.2%.

**Seeds.** No branch on the remote holds an E41 receipt for seed 93683 or 116089 (`git ls-tree` on every fetched head). The brief says every world uses seed 73129's births "until the other seeds' births are on the default branch". If they arrived during the calibration, its 48 lives would not share a learner. Change 9 fixes the learner for the calibration.

**Width.** The brief keeps the memo's condition and gives it a rule: half width if it reaches 80 on R at half the pairs with the fewest weight-items in two of three seeds under backpropagation. On seed 73129 it does (2.65e11 against 4.60e11 and 8.97e11). One seed is enough to calibrate on, not to fix the width. Two things are missing: what happens if the other seeds never report, and that a different width means the calibration is repeated, where the brief says "I decide".

**The optimizer.** E41 saved weights only, so the life starts a fresh AdamW; one optimizer then runs through the whole life. The brief says so, but names `exp26_candidate.make_opt(m)`, which takes no rate or decay; the runner's is `sw_net.make_opt(m, lr, wd)`.

## 3. The schedule (question 3)

`python3 -I $C D`; worlds 1 to 8, and 200 worlds for the distributions.

| | 64 visits | 256 visits | 1,024 visits |
|---|---|---|---|
| Families met, worlds 1 to 8 | 6, 3, 3, 4, 4, 2, 4, 4 | 17, 13, 9, 17, 13, 13, 13, 10 | 52, 52, 52, 47, 52, 49, 45, 41 |
| Families met, 200 worlds: least, median, most | 1, 4, 9 | 5, 13, 22 | 36, 51, 52 |

- **The brief's sentence is not exact.** "A life of 1,024 visits meets all 52 (worlds 1, 2 and 3 do)" holds in 99 of 200 worlds. Worlds 4, 6, 7 and 8 of the calibration's eight meet 47, 49, 45 and 41.
- **A life of 64 visits is nearly one family.** The second family arrives at visit 31, 12, 44, 17, 1, 18, 12, 17 in worlds 1 to 8; over 200 worlds at a median of 15, a tenth of them by visit 2 and a tenth after visit 51. Ten worlds of 200 meet family 0 alone in 64 visits, and 41 meet two or fewer. Quantities over families, such as the trend of learning, cannot be read on lives of 64 visits.
- **Nineteen visits in twenty are returns.** Of 1,024 visits about 975 are returns. Their absences: 1 to 4 visits 43%, 5 to 32 visits 32%, 33 or more 25%. An absence of one visit, which is no absence at all, is 16% of returns.
- **Returns beyond the reach of decay.** Decay alone leaves nothing of a family that receives no gradient after about 2,000 updates (memo 24.6). Returns after more than 2,048 updates are 49% of all at visits of 256 updates, 39% at 128 and 25% at 64. After more than 1,024 updates they are 57%, 49% and 39%.

**Decay in the brief.** It states the decay (0.25, an argument), and that what the stream stops pushing on is gone within about 2,000 updates. It gives no reference for this learner and no arm without decay. For the half-width births, decay alone gives (`python3 -I $C F`; the birth family's reduction score as a share of its start):

| Updates | Sweep's birth | Backpropagation's birth |
|---|---|---|
| 256 | 91% | 94% |
| 512 | 62% | 77% |
| 1,024 | 11% | 23% |
| 2,048 | 0% | 0% |

So in a calibration at 256 updates a visit, a family four visits old is at 11 to 23% of its level by decay alone, and the figures the brief reports as "what is kept" could not be told from it. Change 4 puts the decay-alone figure beside each; recommendation 1 and the note for L2 go further.

## 4. The calibration (question 4)

**The level the gate reads changes its meaning with the visit length.** It is "the share right over the task's asked reduction items in the last 64 updates of its first visit".

- At visits of 256 updates that is the last quarter: the level reached.
- At visits of 64 updates it is the whole visit: the mean over the learning curve. A network at the row-0 level for six updates and right ever after scores 0.90, so the upper limit cannot be passed at 64, and the step named for a ceiling at 64 could not be called. The gate says "use the shortest length that passes", and 64 passes unless the learner is at the floor.
- At 32 updates, the step named for a ceiling, the window is longer than the visit.

The last quarter of the visit at every length (16, 32, 64 updates) reads one thing throughout, with about 62 items a family at 64 updates (60 to 72 in the fixture of section 7; a standard error of 0.06 at most). Change 1.

**The lower limit sits inside the floor.** Before any update a born network answers 0 to 5% of a new family's reduction pairs (check 4). A fresh network that has stored a tenth of a family without its rule answers 6.0 to 11.8% after 8,192 updates (E40's six fresh runs) and up to 16.9 and 17.5% under backpropagation at the present and double width (E41 at a tenth). A tenth of the asked add reduction pairs have their mirrored pair among the taught ones (0.080 to 0.115 over 168 families), which is worth 0.05 on the family's figure to a network that recalls and commutes. A median of 0.10 would pass the gate from inside that range (`python3 -I $C H` and `C`). Change 2 sets 0.20.

**The gate does not say what to do in every case.** Two failures have a named step: above 0.90 at 64 in six worlds, below 0.10 at 256 in six worlds. A third is possible: no length passes and neither case holds, for example five worlds inside the limits at each length. Change 3.

**What is kept is reported but nothing measures it.** The report is to give "the earlier families' reduction pairs at the end of each later visit", and P44.9 reads it. The four listed changes to the runner and scorer produce no such figure: the stream asks only the family being visited, the runner saves states at visits 1, 2, 4 and 8 (`is_pow2`, `sw_run.py` line 555), and the event file keeps right or wrong for stream items only. Change 4 gives the list, the times and the figure.

**No return is seen.** The calibration takes eight new families in turn. In the lives it calibrates, nineteen visits in twenty are returns, and L2 follows by a roster note without an audit. E36 measured how fast a family comes back, at the present width, on the birth family, after a whole life. Change 5 adds one second lap at the adopted length through the runner's own `branch` command; the generator is not touched.

**Earlier families at every visit.** With change 4 the calibration records, at the end of every visit, every family taken so far and the birth family. That is the standing request of memo section 9.

**Rules and arms.** Both arms run in all eight worlds at all three lengths; the gate is read under backpropagation and the sweep's figures are tabled. The two arms differ in the birth as well as in the life (recommendation 2).

**Further steps.** One is named for the floor and one for the ceiling. The step at 512 updates forecasts 5.1 CPU-hours for a life of 1,024 visits under backpropagation and 6.3 under the sweep, by E41's rates (`python3 -I $C G`).

## 5. What E43 changes (question 5)

- **Fresh or continuing optimizer.** Fresh at the start of the life, continuing through it (section 2). The fresh optimizer's first steps fall on the first family's first visit, which the gate leaves out. That is E40's condition, where a new family was learned within 64 updates.
- **What the old answers turn into is not recorded.** Review 9.146 says of E43: "I should have asked what the old answers would turn into." The event file here keeps `correct` and nothing of the answer. The runner's `ask` command writes the generated bytes; the stream does not. A life of 1,024 visits answers 1 to 4 million asked items, and whether a family at its return is answered with another family's reduction, with the unreduced sum or with the end symbol alone could then be learned only by running it again. Three small columns settle it. **Required** (change 6): the file's format should be fixed before the long lives, and the calibration lives are the scorer's test data.
- **A reminder probe.** In a life every return is a reminder, and the instrument already gives "kept at return" and "relearning at return"; no separate probe is needed there. The calibration has no return. **Required** in the small form of change 5; nothing more.
- **Decay.** See section 3. Beside each kept figure the decay-alone figure is **required** (change 4); a calibration without decay is **recommended**; an arm without decay in L2 belongs in its roster note.

## 6. Checks, cost, predictions (question 6)

**Checks.**

- Check 1 compares the worker's file with the specification and three worlds' output. It should also compare the 48 receipts' stream digests with the reader's, since the calibration runs eight worlds (change 7).
- Check 2 says the properties are "asserted in the file". The file asserts that taught and asked are disjoint and that codes are distinct only when run as a script (line 115), not when imported. The runner's own assertion on every learned item covers the first.
- Check 3 (the learner is E41's), check 4 (row 0 to 2 points; the reader agrees exactly), check 5 (the small world still runs) and check 6 (asking does not teach, same stream, resume) are sufficient as written. With change 6, check 5 compares the nine columns the small world's file had.
- Check 7 needs a hand-made event file. Section 7 gives one made without a network, with the figures the scorer should return.

**Cost** (`python3 -I $C G`). E41 measured 0.035 CPU-seconds an update at the half width under backpropagation and 0.043 under the sweep, learning from 64 items. The calibration's 57,344 updates are 0.62 CPU-hours at those rates; an update here learns from 48 items and answers 16. "Under an hour" stands for the brief as written. With changes 4 and 5 and both recommendations 1 and 2 it is about 2 CPU-hours. A life of 1,024 visits is 0.6 to 0.8 CPU-hours at 64 updates a visit, 1.3 to 1.6 at 128 and 2.5 to 3.1 at 256; at the present width 2.0 to 2.2, 4.0 to 4.4 and 8.1 to 8.7, so 256 would fail the forecast there.

**Predictions.** They match the readings once the family numbers and the limits follow changes 1 and 2. P44.5 and P44.6 are the two corners that call no step (the ceiling at 256, the floor at 64); the two that do call one have no prediction (recommendation 5). P44.9 does not say over which families "an earlier family" is taken (change 2).

**Other findings.**

- `$K/__pycache__/lw_world_spec.cpython-311.pyc` is committed (17,044 bytes; the only compiled file in the tree; `.gitignore` lists `__pycache__/`). I read its 16-byte header only: CPython 3.11, source size 7,249, which is the committed source's. It was not loaded. It should be removed.
- The specification's line 4 says the file also runs the row-0 check; it does not.
- The brief's header says "Edit nothing existing"; its section on the runner and scorer edits both.
- The runner's `ITEM_MAX` is 15 bytes, the small world's longest item. Here the longest is 23. It sets the replay buffer's capacity, which the calibration does not use.

## 7. Figures for the worker to compare

All from the reader; first 16 digits of each SHA-256 (`python3 -I $C A`).

| World | In turn, 8 visits of 64 | of 128 | of 256 | Schedule of 1,024 visits | Tasks (codes, moduli, splits) |
|---|---|---|---|---|---|
| 1 | a6832371bb578e25 | 66fac5d10c8e7e3f | eb7652f02be6e59c | 536b65c6f741acd6 | d92e5208663e54f7 |
| 2 | 6a655c90f09d3f96 | bfe7923d52793b94 | 9b018b82e282e5eb | c76e6f3dc61f6008 | bc34eedf690ad84a |
| 3 | 51ad093c04e2ceee | c4000b185954587e | 9df1279deb75a607 | a55de8c310fb6b6f | 556d1c0358fc3b0f |
| 4 | 429f1f2e7259b16d | 352db682c6aa352e | 322a7cf9b19ddc33 | be38d42cba646c82 | 4bfb89d4f736e930 |
| 5 | 9ba7e2068abd0293 | af5914c14a03f1d7 | a1b0df5bd052fe7f | 96cf5f8888c9a72b | 0325a29d6d4c68ea |
| 6 | ae8dc5fabfd624f8 | f6898122baec099c | efeb69b3eb1b1110 | 6156fdf0d08e1d01 | f1dbfc7588f243a0 |
| 7 | c6c604cc5e0f4a26 | 1c70119bd3a09479 | b5d17a901c4a68d1 | b2e47a8776a34b7a | 0d2a50b56b83672e |
| 8 | c5b00a855784be08 | d4396f7c1fc6ea8f | 878bf9f7ae483272 | 1f3d18490052875c | 89213030751632a3 |

- *In turn:* the world digest's rule over the stream of the calibration life. The first column of worlds 1 to 3 is the committed one.
- *Schedule:* SHA-256 of the 1,024 family numbers, one byte each.
- *Tasks:* for each task in id order, SHA-256 over its two code bytes, the modulus, the symbol, the number of taught pairs as two bytes, then every taught pair and every asked pair as the bytes a, b in the split's order.

| World | Second lap by `branch` from the state after 8 visits, at 64 | at 128 | at 256 | The list of change 4, families 0 to 7 |
|---|---|---|---|---|
| 1 | fe57803b545231c8 | 8767dfef71110f56 | 1fca17d80fb6dc53 | 30ff746fca187dfc |
| 2 | 99b3b098fa55b872 | e81c60b4fffab05c | f0938bafb395cb2d | 1595e706a290dc75 |
| 3 | 9744af6bf53e1056 | 2b72ba9ce738e865 | 24fc7f10b3b3160b | 01570d69b34e3939 |
| 4 | 547c7777c21e0313 | 8cafde91d52e15be | 304f8b2aa1843d62 | ffe731cb779ae080 |
| 5 | 347e952e0de7cc03 | 19e1e2587603d446 | cc2e3f19671f24c0 | d4a34cbdd1e2f9b4 |
| 6 | 34e5d891bd52718f | eed1945b1002cc51 | 65221c59e43c6f76 | 405abaeb3d76fed8 |
| 7 | 4b9c94ec7b07aa8f | 0885180cc1967f80 | 7b7a58436c12a05c | e6c0e03a3eed132e |
| 8 | 78095c63df0bf937 | 6d99da6cc4ef419d | 840a056f53dc8bd4 | 8b6daf6e67d8060d |

- *Second lap:* the runner's `branch` over families 0, 1, …, 7 from the state after 8 visits. Its k-th visit draws by the visit's item rule from `Random(f"world|{S}|branch|8|{k}")`, the string the runner has at 436d9677. The digest is the world digest's rule over the eight visits, the `stream_sha256` of the branch's receipt.
- *The list:* for add and then sub of each family, the first 100 asked pairs in the split's order that need the reduction; SHA-256 over `[id // 256, id % 256, a, b]` of the 1,600 entries.

**An event file made without a network** (`python3 -I $C I`), for check 7. World 1, eight families in turn; an asked item counts as right when `(7 * a + 3 * b + update + task id) % 4 == 0`, with `update` counted from 0 over the life.

| Visit length | SHA-256 of `correct`, `wrap`, `mirror` (one byte an entry) | Median of the family's figure over families 1 to 7: last 64 updates / last quarter / whole first visit |
|---|---|---|
| 64 | 1758aef2b46fe2cf, 7e86dcd6b983160c, ba4723d24ac8fd11 | 0.2502 / 0.2303 / 0.2502 |
| 256 | ab4bb15cdaf186a5, 5e741a17769666a0, 4362a866676f1023 | 0.2517 / 0.2517 / 0.2586 |

The script prints each family's figure for add and for sub with the number of items.

## 8. The nine required changes, as text to apply

**1. The level the gate reads (the scorer's item 3, the gate, the list of what is reported).**

- "**First-visit level on reduction pairs:** the share right over the task's asked reduction items in the last quarter of its first visit: the last 16, 32 and 64 updates at visits of 64, 128 and 256 updates, the last 8 at 32 and the last 128 at 512. The family's figure is the mean of add and sub. Beside it the scorer gives the same share over the whole first visit and over its first 16 updates."
- In "Reported", replace "the same at the end of the first 16 updates" by "the same over the whole first visit and over its first 16 updates".
- Reason to state: with a window of 64 updates the level at a visit of 64 updates would be the mean over the visit and at 256 the level reached, and at 64 the upper limit could not be passed.

**2. The gate's limits, the families it counts, the trend, and two predictions.**

- "In at least six of eight worlds, the median over the second to the eighth family taken (families 1 to 7 in the file's numbering) of the first-visit level on reduction pairs is between **0.20** and 0.90."
- Reason to state: "Before any update a born network is at 0 to 5% (check 4). A fresh network that has stored a tenth of a family without its rule is at 6 to 12% after 8,192 updates (E40) and up to 17% (E41), and recall of mirrored pairs alone is worth 0.05."
- "The trend of learning in a calibration life is the least-squares slope of the family's figure against ln n over n = 2 to 8, the first family taken being n = 1."
- P44.6: "below 0.20". P44.7: "the mean first-visit level of families 5 to 7 is above that of families 1 to 3 (file's numbering)". P44.9: "the median over families 0 to 6 of a family's share right on the list of change 4 one visit after its own, divided by its share at the end of its own visit, is 0.20 or less".

**3. Every way the gate can fail.** After the two named steps add: "If no length passes and neither case holds, stop and report the three medians of each world." And to the step at 512 updates: "its life of 1,024 visits is forecast at about 5 CPU-hours under backpropagation and 6 under the sweep."

**4. What is kept: a list, when it is answered, and decay alone beside it.** Add to the changes to the runner and scorer:

- "**At the end of every visit of a calibration life** the runner answers a fixed list without learning and writes every generated answer (by a state at every visit and `ask`, or within the run). The list: for add and for sub of every family taken so far, the family just visited included, the first 100 asked pairs in the split's order that need the reduction, without letters; and the birth family's accuracy R on E41's two samples. The list's digest for families 0 to 7 is in the audit's section 7."
- "**Kept** for a family at a later visit is its share right there divided by its share right at the end of its own visit. The report tables it by visits since, with the modulus."
- "**Decay alone, no run:** beside each kept figure, the same list answered by the weights the network had at the end of that family's own visit, every parameter but the byte table multiplied by 0.99975^u, u the updates since."
- Reason to state: "For the half-width births decay alone leaves 91 to 94% of the birth family's reduction score after 256 updates, 62 to 77% after 512, 11 to 23% after 1,024 and none after 2,048 (the reviewer's computation)."

**5. One second lap, at the adopted length.** "At the adopted visit length each of the 16 lives is continued by `branch` from its state after 8 visits over families 0, 1, …, 7 (the runner's generator, `world|S|branch|8|k`; the reviewer's stream digests are in the audit's section 7). Reported for each family: its share right on reduction items in the return's first update and first 16 updates, the list of change 4 at the end of the return, and its share over the last quarter of the return beside that of its first visit. The gate does not read it." Cost: at most 32,768 updates.

**6. The answer given is recorded.** "The event file gains three columns `g1`, `g2`, `g3` (uint16): the bytes generated for the asked item in order, 65535 where generation had ended. They are written in life, branch and solo files. Check 5 compares the nine columns the small world's file had."

**7. Check 1.** Add: "The stream digest in each of the 48 calibration receipts equals the reviewer's for its world and visit length, and the schedule of 1,024 visits and the tasks of worlds 1 to 8 have the reviewer's digests (audit, section 7)." And to check 7: "The hand-made file is the audit's (section 7): the scorer returns its medians under the window of change 1."

**8. Statements to correct.**

- "At 0.05 a life of 1,024 visits meets 36 to 52 of the 52 families, all of them in 99 of 200 worlds; of the first eight worlds, 1, 2, 3 and 5 meet all and 4, 6, 7 and 8 meet 47, 49, 45 and 41. A life of 256 visits meets 5 to 22 (median 13). A life of 64 visits meets a median of 4, and family 0 alone in 10 of 200 worlds, so quantities over families are read at 256 and 1,024 visits."
- "one fresh `sw_net.make_opt(m, lr, wd)` for the life, as E39's runner (`exp26_candidate.make_opt` takes no rate or decay)."
- Header: "Edit nothing existing but `smallworld/sw_run.py` and `smallworld/sw_score.py`."
- Check 2: "the runner asserts on every learned item that its pair is taught and not asked; the file's own assertions run only when it is run as a script."
- Split: "the first `round(0.10 · p²)` by Python's `round`, which sends a half to the even number: 202, 302, 422, 562, 722 and 902 at the six moduli ending in 5."
- Groups and visits, schedule, digest: after "as in the small world" add "(lines 33, 34 and 36 of the E39 brief)".
- Remove `$K/__pycache__/lw_world_spec.cpython-311.pyc`, and the words "with torch and the harness present, also the row-0 check" from line 4 of the specification. The digests do not change.
- "Done": "the report lists the committed files from `git ls-tree`; the event files and receipts of all 48 lives and of the second lap, the answers of change 4, and the states after 8 visits of world 1."

**9. The learner.**

- "The 48 calibration lives, and the second lap, use seed 73129's births in every world, and every receipt records the birth file's sha256."
- "If E41's three-seed table gives the present width, the calibration is repeated at the present width before L2 is rostered."
- "If E41 has not reported for the other two seeds when L2 is rostered, the rule is read on the seeds on file and the roster note says on how many."

## 9. Recommendations (none holds the release)

1. **The calibration once more without decay**, under `naive` only: 24 short lives with decay 0 in the life, 28,672 updates, about 0.3 CPU-hours. It shows whether the first-visit level and what is kept depend on the decay at all, before any life is run.
2. **A third arm, the sweep's life from backpropagation's birth** (E39's `sweep-after-naive`), 24 short lives. As written the two arms differ in the birth and in the life; the project's constraint allows any optimizer before life.
3. **Levels against size.** Table the family's figure, pooled over worlds, against its number of taught pairs (264 or fewer; 265 to 688; 689 or more) and against its place in turn, and fit the trend with the logarithm of the taught pairs beside ln n. Name now what follows if the small families are at the floor and the large ones are not, and make the lowest modulus an argument of the constructor (default 41) so that raising it is not a change to the generator.
4. **Name the argument `--learner`,** not `--birth`: E39's revision 4 uses "birth" for a birth inside the world, and the runner's receipts carry a `birth` entry for it.
5. **Two more predictions:** the level is above 0.90 at 64 updates in six worlds, and below 0.20 at 256 in six worlds. These are the outcomes that call a further step.
6. **In "what this cannot show":** "Whether a family of 168 to 264 taught pairs can be learned by a born network. No result on file bears on it."

**For the roster note of L2** (not part of this release): an arm without decay, since a quarter to a half of all returns come after more than 2,048 updates; absences of one visit tabled apart from absences of 2 to 4; the replay buffer's capacity computed with 23 bytes an item; and the lives of 64 visits not used for any quantity over families.

## 10. Can E44 be released?

Yes, after the nine changes of section 8, without a re-audit. The generator's rules are untouched by them.

## 11. What I did not check

- **Anything that needs torch.** The runner's and scorer's changes are not written yet; check 3, the costs and the births' training are the workers'.
- **Whether a born network learns a tenth of a family,** at any modulus. It is the calibration's subject.
- **The CPU rates.** They are E41's check 6, measured learning from 64 items an update without asking.
- **The compiled file's contents.** I read its header and did not load it.
- **The digests under Python 3.11.** The reader ran under 3.13.17 and equals the output made under 3.11; I ran nothing under 3.11.
- **E41 for seeds 93683 and 116089.** No receipt exists on any branch I fetched.
