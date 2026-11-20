# Step 5: one frozen setting chosen on development material, then the claims (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and "继续大胆尝试。" (the coordinator's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." and "Go on trying boldly."). **Source.** `claude/nano-layers-8mmhfm` at the commit that carries this file. Written before the runs it describes; when it was written the prose run of step 4 had been read (`res/life_armsstep4_booksworn.json`) and the sums run of step 4 had not finished (it was restarted after an endless draw: eighteen slots asked of sixteen delays; that arm now has sixteen). A first start of this plan's prose run was stopped unread and its files removed, because this plan had not been committed. Development level; no checker gate, seal or review is claimed.

## 1. What the steps so far give (development material only)

One wide further cell is the best arrangement (step 3). Its families so far: the delays' parts without the level (nl) and with it (fine), the pair at delays 1 and 2, a trust the sending cell owns, the first cell's sureness. Step 4's prose run adds one observation: among gates drawn at random, **the level of counts of the sender of one kind** earns its keep (slots drawn from all sorts, twelve of them, gave 0.0323 nats over the first cell against 0.0292 for nl, and nl with six drawn slots gave 0.0372), while single far bytes do not. The level of every kind as a gate by hand would be 288 families on sums, so the drawn form is the one that is the same on both texts.

## 2. Part A: the choice, by a rule fixed here

Candidates, each one further cell that hears the first cell (C6: two), run together on sums seed 1 and on the life with the three worn books, with the single second cell and nl beside them:

| | Families |
|---|---|
| C1 | nl + pair + own + sure (all without the level) |
| C2 | C1 + the fine parts of the delays |
| C3 | C2 + D slots drawn from all three sorts, reviewed as in `PLAN_4.md` |
| C4 | C2 + 2D drawn slots |
| C5 | C1 + D drawn slots |
| C6 | C2 on C2 (the second copy's sureness gate reads the first copy) |
| C7 | C2 + the pair and the sureness also with the level |

*Rule.* Among the candidates within 1 point of the best candidate on sums (mean of the last three readings after a read line), the one with the largest mean gain on the three worn books; candidates within 0.001 nats of it give way to the one with fewer parts used on prose. The chosen cell is written into `layers.py` as `FROZEN` in an addition to this plan, committed before any claim run.

## 3. Part B: the claims, on material where nothing was chosen

*Sums.* Seeds 73129, 93683, 116089, 406364, 545353; 40,000 lines; `nano/merge`'s system with classes; never-shown sums after a read line (five draws) with the empty past beside it; mean of the last three readings and the last; the first cell, the single second cell, nl and the frozen cell from the same run. Median over the five seeds.

*Prose.* (a) The life "the test text's stream, then `edgeworth-parents`" (`minifly_v149_input_ceiling_dev_20261025/data/edgeworth-parents.txt`; no line has opened it; this line opens it once, here): T2; its first nine tenths read, its last tenth scored right after with nothing changing. (b) A second reading: the same life with `chesterton-ball`, `austen-emma`, `melville-moby_dick` (opened twice by the coordinator). In both: the test text's last tenth right after its own stream with nothing changing (the worn measure on which 1.454 was measured for the full nanoGPT network of `nano/ref`, one seed; scored for some fifty arms of the coordinator's line and for every run of this line: it carries little) and again at the end of the life (what is kept), for each cell.

*The hypothesis of the whole line* (`PLAN_1.md`, section 1): with the one frozen setting the frozen cell is at least 1 point above the single second cell on sums (median of the five seeds, mean of the last three readings, after a read line) and at least 0.010 nats below it on `edgeworth-parents`. **Refuted if neither is reached.** If one is reached and not the other, that is what is reported.

| | Statement | Chance |
|---|---|---|
| F1 | sums: the frozen cell at least 1 point above the single second cell at the median | 0.20 |
| F2 | sums: the frozen cell at or above the single second cell at the median (not below it) | 0.75 |
| F3 | sums: nl at or above the single second cell at the median | 0.65 |
| F4 | `edgeworth-parents`: the frozen cell at least 0.010 nats below the single second cell | 0.85 |
| F5 | the same at least 0.015 below | 0.60 |
| F6 | second reading: at least 0.010 below on each of the three books | 0.80 |
| F7 | `edgeworth-parents`: nl at least 0.003 below the single second cell | 0.75 |
| F8 | kept: at the end of life (a) the frozen cell is still below the first cell on the test text's last tenth, by less than a third of its gain right after the stream | 0.60 |

## 4. Size, cost, what is not run

Parts held and used are reported for each cell. Part A: about eight CPU-minutes on sums, two on prose. Part B: about three CPU-minutes for each seed of sums (three at a time), one for each life. *Also reported if time allows, as a reading and not a claim:* the frozen cell on `all.py`'s system with all kinds of cells but the relay system (the coordinator's 1.4707 system is that with the relay system; over 4 GB), the test text's last tenth with nothing changing. *Not run:* sums over 400,000 lines; sums without classes; the life of two tasks; other seeds of the draws; the control with the step doubled (owed from step 3).

## 5. Addition 1 (20 November; written after the development runs of part A and before any claim run): the choice

`res/sums_armsstep5_seed1.json` (468 CPU-seconds), `res/life_armsstep5_booksworn.json` (40 CPU-seconds), `report_5a.txt`. Sums seed 1, % after a read line, mean of the last three readings; prose, mean gain over the first cell on the three worn books (and the test text's last tenth right after its stream; kept at the end of the life):

| | Sums | Prose gain | Test text; kept | Parts used, sums / prose |
|---|---|---|---|---|
| first cell | 92.7 | 0 | 1.4973; 1.5067 | 7,707 / 323 |
| the single second cell | 96.7 | 0.0239 | 1.4820; 1.5062 | 702,110 / 83,709 |
| nl | 97.4 | 0.0292 | 1.4761; 1.5061 | 33,547 / 3,857 |
| C1 nl + pair + own + sure | 97.8 | 0.0390 | 1.4693; 1.5019 | 1,287,361 / 582,188 |
| C2 C1 + fine | 97.2 | 0.0444 | 1.4654; 1.5019 | 1,989,471 / 665,897 |
| C3 C2 + D drawn | 97.5 | 0.0497 | 1.4603; 1.5005 | 2,120,752 / 684,859 |
| **C4 C2 + 2D drawn** | 97.1 | **0.0522** | 1.4541; 1.4965 | 2,192,818 / 739,051 |
| C5 C1 + D drawn | 97.2 | 0.0453 | 1.4610; 1.4957 | 1,463,967 / 585,033 |
| C6 C2 on C2 | 97.1 | 0.0481 | 1.4631; 1.5124 | 1,989,455 / 665,897 |
| C7 C2 + fine pair + fine sure | 96.8 | 0.0458 | 1.4647; 1.5019 | 2,594,419 / 774,301 |

*The rule:* the best candidate on sums is C1 (97.8); every candidate is within 1 point of it (C7 at exactly 1.0); the largest gain on prose is C4's, and no other is within 0.001 nats of it. **The frozen setting is C4**: one further cell that hears the first cell's branches, with these families of parts, every part born at nothing and taught by the cell's own mismatch with the system's own step: the byte at each of the system's delay lines, indexed by (kind, byte) and by (kind, level, byte); the two bytes at delays 1 and 2, by (kind, pair); one part owned by each sending cell; the first cell's largest probability in ten bins, by (kind, bin); and 2D slots (D the number of the system's delay lines) whose gates are drawn at random from the three sorts of `PLAN_4.md` and reviewed as there, by (kind, gate value). `FROZEN` in `layers.py` is set to it and nothing else is changed. The draws use one seed of the generator on every run. My note before the claims: the review is known not to search (`PLAN_4.md`, reading 1); the frozen cell carries it as it is, because the rule chose it so; on sums C4 is 0.7 points under C1 on seed 1, which one seed cannot tell from nothing.

## 6. Outcome of part B, the claims (entered 20 November after the runs; `res/sums_armsclaim_seed*.json`, `res/life_armsclaim_booksedge.json`, `res/life_armsclaim_booksnew.json`, `report_claim.txt`)

The frozen setting of addition 1, unchanged, on both texts; the first cell, the single second cell (the coordinator's) and nl from the same runs. The draws of the frozen cell's slots differ from the development run's (one generator serves the cells of a run in turn).

**Sums** (five reported seeds 73129, 93683, 116089, 406364, 545353; 40,000 lines; `nano/merge`'s system with classes; 1.23 million cells; about 155 CPU-seconds a run for the three further cells). Never-shown sums, whole answer, %:

| Cell | After a read line, mean of the last three readings; median | The same, last reading; median | Empty past, mean of the last three; median | Parts used (median) |
|---|---|---|---|---|
| first cell | 92.9, 94.0, 92.1, 89.5, 93.0; **92.9** | 93.2, 95.2, 92.4, 89.6, 92.9; 92.9 | 91.8, 93.3, 91.7, 88.0, 92.1; 91.8 | 7,779 trusts |
| single second cell | 97.4, 98.0, 96.6, 96.2, 97.1; **97.1** | 97.4, 98.2, 96.4, 96.8, 97.4; 97.4 | 97.1, 98.1, 96.8, 95.9, 96.7; 96.8 | 708,574 |
| nl (delays' parts by kind and byte) | 97.8, 98.5, 97.2, 96.9, 97.9; **97.8** | 97.9, 99.0, 97.2, 97.6, 97.5; 97.6 | 96.9, 97.1, 95.3, 95.6, 96.7; 96.7 | about 33,500 |
| **frozen cell** | 97.8, 98.2, 96.6, 96.5, 97.1; **97.1** | 97.9, 98.4, 96.4, 97.7, 96.3; 97.7 | 98.2, 97.8, 96.1, 97.4, 95.6; 97.4 | about 2.2 million |

The frozen cell less the single second cell, by seed (after a read line, mean of the last three): +0.5, +0.3, 0.0, +0.3, 0.0; of the medians 0.0. nl less the single second cell: +0.5, +0.6, +0.6, +0.7, +0.8; of the medians +0.7. (The first and the single second cell repeat the coordinator's figures for these seeds.)

**Prose** (T2; nats a byte; each last tenth right after its stream, nothing changing):

| Cell | (a) `edgeworth-parents`, opened once, here | (b) chesterton-ball, austen-emma, melville-moby_dick | Test text's last tenth right after its stream (worn) | The same at the end of life (a); of life (b) | Parts used, life (a) |
|---|---|---|---|---|---|
| first cell | 1.3424 | 1.6093, 1.2056, 1.4387 | 1.4973 | 1.5026; 1.5061 | 327 trusts |
| single second cell | 1.3209 | 1.5900, 1.1861, 1.4177 | 1.4820 | 1.5049; 1.5072 | 85,706 |
| nl | 1.3165 | 1.5880, 1.1814, 1.4134 | 1.4761 | 1.5059; 1.5066 | 3,857 |
| **frozen cell** | **1.2949** | **1.5704, 1.1624, 1.3926** | 1.4549 | 1.4988; 1.4901 | 828,155 |
| frozen below the single second cell | **0.0260** | 0.0196, 0.0237, 0.0251 | 0.0271 | | |

Life (a): 2.96 million cells, 21 CPU-seconds; life (b): 5.00 million cells, 39 CPU-seconds. Share of bytes named right on `edgeworth-parents`: 59.61, 60.13, 60.28, 60.78%. The frozen cell's drawn slots at the end of life (a): the levels of eight kinds of sender (kinds 0, 1, 2, 3, 5, 7, 8, 9), the bytes at delays 3 and 4, the pairs (14, 15) and (4, 8).

*Predictions.* F1 failed (sums: the frozen cell equals the single second cell at the median, 97.1 and 97.1). F2 held (not below it at the median; on no seed below it on the mean of the last three). F3 held (nl is above the single second cell on each of the five seeds, by 0.5 to 0.8). F4 held (0.0260). F5 held. F6 held (0.0196, 0.0237, 0.0251). F7 held (nl 0.0044 below). F8 held (at the end of life (a) the frozen cell is 0.0038 below the first cell on the test text, of 0.0424 right after the stream).

**The hypothesis of the whole line: half is reached.** With one frozen setting the further gates are 0.026 nats below the single second cell on a book that no line had opened (0.020 to 0.025 on three more books), where 0.010 was asked; on sums they are level with it (0.0 at the median, where 1 point was asked). It is not refuted, and it does not hold as a statement about both kinds of text.

Readings (mine).
1. **On prose the reading had more to give than the single second cell took: about as much again.** The first cell to the single second cell is 0.0215 nats on the unopened book; the single second cell to the frozen cell is 0.0260 more. Nothing in the frozen cell was chosen on that book or names anything of English.
2. **On sums at 40,000 lines the single second cell had already taken what the reading can give**, and the simplest further cell is the best there: nl, with a twentieth of the parts, is above the single second cell on five seeds of five (by 0.5 to 0.8 points; I did not predict a size, only the direction), and the frozen cell with sixty times nl's parts is level with the single second cell. Both digits that can err stand near 98% for every further cell. My reading remains that the rest is held in the cells.
3. **One setting did not serve both texts equally, and the rule of choice shows why**: sums seed 1 could not tell the candidates apart (all within a point), so prose alone chose, and it chose the largest cell. A rule that asked for the fewest parts within the noise of sums would have chosen nl or C1 for sums; that is a second setting only if it is chosen per text, and it was not tried.
4. **What is kept is still the weak point.** After another book the frozen cell keeps 0.004 of its 0.042 nats on the old text; after three books 0.016 (it is the only cell that ends below the first cell there, and the drawn level gates seem to be why: C5 and C4 kept most in development). The parts follow the recent text, as the coordinator found for the single second cell.
5. The worn measure: T2 with the frozen cell reads 1.4549 on the test text's last tenth with nothing changing (first cell 1.4973; the full nanoGPT network of `nano/ref` 1.454 on one seed). This tenth has been scored for every arm of this line and some fifty of the coordinator's; one text; it carries little.

## 7. A reading beside the claims, not a claim: the same cells on the system with all kinds of cells but the relay system (`res/life_armsclaim_booksworn_withoutrelay.json`; entered 20 November)

`all.py` without the relay system only (164 kinds of sender: the table's cells, the word cells, fading companions, timing cells, the pool; 5.21 million cells; 6.8 GB; 306 CPU-seconds), the life with the three worn books, the same three further cells, nothing changed. Nats a byte:

| Cell | Test text's last tenth, nothing changing (worn) | bryant-stories, burgess-busterbrown, carroll-alice | Test text at the end of the life | Parts used |
|---|---|---|---|---|
| first cell | 1.4740 | 1.4546, 1.2413, 1.3418 | 1.5953 | 6,403 trusts |
| single second cell | 1.4444 | 1.4342, 1.2357, 1.3165 | 1.6176 | 1,485,001 |
| nl | **1.4421** | 1.4353, 1.2379, 1.3146 | 1.6330 | 63,622 |
| **frozen cell** | **1.4884** | **1.4617, 1.3255, 1.3817** | 1.7151 | 4,914,054 |

**On this system the frozen cell is worse than the first cell** (by 0.014 on the test text and by 0.007, 0.084 and 0.040 on the books), though it names more bytes right (61.46% against 61.10% on the last book): it is too sure. The single second cell and nl gain here as on T2 (0.030 and 0.032 on the test text). I did not predict this and did not look for the cause; my guess, untested, is the one of step 1's reading 2: with 164 kinds, twenty-seven families of parts move for every speaking sender at every byte with the first cell's step, and what was a gain with 10 kinds overshoots. **The frozen setting therefore does not transfer across systems of prose, and the claim of section 6 stands for T2 only.** nl does transfer (both prose systems and sums).

The worn measure, for the record: this system with nl reads 1.4421 on the test text's last tenth with nothing changing and 1.4444 with the single second cell (the full nanoGPT network of `nano/ref`: 1.454, one seed; the coordinator's best system, which also has the relay system, 1.4707 without a second cell). One text, one tenth scored many times; and what this system keeps is poor: after three other books the same tenth reads 1.5953 for the first cell and 1.6330 for nl.
