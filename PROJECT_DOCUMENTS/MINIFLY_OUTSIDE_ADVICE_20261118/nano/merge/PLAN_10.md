# Line `nano/merge`, tenth plan: kinds that the system makes while reading, so that a cell can carry its place (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and then "继续大胆假设，基于事实努力尝试。" (the coordinating session's translation is in `PLAN_1.md`). **Source.** `claude/nano-merge-8mmhfm` at the commit that holds this file. Written before any run of `merge10.py` other than its check (`check10_output.txt`; in the check, 6,000 lines of the development seed, I saw which kinds were made: four, among them delays 2, 3, 4, 6, 7). Development level; no gate, seal or review is claimed.

## 1. The evidence this follows

`PLAN_7.md` and `PLAN_8.md`: a kind is a set of delays with no place in the line, so its cells mix what follows at several places. The cell of "the whole pair" has counted on 98.6% of never-shown sums; the measure of alikeness, taken on such cells, prefers a grouping that is not the carry; on a kind with one more delay, which holds the '=' where the tens digit is named, the same measure gives the carry exactly (`look8.py`, a kind laid down by hand for the look). The base has kinds of at most four delays on sums and runs of at most six bytes on prose, listed by hand for each task.

## 2. The one addition (to setting C2 of `PLAN_6.md`; nothing else changes)

**At each finding, a kind of largest trust may be refined by one more delay.** The candidates are every (R, d): R one of the 16 kinds of largest trust (trust counted as in C2, a kind's own and that of the kinds made from it; plain kinds and made kinds alike), d a delay of the chain that R does not have. The worth of a candidate is measured on the system's own last 50,000 bytes: the sum over those bytes of the log of what the cell of (R with d) says of the byte, the byte left out of its own cell, less the same for R. **The 16 of largest positive worth are made**; one that is there already is kept with its cells and its trusts; a new one starts with no cells and trusts at nothing; one that is no longer among the 16 goes, with the class kinds found in it. A made kind is a kind like any other: it counts and speaks, it can be refined again at a later finding, and classes are sought in it by the same search when its trust puts it among the candidates.

Nothing says which delay, which kind, or that a place matters. *Laid down:* one delay at a time; 16 made kinds; 16 by trust and the 50,000 bytes as in `PLAN_6.md`. Numbers set once here and not chosen. `merge10.py` (a subclass of `merge6.py`'s system; its check counts the cells of the made kinds again by hand).

## 3. Hypothesis, what refutes it, predictions (mine, before any run)

*Hypothesis.* With kinds made by the system, (a) the classes of the two units digits found in a made kind are the carry (at most two pairs of a hundred against their class) on at least four reported seeds of five; (b) never-shown two-digit sums are at least 96% at 40,000 lines at the median of five seeds; (c) no book is more than 0.002 nats above C2. *Refuted if* (b) is under 94, or (c) fails. (a) failing with (b) holding would mean the gain came some other way; I would say so.

| | Statement | Chance |
|---|---|---|
| G1 | development seed: a made kind holds delays 3, 4, 6, 7 and one more, and a pair inside it with X = delays 3, 6 is kept at the last finding | 0.50 |
| G2 | development seed: sums at least 96 (C2: 95.5) | 0.40 |
| G3 | five seeds, median at least 96 (C2: 93.1) | 0.30 |
| G4 | five seeds, median under 94 (refutation of b) | 0.35 |
| G5 | part (a): pure classes in a made kind on at least four seeds | 0.30 |
| G6 | 400,000 lines, two seeds: at least 95 on both | 0.25 |
| G7 | three-digit sums, five seeds: at least 96 (the frozen setting 94.1, C2 93.9) | 0.30 |
| G8 | prose, last twentieth of the stream: not more than 0.002 above C2's 1.5482 | 0.65 |
| G9 | prose: at least 0.005 below it (a run of seven bytes or a run with a gap is worth having) | 0.30 |

## 4. Runs, size, cost

Development material first (two-digit seed 1; prose's last twentieth), for the cost and to see that it does no harm; if sums there are not below C2 by more than 1 point and prose not above it by more than 0.002: the five reported seeds on two-digit sums, 400,000 lines on two seeds, five seeds on three-digit sums, and the first life. The made kinds add up to 16 kinds; on sums a kind of five delays can have up to half a million cells, on prose a made kind can have as many cells as the stream has bytes; all are counted and reported. Cost unknown; my estimate is twice C2. Not run: more than one delay at a time; kinds made from class lines; a second life.

## 5. Addition 1 (20 November; written after the development run on two-digit sums was read, before the runs it describes)

*What was run* (`res/dev10_sums.json`, development seed, 40,000 lines): **95.7** (last reading 96.0; C2 95.5 and 95.8); 64 CPU-seconds; 1,405,441 cells of which 170,553 in five made kinds. The made kinds at the end: delays (2, 3, 4, 6, 7), (3, 4, 6, 7, 8), (1, 3, 4, 6, 7), (1, 3, 4, 7, 9), (3, 4, 7, 8, 9), with largest trusts 0.54, 0.54, 0.42, 0.12, 0.05. So the system does make the kind with the place in it, and trusts it. **But no pair inside a made kind is kept**: none of the made kinds is among the 16 kinds of largest trust (trust taken as the mean over uses), so the search for classes never looks in them. G1 failed; G2 failed narrowly on the mean of the last three readings (95.7).

*One change* (`gc=1`): **the made kinds are candidates for the search for classes beside the 16 kinds by trust.** They were chosen by their worth on the system's own recent bytes, which is as much the system's signal as trust is; and there are at most 16 of them. Whether a pair inside one is kept is still decided by worth among all pairs. (The risk is the one `PLAN_6.md` met with K: more candidates, and the 32 places going to pairs of large worth and little use.)

*Predictions (mine).* H1: development seed, a pair with X = delays 3, 6 inside a made kind is kept at the last finding and its classes have at most two strays (0.5). H2: development seed, sums at least 96 on the mean of the last three readings (0.4); under 94.5, that is harm (0.25). The rule of section 4 for going on to the reported seeds is applied to this form.

## 6. Development outcome of addition 1 (entered 20 November, before any run of this plan on a reported seed or a book; `res/dev10_sums_gc.json`, `res/dev10_prose.json`)

Two-digit sums, development seed, with the made kinds as candidates: **96.9** (last reading 97.4; hundreds, tens, units 97.6, 97.6, 100; C2 95.5); 97 CPU-seconds (69 in the search); 1,433,483 cells. Prose's last twentieth without that change: 1.5470 (C2 1.5482; two made kinds with 5,427 cells); with it: run with the others below. H2 held. H1 failed as written: pairs inside made kinds are kept (inside delays 2, 3, 4, 6, 7: the two tens digits, with the classes by sum modulo ten; inside delays 1, 3, 4, 6, 7 another), but the pair of the two units digits inside a made kind is not among the 32 kept; the units-digits classes in use are still those of the plain kind (53 and 47, two strays). So the gain on this seed does not come by the route I set out in the hypothesis, part (a); see the readings after the seeds.

By the rule of section 4 the reported seeds, the long runs, three-digit sums and the first life are now run with this form, named **M** (C2 with made kinds, the made kinds candidates for the search).

## 7. Outcome on the reported seeds and the first life (entered 20 November; `res/s40_M_seed*.json`, `res/s400_M_seed*.json`, `res/s3_M_seed*.json`, `res/life_M.json`, `res/dev10_prose_gc.json`; `report_s40.txt`, `report_s400.txt`, `report_s3.txt`, `report_life.txt`)

Setting M = C2 with kinds made while reading, the made kinds candidates for the search for classes. Never-shown additions, own answer bytes, whole answer, %, mean of the last three readings; seeds 73129, 93683, 116089, 406364, 545353.

| | no classes | frozen setting | C2 | **M** |
|---|---|---|---|---|
| two-digit, 40,000 lines, each seed | 79.0, 77.9, 77.5, 76.1, 77.2 | 91.8, 93.3, 91.7, 88.0, 92.1 | 92.8, 96.7, 95.1, 89.7, 93.1 | **98.5, 98.0, 96.8, 97.9, 98.5** |
| median (last reading) | 77.5 (77.6) | 91.8 (92.2) | 93.1 (92.8) | **98.0** (98.2) |
| hundreds, tens, units (last, median) | 98.4, 77.6, 100 | 98.0, 92.4, 100 | 98.2, 92.8, 100 | 98.2, 99.4, 100 |
| median at 5,000; 10,000; 20,000 lines | 64.8; 72.4; 77.8 | 86.8; 88.4; 90.0 | 92.6; 94.0; 91.2 | 92.6; 95.8; 97.2 |
| two-digit, 400,000 lines, seeds 73129, 93683 | 62.1, 59.1 | 78.9, 81.3 | 81.3, 98.7 | **99.3, 98.7** |
| three-digit, 80,000 lines, each seed | 92.5, 89.9, 92.7, 90.8, 92.3 | 94.9, 93.0, 94.0, 94.1, 95.0 | 93.9, 92.9, 94.1, 93.2, 95.1 | 96.3, 95.3, 95.7, 95.9, 96.3 |
| median (thousands, hundreds, tens, units at the last) | 92.3 (99.6, 95.2, 98.0, 100) | 94.1 (99.6, 95.6, 99.4, 100) | 93.9 | **95.9** (99.6, 96.0, 100, 100) |
| cells, two-digit at 40,000 lines (median); CPU s | 1,213,003; 19 | 1,231,175; 52 | 1,238,148; 67 | 1,440,880 (176,657 in five made kinds); 99 |

At 400,000 lines M does not fall: seed 73129 reads 98.8, 98.8, 99.6, 99.4 at 40,000, 100,000, 200,000 and 400,000 lines, seed 93683 reads 98.2, 98.8, 99.4, 98.8; no reading after 20,000 lines is under 98. 1.58 million cells, about 510 CPU-seconds a run. Two seeds: a reading.

With only the class kinds and the kinds keyed on two classes speaking (40,000 lines, last reading): the tens digit is named on 100, 100, 100, 100, 99.8% of never-shown sums (whole answer 93.4, 76.2, 94.0, 97.0, 94.4; they do not always hold the hundreds digit).

The kinds the system made on two-digit sums (end of the 40,000-line runs): on four seeds of five delays (2, 3, 4, 6, 7), on all five (3, 4, 6, 7, 8) and (1, 3, 4, 6, 7); beside them (1, 3, 4, 7, 9), (3, 4, 7, 8, 9), (1, 3, 4, 5, 7). Where the tens digit is named, delay 2 holds '=', delay 8 holds the end of the line before, delay 1 holds the hundreds digit: each of the first three is the whole pair together with a byte that says where in the line it is. On three-digit sums: (4, 5, 7, 8, 9), (3, 4, 5, 8, 9), (1, 4, 5, 8, 9) and others of five delays. On prose: delays (1, 3), (1, 4), (1, 5), (1, 6), (1, 3, 4), (1, 2, 4): runs with a gap, 31,606 cells in all.

Prose, nats a byte (the first life; the last twentieth of the stream on development material):

| | development | bryant-stories | burgess-busterbrown | carroll-alice | test text, at the end | cells | CPU s |
|---|---|---|---|---|---|---|---|
| no classes | 1.5566 | 1.5023 | 1.3645 | 1.4456 | 1.5306 | 579,255 | 6 |
| frozen setting | 1.5488 | 1.4920 | 1.3487 | 1.4307 | 1.5264 | 2,961,530 | 200 |
| C2 | 1.5482 | 1.4885 | 1.3349 | 1.4019 | 1.5550 | 3,032,556 | 215 |
| M | 1.5470 | 1.4883 | 1.3371 | 1.3921 | **1.5685** | 2,779,094 | 235 |

**The hypothesis as written: part (b) holds (98.0, at least 96 asked; 94 would have refuted); part (a) fails; part (c) fails narrowly** (the second book is 0.0022 above C2, where 0.002 was allowed; the other two are below it). By its own terms the hypothesis is refuted, by 0.0002 nats on one book; I report it so and report what stands.

| | Statement | Outcome |
|---|---|---|
| G1 | a pair of the two units digits inside a made kind is kept (development seed) | failed |
| G2 | development seed at least 96 | failed for the first form (95.7); held with addition 1 (96.9) |
| G3 | five seeds, median at least 96 | held (98.0; lowest seed 96.8) |
| G4 | five seeds, median under 94 | did not happen |
| G5 | part (a): units-digits classes pure in a made kind on four seeds | failed: the units-digits classes in use are still the plain kind's (90, 98, 98, 90, 90 of 100 with their class) |
| G6 | 400,000 lines at least 95 on both seeds | held (99.3, 98.7) |
| G7 | three-digit sums at least 96 | failed narrowly (95.9) |
| G8 | prose development: not more than 0.002 above C2 | held (0.0012 below) |
| G9 | prose development: at least 0.005 below C2 | failed |

Readings (mine).
1. **With kinds the system makes for itself, never-shown two-digit sums are answered at 98% within 40,000 lines and at 99% at 400,000, with no look-ahead and nothing about sums laid down**; 95.8% at 10,000 lines. The plain cells have 77.5 and fall to 60; the hand-laid look-ahead has 91.4 and falls to 63 without a cap (`nano/sparse`). This is the first system of this line that does not fall with reading, on the two seeds run.
2. **The route is not the one I wrote down, and it is the one the evidence of `PLAN_7.md` and `PLAN_8.md` pointed to.** I expected the carry to come out pure in a kind with the place in it. What the system did: it made kinds that hold the whole pair and a byte that fixes the place; in those it found the classes of the two tens digits by their sum modulo ten; and a cell keyed on that class, the place and the two units digits holds the tens digit exactly. The class kinds alone name the tens digit on 100% of never-shown sums on four seeds and 99.8% on the fifth. The cells that mixed places are no longer what the answer rests on.
3. **Three-digit sums gain 1.8 points over the frozen setting** (95.9; the tens digit is now 100%), and the hundreds digit moves only from 95.6 to 96.0: it still needs two classes of which one is of no use alone, and nothing here seeks that.
4. **On prose M changes little where it was chosen and it harms the old text.** Development material 1.5470; the three books within 0.002 of C2 or below it by more; but the test text's last tenth at the end of the life rises to 1.5685, which is 0.038 above the system with no classes. As with C2, the kinds whose cells start empty at a finding hold recent text, the answering cell comes to lean on them, and what was read first is answered worse. **For keeping on prose, M as built is worse than the frozen setting and worse than no classes.** The cure is known in outline (cells that are kept from one finding to the next, as the made kinds already are, for the kinds keyed on two classes too) and is NOT BUILT.
5. Cost: 99 CPU-seconds for 40,000 lines of two-digit sums (plain cells 19), 1.44 million cells (plain 1.21 million); on three-digit sums 397 CPU-seconds and 2.9 million cells (plain 50 and 2.0 million). The made kinds are large (a kind of five delays has up to 180,000 cells on two digits and 780,000 in all on three), and that is the opposite of a smaller store.

## 8. Addition 2 (20 November; written after section 7 and before its runs): the control by chance, and the second life

(a) **Control: made kinds drawn by chance.** As M, but at each finding the kinds made are as many as worth would make, drawn by chance from all the candidates (R with one more delay, R among the 16 by trust); one made earlier is kept (`gchance=1`). Five reported seeds, two-digit sums, 40,000 lines. If this does as well as M, the worth measured on the system's own recent bytes is not what finds the place. *Prediction (mine):* median at most 95.5, that is at least 2.5 points under M (0.65); not above C2's 93.1 by more than 1 point (0.4).
(b) **M on the second life** (`chesterton-thursday`, `chesterton-brown`, `austen-persuasion`), for the table. *Prediction (mine):* each book within 0.003 of C2 (1.4242, 1.3663, 1.2446) or below it (0.7).

## 9. Outcome of addition 2 (entered 20 November; `res/s40_Mchance_seed*.json`, `res/life2_M.json`)

(a) **Made kinds drawn by chance**, two-digit sums, 40,000 lines, seeds 73129, 93683, 116089, 406364, 545353, mean of the last three readings: 92.5, 98.1, 95.7, 90.3, 93.9; median **93.9** (M: 98.5, 98.0, 96.8, 97.9, 98.5, median 98.0; C2, which has no made kinds: 93.1). Tens digit at the last reading, median: 94.2 (M: 99.4). The same number of made kinds and of cells as M (1,440,021 cells; 96 CPU-seconds). Both predictions held. The one seed where chance does as well as M (93683, 98.1) is the one where the draw happened to include delays (2, 3, 4, 6, 7), the whole pair with the '=' sign; on seed 116089 it drew (3, 4, 6, 7, 8) and has 95.7. **Kinds made by chance give what no made kinds give; the worth measured on the system's own recent bytes is what finds the kinds that matter.**

(b) **M on the second life**, nats a byte: chesterton-thursday 1.4253, chesterton-brown 1.3673, austen-persuasion 1.2459 (C2: 1.4242, 1.3663, 1.2446; the frozen setting: 1.4263, 1.3856, 1.2594; no classes: 1.4411, 1.3996, 1.2700). Each within 0.0013 of C2: the prediction held. On prose M made two kinds with 5,851 cells in this life; it is C2 there, to the third place. 245 CPU-seconds, 2,994,942 cells.

## 10. Addition 3 (20 November; written before its runs): which part harms the old text on prose

The test text's last tenth at the end of the first life: no classes 1.5306, frozen setting 1.5264, C2 1.5550, M 1.5685. My reading has been that the kinds keyed on two classes do it, because their cells start empty at every finding. Two runs of the first life to test that: (x) M without the kinds keyed on two classes (`P2=0`: classes and made kinds only); (y) the frozen setting with only the repair of `PLAN_6.md` (`keep=2`; no kinds keyed on two classes, no made kinds). Each scores the test text's last tenth once, at the end.

*Predictions (mine).* (y) within 0.002 of the frozen setting's 1.5264 (0.8). (x) at most 1.535, that is, most of the harm goes when the kinds keyed on two classes go (0.6); if (x) stays above 1.55 the made kinds are the cause and my reading is wrong.

*A slip in arm (x), stated (entered before the rerun).* With `P2=0` the program also stopped keeping its last 50,000 bytes (the buffer was kept only where kinds keyed on two classes are made), so no kind could be made and the run reproduced the frozen setting to the last place (1.4920, 1.3487, 1.4307; 1.5264; `res/life_Mx_no_buffer.json`). It did not test what the addition says. The program is corrected (the bytes are kept wherever kinds are made; the check passes; nothing run before is touched, since every earlier run of `merge10.py` had P2 = 32) and arm (x) is run again; predictions unchanged. Arm (y): 1.5264 at the end, equal to the frozen setting (books 1.4920, 1.3487, 1.4307): its prediction held.

*Outcome of arm (x), run again* (`res/life_Mx.json`): M without the kinds keyed on two classes: books 1.4922, 1.3391, 1.3983; the test text at the end **1.5573**. The prediction failed (at most 1.535 was predicted), and by what I wrote beforehand **my reading was wrong, or half of it**: the kinds keyed on two classes are not the only cause (C2 alone has 1.5550); the made kinds alone do the same harm. Four of the six made kinds at the end of that life were made at 1,280,000 bytes, inside the books, and hold nothing of the test text. The cause common to both: **a kind that is born late holds only what has been read since its birth**, the answering cell learns on the current text to trust it, and the old text is then answered from it. `PLAN_11.md` follows from this.
