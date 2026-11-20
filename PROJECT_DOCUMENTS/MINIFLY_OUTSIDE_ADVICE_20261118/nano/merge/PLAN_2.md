# Line `nano/merge`, second plan: the frozen setting on five seeds of sums and on the life on prose, with the controls (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and then "继续大胆假设，基于事实努力尝试。" (the coordinating session's translation is in `PLAN_1.md`). **Source.** `claude/nano-merge-8mmhfm` at the commit that holds this file; base `claude/friendly-darwin-8mmhfm` at 71172671f. Written before any run on a reported seed, on a book, or on the last tenth of the test text. Development level; no gate, seal or review is claimed.

## 1. The system and the setting, frozen

`merge.py` as `PLAN_1.md` describes it, with the numbers chosen in step 1 on development seed 1 and on the last twentieth of the test text's stream: classes beside the cells, at most P = 32 kept pairs, e_J = 0.02; 16 kinds by trust as candidates, X of at most two delays, findings when the bytes read have doubled (first at 10,000), two sweeps. The base on each task is its comparator's (sums: 255 kinds of at most four of nine delays, step 0.1 shrinking, e = 2, born 0.02; prose: runs of 0 to 6 bytes, step 0.00005, e = 0.02, born 0.25). The same class mechanism and numbers on both tasks.

## 2. Runs

Sums, five reported seeds (73129, 93683, 116089, 406364, 545353), 40,000 lines, readings every 5,000, never-shown additions with the system's own answer bytes:
- `none`: no classes. `cls`: the frozen setting. `chance`: at every finding the classes of each kept pair are replaced by classes of the same sizes with members drawn by chance. `half`: classes found once, at 20,000 lines, their cells starting empty and counting only the second half. `rep`: replacement (not the frozen setting; reported beside it because step 1 showed it higher on sums and worse on prose).
- 400,000 lines, `none` and `cls`, on two seeds (73129, 93683) first; the other three if the cost allows. Two seeds are a reading and not a claim.

Prose, the life (the stream of the test text, then the three books; each book's last tenth scored right after its stream, nothing changing during a scoring): `none`, `cls` (with the last tenth of the test text scored once, at the end of the life), `chance`, `half` (classes found once at 500,000 bytes), `rep`. The number of cells at every 125,000 bytes.

## 3. Hypothesis, what refutes it, predictions

The hypothesis, its refutation and the predictions M1 to M11 are those of `PLAN_1.md`, section 3, unchanged, with my chances as written there before any run. After step 1 (one seed; these are new and written now): 

| | Statement | Chance |
|---|---|---|
| N1 | sums `cls`, median of five seeds, last three readings: at least 88 | 0.70 |
| N2 | sums `rep` above `cls` at the median | 0.70 |
| N3 | each book: `cls` below `none` by at least 0.005 nats (this is M7; my chance for it is now higher than the 0.20 written before step 1) | 0.50 |
| N4 | prose `chance`: no book better than `none` by more than 0.002 | 0.70 |
| N5 | prose `half`: at least half of `cls`'s gain on the mean of the three books | 0.40 |
| N6 | the last tenth of the test text at the end of the life: `cls` below `none` | 0.65 |
| N7 | number of cells on prose: with classes it grows with a power of the bytes read that is not below the base's | 0.80 |

## 4. Size and cost

Sums: about 1.2 million cells, 50 to 60 CPU-seconds a run at 40,000 lines with classes (18 without); at 400,000 lines my estimate is ten CPU-minutes. Prose: 0.45 million cells without classes and about 2.3 million with, at the end of the stream; a life about four CPU-minutes. Four cores, three or four runs at a time.

Not run: subtraction and keeping; three-digit sums; the word cells as base; more than 32 pairs; X of more than two delays; classes of classes; a change to the answering cell.

## 5. Outcome (entered 20 November after the runs; `res/s40_*.json`, `res/s400_*.json`, `res/life_*.json`; tables `report_s40.txt`, `report_s400.txt`, `report_life.txt`)

**Sums**, never-shown additions, the system's own answer bytes, whole answer, %, 40,000 lines, seeds 73129, 93683, 116089, 406364, 545353; mean of the last three readings; in brackets the last reading.

| Arm | each seed | median | median, last | hundreds, tens, units (last, median) | cells (median) | CPU s |
|---|---|---|---|---|---|---|
| no classes | 79.0, 77.9, 77.5, 76.1, 77.2 | 77.5 | 77.6 | 98.4, 77.6, 100 | 1,213,003 | 19 |
| **classes, the frozen setting** | 91.8, 93.3, 91.7, 88.0, 92.1 | **91.8** | 92.2 | 98.0, 92.4, 100 | 1,231,175 (18,596 class cells) | 52 (30 in the search) |
| classes by chance, the same sizes | 80.9, 78.3, 78.1, 77.3, 77.9 | 78.1 | 78.8 | 98.4, 78.8, 100 | 1,249,586 | 64 |
| classes found once at 20,000 lines, cells counting the second half only | 87.6, 89.5, 89.3, 87.5, 88.1 | 88.1 | 89.6 | 98.2, 89.6, 100 | 1,238,417 | 32 |
| replacement (not the frozen setting) | 93.1, 94.9, 92.5, 91.5, 93.1 | 93.1 | 93.0 | 98.2, 93.0, 100 | 1,230,553 | 55 |

Beside them, as the other lines report: look-ahead 91.4; the relay system 66 at 40,000 lines and 84 at 400,000.

400,000 lines, two seeds (73129, 93683), a reading and not a claim: without classes 62.1 and 59.1 (last three readings; 79 at 40,000 lines); with classes 78.9 and 81.3 (92 and 94 at 40,000 lines). Both fall with reading, as `nano/sparse` found for the plain cells; the classes stay 17 to 22 points above. 239 CPU-seconds with classes, 175 without. The other three seeds: NOT RUN.

*What the classes are* (last finding; from the result files). The pair inside the kind of delays 3, 4, 6, 7 with X = delays 3, 6 (where the tens digit is named: the two units digits against the two tens digits) is kept on all five seeds. Its classes of pairs of digits: 53 and 47 on two seeds (98 of 100 pairs lie with their class by whether they sum to ten or more), 45 and 55 on three seeds (90 of 100). The pairs that lie against their class are in every case pairs that sum to nine (`PLAN_4.md`, note on R1). With X = delays 4, 7 (the two tens digits) the classes are, on every seed, the ten sets of pairs with the same sum modulo ten (nine whole and the tenth split 8 and 2, or 6 and 4). With classes by chance the same pairs have 54 to 67 of 100 with their class.

**Prose**, the life (the test text's stream, then the three books; nats a byte, each book's last tenth right after its stream; the test text's last tenth once, at the end):

| Arm | bryant-stories | burgess-busterbrown | carroll-alice | test text, at the end | cells at the end | CPU s |
|---|---|---|---|---|---|---|
| no classes | 1.5023 | 1.3645 | 1.4456 | 1.5306 | 579,255 | 6 |
| **classes, the frozen setting** | **1.4920** | **1.3487** | **1.4307** | 1.5264 | 2,961,530 | 200 (134 in the search) |
| classes by chance, the same sizes | 1.5003 | 1.3619 | 1.4428 | not scored | 3,398,416 | 202 |
| classes found once at 500,000 bytes, cells empty | 1.4865 | 1.3336 | 1.4244 | not scored | 2,488,554 | 68 |
| replacement (not the frozen setting) | 1.5031 | 1.3554 | 1.4511 | not scored | 2,961,530 | 194 |

The base equals `nano/table`'s T0 to the fourth place (1.5023, 1.3645, 1.4456). Beside: T2 with word cells 1.4802, 1.3162, 1.4109 (that line's). The frozen setting is below the base by 0.0103, 0.0157, 0.0149; classes by chance by 0.0020, 0.0025, 0.0028.

*The store.* Cells that have counted, against bytes read (fitted from 125,000 to 1,000,000 bytes of the test text): without classes the number grows as the bytes read to the power 0.60 (this base has runs only; `growth.py`'s 0.80 includes the word cells); with classes to the power 0.48, with 5.1 times as many cells at the end of the life (6.7 times at 125,000 bytes). Leaving out the R-cells whose X-cell is in a class still leaves 4.1 times the base. Scored with those cells silent, the answering cell not having learned so: 2.5 to 2.7 nats. On sums the class cells are 1.5% of the store.

**The hypothesis, on the frozen setting.** (i) **holds**: 91.8% at the median with no look-ahead (at least 90 asked; under 83 would refute; lowest seed 88.0). (ii) **holds**: each of the three books is at least 0.010 nats below the same system without classes (0.005 asked). (iii) **fails**: there are more cells, not fewer, on both tasks, and on prose withdrawal harms. One setting does (i) and does no harm on prose, so the refutation "no single setting" is not met.

| | Statement | Outcome |
|---|---|---|
| M1 | sums at least 90 | held (91.8) |
| M2 | sums under 83 | did not happen |
| M3 | the units-digits pair kept on at least four seeds and every class pure for the carry | failed: kept on five, pure on none (98, 98, 90, 90, 90% with their class) |
| M4 | classes by chance within 3 points of no classes | held (78.1 against 77.5) |
| M5 | found at half, counting the second half: at least 85 | held (88.1) |
| M6 | 400,000 lines at least 90 | failed (78.9 and 81.3 on two seeds) |
| M7 | each book at least 0.005 below the base | held (0.010, 0.016, 0.015); I had given it 0.20 |
| M8 | no book more than 0.002 above the base | held |
| M9 | with the R-cells of merged classes withdrawn: fewer cells and the score not lower on sums | failed as stated: scored with those cells silent the frozen setting has 91.6, 95.2, 93.8, 89.0, 94.0 (not lower on four seeds of five), but the cells are 2% fewer than the base, not a store of another size |
| M10 | the same on prose | failed (2.5 nats) |
| M11 | the whole hypothesis | failed, by part (iii) |
| N1 | sums at least 88 | held |
| N2 | replacement above the frozen setting on sums | held (93.1 against 91.8) |
| N3 | each book at least 0.005 below | held |
| N4 | classes by chance: no book better than the base by more than 0.002 | failed narrowly (0.0020, 0.0025, 0.0028) |
| N5 | found once at half: at least half of the frozen setting's gain | held, and more: it gains more than the frozen setting on every book (0.016, 0.031, 0.021) |
| N6 | the test text's last tenth at the end: classes below no classes | held (1.5264 against 1.5306) |
| N7 | the number of cells grows with a power not below the base's | failed (0.48 against 0.60); the number itself is five times the base's |

Readings (mine; development level).
1. **Cells that are refined alike can be found by the system, and what it finds on sums is the arithmetic**: the carry (up to the pairs that sum to nine) and the sum modulo ten. Nothing about sums was laid down, and classes by chance do nothing.
2. **Cells keyed on the class answer combinations never shown**: 92% without look-ahead, level with the look-ahead device (91.4), at half its cost in cells (1.2 million against 3.3 million).
3. **The same setting gains on prose**, 0.010 to 0.016 nats on three books; a fifth of that with classes by chance. It is about a third of what the word cells gain (0.02 to 0.05), with nothing about words laid down.
4. **The store is not smaller.** The class cells are added; what they stand for cannot be withdrawn on prose.
5. **Classes found on one part of the stream serve on another** (both controls). On the books they serve better than classes found anew at every doubling; I did not expect that (`PLAN_4.md`).
