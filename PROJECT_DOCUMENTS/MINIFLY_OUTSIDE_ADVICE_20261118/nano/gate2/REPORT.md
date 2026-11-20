# Line `nano/gate2`: the second answering cell repeated from its description, and taken to long reading, a second task and three-digit sums. Report of 20 November 2026

By a separate session started by the coordinating session (session_017dSEpVed7SthEH3ngGAdNc); the frame is `../../ROADMAP_SECOND_REVISION_20261120.md`. **Authorisation**, as quoted to this session by the coordinating session: Mr. Xu (in Chinese): "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and, most recently: "继续大胆尝试。" (the coordinating session's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." and "Go on trying boldly."). **Status:** development level. Five reported seeds (73129, 93683, 116089, 406364, 545353) for most of what is said, five further seeds for the repeat, two or three seeds where marked; nothing here has been repeated by anyone else; no checker gate, seal or review is claimed. The mechanism and the questions of steps 1 to 4 are the coordinating session's (`../table/PLAN_15.md`); the programs, the further plans (the diagnostics of the second plan, the fifth and the sixth plan) and every reading marked "mine" are this session's. **Source:** `claude/friendly-darwin-8mmhfm` at a859b2f224f0adca576616c0fc96bf82699ec293; branch `claude/nano-gate2-8mmhfm`; files changed only in this folder.

Every step has a plan file committed before its runs (`PLAN_1.md` to `PLAN_6.md`), with its outcome and every prediction scored below it. All figures are never-shown sums, whole answer right, %, each answering cell continuing its own answer bytes, scored after a read line unless said (`loop_small_case/ERRATA_20261120.md`, section 3); subtraction on the clean pairs (section 1). Sizes: 1.21 to 1.24 million cells that count on two-digit sums (1.38 to 1.40 million after a second task; 2.03 million on three-digit sums); the second cell holds 1,451,520 parts with classes, of which 700,000 to 800,000 are ever used. CPU, one core: 80 s for 40,000 lines with classes, 7 to 8 minutes for 400,000 lines, 3.5 minutes for 80,000 lines of three-digit sums, 11.5 minutes for the long life of two tasks.

## 1. In brief

1. **The result repeats, exactly** (`PLAN_1.md`). Written from the description in `PLAN_15.md` and not from its program, the second answering cell gives 97.4 against 92.9 for the first cell on the five reported seeds with classes (control 92.8), and 96.2 against 90.8 on five seeds no line had used. When I then opened the first program's result files, all 20 that have a counterpart here were the same at every reading to the last digit.
2. **It transfers to three-digit sums with nothing chosen there** (`PLAN_4.md`): 96.9 against 94.1, on five seeds of five; the control gives nothing.
3. **Through a second task it keeps more than the first cell and learns the new task better, over a short life; over a long life it can lose the first task and not recover it** (`PLAN_3.md`). Addition 40,000 lines, then subtraction 20,000: it keeps 88.8% of never-shown addition with classes (the first cell 81.1%) and 95.8% without (96.1%), and answers 89.4% of clean never-shown subtractions (80.4%). The coordinating session's hypothesis (at least 90% kept; refuted under 75%) is neither held nor refuted there. Addition 400,000 lines, then subtraction 200,000, two seeds: it keeps 18% and 78%; on one seed it is refuted.
4. **It does not hold its level over a long reading in the frozen setting, and the cause of the large falls is not in the second cell** (`PLAN_2.md`). At 400,000 lines it ends at 76.5 at the median of five seeds (first cell 78.8), from 97.4. The falls are steps at the moments classes are found anew, and at each of them the base system's search has let go of the one kind of cell its best classes are found in. With that line's own repair of where it searches (its setting C, unchanged) the second cell has no step falls and ends at 93.0, 90.0 and 95.1 on three seeds, 10 to 12 points above the first cell.
5. **A slower, even loss with long reading remains and is the base system's**: without classes both cells lose 17 to 18 points together and the second keeps its margin of 6 to 7.
6. **One new mechanism of mine was tried and is refuted** (`PLAN_5.md`): a second cell that hears each sender with probability one half while it learns. It is worse everywhere it was run.
7. **A measurement** (`PLAN_6.md`): see section 7.

## 2. The system, and what was written

`gate2.py` imports `../merge/merge.py` as it is (cells that count, a kind of cell for every set of at most four of nine delays, an answering cell with a learned trust for each kind and level of counts, classes of cells found by search at that line's frozen setting) and adds, beside its answering cell, second answering cells that the first never hears:

- **the second cell:** its branch for a byte value is the first cell's branch plus, for every speaking sender, (the sum over the nine delays of a part indexed by kind, level and the byte at that delay) times the sender's message. Parts are born at nothing and change by the trust's own step rule times the second cell's own mismatch. Across a finding of classes the parts of a class kind follow the first cell's trust for it.
- **the control:** the same with every part reading a constant in place of the byte.

Both run in one run from the same first cell. Checks (`check_output.txt`, `check3_output.txt`): the compiled loop against a numpy reference (5e-16); the first cell of my loop bit-equal to `merge.py`'s compiled loop; my runner's first cell equal to `merge.run_sums` reading by reading; and in the runs, my first cell equal at every reading to that line's stored files on all five seeds for two-digit sums with and without classes and for three-digit sums.

## 3. Step 1: the repeat (`PLAN_1.md`; `report_1.txt`)

Never-shown two-digit additions, 40,000 lines, last reading; medians of five seeds (mean of the last three readings in brackets).

| Setting, seeds | First cell | Second cell | Control |
|---|---|---|---|
| with classes, five reported seeds | 92.9 (92.9) | **97.4** (97.1) | 92.8 (92.5) |
| without classes, the same seeds | 80.1 (79.4) | **85.6** (85.7) | 83.4 (83.6) |
| with classes, seeds 3001 to 3005 | 90.8 (90.3) | **96.2** (96.0) | 91.2 (90.1) |

The second cell is above the first on ten seeds of ten with classes (by 4.0 to 6.7). The whole difference is at the tens digit (93.0 to 97.6% right). Without classes more than half of the gain is not from the bytes (the control gives 3.3 of 5.6). I opened `../table/gate.py` and the result files (from the coordinating session's branch at 681f168) at 20:25 UTC, after my fifteen runs were tabulated and pushed; `PLAN_1.md`, section 8, lists the differences between the programs, none of which changes a figure. A limit: this is a repeat on the same base system, text, split and random draws; that it is not an accident of one program is shown, and that it holds on other seeds; nothing more.

## 4. Step 4: three-digit sums (`PLAN_4.md`; `report_4.txt`)

80,000 lines, with classes, five reported seeds, mean of the last three readings: first cell 94.1 (94.9, 93.0, 94.0, 94.1, 95.0); second cell **96.9** (97.1, 96.9, 96.1, 96.5, 97.7); control 93.5. By place at the end (thousands, hundreds, tens, units): first 99.6, 95.6, 99.4, 100; second 99.6, 97.0, 100, 100. My one prediction (at least 1.5 points above the first cell, chance 0.60) held. What remains is at the hundreds digit, whose rule needs more than a kind of four delays holds: a loss in what the cells hold.

## 5. Step 3: a second task (`PLAN_3.md`; `report_3.txt`, `report_3long.txt`)

*The short life* (addition 40,000 lines, then subtraction 20,000 with nothing of addition shown; five seeds; medians). "Kept": never-shown additions after a read subtraction line at the end of the life over the same after a read addition line at the end of the first task (mean of the last three readings of each).

| Setting | Cell | Addition before | at the end | Kept, % (five seeds) | Lowest reading after the change | Clean subtractions at the end |
|---|---|---|---|---|---|---|
| with classes | first | 92.9 | 75.4 | 81.1 (87.7, 83.5, 79.3, 78.9, 81.1) | 50.2 | 80.4 |
| | second | 97.1 | 86.2 | **88.8** (81.7, 89.4, 91.0, 87.2, 88.8) | 63.2 | **89.4** |
| | control | 92.5 | 61.3 | 65.8 | 18.6 | 83.1 |
| without classes | first | 79.4 | 76.3 | 96.1 | 74.3 | 80.4 |
| | second | 85.7 | 82.6 | **95.8** | 73.3 | **87.2** |
| | control | 83.6 | 71.6 | 83.4 | 66.8 | 83.1 |

The loss is sudden and mends while only subtraction is read: addition is lowest 1,000 lines after the change and climbs for the rest of the second task (second cell, with classes: 63.2, 74.8, 76.1, 83.7, 87.6, 88.1 at 1,000 to 20,000 lines). The cells that hold addition are untouched; the trusts and parts that weigh them are shared with the new task and move at once. The control, whose nine equal parts make its step nine times the first cell's, falls to 19%.

*The long life* (addition 400,000 lines, then subtraction 200,000; with classes; seeds 73129 and 93683): the first cell keeps 93.8% and 87.0% (78.8 to 73.9; 82.9 to 72.1); the second cell **18.2% and 77.7%** (76.5 to 13.9; 91.9 to 71.4). On seed 73129 all three cells fall to 1 to 17% within 1,000 lines of the change; the first cell and the control are back near 80 by 40,000 lines and the second cell stays at 12 to 24% to the end. Clean subtractions at the end: second cell 78.5 and 77.2, first cell 68.9 and 60.5. On that seed the second cell had already fallen to the first cell's level before the change (section 6), so these two runs do not say whether the lasting loss comes from the state the long reading left or from the change alone.

## 6. Step 2: long reading (`PLAN_2.md`; `report_2.txt`, `report_2d.txt`, `report_2none.txt`, `report_2C.txt`, `report_1C.txt`)

Never-shown additions; the reading at 40,000 lines, then the mean of the last three readings at 400,000 lines.

| Setting | Seed | First cell | Second cell | Control |
|---|---|---|---|---|
| frozen setting, with classes | 73129 | 93.2, 78.8 | 97.4, **76.5** | 92.2, 83.6 |
| | 93683 | 95.2, 82.9 | 98.2, **91.9** | 93.8, 85.4 |
| | 116089 | 92.4, 77.9 | 96.4, **75.4** | 92.8, 84.9 |
| | 406364 | 89.6, 75.5 | 96.8, **74.7** | 88.5, 81.7 |
| | 545353 | 92.9, 83.4 | 97.4, **92.5** | 93.6, 84.4 |
| without classes | 116089 | 80.1, 62.3 | 86.3, **68.1** | 83.8, 66.7 |
| | 406364 | 78.6, 61.8 | 84.6, **69.0** | 80.0, 65.5 |
| | 545353 | 77.8, 62.4 | 85.6, **69.1** | 82.5, 63.5 |
| `nano/merge`'s setting C, with classes | 116089 | 92.0, 82.8 | 96.2, **93.0** | 91.9, 86.6 |
| | 406364 | 89.0, 78.3 | 97.0, **90.0** | 88.8, 82.7 |
| | 545353 | 91.8, 83.4 | 96.6, **95.1** | 92.5, 84.9 |

(The frozen setting's runs on seeds 73129 and 93683 equal the coordinating session's two at every reading.)

*Where it falls* (the second plan's first addition; readings 1,000 lines on either side of each finding of classes, and the same scores with only the class kinds or only the plain kinds speaking; no mechanism changed). In the frozen setting the second cell falls in steps: across the finding at 256,000 lines it goes from 89.9 to 77.8 and from 91.2 to 80.2 within 2,000 lines (seeds 116089, 406364) and regains nothing in the 143,000 lines after. I had supposed that its many parts could not follow class cells made anew; that was wrong, because the first cell's reading of the class kinds alone falls as far at the same moment (87.4 to 71.6; 91.3 to 73.2). At that finding, on both seeds, every pair of the kind of delays (3, 4, 6, 7) is dropped from the kept list: the kind that holds the two digits of both operands when the tens digit is named, in which the classes of the carry and of the sum modulo ten are found. `merge.py` searches the sixteen kinds of largest trust, and a kind whose classes succeed loses its own trust to them; that line's report names the fault and carries two repairs as options, neither in its frozen setting. While that kind is kept, the second cell stays 6 to 10 points above the first.

*With the repair* (that line's setting C, `keep=2`, used as it stands): the kind keeps six pairs at every finding from 32,000 lines on, and the second cell ends at 93.0, 90.0, 95.1: 1.5 to 7.0 points under its level at 40,000 lines and 10.2 to 11.7 above the first cell. At 40,000 lines setting C costs the second cell half a point (96.9 against 97.4, five seeds). That line found setting C worth nothing to its own answering cell over the long reading (80.1 and 81.3 against 78.9 and 81.3 on two seeds, old score).

*The even loss.* Without classes nothing is found anew, and both cells fall evenly, together, by 17 to 18 points; the second keeps its margin (+5.8, +7.2, +6.7 at the end against +6.2, +6.0, +7.8 at 40,000 lines). In the runs with classes the first cell's reading of the plain kinds alone falls from 79% to between 7 and 21%, and from 200,000 lines on the first cell with all kinds speaking is below the class kinds alone.

**So** (mine): what the second cell adds stands through the even loss and was lost at the base's search fault; with that fault repaired by the base line's own option the second cell is, on three seeds, the first counting system of this programme reported at 90 or more after 400,000 lines (the others stand at 79 to 83; the relay system at 95.5). It is three seeds and one setting that is not the frozen one.

## 7. The fifth and sixth plans

FILLED_BELOW

## 8. Predictions, all of them

`PLAN_1.md`: nine of nine held (I had been told the first figures). `PLAN_2.md`: two of six held (L4, L6; the hypothesis L1 refuted); its first addition: four of six (D1 to D4 held; my account of the cause was wrong though the statement tested was not refuted); its second addition: five of five. `PLAN_3.md`: five of nine (K2, K5, K6, K8, K9 held; K1 failed narrowly; K3, K4, K7 failed). `PLAN_4.md`: the one prediction held; of two beside it one held. PLAN_5_SCORES PLAN_6_SCORES

## 9. What is not shown, and limits

- Nothing here is on prose. The coordinating session reports the second cell's gain on prose; this line neither repeated nor extended that.
- The repeat shares the base system, the text, the split and the random draws with the first program.
- The long reading with setting C is three seeds; the long life of two tasks is two seeds, in the frozen setting only, where the second cell had already been brought down on one of them.
- The even loss with long reading, the fall of every cell right after a change of task, and the weak keeping of the first cell with classes (81%) are faults of the base system that this line measured and did not cure.
- Why hearing half the senders harms is given in `PLAN_5.md` as my account; it was not tested.

## 10. What I would try next, and why (mine; nothing here is started)

1. **Make setting C (or a better rule for where to search) the base under the second cell, and redo this line's steps on it with five seeds**: the long reading on all five; the long life of two tasks (does the lasting loss of seed 73129 go away when the second cell has not been brought down before the change?); three-digit sums. Reason: the largest single effect found here is not in the answering cells but in the search letting go of the kind its best classes come from.
2. **A search that cannot forget what the answering cells use.** Setting C keeps a kind by adding its children's trust to its own; the second cell's parts are a second, finer record of which class kinds are used at which place, and could be what the search asks. Reason: the fault was found by watching the second cell, and the second cell knows something the first cell's trust does not.
3. **The even loss**: section 7's measurement says where to look.
4. **The upset after a change of task.** Every cell falls within 1,000 lines of a change though no cell that holds the old task is touched, and it mends by itself. The parts indexed by the operator's byte are the only ones a change of operator leaves alone, and they are one part in nine; a second cell whose parts are indexed by pairs of delays, or a third cell over the second with parts indexed by the byte the second cell leans on, would let the system hold a separate reading for each task without being told there are tasks. This is a new mechanism and needs its plan; I give it one chance in three.
5. **Prose** for whatever of the above holds on sums, with the same program and numbers.
