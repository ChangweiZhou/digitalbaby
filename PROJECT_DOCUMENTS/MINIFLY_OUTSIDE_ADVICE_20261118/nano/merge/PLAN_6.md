# Line `nano/merge`, sixth plan: the class used as a line; cells keyed on two classes (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and then "继续大胆假设，基于事实努力尝试。" (the coordinating session's translation is in `PLAN_1.md`). **Source.** `claude/nano-merge-8mmhfm` at the commit that holds this file. Development level; no gate, seal or review is claimed.

**A slip, stated.** While testing the program I ran it once on development seed 1 for 6,000 lines of two-digit sums (P = 32) to see why its check found no kinds, and looked at the result: 92.0% at 6,000 lines, and among the kinds made, the one keyed on the class of the two tens digits and the class of the two units digits. That was before this file was committed. Nothing was chosen on it; the numbers below were written into the program before it. The first run on three-digit sums with classes (seed 73129, `PLAN_5.md`: 94.9 against 92.5) had also been seen.

## 1. The evidence this follows

1. The system finds, inside the kind of the whole pair, the classes of the two units digits (the carry) and the classes of the two tens digits (their sum modulo ten). Each is used with the *bytes* of the other pair: a cell keyed on (carry class, the two tens digits) or on (class modulo ten, the two units digits). The regularity itself is smaller: the tens digit is fixed by the two classes together, thirty cells or so.
2. `PLAN_5.md`, my expectation: on three-digit sums the hundreds digit needs the class of the tens digits and the class of the units digits together with the two hundreds digits: a cell keyed on two classes, which the system as built cannot make.
3. The hypothesis this line was given says "other cells should be keyed on the class instead of on the cell". So far only one kind was: the kind the class was found in.

## 2. The one addition

The class of an X-cell is a line: at every byte it has a value (the class of the X-cell that is active, or none). **For two kept pairs p and q whose X's share no delay, a candidate kind is keyed on (the class of p's X-cell, the class of q's X-cell, the delays of p's Z that are not in q's X).** Its cells count the next byte like any cell; trusts born at nothing; cells start empty at each finding (the classes are made anew there).

*Which candidates are made* is decided on the system's own last W = 50,000 bytes, which it keeps: a candidate's worth is the sum over those bytes of the log of what the candidate's cell says of the byte with that byte left out of the cell, less the same for the better of the two class kinds it is made from, on the same bytes. The candidates are the pairs (p, q) with p among the 16 class kinds of largest trust and q any kept pair. The P2 = 32 of largest positive worth are made. A kind that is made again at the next finding keeps its trusts.

*Laid down, and the cost in generality:* two classes and no more; p's Z as the rest; 16 by trust; the 50,000 bytes kept; 32 kinds. These numbers were set once, here, and are not chosen on anything. `merge6.py` (a subclass of `merge.py`'s system; `check6_output.txt`).

## 3. Hypothesis, what refutes it, predictions (mine)

*Hypothesis.* With the same numbers on every task, cells keyed on two classes (a) lift three-digit never-shown sums by at least 2 points over the frozen setting, (b) do not lower two-digit sums by more than 1 point, (c) do not raise the loss on any book by more than 0.002 nats. *Refuted if* (a) gains under 1 point, or (b) or (c) fails.

| | Statement | Chance |
|---|---|---|
| V1 | three-digit sums, five seeds, median: at least 2 points above the frozen setting | 0.40 |
| V2 | three-digit: the hundreds digit at least 98% at the last reading (median) | 0.35 |
| V3 | two-digit sums, five seeds, median: at least the frozen setting's 91.8 | 0.60 |
| V4 | two-digit: at least 94 | 0.30 |
| V5 | two-digit: the kinds keyed on two classes alone answer at least 85% whole with under 3,000 cells (median) | 0.35 |
| V6 | prose, each of the three books of the first life: not more than 0.002 above the frozen setting | 0.65 |
| V7 | prose: below the frozen setting by at least 0.002 on at least two books | 0.25 |
| V8 | the whole hypothesis | 0.25 |

## 4. Runs, size, cost

First, development material (sums seed 1 at 40,000 lines; prose's last twentieth of the stream; three-digit seed 1), to see that it runs and what it costs; nothing is chosen on it, and if it harms prose there by more than 0.002 nats the runs on the books are still made, as the hypothesis asks. Then the five reported seeds on two-digit and on three-digit sums, and the first life (three books). Kinds: at most 32 more; their cells are counted and reported beside the others. Cost by estimate: a third more than the frozen setting. Not run: three classes; a class of cells that are themselves keyed on a class; the second life (run only if the first life shows a difference of 0.002 either way).

## 5. Addition 1 (20 November; written after the first development run on two-digit sums was read, before the runs it describes)

*What was run* (`res/dev6_sums.json`, development seed 1, two-digit sums): never-shown sums at each reading 92.4, 95.2, 95.0, 95.2, 95.6, 96.0 (5,000 to 30,000 lines), then 76.4 and 75.8 (35,000 and 40,000). The frozen setting on this seed: 86.0 to 91.6, ending at 90.6.

*What happened* (from the result file). Up to 30,000 lines the kinds keyed on two classes do what was hoped: 95 to 96% from 10,000 lines on. At the finding at 320,000 bytes (32,000 lines) the kind of the whole pair, delays (3, 4, 6, 7), is no longer among the kinds whose pairs are kept. It was not searched: the candidates for the search are the 16 kinds of largest trust, and by then the answering cell had moved its trust from that kind to the kinds made from its classes. Its classes were therefore not made again, the kinds keyed on them lost their lines, and the answer fell below the plain system. **The rule for candidates (`PLAN_1.md`, section 2, point 4) forgets a kind as soon as its classes succeed.** The same thing is in the 400,000-line run of the frozen setting on seed 73129: at the finding at 2,560,000 bytes the kind (3, 4, 6, 7) is absent for the first time.

*The repair* (`keep=1` in `merge.py`): the candidates are the 16 kinds of largest trust **and every kind that has a kept pair now**. A kind whose classes are in use is searched again; whether its pairs are kept is still decided by their worth. This is a repair to the first mechanism, found on development material; it changes the frozen setting, so it gets its own name: **setting K** = the frozen setting with `keep=1`; **setting K2** = setting K with the kinds keyed on two classes.

*Runs.* Development material first (two-digit seed 1; prose's last twentieth; three-digit seed 1) for K and K2. Then on the reported seeds and the books, section 4's runs for K2, with K beside it on two-digit sums at 40,000 lines; and K and K2 at 400,000 lines on the two seeds 73129 and 93683. The hypothesis of section 3 is scored on K2 against the frozen setting as written.

*Predictions (mine).* W1: K2 on the development seed ends at 94 or more (0.65). W2: K alone is within 1.5 points of the frozen setting at 40,000 lines on the development seed (0.75). W3: at 400,000 lines K is above the frozen setting on both seeds (0.6), and K2 is at least 85 on both (0.35; the fall of the plain cells with reading is still there).

## 6. Addition 2 (20 November; written after the two development runs of addition 1 on two-digit sums were read, before the runs it describes)

*What was run* (development seed 1, two-digit sums, 40,000 lines; `res/dev6_sums_K.json`, `res/dev6_sums_K2.json`): **K2 94.9** (last three readings; 91.4 at 5,000 lines, 95.2 at the end; no collapse; 78 CPU-seconds, 1,240,007 cells of which 5,756 are keyed on two classes; those alone name the tens digit on 98.8%). **K alone 81.9**, where the frozen setting has 90.8. W1 held; W2 failed.

*Why K alone fails* (my reading from the result file): a kind that once had a kept pair now stays a candidate for good, so the candidates grow from 16 to about 27 kinds, and the 32 places go to the pairs of largest worth among all of them. Worth (how much better the classes foretell than one class) is large for many pairs that are of little use; with 16 candidates by trust the answering cell had been doing part of the choosing, and `keep=1` took that away. With the kinds keyed on two classes the harm does not show on this seed, but the repair is not right as it stands. The runs of addition 1 for K alone on prose and on three-digit sums were stopped unread and are NOT RUN.

*The repair, second form* (`keep=2`): the candidates are again the 16 kinds of largest trust, but **the trust of a kind counts the trust of the kinds made from it**: to a kind's mean trust is added the mean trust (where above nothing) of each class kind found in it and of each kind keyed on two classes of which one was found in it. A kind whose classes succeed keeps its place because of them; a kind whose classes earn nothing does not. **Setting C** = the frozen setting with `keep=2`; **setting C2** = C with the kinds keyed on two classes.

*Choice between K2 and C2*, on development material only: the higher on two-digit sums (seed 1, last three readings) among those not above the frozen setting on prose's last twentieth by more than 0.002 nats; within half a point, C2 (its rule for candidates is the one that does not harm the first mechanism, if W5 holds). The chosen one is run on the reported seeds and books as section 4 and addition 1 say; the other is reported as development only.

*Predictions (mine).* W4: C2 on the development seed at least 94 (0.6). W5: C alone within 1.5 points of the frozen setting (0.7). W6: C2 on prose's last twentieth not above the frozen setting's 1.5488 by more than 0.002 (0.8).

## 7. Outcome on development material, and the choice (entered 20 November, before any run of this plan on a reported seed or a book; `res/dev6_*.json`, `out_dev6_*.txt`)

| Setting | two-digit sums, seed 1, last three (last) | prose, last twentieth | three-digit, seed 1 |
|---|---|---|---|
| no classes | 76.8 | 1.5566 | (89.7, as `nano/ahead2` reports) |
| frozen setting | 90.8 (90.6) | 1.5488 | stopped for the cores before it ended, NOT RUN (the five reported seeds are in `PLAN_5.md`) |
| two-class kinds, candidates as frozen | 82.7 (95 to 96 up to 30,000 lines, then 76) | 1.5482 | not run |
| K (candidates kept for good) | 81.9 (82.2) | 1.5488 | stopped, NOT RUN |
| K2 | 94.9 (95.2) | 1.5482 | 92.3 (93.4) |
| C (a kind's trust counts its children's) | 90.5 (89.6) | not run | not run |
| **C2** | **95.5 (95.8)** | **1.5482** | not run |

(The prose run of K alone had finished before I stopped the K runs; its figure, 1.5488, is entered. On prose K, K2 and C2 do not differ from their parents: the seven kinds are always all candidates.)

W4 held (95.5), W5 held (90.5 against 90.8), W6 held (1.5482). **Chosen by the rule of addition 2: C2.** On the development seed its 21 kinds keyed on two classes hold 3,647 cells and alone name the tens digit of never-shown sums on 98.8%.

A caution before the reported seeds (mine): on three-digit sums K2 on the development seed has the hundreds digit at 94.6%, no better than the plain cells; the kind I said was needed (the class of the tens digits, the class of the units digits, the two hundreds digits) may not have been made. V1 and V2 look doubtful.

Runs now made with C2: five reported seeds on two-digit sums (C beside it); five on three-digit sums; the first life; 400,000 lines on two seeds for C and C2.

## 8. Outcome on the reported seeds and the books (entered 20 November; `res/s40_C*_seed*.json`, `res/s3_C2_seed*.json`, `res/life_C2.json`, `res/life2_C2.json`, `res/s400_C*_seed*.json`; tables `report_s40.txt`, `report_s3.txt`, `report_life.txt`, `report_life2.txt`, `report_s400.txt`)

Sums: never-shown additions, own answer bytes, whole answer, %, mean of the last three readings; seeds 73129, 93683, 116089, 406364, 545353.

| | frozen setting | C | C2 |
|---|---|---|---|
| two-digit, 40,000 lines, each seed | 91.8, 93.3, 91.7, 88.0, 92.1 | 90.9, 92.9, 91.5, 87.4, 91.5 | 92.8, 96.7, 95.1, 89.7, 93.1 |
| median (last reading; tens digit) | 91.8 (92.2; 92.4) | 91.5 (90.8; 90.8) | **93.1** (92.8; 92.8) |
| median at 5,000; 10,000; 15,000 lines | 86.8; 88.4; 91.6 | 88.6; 89.2; 92.0 | 92.6; 94.0; 95.0 |
| three-digit, 80,000 lines, each seed | 94.9, 93.0, 94.0, 94.1, 95.0 | not run | 93.9, 92.9, 94.1, 93.2, 95.1 |
| median (hundreds digit, last) | 94.1 (95.6) | | **93.9** (96.4) |
| two-digit, 400,000 lines, seeds 73129 and 93683 | 78.9, 81.3 | 80.1, 81.3 | 81.3, **98.7** |

With C2 on two-digit sums the kinds keyed on two classes number 24 to 30 and hold 3,494 to 5,093 cells; alone they name the tens digit of never-shown sums on 98.2, 90.6, 89.4, 90.6, 97.6% (they do not cover the hundreds digit: 23 to 51% whole). 67 CPU-seconds a run at 40,000 lines (52 for the frozen setting), 167 on three digits (145).

Prose, nats a byte (the first life with the test text's last tenth at the end; then the second life):

| | bryant-stories | burgess-busterbrown | carroll-alice | test text, at the end | chesterton-thursday | chesterton-brown | austen-persuasion |
|---|---|---|---|---|---|---|---|
| no classes | 1.5023 | 1.3645 | 1.4456 | 1.5306 | 1.4411 | 1.3996 | 1.2700 |
| frozen setting | 1.4920 | 1.3487 | 1.4307 | 1.5264 | 1.4263 | 1.3856 | 1.2594 |
| C2 | 1.4885 | 1.3349 | 1.4019 | **1.5550** | 1.4242 | 1.3663 | 1.2446 |

(On prose C2 made 5 and 8 kinds keyed on two classes at the last finding, with 71,020 and 72,174 cells; where the second book of the first life is scored they hold 20,300 cells, having been made anew at 1,280,000 bytes.)

**The hypothesis is refuted** by its first part: on three-digit sums C2 gains nothing over the frozen setting (93.9 against 94.1; under 1 point refutes). Parts (b) and (c) hold as written (two-digit 93.1, not lower; no book above the frozen setting), but (c) holds for a reason that is not a merit: see reading 3.

| | Statement | Outcome |
|---|---|---|
| V1 | three-digit: at least 2 points above the frozen setting | failed (0.2 below) |
| V2 | three-digit: the hundreds digit at least 98 | failed (96.4) |
| V3 | two-digit: at least 91.8 | held (93.1) |
| V4 | two-digit: at least 94 | failed (93.1; three seeds of five are under 94) |
| V5 | the kinds keyed on two classes alone: at least 85% whole with under 3,000 cells | failed (23 to 51% whole; 3,500 to 5,100 cells; they hold the tens digit and not the hundreds) |
| V6 | no book of the first life more than 0.002 above the frozen setting | held |
| V7 | below the frozen setting by at least 0.002 on at least two books | held (0.004, 0.014, 0.029) |
| V8 | the whole hypothesis | failed |
| W3 | at 400,000 lines C above the frozen setting on both seeds; C2 at least 85 on both | failed, both halves (C: 80.1 and 81.3 against 78.9 and 81.3; C2: 81.3 and 98.7) |

Readings (mine).
1. **On two-digit sums a class used as a line helps, and helps early.** 92.6% at 5,000 lines and 94 to 95% at 10,000 to 15,000, where the frozen setting has 87 to 92 and the plain cells 65 to 76. The kind made is the one hoped for: the class of the two tens digits with the class of the two units digits, a few dozen cells for the tens digit. The median at 40,000 lines (93.1) is held down by the kinds starting empty at each finding (the dip at 20,000 and 25,000 lines is there on the median) and by the answering cell, as before.
2. **On three-digit sums it gains nothing**, and the reason is the search, not the idea: the kind that would fix the hundreds digit needs the class of the two units digits as seen from where the hundreds digit is named, and no pair with that X is kept, because such a class is of use only together with another and the worth of a pair is measured alone.
3. **On prose the gain on the books is recency and the old text pays for it.** The cells keyed on two classes start empty at every finding, so they hold only the text since then; the three books of the first life gain 0.004 to 0.029 and the test text at the end of the life loses 0.029, ending above the system with no classes (1.5550 against 1.5306). For keeping, C2 as built is worse than the frozen setting. The second life agrees on its books (0.002, 0.019, 0.015).
4. **One long run is worth looking at and proves nothing yet**: on seed 93683 C2 stands at 98.6 to 98.8% from 300,000 to 400,000 lines, where every other long run of this line ends near 80. On seed 73129 it does not.
5. **The repair C** (a kind's trust counts the trust of the kinds made from it) does not change the first mechanism at 40,000 lines (91.5 against 91.8) and keeps the kind of the whole pair in the search to the end of both long runs, where the frozen setting loses it once on each seed; it does not stop the fall with reading.
