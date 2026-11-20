# Line `nano/merge`, fourth plan: where the frozen setting loses, taken one change at a time on development material (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and then "继续大胆假设，基于事实努力尝试。" (the coordinating session's translation is in `PLAN_1.md`). **Source.** `claude/nano-merge-8mmhfm` at the commit that holds this file. Written after step 2's runs at 40,000 lines and its life on prose were read (the 400,000-line runs with classes still running) and before any run this plan describes. Development level; no gate, seal or review is claimed. **What I had seen when writing this:** the five reported seeds at 40,000 lines and the three books under the frozen setting and its controls. Anything chosen below is therefore chosen with those in sight; section 4 says how that is met.

## 1. The evidence this follows (step 2; figures in `report_s40.txt`, `report_life.txt`; scored in `PLAN_2.md` when the long runs end)

1. Sums, five seeds, 40,000 lines: 77.5 without classes, **91.8** with (88.0 to 93.3), 78.1 with classes by chance, 88.1 with classes found once at half, 93.1 with replacement.
2. On three seeds of five the pairs of units digits fall into two classes that are *not* the carry: one holds the 45 pairs that sum to ten or more and the ten that sum to nine, the other the 45 that sum to eight or less (on the other two seeds: 47 and 53, two strays). My account, from the measure J: the same X-cell also stands where the hundreds digit is named, as the two tens digits, and there the pairs that sum to nine behave like neither group. Their best place is a class of their own (I computed J for it by hand: far higher), but the search moves one cell at a time and a single cell leaving for an empty class loses. **A limit of the search, not of the measure.**
3. With only the class kinds speaking (development seed; `alone.py`): 93.8% of never-shown sums with 17,885 cells, against 90.6% with all 1.23 million cells speaking; four class kinds with 3,628 cells give 94.8%. **The plain cells outvote the class cells where they are wrong.**
4. Prose, the three books: classes lower the loss by 0.010, 0.016, 0.015 nats; classes by chance by 0.002 to 0.003; replacement raises it on two books. **Classes found once at 500,000 bytes, their cells starting empty, lower it by 0.016, 0.031, 0.021**: more than classes found anew at every doubling with pooled counts. I did not expect this. Of the 32 kept pairs 30 or 31 persist from one finding to the next, so it is not the trusts being born again; what differs is that the classes themselves are made anew eight times, and that the cells start pooled.

## 2. The arms: each is one change from the frozen setting, on development material only

Sums: development seed 1, 40,000 lines. Prose: taught on the stream before its last twentieth, scored on that twentieth (frozen: 90.8 and 1.5488; no classes: 76.8 and 1.5566).

| Arm | The change | What it tests |
|---|---|---|
| F | none (the frozen setting run again with the program as it now is) | that the program still gives 90.8 and 1.5488 |
| D | `dissolve=1`: after the sweeps each class of two or more is dissolved once, its members seated again one by one (free to found classes), the result kept only if J rises; then one more sweep | evidence 2: more search |
| Oe | classes found once, at half the reading, cells starting empty (`mode=half`) | evidence 4 on development material |
| Op | the same with the cells starting pooled (`mode=half pool=1`) | whether "once" or "empty" is what helps |
| E | found at every doubling, cells starting empty (`empty=1`) | the same from the other side |
| Pr | `practice=0.5`: while reading, at half of the bytes (drawn by chance) the R-cells that a class stands for are silent as in replacement; when answering, nothing is silent | evidence 3: the trust of class kinds is taught only where the member's own cell does not already answer |
| P64 | at most 64 kept pairs | more search |

*A second setting.* A change is "good" if on development material it is not below the frozen setting by more than 1 point on sums nor above it by more than 0.002 nats on prose, and gains at least 1.5 points on sums or 0.002 nats on prose. The good changes that can be combined are combined (Oe, Op and E exclude one another: the best of them on prose that is good), the combination is run once on development material, and it is adopted as the second setting if it is itself good and not below every one of its parts on both tasks; otherwise the single best good change is the second setting (best: the largest sum of its gain in points on sums and a thousand times its gain in nats on prose). If none is good, there is no second setting.

## 3. Predictions (mine, before any run)

| | Statement | Chance |
|---|---|---|
| R1 | D: on the development seed the pair of units digits has every class of two or more pure for the carry | 0.55 |
| R2 | D: sums at least 92.3 | 0.35 |
| R3 | Oe: prose below the frozen setting by at least 0.002 | 0.60 |
| R4 | Oe: sums below the frozen setting by more than 1 point | 0.75 |
| R5 | Op: prose not below Oe | 0.55 |
| R6 | E: prose below the frozen setting by at least 0.002 | 0.45 |
| R7 | Pr: sums at least 92.5 | 0.50 |
| R8 | Pr: prose not above the frozen setting by more than 0.002 | 0.50 |
| R9 | P64: better than the frozen setting on both tasks | 0.45 |
| R10 | a second setting is adopted | 0.55 |

## 4. If a second setting is adopted

It is run (a) on the five reported seeds at 40,000 lines; (b) on the life with the three books, named as a setting chosen after those books had been seen under the frozen setting; (c) on a **second life with three books this line has not opened**: `chesterton-thursday`, `chesterton-brown`, `austen-persuasion` (from `minifly_v153_bits_dev_v1_20261025/data/` and `minifly_v149_input_ceiling_dev_20261025/data/` at the base commit; the three smallest of the texts not yet used, with at most two by one author), read after the test text's stream in that order, each last tenth scored right after its stream; with no classes, with the frozen setting and with the second setting. The frozen setting on the second life is a repeat of part (ii) of the hypothesis on new material. *Prediction (mine):* on each of the three new books the frozen setting is at least 0.005 nats below the base (chance 0.75); the second setting is below the frozen setting on at least two of the three (0.5).

## 5. Size, cost, what is not run

As step 2: a development run about one CPU-minute on sums and one and a half on prose; fourteen runs. The second life about seven CPU-minutes a run. Not run: a trust that depends on which other cells speak; classes kept from one finding to the next and only refined; X of more than two delays; classes of classes; subtraction and keeping.
