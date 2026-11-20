# Line `nano/merge`, eleventh plan: a kind that is made is born with the past, where the system holds it (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and then "继续大胆假设，基于事实努力尝试。" (the coordinating session's translation is in `PLAN_1.md`). **Source.** `claude/nano-merge-8mmhfm` at the commit that holds this file. Written before any run with `full=1` other than the check (`check10_output.txt`, third line). Development level; no gate, seal or review is claimed.

## 1. The evidence this follows

`PLAN_10.md`, addition 3: on prose the test text's last tenth at the end of the life is 1.5264 with the frozen setting and 1.5550, 1.5573, 1.5685 with kinds keyed on two classes, with made kinds, and with both. A kind born late holds only what has been read since its birth; the answering cell learns on the current text to trust it; the old text is then answered from it. The class kinds of the frozen setting do not do this, because a class cell is born with the pooled counts of its members' cells, which hold the whole past.

## 2. The one change (to setting M; nothing else)

The same rule for every kind that is made. **If some kind the system has holds all the delays of the new kind, the new kind's cells are born as the pooled counts of that kind's cells** (of the one with the fewest delays, if there are several), and count on from there. Otherwise the kind is born empty, as before. For a kind made by refining (R with one more delay) that is a kind whose delays contain R's and the new one; for a kind keyed on two classes it is a kind that holds the delays of both X's and of the rest, each of its cells being counted into the cell of (its two classes, its rest). On prose such a kind nearly always exists (a run with a gap lies inside a run); on sums seldom (a kind of five delays lies in no kind of four). Nothing new is laid down and there is no new number. `full=1` in `merge10.py`; the check counts such cells again by hand over the whole stream.

## 3. Hypothesis, what refutes it, predictions (mine, before any run)

*Hypothesis.* With this rule the test text's last tenth at the end of the first life is not above the frozen setting's 1.5264 by more than 0.003 nats, no book is above the frozen setting, and two-digit sums are not below M by more than 1 point at the median of five seeds. *Refuted if* the test text at the end is above 1.5306 (the system with no classes), or sums fall under 96.

| | Statement | Chance |
|---|---|---|
| K1 | test text at the end of the first life: at most 1.5294 | 0.45 |
| K2 | the same above 1.5306 (refutation) | 0.35 |
| K3 | each book of the first life below the frozen setting (1.4920, 1.3487, 1.4307) | 0.60 |
| K4 | each book of the first life within 0.003 of M (1.4883, 1.3371, 1.3921) or below | 0.25 (the gain of M on the books was recency; born full, it should go) |
| K5 | two-digit sums, five seeds, median at least 97 (M: 98.0) | 0.70 |
| K6 | prose development material: at most 1.5490 | 0.70 |

## 4. Runs, size, cost

Because of the time left in this session the runs are started together and not in two stages: development material (two-digit seed 1; prose's last twentieth), the five reported seeds on two-digit sums at 40,000 lines, the first life with the test text's last tenth scored once at the end, and the second life. Cost as M. Not run: 400,000 lines; three-digit sums; the fresh seeds.

## 5. Outcome (entered 20 November; `res/life_F.json`, `res/life2_F.json`, `res/dev11_*.json`, `res/s40_F_seed*.json`, `out_*_F*.txt`)

Setting F = M with the rule of section 2. Prose, nats a byte:

| | development | bryant-stories | burgess-busterbrown | carroll-alice | test text, at the end | chesterton-thursday | chesterton-brown | austen-persuasion |
|---|---|---|---|---|---|---|---|---|
| no classes | 1.5566 | 1.5023 | 1.3645 | 1.4456 | 1.5306 | 1.4411 | 1.3996 | 1.2700 |
| frozen setting | 1.5488 | 1.4920 | 1.3487 | 1.4307 | 1.5264 | 1.4263 | 1.3856 | 1.2594 |
| M | 1.5470 | 1.4883 | 1.3371 | 1.3921 | 1.5685 | 1.4253 | 1.3673 | 1.2459 |
| **F** | 1.5477 | 1.4919 | 1.3518 | 1.4305 | **1.5272** | 1.4273 | 1.3853 | 1.2593 |

Two-digit sums, never-shown, whole answer, %, 40,000 lines, seeds 73129, 93683, 116089, 406364, 545353, mean of the last three readings: **97.9, 97.9, 96.7, 96.7, 98.3; median 97.9** (M: 98.0); last readings 98.0, 98.0, 96.8, 97.0, 98.4; hundreds, tens, units at the last reading (median) 98.0, 99.0, 100; median at 5,000, 10,000, 20,000 lines 90.2, 95.4, 97.0; 1,434,990 cells; 98 CPU-seconds. Development seed 95.6 (M: 96.9). On sums 15 to 24 of the 22 to 32 kinds keyed on two classes are born with the past at the last finding; no made kind is (none of the system's kinds holds five delays). On prose every made kind and every kind keyed on two classes is. First life 249 CPU-seconds and 2,336,138 cells; second life 265 and 2,979,800.

**The hypothesis holds in the two parts that its refutation was stated on, and fails one clause by 0.003 nats.** The test text's last tenth at the end of the first life is 1.5272: 0.0008 above the frozen setting, below the system with no classes, and 0.041 below M. Sums are 0.1 point below M. The clause "no book above the frozen setting" fails: burgess-busterbrown is 0.0031 above it and chesterton-thursday 0.0010; the other four are within 0.0003. K1 held. K2 did not happen. K3 failed (one book of the first life above the frozen setting). K4 failed, as I expected: born with the past, the made kinds and the kinds keyed on two classes no longer give the books the 0.004 to 0.039 nats that M had over the frozen setting. K5 held (97.9). K6 held (1.5477).

Readings (mine).
1. **The harm to the old text was the birth of kinds with no past, and one rule removes it**: a kind that is made is filled from a kind that refines it. This is the line's own first idea turned round (a class cell is the pool of the cells that refine it), applied to every kind the system makes.
2. **It also settles what M's and C2's gain on the books was**: recency. With the past in their cells the same kinds are worth nothing on prose beyond the frozen setting (six books within 0.003 of it). On prose the classes of the frozen setting are, so far, all that this line's mechanisms gain: 0.010 to 0.016 nats.
3. **F is one setting that does the first two parts of the line's hypothesis together, with the larger figure**: 97.9% of never-shown two-digit sums with no look-ahead, and each of six books 0.010 to 0.015 nats below the system with no classes, the old text kept. It was reached on the development seed after the reported seeds and the books had been seen under earlier settings; its figures on the reported seeds and the books are each a first and only run of that setting there. 400,000 lines, three-digit sums and the fresh seeds are NOT RUN for F (they were for M: 99.3 and 98.7; 95.9; 98.0); on sums F differs from M only in the birth of the kinds keyed on two classes.
4. The store is still larger than the base's on both tasks.
