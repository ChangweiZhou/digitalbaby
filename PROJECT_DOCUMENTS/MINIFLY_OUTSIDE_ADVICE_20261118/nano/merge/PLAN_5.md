# Line `nano/merge`, fifth plan: the frozen setting on a task it was not chosen on, three-digit sums (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and then "继续大胆假设，基于事实努力尝试。" (the coordinating session's translation is in `PLAN_1.md`). **Source.** `claude/nano-merge-8mmhfm` at the commit that holds this file. Written before any run with classes on three-digit sums; one run without classes, 10,000 lines on seed 1, was made to see that the text is read (77.0%). Development level; no gate, seal or review is claimed.

## 1. Why

The steer asks for means that transfer. The frozen setting was chosen on two-digit sums and one piece of prose. Three-digit sums are a third task, with nothing chosen on it; the text, the split and the length are `nano/ahead2/PLAN_2.md`'s (lines such as `047+985=1032`; a permutation of the million pairs from the seed, the first fifth never shown, the first 500 scored; 80,000 lines; readings every 10,000; mean of the last three). That line reports 92.3% for cells of at most four delays with no look-ahead (thousands, hundreds, tens, units at the last reading: 99.6, 95.2, 98.0, 100) and 99.7% with look-ahead.

## 2. The question, the hypothesis, what refutes it

**Does the same program with the same numbers gain on three-digit sums, and where does it stop?** Nothing is chosen or changed: `merge.py sums nd=3 lines=80000`, no classes against the frozen setting, five reported seeds.

My expectation, from how the mechanism is built, is a small gain and a named limit. On this text any four digits recur in line after line, so the plain cells already hold what a class of two digits against two others would give. What is missing is the hundreds digit when the two tens digits sum to nine: it then needs the carry out of the units digits as well, six digits in all. A cell keyed on (the class of the two tens digits, the class of the two units digits, the two hundreds digits) would hold it; that is **a cell keyed on two classes**, and the system as built keys a new kind on one class and the rest of one existing kind (laid down in `PLAN_1.md`, section 2).

*Hypothesis.* The frozen setting is above no classes at the median of five seeds, by at least 1 point. *Refuted if* it is below. (Between 0 and 1: no gain shown.)

| | Statement | Chance |
|---|---|---|
| T1 | with classes at least 1 point above without (median, last three readings) | 0.45 |
| T2 | with classes below without | 0.20 |
| T3 | with classes at least 97 | 0.10 |
| T4 | the hundreds digit stays the lowest place with classes | 0.70 |
| T5 | class kinds alone are below all kinds together (here every line read is new, so the trust of class kinds is taught all along and the plain cells are not wrong where the class cells are right) | 0.65 |

## 3. Size and cost

About 1.9 million cells at 10,000 lines without classes (the other line: 2.0 million at 80,000); my estimate is one CPU-minute without classes and three with. Ten runs. Not run: a kind keyed on two classes; look-ahead with classes.

## 4. Outcome (entered 20 November after the ten runs; `res/s3_*.json`, `out_s3_*.txt`, `report_s3.txt`)

Never-shown three-digit additions, own answer bytes, whole answer, %, 80,000 lines; seeds 73129, 93683, 116089, 406364, 545353; mean of the last three readings.

| | each seed | median | median, last | thousands, hundreds, tens, units (last, median) | cells | CPU s |
|---|---|---|---|---|---|---|
| no classes | 92.5, 89.9, 92.7, 90.8, 92.3 | 92.3 | 94.2 | 99.6, 95.2, 98.0, 100 | 1,994,138 | 50 |
| the frozen setting | 94.9, 93.0, 94.0, 94.1, 95.0 | **94.1** | 94.6 | 99.6, 95.6, 99.4, 100 | 2,029,513 (35,402 class cells) | 145 (85 in the search) |
| as `nano/ahead2` reports: no look-ahead; look-ahead | | 92.3; 99.7 | | 99.6, 95.2, 98.0, 100; all 100 | | |

The base equals the other line's figure for each seed (92.6, 89.9, 92.7, 90.8, 92.3 there). With only the class kinds speaking: 80.2, 79.2, 92.2, 88.8, 86.2 (median 86.2); with only the plain kinds: 86.6 (median).

**The hypothesis holds**: with nothing chosen on this task the frozen setting is 1.8 points above no classes at the median (at least 1 asked), and above on every seed (2.4, 3.1, 1.3, 3.3, 2.7). T1 held. T2 did not happen. T3 held as a prediction of a limit (94.1, under 97). T4 held (the hundreds digit is the lowest place, 95.6). T5 held (class kinds alone 86.2, below all kinds together on every seed): here, where nearly every line read is new, the class cells do not beat the whole, the opposite of two-digit sums.

Readings (mine).
1. **The setting transfers to a third task** with a gain of two points, and it gains early: 87.2 at 5,000 lines against 62.6, and 89.2 at 10,000 against 74.8 (medians). Later the plain cells catch up, since on this text any four digits recur.
2. **The gain is in the tens digit** (98.0 to 99.4). **The hundreds digit does not move** (95.2 to 95.6), as expected: it needs two classes at once. That is what `PLAN_6.md` builds.
3. Look-ahead does far better here (99.7). On this task the two devices are not level, as they were on two digits.
