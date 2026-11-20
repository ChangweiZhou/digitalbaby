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
