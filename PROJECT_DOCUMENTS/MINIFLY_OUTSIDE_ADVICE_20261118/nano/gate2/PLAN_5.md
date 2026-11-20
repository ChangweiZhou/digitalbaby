# Line `nano/gate2`, fifth plan: a second answering cell that, while it learns, hears each sender with probability one half (20 November 2026)

**Authorisation**, as quoted to this session by the coordinating session (session_017dSEpVed7SthEH3ngGAdNc): Mr. Xu (in Chinese): "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and, most recently: "继续大胆尝试。" (the coordinating session's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." and "Go on trying boldly."). The coordinating session's brief for this line allows a further mechanism after step 4 if its plan is written first. **Source:** `claude/friendly-darwin-8mmhfm` at a859b2f224f0adca576616c0fc96bf82699ec293; this branch at the commit that carries this file. Development level; no gate, seal or review is claimed. Written before any run of the mechanism except the check of its program (`check3_output.txt`), which includes one short run on development seed 1 (4,000 lines, 8 pairs of classes) whose one reading I saw: after a read line and with an empty past, second cell 66.4 and 70.6, the new cell 71.7 and 73.4, control 61.6 and 61.6, the control with half heard 70.4 and 65.0. Nothing was chosen on it.

## 1. The evidence this follows

1. `../merge/REPORT.md`, section 1, point 6: on a text of repeated lines the trust of class kinds is taught only at first meetings. The roadmap, section 9: in the second half of a reading, lines met before carry twenty times the weight of first meetings in what changes the trusts.
2. This line, `PLAN_2.md`, section 7: over 400,000 lines the first cell's reading of the plain kinds alone falls from 79% to between 7 and 21%, with no finding of classes involved; without classes the first and the second cell lose 17 to 18 points together. After 40,000 lines every shown line has been met many times; on a line met before, the cell that holds the whole line answers exactly, and every trust is fitted in its presence. A never-shown line is answered with exactly that cell silent: a situation in which, after the first pass through the shown lines, no trust is ever taught again.
3. `PLAN_3.md`, sections 5 and 6: after a change of task the reading is upset at once (never-shown addition falls to 50% within 1,000 lines, and to 7% after a long first task) though the cells that hold addition are untouched: the trusts lean on a balance among senders that the new lines disturb.

In all three the trusts are fitted with a set of senders speaking that is not the set that speaks when it matters.

## 2. The one change

A second answering cell as in `PLAN_1.md` (its parts indexed by kind, level and the byte at each delay; the first cell untouched), with one difference **while it learns**: at each byte it hears each speaking sender with probability one half, independently, and doubles what it hears, so that its branches are

    a2(v) = sum over the senders heard of 2 (T[k][level] + w_k) s_k(v),     w_k as in PLAN_1.md,

(the first cell's trust T is read, not changed); its mismatch is taken from that answer, and only the parts of the senders heard change, by the same step rule. **When it answers** (every score) it hears all senders, undoubled, exactly as the second cell of `PLAN_1.md` does. So on every line it reads, about half of the time the cell that holds the whole line is silent for it, and its parts are taught in something like a first meeting for as long as it reads.

Nothing in the rule names a kind, a byte, a place or a text. The one number is one half; it is fixed here and will not be varied (it is the value at which the sets of senders heard are most varied, and the customary one). This is the known means of silencing units at random while weights are learned ("dropout"), applied to the senders of an answering cell that learns from its own mismatch.

*Controls, in the same run from the same first cell:* the second cell of `PLAN_1.md` (all senders heard); the control of `PLAN_1.md` (parts read a constant); and the control with half the senders heard (parts read a constant), which separates what silencing does to a larger total step from what it does to parts that know the bytes.

*Program.* `gate3.py`: `gate2.py` with the kernel extended (four second cells; a random stream inside the kernel, xorshift64*, seeded from the run's seed), a numpy and plain-Python reference of the whole loop including the random stream, and the same checks (`check3_output.txt`: compiled against the reference to 1e-14; the first cell bit-equal to `merge.py`'s; the runner's first cell equal to `merge.run_sums`). One note on the check: with 70 senders and a constant step of 0.05 the control cells are numerically unstable and rounding differences between the compiled loop and numpy grow to the first place within 1,500 bytes (the cells whose parts read bytes agree to 1e-14, and all agree to 1e-15 at 300 bytes or with the shrinking step); the check's constant-step case therefore uses a step of 0.002.

*Size.* Each second cell 1,451,520 parts with classes (four of them in a run; a system would carry one). CPU about a quarter more than `gate2.py`.

## 3. Hypothesis, what refutes it, predictions (mine, before any run)

*Hypothesis.* The cell that hears half its senders while it learns reads better where the specific cells are silent: (i) over 400,000 lines without classes (where the loss is even and the base's search plays no part), its mean of the last three readings after a read line is at least 5 points above the second cell's on at least two seeds of three; and (ii) at 40,000 lines with classes it is not worse: its median over the five reported seeds is no more than 1 point below the second cell's 97.4. *Refuted if* in (i) it is not above the second cell on two seeds of three, or in (ii) it is more than 2 points below.

| | Statement (never-shown additions after a read line; medians over the seeds run) | Chance |
|---|---|---|
| H1 | 400,000 lines without classes: at least 5 points above the second cell on at least two seeds of three (mean of the last three) | 0.40 |
| H2 | the same: above the second cell on at least two seeds of three, by any margin | 0.65 |
| H3 | 40,000 lines with classes, five seeds: its median no more than 1 point below the second cell's | 0.55 |
| H4 | the same: its median at least half a point above the second cell's | 0.30 |
| H5 | 40,000 lines without classes: its median at least 1 point above the second cell's 85.6 | 0.45 |
| H6 | 400,000 lines with classes (frozen setting, three seeds): above the second cell at the end on at least two seeds (the step falls of the base remain) | 0.50 |
| H7 | three-digit sums, five seeds: its median no more than 1 point below the second cell's 96.9 | 0.50 |
| H8 | the life of two tasks (40,000 and 20,000 lines, with classes): it keeps a larger share of addition than the second cell (88.8% at the median) | 0.50 |
| H9 | the same life: its lowest reading after the change is above the second cell's (63.2 at the median) | 0.55 |
| H10 | the control with half heard is at least 1 point above the plain control at 40,000 lines with classes (silencing alone gives something) | 0.45 |

## 4. Runs, in this order (three at a time)

`gate3.py`, all five cells from one run. (1) 40,000 lines, five reported seeds, with classes and without. (2) 400,000 lines without classes, seeds 116089, 406364, 545353. (3) 400,000 lines with classes, the same seeds. (4) Three-digit sums, 80,000 lines, five seeds, with classes. (5) Addition 40,000 lines then subtraction 20,000, five seeds, with classes. Scores as in `PLAN_1.md` to `PLAN_4.md`. The first, second and control cells of these runs must equal the earlier runs' to the digit (the same first cell, the same arithmetic); this is checked and reported. Cost: about 100 to 110 CPU-minutes in all.

## 5. Not run

Prose (the mechanism's transfer to the other kind of text is owed and is not claimed); any probability other than one half; silencing while answering; silencing in the first cell or in the counting of cells; `nano/merge`'s setting C together with this cell (unless time remains after the runs above: then seeds 116089, 406364, 545353 over 400,000 lines, as a reading).
