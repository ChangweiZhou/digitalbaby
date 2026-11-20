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
