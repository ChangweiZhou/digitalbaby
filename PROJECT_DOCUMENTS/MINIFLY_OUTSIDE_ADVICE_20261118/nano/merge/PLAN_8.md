# Line `nano/merge`, eighth plan: one more move for the search; a class is split around its worst-foretold member (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this session by the coordinating session: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and then "继续大胆假设，基于事实努力尝试。" (the coordinating session's translation is in `PLAN_1.md`). **Source.** `claude/nano-merge-8mmhfm` at the commit that holds this file. Written before any run with `dissolve=2` other than the check (`check_output.txt`: the compiled search against brute force in plain Python, three cases). Development level; no gate, seal or review is claimed.

## 1. The evidence this follows

1. On every reported seed the pairs of units digits that lie against their class are pairs that sum to nine; on three seeds of five all ten of them sit with the carrying pairs (`PLAN_2.md`, `PLAN_4.md` note on R1). Their best place by the measure J is a class of their own, which moves of one cell cannot found, and which dissolving a class and seating its members again did not found either.
2. The two long runs with cells keyed on two classes (`PLAN_6.md`): on seed 93683 the system stands at 98.6 to 98.8% of never-shown sums from 300,000 to 400,000 lines; on seed 73129 it ends at 79.4. Looked at after the outcome was entered (`res/s400_C2_seed*.json`): on the first, at the last finding the units-digits classes are 56 and 44 with one stray, and the kinds of largest trust are keyed on (the class of the two tens digits, the class of the two units digits, the byte two back, which is the '=' where the tens digit is named); on the second the classes are 55 and 45 with ten strays, and the same kinds are there. **My reading: where the search seats the pairs that sum to nine rightly, the system holds 98% at long reading; where it does not, it falls. What stands between is the search.**

## 2. The one change (to the search; everything else is setting C2 of `PLAN_6.md`)

After the sweeps, for each class of three or more: its **worst-foretold member** (the member whose leaving costs J least for each of its counts) founds a new class; every other member of the old class then chooses between the old class and the new one by J; the split is kept only if J of the whole grouping rises; this is repeated on the same class up to four times, stopping at the first split that is not kept; then one more sweep over all cells and all classes. `dissolve=2` in `merge.py`. "Up to four times" and "three or more" are laid down; nothing says what a class should be.

## 3. Hypothesis, what refutes it, predictions (mine, before any run)

*Hypothesis.* With the split move, on two-digit sums the units-digits classes are pure for the carry (at most two strays of a hundred) on at least four reported seeds of five, and never-shown sums at 400,000 lines are at least 95% on both seeds of the long reading. *Refuted if* the classes are pure on fewer than three seeds, or either long run ends under 90.

| | Statement | Chance |
|---|---|---|
| X1 | development seed, 40,000 lines: units-digits classes with at most two strays, and the pairs that sum to nine in a class of their own or with the pairs that do not carry | 0.55 |
| X2 | development seed: sums at least 95 (C2 without the split: 95.5) | 0.50 |
| X3 | prose, last twentieth of the stream: not more than 0.002 above C2's 1.5482 | 0.70 |
| X4 | five seeds, 40,000 lines: pure on at least four | 0.45 |
| X5 | five seeds, 40,000 lines, median at least 95 (C2: 93.1) | 0.35 |
| X6 | 400,000 lines: at least 95 on both seeds | 0.30 |
| X7 | three-digit sums, five seeds: within 1 point of C2's 93.9 | 0.60 |

## 4. Runs, size, cost

Development material first (two-digit seed 1; prose's last twentieth): to see the cost of the search on prose (my estimate: three to five times the search of the frozen setting) and that it does no harm. Then the five reported seeds at 40,000 lines, the two long runs, five seeds on three-digit sums; the first life only if the search's cost on prose allows it within this session, else NOT RUN and said so. Not run: moves of more than one founder; a search that merges classes.

## 5. Outcome (entered 20 November; `res/dev8_sums.json`, `res/dev8_prose.json`, `out_dev8_*.txt`). The plan stops on development material; its premise was wrong.

| | two-digit sums, seed 1, last three (last) | prose, last twentieth | CPU s (search), sums; prose |
|---|---|---|---|
| C2 (`PLAN_6.md`) | 95.5 (95.8) | 1.5482 | 64 (40); 108 (78) |
| C2 with the split move | 95.5 (95.4) | 1.5486 | 113 (86); 215 (182) |

On the development seed the units-digits classes are the same with the split move as without: 53 pairs that do not carry, and 47 of which 45 carry and two, (6, 3) and (1, 8), sum to nine. X1 failed in its second clause (two of the pairs that sum to nine still sit with the carrying pairs); X2 held (95.5) without being due to the change; X3 held (0.0004 above).

**Why the split does nothing: the measure itself prefers the grouping that was found.** Computed directly on the counts of the development seed at 32,000 lines (the kind of delays 3, 4, 6, 7, X = delays 3, 6, e_J = 0.02; the script is in the session's history and the function is `j_total`):

| Grouping of the pairs of units digits | J, nats |
|---|---|
| as the search finds it (two pairs that sum to nine with the carrying pairs) | -454,931 |
| those two moved to the pairs that do not carry | -460,471 |
| all ten pairs that sum to nine in a class of their own | -458,621 |

**My account in `PLAN_4.md` (note on R1), `PLAN_2.md` and section 1 of this plan, "a limit of the search, not of the measure", was wrong.** No move of the search can find a grouping the measure scores lower. Seeds, long runs, three-digit sums and the life with the split move are NOT RUN: there is nothing for them to test. X4 to X7 are not scored.

What I now think is the cause (mine; consistent with `PLAN_7.md`, section 6, and not tested further): a kind is a set of delays with no place in the line, so the cells of delays 3, 4, 6, 7 that the measure reads hold three things at once. Where the tens digit is named they hold the whole pair and its tens digit; where the second number's first digit is named they hold four other digits followed by a digit that is there by chance (about three such counts in each cell, beside about four of the first sort, at 32,000 lines); where the hundreds digit is named, the same X-cell is the two tens digits. The measure is taken over all of it. With e_J = 0.02 a byte that a class cell has never counted costs nine nats, so the counts that are there by chance pull cells toward whichever class already holds every digit as a follower, and the grouping that is best for the mixture is not the grouping that is the carry. That the classes come out 90 to 98% right in spite of this is the result; that they do not come out exact is, on this evidence, a consequence of kinds without a place, not of the search and not of the idea of the measure.

What follows for the long runs (`PLAN_6.md`, reading 4): on seed 93683 the mixture happened to seat the pairs that sum to nine with the pairs that do not carry at the last finding, and the system held 98.7%; on seed 73129 it did not. Both are the same measure on different counts.

What would test it: kinds that carry a place (a kind of largest trust refined by one more delay, so that the '=' can be in it beside the four digits), and the same measure on their cells. NOT BUILT; it needs kinds that are made while reading, which `merge.py` does not have.

## 6. A look that tests the cause (20 November; `look8.py`, `look8_output.txt`; a diagnostic with one kind laid down by hand for the look only, not a system and not a claim)

Development seed, 32,000 lines. The measure J (e_J = 0.02) for three groupings of the hundred pairs of units digits, every other X-cell left alone: (1) on the cells of the system's kind, delays 3, 4, 6, 7; (2) on the cells of a kind with the place in it, delays 2, 3, 4, 6, 7; (3) on those of its cells in which delay 2 holds '=' (the place where the tens digit is named and no other).

| Grouping | (1) delays 3, 4, 6, 7 | (2) delays 2, 3, 4, 6, 7 | (3) the same, '=' at delay 2 only |
|---|---|---|---|
| by the carry: ten or more / nine or less | -610,882 | **-575,396** | **-54** |
| nine or more / eight or less | **-608,705** | -576,858 | -8,785 |
| ten or more / nine / eight or less | -608,983 | -583,611 | -85 |

(The figures of section 5 are for the groupings as the search left them, with the other X-cells in their classes; these are for the three groupings alone, so the two tables are not on one scale.)

**The cause is confirmed as far as a look can.** On the system's own kind the measure prefers the grouping that puts the pairs that sum to nine with the carrying pairs, by 2,177 nats: the search was right to leave them there. On a kind that has the place in it the same measure prefers the carry, and where only the place of the tens digit is counted the carry is exact to 54 nats in 32,000 counts while the other grouping costs 8,785. Nothing is wrong with the measure or the search; **the cells of a kind without a place are the wrong thing to measure on.** A system that makes such kinds itself (a kind of largest trust refined by one more delay) is the next step and is NOT BUILT.
