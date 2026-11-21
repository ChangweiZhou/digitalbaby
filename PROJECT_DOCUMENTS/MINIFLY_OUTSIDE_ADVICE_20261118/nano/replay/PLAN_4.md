# Keeping by replay, fourth plan: dreams (the relay system writes what it knows of never-shown lines into the store) (20 November 2026)

By the same separate agent session as `PLAN_1.md` to `PLAN_3.md`; same folder, same source. Written before any run of this step; the check of the program was run first (`check_replay4_output.txt`; lives of 900 lines and one of 30,000). Development level; repeated by no one; no checker gate, seal or review is claimed.

**Authorisation and origin.** Asked by the coordinating session in a message of 20 November, which relays Mr. Xu as having said since (in Chinese; the coordinating session's translation, which is all I have): "You can go on trying boldly; we keep to the facts; besides the ideas I have mentioned, try others as well." The idea, the measures, the hypothesis and the controls are the coordinating session's; what I fixed or added is marked as mine.

## 1. The question

`PLAN_1.md` ended on a limit: replay rests on a store that holds every shown line whole, and the system answers never-shown lines of the old task because the relay system does, not because the store holds them. **If the relay system writes its own answers to sums it was never shown into the store while it still has them, (1) does it keep more of never-shown addition through subtraction, and (2) does the store then answer never-shown addition by itself?**

## 2. The change (one, from arm R of `PLAN_1.md`: r = 1, K = 8, add:400000 then sub:200000)

Of the extra lines, a share q are not replays but **dreams**.

- *The two numbers* (mine, to make "digit by digit from short cells" exact): the first number and the sign after it are drawn byte by byte from the cells that begin at the newline (runs of 1 to 3 bytes), as a replay would draw them. The second number and the sign after it are drawn from cells that begin at the first sign (the sign; the sign and one digit; the sign and two digits), which do not see the first number. So the two numbers are paired by chance, and about one dreamed pair in five is a pair never shown. These cells of 1 to 3 bytes that begin at a sign are new: the program now holds them and counts them on every counted line; they are in the cost.
- *The answer* is written by the relay system: it reads the six bytes, names the next byte, reads its own byte, and so on four times. The dream is kept if the three answer bytes are digits and the fourth is the newline (a test of form only; the program does not know whether the sum is right when it keeps a dream).
- *A kept dream is counted by the counting cells like a real line, at weight w.* The relay system does not learn from it at that moment. Later replays draw from everything counted, dreams included, and the relay system learns from those replays.
- *Chunks* (mine, for speed, stated because it is an approximation): the life is read in chunks of 500 real lines; the dreams of a chunk are made at its start, from the cells and the relay system as they then are, and counted at their places in the chunk. A dream's answer is therefore the relay system's of at most 500 real lines earlier.

What I see beforehand as the way this fails, as the coordinating session also says: a dream is right only as often as the relay system is at that moment, counts do not fade, and a wrong dream that is replayed teaches the relay system its own error. Early in the life the relay system is wrong about nearly everything. For a shown pair the real lines outnumber its dreams; for a never-shown pair the store holds nothing but dreams. Once subtraction lines are counted, the sign after the first number is drawn from both texts, so dreams of subtraction occur, among them lines with the smaller number first, a form the text never writes; they are counted like the others (the system cannot know) and reported apart.

## 3. Arms and settings

- **q = 0**: arm R (the check: the same strengths as `replay.py`'s arm R after the same life, to the last digit).
- **Dreams** with (q, w) from the grid {0.1, 0.3} x {1, 1/4}, chosen on development seed 1 by the highest never-shown addition at the end (the relay system's); within one point, the larger w, then the smaller q.
- **Dreams that begin late** (an addition of mine): the same, with no dream before 300,000 real lines (q in {0.1, 0.3}, w = 1), so that only a relay system that already answers most sums writes into the store. The better of the two on seed 1 goes to the five seeds as a separate arm.
- **Control, chance**: dreams at the chosen (q, w) whose three answer digits are drawn by chance: the store filled with wrong lines of the right form.

Five reported seeds for q = 0, the chosen dreams arm, the late arm and the chance control; 1001 to 1010 where two arms are within five points and it matters.

## 4. Measures

1. Never-shown addition at the end, answered by the relay system (the 500 pairs of before).
2. Never-shown addition answered by the counting cells alone: for each of the 500 held pairs (and of all 2,000 never-shown pairs), whether the store has a line for it, and whether the store's answer (the largest count in the longest cell, byte by byte) is right. With no dreams this is 0 of 500 by construction. Taken after each phase.
3. Cost and errors: cells and counts above zero; dreams made and kept; of the kept addition dreams, how many are right, by reading; of the replays the relay system learns from, how many are wrong sums, by reading (does the share grow over the life?).
4. Subtraction on the clean score (never-shown pairs, of all 2,000, whose line no shown pair writes).

## 5. Hypothesis and what would refute it (the coordinating session's)

**Never-shown addition at the end is at least 3 points above arm R's (77.2 at the median of the five reported seeds). Refuted if it is lower than arm R's.** Between the two it fails without being refuted. I score it on the chosen dreams arm, paired with q = 0 on the same seeds.

## 6. Predictions (mine), with chances

- D1 (0.35): the hypothesis holds for the chosen dreams arm (at least 3 points above q = 0 at the median of the paired differences).
- D2 (0.4): the chosen dreams arm is below q = 0 at the median of the paired differences (the hypothesis refuted): wrong early dreams of never-shown pairs are replayed for the rest of the life.
- D3 (0.6): the late arm is at or above q = 0 at the median of the paired differences.
- D4 (0.7): in the chosen dreams arm, at the end, the store has a line for at least 90% of the 500 held pairs.
- D5 (0.5): of those, the store's answer is right for at least 60%.
- D6 (0.6): in the late arm the store's answer is right for at least 75% of the held pairs it has.
- D7 (0.8): the chance control leaves never-shown addition at least 10 points below q = 0.
- D8 (0.6): in the chosen dreams arm the share of wrong sums among replays is lower in the last reading of the first phase than in its first reading (errors do not grow while addition is being learned).

## 7. Cost, not run

A run takes about 70 to 90 seconds of one core. Development: 7 runs (q = 0, four dreams, two late). Reported: 4 arms on five seeds. About 15 minutes on four cores. The numpy loop is checked on short lives with dreams made, counted and replayed; no full-length numpy run.

Not run: a rule by which the relay system writes only what it is sure of; counts that fade; dreams answered by a vote of several moments; dreams in `PLAN_3.md`'s harsh change; other chunk sizes.

---

## 8. Entered 20 November after the check and the development runs, before any run on the reported seeds

**The check** (`check_replay4_output.txt`): passed. Compiled against numpy with dreams made, counted and replayed, over two phases: 5e-14 to 1e-13, the same dreams kept. With q = 0 the program gives arm R of `replay.py` exactly (difference 0.0). The relay system's written answers agree with its scored answers (72.3 and 72.3 after 30,000 lines).

**An error of mine in a measure, found on the first development runs and corrected before anything was entered.** The store's own answer was compared with the true answer over the wrong bytes, so "right" read 0 everywhere. Corrected (a case added to the check), and the seven development runs made again; every other figure came out the same.

**Development seed 1** (`res/d_v_*.json`). q = 0: never-shown addition at the end 79.0 (arm R's figure on this seed), clean subtraction 84.0.

| q, w | addition at the end (relay) | clean subtraction | held pairs in the store at the end, of 500 | of them the store has right | of all 2,000 never shown: in the store, right | kept addition dreams right, first reading to last, % | wrong sums among replays, first to last, % |
|---|---|---|---|---|---|---|---|
| 0.1, 1 | **81.2** | 79.8 | 499 | 432 (86.6%) | 1,996; 1,702 (85.3%) | 64 to 89, then 83 | 2.6 to 1.5, then 1.6 |
| 0.3, 1 | 80.2 | 88.1 | 500 | 422 | 2,000; 1,715 | 63 to 86, then 83 | 7.1 to 4.5, then 4.2 |
| 0.1, 1/4 | 80.6 | 81.7 | 498 | 425 | 1,995; 1,705 | 67 to 89, then 83 | 0.6 to 0.4 |
| 0.3, 1/4 | 77.6 | 86.7 | 500 | 418 | 2,000; 1,680 | 65 to 84, then 81 | 2.0 to 1.4 |
| 0.1, 1, from 300,000 lines | 80.0 | 86.2 | 452 | 383 (84.7%) | 1,841; 1,563 | 91, 90, then 82 | 0.1 rising to 0.7 |
| 0.3, 1, from 300,000 lines | 77.6 | 84.7 | 500 | 431 | 1,998; 1,713 | 90, 90, then 79 | 0.2 rising to 2.2 |

**Choice by the rule of section 3:** the dreams arm is **q = 0.1, w = 1** (81.2; the two within one point of it have the smaller w or the larger q). The late arm is q = 0.1. The chance control is at q = 0.1, w = 1.

Seen on this seed, to be tested on the reported ones: the store alone has 86.6% of the held additions right at the end, which is above the relay system's own 81.2 at that moment and near its 89 to 90 at the end of the addition phase; the share of wrong sums among replays falls through the addition phase and does not grow in the subtraction phase at q = 0.1.

**Runs on the reported seeds, queued together before any is read:** q = 0, dreams, late, chance on the five reported seeds; and, because seed 1 puts dreams 2.2 points above q = 0, which is inside what five seeds carry, dreams also on 1001 to 1010. For q = 0 on those ten seeds I use `PLAN_1.md`'s arm R runs (`res/p_e_R1_*.json`), which the check shows to be the same life to the last digit.

## 9. Outcome (entered 20 November, after all runs of this plan)

Tables: `report4.txt` (made by `report4.py` from `res/d_*.json`). Arms: q = 0 (arm R; no dreams); D (dreams, q = 0.1, w = 1); L (dreams only after 300,000 real lines; my addition); C (control: dreams answered by chance). Seeds 73129, 93683, 116089, 406364, 545353; median.

**(1) Never-shown addition at the end, answered by the relay system.**

| | five seeds | median | paired against q = 0 |
|---|---|---|---|
| q = 0 | 77.2, 85.0, 77.2, 75.6, 79.4 | **77.2** | |
| D | 75.2, 77.0, 80.8, 76.4, 84.8 | **77.0** | -2.0, -8.0, +3.6, +0.8, +5.4; median +0.8 |
| L | 74.2, 84.0, 73.6, 75.2, 76.4 | **75.2** | median -3.0; five of five below |
| C | 68.8, 72.2, 76.6, 78.0, 88.6 | **76.6** | median -0.6; -12.8 to +9.2 |

On fifteen seeds (D on 1001 to 1010 beside `PLAN_1.md`'s arm R, the same life): q = 0 79.4, D 76.6; paired D less q = 0: -2.2 at the median, above on six seeds and below on nine (-13.6 to +6.4).

**The hypothesis fails.** Dreams do not put never-shown addition 3 points above arm R: +0.8 on the five reported seeds, which is neither the 3 points nor "lower"; on fifteen seeds it is lower (-2.2, nine of fifteen), which by the letter of section 5 refutes it, though six seeds go the other way and I would say only that **dreams do not help the relay system keep, and may cost it a little.**

**(2) Never-shown addition answered by the counting cells alone.**

| | held pairs with a line in the store, of 500 | the store's own answer right, % of the 500 | the same at the end of the addition phase | % of all 2,000 never-shown pairs | the relay system at the end, same 500 |
|---|---|---|---|---|---|
| q = 0 | 0 | 0.0 | 0.0 | 0.0 | 77.2 |
| D | 498 | 83.4, 84.8, 87.0, 80.8, 88.2; **84.8** | 83.8 | 83.6 | 77.0 |
| L | 458 | **76.0** | 55.4 | 77.0 | 75.2 |
| C | 499 | 0.2 | 0.0 | 0.2 | 76.6 |

**This is the result of the plan.** With dreams the store holds a line for 498 of the 500 never-shown pairs and its own answer is right for 84.8% of the 500 (83.6% of all 2,000 never-shown pairs), from counts alone, with no relay system consulted. That is near the relay system's own level at the end of the addition phase (86.9%) and **above the relay system's level at the end of the life** (77.0%): on fifteen seeds the store's answer is ahead of the relay system's by 4.4 points at the median, and behind on none (0.0 to +8.2). The store does not lose it through subtraction: 83.8% at the end of the addition phase, 84.8% at the end of the life. The control by chance has the same lines in the store and 0.2% right, so the answers are the relay system's and not an accident of the store.

So the coordinating session's reading holds for the store and not for the relay system: what the relay system generalised is now held in a form that does not forget, and a system that answered from its store wherever the store has a line would stand at about 85% of never-shown addition after subtraction, where the relay system stands at 77. (I did not run that combined answer as an arm; the two figures are measured apart.)

**(3) Cost and errors** (medians of five seeds).

- Cells: 84,909 in D against 66,272 in q = 0 (both with the new short cells, which are about 220); counts above zero 104,471 against 79,307. The store grows towards the whole domain: it holds a line for 99.6% of the never-shown addition pairs beside the 8,000 shown ones, that is, nearly all 10,000 sums. 59,329 dreamed lines are counted over the life beside 600,000 real ones.
- Of the kept addition dreams 83.5% are right over the life (83.1% of the 10,964 on never-shown pairs): by reading on seed 73129 from 64% in the first 50,000 lines to 86 to 89% late in the addition phase, and 83% in the subtraction phase. The 16.5% that are wrong stay in the store for good.
- Wrong sums among the replays the relay system learns from: 1.7% over the life; by reading (medians) 2.2, 2.1, 1.9, 1.7, 1.6, 1.5, 1.6, 1.5 in the addition phase and 1.5, 1.6, 1.6, 1.6 in the subtraction phase. **Errors do not grow over the life**; they fall while addition is learned and then stay level. The failure the plan feared (the relay system taught its own errors until they compound) does not show at q = 0.1 over 600,000 lines.
- Dreams of subtraction: 3,418 kept, 62.4% right; 535 of them have the smaller number first, a form the text never writes, and are counted like the others. At the end the store has a line for 39% of the clean never-shown subtraction pairs and is right on 73% of those.
- CPU: 63 seconds a run, as arm R.

**(4) Subtraction, clean score** (never-shown pairs whose line no shown pair writes), the relay system at the end: q = 0 78.9, D 80.1, L 77.2, C 79.4 (five seeds); on fifteen seeds q = 0 77.5, D 80.2, paired +0.9 (eight above, six below). No difference.

**Two things I did not expect.**

1. *Dreams answered by chance do not harm the relay system measurably* (76.6 against 77.2; paired -0.6, from -12.8 to +9.2), although 9.0% of the replays it learns from are then wrong sums and the store holds 230,422 cells. Wrong answers drawn by chance agree with each other on nothing, and the relay system does not learn them; I had given 0.8 to a loss of 10 points (D7). It also means the relay system's score is a blunt measure of what is in the store: D and C differ by 85 points in the store and by nothing in the relay system.
2. *Dreams that begin late are the worse arm*, for the relay system (-3.0, five of five below) and for the store (76.0% against 84.8), although a late dream is more often right when made (90% against 64% early). Beginning late leaves fewer dreams (30,016) and 42 held pairs with no line; and the store's answer is a vote of the dreams of a pair, which more dreams make steadier. I had expected the opposite (D3).

**Predictions.** D1 failed (+0.8). D2 failed on the five seeds as registered (+0.8 is not below); on fifteen the difference is below (-2.2). D3 failed (-3.0). D4 held (498 of 500). D5 held (85.1% of those in the store). D6 held (82.6%). D7 failed (-0.6). D8 held (2.2% to 1.5%).

**Limits.** The dreams of a chunk are answered by the relay system of up to 500 lines earlier. One q and w on the reported seeds, chosen on one seed by the relay system's score, which this plan shows to be a blunt measure for the choice. The store's answer is taken by the largest count; ties and near-ties are not looked at. The store with dreams is a store of nearly the whole domain of sums, so this shows keeping of what was generalised at the price of holding every sum, not a smaller store. On this testbed the domain is 10,000 sums; nothing here says what the price is where the domain is not small.

**Not run:** the combined answer (store first, relay system where the store has no line) as an arm; D on `PLAN_3.md`'s harsh change; q and w other than the chosen on the reported seeds; L and C on the further seeds; a relay system that writes only what it is sure of.
