# nano/skip, tenth plan: a higher bar for closing, against the entries that close by chance early and stay closed (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026, as quoted to this line by the coordinating session: "Keep experimenting with the agents." and "Respect evidence and keep trying new ideas." **Source.** `claude/nano-skip-8mmhfm` at c7611a6; `ctx.py` and the kernel unchanged. **Status.** Development level; no checker gate, seal or outside review is claimed. Written before its runs.

**What this follows.** `PLAN_9.md`, reported seeds: with entries for a byte after a byte (472 tendencies) every filler is closed as a whole on every seed, but on two seeds of five an entry of the answer's first digit after `=` is closed at the first reading and never reopens, at a cost of up to 40 points.

**Reading (mine) of the fault, which this plan tests in part.** An entry closes when its tendency passes zero, that is when its evidence for closing is 0.7 of its own uncertainty or more (alpha 2, starting chance 0.2). Among 472 entries with no evidence yet, about a quarter are past that bar by chance at any moment. Most reopen as evidence arrives. One whose byte the cells have not yet come to use does not: closed, it costs nothing, and opened one time in a hundred it finds cells that never learned it. The bar was set on a text with 18 entries. The cheapest change is the bar itself; the dearer one (making a closed entry reopen for long enough for the cells to learn its byte) is left for after.

**One change.** alpha, which sets the bar: 2 (bar 0.7), 1 (bar 1.4), 0.5 (bar 2.8). Everything else as `PLAN_9.md`, addition 2 (`key=pair`, `listen=16`, `bit=1`; 256 cells; 55,570 numbers that learn and 1,888 at the entries).

**Runs.** `h2`, 2,400,000 lines. (A) Seeds 1001 to 1005, which have not been used on this text: alpha 2, 1 and 0.5. (B) The five reported seeds at alpha 1 and 0.5, beside alpha 2 from `PLAN_9.md`. Since alpha is now being moved after the reported seeds were seen, (B) is a look and (A) is the comparison that counts; neither is a confirmation.

**Question.** Does a higher bar remove the wrong closings without losing the fillers?

**Hypothesis H10.** On the seeds 1001 to 1005, with alpha 0.5 no entry other than the fillers' is closed at the end on at least four seeds, and the median is at least 85% with 0, 2 and 4 fillers. **Refuted if** two or more seeds end with a wrong entry closed, or the median is under 85%.

| | Prediction (mine, before the runs) | Chance |
|---|---|---|
| A1 | alpha 2 on the seeds 1001 to 1005: a wrong entry closed at the end on at least one seed (the fault is not peculiar to two seeds) | 0.70 |
| A2 | H10 holds | 0.45 |
| A3 | alpha 0.5: the first reading with at least 50% at every number of fillers is later than with alpha 2 (medians over the seeds that arrive) | 0.70 |
| A4 | alpha 0.5 on the reported seeds: 116089 and 406364 both end with no wrong entry closed | 0.45 |

**Cost.** 25 runs of about 3.2 CPU-minutes: about 80 CPU-minutes. **Not run.** Making a closed entry reopen; the expectation rule as the order of trial; a longer listening time.

## Outcome (entered 20 November 2026; `res/p10_*.json`, and `res/p9H_bit_*.json` for alpha 2 on the reported seeds)

`h2`, 2,400,000 lines, never-shown pairs with 0 / 2 / 4 fillers, mean of the last three readings.

**(A) Seeds 1001 to 1005 (not used on this text before).**

| alpha (bar) | 0 fillers: each seed; median | 2 fillers | 4 fillers | A wrong entry closed at the end | First reading at 50% with every number of fillers (thousands of lines) |
|---|---|---|---|---|---|
| 2 (0.7) | 49.7, 97.5, 54.4, 95.1, 96.1; **95.1** | 94.1, 97.5, 93.1, 95.1, 96.1; **95.1** | the same; **95.1** | on 1001: the `+` after each of the ten digits; on 1003 none is listed and the lines without fillers are still at 54.4 | 2000, 600, 1000, 600, 800 |
| 1 (1.4) | 95.6, 96.4, 94.8, 95.1, 96.7; **95.6** | the same | the same | none on any seed | 800, 800, 800, 600, 800 |
| 0.5 (2.8) | 75.4, 81.7, 80.6, 79.3, 84.3; **80.6** | 75.8, 82.1, 80.3, 79.3, 84.0; **80.3** | 73.4, 81.3, 76.3, 79.3, 82.1; **79.3** | none on any seed | 1400, 1400, 1400, 1200, 1400 |

At every alpha all 40 entries of a filler digit after a letter that was passed over are closed on every seed. Of the 40 entries of a letter after a filler digit that was passed over, all are closed at alpha 2 and 1, and 32 to 34 at alpha 0.5.

**(B) The reported seeds 73129, 93683, 116089, 406364, 545353 (a look: alpha was moved after these seeds were seen).** alpha 2 (`PLAN_9.md`): 96.2, 97.5, 55.7 to 91.2, 54.5 to 59.9, 97.2; a wrong entry on two seeds. alpha 1: 95.9, 95.9, 95.9, 37.5 / 71.6 / 71.6, 97.1; a wrong entry on one seed (406364: `1` after `=`). alpha 0.5: 87.9, 81.2, 81.5, 82.2, 79.6 without fillers and 86.8, 78.7, 80.3, 77.6, 75.1 with four; no wrong entry; 29 to 35 of the 40 letter entries closed.

**H10 is refuted**: with the highest bar no wrong entry is closed on any of the ten seeds, but the median is 80.6 / 80.3 / 79.3, under the 85% asked; the fillers are not all closed by the end. **A1 held** (the fault is not peculiar to two seeds: it shows on one or two of the five fresh ones at alpha 2). **A2 failed. A3 held** (1,400,000 lines against 800,000). **A4 held.**

**Reading (mine).**
1. The bar does what I said it would to the wrong closings: at 0.7 of the uncertainty three or four seeds of ten end with a wrong entry or a lasting loss; at 1.4, one of ten; at 2.8, none.
2. It is not free, and I chose the wrong end of it for the hypothesis. At 2.8 the fillers are found about twice as late and not completely within the run. The middle value is the best of the three on this text: all five fresh seeds at 94.8 to 96.7% (median 95.6, the oracle's level) with nothing wrong closed, and four of the five reported ones.
3. The fault is made rarer by the bar, not removed: one seed of ten at the middle value still holds the answer's first digit closed from the start. What would remove it is what the plan left out, a closed entry that reopens for long enough for the cells to learn its byte.
4. alpha is therefore a number that had to be moved when the entries went from 18 to 472, which I report as such: it was not tuned for a text, but it does depend on how many entries are being tried at once.

**Cost.** 25 runs, about 80 CPU-minutes.
