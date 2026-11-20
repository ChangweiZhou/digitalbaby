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
