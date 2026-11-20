# Step 5: one frozen setting chosen on development material, then the claims (20 November 2026)

**Authorisation.** Mr. Xu, 20 November 2026 (in Chinese): "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" and "继续大胆尝试。" (the coordinator's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas." and "Go on trying boldly."). **Source.** `claude/nano-layers-8mmhfm` at the commit that carries this file. Written before the runs it describes; when it was written the prose run of step 4 had been read (`res/life_armsstep4_booksworn.json`) and the sums run of step 4 had not finished (it was restarted after an endless draw: eighteen slots asked of sixteen delays; that arm now has sixteen). A first start of this plan's prose run was stopped unread and its files removed, because this plan had not been committed. Development level; no checker gate, seal or review is claimed.

## 1. What the steps so far give (development material only)

One wide further cell is the best arrangement (step 3). Its families so far: the delays' parts without the level (nl) and with it (fine), the pair at delays 1 and 2, a trust the sending cell owns, the first cell's sureness. Step 4's prose run adds one observation: among gates drawn at random, **the level of counts of the sender of one kind** earns its keep (slots drawn from all sorts, twelve of them, gave 0.0323 nats over the first cell against 0.0292 for nl, and nl with six drawn slots gave 0.0372), while single far bytes do not. The level of every kind as a gate by hand would be 288 families on sums, so the drawn form is the one that is the same on both texts.

## 2. Part A: the choice, by a rule fixed here

Candidates, each one further cell that hears the first cell (C6: two), run together on sums seed 1 and on the life with the three worn books, with the single second cell and nl beside them:

| | Families |
|---|---|
| C1 | nl + pair + own + sure (all without the level) |
| C2 | C1 + the fine parts of the delays |
| C3 | C2 + D slots drawn from all three sorts, reviewed as in `PLAN_4.md` |
| C4 | C2 + 2D drawn slots |
| C5 | C1 + D drawn slots |
| C6 | C2 on C2 (the second copy's sureness gate reads the first copy) |
| C7 | C2 + the pair and the sureness also with the level |

*Rule.* Among the candidates within 1 point of the best candidate on sums (mean of the last three readings after a read line), the one with the largest mean gain on the three worn books; candidates within 0.001 nats of it give way to the one with fewer parts used on prose. The chosen cell is written into `layers.py` as `FROZEN` in an addition to this plan, committed before any claim run.

## 3. Part B: the claims, on material where nothing was chosen

*Sums.* Seeds 73129, 93683, 116089, 406364, 545353; 40,000 lines; `nano/merge`'s system with classes; never-shown sums after a read line (five draws) with the empty past beside it; mean of the last three readings and the last; the first cell, the single second cell, nl and the frozen cell from the same run. Median over the five seeds.

*Prose.* (a) The life "the test text's stream, then `edgeworth-parents`" (`minifly_v149_input_ceiling_dev_20261025/data/edgeworth-parents.txt`; no line has opened it; this line opens it once, here): T2; its first nine tenths read, its last tenth scored right after with nothing changing. (b) A second reading: the same life with `chesterton-ball`, `austen-emma`, `melville-moby_dick` (opened twice by the coordinator). In both: the test text's last tenth right after its own stream with nothing changing (the worn measure on which 1.454 was measured for the full nanoGPT network of `nano/ref`, one seed; scored for some fifty arms of the coordinator's line and for every run of this line: it carries little) and again at the end of the life (what is kept), for each cell.

*The hypothesis of the whole line* (`PLAN_1.md`, section 1): with the one frozen setting the frozen cell is at least 1 point above the single second cell on sums (median of the five seeds, mean of the last three readings, after a read line) and at least 0.010 nats below it on `edgeworth-parents`. **Refuted if neither is reached.** If one is reached and not the other, that is what is reported.

| | Statement | Chance |
|---|---|---|
| F1 | sums: the frozen cell at least 1 point above the single second cell at the median | 0.20 |
| F2 | sums: the frozen cell at or above the single second cell at the median (not below it) | 0.75 |
| F3 | sums: nl at or above the single second cell at the median | 0.65 |
| F4 | `edgeworth-parents`: the frozen cell at least 0.010 nats below the single second cell | 0.85 |
| F5 | the same at least 0.015 below | 0.60 |
| F6 | second reading: at least 0.010 below on each of the three books | 0.80 |
| F7 | `edgeworth-parents`: nl at least 0.003 below the single second cell | 0.75 |
| F8 | kept: at the end of life (a) the frozen cell is still below the first cell on the test text's last tenth, by less than a third of its gain right after the stream | 0.60 |

## 4. Size, cost, what is not run

Parts held and used are reported for each cell. Part A: about eight CPU-minutes on sums, two on prose. Part B: about three CPU-minutes for each seed of sums (three at a time), one for each life. *Also reported if time allows, as a reading and not a claim:* the frozen cell on `all.py`'s system with all kinds of cells but the relay system (the coordinator's 1.4707 system is that with the relay system; over 4 GB), the test text's last tenth with nothing changing. *Not run:* sums over 400,000 lines; sums without classes; the life of two tasks; other seeds of the draws; the control with the step doubled (owed from step 3).
