# Return: the living set of trusts goes back to a bookmark that foretells better (21 November 2026)

**Authorisation and steer.** As `PLAN_1.md` (Mr. Xu's words quoted there). The choice of this step is mine. **Source.** Branch `claude/friendly-darwin-8mmhfm` at the commit that holds this file. Written after the two development lives of `PLAN_3.md` (the pace 65,536 was chosen there by that plan's rule), while that plan's reported orders were running and before any of them was read; before any run of the program of this plan. Development level; no checker gate, seal or review is claimed.

## 1. The evidence this follows

`PLAN_3.md`, development order, at the end of the life, going on: with bookmarks mixed by shares, tiny Shakespeare's test costs 1.443 (plain 1.504; at its own stage 1.343), the book 1.477 (1.559; 1.402), Python 0.996 (1.060; 0.963). `PLAN_1.md`'s diagnostic D3a, in which the living set itself starts again from the trusts of the stream's own stage and goes on learning, gave 1.394, 1.456 and 0.981. The difference (0.05, 0.02, 0.015) is what a bookmark cannot do: it never changes, so when the old kind of text is back the system answers from old trusts that no longer learn, beside living trusts that are far away.

## 2. The mechanism, exactly (`wake.py`)

`PLAN_3.md` with one addition. **When, before a byte, a bookmark's share is above one half,** then after that byte has been learned: the living set is kept as a new bookmark (so nothing is lost), the living set takes that bookmark's trusts and parts (its sums of uses and of squared drivers stay as they are), the living set takes over that bookmark's share, the new bookmark keeps the share the living set had, and the old bookmark's share is set to nothing (it is now the same as the living set). Nothing else changes. The one half is not a number to choose: it is the point at which one set holds more belief than all the others together.

Since the living set is then no longer the plain system, the comparison is with the runs of `PLAN_3.md` in the same orders (same pace, 65,536).

**Check before any run** (`python3 wake.py check`): the compiled kernel against the reference written out in full, with returns occurring in the cases; with the addition switched off the losses are those of `PLAN_3.md`'s kernel; no answer moves when its own byte and all later ones are replaced.

## 2a. Addition 1 (21 November, written after the first development life was stopped, before any other run)

The development life with the rule as written above was stopped while it read its first stream (`out/W_dev_first_rule_stopped.txt`): in the first megabyte of tiny Shakespeare alone the living set returned 201 times and 216 bookmarks stood where `PLAN_3.md` has 15. The cause is plain once seen: a bookmark just taken is a twin of the living set, the two foretell alike, the belief passes between twins by small differences, and each passing was a return that added one more twin. **The rule is changed in one point before any run is scored: at a return the living set is not kept as a new bookmark.** It takes the bookmark's trusts and parts and adds the bookmark's share to its own; the bookmark's share is set to nothing. What the living set held is lost only as far back as the last bookmark of the pace (at most 65,536 bytes of learning). Returns to twins then add nothing to the store and change almost nothing in the trusts. Predictions W1, W2, W4 and W5 stand as written for the changed rule; W3 was written for the first rule and is scored as it stands.

## 3. Hypothesis, what refutes it, predictions (mine)

**Hypothesis.** With return, at the end of a life, going on, the mixture's excess over the stream's own stage is smaller than without it by 0.01 or more in at least half of the cases (of the five reported orders) in which `PLAN_3.md`'s mixture still has an excess above 0.02; and on no test at any stage is the mixture with return worse than without by more than 0.005.

**Refuted if** either part fails.

| | Statement (small system, pace 65,536; mixture with return against `PLAN_3.md`'s mixture; reported orders) | Chance |
|---|---|---|
| W1 | The first part of the hypothesis holds | 0.50 |
| W2 | The second part holds (nowhere worse by more than 0.005) | 0.60 |
| W3 | Fewer than 300 returns in every life | 0.60 |
| W4 | Against the plain system: the going-on excess removed at the median is 0.60 or more | 0.40 |
| W5 | Streams read alone: the mixture with return is within 0.002 of the mixture without on all six | 0.65 |

## 4. Runs, size, cost, what is not run

Development: the development order once (to see that returns happen and how many; nothing is chosen). Reported: the five orders and the six singles. Size and cost as `PLAN_3.md`, with one bookmark more for each return. **Not run:** bookmarks that keep their own sums; a set that is young (reborn at the pace); the large system.

## 5. Outcome (entered 21 November; `out/W_dev_first_rule_stopped.txt`, `out/W_dev_second_rule_stopped.txt`)

**Return, in both forms, failed in development and was stopped; nothing was run on the reported orders, and no prediction is scored on them.**

| while tiny Shakespeare's body was read (first stream of the development order), nats a byte | the living set | the mixture | bookmarks | returns |
|---|---|---|---|---|
| `PLAN_3.md` (no return) | 1.4110 | 1.4110 | 15 | none |
| return as first written (the living set kept at each return) | 1.4118 | 1.4115 | 216 | 201 |
| return as in addition 1 (the living set not kept) | 1.4214 | 1.4212 | 15 | 173 |

With the second form the system is 0.010 nats a byte worse while reading one stream alone, more than twice the 0.005 the hypothesis allows (W2 fails in development; W3, fewer than 300 returns in a life, would fail on the first stream or the second).

**Why (my reading).** A bookmark of the same kind of text is a twin of the living set. Between twins the shares say nothing: their losses differ by chance at each byte, the differences add up along the stream faster than the small share for switching pulls them back, and one twin or another holds more than half of the belief by turns. "More than half of the belief" is therefore no evidence that the living set should give way, and each time it gave way to an older twin it lost up to 65,536 bytes of its own learning. The rule would need evidence of a lasting difference between the two sets, which the share is not.

**What stands:** the gap this plan was aimed at is real (section 1). **What I would build instead, not built here:** the set that holds the largest share goes on learning beside the living set, each from its own mismatch, so that nothing is ever overwritten: an old kind's bookmark wakes when its text returns and learns on from where it stood, and a twin that wakes does no harm. It needs each bookmark to keep its own sums.
