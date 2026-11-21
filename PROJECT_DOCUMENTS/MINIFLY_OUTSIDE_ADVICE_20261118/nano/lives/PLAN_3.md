# Remembered trusts: the answering cell keeps copies of its trusts as they were and can return to them (21 November 2026, 00:10 UTC)

**Authorisation and steer.** As `PLAN_1.md` (Mr. Xu's words quoted there). The choice of this step is mine. **Source.** Branch `claude/friendly-darwin-8mmhfm` at the commit that holds this file. Written after the outcomes of `PLAN_1.md` (small system, five reported orders) and `PLAN_2.md` (development only), before any run of the program of this plan. Development level; no checker gate, seal or review is claimed.

## 1. The evidence this follows

`PLAN_1.md`: a life over six streams costs the small system 5 to 11% of what six separate systems pay. Putting back the trusts and parts the system had when a stream was read removes about three quarters of the loss going on (0.74 at the median) and more than half with nothing changing (0.57). The trusts are few numbers (17 kinds by 40 levels, and the parts of the second cell). `PLAN_2.md`: deciding where a kind of text begins, by the bytes of the block just read, failed; its lesson was that the worth of keeping something shows only when an old kind returns.

So: keep the trusts as they were at many moments, at a fixed pace, with no decision about kinds; and let the answer itself say which copy is right now, byte by byte, by how well each has just foretold. In the theory of prediction with experts this is known as mixing past posteriors; I have not checked the literature tonight and claim no result from it.

## 2. The mechanism, exactly (`books.py`)

* **Bookmarks.** After every N bytes read with learning, the second answering cell's whole set of trusts and parts is copied and kept (a bookmark). A bookmark never changes. The living set goes on learning exactly as before: the bookmarks touch nothing of it, and its own answer, which is the plain system's of `PLAN_1.md`, is still computed and scored at every byte (the comparison is built in).
* **Every bookmark answers every byte**, hearing the same senders with the same messages as the living set, each with its own trusts and parts.
* **Shares.** The cell holds a share for the living set and for each bookmark, summing to one. Before a byte: each share becomes (1 - s) x share + s / (number of sets), with s = 1 / N. The cell's answer is the sum over sets of share x that set's answer. After the byte b: each share is multiplied by what its set gave to b, and the shares are made to sum to one. A new bookmark starts with no share and gets one by the rule.
* **One number, N** (the pace of bookmarks; it also sets s). Chosen on the development order between 65,536 and 262,144 by the mean over the six tests of the mixture's going-on loss at the end of the life, then frozen. All bookmarks are kept; in a life of 8 million bytes that is 30 to 120 sets. A rule for letting bookmarks go, which a much longer life needs, is not built here.
* With **nothing changing**, no bookmark is taken and no trust moves; the shares still move (as the clocks do).
* Only the second answering cell has bookmarks here; the first is unchanged.

Not touched: the counts. The quarter of the loss that sits in the counts of contexts shared by several streams is not addressed by this plan.

**Check before any run** (`python3 books.py check`): the compiled kernel against a reference written out in full (bookmarks, every set's answer, shares, the mixture); the living set's losses against the plain kernel's (equal to the rounding of the machine's numbers: the two kernels are compiled separately); with no bookmark the mixture is the living set's answer; no answer of the mixture moves when its own byte and all later ones are replaced; a measurement in a forked copy leaves the system as it was.

## 3. Hypothesis, what refutes it, predictions (mine)

**Hypothesis.** With remembered trusts, at the end of a life over six streams, going on, the excess of a stream's test over its own stage falls by half or more against the plain system in the same life (at the median over the cases, of the five reported orders, in which the plain system's excess is above 0.02); and the mixture is nowhere worse than the plain system by more than 0.005 nats a byte (any test, any stage, either measure).

**Refuted if** the median removal is under a quarter, or the mixture is worse than the plain system by more than 0.005 anywhere.

| | Statement (small system; mixture against the plain answer of the same run; reported orders) | Chance |
|---|---|---|
| B1 | The mixture is nowhere worse than the plain answer by more than 0.002 (every test, every stage, both measures, every order) | 0.80 |
| B2 | Going on: at least half of the excess removed at the median (the hypothesis) | 0.55 |
| B3 | The cost of the life (mean over six tests at the end less the mean of the plain singles, 0.8357) is cut by at least 40% in at least four of five orders | 0.45 |
| B4 | Nothing changing: at least half of the excess (end less own stage) removed at the median over cases with a plain excess above 0.02 | 0.50 |
| B5 | What is read later is still learned worse: at its own stage, going on, the mixture is within 0.01 of the plain answer in at least 20 of the 25 later positions (bookmarks of other kinds do not help a new kind) | 0.60 |
| B6 | On a stream read alone, the mixture is below the plain answer by 0.001 or more on at least three of six streams | 0.35 |
| B7 | At the end of a life, on the test of a stream read earlier, the largest share is on a bookmark taken while that stream was read for more than half of the test's bytes, in at least two thirds of the cases | 0.50 |

## 4. Runs, size, cost, what is not run

Development: the development order at N = 65,536 and 262,144; the six singles at the chosen N. Reported: the five orders at the chosen N. Size: the plain system's cells (2.3 million at the end of a life) and, for each bookmark, 26,792 numbers (17 x 40 trusts and 6 x 17 x 256 parts): 0.8 to 3.2 million numbers more in a life. Cost: each bookmark answers each byte; expected 5 to 20 CPU-minutes a life. **Not run:** the large system with bookmarks (its parts are twenty times as many; after the small one is scored); a rule for letting bookmarks go; returning the living set to a bookmark; bookmarks for the first cell; anything for the counts.

## 5. Outcome

(to be entered after the runs)
