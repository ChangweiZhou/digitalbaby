# A life over twelve kinds, six of them not text: do the findings hold where they were not made? (21 November 2026)

**Authorisation and steer.** As `PLAN_1.md` and `PLAN_6.md` (Mr. Xu's words quoted there). The choice of this step is mine. **Source.** Branch `claude/friendly-darwin-8mmhfm` at the commit that holds this file. Written before any life over more than six streams was run. Development level; no checker gate, seal or review is claimed.

## 1. Why

Everything this folder has found about a life (that the plain system does not keep it; where the loss sits; that counts in two memories take away most of the loss on streams read late) was found on six streams of text, and the remedies were shaped by what those six showed. `nano/suite` was held to account for the same thing (roadmap, section 26). The six streams of `PLAN_6.md` are bytes of other kinds (machine code from two compilers, fonts, compiled Python, a compiled table, a picture); no life has been run over them and nothing was chosen on them. A life over all twelve is twice as long, and its changes of kind are harsher.

## 2. What is run (`life12.py`)

The twelve streams: the six of `nano/suite` and the six of `PLAN_6.md` (not the control of bytes drawn by chance). Orders: the permutations of the twelve drawn by `numpy.random.default_rng(seed).permutation(12)` for the first three reported seeds (73129, 93683, 116089). Systems, each with the numbers it has, nothing changed: the plain small system of `PLAN_1.md`; with the bounded kinds of `PLAN_5.md` (M); with the cell's mixture of `PLAN_7.md` as well (MC). Measured as in `PLAN_1.md` (every stream's test after every stage, in forked copies, both measures) and against the same system's twelve singles. Remembered trusts are not run here (a life of twice the length holds twice the bookmarks, and each answers every byte): they are tested on the six streams and that limit is stated.

## 3. Hypothesis, what refutes it, predictions (mine)

**Hypothesis.** The findings hold on the twelve: (a) the plain system does not keep the life: the mean over the twelve tests at the end is above the mean of its twelve singles by 3% or more in every order; (b) with bounded kinds the loss on streams read late (own stage less own single, going on) is cut by half or more at the median over the later positions in which the plain system's loss is 0.02 or more; (c) with bounded kinds the system ends the life below the plain system on at least ten of the twelve tests in every order.

**Refuted if** (b) is under a quarter or (c) fails in two or more orders. If (a) fails, a life over these kinds costs the plain system little and there is less to remedy than six streams said.

| | Statement (going on, second answering cell; three orders) | Chance |
|---|---|---|
| L1 | (a) holds | 0.85 |
| L2 | (b) holds | 0.55 |
| L3 | (c) holds | 0.70 |
| L4 | The cost of the life with bounded kinds is lower than the plain system's by a quarter or more in every order | 0.50 |
| L5 | MC ends the life below M on at least nine of the twelve tests in every order | 0.60 |
| L6 | The six streams that are not text lose less in a life than the six of text (mean excess at the end over the own stage, plain system, in at least two of three orders): they share fewer contexts with the others | 0.50 |
| L7 | For the plain system the loss on a stream read late (own stage less single) is larger for the streams of text than for the others in at least two of three orders | 0.50 |

## 4. Runs, size, cost, what is not run

Six singles of MC on the streams that are not text (the other singles exist); nine lives (three systems, three orders). A plain life over twelve streams is about four times the work of one over six (twice the reading, twelve tests after each of twelve stages): about 20 CPU-minutes on a free machine, the others up to twice that. Size: about 11 million cells for the plain system at the end (the sum of the singles is 2.75 + 10.3 million). **Not run:** remembered trusts and the waking of bookmarks on twelve streams; the other two reported orders; the large system.

## 5. Outcome

The runs were made by this working line on its own machine (4 processors, 15 GB; Python 3.13.18, numpy 2.5.3, gcc 13.3.0), from commit 61e9b88, and were repeated by nobody. No program was changed. Development level; no checker gate, seal or review is claimed. Figures: going on, second answering cell (`report_9.txt`).

**Before the lives.** The four checks passed on this machine (`check_life12_*.txt`: `lives.py` 24 of 24, `books.py`, `mem.py`, `cellmix.py` 7 of 7 each, every exit code 0; largest differences 4.7e-05, 7.6e-08, 5.6e-06, 3.4e-07, the figures of the committed check files). `kinds6.py build` rebuilt the seven streams with no hash assertion failing; the six text streams passed theirs when loaded. Two stored singles were repeated and are equal bit for bit (ZA on py: going on 0.941786, nothing changing 1.438716; ZAM32x512 on c: 0.631920, 2.250529).

| system | order | mean at the end | mean of its singles | the life costs |
|---|---|---|---|---|
| ZA | 73129 | 1.0069 | 0.9427 | +0.0642 (+6.8%) |
| ZA | 93683 | 1.0160 | 0.9427 | +0.0733 (+7.8%) |
| ZA | 116089 | 1.0144 | 0.9427 | +0.0717 (+7.6%) |
| M | 73129 | 0.9460 | 0.9075 | +0.0385 (+4.2%) |
| M | 93683 | 0.9526 | 0.9075 | +0.0450 (+5.0%) |
| M | 116089 | 0.9522 | 0.9075 | +0.0447 (+4.9%) |
| MC | 73129 | 0.9381 | 0.9027 | +0.0354 (+3.9%) |
| MC | 93683 | 0.9432 | 0.9027 | +0.0405 (+4.5%) |
| MC | 116089 | 0.9433 | 0.9027 | +0.0406 (+4.5%) |

Orders: 73129: c, book, fonts, xml, elf_go, magic, table, pixels, py, elf_c, shakespeare, pyc. 93683: c, pyc, elf_go, py, xml, table, magic, fonts, elf_c, pixels, book, shakespeare. 116089: xml, table, magic, pixels, py, fonts, shakespeare, book, elf_c, pyc, c, elf_go. Cells at the end: about 8.79 million (ZA), 10.58 million (M and MC).

**Streams read late** (own stage less own single, later positions). Plain system, mean: text +0.0332, -0.0035, +0.0296; not text +0.0399, +0.0327, +0.0415 (orders 73129, 93683, 116089). Cut at the median over the positions where the plain loss is 0.02 or more (8, 6, 8 positions): M 0.74, 0.61, 0.68; MC 0.77, 0.62, 0.66.

**The end against the own stage** (plain system, mean of end less own stage): text +0.0065, +0.0492, +0.0204; not text +0.0651, +0.0759, +0.0681. At the end M is below the plain system on 12 of 12 tests in each order; MC is below the plain system on 12 of 12 and below M on 12 of 12 in each order.

**Predictions.**

| | Score | The figure that decides it |
|---|---|---|
| L1 | held | the plain system ends above the mean of its singles by 6.8%, 7.8%, 7.6% (3% or more in every order) |
| L2 | held | cut at the median with bounded kinds (M): 0.74, 0.61, 0.68 (half or more) |
| L3 | held | M below the plain system at the end on 12, 12, 12 of the twelve tests (ten or more) |
| L4 | held | cost of the life with bounded kinds lower than the plain system's by 40.0%, 38.6%, 37.6% (+0.0385 against +0.0642; +0.0450 against +0.0733; +0.0447 against +0.0717) |
| L5 | held | MC below M at the end on 12, 12, 12 of the twelve tests (nine or more) |
| L6 | failed | the streams that are not text lose more, not less, in all three orders: +0.0651, +0.0759, +0.0681 against +0.0065, +0.0492, +0.0204 for text |
| L7 | failed | the loss on a stream read late is smaller for text in all three orders: +0.0332, -0.0035, +0.0296 against +0.0399, +0.0327, +0.0415 |

**The hypothesis.** (a) holds (6.8%, 7.8%, 7.6%). (b) holds (0.74, 0.61, 0.68; the bar of refutation, under a quarter, is not approached). (c) holds (12 of 12 in every order). The hypothesis is not refuted on these three orders.

**Remarks (mine).** In the plain system py is far below its own single wherever it is read after c (own stage less single -0.184 in order 93683) and gains further after its own stage in the other two orders (end less own stage -0.210 and -0.169); these three entries pull the text means of L6 and L7 down, and both predictions would have to be read again without py before anything is said about kinds. I have not done that reading: it is not in the plan. The cost of the life is on three orders only and on one machine.

**Not run.** Nothing of section 4's runs was left out: six singles and nine lives ran, each with exit code 0 (wall 7 to 15 minutes a life, three at a time). As the plan states, not run: remembered trusts and the waking of bookmarks on twelve streams; the other two reported orders; the large system.
