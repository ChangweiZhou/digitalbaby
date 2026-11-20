# The best general system put together, and the last table (20 November 2026)

**Authorisation and steer.** Mr. Xu's words as quoted at the head of `PLAN_1.md` (in Chinese: "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。"; "继续大胆尝试，小心求证。"; "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。"). The coordinating session asked for one table at the end and for the best general system. **Source.** This branch at the commit that holds this file. Written while runs of `PLAN_2.md` (pool, since, classes, made kinds on the last streams) and `PLAN_4.md` (sums) were still going, and before any run of a system put together. Development level; no gate, seal or review is claimed.

## 1. How the system is put together (laid down before the runs, so that nothing is picked afterwards)

An idea goes in if it was kept by the rule of `PLAN_2.md` on the going-on measure in its own step, and if it lives in this line's kernel. Two systems:

* **Z, with no more cells than the base**: runs of 0 to 6 bytes; the second answering cell with parts by kind and byte (step 2, i); three recalling senders (step 3 and its variant); for each run a sender that says what the run's cell counted last (step 5); the one rule for the step at eta = 0.03 (step 4); e = 0.02, runs born at 0.25, everything else at nothing. `kinds=runs+last gate=1 recall=3 rule=2 eta=0.03`. Its figures so far, each idea alone on the one before, are in the plans; together they have not been run.
* **ZP, Z with the cells of step 2 (ii) that are kept by the rule** when its last runs are in: the pool, the since-cells, both or neither; said in the outcome. Five to eight times the cells.

Not in either: classes found by search and kinds made while reading (they live in `nano/merge`'s program, which holds every count of every cell and has no second cell, recalling sender or rule; whether they are kept is said in `PLAN_2.md`'s outcome; putting them into this kernel is work for another day); birth only where surprised (a saving, not a gain).

The answer of a system is its second cell's. One setting on all six streams. The four streams other than the two prose streams have by now been scored by this line about a dozen times each, once for each idea; nothing was chosen on them at any point (every number was fixed by an earlier line, by the coordinating session, or on the two prose streams), and that remains so here.

## 2. Predictions (mine)

| | Statement (going on) | Chance |
|---|---|---|
| F1 | Z is below xz on all six streams | 0.80 |
| F2 | Z is below bzip2 and gzip on all six streams | 0.90 |
| F3 | Z's gain over the base (first cell, `PLAN_1.md`) is at least 0.30 nats on c and on table and at least 0.05 on each prose stream | 0.65 |
| F4 | the ideas add up less than fully: Z's gain over the base is under the sum of the gains each idea showed in its own step, on at least five streams | 0.80 |
| F5 | with nothing changing, Z is above the base on at least four streams (the ideas that hear what was just read are tuned by the system for a present that then stands still) | 0.80 |
| F6 | if the pool is kept: ZP is below Z going on, on every stream, by at least 0.01 | 0.50 |

## 3. What the comparison with compressors can and cannot say

xz, bzip2 and gzip are the general compressors this machine has; nothing may be downloaded. They are not the strongest of their trade: compressors that mix the opinions of many contexts (the family this system resembles most) are not installed, and on text they are known to do clearly better than xz. "Below xz" therefore means "past the dictionary coders", not "at the state of the art of compression". The system also stores far more than a compressor's working memory per byte read, and is slower.

## 4. Cost, not run

Z: about 8 CPU-seconds a stream. ZP: 2 to 5 CPU-minutes and 1 to 2 GB. Not run: Z on sums (its recalling and last-byte senders are laid down for the kinds of runs; sums are in `PLAN_4.md` on their own kinds); any change of a number.

## 5. Addition before the runs of ZP (entered 20 November, when the last runs of `PLAN_2.md` (ii) were in and before any run of ZP)

Both the pool and the since-cells are kept by the rule on the going-on measure (`PLAN_2.md`, outcome: each lowers all five kinds and harms none), so **ZP is Z with both** (`kinds=runs+last+since+pool`; the last-byte senders stay on the seven runs only). Since this plan was written, `PLAN_7.md` found the trusts by age kept on both measures; by the rule of section 1 they go in. I therefore run both forms and report both: ZP as laid down here, and **ZPA, ZP with `age=1`** (and ZA = Z with age is already in `res/*__Zage.json`). Prediction F7: going on, ZPA is below ZP on at least four streams (chance 0.55). One run each, three at a time.
