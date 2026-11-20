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

## 6. Outcome (entered 20 November, after the runs; `res/*__Z.json`, `*__ZP.json`, `*__ZPA.json`, `*__Zage.json`; `report_final.txt` has every system on every stream, both measures, sizes and cost)

Going on, second cell, nats a byte on the last tenth:

| system | shakespeare | book | py | c | table | xml | cells (millions) | CPU s |
|---|---|---|---|---|---|---|---|---|
| base (first cell) | 1.4136 | 1.4476 | 1.1503 | 1.0301 | 0.8141 | 0.4570 | 0.30 to 0.56 | 2 to 6 |
| Z | 1.3389 | 1.3910 | 0.9496 | 0.7068 | 0.3415 | 0.2844 | the same | 4 to 7 |
| ZA (Z with trusts by age, `PLAN_7.md`) | 1.3344 | 1.3894 | 0.9358 | 0.6908 | 0.3431 | 0.2846 | the same | 5 to 7 |
| ZP (Z with since and pool cells) | 1.3497 | 1.3972 | 0.9677 | 0.5970 | 0.2494 | 0.2790 | 1.8 to 4.0 | 111 to 181 |
| ZPA (ZP with trusts by age) | 1.3459 | 1.3959 | 0.9582 | 0.5921 | 0.2479 | 0.2777 | the same | 112 to 191 |
| xz -9e | 1.7496 | 1.7876 | 1.1265 | 0.7244 | 0.3814 | 0.4333 | | |

With nothing changing (second cell): Z 1.5470, 1.4580, 1.4661, 2.2396, 2.0140, 0.8431; ZP 1.5526, 1.4728, 1.5594, 2.5663, 2.6307, 0.9723; the base 1.5226, 1.4676, 1.4568, 2.1222, 1.7567, 0.8077. The first cell of ZPA with nothing changing: 1.4868, 1.4322, 1.3852, 2.1363, 1.7228, 0.7410, below the base on five streams and 0.014 above on c.

**Predictions scored.** F1 held: Z is below xz on all six streams (by 0.41, 0.40, 0.18, 0.02, 0.04, 0.15). F2 held. F3 held (0.32 and 0.47 on c and table; 0.075 and 0.057 on prose). **F4 failed, and was badly put**: the ideas were measured each on top of the one before, so their gains add up to Z's by construction except for the two further recalling senders, which give less inside Z than in their own step on three streams and slightly more on three. F5 held (Z is above the base with nothing changing on five streams). **F6 failed**: the since and pool cells lower Z by 0.110 and 0.092 on c and table and by 0.005 on xml, and **raise it on the two prose streams and py (+0.011, +0.006, +0.018)**, though each of them lowered the base on every stream. F7 held: age lowers ZP on all six (0.001 to 0.010).

**There is no one best system.** With no more cells than the base, ZA is the best on prose and py and is below xz everywhere. With three to twelve times the cells, ZPA is far better on c and table (0.59 and 0.25 against xz's 0.72 and 0.38) and slightly worse than ZA on prose and py.

**My reading of F6, not tested.** The since and pool cells were kept as additions to the first cell at the constant step. In ZP they stand under the rule of `PLAN_4.md`, whose number was chosen on a system of eleven senders; with 170 to 230 senders each moving its trust by the same rule, the second cell hardly improves on the first (py: 0.9677 against 0.9754) where in Z it gains 0.05. The same shape appeared on sums with 256 kinds and the prose smoothing. If that is the cause, the rule's one number is not free of the number of senders, and the claim of `PLAN_4.md` must carry that limit. Not run: the rule with its number divided by the root of the senders that speak; ZP at the constant step.
