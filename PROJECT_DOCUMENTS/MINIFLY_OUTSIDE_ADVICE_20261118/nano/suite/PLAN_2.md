# The day's general ideas, one at a time on the base, the same setting on every stream (20 November 2026)

**Authorisation and steer.** Mr. Xu's words as quoted at the head of `PLAN_1.md` (in Chinese: "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。"; "继续大胆尝试，小心求证。"; "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。"; the coordinator's translations are there). The list of ideas and their order were set by the coordinating session. **Source.** This branch at 64fea94; `nano/merge/merge6.py` and `merge10.py` copied to `ext/` from `claude/friendly-darwin-8mmhfm` at 32b631d91, unchanged. Written before any run of these ideas on the streams, with two exceptions said in section 4. Development level; no gate, seal or review is claimed.

## 1. The evidence this follows

`PLAN_1.md`, outcome: with one setting the base is below every compressor on prose and above xz on the four other streams. Each idea below was found, and its numbers chosen, on English prose or on two-digit sums. This step asks of each whether it is general.

## 2. The rule (fixed here, before the runs)

The six streams are of five kinds: prose (shakespeare and book), py, c, table, xml. An idea **lowers** the loss on a kind if the loss is lower by at least 0.001 nats a byte (on prose: on both streams). An idea is **kept** if it lowers the loss on at least three kinds and harms no stream by more than 0.005. The rule is applied to the measure with everything going on (the one the compressors stand beside; `PLAN_1.md`, reading 2) and the verdict on the measure with nothing changing is given beside it; where the two disagree I say so and do not call the idea kept without the qualification. Every comparison is inside one program and one run where that is possible (the second cell beside the first in the same run), otherwise against the base of the same program. Cost is given as cells that have counted, numbers stored (counts above zero and parts used) and CPU seconds.

Nothing is chosen in this step: every idea runs with the numbers its own line fixed. So the four other streams are opened for the claim directly, once for each idea.

## 3. The ideas, exactly

All on the base of `PLAN_1.md` (runs of 0 to 6 bytes; e = 0.02, step 0.00005, born 0.25; 0.3 to 0.56 million cells).

* **(i) the second answering cell, parts by kind and byte** (`suite.py gate=1`; `nano/table/gate_all.py nolevel=1`): for each of the last six delays a part of the trust indexed by (kind, byte at that delay), born at nothing, step 0.00005; the second cell's branch sums are the first cell's plus (sum of the six parts) times the message. Parts: 6 x 7 x S, at most 5,376.
* **(ii-a) cells that time how long ago each byte value last came** (`kinds=runs+since`; `nano/table/place.py` arm B): for each value v a kind keyed on (bytes since v last came, capped at 63; the byte just read); S kinds more, trusts born at nothing.
* **(ii-b) the pool** (`kinds=runs+pool`; `place.py` pool): a kind for every set of two or three places among the last eight, less the two that are runs; 82 kinds more, trusts born at nothing. (ii-a) and (ii-b) are judged apart, and together once (`runs+since+pool`).
* **(iii) classes found by search, the frozen setting of `nano/merge`** (`suite_merge.py arm=classes`): 32 pairs, e_J = 0.02, class cells beside their members' cells, found anew whenever the bytes read have doubled. Compared with `arm=none` of the same program.
* **(iv) kinds made while reading, born with the pooled past** (`arm=F`: `merge10.py` with `gc=1 full=1 keep=2`, as `nano/merge/PLAN_11.md` lays down; the program as this checkout has it).
* **(v) a cell is born only where the first answering cell gave the byte that came less than one chance in two** (`birth=0.5`; `nano/table/PLAN_19.md`). This is an idea for the store: it is judged by the cells saved against the nats lost, not by the rule of section 2.

Not used: the cells that step once a word; anything that names a byte value, a place or a kind of text.

## 4. Checks and what has been seen

`suite.py check` covers every option used here against the plain-Python reference. `suite_merge.py arm=none` on tiny Shakespeare gives 1.5226 and 1.4136 with 459,129 cells, the figures of `suite.py`. Seen before this plan, on the development stream only: the second cell *with* the level on tiny Shakespeare (1.5058 against 1.5226, in `check_gate`); and `arm=F` on the first 200,000 bytes of tiny Shakespeare, run to see that the program runs (1.5734 frozen; I have no base figure for that prefix and draw nothing from it).

## 5. Hypothesis, what refutes it, predictions (mine)

*Hypothesis.* The ideas that add a general means of hearing more of the context (i, ii-b) are kept by the rule; the ideas whose worth came from the make of English or of sums (ii-a: line lengths; iii, iv: classes of letters and of digit pairs) are not. *Refuted if* neither (i) nor (ii-b) is kept, or if both (iii) and (ii-a) are.

| | Statement (going on unless said) | Chance |
|---|---|---|
| B1 | (i) is kept | 0.65 |
| B2 | (ii-a) since is kept | 0.30 |
| B3 | (ii-b) the pool is kept | 0.55 |
| B4 | (iii) classes are kept | 0.30 |
| B5 | (iv) F is kept | 0.15 |
| B6 | (iv) is above (iii) on tiny Shakespeare with nothing changing | 0.60 |
| B7 | (v) saves at least 15% of the cells on every stream | 0.70 |
| B8 | (v) costs more than 0.005 nats going on, on at least one stream | 0.60 |
| B9 | no single idea of the five closes more than a third of the base's distance to xz on c and on table (1.0301 against 0.7244; 0.8141 against 0.3814) | 0.80 |
| B10 | on the measure with nothing changing, the verdict of the rule differs from the going-on verdict for at least one of (i), (ii-a), (ii-b), (iii) | 0.50 |

## 6. Cost, what is not run

(i), (v): about 10 CPU-seconds a stream. (ii): 1 to 4 CPU-minutes a stream and up to 1 GB (89 to 217 kinds). (iii): 3 to 10 CPU-minutes a stream, 1 to 2 GB (the copied program holds every count of every cell); (iv): 10 to 20. At most two large jobs at a time. Not run: combinations other than since with pool (the best general system is put together in a later step from what is kept); other numbers for any idea; the second cell with the level (it is not the form asked; it is in the kernel and its check).

## 7. Outcome (entered 20 November, after the runs; `res/*__gate1.json`, `*__since.json`, `*__pool.json`, `*__since_pool.json`, `*__birth.json`, `*__merge_*.json`; `report_2.txt`)

Nats a byte on the last tenth; the difference from the base of the same program in brackets (first answering cell against first answering cell, except (i), which is the second cell against the first in the same run). `nano/merge`'s program without classes gives this line's base to the fourth decimal on all six streams, with the same cells. The third row of each idea: cells as a multiple of the base's; numbers stored as a multiple; CPU seconds of the run.

| idea | measure | shakespeare | book | py | c | table | xml | kinds lowered | largest harm | by the rule |
|---|---|---|---|---|---|---|---|---|---|---|
| (i) second cell, parts by kind and byte | going on | 1.3876 (-0.0260) | 1.4184 (-0.0292) | 1.1228 (-0.0275) | 0.9470 (-0.0831) | 0.6944 (-0.1196) | 0.4398 (-0.0172) | 5 of 5 | -0.0172 | KEPT |
| (i) second cell, parts by kind and byte | nothing changing | 1.4978 (-0.0248) | 1.4388 (-0.0289) | 1.4349 (-0.0220) | 2.1137 (-0.0085) | 1.8376 (+0.0809) | 0.7915 (-0.0161) | 4 of 5 | +0.0809 | not kept |
| (i) second cell, parts by kind and byte | cells x; numbers stored x; CPU s | 1.00; 1.00; 4 | 1.00; 1.00; 4 | 1.00; 1.00; 5 | 1.00; 1.00; 6 | 1.00; 1.00; 5 | 1.00; 1.00; 5 | | | |
| (ii-a) since | going on | 1.4022 (-0.0114) | 1.4316 (-0.0160) | 1.1346 (-0.0157) | 0.8947 (-0.1353) | 0.5981 (-0.2160) | 0.4033 (-0.0537) | 5 of 5 | -0.0114 | KEPT |
| (ii-a) since | nothing changing | 1.5110 (-0.0116) | 1.4518 (-0.0159) | 1.4451 (-0.0117) | 2.1619 (+0.0397) | 1.9521 (+0.1954) | 0.7783 (-0.0293) | 3 of 5 | +0.1954 | not kept |
| (ii-a) since | cells x; numbers stored x; CPU s | 1.46; 3.40; 32 | 1.44; 2.94; 44 | 1.84; 5.38; 78 | 2.02; 7.20; 80 | 1.25; 2.67; 53 | 3.05; 8.39; 86 | | | |
| (ii-b) pool | going on | 1.4015 (-0.0120) | 1.4345 (-0.0131) | 1.1036 (-0.0467) | 0.8402 (-0.1898) | 0.5512 (-0.2629) | 0.4075 (-0.0495) | 5 of 5 | -0.0120 | KEPT |
| (ii-b) pool | nothing changing | 1.5112 (-0.0114) | 1.4557 (-0.0119) | 1.4139 (-0.0429) | 2.1724 (+0.0502) | 1.8016 (+0.0448) | 0.7550 (-0.0527) | 3 of 5 | +0.0502 | not kept |
| (ii-b) pool | cells x; numbers stored x; CPU s | 4.78; 12.91; 62 | 4.51; 11.56; 74 | 7.23; 15.51; 80 | 8.51; 17.87; 64 | 2.97; 6.80; 62 | 10.35; 18.30; 53 | | | |
| (ii) since and pool together | going on | 1.3906 (-0.0230) | 1.4182 (-0.0294) | 1.0936 (-0.0567) | 0.7950 (-0.2350) | 0.5003 (-0.3138) | 0.3817 (-0.0753) | 5 of 5 | -0.0230 | KEPT |
| (ii) since and pool together | nothing changing | 1.5010 (-0.0216) | 1.4403 (-0.0273) | 1.4098 (-0.0470) | 2.1688 (+0.0465) | 1.9737 (+0.2170) | 0.7473 (-0.0603) | 3 of 5 | +0.2170 | not kept |
| (ii) since and pool together | cells x; numbers stored x; CPU s | 5.24; 15.31; 108 | 4.95; 13.49; 137 | 8.07; 19.89; 175 | 9.52; 24.07; 168 | 3.21; 8.47; 131 | 12.40; 25.68; 160 | | | |
| (iii) classes, frozen setting | going on | 1.4003 (-0.0132) | 1.4332 (-0.0143) | 1.1324 (-0.0179) | 0.9397 (-0.0904) | 0.6373 (-0.1768) | 0.4484 (-0.0086) | 5 of 5 | -0.0086 | KEPT |
| (iii) classes, frozen setting | nothing changing | 1.5116 (-0.0110) | 1.4536 (-0.0141) | 1.4300 (-0.0269) | 2.1532 (+0.0310) | 1.7906 (+0.0339) | 0.8019 (-0.0057) | 3 of 5 | +0.0339 | not kept |
| (iii) classes, frozen setting | cells x; numbers stored x; CPU s | 5.13; 6.00; 106 | 5.25; 6.30; 130 | 6.85; 7.29; 179 | 7.14; 7.52; 144 | 3.89; 4.42; 134 | 5.90; 6.39; 99 | | | |
| (iv) made kinds, F | going on | 1.3993 (-0.0143) | 1.4328 (-0.0148) | 1.1250 (-0.0253) | 0.8942 (-0.1359) | 0.6210 (-0.1931) | 0.4494 (-0.0076) | 5 of 5 | -0.0076 | KEPT |
| (iv) made kinds, F | nothing changing | 1.5099 (-0.0127) | 1.4530 (-0.0146) | 1.4307 (-0.0262) | 2.1454 (+0.0232) | 1.7406 (-0.0162) | 0.8056 (-0.0021) | 4 of 5 | +0.0232 | not kept |
| (iv) made kinds, F | cells x; numbers stored x; CPU s | 5.33; 6.61; 124 | 4.70; 6.20; 165 | 6.97; 8.89; 782 | 6.45; 9.07; 746 | 2.30; 3.37; 311 | 4.23; 7.29; 437 | | | |
| (iv) F against (iii) | going on | 1.3993 (-0.0010) | 1.4328 (-0.0005) | 1.1250 (-0.0074) | 0.8942 (-0.0455) | 0.6210 (-0.0163) | 0.4494 (+0.0010) | 3 of 5 | +0.0010 | KEPT |
| (iv) F against (iii) | nothing changing | 1.5099 (-0.0017) | 1.4530 (-0.0005) | 1.4307 (+0.0007) | 2.1454 (-0.0078) | 1.7406 (-0.0500) | 0.8056 (+0.0037) | 2 of 5 | +0.0037 | not kept |
| (iv) F against (iii) | cells x; numbers stored x; CPU s | 1.04; 1.10; 124 | 0.90; 0.98; 165 | 1.02; 1.22; 782 | 0.90; 1.21; 746 | 0.59; 0.76; 311 | 0.72; 1.14; 437 | | | |
| (v) birth under one half | going on | 1.4172 (+0.0036) | 1.4518 (+0.0042) | 1.1623 (+0.0120) | 1.0422 (+0.0121) | 0.8141 (+0.0000) | 0.4587 (+0.0017) | 0 of 5 | +0.0121 | not kept |
| (v) birth under one half | nothing changing | 1.5272 (+0.0046) | 1.4719 (+0.0042) | 1.4651 (+0.0083) | 2.1740 (+0.0518) | 1.7650 (+0.0082) | 0.8091 (+0.0015) | 0 of 5 | +0.0518 | not kept |
| (v) birth under one half | cells x; numbers stored x; CPU s | 0.78; 0.88; 3 | 0.82; 0.90; 4 | 0.75; 0.84; 5 | 0.75; 0.84; 5 | 0.68; 0.80; 6 | 0.75; 0.83; 5 | | | |

**By the rule, on the going-on measure, every idea of the list is kept**: the second cell with parts by kind and byte, the since-cells, the pool, classes found by search, kinds made while reading. Each lowers all five kinds of stream and harms none; each gains several times more on c and table than on prose. **On the measure with nothing changing none is kept**: each harms c or table (or both) by 0.02 to 0.20. Birth only where surprised is, as laid down, a matter of the store: 18 to 32% fewer cells for 0.000 to 0.012 nats going on (0.002 to 0.052 with nothing changing).

**My hypothesis is refuted.** I had written that the ideas whose worth came from the make of English or of sums (since; classes; made kinds) would not be kept, and that I would be refuted if both classes and the since-cells were. Both are, on every kind of stream. What I took for a property of English (line lengths for the since-cells, classes of letters for the search) is on these streams a general means: a cell keyed on how long ago a value came is a cell for position in a row, a line or a tag; classes of contexts exist in source code and in rows of numbers as they do in prose.

**Predictions scored** (going on unless said). B1 held. B2 held (I gave it 0.30). B3 held. B4 held (0.30). B5 held (0.15). **B6 failed**: with nothing changing on tiny Shakespeare the made kinds have 1.5099 against the classes' 1.5116. B7 held (18 to 32% of cells saved). B8 held (0.012 on py and on c). **B9 failed**: the pool closes 62% and 61% of the base's distance to xz on c and table, the since-cells 44% and 50%. B10 held, for all four. Five of my ten predictions were given under one chance in two; three of those five came true. My sense of what is specific to English was wrong.

**Cost.** The second cell is free (at most 5,376 parts). The since-cells cost 1.3 to 3 times the cells and 3 to 8 times the numbers; the pool 3 to 10 times the cells and 7 to 18 times the numbers; classes 4 to 7 times the cells and 100 to 180 CPU-seconds; made kinds the same store and 120 to 780 CPU-seconds (the base: 2 to 6). Per nat gained the since-cells are the cheapest of the four; made kinds add 0.000 to 0.046 to classes going on (0.046 on c, 0.016 on table, under 0.008 elsewhere, 0.001 worse on xml) for up to five times the CPU.

**My readings.** (1) The rule as I fixed it separates nothing among these ideas: all pass going on, all fail with nothing changing. What it does show is the split between the two measures, which is a property of the streams (c and table end unlike they begin), not of the ideas: with nothing changing, whatever learns more from the nine tenths is more wrong about a tenth that is unlike them. On the four streams whose end is like their beginning (the two prose streams, py, xml) every idea but birth is lower on both measures. (2) Ideas chosen on one English text did transfer. That is evidence against the fear that the programme's prose line was fitting English; it is not evidence about sums, where the per-text numbers differ (`PLAN_4.md`).
