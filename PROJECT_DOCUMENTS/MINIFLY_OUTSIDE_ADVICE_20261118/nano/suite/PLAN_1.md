# A test-bed of byte streams of different kinds, and the base system on it with one setting (20 November 2026)

**Authorisation and steer.** Mr. Xu, 20 November 2026 (in Chinese, newest first): "鉴于bitter lesson，执着于加法这个任务可能并非最聪明的做法，继续努力尝试通用的想法。"; before it "继续大胆尝试，小心求证。" and "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" (The coordinator's translation: "Given the bitter lesson, holding on to the task of addition may not be the cleverest thing to do; go on trying general ideas." "Go on trying boldly, and verify with care." "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas.") The task of this line was set by the coordinating session of the outside-advice work.

**Source.** `claude/friendly-darwin-8mmhfm` at 32b631d91; this line is the folder `nano/suite` on `claude/nano-suite-8mmhfm`. Copied into `ext/`: `nano/merge/merge.py`, `nano/table/gate.py`, `nano/table/score.py`, `nano/ref/tables.py` from that commit; changed in the copies: room for 128 values in the two kernels (`SMAX`, one assertion), the path of the test text in `score.py`; nothing else. Development level. No checker gate, seal or review is claimed; the checks named here are executed checks by the author of the programs.

## 1. Why

Every idea of the outside work was so far judged on two-digit sums and on one or a few English texts, with numbers set for each. This step builds a set of streams of different kinds and puts the base system on all of them with ONE setting, beside references that are general methods. It measures; it does not yet change the system.

## 2. The streams (`make_streams.py`; one file in `manifests/` for each: paths in order, sizes, hashes)

| name | what | bytes | values |
|---|---|---|---|
| shakespeare | the programme's test text (`nano/data/tinyshakespeare.txt`), identical value for value to `nano/table/score.py`'s stream | 1,115,394 | 65 |
| book | Moby Dick as this checkout has it (`minifly_v155_keys_dev_v1_20261026/data/melville-moby_dick.txt`), byte 13 removed | 1,220,066 | 81 |
| py | Python source: `/usr/lib/python3.12/**/*.py` in sorted order of path, joined, cut at 1,500,000 (70 files) | 1,500,000 | 96 |
| c | C headers: `/usr/include/**/*.h` the same way (171 files) | 1,500,000 | 105 |
| table | a data file with numbers in rows: the Unicode collation table shipped with perl (`allkeys.txt`), whole | 1,939,332 | 77 |
| xml | markup with short texts in many languages: the shared MIME database (`freedesktop.org.xml`), first 1,500,000 bytes | 1,500,000 | 128 |

The first nine tenths (to `int(0.9 * length)`) are read once, byte by byte; the last tenth is the test. Values: those met in the first nine tenths in ascending order; where there are more than 128 (xml only: 194 distinct bytes), the 127 most frequent there and one value for all others (16,257 bytes of xml, 1.1%; 1,416 of them in its last tenth). In c three bytes of the last tenth have a value not met before and go to a value of their own kind ("others").

*What I looked for as the fifth and sixth kinds, and what there was.* No word list (`/usr/share/dict` is empty); man pages are 0.3 MB in all. There were: the collation table (rows of hexadecimal numbers with names; taken); the MIME database (taken: its comments are translated into some fifty languages, interleaved through the file); the vim tutor in thirty-two languages (1.2 MB joined; not taken, because joined in order of path its last tenth would be Vietnamese and Chinese alone, never met before, and the measure with nothing changing would say nothing); vim's own documentation and perl's (English with markup; not taken, a third English text would add least). Nothing is downloaded.

*What was seen of the streams before this plan.* Tiny Shakespeare and the book are the development streams (the tenth of tiny Shakespeare has been scored some seventy-five times by the programme; `nano/table/PLAN_16.md` section 7). Of the other four I have seen: the sizes and counts above; about ten lines of the collation table's head and five rows of its middle, to know what it is; the first 2,500 bytes of each inside `suite.py check` (as numbers compared with the reference, not read). No loss on any of them has been computed.

## 3. The exact system, its size and its setting

`suite.py`: the system of `nano/merge/merge.py` in its prose form without classes ("runs of 0 to 6 bytes"): seven kinds of cell, a cell for each run of k = 0..6 bytes met, counting the byte that follows; a cell that has counted says log((c(v) + e) / (n + S e)); the first answering cell's branches compete on the sum over kinds of (trust of the kind at the cell's level of counts) times (message); each used trust changes by its step times (message at the byte that came less the expected message). Forty levels (ten by the count, four by the number of different bytes met). **One setting for all streams, the programme's prose numbers: e = 0.02, the trust's step 0.00005 (constant), trusts born at 0.25.** The number of values S is a parameter.

The program is new (a cell stores only the counts it has; cells are born when first met) and is checked two ways, both run before this plan: against a plain-Python reference in which every cell holds all S counts and the message is written out in full (`suite.py check`: ten cases over all options of the later steps, largest difference 2e-11, cells and counts equal); and against `gate.py` itself (`suite.py check_gate`: the same answers to 1e-15 on a prefix; on the whole of tiny Shakespeare the last tenth with nothing changing is **1.5226**, `gate.py`'s figure, with 459,129 cells and 872,748 counts above zero). Expected size on the other streams: 0.4 to 1.5 million cells, 1 to 3 million counts above zero, 280 trusts. Cost: 10 to 30 CPU-seconds a stream.

**Measures**, each on the last tenth, nats a byte, all bytes: with nothing changing; with everything going on (cells count and trusts move through the tenth; each byte answered before it is read); cells that have counted and counts above zero (at the scoring with nothing changing); CPU seconds.

**References** (`refs.py`), same split: xz -9e, bzip2 -9, gzip -9 on the mapped stream: size of the whole compressed less size of the first nine tenths compressed, in nats a byte of the tenth (these go on learning, so they stand beside "going on"; xz reaches the whole stream; bzip2 sorts blocks of 900,000 bytes apart, so the tenth is helped only by what shares its block; gzip reaches 32,768 bytes back); tables of the last k = 3..6 bytes with interpolated Kneser-Ney smoothing counted on the nine tenths (`nano/ref/tables.py`; these change nothing, so they stand beside "nothing changing"). On tiny Shakespeare the reference program was run once before this plan as a check of the copy: Kneser-Ney 1.7191, 1.5612, 1.5226, 1.5165 for k = 3..6, as `nano/ref/REPORT.md` has them; xz 1.750, bzip2 1.662, gzip 2.146.

## 4. The question and the predictions (mine, before any run on the five other streams)

**Question.** With the prose numbers unchanged, where does the base system stand on each kind of stream against methods that are general? There is no mechanism on trial in this step; the predictions say what I expect of a measurement.

| | Statement | Chance |
|---|---|---|
| A1 | going on, the base is below xz on both prose streams (shakespeare, book) | 0.90 |
| A2 | going on, xz is below the base on at least two of the four other streams (py, c, table, xml) | 0.45 |
| A3 | with nothing changing, the base is within 0.02 nats of the best Kneser-Ney table (k = 3..6) on both prose streams | 0.75 |
| A4 | with nothing changing, the base is below the best Kneser-Ney table on at least three of the four other streams | 0.45 |
| A5 | the gain from going on (nothing changing less going on) is larger on each of the four other streams than on tiny Shakespeare | 0.75 |
| A6 | on the collation table the loss with nothing changing is more than 1.5 times the loss going on (its rows are sorted, so its end is unlike its beginning) | 0.40 |

**What would count against the test-bed itself:** a stream where the last tenth is so unlike the rest that every method, the compressors included, does no better than its own order-0 cost; I would then say so and keep the stream only for the going-on measure.

## 5. Not run in this step

No idea is tried. No number is tuned (the setting is given). The systems with word cells (their ends are given by hand). Sums. Seeds: there are none; the system and the streams are deterministic.

## 6. Outcome (entered 20 November, after the runs; `res/*__base.json`, `res/ref__*.json`, `report_1.txt`)

Nats a byte on the last tenth. The base is 7 kinds, 280 trusts; one setting (e = 0.02, step 0.00005, born 0.25). The Kneser-Ney table changes nothing during the tenth and stands beside "nothing changing"; the compressors go on learning and stand beside "going on".

| stream | values | base, nothing changing | best Kneser-Ney table (k) | base, going on | xz -9e | bzip2 -9 | gzip -9 | cells | counts above zero | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|
| shakespeare | 65 | 1.5226 | 1.5165 (6) | 1.4136 | 1.7496 | 1.6621 | 2.1463 | 459,129 | 872,748 | 2.4 |
| book | 81 | 1.4676 | 1.4570 (6) | 1.4476 | 1.7876 | 1.7035 | 2.2479 | 531,809 | 990,711 | 3.0 |
| py | 96 | 1.4568 | 1.3974 (6) | 1.1503 | 1.1265 | 1.2079 | 1.3972 | 475,303 | 768,177 | 3.5 |
| c | 105 | 2.1222 | 2.0578 (6) | 1.0301 | 0.7244 | 0.9426 | 1.0363 | 420,192 | 658,473 | 3.4 |
| table | 77 | 1.7567 | 1.8000 (4) | 0.8141 | 0.3814 | 0.5976 | 0.8122 | 556,055 | 913,601 | 3.3 |
| xml | 128 | 0.8077 | 0.7893 (6) | 0.4570 | 0.4333 | 0.3845 | 0.6184 | 296,978 | 453,416 | 6.1 |

Half of all cells have been met once (43 to 65% by stream). A repair made during the runs, to the reference only: on the collation table the published discounts of the Kneser-Ney form fall outside (0, r) in two orders and give negative "probabilities"; `refs.py` now keeps each discount inside (0.05, r - 0.05). This also moved one discount on tiny Shakespeare (its four figures are unchanged to the fourth decimal) and none on the other four streams.

**Predictions scored.** A1 held (prose: the base going on is 0.34 below xz on both). A2 held, and more than I gave it: xz is below the base on all four other streams (by 0.02, 0.31, 0.43, 0.02). A3 held (0.006 and 0.011 above the best table). **A4 failed**: with nothing changing the base is below the best Kneser-Ney table on one of the four other streams (the collation table), not three; it is 0.06, 0.06 and 0.02 above on py, c and xml. A5 held (the gain from going on: 0.11 on tiny Shakespeare; 0.31, 1.09, 0.94, 0.35 on the four). A6 held (2.2 times).

**My readings.**
1. *The prose numbers were hiding a kind of stream the system is poor at.* On prose the base is well below every compressor; on source code, rows of numbers and markup the strongest compressor is below it everywhere, by up to 0.43 nats. What a compressor has and the base has not is the use of a long repeat and fast adaptation to what was just read.
2. *On streams joined from files or sorted by key, the measure with nothing changing is mostly a measure of how unlike the end is to the rest.* On c and table every frozen method stands near 2 nats where going on gives 1.0 and 0.8 and xz 0.7 and 0.4. For such streams the going-on measure is the informative one; I report both everywhere.
3. *With nothing changing the base is not ahead of a well-smoothed table of the same contexts* on any stream but one. Its advantage over the table is that it goes on.
4. The test-bed is not broken by the criterion of section 4: on every stream every method is far below its order-0 cost.
