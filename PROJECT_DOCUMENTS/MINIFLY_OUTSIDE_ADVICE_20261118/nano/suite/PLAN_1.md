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
