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
