# Line `nano/search`, second plan: the frozen setting on the tasks (20 November 2026)

**Authorisation and steer.** Mr. Xu's steer, as quoted to this session by the coordinating session, exactly: "我们接受bitter lesson, 因此你可以尝试，但避免spot fix, ad hoc tricks等等在其他的任务上无法迁移的手段。多尝试新想法。" (the coordinating session's translation: "We accept the bitter lesson, so you may try things, but avoid spot fixes, ad hoc tricks and the like: means that do not transfer to other tasks. Try more new ideas."). **Source.** `claude/nano-search-8mmhfm` at the commit before this one (program and check as committed with `PLAN_1.md`, addition 1). Development level. No checker gate, seal or outside review is claimed. Written before any run on a reported seed, on the books or on the last tenth of tiny Shakespeare.

## 1. The frozen system and setting (the same for every task below)

The system of `PLAN_1.md`, section 2 and addition 1. Numbers: K = 128 kinds alive; R = 8192 bytes a period; `prot` = 2 periods; `nrep` = 16 kinds withdrawn and born a period (half of them changes of kinds that earned); B = 131,072 bytes kept; shrinking step, eta = 0.03; e = 2; no cap on a cell's weight; 16 places back, guesses 1 to 4 ahead from two layers; store 2^22 cells. Nothing is chosen again on any task. The only thing that differs between texts is S (14 or 65).

## 2. The tasks and the runs

1. **Two-digit sums**, five reported seeds (73129, 93683, 116089, 406364, 545353), never-shown pairs, own answer bytes, whole answer right, mean of the last three readings and the last: at 40,000 lines (readings every 5,000) and at 400,000 lines (readings every 10,000).
2. **The life** addition 40,000 lines then subtraction 20,000, nothing shown again: addition before and after; subtraction on the clean score (held-out pairs whose written line no shown pair writes). Note: kinds born early in subtraction count over the kept bytes, which then still hold up to 13,000 lines of addition.
3. **Prose**: tiny Shakespeare's stream read once, then the three books (first nine tenths of each), each book's last tenth scored right after its stream; the last tenth of tiny Shakespeare scored once at the end (reported, no claim: it is worn).
4. **Three-digit sums** (`aaa+bbb=cccc`, mine): 2,000 pairs held out of the million, 500 of them scored, lines drawn from all the other pairs; 40,000 and 400,000 lines; five reported seeds.
- Beside each: **the control** (the same 128 random kinds born once, `nrep` = 0; the 400,000-line control for two-digit sums only), and **the number tried halved and doubled** (`nrep` = 8 and 32) on two-digit sums at 40,000 lines and on prose. For orientation on prose, in this program and not a claim: six hand-given kinds (runs of 1 to 6 bytes), no search.
- The other lines' hand-built figures, as they report them: sums 77 to 81% with at most four of the last nine bytes, 91% with look-ahead; prose T0 1.5023, 1.3645, 1.4456 and T2 1.4802, 1.3162, 1.4109.

## 3. Hypothesis (the coordinating session's), what refutes it, predictions (mine, before the runs)

The hypothesis as in `PLAN_1.md`, section 4. It is **refuted** if either text is not better than unsearched random kinds or no single setting does both; each of its parts is scored separately.

| | Statement | Chance |
|---|---|---|
| P1 | two-digit sums at 40,000 lines, median of five seeds of the last-three mean, at least 85% | 0.15 |
| P2 | search above the control on two-digit sums on every seed | 0.90 |
| P3 | below T0 by 0.01 nats on each of the three books | 0.05 |
| P4 | search below the control on each book | 0.95 |
| P5 | `nrep` 32 better than 16 on sums (median) and on all three books | 0.15 |
| P6 | `nrep` 16 better than 8 on sums (median) and on all three books | 0.40 |
| P7 | two-digit sums at 400,000 lines at least 85% (median) | 0.50 |
| P8 | three-digit sums: at least 50% at 40,000 lines; at least 70% at 400,000 | 0.25; 0.30 |
| P9 | the life: addition after subtraction at least 85% of its level before (median); subtraction clean score at least 60 | 0.50; 0.40 |
| P10 | on at least three of five seeds a kind made of digits of the numbers together with a guess is among the five that earned most on two-digit sums | 0.60 |
| H | the whole hypothesis | 0.02 |

**Size and cost.** Store 2^22 cells and 131,072 kept bytes with eight guesses each, for every run; about 150 CPU-minutes in all. **Not run:** a fifth task; more seeds; another setting; a larger store.
