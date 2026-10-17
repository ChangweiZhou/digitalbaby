# Why a byte-in/byte-out learner memorises addition but does not generalise it

**Development diagnostic only.** Not formal science, no preregistration, no adoption, no registry change. It used
the three technical worlds of `addition_core_lineage_comparison_dev_v1_20261015` (202610143901–03), not its 16
formal worlds. 135 runs (3 worlds × 3 seeds per condition), about 1.5 CPU-hours.

## Question

The recurrent references (KEEP, G62) learn all taught single-digit sums but get 0–5% of the unseen ones. Is the
cause missing information, the decimal output (carries), an inability to represent order, or something else?

## Setup

Same task and split as the lineage comparison: operands 0–9, raw bytes `"3+8="` in, `"11"` + EOS out. HELD = 10
commutative orbits (20 ordered facts) never shown in either order. Train = OLD + NEW (80 ordered facts). The learner
is a KEEP-sized GRU: a byte embedding of 24, width 64, about 40k parameters, and a softmax over 256 bytes + EOS. It
reads BEGIN, the prompt and REPLY, then emits answer bytes. Scoring is exact own-output match on HELD.

One factor was changed at a time:

| Factor | Registered task | Diagnostic alternatives |
|---|---|---|
| input | digits `3+8=` | tally marks `\|\|\|+\|\|\|\|\|\|\|\|=` (quantity carried by the stream) |
| output | decimal `11` | one arbitrary byte per sum, `chr(65+s)` (no carry / mod-10) |
| training | online batch-1 RMSprop, 16,384 records (lineage-like dose) | full-batch AdamW, 20k steps, weight decay 0 or 1 |
| experience | addition facts only | + 20 counting records (`3=`→`\|\|\|`, `\|\|\|=`→`3`); control: same records with the digit↔tally pairing permuted |
| width | 64 | 32, 16 |

The tally and symbol formats only localise the difficulty. They are not proposed task encodings.

## Results (`results/SUMMARY.json`, `results/ANALYSIS.json`)

| Condition (digits in, decimal out unless stated) | Taught | Unseen (HELD), final | range over 9 runs |
|---|---:|---:|---|
| online, lineage-like dose | 99% | **6%** | 0–15% |
| full batch, wd 0 | 100% | 6% | 0–15% |
| full batch, wd 0, width 32 / width 16 | 100% / 100% | 16% / 4% | 0–30% / 0–10% |
| full batch, wd 0, one-symbol output | 100% | 0% | 0–0% |
| **tally input**, decimal output | 100% | **98%** | 95–100% |
| **tally input**, one-symbol output | 100% | **100%** | 100% |
| full batch, **wd 1** | 100% | 44% | 15–90% |
| full batch, wd 1, one-symbol output | 100% | 17% | 0–65% |
| online + counting / + permuted counting | 100% / 100% | 17% / 7% | 0–40% / 0–20% |
| full batch wd 1 + counting / + permuted counting | 100% / 100% | **65%** / 52% | 25–95% / 20–80% |

The online, lineage-like dose reproduces the lineage KEEP outcome (0–5% unseen). Rerunning one stage-1 run under
the final code gave identical curves and outputs.

## What the results rule out

1. **Missing information.** Fit "answer = u(first digit) + v(second digit)" to the 80 taught facts only. It predicts
   all 20 unseen facts exactly in all three worlds (design rank 19). The data determine every held answer.
2. **Decimal output and carries.** Tally input with decimal output reaches 98%: the network counts and formats the
   carry correctly. With digit input, decimal output generalises *better* than one arbitrary symbol per sum
   (+27 pp at wd 1, 8 of 9 paired runs better). The shared digit bytes help rather than hurt.
3. **No sense of order.** The state after reading `a+` is ordered by `a` in every digit run: the first principal
   component's rank correlation with `a` is |ρ| ≈ 0.9–1.0, including the runs that memorise.
4. **Too much capacity.** At width 16 (12.6k parameters) the network still memorises all 80 facts and gets 4% unseen.

## What the results point to

The learner gets the *shape* of the number line but not its *precision*, and it never ties the two operand slots
to one meaning:

* **Errors are near misses.** On the registered format, 76–78% of wrong unseen answers are off by exactly one and
  most of the rest by two. They are balanced over and under (online: 89 over, 80 under). They cross the carry
  boundary consistently (9+0 → `10`).
* **The two slots disagree.** A memorising net gives `a+b` and `b+a` the same answer for only 48–50% of unseen
  orbits. Agreement rises with generalisation: 73% with weight decay, 81% with weight decay + counting.
* **Generalisation comes first and is then overwritten.** Mean unseen accuracy peaks early (21–24% at step 500–3,072;
  single runs reach 60%) while taught accuracy is still climbing. It then falls as the last taught facts are fitted.

**Interpretation.** To answer an unseen fact, the learner must treat each digit symbol as one quantity, the same in
both slots, evenly spaced and combined additively. The data support exactly that explanation, but a generic
recurrent learner has a cheaper one: fit each taught fact with small local corrections. Once the taught facts are
right, the loss is about zero and nothing rewards making the explanation reusable. The leftover imprecision is
invisible on taught facts and shows up as ±1 errors on unseen ones. This is an inductive-bias problem, not a data
problem.

Evidence that the remedy is a reuse pressure, not extra structure:

* **Tally input (98–100%).** The quantity is built by repeating one transition per mark, so the same weights produce
  every number in both slots. Weight sharing over time enforces the reuse. *Caveat:* tally addition reduces to
  counting all marks, so this also removes the composition step. It shows the bottleneck is upstream of output
  formatting; it is not a fair test of composing digit symbols.
* **A generic simplicity pressure (weight decay).** +38 pp, 8 of 9 paired runs better, but bimodal (15–90%).
* **Counting experience, as data rather than wiring.** +11 pp online (7 better, 1 worse) and +21 pp with weight
  decay (8 better, 1 worse). It beats the permuted control in 6–7 of 9 pairs. The best runs reach 95%. It helps but
  does not settle the problem, and errors stay mostly ±1.

## Relation to the earlier results

* **No transfer from addition to multiplication.** STREAM-TRANSFER found untaught × at 0–2.7%, and multiplication
  training erased untaught +. A digit-quantity representation that both operations could reuse is never formed, so
  there is nothing to transfer. The savings result (correct OLD speeds NEW: 100% vs 17.5%) fits partial structure,
  such as order, without the precision needed for zero-shot answers.
* **The native cores** (N151, C151, RC151 and the native-bind variants) fail earlier: they do not acquire the taught
  facts through the byte interface (4-byte context ambiguity; retrieval no longer matching the adaptation state
  after rest). This diagnostic covers only learners that do acquire the facts.

## Limits

3 technical worlds × 3 seeds per cell, wide seed variance in the wd-1 cells, paired sign counts and no tests. The
learner is a `torch.nn.GRU` with a full softmax, not the exact KEEP code (no C16 head, no context matrix). The
batch regime trains OLD and NEW together; the lineage trained them in sequence. Stage-1 runs predate the `--aux` and
`--width` options; the default code path is unchanged, and one replay matched exactly.

## Files

`diag.py` (one run), `grid.py` (resumable 4-worker grid, 135 jobs), `summarize.py` → `results/SUMMARY.json`,
`analyze.py` → `results/ANALYSIS.json`, `results/runs/*.json` (per-run curves, probe and held outputs).
