# Why a byte-in/byte-out learner memorises addition but does not generalise it

**Development diagnostic only.** Not formal science, no preregistration, no adoption, no registry change. It used
the three technical worlds of `addition_core_lineage_comparison_dev_v1_20261015` (202610143901–03), not its 16
formal worlds. Stages 1–3: 135 runs (3 worlds × 3 seeds per condition), 5.1 CPU-hours. Stage 4 (transfer to
multiplication): 45 runs, 4.1 CPU-hours.

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

## Stage 4: does a learner that generalised addition do better on multiplication?

`transfer.py`, `transfer_grid.py`, `transfer_summary.py` → `results/TRANSFER_SUMMARY.json`, `results/transfer/*.json`.

**Design.** Phase A trains addition with the stage 1/2 code path. Every MEM, WD and GEN phase-A outcome was checked
against its stored stage-1/2 run and matched exactly (27/27). Phase B then trains **multiplication only** (`a*b=`),
with a fresh optimizer and the same regime in every arm: full-batch AdamW, weight decay 0 or 1, both run from the
same phase-A state. It uses the **same split**: the 80 taught pairs are taught as products; the 20 HELD pairs stay
unseen in both operations. Addition is not rehearsed in phase B; its retention is measured throughout.

| Arm | Phase A | Phase-A unseen + |
|---|---|---:|
| FRESH | none (newborn) | — |
| MEM | addition, wd 0 | 6% |
| WD | addition, wd 1 | 44% |
| GEN | addition + counting, wd 1 | 65% |
| SHUF | addition with answers permuted across the taught prompts + counting, wd 1 | 5% |

SHUF did not fit its permuted table: taught + was 4–11% after 20k steps (weight decay stopped it memorising
arbitrary answers). It is therefore a "same bytes and regime, no addition learned" control, not a memorised-table
control.

**Results** (9 runs per cell; phase-B wd 0 / wd 1):

| Arm | Steps to 100% taught × (median) | Unseen × | – with an operand 0 or 1 | – both operands ≥ 2 | Median relative error, both-≥2 answers | + taught / + unseen after 10 × steps | + taught / + unseen at end |
|---|---|---|---|---|---|---|---|
| FRESH | 500 / 600 | 17% / 18% | 30% / 31% | 3% / 3% | 0.25 / 0.35 | — | — |
| MEM | 200 / 250 | 14% / 16% | 26% / 27% | 3% / 3% | 0.24 / 0.26 | 88% / 6% · 88% / 7% | 9% / 3% · 3% / 2% |
| WD | 250 / 400 | 21% / 18% | 36% / 30% | 6% / 6% | 0.14 / 0.33 | 63% / 22% · 63% / 21% | 3% / 4% · 3% / 2% |
| GEN | 300 / 450 | 18% / 19% | 28% / 29% | 7% / 7% | 0.15 / 0.24 | 25% / 17% · 24% / 16% | 3% / 4% · 3% / 3% |
| SHUF | 200 / 300 | 15% / 17% | 29% / 31% | 0% / 1% | 0.44 / 0.48 | 6% / 5% · 6% / 6% | 3% / 3% · 3% / 2% |

40% (8–12 per world) of the 20 unseen × facts have an operand 0 or 1. Every arm, FRESH included, picks up "×0 → 0"
and "×1 → copy" from the taught facts; that is where most unseen-× accuracy comes from.

**Answers to the three questions:**

1. **Faster multiplication: yes, but not because of addition.** Every pretrained arm reaches 100% taught × sooner
   than FRESH: 71 of 72 paired comparisons, one tie, none later; on average 167–367 steps sooner. SHUF, which learned
   no addition, gains about as much as MEM (median 200 vs 200 steps at wd 0, 300 vs 250 at wd 1). The saving comes
   from generic machinery: reading the prompt bytes and emitting digits + EOS.
   The networks that generalised addition are *slower* than MEM and SHUF: GEN vs SHUF 8 of 9 slower in both regimes;
   GEN vs MEM 8–9 of 9 slower.
2. **Better on unseen multiplication: marginally, and only approximately.** For facts with both operands ≥ 2, GEN and
   WD get 6–7% right vs 3% for MEM and FRESH and 0–1% for SHUF. GEN vs SHUF unseen × overall: +3 pp (4 better, 1
   worse, 4 tied) at wd 0, +2 pp (3/2/4) at wd 1. GEN's and WD's both-≥2 answers are closer to the true product: median relative error
   0.14–0.15 vs 0.24–0.25 for MEM and FRESH and 0.44 for SHUF at phase-B wd 0. This is a weak sign that some magnitude
   is reused; it is not exact transfer.
3. **Forgetting addition less: no, the opposite.** Addition collapses in every arm. Ten full-batch multiplication
   steps take GEN from 100% taught / 65% unseen + to 25% / 17%. MEM still holds 88% taught at step 10 but is at about
   50% by step 50. Every pretrained arm ends at 3–9% taught + and 2–4% unseen +. Both wd-1 arms, with and without counting, forget
   fastest. With per-parameter step sizes set by the optimizer, small-weight nets move proportionally further per
   step, so this likely tracks phase-A weight decay, not generalisation itself.

**Interpretation.** In this learner, the addition structure gained in phase A (ordered and roughly spaced digit
quantities) is not protected and not reused as a unit. Multiplication training overwrites it within tens of steps.
What survives and speeds learning is the format machinery, which even the no-addition SHUF arm has. A core that
"learned addition, so multiplication is easy" would need the digit-quantity representation to persist and to be
shared by both operations. This plain recurrent learner does neither: no mechanism keeps a learned representation
stable while a new task is learned on top.

## Relation to the earlier results

* **No transfer from addition to multiplication.** STREAM-TRANSFER found untaught × at 0–2.7%, and multiplication
  training erased untaught +. Stage 4 reproduces both: unseen × for both-≥2 facts at 0–7%, and unseen + erased
  within tens of × steps. It adds that a learner which *had* generalised addition still loses it and gains almost
  nothing on ×. The savings result (correct OLD speeds NEW: 100% vs 17.5%) matches stage 4's generic format savings;
  SHUF shows that part needs no addition knowledge.
* **The native cores** (N151, C151, RC151 and the native-bind variants) fail earlier: they do not acquire the taught
  facts through the byte interface (4-byte context ambiguity; retrieval no longer matching the adaptation state
  after rest). This diagnostic covers only learners that do acquire the facts.

## Limits

3 technical worlds × 3 seeds per cell, wide seed variance in the wd-1 cells, paired sign counts and no tests. The
learner is a `torch.nn.GRU` with a full softmax, not the exact KEEP code (no C16 head, no context matrix). The
batch regime trains OLD and NEW together; the lineage trained them in sequence. Stage-1 runs predate the `--aux` and
`--width` options; the default code path is unchanged, and one replay matched exactly. Stage 4 trains phase B
without rehearsal and full-batch; 10 phase-B steps are 10 updates on all 80 product facts, so "fast forgetting" is
measured in updates, not presentations.

## Files

`diag.py` (one run), `grid.py` (resumable 4-worker grid, 135 jobs), `summarize.py` → `results/SUMMARY.json`,
`analyze.py` → `results/ANALYSIS.json`, `results/runs/*.json` (per-run curves, probe and held outputs).
Stage 4: `transfer.py` (one arm/world/seed: phase A, then phase B at wd 0 and wd 1), `transfer_grid.py` (45 jobs),
`transfer_summary.py` → `results/TRANSFER_SUMMARY.json`, `results/transfer/*.json`.
