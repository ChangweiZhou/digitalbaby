# Memo: what stops us teaching the fly basic arithmetic (2026-10-17)

Development diagnostics and discussion. Not formal science, no adoption, no registry change. Evidence: `REPORT.md`
(stages 1–4), `results/SUMMARY.json`, `results/ANALYSIS.json`, `results/TRANSFER_SUMMARY.json`,
`results/ORDER_PROBE.json`. Standing constraints: byte in / byte out, no hard-coded task structure, no binary
combinator, no transformers, attention or ELM, nothing that must be redesigned when the task changes.

## 1. Summary

* **Memorising, not generalising, is the default.** A small byte-level recurrent learner (KEEP-sized GRU) learns
  every taught single-digit sum and gets about 6% of unseen ones. The 80 taught facts alone determine all 20 unseen
  answers, so information is not what is missing. The learner does not prefer the reusable explanation.
* **It gets the shape, not the precision.** Digit states are ordered, but 76–78% of wrong unseen answers are off by
  exactly one, and `a+b` / `b+a` get the same answer only about half the time.
* **Quantity in the stream fixes it.** With tally input (`|||+|||||=`) the same learner gets 98–100% of unseen sums.
  Weight decay (+38 pp) and counting experience (+11 to +21 pp) help only partly.
* **Learning addition gives multiplication nothing specific.** Pretraining speeds multiplication equally with or
  without addition knowledge. Learners that generalised addition gain only 3–4 pp on unseen products with both
  operands ≥ 2, and lose addition fastest (taught + 100% → 25% after 10 multiplication updates). Every arm
  keeps under about 20% of taught addition after 200 updates.
* **The native fly barely keeps byte order.** Its input is a fading mix of recent bytes. Reordering the same bytes
  changes its code less than changing the last digit does, and brackets are not represented as grouping. Without
  order there are no arithmetic expressions and no language.

## 2. What is stopping us

The difficulty comes mostly from how the task is posed, not from the learner being too weak.

1. **One-shot answering turns the rule into a table.** A small learner acquires a rule as a small step it can
   repeat. Tally works because the step ("one more mark") is iterated. From `3+8=` straight to `11`, the "rule" must
   be a 100-entry function computed at once, which is a table. Addition helps multiplication only if both are
   procedures (count on; add repeatedly). A table of sums contains nothing a table of products can reuse, which is
   what stage 4 found.
2. **The digits mean nothing to the learner.** `3` and `4` are arbitrary bytes. The learner must infer the symbols'
   meaning and the operation together, from the operation's results alone. Order emerges; exact values do not.
3. **We show answers but never teach.** Teaching normally supplies what symbols mean, a demonstration of the
   procedure, and feedback on the steps. The 10-13 requirement forbids solution traces, so all the work falls on the
   learner. Traces delivered as bytes would be teaching, not wiring: nothing in the core would change per task.
   Whether they are allowed is the project's decision.
4. **A small task is memorisable, and the fly is built to memorise.** A rule pays off only when the world is bigger
   than memory. 100 facts never force it: even a 12.6k-parameter net memorises all 80 taught facts. The mushroom
   body is a pattern-separation device that gives every input a distinct sparse code. That is ideal for "odour →
   shock" and the wrong bias for seeing that 3+8 and 8+3 are the same thing.

Two practical blockers come on top:

5. **Nothing keeps what was learned.** New training overwrites earlier structure within tens to hundreds of
   updates, so even a perfectly taught first step cannot be built on.
6. **The native fly cannot handle the byte stream yet.** It cannot reliably hold a prompt and emit a multi-byte
   answer (4-byte context ambiguity; retrieval no longer matching the adaptation state after rest). Section 4 shows
   why its input loses order.

## 3. Grounded symbols, and why they are not the answer

A symbol is **grounded** when its meaning is tied to something the learner experiences directly, not only to other
symbols. To the fly, `3` is byte 0x33; nothing in the stream relates it to three of anything, except the addition
table itself. Tally marks are grounded by construction: the quantity is in the stream as repetitions. Grounding
digits would mean experiences where quantity is visible, paired with the digit (`|||=`→`3`, `3=`→`|||`), ideally
forcing one internal "3" for both forms. That is experience, not hard-coding.

**Project view:** grounding of this kind does not scale to language and is not the direction to take. The scalable
counterpart is **meaning from prediction**: a symbol's meaning is what it predicts and what predicts it in the
stream. Text models learn this way without grounding. It depends on a memory that keeps order, which the native fly
lacks (section 4).

## 4. Sequencing: does anything in the model tell byte order apart?

**Native fly (N151, C151, RC151 first bank).** The only memory of earlier bytes is the FE0 front end:
* 88 receptor channels; each byte switches on a fixed random 6 of them;
* the trace fades with τ = 10 s, and a byte arrives every 2.14 s;
* by age, the weights are 1, 0.81, 0.65, 0.53, 0.42, 0.34, 0.28, 0.22, 0.18;
* the sum is normalised and projected onto the KCs, and the top 5% fire.

That KC code is the only key the fast and slow stores see. The `adapt` store is per-KC fatigue, also a fading sum of
recent firing. Order appears only as slightly different weights, and anything more than about 5 bytes back has
mostly faded.

Measured on the frozen newborn input path (`order_probe.py`, no learning). Share = fraction of active KCs the two
inputs have in common (Jaccard):

| Pair | What differs | Active KCs shared |
|---|---|---:|
| `3+8=` vs `8+3=` | same bytes, reordered | 69% |
| `5+3*2=` vs `5*3+2=` | same bytes, reordered | 74% |
| `(5+3)*2=` vs `(5*3)+2=` | same bytes, reordered | 75% |
| `(5+3)*2=` vs `5+3*2=` | brackets removed | 63% |
| `1+9=` vs `2+9=` | first number changed | 84% |
| `3+8=` vs `3+9=` | last number changed | 39% |
| `(5+3)*2=` vs `(6+3)*2=` | first number changed | 86% |
| `the cat sat.` vs `the dog sat.` | subject changed | 64% |

Reordering costs less than changing the last digit, early bytes barely register, and brackets are just two more
ingredients in the mix. **RC151's second bank** hashes the last 4 visible bytes into a random code: exact order, but
only 4 deep and with no similarity between related inputs. `(5+3)*2=` and `(6+3)*2=` both end in `)*2=` and get an
identical code. When two inputs share a code exactly, no learning downstream can separate them.

**Recurrent references (KEEP, G62).** The state is updated byte by byte by a nonlinear function, so order matters by
construction. They over-separate it if anything (a+b vs b+a agree about half the time on unseen pairs). Brackets
would need a learned stack-like memory, which is possible for shallow nesting, but nothing pushes them to learn
precedence as a reusable rule rather than memorising whole expressions.

**Consequence.** Order and grouping are a missing basic capacity in the native fly and a prerequisite for arithmetic
expressions and language. No change to the learning rule can recover information the input path has already blurred
or discarded. The fly needs a state that each byte updates in a way that depends on what came before (a generic
recurrent state, not necessarily a GRU), not a fading sum. That stays byte in / byte out with nothing task-specific,
but it departs from the fly's own input path: an odour is a mix, and the mushroom body is designed to tell mixes
apart, not sequences.

## 5. Open decisions

1. **Order-keeping state for the native fly.** Replace or supplement FE0 with a generic, learned, order-dependent
   state, before any further arithmetic work on native cores.
2. **Time to think.** Allow the learner extra steps before answering, either silent internal steps or its own output
   bytes before the final answer, so rules can be learned as repeated small steps.
3. **Teaching through the stream.** Decide whether worked steps or traces delivered as bytes are allowed (the 10-13
   requirement currently forbids solution traces). Grounding by tally is set aside as not scalable (section 3).
4. **Keeping what is learned.** A mechanism that protects learned structure while new tasks are learned; without it,
   no ladder of skills can be built.
5. **Success bar.** Exact answers on unseen facts, or approximate magnitude as a first milestone (which the reference
   learners already reach).

## 6. Results in brief

| Condition (digits in, decimal out) | Unseen + |
|---|---:|
| online, lineage-like dose | 6% |
| full batch, wd 0 / wd 1 | 6% / 44% |
| full batch wd 1 + counting / + permuted counting | 65% / 52% |
| width 16 / 32 (wd 0) | 4% / 16% |
| tally input, decimal out | 98% |

| Transfer arm (phase-B wd 0) | Steps to 100% taught × | Unseen ×, both operands ≥ 2 | + taught after 10 × steps |
|---|---:|---:|---:|
| FRESH | 500 | 3% | — |
| MEM (memorised +) | 200 | 3% | 88% |
| GEN (generalised +, 65%) | 300 | 7% | 25% |
| SHUF (same bytes, no + learned) | 200 | 0% | 6% |

Files: `diag.py`, `grid.py`, `summarize.py`, `analyze.py` (stages 1–3); `transfer.py`, `transfer_grid.py`,
`transfer_summary.py` (stage 4); `order_probe.py` (section 4); `REPORT.md` (full report).
