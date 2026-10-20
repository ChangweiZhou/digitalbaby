# MiniFly v4 — learning arithmetic rules from pretraining data by consolidating what recurs (2026-10-20)

**Design document. Nothing has been built or run.** Development only until separately approved; no formal science,
no adoption. Successor to v0 (`minifly_cls_dev_20261017/`), the v2 proposal (`minifly_v2_design_20261018/`) and
Codex's v3 (`minifly_v3_dev_20261020/` on the codex branch, NO_ADVANCE: 100% taught, 0% never-taught sums).

## 1. Goal and constraints

**Goal.** The fly learns arithmetic rules from pretraining data: it answers problems it has never seen (multi-digit
addition, subtraction in both orders, later decimals), with a slow warmup that does not lock the brain into early,
item-specific memories.

**Constraints.**

* No arithmetic-specific structure: no neuron or module designated as digit, operand, carry, place or operator; no
  parser; no supplied numeric features. Arithmetic is in the data only.
* Raw bytes in and out: each input byte as its 8 bits; each output byte read from 8 bit units plus an end-of-answer
  unit. No codebooks. Problems and answers are ordinary written byte strings; there is no format engineering (no
  reversed answers or other conventions chosen to make learning easier).
* No attention, no transformers or anything derived from them. No backpropagation.
* The native mushroom-body learning equations are kept. A **larger model is allowed** (more memory cells), and
  evolution may set innate parameters.
* Every added mechanism is generic (it would serve text as well as arithmetic) and has biological grounding.

## 2. Why every earlier design memorised (first principles)

A system generalises only if (a) **parts** of examples recur far more often than whole examples, and (b) its
features **expose those parts**, so that an unseen example activates memory already built from seen ones.

* **The single-digit table violates (a).** `3+8=11` is atomic: no smaller unit that recurs elsewhere determines the
  answer. And every whole problem was shown hundreds of times (v3: 128 passes over 60 facts), so item traces were
  refreshed as often as anything shared.
* **The memory consolidates everything.** The native slow store (time constant ≈ 142 days) keeps whatever is written.
  The error-correcting write fits all taught facts quickly, after which the error that any representation learner
  receives is near zero (v3: 98% of taught bytes right after warmup, mean residual 0.085). The lookup always wins.
* **The features violated (b).** Codebook inputs carried no values. The order-blind or order-keeping trace exposed
  "this byte at this recency", never "the units digits of both numbers together".
* **v3's slow warmup** lowered learning rates, but nothing regulated *what becomes permanent*. The representation
  adaptor moved ≈ 0.0004 per weight over a whole life and could shift the input drive by at most 0.125.

## 3. Central hypothesis: consolidate only what recurs

In a stream of random 2–4-digit additions:

| | Expected recurrences in N samples |
|---|---|
| A specific whole problem (4-digit + 4-digit) | ≈ N / 10⁸ — effectively never |
| A specific column pattern (digit, digit, carry-in → digit, carry-out; 200 patterns) | ≈ N / 50 |

If (i) unconfirmed memory traces fade and (ii) a trace becomes permanent only after repeated, consistent
confirmation across separate experiences, then whole problems fade and column patterns consolidate. What survives is
the rule; unseen problems are answered from consolidated parts they share with training. The mechanism is generic: it
applies wherever wholes are rare and parts recur (words and phrases recur, sentences do not).

This turns "slow warmup, don't bake the brain" into a precise requirement: **commitment follows repetition**, and
codes settle only after broad experience.

## 4. Answers in ordinary written form: what that requires

Decision: problems and answers are ordinary byte strings (`1234+5678=6912`), as written. The fly must learn whatever
the bytes require. What they require, stated precisely so the design can provide generic means for it:

* An answer digit is emitted most-significant first, but it depends on the aligned column and on the carry arriving
  from every column to its right. So when the fly emits answer position k (counted from the right), the relevant
  inputs are the digits at positions k, k−1, …, 1 of both held numbers; in practice a carry chain rarely runs far, so
  the nearest few columns carry most of the information.
* The sign of a subtraction depends on the most significant position where the two numbers differ.
* Therefore the fly needs both numbers **held** with positions counted from their right ends (A3), and conjunctive
  cells that combine several neighbouring columns of both held numbers with the output position being emitted (A4).
  This is generic multi-position conjunction over held segments, not arithmetic structure; the larger model is what
  makes enough such conjunctions available. Gate G0 (§9) measures whether they are.

## 5. Architecture (generic modules; every one a switch, parameters are genes)

Brain time advances by Δ = 30/14 s per byte, as in the native fly.

**A1 Raw-bit sensors.** 8 bit lines + a byte-present line.

**A2 Segment boundaries, decided by the fly.** The fly learns for itself which bytes restart its position count; no
byte is listed, chosen or evolved by us. A boundary unit reads the raw bits (8 weights + threshold, neutral at
birth). Its firing is slightly variable, and its weights change by a reward-modulated local rule: when a restart
(or its absence) is followed by a smaller answer error than usual, the weights move towards repeating that decision
(error-signal × eligibility, three-factor; no backprop). Bytes whose restarts make answers more accurate become
boundaries.

**A3 Segment-relative position codes.**
* Time cells: a chain restarted at each boundary gives "steps since the last boundary" (used for the output step k
  after the answer boundary and for positions after a decimal point).
* Closure snapshot: at a boundary, multi-rate traces of the segment just closed give each of its items its distance
  from the segment's end (units = 1, tens = 2, …). The last S closed segments (gene, default 3) are held in
  activity-silent facilitation registers (v2 M6), each item tagged (segment age, position from end, bits).
This is generic chunk structure: it codes position within words and phrases equally.

**A4 Conjunctive memory cells (KCs), larger population.** N_KC cells (genes; candidates 2¹⁶, 2¹⁸ and 2²⁰; native
is 5,177), sparse top-k coding with v0's one-sided homeostasis. Fan-in follows a **topographic prior** (accepted as
a generic wiring rule): each cell samples inputs from a local window in segment-relative position — a few
neighbouring positions across the held segments, the time-cell step, the most recent emitted byte. Window widths vary
across cells (gene for the distribution), so some cells span several neighbouring columns, which ordinary written
answers need (§4). Local connectivity in a sensory map has precedent in the ledger (M016, fixed sparse graphs with a
retinotopic prior). A random (non-topographic) fan-in arm measures what the prior buys.

**A5 Output compartments on one shared KC population.** Nine output channels (8 bits + end-of-answer), each with
native fast/slow value stores over all KCs, native readout and v0's error-correcting signed write. One mushroom body
with several output neurons, instead of v0's 16 separate flies.

**A6 Recurrence-gated consolidation (the key change).** For every KC→output synapse:
* Writes go to the fast store first (native fast decay, time constant a gene, default native ≈ 3,400 s).
* A confirmation variable q accumulates sign-consistent writes and decays slowly
  (q ← q·e^(−Δt/τ_q) + sign(write)·[record differs from the last confirming record]).
* The share of each write that goes to the slow store is f(q) = f_min + (f_max − f_min)·σ((|q| − q₀)/s), replacing
  the native allocation fraction. Unconfirmed writes stay fast and fade; repeatedly confirmed ones become permanent.
Grounding: synaptic tagging and capture, the commitment cascade and spaced-training gating (ledger U013, U030,
U029). τ_q, q₀, s, f_min, f_max are genes.

**A7 Rest: downscaling and replay.** During simulated rest: a small multiplicative downscaling of the slow store
(pressure towards compact structure) and bounded replay of recent records (v0) so earlier stages are not overwritten.

**A8 Own-output feedback.** Emitted bytes enter the sensors like any input (already in the byte loop).

Not included: attention, transformers, positional encodings, key–value retrieval, backprop, codebooks, ELM cells,
arithmetic-designated units. The v3 plastic adaptor is not needed for the central hypothesis; it may be added later as
a slow, optional component (§12).

## 6. Learning and slow warmup

**Learning.** Only the KC→output values learn (native write, error-correcting coefficient, A6 allocation). Fixed
generic features plus local learning: no deep credit assignment is required by the hypothesis.

**Slow-warmup rules ("don't bake").**
1. **Codes settle over broad data.** Homeostatic thresholds adapt during a long early window covering the full data
   distribution (window length a gene, measured in distinct records, not passes).
2. **Commitment follows repetition** (A6). Nothing is permanent until confirmed across separate records.
3. **Unconfirmed traces fade** (native fast decay).
4. **Intensity follows stability.** The teaching multiplier ramps from a₀ to 1 as measured key drift (overlap of
   fixed probe prompts' KC codes between checkpoints) falls below a threshold, not on a fixed schedule.
5. **No pass-based repetition.** Problems are sampled fresh; repetition of wholes is rare by construction.

## 7. Data

A generator of raw byte records `prompt` → `answer` + end marker, sampled fresh (no fixed sample count; budgets
follow measured learning curves and CPU):

| Stage | Content |
|---|---|
| S0 (optional) | single-digit `+ −` facts (tests whether earlier facts are reused) |
| S1 | 2–4-digit addition, equal and unequal lengths |
| S2 | subtraction in both orders with sign (`52-25=27` / `25-52=-27`) |
| S3 | decimals and unequal lengths (`5.2-3.7=1.5`, `12.5+3=15.5`) |

All records are ordinary written byte strings.

Each stage keeps replaying earlier ones at a fixed fraction. Operand digits are uniform; lengths uniform within a
stage.

## 8. Arms, controls and tests

**Arms (paired births, same streams):**

| Arm | A3 segment codes + topographic fan-in | A6 recurrence-gated consolidation |
|---|---|---|
| B0 | off (v0 permutation trace, random fan-in) | off (native allocation) |
| B1 | off | on |
| B2 | on | off |
| B3 (full) | on | on |
| B4 | B3 + S0 single-digit phase first | on |

**Controls.** Shuffled answers (SH); no-write history (N); random instead of topographic fan-in at equal N_KC;
per-module ablations for trimming (§10).

**Tests.** Exact own-output answers on never-seen problems of trained lengths; per-column accuracy; carry-column
accuracy separately; sign accuracy and `a−b` vs `b−a`; numbers one digit longer than trained; retention of earlier
stages. Chance levels measured.

**Predictions (to be locked before running).** B0 and B1 near per-digit chance on unseen problems (no exposed parts,
or parts not consolidated selectively). B2 above chance but degrading as item traces accumulate. B3 well above chance
and improving with experience; positions that depend on long carry chains lag the rest, and the leading digit is
hardest. Length +1 above chance only with A3.

## 9. Gates

* **G0 Coverage (before any learning run).** An offline analysis tool (outside the learner) fits the best linear
  readout from KC codes to the target bits on training problems and tests it on unseen problems. If even the best
  linear readout cannot reach ≥ 95% per digit on unseen problems, the fly cannot either: increase N_KC or revise
  fan-in before running learning. This bounds what the local learner can achieve.
* **G1 Learning sanity.** Recent problems are recalled from the fast store; confirmation variables rise only for
  recurring patterns (checked on fixtures).
* **G2 Rules.** B3: ≥ 90% per digit and ≥ 70% exact on unseen S1 problems; B0 near chance.
* **G3 Order and sign.** ≥ 90% correct sign on unseen S2 problems in both orders.
* **G4 Length +1.** Above chance (exploratory threshold).
* **G5 Retention.** Earlier stages ≥ 90% of their end-of-stage accuracy after later stages.

## 10. Trimming

After the full arm passes: switch off one module at a time and remove it if the battery drops by less than δ
(default 2 pp); then evolution with fitness = battery − λ·(modules + N_KC + state + CPU) to find a smaller
configuration. Report the minimal set.

## 11. Compute and implementation plan

* **P0** Vectorised reimplementation of the native store equations for one shared KC population with several output
  channels; parity tests against the native code at native size (fixed fixtures, declared float tolerances).
* **P1** Generator, raw-bit sensors, boundary unit, segment codes, registers; unit tests; G0 coverage analysis at
  2¹⁶ and 2¹⁸ cells.
* **P2** A6 consolidation and warmup rules; G1 on fixtures.
* **P3** Stage S1 with arms B0–B4 on technical worlds; G2.
* **P4** S2 (G3), S3, length test (G4), retention (G5).
* **P5** Trimming; sample-efficiency curves.

Technical worlds only; results committed with code identity; formal science only on separate approval.

## 12. Risks and open questions

* **Coverage.** Ordinary written answers need conjunctions spanning several neighbouring columns of both held numbers
  plus the output position. G0 measures whether the chosen N_KC and window widths expose them; larger N_KC (the model
  may grow) and structural plasticity that grows useful conjunctions (ledger U043, U005) are the remedies.
* **Long carry chains** (`9999+1`) need the widest conjunctions; they are rare in uniform data, so accuracy on them is
  reported separately and expected to lag.
* **Boundary learning** may be slow or settle on unhelpful boundaries; its decisions and reward history are logged.
* **Consolidation parameters** may consolidate too little (nothing permanent) or too much (items consolidate);
  confirmation counts are logged per synapse class to diagnose.
* **Scale and CPU.** 2¹⁸–2²⁰ cells × 9 channels requires the P0 engine; budgets are set from measurements.

## 13. Decisions taken

* **Data format:** ordinary written byte strings only; no format engineering.
* **Boundaries:** decided by the fly itself through a reward-modulated local rule (A2); nothing listed or evolved.
* **Topographic fan-in prior:** accepted as a generic wiring rule; the random fan-in arm measures its contribution.
* **Model size:** a larger memory-cell population is allowed.
