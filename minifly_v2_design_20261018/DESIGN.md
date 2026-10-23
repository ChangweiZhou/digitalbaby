# MiniFly v2 — design: order and mid-range interaction from biologically grounded mechanisms (2026-10-18)

**Design document. Nothing here has been built or run.** Development only until a separate approval; no formal
science, no adoption. Successor to `minifly_cls_dev_20261017/` (v0) and its memos.

## 1. Goal and constraints

**Goal (deliberately narrow).** A lean fly that (a) tells orders apart — `5-2` is not `2-5`, for pairs it has never
seen — and (b) learns the rules of basic arithmetic from examples, including operations whose digits interact over
several bytes: multi-digit carry/borrow, decimals and unequal lengths, and later mixed operations with brackets.

**Hard constraints.**

* Raw bytes in and out: each input byte enters as its 8 bits; each emitted byte is read from 8 output bit units plus
  one end-of-answer unit. **No codebooks** anywhere (v0's random receptor table and random 16-bit output code are
  removed — they discarded the byte's bits, e.g. `'5'` = 0x35 = `0011 0101`, low four bits 4 + 1).
* **No attention, no transformers**, and nothing derived from them (no positional encodings or rotary codes, no
  segment embeddings, no key–value fast-weight retrieval).
* **No backpropagation.** Learning is local (error-correcting writes, three-factor rules with eligibility traces);
  credit may cross at most two layers. Evolution may set structure and innate parameters.
* **Lean.** The native mushroom body stays the memory; every added mechanism is a switch with parameters (a gene) so
  that it can be removed later.
* **Biological basis.** Every added mechanism must correspond to a mechanism with direct biological evidence (§4).

**Approach.** Build a superset fly with all mechanisms of §5 switchable → make it pass the test ladder (§8) → trim
by ablation and by evolution with a cost per mechanism (§10) → only then reduce the amount of training data (§11).

## 2. What v0 established (carried forward)

* The native Full151 mushroom body can memorise taught facts through the byte stream once (i) the teaching signal
  is error-correcting, (ii) KC adaptation is reset per record, (iii) one-sided KC homeostasis removes always-on KCs,
  (iv) the input keeps order: 84–88% of 80 facts after 24 passes (81–85% after a day), 100% on order-contrast prompts.
* It cannot generalise: a fixed random expansion with a learned linear readout answers an unseen input with a mix of
  trained answers weighted by KC overlap; with codebook inputs that overlap carries no arithmetic structure.
* Replay into both systems kept addition (70–85%) while multiplication was learned.
* A backprop recurrent network next to the fly produced the only generalisation (13%); e-prop 0%. Both are retired;
  v2 uses no backprop.

## 3. Requirements, from first principles

To produce the tens digit of `52-25=` some unit must respond to the specific combination *(tens digit of operand 1
= 5) ∧ (tens digit of operand 2 = 2) ∧ (borrow) ∧ (operator = minus)*, where those items arrived at different times.
Therefore:

| ID | Requirement |
|---|---|
| Q1 | Each held item carries its **role**: which operand, which place value (or rank). Order discrimination is the special case where two inputs differ only in roles. |
| Q2 | **Alignment**: place values must match across operands of different length and with a decimal point. Integer digits align from the right (distance to the end of the integer part); fractional digits align from the left (distance after `.`). |
| Q3 | **Holding**: operand 1 must persist while operand 2 is read (several bytes, seconds of brain time). |
| Q4 | **Reactivation**: held items must be active again, together with the current input, when the step is computed. |
| Q5 | **Conjunctive units** that respond to item × role × state combinations. |
| Q6 | **State across steps**: carry/borrow; the current operator; the current place being produced. |
| Q7 | **Control**: step through places; emit result bytes most-significant first although carries flow from the units. |
| Q8 | **Step tables**: digit ± digit ± borrow → digit, borrow (small, memorisable — the native mushroom body). |
| Q9 | **Gating**: decide when to latch, hold and release a register (at boundaries such as operators, `.`, `=`, brackets). |
| Q10 | **Credit across time without backprop**: the answer's error must reach synapses active seconds earlier. |
| Q11 | **Nesting** (brackets, precedence) — last stage only. |

## 4. Biological evidence base

| Mechanism | Evidence | Serves |
|---|---|---|
| Temporal context cells with a spectrum of decay rates | Lateral entorhinal neurons integrate time from experience over many time scales (Tsao et al. 2018, Nature); temporal context model of human list memory (Howard & Kahana 2002) | Q1, Q2 (age of each item → order; ages at a boundary = right-aligned place value) |
| Time cells reset by events | Hippocampal/entorhinal time cells, including in humans during sequence memory (Umbach et al. 2020, PNAS) | Q1, Q2 (position counted from a boundary → fractional place) |
| Ordinal-position (rank) neurons | Monkey prefrontal neurons tuned to an item's rank within a sequence (Tanji and colleagues); competitive-queuing models of human serial recall | Q1, Q7 |
| Theta–gamma phase code | Ordered items at successive phases of a slower rhythm (Lisman & Idiart 1995); human sequence memory supported by a phase code (Heusser et al. 2016); letters held in working memory reactivated in serial order (Bahramisharif et al. 2018) | Q4, Q7 |
| Activity-silent working memory via short-term synaptic facilitation | Theory: Mongillo, Barak & Tsodyks 2008 (Science); human hidden states revived by a neutral "ping" (Wolff et al. 2017, Nature Neuroscience) | Q3, Q4 |
| Persistent attractor activity; abstract rule neurons | Prefrontal delay activity; human medial-temporal neurons holding "addition" vs "subtraction" abstractly across a calculation (Kutter et al. 2022, Current Biology) | Q6 |
| Basal-ganglia gating of working memory, taught by dopamine | Prefrontal–basal ganglia working-memory gating (O'Reilly & Frank 2006) | Q9 |
| Mixed-selectivity neurons | Prefrontal neurons with nonlinear selectivity to combinations of task variables (Rigotti et al. 2013, Nature) | Q5 |
| Chunk building and closure | Human intracranial activity builds word by word and collapses when a phrase completes (Nelson et al. 2017, PNAS) | Q2, Q11 |
| Articulatory rehearsal loop (phonological loop) | Human verbal working memory with rehearsal of held items (Baddeley) | Q7 (inner rehearsal of the result before output) |
| Eligibility traces on behavioural time scales | Dopamine strengthens synapses active up to ~1 s earlier (Yagishita et al. 2014, Science); plasticity linking inputs seconds apart (Bittner et al. 2017, Science) | Q10 |
| Mushroom-body three-factor learning | Native MiniFly (dopamine-gated KC→output plasticity); KC conjunctive coding | Q5, Q8 |

Kutter et al. (2018, 2022) and Varley et al. (2005, arithmetic and brackets intact in agrammatic aphasia) were
checked online in this project; the remaining citations are from prior knowledge and are to be verified before any
write-up beyond development.

## 5. Architecture (every module is a switch; parameters are genes)

Brain time advances by Δ = 30/14 s per byte, as in the native fly. Notation: xₜ ∈ {0,1}⁸ the bits of byte t.

**M1 Raw-bit sensors.** Eight bit lines plus one "byte present" line. Replaces the receptor table.

**M2 Temporal context traces (multi-rate).** For each bit line i and K rates (genes τ₁…τ_K, default K = 4, τ from 1
to 16 byte steps): cᵢₖ(t) = e^(−Δ/τₖ)·cᵢₖ(t−1) + xᵢₜ. The ratios across k encode each item's age, hence order. At a
boundary, the ages of the digits just read are their place values counted from the right (units = age 1).

**M3 Boundary-reset time cells (rank).** A chain of L cells (gene, default 8). A boundary resets the chain to its
first cell; each byte advances it by one. Gives position counted from the last boundary (fractional places after
`.`; rank within a number). Boundary detection: a unit driven by the raw bits through weights that are learned
(three-factor) or evolved — not a hand-written list (open decision D1).

**M4 Role binding by conjunction.** Item cells respond to (bits of byte) × (M3 position) and to (bits) × (M2 age
at the last boundary): a fixed sparse random projection with native-like fan-in and sparsity, plus v0's one-sided
homeostasis. Order is thereby part of each item's identity.

**M5 Chunk closure into registers.** At a boundary, the role-tagged item cells of the finished number (integer part
aligned by M2 ages, fractional part by M3 positions) are written into the current register (M6); the trace for the
next number starts clean.

**M6 Activity-silent registers (short-term facilitation).** Synapses from item cells onto register cells carry a
facilitation variable u: when the presynaptic item is active, u ← u + U(1 − u); u decays with τ_f (gene, seconds
to tens of seconds). A neutral ping (boundary byte or internal tick) reactivates register cells in proportion to u.
Two registers by default (operands) plus a stack depth for brackets (gene, default 0 until T5).

**M7 Phase-ordered reactivation.** During internal ticks, held items of a register are reactivated one rank per
gamma slot within a cycle of G slots (gene, default 7), so a held number is read out in order (Q7) and its digits
meet the matching digits of the other register at the same phase.

**M8 Persistent state units.** Small self-exciting groups: (a) carry/borrow (bistable), (b) operator rule (one
attractor per operator, set from the raw bits of the operator byte through learned weights), (c) output place
counter (a time-cell chain driven by internal ticks). Each is set/reset by learned input weights.

**M9 Gates (basal-ganglia-like).** Per byte, gate units decide: latch into register 1 / register 2, hold, push, pop.
g = σ(w·features); w learned by a three-factor rule (teaching signal × eligibility trace), initial values evolved.

**M10 Mixed-selectivity layer.** Mushroom-body KCs receive: current raw bits, M2 traces, M3 position, reactivated
register items (M6/M7), carry, operator and place state (M8). Fixed sparse random fan-in with native statistics;
optional input plasticity driven by the teaching signal through fixed feedback weights (switch; ≤ 2 layers, open
decision D2).

**M11 Step memory and output.** Native mushroom-body compartments reading M10, with v0's error-correcting signed
write, adaptation reset per record and homeostasis. Output: 8 bit compartments + 1 end-of-answer compartment; the
emitted byte is the sign pattern of the bit compartments. Native equations unchanged.

**M12 Rehearsal buffer (inner loop).** Result digits produced from the units upward are written into a third
register and replayed most-significant first through M7 before emission (Q7).

**M13 Eligibility traces.** On M8/M9 input weights, M6 register synapses and (if D2) M10 input synapses: e ← λe +
pre·post; Δw = η·(teaching signal)·e; trace time constant ~1–3 s (gene).

**M14 Replay during rest.** v0's bounded episode replay into the step memory (keeps learned tables while the
curriculum grows).

Excluded explicitly: attention, transformers, positional encodings, key–value fast-weight retrieval, backprop,
ELM cells, codebooks.

## 6. Learning summary

| Part | Rule |
|---|---|
| Step tables (M11) | native signed write, error-correcting (v0) |
| Gates (M9), state set/reset (M8), boundary unit (M3) | three-factor: teaching signal × eligibility trace (M13) |
| Register synapses (M6) | facilitation dynamics (fixed form) + three-factor tuning of weights |
| M10 input weights (if D2) | teaching signal through fixed feedback weights, ≤ 2 layers |
| Time constants, sizes, which modules are on | evolution (genes), as in the native EvoLearner genome |

The teaching signal per output byte is the native per-compartment error (target bit vs current output), as in v0.

## 7. Data

Generated arithmetic as raw byte strings `prompt=` → `answer` + end marker. Curriculum stages, each mixed with
replay of earlier stages:

1. single digits, `+ −` (both orders);
2. 2–3-digit integers, `+ −` with carry/borrow, negative results;
3. unequal lengths and decimals (`5.2-3.7=1.5`, `12.5+3=15.5`);
4. `×` (single × multi-digit first);
5. mixed operations and brackets.

No fixed sample count: budgets adapt to learning curves; the amount needed is itself a result (§11).

## 8. Test ladder

| Test | Example | First module set expected to pass |
|---|---|---|
| T1 order | `5-2=3` vs `2-5=-3`, unseen pairs held out in each order | M1–M4, M10, M11 |
| T2 alignment across a gap | `53?58` same tens digit; `52-25` vs `25-52` | + M5, M6, M9 |
| T3 multi-digit carry/borrow | unseen 2–3-digit problems; **numbers one digit longer than trained** | + M7, M8, M12 |
| T4 decimals, unequal lengths | `5.2-3.7`, `3.7-5.2`, `12.5+3` | + M3 fractional alignment |
| T5 brackets / precedence | `(5+3)*2` vs `5+3*2` | + stack (M6 depth), M9 push/pop |
| D distance sweep | accuracy vs number of bytes between interacting digits | reports the mid-range reach |

Scoring: exact own-output bytes; also per output bit and per digit position, so partial structure is visible.
Held-out sets: unseen operand pairs (each order separately), longer operands than trained, unseen operator
combinations. Chance levels measured, not assumed.

**Proposed pass criteria (to be locked before running):** T1 ≥ 95% on unseen pairs in both orders; T2 ≥ 90%; T3 ≥
90% on unseen problems of trained length and ≥ 70% at one extra digit; T4 ≥ 85%; T5 exploratory.

## 9. Controls

* **Role scramble**: M3/M2 role tags permuted per record → T1/T2 must fall to chance (shows order comes from roles).
* **Per-module ablation** at every stage (the basis of trimming).
* **v0 baseline** (permutation trace + memory, no new modules) on the same tests.
* **Random gates** (M9 weights frozen at random) → tests that learned gating matters.
* **No reactivation** (M6 ping disabled) → T2/T3 should fail (shows held items must meet the current input).

## 10. Trimming protocol

1. The superset passes T1–T4 (T5 if reached).
2. Backward elimination: switch off one module at a time; remove it permanently if the battery score drops by less
   than δ (default 2 pp); repeat until no module can be removed.
3. Evolution with fitness = battery score − λ·(enabled modules + state bytes + CPU), started from the trimmed set,
   to find smaller combinations greedy removal misses.
4. Report the minimal set and what each removed module cost or saved.

## 11. Sample efficiency (after trimming)

Learning curves (accuracy vs examples seen) for the trimmed fly; then evolution over learning-rule parameters
(rates, trace constants, replay schedule, what is innate vs learned) with fitness measured at shrinking budgets.
Report examples to criterion per test.

## 12. Compute and implementation plan

* **P0** Fast reimplementation of the native mushroom-body equations (vectorised), with bit-for-bit parity tests
  against the native code on fixtures. Prerequisite: the native port costs ~0.1 CPU s per record.
* **P1** M1–M4, M10, M11 (raw-bit interface) → T1.
* **P2** + M5, M6, M9 → T2.
* **P3** + M7, M8, M12, M13 → T3; distance sweep.
* **P4** fractional alignment → T4.
* **P5** stack → T5.
* **P6** trimming; **P7** sample efficiency.

Each phase: unit tests for the new module, the controls of §9, results committed with the code identity. Technical
worlds only; formal science requires separate approval.

## 13. Open decisions

* **D1** Boundary bytes: learned/evolved boundary unit (proposed) or a fixed list as a first step.
* **D2** Teaching-driven input plasticity on the mixed-selectivity layer (≤ 2 layers): include as a switch?
* **D3** Reuse the native PN→KC matrix for part of M10's input, or a new fixed projection with native statistics
  for the enlarged input bundle.

## 14. Risks

* **Gate learning without backprop** may be slow or unstable; evolution of initial gate weights is the fallback.
* **Interference between registers** sharing item cells; register-specific projections may be needed.
* **Output ordering** (M12) adds a second hold-and-reactivate cycle; if T3 fails only on emission order, test it in
  isolation first.
* **Cost**: the superset is heavier than v0; P0 is a prerequisite, and evolution screens on small budgets first.
