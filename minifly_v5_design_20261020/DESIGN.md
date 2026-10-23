# MiniFly v5 — corrected v4, run as an ordered ladder (2026-10-20)

**Design document. Nothing has been built or run.** Development only until separately approved. Revises
`minifly_v4_design_20261020/DESIGN.md` according to `REVIEW_V4.md` (this folder). The central hypothesis is
unchanged and still untested: in a stream where whole problems are rare and their parts recur, a memory that keeps
only repeatedly confirmed traces ends up holding the rules.

## 1. Constraints (unchanged from v4)

* No arithmetic-specific structure: no neuron or module designated as digit, operand, carry, place or operator; no
  parser; no supplied numeric features. Arithmetic is in the data only.
* Raw bytes in and out: input bytes as their 8 bits; output bytes from 8 bit channels plus an end-of-answer channel;
  ordinary written byte strings; no codebooks, no format engineering.
* No attention, no transformers or anything derived from them. No backpropagation.
* Native mushroom-body learning equations kept; a larger model is allowed; topographic wiring accepted.
* Every mechanism generic (it would serve text as well) and biologically grounded.
* Slow warmup, defined precisely in §5.

## 2. What changes from v4 and why

| v4 element | Problem (review finding) | v5 change |
|---|---|---|
| Output byte = sign of each bit channel's raw value | common offset ≈ −0.98: every bit 0, end-of-answer never fires (F2) | each output channel compares its value with its own running reference (§3, M5) |
| Memory cells sum 12 random signed raw-bit lines | digit-pair combinations arise only by accident (F5) | an unsupervised byte-value stage first; cells combine byte-value units across positions (M2, M4) |
| Boundaries learned from answer error | circular; no segments ever formed (F3) | boundaries from input statistics alone: a change of learned byte class (M3) |
| Warmup scales all teaching, ramp gated by drift | ramp never advanced; teaching stuck at 1/16 (F7) | warmup governs commitment and code formation only; fast-store teaching at full strength (§5) |
| Consolidation sigmoid with 0.9% at zero confirmation | not strict (F8) | hard floor: zero slow allocation below the confirmation threshold |
| G0 fitted on 128 problems, before segments existed | uninformative (F4) | coverage curve on learned codes over 10³–10⁵ fitted problems (rung R4) |
| G1 recall of one-off problems | contradicts the hypothesis (F6) | rung checks on recurring regularities: format, then columns |
| 64 records per history | far below what learning needs (F1) | doses computed in advance per rung (§7) |

## 3. Architecture

Brain time advances by Δ = 30/14 s per byte as in the native fly. Every module is a switch with genes for its
parameters.

**M1 Raw-bit sensors.** 8 bit lines + byte-present line (as v4).

**M2 Byte-value units (unsupervised).** A layer of K units (gene, default 64) receives the 8 raw bits of the current
byte. Units compete (winner-take-most); the winner's weights move towards the current bit pattern (competitive
Hebbian learning), with homeostatic thresholds so all units get used. After listening to the stream, units become
selective to the byte values that actually occur; nothing lists which bytes exist. Raw bits remain available to
later stages alongside the units.

**M3 Byte classes and boundaries (unsupervised, decided by the fly).**
* Each byte-value unit accumulates, by Hebbian co-activity, a running record of which units were active just
  before and just after it (its context).
* A small layer of class units (gene, default 6) clusters byte-value units by context similarity (competitive
  learning on the context records). Bytes used in the same contexts fall into the same class; bytes with unique roles
  get classes of their own. Nothing is told which bytes are digits or operators.
* A **boundary** fires when the class of the current byte differs from the class of the previous byte. In text this
  separates letters from spaces and punctuation; here it separates digit runs from `+ - = .` without being told.
* Boundaries are frozen when class assignments stop changing (measured), before answer learning begins.

**M4 Segment-relative positions and combination cells.**
* Closure snapshot (as v4): at a boundary, the segment just closed is held in a facilitation register with each
  item's position counted from the segment's right end; the last S segments are held (gene, default 4, so `a`, `+`,
  `b`, `=` all fit). Time cells count steps since the last boundary (output position after `=`; fractional positions
  after `.`).
* Combination cells (memory cells): N_KC (genes; 2¹⁶–2²⁰) with small fan-in (gene, default 4–6) drawn
  topographically: from **byte-value units** (not loose bits) at a few neighbouring right-aligned positions across
  the held segments, plus the time-cell step, plus the byte-value unit of the most recently emitted byte. A cell with
  fan-in 4 can respond to "value u at position p of segment s ∧ value v at position p of segment s′ ∧ step k ∧ last
  emitted value w" — an aligned column pattern — which v4's bit-summing cells could not form except by accident.
  Window widths vary across cells so some span several neighbouring positions (needed for carries).
* One-sided homeostasis on combination cells (as v0).

**M5 Output channels with their own reference.** Nine channels (8 bits + end-of-answer), native fast/slow value
stores over all combination cells, native readout. Each channel keeps a running reference rⱼ: the mean of its value
at output events (slow average). The emitted bit is 1 if zⱼ > rⱼ. The error-correcting teaching coefficient uses the
same reference: sⱼ = (tanh((zⱼ − rⱼ)/T) − yⱼ)/2, with the native positive prediction-minus-target sign convention
verified by a unit test before any run.

**M6 Recurrence-gated consolidation.** As v4, with a strict floor: slow allocation is exactly zero until the
confirmation variable exceeds q₀; above it the slow share rises smoothly to f_max. Confirmations count only
sign-consistent writes in different records, spaced in time.

**M7 Rest.** Small multiplicative downscaling of the slow store; bounded replay of recent records.

**M8 Own-output feedback.** Emitted bytes enter the sensors as input (as before).

Excluded: attention, transformers, positional encodings, key–value retrieval, backprop, codebooks (M2 is learned
from the bits, not supplied), arithmetic-designated units.

## 4. Learning summary

| Part | Rule | When |
|---|---|---|
| Byte-value units (M2) | competitive Hebbian + homeostasis | listening phase, then slow |
| Context records and classes (M3) | Hebbian co-activity; competitive clustering | listening phase; frozen when stable |
| Combination-cell thresholds (M4) | one-sided homeostasis | slow throughout |
| Output values (M5) | native error-corrective signed write, full strength | from answer learning onward |
| Consolidation (M6) | confirmation-gated slow allocation | throughout answer learning |

No answer information reaches M2/M3; they learn from the byte stream alone (prompts and answers as heard).

## 5. Slow warmup, made precise

"Don't bake the brain" means nothing becomes permanent, and no code freezes, before broad experience:

1. **Codes form while listening.** M2 and M3 learn from a long listening phase over the full data distribution, with
   no teaching, and freeze only when measured stable.
2. **Fast learning at full strength.** The fast store responds from the start; its traces decay (native ≈ 1 h), so
   they cannot bake anything in.
3. **Commitment follows repetition.** The slow store receives nothing until a synapse's change has been confirmed
   across separate, spaced records (M6).
4. **Representation change is slow.** Homeostatic thresholds and any later plasticity of M2/M3 run at low rates after
   freezing.
5. **No pass-based repetition of wholes.** Problems are sampled fresh.

## 6. Data

Fresh ordinary written byte strings, generated on demand:

| Stage | Content |
|---|---|
| D0 listening | the stream of all stages below, heard without teaching (for M2/M3) |
| D1 | 1–4-digit addition, equal and unequal lengths |
| D2 | subtraction in both orders with sign |
| D3 | decimals and unequal lengths |

Each stage keeps replaying earlier ones at a fixed fraction.

## 7. The ladder (each rung must pass before the next; doses fixed in advance)

The native write moves an output value by ≈ 0.01–0.02 per event, so a bit needs tens of consistent writes on the same
cells. A specific aligned column pattern (digit, digit, step, previous emitted digit) recurs about once per few
hundred records per position. Doses below follow from these two numbers and are upper budgets, not targets.

| Rung | Ability | Measured by | Pass | Dose budget |
|---|---|---|---|---|
| R1 | Byte-value units | selectivity: each occurring byte has a stable winning unit; units used evenly | ≥ 95% of occurring bytes stably assigned | 10⁵ bytes heard |
| R2 | Classes and boundaries | boundary positions consistent across records; class assignments stable | ≥ 95% consistent boundaries on held-out records | 10⁵ bytes heard |
| R3 | Output format | end-of-answer at the right length; emitted bytes are digit bytes in the right places | ≥ 95% format-correct answers on unseen problems | 3×10³ records |
| R4 | Coverage on learned codes (evaluator tool) | external ridge readout from frozen combination-cell codes, fitted on 10³, 3×10³, 10⁴, 3×10⁴, 10⁵ problems; unseen per-digit accuracy curve | curve rising and ≥ 95% by 10⁵ at some N_KC | analysis only |
| R5 | Columns without carry, learned by the fly | per-digit accuracy on unseen problems for digits with no incoming carry | ≥ 90% | 3×10⁴ records |
| R6 | Carries, subtraction with sign, then exact whole answers | per-digit, carry-digit, sign and exact accuracy on unseen problems; one extra digit length; retention | v4 gates G2–G5 | 10⁵ records per stage |

R4 is an external diagnostic: if the best possible linear readout from the fly's own codes cannot generalise even
with ample data, the local learner cannot either, and N_KC or wiring must change before R5.

## 8. Comparisons (from R5 on, where they can matter)

| Arm | M6 consolidation | M4 topographic wiring | M3 segments |
|---|---|---|---|
| C-full | recurrence-gated | on | on |
| C-native | native allocation | on | on |
| C-random | recurrence-gated | random wiring, equal N_KC | on |
| C-noseg | recurrence-gated | on | off (position since record start only) |

Controls: shuffled answers; no-write history. Predictions (to be locked before R5): C-full exceeds C-native on unseen
problems, increasingly with experience; C-random and C-noseg near chance on unseen columns.

## 9. Reporting

At every rung: partial-ability curves against records (format, per-position digits, carry digits, sign, exact), cost,
and the rung verdict. Exact whole answers on unseen problems remain the endpoint for any claim; partial curves guide
development and make failures diagnosable.

## 10. Build order

* **B0** Fast engine for one shared combination-cell population with nine native output channels (Codex's
  `native_engine.py` from v4 is the starting point; its 5,177-cell parity tests carry over).
* **B1** M1–M3 and the listening phase → R1, R2.
* **B2** M4, M5 with the reference and sign unit test → R3.
* **B3** R4 coverage curve at 2¹⁶, 2¹⁸, 2²⁰ cells.
* **B4** M6, M7 → R5 with arms C-full, C-native.
* **B5** R6 with all arms; trimming; sample-efficiency curves.

Technical worlds only; formal science only on separate approval.

## 11. Risks

* **Classes may not separate digits from operators** if contexts are too similar; R2 measures it directly before
  anything depends on it.
* **Coverage may require very large N_KC** for carries spanning several columns; R4 quantifies it.
* **Consolidation thresholds** may admit too little or too much; confirmation statistics are logged per channel.
* **Standard written answers** put the leading digit last in the carry chain; it is expected to lag and is reported
  separately.
