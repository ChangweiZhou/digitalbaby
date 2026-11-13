NOTE (planning record; no experiment, no claim promotion). It reviews the newest progress (Claude V172, 8c808c38; Codex WRITER-INTERFERENCE V1, `WCFI-V1`, 52e73caa) and assesses two proposals the owner relayed from ChatGPT: gated recurrent microcircuits with within-circuit temporal sensitivity, and learned upper-layer feedback. Reports were read, not rerun.

# Next steps after V172 and WCFI-V1 (28 October 2026)

## 1. Newest progress
**V172 (Claude; one dev world; arithmetic 0–19):**
* With the stronger delta writer, final complete taught recall rose from 10/80 (N2B, the lineage parent) to 29–39/80. The writer effect is +23 items (SD 4.1).
* Homeostasis added 6 items, below its pre-stated threshold.
* DELTA-H acquired 50 of 64 OLD items. 128 NEW records then erased 27 of those 50.

**WCFI-V1 (Codex; two worlds; opaque byte associations with a shared suffix):** it resolves the two open points of V172.
1. **Strength, not form.** At matched write strength, measured as the displacement each write produces in the readout logits, the N and D update forms are nearly equivalent. Same-state differences in target log-probability are at most 0.02. V172's writer effect was therefore write strength.
2. **The interference is a stability–plasticity trade-off caused by writing.**
   * Both write forms are tested at a low and a high strength, starting from a shared OLD checkpoint with 50 of 64 correct.
   * Low strength keeps OLD (48–53 of 64) but learns almost nothing NEW (0–3 of 16).
   * High strength learns NEW (14–16) but drops OLD to 25–35.
   * Two controls show neither time nor exposure causes the loss:
     * OBS (same events, writes inhibited) gives the same final outputs as IDLE (exact-time rest);
     * OBS keeps OLD.
   * The high-strength totals (49/80 and 41/80) do not beat OBS (51/80 and 52/80): about one OLD item is lost for each NEW item gained.

**Reading.** The native associative memory now acquires taught facts. Its binding limit is interference among items that share context, which is E1-R. Codex's revised design proposes local protection or consolidation next.

## 2. Assessment of the relayed proposals
The proposals were written before V172 and WCFI-V1, and parts are stale. Their premise about V171's controller is superseded: that line is closed. They also do not address the native memory's interference, now the clearest measured limit. On their own merits:

### 2.1 Gated recurrent microcircuits with within-circuit temporal sensitivity
16 circuits × 8 gated neurons, with exact forward sensitivities within each circuit.

**Strengths:**
* It targets the right mechanism. V168 showed that the loop needs about 3 steps of cross-unit credit. Diagonal traces cannot carry it (e-prop equals one-step truncation), and stochastic RTRL was too noisy.
* Exact sensitivities within a bounded circuit are deterministic: no estimator variance.
* Learned retention gates address persistence directly.
* It needs no content-addressed read, so it is compatible with the no-attention requirement.
* The planned controls are the right ones: no gate, and no cross-neuron sensitivity terms.

**Corrections:**
1. **Test it where temporal credit has headroom, not first on text.**
   * On text at 40 KB, exact BPTT gained nothing over one-step truncation: 3.015 against 3.006 bits (V164; also V163).
   * A text-first test would therefore likely be null whatever the mechanism's merit.
   * V164's synthetic testbed has qualified references: on S-COPY, BPTT 1.152 against TBPTT-1 2.473, a 1.32-bit gap.
   * Byte-level addition streams under the next-byte meta-task are the later target.
2. **Independent circuits cannot combine information across circuits non-linearly, as the proposal itself notes.** Addition needs operand and carry interactions. A downstream layer trained by broadcast error (DFA) is the minimal addition.
3. **Stability:** the convex gate update (1 − g)s + g·q keeps the within-circuit Jacobian bounded. No sensitivity clipping should be needed, and none should be added silently.

### 2.2 Plastic upper feedback (weight mirror)
1. **The ceiling is already known.** A perfect mirror of a linear readout gives exactly the one-hop back-propagated update at T. That is V165's configuration: DFA for H, back-propagation for the top, with own calibration bits of 3.051 against 3.430 for the fixed-random V167. The maximum gain is about 0.38 bits of own fit at equal epochs.
2. **Part of the gap is probably budget.** V169's learners ended at the 3-epoch budget edge. A converged baseline is needed before attributing the gap to credit.
3. **A cheaper mechanism exists.** For a linear readout, the Kolen–Pollack rule converges to the transpose without perturbation pulses:
   * B receives the same local update as W_oᵀ (hidden activity × output error), plus decay, so B − W_oᵀ shrinks geometrically.
   * The perturbation-calibrated mirror needs 2,048 perturbation directions per full snapshot, a cost KP avoids.
   * Akrout et al. (2019) evaluate both.
4. **Relevance:** it improves the generic predictor's own fit (issues T1 and C4). It addresses neither temporal credit nor E1-R.

## 3. Options for the next experiment (V173)
| Option | What it tests | Cost (estimate) | Objective distance |
|---|---|---|---|
| A. Gated microcircuits on V164's synthetic testbed | Arms: within-circuit exact sensitivity (GATED-RTRL), against diagonal-only sensitivity (GATED-DIAG) and a fixed leak (LEAK-RTRL), at matched parameters. References: V164's BPTT and TBPTT-1. Pass: ≥ 50% of the S-COPY gap recovered. | about 0.3–0.6 CPU-h; about 4,000–6,000 records | Temporal-credit mechanism (C1) on synthetic delayed tasks; no text or E3 claim |
| B. Native interference remedies at high write strength | Blocked NEW (reference), against interleaved replay of observed OLD records at matched total writes, against homeostasis on. OBS control. | about 0.4 CPU-h; about 6,000–8,000 records | E1-R of the native memory; no E3 |
| C. Learned upper feedback (KP) in the V169 learner | Fixed random B, against KP, against exact transpose (instrument), all at a converged budget. | about 0.3 CPU-h; about 8,000–12,000 records | Generic-predictor own fit (T1, C4); no E3 |
| D. Stop and write up | – | 0 | none |

**Recommendation (separate from the options):** A first, in parallel with Codex on B.
* A answers the largest open mechanism question, temporal credit in an admissible architecture, against references already qualified.
* B overlaps Codex's announced next step, local protection. Dividing the work avoids duplication: Codex on protection, and B's replay and homeostasis arms only if Codex does not cover them.
* C is lower priority until a converged baseline shows how much of the deficit is credit rather than budget.

Option B and probably C exceed the default 5,000-record cap; A may fit it. Each needs the owner's go-ahead, and any number above the defaults must be written by the owner.
