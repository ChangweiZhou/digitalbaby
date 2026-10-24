# V26 recommendation: close the encoding gap

Written 7 September 2026, after the complete V24 archive, the Codex V24 review, and the
completed V25 run (`elm_fly_v25_20260907_094640_43957`, 348/348 trajectories).

**The storage problem is solved well enough to stop working on it. The binding constraint is now
the encoder, and the specific failure is feature dilution, not temporal order. V26 should be a
small encoder experiment on the minimal encoder, with afferent selection and sparsification
policy as the manipulated variables.**

All intervals below are paired Student 95% intervals on independent seed differences, six seeds,
taken from `PAIRED_SUMMARY.csv` (V25) or `PROTECTION_GATES.csv` / `SUMMARY.csv` (V24). They are
descriptive and unadjusted for the size of the screen.

---

## 1. What is settled

These should be fixed as substrate in V26 and not re-tested as primary contrasts.

| Component | Evidence | Δ final pp | CI95 |
|---|---|---:|---|
| Eight locally routed banks | `e3_local_base` vs `shared_frozen_ordinary` | +30.28 | [+26.34, +34.22] |
| Slower bank writes | `e3_local_slow` vs `e3_local_base` | +1.83 | [+0.09, +3.57] |
| | `dual_FF_slow` vs `dual_FF_base` | +2.19 | [+0.60, +3.78] |
| FIFO delayed credit | `target_variable_fifo` vs `target_immediate` | +1.56 | [+0.02, +3.10] |
| int8 stored readout | `local_base_int8` vs `e3_local_base` | +0.44 | [−0.44, +1.32] |
| Protection with release | `split_stable_base_release` vs `..._local` | +1.60 | [+1.22, +1.98] |
| Address-pathway freezing | `dual_FL_base` vs `dual_LL_base` | +6.67 | [+3.37, +9.97] |

Two of these deserve comment.

**FIFO delayed credit is the major capability gain of V25.** V24 could only recover delayed
teaching with an exact-latency oracle, which the Codex review correctly identified as a
substantial prior. A FIFO trace-selection policy under *variable* delay now matches immediate
teaching. Broadcast still fails badly (−25.34 [−28.04, −22.63] at fixed delay; −17.07
[−19.62, −14.51] at variable delay). Any sequential extension depends on this, because in a
sequence every teaching signal arrives with intervening events.

**int8 is free.** Both precision contrasts have intervals spanning zero with point estimates
slightly positive. This is a real memory reduction at no measurable accuracy cost and should be
the default.

**Blended reading should be the default for anything with transferable structure.** On the
structured task, blended beats hard by 7–12pp across every bank arm (e.g. `structured_fixed`
62.84% blended vs 50.74% hard). Selective writes and broad reads are not in tension.

---

## 2. What is closed — do not revisit

| Mechanism | Result | Source |
|---|---|---|
| Output-column gain modulation | No effect; all arms identical | V20 |
| Readout-side compartmented addressing | 1.43× *more* collateral damage than uniform | V22 |
| Learning-rate cascades (`fast_slow`) | 35.7% vs ordinary 42.9% | V23/V24 live |
| Sparsity increase on top of banks | `active_fraction` 0.125 → −1.0pp vs 63.38% ref | V24 circuit |
| Recurrent readout competition | `recurrence:off` → −1.3pp only | V24 circuit |
| Write spillover | −0.32 [−1.32, +0.69]; −0.27 [−1.37, +0.83] | V25 |
| Broadcast eligibility traces | −25.34 [−28.04, −22.63] | V25 |
| Address/content split as an *upgrade* | Best split 68.88% < shared frozen 69.97% | V25 |

The spillover entry corrects both the earlier Claude reading and the Codex review, which each
gave weight to a +0.4pp V24 point estimate. It does not replicate.

The last entry needs care. The address/content asymmetry is **real** — freezing the address
pathway rescues +6.67 [+3.37, +9.97] from all-live, while freezing the content pathway rescues
only +1.83 [−0.67, +4.33]. But every split architecture tested still loses to simply freezing the
whole encoder. The split is a **mitigation for unavoidable representation drift**, not an
improvement. Keep it available for the live regime (`split_stable` beats `split_live` by +8.21
and +10.63) and do not adopt it otherwise.

---

## 3. The diagnosis

### 3.1 The gap

| Reference | Structured accuracy |
|---|---:|
| Observable-input diagnostic (`structured_observable`, hard) | 79.15% |
| **Order-blind ceiling (analytic)** | **75.00%** |
| Best encoder arm (`structured_minimal`, blended) | 64.88% |
| Ordinary learning (`structured_ordinary`, hard) | 56.00% |

Paired: `structured_observable` vs `structured_ordinary` = **+23.15 [+19.82, +26.47]**.
On the context task the same comparison is **+31.14 [+29.06, +33.23]**.

### 3.2 Correcting an earlier claim

An earlier reading of these results (mine, in conversation) attributed the gap to the encoder
being temporally order-blind. **That was wrong**, and the arithmetic shows why.

The task label is `2*first_category + second_category`, with category = sign of the pulse's first
coordinate. A learner that recovers both signs but not their order distinguishes `{00}`, `{11}`,
and `{01,10}` — giving `0.5 × 1.0 + 0.5 × 0.5` = **75.0%**.

- The observable diagnostic reaches 79.15%, so *order is worth only about +4.15pp*.
- The measured order-erasure cost is 2.95pp [−5.30, −0.59], consistent with that.
- The best encoder arm reaches 64.88% — **10.12pp below the order-blind ceiling.**

So the encoder is not primarily failing to represent order. It is failing to extract the two
category signs at all, and order accounts for at most a third of the loss.

### 3.3 What is actually going wrong

From `cue_bases` / `render_cues` in the V24 stream generator:

- Informative coordinate 0: `±U(0.75, 1.15)`, SD ≈ **0.957**
- Nuisance coordinates 1–31: `N(0, 0.65)`, SD = **0.650**
- Variance ratio: **2.17×**

`V25MinimalEncoder` gives each of 96 cells 8 randomly chosen contacts out of 32:

- P(a cell sees coordinate 0) = 8/32 = 0.25 → roughly **24 of 96 cells** carry any signal
- Per-cell SNR against its 7 nuisance contacts ≈ **0.56**
- Top-k sparsification at `active_fraction` 0.25 then keeps 24 cells, selected by **total
  activation** — which is dominated by nuisance magnitude, not by signal

The observable diagnostic wins because it does none of this: it hands the readout both pulse
windows as 64 dense coordinates with no mixing and no sparsification, so the readout can weight
coordinate 0 directly.

**This is a feature-selection and dilution failure.** And critically, the informative coordinate
carries 2.17× the variance of nuisance — it is discoverable by a **label-free, unsupervised
salience criterion**, computable within the 64-sample calibration pass the minimal encoder
already performs. The fix is available without new supervision and without new exposure budget.

### 3.4 The minimal encoder's status

| Contrast | Δ final pp | CI95 |
|---|---:|---|
| `minimal_base` vs `e3_local_base` | +4.55 | [+0.03, +9.06] |
| `minimal_slow` vs `e3_local_slow` | +4.52 | [−0.90, +9.95] |
| `structured_minimal` vs `structured_local` | +3.22 | [−8.35, +14.79] |
| `context_minimal` vs `context_e3` | +1.05 | [−5.95, +8.05] |

Gate `minimal_temporal_encoder`: passes on two rows, **inconclusive** on one.

The honest statement is **at least as good, not proven better**. But the cost asymmetry is not
statistical. `V25MinimalEncoder` is 8 random contacts and one leaky integrator per cell, three
hardcoded time constants {2, 6, 16}, no recurrence, no micro-MLP, no learned dynamics, and
**zero development training presentations** (64 calibration passes only). E3 costs 384
development training presentations plus recurrence, a per-cell-type MLP, six private states per
cell, and learned taus and gains.

Equal accuracy at zero development cost is the result. Use `minimal` as the V26 base — it is
cheap, simple enough to modify cleanly, and removes a confound (developed-encoder idiosyncrasy)
from every arm.

---

## 4. V26 design

**Base configuration, held fixed:** minimal encoder, 8 locally routed banks, slow writes,
FIFO delayed credit, int8 stored readout, blended read reported alongside hard.
**Tasks:** structured, context, association (association as a non-regression check).
**Seeds:** 6 comparison + 6 fresh confirmation, matching V24/V25 practice.

### Factor A — afferent selection (primary hypothesis)

| Arm | Description |
|---|---|
| A0 | Random 8-of-32 contacts (current minimal) — control |
| A1 | **Variance-ranked contacts**: each cell prefers high-variance coordinates, ranked during the existing calibration pass. Label-free, no added exposure. |
| A2 | **Competitive selection**: local anti-Hebbian competition so cells spread coverage rather than all converging on the same coordinates |
| A3 | Contact-count sweep: 4 / 8 / 16 per cell |

### Factor B — sparsification policy

| Arm | Description |
|---|---|
| B0 | Top-k on total activation (current) — control |
| B1 | Top-k on **z-scored** activation, so a high-variance cell is not selected merely for being loud |
| B2 | **Dense, no sparsification** — isolates how much the top-k stage itself costs. The observable diagnostic is dense; this arm tests whether that is the reason it wins. |

### Factor C — order representation

| Arm | Description |
|---|---|
| C0 | Three taus {2, 6, 16} (current) — control |
| C1 | Wider tau spread {1, 4, 16, 64} |
| C2 | Explicit **asymmetric paired traces**: per-cell fast-minus-slow contrast, an order-sensitive primitive |

Full factorial is 3×3×3 = 27 configurations before the A3 sweep; prune to a screening design
(A × B full, C tested only on the best A×B cell plus the A0B0 control) to keep the run near
V25's ~350 trajectories rather than V24's ~3,500.

### The falsifiable prediction

If the section 3.3 diagnosis is correct, **A and B dominate C**. Specifically:

- A1 or B1/B2 should recover the bulk of the 10.12pp deficit against the 75.0% order-blind ceiling
- C should add at most ~4pp on top, since that is all order is worth on this task

If instead C dominates and A/B are flat, the diagnosis is wrong, order was the binding constraint
after all, and the encoder needs temporal machinery rather than better afferent selection. Either
outcome is informative; state the prediction in the manifest before running.

### Decision gates

1. **Primary**: structured blended accuracy against the 75.0% order-blind ceiling and the 79.15%
   observable ceiling. Report distance to both, not just deltas against `structured_ordinary`.
2. **Non-regression**: association final must not fall more than 5pp below `e3_local_slow`
   (69.97% hard) or `minimal_slow` (74.49% hard).
3. **Non-regression**: context blended must not fall below `context_e3` (56.84%).
4. Any arm passing (1) must also hold (2) and (3). One passing row is not sufficient — this is
   the same standard the V25 manifest applied to `minimal_temporal_encoder`.

---

## 5. What not to do yet

**Do not start the bit-level language proof of concept.** The unblocking condition is an encoder
that reaches within ~5pp of the observable ceiling on structured while holding context and
association. Until then, a language run would be limited by the same dilution failure and would
produce an uninterpretable result. One encouraging sign: `context_reset` vs `context_e3` is
**−3.22 [−5.96, −0.48]** — on a task that genuinely needs context, resetting state hurts. This is
the properly controlled version of V24's confounded `reset:continuous` arm and vindicates the
Codex review's suspicion that the V24 collapse was a probe-protocol artifact. Persistent state is
not disqualified.

**Do not build the hierarchical / scratchpad / multi-horizon architecture** proposed in the
earlier conversation. The Codex review rejected it on specification grounds (no update path for
stacked fixed encoders; stacking random projections yields no abstraction hierarchy) and V25
strengthens the rejection: the encoder was never the part doing the work, and adding structure to
it is the opposite of the indicated move.

**Do not run another broad tournament.** V23 screened 52 mechanisms; V24 ran 3,498 trajectories.
The mechanism space for *storage* is exhausted. V26 should be small.

---

## 6. Deferred, but still open

- **Scaling.** Nothing since V20 has left 4 classes / 264 identities. The `coeff384` result
  (65.6% → 38.2% when each bank is restricted to 12 features) proves capacity per bank is
  load-bearing, so the banks × features × identities surface is both open and answerable. Run it
  after V26, when the encoder is no longer the limiting term — measuring a scaling law through a
  known bottleneck would measure the bottleneck.
- **Reward teaching.** `reward_optimistic` vs `reward_existing` is +3.03 [−0.04, +6.10] on final
  but +14.4pp on late acquisition (61.19% → 75.62%). The Codex exploration-confound hypothesis is
  partly confirmed. Worth one clean arm with exploration controlled at matched budget, not a panel.
- **Confirmation of `minimal`.** Its advantage is a point estimate with intervals mostly spanning
  zero. V26 uses it as the base regardless, on cost grounds, but a fresh-seed confirmation should
  ride along.

---

## 7. Provenance

- V24 archive: `elm_fly_v24_20260906_191734_29987`, 3,498/3,498 trajectories, 0 failures.
- V25 archive: `elm_fly_v25_20260907_094640_43957`, 348/348 trajectories, source SHA256
  `eafb1cf48e5014619a13669b10cc6ce327cdeb20d8fa718723f7cf967e5e56b3`.
- Codex V24 review: `v24_complete_results_review.md`, 7 September 2026. Its two reporting
  corrections (merged circuit overrides under an unnamed baseline; contaminated order-erased
  controls) are accepted and reflected above.
- Task signal structure read from `cue_bases` / `render_cues` and the structured stream builder in
  `elm_fly_local_memory_v24.py`; encoder definitions from `V25MinimalEncoder` and
  `V25ObservableEncoder` in `elm_fly_address_content_v25.py`.

Two corrections to earlier Claude analysis are recorded in this document: the spillover benefit
(section 2) and the order-blindness diagnosis (section 3.2). Both were stated with more
confidence than the data supported.
