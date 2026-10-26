V158 DESIGN (owner-approved option A; written before any V158 code)

# V158 design: can a predictive local rule make contexts with a shared future share KCs? (26 October 2026)

**Basis:** V155 (`minifly_v155_keys_dev_v1_20261026/REPORT.md`). L-PRED passed every formula check but failed its fixture. Both local rules lowered the same-future code-overlap index. The diagnosis: they update only already-active KCs, which raises selectivity and cannot recruit a KC into a new context.

**Owner's decision, verbatim:** "Okay, we do option A. Call this V158."

**Scope:** the testbed and technical cycles only, on synthetic streams plus a small cost run on *Thursday*. No dev, rehearsal or official run. Default caps (2 CPU-h, 5,000 learner records, 3 test cycles).

## 1. The single revision (L-PRED-R)
Only one thing changes from V155's L-PRED (design §12.5 and §12.10): **which KCs may update their keys.**
* **V155:** only the active units of each code (the top 92 per side).
* **V158:** for each code, a **band** on each side: the 2 × 92 = 184 units with the largest positive pre-activation under the current keys. That is the active 92 plus the 92 just below the threshold.
  * The forward term updates the band of the predicted input (q_{t+1}, or the replayed q_n), using the prediction (W ã_t)_j = mean of W[j, a_t].
  * The retrograde term updates the band of the context input q_t, using (Wᵀ b̃)_j = mean of W[b, j].

**Unchanged:**
* the score s and the W update, which still use the active codes only;
* η, the hinge margin 0.5, the seeded negative replay, q̂^(j), clipping at 0, rescaling to the birth column sums with resets counted, and the native wiring.

**Why this, and only this:** a near-threshold unit that the context predicts (or that predicts the future) can now strengthen its synapses from the current input and be recruited into that context's code. That is the mechanism by which contexts with a common future can come to share KCs. A unit with no synapse on the current input's channels still cannot change, because its q̂ sum is 0. The native wiring is kept.

## 2. C0 testbed (evaluator-side, before any V158 learner code)
Can the fixture's index detect sharing when sharing is built in? V155's fixture had no such positive control.
* **Streams:** V155's two fixture streams (seeds 0 and 1; S1 = a…h → F1 = q…x, S2 = i…p → F2 = y, z, A…F; F → S1 ∪ S2; 300 lines of 40 bytes). The traces come from the same front end (P81, untaught) and are cached once for C0 and C1.
* **The index (unchanged from V155):**
  * mean code overlap (|x ∩ x′| / 184) over 2,000 random pairs of positions ending in different bytes of the same set,
  * minus that over 2,000 pairs ending in bytes of different sets,
  * with pair sampling seeded 11 + stream seed.
* **ORACLE keys (an evaluator construct, not a learner):**
  * Choose 46 KCs per side for S1 and 46 others per side for S2 (seeded).
  * Replace each chosen KC's incoming weights with the mean PN input over that stream's positions ending in a byte of its set. The weights are dense over all 302 PNs and scaled to sum to 1.114 (the 95th percentile of birth column sums).
* **Verdict:**
  * CONTINUE if ORACLE − FROZEN ≥ 0.05 on both streams (ceiling) and FROZEN ≤ 0.01 on both streams (floor);
  * otherwise CANNOT_TEST_CLAIM_AS_SPECIFIED: the index cannot show sharing even when it is built in, and V158 stops.

## 3. C1 technical
1. **The L-PRED-R update** equals the formula computed independently (dense loops) for one position: forward and retrograde band terms, rescaling and W.
2. **Column sums, signs and wiring** are kept.
3. **Determinism:** two runs give identical keys and W.
4. **Fixture (pass mark unchanged from V155):** L-PRED-R (η 0.03, 2 passes) raises the index over FROZEN by ≥ 0.01 on both streams. V155's L-COMP and L-PRED results on the same streams are cited for comparison and not re-run.
5. **Cost:** CPU per position on 2 KB of *Thursday*.

## 4. Decision rules (fixed here)
* **C0 fails:** stop and report.
* **C1 fixture fails:** stop. No further rule variant is tried without a new owner decision. The write-up records that neither surrogate (active-only, or near-threshold band) produces same-future sharing under the native wiring at this exposure.
* **C1 fixture passes:** report, and ask the owner for dev and rehearsal caps, with C1's measured cost, to run V155's design §5 yardsticks (same-text gain and carry-over) with L-PRED-R in place of L-PRED.
  * The pass shows only that the rule *can* produce same-future sharing in a constructed stream. It is not evidence about text.

## 5. Prediction (recorded before any code)
* P1: L-PRED-R passes the fixture. My probability is 0.4.
* My record on such predictions is poor, which is why the positive control comes first.

## 6. Cost
* C0: fixture traces (about 1 CPU-minute) and two code computations per stream; well under 0.1 CPU-h.
* C1: about 1,300 learner records; at most about 0.2 CPU-h. L-PRED cost 5.6 ms per position in V155, and the band roughly doubles the key-update work.
