V169 NATIVE V1: P1 PASS (the fully local learner replicates on a second text pair); P2 FAIL (under a matched learn-A-then-B protocol, calibrated counting beats the persistent local learner by 0.59 bits). Q3: no forgetting was observed. DEV_NOT_EVIDENCE.

# V169 report: a persistent local learner against calibrated counting (27 October 2026)

**Questions** (design revision 2):
* **Q1, replication.** Does V167's fully local learner replicate on *Emma* (A) → *The Ball and the Cross* (B)?
* **Q2, matched comparison.** A persistent local learner trained on A and then B, against the strongest calibrated counting model fitted on A and then B.
* **Q3, retention.** How much of A is retained after continuing on B, with and without replay?

**Answers:**
* **Q1: yes.** The locally learned layer, read by the back-propagated probe on B, beats the direct KC readout by 0.112 bits (SD 0.010). It recovers 76% of back-propagation's gain (φ), against 85% on *Thursday* → *Edgeworth*.
* **Q2: no.** Interpolated Kneser–Ney scores 2.819 bits/symbol on B's held-out lines. The persistent learner scores 3.407. The gap is dominated by the learner's own predictive quality at the 3-epoch budget (see "What it shows").
* **Q3: no forgetting.** Continuing on B *improved* the learner's predictions on A. The pre-stated replay rule passes literally, but for a reason other than the one it was designed to detect.

**Files:**
* Design: `minifly_v169_native_design_20261027/DESIGN.md`, revision 2. It was revised before any code, after the memo erratum (commit f88fe8e1) withdrew revision 1's premise.
* Grant: `AUTHORIZATION.json` (default caps).
* Ledgers in `runlog/`: 4 attempts; cycles testbed *continue*, technical *continue*, dev *stop*; amendment 1 (operational).
* Memo: `minifly_memo_credit_horizon_20261027/MEMO.md`, with its erratum.

## C0 testbed (counting; evaluator; no records)
**Checks, all passing:**
* V167's codes and code file match by SHA;
* interpolated Kneser–Ney sums to 1 over 257 symbols (maximum error 8e-15);
* PPM-4 fitted on A then B reproduces the corrected memo value, 2.9548.

**Counting on B's held-out lines** (best configuration chosen on B's calibration lines):

| Training text | PPM-C | Interpolated Kneser–Ney |
|---|---:|---:|
| A only | – | 3.014 (order 5, D 0.9) |
| B only | 3.144 (order 3) | 3.018 (order 5, D 0.9) |
| A then B | 2.955 (order 4) | **2.819** (order 5, D 0.9) = **COUNT\*** |

COUNT\* lies at the edge of the grid (the highest order and largest discount), so counting may be stronger still.

## C1 technical: 5 of 5 checks pass (after amendment 1)
* **Amendment 1:** the first attempt failed the determinism check, because the continuation function did not seed the generator that the dropout uses. It now seeds it at its start, as V167's trainer does. The failed output is kept.
* Replay batches are half B and half A positions.
* Continued learning leaves the starting network untouched.
* Training is deterministic.
* B symbols unseen in A are scored as OTHER.

## C2 dev
**P1, replication** (3-seed back-propagated probe fitted on B over the frozen H; V167's references on the same seeds and lines):

| Layer | B held-out |
|---|---:|
| BP-H (back-propagation, V167 C0) | 3.172 |
| **LOCAL-A's H (no back-propagation anywhere)** | **3.262** (per seed 3.257 / 3.264 / 3.265) |
| TOP (KC code directly) | 3.374 |
| RAND-H | 3.552 |

LOCAL beats TOP by 0.112 (SD 0.010): **pass**. φ = 0.76.

**Own-top held-out bits** (the learner's own predictions; A's classes):

| Arm | A held-out | B held-out | Best epoch / budget |
|---|---:|---:|---:|
| LOCAL-A (trained on A) | 3.273 | 3.943 | 3 / 3 |
| CONT (then B) | 3.215 | 3.443 | 1 / 3 |
| CONT-REPLAY (then B, with A replay) | **2.923** | **3.407** | 3 / 3 |
| LOCAL-B (B only, from scratch) | 4.106 | 3.917 | 3 / 3 |
| *COUNT\* (Kneser–Ney, A then B)* | *2.304* | *2.819* | – |

**P2:** the persistent learner, chosen on B's calibration lines, is CONT-REPLAY. COUNT\* beats it by **0.588** bits (SD 0.032): **fail.**

**Q3:**
* **Forgetting** F = A_after − A_before: −0.058 for CONT and −0.350 for CONT-REPLAY. Both are negative: A *improved*.
* **The replay rule** (F(CONT) − F(CONT-REPLAY) ≥ 0.05 bits and ≥ 3 SDs, with B no worse by more than 0.05) passes literally: 0.291 bits, SD 0.027, and B is better by 0.036.

**Other comparisons (not gated):**
* **Learning A helps the learner on B:** LOCAL-B is worse than CONT by 0.474 (SD 0.025).
* **Learning A helps counting on B:** by 0.199.
* **Counting's retention of A** improves slightly when B is added: 2.341 → 2.304.

## What it shows
1. **The representation result replicates.** On a second, previously unused text pair, the layer learned with no back-propagation anywhere carries over to a new author, beating the KC readout by 0.11 bits and recovering 76% of back-propagation's gain.
2. **The reading fixed in advance for P2: counting's advantage survives a matched protocol.** Kneser–Ney on 50 KB beats the persistent local learner by 0.59 bits.
3. **Qualification (non-gating): most of that gap is the learner's own predictive quality at the 3-epoch budget, not its representation.**
   * The learner's own top predicts its training author at 3.273 bits, against 2.341 for counting on the same text: a gap of about 0.9 bits.
   * Its best epochs are at the budget edge.
   * The same layer read by a fitted probe does better on B (3.262) than the learner's own top does after continued training (3.407).
   * V167 found that the local top fits its training text 0.38 bits worse than a back-propagated top at equal epochs.
   * For comparison, fully trained back-propagated learners came within about 0.1 bits of counting on the same text (headroom check).

   How much of the 0.59-bit gap would remain with a converged local learner is therefore not known. It needs a longer training budget than the default caps allow.
4. **Q3: no forgetting at this scale; the replay result reflects undertraining.**
   * Continuing on B improved A, so the two authors' shared statistics outweigh interference.
   * Replay improved A further because it gave an undertrained learner more training on A, not because it prevented forgetting.
   * The pre-stated rule passes, but its premise (forgetting that replay removes) is absent. The result is not evidence of consolidation by replay.

## Limits
* **Scale:** one text pair, one learner seed; DEV_NOT_EVIDENCE.
* **Budget:** all learners are budget-limited (best epoch at the edge, except CONT, which overfits B's 10 KB after one epoch). COUNT\* is also at its grid edge.
* **Forgetting:** the persistence test spans two related authors and three epochs. Forgetting under longer or more dissimilar sequences is untested.

## Accounting
* **CPU:** 0.055 CPU-h over 4 attempts:
  * C0: 0.002;
  * C1: 0.006, including the failed attempt;
  * C2: 0.046.
* **Records:** 4,111 learner records (C1 100, C2 4,011).
* **Cycles:** 2 of 3 test cycles; designs tried k = 3.
