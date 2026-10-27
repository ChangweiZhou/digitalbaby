V169 DESIGN, REVISION 2 (owner go-ahead; revised before any V169 code was written or run)

# V169 design: a persistent local learner against counting under a matched protocol, with replication and retention (27 October 2026)

**Owner's words:** "No need. 1) Push all the discussions above as a memo after streamlining to the repo in academic language. Be well organized. 2) Launch the two experiments using the skills." V169 is the memo's first experiment (`minifly_memo_credit_horizon_20261027/MEMO.md`, section 7). V170 runs in parallel, by a separate agent.

**Why revision 2:** revision 1 (commit 1ba07110) was premised on the memo's exploratory finding that learners beat counting in low-data transfer. That finding came from a script error, which added 1 bit to every counting number (memo erratum, commit f88fe8e1). Corrected, counting pre-trained on the first text beats every learner by about 0.4 bits.

As the corrected memo notes, the comparison was not like for like: counting carries its whole predictor from A to B, while the learners refit only a readout on B. Revision 2 replaces revision 1's question Q2 with the matched comparison, and drops the native online readouts to keep within the record cap. No V169 code existed when this revision was made.

**Caps:** the defaults: 2 CPU-h, 5,000 learner records, 3 test cycles. No official run, no adoption.

## 0. Questions
* **Q1, replication.** Does V167's fully local learner (no back-propagation anywhere) replicate on a second, unused pair, *Emma* (A) → *The Ball and the Cross* (B)?
* **Q2, matched comparison with counting.** A persistent learner learns A and then continues learning on B. Compare its held-out bits on B with the strongest calibrated counting model fitted on A and then B.
* **Q3, retention.** After continuing on B, how much of A does the learner retain, with and without replay? Counting's retention is computed for comparison.

## 1. Fixed components
* **Texts, codes and references:** V167's rehearsal pair, with *The Ball and the Cross* cut at its Gutenberg end marker (V167 amendment 1). KC codes are V167's cached export, SHA-checked. V167's C0 references are reused for Q1: RAND-H 3.552, TOP 3.374, BP-H 3.172.
* **The learner, LOCAL:** V167's code unchanged (`v167_local.train`, mode `local`).
  * Network: KC code → H (512) → T (2,048, dropout 0.5) → readout.
  * Learning: direct feedback alignment for H and T, a delta-rule readout, AdamW, batch 256, at most 3 epochs, patience 3.
  * Its own top (T and readout) makes its predictions; classes are A's seen symbols plus OTHER over 257 symbols.

## 2. Arms (learners; records counted)
* **LOCAL-A:** LOCAL trained on A, with the best epoch chosen on A's calibration bits.
* **CONT:** LOCAL-A continues learning on B's taught text with the same rule, for at most 3 epochs. The epoch with the best B calibration bits is kept. A's classes are kept; B symbols unseen in A map to OTHER.
* **CONT-REPLAY:** as CONT, but each batch of 256 B positions is accompanied by 256 A positions drawn from A's taught text (fixed seed).
* **LOCAL-B:** LOCAL trained from scratch on B's taught text only, with the same rule and budget, using A's classes for comparability. This shows whether learning A helps the learner on B.

## 3. Counting (evaluator; no records)
* **Models:**
  * PPM-C (V148's ladder, unchanged), order chosen from {2, 3, 4, 5};
  * interpolated Kneser–Ney: absolute discount D, continuation counts at lower orders, uniform base over 257 symbols, contexts that never cross a line; order chosen from {2, 3, 4, 5} and D from {0.5, 0.75, 0.9}.
* **Training sets:** A only; B only; A then B.
* **Selection:** hyperparameters are chosen on B's calibration lines for the B readings, and on A's calibration lines for the A reading.
* **COUNT\*** is the best model fitted on A then B, by B's calibration bits. It is the yardstick for Q2.
* **Retention reading for counting:** A's held-out bits under the A-only model and under the A-then-B model.

## 4. Cycles and rules
**C0 testbed** (evaluator; no learner code; no records):
* V167's codes and references are SHA-matched.
* Kneser–Ney probabilities sum to 1 over 257 symbols, within 1e-9, at 200 random contexts.
* The ladder's PPM-C reproduces the corrected memo numbers (PPM-4, A then B: 2.9548 on this pair) within 1e-4.
* The counting grid is run and COUNT\* selected.

**C1 technical:**
* a CONT-REPLAY batch is exactly half B and half A positions, with A's positions drawn from A's taught text only;
* continued learning starts from LOCAL-A's exact parameters and optimiser-free state (a fresh AdamW, as specified);
* continued learning is deterministic (10 lines);
* B symbols unseen in A are scored as OTHER;
* cost per epoch is measured.

**C2 dev:**
* **P1, replication:** LOCAL-A's frozen H, with the 3-seed back-propagated probe fitted on B, beats TOP (3.374) by ≥ 0.03 bits and ≥ 3 line-bootstrap SDs.
* **P2, matched comparison:** the persistent learner (CONT or CONT-REPLAY, whichever has the better B calibration bits) beats COUNT\* on B's held-out lines by ≥ 0.03 bits and ≥ 3 SDs.
* **Reported, not gated:**
  * the signed gap between the persistent learner and COUNT\*;
  * LOCAL-B against CONT (does A help the learner?);
  * COUNT\* against counting on B only (does A help counting?);
  * φ for P1.
* **Q3, readings fixed in advance:**
  * forgetting F = A_after − A_before (the learner's own A held-out bits), for CONT and CONT-REPLAY;
  * **replay is effective** if F(CONT) − F(CONT-REPLAY) ≥ 0.05 bits and ≥ 3 SDs, with CONT-REPLAY's B held-out bits no worse than CONT's by more than 0.05;
  * counting's change in A held-out bits from adding B is reported beside them.

## 5. Readings fixed in advance
* **P2 passes:** under a matched protocol, a persistent learner with no back-propagation anywhere beats calibrated counting on a new author. The memo's corrected section 4 attributed the gap to the protocol, not to the learner, and this would confirm it.
* **P2 fails:** counting's advantage survives a matched protocol. The signed gap measures how far the learner's representation and memory fall short of exact n-gram statistics at 50 KB. That deficit then becomes the first-order problem on the representation side, ahead of native integration.
* **P1 fails:** V167's *Thursday* result is pair-specific.
* **Q3:**
  * a large F removed by replay means the slow learner needs replay consolidation (as in the CLS round);
  * a small F means the layer is robust to continued learning at this scale.

## 6. Accounting and predictions
**Learner records:**

| Item | Records |
|---|---:|
| LOCAL-A, 3 × 657 | 1,971 |
| CONT, 3 × 170 | 510 |
| CONT-REPLAY, 3 × (170 + 170 A-equivalent) | 1,020 |
| LOCAL-B, 3 × 170 | 510 |
| C1 | about 60 |
| **Total** | **about 4,070** (within 5,000) |

**Estimated CPU:** about 0.3 h.

**Predictions:**
* P1 passes: 0.7;
* COUNT\* is Kneser–Ney: 0.6;
* P2 passes: 0.15;
* CONT beats LOCAL-B: 0.85;
* replay is effective: 0.4.
