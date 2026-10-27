V169 DESIGN (owner go-ahead; written and committed before any V169 code)

# V169 design: a fully local learner against calibrated counting in low-data transfer, with replication and a persistence test (27 October 2026)

**Owner's words:** "No need. 1) Push all the discussions above as a memo after streamlining to the repo in academic language. Be well organized. 2) Launch the two experiments using the skills." V169 is the memo's first experiment (`minifly_memo_credit_horizon_20261027/MEMO.md`, section 7). V170 runs in parallel, by a separate agent.

**Caps:** the defaults, since no numbers were written: 2 CPU-h, 5,000 learner records, 3 test cycles. No official run, no adoption. V169 is sized to fit (§6).

## 0. Questions
* **Q1, replication.** Does the fully local learner of V167 (no back-propagation anywhere) replicate on a second, unused text pair, *Emma* → *The Ball and the Cross*?
* **Q2, counting with native readouts.** With MiniFly-style readouts in place of the evaluator's fitted probes, does that learner beat the **strongest calibrated counting baseline** on the new author with 10 KB? The baseline is interpolated Kneser–Ney or tuned PPM, fitted on both texts.
* **Q3, persistence.** When the same network continues learning on the second text, how much of the first does it retain, and does replay reduce the loss?

## 1. Fixed components (reused unchanged)
* **Texts, codes and references:** V167's rehearsal pair, with *The Ball and the Cross* cut at its Project Gutenberg end marker (V167 amendment 1). The KC codes are V167's cached export, SHA-checked. V167's C0 references on this pair are reused: RAND-H 3.552, TOP 3.374, BP-H 3.172 (3-seed carry-over probe).
* **The learner (LOCAL):** V167's code unchanged (`v167_local.train`, mode `local`), on *Emma*. KC code → H (512) → T (2,048, dropout 0.5) → readout:
  * direct feedback alignment for H and T;
  * a delta-rule readout;
  * AdamW, at most 3 epochs, patience 3 (V167's reduced form).

## 2. Counting baselines (evaluator, no records)
* **PPM-C:** V148's ladder implementation, with the order chosen from {2, 3, 4, 5}.
* **Interpolated Kneser–Ney:**
  * absolute discount D for each order, continuation counts for the lower orders, and a uniform base over the 257 symbols;
  * contexts never cross a line boundary, as in the ladder;
  * order chosen from {2, 3, 4, 5} and D from {0.5, 0.75, 0.9}.
* **Two training sets each:** B's 10 KB only; A's 40 KB followed by B's 10 KB.
* **Hyperparameters** are chosen on B's calibration lines. B's held-out lines are scored once.
* **COUNT\*** is the best of the four (method × training set) on calibration. It is the yardstick for Q2.
* **Computed for both text pairs.** *Thursday* → *Edgeworth* is reported, not gated, because the native readouts are run only on the second pair (§6).

## 3. Native readouts on B (learners; records counted)
* **ONLINE:** V153's online delta rule with warm-up, reimplemented and checked equal to V153's code (C1). For each position: predict, then update by the delta rule; classes are B's seen symbols plus OTHER.
  * η ∈ {0.01, 0.003} and passes ∈ {1, 2}, chosen on B's calibration.
* **Applied to two inputs:**
  * **NATIVE-TOP:** the frozen KC code, with no learned layer;
  * **NATIVE-LOCAL:** LOCAL's frozen H.
* **Fixed scaling:** H's features are scaled so that their mean squared norm on B's training text equals the KC code's. This keeps η comparable between the two inputs.

## 4. Persistence (Q3; learners; records counted)
1. **Start:** LOCAL, trained on *Emma* (§1), at its best epoch.
2. **Continue on B:** the same network keeps learning on B's taught text with the same rule (direct feedback for H and T, delta-rule readout) for 2 epochs. *Emma*'s classes are kept; B symbols not seen in *Emma* map to OTHER.
   * **CONT:** B only.
   * **CONT-REPLAY:** each batch of 256 B positions is accompanied by 256 positions drawn from *Emma*'s taught text (fixed seed).
3. **Measures:** the network's own held-out bits on *Emma* and on B, before and after.
   * **Forgetting:** F = Emma_after − Emma_before.
   * **Learning:** L = B_before − B_after.
   * Line-bootstrap SDs throughout.

## 5. Cycles and rules
**C0 testbed** (evaluator; no learner code; no records):
* V167's code file and references are SHA-matched.
* The counting baselines on both pairs:
  * the Kneser–Ney probabilities sum to 1 over the 257 symbols, within 1e-9, at 200 random contexts;
  * the ladder's PPM-C reproduces the memo's exploratory numbers.
* COUNT\* is selected on B's calibration for each pair.
* **Floor:** COUNT\* on *The Ball and the Cross* is no better than V167's BP-H reference (3.172). Otherwise Q2 is CANNOT_TEST, because counting would already beat the learned ceiling.

**C1 technical:**
* the ONLINE reimplementation equals V153's `online_delta` exactly on random sparse data, and the dense version equals the sparse one on the same data;
* H's scaling follows its formula;
* a CONT-REPLAY batch is exactly half B positions and half *Emma* positions;
* continued learning is deterministic (10 lines);
* cost per epoch is measured.

**C2 dev (*Emma* → *The Ball and the Cross*):**
* **P1, replication:** LOCAL's frozen H with the 3-seed back-propagated probe on B beats TOP (3.374) by ≥ 0.03 bits and ≥ 3 line-bootstrap SDs.
* **P2, counting:** NATIVE-LOCAL beats COUNT\* on B's held-out lines by ≥ 0.03 bits and ≥ 3 SDs.
* **Reported, not gated:**
  * NATIVE-TOP against COUNT\* (does the frozen KC code alone beat counting?);
  * NATIVE-LOCAL against NATIVE-TOP;
  * φ = (RAND-H − LOCAL) / (RAND-H − BP-H) for P1.
* **Q3, descriptive with readings fixed in advance:** F and L for CONT and CONT-REPLAY. **Replay is effective** if F(CONT) − F(CONT-REPLAY) ≥ 0.05 bits and ≥ 3 SDs, with CONT-REPLAY's L at least half CONT's.

## 6. Readings fixed in advance
* **P1 and P2 pass:** a system with no back-propagation in any component, including its readout, beats calibrated counting on a new author with 10 KB, and the representation result replicates on a second pair. This is the memo's low-data claim in native form.
* **P1 passes, P2 fails:** the learned layer replicates, but the native online readout loses what the evaluator probes had. Next: the readout. V153 found an online-learning cost of 0.15 bits.
* **P2 passes only for NATIVE-TOP:** the frozen code's data efficiency, not the learned layer, beats counting.
* **P1 fails:** V167's *Thursday* result does not replicate; report it as pair-specific.
* **Q3:**
  * a large F that replay removes means the slow learner needs consolidation by replay, in line with the CLS round;
  * a small F without replay means the layer is robust to continued learning at this scale.

## 7. Accounting and predictions
**Learner records:**

| Item | Records |
|---|---:|
| LOCAL on *Emma*, 3 × 657 lines | 1,971 |
| ONLINE, 2 inputs × 2 η × 2 passes × 170 B lines | 1,360 |
| CONT, 2 × 170 | 340 |
| CONT-REPLAY, 2 × (170 B + 170 *Emma*-equivalent) | 680 |
| C1 | about 60 |
| **Total** | **about 4,410** (within 5,000) |

**Estimated CPU:** about 0.4 h.

**Predictions:**
* P1 passes: 0.7;
* COUNT\* is interpolated Kneser–Ney fitted on A then B: 0.6;
* P2 passes: 0.5;
* NATIVE-TOP beats COUNT\*: 0.55;
* replay is effective: 0.4.
