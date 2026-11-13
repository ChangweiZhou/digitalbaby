V167 DESIGN (owner go-ahead; written and committed before any V167 code)

# V167 design: a learner above the KC code with no back-propagation anywhere, and a second text pair (26 October 2026)

**Owner's words:** "Okay, you can do these." They answer my listing of the next steps fixed in advance:
* **V165's next step:** "make the top block local too, then test on a second text pair, with caps large enough that training isn't cut off". That is V167, in the main session.
* **V166's next step** is V168, run by a separate agent.

**Caps:** the owner wrote no numbers, so the defaults apply: 2 CPU-h, 5,000 learner records, 3 test cycles. I have asked the owner to write **3 CPU-h, 20,000 records, 3 cycles** for V167.
* Until then, C0 (reference runs, no records) and C1 run.
* C2 and C3 need the raised caps (§6). Without them, C2 runs in its reduced form and C3 does not run.

## 0. Basis
* **V165:** H learned from a broadcast next-byte error through fixed random feedback (DFA-H), under a back-propagated top.
  * *Edgeworth* carry-over 3.126, against BP-H 3.070, the direct KC readout (TOP) 3.330 and the shuffled control 3.440.
  * That recovers 86% of back-propagation's gain.
  * Its best epoch was the last of 3 (the budget edge), and the top still used back-propagation.
* **V167 removes the last back-propagation and replicates on a pair of texts not used so far.**

## 1. The fully local learner (LOCAL)
**Network** (as in V163–V165): KC code x → H (512 ReLU) → T (2,048 ReLU, dropout 0.5) → softmax readout over A's classes.

**Learning: no gradient crosses a layer.** Each layer's input is detached. With e = (p − onehot(y)) / N:
* **Readout:** the delta rule, ΔW_out ∝ −eᵀ z_T, where z_T is T's output after dropout. This is the readout's own local gradient.
* **T:** DFA. δ_T = m ⊙ ReLU′(a_T) ⊙ (e F_T) / (1 − 0.5), where m is the dropout mask and F_T (K × 2,048) is fixed random with variance 1/K, seed 22.
* **H:** DFA as in V165. δ_H = ReLU′(a_H) ⊙ (e F_H), with F_H from V165 (seed 21).
* **Optimiser:** AdamW, learning rate 1e-3, weight decay 1e-4, batch 256.
* **Initialisation:** V165's (RAND-H's H; T and the readout as V163's top at seed 0).

**Arms trained on text A:**
* **LOCAL:** as above.
* **LOCAL-SHUF:** as LOCAL, but the error rows sent to H and T are permuted within each batch (one permutation per batch, fixed seed). The readout keeps its true error.
* **DFA-H(5):** V165's rule (back-propagation within the top), retrained under V167's longer budget. It measures what making the top local costs, and whether V165's gap to BP-H shrinks with more epochs.

**Budget for every learner:** at most **5 epochs**, stopping after 2 without improvement in A calibration bits; best epoch kept.

## 2. Readings
* **Carry-over, the gating reading:** as in V163–V165. H is learned on A and frozen; a fresh back-propagated probe (2,048 hidden units) is fitted on B with seeds 0, 1 and 2; B's held-out lines are scored. The probe is the shared instrument; it is not part of the learner.
* **Fully local carry-over (C2 only, non-gating):** a fresh **local** top (T plus readout, trained as in §1, seed 0, at most 5 epochs) on B over frozen H, for LOCAL's H and for RAND-H's. This tests the pipeline local end to end.
* **Same text:** the jointly trained model's A calibration bits. A probe refitted on A is invalid for layers trained with labels (V163).

## 3. Texts
* **C2 dev:** A = *Thursday*, B = *Edgeworth*, as in V163–V165. The references (RAND-H 3.487, TOP 3.330, BP-H 3.070) are reused, with the same probe, seeds and lines.
* **C3 rehearsal:** A = *Emma*, B = *The Ball and the Cross*. This is V155's rehearsal pair, unused so far.
  * *Emma*: 40 / 5 / 10 KB, 657 taught lines.
  * *The Ball and the Cross*: 10 / 5 / 10 KB, 170 taught lines.
  * Settings are frozen after C2; nothing is tuned on this pair.

## 4. Cycles and rules
**C0 testbed** (no learner code, no records):
* **Codes:** traces of *Emma* (role A) and *The Ball and the Cross* (role B) are computed with V155's untaught P81 front end and coded with the birth keys.
  * Labels and line lengths are checked against V155's corpus rules.
  * The cached-code path equals MiniFly's own encode at every position, as in V155.
* **References on *Emma* → *The Ball and the Cross*** (3-seed carry-over probe):
  * RAND-H;
  * TOP;
  * BP-H, learned on *Emma* by V163's protocol (at most 60 epochs, patience 6).
* **Ceiling:** BP-H beats TOP on this pair by ≥ 0.06 bits. Otherwise C3 is CANNOT_TEST.

**C1 technical:**
* each layer's update equals its explicit formula (float64), including the dropout mask;
* no gradient crosses a layer;
* with the top switched to back-propagation, the code reproduces V165's DFA-H exactly (one epoch, 25 lines);
* the shuffle is a within-batch permutation;
* runs are deterministic;
* **learns:** after one epoch on A's first 200 lines, LOCAL's A calibration bits are ≥ 0.01 below LOCAL with F_H = 0 (H frozen, top local).

**C2 dev (*Thursday* → *Edgeworth*):**
* **Pass:** LOCAL beats **TOP** by ≥ 0.03 bits and ≥ 3 line-bootstrap SDs, **and** beats **LOCAL-SHUF** by ≥ 0.03 bits and ≥ 3 SDs.
* **Reported:**
  * φ = (RAND-H − LOCAL) / (RAND-H − BP-H);
  * LOCAL against DFA-H(5), the cost of a local top;
  * DFA-H(5) against BP-H;
  * the fully local carry-over reading;
  * the alignment of each layer's DFA update with the back-propagated gradient at each epoch.

**C3 rehearsal (*Emma* → *The Ball and the Cross*;** runs only if C2 passes and the cycle closes as *continue***):**
* LOCAL and DFA-H(5) are learned on *Emma* with the settings frozen after C2.
* **Pass:** LOCAL beats TOP by ≥ 0.03 bits and ≥ 3 SDs on the held-out lines of *The Ball and the Cross*.
* **Reported:** φ, and LOCAL against DFA-H(5).

## 5. Readings fixed in advance
* **C2 and C3 pass, and LOCAL is within 0.05 bits of DFA-H(5):** a learner above the KC code with no back-propagation anywhere learns features that carry over to a new author, on two text pairs. Next: native integration (KC → H → MBON, with the error broadcast as a dopamine-like signed signal) and own-byte generation.
* **C2 and C3 pass, but LOCAL is more than 0.05 bits behind DFA-H(5):** a local top works but costs accuracy. Next: feedback that learns to align (for example Kolen–Pollack) for T.
* **C2 passes, C3 fails:** the *Thursday* result does not replicate. Report it as pair-specific.
* **C2 fails while DFA-H(5) beats TOP:** making the top local breaks the learner. DFA through two layers is then the bottleneck.
* **LOCAL does not beat LOCAL-SHUF:** no input-matched credit reaches the layers.
* **A budget edge** (best epoch 5) means a lower bound, not a different rule.

## 6. Accounting and predictions
**Learner records:**

| Item | Records |
|---|---:|
| C1 | about 450 |
| C2: LOCAL, LOCAL-SHUF, DFA-H(5), 5 × 705 lines each | 10,575 |
| C2: two local tops on *Edgeworth*, 5 × 156 lines each | 1,560 |
| C3: LOCAL and DFA-H(5) on *Emma*, 5 × 657 lines each | 6,570 |
| **Total** | **about 19,200** |

* BP-H, RAND-H, TOP and the probes are evaluator fits.
* **Without raised caps:** C2 runs LOCAL and LOCAL-SHUF at 3 epochs (4,230 records), and C3 does not run.

**Estimated CPU:** about 1 h:
* C0 about 0.4 h, most of it the traces of the new texts;
* C1 about 0.05 h;
* C2 about 0.3 h;
* C3 about 0.2 h.

**Predictions:**
* LOCAL passes C2: 0.7;
* LOCAL within 0.05 of DFA-H(5): 0.5;
* C3 passes: 0.65;
* DFA-H(5) within 0.03 of BP-H: 0.35.
