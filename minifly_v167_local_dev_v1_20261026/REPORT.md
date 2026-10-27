V167 LOCAL V1: PASS (reduced form). A learner above the KC code with no back-propagation anywhere learns a layer that carries over to a new author as well as V165's layer under a back-propagated top. The second-pair rehearsal was not run (record cap). DEV_NOT_EVIDENCE.

# V167 report: no back-propagation anywhere in the learner (27 October 2026)

**Question:** V165's layer H learned from a broadcast error, but its top block still used back-propagation. If the top is made local too (direct feedback for its hidden layer, a delta rule for its readout), does the layer still learn features that carry over to a new author?

**Answer: yes.**
* On *Edgeworth* carry-over, the fully local learner's layer scores 3.134. V165's layer under a back-propagated top scored 3.126; the difference is 0.008 bits (SD 0.004).
* It beats the direct KC readout by 0.196 bits and its shuffled control by 0.317 bits.
* It recovers 85% of back-propagation's gain over the random layer.
* **The cost of a local top shows in the learner's own fit to its training text, not in the transferred layer.** Its *Thursday* calibration bits are 3.430, against 3.051 for V165.

**Files:**
* Design: `minifly_v167_local_design_20261026/DESIGN.md`.
* Grant: `AUTHORIZATION.json`.
  * Grant 1, "Okay, you can do these.", had default caps.
  * Grant 2, "3 hours should be fine. No need for a lot of work as we are not testing scaling law etc.", raised the CPU cap to 3 h. Records stay at the default 5,000.
* Ledgers in `runlog/`: 6 attempts; cycles testbed *revise*, testbed *continue*, technical *continue*, dev *stop*; amendment 1.

## C0 testbed (references; no learner records)
**Second text pair (*Emma* → *The Ball and the Cross*, V155's unused rehearsal pair):**
* The cached-code path equals MiniFly's own encode at every position of both texts.
* **Amendment 1 (testbed_invalid):** the held-out block of *The Ball and the Cross* was the Project Gutenberg licence footer and release table at the end of NLTK's file, so every reference scored about 5.2 bits.
  * The text is now cut at its "End of Project Gutenberg" record (251 records dropped), and the export and references were re-run.
  * *Emma* is unchanged.
  * None of the other five V155 texts has a footer, the reserved official pair included.
  * Made before any learner touched the pair.
* **References after the fix** (3-seed carry-over probe on *The Ball and the Cross*):

  | Layer | Bits |
  |---|---:|
  | RAND-H | 3.552 |
  | TOP | 3.374 |
  | BP-H (learned on *Emma*) | 3.172 |

  * Ceiling met: BP-H beats TOP by 0.202 (SD 0.012; 0.06 required).
  * The pair is ready for a rehearsal, which this delivery did not run.

## C1 technical: 6 of 6 checks pass
* **Updates equal their explicit formulas:** readout delta rule, direct feedback for T with the dropout mask, direct feedback for H (maximum difference 4.2e-17). The gradients are nonzero (checked on random input).
* **No gradient crosses a layer:** with both feedback matrices at 0, only the readout learns.
* **Parity:** with the top switched to back-propagation, the training loop reproduces V165's DFA-H exactly.
* **The shuffle** is a within-batch permutation.
* **Runs are deterministic.**
* **Learns:** after one epoch on 200 lines, 3.896 against 4.051 with H's feedback switched off.

## C2 dev, reduced form (*Thursday* → *Edgeworth*; 3 epochs; the design's form under 5,000 records)
**Carry-over** (held-out bits/symbol, fresh back-propagated probe on *Edgeworth*, mean of seeds 0, 1, 2):

| Arm | Carry-over | Source |
|---|---:|---|
| BP-H: back-propagation into H and through the top | 3.070 | V163 |
| DFA-H(3): direct feedback for H, back-propagated top | 3.126 | V165 |
| **LOCAL: no back-propagation anywhere** | **3.134** | V167 (per seed 3.134 / 3.122 / 3.147) |
| TOP: no extra layer | 3.330 | V163 |
| LOCAL-SHUF: errors to H and T from other positions | 3.451 | V167 |
| RAND-H: random layer | 3.487 | V163 |

**Gating comparisons:**
* LOCAL beats TOP by **0.196** (SD 0.012): pass.
* LOCAL beats LOCAL-SHUF by **0.317** (SD 0.013): pass.
* **φ = 0.85.**

**Also reported:**
* LOCAL against DFA-H(3): **+0.008** (SD 0.004), so making the top local costs almost nothing in the transferred layer.
* LOCAL against BP-H: +0.065 (SD 0.009).

**Training (*Thursday* calibration bits of the learner's own top):**
* LOCAL: 3.638, 3.620, 3.430. The best epoch is 3, the budget edge.
* LOCAL-SHUF: 4.229, 4.491, 4.544. It gets worse: errors from other positions actively damage the fit.
* V165's DFA-H for comparison: 3.298, 3.122, 3.051.

**Alignment** with the back-propagated gradient after epochs 1–3 (non-gating):
* LOCAL: H 0.44 / 0.60 / 0.74; T 0.15 / 0.33 / 0.43.
* LOCAL-SHUF: H 0.16 / −0.04 / 0.23; T 0.01 / −0.07 / 0.11.

**Not run** under the record cap (design §6): DFA-H(5), the fully local carry-over reading on *Edgeworth*, and the C3 rehearsal on *Emma* → *The Ball and the Cross*.

## What it shows
1. **A learner with no back-propagation anywhere learns transferable features above the KC code.**
   * Each synapse changes by presynaptic activity × the unit's own derivative × an output error broadcast through a fixed random pathway.
   * At the readout, the change is presynaptic activity × the output error.
   * The layer this produces is as good for a new author as V165's under a back-propagated top (difference 0.008 bits). It recovers 85% of what full back-propagation gains.
2. **Locality has a cost, but in the top's own fit, not in the representation.**
   * The local top fits *Thursday* 0.38 bits worse than a back-propagated one after 3 epochs, and its alignment to the true gradient is lower (0.43 against 0.74 for H).
   * Yet H, which is what transfers, is unaffected. The first hidden layer aligns faster than the second.
3. **The credit must be input-matched,** as in V165. Shuffled errors make the learner worse with training, and its layer carries over worse than the KC readout.
4. **Reading fixed in advance:** only C2 could run, so the full reading (C2 and C3 pass) is not reached. C2 passes, and LOCAL is within 0.05 of V165's DFA-H. The open item is the replication on the second pair, which needs about 6,600 learner records (LOCAL and DFA-H(5) at 5 epochs on *Emma*). With LOCAL alone at 3 epochs it needs about 2,000.

## Limits
* **Budget:** best epoch 3 is the budget edge, so 3.134 is an upper bound on LOCAL's bits.
* **Evidence:** one text pair and one learner seed; DEV_NOT_EVIDENCE. The second pair is prepared (codes, references, ceiling met) but not used.
* **The carry-over probe** on *Edgeworth* is a back-propagated evaluator instrument, shared with V163–V165. It is not part of the learner. The fully local carry-over reading was not run.
* **Not yet native:** the error e must still reach each unit as a signed, position-specific signal. This is not a native MiniFly result.

## Accounting
* **CPU:** 0.386 CPU-h of child CPU over 6 attempts:
  * C0: 0.324, including the superseded export and references;
  * C1: 0.005;
  * C2: 0.057.
* **Records:** 4,730 learner records (C1 500, C2 4,230).
* **Cycles:** 2 of 3 test cycles; designs tried k = 4.
