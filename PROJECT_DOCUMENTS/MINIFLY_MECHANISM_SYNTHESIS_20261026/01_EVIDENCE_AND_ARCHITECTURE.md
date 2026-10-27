# Memo 01 — What is established, and what remains architectural speculation

## 1. Correcting the original architectural assumption

The PN→KC→MBON-centered Full151 lineage is a tested associative-memory substrate, not a demonstrated sufficient architecture for reusable sequential computation or human-style learning. It has sensory traces, adaptation, fast/slow native plasticity, KC competition and autoregressive byte feedback; it is **not** a stateless three-layer perceptron. Nevertheless, it has not demonstrated a learned cycle in which **retrieved knowledge changes subsequent internal computation**.

The absence of human-brain anatomical similarity is not a proof of incapacity. Conversely, an external free-readout finite interpolation (V156) or a universal approximation argument cannot establish that native local learning reaches the needed computation in feasible training.

Separate **finite expressivity**, **reachable learning dynamics**, **causal memory use**, **relevant retention**, **never-taught transfer**, and **actual cost**.

## 2. Outcome ledger (source reports, not reproduced)

| Version | Measured result | Interpretation boundary |
|---|---|---|
| V157 (Codex) | At matched final eight-per-cue dose, clustered vs interleaved improves taught NEW exact by 0–1/16. Held W minus SH_OLD is 0 in both worlds; one world lost five initially correct OLD over longer history. | No established unseen addition or unique cause of retention loss; missing 992 immediate-margin arrays in secondary artifact. |
| V158 | The chosen oracle keys never entered top-k in a constructed same-future-sharing fixture. | A failure of this constructed oracle and competition regime, not a universal impossibility of learned KC keys. |
| V159–V161 | A learned next-receptor predictor has shared-class information; standalone G ~0.80–0.82 synthetic unseen accuracy. Joint feature readout suppresses G; gating improves arbitration but misses declared threshold on one seed. | Synthetic, external fitted readouts; not memory-guided computation. |
| V162 | On Thursday→Edgeworth, calibration selects specific pathway S alone; held 3.320 bits/symbol; no carry-over gain. | No evidence the proposed auxiliary G adds conditional value on this text assay. |
| V163 | Locally contrastive SSL-H carry-over 3.358 vs RAND-H 3.487, TOP KC 3.330, BP-H 3.070; relative gain over random layer ~31% of BP-H's. | It does not beat the stronger original KC baseline. BP-trained evaluator top; one development text pair. |
| V164 | S-COPY x(t−8): BPTT 1.152, TBPTT-1 2.473, COLA 2.474 bits. | A bounded negative result for this COLA implementation; not an algorithm-family impossibility theorem. |
| V165 | DFA-H 3.126, BP-H 3.070, TOP 3.330, shuffled-feedback 3.440. | Task-matched local feedback can train transferable features; top still uses BP. |
| V166 | Memory LOOP with BPTT: two-hop Q2 0.009 bits /99.75%, NO-LOOP Q2 3.092 bits/13%; deranged memory control fails; query targets cue then intermediate. | Strong synthetic architectural witness, NOT native, raw-byte, or local-learning success. Structured episodic input and hand-gated study writes. |
| V167 | LOCAL source model without cross-layer BP: transferred layer 3.134 vs TOP 3.330, shuffled 3.451, BP-H 3.070; source calibration worse than V165, 3.430 vs 3.051. | External BP-trained carry-over probe remains; second text-pair replication and fully local transfer readout not run. |
| V168 | Two-hop Q2: BPTT 0.526, TBPTT-3 0.605, TBPTT-2 3.302, TBPTT-1 3.431, KF-RTRL 3.335, KF-SHUF 3.657. | Cross-step credit is essential and three-step suffices in this task; the tested unbiased noisy estimator recovers only 3.3% of full BPTT improvement at this budget. |

Details:
- V165's **86%** means (RAND-H − DFA-H)/(RAND-H − BP-H) using this development assay, not 86% of general intelligence.
- V167's **85%** is likewise based on a BP-trained *evaluator* probe for carry-over, not an end-to-end local B learner.
- V166's exact two-hop behavior is obtained through BPTT with explicit item symbols, study/query markers and predefined memory writes; it does not settle biological learnability.
- V168's 3.3% is a **training-performance gap**, not 3.3% accuracy. KF beat its shuffled carried-credit control by 0.322 bits, meaning some line-specific signal survives.
- V168's reference testbed underwent documented budget-grid revisions *before learner-code outcomes*; leave that distinction visible. It is not an untouched independent replication of V166.
- Neither Claude's development versions nor Codex V157 establish target E3: complete autonomously generated, never-taught addition under the native lineage.

## 3. Where positive results align

**Observation 1 — task-directed spatial feedback:** V163's next-state contrastive objective is almost perfectly solved, but representation transfer is weak. On the same forward model, V165 direct fixed-random output-error feedback and V167 fully locally trained source network produce much more useful features. Input-matched shuffled-feedback controls strongly constrain an explanation based only on error-like noise.

**Observation 2 — retrieval as an intermediate operation:** V166's learned state/query route depends causally on memory contents. The LOOP and matched NO-LOOP models both handle one hop; only LOOP manages two hops with BPTT. This identifies a computational role missing from the V159 auxiliary predictive trace, which is a fixed linear projection of a fading KC history at evaluation.

**Observation 3 — temporal credit path:** V168 TBPTT-3 nearly matches full BPTT while TBPTT-2 and immediate learning fail. Correctly routed credit for the *first query* needs two earlier transitions. One-step local outputs and purely diagonal traces, under the tested decomposition, omit that path.

## 4. Main unresolved interfaces

1. How to train the dynamic query and read-to-state controller with a low-variance, bounded-cost, eventually biologically plausible signal.
2. Whether V167 representation gains persist with a fully local new-author readout and on independently held text pairs.
3. How a memory-conditioned error direction can reach earlier query synapses without using BP or dense exact Jacobians.
4. How to replace synthetic markers, hard-gated fact writes and symbolic item coding with learned control while preserving own byte/EOS performance.
5. How to preserve **relevant** earlier knowledge through intervening learning, rather than treating dilution of obsolete facts as failure.

## 5. Sources

- [Codex V157](https://github.com/ChangweiZhou/digitalbaby/blob/ec38e8be63e0a448d387e4698fd619c7368ddf12/minifly_v157_dev_20261026/REPORT.md)
- [V163](https://github.com/ChangweiZhou/digitalbaby/blob/7c2a4d94791adbf2b3ac8516b53acf1695d3cc0c/minifly_v163_sslcola_dev_v1_20261026/REPORT.md)
- [V165](https://github.com/ChangweiZhou/digitalbaby/blob/7c2a4d94791adbf2b3ac8516b53acf1695d3cc0c/minifly_v165_dfa_dev_v1_20261026/REPORT.md)
- [V166](https://github.com/ChangweiZhou/digitalbaby/blob/7c2a4d94791adbf2b3ac8516b53acf1695d3cc0c/minifly_v166_memloop_dev_v1_20261026/REPORT.md)
- [V167](https://github.com/ChangweiZhou/digitalbaby/blob/7c2a4d94791adbf2b3ac8516b53acf1695d3cc0c/minifly_v167_local_dev_v1_20261026/REPORT.md)
- [V168](https://github.com/ChangweiZhou/digitalbaby/blob/7c2a4d94791adbf2b3ac8516b53acf1695d3cc0c/minifly_v168_kfloop_dev_v1_20261026/REPORT.md)
