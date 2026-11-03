# MiniFly: review of the memorization → representation link

Reviewed source: `ChangweiZhou/digitalbaby`, branch `codex/minifly-consolidated-20261023@9203640c` (V184-R1 is the most recent). I also checked `claude/charming-clarke-u9g850@f4146856` (V164–V174), `claude/vibrant-cerf-sje726` (the checker/skill kit) and `codex/gemini-human-neuronal-mechanisms-audit-20261025`.
Status: an external diagnostic review. Nothing here is an official MiniFly version, core adoption or E1/E2/E3 claim. The experiments use the project's two development worlds (M1 Persuasion→Sense, M2 Hamlet→Macbeth), one seed each. Code and logs are in `minifly_review_experiments/`.

---

## 0. Main finding

**The fixed sensory code, not the plastic learner, is what currently breaks the memorization → representation link.** I recomputed the FE-PERM/KC sensor from the pinned birth assets; its digest `7e0b78ae…` matches V181's `DATA_AUDIT.json`. Every recent design treats this sensor as immutable. It encodes the whole ~20-byte fading history (decay 0.95) into a 3.5%-active, 5,177-unit random expansion. Nearly every position therefore gets a unique code. Even the best possible linear readout of that code memorizes the training text and generalizes about as badly as a unigram model. If the sensor's decay is changed to about 0.3–0.65, keeping everything else (same PN→KC wiring, same receptors and permutation, same top-k), the same readout beats a 4/5-gram count model. The *unchanged* V181 local learner then reaches 3.20 / 3.67 held bits instead of 4.58 / 5.34.

V175–V184 changed the plastic H/T layers (inhibition, coherence gating, credit signals, shallow variant, H residual). All of them sat downstream of this bottleneck. That explains why every one of them came back negative by a few hundredths of a bit.

---

## 1. Sub-question A: what has been tried, and where exactly is it stuck?

The repository has a clear trail.

* **Memorization works.** The Claude-branch native writer (V172) raised complete taught recall from 10/80 to 29–39/80 [P7]. Codex V171-R1 showed that per-prefix oracle gates could reach 47–57/80 [P9, §2; P11]. Storage is not the problem.
* **Representation has only appeared through fitted probes.** V167/V176 found that locally learned H contains useful signal when read by an *external fitted* probe (≈0.15–0.22 bits) [P12, P13]. The learner's own output did not pick it up (`NO_CONFIRMED_NATIVE_FEATURE_DEPLOYMENT_SIGNAL`) [P13].
* **The continuing learner does not even beat unigram on held text.**
  * V181–V184 NORMAL, M1, A2 held: **4.582 bits**, against a train-fitted unigram of 4.403 [P5] (`minifly_v183_h_credit_dev_20261102/campaign/DECISION.json`). The V181 count controls report interpolated KN-5 at **2.639** [P3].
  * V182 (SHALLOW) was 0.53–0.61 bits worse [P4]. V183 (HPC credit) gave 4.588 vs 4.582 [P5]. V184 (H-residual) was +0.053 bits worse, CI [0.040, 0.065] [P6].
* **V179-DIAG** found train loss falling while held loss rose in all six branches [P2]. The project labelled this "overfitting-like" but said it could not attribute a cause. My results below supply one.

The process documents already flag "goal substitution plus constraint accumulation" [P8]. One constraint was never questioned: the sensory birth (FE-PERM decay 0.95, top-k KC), inherited from Full151 and pinned by hash in every design since [P10].

## 2. Sub-question B: is the bottleneck the learning rule, the dose, or the representation?

I ran four diagnostics. All numbers are bits per byte/EOS target on held records unless marked otherwise.

### B1. Reproduction
I re-implemented the V181 NORMAL learner from `learner.py`: 37,313 parameters, DFA-style fixed random error broadcast to H/T, local output delta rule, Adam lr 1e-3, batch 256, 2 passes over A. Held A2 came out at **4.582**, exactly matching the published value.

Train loss at A2 is **4.659**, which is above unigram *on the training set*. With ~10,178 targets per pass and batch 256, the learner gets about **80 optimizer updates** in total. At A2 it is plainly under-trained.

### B2. Dose / step-count sweep with the unchanged learner and sensor (10 passes, held at fixed pass 10)

| World | lr 1e-3 bs 256 | lr 1e-3 bs 32 | lr 3e-3 bs 256 | best held seen anywhere | unigram | bigram |
|---|---|---|---|---|---|---|
| M1 | 4.291 | 4.171 | 4.163 | ≈4.16 | 4.38 | 3.53 |
| M2 | 4.569 | 4.397 | 4.422 | ≈4.40 | 4.62 | 3.79 |

With more updates the learner fits the training text (train 3.0–3.4) but **held loss saturates just below unigram and then rises**. That reproduces V179-DIAG's divergence. Extra dose turns underfitting into memorization; it does not produce generalization.

### B3. Ceiling of the fixed sensor (converged linear softmax readouts, L2 chosen on calibration)
A linear readout trained to convergence is the best any output-local delta rule could do on a fixed code. I fitted it by full-batch Adam as a diagnostic.

| Input to the readout | M1 held | M2 held | M1 train |
|---|---|---|---|
| KC code, decay 0.95 (current birth) | **4.050** | **4.354** | 1.686 |
| one-hot of the last byte (= bigram) | 3.545 | 3.728 | 3.456 |
| one-hot of the last 2 bytes | – | 3.610 | – |
| count n-gram, absolute discount (n=2/3/4) | 3.53 / 3.04 / 2.87 | 3.79 / 3.54 / 3.52 | – |

On M2, the last byte's identity can be linearly decoded from the KC code with only **72.6%** held accuracy.

**The current sensor is worse than a plain one-hot of the previous byte.** It reaches 1.7 bits on train and 4.05 on held, which is a lookup-table regime. No learning rule placed after this code can exceed this readout's generalization without first rebuilding similarity structure that the sensor has destroyed. H/T could in principle do that, but they see only 256 KCs each and get almost no data.

### B4. Same KC machinery, different decay (only `p ← p·d` changed)

| Decay d | M1 held | M2 held |
|---|---|---|
| 0.05 | 3.056 | 3.459 |
| 0.30 | 2.848 | 3.264 |
| **0.50** | **2.778** | 3.285 |
| 0.65 | 2.833 | 3.383 |
| 0.80 | 3.117 | 3.651 |
| 0.95 (birth) | 4.050 | 4.354 |
| 0.05 ⊕ 0.95 concatenated | 3.395 | 3.816 |
| 0.5 ⊕ 0.95 concatenated | 3.034 | 3.576 |

Generalization peaks at intermediate decay, which is a textbook discrimination–generalization curve. Two points stand out:

* At d=0.5 the fixed code plus a linear readout beats my discounted 4-gram on both worlds.
* *Adding* the long 0.95 trace makes results worse. The extra unique-history dimensions mainly give the readout something to memorize.

### B5. Unchanged V181 local learner on the d=0.5 sensor
Held at a fixed final pass; no checkpoint picking.

| World | lr 1e-3 bs 256, 2 passes (original dose) | lr 1e-3 bs 32, 10 passes | lr 3e-3 bs 32, 10 passes |
|---|---|---|---|
| M1 | 4.547 | **3.196** | 3.236 |
| M2 | 5.332 | **3.666** | 3.803 |

Short A→B lifetime: A for 10 passes, then B for 3 passes, batch 32, lr 1e-3.

| World / sensor | A held after A | B held after A (transfer, B unseen) | A held after B | B held after B |
|---|---|---|---|---|
| M1 d=0.95 | 4.171 | 4.242 | 4.128 | 4.204 |
| M1 d=0.5 | **3.196** | **3.348** | 3.199 | **3.342** |
| M2 d=0.95 | 4.397 | 4.519 | 4.341 | 4.449 |
| M2 d=0.5 | **3.666** | **3.802** | 3.577 | **3.679** |

With the d=0.5 sensor, the learner beats a bigram on held A. It also transfers to the unseen B text, and A does not degrade after B. The dose here is 1,744 records and about 110k targets, which fits inside V181's own 5,000-record / 600k-target caps. The difference from the original run is more, smaller updates, not more data.

**Interpretation.**
* Under the current birth, the plastic learner is not the binding constraint. The sensor is (B3), with update count a secondary constraint (B1).
* On a sensor with sensible similarity structure, the existing 37k-parameter local learner gets most of the way to a converged linear readout. It is still about 0.4 bits behind (3.20 vs 2.78), which is where H/T credit work becomes meaningful again.

## 3. Sub-question C: what does the literature say about this exact link?

The project's 44-paper bibliography [P14] is almost entirely about credit assignment and continual learning. It is missing the literature that directly addresses why a random sparse expansion memorizes well and generalizes poorly:

* **Babadi & Sompolinsky (2014, Neuron) [[R1]](https://doi.org/10.1016/j.neuron.2014.07.035).** They show that "expansion implemented by feed-forward random synaptic weights amplifies variability in the incoming stimuli, and this noise enhancement increases with sparseness of the expanded representation". They also find that "highly sparse expansive representations obtained through synapses that encode the clustered structure of the input reduce both intrastimulus variability and the excess overlaps between stimuli". MiniFly's KC layer is random and 3.5% active, and its input changes with every one of the last ~20 bytes. To the KC layer, each such change is "intrastimulus variability" that gets amplified.
* **Barak, Rigotti & Fusi (2013, J Neurosci) [[R2]](https://www.jneurosci.org/content/33/9/3844).** They argue that "the coding level or sparseness of these neurons' activity controls a trade-off between generalization and discrimination, with the optimal level depending on the task at hand", and warn that "transformations that decorrelate tend to destroy the information about relative distances in the original space, making it harder for the readout neurons to generalize". My decay sweep (B4) is an instance of that trade-off along the temporal axis.
* **Litwin-Kumar et al. (2017, Neuron) [[R3]](https://doi.org/10.1016/j.neuron.2017.01.030).** They found that KC-like sparse random wiring maximizes dimension, but "when input synapses are subject to supervised plasticity, however, dense wiring becomes advantageous". MiniFly's H units see 256 fixed KCs each, which fits the "fixed random expansion" regime rather than learned features.
* **Sun, Advani, Spruston, Saxe & Fitzgerald (2023, Nature Neuroscience; Go-CLS) [[R4]](https://doi.org/10.1038/s41593-023-01382-9).** This paper is closest to Mr. Zhou's three-way split. It identifies a tension in which "unregulated neocortical memory transfer can cause overfitting and harm generalization in an unpredictable world", and proposes "that memories only consolidate when it aids generalization". MiniFly's V179-DIAG pattern (train down, held up) and the replay results in V172/V174 are this phenomenon. Consolidation needs a gate based on predictability, and a predictability estimate is only meaningful on a representation where similar contexts are near each other.

* **Turner, Bazhenov & Laurent (2008, J Neurophysiol) [[R5]](https://doi.org/10.1152/jn.01283.2007).** In *Drosophila*, KC odor responses are "highly selective and, thus sparse". One cause the authors identify is that excitatory inputs from projection neurons "decay rapidly, curtailing temporal integration". So the biological KC layer integrates over a *short* window. MiniFly's sensor instead feeds a ~20-byte decaying history (decay 0.95) into a KC-style hash. That borrows the circuit's pattern-separation role, which V172 shows is good for memorization, and also asks it to generalize, which R1–R3 say random sparse expansions do poorly. The mapping from odor-response integration time to byte-history decay is my analogy, not a claim made by R5.


## 3b. Where sources disagree, and which I find more credible

| # | Disagreement | Sources | My judgement and reason |
|---|---|---|---|
| D1 | **Cause of V179's train-down / held-up divergence.** The project says the data "do not distinguish training underfit, overfit, recency bias, writer magnitude or local-credit geometry" and declines to name a cause. This review says it is mainly the sensor. | P1, P2 vs this review §2 B3–B5 | **This review is more credible on the narrow point, but only on development data.** B3 holds the learner fixed and uses a convex readout, and still shows train 1.7 / held 4.05. A gap that large cannot come from local credit or writer magnitude, because neither exists in a convex readout. The project's caution remains right in that recency and distribution shift are not excluded, since I used the same contiguous splits. |
| D2 | **Where the next fix belongs.** V180–V184 put it in H/T plasticity and credit. The research-reset memo [P8] says the program drifted into proxy constraints but keeps the sensory birth fixed. This review says the sensor comes first. | P4–P6, P8 vs this review | **This review.** Three consecutive H/T changes moved held loss by ≤0.05 bits in either direction [P4–P6]. Changing one sensor scalar moved the same learner by 1.0–1.4 bits (B5). |
| D3 | **Is the fixed KC code a good substrate?** The project's design treats the Full151/KC birth as an adopted core. The theory literature says random sparse expansions amplify within-class variability (R1), trade generalization against discrimination (R2), and favor different wiring when inputs are plastic (R3). | P10 vs R1–R3 | **The literature, together with B3/B4.** R1–R3 are peer-reviewed analytical results. The project never tested the birth's generalization ceiling; its fitness was assumed. Caveat: R1–R3 analyse clustered static stimuli, not byte sequences, so they predict the direction of the effect, not its size. |
| D4 | **Does the local learner have useful features at all?** V167/V176 say yes, via fitted probes (+0.15–0.22 bits) [P12, P13]. V181–V184 own outputs say no, staying worse than unigram [P5, P6]. | P12/P13 vs P5/P6 | **Both are right about different things.** Probe gains measure linearly accessible information; own-output loss measures what the learner deploys after about 80 updates. B1/B2 show the own-output figure is dominated by under-training, so the "no" partly reflects how much training there was, not whether features exist. |
| D5 | **Count-model reference.** The project's calibrated IKN-5 gives 2.639 on M1 [P3]. My untuned discounted 4/5-gram gives 2.87–2.89. | P3 vs base.py | **The project's figure.** Its discount and order were tuned on calibration data; mine were not. I therefore state that the d=0.5 readout (2.778) beats my n-gram but not theirs. |
| D6 | **Is replay/consolidation the main lever?** V172/V174 report retention benefits from replay. V169 notes no forgetting occurred anyway [P9]. Go-CLS (R4) argues unregulated consolidation harms generalization. | P15, P9, R4 | **R4 and P9.** A replay gain cannot be credited to preventing forgetting when there was no forgetting to prevent, and R4 gives a normative reason to gate consolidation by predictability. B5 also shows no A forgetting after B on the d=0.5 sensor, so replay is not the bottleneck at present. |

## 4. Sub-question D: what to do next (concrete steps, no schedule)

1. **Reopen the sensory birth as a variable.** Register a sensor-only experiment that keeps the V181 learner, update rule, data and caps unchanged and varies only FE decay (for example 0.95 vs 0.5). Run it with seeds and fresh worlds. The required falsifier: d=0.5 must beat both unigram and bigram on own-output held loss across the full A→B→corrected-A lifetime, including the EDIT/NOEDIT revision forks I did not run.
2. **Separate the two roles explicitly**, which is Mr. Zhou's memorization/representation split made physical:
   * **Memorization path:** keep a long-history, highly sparse code (the current KC) feeding the native writer, for episodic recall of specific taught items.
   * **Representation path:** a short-timescale or multi-timescale code with graded similarity, for the generalizing predictor.

   Concatenating both into one readout hurt in B4. They should be separate pathways whose outputs are combined, for example through a learned mixture, rather than one feature vector.
3. **Use the converged linear readout as a standing ceiling gauge.** Before testing any new plastic mechanism, report "best convex readout of the fixed code". If the learner is already near that ceiling, more credit-assignment work cannot help.
4. **Fix the update-count starvation.** About 80 updates per A phase is too few for Adam at 1e-3, regardless of mechanism. Smaller batches (32) keep the same data and target budget and give 8× more updates.
5. **Only then return to H/T credit.** The remaining gap is about 0.4 bits on M1 between the learner and the d=0.5 readout ceiling. That is the measurable target for V183/V184-style credit ideas.
6. **For "generalization (reasoning)"**, apply Go-CLS-style selective consolidation. Writer contents should move into the generalizing pathway only where they reduce held or calibration prediction error on the representation path, not on surprise alone. The project itself already flagged that surprise-gating needs testing (`04_POST_V171_CRITICAL_REVIEW` D5).

## 5. Confidence and gaps

**Where sources conflicted** (details in §3b):
- D1: the project's agnostic diagnosis vs the sensor diagnosis here. Resolved toward the sensor for these splits; recency and distribution shift remain open.
- D3: the project's adopted KC birth vs the expansion-coding theory. Resolved toward the theory, since the birth's ceiling had never been measured.
- D4: probe-based "features exist" vs own-output "no gain". Both stand; they measure different things.
- D5: my untuned n-gram vs the project's tuned IKN-5. The project's number is authoritative.
- D6: the replay retention reports vs V169's no-forgetting note and Go-CLS. Credit to replay is not supported.
- **No external source addresses the specific claim** that a 0.95-decay byte trace into a KC hash generalizes poorly. That claim rests only on my experiments (one seed, two previously opened worlds).


* **High confidence:**
  * B1 reproduces the published 4.582 exactly.
  * The sensor-ceiling result (B3) uses a convex fit with L2 chosen on calibration, and the decay effect is large (≈1.3 bits) and consistent across both worlds.
* **Medium confidence:**
  * The optimal decay value. I swept it while looking at held data, though calibration tracks held closely; d≈0.3–0.65 is a broad optimum rather than a tuned point.
  * The lifetime results (B5). They use one seed, two *previously opened* worlds, and no EDIT/NOEDIT revision, EOS or generation checks. By the project's own rules, they are development evidence only.
* **Not shown:**
  * Anything about never-taught addition (E3). For arithmetic, a short trace may lose the operand distances the task needs. A multi-timescale code that keeps similarity at each scale is the more general answer, and it is untested.
  * Whether d=0.5 hurts the native writer's item recall (memorization). That should be measured, and it is part of why step 2 keeps the old KC code for memory.
* **Methodological caveats:**
  * The linear readouts were fitted with autograd/Adam as an evaluator-side diagnostic, the same status as the project's count controls. They are not a native learner. A single-layer softmax gradient is the local delta rule, so the ceiling is attainable in principle by an output-local learner.
  * My n-gram baseline is untuned (absolute discount 0.8), so it is weaker than the project's calibrated IKN-5 (2.639 on M1). The d=0.5 readout (2.778) beats my 4/5-gram but **not** their tuned KN-5. The count-model gap shrinks from about 1.9 to 0.14 bits; it does not disappear.
* **Sources:**
  * The repository results are the project's own primary reports, cited by path.
  * The five neuroscience papers are peer-reviewed primary sources, cited from publisher, PubMed or PMC pages.
  * I did not find prior work testing exactly "fading-trace decay vs generalization in a KC-style byte predictor". The link to the expansion/CLS literature is my synthesis.


## References

Primary literature (peer-reviewed):
- [R1] Babadi B, Sompolinsky H. Sparseness and expansion in sensory representations. *Neuron* 83(5):1213–1226 (2014). https://doi.org/10.1016/j.neuron.2014.07.035
- [R2] Barak O, Rigotti M, Fusi S. The sparseness of mixed selectivity neurons controls the generalization–discrimination trade-off. *J Neurosci* 33(9):3844–3856 (2013). https://www.jneurosci.org/content/33/9/3844
- [R3] Litwin-Kumar A, Harris KD, Axel R, Sompolinsky H, Abbott LF. Optimal degrees of synaptic connectivity. *Neuron* 93(5):1153–1164 (2017). https://doi.org/10.1016/j.neuron.2017.01.030
- [R4] Sun W, Advani M, Spruston N, Saxe A, Fitzgerald JE. Organizing memories for generalization in complementary learning systems. *Nat Neurosci* 26:1438–1448 (2023). https://doi.org/10.1038/s41593-023-01382-9

- [R5] Turner GC, Bazhenov M, Laurent G. Olfactory representations by Drosophila mushroom body neurons. *J Neurophysiol* 99(2):734–746 (2008). https://doi.org/10.1152/jn.01283.2007

Project primary sources (repository reports; numbers quoted in §1–§2):
- [P1] V179 report: https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v179_continuous_local_life_dev_20261102/REPORT.md
- [P2] V179-DIAG: https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v179_lifetime_diagnostic_20261102/REPORT.md
- [P3] V181 count controls (unigram 4.403, IKN-5 2.639): https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v181_mini_local_plasticity_dev_20261102/campaign_20261102/COUNT_CONTROLS.json
- [P4] V182 result: https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v182_shallow_local_dev_20261102/results/REPORT.md
- [P5] V183 decision (NORMAL 4.582 / HPC 4.588 / unigram 4.403): https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v183_h_credit_dev_20261102/campaign/DECISION.json
- [P6] V184 decision (+0.053 bits, CI 0.040–0.065): https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v184_h_residual_deployment_dev_20261102/campaign/DECISION.json
- [P7] V172 writer recall 29–39/80 (Claude branch): https://github.com/ChangweiZhou/digitalbaby/blob/claude/charming-clarke-u9g850/minifly_v172_native_e1_dev_v1_20261027/REPORT.md
- [P8] Research reset memo: https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/PROJECT_DOCUMENTS/MINIFLY_RESEARCH_RESET_20261102/MEMO.md
- [P9] Post-V171 critical review: https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/PROJECT_DOCUMENTS/MINIFLY_MECHANISM_SYNTHESIS_20261026/04_POST_V171_CRITICAL_REVIEW_20261027.md
- [P11] V171-R1 report: https://github.com/ChangweiZhou/digitalbaby/blob/16623c73af101bbf9b017980795bfb957d444f0b/minifly_v171_r1_dev_20261026/REPORT.md
- [P12] V167 report (Claude branch): https://github.com/ChangweiZhou/digitalbaby/blob/claude/charming-clarke-u9g850/minifly_v167_local_dev_v1_20261026/REPORT.md
- [P13] V176 campaign report: https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v176_feature_deployment_dev_20261101/campaign_20261102/REPORT.md
- [P14] Project bibliography: https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/PROJECT_DOCUMENTS/LOCAL_LEARNING_LITERATURE_20261025/PAPERS.md
- [P15] V174 replay report (Claude branch): https://github.com/ChangweiZhou/digitalbaby/blob/claude/charming-clarke-u9g850/minifly_v174_replay_dev_v1_20261028/REPORT.md
- [P10] Sensor source (FE decay 0.95, top-k KC): https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_v181_mini_local_plasticity_dev_20261102/sensor.py

Experiments by this review: `minifly_review_experiments/` (run1.py = B1/B2, base.py = B3, sens.py = B4, run2.py/life.py = B5; logs alongside).
