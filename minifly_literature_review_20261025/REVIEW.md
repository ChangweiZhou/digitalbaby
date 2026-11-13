# Literature review for MiniFly's next step (25 October 2026)

**How this was read**
* **Read in full**, main text and appendices, from the PDFs supplied:
  * Zihan, Delrocq, Gerstner & Bellec, *Can Local Learning Match Self-Supervised Backpropagation?*, ICML 2026 (Codex list P27);
  * Wang, Gao & Chen, *Global credit assignment via dynamical criticality* (COLA), ICML 2026 (Codex list P21).
* **The other entries of Codex's 44-paper list** (`PROJECT_DOCUMENTS/LOCAL_LEARNING_LITERATURE_20261025/`, codex branch) are discussed from Codex's summaries and from my own knowledge of the well-known papers. Those statements are marked *(from memory)*. Full texts could not be fetched: this container's network policy blocks arxiv.org, neurips.cc, nature.com and similar hosts.
* **One relevant paper is missing from the list.** It was found by search and is read at abstract level only: FlyVec (Liang et al., ICLR 2021).

## 1. What the review has to answer
MiniFly's measured gaps on raw text:
1. **Context.** The native KC state carries less than bigram-level context. The best possible readout of the V148 front end's state scores 3.77 bits/byte against bigram 3.55. FE0 does better at 3.38. PPM-2 is 3.05. (V149, `minifly_v149_input_ceiling_dev_20261025/REPORT.md`.)
2. **Output.** The fixed random 16-bit output code caps prediction at 7.03 bits/byte with no context. A graded 16-number code learned from the text would suffice. (`minifly_v148_explore_s0_20261025/EXPLORE.md`.)
3. **Teaching.** Native writes do execute a local error signal linearly (Codex V151). Output-only fixes did not help natively at Codex's tiny V150 dose.

The user's constraints:
* no back-propagation, Transformers, ELM or pretrained models;
* local rules and gradual warmup;
* own bytes and own end-of-line;
* small data;
* the result must carry over to later language tasks.

## 2. Zihan et al. 2026: local self-supervised learning can match self-supervised back-propagation

**What it shows**
* **Theory (Theorem 3.1).** In a deep *linear* network with orthonormal weight matrices, a layer-local contrastive loss gives exactly the same weight update as back-propagating the last layer's loss. This holds for CLAPP-type losses with a trainable lateral matrix B, and for Forward-Forward-type losses with B = I.
  * The conditions are strict: B must be learnable and unconstrained, every layer must have the same width, and the network must be linear.
  * Numerically, the gradient alignment falls as each condition is removed, most strongly in early layers (Fig. 2).
* **When widths shrink across layers,** adding direct feedback from the top layer as the reference makes local updates closer to back-propagation (Theorem 3.3, Fig. 3).
* **For convolutional networks,** letting B depend on spatial location makes exact equivalence possible in the linear case (Prop. 3.5). This was the largest empirical gain.
* **Results:**
  * CLAPP++ reaches 80.5 / 78.7 / 36.6 / 42.6 % linear-probe accuracy on CIFAR-10 / STL-10 / Tiny-ImageNet / ImageNet. Back-propagation with the same loss reaches 80.5 / 80.4 / 37.6 / 48.5 (Table 2).
  * SoftHebb, a purely competitive Hebbian rule with no feedback, reaches 80.3 / 76.2 / – / 27.3. Naive predictive coding and direct feedback alignment on the same loss do much worse: 36.8–52.3 on STL-10.

**The learning rule** (Eq. 8):
`ΔW_ji = γ · (B c)_j · ρ'(a_j) · z_i`
* γ ∈ {−1, 0, +1} is a neuromodulator.
* (B c)_j is a "dendritic prediction" from a context vector c.
* ρ'(a_j)·z_i is the Hebbian term.
* B itself learns by plain Hebbian updates, `ΔB_jk = γ z_j c_k`.
* Positive pairs are two views of the same input. In the original CLAPP for sequences, the positive pair is a representation and its *future* in the same stream. Negative pairs come from other inputs.

**Limits relevant to us**
* The data is images only, judged by linear-probe classification; there is no next-token or text result.
* Training used Adam (learning rate 5·10⁻⁵), batches of 128–256, SimCLR augmentations and about 100 epochs: large data and many passes (Appendix B).
* The implementation uses stop-gradients, which is mathematically equivalent to the local rule; the explicitly local algorithm was costed but not run end to end (§4, Table 3).
* The theory is linear and orthonormal, and does not extend to the activity normalisation that Forward-Forward needs.

**Meaning for MiniFly**
* A local rule that is label-free, uses one modulator and is Hebbian can learn a *representation* about as well as back-propagation on the same objective, at least for images.
* The objective is to predict a later representation and contrast it with others. For text, that means a context code that predicts the code of the next step. This is the principled form of the rule sketched in V148 design §9b ("contexts followed by the same next byte move towards a shared code").
* **Methodological lesson:** always compare the local rule against back-propagation *on the same loss and architecture*, and measure the cosine between the local update and the back-propagation gradient. That is a cheap, interpretable diagnostic.

## 3. Wang et al. 2026 (COLA): online temporal credit in a near-critical recurrent network

**What it shows**
* **Setting:** a vanilla tanh recurrent network with *learned* recurrent and input weights.
* **Two assumptions** about the exact back-propagation-through-time signal, which hold in a stable near-critical regime: it decays smoothly in time, with a per-unit scalar α (Assumption 1), and it is nearly rank 1 across units (Assumption 2).
* **The resulting teaching signal** is closed-form and per unit:
  `δ̃_t = (∂L_t/∂x_t) ⊘ (1 − μ ⊙ φ'(x_t))`
  with two scalars per unit (α, μ) estimated online, so only O(H) extra state is needed.
* **Updates** keep the outer-product form.
* **Initialisation** places the network near the edge of chaos using a Lyapunov estimate (§3.2).

**Results** (Table 2):
* COLA is strongest on long-horizon regression: Adding 0.0059 against back-propagation through time 0.14; Lorenz rollout 0.0058 against 0.68.
* It is competitive on classification.
* **Character language modelling**, H = 128, 5.1 M training characters × 50 epochs, plain SGD with learning rate 10⁻³ (Table 5). Test cross-entropy is stated without a unit, probably nats:

  | Method | PTB-Char | WikiText-2 Char |
  |---|---:|---:|
  | full back-propagation through time | 2.87 | 3.36 |
  | one-step truncation (TBPTT-1) | 1.83 | 2.28 |
  | SnAp-1 | 1.83 | 2.28 |
  | COLA | 1.84 | 2.42 |

**Points to read carefully**
1. **The full back-propagation-through-time baseline is undertrained.** It is worse than one-step truncation on language (2.87 against 1.83), and still 2.88 after gain retuning (Table 16). So "COLA beats back-propagation" on language reflects a weak baseline. This is the same class of problem as Codex's V152 reference, which never learned.
2. **On character language modelling, COLA ≈ one-step truncation.** Beyond one step, temporal credit added nothing at this scale: the gain came from learning the input and recurrent weights with an instantaneous output error.
3. **"Strictly local" means local in time.** The instantaneous error reaches hidden units through the *transpose of the readout weights* (Eq. 5), which is a one-layer weight transport. Replacing it with random feedback would make it RFLO-like (Murray 2019).
4. **The data budget is large:** 255 M character presentations, against our 40 KB.

**Meaning for MiniFly**
* Temporal credit through learned recurrence is **not the first bottleneck** for next-byte prediction. MiniFly has no learned recurrence, and its history comes from the fixed front-end trace.
* COLA becomes relevant only after MiniFly has a learned recurrent state, and for long-range dependencies.
* Its output update (Eq. 33, ΔW_hy = −η (p − y) hᵀ) is the standard local delta rule with the full categorical error. That is the output-side rule Codex V150/V151 qualified natively.

## 4. The rest of the list, by MiniFly gap

**Learning the representation (the context gap)**
* **CLAPP (P26, Illing et al. 2021).** The sequence-predictive original of §2. Its updates are Hebbian, gated by a scalar modulator and a predictive dendritic term *(from memory)*.
* **SoftHebb (P28).**
  * Soft winner-take-all Hebbian learning with no feedback at all.
  * Strong on small images, but drops on ImageNet (27.3 against CLAPP++ 42.6; Table 2 of P27).
  * It learns input statistics, not prediction.
* **Forward-Forward and SCFF (P29, P30).** Local contrastive "goodness" learning that needs negative samples. Hinton's paper includes a small next-character experiment *(from memory)*.
* **FlyVec** (Liang, Ryali, Hoover, Grinberg, Navlakha, Zaki & Krotov, ICLR 2021; *not in Codex's list*; abstract level).
  * A mushroom-body model, with a Kenyon-cell layer inhibited by one APL-like neuron, learns word–context correlations from raw text.
  * It uses a local competitive rule in the Krotov–Hopfield family and produces sparse binary codes.
  * It reports performance comparable to standard word embeddings on word similarity and classification, with less training time and memory.
  * **This is the closest published precedent for MiniFly**: learned plasticity on the fly's input-to-KC connections, from text, without back-propagation. It learns *word* codes from co-occurrence; it does not do next-byte prediction.
* **Local-loss, target-propagation and feedback-alignment family (P01–P04, P06–P13).**
  * These send errors into multi-layer networks.
  * MiniFly first needs *one* learned layer, so the single-layer local rules suffice.
  * Bartunov et al. (P05) is the standing caution that these do not scale uniformly *(from memory)*.

**Output and teaching**
* A local delta rule on a learned readout with the full categorical error is uncontroversial: COLA Eq. 33, and Codex V150/V151.
* Fast-weight programmers (P41, Schlag et al.) show that a delta-rule associative memory works for language **when its keys and queries are learned** *(from memory; outer training uses back-propagation)*.
* MiniFly's native store is such a delta-rule memory, but with *fixed random keys* (the KC codes). V149 measured that those keys carry less than bigram-level context.

**Temporal credit**
* RTRL, e-prop, RFLO, UORO, KF-RTRL, SnAp and COLA (P15–P21) are reference methods for learned recurrence.
* e-prop was tried in CLS v0 and failed there.
* Per §3, they are a later step.

**Retention** (V148: early text erased; repetition hurts)
* Complementary learning systems (P36), cascade synapses (P35), synaptic intelligence (P34), two-factor consolidation (P33) and context-gated plasticity (P32).
* These matter once something worth keeping is learned. Not first.

**Evaluation**
* SCAN and COGS (P39, P40) test composition. Later.
* TEM (P38) separates structure from content but is trained with back-propagation *(from memory)*. It is mechanistic guidance, not a recipe.
* **FLM** (connectome-derived recurrent language model, trained with back-propagation through time; Codex V152 review) reports that removing temporal state hurts far more than removing lateral wiring. That is consistent with V149's finding that the trace's timescale matters.

## 5. What the literature says, taken together
1. **The anomaly is MiniFly's fixed random keys.** Every working memory-based sequence model learns its keys and representation: fast-weight programmers, TEM, FlyVec's learned KC layer. MiniFly's input-to-KC expansion is fixed at birth, and V149 measured that it carries little context.
2. **Learning those keys locally and without labels is possible.**
   * A predictive, contrastive local rule (CLAPP++) matches back-propagation on the same objective for images.
   * A competitive Hebbian rule (SoftHebb, or FlyVec on text) is simpler but does not scale as well (ImageNet 27.3 against 42.6).
   * A prediction objective is the one aligned with next-byte prediction, and the one that can make contexts with similar futures share codes. That sharing is the only route to beating counting.
3. **Output:** use a local delta rule with the full categorical error; this is standard.
4. **Temporal credit can wait.** One-step credit matched COLA on character language modelling at this scale.
5. **Methodology:**
   * compare each local rule with back-propagation on the same loss and architecture, with a gradient-cosine diagnostic (Zihan);
   * qualify every reference first (COLA's language baseline; Codex V152);
   * test the method's own assumptions (COLA §4.6).
6. **Tension with "small data".** Both papers train on far more data than 40 KB: 5.1 M characters × 50 epochs, and about 100 epochs of image datasets. A local representation learner may need many passes over a small text, or more text. Every pass must be counted.

## 6. Next-step options (for discussion)

| | Option | Basis in the literature | Change | Fit to "carries over to language tasks" |
|---|---|---|---|---|
| A | **Learn the input-to-KC expansion with a local predictive rule** (single-layer CLAPP: a KC code at step t predicts the code at t+1 against a code from another position; `ΔW = γ·(B c)·ρ'(a)·x`, B Hebbian) | P27 (theory and images), P26 (sequences) | one component: input-side plasticity (fly catalogue §2, P family) | yes: a CPC/word2vec-like objective at fixed width |
| B | **Learn the same expansion with a competitive Hebbian rule** (SoftHebb or FlyVec style) | P28, FlyVec | the same component, no prediction signal | yes, but objective not aligned with prediction |
| C | Change the front-end timescale (decay) only | FLM ablation, V149 FE0 | a parameter; nothing learned | partial |
| D | Learned recurrence with online temporal credit (COLA, e-prop) | P21, P16, P17 | a new recurrent layer | later; one-step credit sufficed at this scale |

**Recommendation (separate from the options).** A, with B as its comparison arm and the frozen random expansion as the baseline.
* **Measurement:** the V149 probe unchanged (the best-possible readout of the learned state), on the same splits, so the result is directly comparable with V149.
* **Reference:** back-propagation on the same loss, evaluator-only, qualified first.
* **Diagnostic:** the gradient cosine between the local and back-propagation updates.
* **Question:** does locally learned, label-free predictive structure raise the context ceiling above bigram and towards PPM-2, using 40 KB with a counted number of passes?
* **Out of scope for this step:** native integration, output changes and retention.

**Open decision for the owner:** the data budget. Keep 40 KB with several counted passes, or allow more text, given that the literature's local learners use far more.
