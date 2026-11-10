# Literature check: local credit through time and layer-local targets (10 November 2026)

**Commissioned by:** brief `OPUS_TASK_LITERATURE_CHECK_LOCAL_CREDIT_20261110.md`.
**Written by:** a cloud session, read-only. It ran no experiment and edited no existing file.

**How this memo was made.**
- **Reading.** Four research sub-agents of this session read the papers, one per question (Q1–Q4). Every number and section reference below comes from a page one of them opened in this session.
- **Spot-checks.** I opened eight of the cited pages myself and checked the central numbers against the text. All matched:
  - Xiao et al. 2018, Table 1;
  - Launay et al. 2020, Table 5;
  - Handler et al. 2019, Figures 2D–E and 6F–G;
  - Moon et al. 2026, Table 1 and Sections 4.2 and 4.4;
  - Zucchet et al. 2023, Table 2 and Section 3.2;
  - Liu et al. 2026 (K-prop), Sections 3.1–3.2, Appendix A and Table 1;
  - the Zihan et al. 2026 and TinyStories abstracts.
- **Scope.** This memo decides nothing.

## Summary: the four verdicts

| # | Claim | Verdict |
|---|---|---|
| 1 | Exact online gradient within a diagonal-recurrent layer costs about one trace per parameter; multi-layer versions approximate credit through an upper layer's memory | **First half confirmed; second half corrected.** The most-cited multi-layer methods (Zucchet 2023, OSTL) do approximate it, by dropping the future-error path through upper-layer recurrence. K-prop (Liu et al., NeurIPS 2026) carries that credit forward with one extra trace per synapse per distinct downstream time constant. It is exact for two leaky layers and approximate beyond. |
| 2 | Layer- or block-local objectives match backprop on small problems, fall behind with scale, and are not shown at language-model scale | **Corrected on both halves.** With few blocks they match backprop on ImageNet; the gap grows with the number of local pieces, not simply with dataset size. Block-local training of 125M and 400M language models now exists (Moon et al., NeurIPS 2026 workshop): it never matches end-to-end, and the gap doubles from 2 to 4 blocks. |
| 3 | Sign-symmetric feedback stays close to backprop at scale; FA and DFA fall behind as depth grows | **Corrected.** Sign symmetry with a per-layer scale or normalisation comes within about 4–11 top-1 points of backprop on ImageNet ResNets, and once matched it on AlexNet. Raw unscaled sign feedback diverged in the earliest study. FA and DFA collapse on ImageNet (82–98% top-1 error), but no source shows a controlled monotone depth effect. **No result was found for sign symmetry through time or in recurrent networks.** |
| 4 | In the fly mushroom body, dopamine timing relative to odor sets the sign of plasticity at the output synapses on a scale of seconds, and dopamine neurons carry non-reward signals | **Confirmed, narrower in scope.** The synaptic sign switch lies within about ±1 s and is shown in particular compartments (γ1pedc, γ2, γ4, γ5). The behavioural window is tens of seconds. Dopamine neurons also carry movement, internal state, novelty/familiarity, and ongoing activity that drives forgetting. |

## Q1. Online credit in stacked element-wise recurrent layers

**Within one layer the gradient is exact and cheap.**
- **Zucchet et al. 2023 (Sec. 3.1–3.2).** For linear recurrent units with diagonal (complex) recurrence, the sensitivities of the recurrence, the input projection B and the output are local to a neuron or synapse. Memory is bounded by neurons plus parameters, and the method costs about twice an inference pass (Abstract, Sec. 1).
- **Irie et al. 2024 (Sec. 2–3.1).** For element-wise LSTMs, exact RTRL costs O(DN) memory and O(N²) time per layer, against O(N³)/O(N⁴) for full RTRL.
- **Elelimy et al. 2024 (App. E.4).** RTRL for their trace units is "linear in the number of parameters".
- **e-prop (Bellec et al. 2020, Methods).** The eligibility traces are exact: one per synapse for leaky neurons, two components for adaptive ones. What is approximate is the learning signal: the partial ∂E/∂z replaces the total derivative.

**Across layers, methods differ.**
- **Zucchet et al. 2023.** Lower layers receive an error computed by spatial backprop at the current step only, ignoring future error routed through upper-layer recurrence. The paper calls this "the only source of approximation" (Sec. 3.2), and it worsens with depth and as |λ|→1 (Sec. 4.1).
  - Long Range Arena, 4 layers, N=128, H=256, about 1M parameters, 3 seeds (Table 2), online against BPTT: sCIFAR 79.59 vs 83.40, IMDB 86.48 vs 87.69, ListOps 37.62 vs 39.75. Truncated or spatial-only training gets 58–60 on sCIFAR.
  - Copy task, small (Table 1): final loss 8.4e-3 online vs 7.6e-6 BPTT.
  - **Upstream parameters.** B inside the unit gets an exact trace. Layer norm and the GLU outside it get spatial credit only, with no temporal credit.
- **OSTL (Bohnstingl et al., arXiv 2020).**
  - Gradient-equivalent to BPTT for one layer. That still holds with stateless layers added (Supp. S4).
  - The multi-layer version drops a residual term (Eq. 11) and is marked "not exact" (Table 1).
  - Results are reported only in figures, as "on par" or "competitive".
- **Irie et al. 2024 (Sec. 5).** Stacking two element-wise layers reintroduces a full sensitivity matrix and cubic or quartic cost, and they propose no remedy. Their vision stem is pre-trained by truncated backprop and frozen, because RTRL for it is intractable (Sec. 4–5).
- **K-prop (Liu, Haider, Benitez, Petrovici, Senn, NeurIPS 2026).** The closest published case to this project: feedforward stacks of leaky neurons with fixed self-leak and no recurrence matrix.
  - **How it works.** It propagates "kernel gains" backward across layers and keeps "deep eligibility traces" per synapse (Sec. 3.1, Eqs. 4–6).
  - **Cost.** One trace per synapse per distinct downstream time constant (Sec. 3.1); in practice 2 × layers (Sec. 5.1).
  - **Approximation.** The only one is that the nonlinearity's derivative varies slowly. It is **exact for two leaky layers** (App. A, Eqs. 18–23). The approximation enters at three layers, and the mismatch grows toward earlier layers (Sec. 3.2).
  - **Results (Table 1), small benchmarks:**

    | Benchmark | Parameters | BPTT | OSTL | K-prop |
    |---|---:|---:|---:|---:|
    | MNIST1D-Lk4 | 14K | 96.60 | 96.08 | 96.53 |
    | MNIST1D-Lk5 | 21K | 97.69 | 97.03 | 97.56 |
    | Google Speech Commands | 0.55M | 92.47 | 90.61 | 91.52 |

  - Identical time constants across layers must be perturbed (App. A).
- **SnAp (Menick et al. 2020).** Keeps the influence entries that are nonzero within n steps. SnAp-1 is "no more expensive than backpropagation". It treats a single recurrent core and does not discuss stacking.
- **Millidge 2025 (arXiv).** Gives a depth recursion for e-prop traces, with no experiments.

**Answer.**
- **Exact:** the per-layer traces of diagonal units, including their input projection.
- **Approximated in most methods:** the error a lower layer receives through an upper layer's memory.
- **Exact through one upper leaky layer at extra trace cost:** K-prop.
- **Parameters before the write** get exact traces when they are the unit's own input weights. They get spatial-only credit when they sit in a separate block (Zucchet 2023), and truncated or frozen training in Irie 2024.
- **Scale.** The largest exact-in-layer comparison found is about 1M parameters (LRA).

## Q2. A target at every layer

**Supervised or contrastive block-local objectives, mostly vision.**
- **Few blocks on ImageNet match end-to-end.**
  - DGL with K=2: ResNet-152 74.5 vs 74.4 top-1 (Belilovsky et al. 2020, Table 3).
  - InfoPro with K=2: ResNet-101 21.85% vs 22.03% error (Wang et al. 2021, Table 4).
  - Layer-wise VGG-11: 67.6 vs 67.9 (Belilovsky et al. 2019, Table 3).
- **Many pieces fall behind.** Greedy ResNet-32 on CIFAR-10 has 10.30% error at K=2 and 24.59% at K=16, against 7.37% end-to-end (Wang 2021, Table 1).
- **Fixes recover most of the gap.**
  - InfoPro adds a reconstruction term against "information collapse" in early modules (Sec. 2–3).
  - AugLocal builds each auxiliary head from the layers that follow: ResNet-101 / ImageNet with 34 local layers reaches 76.70 vs 77.34 (Ma et al. 2024, Table 3).
  - LoCo overlaps adjacent stages for a "look-ahead", because lower layers "are unaware" of what is needed above (Xiong et al. 2020, Sec. 4.1). ResNet-50 linear probe: 69.5 vs SimCLR 69.8, and GIM 64.7 (Table 1).
- **Predictive coding** is comparable to backprop up to about 7 layers and falls far behind on ResNet-18 (Tiny ImageNet about 16% vs 58% top-1). The cause is exponentially small gradients from layer energy imbalance (Pinchetti et al. 2025, Table 1, Sec. 5).

**Sequences and language.**
- **Greedy InfoMax (Löwe et al. 2019), LibriSpeech, small.**
  - Phone accuracy: 62.5 for GIM vs 64.9 for end-to-end CPC.
  - Cutting gradient flow through time drops it to 55.5; removing the recurrent module, to 50.8 (Table 4).
  - The authors conclude GIM works best where temporal context need not be modelled (Sec. 4.2).
- **Laskin et al. 2020 (arXiv).** A 6M-parameter Transformer on LM1B reaches best validation loss 3.623 with backprop, 3.662 with overlapping local updates and 3.757 greedy (Sec. 4.2).
- **Moon et al. (arXiv, October 2026; NeurIPS 2026 CODEC-FM workshop).** Llama-style models at 125M and 400M on FineWeb-Edu, K ∈ {1, 2, 4} blocks, each with its own next-token head.
  - **Every gap to end-to-end is positive** (Table 1): +0.017 to +0.132 nats per token, about doubling from K=2 to K=4.
  - **"Representation myopia" (Sec. 4.2).** Per-block losses collapse onto one curve; blocks above the first "barely improve on the representation they inherit".
  - **Predicting x_{t+2} as well as x_{t+1}** helps at the first block boundary and hurts at deeper ones (Sec. 4.4). A next-token + CPC hybrid fails through shortcuts (App. H).
  - Single seed, token-matched rather than compute-matched (Sec. 5).
- **Forward-Forward (Hinton 2022).** A next-character result was withdrawn as probably a bug (Sec. 5). Hinton calls the inability of later layers to affect earlier ones "a major weakness" (Sec. 3.4).
- **Zihan et al. 2026 (ICML).** Local self-supervised rules (CLAPP) match backprop SSL on CIFAR-10, STL-10 and Tiny ImageNet. On images only.

**Known failure modes and remedies.**

| Failure mode | Sources | Remedies tried |
|---|---|---|
| Myopia: layers learn what is locally predictable | LoCo, AugLocal, Laskin, Moon | Overlap/look-ahead (LoCo); auxiliary built from upper layers (AugLocal); stronger auxiliaries (Moon) |
| Information collapse in early modules | InfoPro | Reconstruction term (InfoPro) |
| Shortcut learning in contrastive targets | Moon App. H | — |
| Loss of temporal credit when gradients are cut in time | GIM | — |

## Q3. Feedback without weight transport

**Scale, feedforward (ImageNet top-1 error, %).**

| Source | Network | BP | Sign symmetry | FA | Other |
|---|---|---:|---:|---:|---|
| Xiao et al. 2018, Table 1 (50 epochs) | ResNet-18 | 33.14 | 37.91 | 90.52 | |
| Xiao et al. 2018, Table 1 | AlexNet | 49.15 | 47.57 | 93.45 | |
| Akrout et al. 2019, Results | ResNet-18 | 30.1 | 39.2 | 97.4 | weight mirrors 30.2, Kolen-Pollack 29.2 |
| Akrout et al. 2019, Results | ResNet-50 | 22.9 | 33.8 | 98.9 | |
| Sanfiz & Akrout 2021 | ResNet-18 | 30.39 | uSF 34.97 | 85.25 | DFA 82.45 |
| Bartunov et al. 2018, Table 2 | locally connected net | 71.43 | | 93.08 | |

- **Xiao et al.'s sign feedback** is sign(W) times a fixed per-layer scale (Sec. 2–3.1).
- **Moskovitz et al. 2018** need explicit normalisation for sign-concordant feedback (Sec. 3.1, Table 1).
- **Unscaled sign feedback** often diverged to chance on CIFAR in Liao et al. 2016 (Table 2). It worked only with batch normalisation or "Batch Manhattan".
- **Disagreement between sources.** Kunin et al. 2020 could not reproduce Akrout's weight-mirror result (63.5% vs a reported 69.73% top-1 accuracy, Table 3).
- **Depth.** FA's CIFAR error does not grow steadily from LeNet to ResNet-56 (Sanfiz & Akrout 2021, Table 2). Moskovitz argues the scale mismatch compounds with depth (Sec. 5.1) but does not run a depth sweep.

**DFA in a transformer.** Launay et al. 2020, 6 layers, d=512, WikiText-103, best validation perplexity (Table 5):
- BP: 34.4, or 29.8 with β2=0.999.
- DFA with feedback per layer: 52.0 at best. This still carries weight transport inside the layer.
- DFA with feedback per sub-layer: 93.3 at best.
- Shallow baseline: over 400.
- The authors' own wording: "not on par with BP".

**Recurrent networks and time.**
- **Murray 2019 (RFLO).** Fixed random feedback plus traces in 30–100-unit networks. Comparable to BPTT on short tasks and worse on long durations; the gap is attributed mainly to truncation, not to the random feedback (Figs. 2–3). These figures were read from the summary text.
- **e-prop (arXiv v2, 2019).**
  - TIMIT: random-feedback e-prop 0.629, symmetric e-prop 0.651, BPTT 0.671.
  - Penn Treebank, 200-unit LSTM, with true readout feedback: perplexity 115 vs truncated BPTT's 113.
- **DFA in RNNs on WikiText-2 (Han et al. 2020, Table 3).** Test accuracy 18.2–19.2% for DFA vs 23.0–24.5% for BP.
- **No paper was found that applies sign-symmetric feedback through time or in a recurrent network.**

## Q4. Where the fly's olfactory pathway is plastic (adult)

- **Receptor neuron → projection neuron, and local interneurons.**
  - **Habituation is established.**
    - Short-term habituation: 30 minutes of exposure, recovering with a half-life of about 20 minutes.
    - Long-term habituation: days of exposure, lasting days.
    - Both need *rutabaga* in GABAergic local interneurons, and adult rescue argues against a purely developmental role (Das et al. 2011, Figs. 3–7).
    - Long-term habituation was induced in flies 0–12 h old.
  - **Receptor-neuron terminal remodelling is largely confined to an early-adult critical period.** About 80% loss of innervation at 0–2 days after eclosion, little or none at 7–9 days. Receptor expression and neuron number do not change (Golovin et al. 2019, Figs. 1–3).
  - **Status:** plasticity in mature adults at this stage is contested or unknown.
- **Projection neuron → Kenyon cell (calyx).** After 24-hour appetitive long-term memory, the trained odor's projection neurons form more, smaller microglomeruli. This depends on *rut* and protein synthesis, with no change after 1-minute memory (Baltruschat et al. 2021, Figs. 2–4). Timescale about a day; a small evidence base, essentially one study opened.
- **Kenyon cell → output neuron.** Established and best characterised.
  - Odor-specific depression within seconds, lasting over 40 min; Kenyon-cell responses themselves unchanged (Hige et al. 2015, Figs. 1–2).
  - Plasticity in both directions set by timing (Handler et al. 2019, Fig. 2; receptors, Figs. 5–7).
  - Compartment-specific rules for learning speed, retention and capacity: some compartments write after one pairing and decay by 24 h, others hold 4-day memories (Aso & Rubin 2016, Figs. 2–3).
  - Familiarity depression recovering in 20–60 min (Hattori et al. 2017).
- **Feedback loops.**
  - Extinction and reconsolidation need output-neuron → dopamine-neuron loops (Felsenberg et al. 2017, Figs. 2–4; 2018, Figs. 1–6).
  - APL's response to the trained odor falls 30 s to 5 min after pairing (Liu & Davis 2009, Fig. 5).
  - DPM plasticity was read only in a secondary summary.
- **What dopamine neurons carry besides valence:**
  - movement and context-dependent locomotion (Cohn et al. 2015, Fig. 2; Zolin et al. 2021, Figs. 1–4);
  - internal state after feeding (Cohn 2015, Fig. 5);
  - novelty (Hattori 2017, Fig. 6);
  - ongoing post-learning activity that causes forgetting (Berry et al. 2012, Figs. 1, 5, 6). Contested by another group, as the authors acknowledge.
  - Omission of expected punishment acts like reward (Felsenberg 2018, Discussion), but **prediction error is not directly measured** in the papers read.
- **On "no frozen first stage".**
  - Adult plasticity is reported at every stage downstream of the receptor neurons: interneurons, calyx, output synapses, loops.
  - The receptor-neuron layer itself looks comparatively stable after an early-adult window.
  - Both statements are supported, and this memo takes no side.

## Q5. What follows for this project (suggestions; they decide nothing)

**Nearest published methods.**

| | E17 Phase 4 arm: weight-sign feedback, credit through time at every layer | E18 local heads: a next-byte target at each bank, no relay |
|---|---|---|
| **Q1** | Zucchet 2023 (exact within a layer, spatial-only between layers) and K-prop (credit through upper leaky layers). The Phase 4 relay is closest to K-prop with sign feedback in place of exact weights. | Zucchet's spatial-only case, at the extreme where nothing passes between layers |
| **Q2** | — | Moon et al. 2026: per-block next-token heads in a language model. The closest published analogue, and it reports myopia and a gap that grows with block count. |
| **Q3** | Sign symmetry (Xiao, Akrout), always with a per-layer scale or normalisation. **No published test through time.** | — |

**Limits the literature reports.**
- Credit through upper-layer memory is the dominant approximation and worsens with depth and with long retention (Zucchet Sec. 4.1, K-prop Sec. 3.2).
- Unscaled sign feedback can diverge (Liao Table 2).
- Local next-token heads make blocks above the first redundant (Moon Sec. 4.2).

**At most three design changes for a two-bank learner with fixed retention rates.**
1. **Carry credit through the upper bank's memory with K-prop-style traces.**
   - Keep, per lower-bank write synapse, one extra trace for each distinct upper-bank retention rate.
   - K-prop is exact for two leaky layers (App. A).
   - The project's retention rates are fixed, so the trace count is fixed and known before birth.
   - *Extrapolation from K-prop, not tested.*
2. **If weight-sign feedback stays, scale it per layer.** Use a fixed per-layer scale as in Xiao et al., or normalise, rather than relaying raw sign(W).
3. **If local heads stay, give the lower bank's head a view of what the upper bank needs.**
   - An auxiliary head built from a copy of the upper bank (AugLocal), or overlapping stages (LoCo).
   - Predict two bytes ahead only at the lower bank's boundary (Moon Sec. 4.4).
   - A reconstruction term against collapse (InfoPro).

## References (all opened in this session)

### Q1: online credit in recurrent layers

- Zucchet, Meier, Schug, Mujika, Sacramento (2023). *Online learning of long-range dependencies.* NeurIPS 2023 (venue from search). https://ar5iv.labs.arxiv.org/html/2305.15947
- Irie, Gopalakrishnan, Schmidhuber (2024). *Exploring the Promise and Limits of Real-Time Recurrent Learning.* ICLR 2024. https://arxiv.org/html/2305.19044v3
- Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein, Maass (2020). *A solution to the learning dilemma for recurrent networks of spiking neurons.* Nature Communications 11:3625. https://pmc.ncbi.nlm.nih.gov/articles/PMC7367848
- Bellec, Scherr, Hajek, Salaj, Legenstein, Maass (2019). *Biologically inspired alternatives to backpropagation through time for learning in recurrent neural nets.* arXiv preprint. https://arxiv.org/pdf/1901.09049
- Bohnstingl, Woźniak, Maass, Pantazi, Eleftheriou (2020). *Online Spatio-Temporal Learning in Deep Neural Networks.* arXiv preprint. https://arxiv.org/html/2007.12723v2
- Menick, Elsen, Evci, Osindero, Simonyan, Graves (2020). *A Practical Sparse Approximation for Real Time Recurrent Learning.* arXiv preprint. https://arxiv.org/abs/2006.07232
- Elelimy, White, Bowling, White (2024). *Real-Time Recurrent Learning using Trace Units in Reinforcement Learning.* arXiv preprint. https://arxiv.org/html/2409.01449v2
- Liu, Haider, Benitez, Petrovici, Senn (2026). *K-prop: Deep Online Learning by Backpropagating Temporal Kernels.* NeurIPS 2026. https://arxiv.org/abs/2611.12530 and https://arxiv.org/html/2611.12530v1
- Millidge (2025). *Generalising E-prop to Deep Networks.* arXiv preprint. https://arxiv.org/abs/2512.24506

### Q2: a target at every layer

- Löwe, O'Connor, Veeling (2019). *Putting An End to End-to-End: Gradient-Isolated Learning of Representations.* NeurIPS 2019. https://arxiv.org/abs/1905.11786
- Nøkland, Eidnes (2019). *Training Neural Networks with Local Error Signals.* ICML 2019. https://arxiv.org/abs/1901.06656
- Belilovsky, Eickenberg, Oyallon (2019). *Greedy Layerwise Learning Can Scale to ImageNet.* ICML 2019. https://arxiv.org/pdf/1812.11446
- Belilovsky, Eickenberg, Oyallon (2020). *Decoupled Greedy Learning of CNNs.* ICML 2020. https://arxiv.org/pdf/1901.08164
- Wang, Ni, Song, Yang, Huang (2021). *Revisiting Locally Supervised Learning: an Alternative to End-to-end Training.* ICLR 2021. https://arxiv.org/abs/2101.10832
- Xiong, Ren, Urtasun (2020). *LoCo: Local Contrastive Representation Learning.* NeurIPS 2020. https://arxiv.org/pdf/2008.01342
- Ma, Wu, Si, Tan (2024). *Scaling Supervised Local Learning with Augmented Auxiliary Networks.* ICLR 2024. https://arxiv.org/pdf/2402.17318
- Laskin, Metz, Nabarro et al. (2020). *Parallel Training of Deep Networks with Local Updates.* arXiv preprint. https://arxiv.org/abs/2012.03837
- Moon, Zaheer, Lee, Kim, Park (2026). *Greedy Local Learning for Language Model Pretraining: Gaps and Objective Design.* NeurIPS 2026 CODEC-FM workshop. https://arxiv.org/abs/2610.04867 and https://arxiv.org/html/2610.04867
- Zihan, Delrocq, Gerstner, Bellec (2026). *Can Local Learning Match Self-Supervised Backpropagation?* ICML 2026. https://arxiv.org/abs/2601.21683
- Hinton (2022). *The Forward-Forward Algorithm: Some Preliminary Investigations.* arXiv preprint. https://arxiv.org/abs/2212.13345
- Pinchetti et al. (2025). *Benchmarking Predictive Coding Networks — Made Simple.* ICLR 2025 (per the PDF header). https://arxiv.org/abs/2407.01163
- Millidge, Tang, Osanlouy, Harper, Bogacz (2024). *Predictive coding networks for temporal prediction.* PLOS Computational Biology 20(4):e1011183. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011183
- Pyeon, Moon, Hahn, Kim (2021). *SEDONA.* ICLR 2021. Abstract only: https://mlanthology.org/iclr/2021/pyeon2021iclr-sedona

### Q3: feedback without weight transport

- Xiao, Chen, Liao, Poggio (2018). *Biologically-plausible learning algorithms can scale to large datasets.* arXiv preprint. https://arxiv.org/abs/1811.03567
- Bartunov, Santoro, Richards, Marris, Hinton, Lillicrap (2018). *Assessing the Scalability of Biologically-Motivated Deep Learning Algorithms and Architectures.* NeurIPS 2018. https://arxiv.org/abs/1807.04587
- Moskovitz, Litwin-Kumar, Abbott (2018). *Feedback alignment in deep convolutional networks.* arXiv preprint. https://arxiv.org/abs/1812.06488
- Launay, Poli, Boniface, Krzakala (2020). *Direct Feedback Alignment Scales to Modern Deep Learning Tasks and Architectures.* NeurIPS 2020. https://arxiv.org/abs/2006.12878
- Liao, Leibo, Poggio (2016). *How Important Is Weight Symmetry in Backpropagation?* AAAI 2016. https://arxiv.org/pdf/1510.05067
- Nøkland (2016). *Direct Feedback Alignment Provides Learning in Deep Neural Networks.* NIPS 2016. https://arxiv.org/pdf/1609.01596
- Akrout, Wilson, Humphreys, Lillicrap, Tweed (2019). *Deep Learning without Weight Transport.* NeurIPS 2019. https://arxiv.org/pdf/1904.05391
- Kunin, Nayebi, Sagastuy-Brena, Ganguli, Bloom, Yamins (2020). *Two Routes to Scalable Credit Assignment without Weight Symmetry.* ICML 2020. https://arxiv.org/pdf/2003.01513
- Sanfiz, Akrout (2021). *Benchmarking the Accuracy and Robustness of Feedback Alignment Algorithms.* arXiv preprint. https://arxiv.org/abs/2108.13446
- Lillicrap, Cownden, Tweed, Akerman (2014). *Random feedback weights support learning in deep neural networks.* arXiv preprint (the 2016 Nature Communications version was behind a login). https://arxiv.org/pdf/1411.0247
- Murray (2019). *Local online learning in recurrent networks with random feedback.* eLife 8:e43299. https://pmc.ncbi.nlm.nih.gov/articles/PMC6561704/
- Han, Park, Ryu, Yoo (2020). *Extension of Direct Feedback Alignment to Convolutional and Recurrent Neural Network for Bio-plausible Deep Learning.* arXiv preprint. https://arxiv.org/abs/2006.12830

### Q4: the fly's olfactory pathway

- Handler et al. (2019). *Distinct dopamine receptor pathways underlie the temporal sensitivity of associative learning.* Cell. https://pmc.ncbi.nlm.nih.gov/articles/PMC9012144
- Hige, Aso, Modi, Rubin, Turner (2015). *Heterosynaptic plasticity underlies aversive olfactory learning in Drosophila.* Neuron. https://pmc.ncbi.nlm.nih.gov/articles/PMC4674068
- Cohn, Morantte, Ruta (2015). *Coordinated and compartmentalized neuromodulation shapes sensory processing in Drosophila.* Cell. https://pmc.ncbi.nlm.nih.gov/articles/PMC4732734
- Aso, Rubin (2016). *Dopaminergic neurons write and update memories with cell-type-specific rules.* eLife. https://pmc.ncbi.nlm.nih.gov/articles/PMC4987137
- Hattori et al. (2017). *Representations of novelty and familiarity in a mushroom body compartment.* Cell. https://pmc.ncbi.nlm.nih.gov/articles/PMC5806120
- Berry, Cervantes-Sandoval, Nicholas, Davis (2012). *Dopamine is required for learning and forgetting in Drosophila.* Neuron. https://pmc.ncbi.nlm.nih.gov/articles/PMC4083655
- Felsenberg, Barnstedt, Cognigni, Lin, Waddell (2017). *Re-evaluation of learned information in Drosophila.* Nature. https://pmc.ncbi.nlm.nih.gov/articles/PMC5392358
- Felsenberg et al. (2018). *Integration of parallel opposing memories underlies memory extinction.* Cell. https://pmc.ncbi.nlm.nih.gov/articles/PMC6198041
- Baltruschat et al. (2021). *Circuit reorganization in the Drosophila mushroom body calyx accompanies memory consolidation.* Cell Reports. https://pmc.ncbi.nlm.nih.gov/articles/PMC8515896
- Das, Sadanandappa et al. (2011). *Plasticity of local GABAergic interneurons drives olfactory habituation.* PNAS. https://pmc.ncbi.nlm.nih.gov/articles/PMC3169145
- Golovin, Vest, Vita, Broadie (2019). *Activity-dependent remodeling of Drosophila olfactory sensory neuron brain innervation during an early-life critical period.* J Neurosci. https://pmc.ncbi.nlm.nih.gov/articles/PMC6468095
- Zolin, Cohn, Pang, Siliciano, Fairhall, Ruta (2021). *Context-dependent representations of movement in Drosophila dopaminergic reinforcement pathways.* Nat Neurosci. https://pmc.ncbi.nlm.nih.gov/articles/PMC8556349
- Liu, Davis (2009). *The GABAergic anterior paired lateral neuron suppresses and is suppressed by olfactory learning.* Nat Neurosci. https://pmc.ncbi.nlm.nih.gov/articles/PMC2680707
- Yu, Akalal, Davis (2006). *Drosophila α/β mushroom body neurons form a branch-specific, long-term cellular memory trace after spaced olfactory conditioning.* Neuron. https://pmc.ncbi.nlm.nih.gov/articles/PMC1779901
- Sachse et al. (2007). *Activity-dependent plasticity in an olfactory circuit.* Neuron. Section headings only: https://www.rockefeller.edu/research/uploads/www.rockefeller.edu/sites/8/2018/09/SachseVosshallNeuron07.pdf

### Roadmap-note pointers

- Eldan, Li (2023). *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* arXiv preprint. Abstract read. https://arxiv.org/abs/2305.07759
- Warstadt, Mueller, Choshen, Wilcox et al. (2023; arXiv 2025). *Findings of the BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora.* Proceedings of BabyLM (CoNLL 2023). Abstract read. https://arxiv.org/abs/2504.08165

## From memory, not verified

- **Latent Predictive Learning (Halvagal & Zenke, Nature Neuroscience 2023)** is believed to use a layer-local temporal-prediction rule that needs a Hebbian variance term to avoid collapse. The pages were blocked, so it is not verified.
- **Kremer et al. 2010** is believed to report adult structural change at projection-neuron → Kenyon-cell microglomeruli. Only search snippets were seen.
- **Yu et al. 2005 (DPM memory trace)** is known only through the 2006 paper's summary.
- **The roadmap note's statement that small language models "train on a laptop"** is not supported by either the TinyStories or the BabyLM abstract, which give no hardware.

## Searched for and not found

- Any application of sign-symmetric feedback through time or in a recurrent network.
- Any FA or sign-symmetry LSTM language model with perplexity.
- Any local or greedy objective on a byte- or character-level language model.
- Any local language-model result above 400M parameters.
- Any Forward-Forward language-model result: the one reported was withdrawn.
- Any exact online multi-layer training of diagonal state-space models at about one trace per parameter.
- Full text not reached:
  - the final Nature Communications versions of Lillicrap 2016 and Bellec 2020 (arXiv versions used);
  - SEDONA's full text;
  - Kremer et al. 2010;
  - the TNNLS version of OSTL;
  - UORO, covered only through SnAp.
