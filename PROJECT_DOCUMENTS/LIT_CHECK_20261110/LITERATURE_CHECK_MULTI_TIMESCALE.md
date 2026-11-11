# Literature check: learning and memory at several timescales (11 November 2026)

**Commissioned by:** brief `OPUS_TASK_LITERATURE_CHECK_MULTI_TIMESCALE_20261110.md` (Mr. Zhou's question, relayed by Mr. Xu, 10 November).
**Written by:** a cloud worker session, read-only. It ran no experiment, trained nothing and edited no existing file. It decides nothing.
**How it was made.** I opened every cited page myself in this session (no sub-agents). Where WebFetch could not read a PDF, I extracted its text locally and read the passage. Central numbers were checked against the raw text: Nested Learning Tables 2 and 6 and the Section 1 and 3.2 brain-wave passages (raw HTML); Saunshi et al. Table 1 (raw HTML); S4D Table 3a, Kaplanis et al. Section 3 and Fig. 4, Laborieux et al. Table 1, CLS-ER Table 1, Synaptic Intelligence Eqs. 3–5, and Plaçais et al. Figs. 3, 6 and 7 (extracted PDF text). `PERSISTENT_CORE_REQUIREMENTS.md`, named in the brief's reading list, is not in this checkout or its git history; the bounded-cost remarks below therefore rest on the review's own rulings (9.82 addendum) and not on that file.

## Summary for Mr. Zhou

**The short answer.** The "frequencies" in the Hope paper are not brain rhythms. They are how often each block of weights is updated. Hope is therefore an example of reading C (weights at several rates), with the brain-wave hierarchy used as inspiration. The brain rhythms the paper cites run at about 0.5–150 Hz (Nested Learning, Sec. 1), and single neurons' intrinsic timescales in monkey cortex are about 50–350 ms (Murray et al. 2014). Neither one by itself holds something for 120 s. Where old information meets new over seconds to minutes, the documented carriers are slower: replay events, ongoing neuromodulator activity, and synaptic processes at several rates. The fly has one direct case. Slow (about 0.05–0.3 Hz) calcium oscillations in two pairs of mushroom-body dopamine neurons are strengthened after spaced training, and silencing these neurons during the 15-minute gaps between trials prevents long-term memory (Plaçais et al. 2012).

**The five verdicts**

| # | Claim | Verdict |
|---|---|---|
| 1 | Nested Learning / Hope: a chain of blocks updated at different frequencies, brain-motivated, with continual-learning gains | **Confirmed in outline; corrected in detail.** The chain and the frequencies are as claimed (Sec. 7.1, Eqs. 70–71). Everything is pretrained by backprop (AdamW), and the inner levels are updated by gradient-type rules. The brain-wave link is an analogy for ordering the levels; Section 7 has no neuroscience. The class-incremental gains are shown only in a figure, on 3B–8B Llama backbones, against ICL, EWC and InCA, with no replay baseline and no seed count. No independent replication was found. |
| 2 | Reading A is old and works; leaky integrators at a spectrum of rates amount to a Laplace transform of the past | **Confirmed, with two qualifications.** Modules at fixed clock rates beat a plain RNN and an LSTM on small tasks (Clockwork RNN, Table 3: TIMIT word error 16.8 vs 66.8 and 34.2, 100 runs), and the Laplace statement holds as theory. But (a) which rates are chosen matters (S4D), and log-spacing is optimal only under a scale-free prior, which was argued and not tested; (b) in the one controlled ablation found, purely real decays lost 5–10 points to complex (oscillating) ones. |
| 3 | Reading C exists at the synapse, extends memory lifetime in theory and buys measured retention in trained networks | **Confirmed, narrowed.** Theory: Benna–Fusi chains, about linear capacity in the number of synapses (random, uncorrelated memories). Trained networks: a deep-RL agent with 30 variables per weight relearns alternating tasks where the control often never does (2 tasks, 3 runs). A one-variable metaplastic rule matches EWC on six permuted-MNIST tasks without task boundaries (5 runs). Nothing was found on sequence, language-like or compositional tasks, or with a sign-type learning rule. |
| 4 | Learned recurrence that is not element-wise makes exact online credit expensive, which is why the banks avoid it | **Confirmed for forward-only credit, which is the reason given in the `arch17.py` header.** Element-wise RTRL costs O(DN) memory per layer; full RTRL costs O(N³) memory and O(N⁴) time (morning memo, Q1, Irie et al. 2024). **Correction of scope:** under the life rule now in force (a backward sweep over the item, 9.82 addendum), an inner loop costs K times the work per byte, not an explosion of traces. |
| 5 | The fly has parallel memories at different rates; oscillation has an established role in ordering or combining old and new information in some nervous systems | **First half confirmed (morning memo, Q4: Aso & Rubin 2016). Second half narrowed.** In rodents, theta sequences and sharp-wave-ripple replay are well documented, and disrupting ripples impairs memory. The theta–gamma code for ordered items is a supported hypothesis whose capacity link is untested. In the fly, about 10 Hz odour-evoked oscillations in the mushroom body have no shown function, and the slow dopamine-neuron oscillations gate consolidation (the causal test silenced the neurons, not the rhythm). |

**A, B and C: what is established and what each would cost here**

| Reading | Established | Cost in this project (estimate from `arch17.py` shapes; `bank2` is about 1.125M parameters, 131,072 bank-write synapses) |
|---|---|---|
| **A. State at several decay rates** | Yes: diagonal state-space models and multi-clock RNNs. Choice of rates matters, and complex eigenvalues beat real ones in one controlled ablation | **Built** (five fixed real rates). Making the rates complex would double the state per unit (2 instead of 1 float), take 4 multiply-adds instead of 1 per unit per byte, and double the traces per write synapse |
| **B. An inner loop** | Shared-weight iteration solves multi-step addition in small transformers, but only as well as a deeper model at matched compute. Its demonstrated advantage is parameters, not iteration itself | **Absent.** One 512×512 loop is about 262K weights (+23%). K iterations cost K×262K multiply-adds per byte forward and about twice that backward, under the backward-sweep rule |
| **C. Weights at several rates** | Yes, at small scale: Benna–Fusi chains, metaplasticity and slow EMA copies (the last with replay) | **Absent.** A synapse-local chain costs m−1 extra floats per synapse and about 2(m−1) multiply-adds per synapse per update. A metaplastic gate costs 0 extra floats |

---

## Q1. The Nested Learning paper

Behrouz, Razaviyayn, Zhong, Mirrokni, arXiv 2512.24695 (31 Dec 2025; comment: "a version … published at NeurIPS 2025").

**What is updated how often, by what rule.**
- **CMS (Sec. 7.1):** a chain of MLP blocks (Eq. 70); block ℓ changes only every C^(ℓ) steps, by its chunk's summed optimizer error terms, for example the gradient of the task loss (Eq. 71); initial states meta-learned across levels (Eqs. 72–74). It replaces the Transformer MLP.
- **Hope (Sec. 8):** "self-referential Titans" whose memories generate their own keys, values, step sizes and decay (Eqs. 83–84) and are updated in context by a delta-rule step on an L2 regression loss (Eqs. 85–88); each memory is a residual two-layer MLP (Eq. 89). "Self-modifying" means exactly this.
- **Learning:** everything is trained with AdamW (Sec. 9.3); in-level updates are gradients through a two-layer MLP; levels communicate through meta-learned initial states. Nothing is local in the project's sense.

**Results (authors' runs only; no seeds or error bars anywhere in the HTML).**
- Table 2 (spot-checked): 1.3B/100B tokens, average 58.04 vs Titans 56.82, WikiText perplexity 14.39 vs 15.60; 760M/30B, 52.28 vs 51.68.
- Table 6: removing CMS, perplexity 12.24 → 13.04, reasoning 58.1 → 57.3.
- Sec. 9.1 continual learning: class-incremental CLINC, Banking, DBpedia on Llama3-8B and "Llama-3B" after 15B tokens of continual pre-training, against ICL (same pre-training, unchanged MLPs), EWC and InCA; "reported in Figure 6", **no numbers in text, no replay baseline**. More levels help in Figs. 7–8.

**Brain link.** Sec. 1 lists gamma 30–150 Hz (sensory), beta (thinking), delta/theta 0.5–8 Hz (consolidation); Sec. 3.2: "Inspired by the hierarchy of brain waves … we use the update rate" to order levels. Nothing ties the chunk sizes (lowest frequency 512 or 2K tokens, Fig. 7) to a rhythm. **An analogy, not evidence**, and rhythms are activity oscillations whereas CMS frequencies are weight-update rates.

**Replication.** None found; search returned only papers reusing the name for a prompt cache.

## Q2. Weights at several rates, for retention without rehearsal

| Method | Per synapse | Local? | Retention shown | Importance estimate? |
|---|---|---|---|---|
| **Benna–Fusi chain** (Benna & Fusi; Kaplanis et al. 2018) | m coupled variables; weight = u₁ (Eqs. 2–3) | Yes, given Δw | Theory: capacity "almost linearly" in synapse count, random uncorrelated memories. Deep RL, Cart-Pole/Catcher alternated, m=30, 3 runs: Benna–Fusi "relearned the task quickly in all epochs", controls often never relearned Cart-Pole (Fig. 4; within-task replay buffer of 2,000, cleared each epoch). With no replay at all, a 100–50-unit net: controls never held Cart-Pole, Benna–Fusi mostly did; on Catcher the control was faster (Sec. 3.3). Runtime 1.5–2× | None: passive flow |
| **Metaplastic hidden weight** (Laborieux et al.) | 1 hidden value; updates toward zero scaled by 1 − tanh²(m·W_h) (Alg. 1) | Yes | Six permuted MNIST, binarized 2×4096, 5 trials (Table 1): earlier tasks 7.8–13.2% plain vs 96.4–97.3% metaplastic; EWC 96.6–97.2; random consolidation 29–48; learning-rate decay 64–87. **No task boundaries, no replay** | Hidden-weight magnitude |
| **Synaptic Intelligence** (Zenke et al. 2017) | Path integral ω (gradient × update, Eq. 3), Ω, reference weight | ω "computed … locally at each synapse"; Ω and reference set **at task ends** (Eqs. 4–5) | Split/permuted MNIST, 2×256 and 2×2000 MLPs, n=5, figures only; tracks EWC | Yes, local, needs boundaries |
| **Slow EMA copies, CLS-ER** (Arani et al. 2022) | 2 extra copies, θ ← αθ + (1−α)θ_W at stochastic rates (Eq. 2) | Copy local; acts through a loss **on replayed exemplars** | S-CIFAR-10 class-IL, 10 runs (Table 1): buffer 200, ER 44.79 vs 66.19; buffer 500, 57.74 vs 75.22 | No; needs replay |

**Answer.** Benna–Fusi chains and metaplasticity buy retention without rehearsal, synapse-locally and with no separate importance estimate, but only on two RL tasks and permuted MNIST (tasks with little input overlap). Slow copies that beat replay need replay. None was tested against a replay-share curve like E16-M3's, on sequences, or with sign-based credit.

## Q3. Inner loops, for multi-step computation

- **Iteration vs computation.** Saunshi et al. (ICLR 2025, Table 1, spot-checked), addition of n 3-digit numbers, n = 8/16/24/32: one layer looped 12× 99.9/100.0/99.9/99.6; one layer once 0.1/0.1/0.1/0.0; 12 layers 100 throughout. Same for p-hop and i-GSM (Table 2: 1⊗8 loop 73.2 = 8 layers 73.2). Small models (width 256 for addition), best of 3 seeds reported. **At matched compute, iteration equals depth; at matched parameters it beats the shallow model.** The loop buys parameters, not something beyond compute.
- **Easy to hard.** Schwarzschild et al. 2021 (abstract): recurrent nets trained with few iterations solve harder prefix sums, mazes and chess when run longer. Their feedforward comparison was not read.
- **Fast weights with an inner loop.** Ba et al. 2016: A ← λA + ηhhᵀ, S settling steps per input (Eqs. 1–2); associative retrieval error 1.81% vs LSTM 60.81% at the table's R = 20 (Table 3). Its state is an H×H matrix per sequence.
- **Credit** (morning memo, Q1). Element-wise layers keep exact per-synapse traces; dense learned recurrence needs a full sensitivity, O(N³) memory and O(N⁴) time for full RTRL (Irie et al. 2024), and stacking already reintroduces it. Approximations (SnAp, e-prop, Zucchet's spatial-only error) drop credit through the loop, worse with depth and retention. **No forward-only method for a within-step loop was found.** Under the backward sweep now ruled acceptable (9.82 addendum) the loop is simply unrolled K steps per byte.
- **Thalamocortical loops:** nothing new opened; the re-screen's 5.3 caution stands, unchecked by me.

## Q4. Oscillation proper

- **One controlled comparison.** S4D (Gu et al. 2022, Table 3a; 4 layers, H=128, about 100K parameters, 2–3 seeds): real-eigenvalue S4D-Real (no oscillation) loses 5.45 points on sCIFAR and 10.17 on autoregressive Speech Commands, and is worse on BIDMC (RMSE +0.066), against the complex initialisations. These are long-range tasks, not ordered items.
- **Uncontrolled.** LRU's complex diagonal beats a *dense real* matrix (Table 2, sCIFAR 86.5 vs 72.2, 3 seeds) with no real-diagonal arm (Apps. C.3, E.3). coRNN and LinOSS ("outperforms Mamba and LRU by nearly 2x" at length 50k), Phased LSTM, and phase-binding models (Complex AutoEncoder, AKOrN) compare with other architectures, not with decay banks (abstracts).
- **Answer.** No controlled evidence that oscillating units beat matched plain decays at holding items in order or binding them in sequence.

## Q5. Biology, briefly

- **Established.** Sharp-wave ripples replay "fragments of waking neuronal sequences in a compressed format", mostly at rest and in sleep, and "selective disruption of SPW-Rs interferes with memory" (Buzsáki 2015, abstract). Intrinsic timescales rise from about 50 to 350 ms from sensory to prefrontal cortex (Murray et al. 2014; 7 areas, 26 monkeys, Figs. 1–2).
- **Supported, open.** The theta–gamma code holds 4–8 ordered items in successive gamma cycles of a theta cycle; support from phase precession, coupling during learning and septal manipulations; the authors call the ratio–span link "not yet rigorously tested" and coherence routing contested (Lisman & Jensen 2013).
- **Fly.** Odours evoke about 10 Hz mushroom-body LFP oscillations that need the widely branching LN2 antennal-lobe interneurons (Fig. 7); a behavioural role is only speculated (Tanaka, Ito & Stopfer 2009). MP1/MV1 dopamine neurons oscillate slowly (peaks about 0.05–0.29 Hz in example flies, Fig. 3e), weaker after massed and stronger 30–60 min after spaced training (Figs. 4, 6); blocking them during the 15-min inter-trial intervals prevents long-term memory (Fig. 7) (Plaçais et al. 2012). No oscillatory loop inside the mushroom body carrying item order was found; the feedback loops are in the morning memo, Q4.

## Q6. What follows for this project (suggestions; they decide nothing)

**Closest published method per reading**

| Reading | Closest method | Reported limits |
|---|---|---|
| A | Diagonal SSMs with fixed or learned decays (S4D, LRU); Clockwork RNN | Initialisation matters; real-only decays lost 5–10 points in S4D's ablation; log-spacing optimality not tested |
| B | Looped shared-weight layers (Saunshi); recurrent "thinking longer" (Schwarzschild) | Equals a deeper model at matched compute; exact forward-only credit is cubic or worse; small benchmarks |
| C | Benna–Fusi chain (Kaplanis); metaplastic hidden weight (Laborieux); CLS-ER | Small tasks only; CLS-ER needs replay; SI needs task boundaries; nothing on sequences or sign rules |

**(i) Fast and slow weights in the window learner.** Give every weight a three-variable Benna–Fusi chain (u₁ visible, written by the sign rule; two hidden variables). Lives at replay shares 0, 0.25 and 0.5 on the E16-M3 seeds; read hard pairs and the never-shown mod-89 operations. Controls: no chain; a global learning-rate decay (weaker but non-trivial in Laborieux Table 1); random consolidation strength. Falsifier: at share 0 or 0.25 the chain fails to beat both controls on mod-89. Cost: 2 floats and about 4 multiply-adds per synapse per update; nothing per byte at inference. The metaplastic gate costs 0 floats, but was shown on binarized weights (its real-valued result, Fig. 2f, not read).

**(ii) An inner loop in the two-bank learner.** One shared 512×512 tanh transform applied K ∈ {1, 2, 4} times per byte before the logits; banks keep their element-wise traces; the loop's credit rides the item-level backward sweep already in force. Controls (Saunshi's): K unshared layers (matched compute) and K=1 (matched parameters); the existing two-hidden-layer readout `cell1d2` in `arch17.py` should be read first. Falsifier: composition stays at the 12.8–18.5 held / 0.8–8.6 hard of 9.70 for every K; if the loop helps no more than unshared layers, it is depth, not iteration. Cost: about 262K parameters (+23%); per byte K×262K multiply-adds forward, about twice that backward, K activations stored. A forward-only version would need an O(N²) sensitivity per loop unit; not recommended.

**(iii) Oscillating units.** At most one cheap probe, and only if a measured failure is about order (the `+-`/`-+` hard pairs are order-sensitive): make the five retentions complex with the same magnitudes and fixed phases, nothing else changed. Cost: 2 floats of state and 4 multiply-adds per unit per byte; traces per write synapse double (about 131K extra). Support: S4D's long-range ablation only.

## References (all opened in this session)

- Behrouz, Razaviyayn, Zhong, Mirrokni (2025). *Nested Learning: The Illusion of Deep Learning Architectures.* arXiv 2512.24695; per its comment, a version at NeurIPS 2025. https://arxiv.org/abs/2512.24695 and https://arxiv.org/html/2512.24695. Read: Secs. 1, 3.2, 7.1, 8, 9.1–9.6; Tables 2 and 6; Figs. 6–8.
- Koutník, Greff, Gomez, Schmidhuber (2014). *A Clockwork RNN.* arXiv 1402.3511. https://ar5iv.labs.arxiv.org/html/1402.3511. Table 3: TIMIT word error RNN 66.8±4.7, LSTM 34.2±5.6, CW-RNN 16.8±3.5 (100 runs, small nets).
- Spears, Jacques, Howard, Sederberg (2017). *Scale-invariant temporal history (SITH).* arXiv 1712.07165. https://ar5iv.labs.arxiv.org/html/1712.07165. Sec. 2.1 (Laplace transform; optimality cited, not shown).
- Tallec, Ollivier (2018). *Can recurrent neural networks warp time?* arXiv 1804.11188. https://arxiv.org/abs/1804.11188. Abstract only.
- Gu, Gupta, Goel, Ré (2022). *On the Parameterization and Initialization of Diagonal State Space Models.* arXiv 2206.11893. https://arxiv.org/pdf/2206.11893. Sec. 5, Table 3a.
- Orvieto, Smith, Gu, Fernando, Gulcehre, Pascanu, De (2023). *Resurrecting Recurrent Neural Networks for Long Sequences.* arXiv 2303.06349. https://arxiv.org/html/2303.06349. Table 2; Apps. C.3 and E.3.
- Benna, Fusi (2015). *Computational principles of biological memory.* arXiv 1507.07580 (preprint of the 2016 Nature Neuroscience paper). https://arxiv.org/abs/1507.07580v1. Abstract.
- Kaplanis, Shanahan, Clopath (2018). *Continual Reinforcement Learning with Complex Synapses.* ICML 2018. https://arxiv.org/abs/1802.07239. Secs. 2.1, 3.2–3.3; Figs. 4–5.
- Laborieux, Ernoult, Hirtzlin, Querlioz (2020/21). *Synaptic Metaplasticity in Binarized Neural Networks.* arXiv 2003.03533v2. https://arxiv.org/abs/2003.03533. Alg. 1, Table 1.
- Zenke, Poole, Ganguli (2017). *Continual Learning Through Synaptic Intelligence.* ICML 2017. https://arxiv.org/abs/1703.04200. Eqs. 3–5, Secs. 5.1–5.2.
- Arani, Sarfraz, Zonooz (2022). *Learning Fast, Learning Slow (CLS-ER).* ICLR 2022. https://arxiv.org/abs/2201.12604. Eq. 2, Table 1.
- Saunshi, Dikkala, Li, Kumar, Reddi (2025). *Reasoning with Latent Thoughts: On the Power of Looped Transformers.* ICLR 2025. https://arxiv.org/html/2502.17416v1. Tables 1–2.
- Schwarzschild, Borgnia, Gupta, Huang, Vishkin, Goldblum, Goldstein (2021). *Can You Learn an Algorithm?* arXiv 2106.04537. https://arxiv.org/abs/2106.04537. Abstract only.
- Ba, Hinton, Mnih, Leibo, Ionescu (2016). *Using Fast Weights to Attend to the Recent Past.* arXiv 1610.06258. https://ar5iv.labs.arxiv.org/html/1610.06258. Eqs. 1–2, Table 3.
- Rusch, Mishra (2020/21). *Coupled Oscillatory Recurrent Neural Network (coRNN).* arXiv 2010.00951. https://arxiv.org/abs/2010.00951. Abstract.
- Rusch, Rus (2025). *Oscillatory State-Space Models.* ICLR 2025. https://arxiv.org/abs/2410.03943. Abstract.
- Neil, Pfeiffer, Liu (2016). *Phased LSTM.* NIPS 2016 (per arXiv comment). https://arxiv.org/abs/1610.09513. Abstract.
- Löwe, Lippe, Rudolph, Welling (2022). *Complex-Valued Autoencoders for Object Discovery.* TMLR. https://arxiv.org/abs/2204.02075. Abstract.
- Miyato, Löwe, Geiger, Welling (2025). *Artificial Kuramoto Oscillatory Neurons.* ICLR 2025. https://arxiv.org/abs/2410.13821. Abstract.
- Lisman, Jensen (2013). *The theta-gamma neural code.* Neuron 77:1002–1016. https://pmc.ncbi.nlm.nih.gov/articles/PMC3648857/
- Murray et al. (2014). *A hierarchy of intrinsic timescales across primate cortex.* Nature Neuroscience 17:1661–1663. https://pmc.ncbi.nlm.nih.gov/articles/PMC4241138/. Figs. 1–2.
- Buzsáki (2015). *Hippocampal sharp wave-ripple: a cognitive biomarker for episodic memory and planning.* Hippocampus 25:1073–1188. https://pmc.ncbi.nlm.nih.gov/articles/PMC4648295/. Abstract and opening sections.
- Tanaka, Ito, Stopfer (2009). *Odor-evoked neural oscillations in Drosophila are mediated by widely branching interneurons.* J Neurosci 29:8595–8603. https://pmc.ncbi.nlm.nih.gov/articles/PMC2753235. Figs. 1, 6, 7.
- Plaçais, Trannoy, Isabel, Aso, Siwanowicz, Belliart-Guérin, Vernier, Birman, Tanimoto, Preat (2012). *Slow oscillations in two pairs of dopaminergic neurons gate long-term memory formation in Drosophila.* Nature Neuroscience 15:592–599. https://www.bio.espci.fr/sites/www.bio.espci.fr/IMG/pdf/placaispreat2012.pdf. Figs. 3–7.
- Cited, not re-read: this morning's memo, `PROJECT_DOCUMENTS/LIT_CHECK_20261110/LITERATURE_CHECK_LOCAL_CREDIT_IN_TIME.md` (Q1: Irie, Zucchet, SnAp, e-prop; Q4: Aso & Rubin and fly loops).

## From memory, not verified

- Shankar & Howard (Neural Computation 2012) as the original "leaky integrators = Laplace transform, inverted by Post's formula" paper: seen in search snippets only.
- Fusi, Drew & Abbott (2005) cascade synapses, and Hinton & Plaut (1987) fast weights for deblurring old memories: not opened.
- Laborieux et al. as published in Nature Communications (2021): the journal page redirected to a login.
- Recall-gated consolidation (eLife 90793, cited in re-screen 5.5): the page failed to load.
- Benna–Fusi's lifetime scaling with the number of variables per synapse: not in the abstract read.

## Searched for and not found

- Any independent replication or critique of Nested Learning / Hope.
- A controlled comparison of oscillating against plain-decay units for ordered multi-item holding or sequence binding.
- Any multi-rate synapse (cascade, metaplasticity or EMA copy) tested on sequence or language-like tasks, against a replay-share curve, or with sign-based or other non-backprop credit.
- A forward-only exact credit method for a within-step inner loop.
- An oscillatory loop within the Drosophila mushroom body carrying item order.
