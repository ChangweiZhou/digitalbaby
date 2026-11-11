# Source check: the delayed-observation memo and three remarks (11 November 2026)

**Job S2** of `OPUS_TASK_SIDE_JOBS_READ_ONLY_20261110.md`. Written by a cloud worker session, read-only. It ran no experiment, trained nothing and edited no existing file. **It decides nothing.**

**Inputs read:** the precedents section of `PROJECT_DOCUMENTS/MINIFLY_E18_DELAYED_OBSERVATION_MEMO_20261110.md`; `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/NEXT_STEPS_NOTE_20261110.md`, 3.1 (the brief gives the path without `PROJECT_DOCUMENTS/`); `LIT_CHECK_20261110/LITERATURE_CHECK_LOCAL_CREDIT_IN_TIME.md`; `PROJECT_DOCUMENTS/TARGET_TEST_DESIGN_DRAFT_20261110.md` for context.

**How sources were opened.** Every source cited below was opened in this session. Most were opened with WebFetch. Four PDFs were downloaded and read as text with `pdftotext`, because WebFetch could not read them: CLAPP (NeurIPS proceedings PDF), P-TNCN (arXiv v4 PDF), Foster & Wilson 2006 (a course mirror of the Nature PDF) and Halvagal & Zenke 2023 (Nature Neuroscience PDF). Two eLife pages (RFLO, fly sensory preconditioning) failed to load, so their PMC copies were read instead.

## Verdicts

| # | Item | Verdict |
|---|---|---|
| 1a | Greedy InfoMax | **Supported.** Cutting BPTT in the context module lowers phone accuracy from 62.5 to 55.5; speaker accuracy barely moves (99.4 to 99.2). GIM's context module itself uses BPTT. |
| 1b | CLAPP | **Supported, qualified.** In the image experiment the "later view" is the next patch of a static image. Fully local CLAPP was run on images only: 73.6% against CPC's 81.1%. |
| 1c | P-TNCN, 1.70 against 1.38 BPC | **Supported exactly** (Table I). The paper is arXiv-only, with no seeds reported. It is level with the authors' own RTRL LSTM (1.71). |
| 1d | e-prop, with a late teacher | **Supported** (Fig. 3). The learning signal arrives only in the last 150 of 2,250 ms. |
| 1e | RFLO: the gap to BPTT grows with delay | **Supported** (Figs. 2–3). The gap comes from the local approximation, not from random feedback. |
| 1f | TCN: pairing views of the same moment | **Supported.** The delayed, committed version is the memo's own extension. |
| 1g | Fly sensory preconditioning | **Supported, narrower.** Behaviour needed 10 pairings at a 1-s interval. One pairing, or a 30-s or 5-min interval, gave none in wild-type flies. A reviewer called the effect "weak". |
| 2 | Remark 3.1 on reverse replay | **Holds for awake replay at the end of a run. "Recent" is contested.** No source opened shows a causal role specific to reverse replay. Credit assignment is proposed, modelled and correlated with learning, not shown. |
| 3 | No sign-symmetric feedback through time | **Revised, not overturned.** Nothing was found with sign feedback through recurrent weights or in a sequence model. One preprint (Goupy et al. 2025/26) uses unscaled sign symmetry in spiking ResNets under 4-step BPTT, and it **collapses to chance** there (CIFAR-10: 10.0–10.8% against 87–92%; 8 seeds). |
| 4 | Local prediction of later observations; shuffled control | **Only LPL** (Halvagal & Zenke 2023) has a control that destroys temporal contiguity, on images. No source uses the memo's stratified target shuffle. |

## Q1. The memo's precedents: the passages behind the verdicts

The numbers are in the verdict table. These are the passages behind them.

- **GIM** (§4.2, ablation table; Table 7 in the ar5iv rendering, Table 4 in the earlier memo). GIM "performs best on downstream tasks where temporal or context dependencies do not need to be modeled". Its autoregressive module "employs Backpropagation Through Time".
- **CLAPP.**
  - §2 models a saccade by "feeding the network with subsequent frames from the same sample ... and then abruptly changing to frames from another sample".
  - For STL-10, "each image is split into 16 × 16 patches and the patches are viewed one after the other in a vertical order" (§4).
  - Speech: CLAPP-s reaches 61.7 against CPC's 64.3, and "still relies on BP through time (BPTT) to train the GRU layer". Using e-prop instead "reduces performance by only 3.1 %".
  - Video: fully local CLAPP had "prohibitively long convergence times" (Table 1 and the text after it).
- **P-TNCN**, Table I and §B.
  - Other test BPC in the same table: truncated BPTT over 25 steps (cited) 1.56; the authors' GRU-BPTT 1.41, LSTM-RTRL 1.71, RNN-BPTT 2.16.
  - Text models have 2 layers, with 1,000–3,000 units searched.
  - Authors: P-TNCN "does not quite yet reach the level of performance of BPTT and SAB".
- **e-prop**, Fig. 3 and Eqs. 2–4. "A teacher tells the subject at the end of each trial what would have been the right decision". The traces are exact; the learning signal "ignores these indirect influences".
- **RFLO.**
  - Fig. 2 (N = 30): BPTT "shows an advantage ... when the number of timesteps became very large".
  - Fig. 3 (N = 100): for delays beyond about 15τ, "BPTT performs better than RFLO learning".
- **TCN.** "Multiple simultaneous viewpoints of the same observation are attracted", "repelled from temporal neighbors" (abstract). A single-view variant takes positives within 0.2 s (§III-A, §IV-A). Baselines: shuffle & learn, ImageNet features.
- **Fly sensory preconditioning.** Odors were 5-s pulses (MCH, then OCT), at intervals of 1 s, 30 s or 5 min.
  - Imaging in MBON-γ1pedc>α/β: S2 depression at 1 s only (p = 0.035, n = 10; Fig. 1C).
  - Behaviour: p = 0.0019 (n = 12) after 10 pairings at 1 s (Fig. 4B).
  - Inhibiting Rac1 extends the window to 30 s (Fig. 4C).
  - Authors: one pairing "is not enough for the S2 odor to drive the learned behavior".

## Q2. Reverse replay

**Established: timing and form.**
- Foster & Wilson 2006 (4 rats). During pauses "immediately after running", sequences recur "in a temporally reversed order", spanning "the entire track, on a timescale of hundreds of milliseconds", together with ripples.
  - It occurred even after the first lap on a new track.
  - It was seen more readily on a new track than a familiar one.
  - None occurred in 42–424 s before running.
- Diba & Buzsáki 2007 (3 rats). 85% of reverse events (395/464) followed the run, at its end. 95% of forward events (841/887) preceded it. Forward events were more common: 36% against 19% of all events.

**Established in two labs: relation to reward.**
- Ambrose, Pfeiffer, Foster 2016 (5 rats): "only reverse replays increased their rate at increased reward, or decreased their rate at decreased reward". The authors "prefer a more conservative interpretation".
- Bhattarai, Lee, Jung 2019 (3 rats): reward raised the rate of reverse replay (p = 0.001) and the fidelity of forward replay.

**Contested.**
- *Recency.* Gupta et al. 2010 (6 rats): about half of the replay of a path not run for a median of 337 s was backward, and 19 never-experienced shortcuts were replayed. "Our observations are not compatible with the hypothesis that replay reflects the recent experience."
- *Reward.* Wu et al. 2017 (4 rats): awake replay toward an avoided shock zone. Its direction was mostly undecidable, because 70% of the cells were bidirectional.
- *Causation.* Jadhav et al. 2012: disrupting all awake ripples impaired outbound learning (6 rats against 4 and 4 controls), with no selectivity for direction. No source opened here blocks reverse replay alone.

**As credit assignment.**
- **Proposed** by Foster & Wilson: "reverse replay could be paired with a fast-onset, slowly decaying dopamine signal to learn a representation of value". This is not tested.
- **Modelled** by Mattar & Daw 2018. Reverse sequences arise from a "gain" term after a prediction error: "learning amounts to propagating value between adjacent states for temporal credit assignment". The authors call the gain computation "admittedly unrealistic".
- **Correlated in humans** by Liu et al. 2021 (n = 29, MEG). Backward replay with a 160-ms lag after reward went with higher non-local learning rates (0.70 against 0.61, p = 0.023), described as "well suited to solve the non-local credit assignment problem". The result is correlational.

**What is replayed.** In every source, what runs backward is a sequence of reactivated states; in the models, value. None of the sources describes an error signal carried backward through synapses.

## Q3. Sign-symmetric feedback in recurrent networks or through time: the wider search

**Found and opened.**
- **Goupy, Tirilly, Bilasco, *Frozen Backpropagation* (arXiv 2505.13741, v2 revised 8 May 2026, preprint).**
  - Sign symmetry "reduces B to the sign of W". The paper mentions no per-layer scale.
  - Tables 1–2, mean ± sd over 8 initialisations, BP against SS:
    - Temporally coded VGGs trained by event-driven BP, with no time steps: CIFAR-10 VGG-7 90.40 against 89.39; VGG-11 92.04 against 91.26. CIFAR-100: 66.05 against 63.43 and 67.45 against 65.34.
    - Rate-coded ResNets trained by **BPTT over 4 time steps**: CIFAR-10 ResNet-18 87.14 against **10.80**; ResNet-26 91.82 against **10.00**. CIFAR-100: 60.57 against **0.90** and 69.37 against **1.00**. Sign-concordant FA reaches 28–56% in the same networks.
  - §4.2.2, tentatively: "sign transport alone can be insufficient"; BPTT "propagates biased feedback errors across both depth and time", which "may cause misalignment to accumulate more rapidly".
  - Appendix A.6: 250 hyperparameter configurations still left SS far below BP.
  - The networks are convolutional, with membrane state across steps and no recurrent weight matrices. The neuron model is not stated (it follows a cited architecture). Parameter counts are not given.
  - So this is sign feedback through *time-stepped neuron state*, unscaled, in a vision task, not in a recurrent or sequence learner.
- **Opened; nearest in time, but none uses sign feedback.**
  - Folchini et al. 2025: DFA in RNNs, where BPTT "still achieves better predictive accuracy". No numbers are on the record page.
  - Anon & Emon 2026: DFA in a recurrent LIF network, 91.5% on Speech Commands.
  - Lin et al. 2024: e-prop and DECOLLE against BPTT.
  - Yang & Li 2022: sign-concordant FA in a spiking microcircuit, where recurrence is not stated.
  - Barretto-Bittar et al. 2026: diffusive credit against e-prop.
  - Hao et al. 2026: diagnostics for FA.
- **Withdrawn.** Berriche et al., *Sign-Symmetry Learning Rules are Robust Fine-Tuners* (arXiv 2502.05925). Version 2 (25 July 2026) is withdrawn: "Paper makes claims that were later found to be false". Do not cite it.

**Not found:** sign-symmetric or sign-concordant feedback through *recurrent weight matrices*, in an LSTM, GRU or vanilla RNN, in a language or byte model, or compared with BPTT on a sequence task.

**Queries run.** Ten WebSearch queries, eight of them (all but 8 and 9) in extended mode:
1. `sign symmetry feedback recurrent neural network backpropagation through time`
2. `sign-concordant feedback alignment RNN LSTM language modeling`
3. `"sign symmetry" OR "sign-symmetric" recurrent spiking network e-prop feedback weights sign`
4. `weight-sign feedback "through time" recurrent learning rule 2025`
5. `"sign-only" OR "sign transport" feedback weights backpropagation recurrent network 2026 arXiv`
6. `sign symmetry spiking neural network BPTT surrogate gradient feedback weights sign(W) temporal`
7. `"signed feedback alignment" OR "sign feedback alignment" OR "sign-concordant" recurrent network time 2025 2026`
8. `"sign symmetry" LSTM OR GRU OR "recurrent neural network" biologically plausible credit assignment`
9. `sign-symmetric feedback transformer language model weight transport 2025 2026 arXiv`
10. `sign-concordant OR sign-symmetric feedback eligibility trace recurrent temporal credit assignment local learning rule`

**arXiv API abstract searches**, newest first:
- `"sign symmetry"`, `"sign concordant"`, `"sign-concordant"`, `"sign feedback"`.
- `"sign symmetry"` AND each of recurrent, spiking, time, "language model".
- `"feedback alignment"` AND each of recurrent, RNN, "through time".
- `"weight transport"` AND each of recurrent, spiking.
- `"weight symmetry"` AND recurrent; `"random feedback"` AND recurrent; `sign` AND `"feedback alignment"`.

**Limits of the search.**
- The API's AND queries were unreliable: `"weight transport" AND spiking` missed Goupy et al.
- One call was rate-limited (HTTP 429).
- Google Scholar, OpenReview and proceedings were not searched directly.
- Titles seen only in search results (for example TESS, arXiv 2502.01837) are not cited.

## Q4. Predicting later observations under local rules

**The three results most relevant to a learner that predicts every next byte.**
1. **P-TNCN** (Q1). Next-character prediction with no unrolling: 1.70 against 1.38 BPC.
   - Scale: 2 layers of 1,000–3,000 units.
   - Controls are algorithm baselines only. There is **no shuffled control** and no seed count.
2. **LPL** (Halvagal & Zenke 2023). Each layer predicts its own next output, plus a Hebbian variance term and decorrelation, with no backprop between layers.
   - Scale: VGG-11, 4 seeds. Layer-local against end-to-end: STL-10 63.2 against 72.5; CIFAR-10 59.4 against 70.4 (Table 1).
   - Without the predictive term the rule falls to 27.0%, below raw pixels (31.6%).
   - **Shuffled control:** "the full LPL learning rule applied to inputs in which temporal contingency is destroyed (LPL shuffled) behaves qualitatively like the purely Hebbian rule" (Fig. 3).
3. **Successor representation by a local rule** (Fang et al. 2023). An STDP-like rule with anti-Hebbian normalisation learns transitions, so the recurrent network's dynamics compute a multi-step prediction of later states.
   - Scale: simulated random walks in 1D and 2D, and one place-cell dataset.
   - Compared with the ground truth and a TD-learned SR. There is **no shuffled control**.
   - Limitation stated by the authors: "one-step transition errors propagate over multi-step predictions".

**Also opened.**
- Temporal predictive coding (Millidge et al. 2024): on par with a Kalman filter on a 3-D tracking task. No shuffled control was found.
- Moon et al. 2026: predicting x_{t+2} helps at the first block boundary and hurts deeper. Contrastive auxiliaries fell into positional and document-identity shortcuts. Single seed, and backprop within blocks.

**Answer.** Among the results opened, only LPL destroys the relation between input and target in time while keeping the inputs. It does so by shuffling temporal order, not by shuffling targets within delay and frequency strata as the memo proposes.

## From memory, not verified

- TCN's venue (believed to be ICRA 2018). The arXiv page lists none.
- That Foster & Wilson's Nature article and the course-mirror PDF are identical. The PDF carries the Nature DOI and header.

## Could not open

- The eLife HTML pages for RFLO and fly sensory preconditioning, and the eLife peer-review page for the fly paper. The PMC copies were used, including the decision letter in the fly paper's PMC copy.
- The bioRxiv full text of LPL (HTTP 429). The Nature PDF was used instead.
- The full text of Folchini et al. 2025: only its Zenodo record was opened.
- Supplementary Fig. 5 of Wu et al. 2017 (the forward/reverse breakdown).
- The content of the 2026 correction to Jadhav et al. 2012.

## References (all opened in this session)

**Q1**
- Löwe, O'Connor, Veeling (2019). *Putting An End to End-to-End: Gradient-Isolated Learning of Representations.* NeurIPS 2019 (from the arXiv comments field). https://arxiv.org/abs/1905.11786 ; https://ar5iv.labs.arxiv.org/html/1905.11786. §4.2, ablation table.
- Illing, Ventura, Bellec, Gerstner (2021). *Local plasticity rules can learn deep representations using self-supervised contrastive predictions.* NeurIPS 2021. https://proceedings.neurips.cc/paper_files/paper/2021/file/feade1d2047977cd0cefdafc40175a99-Paper.pdf. §2, §4, Table 1.
- Ororbia, Mali, Giles, Kifer (2018/2019). *Continual Learning of Recurrent Neural Networks by Locally Aligning Distributed Representations.* arXiv preprint. https://arxiv.org/abs/1810.07411 (v4 PDF). Table I, §B.
- Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein, Maass (2020). *A solution to the learning dilemma for recurrent networks of spiking neurons.* Nature Communications 11:3625. https://pmc.ncbi.nlm.nih.gov/articles/PMC7367848/. Fig. 3, Eqs. 2–4.
- Murray (2019). *Local online learning in recurrent networks with random feedback.* eLife 8:e43299. https://pmc.ncbi.nlm.nih.gov/articles/PMC6561704/. Figs. 2–3.
- Sermanet, Lynch, Chebotar, Hsu, Jang, Schaal, Levine (2017). *Time-Contrastive Networks: Self-Supervised Learning from Video.* arXiv preprint. https://arxiv.org/abs/1704.06888 ; https://ar5iv.labs.arxiv.org/html/1704.06888. Abstract, §III-A, §IV-A.
- Martinez-Cervantes, Shah, Phan, Cervantes-Sandoval (2022). *Higher-order unimodal olfactory sensory preconditioning in Drosophila.* eLife 11:e79107. https://pmc.ncbi.nlm.nih.gov/articles/PMC9566850/. Figs. 1–4 and the decision letter.

**Q2**
- Foster, Wilson (2006). *Reverse replay of behavioural sequences in hippocampal place cells during the awake state.* Nature 440:680–683. https://courses.csail.mit.edu/6.803/pdf/2006%20wilson.pdf. Text p. 2, Figs. 2–4.
- Diba, Buzsáki (2007). *Forward and reverse hippocampal place-cell sequences during ripples.* Nature Neuroscience 10:1241–1242. https://pmc.ncbi.nlm.nih.gov/articles/PMC2039924/. Figs. 1–2.
- Ambrose, Pfeiffer, Foster (2016). *Reverse replay of hippocampal place cells is uniquely modulated by changing reward.* Neuron 91:1124–1136. https://pmc.ncbi.nlm.nih.gov/articles/PMC6013068. Summary, Figs. 3–6, Discussion.
- Bhattarai, Lee, Jung (2019/2020). *Distinct effects of reward and navigation history on hippocampal forward and reverse replays.* PNAS 117:689–697. https://pmc.ncbi.nlm.nih.gov/articles/PMC6955321. Figs. 4–6.
- Gupta, van der Meer, Touretzky, Redish (2010). *Hippocampal replay is not a simple function of experience.* Neuron 65:695–705. https://pmc.ncbi.nlm.nih.gov/articles/PMC4460981. Summary, Figs. 2–6.
- Wu, Haggerty, Kemere, Ji (2017). *Hippocampal awake replay in fear memory retrieval.* Nature Neuroscience. https://pmc.ncbi.nlm.nih.gov/articles/PMC5373994. Figs. 2–3.
- Jadhav, Kemere, German, Frank (2012). *Awake hippocampal sharp-wave ripples support spatial memory.* Science 336:1454–1458. https://pmc.ncbi.nlm.nih.gov/articles/PMC4441285/. Figs. 1–2.
- Mattar, Daw (2018). *Prioritized memory access explains planning and hippocampal replay.* Nature Neuroscience 21:1609–1617. https://pmc.ncbi.nlm.nih.gov/articles/PMC6203620. Figs. 2–5, Discussion.
- Liu, Mattar, Behrens, Daw, Dolan (2021). *Experience replay is associated with efficient non-local learning.* Science 372:eabf1357. https://pmc.ncbi.nlm.nih.gov/articles/PMC7610948. Abstract, Figs. 5–6, Table S2.

**Q3**
- Goupy, Tirilly, Bilasco (2025; rev. 2026). *Frozen Backpropagation: Relaxing Weight Symmetry in Deep Spiking Neural Networks.* arXiv preprint. https://arxiv.org/abs/2505.13741 ; https://arxiv.org/html/2505.13741v2. §2.2, §4.2.2, Tables 1–2, App. A.6.
- Folchini, Cossu, Ceni, Lomonaco, Bacciu, Gallicchio (2025). *Direct Feedback Alignment for Recurrent Neural Networks.* CADL 2025 workshop. https://zenodo.org/records/18459510. Record page only.
- Anon, Emon (2026). *Adaptive Speech-to-Spike Encoding for Spiking Neural Networks.* Interspeech 2026 (per arXiv). https://arxiv.org/abs/2606.19039. Abstract.
- Lin, Lu, Bal, Sengupta (2024). *Benchmarking Spiking Neural Network Learning Methods with Varying Locality.* arXiv preprint. https://arxiv.org/html/2402.01782v1. Table II.
- Yang, Li (2022). *A Computational Framework of Cortical Microcircuits Approximates Sign-concordant Random Backpropagation.* arXiv preprint. https://arxiv.org/abs/2205.07292. Abstract.
- Barretto-Bittar, Levina, Giannakakis, Zeraati (2026). *Diffusion of Neuromodulators for Temporal Credit Assignment.* arXiv preprint. https://arxiv.org/abs/2603.08949. Abstract.
- Hao, Wan, Zhai (2026). *What Accuracy and Gradient Cosine Miss: Evaluating Feedback Alignment ...* arXiv preprint. https://arxiv.org/abs/2606.21126. Abstract.
- Dong (2026). *A Geometric Analysis of Sign-Magnitude Asymmetry in a ReLU + RMSNorm Block under Ternary Quantization.* arXiv preprint. https://arxiv.org/abs/2605.18933. Abstract; not relevant.
- Berriche, Adjal, Baghdadi (2025; **withdrawn** 2026). *Sign-Symmetry Learning Rules are Robust Fine-Tuners.* arXiv. https://arxiv.org/abs/2502.05925. Comments field.

**Q4**
- Halvagal, Zenke (2023). *The combination of Hebbian and predictive plasticity learns invariant object representations in deep sensory networks.* Nature Neuroscience 26:1906–1915. https://www.nature.com/articles/s41593-023-01460-y.pdf. Fig. 3, Table 1.
- Fang, Aronov, Abbott, Mackevicius (2023). *Neural learning rules for generating flexible predictions and computing the successor representation.* eLife 12:e80680. https://pmc.ncbi.nlm.nih.gov/articles/PMC10019889/. Eq. 4, Figs. 2–3, 6.
- Millidge, Tang, Osanlouy, Harper, Bogacz (2024). *Predictive coding networks for temporal prediction.* PLOS Computational Biology 20(4):e1011183. https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011183.
- Moon, Zaheer, Lee, Kim, Park (2026). *Greedy Local Learning for Language Model Pretraining: Gaps and Objective Design.* arXiv 2610.04867v2, CODEC-FM workshop. https://arxiv.org/html/2610.04867. §4.4, App. H, §5.
