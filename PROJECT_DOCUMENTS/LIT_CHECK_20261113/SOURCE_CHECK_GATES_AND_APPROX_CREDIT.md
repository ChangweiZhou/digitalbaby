# Source check: approximate credit, undecided gates and composition

**Brief:** `OPUS_TASK_SOURCE_CHECK_GATES_AND_APPROX_CREDIT_20261113.md` (default branch, commit 7287e071). Read-only literature job, done 14 November 2026. Nothing trained.

**Method.** I opened every primary source listed below myself: the PDF, its figures and, where one exists, the supplement. Figure values were read from the rendered page. Where a bar carries a printed number, that number is given. Where the value is read off a curve, it is marked "read from curve" with its precision.

**Searches.** No person's name, repository name or project detail was put into any search or outside request. Searches used paper titles, author names and technical terms only, for example:
- "Kolen-Pollack learned feedback weights recurrent neural network";
- "learned feedback weights recurrent neural network weight transport local learning rule";
- "target propagation through time trained feedback recurrent network";
- "recurrent network learned error feedback synapses temporal neural coding network";
- "sign symmetry feedback recurrent neural network LSTM GRU feedback alignment gating saturation";
- "feedback alignment LSTM GRU recurrent network gates comparison backpropagation through time";
- "approximate gradient learning rule systematic generalization compositional recurrent network e-prop RFLO";
- "diagonal linear recurrent unit state space model compositional generalization untaught combinations";
- exact-title searches for the e-prop, KeRNL and "Beyond accuracy" papers.

**Status of this note.** It is one reader's account, the same model family as the earlier audit, and it is not an independent review. It decides no design.

---

## Part A. The three points to settle first

### A1. The study cited as "no gain over random feedback": every arm's numbers

**The study.** Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein, Maass, "A solution to the learning dilemma for recurrent networks of spiking neurons", *Nature Communications* 11:3625 (2020), doi:10.1038/s41467-020-17236-y. Peer reviewed.
- Main text read from the bioRxiv v4 PDF: https://www.biorxiv.org/content/10.1101/738385v4.full.pdf
- Supplement read from the publisher's file: https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-17236-y/MediaObjects/41467_2020_17236_MOESM1_ESM.pdf

**What "adaptive e-prop" learns.** It is a Kolen–Pollack-type rule, but only on the **output-to-neuron broadcast weights B_jk**. It does not touch any recurrent backward path through time; e-prop has none, since credit through time is carried by eligibility traces. Supplement, "Adaptive e-prop and weight decay regularization":

> "This was implemented by applying an identical weight update to the broadcast and the output weights, i.e. ∆B_jk = ∆W^out_kj and by subtracting c_decay · W^out_kj (resp. c_decay · B_jk) from the weight W^out_kj (resp. B_jk) after each weight update … The remaining difference of performance between symmetric and adaptive e-prop reported in Supplementary Figure 2 and Supplementary Figure 4 may be explained by the different initializations."

The decay values are c_decay = 1e-2 (framewise TIMIT) and 1e-4 (sequence TIMIT).

**The comparison figures.** These are Supplementary Figs 2 and 4. Every bar carries its value in print. Error in %, lower is better.

| Network (figure) | Measure | Symmetric e-prop | **Adaptive (KP-type)** | **Random** | BPTT |
|---|---|---|---|---|---|
| LSNN, recurrent spiking (Suppl. Fig. 2, left) | TIMIT framewise error | 34.6 | **35.4** | **36.9** | 32.9 |
| LSNN (Suppl. Fig. 2, right) | TIMIT sequence error (CTC) | 26.4 | **27.1** | **34.7** | 24.7 |
| LSTM, rate, gated (Suppl. Fig. 4, left) | TIMIT framewise error | 26.2 | **26.9** | **26.5** | 25.9 |
| LSTM (Suppl. Fig. 4, right) | TIMIT sequence error | 23.8 | **25.4** | **26.2** | 20.0 |

**Reading.**
- Adaptive is better than random in three of the four comparisons:
  - by 1.5 points (LSNN, framewise);
  - by **7.6 points** (LSNN, sequence);
  - by 0.8 points (LSTM, sequence).
- It is 0.4 points worse in the fourth (LSTM, framewise).
- The main text's Fig. 2 caption says only "random and adaptive e-prop produced similar results, see Fig. S2". For the LSNN sequence task, the supplement's own bars contradict that.
- **"No gain over random feedback" is therefore not supported as a general statement.** It holds for LSTM framewise and is clearly false for LSNN sequence recognition.

**Other figures.**
- Suppl. Fig. 1b (the episodic-memory movie task) shows all four learning curves on top of one another, at correlation ≈0.97–0.98 at 5,000 iterations. There are no separate values; the curves overlap to within the line width.
- Suppl. Fig. 8 (the blocking control) has no adaptive arm.

**Seeds.** The TIMIT figures carry no error bars. The supplement sections I read state no run count for TIMIT, so I record it as **not stated**. For contrast, Fig. 3 states 20 runs and the Atari experiments 5 runs.

### A2. Marschall et al.: the result for each rule and task

**The study.** Marschall, Cho, Savin, "A unified framework of online learning algorithms for training recurrent neural networks", *JMLR* 21(135):1–34 (2020). Peer reviewed. https://jmlr.org/papers/volume21/19-562/19-562.pdf

**Setup.**
- **Network:** a 32-unit rate RNN (Table 2), **not gated**, with leak α = 1 (discrete) or α = 0.5.
- **Training:** batch size 1, online, plain gradient descent at lr 1e-4. Hyperparameters were tuned by hand per algorithm.
- **Rules compared:**
  - exact: RTRL and F-BPTT (truncated BPTT, T = 10);
  - approximations: UORO, KF-RTRL, R-KF, RFLO, KeRNL and DNI;
  - DNI(b), a local variant: fixed random output feedback W_fb, and the Jacobian J learned by the local "perceptron" rule ∆J_ij ∝ −(a_i^(t) − Σ_j' J_ij' a_j'^(t−1)) a_j^(t−1);
  - fixed W, a readout-only baseline.

**Fig. 4a, b and 5a, b.** Training loss after 1M online steps. Values are **read from the curves, ±0.005**. Lines are means over 24 seeds, shaded ±1 SEM.

| Rule | Add, α=1 (Fig 4a; optimum ≈0.455) | Add, α=0.5 (Fig 4b) | Mimic, α=1 (Fig 5a) | Mimic, α=0.5 (Fig 5b) |
|---|---|---|---|---|
| RTRL | ≈0.47 | ≈0.49 | ≈0.019 | ≈0.005 |
| F-BPTT (T=10) | ≈0.47 | ≈0.49 | ≈0.019 | ≈0.005 |
| KF-RTRL | ≈0.47 | ≈0.49 | ≈0.020 | ≈0.005 |
| R-KF | ≈0.475 | ≈0.49 | ≈0.034 | ≈0.010 |
| UORO | ≈0.475 | ≈0.49 | ≈0.031 | ≈0.009 |
| DNI | ≈0.50 | ≈0.50 | ≈0.019 | ≈0.005 |
| RFLO (fixed random feedback) | ≈0.50 | ≈0.52 | ≈0.022 | ≈0.006 |
| KeRNL (learned sensitivity A and rates α) | ≈0.50 | ≈0.52 | ≈0.023 | ≈0.008 |
| DNI(b) (learned J by local rule) | ≈0.515 | ≈0.51 | ≈0.020 | ≈0.007 |
| fixed W | ≈0.54 | ≈0.57 | ≈0.077 | ≈0.025 |

**Fig. 4c, d.** Add-task test loss against difficulty t1, with 12 seeds per point.
- The exact methods (RTRL, F-BPTT, KF-RTRL) stay near the optimum until t1 ≈ 13–14 at α = 1, then reach the baseline of 0.66 at t1 = 19.
- R-KF and UORO rise sharply at t1 ≈ 8–10.
- DNI rises at t1 ≈ 8–11.
- RFLO, KeRNL, DNI(b) and fixed W leave the optimum by t1 ≈ 4–6 and reach the baseline by t1 ≈ 10.
- All values above are read from the curves.

**Fig. 5c, d.** Mimic, test loss against target-network size from 8 to 256. Here RFLO, KeRNL and DNI stay close to the exact methods. The paper explains this by the task's short time horizon.

**Key sentences.**
- On learned against fixed sensitivity:
  > "Despite significant effort in exploring this space, we are not able to get KeRNL to perform better than RFLO, suggesting that the procedure for training A_ki and α_i does more harm than good in our setup (see Fig. S1 …)."
  - Fig. S1 shows that KeRNL's test loss equals RFLO's at learning rates ≤ 1 and rises above it at larger rates, in all four task/α combinations.
- On alignment and performance (§5.4, observation iv):
  > "UORO and R-KF each have mean alignments with RTRL that are barely above 0 (Fig. 6b), yet they outperform other algorithms with much higher average alignment, in particular RFLO, KeRNL, and DNI … angular alignment with exact algorithms is not predictive of learning performance."
- On RFLO (§5.5): it "works by training what is essentially a high-capacity feedforward model to predict labels from the natural memory traces of previous inputs contained in the hidden state", rather than "learning dynamics that actively retain useful bits of the past like RTRL and BPTT."

**Gates.** None. The paper says a full comparison "would likely require a complex architecture (LSTM/GRU)" and leaves gated networks out.

### A3. Every study found that learns a recurrent network's backward weights by a local rule

**Count: five confirmed from primary text.** This agrees with the review's "at least five". The earlier audit counted three, one of which, the adaptive e-prop study, is mischaracterised as shown in A1.

| # | Study (peer review) | What is learned, and by which local rule | Network, task | Compared with fixed feedback on the same task? Every arm |
|---|---|---|---|---|
| 1 | Bellec et al. 2020, *Nat. Commun.* (yes) | Output→neuron broadcast B, by a mirror of the readout update plus weight decay (KP-type). It is not a path through time. | LSNN (spiking) and LSTM (rate, gated); TIMIT | **Yes.** See the A1 table: adaptive 35.4/27.1 vs random 36.9/34.7 (LSNN); 26.9/25.4 vs 26.5/26.2 (LSTM). Seeds not stated. |
| 2 | Roth, Kanitscheider, Fiete 2019, *ICLR* (yes) | Sensitivity weights β (the backward map from past to present hidden state) and kernel rates γ, by node perturbation: they are trained to predict the effect of injected noise ξ. | tanh RNN; adding problem, length 200 and 400; also IRNN, sequential MNIST, and LSTM on A^nB^n | **Yes, fixed against learned** (Table 2; MSE, lower is better), lengths 200 / 400: fixed β, fixed γ 0.014 / 0.180; learn β, fix γ 0.170 / 0.171; fix β, learn γ 0.031 / 0.076; learn both 0.008 / 0.031; BPTT 0.020 / 0.171. Seed count not stated (Fig. 2 is "single-trial"). A^nB^n bit-loss: KeRNL 0.149, UORO 0.147, 16-step BPTT 0.144 (Table 5). Appendix C: "KeRNL did not perform well on next word prediction on the Penn Treebank dataset." |
| 3 | Marschall, Cho, Savin 2020, *JMLR* (yes) | (a) KeRNL's A and α, by perturbation. (b) DNI(b)'s Jacobian J, by a local perceptron rule, with fixed random output feedback. | Rate RNN, 32 units; Add and Mimic | **Yes.** KeRNL against RFLO (fixed random): equal or worse (A2, Fig. S1). DNI(b) against DNI (exact J): DNI(b) worse on Add α = 1 (≈0.515 vs ≈0.50) and about equal on Mimic. 24 seeds. |
| 4 | Manchev & Spratling 2020, *JMLR* 21(152) "Target propagation in recurrent neural networks" (yes) | Approximate inverse G of the recurrent step, trained by a local reconstruction (difference target propagation) loss. | Orthogonal-initialised simple RNN; temporal order, 3-bit temporal order, adding, random permutation; sequential MNIST | **No fixed-feedback arm.** Against BPTT only. Table 1, maximum length solved, BPTT/TPTT: 120/150, 70/150, 90/60, 70/300. Sequential MNIST (Table 6): TPTT 54.2 %, LSTM-BPTT 65.68 %, IRNN 97.2 %. Generalisation to length 200 (Table 4): TPTT higher on all four tasks. Runs not stated. |
| 5 | Ororbia, Mali, Giles, Kifer 2020, *IEEE TNNLS* "Continual learning of recurrent neural networks by locally aligning distributed representations" (P-TNCN; yes; arXiv v4 read) | Error synapses E_1, E_2, by a temporal-difference rule, ∆E_1 = (e^1_{t,z} − e^1_{t−1,z})(e^0)^T. | Two-layer predictive recurrent network; bouncing MNIST, NotMNIST, Fashion MNIST, text | **Only in unreported preliminary experiments**, and against a transpose-mirror rule, not random: "this temporal difference rule improved generalization consistently compared to the approximate transpose rule". Table II (5 trials) compares with LSTM trained by BPTT, SAB or RTRL, not with fixed feedback. |

**Found but excluded,** because the backward weights are not learned by a local rule, or the network is not recurrent:
- Kaleb, Feulner, Gallego, Clopath 2024, *NeurIPS* (peer reviewed): "We pre-train W_in, W_h, W_fb, W_o, b_h and b_o using BPTT". The feedback is learned, but by BPTT. It is then used by an RFLO-type rule; 20 seeds.
- Bellec et al. 2019, arXiv 1901.09049 (not peer reviewed):
  - e-prop 2 trains its error module by BPTT in an outer loop;
  - e-prop 3 trains a synthetic-gradient module (bootstrapped) on LSTM.
  - Neither is a local rule.
- Manchev & Garcia-Peraza-Herrera 2025, arXiv 2504.13531 (not peer reviewed): an LRA RNN using the "LRA-diff" variant. No learned feedback was found in the text.
- Bacvanski, Ziyin, Poggio, arXiv 2510.18808: KP in continuous time, feedforward networks only.
- Murray 2019, *eLife* (RFLO): the feedback is fixed and random.

---

## Part B. The review

### Table: one row per study

Network type: R = rate, S = spiking, G = gated. "PR" = peer reviewed.

| Study | URL opened | Exact sentence / table / figure | Network | Task | Seeds | PR | Bears on question |
|---|---|---|---|---|---|---|---|
| Bellec et al. 2020 | supplement (above) | Suppl. Figs 2 and 4 (A1 table) | S (LSNN); R, G (LSTM) | TIMIT framewise and sequence | not stated | yes | Q3 (learned against random broadcast); Q1 (on LSTM, every e-prop arm is worse than BPTT on sequence error: 23.8–26.2 vs 20.0) |
| Marschall, Cho, Savin 2020 | jmlr.org/papers/volume21/19-562/19-562.pdf | Figs 4, 5, 6 and S1; §5.4 (iv); §5.5 (A2) | R, not gated, 32 units | Add (t1 up to 19), Mimic | 24 (curves); 12 (difficulty) | yes | Q1, Q3, Q5; alignment does not predict performance |
| Roth, Kanitscheider, Fiete 2019 (KeRNL) | openreview.net/pdf?id=BJgPCiA5Ym (blocked to scripts); text read from the ICLR text mirror object.cloud.sdsc.edu/…/kernel_rnn_learning_kernl__9d5e3aff.txt | Table 2; Table 5; App. C | R (tanh, IRNN); R, G (LSTM, A^nB^n) | Adding (200, 400), sequential MNIST, A^nB^n, PTB | not stated | yes | Q3, Q5 |
| Manchev & Spratling 2020 (TPTT) | jmlr.org/papers/volume21/18-141/18-141.pdf | Tables 1, 4, 6, 7 | R, not gated | Temporal order, adding, permutation, sequential MNIST | not stated | yes | Q3 (no fixed arm) |
| Ororbia et al. 2020 (P-TNCN) | arxiv.org/pdf/1810.07411v4 | Eq. 21; Table II | R, predictive coding, not gated | Sequential bouncing MNIST, NotMNIST, Fashion MNIST | 5 | yes (TNNLS) | Q3 |
| Liu, Ghosh, Richards, Shea-Brown, Lajoie 2022 | arxiv.org/pdf/2206.00823v3 | Fig. 2: "Bio-plausible temporal credit assignment rules show worse and more variable generalization gap, which can be informed by loss landscape curvature." §3.2: "these bio-plausible rules achieve worse and more variable generalization performance than their machine learning counterpart (BPTT)." | R, not gated | Pattern generation, sequential MNIST, delayed match-to-sample. Rules: symmetric e-prop, RFLO, MDGL. | several initialisations (histograms); 5 runs for the curvature curves | yes (NeurIPS) | **Q1 (closest positive answer):** worse generalisation, linked to sharper minima. Not gated, and not composition. |
| Williams, Payeur, Lajoie 2026 | arxiv.org/abs/2606.00243 (abstract only) | Abstract: RFLO's learned solutions are "restricted to low-rank perturbations of the initial parameters" | R, linear | Analytical | — | ICML 2026 (per arXiv) | Q1, Q5. Not verified beyond the abstract. |
| Zucchet, Meier, Schug, Mujika, Sacramento 2023 | arxiv.org/pdf/2305.15947 | Table 1, copy task, final training loss (5 seeds). One-layer GRU: SnAp-1 3.16e-1 vs BPTT 2.61e-1. Four-layer GRU: 3.27e-1 vs 1.94e-1. Four-layer LRU: online exact 8.44e-3 vs BPTT 7.59e-6. Caption: "online learning of multilayer networks is difficult without element-wise recurrence." | R; G (GRU); linear diagonal (LRU) | Copy; LRA (sCIFAR, IMDB, ListOps; 3 seeds) | 5; 3 | yes (NeurIPS) | Q1 (an approximate rule on a GRU is worse than BPTT); Q4 |
| Irie, Gopalakrishnan, Schmidhuber 2024 | arxiv.org/pdf/2305.19044 | Element-wise LSTM (eLSTM) with exact RTRL, competitive on two DMLab memory tasks. Limits: "one can come up with a synthetic/algorithmic task where such models fail; see Appendix B.2". Tractability is restricted to one layer: "By stacking two such layers, the corresponding RTRL algorithm becomes intractable". | R, G, element-wise | DMLab-30, ProcGen, Atari | per paper (not checked) | yes (ICLR) | Q4, Q5 |
| Li, He, Tian, Chen, Qin, Wang, Liu 2018 (G2-LSTM) | arxiv.org/pdf/1806.02988 | Abstract: training "pushes the output values of the gates towards 0 or 1 … we achieve better or comparable performances due to its better generalization ability". Fig. 1: in a standard LSTM, "most of the values are in the middle state (around 0.5)". Table 2 (BLEU, baseline / sharpened sigmoid / G2): En→De 21.89 / 21.64 / 22.43; De→En 31.00 / 29.73 / 31.95. | R, G (LSTM) | PTB language model; IWSLT and WMT translation | not checked | yes (ICML) | Q2: Gumbel-softmax gates (stochastic, unit-local). BPTT only. |
| Gu, Gulcehre, Paine, Hoffman, Pascanu 2020 (UR gates) | arxiv.org/pdf/1910.09890 | Fig. 5: "activations near 1.0 are important for a model's ability to make progress or solve this task … UR … is able to learn extreme gate values". On Copy, only OR, UR, O- and C- solve it. | R, G (LSTM/GRU variants), 256 units | Copy (cue after a long delay), Adding (N = 2000), pixel images, language model, RL | 3 per setting (appendix) | yes (ICML) | Q2: uniform gate init and refine gate. BPTT only. |
| Ohno, Kanai, Ida 2023 (fast gate) | arxiv.org/pdf/2210.01348 | Abstract: "gradient vanishing of the gate function can be mitigated by accelerating the convergence of the saturating function"; outperforms prior methods "on benchmark tasks involving extremely long time scales". | R, G (LSTM) | Adding (5,000), Copy (500), sequential MNIST and CIFAR, language model | 3 | yes (AAAI 2023; arXiv read) | Q2. BPTT only. |
| Tallec & Ollivier 2018 (chrono) | arxiv.org/pdf/1804.11188 | Fig. 3: "Chrono initialization heavily outperforms standard initialization" on copy and variable copy, T = 500 and 1000 (except one case). Fig. 4: the same on adding, T = 200 and 750. | R, G (LSTM) | Copy, variable copy, adding, text8 | not checked | yes (ICLR) | Q2: a saturation-promoting initialisation. Its ranges are set from the known task length (see the earlier audit). |
| Vecoven, Ernst, Drion 2020/2021 (bistable cells) | arxiv.org/pdf/2006.05252 | Table 1, copy-first-input test MSE at T = 600: BRC 0.158, nBRC 0.0005, GRU 0.993, LSTM 0.999 (chance ≈ 1). "only networks made of bistable cells are capable of outperforming random guessing threshold". Recurrence is per neuron only. | R; bistable, gated | Copy first input (T = 5–600), denoising, sequential MNIST | 3 | journal version (*PLoS ONE* 2021) not opened; arXiv v1 read | Q2: unit-local bistability, 10+ steps up to 600. BPTT only. |
| Kaleb et al. 2024 | proceedings.neurips.cc/…/09236f27…-Paper-Conference.pdf | "We pre-train W_in, W_h, W_fb, W_o … using BPTT" | R, not gated | Centre-out reaching with perturbation | 20 | yes | Q3 (excluded, as above) |

### Answers to the five questions

1. **Approximate rules and gates, memory, generalisation.**
   - **Not found:** any primary report that sign symmetry, feedback alignment, DFA, e-prop, RFLO, KeRNL or target propagation leaves *gates less saturated* in a gated network.
   - **Found: worse generalisation,** in rate RNNs that are not gated (Liu et al. 2022, linked to sharper minima).
   - **Found: worse optimisation of gated nets.** On TIMIT LSTMs, every e-prop arm is worse than BPTT on sequence error (Bellec 2020, Suppl. Fig. 4). On copy, SnAp-1 on a GRU is worse than BPTT (Zucchet 2023, Table 1).
   - **Found: precision of memory over delays.** On Add, the approximate local rules lose the optimum at much shorter delays than exact rules (Marschall 2020, Fig. 4c).
   - **Found (Marschall):** RFLO learns no memory-retaining dynamics.
   - **Not found:** any test of systematic or compositional generalisation, such as untaught operator pairs, under any approximate rule.
2. **Unit-local mechanisms for decisive gates.** These are bistable cells, binary or stochastic gates (Gumbel), refine gates and uniform-range initialisation, fast-saturating gate functions, and chrono initialisation (see the rows above). Each is shown on memory tasks of 10+ steps (copy, adding, up to T = 600–5,000), and **every one was shown with BPTT only.** None was tested under an approximate credit rule.
3. **Learned backward weights in recurrent networks** (A3).
   - There are five studies. Three compare with fixed feedback on the same task: Bellec 2020, Roth 2019 and Marschall 2020.
   - The comparisons are mixed:
     - KP-type broadcast learning beats random in 3 of 4 TIMIT comparisons, once by 7.6 points.
     - Learning KeRNL's sensitivities alone is *worse* than fixing them. Learning sensitivities and rates together is best in Roth's own Table 2, but is not better than fixed random feedback (RFLO) in Marschall's replication.
     - A locally learned Jacobian (DNI(b)) is at most equal to the exact one.
   - No study learns the backward weights *of a gated recurrent unit's recurrent matrices through time* by a local rule. Bellec's LSTM arm learns only the output broadcast.
4. **Cheap, local exact credit with element-wise recurrence.**
   - Yes for learning: Zucchet 2023 (LRU, exact online, copy and LRA) and Irie 2024 (eLSTM, exact RTRL, DMLab).
   - **Not found:** any demonstration that such units compose learned operations on untaught combinations. A search for diagonal or linear recurrent units with compositional generalisation returned none.
   - Irie notes that exactness is lost when two such layers are stacked.
5. **Negative results.**
   - KeRNL on PTB fails (Roth 2019, App. C).
   - KeRNL's learned sensitivities do no better than RFLO in replication (Marschall 2020, Fig. S1).
   - Alignment with the exact gradient does not predict performance (Marschall 2020, §5.4).
   - RFLO's solutions are low-rank perturbations of the initial weights (Williams 2026, abstract only).
   - Element-wise recurrent nets have task classes they cannot solve, and stacked layers make RTRL intractable (Irie 2024).
   - TPTT is weaker than BPTT on adding (60 vs 90) and on sequential MNIST against LSTM (Manchev & Spratling 2020).

### Candidate remedies, at most five, ranked by the strength of evidence

The evidence for every remedy comes from BPTT-trained networks or from rules other than sign symmetry. None was shown under the rule in question.

1. **Push the update gate towards binary values.** Options: a Gumbel or sharpened gate (Li 2018), the fast gate (Ohno 2023), or the refine gate with uniform initialisation (Gu 2020).
   - *Evidence:* three peer-reviewed BPTT studies on gated nets show that near-binary gates help memory over long delays or improve generalisation.
   - *Cheapest test:* in arm S only, replace σ by the fast gate on z. Run one seed from the existing S birth through stage 3 (F2.1). Read the fraction of z in (0.1, 0.9) and the plus-minus score against the existing SS and SB cells at stage 3.
2. **Use exact credit only where it is local and cheap.** Back-propagate the diagonal (element-wise) part of the gate recurrence exactly and keep sign feedback elsewhere. The Zucchet and Irie principle applies to the parts of the GRU state that act unit by unit.
   - *Evidence:* exact online learning with element-wise recurrence is competitive on copy and LRA. Composition is not shown.
   - *Cheapest test:* in the step, use exact backward for the h ⊙ z path and the gate nonlinearities, which already run through autograd, but replace the sign map on W_hh's z-rows with the exact one. Run one seed, stage 3.
3. **Learn the backward matrices by a KP-type mirror with decay.**
   - *Evidence:* mixed. Positive on spiking TIMIT sequence (+7.6 over random). Nil on LSTM framewise. Learned sensitivities do not beat fixed ones in replication. It costs about 589,824 extra parameters, per the earlier audit.
   - *Cheapest test:* a KP mirror on W_hh only, with the update shared and decay 0.25. Run one seed, stage 3, against SS.
4. **A higher learning rate early, then decay.** This is Liu et al.'s suggested remedy against the sharp minima that approximate rules prefer.
   - *Evidence:* a single paper's appendix (preliminary), on rate RNNs that are not gated.
   - *Cheapest test:* arm S with twice the learning rate for the first stage of the life, then the normal rate. Run one seed, stage 3.
5. **Bistable units for the held state** (nBRC).
   - *Evidence:* strong for long memory under BPTT (copy-first-input at T = 600). It is an architecture change, untested under approximate credit.
   - *Cheapest test:* not cheap in this setting, because it is a new cell. Defer.

### Claims I could not verify from a primary source

- Bellec et al. 2020: the number of seeds behind the TIMIT bars in Suppl. Figs 2 and 4 is not stated in the sections I read. The values themselves are printed on the bars.
- Roth et al. 2019: read from a text mirror of the ICLR PDF, because OpenReview refused scripted access. The figures were not seen as images. The seed count for Table 2 is not stated.
- Vecoven et al.: the journal (*PLoS ONE*) version was not opened.
- Ohno et al.: the AAAI version was not opened; the arXiv v1 was read.
- Williams, Payeur, Lajoie 2026: abstract only.
- Liu et al. 2022: exact per-task seed counts were not extracted, beyond "five independent runs" for the curvature curves.
- Kolen & Pollack 1994: not opened, as in the earlier audit.
- Murray 2019 (RFLO), Akrout et al. 2019 and Kunin et al. 2020 were not reopened in this job.
- Marschall et al. 2019 (the conference precursor that introduced DNI(b)) was not opened.
- Whether **any** study reports gate-value distributions under an approximate credit rule: not found, after the searches listed at the top.
- Whether **any** study tests compositional or systematic generalisation under an approximate credit rule, or with element-wise recurrent units trained by exact local credit: not found.
