# Source check: learned feedback weights (13 November 2026)

**Brief:** `OPUS_TASK_SOURCE_CHECK_LEARNED_FEEDBACK_20261113.md` (default branch, 02897c6e). Read-only: no code, no runs.

**Status marks:**
- **confirmed**: the source says it.
- **corrected**: the claim on file is wrong, or narrower than the source.
- **not found**: I could not find it in an opened source.

## Sources opened in this session

| Key | Source | How opened |
|---|---|---|
| **AK** | Akrout, Wilson, Humphreys, Lillicrap, Tweed, "Deep Learning without Weight Transport", NeurIPS 2019, arXiv:1904.05391v5 | full PDF, all pages, appendices A–E included |
| **KN** | Kunin, Nayebi, Sagastuy-Brena, Ganguli, Bloom, Yamins, "Two Routes to Scalable Credit Assignment without Weight Symmetry", ICML 2020, arXiv:2003.01513v2 | full PDF |
| **LA** | Lansdell, Prakash, Kording, "Learning to solve the credit assignment problem", ICLR 2020, arXiv:1906.00889v3 | full PDF |
| **LN** | Launay, Poli, Boniface, Krzakala, "Direct Feedback Alignment Scales to Modern Deep Learning Tasks and Architectures", NeurIPS 2020, arXiv:2006.12878v1 | full PDF |
| **MU** | Murray, "Local online learning in recurrent networks with random feedback" (RFLO), bioRxiv 458570v1 | full text, first 100,000 characters (the last 3,008 were not read); the eLife page did not load |
| **JB** | Ji-An and Benna, "Deep Learning without Weight Symmetry", arXiv:2405.20594 | abstract page only |
| **ON FILE** | the default branch's review file: 9.17 (Codex's second sweep), 9.82 and its addendum, 9.101 and its addendum; the audit session's note, `REVIEW_AND_NEXT_STEPS_NOTE_20261111.md` §4.4 (`claude/vibrant-cerf-sje726`, 78c099fa) | read in the repository |

**Not opened:**
- **Kolen and Pollack (1994)**, IEEE ICNN vol. 3, pp. 1375–1380, DOI 10.1109/ICNN.1994.374486. No free copy was found, and the IEEE page returned no content.
- **Sign symmetry** (Liao et al. 2016; Xiao et al. 2018), **Moskovitz et al.**, **Guerguiev et al.**, and **e-prop** (Bellec et al.). For e-prop only the abstract page loaded, and it does not describe the feedback.

Nothing below rests on any of these.

## 1. The mechanisms

**Kolen–Pollack (KP), as AK states it (§5.1, eqs. 13–14).**
- The forward synapse W and the feedback synapse B "undergo identical adjustments A(t) and apply identical weight-decay factors λ".
- So W − B shrinks by (1 − λ) per step and the two converge.
- The original was not opened. Its content is known here only through AK §5.1, and through KN App. D.3, which says the 1994 constraint is that the weight update itself is transported.
- *Note §4.4's description ("the forward and the feedback synapse receive the same adjustment and the same decay, and so converge"): **confirmed** against AK §5.1, **not found** against the original.*

**Modified KP, AK §5.2.**
- Eq. 15: ΔB_{l+1} = −η y_l δ_{l+1}ᵀ. Eq. 16: ΔW_{l+1} = −η_W δ_{l+1} y_lᵀ − λW_{l+1}. Eq. 17: ΔB_{l+1} = −η_W y_l δ_{l+1}ᵀ − λB_{l+1}.
- Each feedback synapse uses its own presynaptic activity, the matching forward cell's activity y_l, and the teaching signal δ_{l+1} of its own cell. No weight or weight change is sent.
- The backward pathway never reads W: δ_l = φ′(y_l) B_{l+1} δ_{l+1} (eq. 3; Algorithm 2, line 10).
- It needs:
  - one-to-one cross-projections between forward and feedback cells (Fig. 2; App. C.1);
  - the same learning rate and decay in both paths. The text says "If learning rates and weight decay agree" before eq. 16; Algorithm 2, line 9, writes η_B.
- **Confirmed.**

**Weight mirror (WM), AK §4.**
- There is a separate mirror mode. Noise drives one layer, y_l ~ N(0, σ²) (Alg. 1, line 12), and the feedback cells copy the forward activities: δ_l = y_l, δ_{l+1} = y_{l+1} (eq. 6).
- Then B is updated by a Hebbian rule: ΔB_{l+1} = η_B δ_l δ_{l+1}ᵀ (eq. 7), with decay λ_WM B (eq. 21).
- In expectation this is η_B σ² φ′ Wᵀ (eqs. 8–11), so B tracks a scalar multiple of Wᵀ.
- W is not read. Its effect on y_{l+1} is what is sampled.
- On ImageNet the network spent its first two epochs in mirror mode, then mirrored a little after every minibatch (App. D.2: η_B 0.1, λ_WM 0.5).
- **Confirmed.**

**Later variants found and opened.**
- **Information Alignment** (KN §4, Table 1):
  - It is local: ΔB is built from the "amp" (−x_l x_{l+1}ᵀ), "decay" (B) and "null" (B x_{l+1} x_{l+1}ᵀ) primitives, an Oja-style stabilised WM.
  - KN's "Symmetric Alignment" uses the "self" primitive, whose gradient is −Wᵀ (Table 1). KN calls it non-local because "the backward weights can exactly measure the strengths of their forward counterparts" (§3.1). It reads forward weights, so it is outside this project's line.
- **Node perturbation** (LA §2.3–2.4, eqs. 1–3):
  - Noise ξ is injected into a layer and the loss is measured twice, giving λ̂ = (L̃ − L)ξ/c_h.
  - B is fitted by least squares, by gradient descent once per minibatch, to predict λ̂ from the error signal (eq. 3).
  - B needs the noise, the error from above, and the broadcast loss difference. No forward weight is read.
  - B's initial value: **not found** in the parts I read.
- **Not used:** JB (Product Feedback Alignment), which mentions KP only in feedforward convolutional networks, abstract only.

## 2. Evidence

**AK §6** (ImageNet; ResNet-18 and ResNet-50, BatchNorm after ReLU; top-1 test error, standard errors in parentheses):

| | Backprop | WM | KP | Sign symmetry | Feedback alignment |
|---|---|---|---|---|---|
| ResNet-18 | 30.1(4) | 30.2(7) | 29.2(4) | 39.2(4) | 97.4(2) |
| ResNet-50 | 22.9(4) | 23.4(5) | 23.9(7) | 33.8(3) | 98.9(1) |

- With BatchNorm before ReLU, sign symmetry reached 37.8(4) and 32.6(6). AK says the change "made little difference" to the other four methods.
- Matrix angle to Wᵀ: KP near 0° within 20 epochs; WM < 12° in all layers (ResNet-50, Fig. 4a, c).
- The number of runs behind the standard errors is **not found**.
- Note §4.4, "both outperform feedback alignment and sign symmetry and come close to backprop": **confirmed** (§6, Fig. 3; Abstract).

**Replication, KN.**
- **WM did not replicate at the reported level.** Reimplemented with AK's prescribed metaparameters, it reached 63.5% top-1 accuracy on ResNet-18 against AK's 69.73%. The best of 824 searched settings reached 64.07% (§4, Table 3).
- **WM failed entirely on other architectures** with those metaparameters (Fig. 3).
- **KP**, with AK's metaparameters, "matched backpropagation performance for ResNet-18 and ResNet-50", but a "considerable performance gap" emerged for ResNet-50v2, 101v2 and 152v2 (App. D.3, Fig. S5).
- **Information Alignment** reached 67.93% on ResNet-18 against backprop's 70.06% (Table 3), with the gap widening with depth (Fig. 3).
- Seeds in KN: **not found**.
- **Status:** WM **corrected**. It is not replicated at the level AK reports and is fragile across architectures. KP: **confirmed** for the two networks AK used, **corrected** for deeper and v2 networks.

**LA §4.**
- MNIST (784-50-20-10): five runs (Fig. 2).
- MNIST autoencoder: 10 runs (Fig. 3).
- CIFAR CNN, mean of 5 runs (Table 1):

| | Backprop | Node perturbation | DFA |
|---|---|---|---|
| CIFAR-10 test accuracy | 77 | 75 | 72 |
| CIFAR-100 test accuracy | 51 | 48 | 47 |

## 3. Through time

**No study was found that applies KP, WM, Information Alignment or node-perturbation feedback learning to recurrent weights or through time.**

**How I searched:**
- Searches on 13 November: "Kolen-Pollack OR weight mirror recurrent … backpropagation through time"; "learned feedback weights recurrent network Kolen-Pollack BPTT OR e-prop"; "Kolen-Pollack LSTM OR GRU OR recurrent neural network".
- I also read every opened source for recurrence.
- AK, KN and LA test only feedforward and convolutional networks. AK §7 lists "non-convolutional networks and other architectures" as untested.

**The nearest recurrent work, MU (RFLO):**
- Its feedback matrix B is "fixed and random" (Results, "The RFLO learning rule"). It is never adapted, and no learned feedback is tested.
- It does test symmetric feedback as a control (Fig. 2d). It attributes most of RFLO's gap to BPTT at long horizons to the locality approximation, not to the random feedback.

*Note §4.4, "my search found no use of either mechanism on recurrent weights or through time": **confirmed**, with the caveat that a search can miss work.*

## 4. Against sign feedback

**AK is the one opened source that trains sign symmetry beside learned feedback on the same task.**
- On ImageNet top-1 error, KP and WM beat sign symmetry by 9.0–10.0 points on ResNet-18 (29.2 and 30.2 against 39.2) and 9.9–10.4 points on ResNet-50 (23.9 and 23.4 against 33.8).
- With sign symmetry's better BatchNorm placement (37.8 and 32.6), the gap is 7.6–9.2 points.
- Sign symmetry's matrix angles stayed between about 35° and 80° by layer (Fig. 4e).
- **Confirmed** (AK §6, Fig. 3, Fig. 4e–f, App. D.3).

**Other sources:**
- LA reports the sign congruence of B with W (Fig. 2C) but trains no sign-symmetry arm.
- KN trains no sign-symmetry arm.

**Caution for this project:**
- AK's sign arm is a feedforward ImageNet network with BatchNorm, tuned only in learning rate (App. D.1).
- This project's rule uses weight signs with the mean magnitude, through time.
- So the 8–10-point gap is not a forecast for a GRU.

## 5. Unpredictable targets and next-token objectives

**No opened source reports a feedback-approximation method failing when most targets are unpredictable.**

**The only next-token result found is LN Table 5.**
- Set-up: a 6-layer Transformer on WikiText-103, scored by validation perplexity after 20 epochs.
- Backprop: 34.4, and 29.8 with β₂ = 0.999.
- Direct feedback alignment: 52.0 at best in the "macro" placement and 93.3 in "micro".
- A shallow baseline: 428.
- The feedback there is random and fixed, not learned.
- The input embedding still receives backprop gradients, "some amount of weight transport even in the micro case" (§ Transformers).
- **No learned-feedback method under a next-token objective was found.**

## 6. Fit to the project's line

**The line:**
- (1) a backward sweep with weight signs, with exact weight transport outside the line (9.82 addendum);
- (4) a separate feedback pathway that learns by a local rule and never copies a forward weight is inside it (9.101 addendum).

| Mechanism | Copies a weight? | At initialisation? | What a hostile reader would say |
|---|---|---|---|
| **Modified KP** (AK §5.2) | No weight or weight change is sent. B's update is formed from y_l and δ_{l+1} at the feedback synapse. | No: B starts random (AK §1, "initially random feedback weights"). | The update to B is by construction the transpose of the update to W, with matched rate and decay, so B converges to Wᵀ (angle near 0°, AK Fig. 4c). KN App. D.3 calls the exact-update constraint "clearly as biologically suspect as exact weight symmetry", though noisy updates still trained (Fig. S1). It needs a one-to-one forward–feedback cell pairing. In AK's experiments BatchNorm's σ was passed from the forward to the feedback path (App. D.3), one scalar per feedback neuron: a copied parameter, not a weight. |
| **WM** (AK §4) | No. B learns from noise-driven activity; W's effect is sampled, W is not read. | No: random. | It needs a separate offline mode with injected noise and feedback cells forced to the forward activities. The same σ transfer as KP (App. D.3). It did not replicate at the reported level and failed on other architectures (KN §4). |
| **Information Alignment** (KN §4) | No; local primitives only. | **Not found.** | It is a stabilised WM, still below backprop and worse with depth (KN Fig. 3). |
| **Node perturbation** (LA) | No. | **Not found.** | It needs a second, noisy forward pass, a broadcast scalar loss difference, and a regression per layer. |
| **KN's Symmetric and Activation Alignment** | Symmetric reads W through the "self" primitive. Activation Alignment uses the "sparse" primitive, which KN also calls non-local. | n/a | Outside the line. Not a candidate. |

*Note §4.4 and 9.101's addendum, that the two named mechanisms never copy a weight: **confirmed** for KP and WM as written in AK, with the σ transfer above as the one exception in AK's own experiments.*

**On file:** Codex's sweep found "adaptive versus fixed feedback 96–97% vs 97%" on a small proxy, "a caution for the learned-feedback arm" (review 9.17). That result is the project's own, not a source.

## 7. Cost for a GRU of 256 units with a 512-wide input

My arithmetic, from AK App. B's count. Nothing was measured.

**Size:**
- A GRU has three gates, each with an input matrix (256 × 512) and a recurrent matrix (256 × 256): 3 × (131,072 + 65,536) = **589,824 forward weights**, plus biases.
- A learned pathway needs one B for every matrix the backward sweep crosses.
- The three recurrent B's are 196,608 parameters. All six, if credit also passes to the input, are **589,824**: the GRU's weight count doubled.
- The sign rule needs none.

**Work per update:**
- The backward pass costs the same as now: a product with B instead of with the signs.
- **KP** adds one outer-product accumulation per matrix, of the same size as the forward weights' own (2·n_in·n_out per item and step: 1.18 M multiply-adds per token for all six). It also adds the decay and the write of B. AK puts this at n + 4·n_l·n_{l+1} per example per layer.
- **WM** adds the same update, plus a noisy forward pass per mirrored matrix (about 2·n_l·n_{l+1}, AK App. B) and its own offline schedule.
- **Node perturbation** adds a second forward pass per minibatch and a regression per layer.

The memo decides nothing.
