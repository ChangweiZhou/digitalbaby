# Source check: gate timescales, heterogeneous time constants, transmission delays (E29, alongside Part A)

13 November 2026. Opus 5.5 worker, for the brief `OPUS_TASK_E29_SLOW_UNITS_AND_DELAYS_20261113.md` (revision 2, a5cb6259). This page is read-only and **decides nothing**. Each source is marked by how it was read. **PDF** means I extracted and read the paper's own text. **Page** means I read the publisher's or arXiv's HTML page through a fetch tool that summarises it, so the details are second-hand and should be checked before anyone relies on them.

## 1. Initialising recurrent gates to a range of timescales

**Tallec, C. & Ollivier, Y. (2018). "Can recurrent neural networks warp time?" ICLR 2018; arXiv:1804.11188. Read from the PDF.**
- **Coordinator's recollection: confirmed.** The authors, year and topic are right. The venue is ICLR 2018, as the PDF header states.
- **What was changed.** Gate biases are initialised so that the units' memory ranges are spread. The general prescription is to set the gate bias to −log(U([T_min, T_max]) − 1). For the LSTM ("chrono initialization", eq. 16), b_f ~ log(U([1, T_max − 1])) and b_i = −b_f. The paper motivates gating as quasi-invariance to time warping and says the values are "initial values only", still learned. GRUs are covered by the theory and named in the abstract. **The experiments on chrono initialisation use LSTMs** (plus a deep RHN on Penn Treebank), not GRUs.
- **Tested on.** Synthetic warped and padded "previous character" tasks, which compare plain, leaky and gated RNNs, not chrono. Copy (T = 500, 2000) and variable copy (T = 500, 1000). Adding (T = 200, 750). Pixel MNIST and permuted MNIST. text8 (T_max = 8). Penn Treebank word level on an RHN (T_max = 11).
- **Against what.** The standard LSTM initialisation (forget bias 1, "Constant 1" in the figures) and a memoryless baseline on the synthetic tasks.
- **Seeds and runs.** On the warping tasks, "Each setup is run 5 times" (min, mean and max reported). On text8, "This same slight difference is observed on two independent runs". On PTB, "a single run". **For copy and adding I found no run count stated in the text; the figures show single curves.**
- **Results as stated.** Chrono "heavily outperforms" standard initialisation on the synthetic tasks, except variable copy at the smaller T, where both do well. On MNIST there is "no clear difference". On text8 the margin is small (1.37 vs 1.38 bits per character).
- **Credit.** Exact gradients: RMSprop on the synthetic tasks, Adam with gradient clipping on text8. Backpropagation through time is implied by standard training. The PDF does not state truncation for the synthetic tasks.

## 2. Heterogeneous time constants

**Perez-Nieves, N., Leung, V. C. H., Dragotti, P. L. & Goodman, D. F. M. (2021). "Neural heterogeneity promotes robust learning." *Nature Communications* 12, 5791. doi:10.1038/s41467-021-26022-3. Read from the PMC page (open access).**
- **What was changed.** Each neuron in a recurrent spiking network gets its own membrane and synaptic time constant, initialised from Γ(3, τ̄/3) with τ̄_m = 20 ms and τ̄_s = 10 ms. A log-normal variant is also tested. The time constants are either held fixed or trained, through the decay factors, clipped to 3Δt to 100 ms. Heterogeneity in threshold and reset had no appreciable effect.
- **Tested on.** N-MNIST, F-MNIST, DVS128 Gesture, SHD and SSC.
- **Against what.** Homogeneous time constants with weights-only training; on SHD these were grid-searched. Table 1 has rows for both kinds of initialisation and both training regimes. That is the summary's account; I did not see the table myself.
- **Result as stated.** On SHD, 71.7 ± 1.0 (homogeneous, standard training) vs 81.7 ± 0.8 (heterogeneous initialisation and training). The gains are largest on the tasks with the most temporal structure.
- **Seeds and runs.** Fig. 2A says "over 10 trials". The run count for Table 1 is not stated in what I read.
- **Credit.** **Not exact.** Surrogate-gradient backpropagation through time (sigmoid surrogate, steepness 100) with Adam.

## 3. Transmission delays in recurrent networks

**Lin, T., Horne, B. G., Tiňo, P. & Giles, C. L. "Learning long-term dependencies is not as difficult with NARX networks." NIPS 8 (1995), pp. 577–583. Journal version: "Learning long-term dependencies in NARX recurrent neural networks", IEEE Trans. Neural Networks 7(6):1329–1338, 1996. Read from the PDF of the NIPS version; the journal version was not opened.**
- **What was changed.** The network's own output is fed back through a tapped delay line of order D_y: y(t) = f(u(t), y(t−1), …, y(t−D_y)). The delays are **fixed**. The input order is zero, and the output is one-dimensional. The mechanism the authors argue for: when the network is unrolled for backpropagation through time, the delays become "jump-ahead connections" that shorten the gradient's path.
- **Tested on.** A latching problem, with T from 10 to 200 in steps of 2, and a parity problem with trailing noise of length 0 to 50.
- **Against what.** Output order D = 1, which is a conventional recurrent net, against higher orders. On parity, D = 1, 2 and 3 are compared at roughly equal weight counts (21, 21 and 19 weights).
- **Seeds and runs.** Latching: 50 simulations per T. Parity: 10 trials.
- **Result as stated.** On parity with noise lengths of 10 to 35, D = 1 falls to about 60% while D = 2 or 3 stays at about 80%. The abstract says information is retained "two to three times as long", and that NARX "do not circumvent the problem of long-term dependencies".
- **Credit.** Exact (backpropagation through time).

**Mészáros, B., Knight, J. C. & Nowotny, T. "Efficient event-based delay learning in spiking neural networks." arXiv:2501.07331 (2025); *Nature Communications* (Nov 2025), doi:10.1038/s41467-025-65394-8. Read from the arXiv HTML v3 page (first 100,000 of 114,000 characters).**
- **What was changed.** Per-synapse delays are learned together with the weights, in feedforward and **recurrent** spiking layers. The authors call it the first delay-learning method applied to recurrent SNNs. Neuron time constants are not learned.
- **Credit.** Gradients are **exact** in the EventProp (adjoint, event-based) sense for LIF neurons. The stated approximation is that delays are discretised to the timestep.
- **Tested on.** A sequence-detection toy task, Yin-Yang, SHD, SSC and Braille, against networks without delays, a "delay line" EventProp baseline and published delay methods. They report "Mean ± s.d. over 8 runs" for most results.
- **Result as stated.** On SHD, 93.24 ± 1.0 against the delay-line baseline's 93.5 ± 0.7 (not significant), with about 5× fewer parameters. "Recurrent delays are particularly beneficial in small networks". In what I read, there is no direct recurrent-without-delay number on SHD.

**Queant, A., Rançon, U., Cottereau, B. R. & Masquelier, T. "DelRec: learning delays in recurrent spiking neural networks." arXiv:2509.24852 v2 (Oct 2025), a preprint with no venue named. Read from the arXiv HTML page.**
- **What was changed.** One learned axonal delay per neuron on the recurrent connections (per-synapse delays are supported but not used), held in a circular buffer. Non-integer delays use a triangular kernel that narrows during training; delays are rounded at evaluation.
- **Tested on.** SSC (3 seeds), PS-MNIST (1 seed) and SHD (10 seeds; ablations with 3 seeds per point).
- **Against what.** A recurrent SNN with uniform delay 1, a recurrent SNN with **fixed random** recurrent delays, feedforward learned delays (DCLS), and published SNNs.
- **Result as stated.** SSC 82.58 ± 0.08 (recurrent delays only) against the previous best listed, 82.03 ± 0.25. **On SHD, the claim of state of the art is not borne out by its own Table 2**: 93.39–93.73, against DCLS at 93.77 and SE-adLIF at 93.79.
- **Credit.** **Not exact** (surrogate gradients).

**Hammouamri, I., Khalfaoui-Hassani, I. & Masquelier, T. "Learning delays in spiking neural networks using dilated convolutions with learnable spacings." ICLR 2024; arXiv:2306.17670. Read from search summaries only; not opened.**
- Delays are learned as kernel positions (DCLS) in **feedforward** SNNs, without recurrent connections. They report that fixed random delays help, but learned delays matter in sparse networks. **This is not a recurrent-delay result.** It is listed only because the two papers above compare against it.

## What this does and does not bear on (no decision)

- **No source I found tests a GRU with spread gate initialisation.** Tallec & Ollivier's chrono experiments are LSTMs. None of the sources tests delays in a GRU or any non-spiking gated network trained on an exact-byte task.
- **The fixed-delay controls named in Part B already have precedents.** These are "fixed random delays" (DelRec; Hammouamri for feedforward) and "uniform delay 1" (DelRec). In NARX, delays help gradient descent through BPTT's shortened paths, which is the same "shortcut" the audit flagged (finding 1).
- **Exact credit appears only in Tallec & Ollivier, NARX and Mészáros et al.** Perez-Nieves and DelRec use surrogate gradients.
- **Not found or not checked.** Hammouamri's run counts. The NARX journal version. Any work that varies GRU update-gate biases in mixed-distance copy tasks.
