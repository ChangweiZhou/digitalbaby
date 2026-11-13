# Hostile audit of the two source checks of 13 November, and the corrections that stand (16:43 EST by the clock)

The auditor was a fresh session of the same model family: a second reader, not an independent one. It opened the sources itself. The coordinator spot-checked two points (below) and recomputed the arithmetic; the rest is the auditor's reading and is marked so. **Neither memo, nor this note, decides a design.**

## Memo 1, learned feedback weights (`SOURCE_CHECK_LEARNED_FEEDBACK.md`)

The equations and reported numbers were confirmed against Akrout et al. 2019, Kunin et al. 2020, Lansdell et al. 2020 and Launay et al. 2020. Corrections:

1. **"No study applies learned feedback through time" is wrong.** Three were found by the auditor: Bellec et al. 2020 (adaptive e-prop: output-to-neuron feedback weights learned by a local rule in recurrent spiking networks, reported as similar to random feedback); Roth, Kanitscheider and Fiete 2019 (KeRNL: learned sensitivity weights on recurrent networks, below BPTT); Manchev and Spratling 2020 (target propagation through time with a trained feedback matrix). The coordinator confirmed that the Bellec paper exists and is on recurrent spiking networks; the adaptive variant's text was not opened by the coordinator.
2. **Kolen–Pollack does not copy a weight, and that is a thin statement.** The feedback synapse must receive the same product of the two cells' signals, the same rate and the same decay as its forward partner; the difference between the two then shrinks by the decay factor each step. Kunin et al. rate this as about as hard to justify as symmetry itself. Alignment took 10 to 20 ImageNet epochs in Akrout et al., and without decay there is none.
3. **The weight mirror** needs a separate noise phase and learns from the forward cell's response to noise; it reads no forward weight. It was replicated poorly outside its original setting (Kunin et al.: 63.5 against 69.7 on ResNet-18 after a search of 824 settings).
4. **Work per token is about twice the memo's figure.** Extra parameters for a separate feedback matrix per GRU matrix: 589,824 (confirmed by the coordinator). Work by Akrout et al.'s own count: about 2.36 million operations per token for six matrices, not 1.18 million.
5. **The one recurrent test of a rule of this family showed no gain over fixed random feedback** (Bellec et al., as read by the auditor).
6. Not opened by anyone: Kolen and Pollack 1994 itself.

## Memo 2, timescales and delays (`SOURCE_CHECK_TIMESCALES_AND_DELAYS.md`)

Confirmed: chrono initialisation as described (Tallec and Ollivier 2018), and the set-ups of the papers on heterogeneous time constants and on delays. Corrections:

1. **A spread at initialisation alone bought about 2 points; training the time constants bought about 10** (Perez-Nieves et al., SHD: 71.7 homogeneous; 73.7 heterogeneous initialisation only; 82.7 homogeneous initialisation with trained time constants; 81.7 both). The memo paired only the first and last. The coordinator found these four figures in copies of the preprint; the journal table was not opened by the coordinator.
2. **Chrono's ranges were set from each task's known length,** which E29's Part B forbids itself. Where the range was not tied to a long known dependency (text8, MNIST) the gain was small or absent.
3. **Chrono draws timescales uniformly, not log-uniformly;** and a library whose gate multiplies the old state, as torch's GRU does, needs the opposite bias sign from the paper's equation (the auditor's derivation).
4. **Fixed random delays captured most of the benefit of learned ones** in the one recurrent ablation read (DelRec; three seeds, surrogate gradients); a second study saw little benefit from delays in larger networks on one dataset.
5. **NARX is a weaker precedent than the memo says:** in its latching test no learning passes through the delays.
6. The memo's "no source tests a GRU with a spread of gate values" is a statement about the sources opened; no search is described.

## What follows for this project, as questions and not decisions

- For a learned backward pathway on the GRU, the published record on recurrent networks is three studies, none favourable beyond matching random feedback. Any forecast for it here is ours alone.
- For E29's Part B, the nearest published result says the gain comes from letting timescales be learned, not from spreading them at birth. Part A found the same thing in our own networks: training builds the holding units. A fixed spread at birth should be forecast accordingly.
