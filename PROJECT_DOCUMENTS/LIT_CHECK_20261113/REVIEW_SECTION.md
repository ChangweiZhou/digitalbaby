# Source check: learned feedback weights (13 November 2026, read-only)

**What was checked.** The read-only brief of 13 November asked whether the two mechanisms named from memory in the audit session's note (§4.4) hold up against their sources. I opened in full Akrout et al. 2019, Kunin et al. 2020, Lansdell et al. 2020, Launay et al. 2020 and Murray's RFLO preprint. I could not open Kolen and Pollack's 1994 paper, so it is known here only through Akrout's §5.1.

**What the sources say.**
- **Neither named mechanism copies a weight, and both start from random feedback weights.**
  - In the modified Kolen–Pollack rule, the feedback synapse uses the forward cell's activity and its own cell's teaching signal. Its update mirrors the forward update, at the same rate and decay, so the two matrices converge (Akrout §5.2, eqs. 15–17).
  - The weight mirror learns the transpose from noise in a separate mode (§4, eqs. 6–7).
  - Exception: in Akrout's ImageNet runs, BatchNorm's scale was passed from the forward to the feedback path, one scalar per neuron (App. D.3).
- **Akrout's results:** on ImageNet ResNet-18 and ResNet-50, both mechanisms came within about 1 point of backprop's top-1 error and beat sign symmetry by 7.6–10.4 points.
- **Kunin et al.'s replication:**
  - The weight mirror reached 63.5–64.1% top-1 accuracy against the reported 69.7%, and failed on other architectures.
  - Kolen–Pollack matched backprop on Akrout's two networks, but fell well behind on deeper and v2 ResNets.

**What was not found.**
- Any learned-feedback study on recurrent weights or through time. The nearest, RFLO, uses fixed random feedback.
- Any learned-feedback result under a next-token objective. The only next-token result found is random direct feedback alignment on a Transformer: perplexity 52 against backprop's 30–34.
- Any report on mostly unpredictable targets.

**Cost.** For a GRU of 256 units with a 512-wide input, a full learned pathway would add up to 589,824 parameters, doubling the weights, plus an update per matrix the size of the forward one. That is my arithmetic, not a measurement.

**Rulings of the note checked.**
- §4.4's description of both mechanisms: confirmed against Akrout.
- The "close to backprop" claim: confirmed for Akrout's own networks, corrected by Kunin's failed replication of the weight mirror and Kolen–Pollack's gap on deeper ResNets.
- "No use through time": confirmed by the searches recorded in the memo.

Memo: `PROJECT_DOCUMENTS/LIT_CHECK_20261113/SOURCE_CHECK_LEARNED_FEEDBACK.md`. The memo decides nothing.
