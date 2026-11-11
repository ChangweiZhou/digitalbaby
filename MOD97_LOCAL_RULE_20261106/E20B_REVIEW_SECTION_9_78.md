### 9.78 E20-B: windowless reference networks under backprop (11 November 2026)

**What ran.** E20-B ran twelve births and twelve lives at full dose on one host: four networks (G0 = arch17 bank2; G1 = GRU; G2 = LSTM; G3 = bank2 plus a 64×512 feedback matrix) × seeds 73129, 93683 and 116089.

**Verification.** All 24 completion records and their weights hashes (file and tensor content) were verified by `summarize_e20b.py`. The total cost was 98,048 CPU-seconds.

**Predictions.** All four registered predictions hold, read as in `E20B_PREDICTIONS.md`:
- **P20.7:** G1 and G2 are born at a hard mean of 93.5 to 99.8.
- **P20.8:** new instances of taught pairs end at 87.0 to 91.2 for G1 and 85.7 to 88.8 for G2.
- **P20.9:** the plus-minus pairs end at 73.6 to 78.8 for G1 and 69.2 to 78.6 for G2. G3 is at 1.0 to 20.2.
- **P20.10:** G3 − G0 on `within` is +18.5 to +29.2.

**Selection.** Under amendment 13, G1 and G2 qualify in all three seeds and G3 in none. G1 is the windowless candidate, on a higher three-seed plus-minus mean: 75.8 against 73.3.

**Interruption.** A container restart killed the first three G0 lives. They are kept as written, and the lives were rerun from scratch from the same births under an optimizer-state-saving wrapper. That wrapper was shown bit-identical to uninterrupted runs under SIGKILL.

**Limits.** P20.10 is a package result (32,768 weights and their cost), not evidence for the placement of the feedback. "Windowless" refers to the input; the decoder reruns the network per output byte. This is a development feasibility result at E2 with no claim of reuse or promotion, from one host, three seeds, one dose and one teaching protocol.
