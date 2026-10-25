# Cycle 1: implement and audit (25 Oct 2026)

Nothing was scored on the main text. The dev text is Buster Bear.

## Built
* `corpus.py`: V148's record and split rules (V148's `corpus.py` loaded unchanged under its own name), plus the main text.
  * `chesterton-brown.txt`, 406,629 bytes, SHA-256 `67027976…47ec`, extracted from NLTK's Gutenberg archive (SHA-256 `2d3c3ab5…d611`, the same archive as V148).
  * **Design correction, made before any run:** this file is *The Wisdom of Father Brown* (1914), not *The Innocence of Father Brown* as first written. The design now says so; the file and its role are unchanged.
* `states.py`: the native state before each symbol, read exactly as `values()` reads it.
  * `x` is the binary KC code; `dx` is the observed KC activity after adaptation.
  * The FE0 front end and the last-byte-only code are also produced.
  * A one-compartment bank carries the state, since nothing is taught.
* `readouts.py`: the evaluator-only readouts R1 (logistic regression) and R2 (KC vote), the 257-symbol bits convention, and the codes for the readout checks.
* `run.py`: the runner, with ladder, readout checks, arms and line-bootstrap SDs.
* `checks.py`: the unit checks.

## Refinements found while building (design updated before any run)
1. **The raw-trace arm (V148 design 9a) is dropped.** With an untaught memory the thresholds are zero and `encode` takes the top k by rank, so dividing the trace by its maximum cannot change the code. Checked on 290 real positions: the codes are identical.
2. **An "empty state" indicator is added to every arm.** At a line start the trace is empty and the KC code is all zeros, and a linear readout of a zero vector can only output its bias. The indicator is a function of the state, so it adds no outside information.

## Checks: 17/17 pass (`cycles/CYCLE1_CHECKS.json`)
* Corpus pins and splits for the three texts:
  * Buster: train records 0–662, calibration 662–745, held-out 1152–1314.
  * Bryant: train 0–750, calibration 750–831, held-out 3952–4122.
  * Father Brown: train 0–647, calibration 647–730, held-out 6282–6442.
* `x` equals the `values()` code of the full 16-compartment memory. `dx` from the one-compartment bank equals compartments 0 and 15 of the full bank, over 6 records.
* A record's states do not depend on records read before it, and reading states leaves the memory's state hash unchanged.
* Raw and max-normalised traces give identical codes when the thresholds are zero.
* The last-byte-only code is zero at a line start; at position 1 it equals the code of the previous byte alone.
* R1 and R2 probabilities sum to 1, and the R2 toy values match hand computation.
* The runtime closure lists 58 repository modules.
* **Weak checks, stated as such:**
  * "Probabilities over 257 symbols sum to 1" holds almost by construction.
  * "R1 uses only taught states" only checks that the fit is deterministic. The real guarantee is structural: `run.py` passes only training arrays to the fits.

## Cost
* States for all four arms: 0.010 CPU-s per position, about 10 minutes for a full 40 + 5 + 10 KB split.
* R1 cost scales with sparse products: about 0.06 s per gradient evaluation at 4,000 positions, using float32 products. An estimated 3–6 minutes per arm at full size will be measured in cycle 2.

## Tiny end-to-end run (exercise only; Buster at 4 KB / 1 KB / 2 KB, 50 iterations)
This run used unconverged fits on tiny data. The numbers below show only that the runner works end to end, plus two problems to settle in cycle 2:
* **Floor check.** The constant code scored 4.51 bits/byte against the unigram rung's 4.57, so it beats unigram by more than the 0.02 allowed. R1 with a constant code is the exact frequency estimate over seen symbols plus OTHER. V148's unigram rung uses add-one smoothing over all 257 symbols, which wastes probability on small data. The check compares two smoothings, not information. Cycle 2 measures the gap at full size before the check is revised.
* **Ceiling check.** The last-symbol one-hot scored 3.96 against bigram's 3.70, outside 0.15. This may be under-fitting at this size and iteration count; cycle 2 decides.

Results: `results/c1_tiny/RESULTS.json`.
