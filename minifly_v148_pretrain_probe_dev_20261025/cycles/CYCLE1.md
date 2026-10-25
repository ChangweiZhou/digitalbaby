# Cycle 1: implement and audit (25 Oct 2026)

## Implemented
* `corpus.py`:
  * loads three public-domain texts from the NLTK mirror of Project Gutenberg, each pinned by SHA-256;
  * builds line records (title line dropped, lines over 128 bytes cut);
  * makes contiguous splits and nested prefixes.
* `ladder.py`: uniform, add-one unigram, Witten-Bell bigram, and static PPM-C (orders 2–5, full exclusion), over 257 symbols. Contexts reset at each line.
* `probe.py`: the unchanged native `FastMemory` (default settings).
  * Training: `train(b"", line)`.
  * Probe: a teacher-forced clone read with the same record structure, giving the native 257-way logits at every position.
  * One softmax temperature, fitted on a 61-point geometric grid.
  * Greedy own generation.
* `run.py`:
  * one continuous history evaluated at nested prefixes;
  * calibration, held-out, taught first/last 5 KB and generations;
  * the ladder on the same splits;
  * per-checkpoint saving with resume;
  * a lock check for the main run.

## Deviation from the design
Project Gutenberg itself is blocked by this environment's network policy (proxy 403). The same public-domain texts were taken from NLTK's Gutenberg corpus on GitHub (`nltk_data/packages/corpora/gutenberg.zip`, archive SHA-256 `2d3c3ab5…d611`). Those copies already have the licence boilerplate removed. The main text stays *Alice's Adventures in Wonderland* (`carroll-alice.txt`, 144,395 bytes). The dev texts are `burgess-busterbrown.txt` (cycle 2) and `bryant-stories.txt` (cycle-3 rehearsal, a text not used before).

## Audit: `checks.py`, 26/26 PASS (`CYCLE1_CHECKS.json`)
* **Corpus:**
  * corpus pins and deterministic records for all three texts;
  * the Alice split is ordered and disjoint (train records 0–682, calibration 682–762, held-out 2290–2478 of 2478);
  * nested prefixes are increasing.
* **Ladder:**
  * hand-computed toy values for PPM-C order 2 (p(b|a) = 1/4, p(EOS|a) = 1/6, p(unseen|a) = 1/(6·253)) and for the bigram;
  * every rung sums to 1 over 257 symbols (error ≤ 8e-15).
* **Probe:**
  * it leaves the memory's state hash unchanged and is deterministic;
  * generation leaves the state unchanged;
  * calibration returns finite bits;
  * a pickle round trip preserves the state hash and the logits.

## Cost (1 KB of dev text)
0.036 CPU-s per training event and 0.012 CPU-s per probed symbol. The full protocol on one text (40 KB of training plus evaluation at five checkpoints) is therefore about 50 CPU-minutes.
