# V148 PRETRAIN-PROBE V1: report

25 October 2026. Design: `minifly_v148_pretrain_probe_design_20261025/DESIGN.md`. Protocol locked at commit d3a7f815 (`LOCK.json`, 59 files hashed) before the main text was scored. The main run was executed once, on *Alice's Adventures in Wonderland*, 05:34–07:52 UTC; the lock still verified after it.

## Question
What does the unchanged native FastMemory (`minifly_cls_dev_20261017/fastmem.py`) extract from a small amount of raw English text by next-byte prediction? Each line is one record, taught as `train(b"", line)`: every byte, then its own end-of-line symbol. The primary measure is held-out bits per symbol (257 symbols: bytes plus end-of-line), compared with simple counting predictors trained on the same bytes.

## Main result (Alice, locked run)
Bits per symbol; lower is better. Training is one continuous history: nested prefixes, then passes 2 and 4 over the full 40 KB.

| Taught | Pass | MiniFly held-out | MiniFly taught, first 5 KB | MiniFly taught, last 5 KB | Unigram | Bigram | PPM-3 | Best PPM (orders 2–5) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 KB | 1 | 8.006 | – | – | 8.006 | 8.006 | 8.006 | 8.006 |
| 5 KB | 1 | 7.194 | 6.425 | 6.421 | 4.669 | 3.845 | 3.438 | 3.438 |
| 10 KB | 1 | 7.128 | 6.744 | 6.283 | 4.655 | 3.790 | 3.213 | 3.213 |
| 20 KB | 1 | 7.095 | 6.808 | 6.346 | 4.636 | 3.728 | 2.986 | 2.986 |
| **40 KB** | **1** | **7.104** | **7.028** | **6.501** | **4.625** | **3.683** | **2.819** | **2.778** |
| 40 KB | 2 | 7.121 | 7.013 | 6.620 | 4.625 | 3.683 | 2.819 | 2.778 |
| 40 KB | 4 | 7.248 | 7.130 | 6.941 | 4.625 | 3.683 | 2.819 | 2.778 |

Uniform is 8.006 throughout. Held-out top-1 accuracy of MiniFly: 0.100, 0.102, 0.110, 0.099, 0.100, 0.090 (5 KB to pass 4). Full tables: `results/main/SUMMARY.md`; raw: `results/main/RESULTS.json`.

## Registered predictions (scored by the locked `summarize.py`)

| | Registered expectation | Measured | Result |
|---|---|---|---|
| P1 held-out level | between bigram and PPM-3 | 7.104, worse than bigram (3.683) and unigram (4.625) | **Not as predicted** (worse side) |
| P2 memorisation gap | taught text clearly below held-out, and early text not erased (both first and last taught 5 KB ≥ 0.25 below held-out) | last 5 KB 0.604 below; first 5 KB 0.077 below | **Not as predicted** (early text erased) |
| P3 learning curve | held-out improves from 5 to 40 KB by less than every PPM rung | MiniFly 0.090; PPM orders 2–5: 0.425, 0.618, 0.699, 0.690 | **As predicted** |

The dev data had suggested P1 would fail on the worse side (`cycles/CYCLE2.md`); that expectation was recorded before the lock, and P1 was not edited. P2 passed on the rehearsal text (first-5-KB gap 0.37) and failed on the main text.

## What the measurements show
1. **MiniFly predicts raw text worse than a symbol-frequency count, even on the text it was taught.**
   - Held-out: 7.10 against the unigram's 4.63.
   - The last taught 5 KB: 6.50 against the unigram's 4.62 on the same slice.
   - Its held-out top-1 accuracy (0.099) is below the unigram's (0.170, post hoc, below).
2. **More data barely helps.** From 5 to 40 KB, held-out improves by 0.09 bits; the PPM rungs improve by 0.4–0.7 bits on the same bytes.
3. **More passes hurt.** Pass 2 is flat, and pass 4 is worse on held-out (7.25) and on both taught slices.
4. **What is stored fades as later text is taught.**
   - The first 5 KB (the identical 79 records, 5,162 symbols) scored 6.43 right after it was taught. After 35 KB more it scored 7.03, close to held-out.
   - Only the most recent text keeps an advantage.

## Diagnostics (labelled; not primary; computed in the locked run)
* **Mass on never-seen symbols.** About 63% of MiniFly's held-out probability (0.62–0.66 across checkpoints) goes to symbols that never occur in the taught text. At 40 KB, pass 1, its top choice is such a symbol 29% of the time; by pass 4, 42%. The fixed 257-class output code has no notion of which symbols occur.
* **Renormalised over seen symbols.** Restricting the native scores to symbols seen in training, with a separately calibrated temperature, gives 5.55 at 40 KB. That is still 0.92 bits worse than the unigram. The output code accounts for about 1.5 bits of the gap, so it is one cause and not the only one. These diagnostics do not identify the remaining cause.

## Post-hoc descriptive figures (`posthoc.py`; written during the main run, not locked, evaluator-only)
* **Held-out top-1 accuracy at 40 KB:** unigram 0.170, bigram 0.289, PPM-3 0.482; MiniFly 0.099.
* **Own generations** (50 held-out prefixes of 10 bytes, greedy, 40 KB pass 4):
  * 21 stopped at the memory's own end-of-line; mean 52.5 bytes; 68% printable ASCII.
  * 15% of alphabetic tokens occur as words in the taught text, against 80% for the true continuations. Most of those matches are single letters ("e", "a", "o") and contraction fragments.
  * Counting only tokens of two or more letters: 9.7% of 340 are taught words, against 78.5% for the true continuations. The words found are ah, an, and, as, at, end, had, he, her, is, key, me, my, oh, on, or, re.
  * Repetition (1 − distinct/total byte 3-grams): 0.002 own, 0.039 true.
  * The rest is mostly non-text bytes. Five examples are in `results/main/SUMMARY.md`.

## Consistency with the development texts (development results, not evidence)

| Text (role) | Taught | MiniFly held-out | Renormalised (diag.) | Unigram | Bigram | PPM-3 | First / last taught 5 KB gap to held-out |
|---|---:|---:|---:|---:|---:|---:|---:|
| Buster Bear (cycle 2; reduced protocol: 2 KB calibration, 4 KB held-out) | 10 KB | 6.945 | – | 4.468 | 3.539 | 3.006 | 0.45 / 0.84 |
| Bryant (rehearsal) | 40 KB | 7.032 | 5.466 | 4.529 | 3.712 | 3.198 | 0.37 / 0.98 |
| **Alice (main)** | **40 KB** | **7.104** | **5.546** | **4.625** | **3.683** | **2.819** | **0.08 / 0.60** |

The pattern is the same on all three texts. MiniFly sits between uniform and unigram, the curve is flat, and repetition does not help.

## What this does not show
* It tests the unchanged memory only. None of the later components (design section 9: raw trace, self-supervised layer, replay) was tested.
* It does not explain the part of the gap that remains after renormalisation.
* There is one main text. The memory is deterministic given its native birth, so there is no seed variation to report.

## Bearing on the next step
Design section 9 counts a later component as an improvement only if it lowers held-out bits/byte against this baseline and against the best ladder rung at the same size. The baseline is below the unigram rung, so the first rung still to reach is the one that uses no context at all. The memory does not reproduce symbol frequencies, even on text it was taught. Which step to take next is left for discussion.

## Cost and accounting
* **Main:** training 5,930 CPU-s for 163,972 byte events (4 passes × 40,993; 0.036 CPU-s per event). MiniFly evaluations 2,214 CPU-s. Peak memory 348 MB.
* **Rehearsal:** about 2.3 CPU-hours. Cycle 2 dev runs: about 1 CPU-hour (MiniFly training and evaluation).
* **Failed attempts:** none in the main run or the rehearsal.
  * Cycle 1: one check failed to serialise (fixed).
  * Cycle 2: a resume bug reset the pass counter (fixed).
  * One smoke diagnostic was discarded (`cycles/CYCLE2.md`).
  * Gutenberg was blocked by the network proxy, so the NLTK mirror of the same Gutenberg texts was used; recorded in `cycles/CYCLE1.md`.

## Files
* `corpus.py`, `ladder.py`, `probe.py` and `run.py` are the runtime. `checks.py` holds the 26 cycle-1 checks. `summarize.py` is the locked scorer, `make_lock.py` writes `LOCK.json`, and `posthoc.py` produces the post-hoc figures.
* Results:
  * `results/main/` (RESULTS, SUMMARY, POSTHOC) with log `results_main.log`;
  * `results/c3_rehearsal/` with log `results_c3_rehearsal.log`;
  * `results/c2_dev/` and `results/c2_dev_passes/`.
* Cycle records: `cycles/CYCLE1.md`, `CYCLE2.md`, `CYCLE3.md`.
