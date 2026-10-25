# Cycle 3: dress rehearsal and lock (25 Oct 2026)

Rehearsal text: *Stories to Tell to Children* (Bryant, `bryant-stories.txt`). It was not used in cycles 1–2. The protocol is exactly the one to be locked: prefixes 0, 5, 10, 20 and 40 KB in one continuous history, then passes 2 and 4 over the full 40 KB; calibration 5 KB; held-out 10 KB from the end of the text; 50 own generations at the final checkpoint. Code was not changed during the run: the hashes of `run.py`, `probe.py`, `corpus.py` and `ladder.py` were the same at the start and at the end. Development result only; nothing was scored on the main text.

Command: `python run.py --text bryant --prefixes 0,5,10,20,40 --passes 1,2,4 --calib 5 --held 10 --gen 50 --tag c3_rehearsal`. Output: `results/c3_rehearsal/RESULTS.json`, `SUMMARY.md`, log `results_c3_rehearsal.log`.

## Split audit
4122 records. Train 0–750 (40,987 symbols), calibration 750–831, held-out 3952–4122. The blocks are contiguous and do not overlap; held-out and calibration are never passed to `train`. Of held-out target symbols, 99.83% (5 KB) to 100% (40 KB) occur somewhere in the taught text.

## Results (bits per symbol; symbols are bytes plus end-of-line)

| Taught | Pass | MiniFly held | held accuracy | taught first / last 5 KB | never-seen mass | renormalised over seen (diag.) | Unigram | Bigram | PPM-3 | Best PPM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 KB | 1 | 8.006 | 0.000 | – | 0.996 | – | 8.006 | 8.006 | 8.006 | 8.006 |
| 5 KB | 1 | 7.062 | 0.099 | 6.24 / 6.23 | 0.650 | 5.44 | 4.579 | 3.847 | 3.600 | 3.600 |
| 10 KB | 1 | 7.177 | 0.104 | 6.72 / 5.65 | 0.658 | 5.49 | 4.555 | 3.819 | 3.534 | 3.534 |
| 20 KB | 1 | 7.073 | 0.112 | 6.71 / 6.08 | 0.648 | 5.43 | 4.538 | 3.767 | 3.379 | 3.379 |
| 40 KB | 1 | 7.032 | 0.118 | 6.66 / 6.05 | 0.605 | 5.47 | 4.529 | 3.712 | 3.198 | 3.191 |
| 40 KB | 2 | 7.050 | 0.120 | 6.62 / 6.04 | 0.620 | 5.49 | 4.529 | 3.712 | 3.198 | 3.191 |
| 40 KB | 4 | 7.172 | 0.106 | 6.76 / 6.35 | 0.639 | 5.56 | 4.529 | 3.712 | 3.198 | 3.191 |

Own generations (40 KB, pass 4): 50; 19 stopped at the memory's own end-of-line; mean length 54.6 bytes; 72% printable ASCII. They contain fragments such as "he", "the" and spaces among non-text bytes (`SUMMARY.md` lists five).

## Reading (development, not evidence)
* The held-out level is flat from 5 to 40 KB (7.06 to 7.03), while PPM-3 improves by 0.40 bits. MiniFly stays about 2.5 bits worse than the unigram counter.
* The output code is not the whole gap. Renormalising over symbols seen in training, with its own temperature, gives about 5.45. That is still about 0.9 bits worse than unigram.
* Some text is stored: at 40 KB the last taught 5 KB scores about 1 bit better than held-out, and the first 5 KB about 0.37 better.
* Repetition does not help: pass 2 is unchanged and pass 4 is worse on both held-out and taught text.
* Applying the locked scoring rules to this dev run gives P1 not as predicted (worse than bigram), P2 as predicted, and P3 as predicted. These are dev numbers only.

## Decisions
1. **No protocol change.** The rehearsal ran end to end, resumable state was written at every checkpoint, the split is sound and the numbers are consistent with cycle 2.
2. **Prediction operationalisations are fixed in `LOCK.json` before the main run**, taken from the design wording (section 7):
   * P1: held at 40 KB, pass 1, at most bigram and not below PPM-3. Below the best of PPM orders 3–5 = weakened (better than predicted). Above bigram = not as predicted.
   * P2: at 40 KB, pass 1, taught bits on both the last and the first 5 KB at least 0.25 below held. The last 5 KB tests "memorised"; the first 5 KB tests "not erased by interference".
   * P3: MiniFly's held improvement from 5 to 40 KB smaller than that of every PPM rung (orders 2–5).
3. **Lock** (`make_lock.py` → `LOCK.json`): 59 files hashed. They are the runtime closure taken from `sys.modules` after importing the runner (native sources included), the native birth assets, the output codebook, the three corpus files and the scorer `summarize.py`. `run.py --main` refuses to start if any hash differs.

## Cost
Training 6,034 CPU-s for 163,948 byte events (4 passes × 40,987, end-of-line included): 0.037 CPU-s per event, as measured in cycle 1. MiniFly evaluations 2,210 CPU-s over seven checkpoints. Peak memory 350 MB. Total about 2.3 CPU-hours.

## Failed attempts so far
None of the runs crashed. Cycle 1 had one JSON serialisation failure in a check (fixed). Cycle 2 found a resume bug that reset the pass counter (fixed). One smoke diagnostic was discarded (CYCLE2.md).
