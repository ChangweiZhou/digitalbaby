# PREP/readings.py: the readings of a recording window as a tool

**Task.** Message 28 of `MESSAGES_TO_WORKERS_20261118.md` (the reviewer's branch, commit 348e9fb0). The program reads recorded outputs only; it runs no tissue and acts on none. **Definitions** are those of section 7 of `TASK_STEP_1_THE_TISSUE_READS_THE_STREAM_20261118.md`.

## What it computes

`read_window(x, window_bytes, distance, seed, stream=None, start=None)` takes four inputs:

- `x`: the outputs of every cell at each position of a window (positions by cells; values below 0.01 are set to 0, as stored);
- `window_bytes`: the present byte at each position;
- `distance`: the distance of every cell;
- `seed`: the tissue's seed.

It returns four results:

1. **Shares (7.2).** For d = 1, 2 and 4: the number of cells at d; how many are silent (output above 0.01 at fewer than 1/128 of the window's positions); how many are constant (not silent, with a population standard deviation below 0.01); and the share that is neither.
2. **Scores (7.3).** For each d and each lag L in 0, 1, 2, 4 and 8, the score S of a nearest-centroid classifier, defined below. Each score comes with its standard error, the number of cells sampled and the number of byte values present.
   - **Cells.** A sample of at most 128 of the cells at d, drawn by `Random(seed + 900000).sample(sorted(cells), 128)` with a new generator for each window and distance. The same sample serves every lag.
   - **Folds.** Four consecutive equal stretches of the window. The fitting set leaves out the positions within 16 of the test stretch.
   - **Standardising.** By the fitting set's means and population deviations; a feature whose deviation is below 1e-9 is dropped.
   - **Assigning.** Each test position goes to the byte value of the nearest centroid in squared Euclidean distance. An exact tie goes to the smaller byte value, and a test label with no centroid counts as wrong.
   - **Score.** The balanced accuracy of the four folds' pooled predictions.
3. **C0 (7.5).** What the present byte alone gives: the same classifier on a one-hot code of the present byte.
4. **C<L (7.5).** What every nearer byte gives: the same classifier on a one-hot code of the bytes at lags 0 to L − 1. It is computed only when the stream is given.

**Labels.** The label of position t at lag L is the byte at t − L.

- If `stream` and `start` are given, every position with t − L ≥ 0 is a sample, as the brief says.
- Without them, only positions whose byte L back lies inside the window are samples (the first L positions drop out). These scores then differ slightly from the brief's.

## Two choices the brief leaves open, made as the recorded measures make them

- **The standard error** is the *sample* standard deviation (divisor 3) of the four folds' balanced accuracies, divided by 2. With the population standard deviation the errors come out smaller by the factor √(3/4) and do not match the receipt.
- **Exact ties.** The squared distances are computed as ‖z‖² − 2 z·c + ‖c‖². Wherever the two smallest are within 1e-9 (relative), they are recomputed directly as Σ(z − c)² and the first minimum is taken. Since the centroids are in increasing byte order, an exact tie goes to the smaller byte value. In these windows this rule gives the same assignments as the receipt's (every S equal to 1e-16).

## The test

`PREP/test_readings.py` runs on the two windows of the coincidence tissue of E63's stage A, seed 73129: `TISSUE_STEP1/runs/A_coincidence_73129/A1.npz` (positions 2,048 to 6,143) and `A2.npz` (12,288 to 16,383) on `origin/claude/e63-tissue-step1` at 040d3cf0.

- **The stream.** The 16,384-byte stream is regenerated from the stream functions of `check_step1_brief.py`, the reviewer's code, executed only up to its first output. Its SHA-256 equals the run receipt's (`9fed26e7…`).
- **Against the run receipt** (`measures`, written by the E63 worker's `step1_read.py`):
  - all six counts of shares are equal;
  - the largest difference in S is 5.6e-17, and in its standard error 3.9e-18, over 30 scores;
  - C0 and C<L agree to 1.4e-17.
- **Against STAGE_A_REPORT.md.** All 56 printed figures of the two tables "2. coincidence, E10, window A1/A2" are equal: S to four decimals, its standard error to three, C0, C<L and the shares. For example, the brief's outcome figures S(2, 0, A2) = 0.0375 and S(4, 0, A2) = 0.0264, and the shares 0.9960 of 250 and 1.0000 of 202, are reproduced.
- **The output** is in `PREP/test_readings_output.txt`.

The E63 worker's `step1_read.py` was read only to learn that its labels at a lag come from the regenerated stream. The tool was written from section 7 of the brief.

## Limits

- The tool was tested on one tissue of one seed, in two windows of 4,096 positions; it was not run on stage B's windows of 8,192 positions.
- It computes no distances: they are an input, recorded by the tissue (7.1, item 1).
- It applies none of the rules of sections 7.6 and 7.7. The differences S − B, S − C0 and S − C<L are left to the caller.
- It does not compute the fixed tissue's readings (7.4) separately; the same function applies to the fixed tissue's windows.
- Cost: about 1.5 seconds for one window of 4,096 positions without the stream (shares, 15 scores, C0), on one thread.
