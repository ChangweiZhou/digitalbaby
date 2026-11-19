# Where, and at what amplitude, the unchanging tissue carries each of the last bytes (19 November 2026)

**What this measures.** The reviewer's task of 19 November (05:57 UTC) asks where in the unchanging tissue, and at what amplitude, each of the last five bytes is carried. The program is `where.py`; the tables are made by `where_table.py` and `where_clt.py`. The tissue is the reviewer's `tissue.c` (through `tis.make`), with rule `fixed`, on seeds 73129, 93683 and 116089, in two forms:

- **U**: no control.
- **M**: `gain=2 rho=1 gmax=4`.

**How the measurement is made.**

- **Recording.** The output of every cell is recorded after each byte of the last 4,096 bytes.
  - Lives of 16,384 bytes: this is window A2. Tables `where/TABLE_U.md` and `where/TABLE_M.md`.
  - Lives of 262,144 bytes: the run that follows from the first one. Tables `where/TABLE_UL.md` and `where/TABLE_ML.md`.
- **Distances.** Every distance d from 1 to the largest (11) is read, as `drive.py` computes it. The pool of all 1,984 cells that are not feedback cells is read as well.
- **Lags.** 0 to 4.
- **(a)** The readings tool as it is: outputs below 0.01 are set to 0.
- **(b)** The same classifier with no floor, and with each cell's output divided by its own standard deviation. The classifier is unchanged by a positive scale of a feature, so (b) is the tool without its floor.
- **Sampling.** Both (a) and (b) use the same folds and the same sample of at most 128 cells.
- **References.**
  - C0(lag) is what the present byte alone gives.
  - C<L(lag) is what every nearer byte gives (`where/CLT*.json`).
- **The level.** An entry is marked when it exceeds C0(lag) + 0.02 for its own seed.

Each run took 30 to 56 CPU-seconds for 16,384 bytes and 300 to 340 for 262,144 bytes.

## What the runs measured

1. **U, the present byte.** Under (b) the present byte is carried far beyond where the tool can see it. S(d, 0) is 0.83 to 0.97 at every d from 2 to 7 and 0.51 to 0.62 at d = 9, in all three seeds.
   - Under (a), S(d, 0) is at chance (0.022) from d = 3 or 4 on.
   - The mean output falls by about 4 to 5 times a connection: 9e-4 at d = 2, 5e-5 at d = 4, and 1e-8 to 1e-9 at d = 10 and 11.
   - No output at d ≥ 3 exceeds 0.018.
2. **U, the byte before (lag 1).** Under (b) it lies above C0(1) + 0.02 only at d = 9, 10 and 11, in all three seeds: 0.12 to 0.21, against C0(1) of 0.070 to 0.080. The pool is also above the level in all three seeds.
3. **U, older bytes.** Lags 2, 3 and 4 are above the level at no distance and in no seed.
4. **M, the present byte.** With the control, amplitudes fall by about 2 times a connection instead of 4 to 5. Because of that, (a) still reads the present byte at d = 3 and 4 (0.65 to 0.72 and 0.26 to 0.33).
5. **M, lag 1.** The byte before is carried from d = 5 or 6 to d = 10 or 11 under (b), in all three seeds at d = 6 to 10: 0.12 to 0.26. It is at chance under (a).
6. **M, lag 2.** The byte two back exceeds C0(2) + 0.02 at d = 9 to 11 in two seeds over 16,384 bytes, and in all three over 262,144 bytes (0.077 to 0.103). It does not exceed C<2(2) + 0.02 in any seed: 0.087 to 0.099 + 0.02. So the cells carry nothing about the byte two back beyond what bytes 0 and 1 give.
7. **M, lags 3 and 4.** No lag 3 anywhere. One lag-4 entry (d = 8, seed 93683, over 262,144 bytes) is above the level.
8. **The long life changes little.** The tables over 262,144 bytes are close to those over 16,384 bytes for both U and M. The tissue does not change, and the control's gains settle.

## Limits

- **Tissue.** These are the reviewer's `tissue.c` with rule `fixed` only; no rule of strength change was read here.
- **Window.** There is one window per life.
- **Margin.** The level C0 + 0.02 is the step-1 brief's margin, not a test.
- **Float precision.** Outputs down to 1e-10 are read in float32, as `tis_run` records them. A cell whose standard deviation over the window is exactly 0 is left out of (b); its number is in `sampled_varying` in each json.
