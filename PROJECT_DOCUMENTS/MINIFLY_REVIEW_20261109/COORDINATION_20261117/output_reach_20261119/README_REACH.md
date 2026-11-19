# Next-byte reader on the cells near the output lines (19 November 2026)

This folder is a copy of `reviewer_tissue_20261119` at 3f22c216 (`tissue.c`, `tis.py`, `drive.py`, `diag.py`, `cmp_fixed.py`, `deps/`, `README.md`, unchanged). Added:
- `reach.py`, the reader;
- `onehot3.py`, the reader on three one-hots;
- `make_table.py`, `TABLE.md`;
- `batch_16k.txt` and `batch_64k.txt`, run with `SCRIPT=reach.py sh run_batch.sh <batch>`;
- `deps/PREP/score_emissions.py` and `life_stream.py`, copied unchanged from 10e3ab9c on claude/e63-score-emissions.

Results are in `reach/*.json` and logs in `out/`. `out/prov/` holds the first, superseded runs, whose ridge grid started at 1e-3.

**Question.** Is what the next byte depends on present in the cells near the output lines?

**Tissues.**
- U: `rule=fixed`, no control.
- M: `rule=fixed gain=2 rho=1 gmax=4`.
- Seeds 73129, 93683 and 116089, at 16,384 and 65,536 bytes.

**Stream.** The tissue's stream is the first bytes of `life(seed)`; the program asserts this. So the scorer's kinds apply, and only R1 and R2 occur in these windows. "rest" is every other position.

**Cell sets.**
- (a) The 256 driver cells of the output lines plus every cell in a driver's neighbourhood, with feedback cells left out. This is 495, 519 and 495 cells for the three seeds, at 7 to 11 connections from the input lines.
- (b) Every non-feedback cell: 1,984.

**Features.**
- The output at the reading step of byte t, divided by its standard deviation over the fitting part (no floor) and centred.
- A cell whose output does not vary over the fitting part is left out. This affected 0 to 16 cells, counted in the JSON.

**Reader.**
- Ridge regression onto the 256 one-hot values of byte t+1, with a constant term. The byte given is the largest output.
- It is fitted on the first three quarters of the positions and scored on the last quarter.
- The ridge constant is chosen from 10^-6 to 10^4 (times the number of fitting positions) on the last fifth of the fitting part.
- For the cells the smallest constant, 1e-6, was nearly always chosen, so the reader there is practically unregularised least squares.

**Comparisons.** The same reader on one, two and three one-hots of the last bytes, and the scorer's references.

**Results** (share right on the last quarter, all positions; table in `TABLE.md`):

| Reader | 16,384 bytes | 65,536 bytes |
|---|---|---|
| U, cells near the output lines (a) | 12.2–12.9% | 12.8–14.7% |
| U, every non-feedback cell (b) | 13.3–13.8% | 14.1–14.7% |
| M, cells near the output lines (a) | 11.7–15.1% | 12.3–15.3% |
| M, every non-feedback cell (b) | 14.0–16.8% | 14.6–16.6% |
| Present byte alone (one-hot) | 10.1–11.3% | 10.4–10.8% |
| Present and previous byte (two one-hots) | 11.8–13.7% | 12.5–13.0% |
| Last three bytes (three one-hots) | 27.7–29.0% | 27.4–27.7% |
| ref_1 | 9.8–11.2% | 10.5–10.6% |
| ref_2 | 11.4–12.4% | 12.1–12.3% |
| ref_3 | 20.3–21.3% | 25.1–25.4% |

**What the runs measured.**
1. **Cells near the output lines.** In U a linear reader on them gets more than the present byte alone gives on every seed and both lengths. It gets at or above what two bytes give at 65,536 bytes on all three seeds, but at 16,384 bytes only on seed 73129 (93683: 12.2 against 12.6; 116089: 12.9 against 13.7).
   - The previous byte is visible at R2: the one-hot present-byte reader scores 0% there, the cell readers 5 to 18%.
   - In M the reader on set (a) is above the present byte alone on every seed and length. On seed 93683 it is below the two-byte reader (11.7 against 12.6 at 16,384 bytes; 12.3 against 12.5 at 65,536). On seed 116089 at 16,384 bytes it is also slightly below (13.4 against 13.7).
2. **The whole tissue.** M's non-feedback cells give more than the two-byte reader on every seed and length (14.0 to 16.8 against 11.8 to 13.7), and are the highest of the cell readers. U's give more than the two-byte reader in five of the six cases; the exception is seed 116089 at 16,384 bytes (13.4 against 13.7).
3. **Three bytes.** No cell reader comes near what three bytes give (27 to 29% linear; ref_3 20 to 25%).
4. **ref_mode is 0.** The scorer's most-frequent byte alternates between bytes 10 and 61, which are tied in count at almost every position. Its guess is therefore never right; this is a property of that reference, not of the reader.

**Not measured.**
- Whether a reader fitted online, or by a rule local to the output end, reaches these shares.
- Readers other than linear ones.
- Lengths beyond 65,536 bytes.

**Event.** For about a minute, six processes ran instead of four: three `onehot3.py` runs started while two tissue runs were still going. They were stopped and run again afterwards, and the results kept are those of the reruns.
