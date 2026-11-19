# The pruning tissue with matched peaks, on three seeds, with the time constants of the potential (19 November 2026)

**What this measures.** The reviewer's task of 19 November (07:20 UTC) asks for a tissue that keeps both the present byte far from the input and the bytes before it. The tissue is the reviewer's `tissue.c` at 53a8d732 (copied here), with `tissue_tau.c` and `tis_tau.py`, which add the time constant of the potential per cell as in the earlier folders.

- **Base (row 1).** `rule=timing prune_only=1 no_make=1 norm=1 lines_learn=1 we=1/2 wi=1/2 gain=3 rho=1 gmax=8`, with prune_ratio 1 (the default).
- **Rows run.**
  - (2) prune_ratio 2 and 4. Ratios 1.25 and 1.5 were added as a follow-on run, because 2 and 4 diverge.
  - (3) gmax 4 and 16 at ratio 1.
  - (4) we/wi 3/4 and 1/2, and 3/4 and 3/4, at ratio 1.
  - (5) u2-64e and u2-64 on gmax 8 and gmax 4.
  - 65,536 bytes for gmax 4 and for gmax 4 with u2-64e.
- **Seeds.** 73129, 93683 and 116089.
- **Tables.** `where_tau/TABLE_<label>.md` give every distance, lags 0 to 4, the tool as it is and the same without its floor, the mean output, the share of outputs above 0.01, and the divergence. The json files also hold the largest output (`max_output`), the cells with any output (`cells_with_output`) and connections a cell.
- **The question, checked per seed** by `question.py` (`where_tau/QUESTION_16k.txt`, `where_tau/QUESTION_65k.txt`):
  - (i) the tool as it is reads the present byte at d = 1, 2 and 4 above 0.1103;
  - (ii) it reads the present byte above 0.1103 at every d = 8 to 11, with a largest output above 1/2;
  - (iii) the byte one back is above C<L(1) + 0.02 by the tool somewhere;
  - (iv) the byte two back is above C<L(2) + 0.02 by the tool somewhere;
  - (v) two copies differ by less than 1e-6 after 512 bytes.

## What the runs measured, at 16,384 bytes

All readings are by the tool as it is.

| Setting | S(1,0) / S(2,0) / S(4,0) | Present byte at d = 8–11 | Lags 1 and 2 above C<L + 0.02 | Divergence after 512 bytes | Connections a cell |
|---|---|---|---|---|---|
| (1) ratio 1, gmax 8 | 0.98–1.00 / 0.78–0.82 / 0.51–0.53 | 0.09–0.25; above 0.1103 at all four only in seed 116089; largest output 1.0 | none | 0 | 3.3–3.5 |
| ratio 1.25 | 0.87–0.91 / 0.43–0.50 / 0.20–0.25 | 0.04–0.14 | none | 1.0 in two seeds | 4.4–4.6 |
| ratio 1.5 | 0.68–0.73 / 0.17–0.25 / 0.06 | about chance | none | 1.0 in two seeds | 5.1–6.0 |
| ratio 2 | 0.64–0.71 / 0.15–0.22 / 0.04 | chance | none | 1.0 in all seeds | 6.0–6.6 |
| ratio 4 | 0.66–0.68 / 0.12–0.19 / 0.04 | chance | none | 1.0 | 6.7–7.2 |
| gmax 4 | 0.98–1.00 / 0.82–0.85 / 0.58–0.64 | 0.10–0.46; largest output 0.34–0.83 | lag 1 in one seed at d = 11–13 | 0 | 3.4–3.5 |
| gmax 16 | 0.98–1.00 / 0.76–0.82 / 0.42–0.46 | 0.07–0.16 | none | 0 | 3.3–3.5 |
| we 3/4, wi 1/2 | 0.98–1.00 / 0.77–0.80 / 0.40–0.50 | 0.04–0.18 | none | 0 | 3.4–3.6 |
| we 3/4, wi 3/4 | 0.98–1.00 / 0.76–0.83 / 0.35–0.44 | 0.02–0.12 | none | 0 | 3.4–3.6 |
| gmax 8, u2-64e | 0.98–1.00 / 0.84–0.86 / 0.45–0.51 | 0.02–0.07 | lag 1 at d = 2–7 and lag 2 at d = 4–6, all seeds | 0 | 3.3–3.5 |
| gmax 4, u2-64e | 0.98–1.00 / 0.87–0.89 / 0.50–0.56 | 0.03–0.10; largest output 0.15–0.64 | lag 1 at d = 2–8 and lag 2 at d = 3–8, all seeds, and the pool | 0 | 3.4–3.5 |
| gmax 8, u2-64 | 0.97–1.00 / 0.75–0.79 / 0.36–0.39 | 0.02–0.06 | lags 1 and 2 at d = 1–6 and 2–5, all seeds | 0 | 3.3–3.5 |
| gmax 4, u2-64 | 0.99–1.00 / 0.78–0.84 / 0.38–0.41 | 0.03–0.06 | lags 1 and 2 at d = 1–7 and 2–7, all seeds | 0 | 3.4–3.5 |

## Answer at 16,384 bytes

No setting meets the question on all three seeds. Each setting meets one half of it, never both:

- **the far cells:** row (1) and gmax 4 meet (ii) on seed 116089 only;
- **the bytes before:** every setting with u2-64e or u2-64 meets (iii) and (iv) on all three seeds, with (i) and (v), but loses the present byte at the far cells.

Above a prune_ratio of 1 the tissue keeps more connections, and two copies diverge (from 1.25 in two seeds).

The results at 65,536 bytes are below.

## At 65,536 bytes (gmax 4; gmax 4 with u2-64e)

All readings are by the tool as it is.

- **gmax 4:**
  - S(1,0) / S(2,0) / S(4,0): 0.98–1.00 / 0.83–0.85 / 0.56–0.69.
  - Far cells (d = 8 to 11): 0.15–0.22 (seed 73129), 0.10–0.18 (93683) and 0.30–0.38 (116089). The largest outputs are 0.51–0.67, 0.33–0.58 and 0.54–0.76.
  - Condition (ii) holds in seeds 73129 and 116089. In seed 93683 it fails at d = 10 to 11 (S 0.101 to 0.105) and in the largest output at d = 9 to 11.
  - Lag 1 is above C<L + 0.02 only in seed 116089, at d = 11 to 13 and in the pool. Lag 2 is above nowhere.
  - Divergence 0.
- **gmax 4 with u2-64e:**
  - S(1,0) / S(2,0) / S(4,0): 0.99–1.00 / 0.88 / 0.51–0.54.
  - Lag 1 at d = 2 to 8 (to 10) and lag 2 at d = 2 or 3 to 8 (to 10), on all three seeds and in the pool.
  - Far cells: 0.03–0.11, with largest outputs 0.17–0.69.
  - Divergence 0.

The longer life improves the far cells of gmax 4 but does not bring them to the setting with time constants.

## B1, the reviewer's earlier request (`rule=timing norm=1 lines_learn=1 eta_s=1 no_make=1 we=1 wi=1/4`), 16,384 bytes

These were run on two cores that were free, with τ = 2 and with u2-64e (`where_tau/QUESTION_B1.txt`).

- **τ = 2:**
  - S(1,0) / S(2,0) / S(4,0): 0.92–0.95 / 0.60–0.66 / 0.32–0.39.
  - Far cells: 0.14–0.21 in two seeds and 0.07–0.12 in seed 116089, with largest outputs 0.35–0.57.
  - No lags. Divergence 0.
- **u2-64e:**
  - S(1,0) / S(2,0) / S(4,0): 0.92–0.95 / 0.65–0.73 / 0.33–0.41.
  - Lag 1 at d = 2 to 8 on all seeds (to 13 in one), and lag 2 at d = 4 to 7 on all seeds (to 12).
  - Far cells: 0.04–0.13, with largest outputs 0.19–0.46.
  - Divergence 0.

## Answer

No setting run here meets the question on all three seeds.

- **Nearest on the far cells:** gmax 4 at 65,536 bytes meets (i), (ii) and (v) on two seeds. It carries no lag, except lag 1 at the far end in one seed.
- **Nearest on the lags:** every setting with u2-64e or u2-64 meets (i), (iii), (iv) and (v) on all three seeds, and loses the present byte at d = 8 to 11.

The trade between the present byte at the far cells and the bytes before held in every base tried: B, BG, the pruning tissue and B1.
