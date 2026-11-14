# E31: lives under the sweep with one switch each (13 November 2026)

Evidence level E2. Two workers, A and B, write the code independently. Not audited by a second reader: Mr. Xu waived the audit step for this round, and the two independent implementations stand in for it. Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu; at most three processes, one thread each.

## Why

Sweep-born networks that then live under the sweep compose untaught operator pairs at 54.2 / 44.5 / 37.2% (seeds 73129 / 93683 / 116089; `exp45/e26/e26_life_S_*.json`, plus-minus). The same births living under backprop reach 72–83% (`exp45/e26x/e26x_life_SB_*.json`). The outside review traces the gap to imprecise holding and undecided gates. E31 asks which part of the backward pass has to be exact to close it, and whether a learned backward pathway, ruled inside the line on 13 November, does the same job.

## The arms

Every arm is the E26 life (`exp26_candidate.life`: schedule, streams, panels, scorer, fresh AdamW at rate 0.001 and decay 0.25, 36,864 updates) started from the admitted sweep birth `exp45/e26/e26_dev_S_<seed>.pt`. Only the backward pass differs. In `arch26.forward` every map goes through `exp3_state.lin(x, layer, local)`, which with `local=True` sends error back through `sign(W) * mean|W|`.

| Arm | Backward pass | Status |
|---|---|---|
| floor | sweep everywhere | on file (E26 life S) |
| ceiling | exact everywhere | on file (E26-X life SB) |
| **G** | exact weights for the reset and update gate rows (rows 0–511) of `weight_hh_l0` and `weight_ih_l0`; sweep for rows 512–767 and every other map. The mean magnitude is still taken over the whole matrix. | new |
| **H** | exact weights for all of `weight_hh_l0`; sweep elsewhere | new |
| **K** | error crosses the recurrent map through a separate matrix `Bhh` (768 × 256); sweep elsewhere | new |

Arm K's rule, in full:

- `Bhh` is drawn once at the start of the life from U(−1/16, 1/16) by `torch.Generator().manual_seed(seed + 31)`. It never reads `weight_hh_l0`.
- After each optimizer step, with `W0` and `W1` the recurrent matrix before and after: `Bhh ← (1 − 0.00025) · Bhh + (W1 − (1 − 0.00025) · W0)`. That is the same additive step the forward weight received, plus its own decay at the same rate (0.001 × 0.25).
- The difference `Bhh − W` therefore shrinks by 0.99975 per update: half in 2,772 updates, about 4.6% of the start left when composition is first taught (update 12,288).
- G and H are bounds, not candidates: they read forward weights. K is the candidate.

## Rules for the two workers

1. New files only: `arch31.py` (the forward with the switch) and `exp31_switch.py` (runner, by the pattern of `exp26x_cross.py`). Receipts say `experiment="E31"`, `arm`, and go to `exp45/e31/e31_life_<arm>_<seed>.json` with the weights; `git add -f` the `.pt` files.
2. Do not read the other worker's branch until both dry receipts below are pushed.
3. Every stage row of arm K records the cosine between `Bhh` and `weight_hh_l0` and `‖Bhh − W‖ / ‖W‖`.

## Stage 1: checks, then the dry run (both workers)

- **No switch.** With the switch off, `arch31.forward` gives the same logits and gradients as `arch26.forward(m, x, True, True)` on one real batch (largest relative difference ≤ 1e-6), and a dry life (`E11_STAGE_UPDATES=256`) ends with the same `tensors_sha256` as `exp26_candidate.py life S 73129` at that dose.
- **G and H.** On one batch, the error sent back across the recurrent map equals `g[:, :512] @ W[:512] + g[:, 512:] @ S[512:]` for G and `g @ W` for H, with `S = sign(W) * mean|W|` (≤ 1e-6).
- **K.** Changing the birth file's recurrent matrix leaves the starting `Bhh` unchanged. The update identity above holds to 1e-7 at updates 1, 10 and 100. With `Bhh` set equal to `W` by hand, K's gradients equal H's.
- **Dry receipts.** Dry lives of G, H and K for seed 73129 at `E11_STAGE_UPDATES=256`, pushed with their `tensors_sha256`.
- **Agreement.** Then fetch the other branch and compare the three hashes. Equal: report and wait for my release. Unequal: stop, both report, change nothing.

## Stage 2: full lives (after my release)

Worker A runs G and H, worker B runs K, three seeds each. Push after each life. About 1.3 CPU-hours per life.

## Predictions (mine; plus-minus composition at the end of life)

- **P31.1** G is at least 15 points above the floor in at least two of three seeds. About 45%.
- **P31.2** H is within 10 points of the ceiling in at least two of three seeds. About 35%.
- **P31.3** K is within 5 points of H in all three seeds. About 60%.
- **P31.4** K's final cosine between `Bhh` and the forward matrix is above 0.99 in all three. About 85%.

## Readings (fixed)

- G closes at least half the floor-to-ceiling gap in two of three seeds: credit into the gates is the bottleneck.
- H does and G does not: the recurrent pathway as a whole is.
- Neither does: the gap is not in the recurrent map's backward pass, and learned feedback there is not the remedy.
- K is read only against H. Differences under 15 points between single lives are not read; a birth's score swings by up to 14 points between neighbouring checkpoints.

## Done

Code, check receipts, dry receipts, nine complete lives, and `E31_REPORT.md` with the end-of-life table (within, plus-minus, F1, F3) per arm and seed beside the floor and ceiling, and the four outcomes. No readings beyond them.
