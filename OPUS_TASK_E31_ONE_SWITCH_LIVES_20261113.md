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

## Addendum: arm P, a local push on the update gates (13 November 2026, after Mr. Xu's ruling)

Mr. Xu ruled on 13 November that a term each unit computes from its own gate value, which nudges the gate away from one half, is inside the line. Arm P is added. Both workers write it; worker B runs its three lives after K.

**Arm P.** The sweep everywhere, exactly as the floor, plus one extra term on the update-gate rows only (rows 256–511 of `weight_ih_l0`, `weight_hh_l0` and their biases):

- At each position the update gate is recomputed from detached inputs: `a = lin(z1.detach(), ih)[z rows] + lin(hp.detach(), hh)[z rows]`, `z = sigmoid(a)`.
- The term is `−λ · mean((z − 0.5)²)` over items, positions after the first byte, and units, added to the loss. Because the inputs are detached, its gradient reaches only those rows: each weight changes by its own input times `(z − 0.5) · z · (1 − z)` of its own unit. Nothing crosses a unit or a time step.
- `λ` is set once, before the first update, so that on the first batch the term's gradient norm on those rows is 0.25 of the task loss's gradient norm on the same rows. It then stays fixed. Record `λ` and both norms in the receipt.
- The reset gate gets no such term.

**Checks (stage 1).** With `λ = 0` a dry life equals the no-switch dry life bit for bit. The term's gradient is zero on every tensor and row outside the update-gate rows. On one batch the gradient equals the written two-factor product (≤ 1e-6). A dry life of P for seed 73129 joins the agreement check; both workers must get the same `λ` and the same `tensors_sha256`.

**Every stage row of arm P** records the undecided share U as E30 defines it, on the plus-minus panel.

**Predictions (mine).**

- **P31.5** At the end of life U is at most 60% in all three seeds. About 70%.
- **P31.6** P is at least 10 points above the floor on plus-minus in at least two of three seeds. About 30%.

**Reading.** P31.5 holds and P31.6 misses: undecided gates go with the gap but do not cause it. Both hold: decisive gating is a remedy that needs no feedback at all, and it is run again on fresh seeds before anything is claimed.

## Addendum 2: stage 1 read, rulings on the two disagreements, and the release (14 November 2026)

I read both stage 1 branches (`claude/sleepy-knuth-q96dd1` at bc2ddfa9, worker A; `claude/charming-clarke-u9g850` at 96024a77, worker B) and `E31_AGREEMENT_B.md`. No switch, H, K and the zero-strength push agree bit for bit and equal the E26 life at the dry dose. Two arms differ. Both workers stopped correctly.

### Rulings

1. **G: accepted as agreeing.** Both evaluate the rule the brief wrote; one uses two matrix products and one a stacked matrix, and float32 rounding (about 7e-7 per crossing) compounds into a different hash. From here on, agreement between the two implementations means equal hashes, **or** gradients on one common batch within 1e-6 relative when each worker runs the other's function. Worker A's form (two products, as the brief's formula is written) is the one that runs.
2. **P: padding is excluded.** Padding is not data. Worker A's mask (`x[:, 1:] != X.BOS`) is the definition.
3. **P as written is withdrawn before any full life.** In worker B's dry run it took the undecided share U from 65% to 9% within 1,536 updates, far past backprop's 43–53%. That would test whether over-saturated gates hurt, not whether decisive gates help. P31.5 and P31.6 are withdrawn unjudged. The replacement is arm P2 below.
4. **K stays as registered**, with its random start. At the dry dose its feedback is still far from the forward matrix (cosine 0.73 after 1,536 updates), so its early stages are read separately: K is compared with H at the end of life only, as P31.3 says.
5. **`exp16_integrated.py`.** The births were made with it at a301ae8e; the branch has it at deea398c. Worker A records an admission exception and worker B runs with the older file in a work folder; both give the floor's hash. Each keeps its own way and records it in every receipt.

### Arm P2: a push that stops at a limit

As arm P, with the mask of ruling 2, and this term in place of the old one:

`λ · sum(relu(0.4 − |z − 0.5|)²) / (masked positions × 256)`

- It is zero for any gate outside 0.1 to 0.9, so a gate that has become decisive is left alone.
- Three strengths, each set once on the first batch as the ratio of the term's gradient norm to the task's on the update-gate rows: **0.25, 0.05 and 0.01**. Arms `P2a`, `P2b`, `P2c`.
- Every stage row records U on the plus-minus panel.

### Arm Ks: Kolen–Pollack feedback started from the sweep's own matrix (code and dry run now; full lives wait for Mr. Xu)

As arm K, but `Bhh` starts as `sign(W) · mean|W|` of the birth's recurrent matrix, which is what the sweep sends error through at its first step. It reads the forward weight once, at the start, so it is outside the letter of the fifth ruling; Mr. Xu decides whether it runs.

### Stage 1b (both workers, short)

- Dry lives at `E11_STAGE_UPDATES=256`, seed 73129, of P2a, P2b, P2c and Ks. Push hashes and the three λ values.
- Then compare by the rule in ruling 1. Agreement: go straight to stage 2 without waiting for me. Disagreement: stop and report.

### Stage 2: full lives, three seeds each

| Worker | Lives |
|---|---|
| A | G, H, P2b (nine) |
| B | K, P2a, P2c (nine) |

Push after each life. Ks is not run until I say so.

### Predictions for the new arms (mine)

- **P31.7** At strength 0.05 the end-of-life U lies between 35% and 65% in all three seeds. About 40%.
- **P31.8** At least one of the three strengths is 10 points or more above the floor on plus-minus in at least two of three seeds. About 35%.
- **P31.9** Plus-minus does not rise with strength: the strongest push is not the best of the three in at least two of three seeds. About 60%.

Choosing the best of three strengths after the fact flatters it, so a strength that clears P31.8 is run again on fresh seeds before anything is claimed.
