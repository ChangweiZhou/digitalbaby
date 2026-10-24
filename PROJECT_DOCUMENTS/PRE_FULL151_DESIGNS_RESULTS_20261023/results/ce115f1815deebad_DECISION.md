# Decision after the response-time audit

Date: 2026-09-19. Panel: the eight existing blocks, **development reuse**, not
confirmation. Blocks are the independent unit; the 384 P7 decisions and the 64
cohort cells are not independent samples.

## Primary verdict: UNCHANGED — REJECT_AS_CORE_REPLACEMENT

P1 current usefulness is **+0.2387 pp** [−0.2814, +0.7588], 4 blocks down / 4 up.
P1 was the declared primary before any of this and is **not** re-chosen after
seeing P7. The candidate `b_slow = −0.55` is preserved as a development
perturbation, not deleted and not promoted.

## What changed: my previous report was incomplete

Every cell already saved all of P1–P7 in `v88_metrics`. I displayed a
five-endpoint extractor shortlist and omitted P2, P3 and **P7**. Recomputed from
the same 16 saved cells, no new simulation:

| endpoint | ref % | cand % | diff pp | 95% CI | blocks −/+ |
|---|---|---|---|---|---|
| P1 current usefulness | 70.6163 | 70.8550 | +0.2387 | [−0.2814, +0.7588] | 4/4 |
| P2 relevant after write | 67.5967 | 66.5179 | **−1.0789** | [−1.8450, −0.3127] | 6/2 |
| P3 revised relevant after write | 68.5268 | 67.4665 | −1.0603 | [−2.3210, +0.2005] | 5/1 |
| P4 revision | 74.0625 | 76.8750 | +2.8125 | [+1.8055, +3.8195] | 0/8 |
| P5 acquisition | 59.3192 | 59.4959 | +0.1767 | [−0.5448, +0.8983] | 4/4 |
| P6 anomaly recovery | 73.3594 | 75.0000 | +1.6406 | [−0.0337, +3.3149] | 2/6 |
| **P7 reactivation after 4** | 73.9583 | 83.5938 | **+9.6354** | [+4.9863, +14.2845] | **0/8** |

P7 is 284/384 versus 321/384. Post-hoc development finding, on a reused panel,
with a lower nominal bound slightly below +5 pp. Not a confirmed population effect.

P7's definition was verified in the **local executing** V88 (`49b19beb9248c3a8`):
it pools `react_events` at `exposures == 4`. In the transformed FPF8 world there
are 24 reactivation probes at each of 0/1/2/4, 48 episodes at exposures 4 (= 6k),
every exposure-4 fact appears in `reactivation_plan`, and label-versus-actual
renewed exposures inside the return challenge mismatched in **0 of 96** events. So
these are renewed informative exposures per returning fact, not total-life
exposures, duplicated micro-probes, or a rescore of zero-exposure facts.

## The question answered: faster reacquisition, not better first return

16 replayed trajectories (8 blocks × reference/candidate), each reproducing its
saved P1–P7 counts exactly.

| exposures | ref % | cand % | diff pp | 95% CI | blocks −/+ |
|---|---|---|---|---|---|
| 0 (first return, before renewed teaching) | 46.3542 | 43.4896 | **−2.8646** | [−6.463, +0.734] | 6/2 |
| 1 | 59.8958 | 57.5521 | −2.3438 | [−5.492, +0.805] | 5/1 |
| 2 | 65.1042 | 69.5312 | +4.4271 | [+0.218, +8.636] | 2/5 |
| 4 | 73.9583 | 83.5938 | +9.6354 | [+4.986, +14.285] | 0/8 |

There is a **crossover**. The candidate is *worse* on first return and overtakes
between exposures 1 and 2. Mean signed margins cross in the same place:
+0.3296 vs −0.1896 at exposure 0, 3.7358 vs 3.5346 at 1, 5.6671 vs 5.9128 at 2,
8.6182 vs 9.1229 at 4. By the stated rules this is an **acquisition/update-rate
difference on returning facts**, not better first return — the gain is bought
after renewed teaching begins, at a small cost before it.

## The advantage largely survives a matched rest

One native 24 h REST applied to a disposable clone at each exposure-4 return
probe, no extra teaching and no truth change:

| | ref % | cand % | diff pp | 95% CI | blocks −/+ |
|---|---|---|---|---|---|
| exposure 4, +24 h rest | 66.9271 | 73.6979 | **+6.7708** | [+2.139, +11.402] | 0/8 |

Both arms decay (−7.03 and −9.89 pp), and the candidate retains about 70% of its
advantage. By the stated rules this is **not** a disappearing short-timescale
effect; it motivates a *scoped persistent-relearning hypothesis* and nothing more.
It is explicitly not the unrun k = 1 / continuous-life guardrail.

Clone qualification, 16/16 trajectories: no writable-buffer aliasing, originals
unchanged (`array_digest` before/after), zero-delay clone reproduces the native
probe, clone retains the genotype override, frozen anatomy, **zero** extra teach
operations issued.

## Corrections to my previous report

1. "Nothing reaches 5 pp" was true only of the shortlist I displayed. P7 reaches
   +9.6354 pp and was in the data I shipped.
2. "No trade-off" is wrong. P2 −1.0789 and P3 −1.0603; cohort-position effects run
   −7.6389 to +7.6389 pp; and the new ladder shows a first-return cost.
3. A null P1 at one coefficient value does not rule out the allocation family.
4. My 66.97% / 30.00% `tanh` histogram is a **final-state** histogram over all
   5177 KCs, not update-weighted over coordinates actually receiving nonzero
   updates. It supports a sign-switch description at that sampled state only.
5. Terminal slow L1 ×0.6756 is not an isolated intervention on norm; allocation
   changes fast state, slow state, correction dynamics and later history.
6. Conserving the instantaneous alpha update does not conserve future response:
   with distinct taus, `u*((1−p)e^{−l/τ_f} + p·e^{−l/τ_s})` depends on `p`.

On their point about terminal diagnostics: measured **at the scored return
events**, not terminally, `alpha_fast_L1` is 3433 (ref) vs 3549 (cand) at
exposure 0 and 3430 vs 3506 at exposure 4, while `slow_L1` is ≈55,000 vs ≈40,000.
So fast compensation is present at the scored events, and small.

Position analysis reproduces independently: mean by position
[−1.1285, −1.2153, −0.0868, −0.4340, −0.8681, +0.4340, +1.3021, +3.9063], final
position positive in 8/8 blocks, 31 cohort cells improving, 25 declining, 8
unchanged. Post-hoc; position is a real scheduling distinction in the transform,
but this does not establish recency as causal.

## One next action

**Decompose the crossover with the already-qualified hybrid machinery, at the
exposure-0 and exposure-4 return probes, reference ↔ candidate.**

The two halves of this result — a first-return cost and a relearning gain — may be
carried by the same store or by different ones. The localization round's hybrid
apparatus already answers that kind of question, is bit-identity qualified, needs
no new worlds, no new coefficient and no sweep, and costs minutes: clone the
candidate at a return probe, take `slow` (or `alpha_fast`) from the aligned
reference, and see which rung each store moves.

If one store carries both, the trade is intrinsic to re-routing within a conserved
alpha budget and this family is unpromising for P1. If they separate, a policy
that keeps the relearning gain without the first-return cost is worth specifying.

I am not proposing promotion, a new coefficient, a sweep, new worlds, the k = 1 /
continuous-life guardrails, an 8×8 matrix, or topology work. The k = 1 and
continuous-life guardrails remain **unrun**, which is a status, not a result.
