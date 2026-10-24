# Decision on the slow-allocation candidate

Date: 2026-09-19. Panel: the eight existing target-slot blocks, **explicitly
reused as a development panel** — not held out. Eight blocks are the reporting
unit; facts, checkpoints and cohorts within a block are repeated measurements.

## Verdict: REJECT this candidate as a core change

`b_slow` −0.394336 → −0.550000, reference shared FPF8 versus candidate shared
FPF8, identical worlds, teaching, clock, cues and observations, all eight
currently relevant cohorts scored.

| endpoint, all cohorts | candidate − reference | 95% CI | blocks −/+ | ≥ 5 pp |
|---|---|---|---|---|
| **current usefulness** (primary) | **+0.24 pp** | [−0.28, +0.76] | 4/4 | no |
| acquisition | +0.18 pp | [−0.54, +0.90] | 4/4 | no |
| anomaly recovery | +1.64 pp | [−0.03, +3.31] | 2/6 | no |
| revision | +2.81 pp | [+1.81, +3.82] | **0/8** | no |
| stable retention | +1.80 pp | [+0.21, +3.39] | 1/7 | no |
| designated target only | +0.78 pp | [−2.48, +4.04] | 1/5 (2 zero) | no |

Development success required a meaningful improvement across currently relevant
material against the 5 pp practical reference. The primary endpoint moved
**+0.24 pp**. Nothing reaches the threshold, so the candidate fails the screen.

There is also no compensating decline: no endpoint is negative on average, and the
worst single block on the primary is −0.43 pp. This is a clean null on the
primary, not a trade-off.

Because the candidate did not survive the development screen, **Task 4 was not
run**. No k = 1 or continuous-life guardrail comparison was performed, and no
held-out worlds were opened. That is an unrun task, not a result.

## The informative part: the norm story is falsified

The knob did what the source map said it would. Slow L1 fell to **×0.6756** of
reference (range 0.636–0.707 across the eight blocks, so highly consistent) — a
32% reduction in accumulated slow mass. And the primary endpoint did not move.

Per the standing rule, lower norm with unchanged performance is **not** success.
More importantly it is evidence: the localization round showed that the *expressed*
deficit tracks the slow state in both directions, but this round shows that
**reducing accumulated slow mass by a third via local allocation does not recover
it**. The deficit is therefore not simply "too much slow mass", and a
norm-reduction account of the shared-learning deficit is weakened.

Two sub-threshold effects are worth recording rather than discarding, because they
are consistent in sign across every block or nearly so: revision **+2.81 pp,
8/8 blocks, CI excluding zero**, and stable retention **+1.80 pp, 7/8**. Eight
blocks on a reused development panel cannot certify effects of this size, and they
are below the practical reference, so they justify no claim — only a note that
redirecting deposition from slow to fast appears to help re-learning a changed
answer slightly while leaving present usefulness flat.

## Correction carried forward

My localization documentation called the passive slow decay negligible over a
life. That was wrong: survival is 0.6001498 over the full 6,289,920 s FPF8 life
and 0.7232362 across the 12,096 non-target slots. Corrected in
`STATE_DEPENDENCIES.md` and in the signed check.

## What is supported, and what is not

Supported: the candidate is a genuine one-coefficient change with bit-identical
anatomy and readout (the sole differing numeric array is `agent/genes`), the
zero-perturbation path reproduces the reference exactly, and it uses only
coordinate-local pre-event state with no oracle input.

Not supported: any claim that slow-allocation control improves the shared-learning
deficit; any mechanism verdict from the 5 pp reference; any promotion of this
candidate; and any generalization beyond this reused development panel.

Also revised from my previous brief: `b_slow` is **not** graded negative feedback.
With `ss = 1.0`, 67% of coordinates sit at `slow = 0` where the term is exactly
inert and 30% are fully `tanh`-saturated, so it is a sign-dependent constant
offset on already-loaded coordinates. I had proposed it as graded protection; that
description was wrong even though the signed direction was right.

## At most one subsequent action

**Do not tune this knob further, and do not start a sweep.** The result argues
against the family of interventions that only re-route deposition between the
existing fast and slow stores, because total alpha `u` is conserved on that path
(`actual_fast + actual_slow = u`, asserted natively). Re-routing changed the slow
norm by a third and moved present usefulness by 0.24 pp.

The single remaining question I would put next is the one the localization round
left open and this round did not touch: whether the shared-state deficit is
**overwrite of the target's slow coordinates by other cohorts** or **finite
capacity of the shared store** — tested by holding the *amount* of non-target
learning fixed while varying whether it competes for the same coordinates. This
round makes that more, not less, worth doing: if re-routing within a conserved
budget cannot help, the binding constraint is more likely to be *where* writes
land than *how they are split*, which is exactly what that contrast separates.

I am not proposing it unilaterally, because it decides which of M1–M10 the
following run can speak to, and that is your call. Nothing further was started:
no sweep, order reversal, live-topology engineering, repeated parameter change, or
confirmation job.
