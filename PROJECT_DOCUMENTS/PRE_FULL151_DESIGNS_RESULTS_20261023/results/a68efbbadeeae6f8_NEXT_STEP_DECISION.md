# Localization round — findings and ONE next step

Date: 2026-09-19. Reporting unit: the 8 blocks. Checkpoints, facts and hybrid
probes within a block are repeated measurements, not extra replicates. Diagnostic
reuse of the existing 64 target-slot worlds in their original disjoint blocks;
this is not an independent confirmation. The accepted reference contrast is
unchanged: canonical 95.5729167%, slot-only 95.8333333%, shared 71.7013889%,
shared − slot-only = −24.1319444 pp.

## 1. Findings about PRESENT EXPRESSION

Native probes on disposable clones at 12–13 immediate checkpoints per block,
scoring the same native target facts and truth in every hybrid.

| state blocks taken from B | mean R | sd | min | max | blocks > 0 |
|---|---|---|---|---|---|
| `slow` | **+23.70 pp** | 4.45 | +19.44 | +30.56 | **8/8** |
| `alpha_fast + slow` | +24.83 | 4.81 | +19.44 | +31.94 | 8/8 |
| all four (full copy) | +24.13 | 4.88 | +18.75 | +31.94 | 8/8 |
| `alpha_fast` | +1.48 | 0.94 | +0.00 | +2.78 | 7/8 |
| `gamma_fast` | +0.00 | 0.00 | +0.00 | +0.00 | 0/8 |
| `adapt` | **−1.56** | 1.47 | −3.47 | +0.00 | 0/8 |

The present **slow-state** difference can account for essentially all of the
expressed loss: `slow` alone recovers 23.70 of the 24.13 pp full-copy rescue.
Consistently, at `rep = 0` the readout sees `alpha_fast` and `slow` only through
their sum, and **93.5%** of the readout-relevant `C − B` difference sits in
`slow` (mean L1 47,094 of 48,999). Shared-arm `slow` carries **1.77×** the L1 mass
of the target-only arm at the final checkpoint (57,013 vs 32,268).

Rescue is not additive and `alpha_fast + slow` slightly exceeds the full copy;
these are not mediated-effect percentages and are not forced to sum to 100%.

`adapt` is a small *negative*: giving C the rested arm's adaptation makes the
present response slightly worse, 0/8 blocks positive.

`gamma_fast` deserves care. It **is** read by `Fly.expression`, and it differs
enormously between arms (L1 ≈ 5×10³ in C versus ≈ 10⁻⁷⁴ in B), yet its
contribution to the scored margin is ≤ 4×10⁻⁹. That is a *measured*
near-inertness of the gamma pathway at the scored quantity — not a
by-construction null, and not evidence about gamma_fast's role in learning.

## 2. Findings about TARGET UPDATE EFFECTIVENESS

`adapt` is not a learning-only state: the source map shows it reaches the probe
through `observed_activity`'s `dx`, and §1 measures that path. The fork therefore
used a **common-controller post-bout probe** — after the bout, `adapt` is set to
the sham fork's vector in both forks, so only what the bout wrote into
`fast`/`slow` is measured.

| window | mean effect | sd | range | blocks > 0 | mean margin effect |
|---|---|---|---|---|---|
| first target acquisition bout | +0.26 pp | 0.74 | [+0.00, +2.08] | 1/8 (7 exactly 0) | +0.0046 |
| first target challenge bout | −2.08 pp | 3.86 | [−8.33, +0.00] | 0/8 (6 exactly 0) | −0.0624 |

**Null.** Supplying the rested arm's adaptation state at the start of a target
learning bout did not improve the target's update effectiveness; the challenge
window is slightly negative. Margin effects are ≤ 0.07 units. A null at two
preselected windows does not rule out historical effects elsewhere in the life.

## 3. Unresolved causes

The transplant identifies an **expressed-state dependence**, not the process that
produced it. Specifically we have NOT shown:

- that the other cohorts' *direct* `slow` writes caused the difference, as
  opposed to altered allocation of the target's own writes, or both;
- that `adapt` played no earlier role — §2 tests two windows, not the whole life;
- that protecting `slow` would improve a multi-cohort learner. Copying clean
  memory into a damaged learner is a diagnostic, not a deployable improvement.

Also unresolved and deliberately untouched this round: whether the B→C effect is
overwrite or finite storage. §1 localizes *where* the difference is expressed; it
does not separate those two.

## 4. ONE next step

**Proposal: raise the magnitude of the local slow-dependent allocation term
(`genes[3]`) in the existing V82E parameterization, and score all eight cohorts.**

Why this one and not another. `allocation_fraction`
(`model_evo.py:193`) already computes the fast/slow split from coordinate-local
pre-event state only:

```
z = log(f0/(1-f0)) + g0 + g1*(2a-1) + g2*tanh(f/sf)*sign(u) + g3*tanh(s/ss)*sign(u)
```

The `g3` term is the only existing lever that makes a coordinate's *own*
accumulated `slow` limit further `slow` writing, which is exactly the quantity §1
identifies as carrying the expressed deficit and which §1 shows grows 1.77× under
sharing. It reads only local pre-event state — no label, cohort role, donor
memory, future relevance, or oracle mask — so unlike the transplant it is a
candidate actual learner change rather than a diagnostic. It is a single scalar
inside the frozen genome family, so it needs no new mechanism, no topology change
and no live-encoding work.

Preregistered scoring, per the standing requirement: score **all eight currently
relevant cohorts**, with the separate revision / acquisition / anomaly endpoints,
against baseline k = 1 and the continuous life; retired-fact loss is not failure;
and a change that rescues the designated target by suppressing learning of the
other seven cohorts is **not** an improvement and must be reported as a failure.
Keep the 5 pp practical reference for the behavioural endpoints and do not convert
oracle rescue into a performance claim.

Preregistered prediction to make it falsifiable: if the expressed deficit is
driven by accumulated shared `slow` mass, a larger `|g3|` should reduce the
shared-arm `slow` L1 ratio below 1.77× **and** reduce the B→C target deficit,
*without* reducing the other seven cohorts' endpoints. Any of those three failing
falsifies the proposal, and in particular a target gain bought with a non-target
loss falsifies it.

If you would rather not change the learner yet, the equivalent single remaining
**test** is the one ambiguity §3 leaves open: hold the *amount* of non-target
learning fixed while varying whether it competes for the same coordinates, which
separates overwrite from finite storage. We did not pick that unilaterally
because it determines which M-families the following run can speak to.

Stopping here. No dose sweep, order-reversal panel, topology change, family
tournament, or evolution run was started, and none is proposed in this round.
