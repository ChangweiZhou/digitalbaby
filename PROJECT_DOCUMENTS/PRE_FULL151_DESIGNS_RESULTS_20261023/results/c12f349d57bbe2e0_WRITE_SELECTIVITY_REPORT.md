# Write selectivity: headroom exists, this local selector cannot find it

Date: 2026-09-19. Panel: the eight existing blocks, **development reuse**, not
confirmation. Blocks are the reporting unit; windows, facts and checkpoints
within a block are repeated measurements. Reference subject FULL_151 unchanged;
`b_slow = -0.55` remains a rejected replacement.

## Outcome: case 2 of the stopping boundary

**Oracle headroom exists. The tested local selector fails.** Deliverable C was
therefore **not run** — that is an unrun task, not a result.

## D/H closure, independently recomputed

Read-only from the eight uploaded ledgers, no rerun: `phi_D` mean
**-9.902222116818454**, `phi_H` mean **+2.659175085314679**, total
**-7.243047031503775**; max `|R|` and max `|s_C-(s_B+D+H+R)|` = **4.889e-12**.
All match the brief. No further D/H campaign was run.

## Deliverable A — is there a better allowed write?

Write family, exactly as specified: `slow_i = p'_i u_i`, `fast_i = (1-p'_i) u_i`,
`0 <= p'_i <= 1`, on existing support, with the intended slow-deposition L1 budget
held equal to native, `sum p'|u| = sum p|u|` (an evaluator diagnostic constraint,
not a proposed normalisation mechanism). Direction proposed by a sparse linearised
screen `delta M_q = r_q^T delta s` at fixed non-slow state and clipping region,
solved exactly as a fractional knapsack; **every reported number is a native
probe**, immediately and after one prespecified matched 24 h rest on clones.

Windows fixed before outcomes: cohort positions 0/3/7, the first acquisition and
last challenge teach call of each, 6 per block, 48 total. Acquisition windows are
taken only after the first `set_relevance_context`, because before it the
relevance context is empty and the protected set is empty by definition.

### Per-event scale: no useful headroom

| | protected ΔMargin | protected ΔCorrect | target ΔMargin |
|---|---|---|---|
| immediate | +0.0001 (48/48 >0) | +0 | +0.0004 |
| after 24 h rest | +0.0283 (48/48) | +9 | +0.1903 |

### Bout scale: headroom, and it does not cost the incoming fact

| | protected ΔMargin | protected ΔCorrect | target ΔMargin | target ΔCorrect |
|---|---|---|---|---|
| immediate | **+2.8771** (48/48 >0) | **+287/4494 = +6.386 pp** | +4.5992 | +8 |
| after 24 h rest | **+3.5303** (48/48) | **+349/4494 = +7.766 pp** | +5.3726 | +7 |

Per block: +7.130, +5.151, +6.595, +7.651, +6.940, +6.584, +4.626, +6.417 pp —
**8/8 positive**, mean +6.387 pp against a native 67.690%. The incoming target
never loses (+8 decisions overall). So this is headroom in the required sense: it
is not tiny and it does not sacrifice incoming learning.

### Why the two scales differ — write-support / readout reachability

| scale | coordinates touched | share of PROTECTED readout sensitivity | share of TARGET |
|---|---|---|---|
| one plastic event | 47 of 5177 | **0.0114** | 0.361 |
| whole bout | 931 of 5177 | **0.682** | 1.000 |

A single event writes ~0.9% of coordinates, carrying ~1% of the protected facts'
readout sensitivity and depositing ~0.2% of the accumulated slow L1. There is
essentially nothing to trade within one event. Across a bout the same write
support reaches 68% of the protected readout, and the trade becomes available.
**Selectivity is a bout-scale property here, not an event-scale one.**

This narrows the tested write family only. It is not a capacity theorem, and a
strong oracle result demonstrates local headroom, not good long-life behaviour.

## Deliverable B — is the selection locally observable?

One transparent candidate, fixed before outcomes, with a one-statistic complexity
budget: deposit the **same** budget into the **least-loaded** coordinates first,
ordering by ascending `|slow_i|`. Permitted signals only — the coordinate's own
pre-event slow state and the proposed increment. No cohort or target id, no
evaluator truth mask, no future relevance, no donor state, no D/H vector. It has
no fitted parameters, so nothing was tuned on this panel.

| | protected ΔCorrect | blocks positive | target ΔMargin |
|---|---|---|---|
| immediate | **-12/4494 = -0.267 pp** | **2/8** | -0.6540 |
| after 24 h rest | -17/4494 = -0.378 pp | 2/8 | -0.5981 |

**It fails.** It captures none of the ~+6.4 pp oracle gain, sits at chance in sign
across blocks, and slightly costs the incoming fact. A failed selector does not
prove all local signals are insufficient; it is grounds to stop this candidate and
document the gap.

## The ONE precisely identified constraint

The quantity the oracle exploits is `r_p[i]` — how strongly the **other currently
useful facts'** responses depend on coordinate `i`. From the source,

```
r_q[i] = dir_q * mean_c( unclipped[.,c] * dx[.,i] * Q[i, 2+c] )      (rep = 0)
```

so `r_p[i]` is determined by *those other facts'* `dx` (their codes, gated by
adaptation) and by row `i` of the readout matrix `Q`. It is a property of **which
other facts read from coordinate i**, and it is not a function of coordinate `i`'s
own pre-update state — not its `slow`, `fast`, `adapt`, nor the proposed increment
`u`, which is the entire information the native update rule has. That is the
observability gap, stated structurally rather than inferred from one failure.

If an additional observable were ever justified, the minimal one implied by this
is a per-coordinate *readout-load* statistic — how much previously-correct
readout has depended on that coordinate — accumulated from the coordinate's own
participation, with no labels, no cohort identity and no future relevance. I am
**not** starting it: the brief forbids an automatic rule search or topology change
on this outcome, and whether such a statistic is admissible and bounded is a
design question for you.

## Corrections and qualification

**A sign error in my own objective, caught before reporting.** Collateral harm is
`-delta M_protected`, so minimising it means maximising `+lambda * delta
M_protected`. An earlier draft subtracted that term and therefore rewarded
coordinates that *damage* protected facts; at bout scale it produced **-39**
protected decisions. That was a bug in the objective, not a finding. Both the
event and bout arms now use the derived sign, and the fix is recorded in the code.

Qualification, all 48 windows × 8 blocks:

- `p' = p` reproduces the native event **exactly** (post-event slow bit-identical);
- budget equality exact (`realized == intended` to 1e-9 relative);
- the reference trajectory is never modified — all arms run on disposable clones
  restored from a snapshot of the reference shared-C history;
- native decay order preserved: only `allocation_fraction`'s return value is
  supplied, and every native assertion in `event()` (`split_error`, sign, and
  unsupported-plasticity) passes;
- intended allocations and **actual net innovations** both recorded; they are not
  assumed equal;
- pre-teach slow is never restored, so decay is never cancelled.

## Scope

Source labels are evaluator bookkeeping. D being harmful in aggregate does not
make non-target learning unwanted: every cohort is a legitimate objective when
currently relevant, and nothing here licenses a rule that silences other cohorts
or uses their identities. The oracle is a headroom diagnostic and is not
deployable. Deliverable C, the k = 1 and continuous-life guardrails, and fresh
held-out worlds all remain unrun.
