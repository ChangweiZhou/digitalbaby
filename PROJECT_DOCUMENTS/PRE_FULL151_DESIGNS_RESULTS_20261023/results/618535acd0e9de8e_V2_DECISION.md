# v2 verdict: bounded useful headroom, with individual losses on both sides

Date: 2026-09-19. Panel: the 8 existing blocks, the same 48 exact windows,
**development reuse**. Blocks are the unit; windows and facts within a block are
repeated measurements. FULL_151 unchanged; no new worlds; no D/H rerun; no
local-rule sweep; Deliverable C still deferred.

## The audit was right on every point, and I verified each before rewriting

| finding | verified how |
|---|---|
| 6/48 anchors used the **opposite** answer to the one delivered | reproduced exactly: blocks 1/1/3/4/5/6, facts 668, 559, 996, 952, 1240, 340 |
| bout objective frozen to the first anchor | lines 375–381 set the sets; the loop starts at 396 |
| soft sum is not a constraint; "target never loses" false | 7/48 immediate and 6/48 delayed anchor-margin decreases, and **one correctness loss** (block 04, cohort 3, +24 h) |
| qualification overextended from one event to the bout | v1 recorded budget/identity for one plastic event only |
| wrong surrogate scale | correct coefficient `exp(−dt/τs) − exp(−dt/τf) = 0.00884545` vs the `0.99999756` I used — **113.1× off** |
| `--resume` parsed, never read | single occurrence at line 489, never referenced |

Finding 1 is the one that should not have happened: it is the same
`change_truth`-is-a-commit-not-an-onset defect I had already found and fixed in
the metrics extractor, reintroduced in a new file. The report wording
"48 windows × 8 blocks" was also wrong; there are 48 windows in total.

## What v2 changed

Incoming objective and scoring use each teach call's **delivered** role, with
committed truth tracked separately; the **full** incoming set is scored (24
acquisition / 12 challenge facts per window, 864 total) instead of one anchor;
a per-event incoming noninferiority constraint with a frozen λ ladder
{1, .75, .5, .25, 0} and the native allocation as its last rung replaces the soft
sum; the surrogate uses the fast+slow sum coefficient; budget and identity are
recorded at every changed update and for the full window; `--resume` works, keyed
on source identity with atomic per-window records.

Verified rather than assumed: no `set_relevance_context` and no `change_truth`
occurs inside any of the 48 windows, so computing populations once per window is
exact. Tracking the whole incoming set reveals the delivered/committed mismatch in
**24** windows, not 6 — v1 saw it only where it hit the single anchor.

## Results

| timing | population | Δcorrect | pp | blocks +/− | 95% CI | facts worse / better margin |
|---|---|---|---|---|---|---|
| immediate | protected | +241/4320 | **+5.579** | 8/0 | [+4.645, +6.513] | 771 / 3491 |
| immediate | incoming (delivered) | +19/864 | **+2.199** | 6/0 | [+0.654, +3.744] | 290 / 566 |
| +24 h rest | protected | +294/4320 | **+6.806** | 8/0 | [+5.633, +7.978] | 772 / 3475 |
| +24 h rest | incoming (delivered) | +18/864 | **+2.083** | 7/1 | [+0.340, +3.827] | 278 / 574 |

Scored against committed truth instead, incoming is +2.199 pp immediate and
+1.968 pp after rest — the two agree in sign and size.

## Verdict: bounded useful headroom

Both populations improve on net, in the same direction, with block-level intervals
excluding zero, and the protected effect reaches the 5 pp development reference
while the incoming population is not sacrificed. That is headroom in the required
sense — and unlike v1, the incoming claim is now *measured* on all 864 taught
facts rather than asserted from one anchor.

It is bounded, and these limits are part of the result:

- **Individual losses exist on both sides**: 771/4320 protected and 290/864
  incoming facts end with a worse margin. "No sacrifice" is true of the net, not
  of every fact, and I will not repeat v1's stronger phrasing.
- The incoming constraint is enforced **per event against a same-state shadow**.
  λ = 1 was admissible in 76.2% of events; the ladder bound the rest. As the audit
  notes, an event-level guarantee is not a guarantee against the unmodified
  full-window reference — and indeed 290 incoming facts end worse despite 0
  per-event violations. Endpoint behaviour is reported from native probes, not
  inferred from the constraint.
- The allocation is an **oracle**: it uses protected-fact readout sensitivities.
  It is a headroom diagnostic and is not deployable.
- Per-branch cumulative budgets are recorded and intended equals chosen at every
  update, but same-state matching does **not** make the two diverged branches
  spend identical totals; that is a different question and is not claimed.
- Development panel. Not held out, not confirmation, not a P1 result.

## Qualification

48/48 windows: **full-window `p' = p` reproduces the native branch exactly**
(post-state digests identical), **0** per-event violations across all bounds,
support, sign, budget and shadow-`u` replay checks, cumulative intended budget
equals chosen, and the reference P1–P7 counts reproduce the prior intervention
archive in all 8 blocks. Four synthetic fixtures pass and each one **fails on v1's
logic**: stale truth, population construction, multiple incoming facts, and the
soft-objective counterexample (v1's solver harms incoming; v2's rejects it and
preserves the budget exactly).

## One next step

Nothing in this round licenses a local selector claim: v1's `|slow|` heuristic was
retired as unsuccessful in that assay and v2 deliberately did **not** rerun a
local-rule sweep, per the spec. The single next action is therefore the one the
repaired result now makes answerable: **test whether any permitted local ranking
can approximate this constrained oracle**, using the v2 objective and scoring as
the assessment target, with the feature set and complexity budget fixed in advance
and blocks split whole.

I am not starting it, not proposing a new state variable, not inferring an
observability theorem from v1's single failed heuristic, and not touching the core.
Deliverable C, the k = 1 and continuous-life guardrails, and fresh held-out worlds
all remain unrun.
