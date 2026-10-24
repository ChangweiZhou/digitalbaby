# V27 recommendation: does anything move the compatibility frontier?

Written 7 September 2026, after the completed V26 run
(`elm_fly_v26_20260907_111713_46125`, 1350/1350 trajectories, source SHA256
`45fcf4f7…82068b`) and the smaller independent V26 replication
(`v26_results`, 240 trajectories, source SHA256 `159ed699…5c1f7b`).

**The encoder question is answered: temporal contrast is the geometry that survives every
anti-shortcut control, confirmed independently by two runs. The compatibility question is now
correctly characterized as a frontier, and one mechanism in the archive appears to move that
frontier rather than slide along it — but the control that would decide it was not run. V27
should be built around that single question.**

Intervals are descriptive paired Student 95% intervals over independent seed differences,
six seeds, unadjusted for multiplicity.

---

## 1. What the two V26 runs settled

### 1.1 Temporal contrast is the encoder geometry; variance salience is a generator exploit

Terminal accuracy, blended read, across the anti-shortcut task variants:

| geometry | permuted | equal | nuisance | **rotated** |
|---|---:|---:|---:|---:|
| random8 (baseline) | 55.03% | 47.52% | 41.25% | 57.13% |
| variance8 | **71.11%** | — | — | — |
| variance_coverage8 | 64.10% | 49.71% | 40.34% | 58.12% |
| **contrast** | 65.26% | 50.88% | 42.07% | **67.61%** |
| **residual_contrast** | 63.78% | 49.66% | 43.52% | **68.18%** |

The variance family gains +9 to +16pp on the permuted task, falls to +1 to +2pp under
variance-matching and rotation, and goes negative on nuisance. The contrast family gains
+10.5 to +11.1pp on the **rotated** task, where the informative direction is a random
combination that no per-coordinate statistic can isolate.

The independent replication reached the same conclusion by a different route: an asymmetric
fast-minus-slow trace scored **+10.84 [+6.48, +15.19]** structured blended and
**+2.36 [+0.86, +3.86]** on association, with no one-shot cost; its variance-salience arm was
automatically labelled `task_specific_salience_exploit` by a variance-matched control.

Two independently designed experiments, run by different agents, agree on both the positive
and the negative. Adopt contrast/residual-contrast as the encoder default.

**One arm must not be promoted.** `variance8` posts the single highest geometry number in the
archive (73.80% / 71.11%) and was run **only on `structured_permuted`**. Permuting coordinate
indices is not a control against a variance exploit — it relabels which index carries the tell,
and a variance rule still finds it. Its sibling `variance_coverage8`, which does carry the
equal/nuisance/rotated arms, collapses to baseline. The top-ranked geometry is untested against
precisely the controls that would falsify it.

### 1.2 Representation compatibility is a frontier, not a solved problem

Practical arms against their matched frozen ancestor, `shift_structured`, hard read:

| arm | new Δpp | old Δpp | joint gate |
|---|---:|---:|---|
| balanced16_live | +29.40 [+17.06, +41.74] | −26.31 [−31.32, −21.31] | fails |
| live | +23.61 [+10.08, +37.14] | −35.11 [−40.57, −29.64] | fails |
| balanced16_stable | +17.71 [+11.69, +23.73] | −4.94 [−8.58, −1.29] | inconclusive |
| novelty gate | +12.62 [+9.38, +15.86] | −8.87 [−13.43, −4.32] | fails |
| competitive | +11.00 [+6.11, +15.88] | −2.85 [−5.08, −0.63] | inconclusive |
| **random_gate** | +8.80 [+5.03, +12.56] | −2.47 [−4.06, −0.88] | **passes** |
| **rate35** | +8.80 [+5.00, +12.59] | −2.16 [−4.47, +0.15] | **passes** |

Monotone tradeoff: more adaptation buys more new-feature performance and costs more old-memory
compatibility. Only two arms pass both criteria, both at the low-adaptation end.

### 1.3 The novelty gate is an update-budget reducer, not a novelty detector

`practical_random_gate` (a rate-matched random control) passes both gates. `practical_rate35`
(simply a lower rate) lands at the identical frontier point, +8.80 / ≈−2.2. The novelty gate
does not beat either once its retention cost is counted.

The merged V26 plan predicted this exactly — "its benefit could come from reducing the effective
update budget rather than detecting novelty intelligently" — and required the matched control.
The control ran and the mechanism did not survive it. **This is now the standard every selection
mechanism must meet**, including the one recommended below.

---

## 2. The finding neither report surfaced

The causal panel contains a matched factorial that the report tables did not summarize and that
neither review discussed. Address and content pathways are independently frozen (F) or live (L),
with the local Oja rule either dense or competitive. `shift_structured`, hard read:

| arm | new | old |
|---|---:|---:|
| random8_oja_FF | 74.19% | 59.41% |
| random8_oja_FL | 88.89% | 38.12% |
| random8_oja_LL | 93.75% | 33.26% |
| **random8_competitive_oja_FL** | **85.88%** | **58.56%** |
| **random8_competitive_oja_LL** | **89.70%** | **51.31%** |

Paired, matched geometry / path / rate — competitive versus dense Oja:

| path | new Δpp | old Δpp |
|---|---|---|
| FF | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| **FL** | −3.01 [−5.48, −0.54] | **+20.45 [+15.30, +25.59]** |
| LF | −4.40 [−10.66, +1.87] | **+12.27 [+6.99, +17.54]** |
| **LL** | −4.05 [−8.13, +0.03] | **+18.06 [+12.17, +23.94]** |

The FF row is exactly zero — with no adaptation, a plasticity-selection rule cannot act. That is
a clean sanity check that the contrast is well formed.

**This does not look like sliding along the frontier.** On the dense curve, reaching old ≈ 58.6%
requires falling back to FF at new = 74.19%. Competitive FL holds old = 58.56% at new = 85.88%,
**+11.7pp of new-cohort performance at matched retention**.

### What the mechanism is

From the source, `competitive_oja` computes the ordinary Oja proposal for every cell but allows
only the top-k by |response| within each of two 48-cell populations to apply it — k = round(48 ×
0.25) = 12 winners per population, so 24 of 96 cells update per step, at full magnitude.

This is categorically different from lowering the rate. `rate35` shrinks every cell's update;
competitive plasticity shrinks the *number* of cells that update. Each input then rewrites the
feature definitions of a small, input-specific subset, and cells outside that subset keep the
coordinates in which older associations were written.

That is the same compartmentation principle that made banks work in V23–V24, applied to
plasticity instead of to storage. If it holds up, it is the most interesting structural result
the program has produced since the bank architecture itself.

### Why it is not yet established

**The decisive control was not run.** Competitive plasticity updates 25% of cells at full
magnitude. A dense rule at a matched total update budget — roughly `rate25` — was never compared
against it at matched geometry and path. The gating panel's `practical_competitive`
(+11.00 / −2.85) and `practical_rate35` (+8.80 / −2.16) are close enough that a budget-matched
dense control could plausibly erase the difference.

This is precisely the test the novelty gate failed. Competitive plasticity must be held to the
same standard before it is believed.

---

## 3. Corrections to the earlier recommendation

**The Pareto claim was underpowered, not wrong.** The 240-trajectory replication reported
`R1_live` (+7.70 new / +2.48 old) and `R3_stable_scaffold` (+7.03 / −0.08) as Pareto
improvements. At comparable new-cohort gain the larger run measures −2.2 to −2.5pp old loss. With
±5pp intervals on six seeds those are statistically indistinguishable, and a −5pp gate margin let
a real −2.2pp loss read as "no detectable loss."

The design flaw was a **single Oja rate with no rate sweep**. With no high-adaptation arm, the
experiment could not observe that it was a frontier at all. The numbers are not wrong at that
operating point; they are uninformative about the shape of the problem. V27 must sweep the rate.

**Address freezing is conditional, and the direction is now clearer.** V25 found freezing the
address worth +6.67pp on a stream where adaptation bought nothing. Under the shift task, dense
FL (88.89 / 38.12) beats LF (80.09 / 40.97) on new-cohort but not on old — and competitive
plasticity dominates the choice either way. Pathway freezing is a second-order knob compared with
how plasticity is selected.

---

## 4. What is closed

| Mechanism | Result |
|---|---|
| Variance/salience afferent selection | Exploit; collapses under variance-matching and rotation |
| Sensory novelty gate | Does not beat a rate-matched random gate |
| Norm-matched and shuffled gates | Same; no advantage over rate reduction |
| Afferent coverage balancing | +4.48 [−9.75, +18.71] in replication; noise |
| Top-k removal (dense readout) | Helps order-only tasks, costs association 7–10pp |
| Output gain modulation, readout addressing, learning-rate cascades, spillover | Closed in V20–V25 |

Do not spend V27 re-tuning any of these.

---

## 5. V27 design

Small and focused. One question with a decisive control, plus two supporting blocks.

**Substrate held fixed:** 8 locally routed banks, ordinary write rate, no protection, no replay,
no spillover, contrast geometry, float64 screening with one int8 confirmation, hard and blended
reads reported from one state.

### Block A — does competitive plasticity move the frontier? (primary)

Map the frontier deliberately instead of sampling it, then test whether competitive selection
sits off it.

- **A1 — dense rate sweep.** Dense Oja at rates spanning roughly 0.1× to 3× the current base,
  at least six points, at FL and LL. This *is* the frontier; everything else is measured against
  the curve it traces, not against a single comparator.
- **A2 — competitive at matched rates.** The same sweep with competitive selection.
- **A3 — the budget-matched control.** Dense Oja at the rate whose total update mass equals
  competitive's (25% of cells at full magnitude ⇒ approximately `rate25`), at matched geometry
  and path. **This is the arm that decides the question.**
- **A4 — random-subset plasticity.** Update a random 25% of cells per step rather than the
  top-k responders. Separates "sparse plasticity" from "competitive selection" — the analogue of
  the random-gate control that killed the novelty gate.
- **A5 — competition width sweep.** k ∈ {6, 12, 24, 48} winners per population.

**Predeclared prediction, stated before running:** competitive selection sits strictly above the
dense frontier, and A4 (random-subset) recovers part but not all of the benefit. If A3 or A4
matches competitive, the mechanism is an update-budget effect and should be retired exactly as
the novelty gate was.

### Block B — geometry and compatibility are separate axes

The archive shows the geometry that wins structured transfer is not the one that wins
compatibility: on `shift_structured`, `contrast_stable` (+5.09 / −12.65) sits worse on the
frontier than `balanced16_stable` (+17.71 / −4.94). Any combined candidate needs both measured,
never one inferred from the other.

Run contrast, residual_contrast, balanced16 and random8 × {dense, competitive} at two rates on
`shift_structured` and `shift_structured_rotated`. Report the frontier position of each cell.

### Block C — close two gaps in the geometry result

- **C1.** Give `variance8` its missing `equal`, `nuisance` and `rotated` arms so the highest
  number in the archive is either confirmed or retired.
- **C2.** Give the contrast geometry a context arm — it currently cannot clear a context
  non-inferiority criterion because the data was never generated.

### Gates

1. **Primary:** competitive advances only if, at matched total update budget (A3) *and* against
   random-subset plasticity (A4), its old-cohort retention lower bound exceeds both by a
   predeclared margin while new-cohort loss stays within 5pp.
2. Frontier position is reported as (new, old) pairs against the A1 curve, not as a single
   endpoint. An arm "moves the frontier" only if it lies above the A1 curve at matched new-cohort
   accuracy.
3. Non-regression on association final and next-recurrence within 5pp of the contrast baseline.
4. `variance8` is retired unless it holds under rotation.

---

## 6. Not in scope

Language modelling; scaling sweeps; hierarchical, scratchpad or multi-horizon architectures;
dendritic units; further gating heuristics; any re-tuning of the closed mechanisms in section 4.

The scaling question remains genuinely open and genuinely deferred. Measuring a scaling law while
the compatibility frontier is uncharacterized would measure the frontier.

---

## 7. Provenance

- `elm_fly_v26_20260907_111713_46125` — 1350/1350 trajectories, 0 failures. Panels: geometry,
  causal, practical, robustness, live_bedrock, reference, observable, reward, gating, precision,
  teaching. Seeds 601–631 primary, 641–661 confirmation.
- `v26_results` — independent 240-trajectory replication, seeds 401–479 / 487–569.
- `v26_merged_research_plan.md`, `v26_experiment_recommendation.md`,
  `v24_complete_results_review.md`.
- Competitive rule read from `competitive_oja` in `elm_fly_representation_memory_v26.py`;
  paired contrasts computed from `SUMMARY_BY_SEED.csv`.

Two corrections to earlier Claude analysis are recorded here: the Pareto claim (section 3) and,
carried forward, the feature-dilution diagnosis superseded in V26. A third correction applies to
both prior reviews — neither surfaced the causal-panel competitive contrast, which is the
strongest lead in the archive.
