# MiniFly V9.10E — the cyclic failure is not a zero-overlap obstruction

**Result:** A byte-for-byte replay of V9.10D reproduces its old-end head margins to at most **3.51×10⁻¹⁶** across all 64 worlds. The exact NLMS weight/margin accounting closes at machine precision. The 288 actual teaching inputs span **37 effective directions** at all three prespecified eigentolerances (`1e-8`, `1e-10`, `1e-12`). Held-out readouts have nonzero projection into that span, and a *held-out-label oracle* can orient all nine unseen pairs using a vector in the span in **64/64 worlds**. Thus a strict `c_i ⟂ U` impossibility does not explain V9.10D's failure. This oracle uses withheld answers and unlimited coefficient magnitude; it is a reachability diagnostic, not a learner.

![V9.10E key result](results/KEY_FIGURE.png)

## Frozen replay and exposure

This is a **post hoc E0/mechanism diagnostic** of V9.10D's already-known E3 failure, not a new held-out confirmation. The same 64 worlds (2101–2164), 288 ordered 12-byte old teaching records per world, V9.2/FULL_151 byte/KC parent, and unchanged V9.10C online head were replayed. The head received only each actual pre-outcome `pending_x` and its neutral/aversive feedback. Nine scored cyclic pair relations were never reinforced; no withheld label entered the online update, fixed ridge fit, or label-sum fit. Read-only probes were made at the same old-end state as V9.10D. No per-record vectors or parent checkpoints were copied into this result bundle.

V9.10D's old-teaching/no-old-teaching causal comparison remains the behavioral control: online old-end withheld choice **51.04%** versus **50.00%**, paired +1.04 percentage points [−3.23, +5.32]. An exact taught-pair table has no withheld answers and ties at 50%; any scalar per-symbol score is bounded by 6/9 on the three directed withheld triangles. The analytic three-class solver from taught labels can answer all nine, but was never installed. This replay adds *no new behavioral evidence level*.

## What the teaching inputs can reach

Let `X` be the 288 actual teaching KC vectors, `U=span(X)`, and let each old-end held-out readout be `r=x_neutral−x_aversive`. The mean ratio `||P_U r||/||r||` was **60.24%** [59.47, 61.02] across worlds. The weakest of the three true cycle-sum directions per world had mean projection ratio **42.73%** [41.38, 44.09]. These results and rank 37 were unchanged at relative eigentolerances `1e-8`, `1e-10`, and `1e-12`. The cycle direction is therefore not orthogonal to the actual teaching stream. A linear-program oracle using the **hidden held-out orientations** found a vector inside `U` that makes all nine held-out margins positive in every world. Feasibility alone says nothing about whether the *given feedback sequence* selects such a vector, nor whether it has a physiologically or computationally reasonable norm.

For the frozen head, the update `w_{t+1}=w_t+α_t ε_t x_t` was reconstructed from the actual sequence. Maximum per-step vector error was **1.89×10⁻¹⁷**, final weight reconstruction error **1.52×10⁻¹⁶**, and fixed-endpoint held-out margin-sum error **3.37×10⁻¹⁶**. The queried readout vector was held fixed at the old-end state in this identity; using a changing readout at earlier times would require an additional code-change term. Summing increments is exact accounting of the observed trajectory, **not** the counterfactual effect of removing an earlier event.

## Teaching-label-only probes

The one pre-fixed external ridge diagnostic fits **only** the actual 288 teaching `(x_t,y_t)` pairs, with RMS-L2 feature scaling and dimensionless ridge `λ=0.1`. Its weights never see the nine withheld labels. A secondary label-only accumulation computes `Σ_t α_t y_t x_t` from the same stream, deleting the prediction-residual factor without rerunning or changing the sensory state. Neither probe is the fly's scored native behavior.

| Old-end method, 64 known worlds | Taught choice [95% interval] | Never-taught choice [95% interval] |
|---|---:|---:|
| Frozen online NLMS head | **97.57%** [96.65, 98.49] | **51.04%** [46.77, 55.32] |
| Train-label-only ridge | **100.00%** | **48.26%** [44.15, 52.38] |
| Label accumulation, no prediction correction | **97.05%** [96.04, 98.06] | **48.52%** [44.48, 52.57] |

Relative to the online head, ridge's paired held-out difference was **−2.78 pp** [−6.51, +0.96], and label accumulation's was **−2.52 pp** [−5.38, +0.34]. The label-sum weight norm (world mean **0.281**) was close to online NLMS (**0.286**), while ridge's was much larger (**1.856**); these are not exact dose-matched interventions. Removing prediction correction did not reveal cyclic transfer in this replay, but because all three methods are near floor, the data cannot establish whether that term is necessary for a successful future learner.

## Mechanistic reading and limit

The strongest conclusion is **not** “the KC encoder cannot express cycles.” Its actual teaching span has nonzero access to the required endpoint directions, and a leaky oracle inside that span can solve the withheld signs. Nor is this evidence that the existing online rule has secretly learned them: its signed cycle-sum margin averaged **+0.00310** [−0.02037, +0.02658], and all three teaching-label estimators fit taught pairs while scoring near chance on unseen pairs. The results are consistent with training features that permit pair-specific fitting without a useful inductive bias toward a shared cyclic relation. They do not prove literal pair slots or identify one unique missing biological mechanism. The oracle has no norm bound, and failure of one fixed ridge does not rule out other train-only estimators.

This narrows a next *prospective* mechanism test: propose one compact cross-pair representation or update that predicts a positive effect on the same held-out triangles, include a parameter/dose-matched control, and test it on **fresh worlds** with the original no-old-teaching control. More writing, banking, or prediction-correction ablations alone are poorly targeted while training accuracy is already near 100% and none of the fixed simple estimators transfers. This is a criterion for the next design, not a mechanism success claim.

## Audit and compact output

Engineering world 9907 and the separate eight-world pilot passed. Full replay of 64 worlds took **109.8 seconds** without outcome-based stopping. The independent auditor checked source hashes, the frozen V9.10D margins, every world-level metric and interval, numerical tolerances, and a direct re-execution of the first confirmation world. Native-no-write margins matched V9.10D exactly; the maximum head-margin gap was **3.51×10⁻¹⁶**. The project objective contract's distinction between taught-pair accuracy and unseen relation reuse is preserved. The result bundle is under 10 MB and points to unchanged parent assets rather than duplicating them.

See the [design](DESIGN.md), [key metrics](results/KEY_METRICS.csv), [world metrics](results/CONFIRM_WORLD_METRICS.csv), [full summary](results/CONFIRM_SUMMARY.json), [audit](results/AUDIT.json), [source lock](results/SOURCE_LOCK.json), and [manifest](MANIFEST.json).
