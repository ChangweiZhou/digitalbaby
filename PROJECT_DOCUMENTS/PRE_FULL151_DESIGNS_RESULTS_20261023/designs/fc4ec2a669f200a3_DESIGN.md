# MiniFly V9.10A — native-margin aggregation audit

**Locked before computing any V9.10A prediction, 26 September 2026.** This is a read-only, post-V9.9C diagnostic. The parent REF fly and its V9.9C results are immutable. No new fly worlds, model change, retraining, held-out feedback, or promotion test occurs here. The existing V9.9C 64-world source data already establish a bounded E3 effect; this analysis can only diagnose what an **external task-specific aggregation** can recover from those recorded outputs.

## Question and exposure boundary

V9.9C saved native signed margins for 20 taught and eight never-taught rank comparisons in each world. Ask whether a graph-potential fit to the *taught* native margins yields more correct predictions on the eight never-taught pairs than the REF direct pair read. The fit sees neither the hidden rank ordering nor a held-out outcome. Exact-pair recitation has no held-out answer (50% with ties); a taught-label-only graph fit is a positive task-solvability control and can score 100% on this fixture. Hence a successful external fit cannot by itself show that REF internally represents rank or has solved general relational learning.

The archived margin for an edge is `value(neutral orientation) − value(aversive orientation)`, already **truth-oriented**. Before fitting, convert each *taught* edge's margin to a fixed ascending-byte orientation using its known teaching label. Use ascending byte value merely as an anonymous node identifier; never use the numerical hidden rank as a feature. Build held-out queries as **unordered byte pairs**, sorted by byte value. Save predictions for these canonical queries before a separate scorer reveals their true outcome orientation. The prediction phase reads only `old_end/trained` rows from the source metric table. It must not read a held-out margin, outcome, or score.

## Frozen analysis

For 20 taught canonical byte edges `(u,v)`, `u<v`, let `r_uv = value(v,u) − value(u,v)`. Fit eight free symbol potentials with an unregularized least-squares incidence matrix: `min_q Σ[(q_u−q_v)−r_uv]^2`, centered by `Σq=0`. The training graph must be connected and have incidence rank seven. Predict each canonical held-out pair by `q_u−q_v`. No tuning, learned calibration, threshold, or held-out model selection is allowed; ties receive 0.5.

Run this same fixed map on: native `D`, `L`, `E`, and no-old-write `Z` taught margins; write-induced `D−Z` and `L−Z` taught-margin changes; sign-only `D` margins; and a taught-label-only ±1 positive control. These are **external readouts**. As native reference scores, the separate scorer reports direct `D/L/E/Z` choice and the sign of direct `D−Z`/`L−Z` held-out margin changes. For each arm/method, report world-mean held-out choice and a descriptive paired interval against its relevant direct reference; report the per-world signs and trained-fit residual. In particular, compare `D`, `Z`, and `D−Z` under the *same* potential operation.

The output is a localization aid, not a new confirmatory E3 test. A gain of potential over direct native choice shows that this task-specific external graph operation can aggregate taught outputs. It does **not** distinguish a native latent rank from 20 separately stored pair episodes; nor does it identify gamma, alpha, teacher, or reader as the limiting component. A failure of this scalar fit cannot prove that no relational information exists in native state.

## Technical gates and cost

Before prediction, verify the input/source hashes, 64-world roster, exactly 20 taught margins per branch, no duplicate edges, eight distinct held-out unordered pairs, complete rank-seven taught graph, synthetic orientation/solver invariance, and exact-table/label-only analytic controls. Prediction and scoring are separate commands. The prediction file is hashed and frozen before the scorer reads any held-out margin or truth orientation. Independent audit recomputes each published score from the saved predictions and checks that no held-out answer entered prediction. Invalid source or a failed gate stops rather than silently repairing rows.

The bundle contains this design, scripts, compact predictions and final metrics, report, audit, and a manifest referencing the unchanged V9.9C table and code by hash. No parent datasets, checkpoints, prior archives, caches, or raw state snapshots are copied. Target <10 MB. Expected runtime is seconds to minutes on the Mac.
