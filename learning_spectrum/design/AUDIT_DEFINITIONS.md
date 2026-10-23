# Independent prospective audit definitions

Status: advisory before cycle 1 design/test. This file changes no mechanism, tests, or existing experiment.

## Normalized learning spectrum and finite-horizon gain

For a fixed feature map with column feature x_t and the normalized delta update

w_(t+1) = D_t w_t + eta (y_t - x_t^T D_t w_t) x_t / d_t,

where d_t=max(1,||x_t||²), define A_t=(I-eta x_t x_t^T/d_t)D_t and b_t=eta x_t/d_t. For a terminal query q, the exact unconstrained teacher-to-output coefficient for teacher event t is q^T A_(T-1)...A_(t+1)b_t (and any terminal decay/readout multiplier). The full operator K_T stacks these coefficients for all queries. If one target per cue is repeated, compose K_T with the explicit repetition/incidence operator. This formulation must match the actual chronology, decay and teacher-time/read-time conventions. Clipping makes the map piecewise affine; the linear forecast then needs a verified no-clipping certificate or exact piecewise replay.

The normalized Gram S=(1/n) sum_t x_t x_t^T/d_t has eigenvalues nu and exposes input geometry. Record whether an alternative convention includes eta, sum rather than mean, or a query normalization. These conventions alter numerical claims such as nu~1e-2. Under stationary repeated full-batch learning without other dynamics a direction has approximate gain 1-(1-eta*nu)^N; this is not an exact online finite-horizon prediction. A large eigenvalue is not evidence of task-relevant gain unless the target has a measured projection there. Sequential operators may be nonnormal, so their eigenvalues alone do not bound transient amplification or task performance.

Report the singular spectrum of K_T, target-aligned gain <Y,K_TY>/||Y||², relative forecast error ||K_TY-Y||/||Y||, teacher/event weighting, target projection onto eigenspaces (use projectors for degenerate eigenvalues), and F/G/H projections in decision space. Report complete predicted scores and margins before labels/outcomes from fresh worlds are opened. Spectral pass regions, horizon, target-energy coverage, and forecast tolerances must be fixed before a pilot is interpreted. A normalized eigenvalue threshold alone is insufficient.

## Shared/relation preservation

Response-source F/G/H decomposition is an orthogonal ANOVA decomposition on a balanced 4×4 grid. Decision-center the channel axis before decomposition. Shared low-order directions must be specified independently of observed pilot success, and must not be mistaken for task/answer metadata supplied to the learner.

For a nonzero reference shared target U, report aligned gain <U,K_TU>/||U||², distortion ||K_TU-U||/||U||, and worst retained singular value on its declared subspace. Mere F/G RMS preservation admits sign reversal, rotation and task-irrelevant activity. Specify an absolute floor or control-relative noninferiority margin. An undefined/zero denominator cannot pass by convention.

Use a distinct structured relation assay with train/test compositions fixed before evaluation to substantiate relation transfer. Without such an assay, state the narrower claim: preservation of declared feature or response subspaces, not preservation of reasoning, transfer, or relational behavior. A feature-only rank certificate cannot substitute for finite-horizon learnability.

## Causality and forecast boundaries

Causal EMA transformations must be computed using pre-update state; evaluator pair IDs, digit positions, labels and held-out data cannot enter the feature/update path. Freeze whether probes clone EMA state and whether the current cue updates it; repeated probes must not train the continuing model. The complete initial EMA state and all chronological inputs are needed to reproduce forecasts. If EMA evolves, the actual time-varying feature sequence rather than a final-state Gram matrix determines learning.

Separate exact conditional algebraic prediction (given the input stream, labels, native score baseline and fixed parameters) from an empirical forecast of future native scores. Reusing observed final native scores in a purported prospective total-score forecast is leakage. Validate the predicted added component independently; report any inability to predict total margins as a failed or unmet gate, not an approximation silently called a pass.

Calibration learned from response histories is development data. Count wrong→right and right→wrong on a distinct untouched evaluation set with a fixed causal rule. Offline grand-mean centering that uses all probe inputs is an oracle diagnostic and must not be labeled online calibration.
