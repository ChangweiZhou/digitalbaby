# Follow-up for Codex: the path-integration run is half a pass; find where the return goes wrong

**Date:** 7 November 2026. **Branch:** `codex/minifly-consolidated-20261023`, no new branch. **Folder:** `output/minifly-connectome-navigation/`. **Builds on:** `PI_REPORT.md` (commit 44ca55c).

## What the run established

- The wiring gives the vector to integrate. Native-fitted direction within 20° on 59–60/96 main episodes, median 16.4°, against 21–23/96 and 38° for the identically fitted rewired control; magnitude ratio 0.89 inside the gate; compass unchanged. PFNd carries the signal, PFNv adds little. That is the published result (heading-tagged travel direction) reproduced from the connectome under fitted gains, and it stands.
- The integration itself is a supplied mechanism. The hΔB recurrence fitted to 0.0001 and the memory lives in the explicit state update with retention 0.999. Say this in the report's first paragraph, as the first fit now says "five cell types": the wiring supplies the instantaneous vector; the temporal accumulation is the model's.

## What failed, and what it is not

- The return panel: native-fitted median residual 0.65 of the outbound distance (limit 0.30). The angle gate: 60/96 against 72.
- It is **not** a leak. Retention 0.999 over the roughly 85 steps between the outbound midpoint and the end predicts a residual near 0.08, not 0.65.
- A residual of 0.65 is what a directional error of about 38° on the return leg produces. Two candidates remain: compass drift during the 180° turn-in-place (3°/step for three seconds, a schedule the compass was never probed on), and asymmetry in the hand-shaped PFN velocity encoding (the ±45°/±135° body axes), which would cancel on one-way paths and add on a reversal.

## Do these four things, in this order, no refitting until step 4

1. **Decompose every return episode's residual** into the component along the outbound direction and the component perpendicular to it, and record the compass error at the end of the turn-in-place. Report medians and the 48-episode scatter for native-fitted and rewired-fitted. Along-outbound residual with a small compass error means encoding asymmetry; a large compass error at the end of the turn means drift.
2. **Probe the frozen compass alone** on the return schedule (no translation, no integrator): 180° at 3°/step, both directions, 48 episodes, same scoring as the dark-turn probe. If the median error after the turn exceeds 20°, the compass is the bottleneck and the path-integration result is compass-limited, not wiring-limited.
3. **Compute the integrator's ceiling with the true heading clamped**, exactly as the steering circuit's ceiling was found: run the fitted native integrator with the compass replaced by the true heading on both panels. The gap between clamped-heading and own-compass is the honest measure of the compass's cost, and the clamped-heading score is the ceiling the 72/96 gate should have been set against. (That gate was mine and was set without this ceiling; replace it with: native own-compass within 10 episodes of native clamped-heading on the main panel, and return residual within 0.15 of the clamped-heading return residual.)
4. **Only then refit**, and only the thing step 1 names: if drift, refit the 15 compass gains with the return schedule added to the compass training set (the path-integration gains stay frozen); if asymmetry, add one declared parameter for the PFN axis offset and refit the 10 path gains with the compass frozen. Same budget, same rewired procedure, new lock.

## Pass criteria for this round

- Steps 1–3 delivered as tables, three seeds.
- After step 4: main-panel own-compass within 10 of clamped-heading in every seed; return residual within 0.15 of clamped-heading; rewired-fitted lead of at least 30 on the main panel; compass probes unchanged within 3°.
- If the clamped-heading integrator itself fails the return panel (residual above 0.30 with a perfect heading), the supplied accumulator is the problem; report it and stop. That decides whether the next step is a better compass or a real integrator built from the hΔB recurrence the fit currently ignores.

## What not to do

- Do not add cell types, panels or memos. Do not touch reward, plasticity or the arena.
- Do not refit before steps 1–3 are on the record.
- Do not create a branch.
