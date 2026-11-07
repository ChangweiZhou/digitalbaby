# Task for Codex: path integration in the fan-shaped body under fitted gains

**Date:** 7 November 2026. **Branch:** commit on `codex/minifly-consolidated-20261023`, no new branch. **Folder:** `output/minifly-connectome-navigation/`. **Builds on:** `FIT_OFFICIAL_REPORT.md` (compass and steering loop closed, 80/96 on fresh panels).

## What is established and what this task adds

The compass and steering loop closes under fitted gains: EPG, PEN, Δ7, PFL3 and FC2, 15 free parameters, 80/96 goals on fresh panels against 20–24 for the identically fitted rewired control. That answers "does the wiring hold and turn a heading and steer toward a supplied goal?".

This task asks the next question in the same way: **does the wiring integrate self-motion into a displacement vector?** Path integration is the fly's native vector arithmetic. In the model it has been a gated capability since the first qualification; nothing has been run.

## Scope

Add exactly the cell types the published path-integration circuit uses and nothing else: PFN (the d and v populations that carry heading-tagged velocity), hΔB (the velocity integrators), PFR or FC2 as the readout to a stored vector, with their FlyWire v630 edges and signs. Keep EPG, PEN, Δ7, PFL3 and FC2 as fitted in `FIT_OFFICIAL_RESULTS.json`; their gains are frozen for this task. List every new role-pair gain and every new adapter (forward-velocity input, sideslip input if used, displacement readout) in the manifest with counts. Expect a few dozen new free parameters, not hundreds.

## The probe, fixed before fitting

- Episodes of 160 steps at 0.05 s. The agent receives an initial landmark cue, then a scripted sequence of yaw and forward-velocity commands (no steering, no goal). The true displacement vector is the integral of velocity along the true heading.
- Decode the model's displacement vector from the designated readout population at the end of the episode (and at steps 40, 80, 120 for the curve).
- Score: angular error of the decoded vector against the true displacement, and magnitude ratio decoded over true. Pass counts within 20° of angle and within 25% of magnitude.
- Panels: 96 episodes with yaw schedules and speeds drawn from the same distributions as the earlier panels, one newly sampled heading per sector, sealed before fitting. A second sealed panel of 48 episodes with **outbound-then-return** paths, where the true final displacement is near zero, to catch a readout that only tracks heading.

## Fitting

- Development phase: Adam through the full episode, as before. Loss is the mean squared vector error (decoded minus true, in body-length units) at steps 40, 80, 120 and 160, plus the compass term from the previous fit so the compass is not broken by the new gains.
- Free: the new role-pair gains and new adapters. Frozen: the 15 compass and steering parameters. Same 384 training episodes convention, 2,048 updates with learning rate halving every 512, three seeds, same initialization rule.
- The rewired control: rewire the **new** projections only (PFN, hΔB, PFR/FC2 edges) with the same degree-preserving procedure, keep the fitted compass intact, fit with the identical budget. This isolates the path-integration wiring from the compass wiring, which is already established.
- Lock everything in `PI_LOCK.json` before the first gradient step. Score sealed panels only after all six fits finish.

## Pass criteria, fixed in advance

- Native-fitted: at least 72/96 within 20° of angle on the main panel, median magnitude ratio between 0.75 and 1.25, in every seed.
- Outbound-then-return panel: native-fitted median decoded magnitude below 0.3 of the outbound distance (it must know it came back).
- Rewired-fitted: at least 30 episodes below native on the main panel.
- Compass probes from `FIT_OFFICIAL_LOCK.json` must not regress by more than 3° median.
- If the native arm passes the main panel but fails the return panel, report it as heading tracking, not path integration, and stop.

## Deliverables

- `PI_LOCK.json`, `PI_REPORT.md` with the four-row table (native and rewired, hand-set and fitted) per seed on both panels, the displacement-error curves over the episode, and the compass regression check.
- Fitted gain vectors and training curves.
- One paragraph on which new cell types the fitted gains actually use (a gain that fits to zero is a cell type the loop does not need), because that is the finding that feeds the next step.

## What not to do

- Do not add the mushroom body, reward, plasticity or an arena. Those come after displacement works.
- Do not refit the 15 compass and steering parameters. If path integration seems to need them changed, report it and stop.
- Do not change the compass probes.
- Do not write a design memo. This document is the design.
- Do not create a branch.
