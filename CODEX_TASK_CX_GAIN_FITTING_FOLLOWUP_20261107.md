# Follow-up for Codex: the gain fit worked; correct the verdict, converge the fit, find the steering ceiling

**Date:** 7 November 2026. **Branch:** commit on `codex/minifly-consolidated-20261023`, no new branch. **Folder:** `output/minifly-connectome-navigation/`. **Supersedes** the gate in `CODEX_TASK_CX_GAIN_FITTING_20261107.md`.

## First, correct the verdict

Your fit was done correctly: lock, finite-difference gradient check, matched rewired fit, three seeds. The "qualification failed" verdict is wrong because the gate I gave you was infeasible. The combined loop cannot exceed what the steering circuit achieves with a perfect clamped heading, and that ceiling is 73–75/96 in every arm. Measured against it, your result is positive:

| | Hand-set | Fitted (3 seeds) |
|---|---|---|
| Native combined goals within 20° | 42/96 | 55–56/96 |
| Native compass error at 4 s | 21.1° | 10.1–11.3° |
| Rewired, fitted identically | 11/96 | 13–15/96 |
| Gap to the clamped-heading ceiling | 33 | 17–19 |

Rewrite the outcome paragraph of `FIT_REPORT.md` to state this. Keep every number. Replace "Qualification failed" with the gap-to-ceiling result and the 41–42-goal lead over the fitted rewired control.

## Second, state the scope plainly

Only 8 of the 106 role-pair groups enter the simulator; the other 98 are not in the loop. Say in the report's first paragraph that the fitted model is the EPG, PEN, Δ7, PFL3 and FC2 projections with seven adapter and dynamics scalars, 15 free parameters in all. Do not describe the 98 as "frozen at unity"; describe them as absent. Nothing else changes.

## Third, converge the fit

The heading loss plateaued at about 12° RMS. The goal loss was still falling at update 128 (24° RMS, gradient norm 0.15). Rerun the three paired fits with:

- 2,048 updates, same batch and data, Adam with the learning rate halved every 512 updates from 0.025.
- The goal term computed as the mean squared wrapped goal error over the **last 20 steps** of the episode, not the final step alone.
- Everything else identical: same lock fields, same panels, same seeds, same rewired procedure. Write a new `FIT2_LOCK.json` before the first step.

Report the same tables as before, plus the loss at every 128 updates so a plateau is visible.

## Fourth, find the steering ceiling

With the heading clamped, 21–23 of 96 goals do not converge within 20° in four seconds, hand-set or fitted. That number bounds everything downstream. Before or alongside the refit:

- List the 21–23 failing goals by initial goal angle relative to heading and by final error.
- Report whether they cluster (near ±180°, near zero, one side) and whether the failure is slow convergence (error still falling at 4 s) or a wrong fixed point (error stable).
- Report the same for the fitted rewired arm, so the cluster is attributable to the wiring and not to the adapters.

No fix yet. The diagnosis decides whether the next free parameter is the motor readout, the PFL3 threshold, or the 64-phase normalization, and that choice is made after the numbers, not before.

## Pass criteria for this round

- Combined loop, native-fitted: gap to the clamped ceiling at most 10 goals in every seed, compass error under 15°.
- Rewired-fitted stays at least 30 goals below native-fitted.
- Dark-turn median error within 3° of hand-set (a regression is a reportable trade-off, not a failure).
- Steering-ceiling diagnosis delivered as a table, three seeds.

## What not to do

- Do not add cell types, probes, tasks or memos. Do not open the plasticity mask, path integration or reward.
- Do not tune anything on the sealed panels. The refit uses the training episodes only.
- Do not create a branch.
