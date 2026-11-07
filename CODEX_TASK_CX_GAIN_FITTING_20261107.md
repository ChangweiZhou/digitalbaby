# Task for Codex: close the central-complex loop by fitting gains under connectome constraints

**Date:** 7 November 2026. **Branch:** commit on `codex/minifly-consolidated-20261023`. Do not create a new branch. **Folder:** continue in `output/minifly-connectome-navigation/`.

## What you already built, and what it showed

Your 6 November qualification extracted the FlyWire v630 central complex and mushroom body (6,704 neurons, 417,204 signed edges, 106 role-pair gain groups) and tested it with hand-set gains. Isolated circuits passed with large margins over the degree-matched rewired control: heading held in darkness 16/16 versus 1/16, dark turns 8.6° median error versus 54.5°, goal steering 94/96 versus 49/96. The combined loop failed: steering from the compass's own estimate reached 42/96, and compass error grew from 11° to 21° over four seconds.

Every gain in that model was set by hand. Nothing was fitted. That is the one thing this task changes.

## The task, in one sentence

Fit the 106 role-pair gain groups (and any adapter gains you declare) by gradient descent on the combined heading-plus-steering task, with the connectome's wiring, signs and cell types held fixed; fit the rewired control the same way; rerun the closed loop; report both.

## Rules that resolve the earlier confusion

1. **Gradient descent is allowed here.** This is the development phase. It stands in for evolution and sets the parameters the connectome does not record. The project's "no backprop" rule applies to lifetime learning, which is not part of this task. Do not spend time on a local rule for the gain fit.
2. **What is fixed:** which neurons exist, which neuron connects to which, the sign of every edge (from the neurotransmitter prediction), the synapse counts as relative weights within a role pair, and the role-pair grouping. **What is free:** one scalar gain per role-pair group, the time constants and thresholds you already expose, and the declared adapter gains (landmark input, yaw input, goal input, motor readout). List every free parameter in the manifest with its count. Expect a few hundred, not thousands.
3. **The rewired control is fitted with the same procedure, budget and data.** A result counts only as native-fitted versus rewired-fitted. Native-fitted versus rewired-hand-set is not a result.
4. **No new tasks, no new benchmarks, no new circuits.** Use the probes you already have: stationary heading, dark turns, clamped-heading steering, clamped-heading convergence, and the combined loop. Add nothing until the combined loop passes.
5. **Lock before fitting.** Freeze the probe panels, the fitting loss, the parameter list, the training budget and the pass criteria below in a LOCK.json before the first gradient step. Score the sealed panels only after fitting ends.

## Fitting loss

Simulate the combined loop: initial landmark cue, then yaw commands from the agent's own steering output, a supplied goal, and no further landmark. Loss is the mean squared heading error of the decoded compass bump against the true heading over the episode, plus the mean squared goal-angle error at the end of the episode. Use a few hundred episodes with the same yaw and goal distributions as the probes. If the simulation is not differentiable end to end, say so and use the nearest thing you can justify (fit the compass gains on the dark-turn loss first, then the steering gains on the clamped-heading loss, then all gains jointly on the combined loss). Report which you did.

## Pass criteria, fixed in advance

- Combined loop, native-fitted: at least 72/96 goals within 20° at four seconds, compass error under 15° at four seconds.
- Combined loop, rewired-fitted: must stay at least 30 goals below native-fitted on the same panel. If the rewired control also passes, the wiring did not matter and the result is reported as that.
- Isolated probes must not regress below the hand-set results.
- Three seeds of the fitting procedure (different initial gains, same data). Report every seed.

## Deliverables

- `FIT_LOCK.json`, `FIT_REPORT.md` with the table of all five probes for native-fitted, rewired-fitted, native-hand-set and rewired-hand-set, three seeds each.
- The fitted gain vectors per seed and the fitting curves.
- A short paragraph on the plasticity mask question only if it blocks the above; otherwise leave it.

## What not to do

- Do not write a new design memo, audit, catalogue or roadmap. The design is this document.
- Do not open path integration, displacement, reward learning or the arena until the combined loop passes.
- Do not change the probes to make the loop pass. If a probe is broken, say so and stop.
- Do not create a branch.
