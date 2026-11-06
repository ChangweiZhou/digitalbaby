# MiniFly turning and goal-steering qualification

The revised compass and steering subcircuits pass their isolated engineering checks on fresh panels. The combined circuit still loses accuracy after the initial landmark cue. Its remaining failure is visible when the steering circuit relies on the compass estimate during its own turns.

| Probe | Native | Control |
| --- | --- | --- |
| Stationary compass, two seconds without a landmark | **16/16** retained at least one-quarter of initial harmonic strength and stayed within 20° | **1/16** with independently calibrated rewiring, **0/16** with recurrence removed |
| Turning compass, 96 dark turns including reversals and delayed starts | **8.6°** median error, **81/96** within 20° with a retained heading signal | **54.5°** with rewiring at the native settings, **84.8°** after separate rewired calibration, **61.9°** with yaw disabled |
| Goal steering with externally clamped heading | **94/96** correct initial turn directions and both directions correct in **46/48** opposite-goal pairs | **49/96** with separately calibrated rewiring, **48/96** with the goal removed |
| Angular convergence with externally clamped heading | **75/96** reached within 20° after four simulated seconds, with **12.7°** median remaining error | **14/96** with separately calibrated rewiring, **15/96** with the goal removed |
| Combined supplied-goal steering with one initial landmark cue | **42/96** within 20° at four seconds, with **21.9°** median goal error and **21.1°** median compass error | **11/96** within 20° with rewiring, **8/96** with yaw disabled and **11/96** with the goal removed |

## Changes tested

- **Compass.** The runner preserves the observed EPG→PEN1 contacts in PB and PEN1→EPG contacts in EB. It gives EPG recurrence, PEN feedback and Δ7 inhibition separate gains. Yaw oppositely modulates the two PEN1 sides. The fixed update responds faster to this velocity-dependent current than to the resting current. At the selected setting the per-step resting blend is 0.055 and the velocity-dependent blend is 0.45. The anatomical graph shifts the activity pattern. The simulator supplies yaw increments and an initial landmark cue. [Compass physiology](https://elifesciences.org/articles/23496)
- **Steering.** The runner normalizes heading and goal input streams separately, adds them at the observed PFL3 targets, applies a soft threshold and subtracts the two DNa02 outputs. Δ7 inhibition and the PFL3 transfer are model assumptions. The left-right motor polarity remains fixed. Calibration selected zero added FC2 phase offset. The isolated closed loop supplies the current heading to the steering circuit on each step. The combined loop instead decodes the evolving EPG activity and feeds that estimate to the same steering circuit. [PFL3 physiology](https://www.nature.com/articles/s41586-023-07006-3) · [DNa02 physiology](https://elifesciences.org/articles/102230)

The gains and rate constants were selected on development panels and frozen before the qualification probes in each stage. The panels use deterministic, distinct headings, yaw schedules and goal offsets. Rewired arms use the same calibration search and the same input and output conventions. The receipt also includes rewiring at native settings and the heading, yaw, goal and recurrence lesions. These are engineering checks. Spatial displacement and reward learning remain gated on combined navigation performance.

Reducing the motor turn cap from 0.12 to 0.054 radians per step brought it inside the maximum yaw used in the isolated compass panel. On a fresh matched panel both caps reached **39/96** goals within 20° by four seconds, with roughly **24°** median goal error. The native compass error grew from about **11°** at two seconds to **21°** at four seconds under either cap. A smaller turn command alone did not repair the accumulated error.

## Reproduce

From the project root, run the NumPy-based qualification. It reads the original connectome extract and the derived v630 maps. The original interface probe was rerun separately and reproduced its previous receipt byte for byte.

```sh
dir=output/minifly-connectome-navigation
OPENBLAS_NUM_THREADS=1 python -B "$dir/qualify_dynamics.py"
```

[Qualification receipt](DYNAMICS_QUALIFICATION.json) · [Previous interface report](INTERFACE_RESOLUTION.md)
