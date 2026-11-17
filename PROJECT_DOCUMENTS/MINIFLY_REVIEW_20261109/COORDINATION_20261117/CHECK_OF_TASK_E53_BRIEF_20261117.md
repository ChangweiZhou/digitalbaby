# Check of the brief of E53 (revision 1) before its release (17 November 2026)

Work in progress (started 22:34 UTC). This file is written as the check proceeds, because the container may restart; the verdict and the ordered list of defects are entered last. Until then the notes below are the record.

## Working notes (to be replaced by the ordered findings)

Checked object: `TASK_E53_THE_NETWORK_WITH_AN_INTERFACE_AT_A_BOUNDARY_WITHOUT_DECAY_AND_FROM_A_LARGER_START_20261117.md` at commit 4582f5fa.

### Part A, figures recomputed from the trial's receipts (22:40 UTC)

Receipts `e47_32_one_{S,B}_73129.json` of `origin/claude/e47-worker-stage-0-3d0ya6` (578a877b), read with `git show`.

- Updates to 80 (first of three consecutive rows at or above 80): sweep A 13,568, R 17,664; back-propagation A 10,496, R 11,008. Agrees.
- Levels (mean of the last three rows): sweep A 92.28, R 87.80, 40 letters 90.78; back-propagation 95.39, 94.15, 95.22. Agrees.
- Last row: sweep plain A 93.33, R 86.35; far cells at zero 77.50, 77.70; at the mean 84.83, 80.60. Back-propagation plain 96.17, 94.65; zero 78.83, 78.90; mean 90.17, 88.70. Agrees.
- `magnitudes` at row 0: interface 0.151506, far 0.150255. The brief prints 0.151 for the interface; the figure rounds to 0.152. Last row: sweep 0.1985 and 0.1497; back-propagation 0.1789 and 0.2426. Agrees.
- `magnitudes` is `Wc.abs()[:, cells].mean()`: the mean absolute value of the 33 weights *into* each cell of the group (three gates, the weight on the cell's own state and the ten on its sources), not of the weights leaving it.
- Cost: sweep 9,768 + 3,359 CPU-seconds = 3.647 hours; back-propagation 9,790 + 3,258 = 3.624 hours. The brief's 3.6 and 3.65 agree.
- E42's references, from `exp45/e47_qual/e42_reference/e42_{G,F10}_{S,B}_73129.json`: G 5,632 and 5,120; F10 13,312 and 8,704. Agrees.
