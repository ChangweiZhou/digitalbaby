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

### Notes at 22:50 UTC (not yet ordered)

- The decay did not shrink the far cells' weights in the arm `one`. Receipt, `magnitudes.far`: sweep 0.1503 at row 0, least 0.1320 at 2,048, 0.1614 at 8,192, 0.1497 at the end; back-propagation 0.1503 to 0.2426. Decay alone would give 0.37 of the start at 4,000 updates and 0.0021 at 24,576. The verification's table by ring (section 6): sweep 0.178, 0.132, 0.145; back-propagation 0.234, 0.257, 0.234; start 0.150. The loss of weights was seen in the arm `place`, which E53 does not run. So (a) the decay half of the explanation has no support in the arm that is run; (b) the reason for omitting start x 3 with decay 0.25 ("the multiple would be gone", "the run would repeat the trial") is contradicted by the trial's receipts; (c) item 9 of the plan reads "from a start at which differences pass, with and without decay", and the brief has no arm with the changed start and the decay.
- `build(side, arm, seed)` passes the arm's name to `Local(..., periods=arm)`; `draws` treats every name other than `one` as periods by place. A copy of `run` called with the arm `wd0` would build the network of the arm `place` (971,665 weights). The assert on `WEIGHT_COUNT[(side, arm)]` raises a KeyError first; a worker who adds the two keys with 965,905 gets an assertion error; one who adds them from `L.counts` gets the wrong network.
- `opt_record` writes `weight_decay=C26.WD` (0.25) at the top level of `optimizer` and `optimizer_read_back`, whatever the groups hold.
- `receipt_path`, `provenance`, `main`, `OUT`, `ARMS`, the prints and the strings `start`, `update`, `forward` of the receipt are outside `run` or name E47.
- `L.Wc` is a leaf that requires a gradient: an in-place multiplication needs `torch.no_grad()`.
- The scale of the sweep is computed in every forward from `L.Wc` (`forward`, line 132; `IndexSign`); nothing is cached from `Wc` at build time; `plan` caches indices only. `install` holds a reference to `L`. The module's `rnn.weight_hh_l0` takes no part.
- The start file is written after `make_opt`, so it holds the multiplied tensor. On a resume `build` is called again and the multiplication repeated, but `copy_` from the saved state overwrites `Wc`, and the start's hash is read from the file.
- The rows of the lost runs (`lost_at_restart/`, another machine image) equal those of the final receipts in every field but the times, over 49 and 33 rows: bit-equal reproduction across machines of this host is on record, so check 1's "equal" is achievable.
- `signal_passage.py`, parts `start`, `arith`, `trained` rerun: 0.1868, 0.0207, 0.0028; 2.38, 2.46, 3.98 connections from the ring before; 0.0434 a connection; trained 0.587, 0.113, 0.0152 and 0.499, 0.0752, 5.8e-04. Agrees.
- New measurement (own script on the reviewer's functions): only the far cells' 33 weights x 3, the interface cells' rows unchanged: ring 3 holds 0.206 at 12 letters (0.082 without letters), no growth (0.948, 0.982 a position), way back 0.037, interface's mean state unchanged 0.178. At 3.5 it grows.
