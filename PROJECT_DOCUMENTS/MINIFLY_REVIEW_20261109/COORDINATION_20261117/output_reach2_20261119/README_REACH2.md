# The output end on the tissues BAL and B1 (19 November 2026)

This folder is a copy of the reviewer's folder at ba59490c on claude/vibrant-cerf-sje726: `tissue.c`, `tis.py`, `drive.py`, `diag.py`, `deps/` and `README.md`, unchanged. Added:
- `reach.py` and `reach_nbhd.py` (copied from `output_reach_20261119`), and `deps/PREP` (the scorer, copied from 10e3ab9c);
- `yesno3.py`, `lms_diag.py`, `lms_white.py`, `make_table.py`, `run_mixed.sh`;
- the batches `batch_all.txt`, `batch_lms.txt` and `batch_white.txt`.

The table is `TABLE.md`, logs are in `out/`, and results are in `reach/` and `yesno3/`.

With this `tissue.c`, `reach.py` on U at 16,384 bytes (seed 73129) gives exactly the figures of the earlier folder.

**Tissues.**
- U: `rule=fixed`.
- BAL: `rule=timing eta_s=0 no_make=1 norm=1 we=1/2 wi=1/2`.
- B1: `rule=timing norm=1 lines_learn=1 eta_s=1 no_make=1 we=1 wi=1/4`.
- Seeds 73129, 93683 and 116089; 65,536 bytes each.

**What was measured.**
- **A. Fitted ridge readers** (as in `README_REACH.md`). Fitted on the first three quarters and scored on the last quarter, on:
  - set (a), the cells near the output lines;
  - each byte on its own driver's neighbourhood;
  - set (b), every non-feedback cell;
  - one-hot and two-one-hot readers, and ref_1, ref_2, ref_3, beside them.
- **B and C. Online one-pass rules on the cells near each driver** (`yesno3.py`): yes/no rule A (kappa 1/8, eps 0 and 1/16), yes/no rule B (eps 1/16), and the supervised delta rule D (eta 1/64 and 1/16). Each is run with two preparations of the inputs:
  - "run": running standardisation, as in yesno.py;
  - "shared": driver v first subtracts the mean of its own inputs at that step, then standardises each by a running mean and deviation held at that connection.
  - Reported as the last window of 4,096 and the last quarter (the positions of A).
- **D. Why the supervised one-pass rule stays below the fitted reader** (`lms_diag.py`, `lms_white.py`; an addition):
  - one pass with fixed instead of running standardisation;
  - several passes over the fitting part;
  - the conditioning of each driver's inputs;
  - one pass on inputs whitened per driver by the fitting part's covariance. This is a transformation no cell has; it is a diagnostic only.

**Results** (ranges over three seeds; `TABLE.md`).
1. **BAL against U.**
   - Fitted readers on BAL get less than on U: set (a) 10.3–12.1 against 12.8–14.7; own neighbourhood 8.3–10.3 against 8.8–10.3.
   - The supervised one-pass rule gets slightly more on BAL: 6.0–9.3 against 4.5–6.8.
   - The yes/no rules on BAL stay below the end mark's share (A 0.5–4.2; B 2.6–5.0, or 4.0–6.2 with the shared part removed).
2. **B1 against U.**
   - Every fitted reader gets less on B1: set (a) 7.1–8.3, R2 0.0–0.1; own neighbourhood 6.9–8.1.
   - So does the supervised rule: 3.2–3.9.
   - Yes/no rule B is a little higher: 5.3–7.3, with one last quarter above the end mark's share (7.3 against 6.2, seed 73129).
3. **Removing the shared part.**
   - It raises the supervised one-pass rule on U by 0.4 to 1.7 points, changes BAL by −0.2 to +0.5, and lowers B1 slightly.
   - In no case does it reach the fitted own-neighbourhood reader; the closest is U seed 116089, 8.5 against 10.1.
4. **Section D: the gap is the conditioning of the inputs.**
   - Neither the running standardisation (fixed standardisation gives the same, within 0.1 point) nor the single pass (8 passes: +0.5 to +1.1) explains it.
   - Each driver's inputs are dominated by one direction: the first principal component holds a median 0.89 / 0.81 (U), 0.69 (BAL) and 0.60 (B1) of their variance, and the largest eigenvalue is 1e5 to 1e7 times the median one.
   - On inputs whitened per driver, one pass of the same rule reaches 11.0 and 10.0 on U and 9.8 on BAL, at or above the fitted reader on the same inputs (10.3, 8.8, 9.6).
   - On B1 it does not (4.3–5.0 against 8.1).
   - Removing only the mean (C) removes one direction of this spread, and that is not enough.

**Not measured.**
- Whitening by a rule a cell could hold (for example, a local decorrelating rule at the driver's connections).
- Seeds 93683 and 116089 for BAL and B1 in section D.
- Lengths beyond 65,536 bytes.
