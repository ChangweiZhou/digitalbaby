# E30: a registered gate measure, read on files nobody has measured (13 November 2026)

Read-only. Trains nothing. Evidence level: none claimed (a measurement on saved weights). Not audited by a second reader: Mr. Xu waived the audit step for this round. Host: any; state Python and torch in the report.

## Why

An outside review measured, after the fact, that networks which lived under the sweep leave their update gates undecided, and lose accuracy when the letters between operands grow from 12 to 40. Those numbers were read on files already seen. This job writes the measure down first and then reads it on files not yet looked at this way.

## The measure (fixed here)

All in `MOD97_LOCAL_RULE_20261106/`. Use `e29_gates.gates(m, x)` for the gate values and `e29_gates.plus_minus_panel(seed)` and `e29_gates.add89_panel(seed, k)` for items. New file `e30_gates.py`; edit nothing existing.

- **U, the undecided share.** Over every item, every position after the first byte and all 256 units: the percentage of update-gate values `z` with 0.1 ≤ z ≤ 0.9.
- **H, the holding loss.** Whole-answer accuracy on mod-89 addition (births) or mod-97 addition (lives) at 12 letters minus at 40 letters, 300 items each, from `exp11_cell.panel(..., ks=(12, 40), n=300)` on the held half of the split, scored with `exp16_integrated.score`.

## Step 1: calibration on seen files (stop if it fails)

Compute U on the plus-minus composition panel for the twelve lives in `exp45/e26/` and `exp45/e26x/`. The review reports 73–79% for lives updated by the sweep (S and BS) and 43–53% for lives updated by backprop (B and SB). If any file falls more than 3 points outside its range, stop and report the table; the definition above then differs from the review's and I must settle it before step 2.

## Step 2: unseen files

Only after step 1 passes and is pushed.

- E28 births on branch `claude/charming-cori-ljzwz1`: every complete final file, and the saved rows of S-all (rows 4096 to 20480). U on `add89_panel(seed, 12)` and on the plus-minus panel; H as defined.
- The six E26 births in `exp45/e26/`: the same.
- Report one table per measure: file, arm, seed, row, U, H.

## Predictions (mine, written before any of step 2 is run)

- **P30.1** In each seed with both files, U on the add-89 panel is at least 10 points higher in S-all than in B-all. About 55%.
- **P30.2** The same for the E26 births, S against B, in at least two of three seeds. About 55%.
- **P30.3** H is at least 5 points larger in S-all than in B-all in each seed with both files. About 50%.
- **P30.4** Along S-all's saved rows U falls from row 4096 to the end in every seed. About 60%.

Differences under 5 points in U are not read. If P30.1 and P30.2 both miss, undecided gates are a property of a life under the sweep, not of the sweep at birth, and E31's gate arm is read with that in mind.

## Done

`e30_gates.py`, `exp45/e30/E30_CALIBRATION.json`, `exp45/e30/E30_UNSEEN.json`, `E30_REPORT.md` with the tables and the four outcomes, no readings beyond them. Push to your own branch.
