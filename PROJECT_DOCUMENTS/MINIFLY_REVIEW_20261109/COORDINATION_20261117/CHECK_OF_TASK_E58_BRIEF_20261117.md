# Check of the brief of E58 before its release (17 November 2026)

Started 22:16 UTC. This file is written as the check proceeds; while this line stands, the check is not finished.

Brief checked: `TASK_E58_A_GATED_TERM_AGAINST_FREE_LEARNING_DECAY_AND_NUMBER_OF_PAIRS_20261117.md` (revision 1) beside this file, as it stands in the working tree at commit 54901576.

## Working notes (to be replaced by the final document)

- 22:18. `e48_gated.py` on E48's worker's branch and `e48_gated_spec.py` on the default branch have the same SHA-256 (ab324e9c...b1a0). The default branch is not an ancestor of E48's worker's branch (confirmed).
- 22:40. Figures of "What the record says" recomputed from the 12 receipts of E40 (lived, fresh) and the 18 of E48 (script `figs.py`, appendix). All agree with the brief except: the hint under the sweep on seed 73129 is 7.62 (brief: 7.7) and under back-propagation on seed 73129 6.93 (brief: 7.0); the brief subtracted figures already rounded. Cost figures agree (974/130; 135/202; 0.33; 0.39; 0.094; 9.4; 12.7; 22.6).
- 22:40. The lived networks and their receipts are on the default branch (six `.pt` of 4,501,600 to 4,501,619 bytes, six `.json`). A term file of rank 16 is 150,484 to 150,517 bytes. E48's report is at `$M/exp45/e48/E48_REPORT.md`.
- 22:40. `exp11_cell.split` seeds its shuffle with `seed * 1000 + round(frac * 100) + m * 7 + code[0]`: the fraction enters the seed, so the taught pairs at 0.05 are not a subset of those at 0.10, nor these of those at 0.20.
- 22:40. Check 1's parenthesis omits row 64: with rows 16 and 32 added, the mean loss since the last row at row 64 covers updates 33 to 64, not 1 to 64 as in E40.
- 22:40. `exp26_candidate.step` calls `arch26.forward` directly; scoring goes through `exp11_cell.forward`, which `exp48_gated_run.install()` replaces. A free arm must call `arch26.install()` only.
- 22:40. `exp26_candidate.opt_record` writes `weight_decay=WD` (the constant 0.25) beside the groups read back; in `free-wd0` this field would be false.
