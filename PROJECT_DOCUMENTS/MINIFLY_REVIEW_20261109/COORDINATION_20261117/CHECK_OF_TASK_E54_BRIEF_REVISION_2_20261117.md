# Check of the brief of E54, revision 2 (17 November 2026)

**State of this file: working notes, written as the check proceeds, so that a restart of the container loses little. The final text replaces these notes. Started 22:01 UTC; this version 22:20 UTC.**

Written for the outside reviewer by an agent that has not seen the reviewer's reasoning. Read-only; nothing trained; no code of another branch executed.

## Findings so far (to be put in order and given their repair texts)

1. **The rule "has flattened" calls a slow, still rising curve flat.** The limit of 3 points over 8,192 updates is of the size of the dense network's own slope at a tenth under back-propagation (half width, E41, nine-row rise at update 40,960: 5.2, 2.7, 3.8). By the rule the curve of seed 93683 "has flattened" at a level of 61.7 while its three-row level rose by 5.9 points. A learner half as fast has half the slope. "Worse" needs only the network's own curve to have flattened, so a network that is only slower is again called worse. (Figures to be completed.)
2. **The second reading, the sweep against back-propagation, has no margin at f = 0.10**: the margin is defined by rule (10 under the sweep, 20 under back-propagation), and the comparison is between the rules. Its words and its condition of flattening are not written out.
3. **Licence (i) has no margin.** "Generalises from a fraction at which `G`'s curve of R has flattened below 80" can be earned by a difference of one or two points at a threshold that lies inside the scatter of the dense network's late level. "Flattened below 80" is not defined (level below 80, or never three rows at 80).
4. **Line 68 of `exp42_ckpt.py`** hashes `HERE / "exp42_ckpt.py"`. The worker starts from the default branch (`LOG.md`, section 3), where that file does not exist; the list of changes does not mention the line, so the copy stops at its first run. Also not in the list: line 44 (the default number of updates is `R.DOSE` = 24,576), line 96 (the fraction is not in the assertion on resume), lines 179 to 214 and 223 to 224 (the self-test, tied to E42's folders and to `selftest_case.sh`), the place of the hash at updates 12,288 and 24,576 (it must be written into the receipt before the saved state of that update), the replay loop (lines 99 and 100) and change 3.
5. **Change 6 contradicts the copied text**: `R.install(layer)` (line 59) replaces `exp11_cell.forward`.
6. **Part 4 after a restart**: `exp41_scale.run` refuses to start when the receipt exists (line 153), and it writes the receipt every 4,096 updates. The brief says "started again from its beginning" and not that the partial receipt must be moved. A run of part 4 needs about 1.6 CPU-hours without interruption; the reviewer's container was restarted twice between 21:50 and 22:09.
7. **Detached runs die when the session is idle** (`LOG.md`, 22:11, the long lives' worker). The brief's rule "launched detached" is not enough; the worker must hold its turn open.
8. **Check 5** is read "at the first scored rows", where both counts are 0 for any panel; it cannot fail there. An equality of the two lists of items before the first update can.
9. **Check 4** has no limit and cannot fail.
10. **`describe`** takes the doses as arguments; the 512 updates subtracted from the dose are in `thr_desc` (lines 75 to 83), which the brief does not cite.
11. **"What the record says"**: the rises quoted are three-row rises, while the brief's own rule is the nine-row one; "the sweep therefore needs more taught pairs" goes beyond the brief's own rule at a quarter.
12. P54.2: "by update 24,576" is ambiguous by two rows.
13. The base branch of the worker is not named in the brief (`exp41_scale.py` is absent from `$W`; `exp42_ckpt.py` and `e42_tables.py` are absent from `$D`).

## Verified so far

- The first check's script, run again: all figures as the first check gives them.
- `e54_r2_figs.py`: digest at row 4,096 at f = 0.15 is `a6f5362d355c59fc`; at 0.25 and 0.10 the receipts' values.
- `e54_r2_flat.py`: 6 of 18 curves of E42 not flattened by the nine-row rule (G S 73129; F10 S all three; F10 B 73129 and 116089).
- The ratios on seed 73129; "3 to 12"; "7 to 11 against 15 to 62"; the spreads; "0.9 to 1.9"; the cost figures 11.6, 3.3, 38.6 and "up to 51".
- Nothing that the brief takes from `exp41_scale.py` (`item_key`, `taught_keys`, `assert_taught`, lines 126 to 138) depends on `FRACS`.
