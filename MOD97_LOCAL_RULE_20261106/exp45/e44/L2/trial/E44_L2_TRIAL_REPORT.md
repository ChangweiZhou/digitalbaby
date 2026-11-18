# E44, job L2, release 1: the trial's report (world 1, 1,024 visits, `naive` and `replay`), 18 November 2026

The roster note is `OPUS_TASK_E44_L2_ROSTER_FIRST_LONG_LIVES_20261117.md`, revision 2, on the branch `claude/e44-l2-first-long-lives`. Every figure below is in `E44_L2_TRIAL.json` and `E44_L2_TRIAL_TABLES.md` beside this file, written by `python3 smallworld/l2_report.py trial` from the two lives' receipts, event files and list files in `exp45/e44/L2/lives/`. The definitions D1 to D15 are the note's. The trial is one world on one birth (seed 73129); no statement here is a result of the roster, whose comparisons are made over worlds with intervals.

## 1. What was run

Two lives of world 1 at 1,024 visits of 256 updates, from E41's half-width backpropagation birth of seed 73129, with `--list-every 8 --decay-distances none`, decay 0.25: `naive`, and `replay` with ρ = 0.5. The runner is `smallworld/sw_run.py` with sha256 41714c5b… (commit f249b419), the same in both receipts. Host: Python 3.13.18, torch 2.15.0+cpu, numpy 2.5.3, one thread; 4 cores, at most three processes at once. CPU: 3.42 hours (`naive`) and 3.44 hours (`replay`), of which the list 0.08 hours each; 11.65 and 11.73 CPU-seconds a visit.

**Interruptions.** The lives were started at 21:04 UTC on 17 November. The machine was stopped at 21:15 UTC, at visit 57 of each, before any state had been saved; both were started again from the beginning at 22:07 UTC (`lives/*.starts.jsonl`). The container was restarted at about 00:57 UTC on 18 November, at visits 796 (`naive`) and 790 (`replay`); both were continued with `life --resume` from their states after 768 visits, `lives/lw_naive_w1_v1024_state_v768.pt` (sha256 17b8f9254d18b3958509d1bc86a5406e94111c9002cb8c0ea5d7e62d6b205246) and `lives/lw_replay_w1_v1024_rho0.5_state_v768.pt` (sha256 48596256a5b7f2257941cfd30908cd734b74d2c156347aea099fb26511a6982e), and each receipt records that one resume under `resumes` (path and file hash). These states are not in the receipts' list `states`, because a resumed life's list is the one held in the state it was resumed from; they were removed from the branch's tree afterwards and remain in its history. *(The states named on 18 November after the reviewer's verification.)* Check 5 showed that a resumed life is equal to an uninterrupted one in weights, optimizer, events, learned batches and list.

## 2. Checks on the files

Both list files have the item digest of the audit's section 7 for world 1 at 1,024 visits (52 families), and their row keys equal the reviewer's reference (part K of `e44_l2_checks.py`: 5,031 keys, 1,238,400 items, digest e7cdf44f…), with every item in the reviewer's order. The scorer (`smallworld/l2_score.py`) equals the reviewer's part J on every quantity it computes (D2 to D10), at 256 and at 1,024 visits, for both lives (`smallworld/l2_compare.py`).

## 3. The stop conditions

None holds.

| Condition | Figures |
|---|---|
| A loss that is not finite | Every visit's mean loss is finite; the largest is 0.474 (`naive`) and 0.385 (`replay`). |
| The store below capacity after visit 64 | The state after 64 visits holds 55,209 items (the capacity) in 938,638 of 1,269,828 bytes; full at the end; replay from the second update. |
| Score by age on reduction pairs below 0.05 over visits 512 to 1,023 in both arms | 0.673 (`naive`), 0.715 (`replay`). |
| A forecast for release 1 above 75 CPU-hours | 48.0 CPU-hours, from the trial's measured cost a visit with the list included (48.2 before the trial). |

## 4. `replay`'s first-visit level beside `naive`'s (change 8)

The median first-visit level (D6) over the 52 families met is 0.757 under `naive` and 0.595 under `replay`; over the 17 families met in the first 256 visits, 0.653 and 0.393. It is not below 0.20, so the trend of learning is read for `replay`, and release 2 needs no `replay` at ρ = 0.25 on this account.

## 5. Diagnostics and the quantities

Per life, for the first 256 visits and for all 1,024 (one world: no interval; the statements of the note are made over worlds).

| | `naive`, 256 | `replay`, 256 | `naive`, 1,024 | `replay`, 1,024 |
|---|---|---|---|---|
| Families met; first / continuation / return visits | 17; 17 / 68 / 171 | 17; 17 / 68 / 171 | 52; 52 / 151 / 821 | 52; 52 / 151 / 821 |
| Late level (D4) | 0.582 | 0.577 | 0.669 | 0.730 |
| … on returns / continuations / first visits | 0.554 / 0.738 / 0.587 | 0.574 / 0.638 / 0.480 | 0.652 / 0.814 / 0.676 | 0.728 / 0.769 / 0.651 |
| The blind predictor's late level | 0.014 | 0.014 | 0.015 | 0.015 |
| Score trend over the returns of the second half (D5) | −0.028 | −0.068 | −0.009 | +0.079 |
| … over all its visits (not used) | +0.067 | +0.001 | −0.008 | +0.087 |
| Trend of learning (D7) | −0.006 | −0.023 | +0.106 | +0.104 |
| … coefficient of ln n in the joint fit with ln(taught pairs) | −0.015 | −0.030 | +0.110 | +0.106 |

Score by age on reduction items (D2), in blocks of 64 visits, all visits: `naive` 0.782, 0.544, 0.563, 0.582, 0.643, 0.622, 0.624, 0.668, 0.662, 0.667, 0.680, 0.697, 0.685, 0.695, 0.656, 0.640; `replay` 0.748, 0.552, 0.571, 0.577, 0.623, 0.634, 0.641, 0.685, 0.694, 0.684, 0.709, 0.709, 0.721, 0.742, 0.731, 0.728. The first block has 44 continuations of 64 visits; later blocks 2 to 12. Every visit is in `E44_L2_TRIAL_score_by_age.csv`.

**Kept at return (D8) and relearning at return (D10), 1,024 visits.**

| Absence | Returns | Items | Kept, `naive` | Kept, `replay` | First 16 updates, `naive` / `replay` | Last 64 updates, `naive` / `replay` | Previous visit, last 64, `naive` / `replay` |
|---|---|---|---|---|---|---|---|
| 1 visit | 151 | 570 | 0.807 | 0.739 | 0.782 / 0.730 | 0.810 / 0.758 | 0.779 / 0.735 |
| 2 to 4 | 253 | 983 | 0.026 | 0.527 | 0.142 / 0.581 | 0.781 / 0.701 | 0.754 / 0.688 |
| 5 to 32 | 305 | 1,147 | 0.016 | 0.465 | 0.094 / 0.540 | 0.750 / 0.703 | 0.767 / 0.703 |
| 33 or more | 263 | 1,021 | 0.015 | 0.481 | 0.095 / 0.536 | 0.743 / 0.709 | 0.751 / 0.700 |
| 5 or more | 568 | 2,168 | 0.015 | 0.472 | 0.095 / 0.538 | 0.747 / 0.706 | 0.759 / 0.701 |

At 256 visits, kept after absences of 33 visits or more is 0.038 (`naive`, 79 items) and 0.456 (`replay`); after 5 or more, 0.022 and 0.433 (323 items, 85 returns).

**Kept on the list (D9), 1,024 visits** (mean of a family's share at a later eighth visit divided by its share at the end of its most recent visit): `naive` 0.022, 0.017, 0.014, 0.012 for absences of 1, 2 to 4, 5 to 32, 33 or more visits (113, 263, 1,481 and 2,021 points); `replay` 0.759, 0.688, 0.642, 0.642. No point was left out.

**What the wrong answers are (D14), 1,024 visits.** On the stream's first update of a return after 5 to 32 visits (1,147 answers), `naive`: right 0.016; the answer under the modulus of the family taught last 0.483; under an earlier family's modulus 0.175; the sum or difference without the reduction 0.114; another two-digit number 0.189; chance (the never-met modulus in the place of the last family's) 0.007. `replay`: right 0.465; the last family's modulus 0.017; an earlier family's 0.255; no reduction 0.027; another two-digit number 0.203; chance 0.013. On the list after 33 visits or more (404,200 answers), `naive`: last family's modulus 0.533, an earlier family's 0.211; `replay`: right 0.461, an earlier family's 0.355, last family's 0.015. All classes and absences are in the tables.

**D15: does a memory become more lasting as a family is met again?** Kept at return (D8), by the number of earlier visits to the family (1; 2 to 3; 4 to 7; 8 to 15; 16 or more), 1,024 visits, items in brackets:

| Arm | Absence | 1 | 2 to 3 | 4 to 7 | 8 to 15 | 16 or more |
|---|---|---|---|---|---|---|
| `naive` | 2 to 4 | 0.022 (46) | 0.027 (110) | 0.007 (141) | 0.035 (284) | 0.027 (402) |
| `naive` | 5 to 32 | 0.000 (62) | 0.009 (111) | 0.018 (221) | 0.018 (273) | 0.017 (480) |
| `naive` | 33 or more | 0.000 (52) | 0.031 (97) | 0.000 (192) | 0.008 (241) | 0.023 (439) |
| `replay` | 2 to 4 | 0.130 (46) | 0.373 (110) | 0.461 (141) | 0.504 (284) | 0.654 (402) |
| `replay` | 5 to 32 | 0.081 (62) | 0.252 (111) | 0.321 (221) | 0.549 (273) | 0.581 (480) |
| `replay` | 33 or more | 0.154 (52) | 0.340 (97) | 0.349 (192) | 0.519 (241) | 0.588 (439) |

The first 16 updates of the return, same cells (returns in brackets): `naive` 2 to 4: 0.132 (12), 0.128 (28), 0.108 (40), 0.146 (75), 0.157 (98); 5 to 32: 0.068, 0.050, 0.098, 0.106, 0.101; 33 or more: 0.055, 0.109, 0.074, 0.085, 0.112. `replay` 2 to 4: 0.247, 0.434, 0.504, 0.562, 0.709; 5 to 32: 0.334, 0.348, 0.449, 0.579, 0.640; 33 or more: 0.281, 0.426, 0.477, 0.530, 0.626.

- `naive`: kept does not rise with the number of earlier visits at a fixed class of absence; it stays between 0.000 and 0.035 in every cell. The figure after the first 16 updates is higher after 16 or more earlier visits than after one in all three classes (0.157 against 0.132, 0.101 against 0.068, 0.112 against 0.055), though not at every step between them. *(Corrected on 18 November after the reviewer's verification; the first version said that neither quantity rises.)*
- `replay`: both rise with the number of earlier visits in every class of absence; kept from 0.130, 0.081 and 0.154 after one earlier visit to 0.654, 0.581 and 0.588 after 16 or more, and the first 16 updates from 0.247, 0.334 and 0.281 to 0.709, 0.640 and 0.626.

(A family met more often has also offered more items to the store, and its returns come later in the life: under `replay` the median visit of the returns after one earlier visit and after 16 or more is 264 and 539 (absences of 2 to 4 visits), 434 and 642 (5 to 32), 572 and 696 (33 or more). This table separates neither the store nor the age of the life from the number of earlier visits. *(The age of the life was added on 18 November after the reviewer's verification.)*)

**The learning probe from the trial's states.** Each probe family's first-visit level (D6), the probe branch visiting families 52, 53, 54, 55 in that order (`probe/`; one world, so given as figures only): `naive` after 64 visits 0.863, 0.533, 0.408, 0.104; after 256, 0.934, 0.751, 0.616, 0.468. `replay` after 64, 0.570, 0.293, 0.129, 0.081; after 256, 0.529, 0.359, 0.342, 0.394. The branches after 1,024 visits had only their start records on the branch when this report was first committed (118b6047); they completed at 01:43 UTC, and all 24 branches enter the final report (`../E44_L2_REPORT.md`). *(Corrected on 18 November.)*

## 6. What follows

The roster continues without waiting, as the note says: world 2's `replay` and world 3's two lives of 1,024 visits started at 01:43 UTC; world 2's `naive` life is complete.
