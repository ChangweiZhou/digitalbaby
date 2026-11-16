# E44 job F1: the long-life world in the full-scale format — calibration report (16 November 2026)

Brief: `OPUS_TASK_E44_LONG_LIFE_WORLD_IN_THE_FULL_SCALE_FORMAT_20261116.md`, revision 2 (released), on `codex/minifly-consolidated-20261023` at cc00760d. That branch was at 7247682a when merged here; the brief is unchanged there. Audit: `AUDIT_E44_LONG_LIFE_WORLD_BRIEF_20261116.md` on `claude/vibrant-cerf-sje726` at dc05d575.

Work on `claude/gallant-edison-p0qa68`:
- the runner, scorer and checks 1–7 at 39e4e594;
- the driver's tables at 63e25029;
- the 96 calibration lives, the second lap, checks 1–8 complete and the tables at 4b203807;
- this report in the commit that adds it.

Owner's instruction for this session: do revision 2. Do the eight checks first; check 1's digests must match the reviewer's. Then the 48 calibration lives, the two further sets of 24, and the second lap at the adopted length. If the gate fails, take the one named further step and stop. List committed files from `git ls-tree`. Report the gate, the adopted length, the listed figures and P44.1–P44.11.

Evidence level:
- **TECHNICAL_PASS** for the instrument: checks 1–8 all pass.
- The calibration's figures are E2-type readings in this world. They are claims about nothing in E26's own lives.
- The brief's rule gives the visit length **256**.
- L2 is **NOT_RUN**; it follows by the coordinator's roster note. Nothing else is adopted.

Host: Linux x86-64 container, 4 cores, Python 3.13.18, torch 2.15.0+cpu, numpy 2.5.3. One thread a process, at most three processes, launched detached without a time limit.

## Summary

1. **All eight checks pass** (`checks/E44_CHECKS.json`; before the calibration, `checks/E44_CHECKS_PRE.json`).
   - **Check 1:** every digest of the audit's section 7 is equal, for worlds 1 to 8:
     - the in-turn streams at 64, 128 and 256;
     - the schedule of 1,024 visits;
     - the tasks;
     - the second lap at 64, 128 and 256;
     - the list.
   - **Check 1, receipts:** so are the 48 calibration receipts' stream digests and the 16 second-lap receipts' stream and list digests.
   - **Check 1, the world file:** `lw_world.py` is byte-identical to the specification file. Its output for worlds 1 to 3 equals `OUTPUT_LW_WORLD_MAC.txt`, and so does the output of the reviewer's reader on this host.
2. **The gate passes at 256 updates a visit and only there.** Under `naive`, the median over families 1 to 7 of the first-visit level on reduction pairs:

   | Visit length | Worlds in 0.20–0.90 | Medians, worlds 1 to 8 |
   |---|---|---|
   | 64 | 0 (all 8 below) | 0.053 to 0.133 |
   | 128 | 4 (4 below) | 0.096 to 0.434 |
   | 256 | **8** | 0.301 to 0.636 |

   A life of 1,024 visits at 256 updates is forecast at **3.85 CPU-hours** under `naive`, against a bound of 6. **The adopted visit length is 256.** The gate passed, so no further step was called or run.
3. **The second lap at 256** (16 branches, families 0 to 7 again from the state after 8 visits):
   - **First update of a return:** 0.016 under both arms, pooled share right on the family's reduction items (254 items each).
   - **First 16 updates:** 0.084 and 0.080.
   - **Last quarter of the return:** the family's figure is 0.583 (`naive`) and 0.666 (`sweep`), against 0.489 and 0.596 over the last quarter of the first visits.
4. **What is kept, one visit after a family's own visit:** the median over families 0–6 is 0.20 or less in all eight worlds under `naive` at 256 (0.007 to 0.076).
   - Decay alone over the same 256 updates keeps 0.891 (median, `naive`).
   - The birth family's R falls from 0.947 before the life to 0.084 after the first visit at 256 (`naive`, mean of eight worlds); decay alone leaves 0.875.
5. **Cost.** The 96 lives and 16 branches took 2.58 CPU-hours. CPU-seconds a visit: 3.40, 6.81 and 13.54 at 64, 128 and 256 under `naive`; 3.54, 7.26 and 14.89 under the sweep. The list adds about 1.0 a visit and is timed apart. Peak memory is 511 MiB a process.

## The learner and the ledger

| | |
|---|---|
| Learner | E41's half-width births of seed 73129 at f = 0.50 (`exp45/e41/e41_<B\|S>_h256_f50_73129.pt`): 40,960 updates of 64 items (2,621,440 taught items) on E26's first family, mod 89, made by E41. B: file 4c6d4302…, tensors 9be17d70…, A 94.50, R 94.65. S: file da14a18b…, tensors 59c5ea40…, A 95.83, R 91.95. 317,457 parameters (H = 256, U = 128). Every receipt records the file's sha256 |
| Loaded into | `sw_net.SWNet(256, 128)`: arch20.NetRNN('gru')'s parameter names in its order, as `exp41_scale.NetW`, run by `arch26.forward`. The brief names `NetW`, but `exp41_scale` cannot be imported beside this runner: its environment gate (exp20_reference → exp16_integrated) asserts on an `exp11_cell` already imported. Check 3 shows the two are the same network: equal tensors hash, and equal A and R counts under E41's own class and scorer and under the runner's |
| Arms | `naive`: B's birth, lives by backpropagation. `sweep`: S's birth, lives by the sweep. `sweep-after-naive`: B's birth, lives by the sweep. `naive` with decay 0: B's birth, backpropagation, `--wd 0` in the life. The runner asserts each arm's birth rule |
| Optimizer | One fresh `sw_net.make_opt(m, lr, wd)` for the life: AdamW, lr 0.001, betas 0.9/0.98, decay 0.25 (0 in the decay-0 set), the byte table without decay |
| Update | 64 stream items: 16 asked, answered first (greedy, 3 bytes, never learned from), then one step on the 48 taught items (`exp26_candidate.step`'s text). Every learned item is asserted taught and not asked |
| A calibration life | 8 visits, families 0 to 7 in turn (`--in-turn`): 512, 1,024 or 2,048 updates at 64, 128 or 256 a visit; 24,576, 49,152 or 98,304 taught presentations; 8,192, 16,384 or 32,768 answered asked items |
| The list (item 5) | At the end of every visit, without learning: for add and sub of every family taken so far, the first 100 asked reduction pairs in the split's order, no letters; and the birth family's 2,000 R items (E41's two samples, digests asserted against the birth receipt). The birth family's items are also answered once before visit 0. Every generated answer is written (`*_list.npz`) |
| Decay alone (no run) | At the end of a family's own visit, its weights with every parameter but the byte table multiplied by (1 − lr × decay)^u answer its list items for every later visit of the life, u the updates since; the born weights do the same for the birth family. Computed then, for every later visit, rather than from states saved at every visit: the same numbers, without storing a state a visit |
| Event file | E39's nine columns plus `g1`, `g2`, `g3` (uint16): the generated bytes, 65535 once generation has ended |
| States | One after 8 visits (`--save-states 8`). The resume check's life also has one after 4 |

**E41's width rule.** On the default branch E41 has reported for seed 73129 only; its other two seeds' receipts are not there. The calibration runs at the half width, as the brief says. The three-seed rule is to be read when L2 is rostered (brief, change 9).

## Checks 1 to 8

| Check | Result | Record |
|---|---|---|
| 1. The generator | **Passes.**<br>`lw_world.py` is byte-identical to `$K/lw_world_spec.py` (83b7df79…). Its output for worlds 1 to 3, run as a script (its assertions included), equals `OUTPUT_LW_WORLD_MAC.txt`; the reviewer's `e44_world_reader.py` (dc05d575), run here with `-I`, gives the same. The six committed digests are equal in full.<br>For worlds 1 to 8, every figure of the audit's section 7 is equal: in-turn streams at 64, 128 and 256, schedule of 1,024 visits, tasks, second lap at 64, 128 and 256, the list. That is 72 of 72.<br>The stream digest in each of the 48 calibration receipts equals the audit's for its world and visit length, and equals `lw_world.digest`.<br>The 16 second-lap receipts' stream digests (at 256) and list digests equal the audit's. | `E44_CHECKS_PRE.json` check1; `E44_CHECKS.json` check1_receipts, check1_second_lap |
| 2. Asked is never taught; no code or modulus used before | **Passes.**<br>In worlds 1 to 8 taught and asked are disjoint and complete. No modulus is shared with 73, 83, 89, 97 or 8–37. Code bytes run from 128 to 255, the codes are distinct, and the small world's and the birth family's code bytes are all below 128.<br>An asked pair offered as taught is refused by the runner's assertion.<br>The check lives verified every learned item (24,576 of 24,576). | pre check2 |
| 3. The learner is E41's | **Passes.**<br>For B and S, in a process that imports `exp41_scale` first, `NetW` with the file has the receipt's tensors hash. E41's own panels and scorer give the receipt's final A and R counts exactly: B 94.50 / 94.65, S 95.83 / 91.95.<br>The runner's `SWNet` with the same file has the same tensors hash. On the same items, `sw_net.answer` gives the same A and R counts. | pre check3 |
| 4. Row 0 | **Passes.**<br>The runner's scores on the first three families of worlds 1 and 2, before any update, use `lw_row0_check.py`'s items. They differ from `OUTPUT_LW_ROW0_MAC.txt` by at most **0.045 points** (rounding), against a bound of 2.<br>The coordinator's script, run on this host, prints the committed nine lines exactly.<br>The S birth on world 2 is tabled, not compared (no committed line). | pre check4 |
| 5. The small world still runs | **Passes.**<br>With `--world sw` a 4-visit life gives L0's stream digest (7ba33cb3…) and L0's final weights. Its nine original columns equal L0's file.<br>`g1`–`g3` are present: every right item has two digits and the end symbol, and generation ends after the end symbol. | pre check5 |
| 6. Asking does not teach; same stream; resume | **Passes.**<br>Answering leaves weights and optimizer state unchanged (in process, both arms, updates 0, 1, 31 and 63; and at the first update of every visit in the lives).<br>A life stopped after 4 visits and resumed equals the uninterrupted one: final hashes, event file, list file, stream and learned digests.<br>In all 24 world × length pairs, `naive` and `sweep` have the same stream, per-visit streams and learned-item digest, and equal arm-independent event columns. | pre check6; check6_same_stream |
| 7. The scorer's new quantity | **Passes.**<br>On the audit's hand-made file (world 1, eight families in turn) the scorer returns the audit's medians: last 64 updates / last quarter / whole first visit, 0.2502 / 0.2303 / 0.2502 at 64 and 0.2517 / 0.2517 / 0.2586 at 256. The column hashes are equal, and so is a direct count of each window's items.<br>The reviewer's `first_visit_levels` (taken from `e44_brief_checks.py` at dc05d575) agrees with this scorer on every family of `lw_naive_w1_v8_vu128_turn`, last quarter and whole visit. | pre check7; check7_calibration_life |
| 8. Cost | CPU-seconds a visit and the forecast are in the table under "Cost" | check8 |

## The gate (under `naive`)

The median over the second to the eighth family taken (families 1 to 7) of the first-visit level on reduction pairs. That level is the family's figure, the mean of add and sub on asked reduction items, over the last quarter of its first visit.

| Visit length | World 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | Worlds in 0.20–0.90 | above | below | CPU-s a visit | Forecast, 1,024 visits | Passes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 0.059 | 0.112 | 0.068 | 0.065 | 0.068 | 0.053 | 0.084 | 0.133 | 0 | 0 | 8 | 3.40 | 0.97 h | no |
| 128 | 0.169 | 0.434 | 0.169 | 0.322 | 0.213 | 0.096 | 0.195 | 0.301 | 4 | 0 | 4 | 6.81 | 1.94 h | no |
| 256 | 0.368 | 0.566 | 0.301 | 0.457 | 0.491 | 0.368 | 0.436 | 0.636 | **8** | 0 | 0 | 13.54 | **3.85 h** | **yes** |

**The shortest passing length is 256, so 256 is adopted.** The named further steps (32 updates a visit or a taught share of 0.05 for a ceiling at 64; 512 updates for a floor at 256; the stop) are for a gate that fails. None was called, and none was run.

## The listed figures

All per-life and per-family figures are in `F1/E44_F1.json`, with the tables `F1/E44_F1_TABLES.md` and three CSVs:
- `F1/E44_F1_first_visit.csv`: per world, arm, visit length and family, the first-visit level on reduction pairs, on pairs without reduction, max and min, with the modulus and taught pairs, in the last quarter, the whole first visit and the first 16 updates;
- `F1/E44_F1_kept.csv`;
- `F1/E44_F1_score_by_age.csv`.

The second lap is in `F1/lap2/E44_F1_second_lap.csv`. Pooled summaries follow; "family f" is the file's numbering, taken (f + 1)-th.

**The median over families 1–7 in every set**, with the trend of learning (least-squares slope of the family's figure on ln n, n = 2 to 8):

| Set | Arm | Visit length | Median, worlds 1–8 | Mean | Trend: mean slope, worlds positive |
|---|---|---|---|---|---|
| gate | naive | 64 | 0.059, 0.112, 0.068, 0.065, 0.068, 0.053, 0.084, 0.133 | 0.080 | −0.065, 2 of 8 |
| gate | sweep | 64 | 0.186, 0.355, 0.122, 0.182, 0.240, 0.112, 0.152, 0.233 | 0.198 | −0.038, 2 of 8 |
| gate | naive | 128 | 0.169, 0.434, 0.169, 0.322, 0.213, 0.096, 0.195, 0.301 | 0.237 | −0.091, 1 of 8 |
| gate | sweep | 128 | 0.452, 0.458, 0.396, 0.416, 0.420, 0.345, 0.422, 0.441 | 0.419 | −0.063, 3 of 8 |
| gate | naive | 256 | 0.368, 0.566, 0.301, 0.457, 0.491, 0.368, 0.436, 0.636 | 0.453 | −0.101, 2 of 8 |
| gate | sweep | 256 | 0.669, 0.656, 0.538, 0.561, 0.737, 0.537, 0.622, 0.595 | 0.614 | −0.039, 2 of 8 |
| further | sweep-after-naive | 64 | 0.026, 0.031, 0.036, 0.036, 0.042, 0.059, 0.029, 0.031 | 0.036 | −0.047, 2 of 8 |
| further | sweep-after-naive | 128 | 0.040, 0.047, 0.017, 0.067, 0.085, 0.054, 0.051, 0.045 | 0.051 | −0.045, 3 of 8 |
| further | sweep-after-naive | 256 | 0.157, 0.275, 0.117, 0.187, 0.182, 0.200, 0.212, 0.240 | 0.196 | −0.029, 4 of 8 |
| further | naive, decay 0 | 64 | 0.043, 0.098, 0.071, 0.076, 0.058, 0.053, 0.057, 0.124 | 0.073 | −0.062, 3 of 8 |
| further | naive, decay 0 | 128 | 0.169, 0.360, 0.135, 0.289, 0.232, 0.095, 0.178, 0.289 | 0.218 | −0.098, 1 of 8 |
| further | naive, decay 0 | 256 | 0.440, 0.612, 0.289, 0.457, 0.534, 0.319, 0.433, 0.680 | 0.471 | −0.132, 2 of 8 |

**First-visit level in the three windows** (mean over families 1–7 of the eight worlds):

| Set | Visit length | Window | Reduction (family's figure) | Without reduction | Max | Min |
|---|---|---|---|---|---|---|
| naive | 64 | first 16 updates / whole / last quarter | 0.023 / 0.082 / 0.136 | 0.144 / 0.193 / 0.264 | 0.562 / 0.770 / 0.894 | 0.609 / 0.797 / 0.913 |
| sweep | 64 | ″ | 0.042 / 0.137 / 0.230 | 0.286 / 0.406 / 0.523 | 0.730 / 0.850 / 0.918 | 0.718 / 0.847 / 0.919 |
| naive | 128 | ″ | 0.037 / 0.184 / 0.301 | 0.184 / 0.343 / 0.474 | 0.596 / 0.857 / 0.939 | 0.619 / 0.877 / 0.945 |
| sweep | 128 | ″ | 0.061 / 0.275 / 0.415 | 0.335 / 0.554 / 0.685 | 0.735 / 0.900 / 0.952 | 0.728 / 0.901 / 0.949 |
| naive | 256 | ″ | 0.037 / 0.327 / 0.471 | 0.198 / 0.483 / 0.612 | 0.580 / 0.903 / 0.952 | 0.598 / 0.913 / 0.956 |
| sweep | 256 | ″ | 0.063 / 0.434 / 0.588 | 0.342 / 0.654 / 0.749 | 0.721 / 0.926 / 0.956 | 0.736 / 0.929 / 0.960 |
| sweep-after-naive | 256 | ″ | 0.020 / 0.151 / 0.239 | 0.153 / 0.264 / 0.349 | 0.489 / 0.824 / 0.903 | 0.541 / 0.843 / 0.910 |
| naive, decay 0 | 256 | ″ | 0.035 / 0.332 / 0.479 | 0.209 / 0.520 / 0.664 | 0.589 / 0.910 / 0.958 | 0.595 / 0.917 / 0.961 |

(The other lengths of the two further sets are in `E44_F1_TABLES.md`.)

**The family's figure against its taught pairs and its place in turn** (last quarter, pooled over worlds, families 1–7; 14, 29 and 13 families in the three bins). The joint fit is the least-squares fit of the figure on ln n and ln(taught pairs) together:

| Set | Arm | Visit length | 264 or fewer | 265 to 688 | 689 or more | Joint fit: per ln n | per ln(taught pairs) |
|---|---|---|---|---|---|---|---|
| gate | naive | 64 | 0.066 | 0.141 | 0.201 | −0.068 | 0.099 |
| gate | sweep | 64 | 0.160 | 0.264 | 0.231 | −0.040 | 0.062 |
| gate | naive | 128 | 0.200 | 0.328 | 0.348 | −0.095 | 0.119 |
| gate | sweep | 128 | 0.347 | 0.446 | 0.417 | −0.066 | 0.078 |
| gate | naive | 256 | 0.378 | 0.496 | 0.515 | −0.105 | 0.110 |
| gate | sweep | 256 | 0.507 | 0.619 | 0.606 | −0.042 | 0.085 |
| further | sweep-after-naive | 256 | 0.156 | 0.258 | 0.283 | −0.032 | 0.095 |
| further | naive, decay 0 | 256 | 0.383 | 0.505 | 0.523 | −0.136 | 0.114 |

| Set | Arm | Visit length | n = 1 (family 0) | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| gate | naive | 64 | 0.243 | 0.197 | 0.194 | 0.092 | 0.100 | 0.112 | 0.193 | 0.065 |
| gate | sweep | 64 | 0.447 | 0.216 | 0.367 | 0.159 | 0.196 | 0.220 | 0.300 | 0.154 |
| gate | naive | 128 | 0.454 | 0.377 | 0.430 | 0.188 | 0.229 | 0.268 | 0.454 | 0.158 |
| gate | sweep | 128 | 0.561 | 0.426 | 0.581 | 0.295 | 0.395 | 0.398 | 0.493 | 0.315 |
| gate | naive | 256 | 0.616 | 0.536 | 0.641 | 0.376 | 0.354 | 0.442 | 0.616 | 0.332 |
| gate | sweep | 256 | 0.650 | 0.562 | 0.730 | 0.540 | 0.542 | 0.588 | 0.645 | 0.507 |

The moduli of families 0 to 7 differ by world (from the file):
- world 1: 69, 100, 62, 54, 45, 91, 80, 66;
- world 2: 59, 77, 60, 82, 66, 58, 61, 96;
- world 3: 68, 63, 98, 74, 50, 46, 95, 57;
- world 4: 70, 51, 80, 43, 88, 82, 42, 44;
- world 5: 86, 41, 88, 66, 46, 95, 79, 42;
- world 6: 51, 42, 59, 68, 55, 82, 94, 62;
- world 7: 100, 99, 57, 43, 84, 94, 59, 86;
- world 8: 43, 71, 51, 62, 68, 61, 82, 46.

**Score by age and the blind predictor** (share right over all asked items of each visit, mean over worlds). The blind predictor is 0.018–0.019 in every set.

| Set | Visit length | Visit 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|---|
| naive | 64 | 0.434 | 0.413 | 0.449 | 0.428 | 0.445 | 0.463 | 0.522 | 0.503 |
| sweep | 64 | 0.614 | 0.502 | 0.584 | 0.525 | 0.551 | 0.558 | 0.618 | 0.587 |
| naive | 128 | 0.555 | 0.539 | 0.571 | 0.531 | 0.546 | 0.560 | 0.642 | 0.568 |
| sweep | 128 | 0.693 | 0.613 | 0.699 | 0.625 | 0.643 | 0.662 | 0.706 | 0.657 |
| naive | 256 | 0.671 | 0.653 | 0.680 | 0.623 | 0.620 | 0.644 | 0.728 | 0.643 |
| sweep | 256 | 0.755 | 0.694 | 0.781 | 0.717 | 0.723 | 0.742 | 0.764 | 0.729 |
| sweep-after-naive | 256 | 0.502 | 0.499 | 0.497 | 0.476 | 0.494 | 0.533 | 0.592 | 0.551 |
| naive, decay 0 | 256 | 0.677 | 0.663 | 0.693 | 0.638 | 0.629 | 0.663 | 0.740 | 0.661 |

**What is kept, by the list, with decay alone beside it.** A family's share right on its 200 list items at a later visit, divided by its share at the end of its own visit. The table gives the median over the families of the eight worlds; how many families enter falls with the distance (56, 48, 40, 32, 8). In the decay-0 set at 64, one family with a share of 0 at its own visit is left out. Per family, with the modulus: `E44_F1_kept.csv`.

| Set | Arm | Visit length | Kept, visits since 1 | 2 | 3 | 4 | 7 | Decay alone, 1 | 2 | 3 | 4 | 7 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gate | naive | 64 | 0.168 | 0.064 | 0.180 | 0.127 | 0.167 | 1.000 | 0.983 | 0.943 | 0.835 | 0.650 |
| gate | sweep | 64 | 0.085 | 0.031 | 0.000 | 0.071 | 0.015 | 0.959 | 0.901 | 0.844 | 0.757 | 0.590 |
| gate | naive | 128 | 0.054 | 0.022 | 0.030 | 0.078 | 0.056 | 0.935 | 0.849 | 0.696 | 0.576 | 0.212 |
| gate | sweep | 128 | 0.022 | 0.000 | 0.009 | 0.016 | 0.022 | 0.922 | 0.778 | 0.643 | 0.472 | 0.122 |
| gate | naive | 256 | 0.021 | 0.015 | 0.010 | 0.019 | 0.016 | 0.891 | 0.651 | 0.418 | 0.227 | 0.014 |
| gate | sweep | 256 | 0.003 | 0.000 | 0.000 | 0.012 | 0.007 | 0.821 | 0.479 | 0.242 | 0.086 | 0.010 |
| further | sweep-after-naive | 64 | 0.500 | 0.375 | 0.292 | 0.453 | 0.148 | 1.000 | 1.000 | 1.000 | 0.938 | 0.683 |
| further | sweep-after-naive | 128 | 0.197 | 0.103 | 0.138 | 0.211 | 0.121 | 0.938 | 0.902 | 0.750 | 0.598 | 0.254 |
| further | sweep-after-naive | 256 | 0.071 | 0.032 | 0.030 | 0.062 | 0.071 | 0.848 | 0.617 | 0.414 | 0.221 | 0.031 |
| further | naive, decay 0 | 64 | 0.162 | 0.077 | 0.118 | 0.190 | 0.193 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| further | naive, decay 0 | 128 | 0.075 | 0.026 | 0.034 | 0.081 | 0.061 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| further | naive, decay 0 | 256 | 0.031 | 0.017 | 0.008 | 0.038 | 0.026 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

(With decay 0, decay alone multiplies by 1: the figure is the family's own weights unchanged.)

**The birth family** (mod 89, E41's 2,000 R items): share right before visit 0 and after visits 0, 1, 3 and 7, mean over worlds, with decay alone of the born weights:

| Set | Arm | Visit length | Before | After 0 | 1 | 3 | 7 | Decay alone: after 0 | 1 | 3 | 7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gate | naive | 64 | 0.947 | 0.195 | 0.116 | 0.062 | 0.076 | 0.935 | 0.920 | 0.875 | 0.702 |
| gate | sweep | 64 | 0.919 | 0.096 | 0.124 | 0.015 | 0.011 | 0.913 | 0.898 | 0.843 | 0.578 |
| gate | naive | 256 | 0.947 | 0.084 | 0.050 | 0.018 | 0.029 | 0.875 | 0.702 | 0.212 | 0.000 |
| gate | sweep | 256 | 0.919 | 0.059 | 0.074 | 0.004 | 0.003 | 0.843 | 0.578 | 0.099 | 0.001 |
| further | sweep-after-naive | 256 | 0.947 | 0.111 | 0.066 | 0.025 | 0.037 | 0.875 | 0.702 | 0.212 | 0.000 |
| further | naive, decay 0 | 256 | 0.947 | 0.099 | 0.045 | 0.021 | 0.036 | 0.947 | 0.947 | 0.947 | 0.947 |

**The second lap at 256** (16 branches; means over eight worlds; the first update and first 16 updates are the pooled share right on the family's asked reduction items of add and sub):

| Arm | Family | First update | First 16 updates | Last quarter: return | Last quarter: first visit | List: end of own visit | end of lap 1 (visit 7) | end of return |
|---|---|---|---|---|---|---|---|---|
| naive | 0 | 0.014 | 0.102 | 0.672 | 0.616 | 0.587 | 0.021 | 0.636 |
| naive | 1 | 0.000 | 0.161 | 0.633 | 0.536 | 0.534 | 0.057 | 0.621 |
| naive | 2 | 0.075 | 0.119 | 0.736 | 0.641 | 0.611 | 0.032 | 0.705 |
| naive | 3 | 0.000 | 0.066 | 0.500 | 0.376 | 0.360 | 0.034 | 0.475 |
| naive | 4 | 0.000 | 0.055 | 0.437 | 0.354 | 0.370 | 0.005 | 0.431 |
| naive | 5 | 0.000 | 0.056 | 0.562 | 0.442 | 0.437 | 0.021 | 0.547 |
| naive | 6 | 0.000 | 0.071 | 0.692 | 0.616 | 0.573 | 0.034 | 0.661 |
| naive | 7 | 0.000 | 0.049 | 0.434 | 0.332 | 0.298 | 0.298 | 0.399 |
| sweep | 0 | 0.000 | 0.043 | 0.710 | 0.650 | 0.597 | 0.006 | 0.657 |
| sweep | 1 | 0.000 | 0.109 | 0.671 | 0.562 | 0.493 | 0.019 | 0.623 |
| sweep | 2 | 0.050 | 0.127 | 0.763 | 0.730 | 0.689 | 0.004 | 0.722 |
| sweep | 3 | 0.000 | 0.074 | 0.638 | 0.540 | 0.479 | 0.016 | 0.580 |
| sweep | 4 | 0.000 | 0.073 | 0.589 | 0.542 | 0.501 | 0.015 | 0.546 |
| sweep | 5 | 0.042 | 0.070 | 0.653 | 0.588 | 0.545 | 0.022 | 0.627 |
| sweep | 6 | 0.021 | 0.098 | 0.724 | 0.645 | 0.570 | 0.039 | 0.648 |
| sweep | 7 | 0.000 | 0.059 | 0.581 | 0.507 | 0.466 | 0.466 | 0.539 |

A return's first update has about 4 reduction items a family, 254 an arm over all returns. Pooled, the first update is 0.016 under both arms, and the first 16 updates are 0.084 (`naive`) and 0.080 (`sweep`). For family 7 the end of lap 1 is the end of its own visit. The birth family's R after the second lap is 0.024 (`naive`) and 0.003 (`sweep`).

## Cost

CPU-seconds a visit cover the visit's updates with the answering of their asked items; the list is timed apart. The forecast is 1,024 × the mean a visit.

| Set | Arm | Visit length | CPU-s a visit | List a visit | Forecast, 1,024 visits | Peak memory |
|---|---|---|---|---|---|---|
| gate | naive | 64 | 3.40 | 0.99 | 0.97 h | 509 MiB |
| gate | sweep | 64 | 3.54 | 0.97 | 1.01 h | 509 MiB |
| gate | naive | 128 | 6.81 | 1.01 | 1.94 h | 509 MiB |
| gate | sweep | 128 | 7.26 | 1.00 | 2.07 h | 508 MiB |
| gate | naive | 256 | 13.54 | 0.99 | 3.85 h | 510 MiB |
| gate | sweep | 256 | 14.89 | 1.02 | 4.23 h | 508 MiB |
| further | sweep-after-naive | 256 | 14.17 | 0.98 | 4.03 h | 508 MiB |
| further | naive, decay 0 | 256 | 12.90 | 0.95 | 3.67 h | 511 MiB |

That is 0.053 CPU-seconds an update under `naive` and 0.055–0.058 under the sweep. The audit estimated from E41's rates (learning from 64 items, nothing asked) 2.5 to 3.1 CPU-hours for 1,024 visits at 256; the measured forecast is 3.85 and 4.23.

The list in a life of 1,024 visits over 52 families would be about 52 × 200 + 2,000 items a visit, not measured here.

All runs took 2.58 CPU-hours for the 96 lives and 16 branches. The check lives took under 0.03, and the checks' scoring a few minutes.

## Predictions

| | Prediction (the brief's) | Chance | Outcome |
|---|---|---|---|
| P44.1 | The reviewer's implementation gives the six digests of check 1 at the first attempt | 80% | **True**, by the audit (its reader gave the committed output line for line "at the first attempt"). Here its reader, run as committed at dc05d575, prints `OUTPUT_LW_WORLD_MAC.txt` exactly |
| P44.2 | Check 4: the worker's row-0 scores agree with the coordinator's to 2 points | 85% | **True**: largest difference 0.045 points |
| P44.3 | The gate passes at some visit length of 64, 128, 256 | 65% | **True**: at 256 (8 of 8 worlds; 3.85 CPU-hours) |
| P44.4 | The shortest passing length is 64 | 30% | **False**: 256 |
| P44.5 | At 256 the median is above 0.90 in six of eight worlds | 20% | **False**: 0 of 8 (highest 0.636) |
| P44.6 | At 64 it is below 0.20 in six of eight worlds | 35% | **True**: 8 of 8 (0.053 to 0.133) |
| P44.10 | At 64 it is above 0.90 in six of eight worlds | 5% | **False**: 0 of 8 |
| P44.11 | At 256 it is below 0.20 in six of eight worlds | 15% | **False**: 0 of 8 (lowest 0.301) |
| P44.7 | Under `naive` at 128, the mean first-visit level of families 5–7 is above that of families 1–3 in six of eight worlds | 60% | **False**: 2 of 8 (worlds 1 and 3) |
| P44.8 | The sweep's median first-visit level is within 0.15 of `naive`'s at the adopted length, in six of eight worlds | 45% | **False**: 3 of 8 (worlds 2, 4 and 8; differences 0.041 to 0.301) |
| P44.9 | At the adopted length, the median over families 0–6 of a family's list share one visit after its own, divided by its share at the end of its own visit, is 0.20 or less in six of eight worlds under `naive` | 80% | **True**: 8 of 8 (0.007 to 0.076; no family left out) |

## What this does not show

The brief's limits stand:
- no life of more than 52 families;
- whether the level reflects learning the modulus or the codes;
- nothing at the present width;
- a result here is a claim only when repeated on E26's own lives.

In addition:
- **Lives of 8 visits only.** No absence beyond 7 visits (1,792 updates at 256), and every return is in the second lap's fixed order. What 1,024-visit lives give is L2's.
- **One birth seed** (73129) in every world. E41's three-seed table is not on the default branch.
- **The two further sets are descriptions**, outside the gate, each a single run per world.
- **Same-author scoring.** The reviewer's scorer was run on one calibration life and on the audit's fixture (check 7); nothing else was recomputed independently. No formal checker or seal exists; none is claimed.

## Committed files

From `git ls-tree -r -l 4b203807` over `MOD97_LOCAL_RULE_20261106/exp45/e44` and the five sources: 489 files, 68.6 MB. This report is added in the next commit.

| Where | What | Files |
|---|---|---|
| `smallworld/` | `lw_world.py` (the specification, byte-identical), `lw_checks.py`, `lw_f1.py`, `sw_run.py`, `sw_score.py` | 5 |
| `exp45/e44/checks/` | `E44_CHECKS_PRE.json`, `E44_CHECKS.json` | 2 |
| `exp45/e44/checks/lives/` | the check lives: receipts (3), event files (3), list files (2), logs (5) | 13 |
| `exp45/e44/F1/` | receipts of the 96 lives (`lw_<arm>_w<1-8>_v8[_vu64\|_vu128][_wd0]_turn.json`) and `E44_F1.json` | 97 |
| `exp45/e44/F1/` | event files of the 96 lives (`*.npz`) | 96 |
| `exp45/e44/F1/` | the answers of item 5 (`*_list.npz`) | 96 |
| `exp45/e44/F1/` | the states after 8 visits of world 1, all 12 settings (`lw_*_w1_*_state_v8.pt`) | 12 |
| `exp45/e44/F1/` | `E44_F1_TABLES.md`, three CSVs, the driver log | 5 |
| `exp45/e44/F1/logs/` | the lives' logs | 96 |
| `exp45/e44/F1/lap2/` | second lap: receipts (16), event files (16), list files (16), CSV (1), driver log (1) | 50 |
| `exp45/e44/F1/lap2/logs/` | the branches' logs | 16 |
| `exp45/e44/.gitignore` | the storage rule | 1 |

The states of worlds 2 to 8, the second lap's end states and the check lives' states are not committed. Each is recorded by its file hash, weights hash and optimizer hash in its receipt (`states[]`).

The full listing (mode, type, blob, size, path) follows at the end of this report.

## Decision record

- **Anchor.** `claude/gallant-edison-p0qa68`, brief E44 revision 2 (cc00760d).
  - World: `lw_world.py` (83b7df79…).
  - Learner: E41's half-width births of seed 73129 (B 4c6d4302…, S da14a18b…).
  - Runner and scorer: `sw_run.py` and `sw_score.py` at 39e4e594, unchanged through every run. The 112 life and branch receipts record commits 39e4e594 (3) and d7ddf82c (109); both have the same source hashes (`sw_run.py` 5d643e2d…) and no modified source.
- **Question.** At which visit length does a born learner's first-visit level on reduction pairs of a new family sit between floor and ceiling, within the cost bound?
- **Falsifier.** The gate, per visit length, under `naive`.
- **What ran.**
  - Checks 1–8.
  - 48 gate lives, 24 `sweep-after-naive` and 24 `naive` with decay 0, all in turn over eight families at 64, 128 and 256.
  - 16 second-lap branches at 256.
  - 2.58 CPU-hours. Nothing else.
- **Result.** The gate passes at 256 (8 of 8 worlds; 3.85 CPU-hours forecast). It fails at 64 (0 of 8, all below the floor) and at 128 (4 of 8). P44.1, 2, 3, 6 and 9 hold; P44.4, 5, 7, 8, 10 and 11 do not.
- **Disposition.** Visit length 256 by the brief's rule. No further step. L2 NOT_RUN.
- **Next action.** None by me. L2 follows by the coordinator's one-page roster note, which also reads E41's width rule on the seeds then on file.

## Appendix: `git ls-tree -r -l 4b203807` (E44's files)

```
100644 blob 0e498b036f5f9c50ad906513b3ba26a09c103b15     237	MOD97_LOCAL_RULE_20261106/exp45/e44/.gitignore
100644 blob 9e76b3de7e964332d6f087269226267080d4cd0b 3184934	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/E44_F1.json
100644 blob f41e2a441da5ea01bfd88b2bc03cf186cedf33e4   16057	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/E44_F1_TABLES.md
100644 blob b67c52be607c6ba533d8dd59f4d4bfba82ee7fa7  444245	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/E44_F1_first_visit.csv
100644 blob 9c7e15bd796c5ed6e30e2ed523358e721a5064c9  282745	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/E44_F1_kept.csv
100644 blob fc3b7b1baef05357a48dcde98d7c87e68666640d   38426	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/E44_F1_score_by_age.csv
100644 blob 001376dd7eb5649677a541f03b4a14b02bc104b6    3534	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/f1_driver.log
100644 blob 7ebcc4b6ce65118186f2fe9bf74240c5bc68769a   13142	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/E44_F1_second_lap.csv
100644 blob 161f41ffa09ae80240d85c433ec184627eadacce     698	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lap2_driver.log
100644 blob e9659c0fd6264bc4ad36e8543e7e0e648f144e86      51	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/logs/lw_naive_w1_v8_turn_state_v8_lap2.log
100644 blob ede728b37e62244208016e02b6a83ddd9c376bbc      51	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/logs/lw_naive_w2_v8_turn_state_v8_lap2.log
100644 blob 4f974bd56a71e63163447c18031b32b4635f28c2      51	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/logs/lw_naive_w3_v8_turn_state_v8_lap2.log
100644 blob 8c8b370ae095bc6cca55dcbdb3aa7ddfeb280db2      51	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/logs/lw_naive_w4_v8_turn_state_v8_lap2.log
100644 blob daf1b59b9707ac1972744282f536c8569d4a6fd6      51	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/logs/lw_naive_w5_v8_turn_state_v8_lap2.log
100644 blob 467832907b914ce4bfb0f1dfe0dcee79e179d272      51	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/logs/lw_naive_w6_v8_turn_state_v8_lap2.log
100644 blob 1962d8ed9bb184717c92c8e5a82fea5f1fcb611d      51	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/logs/lw_naive_w7_v8_turn_state_v8_lap2.log
100644 blob 044dba6f15371464c8fb49632716c6174e543310      51	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/logs/lw_naive_w8_v8_turn_state_v8_lap2.log
100644 blob a0956f845b88d2a3b9eb860afb5f680a03d82bd4      51	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/logs/lw_sweep_w1_v8_turn_state_v8_lap2.log
100644 blob 2032dca7e4fb41c781dd7e2a8d79d3467d415cd8      51	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/logs/lw_sweep_w2_v8_turn_state_v8_lap2.log
100644 blob 3c19b595b257dbe8d91317494a314e4ef82ad4e4      51	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/logs/lw_sweep_w3_v8_turn_state_v8_lap2.log
100644 blob 9f8c61804d7dd0b7204bdfb4ed471df0d573e187      51	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/logs/lw_sweep_w4_v8_turn_state_v8_lap2.log
100644 blob 71d4a8a90b59169860c40d5b548f40938e83dc88      51	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/logs/lw_sweep_w5_v8_turn_state_v8_lap2.log
100644 blob eca0ef8be467d643adcc063dd5302b3960ab9e13      51	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/logs/lw_sweep_w6_v8_turn_state_v8_lap2.log
100644 blob 70bf4ee2bdf9f806739ce22d7006a744571c6ad1      51	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/logs/lw_sweep_w7_v8_turn_state_v8_lap2.log
100644 blob a4b21df9aaf175242f578db75b494e61cd7ceaf7      51	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/logs/lw_sweep_w8_v8_turn_state_v8_lap2.log
100644 blob 60406a4d70282f40cae7134a7ab2e57809253652   26714	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w1_v8_turn_br8_g0-1-2-3-4-5-6-7.json
100644 blob 07d93f2e4750c0b93150e2353e302b8f66395971  100792	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w1_v8_turn_br8_g0-1-2-3-4-5-6-7.npz
100644 blob c1a787ba21b5eb9ae0409023e142bb7b7c7e453b   41941	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w1_v8_turn_br8_g0-1-2-3-4-5-6-7_list.npz
100644 blob d7ab4a946859ac39dddc1ec30aedcfdcc610ae1f   26680	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w2_v8_turn_br8_g0-1-2-3-4-5-6-7.json
100644 blob 1c24423e81c9bb2172818921c63d594cd24b9199  100528	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w2_v8_turn_br8_g0-1-2-3-4-5-6-7.npz
100644 blob 3d5497ac9f7dfdfd8c6cdb81b80c56ec2914fbbf   42283	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w2_v8_turn_br8_g0-1-2-3-4-5-6-7_list.npz
100644 blob c47aaa2338c2f720cd9331a06d2b9f50f8db9506   26687	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w3_v8_turn_br8_g0-1-2-3-4-5-6-7.json
100644 blob 3761d3dd0b6623b0739f7bfe70b5b03d32353699  100468	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w3_v8_turn_br8_g0-1-2-3-4-5-6-7.npz
100644 blob 205f05e3d7db945197357433274e0f2eec8d563d   41530	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w3_v8_turn_br8_g0-1-2-3-4-5-6-7_list.npz
100644 blob b5b77d80cd59397f65733fb1129f87c0cdbe9aa5   26715	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w4_v8_turn_br8_g0-1-2-3-4-5-6-7.json
100644 blob d7af94770e4c1b77178104d818823e64a9310a60   99990	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w4_v8_turn_br8_g0-1-2-3-4-5-6-7.npz
100644 blob a5cd11679f4d62fc5932782d5440c0ed9691efdc   41272	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w4_v8_turn_br8_g0-1-2-3-4-5-6-7_list.npz
100644 blob 5be2b05bed607410024146db641daac8af123919   26723	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w5_v8_turn_br8_g0-1-2-3-4-5-6-7.json
100644 blob 010e1096fac8dc5e85c0d5806ab2b911d9729e58  100156	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w5_v8_turn_br8_g0-1-2-3-4-5-6-7.npz
100644 blob 97be25bbeec06ba06adcab822c2ed5ed12996137   42222	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w5_v8_turn_br8_g0-1-2-3-4-5-6-7_list.npz
100644 blob 0ab62da0401cf0f7e8464b3b43a33911fdf11b21   26727	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w6_v8_turn_br8_g0-1-2-3-4-5-6-7.json
100644 blob 48e22af689e7fd09937c12f365787e8932167124   99874	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w6_v8_turn_br8_g0-1-2-3-4-5-6-7.npz
100644 blob 53f460428886fba87b36235672bea097a58c4793   41081	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w6_v8_turn_br8_g0-1-2-3-4-5-6-7_list.npz
100644 blob 5ce0bfd08e28bcbd2036f1e29338ac30a44dd6c0   26707	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w7_v8_turn_br8_g0-1-2-3-4-5-6-7.json
100644 blob 4c9813d7e13afdd982ddcc67a30713a4502dd788  101342	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w7_v8_turn_br8_g0-1-2-3-4-5-6-7.npz
100644 blob 16d7346f74567ebc5a2ba0ae54cf6da4ed0b488c   42352	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w7_v8_turn_br8_g0-1-2-3-4-5-6-7_list.npz
100644 blob 766d4d593ac3c26c3aa8d455d2eebb485318741a   26718	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w8_v8_turn_br8_g0-1-2-3-4-5-6-7.json
100644 blob fbd6d207187302b2f0594dc243107895868e0583   99361	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w8_v8_turn_br8_g0-1-2-3-4-5-6-7.npz
100644 blob 87f1350a0edba64e34642d8c4d66238cc5bd85f3   40443	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_naive_w8_v8_turn_br8_g0-1-2-3-4-5-6-7_list.npz
100644 blob 693cff7b4c23b477aae296ef838b1cc99fc655bd   26710	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w1_v8_turn_br8_g0-1-2-3-4-5-6-7.json
100644 blob 42da609a3e65c49d63c5fa78aa6a21387fe4811f  100431	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w1_v8_turn_br8_g0-1-2-3-4-5-6-7.npz
100644 blob 2b4ae4b2145ec433272eb33687f60c5ad15c7be6   41920	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w1_v8_turn_br8_g0-1-2-3-4-5-6-7_list.npz
100644 blob 13fe7a40d61e4d93d42a42317977d255eb1a3897   26702	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w2_v8_turn_br8_g0-1-2-3-4-5-6-7.json
100644 blob 22f7874c79c7ed140dcb49c226ae3ffdf2ddfbfa  100040	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w2_v8_turn_br8_g0-1-2-3-4-5-6-7.npz
100644 blob 5059686c915e682544540a7ca8cddc17ec02838a   41872	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w2_v8_turn_br8_g0-1-2-3-4-5-6-7_list.npz
100644 blob 9f8b0cb930c9304220927687559497c8e36040ef   26757	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w3_v8_turn_br8_g0-1-2-3-4-5-6-7.json
100644 blob 018b717fe0032e59ed21f1ec4e56fa0a1b3539ba  100381	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w3_v8_turn_br8_g0-1-2-3-4-5-6-7.npz
100644 blob 9a5cac93b9ec173d7434e6be71a7dbd1d31a0647   42150	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w3_v8_turn_br8_g0-1-2-3-4-5-6-7_list.npz
100644 blob 364f8fa1cfa164b6332b6a0a9c0bdd87a8f1a7bc   26743	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w4_v8_turn_br8_g0-1-2-3-4-5-6-7.json
100644 blob 035e501acbe2e7fcb68fb1ce675299ebdf941d6b   99600	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w4_v8_turn_br8_g0-1-2-3-4-5-6-7.npz
100644 blob e8356a0e5e4e033c9d21b98bcef9f32bf936b481   41022	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w4_v8_turn_br8_g0-1-2-3-4-5-6-7_list.npz
100644 blob a2fd1f37274fd983d81442347b9cf6f31d4b4d30   26687	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w5_v8_turn_br8_g0-1-2-3-4-5-6-7.json
100644 blob 7ba386f72a7a5a566ca38b3881027707e47f94ce  100173	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w5_v8_turn_br8_g0-1-2-3-4-5-6-7.npz
100644 blob ed0985d109000e783592c024f624cd018562ebf3   41626	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w5_v8_turn_br8_g0-1-2-3-4-5-6-7_list.npz
100644 blob bbae48d8d430bc7e432d08ea23bd343cc03fc53b   26724	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w6_v8_turn_br8_g0-1-2-3-4-5-6-7.json
100644 blob 55b7fc525de8cd5ea7c3ccaac8256502e9e5ec01   99506	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w6_v8_turn_br8_g0-1-2-3-4-5-6-7.npz
100644 blob 7c072965adef5b558b29819ecb4bc04db7ef44e5   41458	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w6_v8_turn_br8_g0-1-2-3-4-5-6-7_list.npz
100644 blob 58c8f9989d44794f16ee34c76bbca174221852fe   26726	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w7_v8_turn_br8_g0-1-2-3-4-5-6-7.json
100644 blob aa554d45fdf9b13c8f89bf9d22fcf3ebc2a0d98e  100933	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w7_v8_turn_br8_g0-1-2-3-4-5-6-7.npz
100644 blob d2a5774659ded52db31195df1f23526519cd9bf9   42574	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w7_v8_turn_br8_g0-1-2-3-4-5-6-7_list.npz
100644 blob e2a9c44027421fbc8293989d84efb53a29e4a2e1   26699	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w8_v8_turn_br8_g0-1-2-3-4-5-6-7.json
100644 blob 2f13541563c25760ca50bafd251b78d652b38cb4   99226	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w8_v8_turn_br8_g0-1-2-3-4-5-6-7.npz
100644 blob dacd278e6fed8e1b2d686b2748b23d08c083e465   40635	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lap2/lw_sweep_w8_v8_turn_br8_g0-1-2-3-4-5-6-7_list.npz
100644 blob a4194ac564c2a0441356cc940df75e7552860ba6     800	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w1_v8_turn.log
100644 blob 18061b8204ee45715cfef16d11719bb6f7af2ea1     852	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w1_v8_vu128_turn.log
100644 blob abd94789928c7d39bb2dcc85cc37597c64237e5f     892	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w1_v8_vu128_wd0_turn.log
100644 blob e926b5fa708e03d8bf0859bf2d037198eef88e08     842	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w1_v8_vu64_turn.log
100644 blob cca9ecfefa06b960b2d76cec0bb8304580a63c36     882	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w1_v8_vu64_wd0_turn.log
100644 blob 8ab68492dd0b35d099d0c6f4c5de1ee183385ba8     840	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w1_v8_wd0_turn.log
100644 blob 785e584d4ba8334c8792828cc87bebe18340dc01     800	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w2_v8_turn.log
100644 blob 68f8a895708bcbc9094bb49a427599721e828036     852	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w2_v8_vu128_turn.log
100644 blob 45cd4dc2f5812fafd1021d25a78c01e8240c7af9     892	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w2_v8_vu128_wd0_turn.log
100644 blob 08c67811d9f17a822524bb5dec5d3614bda38245     842	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w2_v8_vu64_turn.log
100644 blob f29a6089787b2f587d7ab1f5b9a74f6b880b7429     882	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w2_v8_vu64_wd0_turn.log
100644 blob 0421c0c0aa05c1fd23b6968b3a6c1c918c2594e1     840	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w2_v8_wd0_turn.log
100644 blob ddac83a55dfcc7f74901a5af11b5513ec335b6f5     800	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w3_v8_turn.log
100644 blob f4eaafcdc7f5d27e34552a05cbb593af4655cdc2     852	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w3_v8_vu128_turn.log
100644 blob 85996417d3935e692c8687f728a50956601d087b     892	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w3_v8_vu128_wd0_turn.log
100644 blob 9cbb32c18e102d44cc8c5af8eaf167f35257c330     842	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w3_v8_vu64_turn.log
100644 blob 22b9e7ebd4860bd0873ef427ce0da051450c3380     882	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w3_v8_vu64_wd0_turn.log
100644 blob 8547a2ecb28029cf5dcffffeb64bb5c36703de02     840	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w3_v8_wd0_turn.log
100644 blob 4773f357d19d0bc2f6a8d2d60550dea2cf080e43     800	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w4_v8_turn.log
100644 blob 60f8eacb979ef9cc9a503bef1ab0dbec4eaf05ff     852	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w4_v8_vu128_turn.log
100644 blob 5644ee860eabb040f943fbcf807a74ea975fef06     892	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w4_v8_vu128_wd0_turn.log
100644 blob 8eebe672c512b4232e243a1fa0d433a7bc33e5a9     842	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w4_v8_vu64_turn.log
100644 blob 83b7c2e5fd922b8aa4715af1f88863e4ee11e125     882	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w4_v8_vu64_wd0_turn.log
100644 blob 2e7b02d1c3ecd78851e1774544eaa43d7ce35f4b     840	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w4_v8_wd0_turn.log
100644 blob 9f9638a079ea7f1edd8ef5987602c4327af338f2     800	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w5_v8_turn.log
100644 blob 0317e248b58d160a6bbf6b2ff05d02ea47841455     852	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w5_v8_vu128_turn.log
100644 blob 763ff725d7bb8f76a83f14c4119ce6cc0ee33e3a     892	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w5_v8_vu128_wd0_turn.log
100644 blob 7bd2758ace9d008bd7ce577424bf5f6e0dc0c2b9     842	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w5_v8_vu64_turn.log
100644 blob 9ff314bcdb5a1f3063ffaeaa15f1503194bb37e4     882	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w5_v8_vu64_wd0_turn.log
100644 blob 5b6b1741b61e8ebd7aa9a6ac7695fdba7e851f4f     840	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w5_v8_wd0_turn.log
100644 blob be6fe5f8f73afa7d1ae5b71b639954c12e3ae4ec     800	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w6_v8_turn.log
100644 blob 1d3143c9438f96215aa66e52269f81d119197aac     852	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w6_v8_vu128_turn.log
100644 blob 1854cf430b3189b738e41f4d822d87063d59df4b     892	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w6_v8_vu128_wd0_turn.log
100644 blob 0308de1d04213dcbde5c790d1f49b6d25478a5d7     842	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w6_v8_vu64_turn.log
100644 blob fd3f8f22995a818e5a8e0186e2afc39defbd0ad5     882	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w6_v8_vu64_wd0_turn.log
100644 blob 95a9c238ae9b8d6cb34c7d9cc209cedde16044d1     840	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w6_v8_wd0_turn.log
100644 blob d94353c92e17fe9a4d8497bec5f96ca1e2db2a11     800	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w7_v8_turn.log
100644 blob c0c1165a0f14c7f48ca20b29841574660b55349f     852	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w7_v8_vu128_turn.log
100644 blob 1f316c4755ad92f2180ee0a993d68519351becf2     892	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w7_v8_vu128_wd0_turn.log
100644 blob 79dc46b42dd922a089c82161ba93a8ece525245a     842	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w7_v8_vu64_turn.log
100644 blob bc000a529ef8de8d49ce3ed6968e66372788fc19     882	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w7_v8_vu64_wd0_turn.log
100644 blob f724b5ae55d0165467095aa3b3f2eb83847aa740     840	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w7_v8_wd0_turn.log
100644 blob 4749b981ffeb4182fee5a31b4f9cc9f4be463132     800	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w8_v8_turn.log
100644 blob 61fd2c6c789584d71828245733ac059bc1f74cdf     852	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w8_v8_vu128_turn.log
100644 blob 005c92a177bf2c5f71f412be82e559bdd15cb5d0     892	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w8_v8_vu128_wd0_turn.log
100644 blob 77d90ef8445b44f54c75486713e485265774d775     842	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w8_v8_vu64_turn.log
100644 blob 1b9f87c297bb2cabce258eabd987a32757d1579d     882	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w8_v8_vu64_wd0_turn.log
100644 blob a49b42808c408bc0c5f94fdd022f603dfeb82e61     840	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_naive_w8_v8_wd0_turn.log
100644 blob 51a87f4c5f68780768bd55a1304e106bfe53ebac     920	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w1_v8_turn.log
100644 blob 6bdf0f9fc3399ff2819152521a27580290006556     972	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w1_v8_vu128_turn.log
100644 blob ebaacbad5d6efd0c3669e3cd3f9c5b034c6502e9     962	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w1_v8_vu64_turn.log
100644 blob 2521d0de54ddf3b2ebf956e2987e6705e9c47439     920	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w2_v8_turn.log
100644 blob d2a40b50a278863a020796afa04907e5d1ae7e4a     972	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w2_v8_vu128_turn.log
100644 blob 4dd9c39acf337ff3dfae6f7af2b7bc0cd19d395b     962	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w2_v8_vu64_turn.log
100644 blob 66006a8f7075079d32068bb2f0515a58b7c720be     920	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w3_v8_turn.log
100644 blob 69d1526ad026de1857316dfbe0d91f08032118d0     972	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w3_v8_vu128_turn.log
100644 blob 40a7372b27aff882df67cc77598621e362ebbf20     962	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w3_v8_vu64_turn.log
100644 blob da17cbd451e39a219cc73a0cb1521905c326a1d2     920	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w4_v8_turn.log
100644 blob 30dce807f6a9b65d89f4315089645df85a54a7f2     972	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w4_v8_vu128_turn.log
100644 blob 0b331877378a6f38a54198255b36dd2993943a84     962	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w4_v8_vu64_turn.log
100644 blob a08a8a62445f818396f572c77c20889768c0e69c     920	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w5_v8_turn.log
100644 blob f88e149d9422ecb68432861c7a8f6083760ee657     972	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w5_v8_vu128_turn.log
100644 blob a7ce65e7cec605250c8981322fb43114ee4d75dc     962	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w5_v8_vu64_turn.log
100644 blob dd7c065c76933a0069915202696ef72848070793     920	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w6_v8_turn.log
100644 blob ef606e910677b3be59104d338d1c1ced92a5a88c     972	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w6_v8_vu128_turn.log
100644 blob cd35c7e0f02f1bd19debc30c973593df64cad0db     962	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w6_v8_vu64_turn.log
100644 blob 33fe23c01f9bcc3cb818d3c98c9191ff8f9c2f0b     920	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w7_v8_turn.log
100644 blob 422f049224774c410bd8e5d21abb1bfcce5a92ec     972	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w7_v8_vu128_turn.log
100644 blob c07a47f857ad28114d5550e03f9ce9b1fd455d30     962	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w7_v8_vu64_turn.log
100644 blob d31281691ba52a39343fa790106c31200e8e066a     920	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w8_v8_turn.log
100644 blob e6303a13365b39dd54f34ae9ca865f26001c0f21     972	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w8_v8_vu128_turn.log
100644 blob a9fc895a1f9da2c24df8ab7da37a746ac15506f6     962	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep-after-naive_w8_v8_vu64_turn.log
100644 blob 44170b8861dd80dea2a28288ef1e25cf6e6c104b     800	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w1_v8_turn.log
100644 blob 1f345a07ef6254853c7b20ec769c3f1f9392bfaf     852	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w1_v8_vu128_turn.log
100644 blob 5f300a4c5e9e299b71d76c8d64a07797e3815759     842	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w1_v8_vu64_turn.log
100644 blob 45a032dc31c9bea3075e00ee5fe23c80df785a49     800	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w2_v8_turn.log
100644 blob 56e2275d7b68a7d2dcdffb835c7c3657e3832349     852	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w2_v8_vu128_turn.log
100644 blob 5e2f3299936187e6804186b7343749f44a453b05     842	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w2_v8_vu64_turn.log
100644 blob e5117fb2911a24e49485e1ba4da302c869b09c34     800	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w3_v8_turn.log
100644 blob a782fcac602e8478d6e2dad8e2c80904a65a8f77     852	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w3_v8_vu128_turn.log
100644 blob f939797f696536183d4eb295d9399de46f4ec857     842	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w3_v8_vu64_turn.log
100644 blob 3d1457aaf5e38d69a023bba6cdf81a7545140a9a     800	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w4_v8_turn.log
100644 blob 9716c22c656ffac749126243940004eaaf26369d     852	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w4_v8_vu128_turn.log
100644 blob e7828b3f8bb29794ec248cb328d46099ed43410c     842	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w4_v8_vu64_turn.log
100644 blob b61be803d220bdea58103c37bc0669b6a27062aa     800	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w5_v8_turn.log
100644 blob 7aa1ebf2ed31f043237f8126b41b5f2a74afe97c     852	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w5_v8_vu128_turn.log
100644 blob a51e083993d5c59a7a9dd19e5d864704731e5627     842	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w5_v8_vu64_turn.log
100644 blob 94b0d16d6271cf9bc9499226c3862555036ad2db     800	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w6_v8_turn.log
100644 blob c2bc7476cfd95c4f9bed4adc36943ada39a1fa18     852	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w6_v8_vu128_turn.log
100644 blob d73bb62a2a79daa755fc47c06dfbcef7b8306f67     842	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w6_v8_vu64_turn.log
100644 blob 9ebc28c12771933db28f0c0982e8754ccc2a24f6     800	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w7_v8_turn.log
100644 blob 7bb0399a89918455973bbae5dea56b0a71eee206     852	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w7_v8_vu128_turn.log
100644 blob 56a22b0d78190bb86d1d37a1d0a8f5cc49517241     842	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w7_v8_vu64_turn.log
100644 blob 8a4edde8dad75afa00171c910db0f8de5910ebf8     800	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w8_v8_turn.log
100644 blob 2e2f3579ea1a92db18be916de2723b1b4e1b2b37     852	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w8_v8_vu128_turn.log
100644 blob cdad898e7268d1cf8d3b841c27f0d906e207df5a     842	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/logs/lw_sweep_w8_v8_vu64_turn.log
100644 blob 4951ee0e3d97ce36564f8479f41cacc5732189b9   27914	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_turn.json
100644 blob 2e75145539c7ce2c971102e4bdb45744b60ea041   99744	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_turn.npz
100644 blob d7400d5a1d68eea61787a5f3acec697ca8ce81a7   59012	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_turn_list.npz
100644 blob f7c04ed9fd63dfe32c182337576aee0a4ca0adfb 4081271	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_turn_state_v8.pt
100644 blob d94b2b721492b5e7308e7a234dec762094360b43   27939	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_vu128_turn.json
100644 blob 8e2e6544907fa86725b39c51856f90d9be61a30f   52467	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_vu128_turn.npz
100644 blob 0257ab586e889858ec6d479184282ee500d0657d   53722	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_vu128_turn_list.npz
100644 blob e5688a3138fd4e72631af22f8e3f19f26e636205 4000979	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_vu128_turn_state_v8.pt
100644 blob 8ac825729381f71fb7847396f175443fab325aac   27964	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_vu128_wd0_turn.json
100644 blob 600e3885554bb1c3f2d76b85581641fe5477c768   52424	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_vu128_wd0_turn.npz
100644 blob 713776c966519b199ba18a9964794477a7a21a68   41049	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_vu128_wd0_turn_list.npz
100644 blob 94dc66748476fe29ea8c71f1c089cf9978ebc82e 3982075	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_vu128_wd0_turn_state_v8.pt
100644 blob f1cca5b2b03a4b24a465f72c7f97a6b6a38bb0f9   27900	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_vu64_turn.json
100644 blob a448e128f1c53536857ead5d7b5e5da8617a08e8   28005	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_vu64_turn.npz
100644 blob 1155f650f417362876eed41bd409b08224d87a47   46413	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_vu64_turn_list.npz
100644 blob 8c414c98f1e09d1c6909978699c1e5b6d20d44a6 3952857	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_vu64_turn_state_v8.pt
100644 blob 635b8146a11d536ae12d5b9042030efcd374840f   27935	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_vu64_wd0_turn.json
100644 blob 7fea251e42432ce70d993961a26bb05ec2bacb55   28060	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_vu64_wd0_turn.npz
100644 blob 04b9daa1983cdb901f1674104e2d637aa04cbe9b   41217	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_vu64_wd0_turn_list.npz
100644 blob 8bde3ecf3264de8af690e25364f29c5fb8e8dae0 3945217	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_vu64_wd0_turn_state_v8.pt
100644 blob 8b8812e1a79d9d5e75e6e66afbd336c26adb8070   27962	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_wd0_turn.json
100644 blob efd284861ffc5dd723be3781bf87196be1b50e98   99656	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_wd0_turn.npz
100644 blob e469cf663efa63a2848ac9ee466e228536c7f009   41014	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_wd0_turn_list.npz
100644 blob 5e029f7ed362eade03fccad7914013c7c686d822 4053471	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w1_v8_wd0_turn_state_v8.pt
100644 blob cc795dfcefa8331287a89ca78a510b6c0e3e4bd4   27929	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w2_v8_turn.json
100644 blob 9d3ae80d1fda8a922a13cf7dd09494f0a2c379c3   99674	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w2_v8_turn.npz
100644 blob c97b7e631d2270098e147584dfc2b0ec76c979d3   58923	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w2_v8_turn_list.npz
100644 blob 1e9541982fd12bd75ca0687df7f1a711d1f15ffe   27944	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w2_v8_vu128_turn.json
100644 blob 12364fef8446bb9fdb6990b3fbbea67131107ae1   52276	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w2_v8_vu128_turn.npz
100644 blob bdb31f2d7a0a9d60d9d781e04295240c4c7b6094   54152	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w2_v8_vu128_turn_list.npz
100644 blob 960d3e22739e33aebcc78913798f39e0abb03201   27946	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w2_v8_vu128_wd0_turn.json
100644 blob a0b0749cba39d2d904193eed5967b5d883bbde18   52271	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w2_v8_vu128_wd0_turn.npz
100644 blob 32fe17072d4e31ead2ff8969e0f9092057395e82   41584	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w2_v8_vu128_wd0_turn_list.npz
100644 blob 61ff365ce9fd9018ce673f6da837a085739eb7af   27925	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w2_v8_vu64_turn.json
100644 blob 47ebfdfde39a1327d80f86f0eac971c2285d50a3   27765	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w2_v8_vu64_turn.npz
100644 blob 2e1681012ddb4ae09835663c25fd4202fbe1c43a   46689	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w2_v8_vu64_turn_list.npz
100644 blob 79019e5807d6c06244f69fb17829febc09099638   27909	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w2_v8_vu64_wd0_turn.json
100644 blob 9448f9824781a46f60ed7fc56b30eab73d1d9a73   27777	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w2_v8_vu64_wd0_turn.npz
100644 blob 56f06363ed58ba313c8c64f8777f82cd999a928b   41315	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w2_v8_vu64_wd0_turn_list.npz
100644 blob ad9cc3a56f46330d187ab77f36e32f14a3624b45   27985	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w2_v8_wd0_turn.json
100644 blob 13ac35b22a81b05523231e4d490cc94fa165bd17   99573	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w2_v8_wd0_turn.npz
100644 blob 10e753a40c6c9ec57282ad53d58e4d21644d584e   41432	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w2_v8_wd0_turn_list.npz
100644 blob b34f4efe3b4501c93194d5cd1b8b1b4e32f67207   27954	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w3_v8_turn.json
100644 blob 86d4cb74de71915725dd892ecef04daf749b4cad   99723	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w3_v8_turn.npz
100644 blob 0a5e3e9396c6ee901cff7dfe39bc3dbdd8c57abf   58527	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w3_v8_turn_list.npz
100644 blob 63d63abfaa4127012f85718420ca6cc7376e2e6e   27906	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w3_v8_vu128_turn.json
100644 blob 2eed252d9e9c16177a5dd51bfdce8e7e463337ee   52296	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w3_v8_vu128_turn.npz
100644 blob 0a16d9312ccc2325c2b7025d9ec227e50854a143   53899	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w3_v8_vu128_turn_list.npz
100644 blob 44cead9423c8cf237b8fd6de3215febbbea0f840   27944	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w3_v8_vu128_wd0_turn.json
100644 blob bd7922399f3ec3c974dbfa9e7a4a8aa37a212a6b   52261	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w3_v8_vu128_wd0_turn.npz
100644 blob 1c1b995fc3233a580e0206dcf8968d4875081c66   41175	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w3_v8_vu128_wd0_turn_list.npz
100644 blob 2a439a2239045c69c73f81df5b1e19d41624bb37   27905	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w3_v8_vu64_turn.json
100644 blob 30f4b16078fcb26eb1c9dcaa4b952b443e268bdc   27849	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w3_v8_vu64_turn.npz
100644 blob b11572eefda2e716e08c026fcd2b52fc39711700   46516	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w3_v8_vu64_turn_list.npz
100644 blob 8550fdaa76c671e26c3d910c7a73ea52b8545823   27886	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w3_v8_vu64_wd0_turn.json
100644 blob 249f1eeb94d35e0c7852269eaaef09811f142bbe   27850	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w3_v8_vu64_wd0_turn.npz
100644 blob 0d66baa846a0b2d82921d03d3126e0708e134296   41345	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w3_v8_vu64_wd0_turn_list.npz
100644 blob a91b0063a17f66ee6769bfac57ccab7c8a5f527d   27966	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w3_v8_wd0_turn.json
100644 blob 5e6fabeaa21cad0163e94086e51ecc63d8b6c854   99687	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w3_v8_wd0_turn.npz
100644 blob e1a168d3c8a049a5dc5a8598499fb455f8e140db   41135	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w3_v8_wd0_turn_list.npz
100644 blob cffd9c7fa5752369bc4e65c8e70d45e8c50ebff0   27943	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w4_v8_turn.json
100644 blob db6bb53ee417c8659308c2de3302bf3b9249023c   98691	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w4_v8_turn.npz
100644 blob add1fbe28bc5c675f8a1235d6d2a2ffa12f15e6f   58586	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w4_v8_turn_list.npz
100644 blob 7bbcd02a5f1d1cef1a1981263a5bc372e12cebe1   27895	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w4_v8_vu128_turn.json
100644 blob 5cb2f4b16ec8a737fb096d31c3963795045e1eb5   51840	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w4_v8_vu128_turn.npz
100644 blob 09e62078aeb47f3d6341473586fe1579efbdabbd   53657	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w4_v8_vu128_turn_list.npz
100644 blob 43da8e7ccfacf805a890cb973e95d91a266eb450   27935	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w4_v8_vu128_wd0_turn.json
100644 blob 27c5c0d2f679364aa0b7cab3d33790696280310b   51842	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w4_v8_vu128_wd0_turn.npz
100644 blob ed0400e0d420a244c1a98c1de42c031b3609ff49   40987	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w4_v8_vu128_wd0_turn_list.npz
100644 blob 05d3eb6e4ea38d2b2091d99256b741e7eb18220c   27923	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w4_v8_vu64_turn.json
100644 blob 016ca7c18c11fb94a4d7d798f368d83da7579a53   27618	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w4_v8_vu64_turn.npz
100644 blob 8a816447cd2760f9e9161adeddccefbfb9de5623   46429	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w4_v8_vu64_turn_list.npz
100644 blob a8284f21d776285b4a00dd9ead586d298d350988   27964	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w4_v8_vu64_wd0_turn.json
100644 blob ed78b0bd966aaac68f90596ceb993e8430d84ea8   27615	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w4_v8_vu64_wd0_turn.npz
100644 blob 34acf16133d572e87edbac381e3da82391b5bed6   41029	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w4_v8_vu64_wd0_turn_list.npz
100644 blob d6c7183cf033d4b3cd9f33db4c08ea2c3dd08f46   27995	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w4_v8_wd0_turn.json
100644 blob 3f0621f6734abcb84b0bf28f2d70a026b56df0b3   98689	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w4_v8_wd0_turn.npz
100644 blob 69dec7fe6746c969776830033655056e21beb20e   40848	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w4_v8_wd0_turn_list.npz
100644 blob b84bd9be5297f550345056d032d7ee87c307b1ba   27929	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w5_v8_turn.json
100644 blob d9ae2b05ec0575b464a46e0e8375bb1a3e20e4e1   99719	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w5_v8_turn.npz
100644 blob f7cfee68c9bded71f005aa6323a00752575ebd7e   59111	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w5_v8_turn_list.npz
100644 blob 176cb66e7b12b86aa092cec9d72e1bd2961bf091   27875	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w5_v8_vu128_turn.json
100644 blob 41369e6108122e778502cef00efda194c25e6b54   52470	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w5_v8_vu128_turn.npz
100644 blob 85eb012cf97d77e458559c0d60d5e7bf90e4e623   54517	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w5_v8_vu128_turn_list.npz
100644 blob 0cdb492783d92113ddb0666ec09d989a0cb7f1a0   27963	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w5_v8_vu128_wd0_turn.json
100644 blob 116f2fc99b95ba80b38abc5c681e7563557b68ba   52523	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w5_v8_vu128_wd0_turn.npz
100644 blob 27d8ac46915a7c41b8794c2b4950359a5f4a15fe   41626	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w5_v8_vu128_wd0_turn_list.npz
100644 blob 56b461823c0eaab6ec56ffecb416f6785df6d439   27916	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w5_v8_vu64_turn.json
100644 blob 15b200a95240075ba3c3ca4a885b97d1e53a94e7   28068	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w5_v8_vu64_turn.npz
100644 blob 5ba06289c4ee66e34900e8e7c045bf1e13a7c677   47113	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w5_v8_vu64_turn_list.npz
100644 blob a65dee9937425be1e0c7cb6c00be0497019126e3   27930	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w5_v8_vu64_wd0_turn.json
100644 blob 71571a4c2fb22f887b7d02dc4e6df2d3e8c9a750   28044	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w5_v8_vu64_wd0_turn.npz
100644 blob 8f0bd2539bf434d9f2ccd6a7fe0d9b850fa9d654   41592	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w5_v8_vu64_wd0_turn_list.npz
100644 blob a6c8b4ad8f0dfde2e90ffb054af2b10f3158b189   27956	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w5_v8_wd0_turn.json
100644 blob 662b2e706f4917a6ff276033f6049bc51a18dd1e   99688	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w5_v8_wd0_turn.npz
100644 blob b9675ca92c0372b3b165103eace0010642157dae   41560	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w5_v8_wd0_turn_list.npz
100644 blob d15a83a28efa8ff9da340419da26e59afc37d11f   27944	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w6_v8_turn.json
100644 blob 53cc305cddb5f52a8d534d46c4b1c6c48835837d   99322	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w6_v8_turn.npz
100644 blob 4d383069feb6f4dcbef5a7624cbeee0c429985e2   58380	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w6_v8_turn_list.npz
100644 blob 3376398ec3067b91239ae6ce94377198d638b84c   27927	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w6_v8_vu128_turn.json
100644 blob 33e625253b0e4dc928cff71289ba4259c39ed6f2   51947	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w6_v8_vu128_turn.npz
100644 blob 359c65440b323ee2cfd8ff7d82d0cfaa35fb9a8f   53974	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w6_v8_vu128_turn_list.npz
100644 blob 068a8332163092304d57a3dbd163558b9097d299   27896	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w6_v8_vu128_wd0_turn.json
100644 blob 4ece51e76c530a9a461da84c94600facd1d03840   51954	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w6_v8_vu128_wd0_turn.npz
100644 blob 56350f27902f34ac9dbee290c834676d85f8cf71   41372	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w6_v8_vu128_wd0_turn_list.npz
100644 blob 066e0470c10b3e5fbba80c4e7661b999bb9388ba   27901	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w6_v8_vu64_turn.json
100644 blob a4f1cd8b7cf10616a1db271fd510e5cfb3f8219d   27715	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w6_v8_vu64_turn.npz
100644 blob 436fc2184006954756e8f5662113f9ae6918cee7   46408	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w6_v8_vu64_turn_list.npz
100644 blob 1eb3802582e8f0db208f9baa0455bf7062b57109   27888	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w6_v8_vu64_wd0_turn.json
100644 blob bdd2c46e5ca83f6897651b0029d836929285e9d2   27729	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w6_v8_vu64_wd0_turn.npz
100644 blob 763edf8b2328b15fe0333b85859f7e0f66ddb719   40921	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w6_v8_vu64_wd0_turn_list.npz
100644 blob 5fbdd1378c9415e406ebedd4ccacf2dd5ea738dc   27949	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w6_v8_wd0_turn.json
100644 blob e8f36de0779866ee1b117dc087a0d717ad12e5ef   99162	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w6_v8_wd0_turn.npz
100644 blob 75b0fce50bc4a477494d90a2d50fdec103f83268   40985	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w6_v8_wd0_turn_list.npz
100644 blob c1e3a53895979094369ea7d6641283945ac5d80a   27922	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w7_v8_turn.json
100644 blob b98a8dc709fb703fcc40152dd1d832056e6e13c5  100317	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w7_v8_turn.npz
100644 blob fe18cc43aa69af6fd86c7e7b5b18c51db6ac1bd6   59529	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w7_v8_turn_list.npz
100644 blob 3ace2c77db585bc3900fed0a31da747cbcaab87d   27900	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w7_v8_vu128_turn.json
100644 blob c2b1f2d5d941f56dfd76a778d9a99fe890151698   52636	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w7_v8_vu128_turn.npz
100644 blob aee94dd75b79bce320c9b0898b1e11150e62a6f0   54178	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w7_v8_vu128_turn_list.npz
100644 blob c237784b1cf3817ad842ac4522b9d75b6a8fde14   27939	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w7_v8_vu128_wd0_turn.json
100644 blob e7e38a0c0a2934c5358eeb28e39c4881960bbe0c   52668	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w7_v8_vu128_wd0_turn.npz
100644 blob 8d369b9dca224ccf26079368466cdefb4fe2f544   41574	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w7_v8_vu128_wd0_turn_list.npz
100644 blob 04048303e37a0fc4abd4f1b10f113923e8c73246   27903	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w7_v8_vu64_turn.json
100644 blob 11afc7d8a94b0989ead769a4bf22fe7a0c307e93   28142	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w7_v8_vu64_turn.npz
100644 blob 7f5a811f3a44e8030c5330af5159cf30b5645e98   47555	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w7_v8_vu64_turn_list.npz
100644 blob 2e55ba941453dccc89e57ed250f25931aa643527   27934	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w7_v8_vu64_wd0_turn.json
100644 blob 3ca6ad5680b6c4905b3e89bc9e89203575e632b8   28151	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w7_v8_vu64_wd0_turn.npz
100644 blob 83be1b0932dd400d07371db1be2341f8924c0b0c   42280	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w7_v8_vu64_wd0_turn_list.npz
100644 blob 1090f574de813b9f2f938f63b47927fb6534b548   27942	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w7_v8_wd0_turn.json
100644 blob 42f178bda0bb4dd70e55ae29aefa8975106c7ea8  100222	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w7_v8_wd0_turn.npz
100644 blob 41ba83c3d9d56a10bf01debb82f6a8af3bf9725f   41377	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w7_v8_wd0_turn_list.npz
100644 blob 8d313189f48d1651703fa7a986a48cd45f384354   27894	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w8_v8_turn.json
100644 blob 58c9fac5407dae87ce3bf3e6bf9425621e989cc6   98500	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w8_v8_turn.npz
100644 blob 19a3bd81fb64d926f66a1fc59edaba85a2fc8826   58255	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w8_v8_turn_list.npz
100644 blob c1c8ac5e7f97b59d8f075667c985b10ed275f22c   27921	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w8_v8_vu128_turn.json
100644 blob 552379d8500b9b0547c83b8963d500641585706f   51684	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w8_v8_vu128_turn.npz
100644 blob 5e60b218e5aa9f72364c12f9619c5b5994f00024   53355	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w8_v8_vu128_turn_list.npz
100644 blob 01193a4c4ef04fc42c1ff1da1dbe37f597ddde91   27925	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w8_v8_vu128_wd0_turn.json
100644 blob 1a837790ca00c68b6fc62eb7b0c5144295fb9974   51759	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w8_v8_vu128_wd0_turn.npz
100644 blob e8c710472986be52d9c73080e6d0ffb69a23d06c   40993	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w8_v8_vu128_wd0_turn_list.npz
100644 blob 59fe8228f4b1b0431f1b647def1fcdfd8b950923   27921	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w8_v8_vu64_turn.json
100644 blob d5cd0f72e65c6501adc41df3195589d34684ef87   27571	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w8_v8_vu64_turn.npz
100644 blob c76852a44bfe964c6705486b80bb4b4a29c3fc99   46222	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w8_v8_vu64_turn_list.npz
100644 blob 0bd10b58daa97e5267ba694c10adb6c5c07a75e3   27919	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w8_v8_vu64_wd0_turn.json
100644 blob dc52aaee7c88ac669a5940cf16610b365ab6f1e5   27596	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w8_v8_vu64_wd0_turn.npz
100644 blob 474df5fcb321a1bef1f037e1aec75c441f51d876   41059	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w8_v8_vu64_wd0_turn_list.npz
100644 blob e938effd7f97d1c4b050574f6ef48bfed00242d4   28010	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w8_v8_wd0_turn.json
100644 blob 8306094c0784ab773397eb1e4b9bea09c4d5c734   98481	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w8_v8_wd0_turn.npz
100644 blob fc4ef66c5f16b34aec71bcd24e78f1404eb7b3c9   40967	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_naive_w8_v8_wd0_turn_list.npz
100644 blob d042dd1804cdaa6c86b4e0a7041c53842eee1967   27992	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w1_v8_turn.json
100644 blob 355755bca5569d47677d5cb727fcbec352295131  100462	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w1_v8_turn.npz
100644 blob 45c8f267d08faa29d2abdc98caa27e4a448e229e   58844	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w1_v8_turn_list.npz
100644 blob 43d50efc85ebef6d66cf23b56674c6d6fdedede7 4083311	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w1_v8_turn_state_v8.pt
100644 blob f48abf7af09d63cd726f596a974e1c27d2115fe6   27978	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w1_v8_vu128_turn.json
100644 blob 5270880e4af64df31f341109d24c6ac41287e94f   52599	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w1_v8_vu128_turn.npz
100644 blob c3cbf68c95f9bd03a5ff40ae3502dc946bc4d085   53580	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w1_v8_vu128_turn_list.npz
100644 blob 82c886ce763f058fe7b56b7343982c21cebccb49 4002187	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w1_v8_vu128_turn_state_v8.pt
100644 blob fc091ead0c44bcef3f8630873c2bb93c2eeae93c   27959	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w1_v8_vu64_turn.json
100644 blob 642350abc5759a53d696124f9175e2381180b153   27988	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w1_v8_vu64_turn.npz
100644 blob a2ac68e48601f588ad7e6eccbd949ecabe9ff352   46088	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w1_v8_vu64_turn_list.npz
100644 blob b4ecc40f44b28773f58fef32a66f1f038951cc23 3953297	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w1_v8_vu64_turn_state_v8.pt
100644 blob d27dcd81248ae94933d5bf8c7aece4213d6f754c   27987	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w2_v8_turn.json
100644 blob 723351940d6796822ac41755ad6b25cebda4d818  100116	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w2_v8_turn.npz
100644 blob cbf507e26233c20cc6f0f0796c598e5de61895d4   59209	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w2_v8_turn_list.npz
100644 blob 89fa69ced50093c8cc0ad5c7dd3a5dc2f8f7b805   27954	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w2_v8_vu128_turn.json
100644 blob 56b910e578ddd59cde2890286052a6fc275739e7   52451	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w2_v8_vu128_turn.npz
100644 blob e7f3b2f33c302a7f8562cb1668117d145c9648bd   54430	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w2_v8_vu128_turn_list.npz
100644 blob 6f6251f72a61672510cba173dde552068ed83afb   27932	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w2_v8_vu64_turn.json
100644 blob a8fc003a812a5d2772d922c6945aa9380416f28d   27703	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w2_v8_vu64_turn.npz
100644 blob 39496f8d06cf9dc0994d194336186d5cfcf58cc4   46309	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w2_v8_vu64_turn_list.npz
100644 blob 74dde69a637a0c801d40e3775698f343b3877ba5   27979	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w3_v8_turn.json
100644 blob c0da341e90dad52cec43e623fecb324154516614  100225	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w3_v8_turn.npz
100644 blob bb5fa98eb14f6320227b62e5427ee709c9e64eb6   59101	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w3_v8_turn_list.npz
100644 blob 5bcc09fbfcef77e5f8cb7d54cc5ce9b41596d1ec   27951	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w3_v8_vu128_turn.json
100644 blob 3cacab3541e06a00144563728681882677c9ef47   52269	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w3_v8_vu128_turn.npz
100644 blob c18100c178c8715c71bc0b74fd1c76d6bd26dab2   52694	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w3_v8_vu128_turn_list.npz
100644 blob 994cc91e4f2ac3edc8e39ce2f733179a591fbaa9   27940	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w3_v8_vu64_turn.json
100644 blob 87ec00a62cf4b4673694a929a2312654c8cabec7   27650	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w3_v8_vu64_turn.npz
100644 blob 2122fda8b4d41f9df8b380a23a2a7b852b52813b   44758	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w3_v8_vu64_turn_list.npz
100644 blob c4bd63bda8f105d543aed60daa1b66c46f4d7b0c   27999	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w4_v8_turn.json
100644 blob 110fbfdcba4417d547b68e8f252a010073f028a0   99197	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w4_v8_turn.npz
100644 blob 03436540a973a86a2dbe31f64f92cc658062ed03   58940	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w4_v8_turn_list.npz
100644 blob 9f4ea003e059845090ed97259e18da26e7958de7   27986	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w4_v8_vu128_turn.json
100644 blob a11b743740d63d92ac6ad666df16f604f000766c   51860	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w4_v8_vu128_turn.npz
100644 blob 730953b006d225aa5d4ff1d66fa57bb563dd5e80   53365	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w4_v8_vu128_turn_list.npz
100644 blob 6beb7bf12b04feb04dcb2541500bf728108c4720   27957	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w4_v8_vu64_turn.json
100644 blob a48c95b1b35b776f376306788c914268587d40db   27443	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w4_v8_vu64_turn.npz
100644 blob f5dba5ab18889b7518bf26e90cfe87e5f07c8f90   44898	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w4_v8_vu64_turn_list.npz
100644 blob f1b1c5c998cb9a5f132c3e76ecd4d3c57942daff   27947	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w5_v8_turn.json
100644 blob 8735c4e421d5bc8b60be99d6047aacfe67ab42c4  100147	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w5_v8_turn.npz
100644 blob 71fc99957e74967d3b5b7610806af9625cdf1fa8   59837	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w5_v8_turn_list.npz
100644 blob 8b188abdc3b46a13d74fa2ac31a1a94786c77563   27955	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w5_v8_vu128_turn.json
100644 blob 870cce35cd3e41f93ee3e4727a4a8b0525bab374   52489	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w5_v8_vu128_turn.npz
100644 blob f85e81635b2f384d499bb9c9c183ffbccbd09f36   54718	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w5_v8_vu128_turn_list.npz
100644 blob 9e9d11a2356211c9186d63b922df837147be645f   27950	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w5_v8_vu64_turn.json
100644 blob f02b0a6d4e58991c08faace7662792a2443214f7   27880	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w5_v8_vu64_turn.npz
100644 blob 1cfef729ea0af0ae67b45d7a535c8657036194e9   46176	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w5_v8_vu64_turn_list.npz
100644 blob 748b087a6da77e70849361af33e34e74ab3dd7f6   27949	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w6_v8_turn.json
100644 blob 61defa1915b6b4148800d708fa4baf4d0df13102   99504	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w6_v8_turn.npz
100644 blob e99458534d6116a53c92e07ded93fa007d695a6b   58007	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w6_v8_turn_list.npz
100644 blob e4915864531be405660d317759d317239c897c44   27961	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w6_v8_vu128_turn.json
100644 blob d50e9d8d423dbf5eff8d924e5d22fb74c6481178   52004	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w6_v8_vu128_turn.npz
100644 blob 1aa4e50ba9ba453f330addfd51da3112f39cbfd7   53596	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w6_v8_vu128_turn_list.npz
100644 blob 199db3d48f0a2aeba69f869f968dbc95fac055a6   27907	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w6_v8_vu64_turn.json
100644 blob 7c70ed27d36fdff3eb806aafe2b6410afd4b9cb4   27620	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w6_v8_vu64_turn.npz
100644 blob d6065dbfe3cee8d19eb2a59a2499ed95d3e36d1d   45326	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w6_v8_vu64_turn_list.npz
100644 blob bf2c1d282909265e3b9355a979232ac10436e238   27982	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w7_v8_turn.json
100644 blob 9508c7a89704f5824e4b39b2b268e20f756adbe1  101011	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w7_v8_turn.npz
100644 blob 187cceba8f1959c112da76b6ac8cd1e729b1714a   59980	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w7_v8_turn_list.npz
100644 blob 96826f808ad6166409d0f5aac4c902b61d1a425d   27954	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w7_v8_vu128_turn.json
100644 blob f46cedcfee272353542e49582caffd9d070f0f3a   52925	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w7_v8_vu128_turn.npz
100644 blob 99b8838a96169600a5bc5c449babb8d13635133c   54002	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w7_v8_vu128_turn_list.npz
100644 blob 973b9a9d22ea97d5eee470948523eb59e1c2d394   27970	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w7_v8_vu64_turn.json
100644 blob 3d65e57e54259d46e6ce3bea9fade63d2bc85124   28187	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w7_v8_vu64_turn.npz
100644 blob 79b30d2bb21da4b18a44683909bf3c904901261d   45898	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w7_v8_vu64_turn_list.npz
100644 blob e638680560878f71d0216095a10c010dd00288c5   27967	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w8_v8_turn.json
100644 blob cf1387a6636dff1b945af446e5e48f94dec5839c   98934	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w8_v8_turn.npz
100644 blob a76679ebc56765c3c26e5ec856578af3c13bbe08   58143	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w8_v8_turn_list.npz
100644 blob 33be06f0d276832cb7c26174cca9c012d00e6b97   27937	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w8_v8_vu128_turn.json
100644 blob 4b7912795bedb8c7157c352d7d9828ec1a22fbba   51637	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w8_v8_vu128_turn.npz
100644 blob d34dee3e27f8d74bc85009dfd3bb554371ecefea   51909	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w8_v8_vu128_turn_list.npz
100644 blob 18eaa73ab26938c5339d3ca479463a70f9c39cac   27940	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w8_v8_vu64_turn.json
100644 blob 2f2894bf5ea04b7425d4add71b9c1712b45e8370   27410	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w8_v8_vu64_turn.npz
100644 blob cd1db45c84dc175d059bea60e8d2a8400345a887   43596	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep-after-naive_w8_v8_vu64_turn_list.npz
100644 blob e380e4314323f84679fd31e8578658f268c45474   27903	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w1_v8_turn.json
100644 blob 6b57523f3e1b09de887046d8a66a2827765bbefc   99615	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w1_v8_turn.npz
100644 blob d74705ce9783bb716d8d3bfb88fe2e309f5ed217   60483	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w1_v8_turn_list.npz
100644 blob d7de15b1cc1797c3cbc81591de8cee5c364f5831 4083639	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w1_v8_turn_state_v8.pt
100644 blob 0abda13683adb0641656ead0ddc6cfd6c1c7ecf2   27945	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w1_v8_vu128_turn.json
100644 blob b177806779892240a4a44e517edd8f5f7763e587   52422	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w1_v8_vu128_turn.npz
100644 blob 471d68e4230ada881cb009183c0d78361e3dc3b9   55917	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w1_v8_vu128_turn_list.npz
100644 blob cfe527ed46fbbefbd699efd67974eb6c0e06d405 4004563	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w1_v8_vu128_turn_state_v8.pt
100644 blob 065aa2fd32193330edca26515eb1013001462d66   27906	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w1_v8_vu64_turn.json
100644 blob ed4e1f13abf19c269d57ac2606a00676a7203212   28068	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w1_v8_vu64_turn.npz
100644 blob d764a91c83b207e12978da61298d2c2e2d02cd82   48023	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w1_v8_vu64_turn_list.npz
100644 blob ae5adccbcecc7ac2984a03033260a8641d4b310f 3955481	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w1_v8_vu64_turn_state_v8.pt
100644 blob 731188003e20094546a653b87cfc1d2080b5c985   27919	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w2_v8_turn.json
100644 blob f9790d0ef772f0e321396fe7043095b9f7228441   99251	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w2_v8_turn.npz
100644 blob 4af09490bf9fabbb39d2723c135829e19a467ee0   60070	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w2_v8_turn_list.npz
100644 blob ba6bd950b945abc089e8accaf1e5758c0f3404f8   27925	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w2_v8_vu128_turn.json
100644 blob c85ebdd51adf7351a8165d63d82e11aa2ee206ac   52232	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w2_v8_vu128_turn.npz
100644 blob 82d9e58c69a06a63fcce41cd6f6633c36273871e   55988	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w2_v8_vu128_turn_list.npz
100644 blob 0bd4984293533fd24223c3b3a997d8f37c9c0ae7   27927	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w2_v8_vu64_turn.json
100644 blob 85005e0aa675643074dae4c7ceae4790d472e020   27873	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w2_v8_vu64_turn.npz
100644 blob 3bf4a7e2ffb0ef31f7e151587a82b31ee6227e63   47921	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w2_v8_vu64_turn_list.npz
100644 blob b297cdf43938a2824da18c70df7044f1412beda3   27960	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w3_v8_turn.json
100644 blob 38f26adaab25fc02efa7829d1da718ff1b51a992   99472	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w3_v8_turn.npz
100644 blob 25b68ce318a2f96cccf278de304d070e8e04cfd9   60111	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w3_v8_turn_list.npz
100644 blob 05667f06727d980e3c992162b038e0286c704976   27923	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w3_v8_vu128_turn.json
100644 blob 0e6277e254a51bd2a980a98d1a726070e80c1fa3   52180	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w3_v8_vu128_turn.npz
100644 blob cbaf96c1ba0cd1405d026050c8be49e4f1feb134   56119	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w3_v8_vu128_turn_list.npz
100644 blob 884f56fe2e548911a4e03fd6fd8257c039fba33f   27923	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w3_v8_vu64_turn.json
100644 blob e59d0089339d8935b4635fd355efd3351da8520b   27855	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w3_v8_vu64_turn.npz
100644 blob c64ee78a21d65d627d0e2befc54d9feaeb9d357b   48012	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w3_v8_vu64_turn_list.npz
100644 blob c79081bb986623b504c4a97e36934f7088ea718d   27965	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w4_v8_turn.json
100644 blob 0af19e80aa49965dbd1d92ca4f3941070bf3ee8a   98423	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w4_v8_turn.npz
100644 blob 8626664f35a99f617bfcb68e8f1b9bd86ebcb09e   58962	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w4_v8_turn_list.npz
100644 blob cf384baeda4502f8d6cbe7c2937b761a40fdddfb   27965	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w4_v8_vu128_turn.json
100644 blob 611ec275cfec72094c3cf7d6116cbcfff3978a5e   51799	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w4_v8_vu128_turn.npz
100644 blob b56d250177a30a13f815e824268b5d2f98e12e62   55393	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w4_v8_vu128_turn_list.npz
100644 blob 344a279d483f600eb01e1673561b826a406db597   27890	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w4_v8_vu64_turn.json
100644 blob 8f2a1338e0bd541e2747411b31b44024cdd0f623   27638	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w4_v8_vu64_turn.npz
100644 blob e12b969bd5eff95d4b1b71f890b86a13290dcd36   47695	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w4_v8_vu64_turn_list.npz
100644 blob 8d117cce91910f599b32efd51247120b4393f2f3   27944	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w5_v8_turn.json
100644 blob 624134c43a251c0c97914713789a25282f0ca8bc   99484	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w5_v8_turn.npz
100644 blob 236583748cfadc8bb7e3c67f31c018dd66bfc7b4   60538	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w5_v8_turn_list.npz
100644 blob 489a065be6284180d65fb65be7b22bb1ad50d07f   27949	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w5_v8_vu128_turn.json
100644 blob 79255869544ef3714f41a20dd1febf4a7a8ead61   52437	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w5_v8_vu128_turn.npz
100644 blob cfc7f0a68bf36e3392d97ca3740fbe4c9f31b87c   56155	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w5_v8_vu128_turn_list.npz
100644 blob 2f23da0668c4c58b17beb5dde3225acc8212dadb   27927	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w5_v8_vu64_turn.json
100644 blob cf2e8c2fa29e110e2a9e85393eaf89d20a60aaf0   28003	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w5_v8_vu64_turn.npz
100644 blob 6057ba60bbf9d8c4ed74af236ed36f702121e430   48636	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w5_v8_vu64_turn_list.npz
100644 blob 6ac7b7d8a2ec8f320db8f6eef9f552525802722c   27952	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w6_v8_turn.json
100644 blob c345e2e8d6f684124cd717e8a6b7f3ccd48d450e   98969	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w6_v8_turn.npz
100644 blob c87bc975aab4bb80668a52b00ec6379359e0aadc   60298	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w6_v8_turn_list.npz
100644 blob 880da77fe458615daa9b23aee6a7a7740c56113f   27917	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w6_v8_vu128_turn.json
100644 blob f129e304f433d3bd863dad759981bb3f669f7cf2   51913	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w6_v8_vu128_turn.npz
100644 blob 8a89a9553dbd1ff4541c33e9a3ff9bf84294ed15   55355	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w6_v8_vu128_turn_list.npz
100644 blob f997222e36090263c39740730bd4ce77efa8fafc   27929	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w6_v8_vu64_turn.json
100644 blob 743764432e3b0808d84edbca1390870e0c1bdf51   27766	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w6_v8_vu64_turn.npz
100644 blob 1939da16ff49587929f38ddc7f9182b56f72a8e9   47944	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w6_v8_vu64_turn_list.npz
100644 blob 8f41dd68b9aac8b304b679f0b96ac0ea00086625   27947	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w7_v8_turn.json
100644 blob f19d9000ea693ba463e6e6d4f88b73264d510f54  100182	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w7_v8_turn.npz
100644 blob ae079066f80b45922b5badadab8265cddd33c7b9   61412	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w7_v8_turn_list.npz
100644 blob b7376bf007d5f29c9c8c582f66e177ac3f12f9d6   27927	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w7_v8_vu128_turn.json
100644 blob d83ffd64d35d9f13231182049f9c8e8f9f3b3e0c   52652	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w7_v8_vu128_turn.npz
100644 blob 96810edc9fe2f4da472afa358ca7ba5dfa201bc4   56730	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w7_v8_vu128_turn_list.npz
100644 blob ad8896c66ddf7618e24f5fcf146ad686d935f10a   27914	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w7_v8_vu64_turn.json
100644 blob c188d5a7d59d27ee909a9f86e3eb2641a3d50f84   28144	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w7_v8_vu64_turn.npz
100644 blob 61c437062704a8d80505ed610d3f12a810a5cefc   49098	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w7_v8_vu64_turn_list.npz
100644 blob 3b7642fcbc7b0ed81683c02d7449be1d1e03ed5e   27942	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w8_v8_turn.json
100644 blob 011c32abf2f90dab151068e5a49990989380e9ef   98518	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w8_v8_turn.npz
100644 blob 4f085e72d60ce5ae3de956d6cb0683c40cf73465   59663	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w8_v8_turn_list.npz
100644 blob 11cf207928b4b9309f06b6f1a68d0931adb2ed08   27940	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w8_v8_vu128_turn.json
100644 blob a7638a48eb399f0ddeed8dde2f0381c9878d76d5   51784	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w8_v8_vu128_turn.npz
100644 blob e741dc33f700f2cc604ac42804d42d6f166759b5   56162	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w8_v8_vu128_turn_list.npz
100644 blob 405b5ae4b4db1d93897686bc07a935d9fd4b298f   27885	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w8_v8_vu64_turn.json
100644 blob 924661c8525383c44a82610609094a74feb7372b   27641	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w8_v8_vu64_turn.npz
100644 blob 3171868c75230ba22dd8124b8f043c486929fef4   48274	MOD97_LOCAL_RULE_20261106/exp45/e44/F1/lw_sweep_w8_v8_vu64_turn_list.npz
100644 blob d945def51fa14fb54339064e466c52834747f9cf   66677	MOD97_LOCAL_RULE_20261106/exp45/e44/checks/E44_CHECKS.json
100644 blob eb4933eb30ef9557994dc6b54de8c46238ebdfab   38147	MOD97_LOCAL_RULE_20261106/exp45/e44/checks/E44_CHECKS_PRE.json
100644 blob 7cba038c794b89f028e1b2e38ac232aa4f35af06     392	MOD97_LOCAL_RULE_20261106/exp45/e44/checks/lives/c5/log_0.txt
100644 blob 8ff5bea37b33e379c5dbef57a1de432ad11a19c8    9335	MOD97_LOCAL_RULE_20261106/exp45/e44/checks/lives/c5/sw_naive_w1_v4.json
100644 blob dc6667efd4a4f9f4adb4c8776ddeaae52c80ff24   36888	MOD97_LOCAL_RULE_20261106/exp45/e44/checks/lives/c5/sw_naive_w1_v4.npz
100644 blob d83aa715a8651e973e369d5f4bfe9ff0c9580021     842	MOD97_LOCAL_RULE_20261106/exp45/e44/checks/lives/c6_full/log_1.txt
100644 blob 34a0d581f6ec8bae8c85e8a78755bb21443c4c5b   28444	MOD97_LOCAL_RULE_20261106/exp45/e44/checks/lives/c6_full/lw_naive_w1_v8_vu64_turn.json
100644 blob a448e128f1c53536857ead5d7b5e5da8617a08e8   28005	MOD97_LOCAL_RULE_20261106/exp45/e44/checks/lives/c6_full/lw_naive_w1_v8_vu64_turn.npz
100644 blob 1155f650f417362876eed41bd409b08224d87a47   46413	MOD97_LOCAL_RULE_20261106/exp45/e44/checks/lives/c6_full/lw_naive_w1_v8_vu64_turn_list.npz
100644 blob f19ff50c3035eef3d905b0df944104bdf19b6286     477	MOD97_LOCAL_RULE_20261106/exp45/e44/checks/lives/c6_split/log_2.txt
100644 blob 6e29eec090646005a16378ddc4be10e4532faa30     458	MOD97_LOCAL_RULE_20261106/exp45/e44/checks/lives/c6_split/log_resume.txt
100644 blob f76f313e5d7b83218020cf46117189dfa1a9d9fc   28670	MOD97_LOCAL_RULE_20261106/exp45/e44/checks/lives/c6_split/lw_naive_w1_v8_vu64_turn.json
100644 blob a448e128f1c53536857ead5d7b5e5da8617a08e8   28005	MOD97_LOCAL_RULE_20261106/exp45/e44/checks/lives/c6_split/lw_naive_w1_v8_vu64_turn.npz
100644 blob 1155f650f417362876eed41bd409b08224d87a47   46413	MOD97_LOCAL_RULE_20261106/exp45/e44/checks/lives/c6_split/lw_naive_w1_v8_vu64_turn_list.npz
100644 blob b7ec7a79a0e7785d8d9c1373d33bebdb801e6bbf     382	MOD97_LOCAL_RULE_20261106/exp45/e44/checks/lives/driver.log
100644 blob fc438ec182dec7d738e0c7f5a6360b5724e9d369   32293	MOD97_LOCAL_RULE_20261106/smallworld/lw_checks.py
100644 blob 58633041a2b4472d5087ace349734897684446d2   24063	MOD97_LOCAL_RULE_20261106/smallworld/lw_f1.py
100644 blob 37d7b702869f6e8c0cc06c3a2dd9c218d3210fff    7244	MOD97_LOCAL_RULE_20261106/smallworld/lw_world.py
100644 blob ec56af2e59fe551e047c7eaa82459e06ad748bd2   60930	MOD97_LOCAL_RULE_20261106/smallworld/sw_run.py
100644 blob 36f190ef9bafb33df7050d18b35f76c65112564b   20910	MOD97_LOCAL_RULE_20261106/smallworld/sw_score.py
```
