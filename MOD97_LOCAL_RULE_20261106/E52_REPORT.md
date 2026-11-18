# E52 report: the two learning rules under other learning rates (brief revision 2, 17 November 2026)

**Status.** This is a trial on one seed (93683), at evidence level E2. All 28 runs of the brief and the 7 further runs that the brief's rule calls for are complete, and none diverged. Every figure below comes from a receipt in `exp45/e52/` or from `python3 exp52_tables.py`, which writes `exp45/e52/E52_TABLES.md` and `exp45/e52/E52_SUMMARY.json` from those receipts. The checks before the runs are in `exp45/e52_qual/`.

The report is the worker's and has not been reviewed. Its readings use only the rules the brief fixed in advance. It claims nothing about more than one seed, it does not stand in for the programme's independent audit, and it claims no checker gate.

- **Worker branch:** `claude/e52-learning-rates-sdga8c`. No branch was named in the task, so the worker created this one in its own namespace.
- **Source:** the default branch `codex/minifly-consolidated-20261023` at `7cabb13a`.
- **Host:** Linux x86-64 with 4 cores, Python 3.13.18 and torch 2.15.0+cpu in a virtual environment. Each process ran one thread (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `torch.set_num_threads(1)`), with at most three processes at once. These values are recorded in every receipt under `provenance.host`.

## 1. The question and the terms

The question is this. Is the sweep's lag behind back-propagation a property of the rule, or of the one learning rate (0.001) at which both rules had been run? Every comparison in the record so far used AdamW at learning rate 0.001.

The terms below are those of the brief.

- **The network** is `arch20`'s recurrent network at a width (H, U).
  - The *present width* is H = 512, U = 256 (1,124,113 parameters).
  - The *half width* is H = 256, U = 128 (317,457 parameters).
- **The two learning rules.**
  - *Back-propagation* (rule B) uses the exact gradient.
  - *The sweep* (rule S) passes the error backward through each linear map by the signs of the map's weights, times one scale per map: the mean size of the map's weights.
- **The task.** The first family is add, sub, max and min under modulus 89.
  - *The taught fraction* f is the share of operand pairs that the training stream teaches; the other pairs are never taught.
  - *Score A* is the share right on never-taught pairs of add and sub at 12 letters.
  - *Score R* is the share right on never-taught pairs of add and sub whose answer needs the modulus, with no letters.
- **Updates to 80** on a score is the first scored row from which three consecutive scored rows are at or above 80%. Rows come every 256 updates.
  - A run that never reaches 80 has no count. Its last recordable count is its dose less 512.
- **A case** is one rule at one width and one taught fraction; there are eight.
- **The lag** is the sweep's updates to 80 divided by back-propagation's, at one width and fraction.
- **The optimizer** is AdamW with betas (0.9, 0.98) and weight decay 0.25 on every parameter except the byte table, which has no decay. The run's learning rate is the only setting that changes.
- **The product control** holds rate × decay at 0.00025: its decay is `0.25 * 0.001 / rate`.
- **The length test** is add under modulus 89 at 12 and 40 letters, 300 items each, drawn from pairs that the run's own stream never teaches.
- **The seven norms** are the Euclidean norms of seven weight groups (`exp46_decay.norms`): book, l1, input map, recurrent, lm, l2, l3.

## 2. What was run

| Factor | Values |
|---|---|
| Learning rate | 0.0003, 0.001 and 0.003 (main runs). Also 0.01 (further runs, where the rule called for them) |
| Rule | S, B |
| Width | half, present |
| Taught fraction | 0.50, 0.25 |
| Seed | 93683 |
| Decay | 0.25 in the main runs and the further runs. 0.25 × 0.001 / rate in the four product controls (half width, f = 0.50, both rules, rates 0.0003 and 0.003) |
| Dose | E41's at 0.001, 0.003 and 0.01: 40,960 updates at half width, 12,288 at present width. E41's × 10/3, rounded up to a row, at 0.0003: 136,704 and 40,960 |

That gives 24 main runs, 4 product controls and 7 further runs at 0.01, which is 35 runs in all.

The runner is `exp52_rate.py`: E41's birth (`exp41_scale.run`) with the learning rate and the decay as arguments.
- Every parameter group's `lr` is set to the run's rate after `exp26_candidate.make_opt`.
- The decay is left as `make_opt` sets it, except in the product controls, where group 0's is set to `0.25 * 0.001 / rate`.
- The receipt writes the run's own rate and decay. It does not use `exp26_candidate.opt_record`, which would write 0.001 and 0.25 whatever the groups hold.
- The optimizer groups are asserted equal to the expected ones at the start and at the end of every run.
- At every scored row the runner also records E46's seven norms and the length test.
- The batch digest at row 4,096 is asserted equal to E41's receipt of the same rule, width, seed and fraction.
- At 12 letters, the length test uses exactly the same 300 items as A's add count, at both fractions. The two counts were asserted equal at every row of every run, and every assertion held.

The runs at 0.0003 and at 0.01 used `exp52_resume.py`, which adds saving and resuming to the same run (Section 5).

`exp46_decay.py` is a byte-for-byte copy of E46's runner (`git show origin/claude/wonderful-edison-8hx6i6:MOD97_LOCAL_RULE_20261106/exp46_decay.py`).
- It came from commit `b76f5979`, blob `9fb214f5`, sha256 `d7e98b3001629589…`.
- It is imported unchanged, for `norms` and `GROUPS`.

No existing source file was edited.

## 3. Checks before the runs (`exp45/e52_qual/`)

| Check | Result |
|---|---|
| 1a | **PASS** (`check1a.json`). Rate 0.001, decay 0.25, 2,048 updates, present width, f = 0.50, seed 73129. The final `tensors_sha256` is `79618ab2…` under S and `adf933d7…` under B, the values of E41's check 1 (`exp41_scale.DRY2048`). The batch digest `4f622d94…` equals E41's |
| 1b | **PASS** (`check1b.json`). Each of the eight runs at 0.001 on seed 93683 ends on the `tensors_sha256` of E41's receipt for the same case. S: `cab8fe0b`, `31eaf8e4`, `0d91fdb8`, `0b0da897` (h256 f25, h256 f50, h512 f25, h512 f50). B: `0a194f97`, `ea2dcea0`, `e0912291`, `08ff7716`. Their A, R and taught counts and their batch digests equal E41's at every row (161 rows at half width, 49 at present width) |
| 2 | **PASS** (`check2.json`). From E41's start (seed 73129, present width, f = 0.50, first batch), under back-propagation, one update was taken at each new rate. (a) The decay recovered from the difference between (rate, d) and (rate, 0) lies between 0.24994 and 0.25000 for d = 0.25. For the product decay it is 0.83327–0.83328 (d = 0.8333) and 0.083326–0.083328 (d = 0.08333). The byte table is bit-equal. (b) The change at (rate, 0) is 0.300000, 3.00000 and 10.0000 times the change at (0.001, 0) for the rates 0.0003, 0.003 and 0.01; the largest residual is 0.1% of the largest change. (c) and (d) are asserted inside every run: the groups at the start and the end, and the batch digest at row 4,096 |
| 3 | **PASS** (`check3.json`). Dry runs of 256 updates: 0.038 (B) and 0.042 (S) CPU-seconds per update at half width; 0.108 and 0.118 at present width; 2.2–2.5 and 7.2–7.3 CPU-seconds per scored row. The forecast for the 28 runs was 27.1 CPU-hours, against the brief's 26 and a limit of 32.5. The 28 runs used 27.9 CPU-hours (Section 5) |
| 4 (added) | **PASS** (`check4.json`), on the second attempt (Section 5). Under each rule, half width, f = 0.25, rate 0.003, 3,072 updates, with a checkpoint every 1,024, three runs were compared: (i) `exp52_rate.run` uninterrupted; (ii) `exp52_resume.run` uninterrupted; (iii) `exp52_resume.run` killed with `os._exit` after update 2,000 and resumed from its checkpoint at 1,024. All three end on the same `tensors_sha256` (S `172933b4…`, B `b78c97f4…`). All 13 rows are equal in counts, length counts, norms, losses and batch digests, and the optimizer steps equal 3,072 in each |

## 4. Results

### 4.1 Every run

Table 1 is produced by `exp52_tables.py` from the receipts.
- "–" means the score did not reach 80 within the dose.
- Final levels are means of the last three rows.
- The length test at 40 letters is given at the last row, as the mean of the last three rows, and as its range over the last twelve rows. One row of the length test is a weak figure: in E46 it moved by 15 to 30 points over the last twelve rows of a run, and here its range over the last twelve rows reaches 30 points (sweep, half width, f = 0.50, rate 0.01).
- "RMS of R" is the root mean square change of R from row to row over the last 16 rows.

| Width, f | Rule | Rate | Decay | Dose | Updates to 80, A | Updates to 80, R | Final A | Final R | Final taught | Length 40: last, mean of 3 (range of 12) | RMS of R |
|---|---|---|---|---|---|---|---|---|---|---|---|
| half, 0.50 | S | 0.0003 | 0.25 | 136,704 | 20,992 | 32,000 | 94.78 | 92.37 | 99.06 | 78.7, 82.9 (73.3–87.3) | 1.91 |
| half, 0.50 | S | 0.001 | 0.25 | 40,960 | 10,240 | 13,312 | 91.89 | 91.22 | 93.72 | 85.3, 83.2 (75.7–90.7) | 4.24 |
| half, 0.50 | S | 0.003 | 0.25 | 40,960 | 6,912 | 7,424 | 91.72 | 90.65 | 93.72 | 88.7, 91.4 (80.7–95.7) | 3.35 |
| half, 0.50 | S | 0.01 | 0.25 | 40,960 | – | – | 55.83 | 53.37 | 54.22 | 46.7, 53.7 (24.7–62.0) | 14.48 |
| half, 0.50 | S | 0.0003 (control) | 0.8333 | 136,704 | 76,800 | – | 82.17 | 64.60 | 85.33 | 44.0, 45.3 (36.7–56.0) | 6.83 |
| half, 0.50 | S | 0.003 (control) | 0.08333 | 40,960 | 5,376 | 5,632 | 97.56 | 97.78 | 98.56 | 98.0, 96.7 (93.0–98.3) | 1.68 |
| half, 0.50 | B | 0.0003 | 0.25 | 136,704 | 36,096 | 37,376 | 99.00 | 98.83 | 99.67 | 95.0, 94.8 (80.7–97.7) | 1.51 |
| half, 0.50 | B | 0.001 | 0.25 | 40,960 | 7,424 | 8,704 | 96.56 | 93.97 | 97.11 | 95.7, 95.6 (93.0–98.3) | 2.35 |
| half, 0.50 | B | 0.003 | 0.25 | 40,960 | 4,864 | 4,864 | 92.50 | 91.25 | 92.89 | 90.3, 89.9 (87.7–98.3) | 3.97 |
| half, 0.50 | B | 0.01 | 0.25 | 40,960 | – | – | 72.39 | 70.68 | 76.33 | 61.0, 75.3 (61.0–85.0) | 7.20 |
| half, 0.50 | B | 0.0003 (control) | 0.8333 | 136,704 | 23,808 | 35,072 | 93.39 | 90.17 | 94.67 | 89.0, 90.1 (81.0–91.3) | 5.38 |
| half, 0.50 | B | 0.003 (control) | 0.08333 | 40,960 | 5,376 | 6,912 | 97.61 | 97.13 | 97.11 | 98.7, 97.7 (95.7–99.3) | 1.45 |
| half, 0.25 | S | 0.0003 | 0.25 | 136,704 | 28,416 | 54,272 | 88.72 | 84.17 | 99.22 | 74.7, 73.1 (63.7–74.7) | 1.77 |
| half, 0.25 | S | 0.001 | 0.25 | 40,960 | 14,080 | 24,576 | 88.56 | 84.33 | 97.39 | 78.7, 82.3 (74.3–87.7) | 2.45 |
| half, 0.25 | S | 0.003 | 0.25 | 40,960 | 7,424 | 11,520 | 94.72 | 94.58 | 96.67 | 94.7, 91.9 (88.3–98.3) | 3.14 |
| half, 0.25 | S | 0.01 | 0.25 | 40,960 | – | – | 55.44 | 54.55 | 57.33 | 42.0, 51.6 (37.7–64.3) | 12.03 |
| half, 0.25 | B | 0.0003 | 0.25 | 136,704 | 17,152 | 22,784 | 96.89 | 94.53 | 99.22 | 91.0, 92.6 (82.7–94.7) | 3.39 |
| half, 0.25 | B | 0.001 | 0.25 | 40,960 | 6,144 | 7,424 | 97.72 | 96.23 | 98.78 | 98.0, 97.0 (87.0–98.0) | 1.96 |
| half, 0.25 | B | 0.003 | 0.25 | 40,960 | 3,840 | 5,632 | 96.61 | 96.30 | 97.56 | 90.7, 96.3 (90.7–99.7) | 2.15 |
| half, 0.25 | B | 0.01 | 0.25 | 40,960 | 12,032 | 12,032 | 94.61 | 93.72 | 95.39 | 92.7, 93.8 (82.7–96.7) | 3.26 |
| present, 0.50 | S | 0.0003 | 0.25 | 40,960 | 12,544 | 14,336 | 93.28 | 92.33 | 98.06 | 93.7, 85.0 (73.7–93.7) | 2.65 |
| present, 0.50 | S | 0.001 | 0.25 | 12,288 | 4,864 | 4,864 | 94.56 | 94.18 | 98.44 | 95.7, 91.1 (82.3–95.7) | 3.17 |
| present, 0.50 | S | 0.003 | 0.25 | 12,288 | 4,864 | 4,864 | 94.83 | 94.72 | 96.72 | 91.0, 92.4 (91.0–98.7) | 3.44 |
| present, 0.50 | B | 0.0003 | 0.25 | 40,960 | 12,544 | 12,800 | 98.83 | 96.82 | 99.39 | 96.7, 97.1 (87.0–99.0) | 1.73 |
| present, 0.50 | B | 0.001 | 0.25 | 12,288 | 4,096 | 4,096 | 93.11 | 94.27 | 95.89 | 91.3, 92.0 (91.3–97.7) | 1.77 |
| present, 0.50 | B | 0.003 | 0.25 | 12,288 | 2,816 | 2,816 | 95.89 | 93.77 | 95.22 | 91.3, 95.0 (88.0–97.3) | 3.51 |
| present, 0.50 | B | 0.01 | 0.25 | 12,288 | – | – | 71.39 | 72.12 | 71.22 | 68.3, 71.0 (61.3–81.0) | 7.96 |
| present, 0.25 | S | 0.0003 | 0.25 | 40,960 | – | – | 78.39 | 72.53 | 99.33 | 72.0, 68.7 (64.0–72.0) | 1.80 |
| present, 0.25 | S | 0.001 | 0.25 | 12,288 | 9,984 | – | 83.06 | 78.78 | 96.33 | 70.3, 73.2 (68.0–79.7) | 3.14 |
| present, 0.25 | S | 0.003 | 0.25 | 12,288 | 4,096 | 5,376 | 96.56 | 95.05 | 97.89 | 92.3, 95.4 (89.7–97.7) | 3.66 |
| present, 0.25 | S | 0.01 | 0.25 | 12,288 | – | – | 22.94 | 21.13 | 24.61 | 14.7, 19.9 (14.7–43.0) | 5.60 |
| present, 0.25 | B | 0.0003 | 0.25 | 40,960 | 11,008 | 12,288 | 92.00 | 89.80 | 99.44 | 86.3, 89.0 (84.3–92.3) | 2.59 |
| present, 0.25 | B | 0.001 | 0.25 | 12,288 | 3,584 | 4,352 | 92.44 | 90.62 | 97.33 | 88.3, 90.9 (83.7–95.0) | 4.64 |
| present, 0.25 | B | 0.003 | 0.25 | 12,288 | 3,072 | 3,072 | 95.17 | 95.08 | 96.06 | 94.0, 94.9 (83.0–95.3) | 5.25 |
| present, 0.25 | B | 0.01 | 0.25 | 12,288 | 11,776 | 11,776 | 85.61 | 85.17 | 86.28 | 87.3, 82.1 (68.7–87.3) | 4.15 |

The receipt of each run is `exp45/e52/e52_<rule>_h<H>_f<25|50>_lr<rate>[_product]_93683.json`, with its final weights in the `.pt` file of the same name. The receipt names are also listed in `E52_TABLES.md`, Table 1.

Final levels at 0.0003 are shown at the end of the run and also at E41's dose: `E52_TABLES.md`, Table 2. At E41's dose every run at 0.0003 is below its own final level, and below the run at 0.001 at the same dose, on both A and R. For example, the sweep at present width with f = 0.25 has R = 54.80 at 12,288 updates and 72.53 at 40,960.

The seven norms at 80 on A, at 80 on R, at the end, and (at 0.0003) at E41's dose are in `E52_TABLES.md`, Table 3. Within each case, at the decay of 0.25, the norms of the input map and of the recurrent map at 80 on R grow with the rate. For example, under the sweep at half width with f = 0.50, the recurrent map's norm at 80 on R is 19.5, 21.5 and 24.6 at 0.0003, 0.001 and 0.003. Under the sweep at present width with f = 0.25, the recurrent map's norm at the end is 23.0, 29.0, 40.4 and 78.3 at the four rates. This is so although the run at 0.0003 is the longest.

### 4.2 Readings for each case

The readings below are computed by `exp52_tables.py` using the brief's rules.
- *Best rate on a score:* the rate with the fewest updates to 80. Counts within 256 of each other are tied, and a tie that includes 0.001 is given to 0.001.
- *Faster / the same / slower:* E41's `describe_step`, applied with the run at 0.001 in the place of the narrower run. The band is two thirds to 1.5.
- *Gains from its own rate:* faster at some rate on R and not slower on A at that rate.

| Width, f | Rule | Best rate on R | Best rate on A | 0.0003 against 0.001: R, A | 0.003 against 0.001: R, A | 0.01 against 0.001: R, A | Gains from its own rate |
|---|---|---|---|---|---|---|---|
| half, 0.50 | S | 0.003 | 0.003 | slower, slower | faster, the same | slower, slower | yes, at 0.003 |
| half, 0.50 | B | 0.003 | 0.003 | slower, slower | faster, faster | slower, slower | yes, at 0.003 |
| half, 0.25 | S | 0.003 | 0.003 | slower, slower | faster, faster | slower, slower | yes, at 0.003 |
| half, 0.25 | B | 0.003 | 0.003 | slower, slower | the same, faster | slower, slower | no |
| present, 0.50 | S | 0.001 (tied with 0.003) | 0.001 (tied with 0.003) | slower, slower | the same, the same | not run | no |
| present, 0.50 | B | 0.003 | 0.003 | slower, slower | the same, the same | slower, slower | no |
| present, 0.25 | S | 0.003 | 0.003 | neither reaches, slower | faster, faster | neither reaches, not decided | yes, at 0.003 |
| present, 0.25 | B | 0.003 | 0.003 | slower, slower | the same, the same | slower, slower | no |

**Further runs at 0.01.** The brief's rule asks for a run at 0.01 in every case whose best rate on R is 0.003; that is seven cases. In the eighth (the sweep, present width, f = 0.50) the best rate is a tie that includes 0.001.
- None of the seven runs at 0.01 is faster than the run at 0.003, and every one is slower than the run at 0.001 on R (or, for the sweep at present width with f = 0.25, neither run reaches 80 on R). The best rate therefore stays 0.003 in every one of the seven cases, and no case is reported as "at or above 0.01".
- No best rate fell at the lowest rate, 0.0003.

### 4.3 The lag at each width and fraction

The lag is formed on each score separately, from each rule's count at its best rate on that score. The lag at its own rates is *smaller* (or *larger*) only if it is below two thirds of (or above 1.5 times) the lag at 0.001. It is *gone* if it is smaller and at most 1.0, and otherwise *the same within the scatter of one seed*.

| Width, f | Score | Lag at 0.001 | Sweep's best rate | Back-propagation's best rate | Lag at the rules' own best rates | Reading |
|---|---|---|---|---|---|---|
| half, 0.50 | R | 1.53 (13,312 / 8,704) | 0.003 | 0.003 | 1.53 (7,424 / 4,864) | the same within the scatter of one seed |
| half, 0.50 | A | 1.38 | 0.003 | 0.003 | 1.42 | the same within the scatter of one seed |
| half, 0.25 | R | 3.31 (24,576 / 7,424) | 0.003 | 0.003 | 2.05 (11,520 / 5,632) | **smaller** |
| half, 0.25 | A | 2.29 | 0.003 | 0.003 | 1.93 | the same within the scatter of one seed |
| present, 0.50 | R | 1.19 (4,864 / 4,096) | 0.001 (tie with 0.003) | 0.003 | 1.73 (4,864 / 2,816) | the same within the scatter of one seed |
| present, 0.50 | A | 1.19 | 0.001 (tie with 0.003) | 0.003 | 1.73 | the same within the scatter of one seed |
| present, 0.25 | R | above 2.71 (the sweep is not reached; bound 11,776 / 4,352) | 0.003 | 0.003 | 1.75 (5,376 / 3,072) | **smaller** |
| present, 0.25 | A | 2.79 (9,984 / 3,584) | 0.003 | 0.003 | 1.33 (4,096 / 3,072) | **smaller** |

Two of these readings sit close to the band's edge.
- At present width with f = 0.25, R at 0.001 gives only a lower bound, and 1.75 lies just below two thirds of 2.71 (1.81). On A at the same width and fraction, where both counts exist, the lag falls from 2.79 to 1.33.
- At present width with f = 0.50, the lag rises from 1.19 to 1.73, a ratio of 1.45, inside the band. It rises because back-propagation is faster at 0.003 while the sweep is not.

### 4.4 The sentences the job can license, for seed 93683 only

> - **Half width, half of the pairs:** at this width and fraction the sweep's lag behind back-propagation is *the same within the scatter of one seed* when each rule is run at its own best learning rate among those tried (on R and on A).
> - **Half width, a quarter of the pairs:** at this width and fraction the sweep's lag behind back-propagation is *smaller* on R and *the same within the scatter of one seed* on A when each rule is run at its own best learning rate among those tried.
> - **Present width, half of the pairs:** at this width and fraction the sweep's lag behind back-propagation is *the same within the scatter of one seed* when each rule is run at its own best learning rate among those tried (on R and on A).
> - **Present width, a quarter of the pairs:** at this width and fraction the sweep's lag behind back-propagation is *smaller* when each rule is run at its own best learning rate among those tried (on R and on A).

Nowhere is the lag *gone*: the smallest lag at the rules' own rates is 1.33. Nowhere is it *larger*.

**The explanation under test.** The explanation is that the sweep's lag at half width with a quarter of the pairs is in part a matter of the learning rate. The brief says it is refuted for this seed in either of two cases:
- the sweep gains at no width and fraction; or
- wherever the sweep gains, back-propagation gains too and the lag is not smaller.

On this seed the sweep gains at three widths and fractions: half 0.50, half 0.25 and present 0.25.
- At half 0.25 and at present 0.25, back-propagation does not gain and the lag is smaller, on R.
- At half 0.50, back-propagation also gains and the lag is the same.

The explanation is therefore **not refuted for seed 93683**.

### 4.5 The product control (rate × decay held at 0.00025), half width, half of the pairs

| Rule | Rate | Decay | Updates to 80, A | Updates to 80, R | Class against 0.001: R, A | Final A | Final R | Length 40: last, mean of 3 | Norms at the end (book / l1 / input / recurrent / lm / l2 / l3) |
|---|---|---|---|---|---|---|---|---|---|
| S | 0.0003 | 0.25 (main) | 20,992 | 32,000 | slower, slower | 94.78 | 92.37 | 78.7, 82.9 | 49.0 / 3.3 / 17.1 / 17.3 / 17.7 / 3.7 / 57.0 |
| S | 0.0003 | 0.8333 (control) | 76,800 | – | slower, slower | 82.17 | 64.60 | 44.0, 45.3 | 59.4 / 2.7 / 11.9 / 13.2 / 13.9 / 1.8 / 87.8 |
| S | 0.003 | 0.25 (main) | 6,912 | 7,424 | faster, the same | 91.72 | 90.65 | 88.7, 91.4 | 60.8 / 4.2 / 33.5 / 22.9 / 20.2 / 8.6 / 69.7 |
| S | 0.003 | 0.08333 (control) | 5,376 | 5,632 | faster, faster | 97.56 | 97.78 | 98.0, 96.7 | 42.7 / 9.1 / 50.7 / 36.4 / 32.1 / 17.5 / 53.7 |
| B | 0.0003 | 0.25 (main) | 36,096 | 37,376 | slower, slower | 99.00 | 98.83 | 95.0, 94.8 | 49.5 / 3.7 / 14.6 / 15.9 / 15.0 / 1.8 / 49.8 |
| B | 0.0003 | 0.8333 (control) | 23,808 | 35,072 | slower, slower | 93.39 | 90.17 | 89.0, 90.1 | 59.3 / 3.1 / 10.1 / 13.1 / 13.0 / 1.1 / 63.7 |
| B | 0.003 | 0.25 (main) | 4,864 | 4,864 | faster, faster | 92.50 | 91.25 | 90.3, 89.9 | 114.7 / 3.6 / 31.1 / 21.7 / 19.6 / 7.4 / 63.6 |
| B | 0.003 | 0.08333 (control) | 5,376 | 6,912 | the same, the same | 97.61 | 97.13 | 98.7, 97.7 | 57.9 / 5.0 / 38.6 / 32.6 / 29.5 / 16.8 / 43.1 |

The brief asks for one sentence for each rule and rate on whether the count to 80 falls in the same class under both ways of setting the decay:
- **The sweep at 0.0003:** the same class (slower on R and on A) under both. With the decay at 0.833 it never reaches 80 on R in 136,704 updates, and its final R is 64.60 against 92.37.
- **The sweep at 0.003:** the same class on R (faster under both). The classes differ on A: the same with decay 0.25, faster with decay 0.0833.
- **Back-propagation at 0.0003:** the same class (slower on R and on A) under both.
- **Back-propagation at 0.003:** different classes. It is faster on R and A with decay 0.25, and the same on R and A with decay 0.0833 (6,912 and 5,376 against 8,704 and 7,424 at 0.001).

Holding the product fixed moves the size of the weights with the rate, as the brief expected. At 0.003 the control's input and recurrent norms at the end are 50.7 and 36.4 under the sweep, against 33.5 and 22.9 with decay 0.25. At 0.0003 they are 11.9 and 13.2, against 17.1 and 17.3. The control does not change the main readings, which are made at decay 0.25. It does show that at half width with f = 0.50, the sweep's count to R80 at 0.003 depends on the decay (5,632 against 7,424). That difference lies within the band.

## 5. Interruptions, departures from the brief, and cost

**Worker restarts.** The session's worker process was restarted three times, at about 00:44, 01:31 and 05:55 UTC on 18 November. Each restart killed every process; the disk was kept.

- **00:44.** The runs at 0.003 of the sweep at half width with f = 0.25, and both product controls at 0.003, were killed.
  - Their partial receipts and logs were committed as left and moved to `exp45/e52/interrupted/`.
  - All three were started again from the beginning with `exp52_rate.py`, as the brief requires.
- **01:31.** The runs at 0.003 at present width of B with f = 0.50, S with f = 0.50 and B with f = 0.25 were killed about 10 minutes in, and were started again from the beginning in the same way.
  - The first attempt at check 4 had already failed before this restart (see below); its second attempt was started after the restart.
- **05:55.** Three runs at 0.0003 were killed:
  - the sweep at half width with f = 0.25;
  - both product controls at 0.0003.
  
  They were **resumed** from their checkpoints at updates 69,632, 73,728 and 61,440; the resumption is recorded in each receipt's `resumed` field.

**Departure 1: saving and resuming.** After the first restart, the reviewer asked for saving and resuming to be added if it could be shown to reproduce an uninterrupted run (message of 17 November, 22:11 UTC). The worker wrote `exp52_resume.py`.
- It is `exp52_rate.run` with two additions: a checkpoint of the model, the optimizer, the receipt so far and the CPU counters every 4,096 updates, and `--resume`.
- On resume it rebuilds the stream and the batch digest by drawing and digesting the same batches again, about 45–55 seconds per resumption. It asserts that the rebuilt digest equals the checkpoint's before it loads the states.
- Check 4 (Section 3) showed that a killed and resumed run equals an uninterrupted one in `tensors_sha256`, in every row, and in the batch digest. It also showed that `exp52_resume.py` equals `exp52_rate.py` uninterrupted.
- All runs at 0.0003 and at 0.01 used `exp52_resume.py`. All runs at 0.001 and 0.003 used `exp52_rate.py`.
- The receipt records the runner in `provenance.runner` and `runner_sha256`.

**Departure 2: check 4's first attempt.** The first attempt failed before any comparison.
- Its child process inherited `E11_`/`E16_` variables that the harness sets at import, and `exp20_reference`'s environment gate refused them.
- The check now launches its children without those variables.
- The failed attempt is kept as `exp45/e52_qual/check4_attempt1_failed_runs_20261118T013037Z/` and `exp45/e52_qual/logs/check4_attempt1_failed_20261118T013037Z.log`.

**Departure 3: the driver.** The first driver, `exp52_drive.py`, was stopped by hand after the first restart; the runs it had started were left alive. It was replaced by `exp52_drive2.py`.
- The new driver counts every live run on the host toward the three slots.
- It resumes a run of `exp52_resume.py` that has a checkpoint.
- It runs check 4 in a free slot.
- It uses `exp52_resume.py` at 0.0003 only if check 4 passed.
- The launch log `exp45/e52/logs/LAUNCHES.txt` records every start, exit, move to `interrupted/`, restart and resumption.

**Departure 4: the timing of the further runs.** The runs at 0.01 of back-propagation and of the sweep at present width with f = 0.25 were started at 09:43. The sweep's run at 0.0003 in that case was still running at the time.
- That run was already past 36,000 updates without reaching 80 on either score, so it could no longer become the best rate against 0.003's 5,376 and 4,096.
- It finished without reaching 80, as Table 1 shows.

**Departure 5: commit trailers.** `PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md` asks for `Acting-Agent: claude:<session>`. Commits carry only that trailer, so that no language model is named, as the standing rules require.

**Cost** (CPU-hours, from the receipts).

| Item | CPU-hours |
|---|---|
| The 28 runs of the brief | 27.88 (updates 23.25, scoring 4.63) |
| The 7 further runs at 0.01 | 3.73 (updates 3.09, scoring 0.63) |
| Partial runs killed by the restarts (`interrupted/`) | 1.45 |
| Checks 1a, 2, 3 and 4 | 0.40 |

- The cost of the 28 runs is within the brief's limit of 32.5.
- The CPU counters of a resumed run carry over its checkpoint's counters. The updates between a checkpoint and the restart, at most 4,096 per run, were done twice; the repeats are counted only once.
- Each run records in its receipt how many other runs were alive when it started (`processes_beside`).

## 6. The predictions (the reviewer's, on file before any run)

| | Statement | Chance | Outcome |
|---|---|---|---|
| P52.1 | Under back-propagation the best learning rate on R is 0.001 or 0.003, or a tie that includes one of them, at all four widths and fractions | 75% | **Occurred.** The best rate is 0.003 at all four |
| P52.2 | The sweep gains from its own rate at one or more of the four widths and fractions | 30% | **Occurred.** It gains at half 0.50, half 0.25 and present 0.25, at 0.003 in each |
| P52.3 | At the present width with a quarter of the pairs taught, the sweep's mean of R over the last three rows is at least 85 at some rate tried | 35% | **Occurred.** 95.05 at 0.003; 72.53 at 0.0003, 78.78 at 0.001 and 21.13 at 0.01 |
| P52.4 | At the half width with a quarter of the pairs, the lag on R at the rules' own rates is *smaller* than at 0.001 (3.31 on this seed) | 25% | **Occurred.** 2.05, below two thirds of 3.31 (2.21) |
| P52.5 | Back-propagation gains from its own rate at none of the four widths and fractions | 75% | **Did not occur.** Back-propagation gains at half width with f = 0.50 (R 4,864 against 8,704; A 4,864 against 7,424, at 0.003) |
| P52.6 | At the half width with a quarter of the pairs, the sweep's R is steadier at 0.0003 than at 0.001 (a smaller root mean square change over the last 16 rows) | 65% | **Occurred.** 1.77 against 2.45 |

## 7. What the trial cannot show

The brief's limits apply.
- The trial says nothing about the other settings of the optimizer, another optimizer, or the scale inside the sweep.
- It says nothing about a learning rate that changes during a run.
- It covers only these two widths, these two fractions and this one seed.
- It says nothing about a life. A birth made at a rate other than 0.001 is not admitted to a life by `exp26_candidate.admit_e26`.

Two further limits come from what was seen.
- The best rate is 0.003 in seven of eight cases, and every run at 0.01 is slower. The best rate therefore lies between 0.001 and 0.01, but the trial cannot place it more finely than the three-fold grid.
- A rate that is faster to 80 is not shown to be better at the end of the dose. For example, back-propagation at half width with f = 0.50 reaches 80 on R in 4,864 updates at 0.003, against 8,704 at 0.001, but ends lower after the same dose: final R 91.25 against 93.97. The difference is about one row-to-row RMS change (3.97), so one seed cannot settle it. The runs at 0.0003 end highest in several cases, but their dose is 10/3 as long.

## 8. After the trial (for the reviewer)

The brief says the other two seeds are then run for every width and fraction at which either rule gains from its own rate, or at which the lag changes class. On this seed those are:
- half width, f = 0.50 (both rules gain);
- half width, f = 0.25 (the sweep gains; the lag on R is smaller);
- present width, f = 0.25 (the sweep gains; the lag is smaller).

Present width with f = 0.50 does not qualify. The sweep gains at half width, so the brief's condition for asking again for the sweep's long lives (plan item 6) at the rate 0.003 is met. Running those lives is a decision for the reviewer and the coordinator, and nothing beyond the brief was run.

## 9. Committed files

The list below is from `git ls-tree -r --name-only HEAD` on the worker's branch, restricted to the files E52 added. It is regenerated after this report is committed, and given in Section 9.1.
