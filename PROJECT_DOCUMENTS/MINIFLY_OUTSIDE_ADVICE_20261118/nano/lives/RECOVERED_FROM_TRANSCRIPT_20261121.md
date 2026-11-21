# Run results recovered from the stopped session's transcript (21 November 2026)

Written by session `session_017dSEpWqUVkyw5SyyeE6M13` on branch `claude/magical-albattani-sei232`, after merging `claude/friendly-darwin-8mmhfm` at `f31b2dee5`. No run was made to write this file; nothing here was recomputed.

**Where these come from.** The transcript of the stopped session `session_017dSEpVed7SthEH3ngGAdNc`, read through the remote-session event listing (user and assistant events; the newest page, which runs from 03:34:25 to 03:47:23 UTC on 21 November). That session's last commit is `f31b2dee5` (03:37:13 UTC, `PLAN_16.md` written before any run). It stopped at a usage limit at 03:47:21 UTC. Below are, verbatim and in order, the outputs of its commands printed after that commit that contain run results; each is headed by its timestamp and by the session's own description of the command. Nothing was edited, except that each output is set in a code block. Where a line ends abruptly, the stopped session itself cut it (`cut -c1-520`, `cut -c1-330` and similar in its commands); the full lines were never printed.

**What was not committed.** The result files behind these figures (`res/aba__dev16_*.json`, `res/relay15__r_*.json`, `res/mem__ZAM32x512_N65536_seed545353.json`, and the logs in `out/`) were never committed and are presumed lost with that session's container. The files of these runs that were committed are `res/relay15__dev_R_rho0.01.json`, `res/relay15__dev_Rgtqr_rho0.01.json`, `res/mem__ZAM32x512_N65536_seed73129.json` and `res/mem__ZAM32x512_N65536_seed116089.json`, and the checks `check15_output.txt` and `check16_output.txt`.

**Status of the figures.** Unverified and development level throughout: copied from a transcript, not from files; not repeated by anyone; no outcome of any plan is stated here, and the stopped session stated none. No checker gate, seal or review is claimed.

**What was unfinished at 03:47 UTC.**

* `PLAN_16.md`: the six development runs (forms A and B, mu 0.01, 0.03, 0.1, seed 1) had finished; the choice among them that the plan prescribes (its section on numbers) had not been made by the session, and none of the five reported seeds (73129, 93683, 116089, 406364, 545353) had started. The rows `dev_S1`, `dev_P`, `dev_P_a0.1`, `dev_P_a0.3` in the last table are earlier runs of `PLAN_11.md`, printed beside them.
* `PLAN_15.md`: twenty reported runs are prescribed (five seeds, four arms; `jobs_15.sh`). Twelve result files `relay15__r_*` existed at 03:45:59 (the count printed there); the rehearsal table last printed (03:43:57) covers 3, 2, 4 and 2 seeds of its four arms. At least eight runs were unfinished; no outcome was written.
* The bounded-kinds system with remembered trusts (`mem.py`, `jobs_5.sh both ZAM32x512 65536` over five seeds; the session called it "the fifth plan's last open prediction"): three seeds had finished (73129, 116089, 545353), seed 93683 was running and seed 406364 had not started.
* `PLAN_12.md` has no outcome either; nothing for it is in this page of the transcript.

**Reading the rehearsal rows.** The columns of the `report_13.py` table are, as its committed output `report_13.txt` heads them: Arm | seeds | end of the sums | straight after English 1; 2; 3 | kept (share of the level just before) | the same, only digit branches answering | after 100 lines of return 1; 2; 3 | after 5,000 lines | held-out English: end of English 1; 2; 3 | its fall over return 1; 2; 3. Each cell is a median with the range in brackets.

## The outputs, verbatim

### Tool result at 2026-11-21T03:37:21.020073Z (command issued 2026-11-21T03:37:20.227350Z)

The stopped session's description of the command: "Wait for more reported rehearsal runs and peek at the own-step development runs"

```
Sat Nov 21 03:37:20 UTC 2026
6
| R, rehearsal 1/100 (plan 15) | 1 | 96.3 (96.3 to 96.3) | 0.0 (0.0 to 0.0); 0.0 (0.0 to 0.0); 0.0 (0.0 to 0.0) | 0.00 (0.00 to 0.00); 0.00 (0.00 to 0.00); 0.00 (0.00 to 0.00) | 14.0 (14.0 to 14.0); 6.1 (6.1 to 6.1); 2.0 (2.0 to 2.0) | 1.1 (1.1 to 1.1); 0.0 (0.0 to 0.0); 0.2 (0.2 to 0.2) | 97.9 (97.9 to 97.9); 95.0 (95.0 to 95.0); 96.4 (96.4 to 96.4) | 41.3 (41.3 to 41.3); 40.0 (40.0 to 40.0); 41.1 (41.1 to 41.1) | 9.8 (9.8 to 9.8); 3.2 (3.2 to 3.2); 2.5 (2.5 to 2.5) |
| R, rehearsal 1/20 (plan 15) | 1 | 96.0 (96.0 to 96.0) | 2.7 (2.7 to 2.7); 1.9 (1.9 to 1.9); 0.0 (0.0 to 0.0) | 0.03 (0.03 to 0.03); 0.02 (0.02 to 0.02); 0.00 (0.00 to 0.00) | 68.0 (68.0 to 68.0); 47.2 (47.2 to 47.2); 44.8 (44.8 to 44.8) | 38.6 (38.6 to 38.6); 23.1 (23.1 to 23.1); 2.8 (2.8 to 2.8) | 98.1 (98.1 to 98.1); 96.5 (96.5 to 96.5); 97.5 (97.5 to 97.5) | 40.9 (40.9 to 40.9); 39.8 (39.8 to 39.8); 40.3 (40.3 to 40.3) | 10.5 (10.5 to 10.5); 1.3 (1.3 to 1.3); 1.8 (1.8 to 1.8) |
| Rgtqr, rehearsal 1/100 (plan 15) | 2 | 98.4 (98.3 to 98.6) | 80.9 (72.6 to 89.3); 93.4 (90.8 to 96.0); 95.4 (92.4 to 98.4) | 0.82 (0.74 to 0.91); 0.95 (0.93 to 0.97); 0.98 (0.95 to 1.00) | 96.5 (95.4 to 97.6); 97.4 (96.6 to 98.2); 97.7 (96.7 to 98.6) | 92.2 (88.4 to 96.0); 97.3 (96.3 to 98.3); 96.7 (95.0 to 98.4) | 98.5 (98.0 to 99.0); 97.6 (96.9 to 98.4); 97.0 (95.5 to 98.6) | 46.5 (46.4 to 46.6); 44.5 (44.4 to 44.7); 43.8 (43.6 to 44.0) | 0.4 (0.3 to 0.5); 0.1 (-0.3 to 0.5); -0.2 (-0.6 to 0.2) |
| Rgtqr, rehearsal 1/20 (plan 15) | 2 | 98.6 (98.1 to 99.1) | 96.8 (96.7 to 96.9); 96.5 (95.2 to 97.8); 97.5 (97.1 to 98.0) | 0.98 (0.98 to 0.99); 0.98 (0.96 to 0.99); 1.00 (1.00 to 1.00) | 98.0 (97.8 to 98.2); 97.5 (96.6 to 98.5); 97.8 (97.4 to 98.1) | 97.7 (97.6 to 97.8); 98.2 (97.8 to 98.5); 97.1 (96.3 to 98.0) | 98.9 (98.6 to 99.1); 97.4 (97.2 to 97.6); 97.4 (96.6 to 98.2) | 46.8 (46.4 to 47.1); 44.6 (44.6 to 44.7); 44.5 (44.3 to 44.7) | 0.7 (0.3 to 1.1); -0.4 (-1.6 to 0.9); -0.4 (-0.4 to -0.3) |
```

### Tool result at 2026-11-21T03:37:30.876864Z (command issued 2026-11-21T03:37:30.306932Z)

The stopped session's description of the command: "Check the lives of the bounded-kinds system with remembered trusts (the fifth plan's last open prediction)"

```
mem__ZAM32x512_N65536_seed116089.json
mem__ZAM32x512_N65536_seed545353.json
mem__ZAM32x512_N65536_seed73129.json
10868     434 python3 mem.py life tag=ZAM32x512_N65536_seed93683 order=seed:93683 sys=ZAM32x512 N=65536
11108     126 python3 mem.py life tag=ZAM32x512_N65536_seed93683 order=seed:93683 sys=ZAM32x512 N=65536
11162      18 python3 mem.py life tag=ZAM32x512_N65536_seed93683 order=seed:93683 sys=ZAM32x512 N=65536
29566   10017 /bin/bash ./jobs_5.sh both ZAM32x512 65536 73129 116089 545353 93683 406364
_seed116089 mean at end: living 0.8781 mixture 0.8361 | cost of life: living 0.073 mixture 0.031 | singles mean 0.8052 bookmarks 120
_seed545353 mean at end: living 0.8401 mixture 0.8326 | cost of life: living 0.0349 mixture 0.0274 | singles mean 0.8052 bookmarks 120
6_seed73129 mean at end: living 0.8327 mixture 0.825 | cost of life: living 0.0275 mixture 0.0198 | singles mean 0.8052 bookmarks 120
```

### Tool result at 2026-11-21T03:40:08.503390Z (command issued 2026-11-21T03:38:13.188006Z)

The stopped session's description of the command: "Wait for the first own-step development runs"

```
Sat Nov 21 03:40:08 UTC 2026
8
dev16_A_mu0.03 end of the sums {'cell1': 74.2} cells 1213119
dev16_B_mu0.03 end of the sums {'cell1': 80.1} cells 1213119
```

### Tool result at 2026-11-21T03:40:38.926173Z (command issued 2026-11-21T03:40:13.538879Z)

The stopped session's description of the command: "Wait for the first two own-step development runs to finish and print them"

```
Sat Nov 21 03:40:38 UTC 2026
dev16_A_mu0.03 | never-shown sums, whole answer: end of the sums: 74.2 | end of the English: 73.3 | return 500 lines: 73.8 | return 1000 lines: 73.8 | return 2000 lines: 73.9 | return 5000 lines: 73.2 | English, nats a byte, first and last tenth: [3.427, 1.869] | settings {'mu': 0.03, 'tau': 10000.0, 'alpha0': 0.03} | wall 202.2

dev16_B_mu0.03 | never-shown sums, whole answer: end of the sums: 80.1 | end of the English: 79.8 | return 500 lines: 80.5 | return 1000 lines: 79.3 | return 2000 lines: 80.2 | return 5000 lines: 79.8 | English, nats a byte, first and last tenth: [2.264, 1.798] | settings {'mu': 0.03, 'tau': 10000.0, 'alpha0': 0.03} | wall 200.8

8
```

### Tool result at 2026-11-21T03:42:50.993381Z (command issued 2026-11-21T03:40:55.661750Z)

The stopped session's description of the command: "Wait for the next pair of own-step development runs"

```
Sat Nov 21 03:42:50 UTC 2026
dev16_A_mu0.03 | never-shown sums, whole answer: end of the sums: 74.2 | end of the English: 73.3 | return 500 ... English, nats a byte, first and last tenth: [3.427, 1.869] | settings 
dev16_A_mu0.1 end of the sums {'cell1': 74.3} cells 1213119 ... 
dev16_B_mu0.03 | never-shown sums, whole answer: end of the sums: 80.1 | end of the English: 79.8 | return 500 ... English, nats a byte, first and last tenth: [2.264, 1.798] | settings 
dev16_B_mu0.1 end of the sums {'cell1': 81.5} cells 1213119 ... 
11
```

### Tool result at 2026-11-21T03:43:57.579249Z (command issued 2026-11-21T03:42:56.666313Z)

The stopped session's description of the command: "Wait for the second pair of own-step runs to finish; print them and the rehearsal table so far"

```
Sat Nov 21 03:43:56 UTC 2026
dev16_A_mu0.1 | never-shown sums, whole answer: end of the sums: 74.3 | end of the English: 72.4 | return 500 lines: 73.7 | return 1000 lines: 73.7 | return 2000 lines: 74.2 | return 5000 lines: 73.9 | English, nats a byte, first  ... English, nats a byte, first and last tenth: [3.606, 2.006] | settings 
dev16_B_mu0.1 | never-shown sums, whole answer: end of the sums: 81.5 | end of the English: 80.7 | return 500 lines: 82.3 | return 1000 lines: 81.3 | return 2000 lines: 82.1 | return 5000 lines: 82.2 | English, nats a byte, first  ... English, nats a byte, first and last tenth: [2.275, 1.846] | settings 
| R, rehearsal 1/100 (plan 15) | 3 | 96.3 (95.8 to 97.8) | 0.0 (0.0 to 0.0); 0.0 (0.0 to 0.0); 0.0 (0.0 to 0.0) | 0.00 (0.00 to 0.00); 0.00 (0.00 to 0.00); 0.00 (0.00 to 0.00) | 26.2 (14.0 to 28.1); 9.8 (6.1 to 23.8); 3.8 (2.0 to 22.8) | 1.1 (0.2 to 7.8); 0.0 (0.0 to 1.3); 0.2 (0.0 to 1.2) | 96.9 (93.9 to 97.9); 95.0 (95.0 to 97
| R, rehearsal 1/20 (plan 15) | 2 | 95.7 (95.4 to 96.0) | 4.7 (2.7 to 6.7); 4.0 (1.9 to 6.2); 0.0 (0.0 to 0.0) | 0.05 (0.03 to 0.07); 0.04 (0.02 to 0.07); 0.00 (0.00 to 0.00) | 70.1 (68.0 to 72.1); 57.5 (47.2 to 67.7); 50.8 (44.8 to 56.8) | 42.3 (38.6 to 46.0); 33.8 (23.1 to 44.5); 10.1 (2.8 to 17.4) | 96.7 (95.3 to 98.1); 96.5 
| Rgtqr, rehearsal 1/100 (plan 15) | 4 | 98.4 (98.1 to 98.6) | 83.9 (72.6 to 89.4); 92.1 (88.1 to 96.0); 95.4 (92.4 to 98.4) | 0.85 (0.74 to 0.91); 0.94 (0.89 to 0.97); 0.97 (0.95 to 1.00) | 97.0 (95.4 to 98.3); 97.8 (96.6 to 98.7); 98.1 (96.7 to 98.6) | 94.1 (88.4 to 96.0); 97.0 (96.3 to 98.3); 97.5 (95.0 to 98.4) | 98.5 (98.0 
| Rgtqr, rehearsal 1/20 (plan 15) | 2 | 98.6 (98.1 to 99.1) | 96.8 (96.7 to 96.9); 96.5 (95.2 to 97.8); 97.5 (97.1 to 98.0) | 0.98 (0.98 to 0.99); 0.98 (0.96 to 0.99); 1.00 (1.00 to 1.00) | 98.0 (97.8 to 98.2); 97.5 (96.6 to 98.5); 97.8 (97.4 to 98.1) | 97.7 (97.6 to 97.8); 98.2 (97.8 to 98.5); 97.1 (96.3 to 98.0) | 98.9 (98.6 t
```

### Tool result at 2026-11-21T03:45:59.349469Z (command issued 2026-11-21T03:44:04.074570Z)

The stopped session's description of the command: "Wait for the last own-step development runs"

```
Sat Nov 21 03:45:59 UTC 2026
dev16_A_mu0.01 end of the sums {'cell1': 74.0} cells 1213119 ... 
dev16_B_mu0.01 end of the sums {'cell1': 79.4} cells 1213119 ... 
12
```

### Tool result at 2026-11-21T03:47:21.190902Z (command issued 2026-11-21T03:46:05.678469Z)

The stopped session's description of the command: "Wait for the last own-step development runs and tabulate all six beside the earlier learners"

```
Sat Nov 21 03:47:21 UTC 2026
form mu | end of sums | end of English | lost | English nats last tenth
dev16_A_mu0.01 | 74.0 | 73.1 | 0.9 | 1.845
dev16_A_mu0.03 | 74.2 | 73.3 | 1.0 | 1.869
dev16_A_mu0.1 | 74.3 | 72.4 | 1.9 | 2.006
dev16_B_mu0.01 | 79.4 | 79.2 | 0.2 | 1.801
dev16_B_mu0.03 | 80.1 | 79.8 | 0.3 | 1.798
dev16_B_mu0.1 | 81.5 | 80.7 | 0.8 | 1.846
dev_S1 | 81.2 | 77.3 | 3.9 | 1.509
dev_P | 73.0 | 73.2 | -0.2 | 1.559
dev_P_a0.1 | 78.4 | 77.8 | 0.6 | 1.741
dev_P_a0.3 | 79.5 | 78.7 | 0.8 | 2.774
```

