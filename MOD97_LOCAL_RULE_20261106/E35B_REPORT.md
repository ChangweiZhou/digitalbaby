# E35-B report: inventory 乙 (phase addition and a comparator) — stage 0 and stage 1a, seed 73129

Brief: `OPUS_TASK_E35B_INNATE_CIRCUITS_INVENTORY_YI_20261115.md`, revision 2 (dc05107d on `codex/minifly-consolidated-20261023`). Branch `claude/busy-darwin-dlrwtm`.
Host: Linux x86-64, Python 3.13.18, torch 2.15.0+cpu, one thread per process, at most three processes. Evidence level E2; every network is trained by backpropagation (reference stage). Arm G's figures are the E35-A worker's (`claude/nice-rubin-y5de8k` at d5b89442, `exp45/e35a/e35a_dev_G_f{10,25}_73129.json`, complete); the data digest of its f = 0.10 birth equals that of this worker's M birth (a5f5e3a8…).

## Stage 0: parsing supplied (`exp35b_parsed.py`)

48 runs: the three code forms × two optimizers × two fractions × three seeds for the module (36), and the scrambled module for the angle form (12). The brief says "132 small runs"; the grid it writes is 48, and 48 were run. Measured CPU: 2,916 s in total. Full table with per-operation figures and the curves every 4,096 updates: `exp45/e35b/stage0/E35B_STAGE0_TABLE.md`; receipts beside it.

Final-row held-out mean over four operations (seeds 73129 / 93683 / 116089):

| f | form | E10 optimizer | stream optimizer |
|---|---|---|---|
| 0.25 | angle | 99.8 / 99.9 / 99.8 | 99.7 / 99.7 / 99.8 |
| 0.25 | complex | 98.6 / 99.7 / 99.7 | 99.6 / 99.6 / 99.6 |
| 0.25 | table | 99.3 / 99.4 / 99.5 | 98.8 / 98.9 / 53.6 |
| 0.25 | scrambled (angle) | 33.6 / 21.4 / 21.7 | 8.7 / 6.8 / 9.0 |
| 0.10 | angle | 99.6 / 99.7 / 99.6 | 99.4 / 99.5 / 99.6 |
| 0.10 | complex | 57.3 / 99.4 / 99.6 | 56.2 / 99.3 / 99.5 |
| 0.10 | table | 98.6 / 98.7 / 98.8 | 97.9 / 98.2 / 96.0 |
| 0.10 | scrambled (angle) | 32.3 / 24.4 / 17.2 | 17.5 / 11.8 / 8.8 |

The three runs below 90 among the module runs (complex f = 0.10 seed 73129 under both optimizers; table/stream f = 0.25 seed 116089) have add and sub between 6 and 30 and max and min above 95.

**Gate: PASS.** At f = 0.25 every form/optimizer pair reaches at least 90 in at least two of three seeds (angle: 3 of 3 under both). The stream stage uses the angle form.

## Checks (`exp45/e35b/checks/`, `exp45/e35b/dry/`)

| # | Check | Result |
|---|---|---|
| 1 | Arm G dry birth, f = 0.5, E26's panels, 2,048 updates | `tensors_sha256` adf933d7… equals `e26_dev_B_dry2048_73129.json`: **pass** (226.8 CPU-s) |
| 2 | Frequencies (1, 2, 3, 5, 8, 13, 21, 34), scalar 10 v: mul, div, big, small decode to add, sub, max, min | 7,921 / 7,921 each; `mul(a, div(b, a))` vs b max error 1.8e-7 (float32): **pass**. Scrambled maps: constants 0.090627, 0.090136, 0.088612, 0.090731 (mean norm after scaling 2.828427); raw W sha256 5b794f1e…, scaled sha256 be22608f…, d7d2ba22…, 70fbed86…, fe06d11d… |
| 2b | `module_path` with ideal c, g, o; 4 operations × gaps 0, 12, 40, junk codes at other steps | 7,921 / 7,921 in all twelve cells: **pass** |
| 3 | M, X, F start with every G parameter unchanged; M and X identical | **pass** |
| 4 | module off: M's logits equal a G holding the same shared parameters (initial and perturbed) | bitwise equal, max abs 0.0: **pass** |
| 5 | forward signature `(m, x, local, through_time, keep, silence)`, byte tensor only; no mask/role/position in its source | **pass** |
| 6 | Dry births M, X at f = 0.25, 2,048 updates | finite losses (0.563, 0.668); 291.6 and 296.5 CPU-s; forecast about 3,500 CPU-s per full birth, about 4 CPU-h for stage 1a: under 10, continued |

Parameters: G 1,124,113; M (and X, F) 1,196,964.

## Stage 1a (`exp35b_stream.py`), seed 73129

Births run: M, X and F at f = 0.10. Not run: M and X at f = 0.25 (no family has room there; below). Receipts `exp45/e35b/e35b_dev_<arm>_f10_73129.json`; weights at rows 16,384, 20,480, 24,576 saved and committed beside them. CPU: M 3,413 s, X 3,472 s, F 3,423 s; all losses finite.

Room (arm G held12, last-three mean, below 85): f = 0.10 add/sub 19.6 (room); f = 0.10 max/min 91.0; f = 0.25 add/sub 89.2; f = 0.25 max/min 97.2.

**held12** (last-three mean / final row)

| arm | f | mean | add | sub | max | min | add/sub | max/min |
|---|---|---|---|---|---|---|---|---|
| G (E35-A) | 0.10 | 55.3 / 55.0 | 21.8 / 22.0 | 17.4 / 19.0 | 91.0 / 90.0 | 91.0 / 89.0 | 19.6 / 20.5 | 91.0 / 89.5 |
| M | 0.10 | 55.8 / 56.4 | 23.8 / 26.3 | 16.1 / 16.3 | 90.4 / 90.0 | 92.7 / 93.0 | 19.9 / 21.3 | 91.6 / 91.5 |
| X | 0.10 | 58.2 / 60.4 | 26.1 / 29.0 | 21.1 / 26.0 | 91.9 / 92.7 | 93.6 / 94.0 | 23.6 / 27.5 | 92.7 / 93.3 |
| F | 0.10 | 54.2 / 56.3 | 20.1 / 23.0 | 15.9 / 17.3 | 89.9 / 89.7 | 91.1 / 95.3 | 18.0 / 20.2 | 90.5 / 92.5 |

**held40** (last-three mean / final row)

| arm | f | mean | add | sub | max | min | add/sub | max/min |
|---|---|---|---|---|---|---|---|---|
| G (E35-A) | 0.10 | 53.9 / 53.3 | 21.7 / 23.7 | 18.0 / 19.3 | 88.1 / 85.3 | 87.7 / 85.0 | 19.8 / 21.5 | 87.9 / 85.2 |
| M | 0.10 | 54.7 / 54.9 | 22.7 / 23.3 | 15.1 / 14.7 | 89.9 / 89.7 | 91.0 / 92.0 | 18.9 / 19.0 | 90.4 / 90.8 |
| X | 0.10 | 57.6 / 59.7 | 26.6 / 30.3 | 20.7 / 25.0 | 91.7 / 91.3 | 91.6 / 92.0 | 23.6 / 27.7 | 91.6 / 91.7 |
| F | 0.10 | 53.7 / 55.2 | 20.0 / 24.3 | 15.7 / 16.7 | 89.1 / 88.0 | 90.1 / 92.0 | 17.8 / 20.5 | 89.6 / 90.0 |

**taught12** (last-three mean / final row)

| arm | f | mean | add | sub | max | min | add/sub | max/min |
|---|---|---|---|---|---|---|---|---|
| G (E35-A) | 0.10 | 98.3 / 97.9 | 97.8 / 98.0 | 97.2 / 97.3 | 99.1 / 98.0 | 99.2 / 98.3 | 97.5 / 97.7 | 99.2 / 98.2 |
| M | 0.10 | 99.4 / 99.6 | 98.4 / 99.0 | 99.3 / 99.3 | 99.8 / 100.0 | 100.0 / 100.0 | 98.9 / 99.2 | 99.9 / 100.0 |
| X | 0.10 | 99.4 / 99.5 | 99.3 / 99.7 | 98.7 / 98.7 | 99.9 / 100.0 | 99.8 / 99.7 | 99.0 / 99.2 | 99.8 / 99.8 |
| F | 0.10 | 99.7 / 99.6 | 99.8 / 99.7 | 99.2 / 99.7 | 99.7 / 99.0 | 100.0 / 100.0 | 99.5 / 99.7 | 99.8 / 99.5 |

**held12, module off** (last-three mean / final row)

| arm | f | mean | add | sub | max | min | add/sub | max/min |
|---|---|---|---|---|---|---|---|---|
| M | 0.10 | 24.0 / 25.2 | 9.9 / 11.3 | 7.4 / 7.3 | 32.8 / 36.0 | 45.8 / 46.3 | 8.7 / 9.3 | 39.3 / 41.2 |
| X | 0.10 | 30.0 / 32.4 | 2.7 / 5.0 | 2.7 / 3.3 | 60.6 / 64.3 | 54.1 / 57.0 | 2.7 / 4.2 | 57.3 / 60.7 |
| F | 0.10 | 0.9 / 0.9 | 1.0 / 1.0 | 1.3 / 1.3 | 0.0 / 0.0 | 1.3 / 1.3 | 1.2 / 1.2 | 0.7 / 0.7 |

**held40, module off** (last-three mean / final row)

| arm | f | mean | add | sub | max | min | add/sub | max/min |
|---|---|---|---|---|---|---|---|---|
| M | 0.10 | 22.1 / 23.5 | 9.9 / 11.0 | 7.3 / 7.0 | 30.8 / 34.3 | 40.4 / 41.7 | 8.6 / 9.0 | 35.6 / 38.0 |
| X | 0.10 | 28.5 / 30.2 | 2.3 / 4.3 | 2.7 / 3.3 | 59.9 / 62.0 | 49.0 / 51.0 | 2.5 / 3.8 | 54.4 / 56.5 |
| F | 0.10 | 0.9 / 0.9 | 1.0 / 1.0 | 1.3 / 1.3 | 0.0 / 0.0 | 1.3 / 1.3 | 1.2 / 1.2 | 0.7 / 0.7 |
At f = 0.25 the figures are G's only (E35-A): held12 add/sub 89.2 / 90.3, max/min 97.2 / 97.0; held40 87.1 / 89.3, 95.0 / 96.2; taught12 95.0 / 96.0, 99.2 / 99.5.

Arm F's module-off figures are with the direct read-out's digit logits removed as well, so digits cannot be emitted.

### Gate diagnostics at the end of each module birth

Training batch at update 24,576, by the role of the byte just read (`exp28_everybyte.roles_of`, shifted by one). Tables per role: `exp45/e35b/E35B_STAGE1A_TABLE.md`; every 512 updates in the receipts.

| birth | max op weight at `=` | argmax op at `=` (mul, div, big, small) | gate A / gate B at `=` | q tens / units at `=` | gate B at distractors | gate B at operands |
|---|---|---|---|---|---|---|
| M | 0.932 | 0, 0, 64, 0 | 0.049 / 0.004 | 0.667 / 0.703 | 0.001 | 0.189 |
| X | 0.876 | 0, 45, 0, 19 | 0.034 / 0.965 | 0.982 / 0.932 | 0.897 | 0.770 |
| F | 0.976 | 0, 0, 64, 0 | 0.010 / 0.012 | 1.000 / 1.000 | 0.003 | 0.318 |

## Outcome rows

| family | f | pattern | row |
|---|---|---|---|
| add/sub | 0.10 | room (G 19.6); M 19.9, within 15 of G | **3** |
| max/min | 0.10 | G 91.0 | 0 |
| add/sub | 0.25 | G 89.2 | 0 |
| max/min | 0.25 | G 97.2 | 0 |

"Uses the module" (M held12 falls by at least 15 with the module off): add/sub at f = 0.10, 19.9 → 8.7 (11.2 points): no. (max/min at f = 0.10, a row-0 cell: 91.6 → 39.3.)

**After row 3, arm F at f = 0.10, seed 73129:** add/sub held12 18.0, against G 19.6 and M 19.9. F is within 15 of G: **"Forced through the module, the controller still did not learn the interface on this seed. Stage 1b applies the stop rule with arm F."** Stage 1b is released by the coordinator; nothing of it has been run.

## Predictions

| | Prediction | Outcome |
|---|---|---|
| P35B.1 | Stage 0 gate passes with the angle or complex form (50%) | occurred (angle and complex, 3 of 3 seeds under both optimizers) |
| P35B.7 | Digit tables reach 90 at f = 0.25 in two of three seeds under some optimizer (60%) | occurred (E10 optimizer 3 of 3; stream 2 of 3) |
| P35B.2 | Add and sub reach 90 at f = 0.25 in two of three seeds in some form (60%) | occurred (angle form: 99.8 to 100 on all seeds, both optimizers) |
| P35B.3 | Scrambled arm at or below 30 at both fractions in all seeds (75%) | did not occur (four-operation mean 32.3 at f = 0.10 and 33.6 at f = 0.25, E10 optimizer, seed 73129; max/min up to 62.2) |
| P35B.4 | Stage 1a gives row 1 for add and sub at one fraction or both (20%) | did not occur (row 3 at f = 0.10; row 0 at f = 0.25) |
| P35B.5 | Stage 1a gives row 3 for add and sub at every fraction with room (25%) | occurred (the only fraction with room for add/sub is 0.10, and it is row 3) |
| P35B.6 | Where M generalises at 12 letters, its held40 is at least 60 (55%) | occurred in the one cell where M generalises: max/min at f = 0.10 (a row-0 cell), held12 91.6, held40 90.4 |

## NOT_RUN

- Stage 1a M and X at f = 0.25: no family has room there.
- Stage 1b (seeds 93683, 116089): awaits the coordinator's release.
- Arm G: run by the E35-A worker; this worker ran only its dry hash (check 1).
