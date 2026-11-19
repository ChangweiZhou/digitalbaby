# An independent program of the tissue, and its comparison with the reviewer's (19 November 2026)

**What this is.** The reviewer asked for a program of the tissue written from `THE_TISSUE_AS_RUN_20261119.md` alone (reviewer's branch
`claude/vibrant-cerf-sje726` at 83dc294d), without opening the reviewer's `tissue.c`, `tis.py` or `drive.py`, and then a comparison
with the reviewer's program. Only E63's `birth(seed)` and `stream(seed, n)` (`deps/TISSUE_STEP1/tissue_step1.py`) and the helper's
`readings.py` (`deps/TISSUE_EVENT/readings.py`) were used from other lines. `tissue.c` was opened only after the first comparison, as
the task allows. Every figure below is from one run on this host (Linux x86-64, 4 cores with FMA, gcc 13.3.0, numpy 2.5.3); one seed
except where three are named; nothing here is confirmed by anyone else.

## Files

- `my_tissue.c`: the tissue, step by step in the order of the text's section 2; plain loops over every cell and connection; double
  precision with flush-to-zero and denormals-are-zero set (the text's "numbers below the smallest normal number treated as zero").
  Built with `gcc -O2 -march=x86-64 -fno-fast-math -ffp-contract=off -o my_tissue my_tissue.c` (no fused multiply-add).
- `my_tissue.py`: prepares the birth, the stream and the time constants (`default_rng(seed + 20261119).choice`), runs `my_tissue`, and
  reads the last 4,096 bytes with `readings.read_window`, with distances by breadth-first search along the final connections from the
  lines of the byte values present in the window (as the reviewer's `far.py` computes them).
- `rev_record.py`: runs the reviewer's program through `tis.make` and `tis_run` (as `far.py` and `diag.py` do) and records every cell's
  output at the reading step of every byte (float32, the only form `tis_run` gives). `rev_record_nofma.py`: the same on a copy of the
  reviewer's folder whose only change is `-ffp-contract=off` added to the compile command in `tis.py` and `drive.py`.
- `compare.py`: the first byte and cell at which two such recordings differ, and the largest difference.
- `out/`: the receipts (`*.stats`, `*.json`), the logs, the reviewer's `out/chk.json`, and `far.py`'s printed tables.

## 1. The places where the text is unclear (sent to the reviewer before any run)

- U1, step 8: does z_j mean the output of step k − 1 (section 2's definition) or x_j of step k? **Resolved by the runs:** the reviewer's
  program uses z_j = x(k − 1), the literal reading. With x_j of step k the outputs part from the reviewer's at byte 128, cell 1.
- U2, step 4: no decay of lead and lag when x_i = 0 and F_i = 0 (as written; the reviewer's program agrees).
- U3, step 5: conservation every step, after the arrival; a group with no senders is left alone (agrees).
- U4, step 9(b): a new sender may be an input line; members in `birth()`'s order (agrees).
- U5, U6: ages as k − made at step k; making tested with θ and M after steps 6 and 8; one draw a making (agrees).
- U7: the command names `rule=timing` and `lines_learn=1`, but the text has no change of strength besides conservation, withdrawal and
  making (agrees: with `prune_only=1` the reviewer's program changes no strength by the rule).
- U8: m_i = 0 for a cell with no excitatory sender; the gain uses this step's Q and P (agrees).
- U9: subnormals: the text says "treated as zero"; implemented as the processor's flush-to-zero and denormals-are-zero (as the reviewer's).
- U10: the reading (window and distance) is not in this text; taken from `far.py`: the last 4,096 bytes, distance by the final connections.
- **Found by the comparison, not in the text:** the arithmetic is not reproducible from the text, because the reviewer's program is built
  with `gcc -O3 -march=native`, which on a machine with FMA fuses multiply-adds (rounding once instead of twice).

## 2. N0, seed 73129 (`τ_i = 2`), the reviewer's command

| | Connections (over 2,048 cells) | Made | Withdrawn |
|---|---|---|---|
| Mine, 2,048 bytes | 7,365 | 0 | 8,507 |
| Reviewer's (native build), 2,048 bytes | 7,365 | — | — |
| Mine, 16,384 bytes | 6,638 (3.241 a cell) | 1,240 | 10,474 |
| Reviewer's, native build (`out/chk.json`) | 6,624 (3.234) | 1,219 | 10,467 |
| Reviewer's, built without fused multiply-add | 6,638 | — | — |
| Mine, built with `-O3 -march=native` | 6,650 | 1,242 | 10,464 |
| The text's section 5 | 3.24 a cell | 1,252 | 10,482 |

Outputs at the reading step of every byte, every cell (`compare.py`):

- Mine against the reviewer's native build: equal to the last float32 bit up to byte 2,375. The first difference is at byte 2,376, cell
  1957 (7.5e-9); more cells differ by 1e-8 to 1e-11 from byte 2,497; the first difference above 1e-6 is at byte 2,629, cell 2 (3.7e-6);
  from about byte 3,072 outputs differ by up to 1.0.
- Mine against the reviewer's program built without fused multiply-add (no other change): **equal at every byte of 16,384 for every
  cell**, and the same 6,638 connections.

So the two programs follow the same rule. The difference between them is the reviewer's compile flag, which lets the compiler fuse
multiply-adds; the rounding difference starts after the first makings (byte 2,376; a long mean first falls below 1/4,096 at about byte
2,130) and is amplified by the tissue to a full difference within about 700 bytes. My program built with fused multiply-adds does not
reproduce the reviewer's native build either, since which operations are fused depends on the shape of the code. The text's own
figures (3.24, 1,252, 10,482) match none of these runs.

S(d, L) by the tool over bytes 12,288 to 16,383 (`far.py`; lags 0, 1, 2):

| d | Mine = reviewer's without FMA | Reviewer's native build |
|---|---|---|
| 1 | 0.979, 0.071, 0.053 | 0.979, 0.071, 0.054 |
| 2 | 0.774, 0.057, 0.039 | 0.814, 0.058, 0.040 |
| 4 | 0.632, 0.062, 0.041 | 0.639, 0.056, 0.038 |
| 8 | 0.376, 0.127, 0.043 | 0.385, 0.137, 0.055 |
| 10 | 0.291, 0.144, 0.056 | 0.305, 0.157, 0.055 |

## 3. N256, seed 73129 (`tau=u2-256`)

- Mine: 7,699 connections, 8,630 made, 16,803 withdrawn. The reviewer's built without fused multiply-add: 7,699, and its outputs equal
  mine at every byte of 16,384 for every cell. The reviewer's native build: 7,751; first difference at byte 2,289, cell 532 (9.3e-10);
  first above 1e-6 at byte 3,661, cell 797 (0.0045).
- S(d, 0) at d = 1, 2, 4, 8, 10: mine 0.985, 0.736, 0.357, 0.091, 0.035; the reviewer's native build 0.984, 0.751, 0.374, 0.038, 0.022.

## 4. The nine entries of step 1, N256, from my program (16,384 bytes; bytes 12,288 to 16,383)

| Seed | Connections, made, withdrawn | d = 1: S at lags 0, 1, 2 | d = 2 | d = 4 | Cells at d = 1, 2, 4 |
|---|---|---|---|---|---|
| 73129 | 7,699, 8,630, 16,803 | 0.985, 0.315, 0.147 | 0.736, 0.450, 0.244 | 0.357, 0.344, 0.261 | 112, 188, 183 |
| 93683 | 8,811, 11,579, 18,640 | 0.981, 0.365, 0.203 | 0.731, 0.510, 0.335 | 0.281, 0.357, 0.319 | 117, 183, 180 |
| 116089 | 8,827, 14,530, 21,575 | 0.995, 0.359, 0.192 | 0.712, 0.471, 0.284 | 0.311, 0.365, 0.292 | 121, 201, 193 |

These are the tool's figures from runs of the rule as written, without fused multiply-adds; they are not compared here with the levels
of step 1.

## 5. Which is at fault

- The rule: neither. Every sentence of sections 1, 2 and 3 of the text is followed by both programs, as shown by bit-equal outputs over
  16,384 bytes in N0 and N256 once the compile flag is the same; the one reading of the text that differs (U1, x_j of step k) is the one
  the text's own definition of z_j excludes.
- The arithmetic: the reviewer's build (`drive.py` and `tis.py` compile with `-march=native`, which fuses multiply-adds on this machine).
  Its results are reproducible only with the same source, compiler and machine, not from the text. On this seed rounding alone moves
  S(2, 0) of N0 by 0.04 and S(8, 0) of N256 by 0.05.
- The text's section 5 quotes figures (3.24, 1,252, 10,482) that none of these runs gives.

## 6. Added after the reviewer's answers of 09:05 and 09:27 UTC

The reviewer confirmed the readings U2 to U10 and, for U1, the literal reading (z_j = output of step k − 1), which is the main run
above; the reading with x_j of step k parts from it at byte 128 (section 2).

**The nine entries against both levels** (`levels.py`; N256, 16,384 bytes, the last 4,096 bytes; `out/*_levels.json`). Level (a): 0.1103
at lag 0, c0(L) + 0.02 at lags 1 and 2. Level (b), the stricter one: the upper end of the level's interval plus 0.02 must lie below the
lower end of the score's interval. I took an interval as the value ± 2 standard errors of the tool, and used 0.1103 at lag 0 with no
interval; the definition of the interval is my choice and is stated here. Result: 9 of 9 above (a) and 9 of 9 above (b) on each of seeds
73129, 93683 and 116089. The smallest margin of (b) is at d = 1, lag 2 on seed 73129 (S 0.147 ± 0.008 against c0 0.056).

**The spread between lives that differ only in the last bit** (`spread.py`; N0, seed 73129, 16,384 bytes; the reference run and five runs
in each of which one cell's potential (cells 100, 600, 1100, 1600 and 2000) is changed by 1e-12 before step 0; `out/spread_*.json`):

| d | Reference | Five changed runs | Smallest to largest over all six | Range |
|---|---|---|---|---|
| 1 | 0.979 | 0.978, 0.978, 0.978, 0.999, 0.978 | 0.978 to 0.999 | 0.021 |
| 2 | 0.774 | 0.814, 0.831, 0.831, 0.834, 0.812 | 0.774 to 0.834 | 0.060 |
| 4 | 0.632 | 0.606, 0.595, 0.613, 0.600, 0.668 | 0.595 to 0.668 | 0.073 |
| 8 | 0.376 | 0.421, 0.391, 0.379, 0.375, 0.420 | 0.375 to 0.421 | 0.046 |
| 10 | 0.291 | 0.256, 0.287, 0.265, 0.240, 0.328 | 0.240 to 0.328 | 0.088 |

Connections at the end: 6,638 (reference), 6,615, 6,685, 6,757, 6,664 and 6,639. On this seed and tissue, two settings compared on one
seed by one life each cannot be told apart by S(d, 0) differences smaller than about 0.02 at d = 1, 0.06 at d = 2, 0.07 at d = 4, 0.05 at
d = 8 and 0.09 at d = 10, since lives that differ only in the last bit differ by that much.

## 7. The reviewer's task of 09:35 UTC: the nine entries under the noise, and when a life stops depending on the last bit

All runs with `my_tissue.c` as now committed (seed 73129 unless named; 16,384 bytes; `-ffp-contract=off`). The added options of the
program (a change of one cell's potential before the first step of any byte, making switched off from a byte on, per-byte counts of made
and withdrawn connections, and recording in double precision) leave the unchanged run bit-equal to the earlier one (checked over 512
bytes and, for the references, over 16,384 bytes).

**Task 1** (`task1.py`, `out/task1.json`): six lives a setting (the reference and five with one cell's potential changed by 1e-12 before
step 0; cells 100, 197, 294, 391, 488, or the next ones that are not feedback cells). Smallest to largest S over the six lives:

| Setting | d = 1: lags 0, 1, 2 | d = 2 | d = 4 | Every entry of every life above levels (a) and (b) |
|---|---|---|---|---|
| N256, 73129 | 0.979-0.985, 0.294-0.315, 0.139-0.165 | 0.709-0.752, 0.419-0.458, 0.196-0.252 | 0.300-0.357, 0.324-0.362, 0.217-0.261 | yes |
| N256, 93683 | 0.979-0.990, 0.349-0.368, 0.191-0.218 | 0.662-0.749, 0.457-0.510, 0.296-0.335 | 0.245-0.309, 0.357-0.387, 0.303-0.347 | yes |
| N256, 116089 | 0.981-0.997, 0.343-0.368, 0.176-0.195 | 0.712-0.768, 0.445-0.481, 0.263-0.300 | 0.196-0.311, 0.234-0.365, 0.197-0.292 | yes |
| N64, 73129 | 0.983-0.992, 0.183-0.213, **0.059-0.067** | 0.752-0.781, 0.331-0.353, 0.103-0.160 | 0.441-0.487, 0.333-0.370, 0.151-0.220 | no: S(1,2) below level (a) 0.076 in all six lives |

**Task 2** (`task2.py`, `out/T2_N0.json`, `out/T2_N256.json`): two copies, one cell's potential changed by 1e-9 before the first step of
byte B (at B = 0 cell 100; otherwise the first cell, not a feedback cell, whose output at the reading step of byte B - 1 lies between
0.05 and 0.95; `out/T2_cells.json`). A first try changed cell 100 at every B; it was silent at those bytes, so its change reached no output,
and those runs were discarded. A first try with float32 records was also discarded, because float32 hides differences below about 3e-8.

| B | N0: difference at +64 / +512 bytes; made / withdrawn in B..B+512 | N0: end | N256: difference at +64 / +512; made / withdrawn | N256: end |
|---|---|---|---|---|
| 0 | 3.8e-9 / 1.1e-3; 0 / 8,430 | differ (1.0), connections differ | 1.8e-7 / 4.0e-5; 0 / 7,905 | differ, connections differ |
| 512 | 7e-35 / 1e-56; 0 / 51 | differ, connections differ | 2e-13 / 2e-13; 0 / 40 | differ, connections differ |
| 1,024 | 0 / 0; 0 / 14 | equal, same connections | 0 / 5e-12; 0 / 1 | equal, same |
| 2,048 | 0 / 0; 935 / 1,544 | equal, same | 3e-13 / 1e-12; 1,427 / 2,409 | differ, connections differ |
| 4,096 | 0 / 0; 16 / 78 | equal, same | 6e-12 / 5e-13; 337 / 169 | differ, connections differ |
| 8,192 | 0 / 0; 8 / 7 | equal, same | 5e-12 / 3e-13; 306 / 272 | 4.8e-11, same |
| 12,288 | 0 / 0; 2 / 6 | equal, same | 6e-11 / 4e-12; 205 / 210 | 1e-10, same |
| 15,872 | 4e-14 / 9e-16; 2 / 5 | 9e-16, same | 1e-16 / 0; 68 / 80 | 0, same |

S(d, 0) at d = 1, 2, 4, 8, 10 of the copies that parted, against the reference (N0: 0.979, 0.774, 0.632, 0.376, 0.291; N256: 0.985,
0.736, 0.357, 0.091, 0.035): N0 B = 0: 0.978, 0.814, 0.627, 0.387, 0.243; B = 512: 0.979, 0.789, 0.627, 0.376, 0.300. N256 B = 0:
0.979, 0.712, 0.308, 0.070, 0.033; B = 512: 0.986, 0.727, 0.382, 0.061, 0.046; B = 2,048: 0.984, 0.691, 0.352, 0.045, 0.037; B = 4,096:
0.986, 0.704, 0.379, 0.064, 0.041. Every difference is within the spread between lives of section 6 except S(8,0) of N256 at B = 2,048
(0.046 below the reference), which is at the size of that spread.

**Task 3** (`out/T3_N0.json`, `out/T3_N256.json`): the same with no making from byte 4,096 on (in both copies; the reference with making off
made 1,048 (N0) and 2,149 (N256) connections in all). Perturbed at B = 4,096, 8,192, 12,288, 15,872: in N0 and in N256 the connections at
the end are the same in every case, and the largest output difference at the end is at most 8e-14. In N256 the perturbation at B = 4,096,
which gives a different life when making goes on, leaves no trace when making is off. Switching making off changes the life itself:
S(d, 0) at d = 1, 2, 4, 8, 10 is N0 0.979, 0.809, 0.645, 0.376, 0.321 and N256 0.986, 0.761, 0.358, 0.042, 0.024.

**What these runs show, on one seed.** A difference in the last bit reaches the connections only while connections are being withdrawn
or made in numbers: in the first ~1,000 bytes in both tissues (the first mass withdrawal) and, in N256, until at least byte 4,096; from
byte 8,192 on in N256 and from byte 1,024 on in N0, and after making is switched off, it does not change any connection, and the outputs
differ at the end by at most 1e-10. In between, the outputs may differ by only 1e-12 to 1e-13 while the slow quantities carry the
difference to a later withdrawal or making. Where two copies part, they hold different connections, and their readings differ by amounts
within the spread between lives of section 6 (one entry at its size).

## 8. The long-axon tissue L8-256 (section 3a; the reviewer's task of 10:23 UTC)

Command compared: `far.py L8_256_s73129 rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1
a_make=1/16384 make_period=1024 start_w=1/64 long=8-1/2 tau=u2-256 seed=73129`, on the reviewer's branch at 33c2809e (its build now
with `-ffp-contract=off`). My option is `my_tissue.long_axons` (front end) and the arguments `start_w` and `a_make` of `my_tissue.c`.

**Unclear places** (sent before any run): L1 "excitatory" read as σ = +1 in `birth()`; L2 "within 1/2" as Euclidean distance squared
at most 1/4, no wrap; L3 the enlarged list is `birth()`'s list followed by the long cells, and the 8 starting connections replace
`birth()`'s draw, in `.choice`'s order; L4 one generator, first the long draws, then the lists and draws cell by cell (feedback cells:
no draw); L5 `start_w` = 1/64 for both starting and made connections, `a_make` = 1/16384 in the making test; L6 making may choose a long
cell; L7 τ drawn as N256; L8 whether "near the output lines" uses the enlarged lists or `birth()`'s. **Resolved by the comparison:**
L1 to L7 as I read them; L8 with the enlarged lists.

**Part 1, seed 73129.** 156 long cells. Outputs of every cell at the reading step of every byte equal to the reviewer's to the last
float32 bit over 16,384 bytes (my double record cast to float32 with float32 subnormals set to zero, as the reviewer's program records);
5,113 connections in both; mine 898 made, 11,657 withdrawn.

**Part 2.** "Near the output lines" (drivers and the members of a driver's neighbourhood, feedback cells excepted) with the enlarged
lists: 577 cells, S at lags 0 to 4 = 0.174, 0.184, 0.143, 0.157, 0.128, the reviewer's figures. With `birth()`'s lists it would be
495 cells and 0.167, 0.167, 0.149, 0.151, 0.126.

**L5 corrected by the reviewer (10:24 UTC):** the 8 starting connections keep 1/8; only made connections take `start_w` = 1/64. On
seed 73129 this changes nothing (`out/L8_73129_sw0_8th.stats`: every output, connection and strength identical to the run with 1/64,
and equal to the reviewer's), because the input line of byte 0 is a starting sender of no cell on that seed, so nothing arrives at step
0, and after step 0 conservation rescales every cell's strengths to 3/4 and 1/4 whatever they started at. On seeds 93683 and 116089 the
line of byte 0 is a starting sender of 3 and 5 cells, so the starting strength matters there; their six lives were run again with 1/8
(`jobs_L8b.txt`; the earlier lives with 1/64 are not committed and their figures are withdrawn). Parts 1, 2 and 4 are on seed 73129 and
stand.

**Part 3** (`L8agg.py`, `out/L8_part3.json`, `out/L8_part3.txt`): six lives a seed (reference + five with one cell's potential changed by
1e-12 before step 0). Smallest to largest S. "Near" is given for both groups: enlarged lists (the reviewer's reading; 577, 595 and 569
cells on seeds 73129, 93683 and 116089) and `birth()`'s 1/8 lists (495, 519 and 495 cells).

| Seed | d = 1: lags 0, 1, 2 | d = 2 | d = 4 | Near, enlarged: lags 0, 1, 2 | Near, 1/8 lists: lags 0, 1, 2 |
|---|---|---|---|---|---|
| 73129 | 0.913-0.937, 0.331-0.366, 0.186-0.207 | 0.495-0.554, 0.329-0.362, 0.208-0.230 | 0.210-0.241, 0.179-0.202, 0.148-0.165 | 0.174-0.216, 0.177-0.211, 0.143-0.173 | 0.167-0.193, 0.167-0.184, 0.146-0.158 |
| 93683 | 0.814-0.863, 0.292-0.310, 0.170-0.193 | 0.346-0.398, 0.223-0.271, 0.140-0.187 | 0.152-0.201, 0.161-0.199, 0.141-0.167 | **0.110**-0.147, 0.125-0.168, 0.117-0.144 | **0.106**-0.148, 0.109-0.158, 0.114-0.141 |
| 116089 | 0.937-0.960, 0.337-0.362, 0.191-0.207 | 0.517-0.627, 0.335-0.389, 0.191-0.221 | 0.230-0.315, 0.226-0.244, 0.160-0.196 | 0.211-0.233, 0.186-0.237, 0.139-0.178 | 0.149-0.210, 0.163-0.216, 0.131-0.162 |

Every one of the nine entries is above levels (a) and (b) in every life on all three seeds. Near the output lines, on seeds 73129 and
116089 every life is above (a) and (b) at lags 0, 1 and 2 in both groups. On seed 93683 the present byte near the output lines is below
0.1103 in one life of six in each group (0.110 enlarged, 0.106 with the 1/8 lists) and above (a) but not (b) in most of the other
lives; per life (reference first): enlarged 0.135, 0.147, 0.113, 0.133, 0.141, 0.110; 1/8 lists 0.127, 0.148, 0.138, 0.135, 0.106,
0.125. The reference life gives the reviewer's 0.135. Lags 1 and 2 near the output lines on 93683 are above (a) in every life in both
groups; (b) fails at lag 1 in one life with the 1/8 lists.

**Part 4** (`task2L8.py`, `out/L8T2_L8.json`): two copies, one cell's potential changed by 1e-9 before the first step of byte B, seed
73129; largest output difference at the reading step 64 and 512 bytes later; made and withdrawn in the reference in B..B+512.

| B | Cell | +64 / +512 | Made / withdrawn in B..B+512 | End of the life |
|---|---|---|---|---|
| 0 | 100 | 5.2e-6 / 0.54 | 0 / 10,784 | outputs differ by 0.74; connections differ; S(d,0) d = 1, 2, 4, 8: 0.923, 0.523, 0.211, 0.096 (reference 0.928, 0.537, 0.229, 0.076) |
| 1,024 | 29 | 1.4e-13 / 2.9e-13 | 0 / 7 | 1.2e-13; same connections |
| 4,096 | 0 | 1.3e-13 / 0 | 22 / 12 | 1.6e-15; same |
| 8,192 | 10 | 7.2e-14 / 8.0e-15 | 21 / 13 | 3.3e-15; same |
| 12,288 | 0 | 3.7e-13 / 2.2e-16 | 17 / 20 | 3.7e-68; same |

In this tissue no cell is at distance 10 at the end. On this seed the dependence on the last bit is confined to the first mass
withdrawal (10,784 connections withdrawn in bytes 0 to 512); from byte 1,024 on a change of 1e-9 leaves the connections unchanged.
(A first evaluation of part 4 took the reference as starting at byte 4,096 and gave differences of 1.0 at +64 and +512; it was an
indexing error of the analysis script, corrected before any figure was sent.)

## 9. Long lives on the three weak seeds; seed 112290 against 73129; L8-128 under the last-bit noise (task of 10:39 UTC)

All with starting connections at 1/8 and made ones at 1/64 (the reviewer's L5), `a_make` 1/16384, long axons as in section 8; L8-128
draws τ from {2, ..., 128} with the same rule. A new option of `my_tissue.c` records windows of 4,096 bytes and writes the connections at
the end of each window (`<prefix>.conn_<end>`), so that each window's distances are those of its own connections; the unchanged run is
still bit-equal to the earlier one. `LLread.py` reads each window; `out/LL_summary.txt` is the table below in full; `(A)` below level
(a), `(b)` above (a) but not (b), no mark: above both.

**Part 1: lags 0, 1, 2 near the output lines, enlarged group | `birth()`'s 1/8 group, in windows ending at 16,384, 32,768, 49,152,
65,536.** The nine entries pass both levels in every window of every life except S(4,0) of L8-256 on 146821 at 65,536 (0.131, (b)).

```
LL_N256_465973 16384 nine: all pass | near enl 0.100(A) 0.137 0.100 | near 1/8 0.107(A) 0.120 0.105
LL_N256_465973 32768 nine: all pass | near enl 0.133(b) 0.157 0.129 | near 1/8 0.128(b) 0.163 0.130
LL_N256_465973 49152 nine: all pass | near enl 0.130(b) 0.168 0.141 | near 1/8 0.136(b) 0.166 0.144
LL_N256_465973 65536 nine: all pass | near enl 0.134(b) 0.179 0.141 | near 1/8 0.139(b) 0.177 0.150
LL_N256_146821 16384 nine: all pass | near enl 0.102(A) 0.121 0.120 | near 1/8 0.092(A) 0.128 0.113
LL_N256_146821 32768 nine: all pass | near enl 0.103(A) 0.113 0.112 | near 1/8 0.099(A) 0.140 0.114
LL_N256_146821 49152 nine: all pass | near enl 0.113(b) 0.110(b) 0.113 | near 1/8 0.098(A) 0.137 0.104
LL_N256_146821 65536 nine: fail d4 L0 0.131(b) | near enl 0.100(A) 0.106 0.108 | near 1/8 0.105(A) 0.135 0.096
LL_N256_112290 16384 nine: all pass | near enl 0.071(A) 0.115 0.136 | near 1/8 0.091(A) 0.112(b) 0.122
LL_N256_112290 32768 nine: all pass | near enl 0.093(A) 0.161 0.163 | near 1/8 0.110(A) 0.162 0.151
LL_N256_112290 49152 nine: all pass | near enl 0.110(A) 0.173 0.175 | near 1/8 0.120(b) 0.166 0.155
LL_N256_112290 65536 nine: all pass | near enl 0.082(A) 0.151 0.166 | near 1/8 0.096(A) 0.152 0.139
LL_N128_465973 16384 nine: all pass | near enl 0.129(b) 0.160 0.128 | near 1/8 0.128(b) 0.129 0.093
LL_N128_465973 32768 nine: all pass | near enl 0.134(b) 0.164 0.124 | near 1/8 0.137(b) 0.143 0.105
LL_N128_465973 49152 nine: all pass | near enl 0.124(b) 0.151 0.125 | near 1/8 0.123(b) 0.133 0.101
LL_N128_465973 65536 nine: all pass | near enl 0.126(b) 0.156 0.128 | near 1/8 0.135(b) 0.138 0.111
LL_N128_146821 16384 nine: all pass | near enl 0.136(b) 0.160 0.133 | near 1/8 0.129(b) 0.160 0.142
LL_N128_146821 32768 nine: all pass | near enl 0.111(b) 0.145 0.135 | near 1/8 0.123(b) 0.168 0.140
LL_N128_146821 49152 nine: all pass | near enl 0.105(A) 0.141 0.123 | near 1/8 0.117(b) 0.154 0.135
LL_N128_146821 65536 nine: all pass | near enl 0.107(A) 0.137 0.126 | near 1/8 0.118(b) 0.158 0.135
LL_N128_112290 16384 nine: all pass | near enl 0.109(A) 0.118 0.111 | near 1/8 0.124(b) 0.145 0.120
LL_N128_112290 32768 nine: all pass | near enl 0.130(b) 0.160 0.140 | near 1/8 0.119(b) 0.152 0.136
LL_N128_112290 49152 nine: all pass | near enl 0.138(b) 0.162 0.143 | near 1/8 0.116(b) 0.152 0.140
LL_N128_112290 65536 nine: all pass | near enl 0.127(b) 0.174 0.140 | near 1/8 0.117(b) 0.154 0.134
```

At 16,384 bytes my program gives the reviewer's figures for the present byte near the output lines in L8-256 (0.100 on 465973, 0.102 on
146821, 0.071 on 112290). With a longer life, in L8-256, the present byte near the output lines passes level (a) from 32,768 on and
stays on 465973, and does not pass on 146821 or 112290 (in the enlarged group). In L8-128 it is above (a) in every window from 32,768 on
for 465973 and 112290, and on 146821 only in the 1/8 group (the enlarged group falls to 0.105 and 0.107 at 49,152 and 65,536). Level
(b) is not reached for the present byte near the output lines in any window of any of these six lives.

**Part 2: seed 112290 against 73129** (L8-256, window ending at 16,384; `describe` in `out/LL_N256_*_LL.json`).

| | 73129 | 112290 |
|---|---|---|
| Long cells | 156 | 147 |
| Drivers' connections; share from long-axon cells | 435; 35.6% | 418; 32.1% |
| Cells near the output lines (enlarged) | 577 | 553 |
| ... by distance from the input lines (0 = not reached) | 0: 3, 2: 6, 3: 54, 4: 175, 5: 183, 6: 109, 7: 36, 8: 9, 9: 2 | 0: 12, 2: 2, 3: 16, 4: 71, 5: 155, 6: 156, 7: 94, 8: 36, 9: 10, 10: 1 |
| ... within 4 connections | 235 | 89 |
| Byte values preferred by at least one near cell that is not silent (preferred: largest mean output) | 23 of 46 | 38 of 46 |
| Share of near cells with largest output above 1/2 | 0.955 | 0.901 |

The 1/8 group gives the same picture (73129: 176 of 495 within 4 connections; 112290: 62 of 493). On 112290 the cells near the output
lines lie one to two connections farther from the input lines than on 73129.

**Part 3: L8-128, six lives a seed** (`L128agg.py`, `out/L128_part3.json`, `out/L128_part3.txt`). Smallest to largest S:

| Seed | d = 1: lags 0, 1, 2 | d = 2 | d = 4 | Near, enlarged: lags 0, 1, 2 | Near, 1/8 lists: lags 0, 1, 2 |
|---|---|---|---|---|---|
| 73129 | 0.931-0.963, 0.249-0.275, 0.129-0.165 | 0.555-0.651, 0.249-0.303, 0.141-0.170 | 0.160-0.226, 0.154-0.190, 0.102-0.148 | 0.152-0.190, 0.147-0.169, 0.090-0.135 | 0.125-0.178 (b fails in some lives), 0.146-0.162, 0.092-0.128 |
| 93683 | 0.821-0.859, 0.262-0.312, 0.156-0.179 | 0.446-0.540, 0.278-0.337, 0.178-0.220 | 0.227-0.254, 0.207-0.249, 0.154-0.200 | 0.156-0.196, 0.175-0.225, 0.162-0.199 | 0.128-0.190 (b fails in some lives), 0.169-0.216, 0.129-0.190 |
| 116089 | 0.933-0.956, 0.321-0.335, 0.141-0.169 | 0.578-0.626, 0.343-0.367, 0.158-0.187 | 0.276-0.313, 0.198-0.246, 0.139-0.168 | 0.193-0.237, 0.198-0.233, 0.139-0.160 | 0.188-0.253, 0.179-0.224, 0.123-0.150 |

Every one of the nine entries and lags 0, 1, 2 near the output lines is above level (a) in every life on all three seeds, in both groups;
level (b) fails only for the present byte near the output lines in the 1/8 group, in some lives of seeds 73129 and 93683. Compared with
L8-256 (section 8), the present byte near the output lines on 93683 rises from 0.106-0.148 to 0.128-0.190 (1/8 group) and from
0.110-0.147 to 0.156-0.196 (enlarged), and no life falls below 0.1103.

## 10. LF-256: the long-axon cells with time constant 2 (task of 11:10 UTC)

`my_tissue.prepare(..., long=True, longfast=True)`: τ drawn as in N256, then set to 2 for the long-axon cells; nothing else changes
(starting connections 1/8, made 1/64, `a_make` 1/16384).

**Part 1, seed 73129.** Outputs equal to the reviewer's program (`tis.py` with `longfast=1`, branch head of 11:10 UTC) to the last float32
bit at every cell over bytes 12,288 to 16,383; 7,302 connections in both (916 made, 9,486 withdrawn). Near the output lines (enlarged)
0.201, 0.166, 0.124, 0.087, 0.072 at lags 0 to 4, the figures of `far/LF_256_s73129.json`; with the 1/8 lists 0.227, 0.163, 0.105.

**Part 2: six lives a seed** (`LFagg.py`, `out/LF_part2.json`, `out/LF_part2.txt`; reference life and five with one cell's potential
changed by 1e-12 before step 0). Smallest to largest S; the number of lives below a level is given where it is not zero.

| Seed | d = 1: lags 0, 1, 2 | d = 2 | d = 4 | Near, enlarged: lags 0, 1, 2 | Near, 1/8 lists: lags 0, 1, 2 |
|---|---|---|---|---|---|
| 73129 | 0.935-0.948, 0.284-0.311, 0.140-0.153 | 0.440-0.513, 0.201-0.232, 0.136-0.149 | 0.184-0.217, 0.139-0.182, 0.115-0.135 | 0.139-0.209, 0.121-0.173, 0.098-0.124 | 0.136-0.227, 0.111-0.163, 0.087-0.119 (below a 0, below b 1) |
| 93683 | 0.820-0.838, 0.256-0.295, 0.128-0.148 | 0.373-0.456, 0.230-0.300, 0.151-0.200 | 0.204-0.234, 0.170-0.215, 0.136-0.165 | 0.175-0.236, 0.151-0.205, 0.121-0.167 | 0.181-0.228, 0.154-0.202, 0.118-0.162 |
| 116089 | 0.938-0.956, 0.301-0.319, 0.159-0.168 | 0.519-0.617, 0.237-0.298, 0.139-0.162 | 0.234-0.298, 0.153-0.191, 0.099-0.120 | 0.223-0.283, 0.149-0.185, 0.089-0.109 | 0.203-0.234, 0.144-0.170, 0.084-0.107 (below a 0, below b 2) |
| 146821 | 0.890-0.913, 0.290-0.310, 0.143-0.155 | 0.395-0.518, 0.222-0.271, 0.118-0.162 | 0.153-0.183, 0.100-0.145 (below a 0, below b 1), 0.086-0.101 (below a 0, below b 1) | 0.111-0.157 (below a 0, below b 3), 0.093-0.112 (below a 0, below b 2), 0.079-0.093 (below a 0, below b 5) | 0.129-0.155 (below a 0, below b 3), 0.103-0.118 (below a 0, below b 2), 0.079-0.088 (below a 0, below b 6) |
| 112290 | 0.911-0.930, 0.360-0.379, 0.185-0.211 | 0.436-0.491, 0.271-0.307, 0.145-0.185 | 0.160-0.191, 0.140-0.164, 0.096-0.133 | 0.134-0.162 (below a 0, below b 2), 0.120-0.160, 0.113-0.138 | 0.135-0.165 (below a 0, below b 1), 0.126-0.161, 0.111-0.147 |
| 224268 | 0.872-0.935, 0.256-0.283, 0.112-0.131 | 0.432-0.502, 0.213-0.261, 0.118-0.145 | 0.159-0.196, 0.135-0.142, 0.093-0.100 | 0.126-0.161 (below a 0, below b 2), 0.102-0.127 (below a 0, below b 3), 0.074-0.094 (below a 0, below b 4) | 0.110-0.168 (below a 2, below b 2), 0.101-0.133 (below a 0, below b 2), 0.072-0.090 (below a 2, below b 5) |

No entry is below level (a) in any life on any of the six seeds, except near the output lines with the 1/8 lists on seed 224268: the
present byte in 2 lives of 6 (smallest 0.110) and the byte two back in 2 of 6. With the enlarged group every entry is above (a) in every
life on all six seeds. Level (b) fails in several lives near the output lines on 146821 and 224268, and in single lives elsewhere.
Against L8-256 (sections 8 and 9), the present byte near the output lines on the weak seeds rises: 112290 from 0.071 (one life, 16,384
bytes) to 0.134-0.162 over six lives, 93683 from 0.110-0.147 to 0.175-0.236.

**Part 3: seed 112290, L8-256 against LF-256** (`diag3.py`, `out/diag3_L8_112290.json`, `out/diag3_LF_112290.json`; the reference
life, window ending at 16,384).

| | L8-256 | LF-256 |
|---|---|---|
| Near cells (enlarged, 553) by distance from the input lines (0 = not reached) | 0: 12, 2: 2, 3: 16, 4: 71, 5: 155, 6: 156, 7: 94, 8: 36, 9: 10, 10: 1 | 2: 3, 3: 27, 4: 125, 5: 210, 6: 137, 7: 44, 8: 7 |
| ... within 4 connections | 89 | 155 |
| Long-axon cells (147) by distance | 0: 1, 1: 6, 2: 25, 3: 33, 4: 31, 5: 33, 6: 13, 7: 3, 8: 1, 9: 1 | 1: 6, 2: 23, 3: 40, 4: 41, 5: 28, 6: 6, 7: 2, 8: 1 |
| Share of the drivers' connections from long-axon cells | 32.1% | 41.0% |
| Share of all near cells' connections from long-axon cells | 25.3% | 32.8% |
| S at lags 0 / 1 from the long cells near the output lines (79) | 0.120 / 0.171 | 0.151 / 0.096 |
| ... from the other near cells (474) | 0.089 / 0.115 | 0.139 / 0.140 |
| ... from all long cells (147) | 0.296 / 0.238 | 0.212 / 0.094 |
| ... from all other cells (the tool's sample of 128 of 1,837) | 0.325 / 0.267 | 0.306 / 0.196 |

With the long-axon cells fast, the cells near the output lines lie about one connection closer to the input lines, a larger share of
their connections comes from long-axon cells, and the present byte rises among both the long cells and the other cells near the output
lines; the long cells themselves carry the byte one back much less (0.238 to 0.094 over all long cells).

## 11. The output end written again from E55's worker's texts (task of 11:32 UTC)

**Part 1.** `own_end.py` was written from README_YESNO.md and README_REACH3.md alone, on the outputs of `my_tissue.c` (L8-256). The
places the text leaves unclear were listed before measure.py was read (O1-O8, message of 11:32 UTC); measure.py takes these readings:
O1 the driver's inputs are the set-(a) cells in its enlarged neighbourhood (members after the long cells), the driver included, input
lines excluded (577 cells near the output lines on seed 73129; 59-178 inputs a driver, median 121); O2 m is updated first, s2 with the
new m, and the input is standardised after this byte's update, m = s2 = 0 at the start, r = 1/min(t+1, 1024); O3 measure.py does not
clip (clipping at 8 is in diag_std.py only; `clip8` here is my addition, with its own generator `default_rng(seed + 1)`); O4 rule D's u
includes the constant 1, a is computed before the update, all 256 units learn; O5 rules A and B step (1/4)/(inputs + 1), rbar is updated
after the step at rate 1/64, the eps draws come from `default_rng(seed)` in the order of the configurations; O6-O8 as listed.

**Part 2.** L8-256, 65,536 bytes (`python3 own_end.py out/OE_L8_<seed>.rec <seed> 65536 L8_<seed>`, `out/own_end_L8_<seed>.json`). On
seeds 73129 and 93683 my figures equal E55's worker's (`e55_L8-256_s<seed>.json`) in every window of 4,096 of every rule (float
equality), as do the first windows reaching each level and the levels themselves. Last quarter, in per cent (A eps 0 / A eps 1/16 /
B / D eta 1/64 / D eta 1/16):

| seed | run | raw | clip8 | ref_1 | ref_2 | end mark |
|---|---|---|---|---|---|---|
| 73129 | 11.57 / 11.33 / 12.02 / 17.02 / 15.52 | 8.94 / 8.64 / 8.43 / 16.85 / 15.57 | 12.54 / 11.82 / 13.90 / 17.00 / 15.51 | 10.55 | 12.34 | 6.21 |
| 93683 | 2.47 / 6.54 / 4.71 / 9.26 / 7.20 | 5.05 / 1.78 / 6.16 / 8.96 / 7.38 | 5.94 / 5.65 / 7.20 / 9.27 / 7.25 | 10.62 | 12.14 | 6.29 |
| 116089 | 9.45 / 9.01 / 9.90 / 16.17 / 15.29 | 7.71 / 7.27 / 6.62 / 16.15 / 15.73 | 9.22 / 8.96 / 11.05 / 16.17 / 15.28 | 10.61 | 12.20 | 6.24 |

**Part 3.** LF-256 (`out/own_end_LF_<seed>.json`), last quarter, same order:

| seed | run | raw | clip8 |
|---|---|---|---|
| 73129 | 9.38 / 6.87 / 10.34 / 13.56 / 11.31 | 7.10 / 6.71 / 7.14 / 13.90 / 11.49 | 8.16 / 8.17 / 7.53 / 13.56 / 11.33 |
| 93683 | 4.08 / 5.17 / 6.96 / 11.17 / 8.48 | 2.98 / 3.56 / 5.90 / 9.64 / 7.67 | 3.02 / 4.25 / 7.09 / 11.16 / 8.48 |
| 116089 | 3.21 / 4.62 / 5.64 / 8.29 / 6.88 | 3.04 / 2.43 / 5.98 / 7.71 / 6.48 | 3.47 / 3.28 / 3.82 / 8.29 / 6.88 |

On the output end LF-256 is below L8-256 on 73129 and 116089 for every rule; on 93683 D rises (9.26 to 11.17), the yes/no rules stay
below ref_1.

**Part 4. Why seed 93683 does not learn (L8-256).** `python3 why93683.py out/OE_L8_<seed>.rec <seed> L8_<seed>` (`out/why_L8_<seed>.json`):
a ridge reader for each byte value on its own driver's inputs, fitted on the first three quarters, scored on the last; against it the
same reader on a one-hot of the present byte (oh1) and of the present and previous byte (oh2).

| | own inputs | oh1 | oh2 | tool S, present byte, on a driver's inputs (mean) | ... previous byte | predecessors of v covered by the inputs' preferred bytes |
|---|---|---|---|---|---|---|
| 73129 | 17.19 | 10.59 | 12.96 | 0.160 | 0.176 | 0.319 |
| 93683 | 10.52 | 10.35 | 12.45 | 0.147 | 0.157 | 0.198 |

On 93683 the drivers' own inputs give no more than the present byte alone, and less than the present and previous byte; on 73129 they
give 4.2 points more than two bytes. The tool reads the present and previous byte on the drivers' inputs at about the same level on both
seeds. The difference lies in the bytes that close and open an item (recall under own / oh2; count in the scored quarter):

| next byte | 73129 | 93683 |
|---|---|---|
| `=` | 0.88 / 0.55 (1,017) | 0.48 / 0.52 (1,030) |
| end mark | 1.00 / 0.48 (1,017) | 0.71 / 0.48 (1,030) |
| `q` | 0.49 / 0.09 | 0.00 / 0.00 |
| `>` | 0.38 / 0.00 | 0.00 / 0.00 |
| `+` | 0.25 / 0.00 | 0.00 / 0.00 |
| `w` | 0.18 / 0.10 | 0.00 / 0.00 |
| `p` | 0.09 / 0.00 | 0.00 / 0.66 |

`python3 why_eq.py <seed>` (`out/why_eq_L8_<seed>.json`): a ridge on the inputs of the drivers of the end mark and of `=`, fitted to
"the byte L back is `=`"; area under the ROC curve on the last quarter, L = 0, 1, 2, 3:

| | driver of the end mark | driver of `=` |
|---|---|---|
| 73129 | 1.000, 1.000, 0.997, 1.000 | 0.848, 1.000, 0.994, 1.000 |
| 93683 | 0.640, 0.673, 0.696, 0.725 | 0.678, 0.683, 0.717, 0.739 |

So what is absent from the neighbourhoods on 93683 is not the present byte or the bytes one and two back but a lasting mark of where the
stream is within an item: on 73129 the inputs of these two drivers tell for at least three bytes that `=` has passed, on 93683 they
barely do. This is a fitted reader on the inputs, a measure of what is there, not of what any learning rule finds.

## 12. Tissue A4: every excitatory cell long, reach 1/4 (task of 12:13 UTC)

A4 is the long-axon option of section 8 with K = 1 and R = 1/4 (`my_tissue.prepare(..., long=True, frac=1.0, reach=0.25,
ltag="_A4")`; the draw `.random() < 1` is still made for each eligible cell, in order), tau = u2-256. Long cells 1,291 / 1,331 / 1,323
on seeds 93683 / 73129 / 116089; cells near the output lines (drivers and their enlarged members, feedback cells excluded) 708 / 678 /
682, as in the reviewer's perbyte files (708, 678). Lives of 65,536 bytes recorded at every byte (`jobs_A4.txt`, about 4 minutes each).

**Part 1. The output end** (`python3 own_end.py out/OE_A4_<seed>.rec <seed> 65536 A4_<seed> A4`, `out/own_end_A4_<seed>.json`). Last
quarter in per cent, A eps 0 / A eps 1/16 / B / D eta 1/64 / D eta 1/16:

| seed | run | raw | clip8 | ref_1 | ref_2 | end mark |
|---|---|---|---|---|---|---|
| 93683 | 6.35 / 4.84 / 6.57 / 10.64 / 9.23 | 4.50 / 6.50 / 5.67 / 10.11 / 8.62 | 5.18 / 5.25 / 6.58 / 10.64 / 9.26 | 10.62 | 12.14 | 6.29 |
| 73129 | 9.81 / 9.83 / 10.07 / 13.80 / 12.82 | 8.82 / 8.04 / 6.81 / 13.78 / 13.00 | 10.07 / 9.29 / 10.01 / 13.81 / 12.82 | 10.55 | 12.34 | 6.21 |
| 116089 | 9.48 / 7.84 / 9.14 / 15.31 / 13.81 | 7.39 / 6.65 / 8.59 / 15.33 / 13.81 | 7.92 / 8.41 / 9.39 / 15.31 / 13.81 | 10.61 | 12.20 | 6.24 |

First windows reaching the levels are in the json files (`first_window_reaching`). Against L8-256 (section 11) A4 is lower on 73129
(run B 12.02 to 10.07, D 17.02 to 13.80) and 116089 (run B 9.90 to 9.14, D 16.17 to 15.31) and higher on 93683 (run B 4.71 to 6.57, D 9.26
to 10.64). No yes/no rule reaches ref_1 in the last quarter on any of the three seeds.

**Part 2. The linear reader byte value by byte value** (`perbyte_own.py`, written from the head comment of the reviewer's perbyte3.py).
Unclear places sent before reading the code (P1-P6); perbyte3.py takes: inputs centred and divided by their deviation over the fitting
half, cells of deviation <= 1e-9 dropped, clipped at 8; a constant input under the ridge, ridge 0.01 trace/size including the constant;
argmax over the values seen in the fitting half; "digits" the mean of the per-digit shares; no floor. Read uncentred without a constant,
L8-256 seed 93683 gives 0.158 / 0.205 / 0.184 / 0.171 instead of 0.166 / 0.207 / 0.191 / 0.176. With the readings above, my figures at
16,384 bytes equal the reviewer's `A4_256_s{93683,73129}_linear.json` (near output, linear) as floats in every entry of overall, end, eq
and digits at the four lags (`out/pb_A4_<seed>_ref_16384.json`).

**Part 3. Six lives a seed** (`pb_lives.py`, `out/pb_A4_lives.json`, `out/pb_A4_ranges.txt`): the reference life and lives with
1e-12 added to one cell's potential at byte 0 (cells 100, 197, 294, 391, 488). On seed 73129 the life with cell 294 is bit-identical to
the reference at 8,192-16,384 (largest difference 0), so cell 1024 (the most active cell in the first bytes) replaces it; the lives with
cells 100 and 488 differ (largest output difference 8e-4 and 0.22 at 8,192-16,384; for cell 100 5e-4 at 57,344-65,536, the same 4,425
connections) but give readings equal to the reference's at both lengths. On 73129 the six lives therefore give four distinct readings.
Range over the six lives, near the output lines, bytes 0 / 1 / 2 / 3 back:

| seed, bytes | overall | end mark | equals sign | digits |
|---|---|---|---|---|
| 93683, 16,384 | 0.275-0.339 / 0.301-0.374 / 0.268-0.336 / 0.204-0.290 | 0.660-0.873 / 0.726-0.873 / 0.649-0.803 / 0.247-0.761 | 0.442-0.942 / 0.636-0.899 / 0.494-0.842 / 0.633-0.849 | 0.345-0.474 / 0.398-0.498 / 0.316-0.437 / 0.232-0.352 |
| 93683, 65,536 | 0.310-0.384 / 0.337-0.407 / 0.324-0.371 / 0.280-0.320 | 0.681-0.950 / 0.719-0.938 / 0.731-0.946 / 0.735-0.938 | 0.642-0.973 / 0.746-0.977 / 0.712-0.950 / 0.604-0.908 | 0.374-0.526 / 0.450-0.535 / 0.365-0.493 / 0.311-0.400 |
| 73129, 16,384 | 0.334-0.365 / 0.370-0.415 / 0.328-0.376 / 0.301-0.326 | 0.938-1.000 / 0.961-1.000 / 0.895-0.996 / 0.785-0.961 | 0.945-1.000 / 0.977-1.000 / 0.965-1.000 / 0.895-0.996 | 0.228-0.290 / 0.238-0.320 / 0.213-0.302 / 0.198-0.282 |
| 73129, 65,536 | 0.354-0.424 / 0.399-0.473 / 0.356-0.434 / 0.326-0.389 | 0.988-1.000 / 0.957-1.000 / 0.949-1.000 / 0.946-0.984 | 0.957-1.000 / 0.984-1.000 / 0.992-1.000 / 0.961-1.000 | 0.256-0.356 / 0.253-0.358 / 0.235-0.353 / 0.204-0.316 |

Seed 116089, reference life only: 16,384 bytes overall 0.354 / 0.343 / 0.330 / 0.297, end 0.936 / 0.939 / 0.917 / 0.765, eq 0.973 /
0.955 / 0.928 / 0.909, digits 0.280 / 0.278 / 0.298 / 0.272; 65,536 bytes 0.339 / 0.321 / 0.328 / 0.310.

Between lives of one seed the overall share moves by 0.03-0.09 and the end mark and equals sign on seed 93683 by up to 0.5 at one lag;
the reviewer's reference life of seed 93683 is the lowest of the six at 16,384 at lags 0, 2 and 3. The ranges of the two seeds overlap
in the overall share; on 73129 the end mark and equals sign stay at 0.79-1.00 in every life, on 93683 they do not.

## 13. Rule Y: the units that did not emit step toward the input on "no" (task of 12:52 UTC)

**Readings.** YESNO/ on claude/e63-yesno has no README; the unclear places Y1-Y8 were sent before yn.py was read. yn.py's head comment
and code take: the emitter's step st = s0 (1/4)/n, n the number of active inputs (one-hots and the constant), divided by
sqrt(1 + m_e/16) under decay s (m_e the unit's earlier emissions); the emitter moves by (ok - r_e) st U, r_e its own share of yes
(rate 1/64, updated after the step, only when it emits); on "no" all 255 other units add up s0 (1/4)/n U, with neither decay nor
baseline; one generator `default_rng(seed)` for all rules, drawn in the order of the rules at each byte, so that a rule's random choices
depend on the rules run beside it; missing earlier bytes at t = 0, 1 are left out. Y1 = Y:eps:1/256:own:s:1, Y2 = Y:eps:1/256:own:c:4.

**Part 1: exact inputs** (`own_yn.py`, `out/yn/`). With the same rules in the same order my figures equal the main worker's as floats in
every window (`cmp_yn.py`, `out/yn/compare_main_worker.txt`): the 18 files x{1,2,3}_s{73129,93683,116089}_{64k,256k} (A, A eps, B, D,
D) and g2_eps_own, g3_eps_own (B, D and 12 settings of Y). In the main worker's step-2 configuration (B, D 1/64, Y1, Y2), last quarter
in per cent (`out/yn/table_exact.txt`; seeds 93683 / 73129 / 116089):

| inputs, bytes | B | D 1/64 | Y1 | Y2 | ref_1 / ref_2 / ref_3 (73129) |
|---|---|---|---|---|---|
| 1 byte, 65,536 | 10.08 / 9.44 / 8.40 | 10.05 / 10.16 / 10.32 | 2.39 / 2.58 / 2.88 | 5.65 / 6.73 / 7.65 | 10.55 / 12.34 / 25.12 |
| 2 bytes, 65,536 | 10.89 / 10.80 / 6.86 | 12.24 / 12.13 / 12.49 | 2.57 / 2.64 / 2.64 | 11.42 / 11.38 / 11.47 | |
| 3 bytes, 65,536 | 17.13 / 12.62 / 13.45 | 26.78 / 26.32 / 26.72 | 20.47 / 20.67 / 20.20 | 18.16 / 14.89 / 14.23 | |
| 1 byte, 262,144 | 8.91 / 8.35 / 7.95 | 9.79 / 9.83 / 9.73 | 1.39 / 1.28 / 1.14 | 3.25 / 3.36 / 3.15 | 10.44 / 12.42 / 27.44 |
| 2 bytes, 262,144 | 10.91 / 10.98 / 7.25 | 12.13 / 12.24 / 12.33 | 1.55 / 1.75 / 1.44 | 9.72 / 10.11 / 10.10 | |
| 3 bytes, 262,144 | 18.76 / 17.59 / 17.38 | 27.25 / 27.29 / 27.46 | 19.14 / 19.54 / 19.90 | 22.07 / 17.09 / 17.02 | |

Five generators of their own (`sep=1 gen=1..5`, B and Y1 only, 65,536 bytes), range of the last quarter: 1 byte B 5.55-10.05, Y1
2.04-2.86; 2 bytes B 6.68-11.00, Y1 2.09-4.26; 3 bytes B 11.55-17.13, Y1 15.33-22.20 (each seed in the table file). Y1 is above B on
three exact bytes on every seed and generator; on one and two bytes it is below the end mark's share (about 6%) on every seed and
generator, and lower at 262,144 than at 65,536 bytes.

**Part 2: recorded driver inputs** (`python3 own_end.py out/OE_<tissue>_<seed>.rec <seed> 65536 Y_<tissue>_<seed> <A4|L8> Y`;
`out/own_end_Y_*.json`, `out/ytab.txt`). Rule Y as yn.py with n = P, the unit's inputs + 1; on "no" each other unit adds
up s0 (1/4)/P_v times its own input vector. L8-128 lives of 65,536 bytes were run for this (`jobs_OE128.txt`). The rules run in the
order B, D, Y1, Y2 with measure.py's generators, so B's random choices, and its figures, differ from section 11's. Last quarter, run |
clip8:

| seed, tissue | B | D 1/64 | Y1 | Y2 | ref_1 / ref_2 |
|---|---|---|---|---|---|
| 93683 L8-256 | 7.52 / 8.23 | 9.26 / 9.27 | 2.46 / 2.51 | 6.22 / 6.09 | 10.62 / 12.14 |
| 93683 L8-128 | 8.79 / 7.44 | 11.99 / 12.02 | 3.31 / 6.01 | 5.69 / 7.06 | |
| 93683 A4-256 | 6.07 / 6.45 | 10.64 / 10.64 | 3.17 / 3.42 | 4.46 / 5.19 | |
| 73129 L8-256 | 12.35 / 13.44 | 17.02 / 17.00 | 8.32 / 8.14 | 9.97 / 11.22 | 10.55 / 12.34 |
| 73129 L8-128 | 12.13 / 10.83 | 15.54 / 15.53 | 6.92 / 7.74 | 8.62 / 9.14 | |
| 73129 A4-256 | 9.95 / 9.36 | 13.80 / 13.81 | 7.79 / 6.66 | 9.31 / 9.48 | |
| 116089 L8-256 | 11.04 / 10.18 | 16.17 / 16.17 | 2.58 / 7.53 | 3.97 / 5.24 | 10.61 / 12.20 |
| 116089 L8-128 | 5.82 / 8.75 | 13.89 / 13.90 | 6.50 / 6.76 | 8.77 / 8.56 | |
| 116089 A4-256 | 10.93 / 8.95 | 15.31 / 15.31 | 8.69 / 6.82 | 6.26 / 7.89 | |

Y reaches the present byte's level (ref_1) in the last quarter in one of the 36 cases (73129, L8-256, clip8 Y2, 11.22) and the two-byte
level in none; on seed 93683 it reaches neither on any tissue, and Y1 is there below the end mark's share on every tissue. Y is above B
in 3 of the 36 pairs (73129 A4-256 clip8 Y2 9.48 against 9.36; 116089 L8-128 run Y1 6.50 and Y2 8.77 against 5.82), all below ref_1.

**Part 3: is the step for the non-emitting units safe** (exact inputs, 262,144 bytes, weight norms every 4,096 bytes,
`out/yn/table_norms.txt`). The weights grow without bound: under Y1 the median unit's weight vector grows in proportion to the life
(1 byte 31 at 65,536 to 127 at 262,144; 2 bytes 21 to 85; 3 bytes 14 to 54), since every "no" adds the same step to 255 units and nothing
takes it back; under Y2 the largest row reaches 500-940; under B the median unit is never changed (0) and the largest row is 47-337. The
share right falls late: Y1 in windows of 4,096 ending at 16k / 64k / 128k / 256k, seed 93683, 1 byte 6.4 / 2.3 / 2.3 / 0.8, 2 bytes 8.8 /
2.4 / 3.0 / 2.1, 3 bytes 19.1 / 21.0 / 22.0 / 19.4; seeds 73129 and 116089 alike; B on three bytes still rises to 256k. On the recorded
inputs at 65,536 bytes Y1's median row norm is 0.4-0.5 and Y2's largest 12-105, against 0 and 6-26 for B. Under decay s the emitter's own
step falls with the square root of its emissions while the step of the other units does not; that this causes the fall is not tested
here.

## 14. Rule Q: the activity as the unit's own estimate of the chance of its byte (reviewer's task of 12:58 UTC)

**Readings.** Written from the reviewer's description alone; the places it leaves open, sent before the reviewer's yq.py was read: Q1 the
generator (here each rule in a life of its own with `default_rng(seed)`, and five others `default_rng([g, 0])`, g = 1..5); Q2 all
activities computed once before any change at a byte; Q3 the random emitter (chance 1/16) is the emitter for the step, and on yes all
255 other units step, values never seen included; Q4 on recorded inputs n = |U_v|^2 of each unit's own input vector; Q5 ties to the
lower value. Rule: W[e] += eta (ok - a_e) U / n; on yes W[v] += eta (0 - a_v) U / n for every v other than e; on no the others do
nothing. Afterwards: yq.py takes the same update (its Q with z1); it runs D, B and its Q rules with one generator in that order. With its
order (`jobs_cq.txt`: D 1/16, B, Q 1/4, Q 1/16, Q without the others' step) my figures equal its files d2_s93683_64k, d2_s116089_64k,
d3_s93683_256k, d3_s116089_256k as floats in every rule (20 of 20). In those files Q without the others' step is as good as Q (3 bytes,
262,144: 23.88 and 25.64 against 24.05 and 24.90).

**(a) Exact inputs** (`out/yn/table_q.txt`, `qtab.py`), last quarter in per cent, seeds 93683 / 73129 / 116089, own life a rule;
five generators in brackets:

| inputs, bytes | Q 1/4 | Q 1/16 | Q 1/64 | D 1/64 | B |
|---|---|---|---|---|---|
| 3, 65,536 | 22.24 / 22.91 / 23.39 (19.96-23.63) | 16.89 / 22.50 / 23.58 (13.78-23.90) | 14.42 / 16.74 / 16.91 | 26.78 / 26.32 / 26.72 | 14.16 / 16.90 / 16.74 (11.55-17.13) |
| 3, 262,144 | 23.68 / 23.73 / 23.81 | 24.65 / 24.17 / 24.27 (24.13-24.66) | 17.79 / 21.00 / 22.56 | 27.25 / 27.29 / 27.46 | 17.59 / 18.99 / 19.55 (17.21-20.55) |
| 2, 65,536 | 10.00 / 10.13 / 9.92 | 11.27 / 11.54 / 11.29 | 11.48 / 11.13 / 11.11 | 12.24 / 12.13 / 12.49 | 11.01 / 10.96 / 10.69 |
| 2, 262,144 | 9.90 / 10.04 / 9.97 | 11.64 / 11.52 / 11.65 | 11.85 / 11.77 / 11.80 | 12.13 / 12.24 / 12.33 | 11.01 / 10.87 / 10.89 |
| 1, 65,536 | 6.77 / 6.90 / 6.84 | 9.81 / 9.01 / 9.35 | 10.24 / 10.01 / 10.10 | 10.05 / 10.16 / 10.32 | 8.81 / 9.30 / 9.03 |
| 1, 262,144 | 6.45 / 6.26 / 6.53 | 9.93 / 9.74 / 9.84 | 10.13 / 10.04 / 10.18 | 9.79 / 9.83 / 9.73 | 7.75 / 7.61 / 7.68 |

**(c) Bounded.** Largest weight-row norm at 16k / 64k / 128k / 256k: Q on three bytes 1.9-3.3 / 3.4-3.7 / 3.6-3.7 / 3.6-3.7, two bytes
about 1.2, one byte about 0.5; B on three bytes 15 / 89-135 / 132-227 / 197-297. Q's share does not fall late (Q 1/16, three bytes,
windows at 64k / 128k / 256k: 23.7 / 23.7 / 24.4 on 73129, 17.5 / 24.0 / 24.1 on 93683).

**(b) Recorded driver inputs** (`python3 own_end.py out/OE_<tissue>_<seed>.rec <seed> 65536 Q_<tissue>_<seed> <A4|L8> Q`,
`out/own_end_Q_*.json`, `out/qreal_tab.txt`; each configuration its own generator `default_rng([seed, c])`), last quarter, run | clip8:

| seed, tissue | B | D 1/64 | Q 1/4 | Q 1/16 | Q 1/64 | best Q / D | ref_1 / ref_2 |
|---|---|---|---|---|---|---|---|
| 93683 L8-256 | 7.61 / 8.52 | 9.26 / 9.27 | 6.08 / 6.68 | 6.34 / 8.32 | 9.40 / 9.00 | 1.02 | 10.62 / 12.14 |
| 93683 L8-128 | 5.91 / 7.40 | 11.99 / 12.02 | 8.86 / 8.33 | 8.06 / 9.55 | 8.12 / 8.00 | 0.80 | |
| 93683 A4-256 | 6.65 / 7.23 | 10.64 / 10.64 | 7.72 / 6.68 | 7.76 / 9.08 | 9.00 / 8.62 | 0.85 | |
| 73129 L8-256 | 11.71 / 10.86 | 17.02 / 17.00 | 13.54 / 13.47 | 15.03 / 14.86 | 14.83 / 11.91 | 0.88 | 10.55 / 12.34 |
| 73129 L8-128 | 9.78 / 9.55 | 15.54 / 15.53 | 12.15 / 11.80 | 13.49 / 13.59 | 14.06 / 11.22 | 0.90 | |
| 73129 A4-256 | 10.49 / 8.45 | 13.80 / 13.81 | 10.72 / 10.50 | 12.12 / 12.23 | 12.16 / 12.26 | 0.89 | |
| 116089 L8-256 | 7.89 / 7.90 | 16.17 / 16.17 | 13.65 / 13.32 | 14.83 / 14.00 | 14.50 / 14.22 | 0.92 | 10.61 / 12.20 |
| 116089 L8-128 | 7.10 / 6.51 | 13.89 / 13.90 | 11.87 / 12.03 | 12.31 / 12.04 | 12.01 / 11.97 | 0.89 | |
| 116089 A4-256 | 11.18 / 8.16 | 15.31 / 15.31 | 12.21 / 12.22 | 14.06 / 12.65 | 12.90 / 12.92 | 0.92 | |

On seeds 73129 and 116089 the best Q comes to 0.88-0.92 of D on every tissue and passes ref_2 (B: 0.49-0.76 of D). On seed 93683 Q is
above B on every tissue but stays below ref_1 on all three; it equals D only on L8-256, where D itself is below ref_1, and is 0.80 and
0.85 of D on L8-128 and A4-256. The largest weight-row norm of Q 1/16 on recorded inputs is 0.12-0.67 (B 3.4-25).

## 15. Memory at the output unit's own connections: slow copies of each input (reviewer's task of 13:25 UTC)

**Readings** (sent before running): T1 the traces s_k(t) = s_k(t-1) + (x(t) - s_k(t-1))/k are taken of the raw output x, and each
component (x, s_2, s_4, s_8) then gets its own running standardisation (measure.py's 'run') and is clipped at +-8; T2 s_k is updated
with x(t) before the unit reads at byte t, s_k = 0 before byte 0; T3 n = |U|^2 of the whole vector with the constant; T4 Q's random
choices from `default_rng([seed, form])`; T5 clipped inputs only. Rules D (eta 1/64) and Q (eta 1/16) as in section 14.
Tissues without time constants: my long-axon options with tau = 2 for every cell (`prepare("N0", ..., long=True)`; A4-0: K = 1,
R = 1/4; L8-0: K = 8, R = 1/2), 65,536-byte lives recorded at every byte (`jobs_T0.txt`). `python3 own_trace.py out/OE_<t>_<seed>.rec
<seed> 65536 <t>_<seed> <A4|L8> forms=13|24`, `out/trace_*.json`, table `out/trtab.txt` (`trtab.py`).

Last quarter in per cent, D | Q, forms (i) x, (ii) x s_2, (iii) x s_2 s_4, (iv) x s_2 s_4 s_8:

| tissue, seed | (i) | (ii) | (iii) | (iv) | ref_1 / ref_2 / ref_3 |
|---|---|---|---|---|---|
| A4-0 73129 | 6.04 / 7.25 | 12.53 / 12.34 | 14.51 / 13.42 | 15.04 / 13.59 | 10.55 / 12.34 / 25.12 |
| A4-0 93683 | 7.08 / 8.70 | 17.89 / 12.74 | 17.76 / 16.67 | 18.16 / 11.90 | 10.62 / 12.14 / 25.33 |
| A4-0 116089 | 8.64 / 8.84 | 16.22 / 12.52 | 18.05 / 14.77 | 18.32 / 15.03 | 10.61 / 12.20 / 25.36 |
| L8-0 73129 | 5.51 / 7.77 | 15.91 / 13.16 | 16.63 / 15.56 | 16.70 / 15.13 | |
| L8-0 93683 | 5.22 / 5.77 | 10.63 / 9.56 | 12.24 / 10.88 | 12.40 / 7.37 | |
| L8-0 116089 | 6.54 / 6.91 | 15.04 / 11.65 | 16.51 / 13.93 | 16.63 / 13.21 | |
| L8-128 73129 | 15.53 / 13.42 | 16.86 / 15.48 | 16.67 / 15.03 | 16.60 / 15.14 | |
| L8-128 93683 | 12.02 / 9.51 | 12.59 / 10.08 | 12.49 / 11.23 | 12.43 / 7.77 | |
| L8-128 116089 | 13.90 / 12.92 | 15.05 / 13.55 | 15.32 / 13.21 | 15.33 / 13.51 | |
| A4-256 73129 | 13.81 / 12.56 | 14.52 / 12.87 | 14.43 / 13.43 | 14.02 / 13.08 | |
| A4-256 93683 | 10.64 / 8.65 | 10.91 / 8.73 | 10.96 / 10.10 | 10.69 / 7.28 | |
| A4-256 116089 | 15.31 / 13.95 | 15.91 / 14.00 | 16.12 / 14.32 | 16.06 / 13.08 | |

On the tissues without time constants x alone gives D 5.2-8.6, at or below the present byte's level; the traces add 7-11 points (D,
(iii)), to 12.2-18.3, mostly with s_2; s_8 adds at most 0.5 to D and lowers Q on 93683 (A4-0 16.67 to 11.90, L8-0 10.88 to 7.37). On the
tissues with time constants the traces add 0.3-1.4 points to D and up to 1.8 to Q. On seed 93683, A4-0 with traces (D 17.76-18.16) is
above every tissue with time constants, with or without traces (at most 12.59). No form reaches ref_3; the best, 18.32, is 0.72 of it.

**What the unit needs, and the weights.** Each input connection holds k - 1 further slow state variables (s_2, s_4, s_8), each with its
own running mean and deviation for the standardisation, and a weight for each; nothing else changes in D or Q. The largest weight-row
norm at 65,536 bytes is 0.20-0.72; it is still rising at the end of the life, more slowly than at the start (for example A4-0 73129, Q
(ii): 0.17, 0.36, 0.47, 0.53, 0.57, 0.62, 0.63, 0.65 every 8,192 bytes); whether it stays bounded beyond 65,536 bytes was not measured.

## 16. A4-0 with copies at the output unit: the long life, where the present byte is lost, the digits (task of 14:07 UTC)

`own_trace2.py` (D 1/64 and Q 1/16 on [x, s_2, s_4], section 15's readings; with the driver's own cells it repeats own_trace.py:
14.51 / 13.42 on A4-0 73129). Where it and the reviewer's outq.py differ (read after section 15): outq.py takes the copies of the
standardised input scaled by sqrt(2k - 1), standardises at rate 1/1024 from the start with mean and variance updated after use, gives
a unit the members of its output cell's neighbourhood (the driver not added), one generator for all rules, last quarter from 3NB/4.

**Part 1. The long life** (A4-0, 262,144 bytes recorded at every byte, `jobs_long.txt`; `out/tr2_L_A40_<seed>_c.json`,
`out/long_blocks.txt`). Share per block of 32,768 bytes and the largest weight-row norm at each block's end:

| seed, rule | share per block (%) | largest row norm | last quarter |
|---|---|---|---|
| 73129 D | 14.03 14.60 14.70 14.51 14.88 14.96 15.15 15.07 | 0.134 0.199 0.256 0.301 0.342 0.380 0.413 0.441 | 15.11 |
| 73129 Q | 11.65 13.33 13.37 13.45 14.15 14.47 14.38 14.29 | 0.285 0.409 0.510 0.580 0.649 0.713 0.774 0.829 | 14.33 |
| 93683 D | 16.59 17.79 17.61 17.86 18.09 17.29 17.30 17.53 | 0.244 0.331 0.376 0.418 0.455 0.445 0.455 0.466 | 17.41 |
| 93683 Q | 11.17 16.58 16.68 17.01 16.97 16.42 16.46 16.62 | 0.439 0.618 0.696 0.753 0.793 0.814 0.872 0.902 | 16.54 |

The share holds (73129 rises slowly; 93683 is flat from the second block); the weights grow more slowly with time; on 93683 D's largest
row levels at 0.45-0.47 from 131,072 bytes; the others are still rising at 262,144 (by 0.03-0.06 a block).

**Part 2. Which cells a unit sees** (A4-0, 65,536 bytes; distances by breadth-first search over the final connections; reader: the
linear reader of section 12 on the outputs x of the same cells, last 8,192 bytes, present byte):

| cells | 73129 D / Q | 93683 D / Q | 116089 D / Q | reader, all values (73129 / 93683 / 116089) | reader, digits |
|---|---|---|---|---|---|
| (a) distance 1 (74 / 62 / 64 cells) | 19.70 / 14.00 | 19.62 / 17.08 | 20.16 / 17.13 | 0.895 / 0.834 / 0.891 | 0.90 / 1.00 / 1.00 |
| (b) distance 1-2 (244 / 214 / 216) | 19.84 / 14.12 | 19.76 / 15.87 | 19.88 / 16.99 | 0.895 / 0.834 / 0.891 | 0.90 / 1.00 / 1.00 |
| (c) driver's own (678 / 708 / 682) | 14.51 / 13.42 | 17.76 / 16.67 | 18.05 / 14.77 | 0.553 / 0.430 / 0.507 (union) | 0.52 / 0.39 / 0.59 |
| (d) all 1,984 | 18.07 / 13.93 | 19.24 / 17.33 | 20.18 / 15.64 | 0.894 / 0.834 / 0.891 | 0.90 / 1.00 / 1.00 |

The reader stops at 0.83-0.90 in every set because at the end of the life 8 / 11 / 8 of the 46 byte values' input lines have no
receiving cell (`out/lines_unconnected.txt`: 73129 end mark + 9 E N P R S; 93683 2 < > C J M P R V v w; 116089 8 A G J K V W v); these
bytes are 17 / 20 / 13% of the read positions, and the reader passes 1 minus that share only by naming one of them by the absence of the
others. A connected line has a median of 2 receivers. So the present byte is lost first at the input lines (their connections
withdrawn), then beyond distance 2 (0.83-0.90 to 0.43-0.55 at the drivers' cells). Every unit seeing only the 62-74 cells at distance
1 gives D 19.6-20.2, 0.87-0.89 of the reviewer's exact-input 22.6, as much as or more than all 1,984 cells.

**Part 3. Values never named; the digits** (`out/never_named.txt`). D at the drivers' own cells never names in the last quarter 18 of
46 values on 73129 (9 B D E F G I K L M O P S T U V Z v), 12 on 93683 (0 9 C D G H O R S T v w), 4 on 116089 (9 B N U); Q 22 / 22 / 26.
With an exact one-hot of the present byte's digit added to every driver's inputs, D 14.51 to 14.73, 17.76 to 17.83, 18.05 to 18.24 (Q
13.42 to 13.46, 16.67 to 16.83, 14.77 to 14.92): 0.07-0.22 of the gap of 4.5-8 points to 22.6. A digit as the next byte is named with
recall 0-0.11 under D with or without it: the present digit does not tell the next one in this stream.

**Added (reviewer's questions of 14:21 UTC; `lines_q.py`, `out/lines_q.json`; no new run).** In my program a connection from an input
line cannot be withdrawn either: the withdrawal loop skips senders that are lines (`my_tissue.c`, step 9, `if (s >= NC) continue;`).
(1) All 8 / 11 / 8 lines with no receiver at the end (73129 / 93683 / 116089) had none at birth; 9 / 11 / 9 lines had none at birth, so
one line on 73129 and one on 116089 gained a receiver by making; lines with a made connection at the end: 5 / 4 / 4; receivers of a line,
median at birth 2 / 1 / 2, at the end 2 / 1.5 / 2. (2) Cells at distance 1 with no receiver of their own at the end: 12 of 74, 9 of 62,
9 of 64. Linear reader of the present byte (all values; digits): all distance-1 cells 0.895 / 0.834 / 0.891 (0.90 / 1.00 / 1.00); only
those with a receiver 0.864 / 0.752 / 0.862 (0.90 / 0.90 / 1.00); those without a receiver alone 0.353 / 0.360 / 0.389 (0.20 / 0.40 /
0.46).

## 17. An output part of M cells on exact inputs (reviewer's task of 14:42 UTC, after Mr. Xu's ruling that the output lines are withdrawn)

`outpart.py`, written from the reviewer's description alone. Readings of the open places (sent before yq_out.py was read): O1 the
copies are of the raw one-hot, s_k += (onehot - s_k)/k before use, scaled by sqrt(2k - 1); O2 a cell's best estimate is the maximum
over all 256 values (ties to the lower value), cells ordered by it (ties to the lower index); O3 in its turn a cell draws, and with
chance eps speaks a seen value at random whatever its threshold, inhibiting like any speaker; O4 inhibition received = sum of G[k, j]
over the cells j that already spoke; G grows in both directions for every pair of speakers on no, unclipped; O5 "one cell (th 0)": the
threshold stays 0 (no change); O6 yes per quarter, byte 0 excluded; one generator default_rng(seed) drawn per cell in the order of the
pass; O7 (c) each cell a fixed random half of the 768 byte features (default_rng(seed + 1)), the constant kept; O8 (d) a one-hot of
every byte spoken at the previous byte, 256 more inputs, no copies.

**Against yq_out.py.** One cell: 7.8 / 11.0 / 16.2 / 16.2 by quarter on 73129, equal to the reviewer's. M = 8 and M = 32 differ (mine
3.2 / 5.1 / 10.0 / 9.9 and 1.7 / 2.7 / 3.1 / 3.3; the reviewer's 2.0 / 9.2 / 12.0 / 14.0 and 4.0 / 5.5 / 5.5 / 9.7). Read afterwards,
yq_out.py draws the random speakers after the threshold pass, all at once, and they inhibit no one; orders the cells by
np.argsort(-best); and counts byte 0 in the first quarter. With that order (`draw=after`) my program equals its pushed files o1, o8a,
o8b, o32 (no threshold change, no inhibition) as floats in all four quarters; the files with threshold change and inhibition behind the
reviewer's figures were not pushed. M = 8 depends strongly on the generator: on 73129 at 65,536 bytes the last quarter over five
generators is 4.57 / 8.39 / 9.86 / 13.95 / 14.43, which contains the reviewer's 14.0.

**(a) 262,144 bytes** (`out/op/table.txt`), yes by quarter of 65,536 bytes, seeds 73129 / 93683 / 116089:

| | 73129 | 93683 | 116089 |
|---|---|---|---|
| one cell | 12.8 16.5 17.8 19.4 | 11.3 16.5 18.2 19.6 | 13.5 16.9 17.9 18.6 |
| M = 8 (eps 1/64, r_g 1/64) | 7.0 11.2 12.4 13.3 | 12.0 15.4 17.0 18.0 | 7.6 14.8 15.5 16.8 |
| M = 32 (eps 1/256) | 2.7 4.8 8.4 10.0 | 1.9 3.2 6.2 10.5 | 2.1 11.0 15.6 16.2 |

M = 8 and M = 32 are still rising at 262,144 bytes but neither reaches the single cell on any seed (last quarter 1.6-6.1 and 2.4-9.4
points below); 1.11-1.12 cells speak, one voice at 88-90% of the bytes, never silence.

**(b)-(d), M = 8, 65,536 bytes**, last quarter (73129 / 93683 / 116089); the same five generators' spread on 93683 and 116089: 8.15-15.39
and 6.96-12.26:

| variant | 73129 | 93683 | 116089 | speakers, one voice, silent |
|---|---|---|---|---|
| as described | 9.86 | 15.39 | 9.98 | 1.10-1.11, 90%, 0% |
| no inhibition (G stays 0) | 3.63 | 4.10 | 3.81 | 3.9-4.0, 13-18%, 13-16% |
| no threshold change (th stays 1/8) | 5.76 | 5.69 | 5.61 | 0.30-0.34, 25-29%, 69-72% |
| neither | 5.76 | 5.69 | 5.61 | as the line above (with th at 1/8 two cells rarely speak, so G does not matter) |
| (c) each cell a random half of the inputs | 9.32 | 13.51 | 9.61 | 1.10-1.11, 90%, 0% |
| (d) spoken bytes fed back | 10.25 | 14.81 | 10.28 | 1.10-1.11, 90%, 0% |
| one cell | 16.22 | 16.24 | 16.60 | |
| one cell, spoken byte fed back | 16.54 | 16.59 | 16.74 | |

Both parts are needed: without inhibition about four cells speak at once and the answer is rarely yes; without the threshold change the
thresholds stay above the estimates and the part is silent at 69-72% of the bytes. The half inputs and the feedback stay within the
generator spread. The largest weight-row norm at 262,144 bytes is 3.9-5.6 (the single cell 5.3-5.5); G reaches about 38 on 73129.

## 18. The whole system in a small case (task of 15:08 UTC; stopped at 15:3x UTC by the change of task)

`sysmini.py` (my own code; the reviewer's sysx.py not read): 64 receiving cells (sensitivity to the text byte, with (ii) the last output
byte, with (iii) also right/wrong; 8 entries of 1/8 at birth among what they receive), 64 cells in between (16 receiving cells each at
1/16), 8 active in each layer, one output cell (outpart.py's, th 0, eps 1/16) on [h2, sqrt(3) s_2, sqrt(7) s_4, 1]; copying rule
r = 1/16. Readings S1-S6 sent with the first row. 65,536 bytes, last quarter, median [range] over five orders (`out/sys/table.txt`):

| seed | (i) text | (ii) + last output | (iii) + error | ref_1 / ref_2 / ref_3 |
|---|---|---|---|---|
| 73129 | 15.09 [11.63-15.96] | 4.03 [3.80-5.09] | 5.42 [4.22-5.97] | 10.55 / 12.34 / 25.12 |
| 406364 | 16.53 [13.31-17.46] | 4.18 [3.92-4.82] | 5.26 [3.41-5.28] | 10.72 / 12.16 / 25.34 |
| 545353 | 15.92 [13.44-16.74] | 4.44 [4.35-4.67] | 5.26 [4.32-5.57] | 10.95 / 12.49 / 25.07 |

The reviewer's finding holds on every seed and order. One rule was tried before the stop: 'tag' (a three-factor rule: an active cell
tags its present inputs and makes the change at the next byte toward 1 if the error input it receives says right, toward 0 if wrong;
the cells in between take the error from the right/wrong sensitivity of their active inputs): (i) 5.96 [5.57-6.14] and 6.07
[5.76-6.13], (iii) 6.26 [6.16-6.34] and 6.56 [4.91-7.24] (n 3) on 73129 and 406364: (iii) slightly above (i), but both far below the
copying rule's (i); it did not help. The rules 'tagH1', 'errgate', 'novel' and the 262,144-byte grid were not run.
