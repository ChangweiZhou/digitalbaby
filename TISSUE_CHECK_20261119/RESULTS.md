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
