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
