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
