# nano/cross2: credit passed over two connections, forward in time (roadmap second revision, track M)

Development level; a small case; repeated by no one; no checker gate, seal or review is claimed. One line of work, one folder. Nothing outside this folder is changed.

## What was copied, and from where

From `loop_small_case/` at commit 260c5d73 (branch `claude/friendly-darwin-8mmhfm`): `hold.py`, `hold_x.py`, `.gitignore` (copied, then extended here); `kern4.c` as `kern5.c` and `fastk4.py` with the needed lines of `fastk.py` as `fastk5.py` (extended); `check_fast.py` rewritten in the same manner for this folder's kernel. With `cross=0` and `cross=1` the copies compute what the originals compute.

## The second crossing (`cross=2`, the reader's own strengths)

The cells are those of `hold.py`: state h, gate z, candidate c, h(t) = z h(t-1) + (1 - z) c; each cell reads the present byte's lines (with `chain=C` the C bytes before it) and K = 16 other cells one byte late. Each strength keeps a trace that runs forward (E(t) = z(t) E(t-1) + sensitivity(t) × input(t)). The answering cell's mismatch reaches cell k as r_k(t).

- No crossing: cell k changes by r_k(t) × E_k(t).
- First crossing (`cross=1`, as in `loop_small_case`): reader k passes back to each cell j it reads r_k(t) × [kc_k(t) Bc_kj + kz_k(t) Bz_kj], its own sensitivity to that connection times its own strength; cell j sums what its readers send, rho_j(t), and changes by rho_j(t) × E_j(t-1).
- Second crossing (`cross=2`, new): cell j passes on to each cell i it reads rho_j(t) × [kc_j(t-1) Bc_ji + kz_j(t-1) Bz_ji]: what reached it over the first crossing, times its own sensitivity to that connection as it was one byte earlier, times its own strength from i. Cell i sums what reaches it this way, rho2_i(t), and changes by rho2_i(t) × E_i(t-2). The changes of the first crossing and of the cell's own path are made as before.

**What a cell has to keep for this.** (1) For each of its traces, the value of one byte before beside the present one: at the moment of a change it then has the trace as it now is, as it stood one byte earlier and as it stood two bytes earlier (in the program: the trace after its update, the trace before its update, and the kept copy `P..`). That is one more number for each strength. (2) Its own two sensitivities (to its candidate and to its gate) of one byte earlier: two numbers for each cell (`kc1`, `kz1`). **No strength of another cell is used**: each cell multiplies by its own strengths only. As with the first crossing, the value passed differs from one cell read to the next (section 12.4 of `loop_small_case/REPORT.md` applies unchanged; this folder does not settle it). Nothing passes backward over the text; the strengths used at the second crossing are the cell's strengths at the present byte.

The numbers that learn are the same with 0, 1 or 2 crossings: for 256 cells, K = 16 and S byte values with C delay lines, 2 × 256 × (S(1 + C) + 16 + 1) in the cells, and 2 × S × 256 + S at the answering cell and its return connections.

## The checks

- `python3 hold_x.py check` (`check_x_output.txt`): at fixed strengths (a step of 1e-9), the forward program's accumulated changes over a line against the instrument's pass backward over that line, for no crossing (X=0), one crossing at the last step (X=1L) and two crossings at the last two steps (X=2L, new), without a chain and with two delay lines (the instrument was given the chain for this). Agreement to 8e-6 of the largest change or better (the level of the original check, set by the step of 1e-9). Eight controls that must differ do differ (by 4% to 30% of the largest change): each number of crossings against the instrument at another, and `cross=2` against two crossings at any time (X=2).
- `python3 check_fast.py` (`check_fast_output.txt`): the compiled kernel against numpy over 300 lines, every strength, trace and kept value: agreement to 1e-10 or better in 30 settings; four controls (kernel at one number of crossings, numpy at the other) differ by 9% and 70%.

## Plans and results

Each experiment has `PLAN_<n>.md`, committed before its runs, with the outcome appended. Result files are in `res/`.

| Experiment | Text | Result (development level) |
|---|---|---|
| E1, `PLAN_1.md` | m1, gap 8, 960,000 lines, five reported seeds | final: cross=1 94.4% at the median (lowest 83.2), cross=2 100.0% (lowest 95.7); lines to 90%: 420,000 against 300,000 at the median; hypothesis not refuted; A1 to A3 held, A4 and A5 failed |
| E2, `PLAN_2.md` | g2, gap 8, two delay lines, 2,400,000 lines, development seed | never-shown sums at the gap of 8 letters: cross=1 7.5%, cross=2 9.5%; hypothesis refuted; the reported seeds not run (condition of 30% not met) |
| E3, `PLAN_3.md` | d2, no delay line, 2,400,000 lines, development seed | never-shown sums: cross=1 2.3%, cross=2 8.0% (tens digit 30% against 68%); hypothesis refuted |

Commands: `sh run.sh jobs_e1_all.txt`, `sh run.sh jobs_e23_dev.txt`, `python3 report.py e1`, `python3 report.py rows <label> ...`. Total cost of the experiments: about 2.5 CPU-hours.
