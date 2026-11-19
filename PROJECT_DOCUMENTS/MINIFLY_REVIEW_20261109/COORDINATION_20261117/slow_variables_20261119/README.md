# The reviewer's tissue program (19 November 2026)

A complete program of the tissue of step 1 in C (`tissue.c`), with a driver (`drive.py`). It was written to try changes to the tissue in
minutes. It is not E63's program and not the fast program of `claude/e63-tissue-event-017dSE`; on the tissue whose strengths do not
change it gives the same outputs as that program to about 5e-9 (`cmp_fixed.py`).

Run one tissue (16,384 bytes, seed 73129 unless given):

    python3 -I drive.py <label> rule=first|coincidence|fixed [floor=1/128] [astar=1/32] [by_use=1] [gain=1] [eta_g=1/4096] [seed=..] [nbytes=16384]

The result is `out/<label>.json` and one printed line. `deps/` holds copies of two files of other lines, unchanged: `tissue_step1.py`
(E63's program; only its birth and its stream are used) and `readings.py` (the helper's readings tool).

Options beyond the definition (revision 5):

- `floor`: outputs below it are zero.
- `astar`: the target of activity a*.
- `by_use=1`: a connection is withdrawn when its use stays below 1/128 of the cell's most used connection for 128 steps at which the cell was active.
- `gain=1`: two-sided control of activity. The definition's control can only lower a cell's activity (the threshold cannot fall below zero).
  With this option every cell multiplies its arrival by a gain g, and g(k+1) = g(k) (1 + eta_g (a* - x(k)) / a*), held between 1/64 and 4096.
  This is the scaling of all of a cell's incoming strengths by one factor that follows the cell's own activity (synaptic scaling); the quantity is the cell's own.

Results are in `out/`; the batches run are `batch1.txt`, `batch2.txt`.

## This copy: slow variables of one cell (19 November 2026)

This folder is a copy of `reviewer_tissue_20261119` at commit 0b0b6f8e, with two options added to `tissue.c` and `drive.py`. Each is a quantity of one cell only.

- (A) Adaptation: `tau_h=.. beta=..`. The cell keeps h(k+1) = h(k) + (x(k) − h(k)) / tau_h, and its output is computed from v − θ − beta·h.
- (B) Depression of sending: `tau_u=.. rho_u=..` (named `rho` in the first batch, before the ratio control took that name). The cell keeps u in [0, 1], with u(k+1) = u(k) + (1 − u(k)) / tau_u − rho·u(k)·x(k). What other cells receive from it at step k+1 is u(k)·x(k) in place of x(k), and the traces of sending cells follow what is received.

Without either option the copy gives exactly the reviewer's program's line. This was checked by running both programs with `rule=coincidence gain=1 astar=1/32 floor=1/128`: the two lines are identical. The stored file `reviewer_tissue_20261119/out/coinc_gain_a32_F.json` does not match today's run of the same program with these options; the reason was not looked into.

The tissues run are listed in `batch_s73129.txt` and `batch_best_seeds.txt`, which `run_batch.sh` runs four at a time. Their results are in `out/`, and `make_table.py` writes `TABLE.md`.

Later the same day the reviewer's ratio control was merged in from commit d81a85c7: the fields `rho` and `ibar` after `gain`, the branch `two_sided == 2` of the step, and `tis.py` and `diag.py`. `tis.py` was given this copy's options. With neither option, `rule=fixed gain=2 rho=1` gives the reviewer's line exactly (S(d,0) 0.982 / 0.811 / 0.243, S(d,1) 0.071 / 0.063 / 0.048).

The batches on that base are `batch_ratio_s73129.txt` (tau 32 and 128) and `batch_ratio_slow_s73129.txt` (tau 512 and 2048). `run_batch.sh` runs each with `drive.py` and then with `diag.py`, whose outputs are in `diag/`. `chain.sh` ran the second batch after the first.
