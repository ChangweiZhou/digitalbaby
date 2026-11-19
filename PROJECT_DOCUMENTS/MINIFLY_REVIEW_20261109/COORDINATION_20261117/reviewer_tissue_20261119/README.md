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
