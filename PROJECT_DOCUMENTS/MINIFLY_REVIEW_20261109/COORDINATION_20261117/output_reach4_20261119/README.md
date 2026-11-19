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

- `gain=2 rho=1`: ratio control. The cell holds a slow mean of its own output, ā (rate 1/128), and a slow mean of the outputs of its
  sources, ī (the plain mean over its connections, rate 1/128), and g(k+1) = g(k) (1 + eta_g (rho ī − ā) / (rho ī + ā)), held between
  1/64 and 4096. A cell so passes on the fraction rho of the activity that reaches it; where nothing arrives the gain does not move.
  Both quantities are the cell's own (its activity, and the activity arriving at its connections).

`diag.py <label> <options>` gives, by distance from the input lines, the share of a cell's output variance that the present and the
previous byte explain (raw outputs), and the divergence of two copies of one tissue after a change of 1e-9 (results in `diag/`).
`tis.py` is drive.py's set-up as a module.

## What the runs of 19 November show (seed 73129, 16,384 bytes, window A2; one seed, not confirmed by anyone else)

1. Tissue whose strengths do not change, no control (`diag/fixed.json`): the present byte explains 1.000 of the variance of the median
   cell's output at every distance from 1 to 6 connections and 0.96 at 8. The mean output falls from 0.0046 at 1 connection to 0.00088,
   0.00022, 0.00005 at 2, 3, 4: about 4.5-fold a connection. The readings tool sets outputs below 0.01 to zero, so S(4,0) = 0.022
   (chance) there is the result of the amplitude, not of a loss of what the cells carry.
2. Control toward a fixed activity a* (`gain=1`): every cell reaches the activity, by bursts of its own: the share explained by the
   present byte falls to 0.23, 0.09, 0.04 at 1, 2, 4 connections; with eta_g = 1/4096 two copies differing by 1e-9 differ by 1.0 after
   512 bytes. S(4,0) stays near 0.035 whatever the speed of the control (eta_g from 1/512 to 1/262144) and at a* = 1/128.
3. Ratio control (`gain=2`), strengths not changing: S(d,0) at d = 1 / 2 / 4 is 0.979 / 0.486 / 0.038 at rho = 1/2; 0.979 / 0.820 / 0.101
   at 3/4; 0.978 / 0.825 / 0.209 at 7/8; 0.982 / 0.811 / 0.243 at rho = 1 (level 0.1103). Lag 1 at rho = 1: 0.071 / 0.063 / 0.048
   against the present byte's own 0.070; the level 0.090 is not reached.

4. Ratio control with no bound on the gain is itself unstable: at rho = 1 two copies differ by 1.0 after 512 bytes, and the mean output
   rises again beyond 5 connections (cells that cannot reach the ratio take the largest gain). rho = 5/4 and 3/2: every gain at the
   bound 4096, S(1,0) 0.25 to 0.36. With the gain bounded (`gmax=4` or `gmax=2`) the divergence is 0 from 4 bytes on, the present byte
   explains 0.93 of the variance at 4 connections and 0.63 at 6, and the PREVIOUS byte explains 0.07 at 4, 0.30 at 6, 0.47 to 0.49 at
   7 and 8 connections (`diag/fixed_ratio_r100_m4.json`): activity takes several steps a connection, so cells far from the input lines
   still carry the byte before. The mean output there is small (0.0015 at 4 connections, 0.00007 at 8).
5. The two rules of strength change of stage A under the ratio control (rho = 1, no bound): S(1,0) 0.22 and 0.21, S(2,0) at chance,
   29 and 35 connections a cell. They destroy what the unchanging tissue carries.

Further options: `gmax=..` (bound of the gain); `rule=timing eta_s=.. tau_f=..`: a connection from an excitatory cell grows when the
sender was active before the receiver and falls when it was active after it, dw = eta_s (F_j x_i(k) − F_i x_j(k−1)), F a trace of a
cell's output with time constant tau_f steps (both ends of the connection, nothing else).

Results are in `out/`; the batches run are `batch1.txt` to `batch8.txt`.

## Added on 19 November after 09:20 UTC

- Build: `drive.py` and `tis.py` compile with `-ffp-contract=off` from 09:31 UTC, so that every multiplication and addition is
  rounded separately and the program equals the independent program of E62's worker at every byte (checked on the reference command:
  6,638 connections, 1,240 made, 10,474 withdrawn). Results in `out/`, `far/`, `diag/`, `variety/`, `band/`, `cond/` and `loops/`
  made before that time came from the build that joined a multiplication and an addition in one rounding; a life built one way is
  another life than the same command built the other way (their readings agree within the yardstick of STATUS row 48).
- `prune_only=2`: besides the test of order in time, a connection from a cell whose long mean of output is below the level of
  making (`a_make`) is withdrawn at age 4,096 steps; `prune_only=3`: only from excitatory cells. Tested and not adopted (batch37).
- `long=K-R` (tis.py only, so `far.py`, `band.py`, `loops.py`, `variety.py`, `diag.py`, `cond.py`; not `drive.py`): one excitatory
  cell in K, not a driver of an output line and not a feedback cell, drawn with `default_rng(seed + 20261120)`, has a long axon: it
  is a member of the neighbourhood of every cell within R of it. The starting connections are then drawn again with the same
  generator, 8 from each cell's members. Batches 39 and 40.
- `loops.py`: the sets of cells that can each reach the others, counted over all connections and over those from excitatory cells.
- Batches: 36 (loops), 37 (silent sources), 38 (inhibitory total), 39 and 40 (long axons).
