# Holders that take part in the network (plan, written before any run; 22 November 2026)

Line: NEWNET (development level, not a MiniFly claim). Source: branch `claude/magical-albattani-sei232` at `cf5ff1d1`, work on
`claude/newnet-meshhold-sei232`. Task given to this helper session: let each module's holding cells read the network, not only the
byte's lines, and measure whether that helps. No existing file is edited; no adopted core is changed.

## The exact input change (`net12.py`)

net10 with `joint=1 mesh=2 mesh_base=0 mesh_dag=1` (the best graph in ARCH_GENERAL.md: Moby Dick 0.395/0.394 at K=1/2, seed 1).
One rule for every module m:

- within a byte the context cells step first, then the holders, so the holders of module m see **module m's own context cells of
  this byte** (2048 binary cells, about 8 active, up to 128 on a surprise burst; they answer to bytes up to this one: no look-ahead);
- each holder i has fixed strengths from every context cell j of its module, Xc[j,i] and Xg[j,i] ~ N(0, 1) (wx=1, the size of the
  byte lines Wc, Wg); candidate input += sum over active j of Xc[j,i] / sqrt(n active), gate input += the same with Xg (the added
  input has the size of one byte line whatever the number of active cells);
- `hin=add`: byte line and context cells; `hin=ctx`: context cells **in place of** the byte line (holders' Wc, Wg zeroed; the
  holders' expansion still reads the byte's line as in net10); `hin=none`: net10.
- C step: net8c's C step reads a per-holder constant (bc, bg) that never learns with fb=none; the added input is passed as that
  constant for the one byte and put back. So the C step is used unchanged; no numpy port. Correctness check before the runs:
  net12 `hin=none` must give net10's held-out to the last digit on a short run (same seeds), and `hin=add wx=0` likewise.

The source-regions variant (holders read the cells of m's source regions instead of m's own) is NOT_RUN: module m's own cells
already read those regions; one variant fewer keeps the comparison narrow.

## Hypothesis and predicted numbers

H: holders that integrate the context cells' codes (conjunctions of the last few bytes) over their clocks carry longer, more
specific context than holders integrating single bytes, so held-out prediction rises; memory across a gap is kept because the
gates and clocks are unchanged.

Predicted (mean of seeds 1, 2; T=16000, NT=10000, L=8):
- Moby Dick: net10 about 0.394; net12 add 0.400-0.410; net12 ctx about the same as add (within 0.01).
- Code: net10 unknown here (L=4 chain joint 0.384); net12 add +0.005 to +0.015 over net10.
- Memory (t18_mem.py, context stream G=16, T=30000, frozen 200 trials, choice between the two end bytes): context alone near 0.5
  (paths reach at most about L+2 = 10 bytes < 17) in both; holders alone and sum: net12 within 0.05 of net10.

## What refutes it

- Held-out help: if net12 add is not above net10 by at least 0.005 on the mean of both seeds for a stream, H is refuted for that
  stream (no help); below net10 by more than 0.005: it hurts. Seed-to-seed spread is reported; a difference smaller than that
  spread is read as no difference.
- Memory: holders-alone or sum frozen share of net12 more than 0.1 below net10's: the change costs holding.
- If hin=none does not reproduce net10 exactly, no comparison is reported until fixed.

## Runs and cost (OMP_NUM_THREADS=1, at most 4 at once)

| Runs | Count | Estimated each |
|---|---|---|
| t17 moby / code, net10 and net12 hin=add, hin=ctx, seeds 1, 2 | 2 streams x 3 nets x 2 seeds = 12 | ~2 min (26,000 bytes; net10 L=8 measured 3,000 bytes in 12 s) |
| t18_mem G=16 T=30000, net10 / net12 add / net12 ctx, seeds 1, 2 | 6 | ~2-3 min (30,200 bytes) |
| equivalence checks (T=2000 NT=1000) | 3 | ~12 s |

About 45 CPU-minutes, about 12 minutes wall at 4 at once. Model: L=8 modules, each 128x16 = 2048 context cells, 256 holders,
2048 expansion cells; net12 adds 2 x 2048 x 256 fixed strengths per module (8.4 M fixed, not learned). Logs go to
`meshhold_runs/`. Report: MESHHOLD_REPORT.md with one table of every run.
