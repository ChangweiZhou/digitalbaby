# Event-driven context kernel (net7e): plan (22 November 2026)

Job (helper session, NEWNET line): make the context-cell kernel cost in proportion to what is active, not to the number of slots,
without changing what it computes. This is an implementation change only: no new learner, no experiment, no claim about results.

## What net7.c costs now
Every byte, for each region d and each cell i (C*M cells), every slot s (S) is read: `a = rw>0 && ctx[gsrc]`, and the counts
n_on (strong active), n_act (active), sH (strong active from the hold part), aH (active from the hold part) are made. That is
L*C*M*S reads a byte, whatever is active (at net10's setting: 8*128*M*64).

## net7e.c: the same step, by events
1. **Inverted index, kept with the slots.** For each global source g (= gsrc, in [0, L*2CM)) a doubly linked list of the slots
   whose gsrc is g and whose strength is > 0 (`head[g]`, `nxt[slot]`, `prv[slot]`; `prv = -2` marks a slot not in any list).
   The arrays are numpy int32 arrays on the Tissue (so a copy of the Tissue copies its index). A slot enters a list only when the
   growth step gives it a source (strength w_new); it leaves its old list at that moment. A strength never becomes <= 0 by
   learning (w - eta_minus*w stays > 0 in doubles; w + eta*(1-w) > 0), and the visit still tests `w > 0` exactly as net7 does,
   so the counts are those of net7 even if it did. The index is built from (rw, gsrc) at the first step (and by `reindex()`),
   so a Tissue whose arrays were set from outside is still right.
2. **Counts.** Per region: scan its 2CM context entries (net7 already does, for np_/nh); for each active source walk its list;
   each slot met is active iff `rw > 0`; add to its cell's four counts; record the slot (for learning) and the cell (first touch).
   Cells never touched have zero counts, so pred = match = 0: exactly what net7 computes for them.
3. **pred/match** only for touched cells (others 0). Surprise: npc, hit from the columns of the touched cells; rsum summed over
   the winning columns' cells in the order c, m ascending (the same floating-point order as net7).
4. **Pick** for winning columns only (a column that does not win fires no cell, and its `best` enters nothing else: the rel update
   multiplies by `cl[c]`, f needs `cl[c]`). The score per cell, ties (first highest m) as net7. Cells output: cleared, then one per
   winning column.
5. **Learning.** rel: for pred cells (touched only), as net7. rw: for each recorded active slot, +eta if its cell has f, else
   -eta_minus if its cell is predicted (each active slot is in exactly one list, so it is updated once, as in net7).
6. **Growth** for f cells in ascending cell order (one per winning column, columns ascending: the same order as net7, so the C
   generator's draws come in the same order). "have": mark the sources of the cell's slots with rw > 0 (stamp array), then the
   missing sources are the active ones not marked, in order (net7: the same test, O(P*S)). Sampling, weakest-slot order and the
   writes are net7's code unchanged, plus the index update of every slot written.
7. use += f only for f cells (adding 0 elsewhere changes nothing, inf included).

Remaining work a byte that is not proportional to the active slots: O(L*2CM) to scan the context and O(L*CM) for clearing outputs
in C; the Python wrapper (as net7.py) builds the context and the float cells arrays with numpy, O(L*C*M). These will be reported as
measured, not hidden.

## Checks (check_net7e.py)
net7 and net7e side by side, every byte: cells, cols, surprise, predicted; at the end rw, rsrc, gsrc, rel, use equal (exact).
Streams: tinyshakespeare (from its start) and the context stream of check_net7.py, at least 3,000 bytes. Settings (n_grow=8,
M=16 and 32, otherwise net10's context settings: L=8, C=128, S=64, lines_in=16, n_values=128, hold=0): ctx=2 mesh=2 mesh_dag=1
mesh_base=0; ctx=2 without mesh; ctx=0. Plus net10 (`L=8 joint=1 mesh=2 mesh_base=0 mesh_dag=1`) with `ctx_e=0` against
`ctx_e=1`: every answer identical (and Oc), over the same bytes. Any mismatch is reported as is.

## Timing
OMP_NUM_THREADS=1, one process at a time: ms a byte of the context stack (`Tissue.step` of the context cells at net10's setting,
mesh=2 mesh_base=0 mesh_dag=1, L=8, C=128) for M = 16, 32, 64, 128, 256, net7 and net7e, on tinyshakespeare after a warm-up,
also split into C kernel time and the rest.

## Files
New: NEWNET/net7e.c, NEWNET/net7e.py, NEWNET/check_net7e.py, NEWNET/EVENTK_REPORT.md (results). Changed: NEWNET/net10.py, only an
option `ctx_e` (default 0, which keeps net7). No other existing file is edited.
