# Event-driven context kernel (net7e): results (22 November 2026)

Plan: `EVENTK_PLAN.md`. Branch `claude/newnet-eventk-sei232`, from 68002bf3. Implementation only, no experiment: the learner is net7
unchanged; net7e computes the same thing by visiting only the slots of the active sources (an inverted index of slots by source,
kept on the Tissue as numpy arrays `ix_head`, `ix_nxt`, `ix_prv`, updated when growth gives a slot a new source).

## Identity (check_net7e.py, output in check_net7e_out.txt)
net7 against net7e at every byte (cells, cols, surprise, predicted) and at the end (rw, rsrc, gsrc, rel, use, hold), 3,000 bytes,
L=8, C=128, S=64, lines_in=16, n_values=128, hold=0, n_grow=8, seed 301: M=16 and 32 × {tinyshakespeare, context stream} ×
{ctx=2 mesh=2 mesh_dag=1 mesh_base=0, ctx=2 no mesh, ctx=0}: **12 of 12 identical**. net10 (L=8 joint=1 mesh=2 mesh_base=0
mesh_dag=1, M=16 and 32, both streams) with `ctx_e=0` against `ctx_e=1`: every answer, Oc, xc, and at the end Ac and the context
strengths **identical in 4 of 4**. No mismatch was seen. In the timing runs below (22,000 bytes of tinyshakespeare, M=16 to 256), the
hash of the cells at every byte is the same for net7 and net7e at every M (and the hash of rw at byte 22,000 too, first net7e version).
The random sample of new sources does take place there (mesh=2: 16 active sources against n_grow=8), so the generator's draws match.

## Speed (time_net7e.py; eventk_timing_net7.jsonl, eventk_timing.jsonl)
net10's context stack alone (`Tissue.step`, L=8, C=128, S=64, ctx=2, hold=0, n_grow=8, mesh=2 mesh_base=0 mesh_dag=1, seed 301),
tinyshakespeare, bytes 20,000-22,000 timed after reading 20,000; OMP_NUM_THREADS=1, one run at a time, this container (4 cores).

| M | slots | slots in use | active slots a byte (net7e counts) | net7 ms/byte (kernel) | net7e ms/byte (kernel) | speed-up |
|---|---|---|---|---|---|---|
| 16 | 1.05 M | 73% | 11,184 | 2.57 (2.16) | 0.90 (0.60) | 2.9× |
| 32 | 2.10 M | 64% | 10,881 | 4.76 (4.25) | 1.20 (0.79) | 4.0× |
| 64 | 4.19 M | 49% | 11,005 | 7.05 (6.42) | 1.28 (0.88) | 5.5× |
| 128 | 8.39 M | 31% | 10,764 | 17.06 (15.86) | 1.59 (1.05) | 10.7× |
| 256 | 16.8 M | 17% | 10,228 | 37.41 (32.42) | 2.06 (1.29) | 18.2× |

The active slots a byte stay about 11,000 at every M, and net7e's kernel time grows only slowly with M (0.6 to 1.3 ms) while net7's
grows with the slots. The rest of a byte (the forward part and making the context and cells arrays in numpy) is 0.3 to 0.8 ms and is
still O(L·C·M) in places (the float cells array returned, the hold-part compare). Measured in the kernel at M=16/64 (5,000-6,000
bytes): about 0.2/0.3 ms walking the lists (random memory access), 0.1 ms growth (net7's code unchanged), 0.13 ms the rest.
A first net7e version (one list at a time, allocation per call, net7's numpy wrapper) gave 1.31 / 1.70 / 2.23 / 2.37 / 5.49 ms
(eventk_timing_v1.jsonl); the final one walks the active lists side by side, keeps its workspace between calls and builds the context
from the active cells only.

Limits: timings are from one container and one seed; the cost of net7e depends on how many slots read the active sources, which grows
with reading (here measured after 20,000 bytes; at 64,000 bytes it was not measured). The index is built at the first step from
(rw, gsrc); if rw, rsrc or gsrc are set from outside afterwards, `reindex()` must be called. The C workspace and generator are
per-process statics (as net7's generator), so one thread at a time.
