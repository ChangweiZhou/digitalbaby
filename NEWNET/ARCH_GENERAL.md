# A general architecture, found by fast iteration (22 November 2026)

Owner, this session: the system may have many kinds of cells with their own local rules, but the architecture must be general ("架构本身
需要是通用的"); "做起来，跑起来……我们的目的就是通过快速迭代找到这个架构". Development level; seed 1 unless stated.

**net10** (`net10.py`): L modules of one plan; L and the module size are the only scale. Module m: context cells (minicolumn region
reading region m-1 one byte late: the last m + 2 bytes), holding cells (gated, clocks log-uniform in a band 2^m to 2^(m+3) bytes,
adaptation, gain by own variance, own sparse expansion; in this form they read the byte's lines), and one output compartment for each
kind. Nothing is set for a task.

**Judged on streams never used in design** (`t17_general.py`): Moby Dick and the repository's Python files outside NEWNET; read 16,000
bytes from the start, then score 10,000 bytes from the middle, learning off. Count tables fitted on the same bytes (reference only):
Moby Dick 0.292 / 0.368 / 0.379 for the last 1 / 2 / 3 bytes; code 0.255 / 0.349 / 0.386.

| Output end | L=2 | L=4 | L=8 | code, L=4 |
|---|---|---|---|---|
| each compartment on its own mismatch (as net9) | 0.366 | 0.355 | 0.336 | 0.347 |
| one shared mismatch, each compartment a full step | 0.343 | 0.314 | 0.241 | 0.338 |
| **one shared mismatch, one step shared by all compartments** (`joint=1`) | **0.385** | **0.386** | **0.381** | |
| the same, step 4 times larger (L=8) | | | 0.368 | |

Readings: (1) with compartments that each learn alone, more modules were worse (votes of many self-confident compartments add up);
(2) the general rule, one delta rule over all of them with one step, removes that: on a never-used text it is above the three-byte
table at every size; (3) size neither helps nor hurts at 16,000 bytes read. Longer reading (64,000) at L=2, 4, 8 is running.

**Reading longer** (64,000 bytes, joint=1; count tables 0.294 / 0.390 / 0.452): L=2 0.409, L=4 0.418, L=8 0.400. Code (16,000, L=4, joint):
0.384 (tables 0.255 / 0.349 / 0.386; own-mismatch form 0.347). More reading helps; more modules beyond 4 still do not.

## A mesh instead of a chain (owner: the nervous system is a net, not layers)

`net7 mesh=K`: each region takes its columns from the byte's lines (its own wiring) and its context from K other regions' cells at the
byte before. Moby Dick, 16,000 bytes, joint=1, L=8 unless stated:

| Graph | held-out |
|---|---|
| chain (net10 as above) | 0.381 |
| random graph with loops, plus each region's own columns of the byte before (`mesh_base=1`), K = 2 / 4 | 0.350 / 0.355 |
| random graph with loops, no own base, K = 1 / 2 (K=2 also L=4: 0.333; L=16 half-size: 0.319) | 0.317 / 0.324 |
| **random graph without loops** (`mesh_dag=1`: region d reads regions before it; a region with none reads the byte before), K = 1 / 2 | **0.395 / 0.394** |
| the same, K = 2, L = 16 half-size modules | 0.385 |

Readings: with loops and no base the codes depend on the whole past (they do not recur: held-out falls); with the base, the byte
before dominates every region's context (128 of its active sources against 8 from each other region), so every region codes about
two bytes (predicted share 0.57 against 0.18 for the chain, yet worse held-out). Without loops the paths of a random graph have many
lengths, so contexts of many lengths, bounded: better than the chain.

**The graph without loops, more runs** (K=2, `mesh_base=0 mesh_dag=1`, joint=1): Moby Dick 16,000, L=8, seed 2: 0.391 (seed 1: 0.394).
Code 16,000, L=8: **0.392** (chain L=4: 0.384; three-byte table 0.386). Moby Dick 64,000: L=8 0.419, L=16 half-size 0.417 (chain L=8
0.400, L=4 0.418; three-byte table 0.452). With more reading the table pulls ahead: the learner gains less from more data than counting
does. My reading, to be tested: the context cells' capacity (2,048 a region) is used up by the many contexts of 64,000 bytes.

**Capacity of the context cells** (Moby Dick 64,000, L=8, graph without loops): as above 0.419; columns doubled (cC=256) 0.431; the
same with half the holders 0.429; **cells a column doubled (M=32) 0.444**; no holders at all 0.415 (three-byte table 0.452). The reading
was right: the context cells were full. More cells for the same byte in different contexts is the size that pays.

**Size that pays, further** (64,000 bytes, L=8, graph without loops, joint=1): Moby Dick M=16 0.419, M=32 0.444 (seed 2: 0.446),
M=32 with cC=256 0.448, **M=64 0.455** (three-byte table 0.452). Code, M=32: **0.445** (three-byte table 0.420). Each doubling of the
cells a column holds raises the held-out share; at M=64 the learner is above the three-byte table on a text never used in design.

**The battery for net10 (L=8, graph without loops, K=2, M=32, joint=1), seed 1:** memory G=16 (128 values, 40,000 bytes): 1.000 at the
gap. Two texts (10,000-byte tests): A 0.383 -> 0.369, B 0.447, control 0.340 / 0.441 (net9: 0.356 -> 0.346, B 0.409, control 0.328 /
0.403): every level higher. **Keeping A on letters: 0.960 -> 0.470 (net9: 1.000 -> 1.000): a regression.** Candidates: paths of the
random graph shorter than the 9 bytes back to A's start, or the shared mismatch (net9's fast compartment learned alone).

**Why keeping regressed** (life=ctx, M=32): chain with joint 1.000 -> **1.000**; chain, own mismatch 0.910 -> 0.470; graph without loops
K=1, joint 0.960 -> 0.470; the same K=2, own 0.530 -> 0.470. Keeping needs a path as long as the context to be told apart (the chain's 8
regions reach A's start 9 bytes back; the random graph's longest path does not), and on the chain the shared mismatch keeps better than
each compartment alone. The chain gives long paths, the random graph many lengths. New topology `mesh_dag=2`: each region reads the
region just before (a path through all) and K-1 random earlier ones (paths of many lengths).

## Search, batch 1 (running): A0-A7, same size (L=8, M=16); results in `search_results.tsv`

**Batch 1** (L=8, M=16; score = mean of moby, code, keep, mem):

| | architecture | moby | code | keep | mem | score |
|---|---|---|---|---|---|---|
| A0 | chain, joint | 0.381 | 0.359 | 1.000 | 1.000 | **0.685** |
| A2 | graph without loops K=1, joint | 0.395 | 0.398 | 0.470 | 1.000 | 0.566 |
| A1 | graph without loops K=2, joint | 0.394 | 0.392 | 0.470 | 1.000 | 0.564 |
| A6 | graph without loops K=4, joint | 0.390 | 0.392 | 0.470 | 1.000 | 0.563 |
| A3 | graph without loops K=2, own mismatch | 0.363 | 0.354 | 0.470 | 0.977 | 0.541 |
| A7 | graph with loops and own base, joint | 0.350 | 0.335 | 0.470 | 1.000 | 0.539 |
| A4 | chain, own mismatch | 0.336 | 0.297 | 0.470 | 0.977 | 0.520 |
| A5 | graph without loops K=2, joint, no holders | 0.396 | 0.398 | 0.470 | 0.507 | 0.443 |

Joint beats own mismatch everywhere; the holders are what carries memory across a gap (without them 0.507); only the chain keeps A, and
the graphs are better on both never-used streams (+0.013 to +0.039). The score is dominated by keep (nearly 1.0 or 0.47).

**Batch 2** (chain plus random earlier sources, `mesh_dag=2`): A8 K=2: moby 0.397, code 0.391, keep 0.470, mem 1.000 (score 0.565); A9 K=3:
0.393 / 0.392 / 0.470 / 1.000 (0.564); A10 L=12 M=12: 0.385 / 0.369 / 0.470 / 1.000 (0.556). A path through all regions is not enough
to keep A: a region that reads the region before and a shorter one takes the union as its context, and the shorter context (common in
text) undoes the separation. Keeping seems to need a pure long chain somewhere. Next: two pathways of different plan (`mesh_dag=5`): a
pure serial chain (as the hippocampal trisynaptic path) beside a random graph without loops (as the neocortex), joined at the output.

**Batch 5** (architectures from biological principles):

| | architecture | moby | code | keep | mem | score |
|---|---|---|---|---|---|---|
| E6 | chain + one random earlier source, **sparser (4 winning columns a region, not 8)** | 0.385 | **0.394** | **1.000** | 1.000 | **0.695** |
| E4 | chain, sparser (4) | 0.375 | 0.384 | 1.000 | 1.000 | 0.690 |
| E5 | chain, denser (16) | 0.382 | 0.353 | 1.000 | 1.000 | 0.684 |
| E2 | two parallel chains of 8 (L=16, half-size regions) | 0.377 | 0.349 | 1.000 | 1.000 | 0.681 |
| E3 | feedforward and feedback (d reads d-1 and d+1: loops) | 0.378 | 0.339 | 1.000 | 1.000 | 0.679 |
| E1 | two parallel chains of 4 | 0.393 | 0.389 | 0.470 | 1.000 | 0.563 |

Sparser codes give both: with 4 winning columns the chain with shortcuts (A8 in batch 2: keep 0.470) keeps A, and reads code best so far
(as the dentate gyrus' sparse code separates similar contexts). Structured feedback loops do not break keeping (unlike random loops).
Parallel pathways keep only when each is long enough.

**Batch 6** (two pathways of different plan, `mesh_dag=5`, L=16 half-size): F2 (graph half K=1): moby 0.388, code 0.378, keep 1.000, mem
1.000 (0.692); F1 (K=2): 0.382 / 0.364 / 1.000 / 1.000 (0.686); F3 (full-size regions, M=8): 0.374 / 0.345 / 1.000 / 1.000 (0.680). A pure
chain beside a graph keeps A and reads new streams nearly as well as the graph alone; E6 (one graph, sparser) is as good and simpler.

**Batch 7** (sparseness): G3 = E6 with M=32 (twice the context cells): moby 0.395, code 0.409, keep 1.000, mem 1.000 (0.701; larger, not
an equal-size comparison); G1 E6 with 2 winners: 0.378 / 0.392 / 1.000 / 1.000 (0.693); G4 feedforward+feedback with 4 winners: 0.349 /
0.321 / 0.660 / 1.000 (0.583); G2 graph without loops with 4 winners: 0.381 / 0.393 / 0.470 / 1.000 (0.561).

**A fault of the battery.** A's start is 9 bytes before the answer, so `keep` can only be passed by an architecture with a path of at
least 8 regions: the test rewards a depth matched to it (the trap of choosing 8 regions for this test, again). From batch 8 on,
`keep4` and `keep2` (the same life with G=4 and G=2) are run beside it, and the score is read with all three.

**Batch 8** (keeping at G = 8, 4, 2; score = mean of the six):

| | architecture | moby | code | keep8 | keep4 | keep2 | mem | score |
|---|---|---|---|---|---|---|---|---|
| H1 | E6, seed 2 | 0.388 | 0.395 | 1.000 | 1.000 | 1.000 | 1.000 | 0.797 |
| H3 | E6 + stricter thresholds (0.9 / 0.75) | 0.382 | 0.389 | 1.000 | 1.000 | 1.000 | 1.000 | 0.795 |
| H4 | E6 with L=12, M=11 | 0.382 | 0.366 | 1.000 | 1.000 | 1.000 | 1.000 | 0.791 |
| H2 | E6, seed 3 | 0.379 | 0.385 | **0.540** | 1.000 | 1.000 | 1.000 | 0.717 |
| H5 | graph without loops, L=6 M=21, 4 winners | **0.396** | **0.402** | 0.470 | 1.000 | 1.000 | 1.000 | 0.711 |
| H6 | graph without loops (baseline) | 0.394 | 0.392 | 0.470 | 1.000 | 1.000 | 1.000 | 0.709 |

Keeping at gaps of 2 and 4 is general (every candidate); only G=8 separates them, and it is passed only by architectures with a long
serial path, and not on every seed (E6 seed 3: 0.540). A fixed depth keeps no context longer than its paths; beyond them the context is
only in the holders, which do not keep it through later text. The graphs without loops remain the best on the never-used streams.

**Batch 9 (queued): separation of long contexts by a sparse expansion of the holders, not by depth.** As the dentate gyrus separates
similar slow inputs by a very sparse, highly combinatorial code. On the graph without loops (H6, which fails keep8): holders' expansion
sparser and wider (`h_g_k` 0.005 / 0.002, `h_g_fan` 16).

**Batch 9** (sparse, wide expansion of the holders, on the graph without loops): I1 h_g_k 0.005 fan 16: moby 0.395, code 0.391, keep8
0.470 (score 0.709); I2 0.002: 0.392 / 0.392 / 0.470 (0.709); I3 0.005 with nG 8192: 0.396 / 0.396 / 0.470 (0.710); I4 0.005 with 4
winners in the context cells: 0.386 / 0.396 / 0.470 (0.709); keep4, keep2, mem 1.000 in all. Refuted: a sparser expansion of the holders
does not keep A at G=8 in the graph. My reading, not checked: at the gap the context compartments (taught by text on the shared letters)
outvote the holders, which do carry the branch.
