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
