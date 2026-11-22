# NEWNET: a tissue designed from the requirements (21 November 2026)

Owner, 21 November 2026: "不要机械使用E63的程序，从头开始设计网络结构来适应任务"; "实验要短，快速迭代". Development level; one session; no outside review.

## The structure (`net2.py`, about 100 lines)

- Cells with random coordinates in a strip cut into regions along its length; region 0 receives the input lines, a cell of region d receives most of its connections from region d - 1 and a share `rec` from its own region and region d + 1 (recurrence).
- **In every region only the k cells with the largest input fire (output 1); the others are silent.** Every region regenerates a code of the same sparseness whatever reaches it: nothing decays from region to region and nothing floods (regional inhibition, all-or-none outputs).
- **A cell keeps its forward input and its recurrent input apart** (two compartments). A recurrent connection learns whether its sender's firing one step before predicts whether the cell's *forward* input alone would win now: dw = eta * pre(t-1) * (y_F(t) - mean of y_F), with a slow decay toward 0. Below a strength `w_on` it is silent (it learns, it does not drive). So a recurrent connection grows only where the stream is predictable, and cannot create the correlation it is judged by.
- One step a byte: region d at step t reflects region d - 1 at step t - 1.

## Test 1: is the byte carried deep? (`t2_depth.py`)

Independent random bytes over 32 values (no structure a decoder could borrow); for every region, the byte presented `lag` steps earlier is decoded from the region's code (nearest centroid, balanced accuracy, 4 contiguous folds); null = largest of 20 label shuffles. 12,000 steps, first 6,000 learning only. 6,144 cells, 48 regions of 128, 16 firing a region, 32 connections a cell, rec 0.25.

| Seed | Score at region 20 / 30 / 40 / 47 | Null at 47 | Farthest region above 1.5 x null | Regions whose best lag is not the region's index |
|---|---|---|---|---|
| 1 | 0.904 / 0.836 / 0.769 / 0.712 | 0.038 | 47 | 0 |
| 2 | 0.886 / 0.823 / 0.761 / 0.723 | 0.036 | 47 | 0 |
| 3 | 0.879 / 0.815 / 0.740 / 0.702 | 0.036 | 47 | 0 |

Confirmed on three seeds: **the byte is carried 47 connections deep**, relayed one region a step (the best lag equals the region at every region), with 16 cells firing in every region at every step.

## How the structure was reached (each one change, rounds of seconds)

| Change | Farthest region (24-region strip) |
|---|---|
| connections from all directions, random fixed strengths | 2 |
| forward only, but "upstream" defined as smaller x (it took cells of the same region) | 2 to 4 |
| region d from region d - 1 only, region 0 from the lines only | **23 (score 1.00 everywhere)**; 47 of 48 |
| plus 25% fixed recurrent connections | 5 |
| recurrent weaker (0.1 to 0.3 of forward) | 8 to 11 |
| recurrent learned against the cell's own firing (self-fulfilling: they grow on a random stream) | 3 to 5 |
| learned against the forward drive, no decay (a random walk held at 0 drifts up) | 5 to 7 |
| plus a slow decay | 23, but falling (0.45 at 12, 0.12 at 23) |
| plus silent below 0.1 | **23 (0.79 at 23)**; 47 of 48 |

## Not yet shown

That learned recurrence carries anything useful on a predictable stream; the output end; feedback; memory and keeping. Strengths of forward connections are random and fixed in test 1.

## Minicolumns (`net3.py`), and the output end (`t4_output.py`), 21 November 2026, later

Forward input chooses the winning columns of a region; context (connections from cells of the same region at the previous byte,
learned by the cell that fires, silent below `w_on`) chooses only which cell of a winning column fires. With `sweep=1` a byte
crosses every region before the next arrives; context comes from the previous byte. One seed (1), development level.

| Test | Result |
|---|---|
| Depth, 24 regions, random bytes and a 7-byte cycle | byte decodable at every region, 1.000, from columns and from cells, on both streams; context predicts the cell in 69% of winning columns on the cycle and 0% on random bytes (nothing to predict) |
| Output end, 12 regions, sweep, teach, 7-byte cycle | 0.89 right (chance 0.143) |
| Output end, yes/no only, 7-byte cycle, no sweep | 0.46 right: learns from yes or no alone |
| Context stream (A B C D / E B C F in random order), teach, output reading all 256 cells of the last region | 0.72 right overall (ceiling 0.875), 0.71 at the byte after C (0.5 without context) |
| The same, yes/no only | 0.53 overall, 0.51 after C: the context is not yet used |
| Is "two back was A" decodable from the last region's cells at C? | 0.98 |
| Random bytes (control) | at chance throughout |

Open: the output end does not yet use all the context the last region holds; yes/no learning of context; speed (25 to 40 s for 6,000 bytes of 24 regions, numpy).

### Context codes made stable (21 November 2026, later; confirmed on seeds 1, 2, 3)

The codes of a context never repeated: when no cell of a winning column was predicted, the least-used cell fired, so the cell
chosen rotated and context weights never settled (55% of winning columns predicted). Fix in `net3.py`: when no cell is above
`w_on`, the cell with the largest sub-threshold context sum fires; only with no context at all does the least-used cell fire.
Context codes now repeat (3 distinct codes per context), and 89% of winning columns are predicted.

`t4_output.py T=6000 L=12 sweep=1 fan_out=256`, last quarter:

| Stream, rule | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| context, teach: overall (ceiling 0.875) / at the byte after C (0.5 without context) | 0.868 / 1.000 | 0.873 / 1.000 | 0.878 / 1.000 |
| context, yes/no only: overall / after C | 0.70 / 0.907 | 0.689 / 0.907 | 0.678 / 0.909 |
| random bytes, yes/no (control, chance 0.031) | – | 0.027 | 0.033 |
| 7-byte cycle, teach / yes/no (seed 1) | 1.000 / 0.905 | | |

The output end now uses the context the last region holds, with the correct byte and with yes/no alone.

## Memory across a gap: surprise-gated hold (`net5.py`), 21 November 2026, later

**Task.** Stream of sequences `start_b m1 ... mG end_b`, branch b drawn at random; the end byte is known only from the start, G bytes
back; everything in between is the same for every branch. `t4_output.py stream=context G=.. NB=..`; decisive measure: right share
at the byte after mG, last quarter (1/NB without the start).

**What failed, measured (`t5_context.py`: branch decodable per region and position; `t6_codes.py`: new codes over a life).**

| Design | What happened |
|---|---|
| net3 (fixed random context sources, every connection of the firing cell decays) | G=2 right on one draw of byte values, 0.47 on another; a context gets 6-7 codes; at G=8 the branches merge within 3-4 bytes |
| net4: a cell makes its own context connections to the cells that fired, keeps those of other contexts | branch no longer fades, but codes split by the whole past (start after branch 0 differs from start after branch 1, and so on back); cells of a column fill up and branches merge again. Stricter matching or more cells did not help |

**net5.** Cell 0 of a column carries no context. A region signals surprise (a regional signal) when its prediction was broader than
what happened (precision of predicted columns below 3/4) or made by cells whose predictions often fail (each cell keeps a running
share of its predictions that came true). On surprise the winning columns fire their default cells and the region's hold cells take
this code; otherwise the hold persists. A cell's context has two parts, the previous byte's cells and the hold, each matched on its
own (the previous byte predicts, the hold chooses the cell); a learning cell makes connections to at most `n_grow` = 8 active sources
of each part, a random sample. The cell that would be chosen learns even on surprise (otherwise only one of two random starts gets a
predicted cell and its lucky predictions keep a stale hold). No window of past bytes: the hold is the activity of a few cells.

Fixes found by tracing one region (`trace_region.py`: per byte, surprise and the cells chosen): thresholds relative to the number of
active sources (an empty hold deadlocked fixed thresholds); per-part matching (a hold constant across the middle predicted every
middle byte and caused false surprises); learning on surprise. Larger codes (C=256, k=16) were not needed once these were in.

**Results**, `net=5 n_grow=8 fan_out=0 out_all=1` (the output end reads the cells of all 12 regions; reading only the last region
failed at G=16 because that region never settled while 7 others carried the branch perfectly), teach, last quarter:

| Case | seed 1 | seed 2 | seed 3 | overall right share (ceiling) |
|---|---|---|---|---|
| G=8, NB=2, T=6000 (output reads last region) | 1.000 | 0.987 | 1.000 | 0.95 (0.95) |
| G=16, NB=2, T=8000 | 1.000 | 1.000 | 1.000 | 0.97-0.98 (0.972) |
| G=32, NB=2, 64 byte values, T=12000 | 1.000 | 1.000 | 1.000 | 0.98-0.99 (0.985) |
| G=8, NB=4, T=8000 | 1.000 | 0.980 | 1.000 | 0.87-0.89 (0.925) |
| random bytes (control, chance 0.031) | 0.030 | | | |
| yes/no only, G=8 / G=16 (seed 1; with 10% exploration the ceiling is about 0.90) | 0.890 / 0.829 | | | |

Cost: about 155 s for 12,000 bytes at G=32 (numpy, one core). Open: speed; yes/no at long gaps; real text; keeping what was learned
when the stream changes; why the last region does not settle at G=16.

**Speed (same numbers).** One flat gather with a stored index, sums over slots in uint8, learning updates only on the rows that
change, no sampling when the sample would take every candidate: 10.7 → 3.8 ms a byte (12 regions), checked identical over 2,000
bytes against the committed version (`check_same.py REF`: cells, strengths, sources, reliabilities and holds compared with ==).

## Real text (tinyshakespeare, bytes as they are), 22 November 2026

`t4_output.py stream=text` reads the file's bytes from the start (65 distinct values; `n_values=128`, `C=128 lines_in=16` so that every
value wins its 8 columns in region 0; with C=64, lines_in=8 some values won none). Yardsticks, reference only (they keep counts of
the whole past and are not learners): the most frequent next byte given the last byte, and given the last two, counted online.

**The output rule was the first bottleneck.** Reading only region 0's columns (the present byte), the rule of the earlier tests
(dw = eta pre (target - gave), total held at 1) gave 0.13-0.18 against the last-byte table's 0.289: it changes only on errors and
does not estimate which byte is most probable. `rule=delta`: each output cell moves its own output towards its target
(dw = eta/n pre (target - O); yes/no: only the cell that answered, target 1 or 0). 0.281 from the present byte alone (one seed).
The context stream still gives 1.000 at G=8 with it.

**The context of a cell was the second.** With net5's context (the previous byte's exact cells, plus the hold), on text no context
ever repeats (every occurrence of ' ' had its own code, nothing was predicted) and the output stays at the present-byte level
(0.27-0.28 with 2 to 6 regions). Tried and not enough: `burst=1` (an unpredicted column fires its default cell; on text the broad
predictions then made 92% of steps surprising and every column fired its default cell). `ctx=1`: the previous-byte part of the
context is the previous byte's columns (every cell of a winning column counts), so a cell codes (present byte, previous byte);
with the hold still required it failed on both streams (the hold, refreshed on most bytes of text, chained again; the context
stream lost the branch: 0.61 at G=8). **`ctx=1 hold=0`** (order 2, no hold): the previous byte is decodable from region 0's
cells at 1.00 (chance 0.12; 11-13 codes for a present byte), and

| text, T=8000, teach, delta, last quarter | seed 1 | seed 2 | seed 3 | count tables (reference): last byte / last two |
|---|---|---|---|---|
| 1 region | 0.399 | 0.395 | 0.402 | 0.289 / 0.407 |
| 2 regions, M=16, S=64 | 0.406 | 0.406 | 0.403 | 0.289 / 0.407 |
| 1 region, T=16000 (seed 1) | 0.368 | | | 0.257 / 0.383 |

So the tissue now reaches the two-byte table on text. Open: beyond order 2 (more context than the previous byte without the chain
coming back); one tissue that does both the context stream (needs the hold) and text (the hold must not be required); yes/no on text.

**Beyond order 2: the order set by depth (`ctx=2`).** Region 0 takes the previous byte's columns as its previous-byte context;
region d > 0 takes region d-1's cells at the previous byte. Region d then codes the last d + 2 bytes, and since a region's context
never depends on itself the chain cannot come back. Text, T=8000, teach, delta, `hold=0`, last quarter (count tables, reference:
last byte 0.289, last two 0.407, last three 0.458):

| Setting | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| 3 regions, M=16, S=64 | 0.440 | 0.433 | 0.426 |
| 2 regions, M=8 | 0.426 | | |
| 3 regions, M=32 (200 s) | 0.441 | | |
| 4 regions, M=16 | 0.425 | | |

Above the two-byte table on three seeds, below the three-byte table; more cells or a fourth region did not add.
