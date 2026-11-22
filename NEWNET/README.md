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

## One tissue for both streams (`net6.py`), 22 November 2026

Two kinds of area given at birth, side by side on the same input lines, read by one output end (fixed architecture, reported
apart from what is learned): order areas (`ctx=2, hold=0`, 3 regions, M=16, S=64) and hold areas (net5's surprise-gated hold,
`burst=1`, 6 regions). Output end: delta rule, teach, all cells of both. Last quarter:

| Case | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| order areas alone, text, eta_out 0.2 (count tables, reference: 0.289 / 0.407 / 0.458) | 0.453 | 0.447 | 0.441 |
| both, context stream G=16, at the byte after the gap | 1.000 (eta 0.1) | 1.000 | 1.000 |
| both, context stream G=8 / G=32 (64 values), seed 1 | 1.000 / 1.000 | | |
| both, text, eta_out 0.1 / 0.2 / 0.3 | 0.393 / 0.405 / 0.404 | 0.393 (0.1) | |
| both with 12 hold regions, no burst, text, eta 0.1 | 0.381 | | |

One tissue now carries a branch across 32 predictable bytes and passes the two-byte table on text, but on text the hold areas
cost 0.04-0.05 against the order areas alone; a larger output rate does not recover it, so it is interference of the hold
areas' activity at the output end, not its rate. Cost: about 190 s for 8,000 bytes of text with both (too slow for the 2-minute rule).

## Speed, second pass (22 November 2026)

Where the time went (text, both kinds of area, per byte): output end 15 ms (dense update of 128 x 14,000 strengths although about 72
cells are active), each tissue part 2.7 ms (numpy over every slot). Changes:

1. **Output end** (`t4_output.py`, when it reads every cell): only the columns of the active cells are read and changed. Same answers at
   every step, compared over 3,000 bytes on text (teach) and on the context stream (yes/no) against the previous version.
2. **Tissue step in C** (`net7.c`, `net7.py`; the forward part and the hold stay in numpy). Same definition as net5, with ties among
   equally weak slots going to the lowest slot; `net5.py stable=1` does the same in numpy (its default is unchanged and checked
   identical with `check_same.py`). `check_net7.py`: cells, surprise and predicted share identical to `net5 stable=1` at every step
   over 3,000 bytes in four settings without random sampling. Where more sources are active than `n_grow` (order areas), the sample
   uses C's own generator, so those runs are not step-identical to numpy; they were re-run instead. `net6.py fast=1` uses it.

| Run (one seed) | before | now | result before / now |
|---|---|---|---|
| text, both kinds of area, 8,000 bytes | 194 s | 21 s | 0.405 / 0.407 |
| text, order areas only, 8,000 bytes | about 90 s | 9 s | 0.453 / 0.447 |
| context stream G=16, both, 8,000 bytes | about 90 s | 10 s | 1.000 / 1.000 |
| context stream G=32, 12,000 bytes, old output rule | 155 s | 61 s | 1.000 / 1.000 |

What is left is spread evenly (C scan of the slots, numpy forward part, Python per byte, about a third each, about 1 ms a byte for 12
regions). The old output rule stays dense (it renormalises every strength every byte); new work uses the delta rule.

## Keeping: A -> B -> A in one life (`t8_life.py`), 22 November 2026

**The life.** A: the context stream (G=8, two branches). B: tinyshakespeare from its start (8,000 bytes unless stated). Then A
again (200 sequences). Nothing is reset and no task signal is given. Tests with learning off, at the end of A, after B and after the
return: A at the byte after the gap, over 100 fresh sequences (chance 0.5); text over 2,000 bytes never read (from byte 600,000).
Diagnostics: the output end of before B put back (`put back`); the branch decoded from the tissue's cells at the gap (no output end);
`t9_where.py`: for each area and region, whether the cells at the gap are those of before B, the predicted share over A's middle and
surprise at its start. Output end: `rule=gated` (the right byte's cell and any cell that spoke wrongly are taught; a silent wrong cell
is not, PLAN_12's letter r in `nano/lives`); `rule=delta` gave the same in every case run with both.

**What was found, in order** (net6 fast=1, order areas 3 regions, hold areas 6 regions burst=1):

1. A was lost after 8,000 bytes of text (0.47), and *reading* 2,000 bytes of text with learning off already lost it. The cause was a
   deadlock in the hold areas: after text the hold held a text code, so nothing of A was predicted, so no surprise (it needed a
   broad prediction), so the hold was never refreshed. **`recall=1`**: surprise also when most winning columns were not predicted.
   Then A was kept on seeds 1 and 3, not on seed 2 (codes drifted), and not after 32,000 bytes of text.
2. After 32,000 bytes the cells of A still had their strong connections (14 to 16 of 16), but some were also used by text contexts,
   whose predictions failed; the reliability kept by the cell (net5) fell to 0.03-0.13, and on A's return the region read their
   predictions as unreliable and refreshed the hold in the middle. **`lam=0`** (no reliability; learning on surprise, added
   earlier, already covers what it was for): A kept on all three seeds, after 8,000 and after 32,000 bytes. The context stream alone
   still gives 1.000 (G=8 hold areas alone, G=16 both).
3. With A on **lowercase letters** (bytes text uses too) the hold areas lost A again: after `b` the region also predicts text's
   successors of `b`; the prediction is broad, so surprise, and the hold is refreshed. For that region this is a real ambiguity.
   The order areas kept A's cells exactly (1.00). **8 order regions** (`o_L=8`: region 7 codes the last 9 bytes, which reach the
   start from the gap) keep A.

| `rule=gated recall=1 lam=0`; A after B (learning off) / tissue branch after B / text after B -> after the return | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| disjoint bytes, B 8,000 | 1.000 / 1.00 / 0.287 -> 0.293 (lam 0.1) | 1.000 / 1.00 / 0.287 -> 0.283 | 1.000 / 1.00 / 0.280 -> 0.280 (lam 0.1) |
| disjoint bytes, B 32,000 | 1.000 / 1.00 / 0.322 -> 0.323 | 1.000 / 1.00 / 0.317 -> 0.319 | 1.000 / 1.00 / 0.328 -> 0.321 |
| letters, 3 order + 6 hold regions | 0.510 / 0.50 | 0.000 / 0.50 | |
| letters, 8 order + 6 hold regions | 1.000 / 1.00 / 0.292 -> 0.279 | 1.000 / 1.00 / 0.285 -> 0.273 | 1.000 / 1.00 / 0.287 -> 0.277 |
| letters, 8 order regions, no hold areas | 1.000 / - / 0.325 -> 0.327 | 1.000 / - / 0.318 -> 0.329 | 1.000 / - / 0.321 -> 0.331 |
| letters, 8 order regions, no hold, B 32,000 | 1.000 / - / 0.342 -> 0.342 | | |

On the return, A is right from the first block of 20 sequences in every kept case. Text read alone with 8 order + 6 hold regions:
0.402 (one seed). Run time 50-120 s. Open: what the hold areas are for once order areas are deep (they are needed for gaps longer
than the order reaches: G=16, 32); why they cost text about 0.01 on the return; two texts (A and B both real text).

## Two real texts in one life (`t10_texts.py`), 22 November 2026

A: tinyshakespeare from its start; B: Austen's *Emma* from byte 50,000; then A again where it stopped (2,000 bytes). Tissue:
8 order regions, no hold areas (net6 fast=1 o_L=8 h_L=0). Output end gated, constant step 0.2. Tests with learning off on 2,000
bytes of each text never read. Control: the same life with no A (TA=0): what reading only B gives on the A test.

| 24,000 bytes each | A test after A | A test after B | control (B only) on A | B test after B | A after B with the output end of after A put back | after the return (A / B) |
|---|---|---|---|---|---|---|
| seed 1 | 0.369 | 0.320 | 0.296 | 0.398 | **0.358** | 0.332 / 0.392 |
| seed 2 | 0.357 | 0.331 | 0.324 | 0.392 | **0.369** | 0.340 / 0.396 |

With 8,000 bytes each the A-specific part was too small to measure (seed 1: 0.326 / 0.325 / control 0.314).

**Reading.** B costs most of what A adds over the control (seed 1: 0.073 above it after A, 0.024 after B; seed 2: 0.033 and 0.007).
It is lost at the output end, not in the tissue: with the output end of after A put back on the tissue after B, A is back at 0.358
and 0.369. The two texts share their contexts (the same English byte sequences, the same cells), and the output end's constant
step makes its reading of a context follow the latest text. Nothing of B is lost on the return.

Tried at the output end, seed 1 unless stated (A test after A / after B; B after B), none of which helps:

| Output end | A after A | A after B | B after B |
|---|---|---|---|
| constant step (above) | 0.369 | 0.320 | 0.398 |
| rehearsal, a store of 1,000 episodes of 40 bytes (every episode read has the same chance to be kept), rho 1/100 | 0.364 | 0.328 | 0.407 |
| the same, rho 1/20 | 0.366 | 0.327 | 0.406 |
| each connection steps 1/n (n: its own updates) | 0.321 | 0.339 | 0.338 |
| the same, at least 0.01 | 0.327 | 0.347 | 0.361 |
| fast (constant) plus slow (1/n) connections on one error, seeds 1 / 2 | 0.367 / 0.350 | 0.316 / 0.332 | 0.400 / 0.391 |

Open: a reading that depends on which text is being read needs the tissue to carry something of the text over a longer range
than 9 bytes (vocabulary, style), so that the two texts' contexts are not the same cells.

**A slow code of which text is read (tried, not enough).** Each column keeps a slow trace of its own activity (time constant tau)
and a long-term mean (20,000); the code is the 8 columns most above their mean. On its own it tells the two texts apart (nearest
centroid 0.85 at tau 1,000, 0.80 at 300, chance 0.5) and changes in 7% of bytes (tau 1,000). (a) As the required second part of the
order areas' context (`net7 slow=1000`): contexts split and learning suffers (A after A 0.274 and 0.265 against 0.369 and 0.357;
tau 300: 0.293). (b) As 4 extra "style" regions beside the plain 8 (`net6 s_L=4`): A after A / after B / control: seed 1 0.359 /
0.317 / 0.313, seed 2 0.363 / 0.325 / 0.306: the A-specific part is still mostly lost; 6 minutes a run.

## net8: a tissue taught by its own mismatch (22 November 2026)

Designed from the task and from biological principles, with what was learned from the outside session's work taken in as principles
only (its programs and structure are not used): the tissue must receive the mismatch, holding must be learned, and a number that
changes only while its own sender is active is not overwritten by another task. `net8.py`, `t11_gate.py` (numpy, 256 cells, 1-20 s a run).

- Cells with a state and a gate (hold or take in), both reading the present byte's line, 16 other cells one byte before, a constant.
- Output cells, one per byte value, read the cells (delta rule; a silent wrong output cell is not taught).
- Feedback cells carry each output cell's mismatch to the region; a tissue cell weighs it with its own feedback strengths, which
  change by the same local product as the output strengths (`fb=kp`), or stay as born (`fb=random`).
- Every strength keeps a trace, run forward with its own cell's state, of how it shaped that state; it changes by the mismatch its
  cell receives times the trace. No pass backward; a strength whose sender was silent has no trace.
- Two general properties were needed, both found by measurement:
  (1) **cells born with different clocks** (`tau_max=150`: the gate's constant gives time constants log-uniform from 1 to 150 bytes);
  with every gate near one half at birth, nothing of the start reaches the end of a gap of 8 and no trace reaches back to it;
  (2) **adaptation** (`center=1000`): a cell sends its state less its own running mean. The branch was decodable from the cells'
  states at the gap (nearest centroid 1.00) but as differences of 0.02 to 0.05 on a large common part, which the output end could
  not learn. With both, the tissue's rate had to be raised from 0.05 (whose changes were too small to matter: about 4e-4 an event).

Context stream, at the byte after the gap, last quarter (chance 0.5):

| G, life | learning tissue and gates (`fb=kp eta_t=0.5`) | gates as born | no tissue learning | feedback as born |
|---|---|---|---|---|
| 8, 12,000 (eta_t 0.05) | 1.000 | 0.977 | 0.927 | 1.000 |
| 16, 16,000 | 0.757 | | 0.586 (eta 0.05) | 0.617 |
| 16, 40,000 | **1.000, 0.989, 0.998** (seeds 1-3) | 0.633 | 0.665 | |
| 32, 80,000 (64 values) | 0.685, still rising | | 0.605 | |

Controls on seed 1 only. Not yet run: text, keeping, yes/no, more cells, more than one region.
