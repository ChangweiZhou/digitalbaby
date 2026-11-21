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
