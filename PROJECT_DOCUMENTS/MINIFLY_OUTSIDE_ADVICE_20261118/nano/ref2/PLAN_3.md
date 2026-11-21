# nano/ref2, plan 3: the network going on through the last tenth (20 November 2026, 22:50 UTC)

By the agent session that owns `nano/ref2`. Development level; a first reading; no gate, seal or review is claimed. **Source commit:** `97e2853db59c888859a6ceb888179e58054a186b`. The word that set the work (Mr. Xu, in Chinese, exactly): "继续大胆尝试，小心求证。" (the coordinating session's translation: "Go on trying boldly, and verify with care."). **This step was asked for by the coordinating session** in a message of 22:42 UTC, in place of the longer schedule of the first brief's step 3; it said itself that the request is not new words of Mr. Xu. Its reason as it gave it (I have not read its roadmap section 20): the systems of cells are now reported by their loss while going on through the last tenth (each byte answered before it is learned from), and the network has only been scored with nothing changing, so the two were not set against each other alike. Written while the four seeds teach: no figure of the network at step 4,000 exists yet.

## 1. The question

How low is the reference network on the last tenth when it, too, goes on learning through it (each byte answered before it is learned from), and how does that stand against the systems of cells going on?

## 2. The procedure (`dyn.py`; fixed here, nothing chosen on the last tenth beyond W)

The network of each seed at step 4,000 (its checkpoint, with the optimiser's state). The last tenth is read in order in windows of **W = 64 and W = 256** bytes (two separate passes, each from step 4,000). For each window:
1. **Answer:** with the strengths as they then are, evaluation mode, every byte of the window is answered as in plan 1's score: 249 to 256 bytes before each byte. No byte is answered by strengths that have learned from it or from a later byte.
2. **One step of teaching on the window:** a batch of one block; the block's last W places name the W bytes of the window (for W = 256 an ordinary block of the schedule; for W = 64 the 192 places before them are context only and carry no loss, so that each byte is learned from once); dropout 0.2 as in the schedule; the same AdamW (0.9, 0.99; weight decay 0.1), **its running averages carried over** from step 4,000; **rate 1e-4**, the rate at the schedule's end; gradient clipped at 1.

My choices, named: batch of one (the schedule's batch is 16 random blocks; here there is one new block); the loss on the new bytes only; the averages carried over; dropout left on in the step. **The one alternative, used only if rate 1e-4 gives a going-on loss above the same seed's loss with nothing changing: rate 3e-5.** No other setting is tried.

The systems of cells going on: `cells.py` now also keeps the per-byte losses of nano/table's own going-on pass (`go(cut, len(x), True, True)` after the frozen scoring, as `gate_all.run_text` does; reported there: T2 1.3863 / 1.3717, without relay 1.3462 / 1.3192, all kinds 1.3470 / 1.3204).

**The comparison** as in `PLAN_2.md` section 3 (per-byte differences, standard error from blocks of 1,000 bytes), for going on against going on. Not alike in one respect that I cannot remove: a cell learns from every byte at once (a window of one byte); the network takes one step per W bytes.

**Cost** (timed on the step-500 checkpoint, a few windows): a teaching step 0.26 CPU-s; a pass with W = 64 about 20 minutes of one core, with W = 256 about 15. All four seeds if the time allows, seed 1337 first. The cells: as plan 2, about 8 minutes of one core.

## 3. Predictions (mine)

| | Prediction | Chance |
|---|---|---|
| D1 | going on with W = 256 at rate 1e-4 is below the same seed's loss with nothing changing | 0.85 |
| D2 | the gain is between 0.01 and 0.06 nats | 0.6 |
| D3 | W = 64 is below W = 256 | 0.75 |
| D4 | the cells with all kinds going on (reported 1.3204) are below every seed of the network going on, by more than twice the block standard error | 0.8 |
| D5 | the cells' going-on figures repeat here to the four digits | 0.95 |

## 4. Not run

The longer schedule (the first brief's step 3). Any search over the rate, the window, the number of steps a window, or a reset of the optimiser. Going on with random blocks of the stream mixed in. Any other text.

## 5. Outcome

(appended below)

**Entry 1 (22:50 UTC): the cells going on, run here** (`cells.py`; per-byte files `res/cells_<name>_cell{1,2}_going_on.f32`; the frozen files came out bit-identical to plan 2's).

| system | first cell going on | second cell going on | second cell, every eighth byte | CPU (frozen and going on) |
|---|---|---|---|---|
| `T2` | 1.3863421 | 1.3717006 | 1.3734 | 7 s |
| `big` (without relay) | 1.3462186 | 1.3191779 | 1.3215 | 94 s |
| `all` | 1.3470264 | 1.3204424 | 1.3237 | 402 s |

**D5 held** (reported: 1.3863 / 1.3717, 1.3462 / 1.3192, 1.3470 / 1.3204).

**Entry 2 (02:45 UTC, 21 November): the network going on, all four seeds** (`dyn.py`; `res/dyn_s<seed>_W<W>.json` and `.f32`; the comparison in `res/compare_going_on.json`). Rate 1e-4 as planned; it made no seed worse, so **the alternative rate was not used**. A teaching step costs 0.25 CPU-s (a batch of one block); a whole pass 1,180 CPU-s with W = 64 (1,743 steps) and 870 with W = 256 (436 steps), nearly all of it the answering.

| seed | nothing changing (plan 1) | going on, W = 64 | named right, % | going on, W = 256 | named right, % |
|---|---|---|---|---|---|
| 1337 | 1.4587 | 1.3432 | 58.80 | 1.3524 | 58.49 |
| 1338 | 1.4588 | 1.3421 | 58.66 | 1.3564 | 58.40 |
| 1339 | 1.4580 | 1.3431 | 58.76 | 1.3562 | 58.48 |
| 1340 | 1.4607 | 1.3420 | 58.64 | 1.3561 | 58.35 |
| **mean; sd** | 1.4590; 0.0011 | **1.3426; 0.0006** | 58.72 | **1.3553; 0.0019** | 58.43 |

The gain from going on is 0.116 nats (W = 64) and 0.104 (W = 256), with block standard errors of 0.014 and 0.012: it is very uneven. By thirds of the tenth (four-seed mean, W = 64): 0.048, 0.056 and **0.246**; the last third is where most is gained. The every-eighth set gives the same figures within 0.006 (1.343 to 1.345 at W = 64).

**Going on against going on** (d = cells - the four-seed mean of the network, all 111,540 bytes; s.e. from blocks of 1,000):

| cells | cells going on | d against W = 64 | s.e. | d against each seed | share of blocks with the cells lower | d against W = 256 | s.e. |
|---|---|---|---|---|---|---|---|
| `T2`, second cell | 1.3717 | +0.0291 | 0.0045 | +0.029 to +0.030 | 0.17 | +0.0164 | 0.0067 |
| `big`, first cell | 1.3462 | +0.0036 | 0.0047 | | 0.36 | -0.0091 | 0.0071 |
| `big`, second cell | 1.3192 | **-0.0234** | 0.0047 | -0.023 to -0.024 | 0.70 | -0.0361 | 0.0073 |
| `all`, first cell | 1.3470 | +0.0044 | 0.0046 | | 0.35 | -0.0082 | 0.0071 |
| `all`, second cell | 1.3204 | **-0.0221** | 0.0047 | -0.022 to -0.023 | 0.71 | -0.0348 | 0.0072 |

**Said plainly (my reading; a first reading).** Going on helps the network about as much as it helps the cells (0.116 against the cells' 0.122 for all kinds), so the order does not change: with both going on, the systems with 164 kinds and the second cell are lower than the network by 0.022 to 0.023 nats a byte against its better window (W = 64), 4.7 to 4.9 block standard errors, on every seed, in about 70% of the blocks; `T2` is higher by 0.029; the first cells of the large systems are level with the network (within one standard error). Setting a going-on cell system (1.32) against the stopped network (1.459) overstated the margin about sixfold; setting like against like, the margin is of the same size as with nothing changing (0.017), a little larger.

**Limits.** One rate, one step a window, two windows, nothing searched: the network's going-on figure is what this first procedure gives, not the best a network can do going on (more steps a window, a tuned rate, or blocks of the stream mixed in were not tried, and any of them tuned on this tenth would spoil it). The cells learn from each byte at once; the network once per 64 or 256 bytes, and the smaller window is the better, so a still smaller window might be better again (not run; W was fixed beforehand). One text, one tenth. The network is the shorter schedule.

**Predictions.** D1 held. **D2 failed**: I expected a gain of 0.01 to 0.06; it is 0.10 to 0.12. D3 held on all four seeds. D4 held (4.7 standard errors against W = 64). D5 held (entry 1).

The checkpoints were deleted after these passes (no strengths are kept; a rerun of plan 1's commands rebuilds them, 3.4 h a seed on one thread).
