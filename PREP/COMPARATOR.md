# The comparator on the stream of one life (`PREP/comparator.py`)

**What it is.** The comparator is the positive control of the roadmap's condition 5: a network trained by back-propagation on the same stream as the tissue, counted in the same units (items read and arithmetic operations). This program prepares it and times it on the stream of one life, `PREP/life_stream.py`, `life(seed)` (E52's worker, commit `f4d8f847`). It has been run only on the first 8,192 bytes of seed 73129. **The full run is the coordinator's to release and has not been started.**

    python3 PREP/comparator.py run 73129 --bytes 8192        # the trial; omit --bytes for the whole life

**The network.** It has three parts:
- a table of 32 numbers for each of the 256 byte values;
- a GRU of 128 units (torch's `nn.GRU`, one layer);
- a linear readout to 256 classes, one per byte value.

That is 103,424 parameters in all, and 94,208 multiply-adds for one position of the forward pass.

**Reading the stream.** The stream is cut into items, each ending at the end-mark byte 10. A stretch without an end mark, such as a quiet stretch of 4,096 spaces, is one item.
- For an item covering positions s to e, the network reads the bytes at s − 1 to e − 1. At position 0, which has no byte before it, it reads the byte 10, as if an item had just ended.
- Its output at each position p is its guess for the byte at p. The byte it gives is the most likely one, recorded before the update that uses the item, so it is the network's answer before the byte is shown.
- The loss is the mean cross entropy of the item's outputs, and one optimiser step follows each item.
- The state is carried from item to item and detached at the item's end, so the gradient is cut there.

**Optimiser.** AdamW with rate 0.001, betas (0.9, 0.98) and eps 1e-8. The weight decay is the record's (`exp26_candidate.make_opt`): 0.25 on every weight except the byte table, 0 on the byte table. The reviewer's message does not name a weight decay; this is my choice, taken from the record, and `Config.weight_decay` changes it.

**The record.** `comparator_runs/comparator_<seed>_<N>.npz` holds:
- `given` (uint8, the byte given at every position);
- `shown` (the life's bytes);
- `prob` (the probability given to the byte shown);
- `loss_item` and `item_end` (one value per item).

A scorer reads `given` against `shown` at the positions that `life(seed)["due"]` marks, as for the tissue. The `.json` beside it carries the settings, the counts, the CPU time and the forecast.

**The trial: the first 8,192 bytes of seed 73129, one thread** (`comparator_runs/comparator_73129_8192.json`).

| Measure | Value |
|---|---|
| Items read | 519 |
| Optimiser steps | 519 |
| Multiply-adds | 2.32e9 (forward 0.77e9; backward counted as twice the forward) |
| CPU time | 1.20 s, 1.46e-4 s a byte |
| Bytes given rightly at the first attempt | 1,433 of 8,192 (17.5%) |
| Mean item loss, first 100 items | 4.09 |
| Mean item loss, last 100 items | 2.44 |

The record has the right form: `given`, `shown` and `prob` have 8,192 entries each, and `loss_item` and `item_end` have 519.

**Timing a long item.** A separate timing, on random bytes and not a run of the life, measured the forward pass, backward pass and step:
- on an item of 16 bytes: 1.49e-4 s a byte;
- on an item of 4,096 bytes, as long as a quiet stretch: 1.15e-4 s a byte.

Long items therefore cost no more per byte, and the trial's figure can be carried to the whole life.

**Forecast, one thread, from the trial's 1.46e-4 CPU-s a byte.**

| Read | Bytes | Items | CPU time | Multiply-adds |
|---|---|---|---|---|
| The whole life | 704,541 | about 64,900 (64,892 end marks, plus the quiet stretches) | about 103 s (0.03 CPU-hours) | 1.99e11 |
| Its first 262,144 bytes | 262,144 | about 16,370 | about 38 s | 7.41e10 |

These figures exclude making the stream with `life(seed)`, which takes a few seconds.

**What it does not do.** It does not score the due positions, which is the scorer's work and is shared with the tissue. It does not run the whole life or other seeds. It does not vary the size, the rate or the cut of the gradient.
