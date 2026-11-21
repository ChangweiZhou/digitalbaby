# nano/ref2: the reference network with seeds and kept answers, and the systems of cells against it (20 to 21 November 2026)

By the agent session that owns `nano/ref2`. Development level; four seeds of one setting on one text; no gate, seal or review is claimed. Nothing outside this folder was changed. The plans were committed before their runs and hold the full outcomes: `PLAN_1.md` (the network again), `PLAN_2.md` (byte by byte against the cells), `PLAN_3.md` (both going on; asked for by the coordinating session in place of a longer schedule). Readings are mine.

All figures: nats a byte on the last tenth of tiny Shakespeare (111,540 bytes). Network: `nano/ref`'s large setting unchanged (10,745,088 numbers, batch 16, 4,000 steps, 16.4 million bytes seen; not nanoGPT's published schedule), seeds 1337 to 1340, one thread each, 12,160 CPU-s of teaching a seed. Each scored byte has 249 to 256 bytes before it; the every-eighth set (13,943 bytes) exactly 256.

| | nothing changing | going on (each byte answered, then learned from) |
|---|---|---|
| network, mean of four seeds (sd) | 1.4590 (0.0011); every-eighth set 1.4653 (0.0078) | 1.3426 (0.0006) with a step every 64 bytes; 1.3553 (0.0019) every 256 |
| cells `T2`, second cell | 1.4820: **+0.023** (s.e. 0.006) | 1.3717: **+0.029** (0.005) |
| cells without relay, second cell | 1.4444: **-0.015** (0.006) | 1.3192: **-0.023** (0.005) |
| cells with all kinds, second cell | 1.4421: **-0.017** (0.006) | 1.3204: **-0.022** (0.005) |

Differences are cells minus network on all bytes, with the standard error from blocks of 1,000 bytes (going on: against the 64-byte window).

1. The copy teaches exactly as `nano/ref`'s program: seed 1337 on four threads repeats its curve to the last digit at step 500. A different number of threads changes the rounding and gives a neighbouring draw.
2. On all bytes the setting is steady (1.458 to 1.461). The earlier 1.454 was measured on the every-eighth set, which is answered at the block's last place; there the seeds run from 1.457 to 1.473 and 1.4541 is below all four.
3. The cells' figures repeat here, run by their own program unchanged.
4. Like against like, the systems of 164 kinds with the second answering cell are lower than this network by about 0.02 nats a byte, three to five standard errors, on every seed, both with nothing changing and going on; `T2` is higher by 0.02 to 0.03; every first cell is at or above the network.
5. Not measured: nanoGPT's published schedule (how much lower a longer run goes); the full 256 bytes before every byte; any other text; any tuning of the going-on procedure. The last tenth is worn on the cells' side. Sizes are not alike.

Files: `res/nll_s<seed>.f32`, `res/dyn_s<seed>_W<W>.f32`, `res/cells_<system>_cell<1|2>[_going_on].f32` (float32, one loss a byte, 446 kB each); `res/*.json`, `res/compare.txt`. Programs: `gpt.py` (the copy), `cells.py`, `compare.py`, `dyn.py`, `jobs1.sh`.
