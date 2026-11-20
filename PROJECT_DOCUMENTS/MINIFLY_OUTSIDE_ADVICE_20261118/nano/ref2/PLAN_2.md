# nano/ref2, plan 2: the systems of cells against the reference network, byte by byte (20 November 2026)

By the agent session that owns `nano/ref2`. Development level; no gate, seal or review is claimed. **Source commit:** `66b1bd4bbaebef8ad21a441f793301fb83bb6b04` (this branch; `nano/table` as at `0b1c9d9`). The word that set the work (Mr. Xu, in Chinese, exactly): "继续大胆尝试，小心求证。" (the coordinating session's translation: "Go on trying boldly, and verify with care."). Written while the four seeds of plan 1 are teaching: **no figure of the large network at step 4,000 exists yet.**

## 1. The question

The comparison the coordinating session could not make: on the last tenth of the test text (111,540 bytes), is a system of cells lower or higher than the reference network, by how much, with what error, and is the difference larger than the spread between the network's seeds?

## 2. The learners

* **The network:** plan 1's four seeds (10,745,088 numbers that learn each; 16.4 million bytes seen; each scored byte has 249 to 256 bytes before it, the every-eighth set exactly 256). Files `res/nll_s<seed>.f32`.
* **The systems of cells** are the coordinating session's, run unchanged by `cells.py` here, which imports `nano/table/gate_all.py`: `SysGate(x, without, 0)`, the stream read once (`go(6, cut, True)`), then the last tenth with nothing changing (`go(cut, len(x), False, True)`), the time standing at the end of the stream (the convention `PLAN_16.md` section 7 names). Two answering cells each (the first, and the second of `PLAN_15/16`):
  * `T2` (without companions, relay, since, pool; 10 kinds): reported 1.4973 and 1.4820; about 9 CPU-s, under 2 GB.
  * `big` (without relay; 164 kinds): reported 1.4740 and 1.4444; about 117 CPU-s, 4 to 7 GB.
  * `all` (every kind and the relay system): reported 1.4707 and 1.4421, the figure section 14 of the roadmap discusses; about 477 CPU-s. Not asked for by the task; added because it is the system whose figure was set against 1.454. Run only if memory allows.
  Their size as the coordinating session states it (`PLAN_16.md` section 7, not counted by me): for all kinds about 18 million numbers and the contexts of 4.2 million cells. They read the stream once (1.0 million bytes); the network sees 16.4 million bytes in random blocks.
* They are run beside the four teaching runs (which hold about 2 GB together; 12 GB are free), one at a time; each takes a core from the teaching for its minutes.

## 3. The analysis (fixed here, before any network figure)

For each system of cells c (each answering cell) and each network seed s, with l the per-byte losses in nats: d = l_c - l_s (negative: the cells are lower).
* the mean of d over all 111,540 bytes; its standard error as if bytes were independent (sd/sqrt(n)); and **the standard error from blocks**: the means of d over the 111 whole blocks of 1,000 bytes (the last 540 bytes left out of the error only), sd of those 111 means / sqrt(111). The blocks' figure is the one I read; neighbouring bytes are not independent.
* the same on the every-eighth set (bytes 0, 8, 16, ...: 13,943 bytes; 125 to a block).
* the spread between seeds: the sd of the four seeds' mean losses, and largest minus smallest; and d against the mean of the four seeds' per-byte losses.
* "Larger than the spread between seeds" is read as: the mean of d against the four-seed mean is beyond twice the sd between seeds **and** beyond twice its block standard error. A different text would give another answer; all of this is one text, one tenth, and that tenth is worn (`PLAN_16.md`, section 7, item 5).
* Nothing is tuned; the kept files are compared as they are. float32 files: their means are checked against the float64 means.

## 4. Predictions (chances are mine)

| | Prediction | Chance |
|---|---|---|
| C1 | `T2` gives 1.4973 and 1.4820, `big` 1.4740 and 1.4444, to the four digits | 0.95 |
| C1b | `all` runs here and gives 1.4707 and 1.4421 | 0.85 |
| C2 | `T2`'s second cell is above every seed on all bytes, by more than twice the block standard error | 0.9 |
| C3 | `all`'s second cell is below the four-seed mean on all bytes | 0.65 |
| C4 | ... and by more than twice the block standard error and twice the sd between seeds | 0.35 |
| C5 | `big`'s second cell is within twice the block standard error of the four-seed mean | 0.6 |
| C6 | the block standard error of d on all bytes is between 0.003 and 0.010 | 0.7 |

## 5. Not run

The system "parts by kind and byte" of `nano/layers` (also reported at 1.4421): its program is another line's. The cells with the time going on, or with everything going on through the tenth (a different measure: the network does not learn from the tenth). An ensemble of the seeds. Any other text.

## 6. Outcome

(appended below)

**Entry 1 (22:03 UTC): the systems of cells run here** (`cells.py`, beside the four teaching runs; `res/cells_*.json`, per-byte files `res/cells_<name>_cell{1,2}.f32`). Nothing in `nano/table` is changed.

| system | first cell, all bytes | second cell, all bytes | second cell, every eighth byte | named right (second cell), % | CPU | peak memory |
|---|---|---|---|---|---|---|
| `T2` (10 kinds) | 1.4972798 | 1.4820230 | 1.4835 | 55.39 | 6.5 s | 0.8 GB |
| `big` (without relay, 164 kinds) | 1.4740423 | 1.4444048 | 1.4475 | 57.06 | 84 s | 5.2 GB |
| `all` (164 kinds and the relay system) | 1.4707107 | 1.4420605 | 1.4463 | 57.06 | 402 s | 5.2 GB |

**C1 and C1b held**: the figures are the reported ones to the digits given (`PLAN_16.md` section 7 gives 1.4972798 and 1.4820230 for T2 and 1.4707107 and 1.4420605 for all kinds; 1.4463 on the every-eighth set). The float32 files' means agree with the float64 means to 1e-9. So the coordinating session's figures repeat on this machine, by its own program; this is a repeat run, not an independent implementation.
