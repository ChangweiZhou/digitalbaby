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

**Entry 2 (02:12 UTC, 21 November): the comparison** (`compare.py`, `res/compare.json`, `res/compare.txt`; a few CPU-seconds). d = cells - network in nats a byte; negative: the cells are lower. Network: plan 1's four seeds, mean 1.4590 on all bytes (sd between seeds 0.0011) and 1.4653 on the every-eighth set (sd 0.0078). Both sides with nothing changing.

| cells (answering cell) | cells, all bytes | d against the four-seed mean, all 111,540 bytes | s.e. as if independent | **s.e. from blocks of 1,000** | d against each seed (1337, 1338, 1339, 1340) | d on the every-eighth set (13,943), s.e. from blocks | share of the 111 blocks in which the cells are lower |
|---|---|---|---|---|---|---|---|
| `T2`, first | 1.4973 | +0.0382 | 0.0026 | 0.0059 | | +0.0334, 0.0094 | 0.14 |
| `T2`, second | 1.4820 | **+0.0230** | 0.0025 | **0.0060** | +0.0233, +0.0232, +0.0240, +0.0214 | +0.0182, 0.0094 | 0.18 |
| `big`, first | 1.4740 | +0.0150 | 0.0027 | 0.0056 | | +0.0107, 0.0093 | 0.26 |
| `big`, second | 1.4444 | **-0.0146** | 0.0027 | **0.0057** | -0.0143, -0.0144, -0.0136, -0.0163 | -0.0178, 0.0094 | 0.53 |
| `all`, first | 1.4707 | +0.0117 | 0.0027 | 0.0059 | | +0.0079, 0.0095 | 0.29 |
| `all`, second | 1.4421 | **-0.0170** | 0.0027 | **0.0059** | -0.0166, -0.0168, -0.0159, -0.0186 | -0.0190, 0.0095 | 0.57 |

Against single seeds the block standard errors are 0.0054 to 0.0069. Seed against seed on all bytes: differences of 0.0001 to 0.0027 with block standard errors of 0.0018 to 0.0027.

**Said plainly (my reading).**
* With nothing changing, **the system with all kinds and its second cell is lower than the reference network by 0.017 nats a byte, with a standard error of 0.006 from blocks** (2.9 standard errors), against every seed (0.016 to 0.019), and the difference is fifteen times the sd between seeds (0.0011). The system without relay: 0.015 lower (2.5 standard errors). By the rule written in section 3 both are "larger than the spread between seeds".
* **`T2` is higher than the network** by 0.023 (3.8 standard errors), on every seed. **The first cells are all higher** than the network (0.012 to 0.038): the systems are below the network only with the second answering cell.
* **The margin is not even over the text**: the cells with all kinds are lower in 57% of the 1,000-byte blocks only; the mean difference comes from a minority of blocks where they are much lower. The standard error as if bytes were independent (0.0027) understates the error by more than half.
* On the every-eighth set the difference is the same size (0.019) with a larger error (0.0095: 2.0 standard errors), and that set flatters or hurts the network by seed (plan 1, entry 3); the earlier "1.4463 against 1.4541, 0.008 lower" was against the network's most fortunate figure.
* **Limits.** One text and one tenth, and that tenth is worn on the cells' side (about seventy-five variants scored on it, `PLAN_16.md` section 7); the network's setting was fixed without it. The blocks' error describes this tenth's bytes, not other texts. The network is this shorter schedule (16.4 million bytes seen), not nanoGPT's published one, which this line has not run: how much lower a longer schedule goes is not known here. The cells' figure uses the convention that the time stands at the end of the stream. Sizes are not alike (about 18 million numbers and the contexts of 4.2 million cells, as the coordinating session counts them, against 10.7 million numbers). I have rerun the cells' program, not written one of my own.

**Predictions.** C1, C1b held (entry 1). C2 held (+3.4 to +4.5 standard errors on the four seeds). C3 held. **C4, to which I gave 0.35, held.** **C5 failed** (the system without relay is 2.5 standard errors below, not within 2). C6 held (0.0059).
