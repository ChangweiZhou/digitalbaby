# E16-RC report: capacity curve of the raw-byte reservoir on the standing seeds (9 November 2026)

Brief: `OPUS_TASK_E16RC_RAW_CAPACITY_CURVE_20261108.md` (commit 7804abe). Predictions: `E16RC_PREDICTIONS.md`, committed at d309b204 before any life was launched and not changed since. Harness unchanged (`exp16_integrated.py` at 7804abe). Tables: `exp45/SUMMARY_E16RC.md`, written by `e16rc_tables.py`. Branch: `claude/vibrant-gauss-tops5f`. That is this session's assigned branch; it is identical to the brief's commit, and this session may not push to `codex/minifly-consolidated-20261023`.

**Outcome.** Two of the four predictions hold, and two miss.

- **Byte size (P16RC.4) holds.** Every slot carries 16.0 bytes.
- **Composition count (P16RC.3, second clause) holds.** At 131,072 slots the reservoir holds about 20,500 distinct composition byte strings, under the 24,000 taught triples.
- **Monotonicity (P16RC.1) misses.** Accuracy does not rise with capacity in any seed except seed 73129's compositions.
- **The gate's lower edge (P16RC.2) misses.** 16,384 slots already meets the 0.80 composition ratio in two seeds of three.
- **Saturation (P16RC.3, first clause) misses in one seed.** At 131,072 slots, seed 116089 is 3.6 points below the every-entry arm.

Read together: from 16,384 to 131,072 slots (0.26 MB to 2.1 MB), retention after block 6 varies little. The variation is about the size of the differences between seeds. Only the lowest capacity in one seed (93683 at 16,384, ratio 0.76) falls below the gate. There is no clean logarithmic curve for the raw arm on these seeds. The smallest budget that met the gate in every seed here is 65,536 slots (1.05 MB).

## Verdicts

Measures as in the standing reports, after block 6:
- **Operations:** mean exact accuracy at k = 12 over the family's four operations.
- **Compositions:** mean final-answer accuracy over the four held pairs.
- **Ratios:** to the twin `e16_life_D_<seed>.json`.

The 65,536 values are E16-R3's receipts, which were not rerun.

### P16RC.1 (monotone in capacity): misses in every seed

"Rise" was fixed in advance as strictly greater at each step: 16,384 → 32,768 → 65,536 → 131,072.

| Seed | Held compositions | mod-97 operations | Verdict |
|---|---|---|---|
| 73129 | 76.7 → 79.2 → 80.2 → 81.5, rises | 89.5 → 91.3 → 93.2 → **91.8** (−1.4 at the last step) | misses |
| 93683 | 69.8 → 78.5 → **74.8** → 78.2 (−3.7 at 65,536) | 89.7 → 90.4 → **89.9** → 90.6 (−0.5) | misses |
| 116089 | 76.1 → **73.1** → 80.1 → 79.8 (−3.0, then −0.3) | 78.0 → 78.8 → 82.2 → **80.5** (−1.7) | misses |

Only seed 73129's compositions rise at every step. Every reversal is 3.7 points or less. That is within what the seeds differ by at a single capacity, which is 3.8–7.3 points on compositions.

### P16RC.2 (gate's lower edge): misses

| Seed | Composition ratio at 16,384 (predicted < 0.80) | Composition ratio at 32,768 (predicted ≥ 0.80 in at least two seeds) |
|---|---:|---:|
| 73129 | **0.822** | 0.849 |
| 93683 | 0.758 | 0.853 |
| 116089 | **0.811** | 0.779 |

The 16,384 clause holds in one seed only. The 32,768 clause holds in two seeds, which is what it required. The prediction as a whole misses: the gate's lower edge is not between 16,384 and 32,768. At 16,384 the other two ratios are also above 0.80 in every seed:

| Seed | mod-97 ratio at 16,384 | mod-73 ratio at 16,384 |
|---|---:|---:|
| 73129 | 0.94 | 0.97 |
| 93683 | 0.95 | 0.98 |
| 116089 | 0.83 | 0.95 |

So 16,384 slots meets the full gate in two seeds. Seed 93683 misses it on compositions only.

### P16RC.3 (saturation at 131,072): misses in seed 116089 on the first clause; the second clause holds

| Seed | Held comp at 131,072 | Every entry (E16-D) | Difference (limit 3) | Distinct composition strings, block 5 / block 6 (limit about 24,000) |
|---|---:|---:|---:|---|
| 73129 | 81.5 | 84.5 | −3.0, holds (inclusive) | 20,461 / 19,333, holds |
| 93683 | 78.3 | 79.8 | −1.6, holds | 20,437 / 19,257, holds |
| 116089 | 79.8 | 83.3 | **−3.6, misses** | 20,492 / 19,283, holds |

Doubling from 65,536 to 131,072 slots moved compositions by these amounts:

| Seed | Change in compositions |
|---|---:|
| 73129 | +1.3 |
| 93683 | +3.4 |
| 116089 | −0.3 |

In two seeds the 131,072 arm still sits 3.0–3.6 points under the every-entry arm, although its reservoir holds about 20,000 of the 24,000 taught composition strings at block 5. The difference between the two arms is therefore not only coverage. The every-entry arm also balances draws per family, so compositions are a third of its replay draws. In the label-free reservoir they are about 16–17% of slots at block 5 at every capacity:

| Capacity | Composition strings at block 5 | Share of slots |
|---:|---:|---:|
| 16,384 | 4,839 | 29.5% |
| 131,072 | 20,461 | 15.6% |

At 16,384 slots the share is 29.5%, which is why that arm holds up better than its size suggests. At 131,072 slots, distinct-count saturation lowers the share to 15.6%.

### P16RC.4 (16 ± 1 bytes per slot at block 5): holds in every life

The range is 15.96–16.03 bytes per slot.

| Capacity | Reservoir size at block 5 |
|---:|---|
| 16,384 | 261,535–261,953 bytes (0.26 MB) |
| 32,768 | 523,872–525,141 bytes (0.52 MB) |
| 65,536 | about 1.05 MB (E16-R3) |
| 131,072 | 2,094,722–2,097,072 bytes (2.1 MB) |

## Supporting numbers

- **Distinct strings at block 5, and their share of slots:**

  | Capacity | Distinct strings | Share |
  |---:|---|---:|
  | 16,384 | 15,785–15,850 | 96–97% |
  | 32,768 | 30,598–30,683 | 93–94% |
  | 65,536 | 57,820–58,043 | 88% |
  | 131,072 | 106,434–106,645 | 81% |

- **Composition loss from block 4 to block 6, in points:**

  | Capacity | 73129 | 93683 | 116089 |
  |---:|---:|---:|---:|
  | 16,384 | 11.9 | 16.3 | 11.7 |
  | 32,768 | 9.8 | 6.7 | 14.4 |
  | 65,536 | 7.5 | 11.8 | 9.1 |
  | 131,072 | 7.8 | 8.8 | 9.0 |
  | every entry | 5.7 | 7.6 | 8.0 |

  The twin loses about 0 points.
- **The cell is still what holds the compositions.** With the cell silenced at 131,072 slots, compositions fall to 0–1% on the carry pairs and 5–21% on the max/min pairs, final answer.
- **mod-73 operations are flat in capacity:**

  | Capacity | Ratio range across seeds |
  |---:|---|
  | 16,384 | 0.95–0.98 |
  | 32,768 | 0.95–0.99 |
  | 65,536 | 0.92–0.98 |
  | 131,072 | 0.94–0.99 |

- **Seed 116089's mod-97 operations sit at 78–82 at every capacity, ratio 0.83–0.87.** The every-entry arm gives 77.7 there too. This is that seed's weak spot, not a capacity effect.

## Machine, pace, and what went wrong

- **Machine:** four cores, 15 GB RAM. PyTorch 2.15.0 (CPU) was installed in this session; it was absent at start.
- **Smoke job:** `E11_STAGE_UPDATES=4 E16_REPLAY=raw E16_CAP=16384 ... T 73129` finished with seven stages, 199 CPU-s and 254 s wall. Receipt and log are in `exp45/archive_e16rc_dry/`.
- **Lives:** nine launched together at 01:16 UTC, detached with `setsid nohup`; all ended at about 04:15 UTC, 3.0 hours of wall time.
  - Each took 4,678–4,719 CPU-s, about 780 CPU-s per block.
  - That is 2.2× the standing machine's 2,150 CPU-s per life. I forecast 1.5× from the smoke job, whose blocks are almost all scoring, so the forecast was too low.
  - Capacity did not change cost.
- **No life crashed or was restarted.** Each receipt has seven stages.
- **Log incident.** Until about 03:13 UTC the session's commit script ran `git pull --rebase --autostash`. Restoring the stash replaced a log file that a live process was writing to: the process kept writing to the unlinked inode.
  - Only `exp45/logs/e16rc_raw131072_73129.log` was affected; I checked every process's stdout target. I restored it from `/proc/<pid>/fd/1` at 03:13 and then stopped all pulls.
  - The watcher re-copied that log every 30 s. The process's last line, after block 6, was written after the final copy and before it exited, so that log ends at block 5.
  - The receipt `exp45/e16_life_T_raw131072_73129.json` is complete: seven stages, block 6 at 4,719 CPU-s. Every number above comes from receipts, not logs.
  - No receipt was affected. The harness rewrites each receipt in full at each block, by path.
- **Comparator caveat.** E16-R3's receipts record `settings.cap = None`; that is the pre-fix field. The capacity of 65,536 is taken from the file name and the buffer counts.

## Limits

- **Development result.** Three seeds, one harness, one stream. These are the same standing seeds on which every setting in the chain was chosen.
- **Single runs.** Each capacity is one life per seed, and the seed-to-seed spread at a fixed capacity (3–7 points on compositions) is as large as the capacity effect measured here. The curve's shape beyond "flat from 16,384 up, with occasional drops below the gate at the low end" is not resolved by this job.
- **Labels used for reading only.** The share of composition strings is read by first byte, which the learner never sees.
- **No realized draw counters.** As before, the receipts carry none.
