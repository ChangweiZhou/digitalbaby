# V216 execution optimization

The engineering entry point now uses the optimized path. The reference path is retained for parity checks. The historical three-cycle receipts were preserved.

## Changes

- Batch the existing sparse PN→KC projection across byte sites within a record. Select the same positive top-k values without fully sorting each side. At the cutoff, retain the original ascending KC-index tie break.
- Cache sensory states and packed activation vectors by exact byte prefix from the engineering episode reset. Each cache is bounded to 8,192 entries. Long records bypass the prefix caches. A future continuous-input path must key any cache by its actual starting sensory state or bypass this cache.
- Compute only the full NORMAL gradient. Skip CGLP subgroup diagnostics unused by this arm. Keep the update order, numerical operations, dropout RNG, optimizer and warmup unchanged. The fork uses the source restore method's copies directly.
- Test two independent single-thread processes for separate lives. Updates within one life remain serial.

## Measurements

| Benchmark | Reference | Optimized |
| --- | ---: | ---: |
| Training CPU on 1,200 matched events | 5.43 s | 1.05 s |
| Post-warmup training CPU per event | 4.36 ms | 1.79 ms |
| Two optimized engineering jobs, serial vs concurrent wall time | 9.25 s | 5.87 s |

The two concurrent jobs' summed child peak RSS was 551,010,304 bytes. This excludes the controller and does not certify process-tree admission. The 255-event tail includes rewarming an empty cache after restoring a checkpoint. At that observed rate, the 1.6M training events plus 25% reserve project to about **1.0 CPU hour** for training alone. Formal layouts may produce different cache-hit rates. Full-roster evaluation, references and I/O remain outside the estimate.

Sensor codes, weights and optimizer moments were equal at every step across 1,200 events. Final RNG and clock matched. All 32 held engineering probe outputs and losses matched. The optimized runner also reproduced the saved Cycle 3 outputs and losses, and the disk checkpoint resumed with the same first update.

Reproduce with the V181-capable Python environment using `optimize_check.py`, `parallel_check.py`, and `profile_tail.py`. The current environment still reports the existing SciPy/NumPy compatibility warning. GPU execution was untested. The status remains ADMISSION_PENDING and the official roster was not run.
