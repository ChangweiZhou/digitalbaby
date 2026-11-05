# MiniFly re-entry and temporal credit — three engineering cycles

**2026-11-05 · Engineering complete.** The reduced rehearsal and timing sample are development checks. Main roster training and the decision thresholds in the design remain pending.

| Cycle | Refinement and audit | Executed result |
|---|---|---|
| 1 · Memory routing | Both routes retain the direct memory-to-head read. Use a hand-initialized two-hop matrix to test whether the first read changes the next key. | The third coordinate of the first read was zero. It became nonzero with re-entry and stayed zero in the output-only path. With the route off, zeroing memory left the workspace bit-identical. The untouched circuits stayed bit-identical. |
| 2 · Forward credit | Differentiate the acquired matrix and the learned write gate through the actual loop. Verify the norm cap on its active branch and reset sensitivity at the declared window boundaries. | Eight float64 fixtures crossed routing, one/eight-symbol windows and active/inactive cap. Maximum discrepancy against independent matched-window Torch gradients was **1.33e-15**. Maximum directional finite-difference error was **1.29e-9**. Changing one circuit's parameters left other circuits' states unchanged. Gate gradients were zero for the one-symbol window and nonzero for the eight-symbol window. |
| 3 · End-to-end rehearsal | Construct full-sized data layouts on three development seeds, then exercise the four local arms through short A → B → A′ histories and both BPTT references through short A histories. Use real sensing and scored native byte/EOS outputs. | All three full data layouts satisfied orbit exclusion and the prescribed held panel counts. All **12 local histories** resumed bit-for-bit through A′ from a checkpoint inside an unfinished B update block. Read-only probes preserved the training state. The EOS fixture emitted EOS through the native head and the scorer rejected it as a premature answer. |

The short rehearsal produced **0/336 exact held answers across the recorded arm/phase probe rows**. The independent 2,048-symbol timing samples also produced **0/48** on their eight-question panels. These outcomes establish execution and a weak early learning point. The rehearsal uses smaller histories and panels than the proposed experiment, so these counts do not estimate performance at the design's larger doses.

## Timing and footprint

Each arm trained on the first 2,048 symbols of the same full-layout development stream. The measurement includes native sensing and parameter updates. The six arms each completed 64 updates.

| Arm | Training symbols per CPU second |
|---|---:|
| R8 | 1,380 |
| O8 | 1,768 |
| R1 | 1,433 |
| O1 | 1,652 |
| R32-BP | 1,114 |
| O32-BP | 1,139 |

Applying the observed per-arm rates to the design's maximum training counts gives **about 11 CPU-hours** for the main training alone, or **13.7 CPU-hours** with the proposed 25% reserve. Probe generation, donor histories, checkpoint I/O and campaign overhead require separate accounting. The observed rates are a short-run projection.

The profiled model held **1,331,472 bytes** in NumPy arrays, including optimizer and sensitivity arrays, and **556,836 bytes** of sensor arrays. The single engineering process peaked at **337,641,472 bytes RSS**, including the numerical libraries and Torch diagnostics. This is a single-process receipt. Parallel process-tree memory and longer-run throughput remain unmeasured.

## Reproduce

Use Python with NumPy 2.4.0, SciPy 1.16.3 and Torch 2.9.1. The included immutable sensor source and assets are copied from the V181 sensor used by V219. The code uses that sensor's fingerprint checks on startup.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python -B qualify.py all
```

The three JSON receipts hold observed read values, the derivative comparisons, stream digests, raw output/EOS rows, resume digests and timings. The engineering runner covers the core, data generator and reduced rehearsal. The final roster, full shortcut suite, matched memory-donor campaign and final reducer still need to be locked and wired for the main run.
