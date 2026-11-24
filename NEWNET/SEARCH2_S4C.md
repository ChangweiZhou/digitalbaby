S4 part c (E3 M=64, seeds 2-3), full battery: 2026-11-23 23:30:51 - 2026-11-24 02:00:10 UTC, 4 cores (nproc --all), wall 149 min elapsed. A container restart at about 00:56 UTC killed the batch: E3s3 life23, life27, zhen and moby were lost and rerun from scratch after the restart at 00:57:25 (finished lines skipped). 20/20 lines full, no nan. Source: branch claude/newnet-s5c-sei232 from 049168ad; ef0a7cc6 (checkpointing) merged at 01:33 while only E3s3 life23/life27 were running, so they ran on the pre-merge code. Candidates and options unchanged.

| cand | moby | keep | keep4 | keep2 | mem | life23 | enzh | code | life27 | zhen | dev | held | depth | options |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E3s2 | 0.406 | 1.000 | 1.000 | 1.000 | 1.000 | 0.630 | 0.992 | 0.428 | 0.615 | 0.948 | 0.8612 (7) | 0.6636 (3) | 8 | net=13 M=64 seed=2 |
| E3s3 | 0.397 | 1.000 | 1.000 | 1.000 | 1.000 | 0.634 | 0.997 | 0.420 | 0.617 | 0.956 | 0.8612 (7) | 0.6642 (3) | 8 | net=13 M=64 seed=3 |
