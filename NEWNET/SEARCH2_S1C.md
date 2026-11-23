S1c, run 2026-11-23 on 4 cores; wall time 99 min (started serially: search2_batch.sh calls nproc after exporting OMP_NUM_THREADS=1, so nproc reported 1; after ~31 min the remaining 28 jobs were pooled 4 at a time with the same per-job command; candidates and options unchanged).

| cand | moby | keep | keep4 | keep2 | mem | life23 | enzh | code | life27 | zhen | dev | held | options |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D6 | 0.406 | 1.000 | 1.000 | 1.000 | 1.000 | 0.621 | 0.995 | 0.419 | 0.607 | 0.958 | 0.8603 (7) | 0.6614 (3) | net=13 M=32 |
| D5 | 0.399 | 1.000 | 1.000 | 1.000 | 1.000 | 0.608 | 0.989 | 0.407 | 0.588 | 0.948 | 0.8566 (7) | 0.6478 (3) | net=13 W=10 |
| D4 | 0.400 | 1.000 | 1.000 | 1.000 | 1.000 | 0.601 | 0.992 | 0.406 | 0.583 | 0.960 | 0.8561 (7) | 0.6495 (3) | net=13 W=6 |
