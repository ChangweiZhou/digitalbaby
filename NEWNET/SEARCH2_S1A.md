S1a, 2026-11-23; machine: 4 cores (nproc --all); wall time 138 min in total: about 100 min with the old one-job-at-a-time batch (11 jobs), then 37 min for the remaining 19 jobs after the nproc --all fix. Source: merge of 1781a486 into claude/newnet-s2a-sei232. All 30 jobs were full runs; none returned nan.

| cand | moby | keep | keep4 | keep2 | mem | life23 | enzh | code | life27 | zhen | dev | held | options |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | 0.399 | 1.000 | 1.000 | 1.000 | 1.000 | 0.609 | 0.989 | 0.406 | 0.594 | 0.949 | 0.8567 (7) | 0.6496 (3) | net=13  |
| R1 | 0.399 | 1.000 | 1.000 | 1.000 | 1.000 | 0.583 | 0.989 | 0.406 | 0.571 | 0.951 | 0.8530 (7) | 0.6429 (3) | net=12 lagro=1 relay=1 |
| R2 | 0.399 | 1.000 | 1.000 | 1.000 | 1.000 | 0.421 | 0.981 | 0.406 | 0.410 | 0.926 | 0.8287 (7) | 0.5805 (3) | net=10  |
