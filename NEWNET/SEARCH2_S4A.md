S4a (F1 net13 M=128), run 2026-11-23 23:29 UTC to 2026-11-24 04:02 UTC: wall 4 h 33 min on 4 cores (OMP_NUM_THREADS=1), branch claude/newnet-s5a-sei232 from 049168ad. Container resets at about 00:07, 00:32 and 03:46 UTC, plus a deliberate stop at 01:32 to merge the checkpoint code (ef0a7cc6). life23 and life27 ran from the start again at 00:08, 00:33 and 01:32. After 01:32 they resumed from search2_ckpt/, and life27 went on from its 03:37 save after the 03:46 reset (CKPT_EVERY=90 from then). The seconds column in the tsv counts only the last process: life27's 923 s is the final resumed segment, not the whole life.

| cand | moby | keep | keep4 | keep2 | mem | life23 | enzh | code | life27 | zhen | dev | held | depth | options |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 | 0.412 | 1.000 | 1.000 | 1.000 | 1.000 | 0.635 | 0.997 | 0.431 | 0.621 | 0.972 | 0.8635 (7) | 0.6748 (3) | 8 | net=13 M=128 |
