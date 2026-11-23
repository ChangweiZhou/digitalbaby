# search2 state (the optimizer's state; updated by the 10-minute monitor)

Results of record: `search2_results_s*.tsv` (one line per candidate x test; a full, non-nan line is never rerun - `search2_batch.sh`
resumes from them after any reset). Logs: `search2_logs/`. Working rule: MEMO_20261123.md.

| batch | candidates | helper branch | state |
|---|---|---|---|
| S1a | R0 net13, R1 net12 lagro relay, R2 net10 | claude/newnet-s2a-sei232 | **done, merged** (SEARCH2_S1A.md): net13 dev 0.8567 / held 0.6496; net12 0.8530 / 0.6429; net10 0.8287 / 0.5805 |
| S1b | D1 rc=32, D2 rc=128, D3 rc_all=1 | claude/newnet-s2b-sei232 | **done, merged** (SEARCH2_S1B.md): rc_all=1 0.8577 / 0.6525; rc=32 0.8563 / 0.6497; rc=128 0.8555 / 0.6508 |
| S1c | D4 W=6, D5 W=10, D6 M=32 | claude/newnet-s2c-sei232 | **done, merged** (SEARCH2_S1C.md): D6 M=32 dev 0.8603 / held 0.6614; W=10 0.8566 / 0.6478; W=6 0.8561 / 0.6495 |
| S1d | D7 cC=128, D8 L=24, D9 mesh=2 mesh_dag=2, D10 eta_c=0.3 eta_h=0.15 | claude/newnet-s2d-sei232 | **done, merged** (SEARCH2_S1D.md): cC=128 0.8609 / 0.6637 (best); L=24 0.8528 / 0.6263; rates 0.8519 / 0.6346; mesh_dag=2 0.8393 / 0.6269 |
| S2a | E1 cC=128 M=32, E2 cC=256, E3 M=64 (seed 1) | claude/newnet-s3a-sei232 | running (18:28) |
| S2b | E4 cC=128 M=32 rc_all=1, E5 cC=256 M=32, E6 cC=128 M=64 (seed 1) | claude/newnet-s3b-sei232 | running |
| S2c | R0, cC=128, M=32 at seed 2 | claude/newnet-s3c-sei232 | running |
| S2d | R0, cC=128, M=32 at seed 3 | claude/newnet-s3d-sei232 | running |

Best so far after S1 (SEARCH2_S1.md, seed 1): D7 cC=128 dev 0.8609 / held 0.6637, D6 M=32 0.8603 / 0.6614; net13 0.8567 / 0.6496. Widening (columns or cells per column) helps; more regions, other topology, faster rates hurt; window size and rule-cell size do not matter.

Fix 17:20: search2_batch.sh pooled only 1 job at a time (nproc honours OMP_NUM_THREADS=1); now counts cores with nproc --all.

Next batch: decided when S1 is complete - combine the dimensions that improved dev without hurting held-out; one general dimension
at a time around the best; seeds 2-3 for the leaders.

Resume procedure (for the monitor): if a helper died or stopped without its summary, tell it (send_message) or a new helper on the
same branch to: fetch its branch, `git merge origin/claude/magical-albattani-sei232` (for the resume-capable batch script), and rerun
`bash search2_batch.sh search2_cands_s1X.txt search2_results_s1X.tsv`.

Owner (17:45): measure held-out log-perplexity (bits/nats per byte, t26_calib.py) of the final architectures at the end, when the search is done - not as a search criterion now.

Serial depth (owner, 23 Nov ~18:45; the real-time argument: a reaction allows only a few serial steps, so prefer width): the summary
now records each candidate's depth (regions on the longest source path of the context tissue). In S1 the two deeper candidates were
the worse ones (L=24: depth 12; mesh=2 mesh_dag=2: depth 16; all others 8). Rule for the search: widen, keep depth <= 8.
Next direction after the search (general, not a device): learning state carried over time (computation spread across steps rather
than deeper per step) - to be designed and judged on the whole battery.
