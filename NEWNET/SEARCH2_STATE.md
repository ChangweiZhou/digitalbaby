# search2 state (the optimizer's state; updated by the 10-minute monitor)

Results of record: `search2_results_s*.tsv` (one line per candidate x test; a full, non-nan line is never rerun - `search2_batch.sh`
resumes from them after any reset). Logs: `search2_logs/`. Working rule: MEMO_20261123.md.

| batch | candidates | helper branch | state |
|---|---|---|---|
| S1a | R0 net13, R1 net12 lagro relay, R2 net10 | claude/newnet-s2a-sei232 | running (started 15:36 UTC 23 Nov) |
| S1b | D1 rc=32, D2 rc=128, D3 rc_all=1 | claude/newnet-s2b-sei232 | running |
| S1c | D4 W=6, D5 W=10, D6 M=32 | claude/newnet-s2c-sei232 | running |
| S1d | D7 cC=128, D8 L=24, D9 mesh=2 mesh_dag=2, D10 eta_c=0.3 eta_h=0.15 | claude/newnet-s2d-sei232 | running |

Best so far (dev mean): - (no full batch finished)

Next batch: decided when S1 is complete - combine the dimensions that improved dev without hurting held-out; one general dimension
at a time around the best; seeds 2-3 for the leaders.

Resume procedure (for the monitor): if a helper died or stopped without its summary, tell it (send_message) or a new helper on the
same branch to: fetch its branch, `git merge origin/claude/magical-albattani-sei232` (for the resume-capable batch script), and rerun
`bash search2_batch.sh search2_cands_s1X.txt search2_results_s1X.tsv`.
