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

S2c, S2d merged (19:59): seeds 1-3 means (+- sd):
| | moby | code | life23 | life27 | enzh | zhen |
|---|---|---|---|---|---|---|
| net13 | 0.393+-0.006 | 0.400+-0.007 | 0.608+-0.000 | 0.593+-0.001 | 0.986+-0.009 | 0.941+-0.013 |
| cC=128 | 0.395+-0.006 | 0.406+-0.008 | 0.613+-0.000 | 0.598+-0.000 | 1.003+-0.011 | 0.979+-0.006 |
| M=32 | 0.400+-0.007 | 0.416+-0.005 | 0.622+-0.001 | 0.606+-0.001 | 0.989+-0.015 | 0.963+-0.020 |
Both widenings hold on three seeds: M=32 mainly prediction (lives, code), cC=128 mainly keeping (zhen +0.04, enzh +0.02).

21:08 S2 partial (seed 1): cC=128 M=32 0.624 / 0.609 (lives), code 0.426; + rc_all=1 0.631 / 0.615 (best lives); cC=256 0.611 / 0.598
(columns saturate on prediction, keeping 0.989); cC=256 M=32 0.622 / -, moby 0.407, enzh 1.021. Diminishing returns from 4x to 8x size.
S3a launched early (branch claude/newnet-s4a-sei232): seeds 2-3 of E4 (cC=128 M=32 rc_all=1) and E1 (cC=128 M=32).

22:40 S2a, S2b merged; SEARCH2_S2.md (seed 1). Cells per column keep scaling: M=16 / 32 / 64 -> life23 0.609 / 0.621 / 0.630,
life27 0.594 / 0.607 / 0.616, code 0.406 / 0.419 / 0.430, moby 0.399 / 0.406 / 0.411 (about +0.01 on lives per doubling, no sign of
saturation yet). Columns saturate: cC=64 / 128 / 256 -> life23 0.609 / 0.613 / 0.611 (keeping zhen 0.949 / 0.986 / 0.989).
Best held-out: E6 cC=128 M=64 (8x): lives 0.631 / 0.617, code 0.432, zhen 0.981, held 0.677. Not a plateau along M: the next
batch (after S3a seeds) should continue cells per column (M=128 with cC=64 and cC=128), seeds for E3/E6.

23:20 owner approved continuing along cells per column. S4 launched: a F1 M=128 (claude/newnet-s5a-sei232), b F2 cC=128 M=128
(s5b), c seeds 2-3 of E3 M=64 (s5c), d seeds 2-3 of E6 cC=128 M=64 (s5d). Then the plateau decision and the held-out log-perplexity.

- 23:52 UTC: S3a stopped at 20/40 lines because its batch ran inside a 2-hour background task. The helper was told to restart it detached (nohup), so lines already done are skipped. S4 helpers were told to make sure their batches are detached too. Rule from now on: start batches with nohup or setsid from an ordinary Bash call, never as the background task itself.

## Resumable long tests (24 November, 01:20-01:45 UTC)
Container restarts every 30-45 minutes kill every process. A life test at M=128 runs 1-4 hours, so S4a/S4b's long tests could never
finish. resume.py now saves a long test's whole state (net, numpy generators, the C library's xorshift state, counters, position) gzip-
pickled every 5 minutes. t23 and t27 save between lines; t13 (keep, keep4, keep2, enzh, zhen) saves at stage ends. search2.py
gives the long tests a file in NEWNET/search2_ckpt/ (git-ignored; not for SMOKE runs). Restarting the same search2_batch.sh line resumes
them. The secs column then counts only the time since the last restart.
Identity check (resume_check.sh: the same command run whole, and run killed twice and resumed; final results compared exactly):
- t23 life (resumed at phase 1 line 138 and at phase 3): IDENTICAL.
- t27 life (resumed mid-text at byte 1051 and at phase 4): IDENTICAL.
- t13 keep/ctx (resumed at stages 2 and 3): IDENTICAL.
The first t13 check, before the C generator was saved, differed in the 3rd decimal (0.292 vs 0.293). It was found and fixed (commit 7a531da1f).

- 01:42 UTC: S3a merged (seeds 2-3 of E1 cC=128 M=32 and E4 cC=128 M=32 rc_all=1). Over seeds 1-3, rc_all=1 adds about +0.006 on both lives (+0.007 life23, +0.005 life27; seed spread ≤0.001) and leaves the rest unchanged. rc_all=1 is kept in the leading line. Running: S4a-d only.
