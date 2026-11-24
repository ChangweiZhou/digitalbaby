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
- 02:18 UTC: S4c merged (seeds 2-3 of M=64). Over seeds 1-3, M=64 beats M=32 by about +0.01 on both lives and on code, beyond the seed spread; moby gains are within the spread. Running: S4a (M=128, lives), S4b (cC=128 M=128), S4d (seeds 2-3 of cC=128 M=64).

## Data-scaling batch (owner approved 24 Nov ~03:00 UTC: "Do this. Send some agents.")
Question: does a net gain more from reading more bytes, and do wider nets gain more than narrow ones? The search so far varied size only,
with the data fixed (moby reads 16 kB). Test: t26_calib.py, seed 1, moby and code. Each net reads T = 16k, 64k or 256k bytes and is scored
on the same 10,000 never-read bytes from the middle of the stream. It reports the right share (clip readout, as in the battery), bits and
nats per byte (learned-temperature readout), and the bits of a back-off count table fitted on the same T bytes (reference).
Runner datascale_batch.sh (resumable; finished jobs skipped), table datascale_summary.py. Helpers started 03:09 UTC:
- a: E6 cC=128 M=64, on claude/newnet-ds-a-sei232
- b: M=64 and net13 (M=16), on claude/newnet-ds-b-sei232
- c: M=128, on claude/newnet-ds-c-sei232
- d: E4 cC=128 M=32 rc_all=1, on claude/newnet-ds-d-sei232
Results go in datascale_results_{a,b,c,d}.jsonl and the summaries in DATASCALE_{A,B,C,D}.md.
Reading: the gap between nets should grow with T if moby is limited by data. Compare also against the count table's own gain.
- 03:18 UTC: helper ds-a was blocked by its own permission layer from launching a detached batch, since no human message in that session authorised it. Part a runs in the parent session instead: datascale_results_a.jsonl is written here, and the monitor restarts it after each reset with `setsid nohup bash datascale_batch.sh datascale_jobs_a.txt datascale_results_a.jsonl`. Helpers b, c, d were still pending at 03:18.
- Disk-rollback test: a detached probe wrote a file every minute in the parent's container. After a reset at about 03:17, the files from 03:14-03:16 survived, though the last turn had ended about 03:12. So the disk is not rolled back to the last turn end, at least here.
- 03:36 UTC: helper ds-b was also blocked by its permission layer. The owner said "Start now", so part b runs in the parent session alongside part a (8 jobs on 4 cores; both slow down). Restart command after a reset: `setsid nohup bash datascale_batch.sh datascale_jobs_b.txt datascale_results_b.jsonl`. Part c runs on its helper (started 03:22). Part d has not reported. Part a resumed from its checkpoints after a reset at about 03:35 (E6_moby_64k from byte 47065).
- 03:47 UTC: resets come about every 10 minutes on some helpers. A resumed test first reloads its checkpoint, then saves only after CKPT_EVERY=300 s, so resets that come sooner leave no progress (S4b: life23/life27/keep2 stuck). From the next restart on, every batch runs with CKPT_EVERY=90, which does not change results. Every session also keeps a tracked background waiter, because resets seemed to stop on S4d once it had one (not proven: S4c reset at 00:56 with one running).
  The parent's restart command: `CKPT_EVERY=90 setsid nohup bash datascale_batch.sh datascale_jobs_$p.txt datascale_results_$p.jsonl`.
- 03:52 UTC: mem (t11_gate.py) is now resumable too, and search2.py gives it a checkpoint. Identity check (killed twice, resumed at bytes 1167 and 2274): IDENTICAL. resume_check.sh now leaves out the timing at the end of t11's line. Helpers pick this up by merging claude/magical-albattani-sei232. Not every test is checkpointed: moby and code (t17) are not.

## Quick battery (24 Nov ~04:15 UTC; owner: fast iteration, each experiment ≤ ~16 min)
QUICK=1 in search2.py:
- t17 T=16000 NT=4000
- keep TA=4000 TB=8000
- mem T=20000
- life23 N0=N1=N3=1200 N4=N5=600 TXT=8000 PROBE=600
- life27 NB=600 TXT=8000 PROBE=600
- lang TA=TB=8000 NT=2000
Life scores use storage references recomputed at these sizes: life23 405,661 and life27 425,369 bits (net=store). Rows are tagged 'quick' and are comparable only with other quick rows.
Validation: search2_cands_q1.txt (11 candidates with known full scores) runs on the S4a helper (claude/newnet-s5a-sei232), writing search2_results_q1.tsv and SEARCH2_Q1.md. It gives the Spearman agreement with the full battery, per test and dev/held, and the wall time per test.
The quick battery is used to choose only if the agreement is good, especially on the lives. Otherwise its sizes are revised.
English and Chinese generation samples are to be added to each round's report (owner, 24 Nov).
