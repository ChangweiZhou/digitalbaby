# Ideas 8 and 9: the blocked schedule, and more data (23 November 2026)

Development level, one machine, seed 1 (plus seed 2 for net12 in idea 8). No gate, seal or outside review is claimed.
Source: branch `claude/newnet-idea89-sei232`, from `claude/magical-albattani-sei232` @ 6dc127e8. Existing scripts unchanged; new script
`t23_mixed.py`. Runs: `idea89_runs/` (`run.sh`, `mixed.jsonl`, `x3.jsonl`, `*.log`). OMP_NUM_THREADS=1, 4 processes.
net12 = `net=12 struct=K8 lagro=1 relay=1`; K8 = `net=10 struct=K8`; storage = back-off n-gram (orders 1-8). All cue=1.

## Idea 8: the same life with the rules interleaved (t23_mixed.py)

Same amounts as t23's life: copy 9,000 lines, rev 9,000, succ 6,000, in one random order from the start; the same 40,000 bytes of
tinyshakespeare in 8 blocks of 5,000 bytes (read after lines 1500, 4500, ..., 22500). Same items, alphabet growth, probes, bits.

| interleaved life | total bits | (blocked life, t23) | copy end | rev end | succ max -> end | text max -> end |
|---|---|---|---|---|---|---|
| storage | 2,019,367 | 1,984,668 | 0.15 | 0.69 | 0.18 -> 0.02 | 0.375 -> 0.375 |
| K8 | 1,207,788 | 1,149,490 | 0.09 | 0.36 | 0.08 -> 0.08 | 0.350 -> 0.287 |
| net12, seed 1 | 872,222 | 828,527 | 1.00 | 1.00 | 0.22 -> 0.22 | 0.345 -> 0.331 |
| net12, seed 2 | 873,845 | - | 1.00 | 1.00 | 0.29 -> 0.29 | 0.351 -> 0.347 |

First-met answer bytes, net12 seed 1 / seed 2 (layer b): copy 0.981 / 0.983, rev 0.916 / 0.922, **succ 0.105 / 0.102**
(K8: 0.12, 0.01, 0.07; storage 0.02, 0.00, 0.01). net12's right share on succ's first-met answers by window of 1,000 lines rises
slowly from 0.02-0.09 to 0.23 (seed 1) / 0.22 (seed 2) at the end; copy reaches 0.99 in the 3rd window, rev 0.98-1.0 by the 5th-6th.

In the blocked life net12 learned succ to 0.98 in 6,000 consecutive lines; interleaved with copy and rev it reaches only 0.22-0.29
with the same 6,000 lines. The text is kept (end 0.331 / 0.347 against its best 0.345 / 0.351, and against storage's 0.375),
where in the blocked life it fell 0.35 -> 0.24. Total bits are 5% worse than the blocked life (872k against 829k).

## Idea 9a: the blocked life with every phase x3 (t23_life_rules.py N0=N1=N3=18000 N4=N5=9000 TXT=120000)

Phase ends: P0 copy 18,000, P1 rev 36,000, text 51,000, P3 succ 69,000, P4 copy 78,000, P5 rev 87,000 (probe clock).

| | total bits | text after text -> end | succ after succ -> 3,000 lines of copy -> end of copy -> end | copy / rev at end of succ | copy / rev end |
|---|---|---|---|---|---|
| net12 | 2,409,637 | 0.361 -> 0.239 (66%) | 1.00 -> 0.95 -> 0.13 -> 0.03 | 0.59 / 0.85 | 1.00 / 1.00 |
| K8 | 3,376,403 | 0.358 -> 0.065 (18%) | 0.08 -> 0.01 -> 0.01 -> 0.00 | 0.00 / 0.01 | 0.00 / 0.37 |

(1x life, t23: net12 text 0.35 -> 0.24 (69%), succ 0.98 -> 0.07; K8 text 0.36 -> 0.09 (25%).)
net12 first-met layer b: P0 copy 0.99 (c 0.72), P1 rev 0.89, P3 succ 0.79, P4 copy 0.97, P5 rev 1.00. Savings windows: P4 copy starts
at 0.66 (P0's first window 0.48; 1x life 0.95), P5 rev 1.00. So three times the data does not keep succ (it survives ~3,000 lines of
copy, then is lost) and keeps the same share of the text; and the longer succ phase now also partly erodes copy (0.59) and rev (0.85),
which the returning rules then recover.

## Idea 9b: natural text with more reading (t17_general.py T=64000, NT=10000, net=12 K8 options, seed 1)

| held-out next-byte right share | plain | + lagro=1 relay=1 | count tables last 1/2/3 (same bytes) |
|---|---|---|---|
| Moby Dick | 0.429 | 0.433 | 0.294 / 0.390 / 0.452 |
| code | 0.428 | 0.429 | 0.250 / 0.355 / 0.420 |

The lag machinery changes natural text by +0.004 / +0.001 at four times the reading (seed 1): no meaningful help, no cost.

## Verdict (plain)
- Idea 8: **no**, net12 does not keep everything when interleaved. Interleaving keeps the text (net12's text probe barely falls) and
  copy/rev stay at 1.0, but the learned mapping succ is then hardly learned at all (0.22 / 0.29, two seeds). The blocked schedule is
  not the whole E2 problem: blocked, succ is learned and then overwritten; mixed, it is not learned. The remaining problem is that
  the mapping and copy compete for the same direct readout, under either schedule.
- Idea 9a: more data of the same blocked life changes nothing qualitative: text kept 66% (net12) / 18% (K8), succ still forgotten
  when copy returns.
- Idea 9b: the lag readout and relay do not help natural text with 64,000 bytes read (+0.004 / +0.001).
Limits: seed 1 only except idea 8 net12 (seeds 1, 2); one interleaving order per seed; cue=1 only.
