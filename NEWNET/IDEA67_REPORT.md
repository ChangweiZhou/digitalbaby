# Ideas 6 and 7 on t22: exploration under yes/no, surprise-scaled plasticity (23 November 2026)

Development level, one machine, **seed 1 only**; no gate, seal or outside review is claimed. Plan with the success criteria written
before the runs: `IDEA67_PLAN.md`. Code: `net12_fb.py` (subclass of `net12.Tissue`; no existing .py changed), `t22_fb.py` (copy of
`t22_rules.py` importing net12_fb; with all new options off it gives the same numbers as `t22_rules.py`, checked on 400 lines).
Branch claude/newnet-idea67-sei232, from claude/magical-albattani-sei232 @ 6dc127e8; code commit 77835c0e. Logs and result
lines: `idea67_runs/` (`full.jsonl` planned runs, `posthoc.jsonl` post-hoc runs, `small.jsonl` short checks).
All full runs: struct=K8, lagro=1, relay=1, 24,000 lines (copy 0-12,000, rev 12,000-18,000, copy 18,000-24,000), ~15.5 min each,
single thread, 4 in parallel. References from `rules_runs/` (same stream, same seed).

## Planned full runs: right share by phase and layer

| run | feedback | P0 copy b | P0 copy c | P1 rev b | P1 rev c | P2 copy b | P2 copy c | answers told wrong |
|---|---|---|---|---|---|---|---|---|
| reference: noise 0.1 | yes/no | 0.137 | 0.018 | 0.327 | 0.208 | 0.817 | 0.429 | - |
| 6a uncertainty noise (unc=1, 0.1 x (1 - max p)) | yes/no | **0.498** | 0.073 | 0.375 | 0.292 | 0.908 | 0.786 | 0.545 |
| 6b inhibition of return (ior=1) + noise 0.1 | yes/no | 0.055 | 0.018 | 0.247 | 0.125 | 0.602 | 0.357 | 0.675 |
| 6c both | yes/no | 0.355 | 0.073 | 0.374 | 0.250 | 0.913 | 0.857 | 0.576 |
| reference | teach | 0.988 | 0.836 | 0.774 | 0.708 | 0.909 | 0.786 | - |
| 7 surprise-scaled rates (surp=1) | teach | 0.993 | 0.836 | 0.779 | 0.708 | 0.911 | 0.786 | - |

(bytes: P0 b 30,448, c 55; P1 b 9,401, c 24; P2 b 17,608, c 14: layer c is tiny.)

## After a change of rule (right share, windows of 250 lines)

| run | copy -> rev, windows 1-12 | rev -> copy, windows 1-12 |
|---|---|---|
| reference yes/no | 0.21 0.21 0.24 0.24 0.26 0.27 0.30 0.29 0.32 0.32 0.32 0.34 | 0.52 ... 0.86 0.85 |
| 6a | 0.33 x 12 (0.33-0.35, flat) | 0.74 0.75 0.77 0.77 0.80 0.83 0.85 0.87 0.89 0.91 0.92 0.93 |
| 6b | 0.12 0.10 0.12 0.14 0.13 0.16 0.20 0.19 0.21 0.24 0.24 0.27 | 0.42 ... 0.54 0.53 |
| 6c | 0.32-0.35 (flat) | 0.73 0.74 0.77 0.78 0.79 0.85 0.86 0.89 0.90 0.91 0.94 0.94 |
| reference teach | 0.34 0.34 0.35 0.34 0.34 0.35 0.38 0.48 0.61 0.80 0.89 0.95 | 0.64 0.67 0.69 0.71 0.71 0.78 0.82 0.89 0.94 0.98 1.00 1.00 |
| 7 | 0.34 0.34 0.35 0.34 0.34 0.35 0.39 0.52 0.65 0.82 0.90 0.95 | 0.64 0.68 0.69 0.71 0.72 0.79 0.82 0.90 0.95 0.98 0.99 1.00 |

Run 7's multiplier m (mean per window): 0.97-0.98 while stable; after copy -> rev 1.24 in the first window, 1.06 in the second, then
0.97-1.00; after rev -> copy 1.09, 1.02, then 0.98. The ceiling (4) was never approached.

## Verdict against the plan
- **Idea 6: fails the criterion (P0 copy b >= 0.5), narrowly.** Best is 6a, uncertainty-scaled noise: 0.498 (reference 0.137, 3.6x).
  It also helps later (P2 copy b 0.91 vs 0.82, P2 c 0.79 vs 0.43 on 14 bytes) and does not help the copy -> rev switch (flat 0.33:
  the answers stay copy-like). Inhibition of return hurts: alone 0.055 (below the reference), and added to 6a it lowers P0 b to 0.355.
  A plausible reading (not tested separately): a refuted symbol in the context "byte just read" is often the right answer at that
  place a few lines later (8-30 symbols early in the life), so suppressing it costs right answers; and in a stream of new items the
  same context rarely repeats with the same wrong candidate.
- **Idea 7: fails.** rev after_change reaches 0.9 only in window 11 (criterion: by window 4); essentially identical to teach without
  it (window 11, 0.89). P0 b/c 0.993/0.836 (not below 0.97/0.75; it does not cost layers b/c). The mechanism as specified barely
  acts: the slow mean (2,000 bytes, ~220 lines) catches up with the raised error within about one window, so m returns to ~1 while
  the error stays high for ~1,750 lines.
