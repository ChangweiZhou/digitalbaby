# XDG report: net12 context-gated cells (xdg=8) and lag_xdg in the t23 one-life rules+text stream

Source: branch `claude/magical-albattani-sei232` at `ada2bd07` (net12.py, t23_life_rules.py unchanged); runs on branch
`claude/newnet-xdg-sei232`, OMP_NUM_THREADS=1, 4 in parallel, each about 25 min (≈1470 s for the 29,000 lines).
Results: `xdg_runs/res.jsonl`, logs `xdg_runs/*.log`, tables made by `xdg_runs/summarize.py` (copy in `xdg_runs/tables.md`).
The baseline rows (`lagro=1+relay=1`, seed 1, cue 0 and 1) are taken from `life_rules_runs/res.jsonl`; the seed-2 baseline was run here.
All runs: net=12 struct=K8 lagro=1 relay=1, plus the options in the row name. The life is the same in every run: copy 6000, rev 6000, text 40,000 bytes,
succ 6000, copy 3000, rev 3000 lines. Probes at the end of each phase are at lines 6000, 12000, 17000, 23000, 26000 and 29000.
Only seed 1 has all variants; seed 2 has only xdg+lag_xdg cue=1 and the baseline cue=1. These are single lives, so no error bars.
No gate, seal or external review is claimed.

## Bits of the whole life

| run | total | input | answer | text | answer b/byte | text b/byte |
|---|---|---|---|---|---|---|
| lagro=1+relay=1 cue=1 s1 | 828527 | 577497 | 104926 | 146104 | 1.457 | 3.653 |
| lagro=1+relay=1 cue=0 s1 | 845244 | 573147 | 127431 | 144666 | 1.770 | 3.617 |
| lagro=1+relay=1+lagro_gate=1 cue=1 s1 | 826281 | 577320 | 102874 | 146086 | 1.429 | 3.652 |
| lagro=1+relay=1+xdg=8+lag_xdg=1 cue=1 s1 | 889862 | 594681 | 146137 | 149044 | 2.030 | 3.726 |
| lagro=1+relay=1+xdg=8+lag_xdg=1 cue=0 s1 | 900181 | 591404 | 160053 | 148724 | 2.223 | 3.718 |
| lagro=1+relay=1+xdg=8 cue=1 s1 | 847565 | 585259 | 109940 | 152366 | 1.527 | 3.809 |
| lagro=1+relay=1 cue=1 s2 | 829414 | 576965 | 105198 | 147252 | 1.461 | 3.681 |
| lagro=1+relay=1+xdg=8+lag_xdg=1 cue=1 s2 | 882755 | 599649 | 129634 | 153472 | 1.800 | 3.837 |

## First-met right shares (layers b, c; n in brackets)

| run | P0_copy_b | P0_copy_c | P1_rev_b | P3_succ_b | P4_copy_b | P5_rev_b |
|---|---|---|---|---|---|---|
| lagro=1+relay=1 cue=1 s1 | 0.977 (15619) | 0.725 (40) | 0.795 (7037) | 0.505 (16485) | 0.991 (7896) | 1.000 (2435) |
| lagro=1+relay=1 cue=0 s1 | 0.977 (15619) | 0.725 (40) | 0.743 (6746) | 0.348 (16485) | 0.799 (7820) | 0.959 (2416) |
| lagro=1+relay=1+lagro_gate=1 cue=1 s1 | 0.977 (15619) | 0.725 (40) | 0.794 (7037) | 0.529 (16485) | 0.986 (7896) | 1.000 (2435) |
| lagro=1+relay=1+xdg=8+lag_xdg=1 cue=1 s1 | 0.915 (15619) | 0.650 (40) | 0.621 (7037) | 0.314 (16485) | 0.914 (7896) | 0.913 (2435) |
| lagro=1+relay=1+xdg=8+lag_xdg=1 cue=0 s1 | 0.915 (15619) | 0.650 (40) | 0.627 (6746) | 0.208 (16485) | 0.783 (7820) | 0.913 (2416) |
| lagro=1+relay=1+xdg=8 cue=1 s1 | 0.978 (15619) | 0.725 (40) | 0.790 (7037) | 0.473 (16485) | 0.989 (7896) | 1.000 (2435) |
| lagro=1+relay=1 cue=1 s2 | 0.976 (15806) | 0.725 (40) | 0.797 (7024) | 0.506 (16350) | 0.988 (7860) | 1.000 (2444) |
| lagro=1+relay=1+xdg=8+lag_xdg=1 cue=1 s2 | 0.934 (15806) | 0.600 (40) | 0.700 (7024) | 0.524 (16350) | 0.959 (7860) | 0.988 (2444) |

## First windows (250 lines) of P4 (copy again) and P5 (rev again), with P0 / P1 for comparison

| run | P0 first 4 | P4 first 4 | P1 first 4 | P5 first 4 |
|---|---|---|---|---|
| lagro=1+relay=1 cue=1 s1 | 0.48 0.86 0.98 1.00 | 0.95 0.98 0.98 0.99 | 0.35 0.36 0.40 0.48 | 1.00 1.00 1.00 1.00 |
| lagro=1+relay=1 cue=0 s1 | 0.48 0.86 0.98 1.00 | 0.20 0.39 0.58 0.72 | 0.35 0.35 0.36 0.38 | 0.56 0.64 0.70 0.76 |
| lagro=1+relay=1+lagro_gate=1 cue=1 s1 | 0.48 0.86 0.98 1.00 | 0.91 0.97 0.98 0.99 | 0.35 0.36 0.40 0.47 | 1.00 1.00 1.00 1.00 |
| lagro=1+relay=1+xdg=8+lag_xdg=1 cue=1 s1 | 0.45 0.71 0.85 0.99 | 0.87 0.78 0.97 0.72 | 0.28 0.34 0.33 0.38 | 0.90 0.98 0.96 0.97 |
| lagro=1+relay=1+xdg=8+lag_xdg=1 cue=0 s1 | 0.45 0.71 0.85 0.99 | 0.36 0.44 0.67 0.57 | 0.32 0.34 0.32 0.36 | 0.55 0.64 0.68 0.78 |
| lagro=1+relay=1+xdg=8 cue=1 s1 | 0.49 0.88 0.99 1.00 | 0.93 0.97 0.98 1.00 | 0.35 0.36 0.39 0.43 | 1.00 1.00 1.00 1.00 |
| lagro=1+relay=1 cue=1 s2 | 0.47 0.85 0.99 1.00 | 0.91 0.98 0.99 0.99 | 0.34 0.38 0.41 0.48 | 1.00 1.00 1.00 1.00 |
| lagro=1+relay=1+xdg=8+lag_xdg=1 cue=1 s2 | 0.47 0.77 0.97 0.94 | 0.86 0.83 0.93 0.95 | 0.35 0.41 0.49 0.52 | 0.98 0.99 0.99 0.99 |

## Probes (learning off) at the end of each phase: copy / rev / succ / text

| run | 6000 | 12000 | 17000 | 23000 | 26000 | 29000 |
|---|---|---|---|---|---|---|
| lagro=1+relay=1 cue=1 s1 | 1.00/0.34/0.00/0.016 | 1.00/1.00/0.00/0.021 | 0.99/1.00/0.01/0.351 | 0.94/0.99/0.98/0.239 | 1.00/1.00/0.17/0.267 | 1.00/1.00/0.07/0.242 |
| lagro=1+relay=1 cue=0 s1 | 1.00/0.34/0.00/0.005 | 0.50/1.00/0.01/0.004 | 0.47/1.00/0.03/0.362 | 0.14/0.26/0.89/0.267 | 1.00/0.54/0.00/0.268 | 0.64/1.00/0.02/0.276 |
| lagro=1+relay=1+lagro_gate=1 cue=1 s1 | 1.00/0.34/0.00/0.016 | 1.00/1.00/0.00/0.021 | 0.99/1.00/0.01/0.351 | 0.91/0.98/0.99/0.239 | 1.00/1.00/0.22/0.269 | 1.00/1.00/0.07/0.242 |
| lagro=1+relay=1+xdg=8+lag_xdg=1 cue=1 s1 | 0.94/0.34/0.00/0.015 | 0.99/1.00/0.00/0.021 | 0.32/1.00/0.01/0.344 | 0.98/1.00/0.33/0.267 | 1.00/0.97/0.01/0.231 | 1.00/1.00/0.02/0.211 |
| lagro=1+relay=1+xdg=8+lag_xdg=1 cue=0 s1 | 0.94/0.34/0.00/0.010 | 0.51/1.00/0.01/0.006 | 0.17/1.00/0.03/0.349 | 0.29/0.48/0.53/0.273 | 1.00/0.59/0.00/0.252 | 0.64/1.00/0.02/0.218 |
| lagro=1+relay=1+xdg=8 cue=1 s1 | 1.00/0.34/0.00/0.017 | 0.99/1.00/0.00/0.025 | 0.99/1.00/0.01/0.334 | 0.92/0.98/0.96/0.250 | 1.00/1.00/0.14/0.254 | 1.00/1.00/0.09/0.219 |
| lagro=1+relay=1 cue=1 s2 | 1.00/0.34/0.00/0.017 | 1.00/1.00/0.01/0.018 | 0.99/1.00/0.00/0.370 | 0.91/0.98/0.95/0.237 | 1.00/1.00/0.15/0.266 | 1.00/1.00/0.03/0.260 |
| lagro=1+relay=1+xdg=8+lag_xdg=1 cue=1 s2 | 1.00/0.34/0.00/0.016 | 0.92/1.00/0.00/0.015 | 0.89/0.92/0.01/0.344 | 0.89/0.83/0.91/0.218 | 1.00/0.98/0.38/0.228 | 1.00/1.00/0.18/0.238 |

## Kept: text probe at 17000 vs 29000; succ probe at 23000 vs 29000; copy at 6000 vs 29000; rev at 12000 vs 29000

| run | text 17k | text 29k | succ 23k | succ 29k | copy 6k | copy 29k | rev 12k | rev 29k |
|---|---|---|---|---|---|---|---|---|
| lagro=1+relay=1 cue=1 s1 | 0.351 | 0.242 | 0.983 | 0.067 | 1.000 | 1.000 | 1.000 | 1.000 |
| lagro=1+relay=1 cue=0 s1 | 0.362 | 0.276 | 0.889 | 0.017 | 1.000 | 0.644 | 1.000 | 1.000 |
| lagro=1+relay=1+lagro_gate=1 cue=1 s1 | 0.351 | 0.242 | 0.994 | 0.072 | 1.000 | 1.000 | 1.000 | 1.000 |
| lagro=1+relay=1+xdg=8+lag_xdg=1 cue=1 s1 | 0.344 | 0.211 | 0.333 | 0.022 | 0.944 | 1.000 | 1.000 | 1.000 |
| lagro=1+relay=1+xdg=8+lag_xdg=1 cue=0 s1 | 0.349 | 0.218 | 0.533 | 0.017 | 0.944 | 0.639 | 1.000 | 1.000 |
| lagro=1+relay=1+xdg=8 cue=1 s1 | 0.334 | 0.219 | 0.961 | 0.089 | 1.000 | 1.000 | 1.000 | 1.000 |
| lagro=1+relay=1 cue=1 s2 | 0.370 | 0.260 | 0.950 | 0.028 | 1.000 | 1.000 | 1.000 | 1.000 |
| lagro=1+relay=1+xdg=8+lag_xdg=1 cue=1 s2 | 0.344 | 0.238 | 0.911 | 0.183 | 1.000 | 1.000 | 1.000 | 1.000 |

## Text and succ probes across the second half of the life (lines 17000, 20000, 23000, 26000, 29000)

| run | text | succ (23000..29000 every 1000) |
|---|---|---|
| baseline cue=1 s1 | 0.351 0.257 0.239 0.267 0.242 | 0.983 0.822 0.367 0.167 0.117 0.061 0.067 |
| xdg+lag_xdg cue=1 s1 | 0.344 0.240 0.267 0.231 0.211 | 0.333 0.083 0.028 0.006 0.011 0.000 0.022 |
| xdg (cells only) cue=1 s1 | 0.334 0.267 0.250 0.254 0.219 | 0.961 0.772 0.367 0.144 0.117 0.067 0.089 |
| lagro_gate cue=1 s1 | 0.351 0.258 0.239 0.269 0.242 | 0.994 0.856 0.444 0.222 0.128 0.072 0.072 |
| baseline cue=0 s1 | 0.362 0.309 0.267 0.268 0.276 | 0.889 0.278 0.006 0.000 0.022 0.006 0.017 |
| xdg+lag_xdg cue=0 s1 | 0.349 0.294 0.273 0.252 0.218 | 0.533 0.206 0.011 0.000 0.022 0.006 0.017 |
| baseline cue=1 s2 | 0.370 0.271 0.237 0.266 0.260 | 0.950 0.772 0.400 0.150 0.117 0.061 0.028 |
| xdg+lag_xdg cue=1 s2 | 0.344 0.237 0.218 0.228 0.238 | 0.911 0.733 0.689 0.383 0.283 0.211 0.183 |

## Plain answer

- **Text:** xdg+lag_xdg does **not** keep text better than the baseline in this life. The text probe at the end (line 29000) is
  0.211 vs 0.242 (cue=1 s1), 0.218 vs 0.276 (cue=0 s1) and 0.238 vs 0.260 (cue=1 s2). It is also lower right after the next copy phase
  (line 26000): 0.231 vs 0.267, 0.252 vs 0.268 and 0.228 vs 0.266. It starts from a slightly lower level at the end of the text phase
  (0.344 vs 0.351, 0.349 vs 0.362, 0.344 vs 0.370), and the loss after that is about the same or larger. So the toy result (text after
  copy lines 0.945 vs 0.85) does not carry over to this stream. xdg without lag_xdg and lagro_gate do not keep text better either
  (0.219 and 0.242 at the end, vs 0.242).
- **Succ:** the seeds disagree. Seed 1 cue=1: succ is learned much worse (0.333 at line 23000 vs 0.983) and is not kept (0.022 vs 0.067).
  Seed 2 cue=1: succ reaches about the same level (0.911 vs 0.950) and is kept better (0.183 vs 0.028 at line 29000; 0.689 vs 0.400 at line 25000).
  Seed 1 cue=0: worse at line 23000 (0.533 vs 0.889) and the same at the end (0.017 vs 0.017). So there is no consistent succ-keeping gain;
  one seed out of two shows one.
- **Copy during text:** with xdg+lag_xdg (s1 cue=1) the copy probe falls from 0.994 at line 12000 to 0.50 at line 13000 and to 0.32 at line 17000.
  The baseline keeps 0.99, and on s2 xdg+lag_xdg keeps 0.89. Copy and rev come back to 1.00 in P4/P5 in every cue=1 run.
- **Cost:** first-met shares are lower in every phase for xdg+lag_xdg. s1 cue=1: P0 b 0.915 vs 0.977, P1 b 0.621 vs 0.795, P3 b 0.314 vs 0.505,
  P4 b 0.914 vs 0.991, P5 b 0.913 vs 1.000. On s2 cue=1 the losses are smaller (P0 0.934 vs 0.976, P1 0.700 vs 0.797, P3 0.524 vs 0.506, P4 0.959 vs 0.988,
  P5 0.988 vs 1.000). The first P4 windows are lower (s1: 0.87 0.78 0.97 0.72 vs 0.95 0.98 0.98 0.99). Bits of the whole life are higher:
  +61,335 (889,862 vs 828,527, s1 cue=1), +54,937 (900,181 vs 845,244, s1 cue=0) and +53,341 (882,755 vs 829,414, s2 cue=1). Most of the extra
  is in answer bytes (+41k, +33k, +24k), then input bytes, then text bytes (+3k, +4k, +6k).
- **The other two variants (seed 1, cue=1):** xdg cells without lag_xdg give +19,038 bits (847,565), mostly text bytes (+6k), with first-met
  shares almost unchanged (P3 0.473 vs 0.505). lagro_gate=1 is the only variant below the baseline: 826,281 bits (−2,246), P3 0.529 vs 0.505,
  and probes almost identical to the baseline. It does not keep text or succ better either (0.242 and 0.072 at the end, vs 0.242 and 0.067).

NOT_RUN: seed 2 of the cue=0 runs, of xdg without lag_xdg and of lagro_gate; seeds ≥3.
