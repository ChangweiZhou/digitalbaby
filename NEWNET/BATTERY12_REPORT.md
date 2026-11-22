# net12 on the general battery and in English/Chinese lives (22 November 2026)

Question: net12 (net10 plus a window of W=8 lag cells with a direct readout `lagro=1`, a gated relay `relay=1` and context-gated
readout groups `ctxg=16`, `ctxg_err`) generalises far better than net10 on rule streams (`RULES_REPORT.md`). Does it keep net10 K8's
quality on natural text, keeping and memory? Development level. No gate, seal or external review is claimed. Everything here was run
on this machine (4 cores, `OMP_NUM_THREADS=1`, 4 jobs at a time) from branch `claude/magical-albattani-sei232` @ `4a841fd0`. No
existing `.py` file was changed. Driver: `battery12_run.py`. Battery lines: `search_results_net12.tsv`. Logs: `battery12_runs/`.
Wall time for all 72 jobs: about 1 hour.

K8 = `L=16 cC=64 joint=1 mesh=1 mesh_base=0 mesh_dag=5 c_k_win=4 c_th_on=0.9 c_th_match=0.75`. Every candidate is `net=12` + K8 + the
options shown. The battery is the one in `SEARCH_SUMMARY.md`, run with `search_arch.py`. The lives use
`t13_life8.py net=10 life=enzh|zhen n_values=256 NT=10000` + options (TA=16000, TB=16000, TR=2000 defaults); `net=12` comes later
in the options and overrides `net=10`.

## Check: net12 with nothing new switched on equals net10 K8

B0 (`net=12` + K8) seed 1 gives moby 0.399, code 0.406, keep/keep4/keep2/mem 1.000. These are identical to the published net10 K8
seed 1 row (`search_results_b13.tsv`, K8). The lives are also identical to `LIFE_EN_ZH.md` K8 seed 1:

- English -> Chinese: 0.371 / 0.364 / 0.337.
- Chinese -> English: 0.336 / 0.311 / 0.367.

So the net12 code path with all new compartments off reproduces net10 exactly.

## Battery (score = mean of the six)

| candidate | options added to net=12 + K8 | seed | moby | code | keep | keep4 | keep2 | mem | score |
|---|---|---|---|---|---|---|---|---|---|
| net10 K8 (published) | - | 1 | 0.399 | 0.406 | 1.000 | 1.000 | 1.000 | 1.000 | 0.801 |
| net10 K8 (published) | - | 2 | 0.391 | 0.397 | 1.000 | 1.000 | 1.000 | 1.000 | 0.798 |
| net10 K8 (published) | - | 3 | 0.389 | 0.389 | 1.000 | 1.000 | 1.000 | 1.000 | 0.796 |
| B0 | (none) | 1 | 0.399 | 0.406 | 1.000 | 1.000 | 1.000 | 1.000 | 0.801 |
| B1 | lagro=1 | 1 | 0.400 | 0.406 | 1.000 | 1.000 | 1.000 | 1.000 | 0.801 |
| B2 | lagro=1 relay=1 | 1 | 0.399 | 0.406 | 1.000 | 1.000 | 1.000 | 1.000 | 0.801 |
| B2 | lagro=1 relay=1 | 2 | 0.393 | 0.402 | 1.000 | 1.000 | 1.000 | 1.000 | 0.799 |
| B2 | lagro=1 relay=1 | 3 | 0.388 | 0.393 | 1.000 | 1.000 | 1.000 | 1.000 | 0.797 |
| B3 | lagro=1 relay=1 ctxg=16 | 1 | 0.400 | 0.408 | 1.000 | 1.000 | 1.000 | 1.000 | 0.801 |
| B3 | lagro=1 relay=1 ctxg=16 | 2 | 0.392 | 0.400 | 1.000 | 1.000 | 1.000 | 1.000 | 0.799 |
| B3 | lagro=1 relay=1 ctxg=16 | 3 | 0.387 | 0.391 | 1.000 | 1.000 | 1.000 | 1.000 | 0.796 |
| B4 | lagro=1 relay=1 ctxg=16 ctxg_err=1.0 | 1 | 0.399 | 0.406 | 1.000 | 1.000 | 1.000 | 1.000 | 0.801 |

Three-seed means:

| candidate | moby | code | score | moby spread over seeds | code spread over seeds |
|---|---|---|---|---|---|
| net10 K8 (published) | 0.393 | 0.397 | 0.798 | 0.389-0.399 | 0.389-0.406 |
| B2 | 0.393 | 0.400 | 0.799 | 0.388-0.399 | 0.393-0.406 |
| B3 | 0.393 | 0.400 | 0.799 | 0.387-0.400 | 0.391-0.408 |

Paired by seed against net10 K8 (B minus K8, seeds 1/2/3):

- B2: moby +0.000 / +0.002 / -0.001; code +0.000 / +0.005 / +0.004.
- B3: moby +0.001 / +0.001 / -0.002; code +0.002 / +0.003 / +0.002.

Keeping at G=8, 4 and 2 and memory stay at 1.000 for every candidate and every seed.

## English / Chinese lives (first text after it -> after the second text; kept share = ratio)

| life | candidate | seed 1 | seed 2 | seed 3 | mean kept | second text after it (mean) |
|---|---|---|---|---|---|---|
| English -> Chinese | net10 K8 (published) | 0.371 -> 0.364 (0.98) | 0.371 -> 0.358 (0.96) | 0.359 -> 0.356 (0.99) | 0.98 | 0.334 |
| English -> Chinese | B0 net12 (check) | 0.371 -> 0.364 (0.98) | NOT_RUN | NOT_RUN | - | 0.337 (s1) |
| English -> Chinese | B2 | 0.372 -> 0.368 (0.99) | 0.373 -> 0.363 (0.97) | 0.363 -> 0.360 (0.99) | 0.98 | 0.347 |
| English -> Chinese | B3 | 0.372 -> 0.366 (0.98) | 0.373 -> 0.362 (0.97) | 0.360 -> 0.359 (1.00) | 0.98 | 0.344 |
| Chinese -> English | net10 K8 (published) | 0.336 -> 0.311 (0.93) | 0.341 -> 0.311 (0.91) | 0.337 -> 0.305 (0.91) | 0.91 | 0.358 |
| Chinese -> English | B0 net12 (check) | 0.336 -> 0.311 (0.93) | NOT_RUN | NOT_RUN | - | 0.367 (s1) |
| Chinese -> English | B2 | 0.349 -> 0.332 (0.95) | 0.354 -> 0.329 (0.93) | 0.347 -> 0.328 (0.95) | 0.94 | 0.360 |
| Chinese -> English | B3 | 0.345 -> 0.326 (0.95) | 0.350 -> 0.320 (0.91) | 0.343 -> 0.321 (0.94) | 0.93 | 0.360 |

After the return (2,000 bytes of the first text):

- B2: English 0.365 / 0.364 / 0.357; Chinese 0.365 / 0.358 / 0.362.
- B3: English 0.363 / 0.364 / 0.357; Chinese 0.362 / 0.353 / 0.359.

These are at or above the first text's level, as for K8.

The net10 K8 seed 2/3 rows are the published numbers in `LIFE_EN_ZH.md`; they were not re-run here. Seed 1 was re-run as B0 and
matched. Not run: the "second text only" control for net12, and `t19_putback.py`, which imports net10 directly and so cannot run net12
without changing or adding a script.

## Answer

**The new compartments cost nothing measurable on natural text, keeping or memory.**

- **Battery.** Score: B2 0.799 and B3 0.799 against net10 K8 0.798 (three seeds each). Moby Dick: identical means (0.393). Code:
  +0.003 on average (paired differences 0.000 to +0.005). Every single-seed difference is within ±0.005, smaller than the spread
  between seeds (0.010-0.017). Keeping at three depths and memory stay at 1.000 everywhere.
- **Seed 1, B1 to B4.** Each compartment added in turn, including `ctxg_err=1.0`, changes the score by at most +0.0005.
- **English/Chinese lives.**
  - English -> Chinese: keeping is unchanged (0.98 for all).
  - Chinese -> English: keeping is slightly better, 0.94 (B2) and 0.93 (B3) against 0.91. This is +2-3 points, higher than K8 on
    every seed for B2 and on 2 of 3 seeds for B3 (seed 2: 0.91 = 0.91).
  - The Chinese text is also read better: 0.350 / 0.346 against 0.338 when it comes first.
  - In English -> Chinese, the second (Chinese) text after it is also higher: 0.347 / 0.344 against 0.334.
  - English is read the same (0.369 / 0.368 against 0.367).

So these runs show no cost from the new compartments on natural text, keeping or memory. On this battery and these lives, net12 B2/B3
stays as general as net10 K8, with small gains on the Chinese text.

Limits:

- One size, short lives, three seeds; the net10 seed 2/3 rows are published numbers, not re-run in this session.
- The keeping/memory tests are at their ceiling (1.000), so they cannot show a gain, only a loss.
- B1 and B4 were run on seed 1 only.
