# Rule cells on the general battery and in English/Chinese lives (23 November 2026)

Question: the persistent rule cells (`net12_rc.py`, `rc=64 rc_gate=1`, with `lagro=1 relay=1`, as run in `RULECELLS_REPORT.md`) keep the
learned succ mapping at full size. Is the architecture still general? Does it lose anything on natural text, keeping or memory?

This is development level. No gate, seal or external review is claimed. Everything here was run on this machine (4 cores,
`OMP_NUM_THREADS=1`, 4 jobs at a time). The helper branch `claude/newnet-rcgeneral-sei232` was made from `claude/magical-albattani-sei232`
@ `32db8c13`. No existing `.py` file was changed. Two new files were added:

- `net121.py`: an alias so that scripts which pick a net by number (`importlib "net%d"`) can select `net12_rc`. It is one line,
  `from net12_rc import Tissue`.
- `rcgeneral_run.py`: the driver, which follows the scheme of `battery12_run.py`.

Battery lines are in `search_results_rc.tsv`; logs are in `rcgeneral_runs/`. All 24 jobs took about 1 hour of wall time. The Chinese
text `data/hongloumeng.txt` (git-ignored, not committed) was downloaded from gutenberg.pglaf.org/2/4/2/6/24264/24264-0.txt.

**RC** = `net=121` + K8 + `lagro=1 relay=1 rc=64 rc_gate=1`. The other rule-cell parameters keep their defaults: rc_k=3, rc_tau=8,
rc_vig=0.4, rc_eta=0.02.

- K8 = `L=16 cC=64 joint=1 mesh=1 mesh_base=0 mesh_dag=5 c_k_win=4 c_th_on=0.9 c_th_match=0.75`.
- The battery and the lives are exactly as in `BATTERY12_REPORT.md`:
  - the battery runs through `search_arch.py`;
  - the lives are `t13_life8.py net=10 life=enzh|zhen n_values=256 NT=10000` + options, where the later `net=121` overrides `net=10`.
- B2 (net12 `lagro=1 relay=1`) and net10 K8 are the numbers from `BATTERY12_REPORT.md`; they were not re-run here.

## Battery (score = mean of the six)

| candidate | seed | moby | code | keep | keep4 | keep2 | mem | score |
|---|---|---|---|---|---|---|---|---|
| net10 K8 (published) | 1 | 0.399 | 0.406 | 1.000 | 1.000 | 1.000 | 1.000 | 0.801 |
| net10 K8 (published) | 2 | 0.391 | 0.397 | 1.000 | 1.000 | 1.000 | 1.000 | 0.798 |
| net10 K8 (published) | 3 | 0.389 | 0.389 | 1.000 | 1.000 | 1.000 | 1.000 | 0.796 |
| B2 net12 | 1 | 0.399 | 0.406 | 1.000 | 1.000 | 1.000 | 1.000 | 0.801 |
| B2 net12 | 2 | 0.393 | 0.402 | 1.000 | 1.000 | 1.000 | 1.000 | 0.799 |
| B2 net12 | 3 | 0.388 | 0.393 | 1.000 | 1.000 | 1.000 | 1.000 | 0.797 |
| **RC** | 1 | 0.399 | 0.406 | 1.000 | 1.000 | 1.000 | 1.000 | 0.801 |
| **RC** | 2 | 0.391 | 0.401 | 1.000 | 1.000 | 1.000 | 1.000 | 0.799 |
| **RC** | 3 | 0.388 | 0.393 | 1.000 | 1.000 | 1.000 | 1.000 | 0.797 |

Three-seed means:

| candidate | moby | code | score | moby spread | code spread |
|---|---|---|---|---|---|
| net10 K8 | 0.393 | 0.397 | 0.798 | 0.389-0.399 | 0.389-0.406 |
| B2 | 0.393 | 0.400 | 0.799 | 0.388-0.399 | 0.393-0.406 |
| **RC** | 0.393 | 0.400 | 0.799 | 0.388-0.399 | 0.393-0.406 |

Paired differences by seed (seeds 1 / 2 / 3):

| comparison | moby | code |
|---|---|---|
| RC minus B2 | 0.000 / -0.002 / 0.000 | 0.000 / -0.001 / 0.000 |
| RC minus K8 | 0.000 / 0.000 / -0.001 | 0.000 / +0.004 / +0.004 |

Keeping at G = 8, 4 and 2, and memory, stay at 1.000 for every seed.

## English / Chinese lives

Each cell reads: first text after it -> after the second text (kept share = ratio).

| life | candidate | seed 1 | seed 2 | seed 3 | mean kept | second text after it (mean) |
|---|---|---|---|---|---|---|
| English -> Chinese | net10 K8 (published) | 0.371 -> 0.364 (0.981) | 0.371 -> 0.358 (0.965) | 0.359 -> 0.356 (0.992) | 0.979 | 0.334 |
| English -> Chinese | B2 | 0.372 -> 0.368 (0.989) | 0.373 -> 0.363 (0.973) | 0.363 -> 0.360 (0.992) | 0.985 | 0.347 |
| English -> Chinese | **RC** | 0.372 -> 0.368 (0.989) | 0.372 -> 0.363 (0.976) | 0.362 -> 0.359 (0.992) | 0.986 | 0.348 |
| Chinese -> English | net10 K8 (published) | 0.336 -> 0.311 (0.926) | 0.341 -> 0.311 (0.912) | 0.337 -> 0.305 (0.905) | 0.914 | 0.358 |
| Chinese -> English | B2 | 0.349 -> 0.332 (0.951) | 0.354 -> 0.329 (0.929) | 0.347 -> 0.328 (0.945) | 0.942 | 0.360 |
| Chinese -> English | **RC** | 0.350 -> 0.332 (0.949) | 0.354 -> 0.328 (0.927) | 0.347 -> 0.329 (0.948) | 0.941 | 0.360 |

RC, second text after it, by seed:

- English -> Chinese: 0.352 / 0.349 / 0.342.
- Chinese -> English: 0.364 / 0.363 / 0.352.

These are within 0.002 of B2 on each seed.

After the return (2,000 bytes of the first text; first text / second text):

- RC English -> Chinese: 0.364 / 0.343, 0.363 / 0.343, 0.358 / 0.341.
- RC Chinese -> English: 0.365 / 0.362, 0.358 / 0.359, 0.361 / 0.349.
- B2: within 0.002 of these on each seed.

## Verdict

**No test gets worse beyond the seed spread. The rule cells change nothing measurable on natural text, keeping or memory.**

- **Battery.** RC equals B2 to three decimals in every test except two:
  - seed 2 Moby Dick: -0.002;
  - seed 2 code: -0.001.

  The spread between seeds is 0.011 (moby) and 0.013 (code). The score is 0.799 against B2 0.799 and K8 0.798. Keeping at three depths
  and memory stay at 1.000 for all seeds.
- **Lives.**
  - Kept share, English -> Chinese: RC 0.986, B2 0.985, K8 0.979.
  - Kept share, Chinese -> English: RC 0.941, B2 0.942, K8 0.914.
  - Per-seed differences from B2 are at most 0.003 in kept share and at most 0.002 in any reading. The seed spread of the kept share is
    0.02.
  - The advantage of B2 over K8 in Chinese -> English keeping (+2.7 points) remains.

On this battery and these lives, net12_rc RC is as general as net12 B2 and net10 K8. Together with `RULECELLS_REPORT.md` (succ at the end
0.82 against 0.07, seed 1, cue=1), these runs show no cost for that gain on these tests.

## Limits / NOT_RUN

- One size, short lives, three seeds.
- The B2 and net10 K8 numbers are published numbers from `BATTERY12_REPORT.md` and `LIFE_EN_ZH.md`; they were not re-run in this
  session.
- The keeping and memory tests are at their ceiling (1.000), so they could show only a loss, not a gain.
- How often the rule cells were recruited or won on natural text was not logged; the scripts do not print it. So these runs do not show
  whether the rule cells' share of the gate is small on text or whether it is used without harm.
- Not run:
  - `rc_gate=2`, other rc settings, and `ag=1`;
  - the "second text only" control;
  - `t19_putback.py`, which imports net10 directly.
