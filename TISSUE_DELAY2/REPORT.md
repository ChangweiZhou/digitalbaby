# Delays in the base of 3b486e0a (19 November 2026)

The reviewer asked for this in a message to E57's worker at 08:12 UTC on 19 November. The material is the reviewer's folder `reviewer_tissue_20261119` at 3b486e0a on `claude/vibrant-cerf-sje726`, copied here as `TISSUE_DELAY2/`.

What I changed in the copy:

- **`tissue.c`**, additions marked `TISSUE_DELAY2`:
  - A delay for every pair (cell, member of its neighbourhood that is a cell), drawn once at birth. A connection made later therefore has the delay of its pair.
  - Histories of outputs and of fast traces.
  - The arrival, the largest-arriving value of the control `gain=3`, and both withdrawal sums of `prune_only` use the sender's activity **as it arrives**: the delayed output, and the sender's fast trace as it was D steps before.
- **`tis.py`**: four fields appended to the structure.
- **`delay.py`** and **`delay_tables.py`** (from `TISSUE_DELAY`, adapted) write `delay/*.json` and `DELAY_TABLES.md`.
- **`q.sh`, `q2.sh`, `q3.sh`**: the runs.

All figures below are generated from the receipts by a script (`delay/*.json`). Each is from one run. Nothing is confirmed independently.

## Terms

- **N**: `rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/4096 make_period=1024`. **Nt**: N with `tau=u2-128`.
- **The delays**: random 0–8, 0–12, 0–16 or 0–32 on every pair, or 32 on a quarter of the pairs. I made the runs with 0–8 and 0–12 after the first rows pointed to smaller delays.
- **Readings**: as in `TISSUE_DELAY`.
  - **(a)**: the tool as it is.
  - **(b)**: no floor, each cell divided by its own standard deviation.
  - **Near cells**: reach.py's set.
- **Levels**: lag 0: 0.1103. Lags 1 to 4: c0(L) + 0.02 of the window; this is my reading of how the reviewer's levels were set.
- **Divergence**: two copies, the second with its most active cell's potential changed by 1e-9; the largest difference 64 and 512 bytes later.

## The question, on 16,384 bytes

**Is there a setting in which, on three seeds:**

- the tool reads the present byte at d = 1, 2, 4, 8 and 10;
- the bytes one, two and three back are above their levels among the near cells;
- the largest output there exceeds 1/2;
- and two copies do not diverge?

**No setting tried meets all of it.** The full table is in `DELAY_TABLES.md`. These are the rows of N:

| Base | Delay | Seed | S0 (a) at d = 1/2/4/8/10 | all five | near L1 (a/b) | near L2 | near L3 | near largest output | no divergence | all | connections a cell | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | no delay | 73129 | 0.979 / 0.814 / 0.616 / 0.370 / 0.297 | yes | yes/yes | no/no | no/no | 1.000 | yes | **no** | 3.3 | 262 |
| N | no delay | 93683 | 0.995 / 0.857 / 0.562 / 0.259 / 0.183 | yes | yes/yes | no/no | no/no | 1.000 | yes | **no** | 3.6 | 279 |
| N | no delay | 116089 | 0.998 / 0.755 / 0.567 / 0.161 / 0.096 | no | no/no | no/no | no/no | 1.000 | yes | **no** | 3.5 | 279 |
| N | random 0–8 (made after the first rows) | 73129 | 0.996 / 0.802 / 0.529 / 0.177 / 0.098 | no | yes/yes | yes/yes | no/no | 1.000 | yes | **no** | 3.3 | 378 |
| N | random 0–8 (made after the first rows) | 93683 | 0.996 / 0.820 / 0.511 / 0.186 / 0.087 | no | yes/yes | yes/yes | yes/yes | 1.000 | yes | **no** | 3.6 | 406 |
| N | random 0–8 (made after the first rows) | 116089 | 0.998 / 0.860 / 0.536 / 0.132 / 0.103 | no | yes/yes | yes/yes | no/no | 1.000 | yes | **no** | 3.5 | 410 |
| N | random 0–12 (made after the first rows) | 73129 | 0.990 / 0.804 / 0.482 / 0.090 / 0.074 | no | yes/yes | yes/yes | yes/yes | 1.000 | yes | **no** | 3.3 | 382 |
| N | random 0–12 (made after the first rows) | 93683 | 0.995 / 0.836 / 0.459 / 0.090 / 0.053 | no | yes/yes | yes/yes | no/no | 1.000 | yes | **no** | 3.6 | 400 |
| N | random 0–12 (made after the first rows) | 116089 | 0.991 / 0.800 / 0.396 / 0.069 / 0.074 | no | yes/yes | yes/yes | no/no | 1.000 | yes | **no** | 3.5 | 400 |
| N | random 0–16 on every pair of cells | 73129 | 0.993 / 0.795 / 0.394 / 0.098 / 0.064 | no | yes/yes | yes/yes | yes/yes | 1.000 | yes | **no** | 3.3 | 397 |
| N | random 0–16 on every pair of cells | 93683 | 0.987 / 0.843 / 0.385 / 0.109 / 0.066 | no | no/no | yes/yes | yes/yes | 1.000 | yes | **no** | 3.6 | 414 |
| N | random 0–16 on every pair of cells | 116089 | 0.989 / 0.769 / 0.512 / 0.106 / 0.091 | no | no/no | yes/yes | no/no | 1.000 | yes | **no** | 3.4 | 398 |
| N | random 0–32 | 73129 | 0.979 / 0.744 / 0.175 / 0.036 / 0.029 | no | no/no | no/no | no/no | 1.000 | yes | **no** | 3.3 | 402 |
| N | random 0–32 | 93683 | 0.984 / 0.697 / 0.207 / 0.111 / 0.067 | no | no/no | yes/yes | no/no | 1.000 | yes | **no** | 3.5 | 418 |
| N | random 0–32 | 116089 | 0.980 / 0.791 / 0.190 / 0.087 / 0.084 | no | no/no | no/no | no/no | 1.000 | yes | **no** | 3.4 | 403 |
| N | 32 on 1/4 of the pairs | 73129 | 0.977 / 0.737 / 0.371 / 0.144 / 0.087 | no | yes/yes | yes/yes | no/no | 1.000 | yes | **no** | 3.3 | 353 |
| N | 32 on 1/4 of the pairs | 93683 | 0.986 / 0.677 / 0.277 / 0.166 / 0.100 | no | yes/yes | yes/yes | no/no | 1.000 | yes | **no** | 3.5 | 369 |
| N | 32 on 1/4 of the pairs | 116089 | 0.988 / 0.773 / 0.313 / 0.087 / 0.058 | no | no/no | yes/yes | no/no | 1.000 | yes | **no** | 3.4 | 357 |

The rows of Nt are in `DELAY_TABLES.md`. In every Nt row the near cells score 0.02 to 0.08 at lags 1 to 3, and S0 at d = 8 is below its level.

**The closest setting, N with random delays 0–8, on all three seeds:**

- **Met**: the present byte is above its level at d = 1, 2, 4 and 8; the bytes one and two back are above their levels among the near cells; the largest output there is 1.000; there is no divergence.
- **Not met at d = 10**: the present byte there is 0.098 / 0.087 / 0.103.
- **Not met for the byte three back** among the near cells: above its level on seed 93683 only.

| Seed | near (b), lags 1 / 2 / 3 | levels |
|---|---|---|
| 73129 | 0.203 / 0.208 / 0.142 | 0.090 / 0.076 / 0.182 |
| 93683 | 0.178 / 0.243 / 0.194 | 0.096 / 0.074 / 0.173 |
| 116089 | 0.144 / 0.226 / 0.176 | 0.100 / 0.075 / 0.180 |

**The base N without delay** reads the present byte at d = 1 to 10 on seeds 73129 and 93683, but not at d = 10 on 116089 (0.096). Among its near cells only the byte one back is above its level, on 73129 and 93683.

**Divergence** is 0 after 64 and 512 bytes in every row but one: Nt with random 0–32 on seed 73129, which shows 7.3e-08 after 512 bytes.

## N with random delays 0–8 over 65,536 bytes, four windows

| Seed | Window | S0 (a) at d = 1 / 2 / 4 / 8 / 10 | near (b), lags 1 / 2 / 3 | levels at lags 1 / 2 / 3 | near largest output | connections a cell |
|---|---|---|---|---|---|---|
| 73129 | 12288–16383 | 0.996 / 0.802 / 0.529 / 0.177 / 0.098 | 0.203 / 0.208 / 0.142 | 0.090 / 0.076 / 0.182 | 1.000 | 3.3 |
| 73129 | 28672–32767 | 0.997 / 0.761 / 0.522 / 0.171 / 0.102 | 0.200 / 0.218 / 0.158 | 0.106 / 0.080 / 0.176 | 1.000 | 3.3 |
| 73129 | 45056–49151 | 0.994 / 0.792 / 0.529 / 0.172 / 0.099 | 0.207 / 0.232 / 0.164 | 0.098 / 0.077 / 0.168 | 1.000 | 3.3 |
| 73129 | 61440–65535 | 0.995 / 0.797 / 0.520 / 0.175 / 0.104 | 0.208 / 0.240 / 0.172 | 0.102 / 0.073 / 0.181 | 1.000 | 3.3 |
| 73129 | divergence after 64 / 512 bytes | 0 / 0 | | | | |
| 93683 | 12288–16383 | 0.996 / 0.820 / 0.511 / 0.186 / 0.087 | 0.178 / 0.243 / 0.194 | 0.096 / 0.074 / 0.173 | 1.000 | 3.6 |
| 93683 | 28672–32767 | 0.993 / 0.891 / 0.498 / 0.179 / 0.095 | 0.186 / 0.234 / 0.176 | 0.097 / 0.073 / 0.173 | 1.000 | 3.6 |
| 93683 | 45056–49151 | 0.995 / 0.889 / 0.519 / 0.195 / 0.089 | 0.192 / 0.226 / 0.181 | 0.103 / 0.077 / 0.166 | 1.000 | 3.6 |
| 93683 | 61440–65535 | 0.995 / 0.892 / 0.510 / 0.171 / 0.089 | 0.183 / 0.210 / 0.153 | 0.100 / 0.081 / 0.186 | 1.000 | 3.6 |
| 93683 | divergence after 64 / 512 bytes | 0 / 0 | | | | |
| 116089 | 12288–16383 | 0.998 / 0.860 / 0.536 / 0.132 / 0.103 | 0.144 / 0.226 / 0.176 | 0.100 / 0.075 / 0.180 | 1.000 | 3.5 |
| 116089 | 28672–32767 | 0.997 / 0.859 / 0.505 / 0.121 / 0.101 | 0.147 / 0.222 / 0.200 | 0.101 / 0.077 / 0.170 | 1.000 | 3.5 |
| 116089 | 45056–49151 | 0.999 / 0.856 / 0.589 / 0.124 / 0.102 | 0.148 / 0.224 / 0.199 | 0.093 / 0.082 / 0.170 | 1.000 | 3.5 |
| 116089 | 61440–65535 | 0.997 / 0.854 / 0.572 / 0.115 / 0.093 | 0.120 / 0.215 / 0.182 | 0.091 / 0.074 / 0.178 | 1.000 | 3.5 |
| 116089 | divergence after 64 / 512 bytes | 0 / 0 | | | | |

- The bytes one and two back are above their levels among the near cells in all 12 windows.
- The byte three back is above its level among the near cells in 6 of 12 windows.
- The present byte is above its level at d = 8 in every window, the lowest being 0.115. At d = 10 it is 0.087 to 0.104, below 0.1103 in every window.
- The largest output of the near cells is 1.000 in every window, and the divergence is 0 on all three seeds.
- The readings show no trend over 65,536 bytes.

## What the numbers say

1. **In this base, short delays (0 to 8 or 0 to 12 steps) bring the earlier bytes to the cells near the output lines.** Delays of 0 to 32 carry them at the middle distances. Near the output they score 0.021 to 0.094 at lags 1 to 3, above a level only at lag 2 on seed 93683 (0.078 against 0.074).
2. **The trade-off is the present byte far out.** Every delay setting lowers S0 at d = 8 below the undelayed base on every seed, and at d = 10 on every seed but one: random 0–8 on seed 116089 gives 0.103 at d = 10, against the base's 0.096. With delays 0 to 8, d = 8 stays above the level and d = 10 does not.
3. **The byte three back among the near cells is at its level's edge.** In the 65,536-byte runs it is above its level in about half of the windows. The lag-3 level itself (c0(3) + 0.02, 0.166 to 0.186) varies between windows.
4. **`tau=u2-128` combined with these delays weakens the near cells** in every row.

## Cost

33 runs took 4.64 CPU-hours (`delay/*.json`), with at most three processes of one thread each.
