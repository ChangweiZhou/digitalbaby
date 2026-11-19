# Long axons with memories (19 November 2026)

The reviewer asked for this in a message to E57's worker at 09:53 UTC on 19 November.

What is in this folder:

- **`tissue.c`**: the reviewer's file at 341741c26 with the delay additions of `TISSUE_DELAY2` applied as a patch.
- **`tis.py`**: mine, with the option `long=K-R` written from the reviewer's message. It is built with `-ffp-contract=off`, as the reviewer's.
- **`delay.py`** and **`delay_tables.py`**: the readings, as in `TISSUE_DELAY2`.
- **`eqcheck.py`**: item 1.
- **`q.sh`, `q2.sh`**: the runs.

All figures are generated from the receipts by a script. Each is from one run. Nothing is confirmed independently.

## 1. The option long=K-R in this program, against the reviewer's

**Line**: the line of the reviewer's batch 40 with `long=8-1/2`, seed 73129, 16,384 bytes. The reviewer's folder at 341741c26 was run as fetched (`qual/eqcheck_73129_16384.json`).

- **Long cells**: equal, 156 cells.
- **Every cell's members and starting sources**: equal.
- **Outputs of every cell after every byte**: identical, largest difference 0.0.
- **Connections and strengths at the end**: identical, 5,312 connections.

## 2. long=8-1/2 with five memories, three seeds, 16,384 bytes

**Line L**: `rule=timing prune_only=1 norm=1 lines_learn=1 we=3/4 wi=1/4 gain=3 rho=1 gmax=8 mk_long=1 a_make=1/16384 make_period=1024 start_w=1/64 long=8-1/2`.

**The memories:**

- (i) none;
- (ii) delays drawn from 0 to 8 for every pair of cells;
- (iii) time constants up to 64 (`tau=u2-64`);
- (iv) time constants up to 256;
- (v) delays 0–8 together with time constants up to 64.

**Readings and levels:**

- **margin**: the smallest of S0 − 0.1103 by the tool at d = 1, 2, 4.
- **near**: reach.py's set (the drivers of the output lines and their neighbourhoods), read by (a) the tool as it is and (b) with no floor and per-cell standardisation.
- **Levels**: lag 0: 0.1103. Lags 1 to 4: c0(L) + 0.02 of the window, my reading of how the reviewer's levels were set.
- **With long axons the tissue is shallow**: the largest distance is about 7, so S0 at d = 8 and 10 is mostly absent.

| Memory | Seed | S0 (a) at d = 1/2/4/8/10 | margin | near L0 | near L1 | near L2 | near L3 | near L0–L3 all above | near largest output | divergence 64/512 | connections a cell | CPU s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| (i) no memory | 73129 | 0.976 / 0.692 / 0.327 / — / — | +0.216 | 0.274 / 0.293 [0.110] | 0.048 / 0.068 [0.090] | 0.042 / 0.040 [0.076] | 0.071 / 0.072 [0.182] | **no** | 1.000 | 0 / 0 | 2.7 | 229 |
| (i) no memory | 93683 | 0.875 / 0.559 / 0.281 / — / — | +0.171 | 0.256 / 0.260 [0.110] | 0.041 / 0.041 [0.096] | 0.027 / 0.027 [0.074] | 0.052 / 0.055 [0.173] | **no** | 1.000 | 0 / 0 | 2.8 | 232 |
| (i) no memory | 116089 | 0.946 / 0.591 / 0.329 / — / — | +0.219 | 0.297 / 0.303 [0.110] | 0.043 / 0.043 [0.100] | 0.026 / 0.026 [0.075] | 0.053 / 0.054 [0.180] | **no** | 1.000 | 0 / 0 | 2.7 | 221 |
| (ii) delays 0–8 | 73129 | 0.968 / 0.640 / 0.329 / 0.030 / — | +0.218 | 0.270 / 0.271 [0.110] | 0.175 / 0.175 [0.090] | 0.087 / 0.087 [0.076] | 0.091 / 0.091 [0.182] | **no** | 1.000 | 0 / 0 | 2.5 | 359 |
| (ii) delays 0–8 | 93683 | 0.847 / 0.473 / 0.308 / 0.079 / — | +0.198 | 0.231 / 0.231 [0.110] | 0.220 / 0.220 [0.096] | 0.122 / 0.122 [0.074] | 0.089 / 0.089 [0.173] | **no** | 1.000 | 0 / 0 | 2.5 | 368 |
| (ii) delays 0–8 | 116089 | 0.913 / 0.639 / 0.389 / 0.024 / — | +0.279 | 0.322 / 0.324 [0.110] | 0.191 / 0.191 [0.100] | 0.027 / 0.027 [0.075] | 0.056 / 0.056 [0.180] | **no** | 1.000 | 0 / 0 | 2.5 | 358 |
| (iii) time constants up to 64 | 73129 | 0.965 / 0.596 / 0.188 / 0.043 / — | +0.078 | 0.163 / 0.163 [0.110] | 0.152 / 0.152 [0.090] | 0.103 / 0.103 [0.076] | 0.076 / 0.077 [0.182] | **no** | 1.000 | 0 / 0 | 3.3 | 265 |
| (iii) time constants up to 64 | 93683 | 0.818 / 0.527 / 0.185 / 0.047 / — | +0.075 | 0.169 / 0.169 [0.110] | 0.182 / 0.182 [0.096] | 0.124 / 0.124 [0.074] | 0.108 / 0.108 [0.173] | **no** | 1.000 | 0 / 0 | 3.0 | 252 |
| (iii) time constants up to 64 | 116089 | 0.933 / 0.653 / 0.270 / 0.103 / 0.023 | +0.159 | 0.215 / 0.216 [0.110] | 0.225 / 0.226 [0.100] | 0.133 / 0.133 [0.075] | 0.085 / 0.086 [0.180] | **no** | 1.000 | 0 / 0 | 2.9 | 243 |
| (iv) time constants up to 256 | 73129 | 0.928 / 0.537 / 0.229 / 0.076 / — | +0.119 | 0.174 / 0.173 [0.110] | 0.184 / 0.184 [0.090] | 0.143 / 0.145 [0.076] | 0.157 / 0.155 [0.182] | **no** | 1.000 | 0 / 0 | 2.6 | 210 |
| (iv) time constants up to 256 | 93683 | 0.863 / 0.398 / 0.152 / 0.041 / — | +0.042 | 0.135 / 0.133 [0.110] | 0.168 / 0.166 [0.096] | 0.141 / 0.140 [0.074] | 0.132 / 0.130 [0.173] | **no** | 1.000 | 0 / 0 | 2.6 | 212 |
| (iv) time constants up to 256 | 116089 | 0.940 / 0.627 / 0.231 / 0.071 / — | +0.121 | 0.228 / 0.225 [0.110] | 0.202 / 0.201 [0.100] | 0.160 / 0.160 [0.075] | 0.124 / 0.123 [0.180] | **no** | 1.000 | 0 / 0 | 2.6 | 214 |
| (v) delays 0–8 and time constants up to 64 | 73129 | 0.922 / 0.561 / 0.171 / 0.047 / — | +0.061 | 0.156 / 0.156 [0.110] | 0.173 / 0.173 [0.090] | 0.110 / 0.110 [0.076] | 0.100 / 0.100 [0.182] | **no** | 1.000 | 0 / 0 | 2.8 | 400 |
| (v) delays 0–8 and time constants up to 64 | 93683 | 0.844 / 0.549 / 0.222 / 0.024 / — | +0.112 | 0.159 / 0.159 [0.110] | 0.200 / 0.200 [0.096] | 0.157 / 0.156 [0.074] | 0.122 / 0.123 [0.173] | **no** | 1.000 | 0 / 0 | 2.9 | 424 |
| (v) delays 0–8 and time constants up to 64 | 116089 | 0.932 / 0.598 / 0.201 / 0.032 / — | +0.091 | 0.163 / 0.164 [0.110] | 0.209 / 0.209 [0.100] | 0.184 / 0.184 [0.075] | 0.111 / 0.111 [0.180] | **no** | 1.000 | 0 / 0 | 2.7 | 392 |
| (iii) time constants up to 64, 65,536 bytes (last window) | 73129 | 0.958 / 0.603 / 0.189 / 0.071 / — | +0.079 | 0.153 / 0.153 [0.110] | 0.150 / 0.150 [0.102] | 0.095 / 0.095 [0.073] | 0.068 / 0.068 [0.181] | **no** | 1.000 | 0 / 0 | 3.3 | 1152 |
| (iii) time constants up to 64, 65,536 bytes (last window) | 93683 | 0.814 / 0.504 / 0.208 / 0.044 / — | +0.098 | 0.168 / 0.168 [0.110] | 0.191 / 0.191 [0.100] | 0.152 / 0.152 [0.081] | 0.112 / 0.113 [0.186] | **no** | 1.000 | 0 / 0 | 3.0 | 1102 |
| (iii) time constants up to 64, 65,536 bytes (last window) | 116089 | 0.941 / 0.612 / 0.295 / 0.117 / 0.047 | +0.185 | 0.228 / 0.228 [0.110] | 0.236 / 0.236 [0.091] | 0.144 / 0.144 [0.074] | 0.094 / 0.094 [0.178] | **no** | 1.000 | 0 / 0 | 2.8 | 1038 |

Near entries are (a) / (b), with the level in brackets.

**The answer to the question** (which memory gives lags 0 to 3 near the output lines on three seeds, with the most margin for the present byte):

- **Lag 3 near the output lines is reached by no memory on any seed.** The closest is (iv): 0.157 / 0.132 / 0.124 by (a), against levels of 0.182 / 0.173 / 0.180. Its level is high, because the stream makes the byte three back partly predictable from the present byte (c0(3) about 0.16).
- **Lags 0 to 2 near the output lines on all three seeds**: (iii), (iv) and (v). Their smallest margins for the present byte over the three seeds are +0.075 (iii), +0.042 (iv) and +0.061 (v), so (iii) has the most margin.
- **(ii)** misses lag 2 near the output lines on seed 116089 (0.027). **(i)** carries only lag 0 there.
- **Near largest output** is 1.000 and **divergence** is 0 after 64 and 512 bytes in all 15 runs.

## 3. (iii) time constants up to 64, 65,536 bytes, four windows

| Seed | Window | S0 (a) at d = 1 / 2 / 4 | margin | near (a), lags 0 / 1 / 2 / 3 | levels | connections a cell |
|---|---|---|---|---|---|---|
| 73129 | 12288–16383 | 0.965 / 0.596 / 0.188 | +0.078 | 0.163 / 0.152 / 0.103 / 0.076 | 0.110 / 0.090 / 0.076 / 0.182 | 3.3 |
| 73129 | 28672–32767 | 0.954 / 0.606 / 0.200 | +0.090 | 0.169 / 0.149 / 0.093 / 0.068 | 0.110 / 0.106 / 0.080 / 0.176 | 3.3 |
| 73129 | 45056–49151 | 0.961 / 0.611 / 0.193 | +0.083 | 0.166 / 0.159 / 0.104 / 0.068 | 0.110 / 0.098 / 0.077 / 0.168 | 3.3 |
| 73129 | 61440–65535 | 0.958 / 0.603 / 0.189 | +0.079 | 0.153 / 0.150 / 0.095 / 0.068 | 0.110 / 0.102 / 0.073 / 0.181 | 3.3 |
| 73129 | divergence after 64 / 512 bytes: 0 / 0 | | | | | |
| 93683 | 12288–16383 | 0.818 / 0.527 / 0.185 | +0.075 | 0.169 / 0.182 / 0.124 / 0.108 | 0.110 / 0.096 / 0.074 / 0.173 | 3.0 |
| 93683 | 28672–32767 | 0.815 / 0.542 / 0.218 | +0.108 | 0.184 / 0.190 / 0.161 / 0.117 | 0.110 / 0.097 / 0.073 / 0.173 | 3.0 |
| 93683 | 45056–49151 | 0.822 / 0.511 / 0.215 | +0.105 | 0.178 / 0.184 / 0.144 / 0.111 | 0.110 / 0.103 / 0.077 / 0.166 | 3.0 |
| 93683 | 61440–65535 | 0.814 / 0.504 / 0.208 | +0.098 | 0.168 / 0.191 / 0.152 / 0.112 | 0.110 / 0.100 / 0.081 / 0.186 | 3.0 |
| 93683 | divergence after 64 / 512 bytes: 0 / 0 | | | | | |
| 116089 | 12288–16383 | 0.933 / 0.653 / 0.270 | +0.159 | 0.215 / 0.225 / 0.133 / 0.085 | 0.110 / 0.100 / 0.075 / 0.180 | 2.9 |
| 116089 | 28672–32767 | 0.939 / 0.635 / 0.276 | +0.166 | 0.231 / 0.239 / 0.142 / 0.095 | 0.110 / 0.101 / 0.077 / 0.170 | 2.8 |
| 116089 | 45056–49151 | 0.937 / 0.623 / 0.265 | +0.155 | 0.223 / 0.247 / 0.141 / 0.091 | 0.110 / 0.093 / 0.082 / 0.170 | 2.8 |
| 116089 | 61440–65535 | 0.941 / 0.612 / 0.295 | +0.185 | 0.228 / 0.236 / 0.144 / 0.094 | 0.110 / 0.091 / 0.074 / 0.178 | 2.8 |
| 116089 | divergence after 64 / 512 bytes: 0 / 0 | | | | | |

- Lags 0, 1 and 2 near the output lines are above their levels by (a) in 12 of 12 windows.
- Lag 3 is above its level in 0 of 12 windows.
- The margin for the present byte is +0.075 to +0.185.
- The divergence is 0 on all three seeds.
- The readings show no trend over 65,536 bytes.

## Cost

18 runs took 2.13 CPU-hours, plus the two runs of `eqcheck.py`; at most three processes of one thread each.
