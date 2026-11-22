# Four streams in one life: report (22 November 2026)

Plan: `LIVES4_PLAN.md` (committed before any run, `7f214e7f`). Script: `t20_lives4.py`. Raw results: `lives4_runs/*.json` and logs.
Source `4af67ad2` + this branch (`claude/newnet-lives4-sei232`). Development level; no gate, seal or outside review is claimed. The learner
is unchanged (`net10.py`, only `n_values=256`). K8 seeds 1, 2 and 3; N1 seed 1 only (N1 seeds 2-3 NOT_RUN).

Life: 8,000-byte blocks en (tinyshakespeare), zh (Hong Lou Meng, UTF-8), code (the repository's tracked `.py` files outside `NEWNET/`,
sorted path order), num (`a+b=c` lines), 3 rounds, 96,000 bytes, nothing reset, no task signal. Offsets (life reads / held-out test of
5,001 bytes = 5,000 predictions): en 0- / 600,000; zh 100,000- / 1,500,000; code 0- / 16,000,000; num 0- / 100,000 (details in the
plan). Every test is learning off, on a copy of the network, so the tests do not change the life. Control: each stream alone for the same
24,000 bytes in 3 blocks. Cost: K8 life about 1,360 s, N1 1,025 s; controls 108-169 s; 20 runs, 4 at once, about 35 min wall.

## Main tables: scores after each block (bold: the stream just read)

### K8 seed 1: scores after each block (learning off, 5,000 held-out predictions)

| # | block | en | zh | code | num |
|---|---|---|---|---|---|
| 1 | r1 en | **0.337** | 0.003 | 0.135 | 0.000 |
| 2 | r1 zh | 0.336 | **0.265** | 0.133 | 0.005 |
| 3 | r1 code | 0.272 | 0.254 | **0.391** | 0.004 |
| 4 | r1 num | 0.264 | 0.224 | 0.392 | **0.300** |
| 5 | r2 en | **0.354** | 0.237 | 0.362 | 0.264 |
| 6 | r2 zh | 0.357 | **0.333** | 0.368 | 0.237 |
| 7 | r2 code | 0.292 | 0.332 | **0.413** | 0.241 |
| 8 | r2 num | 0.296 | 0.324 | 0.412 | **0.322** |
| 9 | r3 en | **0.356** | 0.328 | 0.335 | 0.293 |
| 10 | r3 zh | 0.353 | **0.362** | 0.354 | 0.299 |
| 11 | r3 code | 0.267 | 0.365 | **0.402** | 0.261 |
| 12 | r3 num | 0.297 | 0.362 | 0.395 | **0.330** |

### K8 seed 2: scores after each block (learning off, 5,000 held-out predictions)

| # | block | en | zh | code | num |
|---|---|---|---|---|---|
| 1 | r1 en | **0.339** | 0.001 | 0.139 | 0.004 |
| 2 | r1 zh | 0.323 | **0.264** | 0.130 | 0.001 |
| 3 | r1 code | 0.247 | 0.266 | **0.395** | 0.001 |
| 4 | r1 num | 0.253 | 0.232 | 0.387 | **0.296** |
| 5 | r2 en | **0.355** | 0.229 | 0.244 | 0.237 |
| 6 | r2 zh | 0.343 | **0.331** | 0.287 | 0.248 |
| 7 | r2 code | 0.277 | 0.332 | **0.421** | 0.242 |
| 8 | r2 num | 0.289 | 0.321 | 0.366 | **0.323** |
| 9 | r3 en | **0.365** | 0.328 | 0.308 | 0.282 |
| 10 | r3 zh | 0.357 | **0.356** | 0.343 | 0.291 |
| 11 | r3 code | 0.291 | 0.358 | **0.368** | 0.272 |
| 12 | r3 num | 0.319 | 0.356 | 0.377 | **0.329** |

### K8 seed 3: scores after each block (learning off, 5,000 held-out predictions)

| # | block | en | zh | code | num |
|---|---|---|---|---|---|
| 1 | r1 en | **0.328** | 0.000 | 0.132 | 0.000 |
| 2 | r1 zh | 0.328 | **0.264** | 0.121 | 0.000 |
| 3 | r1 code | 0.246 | 0.255 | **0.376** | 0.004 |
| 4 | r1 num | 0.246 | 0.225 | 0.380 | **0.284** |
| 5 | r2 en | **0.333** | 0.217 | 0.361 | 0.232 |
| 6 | r2 zh | 0.326 | **0.333** | 0.355 | 0.248 |
| 7 | r2 code | 0.268 | 0.327 | **0.411** | 0.258 |
| 8 | r2 num | 0.272 | 0.322 | 0.397 | **0.319** |
| 9 | r3 en | **0.343** | 0.322 | 0.377 | 0.291 |
| 10 | r3 zh | 0.339 | **0.363** | 0.382 | 0.278 |
| 11 | r3 code | 0.260 | 0.357 | **0.375** | 0.270 |
| 12 | r3 num | 0.277 | 0.360 | 0.390 | **0.329** |

### K8: fall of each stream from the end of its block to just before its next block (seeds (1, 2, 3))

| stream | round 1 fall | round 2 fall | round 3 fall (to end of life) |
|---|---|---|---|
| en | +0.081 (24%) [+0.073, +0.087, +0.082] | +0.062 (18%) [+0.058, +0.066, +0.061] | +0.057 (16%) [+0.059, +0.046, +0.066] |
| zh | +0.037 (14%) [+0.028, +0.035, +0.047] | +0.006 (2%) [+0.005, +0.003, +0.011] | +0.001 (0%) [+0.000, +0.000, +0.003] |
| code | +0.050 (13%) [+0.023, +0.108, +0.021] | +0.055 (13%) [+0.059, +0.079, +0.028] | -0.006 (-2%) [+0.007, -0.009, -0.015] |
| num | +0.046 (16%) [+0.059, +0.054, +0.025] | +0.054 (17%) [+0.061, +0.051, +0.049] | - (last block) |

### K8: level at the end of each round (after block 4, 8, 12) against the control after the same own bytes

| stream | round 1 life / control | round 2 | round 3 |
|---|---|---|---|
| en | 0.254 / 0.335 (76%) | 0.286 / 0.360 (79%) | 0.298 / 0.382 (78%) |
| zh | 0.227 / 0.272 (84%) | 0.322 / 0.336 (96%) | 0.359 / 0.373 (96%) |
| code | 0.387 / 0.392 (99%) | 0.392 / 0.405 (97%) | 0.387 / 0.421 (92%) |
| num | 0.293 / 0.284 (103%) | 0.322 / 0.326 (99%) | 0.329 / 0.330 (100%) |

### K8: just after its own block against the control (same own bytes)

| stream | round 1 | round 2 | round 3 |
|---|---|---|---|
| en | 0.335 / 0.335 (100%) | 0.347 / 0.360 (96%) | 0.354 / 0.382 (93%) |
| zh | 0.264 / 0.272 (97%) | 0.332 / 0.336 (99%) | 0.360 / 0.373 (97%) |
| code | 0.387 / 0.392 (99%) | 0.415 / 0.405 (102%) | 0.382 / 0.421 (91%) |
| num | 0.293 / 0.284 (103%) | 0.322 / 0.326 (99%) | 0.329 / 0.330 (100%) |

seconds: lives [1353, 1365, 1364], controls [[162, 168, 157, 155], [159, 166, 162, 152], [163, 169, 162, 157]]

### N1 seed 1: scores after each block (learning off, 5,000 held-out predictions)

| # | block | en | zh | code | num |
|---|---|---|---|---|---|
| 1 | r1 en | **0.340** | 0.003 | 0.133 | 0.000 |
| 2 | r1 zh | 0.331 | **0.273** | 0.120 | 0.002 |
| 3 | r1 code | 0.271 | 0.259 | **0.399** | 0.004 |
| 4 | r1 num | 0.270 | 0.221 | 0.389 | **0.321** |
| 5 | r2 en | **0.364** | 0.226 | 0.377 | 0.277 |
| 6 | r2 zh | 0.348 | **0.327** | 0.378 | 0.259 |
| 7 | r2 code | 0.292 | 0.321 | **0.419** | 0.261 |
| 8 | r2 num | 0.299 | 0.309 | 0.410 | **0.325** |
| 9 | r3 en | **0.369** | 0.308 | 0.408 | 0.311 |
| 10 | r3 zh | 0.360 | **0.370** | 0.393 | 0.314 |
| 11 | r3 code | 0.291 | 0.363 | **0.419** | 0.287 |
| 12 | r3 num | 0.305 | 0.362 | 0.413 | **0.334** |

### N1: fall of each stream from the end of its block to just before its next block (seeds (1,))

| stream | round 1 fall | round 2 fall | round 3 fall (to end of life) |
|---|---|---|---|
| en | +0.070 (21%) | +0.066 (18%) | +0.064 (17%) |
| zh | +0.046 (17%) | +0.019 (6%) | +0.008 (2%) |
| code | +0.021 (5%) | +0.027 (6%) | +0.006 (1%) |
| num | +0.060 (19%) | +0.038 (12%) | - (last block) |

### N1: level at the end of each round (after block 4, 8, 12) against the control after the same own bytes

| stream | round 1 life / control | round 2 | round 3 |
|---|---|---|---|
| en | 0.270 / 0.340 (79%) | 0.299 / 0.376 (80%) | 0.305 / 0.389 (78%) |
| zh | 0.221 / 0.269 (82%) | 0.309 / 0.330 (94%) | 0.362 / 0.369 (98%) |
| code | 0.389 / 0.401 (97%) | 0.410 / 0.429 (96%) | 0.413 / 0.426 (97%) |
| num | 0.321 / 0.320 (101%) | 0.325 / 0.333 (98%) | 0.334 / 0.336 (99%) |

### N1: just after its own block against the control (same own bytes)

| stream | round 1 | round 2 | round 3 |
|---|---|---|---|
| en | 0.340 / 0.340 (100%) | 0.364 / 0.376 (97%) | 0.369 / 0.389 (95%) |
| zh | 0.273 / 0.269 (101%) | 0.327 / 0.330 (99%) | 0.370 / 0.369 (100%) |
| code | 0.399 / 0.401 (100%) | 0.419 / 0.429 (98%) | 0.419 / 0.426 (98%) |
| num | 0.321 / 0.320 (101%) | 0.325 / 0.333 (98%) | 0.334 / 0.336 (99%) |

seconds: lives [1025], controls [[117, 121, 115, 108]]

## Where each fall happens (change of a stream's score over each of the three following blocks; K8 mean of 3 seeds, N1 seed 1)

| stream, round | K8 | N1 |
|---|---|---|
| en r1 | zh -0.006, code **-0.075**, num -0.001 | zh -0.010, code **-0.060**, num -0.001 |
| en r2 | zh -0.005, code **-0.063**, num +0.007 | zh -0.016, code **-0.056**, num +0.006 |
| en r3 | zh -0.005, code **-0.077**, num +0.025 | zh -0.008, code **-0.069**, num +0.014 |
| zh r1 | code -0.006, num **-0.031**, en +0.000 | code -0.014, num **-0.039**, en +0.006 |
| zh r2 | code -0.002, num -0.008, en +0.004 | code -0.005, num -0.013, en -0.001 |
| zh r3 | code -0.000, num -0.001 | code -0.006, num -0.002 |
| code r1 | num -0.001, en **-0.064**, zh +0.015 | num -0.010, en -0.012, zh +0.001 |
| code r2 | num -0.023, en **-0.052**, zh +0.020 | num -0.009, en -0.002, zh -0.015 |
| code r3 | num +0.006 | num -0.006 |
| num r1 | en **-0.049**, zh +0.000, code +0.003 | en **-0.044**, zh -0.018, code +0.002 |
| num r2 | en -0.033, zh +0.001, code -0.022 | en -0.014, zh +0.003, code -0.027 |

## The plan's predictions against the measurements

| prediction | measured | verdict |
|---|---|---|
| 1. zh falls least, K8 round-1 fall <= 0.015 | 0.037 (14%) in round 1, mostly during the num block; then 0.006 and 0.001 | missed in round 1; holds from round 2 |
| 2. en, code, num K8 round-1 falls 0.02-0.06; N1 about twice K8 | en 0.081, code 0.050, num 0.046; N1 0.070, 0.021, 0.060 (zh 0.046) | en outside the range; N1 not twice K8 |
| 3. K8 round-2 fall smaller than round-1 for >= 3 of 4 streams | smaller for en (0.081 -> 0.062) and zh (0.037 -> 0.006); not for code (0.050 -> 0.055) or num (0.046 -> 0.054) | **refuted** (falsifier a: 2 streams did not shrink) |
| 4. K8 at the end of each round >= 0.90 of control | en 76% / 79% / 78%, zh 84% in round 1 | **refuted** (falsifier b: en below 0.80 of control) |
| K8 keeps better than N1 | N1's fall larger than K8's in 4 of 8 (stream, round) cases; N1 seed 1 only | **refuted for this life** (falsifier c), with one N1 seed |

The hypothesis H is refuted as stated: losses do not shrink over rounds for code and numbers, English stays at about 78% of its control
at the end of every round, and K8 does not keep better than N1 here.

## Plain readings (only what was measured)

1. **The losses come from streams that share bytes.** English loses almost all its fall while code is read (-0.06 to -0.08 each
   round, both structures); code loses mostly while English is read (K8); numbers lose while English (and in round 2 code) is read;
   Chinese, sharing nothing but a few ASCII bytes, barely touches English or code (-0.005; it even helps code by +0.015 / +0.020 for K8).
2. **Chinese is the exception that settles:** after round 1 it hardly falls (K8 0.006, then 0.001; N1 0.019, 0.008) and ends at 96%
   (K8) / 98% (N1) of its control. Its one round-1 fall came while numbers were read (-0.031 / -0.039), not code.
3. **English does not settle:** its fall is about the same every round (K8 0.081, 0.062, 0.057; N1 0.070, 0.066, 0.064); at the end of
   each round it is 76-79% of its control (both structures). Each English block brings it back, but to 93-96% of control, falling over
   rounds (K8 100%, 96%, 93%): the alternation also slows English's own learning.
4. **Code and numbers end near their control** (K8 code 92-99%, num 99-103%; N1 code 96-97%, num 98-101%), because each is measured
   right after or soon after its block; their falls between blocks (13-17% for K8) do not shrink from round 1 to round 2.
5. **K8 against N1:** in this life N1 is not worse. N1 falls less on code (0.021 / 0.027 against 0.050 / 0.055) and about the same on
   English; K8 falls less on Chinese (both rounds) and on numbers in round 1; N1 less on numbers in round 2. The English/Chinese difference between them (`LIFE_EN_ZH.md`) does not
   carry over to this mixture. N1 has one seed; K8's seeds vary (code round-1 fall 0.021-0.108), so the comparison is not settled.
6. **Seeds:** the K8 pattern (en falls during code, zh settles, code/num do not shrink) holds on all three seeds; single values vary by
   up to 0.08 (code).

## Limits

One block length (8,000), one order, 3 rounds, 5,000-byte tests (a share of 0.01 is 50 predictions); N1 one seed; tests on copies (the
earlier lives ran their tests on the network itself); the code stream is the repository's own Python (its test region lies 16 MB into
the sorted files, different modules than those read).
