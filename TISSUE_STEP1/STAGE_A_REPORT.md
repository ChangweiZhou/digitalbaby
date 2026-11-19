# E63, stage A: the tissue reads the stream (seed 73129, 16,384 bytes)

Brief: `TASK_STEP_1_THE_TISSUE_READS_THE_STREAM_20261118.md`, revision 3, with its changes of 02:16 and 02:36 UTC on 19 November 2026, on the reviewer's branch `claude/vibrant-cerf-sje726` (folder `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/`, "the brief"). Tissue: `THE_TISSUE_DEFINED_REVISION_5_20261119.md` on the same branch ("the definition"). Released by the coordinator's decision 50; the budget of stage A raised to 5 CPU-hours by the owner's word relayed in this session. A fifth tissue added by decision 62 was started and then stopped at about byte 13,312 when decision 71 (default branch at 3e17cd1f, by Mr. Xu's order) made decision 62 void; its files are left in `runs/A_first_withdrawbyuse_73129/` as they stood at the stop and are not read in this report. The work started from commit f973c1b5 of the default branch (`codex/minifly-consolidated-20261023`). Branch: `claude/e63-tissue-step1`. Stage B was not started.

Terms used below. A *tissue* is one life of the program `tissue_step1.py` from birth to the end of its stream. The *distance* d of a cell is the number of connections on the shortest path from an input line to it (section 1 of the brief; the mode over a stretch). A cell is *silent* in a stretch or window when its recorded output exceeds 0.01 at fewer than 1/128 of the positions, and *constant* when it is not silent and the standard deviation of its recorded output is below 0.01 (section 7.2). S(d, L, window) is the balanced accuracy with which the classifier of section 7.3 tells the byte L positions back from the recorded outputs of at most 128 cells at distance d. C0(L) is the same classifier fitted to a one-hot code of the present byte alone, and C<L(L) to a one-hot code of the bytes at lags 0 to L − 1 (section 7.5). Chance is 1/46 ≈ 0.0217 (the stream uses 46 byte values). Window A1 is bytes 2,048 to 6,143 and window A2 bytes 12,288 to 16,383.

## 1. Outcome

**Outcome 3 of section 11.3 ("anything else"), with the level 0.1103.** Neither outcome 1 nor outcome 2 holds, so the result is reported as found and the coordinator decides.

- In the first-form tissue (tissue 1), the cells at d = 2 and d = 4 do not stay informative in A2. None of the 264 cells at d = 2 is neither silent nor constant (212 are constant and 52 silent), and all 236 cells at d = 4 are silent.
- In the coincidence tissue (tissue 2), the cells at d = 2 and d = 4 are active. Their shares neither silent nor constant are 0.996 and 1.000. But their scores for the present byte are S(2, 0, A2) = 0.0375 and S(4, 0, A2) = 0.0264. Both are below the level of 0.1103 and near chance (0.0217).
- Outcome 1 fails because neither plastic tissue has both distances informative.
- Outcome 2 fails because it requires both tissues to have a share below 1/2 at d = 2 or d = 4. The coincidence tissue's shares are at least 1/2.

In words: the first-form tissue keeps the present byte at one connection from the lines (S(1, 0, A2) = 0.684) and at two connections in cells whose outputs vary by less than 0.01 (S(2, 0, A2) = 0.347, all 264 cells constant or silent). Beyond that it is silent. The coincidence tissue keeps its deeper cells active, but their activity does not tell the present byte better than chance plus 0.016. The fixed tissue tells the present byte at d = 1 (0.978) and at d = 2 (0.826, from cells that are all constant or silent by the level 0.01), and its cells at d = 4 are silent. No tissue tells the bytes at lags 1 and 2 better than C0(L) + 0.02 at any distance (section 9 below).

**Tissue 4 against finding 1.** After operation 7 of step 129, 61 of the 15,872 starting connections remain. This is fewer than 300, so the check's derivation is borne out (X0 holds). The counts by kind are in section 7.

**Decision 51 (a), in its words.** In the first-form tissue, "every cell reads its whole neighbourhood". At byte 16,383 the mean number of connections of a cell is 91.8, and 1,779 of the 1,984 cells that are not feedback cells have an exhausted neighbourhood. It already held at byte 2,047 (91.6 a cell, 1,760 neighbourhoods exhausted). In the coincidence tissue the mean is 25.9, with 152 neighbourhoods exhausted.


## 2. What was run

| No. | Tissue (section 11.1) | Command (in `TISSUE_STEP1/`) | Bytes | CPU s | Wall s | Resumed |
|---|---|---|---|---|---|---|
| 1 | first form, E10 | `tissue_step1.py run first 73129 --stage A --resume --beside 2 --out runs/A_first_73129` | 16384 | 5948 | 5992 | no |
| 2 | coincidence, E10 | `tissue_step1.py run coincidence 73129 --stage A --resume --beside 2 --out runs/A_coincidence_73129` | 16384 | 1346 | 1360 | no |
| 3 | fixed | `tissue_step1.py run fixed 73129 --stage A --resume --beside 2 --out runs/A_fixed_73129` | 16384 | 72 | 72 | no |
| 4 | first form, formula 11 as written (512 bytes) | `tissue_step1.py run first 73129 --bytes 512 --formula11-as-written --stage A --resume --beside 2 --out runs/A_first_f11written_73129` | 512 | 3 | 3 | no |

All four tissues were born from seed 73129 by the same `birth` (check 2) and read the same stream (SHA-256 of the 262,144 bytes 7772341c…8e8775, check 3; the first 16,384 or 512 of them). At most three processes ran at a time, each with one thread (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` set to 1). Tissues 1 and 2 were started at 02:42 UTC by a shell command and tissues 3 and 4 by `stage_a_queue.sh`; no variable beginning `E11_`, `E16_` or `EXP45_` exists in this machine's environment. The field `git_head` of a receipt is the branch's commit when the receipt was written, at the end of the run; the program's SHA-256 is the record of what ran. Every run was started with `--resume`; none was interrupted, so none was resumed (field `resumes` of each receipt is empty) and none was started again. The program's SHA-256 for every stage A run is 49316c23…a1fc1 (commit 0e333253); it differs from the program of checks 1 to 7 and 9 (ea171fff…f56008, commit 0f127aa7) only by the option `--withdraw-by-use` of the void fifth tissue, which none of the four tissues takes. The reviewer's message 18 proposed two changes of speed that change no figure; they were not applied, and every tissue of stage A ran with the same program. Check 8 (save and resume) ran on 49316c23.

## 3. Checks 1 to 10

| Check | Result | Receipt |
|---|---|---|
| 1. Worked example | Passed on the reviewer's ground (message 17): parts (a) to (d) agree to four decimals; part (e), the whole tissue over its first 1,024 steps, agrees with the definition's table up to step 129 and differs after it (cell 333 at step 160: 0.0172 against 0.0322; 13,862 connections made by step 1,024 against 13,875; smallest starting strength between cells 0.0419 against 0.0455). The definition does not say from which generator or in which order of candidates the first new connections (step 128) are drawn; the definition's table after step 129 is withdrawn and this program's figures are the reference. The computed verdict (`pass_as_computed: false`) is kept beside the ground | `qual/check1.json` |
| 2. Birth | Passed | `qual/check2.json` |
| 3. Stream | Passed (digests of all three seeds) | `qual/check3.json` |
| 4, 5, 7 | Passed | `qual/check4.json`, `check5.json`, `check7.json` |
| 6. Audit of giving up and of feedback cells | Passed: 177,443 connections given up, none from a line or a feedback cell, none at or above 1/128 | `qual/check6.json` |
| 8. Save and resume | Passed: a run killed after byte 1,024 and resumed ends bit-equal to the uninterrupted run, with equal records | `qual/check8.json` |
| 9. The analysis | Passed on the reviewer's ground (message 16): (a) C0 equal to section 19, part E; (b) positive control 1.0 at every lag; (c) negative control within 0.0217 ± 0.01; (d) C<L recorded; (e) the pair code scores 0.0903 in A2, not the brief's 0.0946. The code of 0s and 1s produces exact ties between class centroids; this program breaks them toward the smaller byte value as section 7.3 says, with distances within a relative 10⁻⁹ of the smallest counted as ties (function `nearest` of `step1_read.py`). The first attempt, which broke ties by rounding noise, is kept in `qual/check9_first_attempt_ties_by_rounding.json`. The level of "stays informative" is accordingly 0.1103 | `qual/check9.json` |
| 10. Cost | Timings finite; forecast for stage A 2.93 CPU-hours for the four tissues, above the brief's 2.5; the worker stopped and asked; the owner allowed up to 5 CPU-hours | `qual/check10.json` |

## 4. Numbers and choices

The numbers of the first form, the numbers of the reading and the forms of every choice are those of section 5.7 of the brief, used without change, except the level of section 11.3, which is 0.1103 by the change of 02:16 UTC. The decisions of section 5.8 are applied as written (feedback cells present from birth with silent lines; no form (b) cells; feedback cells drawn by `rb.sample` after coordinates and lines; "within 1/8" as squared distance at most 1/64; starting connections made at step 0; all giving up before the makings, makings in increasing cell index; line connections never given up under E10; emissions recorded, not compared; outputs recorded at the reading step, values below 0.01 stored as 0).

Choices of the worker where the brief leaves one:
1. The standard error of S is the sample standard deviation (n − 1) of the four folds' balanced accuracies divided by 2.
2. Ties of the classifier: distances within a relative 10⁻⁹ of the smallest count as ties (check 9 above).
3. Recorded outputs are stored as float16; the analysis reads them as float64.

## 5. Errata (section 5.2 of the brief), as applied

| No. | Applied |
|---|---|
| E1 | The coincidence tissue is run from the start beside the first-form tissue; every measure is read for d = 1, 2, 4 and each lag separately |
| E2 | No feedback at step 1; the receipts give the output-end cells taking a yes-cell and a no-cell as a source at every count |
| E3 | **Not applied**: in the coincidence tissue the output-end cells keep formula 7, so the kinds of sentence 87 are kept in all tissues |
| E4 | Applied: feedback cells have the feedback line only, θ = 0, no sources, no makings, no strength change (check 6 audits this) |
| E5 | **Not applied** (the reviewer's decision on finding 9 of the check): feedback cells draw no input lines |
| E6, E7 | Wordings; no effect on the program |
| E8 | 27 encoder cells at one connection from a feedback cell for seed 73129 (section 19, B.1) |
| E9 | Feedback cells present, with silent lines |
| E10 | Applied in tissues 1, 2, 3 and 5; formula 11 as written only in tissue 4 |

## 6. The coincidence rule and the cell's prediction

The coincidence rule (section 5.5) does not use the cell's prediction. In the coincidence tissue the inner cells therefore do not do what sentence 36 (Mr. Xu's words, section 8 of `WHAT_A_NEURON_HAS_20261118.md`: "predicting the information newly arriving from other cells") describes for every cell; a favourable result for that arm would have to be reconciled with section 8 of Mr. Xu's words before it is carried forward. Stage A gives no favourable result for that arm (section 1).

## 7. Tables of section 11.2

The tables below are printed by `python3 step1_tables.py` from the receipts `runs/A_*/receipt.json`. The receipts are written by `tissue_step1.py`, and their `measures` are written by `python3 step1_read.py runs/<tissue>`. The receipts hold every count (every 128th byte); the tables show every 2,048th byte. Tissue numbers are those of section 11.1.

### Outcome (section 11.3, level 0.1103)

Outcome 3: Anything else.

| Tissue | d | share neither silent nor constant, A2 | S(d, 0, A2) | share ≥ 1/2 | S ≥ level |
|---|---|---|---|---|---|
| 1. first form, E10 | 2 | 0.0000 | 0.3471 | no | yes |
| 1. first form, E10 | 4 | 0.0000 | 0.0255 | no | no |
| 2. coincidence, E10 | 2 | 0.9960 | 0.0375 | yes | no |
| 2. coincidence, E10 | 4 | 1.0000 | 0.0264 | yes | no |
| 3. fixed | 2 | 0.0000 | 0.8261 | no | yes |
| 3. fixed | 4 | 0.0000 | 0.0217 | no | no |

### Tissue 4 (formula 11 as written, 512 bytes): connections after operation 7

| Step | connections | starting remaining | given up | made | input-line connections | from feedback cells |
|---|---|---|---|---|---|---|
| 127 | 15872 | 15872 | 0 | 0 | 754 | 423 |
| 129 | 2045 | 61 | 15811 | 1984 | 106 | 64 |
| 4127 | 1991 | 0 | 77369 | 63488 | 98 | 53 |
| 8223 | 1990 | 0 | 140858 | 126976 | 95 | 49 |
| 12319 | 1989 | 0 | 204347 | 190464 | 100 | 44 |
| 16383 | 1990 | 0 | 265850 | 251968 | 91 | 46 |

X0 (fewer than 300 of 15,872 starting connections remain after step 129): 61 remain, held.

By kind after step 129:

| Kind | n | mean strength |
|---|---|---|
| encoder <- input line of a byte value in use | 29 | 0.1250 |
| encoder <- input line of a value not in use | 77 | 0.1250 |
| encoder <- excitatory cell | 133 | 0.1250 |
| encoder <- inhibitory cell | 43 | 0.1250 |
| encoder <- feedback cell | 6 | 0.1250 |
| inner <- input line of a byte value in use | 0 | — |
| inner <- input line of a value not in use | 0 | — |
| inner <- excitatory cell | 1144 | 0.1250 |
| inner <- inhibitory cell | 311 | 0.1250 |
| inner <- feedback cell | 46 | 0.1250 |
| output-end <- input line of a byte value in use | 0 | — |
| output-end <- input line of a value not in use | 0 | — |
| output-end <- excitatory cell | 233 | 0.1250 |
| output-end <- inhibitory cell | 11 | 0.1250 |
| output-end <- feedback cell | 12 | 0.1250 |

### Connections (section 11.2, item 1; every 2,048 bytes shown, every 128 bytes in the receipts)

#### 1. first form, E10

| Step | byte | connections | mean a cell | starting remaining | given up | made | input-line conns | line values in use | from feedback cells (w min–max) | inh → inner | exc → inner mean w |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 127 | 3 | 15872 | 7.75 | 15872 | 0 | 0 | 754 | 44 | 423 (0.1250–0.1250) | 2518 | 0.1250 |
| 129 | — | 17856 | 8.72 | 15872 | 0 | 1984 | 838 | 44 | 486 (0.1250–0.1250) | 2828 | 0.1250 |
| 1024 | — | 31111 | 15.19 | 15872 | 0 | 15239 | 1522 | 46 | 871 (0.1250–0.1250) | 5001 | 0.1204 |
| 65535 | 2047 | 187656 | 91.63 | 6926 | 177443 | 349227 | 11847 | 46 | 5200 (0.1250–0.1250) | 30257 | 0.0739 |
| 131071 | 4095 | 188166 | 91.88 | 6839 | 296737 | 469031 | 11847 | 46 | 5200 (0.1250–0.1250) | 30257 | 0.0737 |
| 196607 | 6143 | 188275 | 91.93 | 6801 | 404143 | 576546 | 11847 | 46 | 5200 (0.1250–0.1250) | 30257 | 0.0734 |
| 262143 | 8191 | 188292 | 91.94 | 6778 | 513972 | 686392 | 11847 | 46 | 5200 (0.1250–0.1250) | 30257 | 0.0733 |
| 327679 | 10239 | 188398 | 91.99 | 6745 | 619161 | 791687 | 11847 | 46 | 5200 (0.1250–0.1250) | 30257 | 0.0732 |
| 393215 | 12287 | 188164 | 91.88 | 6733 | 724997 | 897289 | 11847 | 46 | 5200 (0.1250–0.1250) | 30257 | 0.0732 |
| 458751 | 14335 | 187725 | 91.66 | 6723 | 840463 | 1012316 | 11847 | 46 | 5200 (0.1250–0.1250) | 30257 | 0.0732 |
| 524287 | 16383 | 187990 | 91.79 | 6719 | 944453 | 1116571 | 11847 | 46 | 5200 (0.1250–0.1250) | 30257 | 0.0730 |

#### 2. coincidence, E10

| Step | byte | connections | mean a cell | starting remaining | given up | made | input-line conns | line values in use | from feedback cells (w min–max) | inh → inner | exc → inner mean w |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 127 | 3 | 15872 | 7.75 | 15872 | 0 | 0 | 754 | 44 | 423 (0.1250–0.1250) | 2518 | 0.1250 |
| 129 | — | 17856 | 8.72 | 15872 | 0 | 1984 | 838 | 44 | 486 (0.1250–0.1250) | 2828 | 0.1250 |
| 1024 | — | 25499 | 12.45 | 15775 | 157 | 9784 | 1423 | 46 | 709 (0.1250–0.1250) | 3885 | 0.8337 |
| 65535 | 2047 | 56309 | 27.49 | 2503 | 204107 | 244544 | 11847 | 46 | 2964 (0.1250–0.1250) | 841 | 0.5579 |
| 131071 | 4095 | 58813 | 28.72 | 2115 | 468847 | 511788 | 11847 | 46 | 4245 (0.1250–0.1250) | 769 | 0.5492 |
| 196607 | 6143 | 56855 | 27.76 | 1998 | 734069 | 775052 | 11847 | 46 | 4758 (0.1250–0.1250) | 812 | 0.5506 |
| 262143 | 8191 | 57345 | 28.00 | 1966 | 987016 | 1028489 | 11847 | 46 | 4959 (0.1250–0.1250) | 994 | 0.5863 |
| 327679 | 10239 | 54988 | 26.85 | 1935 | 1239206 | 1278322 | 11847 | 46 | 5071 (0.1250–0.1250) | 982 | 0.6065 |
| 393215 | 12287 | 54204 | 26.47 | 1910 | 1485590 | 1523922 | 11847 | 46 | 5112 (0.1250–0.1250) | 974 | 0.6180 |
| 458751 | 14335 | 53298 | 26.02 | 1871 | 1722845 | 1760271 | 11847 | 46 | 5132 (0.1250–0.1250) | 1051 | 0.6158 |
| 524287 | 16383 | 53023 | 25.89 | 1857 | 1959103 | 1996254 | 11847 | 46 | 5146 (0.1250–0.1250) | 989 | 0.6360 |

#### 3. fixed

| Step | byte | connections | mean a cell | starting remaining | given up | made | input-line conns | line values in use | from feedback cells (w min–max) | inh → inner | exc → inner mean w |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 127 | 3 | 15872 | 7.75 | 15872 | 0 | 0 | 754 | 44 | 423 (0.1250–0.1250) | 2518 | 0.1250 |
| 129 | — | 15872 | 7.75 | 15872 | 0 | 0 | 754 | 44 | 423 (0.1250–0.1250) | 2518 | 0.1250 |
| 1024 | — | 15872 | 7.75 | 15872 | 0 | 0 | 754 | 44 | 423 (0.1250–0.1250) | 2518 | 0.1250 |
| 65535 | 2047 | 15872 | 7.75 | 15872 | 0 | 0 | 754 | 44 | 423 (0.1250–0.1250) | 2518 | 0.1250 |
| 131071 | 4095 | 15872 | 7.75 | 15872 | 0 | 0 | 754 | 44 | 423 (0.1250–0.1250) | 2518 | 0.1250 |
| 196607 | 6143 | 15872 | 7.75 | 15872 | 0 | 0 | 754 | 44 | 423 (0.1250–0.1250) | 2518 | 0.1250 |
| 262143 | 8191 | 15872 | 7.75 | 15872 | 0 | 0 | 754 | 44 | 423 (0.1250–0.1250) | 2518 | 0.1250 |
| 327679 | 10239 | 15872 | 7.75 | 15872 | 0 | 0 | 754 | 44 | 423 (0.1250–0.1250) | 2518 | 0.1250 |
| 393215 | 12287 | 15872 | 7.75 | 15872 | 0 | 0 | 754 | 44 | 423 (0.1250–0.1250) | 2518 | 0.1250 |
| 458751 | 14335 | 15872 | 7.75 | 15872 | 0 | 0 | 754 | 44 | 423 (0.1250–0.1250) | 2518 | 0.1250 |
| 524287 | 16383 | 15872 | 7.75 | 15872 | 0 | 0 | 754 | 44 | 423 (0.1250–0.1250) | 2518 | 0.1250 |

### Activity per stretch of 1,024 bytes (section 11.2, item 2): share neither silent nor constant (mean output, sd)

#### 1. first form, E10

| Bytes | d = 1 | d = 2 | d = 4 |
|---|---|---|---|
| 0–1023 | 237: 1.000 (0.0253, 0.0438) | 263: 1.000 (0.0302, 0.0475) | 237: 1.000 (0.0347, 0.053) |
| 1024–2047 | 237: 0.996 (0.0114, 0.0265) | 264: 0.076 (0.00152, 0.00626) | 236: 0.047 (0.000618, 0.0046) |
| 2048–3071 | 237: 0.987 (0.01, 0.024) | 264: 0.030 (0.00126, 0.00551) | 236: 0.000 (9.54e-05, 0.00138) |
| 3072–4095 | 237: 0.992 (0.0094, 0.0232) | 264: 0.049 (0.00124, 0.00555) | 236: 0.000 (0.00013, 0.00167) |
| 4096–5119 | 237: 0.987 (0.00875, 0.0221) | 264: 0.026 (0.00118, 0.0053) | 236: 0.000 (6.26e-05, 0.00123) |
| 5120–6143 | 237: 0.975 (0.00798, 0.021) | 264: 0.030 (0.00103, 0.00509) | 236: 0.000 (4.07e-05, 0.000914) |
| 6144–7167 | 237: 0.979 (0.0078, 0.0209) | 264: 0.030 (0.00103, 0.00512) | 236: 0.000 (3.87e-05, 0.000885) |
| 7168–8191 | 237: 0.945 (0.00704, 0.0195) | 264: 0.004 (0.000726, 0.00406) | 236: 0.000 (5.56e-05, 0.00107) |
| 8192–9215 | 237: 0.958 (0.00715, 0.0198) | 264: 0.026 (0.000841, 0.00454) | 236: 0.000 (6.88e-05, 0.00107) |
| 9216–10239 | 237: 0.983 (0.00737, 0.0203) | 264: 0.030 (0.000892, 0.00476) | 236: 0.000 (0.000103, 0.00146) |
| 10240–11263 | 237: 0.949 (0.00677, 0.0185) | 264: 0.015 (0.000753, 0.00417) | 236: 0.000 (9.34e-05, 0.00153) |
| 11264–12287 | 237: 0.916 (0.00584, 0.0173) | 264: 0.004 (0.000609, 0.00371) | 236: 0.000 (3.27e-05, 0.000778) |
| 12288–13311 | 237: 0.924 (0.0061, 0.0177) | 264: 0.004 (0.000563, 0.00343) | 236: 0.000 (7.12e-05, 0.00116) |
| 13312–14335 | 237: 0.882 (0.00544, 0.0165) | 264: 0.000 (0.000404, 0.0028) | 236: 0.000 (2.49e-05, 0.00064) |
| 14336–15359 | 237: 0.785 (0.00503, 0.016) | 264: 0.000 (0.000352, 0.00272) | 236: 0.000 (4.31e-06, 0.000253) |
| 15360–16383 | 237: 0.848 (0.00526, 0.0162) | 264: 0.000 (0.000519, 0.00318) | 236: 0.000 (7.59e-07, 9.74e-05) |

#### 2. coincidence, E10

| Bytes | d = 1 | d = 2 | d = 4 |
|---|---|---|---|
| 0–1023 | 237: 1.000 (0.0221, 0.0451) | 198: 1.000 (0.119, 0.317) | 190: 1.000 (0.126, 0.328) |
| 1024–2047 | 237: 0.949 (0.00969, 0.0288) | 232: 0.996 (0.106, 0.297) | 193: 1.000 (0.109, 0.306) |
| 2048–3071 | 237: 0.857 (0.00677, 0.024) | 252: 0.992 (0.108, 0.3) | 190: 1.000 (0.11, 0.307) |
| 3072–4095 | 237: 0.768 (0.00585, 0.0237) | 256: 0.992 (0.107, 0.298) | 192: 1.000 (0.11, 0.305) |
| 4096–5119 | 237: 0.835 (0.0104, 0.0662) | 256: 0.996 (0.103, 0.291) | 200: 1.000 (0.109, 0.303) |
| 5120–6143 | 237: 0.848 (0.0152, 0.0901) | 256: 0.996 (0.111, 0.299) | 204: 1.000 (0.11, 0.305) |
| 6144–7167 | 237: 0.819 (0.0121, 0.0798) | 256: 0.996 (0.102, 0.289) | 194: 1.000 (0.111, 0.305) |
| 7168–8191 | 237: 0.810 (0.0119, 0.0792) | 257: 0.996 (0.102, 0.288) | 195: 1.000 (0.11, 0.304) |
| 8192–9215 | 237: 0.785 (0.0122, 0.0811) | 260: 1.000 (0.101, 0.286) | 186: 1.000 (0.111, 0.307) |
| 9216–10239 | 237: 0.743 (0.00966, 0.0693) | 258: 0.996 (0.104, 0.292) | 210: 1.000 (0.11, 0.306) |
| 10240–11263 | 237: 0.747 (0.00968, 0.0666) | 257: 0.996 (0.103, 0.289) | 199: 1.000 (0.109, 0.303) |
| 11264–12287 | 237: 0.793 (0.0138, 0.0897) | 259: 0.996 (0.102, 0.288) | 206: 1.000 (0.112, 0.307) |
| 12288–13311 | 237: 0.717 (0.0146, 0.095) | 254: 0.996 (0.105, 0.289) | 205: 1.000 (0.112, 0.306) |
| 13312–14335 | 237: 0.734 (0.0213, 0.12) | 249: 0.996 (0.104, 0.288) | 197: 1.000 (0.109, 0.301) |
| 14336–15359 | 237: 0.717 (0.0204, 0.118) | 250: 0.996 (0.104, 0.289) | 204: 1.000 (0.108, 0.3) |
| 15360–16383 | 237: 0.671 (0.0178, 0.108) | 251: 0.996 (0.102, 0.287) | 195: 1.000 (0.112, 0.306) |

#### 3. fixed

| Bytes | d = 1 | d = 2 | d = 4 |
|---|---|---|---|
| 0–1023 | 102: 1.000 (0.00441, 0.0221) | 215: 0.000 (0.000595, 0.00326) | 220: 0.000 (0, 0) |
| 1024–2047 | 102: 1.000 (0.00437, 0.022) | 215: 0.000 (0.000595, 0.00325) | 220: 0.000 (0, 0) |
| 2048–3071 | 102: 1.000 (0.00445, 0.0222) | 215: 0.000 (0.000588, 0.00323) | 220: 0.000 (0, 0) |
| 3072–4095 | 102: 1.000 (0.00443, 0.0221) | 215: 0.000 (0.0006, 0.00327) | 220: 0.000 (0, 0) |
| 4096–5119 | 102: 0.980 (0.00447, 0.0222) | 215: 0.000 (0.000602, 0.00328) | 220: 0.000 (0, 0) |
| 5120–6143 | 102: 1.000 (0.00443, 0.0221) | 215: 0.000 (0.000601, 0.00328) | 220: 0.000 (0, 0) |
| 6144–7167 | 102: 1.000 (0.00439, 0.022) | 215: 0.000 (0.000606, 0.00328) | 220: 0.000 (0, 0) |
| 7168–8191 | 102: 1.000 (0.00444, 0.0222) | 215: 0.000 (0.000624, 0.00334) | 220: 0.000 (0, 0) |
| 8192–9215 | 102: 1.000 (0.00432, 0.0219) | 215: 0.000 (0.00057, 0.00318) | 220: 0.000 (0, 0) |
| 9216–10239 | 102: 1.000 (0.00419, 0.0215) | 215: 0.000 (0.000559, 0.00316) | 220: 0.000 (0, 0) |
| 10240–11263 | 102: 0.990 (0.00423, 0.0216) | 215: 0.000 (0.000558, 0.00316) | 220: 0.000 (0, 0) |
| 11264–12287 | 102: 1.000 (0.00428, 0.0217) | 215: 0.000 (0.000574, 0.0032) | 220: 0.000 (0, 0) |
| 12288–13311 | 102: 1.000 (0.00439, 0.022) | 215: 0.000 (0.000613, 0.00331) | 220: 0.000 (0, 0) |
| 13312–14335 | 102: 1.000 (0.00452, 0.0223) | 215: 0.000 (0.00064, 0.00338) | 220: 0.000 (0, 0) |
| 14336–15359 | 102: 1.000 (0.00428, 0.0217) | 215: 0.000 (0.000583, 0.00322) | 220: 0.000 (0, 0) |
| 15360–16383 | 102: 0.990 (0.00443, 0.0221) | 215: 0.000 (0.000586, 0.00324) | 220: 0.000 (0, 0) |

### Scores S(d, L) (section 11.2, item 3), beside C0(L) and C<L(L)

#### 1. first form, E10, window A1 (bytes 2048–6143)

| | L = 0 | L = 1 | L = 2 | L = 4 | L = 8 |
|---|---|---|---|---|---|
| S(d = 1), 128 of 237 cells | 0.7552 ± 0.006 | 0.0529 ± 0.002 | 0.0306 ± 0.003 | 0.0446 ± 0.002 | 0.0263 ± 0.002 |
| S(d = 2), 128 of 264 cells | 0.5271 ± 0.033 | 0.0416 ± 0.002 | 0.0353 ± 0.002 | 0.0394 ± 0.003 | 0.0279 ± 0.002 |
| S(d = 4), 128 of 236 cells | 0.0386 ± 0.002 | 0.0238 ± 0.001 | 0.0208 ± 0.001 | 0.0247 ± 0.000 | 0.0250 ± 0.000 |
| C0(L) | 1.0000 | 0.0858 | 0.0497 | 0.0695 | 0.0335 |
| C<L(L) | — | 0.0858 | 0.0885 | 0.1831 | 0.1779 |

Shares neither silent nor constant in A1: d = 1: 0.9916 of 237, d = 2: 0.0417 of 264, d = 4: 0.0000 of 236

#### 1. first form, E10, window A2 (bytes 12288–16383)

| | L = 0 | L = 1 | L = 2 | L = 4 | L = 8 |
|---|---|---|---|---|---|
| S(d = 1), 128 of 237 cells | 0.6841 ± 0.011 | 0.0541 ± 0.004 | 0.0383 ± 0.004 | 0.0555 ± 0.001 | 0.0321 ± 0.002 |
| S(d = 2), 128 of 264 cells | 0.3471 ± 0.026 | 0.0439 ± 0.001 | 0.0319 ± 0.002 | 0.0419 ± 0.001 | 0.0289 ± 0.002 |
| S(d = 4), 128 of 236 cells | 0.0255 ± 0.002 | 0.0236 ± 0.000 | 0.0236 ± 0.000 | 0.0241 ± 0.001 | 0.0217 ± 0.000 |
| C0(L) | 1.0000 | 0.0703 | 0.0561 | 0.0729 | 0.0354 |
| C<L(L) | — | 0.0703 | 0.0870 | 0.1755 | 0.1726 |

Shares neither silent nor constant in A2: d = 1: 0.9536 of 237, d = 2: 0.0000 of 264, d = 4: 0.0000 of 236

#### 2. coincidence, E10, window A1 (bytes 2048–6143)

| | L = 0 | L = 1 | L = 2 | L = 4 | L = 8 |
|---|---|---|---|---|---|
| S(d = 1), 128 of 237 cells | 0.4098 ± 0.031 | 0.0394 ± 0.004 | 0.0289 ± 0.002 | 0.0318 ± 0.003 | 0.0215 ± 0.004 |
| S(d = 2), 128 of 257 cells | 0.0336 ± 0.005 | 0.0199 ± 0.002 | 0.0228 ± 0.002 | 0.0161 ± 0.001 | 0.0255 ± 0.002 |
| S(d = 4), 128 of 192 cells | 0.0233 ± 0.001 | 0.0228 ± 0.001 | 0.0173 ± 0.002 | 0.0219 ± 0.003 | 0.0267 ± 0.001 |
| C0(L) | 1.0000 | 0.0858 | 0.0497 | 0.0695 | 0.0335 |
| C<L(L) | — | 0.0858 | 0.0885 | 0.1831 | 0.1779 |

Shares neither silent nor constant in A1: d = 1: 0.8819 of 237, d = 2: 0.9961 of 257, d = 4: 1.0000 of 192

#### 2. coincidence, E10, window A2 (bytes 12288–16383)

| | L = 0 | L = 1 | L = 2 | L = 4 | L = 8 |
|---|---|---|---|---|---|
| S(d = 1), 128 of 237 cells | 0.2344 ± 0.002 | 0.0310 ± 0.004 | 0.0266 ± 0.004 | 0.0377 ± 0.005 | 0.0250 ± 0.002 |
| S(d = 2), 128 of 250 cells | 0.0375 ± 0.003 | 0.0263 ± 0.001 | 0.0167 ± 0.001 | 0.0252 ± 0.002 | 0.0210 ± 0.003 |
| S(d = 4), 128 of 202 cells | 0.0264 ± 0.002 | 0.0289 ± 0.002 | 0.0194 ± 0.002 | 0.0283 ± 0.001 | 0.0237 ± 0.002 |
| C0(L) | 1.0000 | 0.0703 | 0.0561 | 0.0729 | 0.0354 |
| C<L(L) | — | 0.0703 | 0.0870 | 0.1755 | 0.1726 |

Shares neither silent nor constant in A2: d = 1: 0.7300 of 237, d = 2: 0.9960 of 250, d = 4: 1.0000 of 202

#### 3. fixed, window A1 (bytes 2048–6143)

| | L = 0 | L = 1 | L = 2 | L = 4 | L = 8 |
|---|---|---|---|---|---|
| S(d = 1), 102 of 102 cells | 0.9783 ± 0.000 | 0.0738 ± 0.006 | 0.0481 ± 0.001 | 0.0656 ± 0.002 | 0.0322 ± 0.002 |
| S(d = 2), 128 of 215 cells | 0.8261 ± 0.000 | 0.0642 ± 0.005 | 0.0368 ± 0.002 | 0.0452 ± 0.001 | 0.0289 ± 0.004 |
| S(d = 4), 128 of 220 cells | 0.0217 ± 0.000 | 0.0217 ± 0.000 | 0.0217 ± 0.000 | 0.0217 ± 0.000 | 0.0217 ± 0.000 |
| C0(L) | 1.0000 | 0.0858 | 0.0497 | 0.0695 | 0.0335 |
| C<L(L) | — | 0.0858 | 0.0885 | 0.1831 | 0.1779 |

Shares neither silent nor constant in A1: d = 1: 1.0000 of 102, d = 2: 0.0000 of 215, d = 4: 0.0000 of 220

#### 3. fixed, window A2 (bytes 12288–16383)

| | L = 0 | L = 1 | L = 2 | L = 4 | L = 8 |
|---|---|---|---|---|---|
| S(d = 1), 102 of 102 cells | 0.9783 ± 0.000 | 0.0664 ± 0.003 | 0.0530 ± 0.001 | 0.0668 ± 0.004 | 0.0311 ± 0.003 |
| S(d = 2), 128 of 215 cells | 0.8261 ± 0.000 | 0.0556 ± 0.001 | 0.0419 ± 0.002 | 0.0489 ± 0.003 | 0.0327 ± 0.001 |
| S(d = 4), 128 of 220 cells | 0.0217 ± 0.000 | 0.0217 ± 0.000 | 0.0217 ± 0.000 | 0.0217 ± 0.000 | 0.0217 ± 0.000 |
| C0(L) | 1.0000 | 0.0703 | 0.0561 | 0.0729 | 0.0354 |
| C<L(L) | — | 0.0703 | 0.0870 | 0.1755 | 0.1726 |

Shares neither silent nor constant in A2: d = 1: 1.0000 of 102, d = 2: 0.0000 of 215, d = 4: 0.0000 of 220

### Cost and emissions per stretch (section 11.2, items 4 and 5)

| Bytes | 1. first form, E10: s/byte, conns a cell, emitted | 2. coincidence, E10: s/byte, conns a cell, emitted | 3. fixed: s/byte, conns a cell, emitted |
|---|---|---|---|
| 0–1023 | 0.3680, 90.39, 0.0010 | 0.0769, 26.55, 0.0137 | 0.0063, 7.75, 0.0000 |
| 1024–2047 | 0.3656, 91.63, 0.0029 | 0.0823, 27.49, 0.0137 | 0.0044, 7.75, 0.0000 |
| 2048–3071 | 0.3699, 91.84, 0.0029 | 0.0890, 28.10, 0.0273 | 0.0043, 7.75, 0.0000 |
| 3072–4095 | 0.3774, 91.88, 0.0078 | 0.0865, 28.72, 0.1094 | 0.0042, 7.75, 0.0000 |
| 4096–5119 | 0.3671, 91.92, 0.0020 | 0.0846, 27.62, 0.0947 | 0.0040, 7.75, 0.0000 |
| 5120–6143 | 0.3511, 91.93, 0.0029 | 0.0827, 27.76, 0.1650 | 0.0041, 7.75, 0.0000 |
| 6144–7167 | 0.3618, 92.03, 0.0098 | 0.0844, 27.61, 0.2051 | 0.0041, 7.75, 0.0000 |
| 7168–8191 | 0.3741, 91.94, 0.0068 | 0.0826, 28.00, 0.2080 | 0.0041, 7.75, 0.0000 |
| 8192–9215 | 0.3775, 91.70, 0.0059 | 0.0850, 26.89, 0.2041 | 0.0040, 7.75, 0.0000 |
| 9216–10239 | 0.3960, 91.99, 0.0068 | 0.0812, 26.85, 0.2070 | 0.0044, 7.75, 0.0000 |
| 10240–11263 | 0.3752, 91.97, 0.0020 | 0.0813, 27.04, 0.2812 | 0.0044, 7.75, 0.0000 |
| 11264–12287 | 0.3720, 91.88, 0.0039 | 0.0796, 26.47, 0.2607 | 0.0044, 7.75, 0.0000 |
| 12288–13311 | 0.3434, 91.69, 0.0088 | 0.0806, 26.22, 0.2998 | 0.0041, 7.75, 0.0000 |
| 13312–14335 | 0.3523, 91.66, 0.0088 | 0.0806, 26.02, 0.2979 | 0.0042, 7.75, 0.0000 |
| 14336–15359 | 0.3208, 91.97, 0.0039 | 0.0782, 25.74, 0.2979 | 0.0045, 7.75, 0.0000 |
| 15360–16383 | 0.3362, 91.79, 0.0039 | 0.0783, 25.89, 0.3564 | 0.0042, 7.75, 0.0000 |

### Starting connections from feedback cells at the end (X2)

| Tissue | starting ones alive | strengths |
|---|---|---|
| 1. first form, E10 | 423 of 423 | 0.1250 |
| 2. coincidence, E10 | 423 of 423 | 0.1250 |
| 3. fixed | 423 of 423 | 0.1250 |
| 4. formula 11 as written | 0 of 423 | — |

### Cost over the last 4,096 bytes and the forecast for stage B round 1 (section 11.4)

| Tissue | s/byte, bytes 12,288–16,383 | mean connections a cell |
|---|---|---|
| 1. first form, E10 | 0.3382 | 91.78 |
| 2. coincidence, E10 | 0.0794 | 25.97 |
| 3. fixed | 0.0043 | 7.75 |

Brief's rule: plastic cost = max(first, coincidence) × 1.25 = 0.4227 s/byte; 262,144 × (6 × 0.4227 + 3 × 0.0043) = 185.6 CPU-hours.



## 8. Registered expectations that stage A can score (section 17)

| No. | Expectation | Stage A's outcome |
|---|---|---|
| X0 | Fewer than 300 of the 15,872 starting connections remain after step 129 in tissue 4 | **Held**: 61 remain |
| X1 | No input-line connection is given up. The number of distinct byte values in use with an input-line connection never falls below its value at birth | **Held** in tissues 1 to 3. Input-line connections never decrease (754 at birth, 11,847 at the end in tissues 1 and 2). The number of byte values in use with a line connection is 44 at birth and never lower (46 at the end in tissues 1 and 2) |
| X2 | The 423 starting connections from feedback cells remain at every count, with strength 1/8 | **Held** in tissues 1 to 3: 423 of 423 alive at the end, all at 0.125. A connection given up never returns with birth step 0, so alive at the end means alive at every count. The counts of all connections from feedback cells (5,200 and 5,146 at the end in tissues 1 and 2) include those made later, all at 0.125. In tissue 4 (formula 11 as written) none of the 423 remains; the expectation's "all" did not exclude that tissue |
| X3 | Mean connections of a cell above 32 at byte 16,383 in tissue 1 | **Held**: 91.8 |
| X6 | Outcome 1 | **Not held**: outcome 3 |
| X7 | Outcome 2 | **Not held**: outcome 3 |
| X16 | Checks 1 to 10 pass at the first attempt | **Not held**. Check 1(e) and check 9(e) failed at the first attempt and passed on the reviewer's grounds (section 3). Check 10's forecast exceeded 2.5 CPU-hours |
| X17 | In tissue 2, fewer than half of the connections from inhibitory cells into inner cells remain at byte 16,383, against birth | **Held**: 989 against 2,518 at birth (these counts include connections made later) |

## 9. Lags, emissions and what the scores say

- **Lags.** At lags 1 and 2, no tissue and no distance scores S(d, L) ≥ C0(L) + 0.02 in either window. The highest are the fixed tissue's d = 1 cells: 0.0664 and 0.0530 in A2, against C0 = 0.0703 and 0.0561. At lags 4 and 8 every score lies below C0(L) + 0.02 and below C<L(L). No tissue carries an earlier byte beyond what the present byte allows.
- **Emissions** (not scored):
  - tissue 1 emits at 0.5 % of positions;
  - tissue 2 emits at 19.0 %, its share rising from 1.4 % in the first stretch to 35.6 % in the last;
  - tissues 3 and 4 never emit.

  The five commonest emitted bytes are in the receipts (`emissions`).
- **Constant cells that carry the byte.** In tissues 1 and 3 the cells at d = 2 are counted constant by the level 0.01, yet they score 0.347 and 0.826 for the present byte. Their outputs vary, but by less than the level. Section 11.3 counts them as not informative, and so does this report.

## 10. Cost and the forecast for stage B

| Tissue | CPU s, whole run | s/byte over bytes 12,288–16,383 | Mean connections a cell there |
|---|---|---|---|
| 1. first form, E10 | 5948 | 0.3382 | 91.78 |
| 2. coincidence, E10 | 1346 | 0.0794 | 25.97 |
| 3. fixed | 72 | 0.0043 | 7.75 |
| 4. first form, formula 11 as written (512 bytes) | 3.3 | — | — |

The four tissues of stage A spent 2.05 CPU-hours (7,369 s, from the receipts). The void fifth tissue ran on one core from 02:54 to about 04:00 UTC (about 1.1 CPU-hours) until it was stopped. The checks' times are in their receipts and logs (`qual/`).

**Forecast for stage B, round 1, by section 11.4:**

- plastic cost per byte = max(0.3382, 0.0794) × 1.25 = 0.4227 s/byte;
- 262,144 bytes × (6 × 0.4227 + 3 × 0.0043) s/byte = **185.6 CPU-hours**.

This is above the ceiling of 40 CPU-hours, so by section 9 the coordinator decides on a higher ceiling or a shorter stream. Section 9 estimated 195 ms a byte at full neighbourhoods; the measured cost is 338 ms. Check 10 measured 0.474 s/byte at 91 connections.

The reviewer's message 18 names two changes of the program that change no figure:
- skipping cells whose neighbourhood is exhausted when connections are made, timed at about half the cost;
- passing only over active connections.

They were not applied in stage A, and by message 21 no more time is spent on this program. The forecast above is for this program as it ran.

## 11. Committed files

From `git ls-tree -r --name-only HEAD TISSUE_STEP1` at the commit of this report:

```
TISSUE_STEP1/STAGE_A_REPORT.md
TISSUE_STEP1/qual/check1.json
TISSUE_STEP1/qual/check10.json
TISSUE_STEP1/qual/check10.log
TISSUE_STEP1/qual/check2.json
TISSUE_STEP1/qual/check2.log
TISSUE_STEP1/qual/check3.json
TISSUE_STEP1/qual/check3.log
TISSUE_STEP1/qual/check4.json
TISSUE_STEP1/qual/check5.json
TISSUE_STEP1/qual/check5.log
TISSUE_STEP1/qual/check6.json
TISSUE_STEP1/qual/check6.log
TISSUE_STEP1/qual/check7.json
TISSUE_STEP1/qual/check7.log
TISSUE_STEP1/qual/check8.json
TISSUE_STEP1/qual/check8.log
TISSUE_STEP1/qual/check8/cut/emissions.npy
TISSUE_STEP1/qual/check8/cut/final_state.npz
TISSUE_STEP1/qual/check8/cut/receipt.json
TISSUE_STEP1/qual/check8/cut/summary.npz
TISSUE_STEP1/qual/check8/whole/emissions.npy
TISSUE_STEP1/qual/check8/whole/final_state.npz
TISSUE_STEP1/qual/check8/whole/receipt.json
TISSUE_STEP1/qual/check8/whole/summary.npz
TISSUE_STEP1/qual/check9.json
TISSUE_STEP1/qual/check9.log
TISSUE_STEP1/qual/check9_first_attempt_ties_by_rounding.json
TISSUE_STEP1/runs/A_coincidence_73129/A1.npz
TISSUE_STEP1/runs/A_coincidence_73129/A2.npz
TISSUE_STEP1/runs/A_coincidence_73129/emissions.npy
TISSUE_STEP1/runs/A_coincidence_73129/final_state.npz
TISSUE_STEP1/runs/A_coincidence_73129/log.txt
TISSUE_STEP1/runs/A_coincidence_73129/receipt.json
TISSUE_STEP1/runs/A_coincidence_73129/summary.npz
TISSUE_STEP1/runs/A_first_73129/A1.npz
TISSUE_STEP1/runs/A_first_73129/A2.npz
TISSUE_STEP1/runs/A_first_73129/emissions.npy
TISSUE_STEP1/runs/A_first_73129/final_state.npz
TISSUE_STEP1/runs/A_first_73129/log.txt
TISSUE_STEP1/runs/A_first_73129/receipt.json
TISSUE_STEP1/runs/A_first_73129/saved_state.npz
TISSUE_STEP1/runs/A_first_73129/summary.npz
TISSUE_STEP1/runs/A_first_f11written_73129/emissions.npy
TISSUE_STEP1/runs/A_first_f11written_73129/final_state.npz
TISSUE_STEP1/runs/A_first_f11written_73129/log.txt
TISSUE_STEP1/runs/A_first_f11written_73129/receipt.json
TISSUE_STEP1/runs/A_first_withdrawbyuse_73129/A1.npz
TISSUE_STEP1/runs/A_first_withdrawbyuse_73129/log.txt
TISSUE_STEP1/runs/A_first_withdrawbyuse_73129/saved_state.npz
TISSUE_STEP1/runs/A_fixed_73129/A1.npz
TISSUE_STEP1/runs/A_fixed_73129/A2.npz
TISSUE_STEP1/runs/A_fixed_73129/emissions.npy
TISSUE_STEP1/runs/A_fixed_73129/final_state.npz
TISSUE_STEP1/runs/A_fixed_73129/log.txt
TISSUE_STEP1/runs/A_fixed_73129/receipt.json
TISSUE_STEP1/runs/A_fixed_73129/summary.npz
TISSUE_STEP1/runs/queue.log
TISSUE_STEP1/stage_a_queue.sh
TISSUE_STEP1/step1_checks.py
TISSUE_STEP1/step1_read.py
TISSUE_STEP1/step1_tables.py
TISSUE_STEP1/tissue_step1.py
```


## Limits

Stage A is one seed (73129), one stream and 16,384 bytes. No feedback was given; nothing was scored as an answer; no learned value, answer or association is tested or claimed. The classifier is the experimenter's, fitted to many records; a living cell has no such classifier. The limits of section 18 of the brief hold: nothing is shown about answers, correction, keeping or generalisation, about whether later steps can use what the cells carry, about how feedback-cell connections fare when the lines are active, or about how fast giving up should act in a living tissue.
