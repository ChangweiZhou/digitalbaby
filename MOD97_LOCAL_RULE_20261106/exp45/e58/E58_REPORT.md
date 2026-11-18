# E58: a new family learned in a gated term against free learning — the decay of the lived weights, the place of the change, and the number of taught pairs

**Status.** First measurement at evidence level E2. It confirms nothing on its own, and no independent audit of it exists. The work follows the brief `TASK_E58_A_GATED_TERM_AGAINST_FREE_LEARNING_DECAY_AND_NUMBER_OF_PAIRS_20261117.md`, revision 2. The brief is on `origin/claude/vibrant-cerf-sje726` at commit c0919f75, under `PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/COORDINATION_20261117/`. It was written by the outside reviewer, acting for the coordinator.

The work is on branch `claude/e58-gated-term-kqasnme`, created from the default branch `codex/minifly-consolidated-20261023` at 7cabb13a. All paths below are relative to `MOD97_LOCAL_RULE_20261106/`, and every receipt is under `exp45/e58/`.

The figures of this report come from four kinds of source. The first is the run receipts, which these four commands print, run from `MOD97_LOCAL_RULE_20261106/`:
- `python3 exp45/e58/e58_tables.py A` prints the Part A tables. Their saved output is `exp45/e58/qual/E58_PART_A_TABLE.txt`.
- `python3 exp45/e58/e58_tables.py B` prints the Part B tables. Their saved output is `exp45/e58/qual/E58_PART_B_TABLE.txt`.
- `python3 exp45/e58/e58_norms.py` prints the norms. Its saved output is `exp45/e58/qual/E58_NORMS_PART_A.txt`.
- `python3 exp45/e58/e58_report_appendix.py` prints the tables of the appendix.

The other three sources are:
- the receipts of the checks in `exp45/e58/qual/E58_check<n>….json`, which section 3 cites;
- the incomplete receipts and logs of the interrupted runs in `exp45/e58/interrupted/`, which give the rows at which those runs stopped (section 2);
- the receipts of E40 and E48, to which checks 1, 2 and 5 compare.

The report was corrected after the reviewer's verification (`PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/E58_VERIFICATION_20261118.md` on `origin/claude/vibrant-cerf-sje726` at e16ffd88). The corrections concern the text only; no receipt was changed and no run was added.

## 1. Terms

- **Family.** Four operations on two numbers (add, sub, max, min) under one modulus. A code byte at the start of each item names the operation and the family.
- **New family.** E40's family: modulus 83, code bytes `e f g h`.
- **Lived network.** The network at the end of one of E26's lives. There are six of them: three seeds, each under two training rules. They are stored as `exp45/e26/e26_life_<S|B>_<seed>.pt`.
- **The two training rules.** *Back-propagation* (rule B) uses the exact gradient. *The sweep* (rule S) passes the error backward through each linear map using the signs of the weights multiplied by their mean absolute size.
- **Taught fraction f.** The share of the 6,889 pairs of operands of each operation that the network is taught. The other pairs are *untaught*.
- **Free learning.** Every weight of the lived network may change. The optimizer is AdamW with learning rate 0.001 and weight decay 0.25 on every tensor except the byte table (the embedding table `book.weight`).
- **Gated term.** E48's construction. Every weight of the lived network is frozen. Learning happens only in two products of rank 16 added to the two matrices of the recurrent layer, and in the four rows of the byte table that belong to the new codes. The term is switched on only for items whose code byte belongs to the new family. The program that runs the experiment reads this from the item; the network does not decide it.
- **Decay toward the start.** After each update, every tensor outside the byte table is moved toward its lived value by 0.00025 of its distance from it.
- **Reduction score.** The percentage of answers that are correct in every byte, over all untaught pairs of add and of sub whose answer needs the reduction by 83, each presented without letters. The score is the mean of the add and sub percentages.
- **Common pairs.** For each seed, the pairs of add and of sub that need the reduction, are untaught at f = 0.20 (and therefore at every fraction), and whose two operand values each occur in the same position in some pair taught at f = 0.05. The *common score* is the reduction score taken over these pairs only.
- **Row.** A number of updates at which the network is scored.
- **R-mid.** The mean reduction score over rows 512 and 1,024.
- **R-late.** The mean reduction score over rows 4,096, 5,120, 6,144, 7,168 and 8,192.
- **Life's four panels.** The test panels of add and sub under moduli 97 and 73, the families of the network's life. *The first family's two panels* are add and sub under 89.

## 2. Arms and runs

Every run starts from the lived network of its seed and rule. It makes 8,192 updates in batches of 64 items and is scored at rows 0, 16, 32, 64, 128, 256, 512, 1,024 and every 1,024 updates after that.

| Arm | What may change | Decay |
|---|---|---|
| `free` | every weight (E40's run, with more rows) | 0.25; byte table 0 |
| `free-again` | as `free`, with the items in another order (stream seed + 141) | as `free` |
| `free-wd0` | every weight | 0 everywhere |
| `free-anchor` | every weight | 0 in the optimizer; decay toward the start at 0.00025 an update; byte table none |
| `term` | the gated term of rank 16 and the four code rows | 0.25 on the term's factors |
| `term-wd0` | as `term` | 0 |

- **Part A** uses f = 0.10 (E40's taught pairs), all six arms, three seeds and both rules: 36 runs.
- **Part B** uses f = 0.05 and f = 0.20, the arms `free`, `free-wd0` and `term`, three seeds and both rules: 36 runs.

At f = 0.05 and 0.20 the taught pairs are *nested* in E40's list. At f = 0.05 they are the first 344 of E40's 689 taught pairs. At f = 0.20 they are E40's 689 and the next 689 pairs of E40's shuffled list.

The runner is `exp58_new_family.py`, a new file. It imports `e40_new_family.py` and two other files without change: `exp48_gated_run.py` and `e48_gated.py`. These two were copied byte for byte from `origin/claude/sweet-turing-lwi3no` at 2d9d1726, with `git show`. Their SHA-256 values are recorded in `E58_COPIED_SOURCES.txt`, and the runner asserts them in every receipt.

The only change the runner makes to E40's code is one replacement of `e40_new_family.split`, made before the stream, the scorer and the record of presented items are built. It enforces the brief's assertions (a) to (d) in every run.

**Software and host.** Python 3.13.18, torch 2.15.0+cpu and numpy 2.5.5, on Linux x86-64 with four cores. Each run used one thread, and at most three runs ran at once.

**Restarted runs.** The host restarted twice while runs were in progress, at about 00:19 and 06:07 UTC on 18 November 2026. Each interrupted run's incomplete receipt and log were moved to `exp45/e58/interrupted/`, and the run was started again from its beginning. Six runs were restarted:
- First restart: `free-again` S at f = 0.10 on seed 73129 (it had reached row 1,024), `free-again` B at f = 0.10 on seed 73129 (row 256), and `term-wd0` S at f = 0.10 on seed 73129 (row 7,168).
- Second restart: `free` S (row 3,072), `free` B (row 4,096) and `free-wd0` S (row 3,072), all at f = 0.05 on seed 93683.

The runs are deterministic, and only the completed receipts are used.

## 3. Checks

Each check has a receipt in `exp45/e58/qual/`.

1. **`free` is E40's run.** Pass (`E58_check1.json`). On seed 73129 under both rules, at all eight of E40's rows (64 to 8,192), the batch digest, every count of every score and `tensors_sha256` equal those of `exp45/e40/e40_lived_<rule>_73129.json`. The runner also asserts this inside every `free` run at f = 0.10, and a difference would have stopped the run.
2. **`term` is E48's run.** Pass (`E58_check2.json`). On seed 73129 under both rules, at rows 0 to 1,024, every count of every score, the batch digest and the digests of the earlier families' answers equal those of E48's rank-16 receipts. The five tensors of the term at row 1,024 are bit-equal to E48's term files. All were read from `origin/claude/sweet-turing-lwi3no` with `git show`.
3. **`free-wd0` differs from `free` in the decay only.** Pass (`E58_check3_S.json`, `E58_check3_B.json`). After one update, the largest absolute deviation of (`free-wd0` − `free`) from 0.00025 × start is 4.1e-8 (S) and 7.7e-8 (B). Without the decay the deviation would be at least 6.4e-6 (S) and 2.0e-5 (B). The byte table is bit-equal in the two arms, and the decay read back from the optimizer is 0 in both groups.
4. **`free-anchor`.** Pass (`E58_check4_<S|B>.json`). On a copy with every tensor outside the byte table doubled, the decay toward the start leaves each tensor at 1.99975 × its lived value within 5.7e-8 (S) and 1.1e-7 (B). The byte table stays bit-equal, the lived network itself is unchanged, and the decay read back from the optimizer is 0, 0.
5. **The fractions.** Pass (`E58_check5.json`). Assertions (a) to (d) hold at all three fractions and seeds, and the taught pairs are nested (344 ⊂ 689 ⊂ 1,378). Walking all 8,192 batches presents no untaught pair and every taught pair at least once. No item of an untaught panel has a taught pair, and every item of the taught panel has one. At f = 0.10 the batch digests equal E40's at E40's rows. The digest at every row is stored, and every run except `free-again` asserts its own digests against it.
   - Common pairs, add/sub: 2,491/2,671 (seed 73129), 2,556/2,560 (seed 93683), 2,718/2,629 (seed 116089).
6. **The earlier families are unchanged in `term` and `term-wd0`.** Pass (`E58_check6.json`, over all 24 term runs of Parts A and B; the earlier receipts `E58_check6_seed73129_only.json` and `E58_check6_part_a.json` cover 4 and 12 runs). At every row of every term run, the digest of the earlier families' answers equals row 0's. The hash of the lived tensors, with the code rows reset, equals the life's `tensors_sha256`. The runner asserts both at every row.
7. **Cost.** Pass (`E58_check7.json`). From the 12 Part A runs of seed 73129, a run costs 881 to 1,465 CPU-seconds, or 0.077 to 0.137 CPU-seconds per update and 15.6 to 22.7 per scored row. The forecast for the whole job was 24.1 CPU-hours, below the 35-hour limit. The 72 completed runs took 25.4 CPU-hours (the sum of `cpu_total` over the receipts). The six restarted runs also used the CPU time they spent before they were interrupted, which is not included in that sum.
8. **`term-wd0` differs from `term` in the decay only.** Pass (`E58_check8_<S|B>.json`). After one update:
   - the `down` factors of `term-wd0` are bit-equal to their starting values;
   - those of `term` equal 0.99975 × start within 3.0e-9 (the gradient of `down` at the first update is 0);
   - the `up` factors and code rows are bit-equal in the two arms;
   - the decay read back from the optimizer is 0.25/0 for `term` and 0/0 for `term-wd0`.

## 4. Part A (f = 0.10)

### 4.1 Table

R-late, with R-mid in parentheses. Source: `E58_PART_A_TABLE.txt`.

| Rule | Arm | 73129 | 93683 | 116089 |
|---|---|---|---|---|
| S | free | 79.0 (83.1) | 85.1 (83.7) | 83.5 (84.1) |
| S | free-again | 74.7 (82.4) | 82.5 (84.4) | 83.6 (85.1) |
| S | free-wd0 | 76.7 (83.3) | 84.5 (85.4) | 83.5 (83.7) |
| S | free-anchor | 84.7 (83.5) | 86.0 (86.4) | 85.8 (81.6) |
| S | term | 92.0 (90.9) | 94.4 (90.1) | 87.2 (87.7) |
| S | term-wd0 | 91.6 (91.2) | 92.7 (90.7) | 86.9 (88.0) |
| B | free | 83.0 (83.3) | 91.9 (91.3) | 89.1 (90.7) |
| B | free-again | 83.4 (88.4) | 90.7 (86.0) | 91.8 (91.3) |
| B | free-wd0 | 83.5 (75.5) | 91.6 (92.0) | 93.7 (93.1) |
| B | free-anchor | 85.4 (86.3) | 95.5 (92.0) | 94.6 (95.0) |
| B | term | 94.2 (91.9) | 98.5 (97.4) | 99.0 (98.6) |
| B | term-wd0 | 94.6 (92.4) | 97.6 (97.5) | 98.7 (98.6) |

The same file gives, for every run, the highest single row from row 64 on with its row number, the score at row 8,192 minus the score at row 4,096, R-late on the common pairs, the taught panel at row 8,192, and the score at every row.

### 4.2 Scatter of one arm

N is the absolute difference between R-late of `free-again` and R-late of `free`.
- Under the sweep, N is 4.2, 2.6 and 0.0 on the three seeds.
- Under back-propagation, N is 0.4, 1.2 and 2.6.

N is below 5 points on every seed under both rules, so verdicts are stated under both rules.

### 4.3 Readings in the words fixed by the brief

The figure is R-late. Differences are given for seeds 73129, 93683 and 116089, in that order.

**Point 1.**
- Under the sweep, `term` is above `free` (+13.0, +9.3, +3.7), so the lead of the gated term persists to 8,192 updates. The lead on R-late is larger than the lead on R-mid (+7.8, +6.4, +3.7) on two seeds and smaller by 0.02 on the third (3.68 on R-mid, 3.66 on R-late). Between R-mid and R-late, `free` fell on seeds 73129 and 116089 and rose on 93683. `term` rose on 73129 and 93683 and fell by 0.5 on 116089. Neither "free fell" nor "term rose" holds on all three seeds.
- Under back-propagation, `term` is above `free` (+11.3, +6.6, +9.8), so the lead persists. It is larger on R-late than on R-mid (+8.6, +6.1, +7.9) on all three seeds. `term` rose on all three seeds (91.9 → 94.2, 97.4 → 98.5, 98.6 → 99.0). `free` fell on 73129 and 116089 and rose on 93683.

**Point 2.** The lead is split into L = R-late(`term`) − R-late(`free`), A = R-late(`free-anchor`) − R-late(`free`) and P = R-late(`term`) − R-late(`free-anchor`).

| Rule | Seed | L | A | P | a = A / L |
|---|---|---|---|---|---|
| S | 73129 | +13.0 | +5.7 | +7.3 | 0.44 |
| S | 93683 | +9.3 | +0.9 | +8.4 | 0.10 |
| S | 116089 | +3.7 | +2.2 | +1.4 | (L < 5) |
| B | 73129 | +11.3 | +2.4 | +8.8 | 0.21 |
| B | 93683 | +6.6 | +3.6 | +3.0 | 0.55 |
| B | 116089 | +9.8 | +5.4 | +4.4 | 0.55 |

Under the sweep, the seeds on which L is 5 or more give a = 0.44 and 0.10. Under back-propagation they give a = 0.21, 0.55 and 0.55. **Under both rules the seeds disagree.** The report therefore states none of the three sentences ("the decay of the lived weights accounts for the lead", "the place of the change accounts for the lead", "both contribute").

**Point 3.**
- `term-wd0` against `term`: not distinguished under the sweep (−0.4, −1.7, −0.3). Under back-propagation the comparison is at the ceiling of the score: the figures are 94.6/94.2, 97.6/98.5 and 98.7/99.0, and 100 minus each is 5.4/5.8, 2.4/1.5 and 1.3/1.0. No verdict is stated.
- `free-wd0` against `free-anchor`: not distinguished under the sweep (−7.9, −1.5, −2.3) or back-propagation (−1.9, −3.9, −0.9).
- `term-wd0` against `free-wd0`: above under the sweep (+14.8, +8.2, +3.4) and under back-propagation (+11.1, +6.0, +5.1).
- `term` against `free-anchor`, which is P of point 2 read as a comparison: above under the sweep (+7.3, +8.4, +1.4); not distinguished under back-propagation (+8.8, +3.0, +4.4).
- `free-anchor` against `free`, which is A read as a comparison: not distinguished under the sweep (+5.7, +0.9, +2.2) or back-propagation (+2.4, +3.6, +5.4).

These readings enter no sentence of point 2.

**Point 4.** `free-wd0` against `free`, the effect of removing the weight decay altogether, is not distinguished under the sweep (−2.2, −0.5, −0.0) or under back-propagation (+0.5, −0.3, +4.5).

**Point 5.** Appendix A.3 (and the table file) gives, at every row of the three free arms, the mean of the life's four panels, the mean of the first family's two panels, the norm of the tensors outside the byte table, and the norm of their change.

- **Loss of the life's families.** At row 0 the mean of the life's four panels is 86.6 (S) and 90.5 (B) on seed 73129. At row 16 it has fallen to 60.9 and 56.6 in `free`, while the norm of the weights is larger by 0.21% (S) and 0.23% (B). At row 1,024 it is:

  | Rule | free | free-wd0 | free-anchor |
  |---|---|---|---|
  | S | 45.7, 38.0, 43.5 | 42.2, 41.9, 44.8 | 47.9, 43.6, 47.0 |
  | B | 47.1, 49.2, 47.8 | 45.8, 44.8, 44.6 | 46.3, 50.4, 48.2 |

  The loss of the earlier families in free learning therefore happens in the first updates, and it occurs whether or not the weights decay.
- **Norms** (`E58_NORMS_PART_A.txt`), as the norm of the tensors outside the byte table relative to the lived network, at row 8,192:

  | Arm | S | B |
  |---|---|---|
  | `free` | 0.84 to 0.95 | 0.91 to 0.97 |
  | `free-wd0` | 1.47 to 1.58 | 1.53 to 1.61 |
  | `free-anchor` | 1.20 to 1.21 | 1.18 to 1.26 |

  The change relative to the lived norm at row 8,192 is 0.61 to 0.75 in `free`, 0.82 to 0.96 in `free-wd0` and 0.36 to 0.52 in `free-anchor`.
- **Does the arm that keeps more also score higher?** Within each seed and rule, the three free arms were ordered by the mean of the life's four panels over the five late rows (this ordering is descriptive and was not fixed beforehand). `free-anchor` keeps the most on all six seed-rule combinations: 42.4 to 52.5, against 22.2 to 41.9 in `free`. It also has the highest R-late on all six. The R-late differences behind this ordering are mostly below the 5-point margin (section 4.3, `free-anchor` against `free`), so it is a direction consistent with the explanation under test, not a verdict.

## 5. Part B (f = 0.05, 0.10, 0.20)

All figures in this section are from `E58_PART_B_TABLE.txt`. Appendix A.1 gives R-late of every run on its own untaught pairs and the taught panel at row 8,192 by run; appendix A.2 gives the first rows at 80 and 90. The table file also gives every run's R-mid, highest row, score at row 8,192 minus score at row 4,096, curve and taught panel. The taught panel at row 8,192 is 95.8 to 100.0 in every run of Part B.

### 5.1 R-late on the common pairs

| Rule | Arm | f | 73129 | 93683 | 116089 |
|---|---|---|---|---|---|
| S | free | 0.05 | 42.8 | 50.5 | 57.3 |
| S | free-wd0 | 0.05 | 75.9 | 79.4 | 76.4 |
| S | term | 0.05 | 85.7 | 83.8 | 74.1 |
| S | free | 0.10 | 79.3 | 85.1 | 83.7 |
| S | free-wd0 | 0.10 | 76.9 | 84.6 | 83.6 |
| S | term | 0.10 | 92.1 | 94.6 | 87.2 |
| S | free | 0.20 | 91.3 | 92.4 | 93.1 |
| S | free-wd0 | 0.20 | 92.4 | 94.5 | 93.6 |
| S | term | 0.20 | 96.4 | 95.6 | 94.4 |
| B | free | 0.05 | 55.6 | 70.2 | 75.3 |
| B | free-wd0 | 0.05 | 69.7 | 89.6 | 85.4 |
| B | term | 0.05 | 88.6 | 91.7 | 93.9 |
| B | free | 0.10 | 83.2 | 91.8 | 89.2 |
| B | free-wd0 | 0.10 | 83.9 | 91.4 | 93.7 |
| B | term | 0.10 | 94.3 | 98.5 | 99.0 |
| B | free | 0.20 | 92.5 | 97.1 | 95.3 |
| B | free-wd0 | 0.20 | 94.2 | 96.7 | 96.8 |
| B | term | 0.20 | 97.8 | 99.6 | 99.8 |

**Free learning at f = 0.05 does not hold its level.** The reduction score of `free` at f = 0.05 is 70.9 to 91.1 at row 1,024 and 36.6 to 78.4 at row 8,192. Row 8,192 minus row 4,096 is −14.5 to −19.1 under the sweep and −4.2 to −10.9 under back-propagation, so this arm is still falling when the runs end.

`free-wd0` at f = 0.05 ends at 60.3 to 89.5. `term` at f = 0.05 ends at 68.7 to 92.5. Its largest fall from row 1,024 to row 8,192 is on seed 116089 under the sweep, from 81.6 to 68.7.

### 5.2 The term against free learning at each fraction

The figure is R-late of the reduction score. Differences are for seeds 73129, 93683 and 116089.

| Rule | f | `term` against `free` | `term` against `free-wd0` |
|---|---|---|---|
| S | 0.05 | above (+43.0, +33.1, +16.4) | not distinguished (+9.5, +4.2, −2.5) |
| S | 0.10 | above (+13.0, +9.3, +3.7) | above (+15.3, +9.9, +3.7) |
| S | 0.20 | not distinguished (+5.1, +3.2, +1.3) | not distinguished (+4.0, +1.1, +0.8) |
| B | 0.05 | above (+33.5, +21.4, +18.5) | above (+19.6, +2.0, +8.4) |
| B | 0.10 | above (+11.3, +6.6, +9.8) | above (+10.8, +6.9, +5.3) |
| B | 0.20 | at the ceiling | at the ceiling |

The two ceiling comparisons under back-propagation at f = 0.20:
- `term` against `free`: figures 97.7/92.3, 99.5/97.1 and 99.8/95.3; 100 minus each is 2.3/7.7, 0.5/2.9 and 0.2/4.7.
- `term` against `free-wd0`: figures 97.7/94.1, 99.5/96.6 and 99.8/96.7; 100 minus each is 2.3/5.9, 0.5/3.4 and 0.2/3.3.

### 5.3 Comparisons between fractions (R-late on the common pairs)

For a rule and a pair of fractions (f, g), "deficit" means the higher of `free` and `free-wd0` at g minus `term` at f, on each seed. The brief's conditions are:
- (a) `free` at g is above `free` at f;
- (b) the deficit is at most 2 points on at least two seeds and below 5 points on the third;
- (c) the comparison is not at the ceiling.

**Within the sweep:**
- (0.05, 0.10): (a) holds (+36.5, +34.6, +26.4). The deficit is −6.3, +1.2 and +9.6, so (b) fails. No sentence is stated.
- (0.10, 0.20): (a) holds (+12.0, +7.4, +9.4). The deficit is +0.4, −0.2 and +6.4, so (b) fails on the third seed, where the deficit is 5 or more. No sentence is stated.
- (0.05, 0.20): (a) holds (+48.5, +41.9, +35.7). The deficit is +6.8, +10.6 and +19.5. No sentence is stated.

**Within back-propagation:**
- (0.05, 0.10): (a) holds (+27.7, +21.6, +13.9). The deficit is −4.7, +0.1 and −0.2, so (b) holds, and (c) holds. **Under back-propagation, the gated term reaches from f = 0.05 the level that free learning reaches from g = 0.10.**
- (0.10, 0.20): (a) holds (+9.2, +5.4, +6.1). The deficit is −0.1, −1.4 and −2.2, but the comparison is at the ceiling of the score: the lower figure exceeds 95 on seeds 93683 and 116089. No statement is made.
- (0.05, 0.20): (a) holds (+36.9, +26.9, +20.0). The deficit is +5.6, +5.4 and +2.9. No sentence is stated.

**Across the rules** (`term` under the sweep at f, against `free` and `free-wd0` under back-propagation at g):
- (0.05, 0.10): deficit −1.8, +7.9, +19.6.
- (0.10, 0.20): deficit +2.1, +2.5, +9.6.
- (0.05, 0.20): deficit +8.5, +13.3, +22.6.

Condition (a) was read under back-propagation, the rule of the free arms in this comparison; it also holds under the sweep (section 5.3, within the sweep). Condition (b) fails in all three pairs. As the brief prescribes when (b) fails, only the deficits by seed are given; no sentence is stated.

### 5.4 Number of presented items

Appendix A.2 gives, for every run of Part B, the first row at which the reduction score is 80 or more and 90 or more, with the number of presentations of each taught pair by that row (the row times 64 items, divided by four times the number of taught pairs).
- Under the sweep, `term` first reaches 80 at rows 256 to 512, and `free` and `free-wd0` at rows 64 to 512. The exception is `free` at f = 0.05 on seed 116089, which never reaches 80.
- Under back-propagation, `term` reaches 80 at rows 64 to 256, and the free arms at rows 32 to 128.

The two levels give different pictures. Appendix A.2 compares `term` with `free` and with `free-wd0` in 36 pairs of runs (2 rules × 3 fractions × 3 seeds × 2 free arms).
- **At the level of 80**, `term` is later in 33 of 36 pairs. The three exceptions are all under the sweep: two are equal (f = 0.05 on seed 93683 and f = 0.20 on seed 116089, both against `free`), and in one `free` never reaches 80 (f = 0.05 on seed 116089).
- **At the level of 90**, `term` is earlier in 6 pairs, and in 10 more `term` reaches 90 while the free arm never does: 16 of 36 in all. `term` is later in 11 pairs, the two are equal in 3, and neither reaches 90 in 6.

So the gated term is slower to reach a moderate level in presented items, as in E48, but it reaches the higher level earlier, or alone, in 16 of the 36 pairs. The statements of section 5.3 concern the number of different taught pairs at an equal number of presented items (524,288 at row 8,192), not the number of items a learner needs.

### 5.5 What Part B says about the explanation under test

The following reading was not fixed beforehand, and it is not offered as support for the explanation under test. The gap between `term` and `free` is largest at f = 0.05, where `free` loses most over the run. But at f = 0.05 under the sweep, most of that gap closes in `free-wd0`, which also changes every weight. `term` against `free-wd0` is not distinguished there (+9.5, +4.2, −2.5), where `term` against `free` differs by +43.0, +33.1 and +16.4. The gap therefore cannot be attributed to the place of the change on this evidence.

The explanation's second part says that a change confined to the new family's connections reaches the same level from fewer pairs. Under back-propagation this holds for (0.05, 0.10). It does not hold under the sweep, or across the rules, by the brief's criteria.

## 6. Predictions

| | Statement | Chance given | Outcome |
|---|---|---|---|
| P58.1 | Checks 1 to 5 and 8 pass | 85% | True: all pass (section 3) |
| P58.2 | `term` is above `free` at f = 0.10 under the sweep | 65% | True (+13.0, +9.3, +3.7) |
| P58.3 | `term` is above `free` at f = 0.10 under back-propagation | 70% | True (+11.3, +6.6, +9.8) |
| P58.4 | `free-wd0` is above `free` at f = 0.10 under at least one rule | 25% | False: not distinguished under both rules |
| P58.5 | `term-wd0` is not distinguished from `term` under both rules | 60% | Not decided. It is not distinguished under the sweep and at the ceiling under back-propagation, where no verdict is stated. Recording it as not decided is my choice: the brief has no rule for a prediction whose comparison is at the ceiling. |
| P58.6 | `free-anchor` is above `free` at f = 0.10 under at least one rule | 25% | False: not distinguished under both rules |
| P58.7 | The mean of the life's four panels at row 1,024 is higher in `free-anchor` than in `free` by 10 points or more on two of three seeds, under at least one rule | 15% | False: differences +2.2, +5.6, +3.5 (S) and −0.8, +1.2, +0.5 (B) |
| P58.8 | At f = 0.05, `term` is above `free` under both rules | 50% | True (S +43.0, +33.1, +16.4; B +33.5, +21.4, +18.5) |
| P58.9 | For (0.05, 0.10) the report says "the gated term reaches from f the level that free learning reaches from g" under at least one rule | 30% | True: under back-propagation |
| P58.10 | At f = 0.20 under back-propagation, `term` against `free` is at the ceiling or not distinguished | 60% | True: at the ceiling |
| P58.11 | N is below 5 points on every seed under both rules | 75% | True (4.2, 2.6, 0.0; 0.4, 1.2, 2.6) |

**The explanation under test.** Neither refutation clause of the brief holds.
- Both parts would be refuted if `term` were above `free` under neither rule. It is above under both rules at f = 0.10.
- The second part would be refuted if the decay of the lived weights accounted for the lead under both rules. Under both rules the seeds disagree on the share a, so no sentence of point 2 is stated.

The explanation is therefore not refuted in this setting. This is not a confirmation: the first measurement is at level E2, and no independent audit exists.

## 7. What the results cannot show

The limits listed in the brief apply unchanged:
- The gate is read from the item by the program that runs the experiment; the network does not form it.
- The fourth cell of the design, the decay of the lived weights with the change confined to the term, cannot be realised.
- Part B compares numbers of different taught pairs at an equal number of presented items. It does not compare the number of items a learner needs.
- Pairs vary by no more than a factor of four.
- There is one new family, rank 16 only, a term on the two recurrent matrices only (no term on other matrices), and one learning rate.
- The job has no families in sequence, and memory is not fixed: each family would add a term.
- The job says nothing about Mr. Xu's requirement that long-term memory form in the course of learning without intervention. The separation is supplied from outside.

Further limits concern this report:
- **The one licensed sentence is close to its threshold.** That sentence is section 5.3, back-propagation, (0.05, 0.10), with deficits −4.7, +0.1 and −0.2. On two seeds it meets the 2-point condition by only 1.9 and 2.2 points. The scatter of one arm on the same score (R-late on the common pairs, `free-again` against `free` at f = 0.10 under back-propagation) is 1.1 and 2.5 points on those two seeds.
- **The scatter was measured for one arm only.** N comes from `free` at f = 0.10 alone. It was not measured for the term arms, nor at f = 0.05. At f = 0.05, `free` changes by −4.2 to −19.1 points between rows 4,096 and 8,192, and `free-wd0` by up to −12.2.
- **R-mid scatters more than R-late.** On R-mid, the scatter of one arm under back-propagation is 5.1 and 5.3 points on two seeds (73129 and 93683), so comparisons on R-mid under that rule are less secure than those on R-late.
- **The ceiling hides differences.** Several comparisons under back-propagation (point 3, section 5.2 at f = 0.20, and section 5.3 for (0.10, 0.20)) are at the ceiling of the score. Above 95 a difference of 5 points cannot occur, so these comparisons cannot show a difference that may exist.
- **Some levels may not have settled.** Levels still moving at 8,192 updates may not have reached their end. In particular, `free` at f = 0.05 is still falling at row 8,192 (section 5).
- **Verdicts are thin.** Every verdict rests on three seeds and a 5-point margin. The scatter of one arm, N, reaches 4.2 points on one seed under the sweep.

## Appendix (printed by `python3 exp45/e58/e58_report_appendix.py`)

### A.1 Part B: R-late of the reduction score on each run's own untaught pairs, and the taught panel at row 8,192

| Rule | Arm | f | R-late 73129 | R-late 93683 | R-late 116089 | taught 73129 | taught 93683 | taught 116089 |
|---|---|---|---|---|---|---|---|---|
| S | free | 0.05 | 42.2 | 50.2 | 57.6 | 97.7 | 98.5 | 99.0 |
| S | free-wd0 | 0.05 | 75.8 | 79.1 | 76.5 | 100.0 | 100.0 | 100.0 |
| S | term | 0.05 | 85.2 | 83.3 | 74.0 | 100.0 | 100.0 | 98.3 |
| S | free | 0.10 | 79.0 | 85.1 | 83.5 | 99.0 | 99.5 | 99.7 |
| S | free-wd0 | 0.10 | 76.7 | 84.5 | 83.5 | 99.8 | 99.8 | 99.7 |
| S | term | 0.10 | 92.0 | 94.4 | 87.2 | 100.0 | 100.0 | 99.2 |
| S | free | 0.20 | 91.2 | 92.4 | 93.1 | 96.8 | 95.8 | 99.8 |
| S | free-wd0 | 0.20 | 92.3 | 94.4 | 93.6 | 99.2 | 98.8 | 99.7 |
| S | term | 0.20 | 96.3 | 95.6 | 94.4 | 100.0 | 99.8 | 99.8 |
| B | free | 0.05 | 55.0 | 70.1 | 75.4 | 98.0 | 99.2 | 100.0 |
| B | free-wd0 | 0.05 | 68.9 | 89.5 | 85.4 | 100.0 | 100.0 | 99.3 |
| B | term | 0.05 | 88.5 | 91.5 | 93.8 | 100.0 | 100.0 | 99.8 |
| B | free | 0.10 | 83.0 | 91.9 | 89.1 | 98.5 | 100.0 | 99.0 |
| B | free-wd0 | 0.10 | 83.5 | 91.6 | 93.7 | 99.5 | 98.5 | 100.0 |
| B | term | 0.10 | 94.2 | 98.5 | 99.0 | 99.8 | 100.0 | 100.0 |
| B | free | 0.20 | 92.3 | 97.1 | 95.3 | 98.7 | 98.7 | 99.3 |
| B | free-wd0 | 0.20 | 94.1 | 96.6 | 96.7 | 99.8 | 99.8 | 99.8 |
| B | term | 0.20 | 97.7 | 99.5 | 99.8 | 100.0 | 100.0 | 100.0 |

### A.2 First row at which the reduction score is 80 or more and 90 or more (in parentheses: presentations of each taught pair by that row)

| Rule | f | Arm | 80: 73129 | 80: 93683 | 80: 116089 | 90: 73129 | 90: 93683 | 90: 116089 |
|---|---|---|---|---|---|---|---|---|
| S | 0.05 | free | 64 (3.0) | 512 (23.8) | never | never | never | never |
| S | 0.05 | free-wd0 | 64 (3.0) | 128 (6.0) | 128 (6.0) | never | never | never |
| S | 0.05 | term | 256 (11.9) | 512 (23.8) | 512 (23.8) | never | never | never |
| S | 0.10 | free | 128 (3.0) | 256 (5.9) | 128 (3.0) | never | never | never |
| S | 0.10 | free-wd0 | 128 (3.0) | 256 (5.9) | 128 (3.0) | never | 2048 (47.6) | never |
| S | 0.10 | term | 256 (5.9) | 512 (11.9) | 512 (11.9) | 1024 (23.8) | 1024 (23.8) | 4096 (95.1) |
| S | 0.20 | free | 128 (1.5) | 128 (1.5) | 512 (5.9) | 2048 (23.8) | 512 (5.9) | 2048 (23.8) |
| S | 0.20 | free-wd0 | 128 (1.5) | 128 (1.5) | 128 (1.5) | 2048 (23.8) | 1024 (11.9) | 1024 (11.9) |
| S | 0.20 | term | 256 (3.0) | 512 (5.9) | 512 (5.9) | 512 (5.9) | 1024 (11.9) | 1024 (11.9) |
| B | 0.05 | free | 128 (6.0) | 32 (1.5) | 32 (1.5) | never | 1024 (47.6) | 64 (3.0) |
| B | 0.05 | free-wd0 | 64 (3.0) | 32 (1.5) | 32 (1.5) | never | never | 64 (3.0) |
| B | 0.05 | term | 256 (11.9) | 256 (11.9) | 64 (3.0) | 1024 (47.6) | 256 (11.9) | 128 (6.0) |
| B | 0.10 | free | 64 (1.5) | 32 (0.7) | 32 (0.7) | never | 64 (1.5) | 64 (1.5) |
| B | 0.10 | free-wd0 | 64 (1.5) | 32 (0.7) | 32 (0.7) | never | 64 (1.5) | 64 (1.5) |
| B | 0.10 | term | 256 (5.9) | 128 (3.0) | 64 (1.5) | 512 (11.9) | 256 (5.9) | 128 (3.0) |
| B | 0.20 | free | 64 (0.7) | 32 (0.4) | 32 (0.4) | 512 (5.9) | 64 (0.7) | 64 (0.7) |
| B | 0.20 | free-wd0 | 64 (0.7) | 32 (0.4) | 32 (0.4) | 2048 (23.8) | 64 (0.7) | 128 (1.5) |
| B | 0.20 | term | 256 (3.0) | 128 (1.5) | 64 (0.7) | 512 (5.9) | 256 (3.0) | 256 (3.0) |

Comparisons of `term` with `free` and with `free-wd0` (2 rules x 3 fractions x 3 seeds x 2 arms = 36 at each level):

- level 80: term later 33; equal 2; term earlier 0; free never term reaches 1; term never free reaches 0; neither 0
- level 90: term later 11; equal 3; term earlier 6; free never term reaches 10; term never free reaches 0; neither 6

### A.3 Part A, point 5, every row of the three free arms: mean of the life's four panels / mean of the first family's two panels / norm of the tensors outside the byte table relative to the lived network / norm of their change relative to the lived norm

- **S free 73129**: 0: 86.6 / 33.0 / 1.000 / 0.000; 16: 60.9 / 31.7 / 1.002 / 0.065; 32: 66.0 / 36.5 / 1.002 / 0.078; 64: 69.4 / 37.3 / 1.000 / 0.088; 128: 66.1 / 40.5 / 0.993 / 0.100; 256: 45.2 / 30.0 / 0.990 / 0.142; 512: 46.3 / 35.8 / 0.984 / 0.202; 1024: 45.7 / 39.2 / 0.970 / 0.283; 2048: 43.6 / 38.3 / 0.941 / 0.381; 3072: 40.2 / 34.7 / 0.933 / 0.453; 4096: 37.7 / 32.5 / 0.939 / 0.509; 5120: 39.7 / 31.5 / 0.947 / 0.560; 6144: 39.2 / 32.8 / 0.923 / 0.588; 7168: 37.9 / 29.7 / 0.928 / 0.621; 8192: 32.5 / 25.5 / 0.950 / 0.661
- **S free 93683**: 0: 88.2 / 17.7 / 1.000 / 0.000; 16: 41.6 / 16.8 / 1.003 / 0.068; 32: 39.5 / 15.8 / 1.002 / 0.085; 64: 42.4 / 17.7 / 0.998 / 0.096; 128: 42.4 / 11.7 / 0.990 / 0.108; 256: 49.0 / 14.7 / 0.982 / 0.144; 512: 43.8 / 16.2 / 0.968 / 0.199; 1024: 38.0 / 9.7 / 0.938 / 0.284; 2048: 29.2 / 14.2 / 0.905 / 0.408; 3072: 26.3 / 8.5 / 0.880 / 0.497; 4096: 21.4 / 9.0 / 0.873 / 0.568; 5120: 25.3 / 12.3 / 0.870 / 0.629; 6144: 21.9 / 12.0 / 0.875 / 0.678; 7168: 24.7 / 16.2 / 0.853 / 0.707; 8192: 17.5 / 10.8 / 0.889 / 0.755
- **S free 116089**: 0: 92.8 / 13.5 / 1.000 / 0.000; 16: 46.7 / 14.5 / 1.001 / 0.062; 32: 52.7 / 18.2 / 1.001 / 0.078; 64: 55.9 / 12.3 / 0.999 / 0.091; 128: 57.1 / 18.8 / 0.990 / 0.105; 256: 49.7 / 14.2 / 0.980 / 0.144; 512: 49.3 / 21.7 / 0.961 / 0.203; 1024: 43.5 / 26.0 / 0.926 / 0.289; 2048: 40.6 / 26.3 / 0.883 / 0.409; 3072: 39.2 / 24.5 / 0.862 / 0.495; 4096: 36.4 / 25.8 / 0.845 / 0.562; 5120: 39.9 / 32.2 / 0.817 / 0.605; 6144: 37.9 / 28.5 / 0.831 / 0.649; 7168: 36.4 / 28.5 / 0.839 / 0.684; 8192: 37.8 / 32.2 / 0.836 / 0.710
- **S free-wd0 73129**: 0: 86.6 / 33.0 / 1.000 / 0.000; 16: 61.0 / 32.0 / 1.006 / 0.066; 32: 66.6 / 37.0 / 1.010 / 0.079; 64: 70.3 / 37.8 / 1.015 / 0.090; 128: 66.8 / 40.3 / 1.022 / 0.102; 256: 48.3 / 30.7 / 1.040 / 0.143; 512: 44.3 / 37.0 / 1.073 / 0.209; 1024: 42.2 / 35.3 / 1.126 / 0.299; 2048: 43.9 / 37.2 / 1.213 / 0.433; 3072: 43.1 / 38.2 / 1.289 / 0.539; 4096: 41.9 / 38.7 / 1.357 / 0.631; 5120: 38.8 / 34.8 / 1.416 / 0.708; 6144: 40.8 / 37.7 / 1.473 / 0.782; 7168: 41.4 / 37.2 / 1.525 / 0.848; 8192: 40.4 / 32.7 / 1.576 / 0.911
- **S free-wd0 93683**: 0: 88.2 / 17.7 / 1.000 / 0.000; 16: 42.1 / 16.8 / 1.007 / 0.069; 32: 40.4 / 16.7 / 1.010 / 0.086; 64: 45.1 / 18.2 / 1.014 / 0.097; 128: 45.1 / 12.3 / 1.020 / 0.109; 256: 48.5 / 11.7 / 1.034 / 0.141; 512: 42.1 / 9.0 / 1.064 / 0.203; 1024: 41.9 / 20.5 / 1.109 / 0.288; 2048: 41.6 / 17.8 / 1.183 / 0.407; 3072: 41.6 / 23.5 / 1.251 / 0.509; 4096: 38.7 / 20.7 / 1.309 / 0.593; 5120: 39.4 / 22.5 / 1.366 / 0.670; 6144: 37.9 / 28.5 / 1.415 / 0.739; 7168: 30.0 / 21.3 / 1.462 / 0.797; 8192: 27.1 / 24.7 / 1.509 / 0.857
- **S free-wd0 116089**: 0: 92.8 / 13.5 / 1.000 / 0.000; 16: 46.8 / 14.5 / 1.005 / 0.062; 32: 54.1 / 19.0 / 1.009 / 0.078; 64: 57.4 / 13.0 / 1.014 / 0.091; 128: 56.4 / 18.7 / 1.019 / 0.104; 256: 43.4 / 13.0 / 1.035 / 0.144; 512: 48.4 / 22.3 / 1.061 / 0.203; 1024: 44.8 / 29.8 / 1.104 / 0.287; 2048: 45.2 / 32.5 / 1.174 / 0.401; 3072: 42.0 / 34.2 / 1.234 / 0.496; 4096: 40.8 / 29.0 / 1.290 / 0.578; 5120: 39.4 / 31.8 / 1.341 / 0.648; 6144: 37.6 / 33.3 / 1.390 / 0.712; 7168: 37.3 / 30.8 / 1.431 / 0.766; 8192: 36.7 / 33.0 / 1.475 / 0.823
- **S free-anchor 73129**: 0: 86.6 / 33.0 / 1.000 / 0.000; 16: 61.1 / 32.0 / 1.006 / 0.065; 32: 66.7 / 36.8 / 1.010 / 0.078; 64: 70.8 / 38.0 / 1.014 / 0.089; 128: 66.9 / 40.5 / 1.021 / 0.100; 256: 50.4 / 33.2 / 1.039 / 0.139; 512: 47.4 / 37.2 / 1.068 / 0.199; 1024: 47.9 / 37.2 / 1.110 / 0.267; 2048: 45.1 / 36.7 / 1.165 / 0.357; 3072: 46.7 / 37.0 / 1.195 / 0.405; 4096: 45.3 / 40.0 / 1.215 / 0.436; 5120: 44.5 / 41.8 / 1.220 / 0.442; 6144: 39.1 / 33.7 / 1.231 / 0.461; 7168: 45.4 / 37.3 / 1.241 / 0.475; 8192: 45.5 / 41.5 / 1.212 / 0.421
- **S free-anchor 93683**: 0: 88.2 / 17.7 / 1.000 / 0.000; 16: 42.1 / 17.0 / 1.007 / 0.068; 32: 40.4 / 16.8 / 1.010 / 0.085; 64: 44.7 / 18.2 / 1.013 / 0.096; 128: 45.3 / 12.2 / 1.019 / 0.107; 256: 47.1 / 12.8 / 1.033 / 0.136; 512: 48.9 / 13.2 / 1.057 / 0.189; 1024: 43.6 / 20.0 / 1.091 / 0.254; 2048: 43.2 / 25.8 / 1.138 / 0.331; 3072: 44.1 / 17.7 / 1.171 / 0.385; 4096: 45.9 / 25.5 / 1.185 / 0.407; 5120: 42.8 / 29.0 / 1.199 / 0.427; 6144: 40.6 / 23.8 / 1.207 / 0.441; 7168: 45.2 / 18.7 / 1.212 / 0.448; 8192: 41.7 / 30.0 / 1.200 / 0.427
- **S free-anchor 116089**: 0: 92.8 / 13.5 / 1.000 / 0.000; 16: 46.9 / 14.5 / 1.005 / 0.062; 32: 54.4 / 18.8 / 1.009 / 0.078; 64: 57.5 / 12.8 / 1.013 / 0.090; 128: 57.4 / 19.0 / 1.019 / 0.102; 256: 49.0 / 14.8 / 1.033 / 0.138; 512: 48.4 / 19.2 / 1.056 / 0.192; 1024: 47.0 / 22.7 / 1.088 / 0.255; 2048: 43.8 / 28.2 / 1.131 / 0.332; 3072: 43.4 / 30.8 / 1.158 / 0.375; 4096: 43.6 / 28.5 / 1.176 / 0.403; 5120: 42.3 / 29.0 / 1.186 / 0.421; 6144: 40.2 / 28.7 / 1.190 / 0.423; 7168: 42.4 / 31.5 / 1.198 / 0.438; 8192: 43.6 / 32.2 / 1.202 / 0.441
- **B free 73129**: 0: 90.5 / 37.3 / 1.000 / 0.000; 16: 56.6 / 38.3 / 1.002 / 0.073; 32: 60.4 / 41.3 / 1.003 / 0.089; 64: 61.4 / 41.5 / 1.001 / 0.100; 128: 60.6 / 43.0 / 0.994 / 0.113; 256: 48.9 / 39.7 / 0.992 / 0.156; 512: 44.4 / 38.5 / 0.982 / 0.208; 1024: 47.1 / 33.8 / 0.970 / 0.291; 2048: 40.8 / 37.2 / 0.957 / 0.404; 3072: 40.7 / 32.3 / 0.959 / 0.487; 4096: 39.7 / 33.5 / 0.946 / 0.540; 5120: 39.9 / 35.3 / 0.955 / 0.586; 6144: 40.4 / 36.5 / 0.953 / 0.622; 7168: 38.0 / 33.2 / 0.959 / 0.662; 8192: 33.4 / 31.0 / 0.970 / 0.691
- **B free 93683**: 0: 96.8 / 26.7 / 1.000 / 0.000; 16: 65.6 / 29.8 / 1.000 / 0.080; 32: 67.7 / 34.5 / 0.999 / 0.095; 64: 68.9 / 36.3 / 0.996 / 0.104; 128: 68.3 / 28.7 / 0.988 / 0.114; 256: 58.4 / 28.3 / 0.984 / 0.148; 512: 51.8 / 28.5 / 0.973 / 0.201; 1024: 49.2 / 31.5 / 0.963 / 0.274; 2048: 44.1 / 20.7 / 0.945 / 0.363; 3072: 39.6 / 17.3 / 0.946 / 0.432; 4096: 41.2 / 12.3 / 0.939 / 0.481; 5120: 37.6 / 10.8 / 0.939 / 0.524; 6144: 31.9 / 9.7 / 0.946 / 0.563; 7168: 36.9 / 9.2 / 0.941 / 0.586; 8192: 38.8 / 15.0 / 0.932 / 0.607
- **B free 116089**: 0: 90.8 / 36.3 / 1.000 / 0.000; 16: 54.3 / 41.2 / 0.999 / 0.072; 32: 54.2 / 44.3 / 1.000 / 0.089; 64: 57.7 / 45.8 / 0.999 / 0.098; 128: 57.7 / 46.0 / 0.990 / 0.106; 256: 51.3 / 44.5 / 0.988 / 0.145; 512: 48.6 / 42.7 / 0.982 / 0.198; 1024: 47.8 / 43.2 / 0.967 / 0.267; 2048: 45.0 / 40.7 / 0.946 / 0.365; 3072: 43.8 / 41.8 / 0.944 / 0.435; 4096: 40.7 / 40.3 / 0.949 / 0.491; 5120: 43.2 / 41.0 / 0.953 / 0.540; 6144: 41.1 / 40.3 / 0.952 / 0.579; 7168: 42.6 / 39.2 / 0.944 / 0.607; 8192: 41.6 / 38.2 / 0.915 / 0.621
- **B free-wd0 73129**: 0: 90.5 / 37.3 / 1.000 / 0.000; 16: 56.9 / 38.2 / 1.006 / 0.073; 32: 60.5 / 41.2 / 1.010 / 0.089; 64: 61.2 / 42.0 / 1.016 / 0.102; 128: 59.9 / 42.8 / 1.023 / 0.115; 256: 51.7 / 44.3 / 1.046 / 0.161; 512: 47.9 / 37.3 / 1.079 / 0.221; 1024: 45.8 / 39.8 / 1.139 / 0.319; 2048: 42.4 / 38.2 / 1.236 / 0.464; 3072: 42.1 / 33.5 / 1.317 / 0.578; 4096: 41.8 / 39.7 / 1.383 / 0.666; 5120: 38.6 / 33.3 / 1.450 / 0.753; 6144: 42.2 / 35.3 / 1.505 / 0.826; 7168: 39.1 / 34.5 / 1.557 / 0.892; 8192: 40.6 / 35.7 / 1.609 / 0.956
- **B free-wd0 93683**: 0: 96.8 / 26.7 / 1.000 / 0.000; 16: 65.7 / 29.8 / 1.004 / 0.080; 32: 67.9 / 34.8 / 1.007 / 0.095; 64: 69.9 / 36.3 / 1.011 / 0.105; 128: 68.7 / 31.8 / 1.018 / 0.115; 256: 58.1 / 33.5 / 1.038 / 0.152; 512: 61.4 / 36.0 / 1.072 / 0.214; 1024: 44.8 / 30.7 / 1.125 / 0.296; 2048: 49.6 / 33.7 / 1.212 / 0.425; 3072: 45.9 / 34.7 / 1.276 / 0.513; 4096: 46.4 / 16.3 / 1.334 / 0.592; 5120: 46.0 / 18.7 / 1.395 / 0.672; 6144: 44.8 / 3.8 / 1.446 / 0.739; 7168: 45.4 / 17.0 / 1.495 / 0.801; 8192: 43.1 / 5.0 / 1.532 / 0.849
- **B free-wd0 116089**: 0: 90.8 / 36.3 / 1.000 / 0.000; 16: 54.4 / 41.0 / 1.003 / 0.072; 32: 54.6 / 44.5 / 1.008 / 0.089; 64: 58.1 / 45.5 / 1.014 / 0.099; 128: 58.1 / 44.7 / 1.020 / 0.108; 256: 52.9 / 44.2 / 1.042 / 0.155; 512: 50.5 / 41.5 / 1.075 / 0.210; 1024: 44.6 / 41.8 / 1.138 / 0.307; 2048: 42.7 / 42.5 / 1.238 / 0.447; 3072: 42.4 / 41.2 / 1.319 / 0.557; 4096: 43.4 / 42.8 / 1.383 / 0.641; 5120: 44.2 / 42.3 / 1.435 / 0.708; 6144: 42.2 / 38.3 / 1.497 / 0.789; 7168: 43.4 / 40.7 / 1.553 / 0.859; 8192: 43.9 / 43.8 / 1.569 / 0.876
- **B free-anchor 73129**: 0: 90.5 / 37.3 / 1.000 / 0.000; 16: 57.1 / 38.2 / 1.006 / 0.073; 32: 60.8 / 41.2 / 1.010 / 0.089; 64: 61.8 / 42.0 / 1.015 / 0.101; 128: 59.8 / 42.7 / 1.022 / 0.113; 256: 49.6 / 43.3 / 1.045 / 0.158; 512: 48.6 / 38.5 / 1.074 / 0.210; 1024: 46.3 / 38.2 / 1.120 / 0.289; 2048: 39.1 / 33.8 / 1.177 / 0.378; 3072: 46.8 / 39.0 / 1.211 / 0.431; 4096: 42.7 / 36.8 / 1.232 / 0.462; 5120: 42.2 / 39.5 / 1.241 / 0.479; 6144: 43.6 / 35.0 / 1.251 / 0.498; 7168: 44.4 / 34.2 / 1.255 / 0.508; 8192: 41.7 / 34.7 / 1.259 / 0.515
- **B free-anchor 93683**: 0: 96.8 / 26.7 / 1.000 / 0.000; 16: 65.8 / 30.0 / 1.004 / 0.080; 32: 67.9 / 35.0 / 1.007 / 0.095; 64: 70.2 / 36.3 / 1.011 / 0.104; 128: 68.7 / 30.8 / 1.017 / 0.113; 256: 59.6 / 31.2 / 1.036 / 0.148; 512: 58.2 / 30.2 / 1.065 / 0.200; 1024: 50.4 / 30.2 / 1.109 / 0.270; 2048: 49.2 / 35.5 / 1.160 / 0.347; 3072: 50.6 / 22.8 / 1.189 / 0.391; 4096: 50.1 / 21.7 / 1.210 / 0.419; 5120: 51.8 / 16.5 / 1.224 / 0.443; 6144: 51.6 / 22.8 / 1.228 / 0.450; 7168: 53.4 / 24.8 / 1.214 / 0.426; 8192: 55.9 / 26.3 / 1.180 / 0.360
- **B free-anchor 116089**: 0: 90.8 / 36.3 / 1.000 / 0.000; 16: 54.6 / 41.2 / 1.003 / 0.072; 32: 54.9 / 44.5 / 1.008 / 0.089; 64: 58.4 / 45.5 / 1.014 / 0.098; 128: 59.7 / 45.0 / 1.019 / 0.106; 256: 48.8 / 44.7 / 1.040 / 0.150; 512: 48.6 / 44.8 / 1.071 / 0.197; 1024: 48.2 / 45.3 / 1.119 / 0.274; 2048: 49.1 / 45.5 / 1.173 / 0.350; 3072: 46.8 / 43.7 / 1.215 / 0.413; 4096: 45.8 / 44.7 / 1.237 / 0.446; 5120: 46.6 / 44.8 / 1.253 / 0.466; 6144: 44.4 / 42.8 / 1.256 / 0.475; 7168: 47.2 / 46.0 / 1.231 / 0.436; 8192: 43.8 / 43.2 / 1.235 / 0.448

## 8. Committed files

This list is every file the branch adds, from `git diff --name-only --diff-filter=A 7cabb13a HEAD`, with sizes from `git ls-tree -r -l HEAD`. The commit that adds this report adds the report itself; the commit of the corrections adds `e58_report_appendix.py`. No existing file of the default branch is changed (`git diff --name-only --diff-filter=M 7cabb13a HEAD` is empty).

The committed files under `exp45/e58/` total 202.0 MB, not counting this report.

**Sources** — 4 files

- `E58_COPIED_SOURCES.txt` (801 bytes)
- `e48_gated.py` (7,186 bytes)
- `exp48_gated_run.py` (20,320 bytes)
- `exp58_new_family.py` (31,651 bytes)

**Receipts of runs (72)** — 72 files

- `exp45/e58/e58_free-again_B_f10_116089.json` (62,447 bytes)
- `exp45/e58/e58_free-again_B_f10_73129.json` (62,391 bytes)
- `exp45/e58/e58_free-again_B_f10_93683.json` (62,475 bytes)
- `exp45/e58/e58_free-again_S_f10_116089.json` (62,354 bytes)
- `exp45/e58/e58_free-again_S_f10_73129.json` (62,435 bytes)
- `exp45/e58/e58_free-again_S_f10_93683.json` (62,327 bytes)
- `exp45/e58/e58_free-anchor_B_f10_116089.json` (63,201 bytes)
- `exp45/e58/e58_free-anchor_B_f10_73129.json` (63,270 bytes)
- `exp45/e58/e58_free-anchor_B_f10_93683.json` (63,183 bytes)
- `exp45/e58/e58_free-anchor_S_f10_116089.json` (63,192 bytes)
- `exp45/e58/e58_free-anchor_S_f10_73129.json` (63,241 bytes)
- `exp45/e58/e58_free-anchor_S_f10_93683.json` (63,191 bytes)
- `exp45/e58/e58_free-wd0_B_f05_116089.json` (63,074 bytes)
- `exp45/e58/e58_free-wd0_B_f05_73129.json` (63,051 bytes)
- `exp45/e58/e58_free-wd0_B_f05_93683.json` (63,024 bytes)
- `exp45/e58/e58_free-wd0_B_f10_116089.json` (63,119 bytes)
- `exp45/e58/e58_free-wd0_B_f10_73129.json` (63,035 bytes)
- `exp45/e58/e58_free-wd0_B_f10_93683.json` (63,092 bytes)
- `exp45/e58/e58_free-wd0_B_f20_116089.json` (63,123 bytes)
- `exp45/e58/e58_free-wd0_B_f20_73129.json` (63,132 bytes)
- `exp45/e58/e58_free-wd0_B_f20_93683.json` (63,111 bytes)
- `exp45/e58/e58_free-wd0_S_f05_116089.json` (63,015 bytes)
- `exp45/e58/e58_free-wd0_S_f05_73129.json` (63,040 bytes)
- `exp45/e58/e58_free-wd0_S_f05_93683.json` (62,932 bytes)
- `exp45/e58/e58_free-wd0_S_f10_116089.json` (63,053 bytes)
- `exp45/e58/e58_free-wd0_S_f10_73129.json` (63,103 bytes)
- `exp45/e58/e58_free-wd0_S_f10_93683.json` (63,032 bytes)
- `exp45/e58/e58_free-wd0_S_f20_116089.json` (63,033 bytes)
- `exp45/e58/e58_free-wd0_S_f20_73129.json` (63,052 bytes)
- `exp45/e58/e58_free-wd0_S_f20_93683.json` (63,089 bytes)
- `exp45/e58/e58_free_B_f05_116089.json` (63,078 bytes)
- `exp45/e58/e58_free_B_f05_73129.json` (63,085 bytes)
- `exp45/e58/e58_free_B_f05_93683.json` (62,972 bytes)
- `exp45/e58/e58_free_B_f10_116089.json` (64,051 bytes)
- `exp45/e58/e58_free_B_f10_73129.json` (63,999 bytes)
- `exp45/e58/e58_free_B_f10_93683.json` (63,981 bytes)
- `exp45/e58/e58_free_B_f20_116089.json` (63,188 bytes)
- `exp45/e58/e58_free_B_f20_73129.json` (63,056 bytes)
- `exp45/e58/e58_free_B_f20_93683.json` (63,087 bytes)
- `exp45/e58/e58_free_S_f05_116089.json` (63,017 bytes)
- `exp45/e58/e58_free_S_f05_73129.json` (63,030 bytes)
- `exp45/e58/e58_free_S_f05_93683.json` (63,000 bytes)
- `exp45/e58/e58_free_S_f10_116089.json` (64,059 bytes)
- `exp45/e58/e58_free_S_f10_73129.json` (63,965 bytes)
- `exp45/e58/e58_free_S_f10_93683.json` (63,914 bytes)
- `exp45/e58/e58_free_S_f20_116089.json` (63,017 bytes)
- `exp45/e58/e58_free_S_f20_73129.json` (63,090 bytes)
- `exp45/e58/e58_free_S_f20_93683.json` (62,941 bytes)
- `exp45/e58/e58_term-wd0_B_f10_116089.json` (80,083 bytes)
- `exp45/e58/e58_term-wd0_B_f10_73129.json` (80,026 bytes)
- `exp45/e58/e58_term-wd0_B_f10_93683.json` (80,028 bytes)
- `exp45/e58/e58_term-wd0_S_f10_116089.json` (81,999 bytes)
- `exp45/e58/e58_term-wd0_S_f10_73129.json` (82,086 bytes)
- `exp45/e58/e58_term-wd0_S_f10_93683.json` (81,940 bytes)
- `exp45/e58/e58_term_B_f05_116089.json` (80,087 bytes)
- `exp45/e58/e58_term_B_f05_73129.json` (80,076 bytes)
- `exp45/e58/e58_term_B_f05_93683.json` (80,036 bytes)
- `exp45/e58/e58_term_B_f10_116089.json` (80,119 bytes)
- `exp45/e58/e58_term_B_f10_73129.json` (80,033 bytes)
- `exp45/e58/e58_term_B_f10_93683.json` (80,080 bytes)
- `exp45/e58/e58_term_B_f20_116089.json` (80,059 bytes)
- `exp45/e58/e58_term_B_f20_73129.json` (80,001 bytes)
- `exp45/e58/e58_term_B_f20_93683.json` (80,029 bytes)
- `exp45/e58/e58_term_S_f05_116089.json` (81,982 bytes)
- `exp45/e58/e58_term_S_f05_73129.json` (82,055 bytes)
- `exp45/e58/e58_term_S_f05_93683.json` (81,902 bytes)
- `exp45/e58/e58_term_S_f10_116089.json` (82,028 bytes)
- `exp45/e58/e58_term_S_f10_73129.json` (82,078 bytes)
- `exp45/e58/e58_term_S_f10_93683.json` (81,919 bytes)
- `exp45/e58/e58_term_S_f20_116089.json` (81,983 bytes)
- `exp45/e58/e58_term_S_f20_73129.json` (82,049 bytes)
- `exp45/e58/e58_term_S_f20_93683.json` (81,901 bytes)

**Weights at row 8,192 of `free`, `free-wd0`, `free-anchor` (42)** — 42 files

- `exp45/e58/e58_free-anchor_B_f10_116089_row8192.pt` (4,502,044 bytes)
- `exp45/e58/e58_free-anchor_B_f10_73129_row8192.pt` (4,502,025 bytes)
- `exp45/e58/e58_free-anchor_B_f10_93683_row8192.pt` (4,502,025 bytes)
- `exp45/e58/e58_free-anchor_S_f10_116089_row8192.pt` (4,502,044 bytes)
- `exp45/e58/e58_free-anchor_S_f10_73129_row8192.pt` (4,502,025 bytes)
- `exp45/e58/e58_free-anchor_S_f10_93683_row8192.pt` (4,502,025 bytes)
- `exp45/e58/e58_free-wd0_B_f05_116089_row8192.pt` (4,501,987 bytes)
- `exp45/e58/e58_free-wd0_B_f05_73129_row8192.pt` (4,501,968 bytes)
- `exp45/e58/e58_free-wd0_B_f05_93683_row8192.pt` (4,501,968 bytes)
- `exp45/e58/e58_free-wd0_B_f10_116089_row8192.pt` (4,501,987 bytes)
- `exp45/e58/e58_free-wd0_B_f10_73129_row8192.pt` (4,501,968 bytes)
- `exp45/e58/e58_free-wd0_B_f10_93683_row8192.pt` (4,501,968 bytes)
- `exp45/e58/e58_free-wd0_B_f20_116089_row8192.pt` (4,501,987 bytes)
- `exp45/e58/e58_free-wd0_B_f20_73129_row8192.pt` (4,501,968 bytes)
- `exp45/e58/e58_free-wd0_B_f20_93683_row8192.pt` (4,501,968 bytes)
- `exp45/e58/e58_free-wd0_S_f05_116089_row8192.pt` (4,501,987 bytes)
- `exp45/e58/e58_free-wd0_S_f05_73129_row8192.pt` (4,501,968 bytes)
- `exp45/e58/e58_free-wd0_S_f05_93683_row8192.pt` (4,501,968 bytes)
- `exp45/e58/e58_free-wd0_S_f10_116089_row8192.pt` (4,501,987 bytes)
- `exp45/e58/e58_free-wd0_S_f10_73129_row8192.pt` (4,501,968 bytes)
- `exp45/e58/e58_free-wd0_S_f10_93683_row8192.pt` (4,501,968 bytes)
- `exp45/e58/e58_free-wd0_S_f20_116089_row8192.pt` (4,501,987 bytes)
- `exp45/e58/e58_free-wd0_S_f20_73129_row8192.pt` (4,501,968 bytes)
- `exp45/e58/e58_free-wd0_S_f20_93683_row8192.pt` (4,501,968 bytes)
- `exp45/e58/e58_free_B_f05_116089_row8192.pt` (4,501,911 bytes)
- `exp45/e58/e58_free_B_f05_73129_row8192.pt` (4,501,892 bytes)
- `exp45/e58/e58_free_B_f05_93683_row8192.pt` (4,501,892 bytes)
- `exp45/e58/e58_free_B_f10_116089_row8192.pt` (4,501,911 bytes)
- `exp45/e58/e58_free_B_f10_73129_row8192.pt` (4,501,892 bytes)
- `exp45/e58/e58_free_B_f10_93683_row8192.pt` (4,501,892 bytes)
- `exp45/e58/e58_free_B_f20_116089_row8192.pt` (4,501,911 bytes)
- `exp45/e58/e58_free_B_f20_73129_row8192.pt` (4,501,892 bytes)
- `exp45/e58/e58_free_B_f20_93683_row8192.pt` (4,501,892 bytes)
- `exp45/e58/e58_free_S_f05_116089_row8192.pt` (4,501,911 bytes)
- `exp45/e58/e58_free_S_f05_73129_row8192.pt` (4,501,892 bytes)
- `exp45/e58/e58_free_S_f05_93683_row8192.pt` (4,501,892 bytes)
- `exp45/e58/e58_free_S_f10_116089_row8192.pt` (4,501,911 bytes)
- `exp45/e58/e58_free_S_f10_73129_row8192.pt` (4,501,892 bytes)
- `exp45/e58/e58_free_S_f10_93683_row8192.pt` (4,501,892 bytes)
- `exp45/e58/e58_free_S_f20_116089_row8192.pt` (4,501,911 bytes)
- `exp45/e58/e58_free_S_f20_73129_row8192.pt` (4,501,892 bytes)
- `exp45/e58/e58_free_S_f20_93683_row8192.pt` (4,501,892 bytes)

**Term files at rows 1,024 and 8,192 (48)** — 48 files

- `exp45/e58/e58_term-wd0_B_f10_116089_row1024.pt` (150,616 bytes)
- `exp45/e58/e58_term-wd0_B_f10_116089_row8192.pt` (150,616 bytes)
- `exp45/e58/e58_term-wd0_B_f10_73129_row1024.pt` (150,605 bytes)
- `exp45/e58/e58_term-wd0_B_f10_73129_row8192.pt` (150,605 bytes)
- `exp45/e58/e58_term-wd0_B_f10_93683_row1024.pt` (150,605 bytes)
- `exp45/e58/e58_term-wd0_B_f10_93683_row8192.pt` (150,605 bytes)
- `exp45/e58/e58_term-wd0_S_f10_116089_row1024.pt` (150,616 bytes)
- `exp45/e58/e58_term-wd0_S_f10_116089_row8192.pt` (150,616 bytes)
- `exp45/e58/e58_term-wd0_S_f10_73129_row1024.pt` (150,605 bytes)
- `exp45/e58/e58_term-wd0_S_f10_73129_row8192.pt` (150,605 bytes)
- `exp45/e58/e58_term-wd0_S_f10_93683_row1024.pt` (150,605 bytes)
- `exp45/e58/e58_term-wd0_S_f10_93683_row8192.pt` (150,605 bytes)
- `exp45/e58/e58_term_B_f05_116089_row1024.pt` (150,572 bytes)
- `exp45/e58/e58_term_B_f05_116089_row8192.pt` (150,572 bytes)
- `exp45/e58/e58_term_B_f05_73129_row1024.pt` (150,561 bytes)
- `exp45/e58/e58_term_B_f05_73129_row8192.pt` (150,561 bytes)
- `exp45/e58/e58_term_B_f05_93683_row1024.pt` (150,561 bytes)
- `exp45/e58/e58_term_B_f05_93683_row8192.pt` (150,561 bytes)
- `exp45/e58/e58_term_B_f10_116089_row1024.pt` (150,572 bytes)
- `exp45/e58/e58_term_B_f10_116089_row8192.pt` (150,572 bytes)
- `exp45/e58/e58_term_B_f10_73129_row1024.pt` (150,561 bytes)
- `exp45/e58/e58_term_B_f10_73129_row8192.pt` (150,561 bytes)
- `exp45/e58/e58_term_B_f10_93683_row1024.pt` (150,561 bytes)
- `exp45/e58/e58_term_B_f10_93683_row8192.pt` (150,561 bytes)
- `exp45/e58/e58_term_B_f20_116089_row1024.pt` (150,572 bytes)
- `exp45/e58/e58_term_B_f20_116089_row8192.pt` (150,572 bytes)
- `exp45/e58/e58_term_B_f20_73129_row1024.pt` (150,561 bytes)
- `exp45/e58/e58_term_B_f20_73129_row8192.pt` (150,561 bytes)
- `exp45/e58/e58_term_B_f20_93683_row1024.pt` (150,561 bytes)
- `exp45/e58/e58_term_B_f20_93683_row8192.pt` (150,561 bytes)
- `exp45/e58/e58_term_S_f05_116089_row1024.pt` (150,572 bytes)
- `exp45/e58/e58_term_S_f05_116089_row8192.pt` (150,572 bytes)
- `exp45/e58/e58_term_S_f05_73129_row1024.pt` (150,561 bytes)
- `exp45/e58/e58_term_S_f05_73129_row8192.pt` (150,561 bytes)
- `exp45/e58/e58_term_S_f05_93683_row1024.pt` (150,561 bytes)
- `exp45/e58/e58_term_S_f05_93683_row8192.pt` (150,561 bytes)
- `exp45/e58/e58_term_S_f10_116089_row1024.pt` (150,572 bytes)
- `exp45/e58/e58_term_S_f10_116089_row8192.pt` (150,572 bytes)
- `exp45/e58/e58_term_S_f10_73129_row1024.pt` (150,561 bytes)
- `exp45/e58/e58_term_S_f10_73129_row8192.pt` (150,561 bytes)
- `exp45/e58/e58_term_S_f10_93683_row1024.pt` (150,561 bytes)
- `exp45/e58/e58_term_S_f10_93683_row8192.pt` (150,561 bytes)
- `exp45/e58/e58_term_S_f20_116089_row1024.pt` (150,572 bytes)
- `exp45/e58/e58_term_S_f20_116089_row8192.pt` (150,572 bytes)
- `exp45/e58/e58_term_S_f20_73129_row1024.pt` (150,561 bytes)
- `exp45/e58/e58_term_S_f20_73129_row8192.pt` (150,561 bytes)
- `exp45/e58/e58_term_S_f20_93683_row1024.pt` (150,561 bytes)
- `exp45/e58/e58_term_S_f20_93683_row8192.pt` (150,561 bytes)

**Checks and printed tables (exp45/e58/qual/)** — 18 files

- `exp45/e58/qual/E58_NORMS_PART_A.txt` (2,184 bytes)
- `exp45/e58/qual/E58_PART_A_SEED_73129_TABLE.txt` (8,237 bytes)
- `exp45/e58/qual/E58_PART_A_TABLE.txt` (22,438 bytes)
- `exp45/e58/qual/E58_PART_B_SEED_73129_TABLE.txt` (7,555 bytes)
- `exp45/e58/qual/E58_PART_B_TABLE.txt` (20,775 bytes)
- `exp45/e58/qual/E58_check1.json` (5,128 bytes)
- `exp45/e58/qual/E58_check2.json` (5,441 bytes)
- `exp45/e58/qual/E58_check3_B.json` (5,682 bytes)
- `exp45/e58/qual/E58_check3_S.json` (5,685 bytes)
- `exp45/e58/qual/E58_check4_B.json` (5,330 bytes)
- `exp45/e58/qual/E58_check4_S.json` (5,336 bytes)
- `exp45/e58/qual/E58_check5.json` (20,148 bytes)
- `exp45/e58/qual/E58_check6.json` (6,635 bytes)
- `exp45/e58/qual/E58_check6_part_a.json` (4,562 bytes)
- `exp45/e58/qual/E58_check6_seed73129_only.json` (3,210 bytes)
- `exp45/e58/qual/E58_check7.json` (5,644 bytes)
- `exp45/e58/qual/E58_check8_B.json` (4,025 bytes)
- `exp45/e58/qual/E58_check8_S.json` (4,020 bytes)

**Logs** — 73 files

- `exp45/e58/logs/e58_free-again_B_f10_116089.log` (1,300 bytes)
- `exp45/e58/logs/e58_free-again_B_f10_73129.log` (1,280 bytes)
- `exp45/e58/logs/e58_free-again_B_f10_93683.log` (1,281 bytes)
- `exp45/e58/logs/e58_free-again_S_f10_116089.log` (1,297 bytes)
- `exp45/e58/logs/e58_free-again_S_f10_73129.log` (1,281 bytes)
- `exp45/e58/logs/e58_free-again_S_f10_93683.log` (1,284 bytes)
- `exp45/e58/logs/e58_free-anchor_B_f10_116089.log` (1,317 bytes)
- `exp45/e58/logs/e58_free-anchor_B_f10_73129.log` (1,297 bytes)
- `exp45/e58/logs/e58_free-anchor_B_f10_93683.log` (1,300 bytes)
- `exp45/e58/logs/e58_free-anchor_S_f10_116089.log` (1,313 bytes)
- `exp45/e58/logs/e58_free-anchor_S_f10_73129.log` (1,298 bytes)
- `exp45/e58/logs/e58_free-anchor_S_f10_93683.log` (1,298 bytes)
- `exp45/e58/logs/e58_free-wd0_B_f05_116089.log` (1,272 bytes)
- `exp45/e58/logs/e58_free-wd0_B_f05_73129.log` (1,253 bytes)
- `exp45/e58/logs/e58_free-wd0_B_f05_93683.log` (1,255 bytes)
- `exp45/e58/logs/e58_free-wd0_B_f10_116089.log` (1,270 bytes)
- `exp45/e58/logs/e58_free-wd0_B_f10_73129.log` (1,249 bytes)
- `exp45/e58/logs/e58_free-wd0_B_f10_93683.log` (1,250 bytes)
- `exp45/e58/logs/e58_free-wd0_B_f20_116089.log` (1,266 bytes)
- `exp45/e58/logs/e58_free-wd0_B_f20_73129.log` (1,249 bytes)
- `exp45/e58/logs/e58_free-wd0_B_f20_93683.log` (1,248 bytes)
- `exp45/e58/logs/e58_free-wd0_S_f05_116089.log` (1,270 bytes)
- `exp45/e58/logs/e58_free-wd0_S_f05_73129.log` (1,257 bytes)
- `exp45/e58/logs/e58_free-wd0_S_f05_93683.log` (1,254 bytes)
- `exp45/e58/logs/e58_free-wd0_S_f10_116089.log` (1,267 bytes)
- `exp45/e58/logs/e58_free-wd0_S_f10_73129.log` (1,249 bytes)
- `exp45/e58/logs/e58_free-wd0_S_f10_93683.log` (1,250 bytes)
- `exp45/e58/logs/e58_free-wd0_S_f20_116089.log` (1,264 bytes)
- `exp45/e58/logs/e58_free-wd0_S_f20_73129.log` (1,250 bytes)
- `exp45/e58/logs/e58_free-wd0_S_f20_93683.log` (1,248 bytes)
- `exp45/e58/logs/e58_free_B_f05_116089.log` (1,206 bytes)
- `exp45/e58/logs/e58_free_B_f05_73129.log` (1,187 bytes)
- `exp45/e58/logs/e58_free_B_f05_93683.log` (1,188 bytes)
- `exp45/e58/logs/e58_free_B_f10_116089.log` (1,203 bytes)
- `exp45/e58/logs/e58_free_B_f10_73129.log` (1,182 bytes)
- `exp45/e58/logs/e58_free_B_f10_93683.log` (1,186 bytes)
- `exp45/e58/logs/e58_free_B_f20_116089.log` (1,202 bytes)
- `exp45/e58/logs/e58_free_B_f20_73129.log` (1,185 bytes)
- `exp45/e58/logs/e58_free_B_f20_93683.log` (1,185 bytes)
- `exp45/e58/logs/e58_free_S_f05_116089.log` (1,205 bytes)
- `exp45/e58/logs/e58_free_S_f05_73129.log` (1,186 bytes)
- `exp45/e58/logs/e58_free_S_f05_93683.log` (1,186 bytes)
- `exp45/e58/logs/e58_free_S_f10_116089.log` (1,203 bytes)
- `exp45/e58/logs/e58_free_S_f10_73129.log` (1,186 bytes)
- `exp45/e58/logs/e58_free_S_f10_93683.log` (1,188 bytes)
- `exp45/e58/logs/e58_free_S_f20_116089.log` (1,201 bytes)
- `exp45/e58/logs/e58_free_S_f20_73129.log` (1,185 bytes)
- `exp45/e58/logs/e58_free_S_f20_93683.log` (1,184 bytes)
- `exp45/e58/logs/e58_term-wd0_B_f10_116089.log` (1,275 bytes)
- `exp45/e58/logs/e58_term-wd0_B_f10_73129.log` (1,253 bytes)
- `exp45/e58/logs/e58_term-wd0_B_f10_93683.log` (1,258 bytes)
- `exp45/e58/logs/e58_term-wd0_S_f10_116089.log` (1,268 bytes)
- `exp45/e58/logs/e58_term-wd0_S_f10_73129.log` (1,256 bytes)
- `exp45/e58/logs/e58_term-wd0_S_f10_93683.log` (1,255 bytes)
- `exp45/e58/logs/e58_term_B_f05_116089.log` (1,206 bytes)
- `exp45/e58/logs/e58_term_B_f05_73129.log` (1,193 bytes)
- `exp45/e58/logs/e58_term_B_f05_93683.log` (1,191 bytes)
- `exp45/e58/logs/e58_term_B_f10_116089.log` (1,210 bytes)
- `exp45/e58/logs/e58_term_B_f10_73129.log` (1,187 bytes)
- `exp45/e58/logs/e58_term_B_f10_93683.log` (1,194 bytes)
- `exp45/e58/logs/e58_term_B_f20_116089.log` (1,215 bytes)
- `exp45/e58/logs/e58_term_B_f20_73129.log` (1,190 bytes)
- `exp45/e58/logs/e58_term_B_f20_93683.log` (1,188 bytes)
- `exp45/e58/logs/e58_term_S_f05_116089.log` (1,207 bytes)
- `exp45/e58/logs/e58_term_S_f05_73129.log` (1,194 bytes)
- `exp45/e58/logs/e58_term_S_f05_93683.log` (1,192 bytes)
- `exp45/e58/logs/e58_term_S_f10_116089.log` (1,201 bytes)
- `exp45/e58/logs/e58_term_S_f10_73129.log` (1,189 bytes)
- `exp45/e58/logs/e58_term_S_f10_93683.log` (1,193 bytes)
- `exp45/e58/logs/e58_term_S_f20_116089.log` (1,202 bytes)
- `exp45/e58/logs/e58_term_S_f20_73129.log` (1,190 bytes)
- `exp45/e58/logs/e58_term_S_f20_93683.log` (1,184 bytes)
- `exp45/e58/logs/scheduler.log` (0 bytes)

**Interrupted runs (exp45/e58/interrupted/)** — 13 files

- `exp45/e58/interrupted/e58_free-again_B_f10_73129.json.20261118T001933` (27,250 bytes)
- `exp45/e58/interrupted/e58_free-again_B_f10_73129.log.20261118T001933` (486 bytes)
- `exp45/e58/interrupted/e58_free-again_S_f10_73129.json.20261118T001933` (35,062 bytes)
- `exp45/e58/interrupted/e58_free-again_S_f10_73129.log.20261118T001933` (653 bytes)
- `exp45/e58/interrupted/e58_free-wd0_S_f05_93683.json.20261118T060744` (43,043 bytes)
- `exp45/e58/interrupted/e58_free-wd0_S_f05_93683.log.20261118T060744` (804 bytes)
- `exp45/e58/interrupted/e58_free_B_f05_93683.json.20261118T060744` (47,007 bytes)
- `exp45/e58/interrupted/e58_free_B_f05_93683.log.20261118T060744` (842 bytes)
- `exp45/e58/interrupted/e58_free_S_f05_93683.json.20261118T060744` (43,163 bytes)
- `exp45/e58/interrupted/e58_free_S_f05_93683.log.20261118T060744` (764 bytes)
- `exp45/e58/interrupted/e58_term-wd0_S_f10_73129.json.20261118T001933` (76,564 bytes)
- `exp45/e58/interrupted/e58_term-wd0_S_f10_73129.log.20261118T001933` (1,136 bytes)
- `exp45/e58/interrupted/e58_term-wd0_S_f10_73129_row1024.pt.20261118T001933` (150,605 bytes)

**Queues, scheduler, report** — 9 files

- `exp45/e58/E58_REPORT.md`
- `exp45/e58/e58_report_appendix.py` (3,935 bytes; added by the commit of the corrections)
- `exp45/e58/e58_norms.py` (875 bytes)
- `exp45/e58/e58_tables.py` (11,676 bytes)
- `exp45/e58/queues/a1_73129_first.txt` (104 bytes)
- `exp45/e58/queues/a2.txt` (480 bytes)
- `exp45/e58/queues/all.txt` (1,266 bytes)
- `exp45/e58/run_queue.sh` (418 bytes)
- `exp45/e58/scheduler.py` (2,650 bytes)

