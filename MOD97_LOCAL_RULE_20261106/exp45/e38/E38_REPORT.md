# E38 stage 1 report: what erases the birth family, and does an anchor protect it? (seed 73129)

Brief: `OPUS_TASK_E38_WHAT_ERASES_THE_BIRTH_FAMILY_20261115.md`, revision 2, at 18751457 on `codex/minifly-consolidated-20261023`. Work branch `claude/peaceful-hopper-8ofq4i`; the runner `exp38_erase.py` was committed at 7946c68f before any receipt was written, and every life's receipt records its sources' sha256 unmodified against that commit. Evidence level E2, one seed: stage 1 supports no claim by itself.

Code: `exp38_erase.py` (runner: the life loop of `exp26_candidate.life` lines 196 to 279 copied over `X16.SCHEDULE[:2]`, the step of `exp26_candidate.step` copied with the pull between backward and `opt.step()`; nothing existing edited, called or patched beyond what the brief names), `e38_read.py` (reader), `e38_check2_diag.py` (a diagnostic for checks 2 and 3, not one of the brief's checks). Host: Linux x86-64, 4 cores, Python 3.13.18, torch 2.15.0+cpu (installed in this session: the container had no torch), one thread per process, at most three processes, launched with `setsid nohup` outside any time-limited shell.

**Admission exception (recorded, as E31 did).** `exp26_candidate.admit_e26` refuses both births because `exp16_integrated.py` is now at deea398c (sha c9fcedcf…). The births and E26's lives were made with a301ae8e (f4559971…). deea398c only adds E21b's resume and receipt functions, which `exp26_candidate.life`'s loop does not call. `exp38_erase.admit` follows `exp31_switch.admit_s`: it admits only that one file with that exact pair of hashes, and runs every other check of `admit_e26` unchanged. Check 1 then tests the outcome bit for bit.

## Checks

| # | Check | S (sweep) | B (backprop) | Receipt |
|---|---|---|---|---|
| 1 | `zero` repeats E26: at rows 6,144 and 12,288 the batch digest, last-512 loss, every `evaluate()` score and the exposure counts equal `stages[1]`, `stages[2]` | **pass** | **pass** | `e38_check1_<rule>_73129.json` |
| 2 | anchor-to-zero step against torch's own decay, 64 updates, tolerance 1e-5 of each tensor's largest entry, at κ 0.25 and 2.5 | **FAIL** (largest 5.2e-4, l1.bias) | pass | `e38_check2_<rule>_73129.json` |
| 3 | two forms of the anchor step, 64 updates, 1e-6 relative | **FAIL** (κ 0.25: 4.2e-5; κ 2.5: 7.8e-3) | pass | `e38_check3_<rule>_73129.json` |
| 4 | stored birth values: 1,122,049 in 12 tensors, bit-equal to the birth file, not aliased, every decayed tensor moved, `anchor` ≠ `none` after 64 updates; and again at the end of each anchor life | pass (64 and 12,288) | pass (64 and 12,288) | `e38_check4_*`, `birth_checks` in each anchor receipt |
| 5 | `none`'s optimizer groups read back with weight_decay 0.0, 0.0 | pass | pass | `e38_check5_*` |
| 6 | no presented single-operation item is a held-out pair of its family (birth family and F1, every arm); per-block counts in `rehearse` | pass (0 held pairs in all ten lives; composition-panel assertion also held) | pass | `held_pairs_presented` in each receipt |
| 7 | one dry life of 256 updates per arm under the sweep, CPU time and forecast | 77–88 s each (`dry/`); forecast about 1,600–1,750 s per life with scoring, 4.5 CPU-h for ten | — | `dry/*.json` |

**Checks 2 and 3 fail under the sweep as specified.** I report them as failed and do not reinterpret them as passes. A diagnostic separates arithmetic from rounding (`e38_check2_diag_S_73129_float{32,64}.json`). With every tensor in float64, the two pairs agree to 2e-16 to 6e-15 at every update to 64, for both strengths. That is rounding level, so the pull is the same arithmetic as torch's decay, and its two forms are the same arithmetic as each other. In float32 the largest difference is 1e-7 after one update and stays near 1e-6. It then jumps by two to four orders of magnitude at single updates: for example, from 1.5e-6 at update 32 to 5.2e-4 at 64 for κ 0.25, and to 5.9e-4 at update 8 for the two forms at κ 2.5. The undecayed embedding table also differs (1e-7). Under backprop the same comparisons pass. The pattern fits rounding-level differences tipped into discrete flips by the sweep's sign-based way back. It does not fit a wrong step. The brief offers checks 2 to 4 in place of a second worker's implementation. Under the sweep they stand only with this diagnostic, which the brief did not specify; a reviewer should decide whether that suffices.

Check 1 passed, so the stop condition (reading 0) does not apply.

## Lives

Ten lives: five arms × two rules, seed 73129, 12,288 updates each (F1.1 and F1.2), 17 rows per life. Weights at rows 6,144 and 12,288 are committed (20 files). The other 15 rows per life are in `rows_uncommitted/` (not pushed); their sha256 are in each receipt's `rows`. CPU per life was 1,497 to 1,755 s, of which scoring took 249 to 284 s. Total 16,328 CPU-s (4.5 CPU-h), against the reviewer's estimate of about 6.

Recorded per arm: `decay_target`, `decay_strength`, the groups read back, and the stored birth count. `zero`/`rehearse`: zero, 0.25, groups (0.25, 0.0). `none`: none, 0, (0.0, 0.0). `anchor`: birth, 0.25, (0.0, 0.0), 1,122,049. `anchor10`: birth, 2.5, (0.0, 0.0), 1,122,049.

`rehearse` item counts per block (birth family / F1, of 393,216): block 1 196,298 / 196,918, block 2 196,376 / 196,840. These match the audit's rebuild.

### Curves (held-out add/sub mod 89 at k = 12, mean / wrap pairs of score 2, mean of add and sub / new family add/sub mod 97 at k = 12, mean)

Every row and every column (each task separately, max89, min89, no-wrap pairs) is in `E38_READ_TABLES.md`, and the numbers in `E38_READ.json`. A selection:

| arm, rule | 0 | 64 | 128 | 256 | 512 | 1,024 | 2,048 | 4,096 | 6,144 | 8,192 | 12,288 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| zero S | 98.0/96.6/23.0 | 46.8/17.8/62.5 | 45.0/7.2/77.1 | 42.0/3.2/82.6 | 46.8/1.1/89.2 | 44.7/0.5/89.6 | 46.8/0.1/92.1 | 46.3/0.1/93.0 | 49.0/0.3/92.5 | 47.3/0.0/92.1 | 47.8/0.0/92.5 |
| none S | 98.0/96.6/23.0 | 46.8/17.6/62.9 | 45.0/7.4/77.5 | 44.0/3.0/85.6 | 46.5/0.6/90.4 | 47.5/0.6/92.6 | 47.5/0.3/90.8 | 48.3/0.1/93.9 | 48.8/0.2/95.2 | 48.2/0.1/95.2 | 49.0/0.1/96.2 |
| anchor S | 98.0/96.6/23.0 | 46.7/17.7/62.9 | 45.0/7.6/77.4 | 44.2/3.3/84.9 | 46.2/0.7/92.0 | 45.8/0.2/90.5 | 48.0/0.5/94.0 | 49.0/0.2/95.4 | 47.8/0.1/94.8 | 47.7/0.2/95.6 | 48.7/0.1/97.2 |
| anchor10 S | 98.0/96.6/23.0 | 46.5/19.4/63.0 | 45.8/9.3/76.8 | 44.8/5.0/82.5 | 46.2/2.2/88.1 | 46.7/1.5/90.0 | 43.7/1.2/84.9 | 47.0/0.3/89.5 | 48.0/0.8/90.5 | 46.2/0.5/91.9 | 46.8/0.2/91.0 |
| rehearse S | 98.0/96.6/23.0 | 72.5/65.0/40.9 | 77.8/69.0/54.5 | 85.0/75.6/61.1 | 91.2/86.5/80.0 | 84.7/83.2/76.1 | 95.8/93.5/89.1 | 93.8/92.9/86.1 | 96.8/94.6/88.0 | 97.0/94.0/91.1 | 98.3/96.8/92.4 |
| zero B | 98.5/98.1/6.2 | 33.5/10.1/52.6 | 37.7/3.1/73.8 | 38.2/0.3/79.5 | 42.7/0.2/87.9 | 45.0/0.0/90.1 | 42.2/0.1/90.9 | 44.3/0.0/92.4 | 44.2/0.1/91.6 | 45.0/0.1/93.1 | 47.5/0.0/94.8 |
| none B | 98.5/98.1/6.2 | 33.7/10.1/53.0 | 37.2/2.9/73.2 | 40.3/0.2/80.0 | 46.0/0.0/91.4 | 46.7/0.1/90.8 | 46.5/0.0/93.8 | 47.8/0.0/94.2 | 48.2/0.0/95.0 | 49.0/0.0/96.0 | 44.8/0.0/96.6 |
| anchor B | 98.5/98.1/6.2 | 33.8/10.2/53.2 | 37.3/3.1/73.4 | 40.2/0.3/80.4 | 48.2/0.1/92.0 | 45.8/0.1/89.4 | 47.7/0.0/94.2 | 48.3/0.0/96.6 | 48.5/0.0/93.2 | 47.2/0.0/90.9 | 49.2/0.0/92.8 |
| anchor10 B | 98.5/98.1/6.2 | 33.5/11.5/54.1 | 38.3/3.8/72.2 | 38.7/0.4/79.9 | 43.3/0.3/89.5 | 47.7/0.0/92.2 | 45.3/0.0/90.8 | 47.3/0.1/89.6 | 49.3/0.0/91.0 | 48.3/0.0/88.9 | 44.7/0.1/85.6 |
| rehearse B | 98.5/98.1/6.2 | 72.3/69.8/33.1 | 86.3/81.7/46.5 | 94.7/90.6/66.8 | 92.8/90.1/76.2 | 96.2/93.1/81.5 | 90.0/91.2/84.8 | 96.0/92.4/91.5 | 98.2/97.9/93.0 | 97.5/96.1/93.0 | 96.0/93.9/92.9 |

### Marks at row 12,288

| arm | S: held / wrap / new | S marks | B: held / wrap / new | B marks |
|---|---|---|---|---|
| zero | 47.8 / 0.04 / 92.5 | loses; acquires | 47.5 / 0.00 / 94.8 | loses; acquires |
| none | 49.0 / 0.05 / 96.3 | loses; acquires | 44.8 / 0.00 / 96.6 | loses; acquires |
| anchor | 48.7 / 0.09 / 97.3 | loses; acquires | 49.2 / 0.00 / 92.8 | loses; acquires |
| anchor10 | 46.8 / 0.18 / 91.0 | loses; acquires | 44.7 / 0.05 / 85.6 | loses; acquires |
| rehearse | 98.3 / 96.8 / 92.4 | **keeps; acquires** | 96.0 / 93.9 / 92.9 | **keeps; acquires** |

No-wrap pairs at 12,288 are 88 to 98 in every arm, so the loss is concentrated in the wrap pairs, as the audit found.

## Readings (the brief's table only)

`zero` loses the family by the mark under both rules, so readings 1 to 5 are read for both.

| rule | none (rows 1, 2) | anchors (rows 3 to 5, per strength) | rehearse (row 6) |
|---|---|---|---|
| S | **2**: `none` loses. The new family's own updates erase it; decay is not needed for the loss. | `anchor` loses, `anchor10` loses: neither keeps, so **5**: a uniform pull is not enough on this seed. Rows 3 and 4 do not apply at either strength. | keeps (98.3 / 96.8) and acquires (92.4) |
| B | **2** | **5** (both strengths lose) | keeps (96.0 / 93.9) and acquires (92.9) |

**Rehearsal's cost to the new family** (row 6, reported as facts):
- **Per update** (row 12,288 against `zero`'s row 12,288): S −0.1 points, B −1.9.
- **Per F1 item** (row 12,288, 393,758 F1 items, against `zero`'s row 6,144, 393,216): S −0.1, B +1.3.

### Facts beside the readings

- **First row where held-out mod-89 add/sub falls below 80:** 64, in all ten lives. `rehearse` recovers above 80 by row 256 (S) and 128 (B), then dips at single rows (S 84.7 at 1,024 and B 90.0 at 2,048 stay above 80).
- **First row where wrap pairs fall below 50:** 64 in every non-rehearse life. They are 10 to 19 at row 64 and at most 5 by row 256. They never fall below 50 in `rehearse`.
- **Timing.** The loss is complete within a few hundred updates in all four decay variants. At decay 0.25 the weights keep 98.4% of their size after 64 updates (0.99975^64), so the speed alone separates this loss from decay acting over thousands of updates. That is a fact about timing, not a further reading.
- **Paired difference S − B at 12,288** (held / wrap / new): zero +0.3 / +0.0 / −2.3; none +4.2 / +0.1 / −0.4; anchor −0.5 / +0.1 / +4.5; anchor10 +2.2 / +0.1 / +5.4; rehearse +2.3 / +2.9 / −0.5.
- **`min89` under backprop.** Not in the marks, but large: `zero` B 10.7 and `none` B 21.7 at 12,288, while the anchors (93.7, 98.3) and every sweep life stay above 94.

## Predictions

| | Prediction | Chance | Outcome |
|---|---|---|---|
| P38.1 | Check 1 passes for both rules | 90% | **yes** |
| P38.2 | `none` keeps the birth family (sweep) | 15% | no |
| P38.3 | `anchor` keeps it (sweep) | 25% | no |
| P38.4 | `anchor10` keeps it (sweep) | 55% | no |
| P38.5 | `anchor10` acquires the new family (sweep) | 55% | **yes** (91.0) |
| P38.6 | Some anchor keeps and acquires (sweep) | 35% | no |
| P38.7 | `rehearse` keeps and acquires (sweep) | 70% | **yes** |
| P38.8 | A reading that differs between the sweep and backpropagation | 15% | no (2, 5, 6 under both) |

## Limits

- One seed. Seed 73129 is the one whose E26 lives lost least on the held-out panel in block 1. Stage 2 is the brief author's to release; the brief opens it only for arms that keep and acquire, which here is `rehearse` alone.
- Checks 2 and 3 failed as specified under the sweep. The float64 diagnostic indicates rounding, not a wrong step, but it is my own diagnostic, not the brief's, and no second worker has reviewed it.
- The admission exception for `exp16_integrated.py` is recorded above. Check 1's bit-for-bit pass is the test that it changed nothing.
- No independent review of this run has taken place. Nothing here is confirmed, sealed or adopted.
