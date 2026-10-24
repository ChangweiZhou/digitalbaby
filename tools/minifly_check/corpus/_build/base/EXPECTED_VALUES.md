# Corpus base: hand-computed values

Delivery `minifly_corpus_base_dev_v1_20261101` (T2, MAIN, E3 target, `addition_single_digit`,
`pair_rectangle` witness, commutative). Computed by hand from SPEC Sections 9.1-9.6 without the
checker; `refcalc.py` / `scores_doc.py` repeat the same arithmetic as a cross-check and wrote the
numbers below. Fixture C00a is this base; its EXPECTED.json entry states the verdict and the headline.

## 1. Items

Prompt `a+b=` (hex of ASCII), answer decimal, EOS `0a`, `gen_cap_bytes` 4. Every stream is the
emitted answer followed by EOS bytes up to 4 bytes (`5` -> `350a0a0a`), so `own_eos` is true unless a
fixture fills all 4 bytes.

Taught units (TAUGHT manifest of W, every world, 10 presentations each): the cross through 5,
`(a,5)` for a = 1..8 and `(5,b)` for b = 1..8, b != 5 (15 items). Answers: 6..13 on the column,
6..9 and 11..13 on the row.

| Pool | Items | n_items_per_pool |
|---|---|---|
| TAUGHT | the 15 taught items (t01..t15) | 15 |
| OLD | (1,5) (2,5) (3,5) (4,5) (5,5) (6,5) (7,5) (8,5) (o01..o08) | 8 |
| NEW | (5,1) (5,2) (5,3) (5,4) (5,6) (5,7) (5,8) (n01..n07) | 7 |
| HELD | S2: (1,2) (2,1) (2,2) (1,4) (3,2) (2,4) (4,2) (3,4); S3: (2,8) (3,7) (6,4) (4,8) (8,4) (6,7) (7,7) (8,8) | 16 |

Cells per (world, native arm): TAUGHT@prefix, OLD@old_end, OLD@final, NEW@final, HELD@final
(checkpoints prefix 40, old_end 80, final 160). Testbed arms (CEILING_*, FLOOR_*) on the 8
technical worlds: HELD@final, OLD@old_end, OLD@final, answer_path `evaluator`.

## 2. Strata and determinacy (SPEC 9.3, 9.4)

- S0: no operand is 0 and no answer equals an operand -> no S0 item.
- S1: every taught orbit contains 5; no held item contains 5 -> no S1 item.
- S4: every value 1..8 occurs as a first operand ((a,5)) and as a second operand ((5,b)); the
  longest taught answer has 2 bytes and so does the longest held answer (16) -> no S4 item.
- S2 / S3: the 8 items with a+b <= 9 are S2 (no carry), the 8 with a+b >= 10 are S3 (carry).
- Determinacy under pair_rectangle: for held (a,b) take a' = 5, b' = 5: (a,5), (5,b) and (5,5) are
  taught, so all 16 held items are determined. Counted = 16 per world (strata_counted S2, S3; 8 each,
  meeting min_items_per_stratum 8). Counted items over 8 MAIN worlds: n = 128.

## 3. Shortcut ladder on HELD (SPEC 9.2), every world

- CONST: taught answer counts are 2 for 6, 7, 8, 9, 11, 12, 13 and 1 for 10; the tie goes to the
  lexicographically smallest answer, `11` (0x3131 < 0x36). No held answer is 11 -> 0. (Numeric
  tie-breaking would give `6`, 2/16; the best member is unchanged.)
- LENFREQ, SUFFIX1 (all prompts have length 4 and end in `=`), EXACT, CANON (no held prompt or orbit
  is taught), ID_ZERO (no zero operand): fall back to CONST -> 0.
- SUFFIX2/3 (`b=` / `+b=`): only (5,b) shares them, predicting 5+b != a+b -> 0.
- NEAREST: the best positional matches (3 of 4 bytes) are (a,5) and (5,b), predicting a+5 or 5+b -> 0
  whichever tie rule is used.
- COPY_A, COPY_B -> 0 (no zero operand).
- DIGIT_NO_CARRY: right on every S2 item, wrong on every S3 item -> **8/16 = 0.5 in every world**.
- FIXED_1 of W's own stream: right only on S2 items W answers correctly; W is wrong on at least one
  S2 item per world -> at most 7/16. FIXED_2: right only on S3 items W answers correctly; W is wrong
  on at least one S3 item per world -> at most 7/16. FIXED_0, FIXED_3..8 -> 0.
- MODE_REPEAT: W's modal byte `1` repeated to its modal length (1 or 2 by tie) -> `1` or `11` -> 0.

Best shortcut: **DIGIT_NO_CARRY 0.5**, unique in every world.

## 4. Arms on HELD

| Arm | HELD behaviour | exact on counted |
|---|---|---|
| W (candidate) | right except the per-world wrong items below | 13 or 14 of 16 |
| NPT (no_prior_teaching) | DIGIT_NO_CARRY answers, plus (2,8) = 10 right | 9/16 = 0.5625 |
| SHUF | copies b | 0 |
| FROZEN | copies a | 0 |
| ORACLE (oracle_plumbing) | right on h01..h04, else `11` | 0.25 |

W wrong items (S2 index; S3 index) and the per-world contrast, comparator = max(chance 0.05,
NPT 0.5625, best shortcut 0.5) = 0.5625:

| World | W wrong | W | d_w | OLD end | OLD final |
|---|---|---|---|---|---|
| w_off_1 | S2[0]; S3[0] | 0.875 | 0.3125 | 1.0 | 1.0 |
| w_off_2 | S2[1]; S3[1] | 0.875 | 0.3125 | 1.0 | 1.0 |
| w_off_3 | S2[2]; S3[2,3] | 0.8125 | 0.25 | 1.0 | 0.875 |
| w_off_4 | S2[3]; S3[4] | 0.875 | 0.3125 | 1.0 | 1.0 |
| w_off_5 | S2[4,5]; S3[5] | 0.8125 | 0.25 | 1.0 | 1.0 |
| w_off_6 | S2[6]; S3[6] | 0.875 | 0.3125 | 1.0 | 0.875 |
| w_off_7 | S2[7]; S3[7,0] | 0.8125 | 0.25 | 1.0 | 1.0 |
| w_off_8 | S2[0]; S3[2] | 0.875 | 0.3125 | 1.0 | 1.0 |

A wrong answer v becomes v+1 (1-digit) or v+2 (2-digit), so its first byte (S2) or 2-byte prefix (S3)
is wrong as well.

## 5. Decision rule (SPEC 9.5) and G (SPEC 9.6)

- mean(W) = 109/128 = 0.8515625 >= absolute_floor 0.25.
- d = (0.3125 x5, 0.25 x3): mean 0.2890625, sd 0.0323468 (n-1), t(0.95, 7) = 1.894579,
  LCB = 0.2890625 - 1.894579 x 0.0323468 / sqrt(8) = **0.2673955 > margin_abs 0.05**.
- min d_w = 0.25 >= per_world_floor -0.05; 8 worlds >= min_worlds 8; the plumbing gate stop_gate has a
  negative (false) and a positive (true) fixture.
- old_drop: drops (0, 0, 0.125, 0, 0, 0.125, 0, 0), mean 0.03125, sd 0.0578638, upper bound
  0.03125 + 1.894579 x 0.0578638 / sqrt(8) = 0.0700092 <= max_drop 0.10.
- old_acq: OLD ladder at old_end (exposure-independent members): DIGIT_NO_CARRY, FIXED_1 and FIXED_2
  all 4/8 = 0.5; CONST/LENFREQ/SUFFIX1/ID_ZERO 1/8 ((6,5) = 11); SUFFIX2/3 1/8 ((5,5) = 10).
  OLD_END 1.0 - 0.5 = 0.5 >= margin 0.05 in every world.
- No world is technical, dev or rehearsal -> **verdict PASS**.
- G = mean_w [W - max(NPT, best shortcut, chance)] = 0.2890625 (28.9pp).
- Headline:
  `PASS | earned E3 (attempted E3) | held exact own-EOS 85.2% [S2+ determined, n=128] vs best shortcut DIGIT_NO_CARRY 50.0% | G 28.9pp | OLD final 96.9% (drop 3.1pp) | 8 MAIN worlds | best of k=1`
  (OLD final 0.96875, drop 0.03125; designs_tried_k 1: one (native_closure_sha, params_sha256) pair).

## 6. Admission and testbed

- ADM-ACQ (technical world, TAUGHT@prefix): W 15/15 = 1.0; best exposure-independent member 8/15 =
  0.5333 (SUFFIX2/3, DIGIT_NO_CARRY, FIXED_1) -> margin 0.4667 >= 0.05.
- ADM-SHUF: SHUF swaps the answers of orbits {2,5}<->{4,5} and {6,5}<->{8,5}; agreement 7/15 items
  (4/8 orbits), inside [0.35, 0.65]; orbit-consistent.
- ADM-CTRL: NPT zero channel pair_memory (writes 0, post_first = birth; W writes 150 and changes it);
  FROZEN zero channels pair_memory and readout (writes 0, digests stay at birth); at post_first NPT's
  readout and SHUF's channels equal W's (the first record (1,5) keeps its label under SHUF); ORACLE
  differs from W only on declared_changed readout. Doses: NPT 1.0, FROZEN 1.0 >= 0.05.
- Testbed (8 technical worlds, comparator = max(chance, ladder of the config's own stream)):
  CEILING_EXACT_BP is wrong on h16 in even worlds, CEILING_ORACLE_FEATURES on h15 in odd worlds:
  d = (0.5, 0.4375 alternating), mean 0.46875, LCB 0.446372 -> PASS. Every FLOOR config answers
  DIGIT_NO_CARRY (one extra S2 error in half the worlds): d = (0, -0.0625), mean -0.03125,
  LCB -0.053628 -> FAIL.

## 7. Values for other fixtures

- C00b: W right on 7 S2 and 3 S3 items (6 S3 in w_off_8): d = (0.0625 x7, 0.25), mean 0.0859375,
  sd 0.0662913, LCB 0.0415338 <= 0.05 -> FAIL; held 64.8%, G 8.6pp.
- C00e (dev only, 2 DEV worlds): W 0.875 and 0.8125, d 0.3125 and 0.25, held 84.4% (n=32), G 28.1pp;
  2 worlds < min_worlds, so the dev shadow verdict is FAIL.
- C06a: HELD 8 S2 + 12 S3; FIXED_2 of W's stream 12/20 = 0.6 > DIGIT_NO_CARRY 8/20 = 0.4; W 0.30 (odd
  worlds) / 0.25 (even), NPT 0; mean W 0.275, G -0.325 -> FAIL, best shortcut FIXED_2.
- C06c: OLD_END = 0.5 = best exposure-independent shortcut -> old_acq 0 < 0.05 -> FAIL (the primary
  still clears its LCB).
- K01: Brier (p_pass - outcome)^2 = 0.09, 0.16, 0.81, mean 0.353333; Laplace base rate (1+1)/(3+2) =
  0.4; only 3 scored deliveries (< 5) -> skill INSUFFICIENT_N.
