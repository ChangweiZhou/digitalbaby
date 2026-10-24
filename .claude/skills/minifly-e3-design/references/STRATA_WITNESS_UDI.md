# Strata, determinacy witnesses, UDI and G (addition)

Normative text: `tools/minifly_check/SPEC.md` sections 9.3, 9.4, 9.6 and 9.7. The checker computes all of this from
item content, the candidate's TAUGHT manifest and the lock's units block (`addition.stratum`, `addition.determined`,
STR-01, DET-01, UDI-01). You never assign a stratum, a determinacy flag or a UDI split by hand; read them from
`results/<role>/SCORES.json`. If anything here seems to disagree with the checker, the checker and the spec win and
this file is the bug (report it through `minifly-governance`).

## 1. Strata

| Stratum | Definition | Counts for E3 |
|---|---|---|
| S0 trivial | An operand is 0, or the answer equals an operand | no |
| S1 presentation twin | The canonical orbit (sorted, normalized operands) is taught | no |
| S2 interpolation | Each operand value appears in its position in some taught pair; no carry | yes, if determined |
| S3 carry | As S2, with carry | yes, if determined |
| S4 extrapolation | An operand value is unseen in its position, or the answer is longer than every taught answer | yes, if determined |

- S0 and S1 are exclusions: an item that is trivial or whose orbit is taught never counts for E3, whatever else holds.
- Taught includes any exposure stream that fed the orbit with its answer (SPEC 20.11): the scorer takes those orbits
  from ADM-EXPO's content scan (`admitcheck.exposed_held_orbits`) and makes the item S1 in every world, even when the
  candidate's TAUGHT manifest is clean.
- "In its position" means first operand or second operand. Strata are position-specific even when the task is
  commutative; the witness below may use either orientation.
- Carry: for single-digit operands, a + b >= 10; for multi-digit operands, any column produces a carry.
- E3 counts only S2..S4 items that are also determined, and each counted stratum needs at least
  CONSTRAINTS.min_items_per_stratum distinct items (8, draft; STR-01).

## 2. Witnesses (evaluator-side; never given to the learner)

- pair_rectangle: held (a, b) is determined iff there exist a' != a and b' != b such that (a, b'), (a', b) and
  (a', b') are all taught (either orientation if units.commutative). Under additive structure
  (a, b) = (a, b') + (a', b) - (a', b').
- column_closure (multi-digit): held is determined iff every column triple (a_i, b_i, carry_in_i) it requires occurs
  in some taught item. These column triples are the multi-digit units.

Why determinacy: an undetermined held item cannot be derived from the taught units under the declared structure, so
scoring it measures guessing or supplied priors, not reuse. Counted items are determined items only; too few is a DET-01
FAIL (the testbed cannot test E3).

## 3. Worked example: single-digit addition, pair_rectangle, commutative true

Taught set T (prompt `a+b=`, answer the decimal sum):

| Taught | (1,2) | (1,5) | (4,2) | (6,3) | (8,7) | (8,3) | (2,9) | (0,4) |
|---|---|---|---|---|---|---|---|---|
| Answer | 3 | 6 | 6 | 9 | 15 | 11 | 11 | 4 |

First-position values: {0, 1, 2, 4, 6, 8}. Second-position values: {2, 3, 4, 5, 7, 9}.

| Held | Sum | Stratum and why | Witness | Counted |
|---|---|---|---|---|
| (0,7) | 7 | S0: an operand is 0 | — | no |
| (5,1) | 6 | S1: orbit {1,5} is taught as (1,5) | — | no |
| (4,5) | 9 | S2: 4 is a first operand in (4,2), 5 a second in (1,5); no carry | determined: (4,2), (1,5), (1,2); 9 = 6 + 6 - 3 | yes |
| (6,7) | 13 | S3: 6 first in (6,3), 7 second in (8,7); carry | determined: (6,3), (8,7), (8,3); 13 = 9 + 15 - 11 | yes |
| (1,3) | 4 | S2: 1 first in (1,2), 3 second in (6,3); no carry | undetermined: (1,b') needs b' in {2,5}, (a',3) needs a' in {6,8}, and none of (6,2), (6,5), (8,2), (8,5) is taught in either orientation | no |
| (8,2) | 10 | S3: 8 first in (8,7), 2 second in (1,2); carry | undetermined: (8,b') needs b' in {3,7}, (a',2) needs a' in {1,4} or 9 via (2,9), and no such (a',b') is taught in either orientation | no |
| (9,5) | 14 | S4: 9 never appears as a first operand | determined only because commutative: (9,2) via the taught (2,9), plus (1,5), (1,2); 14 = 11 + 6 - 3 | yes |

Counted: (4,5) in S2, (6,7) in S3 and (9,5) in S4. With units.commutative false, (9,5) would be undetermined and drop
out. A real delivery needs at least 8 counted items per counted stratum; this table only shows the mechanics.

Shortcut note: DIGIT_NO_CARRY predicts (4,5) and (1,3) exactly, because they have no carry. A held pool made mostly of
S2 items gives the ladder a high bar; S3 and S4 items are where a learner can show headroom.

## 4. Worked example: two-digit addition, column_closure

Columns are listed units first as (a_i, b_i, carry_in_i).

| Taught | Sum | Columns |
|---|---|---|
| 12+35 | 47 | (2,5,0), (1,3,0) |
| 28+14 | 42 | (8,4,0), (2,1,1) |
| 57+31 | 88 | (7,1,0), (5,3,0) |
| 17+24 | 41 | (7,4,0), (1,2,1) |
| 38+25 | 63 | (8,5,0), (3,2,1) |
| 31+15 | 46 | (1,5,0), (3,1,0) |

First operands taught: {12, 28, 57, 17, 38, 31}. Second operands taught: {35, 14, 31, 24, 25, 15}.

| Held | Sum | Columns required | Stratum | Determined | Counted |
|---|---|---|---|---|---|
| 17+31 | 48 | (7,1,0) from 57+31; (1,3,0) from 12+35 | S2 (17 and 31 seen in position; no carry) | yes | yes |
| 28+15 | 43 | (8,5,0) from 38+25; (2,1,1) from 28+14 | S3 (seen in position; carry) | yes | yes |
| 12+31 | 43 | (2,1,0) is taught nowhere | S2 | no | no |
| 14+28 | 42 | — | S1 (orbit {14,28} taught as 28+14) | — | no |
| 18+25 | 43 | (8,5,0) from 38+25; (1,2,1) from 17+24 | S4 (18 never appears as a first operand) | yes | yes |
| 92+35 | 127 | (9,3,0) is taught nowhere | S4 (92 unseen as a first operand; the answer is longer than every taught answer) | no | no |

18+25 is the point of column units: the operand 18 was never taught, yet every column it needs was. Its stratum follows
the operand-value reading of 9.3; the checker's `addition.stratum` is authoritative for it.

## 5. UDI: leave-units-out variant worlds (E3 promotion only)

Declare `udi {variants_manifest, alpha}` in the lock; the variants live in `manifests/UDI_VARIANTS.json`.

Construction, per base world:
1. Take the counted held items (determined, S2..S4).
2. Split them into a dependent half D and an independent half I. For SEALED_CONFIRM, D is drawn with the seal-derived
   seed (purpose 'udi'), never chosen by hand.
3. Remove from the taught set enough witness support that every D item becomes undetermined, while every I item keeps a
   witness.
4. The variant world is the base world with those items no longer taught: TAUGHT(variant) = TAUGHT(base) minus the
   removed set.

UDI-01 checks that D items are undetermined in the variant, that I items stay determined, and that the variant's TAUGHT
equals the base TAUGHT minus the removed set. (Checker 1.0.0 registers UDI-01 as planned: not yet enforced by any gate.)

Example on the single-digit set of section 3, counted items (4,5), (6,7), (9,5), with D = {(4,5)} and I = {(6,7), (9,5)}
(in a real delivery D is half of 20 or more counted items):
- Every witness of (4,5) uses (4,2), so remove (4,2). Then (4,5) is undetermined, while (6,7) keeps (6,3), (8,7), (8,3)
  and (9,5) keeps (2,9), (1,5), (1,2). Valid.
- Removing the whole witness {(4,2), (1,5), (1,2)} instead would also break (9,5), which is in I. Invalid (UDI-01 FAIL).

Computation:

```
UDI = mean_w [ (acc_base(D) - acc_var(D)) - (acc_base(I) - acc_var(I)) ]
```

with a world-level LCB. Promotion needs LCB > 0. Without it the claim is E3 with reuse_specificity not_tested (or
udi_fail when it was tested and failed).

| World | acc_base(D) | acc_var(D) | acc_base(I) | acc_var(I) | Contribution |
|---|---:|---:|---:|---:|---:|
| w1 | 0.62 | 0.20 | 0.60 | 0.57 | 0.42 - 0.03 = 0.39 |
| w2 | 0.55 | 0.25 | 0.58 | 0.55 | 0.27 |
| w3 | 0.70 | 0.31 | 0.66 | 0.64 | 0.37 |
| w4 | 0.48 | 0.30 | 0.50 | 0.49 | 0.17 |
| w5 | 0.66 | 0.28 | 0.61 | 0.60 | 0.37 |
| w6 | 0.59 | 0.33 | 0.57 | 0.52 | 0.21 |
| w7 | 0.51 | 0.36 | 0.49 | 0.47 | 0.13 |
| w8 | 0.64 | 0.22 | 0.62 | 0.61 | 0.41 |

UDI = 0.290 with sd 0.110. Computed the paired_lcb way at alpha 0.05, LCB = 0.290 - 1.895 * 0.110 / sqrt(8) = 0.217 > 0.
The checker's computation is authoritative.

Why it separates reuse from triggers: a learner that uses the removed units loses accuracy on D only. A shortcut
ignores the taught units, and a teaching-as-trigger learner (any teaching switches on the same behaviour) loses the same
on D and I, so both give UDI of about 0. For example, acc_base 0.60 on both halves and acc_var 0.30 on both gives
(0.60 - 0.30) - (0.60 - 0.30) = 0.

## 6. G computation example

```
G = mean_w [ exact(W, HELD, final, counted) - max(exact(no_prior_teaching), best shortcut, chance) ]
```

8 MAIN worlds, chance 0.0, exact own-EOS on counted held items at the final checkpoint:

| World | W | no_prior_teaching | Best shortcut | Bar = max | W - bar |
|---|---:|---:|---|---:|---:|
| w1 | 0.62 | 0.08 | DIGIT_NO_CARRY 0.31 | 0.31 | 0.31 |
| w2 | 0.55 | 0.10 | NEAREST 0.25 | 0.25 | 0.30 |
| w3 | 0.70 | 0.12 | DIGIT_NO_CARRY 0.34 | 0.34 | 0.36 |
| w4 | 0.48 | 0.09 | DIGIT_NO_CARRY 0.29 | 0.29 | 0.19 |
| w5 | 0.66 | 0.15 | DIGIT_NO_CARRY 0.33 | 0.33 | 0.33 |
| w6 | 0.59 | 0.11 | NEAREST 0.27 | 0.27 | 0.32 |
| w7 | 0.51 | 0.31 | DIGIT_NO_CARRY 0.28 | 0.31 (no_prior_teaching) | 0.20 |
| w8 | 0.64 | 0.07 | DIGIT_NO_CARRY 0.30 | 0.30 | 0.34 |

G = 2.35 / 8 = 0.294, printed in the headline as `G 29.4pp`. Note w7: when the no-teaching control is the strongest
comparator, it sets the bar.

- G_per_sample divides G by the taught presentations in thousands (from receipts); G_per_cpu_h divides it by the
  lineage-wide CPU-h, failed attempts included. The checker computes both and SCORES.json shows the denominators used.
- These are MAIN worlds, so this G is shown as 'unsealed'. Only G on SEALED_CONFIRM worlds enters the program progress
  series (I11).
