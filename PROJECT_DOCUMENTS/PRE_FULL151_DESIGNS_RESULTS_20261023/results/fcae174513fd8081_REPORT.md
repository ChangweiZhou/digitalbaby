# V83 common-world audit report

**Status:** mechanistic comparison only; V83 does not promote a new persistent core.

## Flags

`ANCHOR_PATTERN_REPRODUCED`, `V81_CHURN_REPRODUCED`, `V82_SHARED_IDENTITY_PROTECTION_REPRODUCED`, `NO_CLEAN_HYBRID`, `V81_ALLOCATION_CONTRIBUTES`

## Common-panel arm means

| arm | macro fixed | macro actual | plasticity floor | old accuracy | old conditional loss | worst old damage | slow L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| REF | 0.7970 | 0.8002 | 0.2844 | 0.7608 | 0.0653 | 2.3797 | 32677.8 |
| V81 | 0.8282 | 0.8606 | 0.3931 | 0.7161 | 0.1013 | 1.7325 | 26105.1 |
| V82 | 0.8174 | 0.8282 | 0.2569 | 0.7827 | 0.0493 | 1.8376 | 34002.4 |
| S81_D82 | 0.7491 | 0.7758 | 0.1854 | 0.6607 | 0.0892 | 1.9885 | 34337.1 |
| S82_D81 | 0.8588 | 0.8708 | 0.5066 | 0.7346 | 0.0993 | 1.2902 | 25177.3 |
| V82_NOSUPPORT | 0.7923 | 0.8043 | 0.2469 | 0.7441 | 0.0488 | 1.4056 | 24186.5 |
| V81_NATIVE_ALLOC | 0.7451 | 0.7720 | 0.1889 | 0.6499 | 0.1093 | 1.8436 | 34240.2 |

## Anchor effects versus REF

- **v81_macro_fixed:** 0.031288
- **v81_macro_actual:** 0.060387
- **v82_old_worst:** -0.542074
- **v81_shared_loss:** 0.012500
- **v82_shared_loss:** -0.021403

## Reader / attribution safeguards

- Primary reader uses the fixed reference-Q projection for all arms.
- Actual-only outcomes are recorded independently of the reader.
- Hybrid PN→KC matrices are copied from the declared static donor, so dynamics swaps do not redraw jitter.
- Per-memory day rows permit shared-identity loss analysis on the same initially-correct memories.

## Interpretation

Use the flags and paired-history contrasts to decide whether the historical V81/V82 polarity survives common worlds. Do not infer independent biological modules from the block swaps; Q/support/feedback participate in coupled computations.

## Secondary continuous-life panel

| arm | stage | overall fixed | REV subset | unchanged subset |
|---|---|---:|---:|---:|
| REF | after_72_day | 0.6892 | 0.5521 | 0.6713 |
| REF | after_96_day | 0.6710 | 0.5521 | 0.6713 |
| REF | after_challenge_day | 0.6936 | 0.5729 | 0.6620 |
| V81 | after_72_day | 0.6013 | 0.3646 | 0.5926 |
| V81 | after_96_day | 0.5846 | 0.3542 | 0.6007 |
| V81 | after_challenge_day | 0.5964 | 0.3750 | 0.5914 |
| V82 | after_72_day | 0.6383 | 0.5312 | 0.6343 |
| V82 | after_96_day | 0.6115 | 0.5104 | 0.6377 |
| V82 | after_challenge_day | 0.6632 | 0.5312 | 0.6331 |
