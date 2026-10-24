# V83 common-world audit report

**Status:** mechanistic comparison only; V83 does not promote a new persistent core.

## Flags

`ANCHOR_PATTERN_NOT_REPRODUCED`, `V82_SHARED_IDENTITY_PROTECTION_REPRODUCED`, `OBSERVER_COUPLING`, `NO_CLEAN_HYBRID`, `V81_ALLOCATION_CONTRIBUTES`

## Common-panel arm means

| arm | macro fixed | macro actual | plasticity floor | old accuracy | old conditional loss | worst old damage | slow L1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| REF | 0.8179 | 0.8182 | 0.1875 | 0.7489 | 0.0891 | 3.1401 | 32633.1 |
| V81 | 0.8117 | 0.8811 | 0.3250 | 0.7248 | 0.1063 | 2.2370 | 26130.2 |
| V82 | 0.7798 | 0.7829 | 0.0000 | 0.7746 | 0.0427 | 2.3307 | 33888.5 |
| S81_D82 | 0.7385 | 0.7557 | 0.0000 | 0.6598 | 0.1254 | 2.7679 | 34533.4 |
| S82_D81 | 0.8567 | 0.8670 | 0.5208 | 0.7170 | 0.1164 | 1.5375 | 25016.6 |
| V82_NOSUPPORT | 0.7676 | 0.7716 | 0.0000 | 0.7503 | 0.0321 | 1.7500 | 24275.2 |
| V81_NATIVE_ALLOC | 0.7354 | 0.7536 | 0.0000 | 0.6508 | 0.1678 | 2.4817 | 34353.2 |

## Anchor effects versus REF

- **v81_macro_fixed:** -0.006205
- **v81_macro_actual:** 0.062931
- **v82_old_worst:** -0.809355
- **v81_shared_loss:** -0.015179
- **v82_shared_loss:** -0.041906

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
| REF | after_72_day | 0.6875 | 0.5000 | 0.6806 |
| REF | after_96_day | 0.6771 | 0.5000 | 0.6667 |
| REF | after_challenge_day | 0.6875 | 0.3750 | 0.6806 |
| V81 | after_72_day | 0.5417 | 0.1250 | 0.5556 |
| V81 | after_96_day | 0.6146 | 0.2500 | 0.6250 |
| V81 | after_challenge_day | 0.5833 | 0.1250 | 0.6250 |
| V82 | after_72_day | 0.6181 | 0.5000 | 0.6250 |
| V82 | after_96_day | 0.6198 | 0.3750 | 0.6806 |
| V82 | after_challenge_day | 0.6667 | 0.3750 | 0.6667 |
