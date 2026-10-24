# V44 results — causal capacity decomposition

**Verdict:** `SLOT_PRESSURE_DOMINANT`

This round diagnoses resource pressure; it does not test a deployable repair.

## Gates

- Q0 independent CAP-BASE reproduction: **True**
- Q1 intervention isolation: **True**
- Q2 capacity materially contributes: **True**
- Q3 removing both caps restores acquisition criterion: **True**
- Q4 trace pressure material: **False**
- Q5 mature-slot pressure material: **True**
- Q6 interaction: **interaction_small_or_unresolved**

## Primary acquisition contrasts

| contrast | mean | 95% interval |
|---|---:|---:|
| TRACE-RELIEF_minus_CAP-BASE | -0.0327 | [-0.0481, -0.0173] |
| SLOT-RELIEF_minus_CAP-BASE | +0.3311 | [+0.2691, +0.3930] |
| BOTH-RELIEF_minus_CAP-BASE | +0.3315 | [+0.2699, +0.3932] |
| factorial_trace | -0.0161 | [-0.0238, -0.0084] |
| factorial_slot | +0.3477 | [+0.2833, +0.4120] |
| interaction | +0.0332 | [+0.0177, +0.0487] |

See `CAUSAL_CONTRASTS.csv`, lifecycle tables, and `PROTOCOL.md` for the locked interpretation rules.
