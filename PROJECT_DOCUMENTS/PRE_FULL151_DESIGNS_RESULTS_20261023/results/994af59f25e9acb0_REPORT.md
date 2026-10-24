# R2/Z1 E3 correction: final result

This report is generated from the source-locked `SUMMARY.json` after the full receipt audit. The locked analysis script wrote that summary, then failed while formatting its Markdown report because `lock` was not in scope in `__main__`. The original failure record is retained. No scientific receipt, source-locked code, endpoint, or contrast was changed.

## Completion and scope

- 32 paired science worlds, four arms per world: 128/128 complete and committed.
- Each receipt passed the independent life and literal-branch audits before `SUMMARY.json` was written.
- Primary endpoint: final choice accuracy on six never-reinforced old-relation orientations, after intervening new learning and delay. The `N_old_rel` branch blocks only old-relation native writes.
- Selected comparisons: R2 against dual-bank R0, and Z1 against native Z0. No global mechanism-family winner is claimed.

## Primary results

| Arm | Final held-out W | Final held-out N_old_rel | W−N |
|---|---:|---:|---:|
| R0 | 52.1% | 49.0% | +3.1% |
| R2 | 55.2% | 54.2% | +1.0% |
| Z0 | 85.4% | 47.4% | +38.0% |
| Z1 | 72.9% | 49.5% | +23.4% |

| Paired contrast of W−N effects | Mean | Simultaneous 95% CI | Decision |
|---|---:|---:|---|
| R2 − R0 | -2.1% | [-13.7%, +9.6%] | not established |
| Z1 − Z0 | -14.6% | [-27.3%, -1.9%] | not established |

R2 did not establish improved E3: its held-out W−N effect was only +1.0 percentage point, versus +3.1 points for R0. Z1 retained a positive within-arm causal effect (+23.4 points), but Z0 was larger (+38.0 points); the paired Z1−Z0 interval lies entirely below zero. Thus this specific eligibility split worsened the prespecified E3 contrast.

These are within-fixture, binary relation choices; a one-bit-per-symbol class rule could solve the held-out pairs. They do not show broad relational learning, autonomous learning, or general intelligence.

Source lock: `7a190f8c9bd8d63848bc898f30c74dafc2353cb5868a24b1b35d54d3431d706b`.
