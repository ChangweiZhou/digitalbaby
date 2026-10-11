# FEEDBACK-EFFICIENCY V1 — REPORT

Verdict: RETAIN_COV_R

E3 improvement test for bounded addition from noisy/missing observed answer bytes. Own outputs on never-taught combinations, delayed probes and intervening learning. Role/additivity and ordinal-output priors are engineered; this is not unrestricted arithmetic or AGI.

```json
{
  "verdict": "RETAIN_COV_R",
  "selected": "COV-EXTRA",
  "gain": {
    "mean": 0.010742187500000003,
    "lower": -0.010771921452269767,
    "method": "paired Student"
  },
  "causal": {
    "new_causal_N": {
      "mean": 0.2884765625,
      "lower": 0.2538426554291025,
      "method": "paired Student"
    },
    "new_causal_shuffle": {
      "mean": 0.269140625,
      "lower": 0.23087793339844018,
      "method": "paired Student"
    },
    "old_causal_N": {
      "mean": 0.508004474638801,
      "lower": 0.4759666062139851,
      "method": "paired Student"
    },
    "old_causal_shuffle": {
      "mean": 0.48899779596507625,
      "lower": 0.45682837319454783,
      "method": "paired Student"
    }
  },
  "checks": {
    "mean_gain": false,
    "gain_lower": false,
    "all_causal_lowers": true,
    "primary_protection": true,
    "standard_protection": true,
    "ordinary_protection": true,
    "resources": true,
    "shortcut_guard": true
  },
  "shortcut_gap": 0.1458984375,
  "committed_full_jobs": 456,
  "committed_external_records": 1167360,
  "committed_training_bytes": 6815268,
  "correlated_state_trajectories": 10488
}
```
