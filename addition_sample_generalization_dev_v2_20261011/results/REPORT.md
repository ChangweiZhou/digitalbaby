# SAMPLE-GEN V2 REPORT

Verdict: ADOPT_BOUNDED_SAMPLE_EFFICIENT_CORE

This tests E3 within bounded addition with explicit identity/role and ordinal-output priors. It does not establish arbitrary arithmetic or broad intelligence.

```json
{
  "verdict": "ADOPT_BOUNDED_SAMPLE_EFFICIENT_CORE",
  "selected_candidate": "COV-R",
  "checks": {
    "minimum_primary_gain": true,
    "primary_lower_positive": true,
    "old_causal_lower_positive": true,
    "shuffled_causal_lower_positive": true,
    "final_old_at_least_75": true,
    "final_new_at_least_60": true,
    "ordinary_score_protected": true,
    "ordinary_final_protected": true
  },
  "primary": {
    "gain": {
      "mean": 0.4941411716407681,
      "lower": 0.4553189582596303,
      "df": 61.6755488523854,
      "method": "stratified paired Student",
      "worlds": 64
    },
    "causal_N": {
      "mean": 0.6106305372605019,
      "lower": 0.5741880701012014,
      "df": 60.583897699277315,
      "method": "stratified paired Student",
      "worlds": 64
    },
    "causal_shuffled": {
      "mean": 0.5543160214919562,
      "lower": 0.5177626954114714,
      "df": 60.59749255279162,
      "method": "stratified paired Student",
      "worlds": 64
    }
  },
  "opaque_means": {
    "STABLE-ID": {
      "score": 0.1883643656197337,
      "final_old_held": 0.25,
      "final_new_held": 0.171875,
      "causal_N": 0.036801865619733684,
      "causal_shuffled": 0.040596706435151,
      "production_cpu_seconds": 0.04673087500000409,
      "persistent_bytes": 110294
    },
    "COMPACT-ID": {
      "score": 0.1883643656197337,
      "final_old_held": 0.25,
      "final_new_held": 0.171875,
      "causal_N": 0.036801865619733684,
      "causal_shuffled": 0.040596706435151,
      "production_cpu_seconds": 0.03523409374999987,
      "persistent_bytes": 7374
    },
    "COV-R": {
      "score": 0.6825055372605019,
      "final_old_held": 0.93984375,
      "final_new_held": 0.60703125,
      "causal_N": 0.6106305372605019,
      "causal_shuffled": 0.5543160214919564,
      "production_cpu_seconds": 0.03876870312499803,
      "persistent_bytes": 14690
    },
    "DIAG-R": {
      "score": 0.3171961146699023,
      "final_old_held": 0.42421875000000003,
      "final_new_held": 0.33906250000000004,
      "causal_N": 0.20313361466990232,
      "causal_shuffled": 0.1874231443890198,
      "production_cpu_seconds": 0.02921257812500099,
      "persistent_bytes": 7250
    }
  },
  "ordinary_guard_means": {
    "STABLE-ID": {
      "score": 0.22389224097109828,
      "final_old_held": 0.3703125,
      "final_new_held": 0.1875,
      "causal_N": 0.08639224097109824,
      "causal_shuffled": 0.06965905359377059,
      "production_cpu_seconds": 0.05660312499999915,
      "persistent_bytes": 110294
    },
    "COMPACT-ID": {
      "score": 0.22389224097109828,
      "final_old_held": 0.3703125,
      "final_new_held": 0.1875,
      "causal_N": 0.08639224097109824,
      "causal_shuffled": 0.06965905359377059,
      "production_cpu_seconds": 0.040964218749999774,
      "persistent_bytes": 7374
    },
    "COV-R": {
      "score": 0.6671143815280429,
      "final_old_held": 0.8984375,
      "final_new_held": 0.7,
      "causal_N": 0.532739381528043,
      "causal_shuffled": 0.5355894800182488,
      "production_cpu_seconds": 0.04395196874999599,
      "persistent_bytes": 14690
    },
    "DIAG-R": {
      "score": 0.44586650563168434,
      "final_old_held": 0.61875,
      "final_new_held": 0.3109375,
      "causal_N": 0.3052415056316843,
      "causal_shuffled": 0.32091086733700663,
      "production_cpu_seconds": 0.03352378124999797,
      "persistent_bytes": 7250
    }
  },
  "committed_full_jobs": 492,
  "committed_external_records": 1180800,
  "committed_training_bytes": 6906441,
  "state_trajectories_with_shared_prefix": 10332
}
```

Postprocessing-only recovery: the original status monitor repeatedly decoded all receipts while holding a shared lock. The replacement reads each accepted receipt once, retains compact metric summaries and reports counters from memory. Original learning/design sources, manifests, selection, receipts and acceptance certificates are unchanged; zero new training records were executed.
