# MiniFly V9.6 — byte-value persistence interventions

Prospective paired worlds 701–780, n=80; fixed B transfer and FB4 transfer, no tuning on these worlds.

| Arm | Old D4 choice | vs REF, pp (95% paired t) | Old D4 margin Hz | New D4 choice (cohorts 1–4) | Old pairs lost/at-risk |
|---|---:|---:|---:|---|---:|
| REF | 65.00% | — | 0.03284 | 68.8%, 67.9%, 77.5%, 72.1% | 38/178 |
| B_TRANSFER | 63.75% | -1.25 [-5.74, +3.24] | 0.02861 | 64.6%, 62.1%, 70.8%, 72.5% | 39/179 |
| FB4_TRANSFER | 72.92% | +7.92 [+1.11, +14.72] | 0.42184 | 72.1%, 70.4%, 80.0%, 82.1% | 32/196 |

The table is descriptive. Judge the complete old-and-new vector, including immediate acquisition and delayed new-cohort retention, not the own write/no-write gap alone.

## Candidate state cost (bytes)

- REF: {'added_bank_mutable_bytes': 0, 'fixed_B_vector_bytes': 0, 'fixed_router_bytes': 0, 'native_shared_mutable_bytes': 165664, 'persistent_deployable_bytes': 3050272, 'predictor_mutable_bytes': 2884608, 'total_allocated_counted_bytes': 3050272, 'transient_alpha_scratch_bytes': 0}
- B_TRANSFER: {'added_bank_mutable_bytes': 0, 'fixed_B_vector_bytes': 41416, 'fixed_router_bytes': 0, 'native_shared_mutable_bytes': 165664, 'persistent_deployable_bytes': 3091688, 'predictor_mutable_bytes': 2884608, 'total_allocated_counted_bytes': 3091688, 'transient_alpha_scratch_bytes': 0}
- FB4_TRANSFER: {'added_bank_mutable_bytes': 331328, 'fixed_B_vector_bytes': 41416, 'fixed_router_bytes': 331392, 'native_shared_mutable_bytes': 165664, 'persistent_deployable_bytes': 3671576, 'predictor_mutable_bytes': 2884608, 'total_allocated_counted_bytes': 3754408, 'transient_alpha_scratch_bytes': 82832}

## Validity and scope

REF was required to reproduce archived V9.5 world 401 exactly; a one-address bank had to reproduce B exactly on a nonzero life. Those engineering gates are in PREFLIGHT.json.
All probe scores were recomputed from three pair margins before aggregation. Every science world used the same byte/reward schedule across arms and kept the predictor identical. The B vector is historical PN calibration transferred unchanged; FB4 additionally spends bank and router state. No anatomical or capacity-only claim follows.
The 80-world intervals describe world variation for one fixed model seed. This experiment is distinct from PN-cue A11.

See RESULTS.json for every old/new choice and margin estimate, matched no-write controls, acquisition effects, pair losses, and receipt hashes.
