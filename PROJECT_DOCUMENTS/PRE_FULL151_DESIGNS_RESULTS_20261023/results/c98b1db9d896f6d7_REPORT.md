# ELM v22: readout-specific memory writes

Status: exploratory 3-seed pilot; no automatic promotion or significance claim.

Four outcomes; chance accuracy is 25%. Probes are noisy variants of learned cue identities.

| Arm | A after A | A after B | B after B | Final unchanged | Final revised | Final all |
|---|---:|---:|---:|---:|---:|---:|
| uniform_delta | 98.61% | 49.54% | 100.00% | 47.84% | 100.00% | 60.88% |
| g1_feedback | 95.83% | 45.83% | 97.22% | 43.83% | 98.15% | 57.41% |
| joint_feedback | 95.37% | 44.91% | 95.37% | 45.68% | 98.15% | 58.80% |
| joint_scrambled | 94.44% | 36.57% | 97.22% | 44.44% | 97.22% | 57.64% |
| joint_outcome_only | 95.83% | 44.44% | 92.13% | 42.90% | 96.30% | 56.25% |
| cumulative_ridge | 100.00% | 98.61% | 99.07% | 99.38% | 46.30% | 86.11% |

## What is changed

The encoder, sensory data, feature sparsity, codebook and labeled stream retain V21 RNG namespaces. The encoder is frozen after separate sensory development; only associative readout weights learn online.

V21 shared-mask memory has the exact form W=V C, where C C^T=104 I. All arms now store V with 96 input coordinates and four task readouts. Predictions are h V, and the feedback ports are (h V) C. This preserves the effective capacity of the ordinary, G1-only and ridge predecessors while preventing joint masks from introducing 100 task-invisible output directions.

Let A be KC-to-MBON counts normalized over MBONs per KC. The four-head compatibility is H[d,i,a] = mean over KCs k in group i of G1[d,k] * sum_j A[k,j] C[a,j]^2/104. Multiplication occurs at the original KC before pooling. The squared code participation and random ELM/profile assignment are engineering correspondences, not biological outcome or compartment identities.

For error e=one_hot(label)-h V, teaching is abs((e C) F). The joint priority is teaching @ H, one vector per task readout. Each readout selects active inputs until the retained-energy floor is met. The update is delta V[i,a] = eta * mask[i,a] * h[i] * e[a] / sum_i(mask[i,a] h[i]^2).

Energy floor: 0.500; learning rate: 0.600. Each local arm corrects the same fraction of its own current residual. Absolute updates between diverging learners and realized selected energies need not match. The norm relative to an ordinary write at that same state is bounded by 1/sqrt(energy_floor). Flat priorities fall back to all active inputs.

| Arm | Operation |
|---|---|
| uniform_delta | Ordinary normalized local learning |
| g1_feedback | V21 shared input mask, in four-readout coordinates |
| joint_feedback | Measured within-KC teaching/readout pairing |
| joint_scrambled | KC readout profiles shuffled only within each fixed G1 group |
| joint_outcome_only | Measured joint routing; prediction removed from teaching |
| cumulative_ridge | Cumulative historical statistics; extra covariance memory |

The joint null preserves G1 exactly, group memberships, and each group's complete multiset of KC readout profiles. It disrupts their within-cell pairing with teaching delivery. The derived joint tensor marginals, realized masks and write norms are not held identical.

## Write diagnostics

| Arm | Selected active weights | Retained energy | Norm ratio | Distinct head masks | Feedback-cut mask change |
|---|---:|---:|---:|---:|---:|
| uniform_delta | 100.00% | 100.00% | 1.0000 | 1.000 | n/a |
| g1_feedback | 51.35% | 53.38% | 1.3705 | 1.000 | 96.94% |
| joint_feedback | 51.59% | 53.34% | 1.3715 | 3.916 | 99.35% |
| joint_scrambled | 51.62% | 53.36% | 1.3712 | 3.908 | 99.35% |
| joint_outcome_only | 51.89% | 53.41% | 1.3704 | 3.897 | n/a |
| cumulative_ridge | n/a | n/a | n/a | n/a | n/a |

MSE, score drift and update norms are in four-task coordinates. V21's old port-space pre-update MSE is four times the task MSE, and its port-space update norm is sqrt(104) times the factored update norm. Current-label MSE improvement and four-label score drift retain their scale.

## Alternative writes from identical states

COMMON_STATE_DAMAGE.csv compares the actual policy write with an ordinary write, and for joint arms also with the shared G1 write, from identical pre-update weights, input and residual. Only the actual write is committed. Each row retains numerator, denominator, probe counts, write counts and squared update norms. Ratios are omitted if the reference drift is effectively zero. These comparisons measure immediate write placement, not the long-term trajectory of following an alternative policy.

For joint_feedback versus the shared G1 alternative, feedback is held present and the comparison isolates the addressing construction. Joint_outcome_only versus g1_feedback changes both addressing and prediction feedback; that particular pair cannot isolate readout geometry.

Protected probes exclude the current cue identity; during revision they exclude all deliberately revised identities. No old probe, cue ID, phase marker, revision flag or diagnostic enters a learner. COLLATERAL_DAMAGE.csv records actual old-label loss changes, score drift and prediction flips. Repeated probe comparisons are not independent statistical samples.

The following ratios sum per-write drift over protected A probes during B, then divide summed numerators by summed denominators. Consult the per-seed CSV before interpreting pooled numbers.

| Actual policy | Alternative write | Actual/reference drift | Probe comparisons |
|---|---|---:|---:|
| g1_feedback | uniform_delta | 1.4682 | 31104 |
| joint_feedback | uniform_delta | 1.4323 | 31104 |
| joint_feedback | g1_feedback | 0.9752 | 31104 |
| joint_scrambled | uniform_delta | 1.4637 | 31104 |
| joint_scrambled | g1_feedback | 0.9775 | 31104 |
| joint_outcome_only | uniform_delta | 1.4347 | 31104 |
| joint_outcome_only | g1_feedback | 0.9608 | 31104 |

## Scope and cost

The 96-profile G1 transfer retains the previous 83.94% cellwise reconstruction error. Whole-cell co-targeting does not identify spatial synaptic compartments. No synaptic sign, stabilization law, memory lifetime or guaranteed write protection is inferred from unsigned counts. The zero G1 pattern represents 2,700 cells as one functional profile, so population frequencies are not preserved. REDUCTION.json includes source checksums, axis verification and measured coverage.

Twenty-eight of 1,852 directly addressed KCs have no recorded MBON target. They keep zero readout profiles. The missing fraction is 1.0266% of original G1 mass and 1.3146% of pooled mean G1 mass; the pooled operator discrepancy is 5.7475%. These are different denominators.

- Seed 17: 48 online cue identities; 96 unique online sensory inputs; 120 distinct input/outcome pairs; 360 presentations per arm. Development: 384 training + 64 calibration; total charged per arm 808. Elapsed 1.80 seconds.
- Seed 29: 48 online cue identities; 96 unique online sensory inputs; 120 distinct input/outcome pairs; 360 presentations per arm. Development: 384 training + 64 calibration; total charged per arm 808. Elapsed 1.22 seconds.
- Seed 43: 48 online cue identities; 96 unique online sensory inputs; 120 distinct input/outcome pairs; 360 presentations per arm. Development: 384 training + 64 calibration; total charged per arm 808. Elapsed 1.23 seconds.

Each arm stores 384 learned associative coefficients; ridge additionally stores 9,216 inverse-covariance coefficients. The 96 ELM cells, 576 private transient values and 288 history values retain their previous roles. Developed encoder parameters, routing arrays, source anatomy and feature caches are additional resources. Fixed anatomy is not counted as learned memory. COSTS.json charges development, online presentations and read-only evaluator work separately. Cached features do not discount repeated learning presentations.

Revised cues receive equal numbers of old and new labels, so cumulative ridge still optimizes conflicting historical targets. Read acquisition, retention and revision jointly. Beating a shuffle alone is insufficient; measured routing must improve on ordinary and predecessor learning. A learned-feedback claim additionally requires a benefit over outcome-only.

## Verification

```json
{
  "status": "passed",
  "checks": [
    "embedded_checksum_dimensions_and_source_aggregation",
    "within_group_shuffle_preserves_profiles_and_G1_predecessor",
    "joint_product_before_pooling_independent_cell_sum",
    "104_port_uniform_and_G1_predecessor_trajectory_equivalence",
    "four_head_and_104_port_ridge_direct_solve",
    "per_readout_fractional_correction_and_energy_norm_bounds",
    "propose_and_old_probe_evaluation_are_read_only",
    "predicted_write_drift_equals_actual_forward_change",
    "zero_feature_and_zero_residual",
    "outcome_only_priority_prediction_independence",
    "flat_priority_fallback",
    "independent_ELM_experience_state_reset"
  ],
  "maximum_104_port_factorization_absolute_error": 8.881784197001252e-16,
  "synthetic_joint_states_with_distinct_head_masks": 12,
  "synthetic_joint_states_changed_by_feedback_cut": 12,
  "synthetic_joint_states_examined": 12,
  "synthetic_responsiveness_is_not_learning_evidence": true,
  "v20_v21_default_stream_compatibility": [
    {
      "seed": 17,
      "status": "matched",
      "note": "Compares data and labeled order, not hardware-dependent encoder weights."
    },
    {
      "seed": 29,
      "status": "matched",
      "note": "Compares data and labeled order, not hardware-dependent encoder weights."
    },
    {
      "seed": 43,
      "status": "matched",
      "note": "Compares data and labeled order, not hardware-dependent encoder weights."
    }
  ]
}
```

The ZIP contains aggregate CSV/JSON/Markdown only. It omits raw graphs, examples, checkpoints and per-update traces. No automatic tuning, extra seed sweep or architecture expansion is performed.
