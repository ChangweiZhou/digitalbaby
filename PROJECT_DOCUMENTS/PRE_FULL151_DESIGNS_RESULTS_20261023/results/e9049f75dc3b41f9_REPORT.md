# ELM v21: teaching-directed input-connection addressing

Status: exploratory 3-seed pilot. No automatic promotion or significance claim.

Held-out probes are noisy variants of learned cue identities. There are four outcomes (25% chance).
Task labels and evaluation results are excluded from fixed anatomical profile construction and assignment. Online outcomes legitimately control writes; no cue identity, phase marker, revision flag or old probe enters a learner.

| Arm | A after A | A after B | B after B | Final unchanged | Final revised | Final all |
|---|---:|---:|---:|---:|---:|---:|
| g1_feedback | 95.83% | 45.83% | 97.22% | 43.83% | 98.15% | 57.41% |
| g1_scrambled | 96.30% | 32.87% | 90.74% | 40.74% | 94.44% | 54.17% |
| outcome_only | 95.37% | 48.15% | 88.43% | 44.44% | 98.15% | 57.87% |
| uniform_delta | 98.61% | 49.54% | 100.00% | 47.84% | 100.00% | 60.88% |
| cumulative_ridge | 100.00% | 98.61% | 99.07% | 99.38% | 46.30% | 86.11% |

## Intervention and controls

The ELM, development, sensory inputs, task codes and stream RNG namespaces are retained from v20. The online representation stays frozen; durable learning occurs in the associative W. The G1 controller selects active input coordinates whose outgoing connections may change. It does not identify separate biological modulation sites on every input-output synapse.

- g1_feedback: abs((target-prediction) F) times the pooled direct teaching-delivery operator ranks active inputs.
- g1_scrambled: each teaching row independently permutes its 95 live profile values; the universal zero profile stays fixed. Per-row strengths and sparsity are preserved; column marginals and shared destinations are not.
- outcome_only: the same measured delivery profiles, with the learned prediction removed from the teaching computation.
- uniform_delta: ordinary normalized local delta learning.
- cumulative_ridge: unchanged cumulative-statistics reference with additional quadratic memory.

Select high-priority active coordinates until at least 50.0% of feature energy is retained. Ties use a fixed label-independent order. Flat scores fall back to updating all active coordinates. Zero direct teaching is a routing observation, not permanent biological nonplasticity.

Each local arm uses W += eta * outer(mask*h, target-prediction)/sum(mask*h*h). For nonzero h this corrects exactly the same fraction eta of that arm's current residual. This matches fractional correction at the current state, not absolute updates between diverging arms. The same-state update norm is bounded by 1/sqrt(energy_floor) times the ordinary update. Ridge is not subject to this local normalization.

All local and ridge updates remain within the four-label code span. The 104-port codes still influence which teaching signals are produced; counterfactual mask sensitivity is logged rather than assumed. No positive result is guaranteed.

## Reduction and its limitations

G1 contains 301 teaching channels, 4,552 original destinations and 3,891 nonzero entries. Of those destinations, 2,700 have no recorded direct input, and 41 teaching rows are zero. These zero observations are not interpreted as proof of no biological plasticity.

The supplied anatomy-only spherical clustering produces 95 nonzero delivery profiles and one zero profile. Each profile ranks one ELM coordinate; the profile-to-ELM assignment is a fixed new permutation per seed. It is not a recovered anatomical identity. Representing the 2,700 zero columns by one functional pattern does not preserve biological population frequencies.

Group sums preserve all source routing mass. The learner uses count means to avoid ranking a profile highly solely because many original cells were assigned to it. This is a coarse transfer, not a faithful reconstruction: cellwise relative Frobenius error is 83.94%, and the nonzero-destination p90 relative error is 224.65%. REDUCTION.json includes further errors and the grouping algorithm. The script embeds original sparse G1, group membership and source fingerprints so the aggregation can be checked without the large connectome.

The prior 96.5% composed mass result concerns dropping indirect teaching delivery before this additional profile reduction; it must not be presented as the fidelity of the 96-profile learner. A negative result rejects this transfer-plus-learning construction, not raw G1 or fly learning generally.

## Write and interference diagnostics

| Arm | Selected active features | Retained energy | Norm ratio to same-state ordinary write | Flat-score fallback | Mask changes when prediction feedback is cut |
|---|---:|---:|---:|---:|---:|
| g1_feedback | 51.35% | 53.38% | 1.3705 | 0.00% | 96.94% |
| g1_scrambled | 51.22% | 53.40% | 1.3704 | 0.00% | 94.07% |
| outcome_only | 51.78% | 53.23% | 1.3724 | 0.00% | n/a |
| uniform_delta | 100.00% | 100.00% | 1.0000 | 0.00% | n/a |
| cumulative_ridge | n/a | n/a | n/a | n/a | n/a |

COLLATERAL_DAMAGE.csv records before/after changes on held-out variants of previously seen identities, excluding the current identity. During revision it also excludes all deliberately revised identities. It reports signed old-label MSE change, its positive part, score drift, accuracy change and prediction flips, aggregated by pass and protected A/B group. Each mean is over repeated probe comparisons, not independent observations. The same held-out probes are used at checkpoints and for evaluator-only diagnostics.

Damage is measured in four-label score coordinates (code projection divided by 104). Current-example label-MSE improvement is logged alongside it. These diagnostics never select masks, modify weights, stop training or tune parameters. Lower damage alone is not enough: read acquisition, retention and revision together.

## Costs and interpretation

- Seed 17: 48 online cue identities; 96 unique online sensory inputs; 120 distinct input/outcome pairs; 360 labeled presentations per arm. Development: 384 training + 64 calibration presentations. Total: 808. Elapsed 1.30 seconds.
- Seed 29: 48 online cue identities; 96 unique online sensory inputs; 120 distinct input/outcome pairs; 360 labeled presentations per arm. Development: 384 training + 64 calibration presentations. Total: 808. Elapsed 0.70 seconds.
- Seed 43: 48 online cue identities; 96 unique online sensory inputs; 120 distinct input/outcome pairs; 360 labeled presentations per arm. Development: 384 training + 64 calibration presentations. Total: 808. Elapsed 0.69 seconds.

Default resource counts: 96 ELM activity cells, 576 private transient values, 288 history values, 9,984 associative coefficients and 7,752 developed encoder parameters. Ridge additionally stores 9,216 inverse-covariance coefficients. Fixed routing, profile maps and calibration are additional memory. Raw examples reside only in the experiment driver. Feature caching does not discount learning presentations. COSTS.json counts the extra read-only probe evaluations separately.

A return is measured after the final B pass, with no A training/reminder. Read-only A probes also occur earlier. Revised cues receive as many new-label presentations as their original-label presentations, so cumulative ridge deliberately retains conflicting historical targets.

If measured and scrambled addressing both improve on uniform learning, that supports a generic addressing operation, without showing the measured structure is better. A learned-feedback claim additionally requires improvement over outcome-only. Compression comparisons are deferred until there is useful behavior to preserve.

## Verification

{
  "status": "passed",
  "checks": [
    "embedded_shapes_checksum_and_exact_group_aggregation",
    "read_only_prediction",
    "current_cue_fractional_correction",
    "write_energy_floor",
    "update_norm_bound",
    "zero_feature_and_zero_error",
    "uninformative_score_fallback",
    "outcome_only_prediction_independence",
    "independent_row_scramble_multisets_and_explicit_zero_profile",
    "cumulative_ridge_direct_solve",
    "independent_ELM_state_reset"
  ],
  "local_arms_checked": [
    "g1_feedback",
    "g1_scrambled",
    "outcome_only",
    "uniform_delta"
  ],
  "task_span_prediction_changes_that_change_mask": 16,
  "task_span_prediction_changes_examined": 16,
  "responsiveness_is_not_a_performance_claim": true,
  "v20_default_stream_compatibility": [
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

No raw graph, per-neuron trace, per-update log or checkpoint is exported. The compact archive contains aggregate CSVs, JSON provenance and this report.
