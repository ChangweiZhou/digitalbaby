# ELM v20: measured fly routing inside online associative learning

Status: exploratory 3-seed mechanism pilot; no automatic promotion or statistical significance claim.

All percentages below are held-out noisy variants of learned cue identities. This tests acquisition, interference and revision, not generalization to new semantic categories.

| Arm | A after A | A after B, before reminder | B after B | Final unchanged A | Final revised | Final all |
|---|---:|---:|---:|---:|---:|---:|
| fly_exact | 98.61% | 49.54% | 100.00% | 39.51% | 100.00% | 60.88% |
| fly_compressed8 | 98.61% | 49.54% | 100.00% | 39.51% | 100.00% | 60.88% |
| fly_shuffled | 98.61% | 49.54% | 100.00% | 39.51% | 100.00% | 60.88% |
| outcome_only | 98.61% | 49.54% | 100.00% | 39.51% | 100.00% | 60.88% |
| feedback_disabled | 98.61% | 49.54% | 100.00% | 39.51% | 100.00% | 60.88% |
| cumulative_ridge | 100.00% | 98.61% | 99.07% | 99.38% | 46.30% | 86.11% |

## What changed

The v17 ELM cell arithmetic now supplies features to an actual online associative matrix W. Learned outputs return through measured BANC feedback mappings and a shared-KC addressing proxy to redistribute the magnitude of local error updates. Every controller has exactly mean-one nominal gains. The encoder is developed on disjoint, unlabeled sensory examples and is then frozen. Durable learning occurs in W; there is no replay or historical classifier inside the fly arms.

- fly_exact: measured direct plus external-relay terms through four hops.
- fly_compressed8: direct and fourth-hop terms unchanged; second/third-hop terms use the measured eight-pattern nonnegative factors.
- fly_shuffled: permutes scaled teaching destinations within hemisphere, keeping the addressing proxy fixed.
- outcome_only: cuts learned-output feedback but retains outcome routing and modulation.
- feedback_disabled: ordinary normalized local delta update, gain one.
- cumulative_ridge: separate sufficient-statistics benchmark with additional quadratic feature memory.

Exact and compressed path terms are summed within an experience; anatomical hop counts are not assigned invented physical delays. Direct output/output and teaching/teaching recurrence are omitted in all arms in this first functional hypothesis, so this is not a complete simulation of the connectome.

The embedded compact mapping has 6.73% raw three-hop routing error. The common normalization and fourth-hop inclusion used by this learner give 18.56% effective operator error; the original 6.7% figure must not be substituted for this model's actual error.

## Interpretation

Read acquisition, old retention and revision jointly. If initial A acquisition is poor, a high retention fraction is not evidence of useful memory. A gain over feedback_disabled alone does not identify learned feedback: also compare outcome_only and fly_shuffled. Compare fly_compressed8 with fly_exact to assess functional preservation under compression. RLS may retain stale associations and adapt slowly under reversal; that tradeoff is intentional. No old-A label is provided between B training and the A-return probe.

Unsigned connectivity does not identify synaptic signs, eligibility or stabilization. Prediction subtraction, absolute-value teaching signals, signed standardized sparse ELM features, mean-one gains, label codes and the shared-KC address mapping are engineering assumptions. Different plasticity rules could change every result. The anatomical source also has unresolved metadata/edge count differences; this model uses the supplied pair-count tables, not an asserted complete physiological circuit.

## Exposure and memory accounting

- Seed 17: 48 cue identities; 96 unique online sensory inputs; 120 distinct input/outcome pairs including revision; 360 online labeled presentations per arm; 384 development training presentations plus 64 calibration presentations. Elapsed 1.1 seconds.
- Seed 29: 48 cue identities; 96 unique online sensory inputs; 120 distinct input/outcome pairs including revision; 360 online labeled presentations per arm; 384 development training presentations plus 64 calibration presentations. Elapsed 0.5 seconds.
- Seed 43: 48 cue identities; 96 unique online sensory inputs; 120 distinct input/outcome pairs including revision; 360 online labeled presentations per arm; 384 development training presentations plus 64 calibration presentations. Elapsed 0.5 seconds.

Feature caching avoids repeated physical encoder evaluation but does not discount repeated learning presentations. Held-out probes are read-only and counted separately in COSTS.json. Each candidate stores N×104 learned coefficients; ridge additionally stores N×N inverse covariance. Raw examples held by the experiment driver are never available to learner updates. Cell count, private states, fixed routing coefficients and learned weights are distinct resources.

## Verification

{
  "status": "passed",
  "checks": [
    "embedded_shapes_and_checksum",
    "read_only_prediction",
    "durable_associative_update",
    "mean_one_bounded_gain",
    "normalized_delta_baseline",
    "factorized_operator_identity",
    "nontrivial_margin_preserving_shuffle",
    "cumulative_ridge_equivalence",
    "independent_experience_state_reset"
  ]
}

The full executed configuration, source hash, embedded-data hash, per-seed stream hashes, matched differences and write diagnostics are included. No checkpoint, raw graph, feature dump or per-neuron trace is exported.
