# Final independent audit

## Verdict

PASS as a completed, negative development preflight with three sequential prospective design–audit–small-test cycles. FAIL for full-run eligibility. No fresh-world behavioral result is asserted, and this audit authorizes no full launch.

The candidate A equation was fixed before cycle 1, remained unchanged through the three cycles, and failed its static and exact finite-horizon learning requirements. Candidate B was separately fixed before cycle 3 and failed its development screen. Neither failure was rescued by parameter selection or relabeling an auxiliary diagnostic as the primary gate.

## Independently checked evidence

- Read all three prospective specifications and wrote their pre-test audit files before their tests. Cycle 3's two conditional requirements were resolved and reviewed before execution.
- Reviewed geometry.py, finite_horizon.py, calibration.py, cycle2.py, cycle3.py, history_data.py, launch_gate.py, and validation tests. These separate label-blind geometry/calibration from evaluator labels and item-basis Jacobians.
- Ran design/verify_final_evidence.py. All 128 original H/J/T/T_OFF receipt SHA256 values match the provenance manifest. Every bundled development field matches the corresponding original receipt exactly; extracted W history records match their original order and fields. The bundle hash and in-bundle provenance match the external manifest.
- Independently recomputed cycle 3 corrections, decisions, repair/break counts and the count identity: raw 202/512, B 186/512, 38 repairs, 54 breaks, so 202+38−54=186.
- Verified cycle 1 and cycle 3 portable replay artifacts equal their original result artifacts exactly after removing only the resources field. The packaging path/extraction change therefore preserves the observed scientific output for those replayed cycles.
- Reviewed the recorded final test output: 13 tests passed. This is a log review, not a second independent pytest execution or an independent rerun of the whole simulator.
- No unopened minifly_a_v3 behavioral outcome was inspected. The source-only FE0 birth/encoding comparison does not require such outcomes. Existing response_mechanisms histories were used only as authorized development data.

Machine-readable verification is in design/FINAL_EVIDENCE_CHECK.json; the reproduction script is in design/verify_final_evidence.py.

## Dynamics and geometry review

The ordered operator decays the bank before each teacher, includes subsequent residual-update attenuation, and applies final query-read-time decay. Its teacher injection uses the fixed 32 old/new item basis and is mapped to the actual four-channel +/-1 targets only on the evaluator side. A separate direct four-channel trajectory includes clipping and agrees to at most 4.44e-16; no direct clipping was recorded. This supports algebraic/implementation consistency, not independent prospective accuracy prediction.

The static minimum normalized interaction eigenvalue is about .00089–.00096, below .00928686. Exact old_end minimum symmetric interaction gain is about .01808–.01842, below .20. Final gain is about .0020446–.0020722, below .02. Each is a binding failure. Singular values, signed target gains, cross-component operator norms, old/new teacher blocks and exact conditional bank score tensors are retained, avoiding substitution of eigenvalues for task-relevant gain.

Native comparison uses the actual sparse KC encode_sparse output from an immutable native birth and FE0 sensory processing. Candidate queries are independent clones of their continuing history. Clock-origin differences between reset native panels and candidate-history panels do not imply additional native learning; the native sensor is reset and evaluated using the same elapsed cue/read timing. The shared-geometry criterion removes the centered isotropic identity component, tests strength and residual cosine, and rejects the identity negative control. Four candidate snapshots pass this narrow geometry test. This does not establish behavioral relation transfer or prove that every learned shared direction is preserved.

## Calibration and prediction review

B's anchor uses only the available prefix, including before the 16th entry; empty history returns zero. It uses pre-teacher raw outputs, literal times and a single query's uniform cue embedding. Evaluator labels and the oracle panel mean enter only scoring. Decision-centered baseline MSE and recomputed best-other margins avoid credit for irrelevant channel-common shifts or a stale rival.

The negative result is development-only: −3.125 percentage points, paired 95% interval approximately [−7.29181,+1.04181] points, mean margin shift −.0595473, and baseline MSE .252833 versus H .0809176. This rejects this frozen surrogate under the stated screen, not every possible causal calibration model.

There is no independently derived prospective TOTAL native-plus-bank score/margin forecast. Exact conditional added-bank prediction is explicitly narrower. Thus NO QUANTITATIVE THEORY PREDICTION for future total scores/accuracy is the correct designation, and it independently blocks a fresh full run. No statement here is a fresh-world falsification of the broader theory.

## Scope and limitations

The eligibility script is a digest-bound checklist, not a robust evidence-validating launch authorization system. It does not reconstruct numerical evidence from arbitrary fabricated true-valued manifests. No simulation-launch implementation exists. The failed actual gates must remain visible in the published decision.

Resource records are completed-process measurements and commands used single-thread/wall-time constraints; continuous OS RSS enforcement is not certified. No new source audit claim is made for future edits after this review. This audit covers the observed development artifacts and code, not publication success, remote commit identity, future source-lock completeness, or a new scientific run.
