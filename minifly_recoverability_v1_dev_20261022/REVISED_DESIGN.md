# RECOVERABILITY V1-R1 — preserve payload; learn stopping separately

22 October 2026. **Post-cycle proposal, not executed.** V1 remains a completed negative pilot; this document does not redefine its endpoint or turn a posthoc recombination into a successful arm.

## Why revise

The [complete V1 pilot](REPORT.md) separates two failures:

1. On STD final NEW, direct payload-only exact is 61.72%, but the joint head reduces it to about 20%. The head improves EOS ordering without reliably locating the fixed decision threshold. Its shared payload/EOS objective can lower loss while damaging a usable payload.
2. On STD OLD, payload-only exact falls from 54.69% after OLD teaching to zero after NEW. The matched NEW-write-off continuation retains 53.91%. Fixing the readout cannot be assumed to fix this write-dependent interference.

The existing 5,801-parameter readout family failed its declared joint criterion. Do not enlarge it, tune on assessment labels, add replay, alter native feedback strength, or change the arithmetic track to obtain a positive result. The immediate question is narrower: can a stopping learner improve complete recall without destroying native payload signs?

## One proposed intervention

Keep the original frozen V1 heads and direct codec as references. Add **STOP-ONLY**, a generic external readout with the same 171 inputs and tanh(32) hidden layer, but only one learned EOS residual. It has 5,537 trainable parameters. At slot j:

`EOS_logit[j] = normalized_native_mean[j, EOS] + MLP_EOS(response_tensor, position_j)`

Payload bytes come directly from the signs of the eight original native averaged margins at that slot. Positive normalization cannot change those signs. The head cannot alter a payload bit. It receives no key, label, true length, OLD/NEW identity, association directory or arithmetic feature. EOS is generated at logit > 0; ties mean continue. At slot 8, a payload proposal is truncation, never a forced EOS. Empty values use the same learned decision at slot 0. No per-query threshold or evaluator-forced length is permitted.

Use MEAN observation as the proposed primary view: SEP supplied no joint advantage at STD in V1. Keep RAW/STD paired because normalization substantially changes payload storage. This is a smaller experiment, not a claim that separate native outputs contain no information. Do not simultaneously add a learned payload correction or native protection mechanism.

## Unchanged native contract

Reuse the exact V1 native sources and anatomy, `gated=False`, nine compartments, `DT=30/14`, no replay, original averaged native residual and EOS payload mask. Preserve

`a(n)=(1/16)*(1/16+(15/16)*min(n/1024,1))`.

Keep target-independent 16-byte-key surrogate mapping, the frozen 2,048-vector calibration, side-specific positive top-k, RAW/STD definitions, BEGIN/common-horizon SNAP reads, two values of each length 0–8 per cohort, eighteen OLD/eighteen NEW keys, fourteen passes/cohort, and both one-hour simulated rests. External EOS never feeds native learning. No arithmetic rule, attention, Transformer, ELM, task-specific representation or raw-record backend is added. The addition experiment pointer and sources remain unchanged.

## Training and the use of existing artifacts

Previously exposed V1 development/assessment labels are diagnostic data now. They cannot become a fresh test. A software feasibility check may fit STOP-ONLY on the already-declared twelve V1 training episodes and inspect existing traces, with all such results explicitly exploratory; **no such fit was performed in this delivery**.

A subsequent prospective test must freeze the EOS head and all choices before generating fresh assessment values. Use the six original equal-weight phase pools and all three paired initialization seeds, without best-seed selection. Only EOS targets through the observed end are supervised; post-EOS slots remain masked and no payload gradient exists. Retain NumPy AdamW settings (lr .001, 100-update linear warmup, beta .9/.999, eps 1e-8, decay .0001, gradient norm clip 1), batch 32, final iterate, and ceilings of 2,000 updates / 2,000,000 binary terms including final loss measurement. Report the actual smaller EOS term count; equal ceilings are not equal compute.

Before any fresh native run, freeze new collision-checked seed namespaces, complete dependency closure, initialization, stream hashes and resource admission. Do not reuse V1 assessment seeds or silently increase the roster. The proposed fresh assessment size remains four counterfactual pairs with two separate value lives, two native storage configurations and unchanged W/N/SH/S_NEW controls: 28,224 native applications / 141,120 events. A separate implementation authorization is needed; this revision launches nothing.

## Qualification before testing

- Preserve every payload sign exactly for arbitrary responses, including zeros, empty values and maximum-length values. Audit that a head update can change EOS only.
- Verify direct/MEAN parity, BEGIN/reset/read-only state identity, native source hashes and gradual coefficients.
- Finite-difference the scalar EOS head at nonzero residual parameters; test empty/nonempty masks and stop/cap behavior.
- Reject any input path from labels, true lengths, keys, cohort/seed identifiers or stored outputs into candidate inference.
- Persist all native calibration/address assets and checkpoints. Log early acquisition probes in training as well as evaluation. Record complete source closure and phase timing before a new run.

## Endpoints and stopping decisions

Keep final OLD and final NEW complete nonempty bytes **plus own EOS** as co-primary descriptive endpoints. Also report all values, per-length exact, empty defaults, bit/byte accuracy, early/late EOS, truncation, and forced-length payload diagnostics. The latter diagnose the storage; they cannot substitute for generated values.

Average three fixed seeds within life, then two lives per assessment pair. A positive joint pilot still requires both cohorts to beat the direct codec and N/SH/ZERO/FOREIGN in every pair. Compare STOP-ONLY against the unchanged V1 joint head and the explicitly posthoc payload/EOS recombination, but do not select a threshold or seed on assessment. Report the exact table and training-only pooled prior as evaluator references. Four pairs remain exploratory, not a statistical confirmation or deployment qualification.

The posthoc recombination raised STD NEW to about 13% while OLD stayed zero; that motivates the intervention but predicts no OLD rescue. If stopping improves NEW while OLD remains poor relative to S_NEW, report failure of the unchanged joint criterion and stop readout expansion. Then design exactly one native interference intervention in its own arm, with its mechanism and retention/plasticity costs justified before execution. Do not stack it into this test.

If STOP-ONLY harms payload accuracy, it violates its invariant and is an implementation failure. If it preserves payloads but not stopping, reject this bounded EOS learner; do not automatically add a larger decoder. Any success remains E1 known-key taught-value recovery, not unseen-addition generalization or a working learned key interface.

## Delivery boundary

The present delivery completes V1's three-cycle exercise and ships this revision. It does not execute V1-R1, alter native warmup, change the addition track, or authorize a host campaign. GitHub publication of the completed work is separately authorized by the user's subsequent “Push the results” request.
