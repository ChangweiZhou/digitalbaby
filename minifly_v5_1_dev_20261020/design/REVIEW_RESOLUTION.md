# v5 review → v5.1 revision

This records written resolutions to the [ten v5 findings](../minifly_v5_design_review_20261020/REVIEW.md). **Resolved in the design does not mean implemented or experimentally qualified.** No new model training or behavioral measurement was performed.

| Finding | Resolution in [DESIGN.md](DESIGN.md) | Required future evidence |
| --- | --- | --- |
| F01: learned-value mean can absorb memory | §5 replaces it with a stateless zero-value counterfactual at matched activity/adaptation. Both teaching and generation use the same pre-event margin. | All-bit/EOS direction, constant/imbalanced target, passive-control, saturation and readonly port tests; then complete taught replies. |
| F02: listening leakage and free information | §2 globally excludes held semantic orbits before all sources. §8 is addition-only; all heard answer bytes and repeat presentations are counted. | Source-level split/provenance audit across listening, teaching, replay, calibration and fitted-head extraction. |
| F03: stable collapse/oversplitting can pass | §4 separates identity formation from context accumulation, permits unused capacity and requires identity confusion/stability plus noncollapsed predictive compression diagnostics. K=64's limit is explicit. | Frozen winner/confusion tables, class/transition/run statistics and prelocked evaluator description-length calculation. No class labels reach the learner. |
| F04: format gate contains arithmetic | §6 separates isolated-port and integrated-encoder taught-reply checks; §§10–11 separate syntax, length and exact arithmetic. No unseen correct-length prerequisite. | Own EOS on finite generic replies; immediate/delayed taught arithmetic and separate semantic-length curves. E1 remains diagnostic. |
| F05: dose extrapolated from attenuated writes | §6 corrects the v4 measurement interpretation and specifies a finite response calibration and smooth fast-write ramp. §8 selects a dose only after measured recurrence/cost projections. | Chosen rate/ramp, actual response curves, real recurrence and resource lock. No automatic 100k-record or larger-brain escalation. |
| F06: closure/output timing ambiguous | §3 retains boundary bytes, leaves `=` open at the first answer frame and distinguishes reply step from boundary age. Target arrives after frame preparation. | Identical prefeedback frames when targets change; byte-by-byte event trace and state hashes for training/generation. |
| F07: ridge is neither an upper bound nor cheap | §10 makes fitting evaluator-only, streamed and capped at 65,536 cells, 1k/3k prefixes, 600 CPU s, 2 GiB RSS and 128 MiB cache. | Solver/source lock, convergence status, cost projection and complete label/compute ledger. No dense feature/Gram matrix or native-capability claim. |
| F08: gate, spacing and freeze undefined | §§6–7 give signed q, strict zero allocation, symmetric signs, time/identity rules, bounded novelty ring, record-end confirmation and replay permissions. Representations really freeze. | Zero-floor/sign/spacing/contradiction/repeated-identity fixtures; exact checkpoint continuation and q/fast/slow utility accounting. |
| F09: random sum mislabeled conjunction | §4 names the primary activation honestly as inherited weighted combination features. It specifies generic locality and matched lesions; no arithmetic columns or untested dendritic AND. | Exact native rescaling/wiring/competition lock, actual selectivity/activity measurements and full numerical bridge checks. |
| F10: causal E3/retention not locked | §§2,7,9,11 define shared-D0 N_OLD, matched SH_OLD, OLD/rest/NEW/rest chronology, recitation controls, own-output scoring, world-level statistics and dual efficiency axes. | Fresh frozen confirmation after engineering/development, if separately authorized. No partial-metric promotion or automatic adoption. |

## Deliberate changes from received v5

- Keep the native lineage and learned sequence stages, while distinguishing inherited primitives from the changed end-to-end model.
- Retain gradual fast-write warmup pending calibration; full-strength fast writes are an unproved alternative, not the definition of safety.
- Keep one initial KC size. Defer the larger population grid and all non-addition tasks.
- Remove unspecified extra slow downscaling. Replay refreshes fast memory only; it cannot create q confirmations or new slow writes.
- Treat q thresholds and finite resource caps as explicit engineering hypotheses/limits, not empirical biological constants or arithmetic success standards.
- Require better retained exact unseen addition per samples **and** learner CPU, with causal controls. A tradeoff or no advance is reported honestly.

## Still pending before execution

The native asset/readout/precision lock, exact M4 wiring and threshold implementation, fixture roster, smoothing/description-length formula, streamed solver and measured dose/resource lock require future implementation work. They are listed in [SPEC.json](SPEC.json). Those pending locks cannot be filled using confirmation outcomes. This design does not authorize that work or assert that the mechanisms will pass it.

The original memo, historical v4 result and review are preserved. The planning pointer identifies v5.1 as a **design**, while v4 remains the last completed technical exercise.
