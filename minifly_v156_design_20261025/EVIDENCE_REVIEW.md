# V156 evidence review

**Design preparation only.** Source inspection and file inventory were performed; no learner, checkpoint inference, feature extraction, LP or continuation was executed.

## Evidence retained

- V154 at [8cd3ad76](https://github.com/ChangweiZhou/digitalbaby/commit/8cd3ad76cbf13e51b0379ced9919b583650ca790): ten native groups, thirty correlated histories, two reused development worlds. Its immediate 848/15,360 responses and final 3/50 versus 0/50 pattern support partial acquisition and an unresolved transfer problem, not core adoption. The complete report and both accepted audits remain unchanged.
- Claude's V153 at [9026a970](https://github.com/ChangweiZhou/digitalbaby/commit/9026a970575322423e25ecac479fdc1e868b197a): the report and machine-readable rehearsal results agree on the quoted rounded BPS figures. The main run was skipped. This design does not convert text-rehearsal loss into an addition score or inherit another chat's run authorization.
- V151's valid sampled native self-response and V154's finite fixed-family diagnostics rule out neither output-family limitations nor cumulative interference. Individual beneficial writes are not a complete learning trajectory.

## Newly shipped V153 follow-up: Claude V155

Read-only inspection of [2501683d4867d2111a004f060edd28aaee67ffb4](https://github.com/ChangweiZhou/digitalbaby/commit/2501683d4867d2111a004f060edd28aaee67ffb4) found a completed [headroom report](https://github.com/ChangweiZhou/digitalbaby/blob/2501683d4867d2111a004f060edd28aaee67ffb4/minifly_v155_headroom_check_20261026/REPORT.md), its machine summary, and a separate [local-key design](https://github.com/ChangweiZhou/digitalbaby/blob/2501683d4867d2111a004f060edd28aaee67ffb4/minifly_v155_keys_design_20261026/DESIGN.md). The report's registered verdict is `ROOM_BUT_NOT_COUNTING`; the key design remains a proposal. The V153 addendum withdraws the immediate class-bias recommendation. Our earlier V155 name was provisional and is retired in favor of V156 to avoid overlap.

Verified summary: T1 raw-trace linear 3.4630767, T2 frozen-KC linear 2.9628954, T3 dense learned trace 2.6994702, T4 explicit-four-symbol learner 2.6246966, PPM-4 2.6062837 BPS. T3−T2 = −0.2634252 with line-bootstrap SD 0.0160984. One text, one seed per setting, calibration selection; T1/T2 reached 300 iterations without formal convergence. The registered report includes 1,559 CPU seconds across four attempts, no MiniFly learning records; evaluator epochs still have computational and data-reuse cost.

Accept the observed improvement and withdrawal. Do not accept the report's stronger prose as established: T2 is a best-found regularized linear fit, not a proof that all output changes are bounded by it; failure to beat counting by these references does not prove impossibility at this data size; train/held gaps do not isolate capacity versus data; T3/T4's different widths/regularization/input classes do not identify trace information loss. A bootstrap over lines is not seed replication. Dense backprop headroom does not establish sparse native headroom or local learnability.

Claude's proposed local-key study excludes arithmetic, native output/write integration and retention, and does not specify V154-style gradual warmup for its new key updates. Its gate failures would not prove the wiring or binary code is an absolute limit without a sound certificate. V156 does not run or rewrite that study, import its global references as a candidate, or add text as a new goal. The finite normalized-OBS decision question and native margin dynamics remain complementary, narrowly scoped questions.

## Corrections to make the proposal testable

1. **Saved features are normalized.** [features154.py](../minifly_v154_dev_20261025/features154.py) stores `v/max(1,||v||₂)` for RHO/X/OBS. Raw features were used transiently for exact-collision grouping. CQ/FQ feasibility on those caches cannot silently be called raw-native representability. Homogeneous no-bias decisions are invariant to positive row scaling; affine comparisons need more care.
2. **Saved states are endpoints.** [histories154.py](../minifly_v154_dev_20261025/histories154.py) preserves OLD_W/OLD_SH and final branch states. It does not preserve a model at every update or all earlier logits. New margin attribution requires a new, explicitly labeled short history. Use a new deterministic NEW order, not a replay claimed as historical reconstruction.
3. **A static witness is not native realizability.** The free map `Uφ` has a different reachable state family from native stores, which also evolve during query preparation. Feasibility only answers the declared finite, fixed-tap question. No exact collisions alone establish neither good conditioning nor a learned reusable representation.
4. **C16 rank does not settle decisions.** `rank(CP/4)≤15`, but V154's individual class-winner check passed all 257 classes. If the finite feature rows are independently interpolatable, even C0 can realize all taught labels. This can be a memorization witness. Do not call it arithmetic learning or a learning-rule success.
5. **BPS differences are not all argmax failures.** V154's full/C16 losses differed greatly while argmax accuracy was similar. Twelve identity-front batch fits were unresolved. No fit is an information-theoretic ceiling.
6. **V153's C16 already has a latent intercept.** Its [readouts.py](https://github.com/ChangweiZhou/digitalbaby/blob/9026a970575322423e25ecac479fdc1e868b197a/minifly_v153_bits_dev_v1_20261025/readouts.py) fits 16 latent offsets for both C16 and C16Q. C16Q adds 257 unrestricted class offsets. The comparison expands prior/calibration expressivity; it is not the introduction of the first possible marginal preference. V156's C0 deliberately matches V154's no-intercept probe, while CQ's class offset subsumes any latent offset.
7. **V153's chain has additional mismatches.** R1/ONLINE use seen classes plus OTHER; C16/native use all 257. ONLINE receives up to three passes, native one. Native temperature is calibration-selected. The native wrapper uses inherited per-bit teaching rather than V154 N_2B, and the inspected FastMemory event rule does not use V154's explicit `s(n)` ramp. Therefore 0.783 BPS is not a uniquely measured native-plasticity penalty. This design retains V154's gradual ramp and does not import V153's native wrapper as its parent.
8. **Explicit bias is not necessary for EOS collapse.** V154's q remains zero in the registered N_2B arms, while saved I_FAST OLD curves already show empty EOS. Shared native responses can encode a common preference. Conversely, V152 bias removal did not recover complete answers. Neither a bias ban nor a guaranteed prior repair is supported.
9. **State affineness is not margin affineness.** [teach_signed](../minifly_a_v3/src/stores.py) is affine in fast/slow state at a fixed parent and event. The operative inherited [expression](../minifly_a_v3/package/REFERENCE_SOURCE/minifly/iteration18/src/model73.py) clips gamma/MBON values. Strongest competitors can switch. Lane M therefore records finite actual effects rather than assumes an exact stationary gradient kernel.
10. **Matched passive is not idle.** Native no-write events still receive bytes, update nonplastic state, advance time and count observed targets. V156 includes a separate matched-duration idle fork, with exact order-dependent telescoping. It does not call their difference a unique cellular mechanism.

## Mechanism catalog corrections and restraint

The full human crosswalk identifies CELL-10 as dendritic nonlinearities, SYS-03 as pattern separation, SYS-08 as retrieval-mediated integration and SYS-14 as the TEM structure/content item. The attachment's broad grouping is useful, but those exact IDs must not be conflated.

The 254 register is a historical configuration census, not 254 independent theories. Eligibility, replay, availability, fast/slow and context allocation already have attempted neighbors. Their actual tested recipes, old ELM-related implementations and evidence scopes cannot be relabeled as missing native mechanisms. U034/SUPPORT-DIR remains an exact closed/diagnostic recipe. COV-R/U042 are historical engineering anchors, not evidence that every native rule-free wrapper inherited transfer.

The [44-paper bibliography](../PROJECT_DOCUMENTS/LOCAL_LEARNING_LITERATURE_20261025/README.md) checked primary pages/abstracts, not every supplement or reproduction. The human crosswalk has background-only and contested rows; some named entries were not freshly retrieved. This design uses the catalogs to constrain conditional hypotheses, not to justify importing a mechanism. No new literature effectiveness claim or broad algorithm ranking is needed for the two registered diagnostics.

## Sources and access

- User feedback attachment: SHA256 and original local path in `PROVENANCE.json`.
- Local immutable V154 source/results and V152 native adapter: file hashes in `PROVENANCE.json` and `ASSETS.json`.
- V153 report, readouts, native wrapper, settings, summary reducer, inherited event excerpt and machine rehearsal results: fetched read-only through the GitHub connector at the full pinned commit above. Cached source text stays under ignored `research_cache/`; no corpus was downloaded and no remote code was executed.
- Human crosswalk: read from Git object `8cd3ad76cbf13e51b0379ced9919b583650ca790:minifly_v144_addition_readiness_design_20261024/MECHANISM_CROSSWALK.csv`; original catalog commit `4c12621d0804d2495c2eacaebf4e17aaeb994771`.

No inference here identifies one dominant cause. The proposed outcome is a verified finite decision witness or certificate, together with an auditable account of native margin changes under a small competing-write history.
