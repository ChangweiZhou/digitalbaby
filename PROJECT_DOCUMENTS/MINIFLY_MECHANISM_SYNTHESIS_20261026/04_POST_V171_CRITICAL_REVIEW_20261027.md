# MiniFly — Critical review and corrections after Claude issue ledger, V170, V171 and V171-R1

**2026-10-27 · Documentation-only research note.** No experiment, new model adoption, or amendment to existing design pointers. Source heads: Claude `fd80e118761d5106d804f7b39f3cab40e424d5f8`; Codex `16623c73af101bbf9b017980795bfb957d444f0b`. The reports and code were reviewed, not rerun.

## 1. Withdrawals of earlier recommendations

1. **Temporal-credit-first is not a proven general prescription.** In native V171, EXACT3 and EXACT_RECORD improved teacher-forced loss, not complete own-byte/EOS answers; its E1 reference gate failed. V171-R1's new exact-one-step comparator confirms delayed recurrent *gradients exist*, correcting the earlier overstatement that short temporal credit was negligible throughout. Neither establishes that a new credit learner fixes native acquisition.
2. **The V166 memory query is not an automatically admissible architecture.** Its outer-product `r=Σ_i v_i(k_i·q)` amounts to unnormalised key/value retrieval and conflicts with the standing no-attention kernel requirement. Keep this as historical scaffolded BPTT evidence, not an approved integration candidate.
3. **“Spatial credit resolved” was too strong.** V165/V167/V169 learned useful transferable representations evaluated with BP readouts, but the continued fully local V169 own predictor scored 3.407 bits/symbol against matched Kneser–Ney 2.819 on B.
4. **KF-RTRL failed at the selected budget; “excessive variance caused failure” is not identified.** Variance, clipping, optimization and objective geometry were not independently separated.
5. **V170 does not prove missing role binding.** Strategy A's preregistered verdict INVALID; B NOT_RUN; the held-out order test confounds role use with unsupplied ordinal comparisons.
6. Earlier exploratory counting numbers contained an explicit +1 bit/symbol arithmetic error. Withdraw related neural data-efficiency claims.

## 2. Results with proper evidentiary boundaries

**Claude V169:** Emma→Ball and the Cross, LOCAL-A's learned H read by external BP probe 3.262 vs KC TOP 3.374. Matched persistent own-top learner 3.407 vs COUNT* 2.819 bits/symbol (0.588 deficit). No observed A forgetting: continuing on B improved A even without replay; replay's registered numerical pass is not proof that it prevented forgetting. Best training budgets and model adequacy remain unresolved.

**Claude V170:** fixed PN/KC re-entry yields trained-pair two-hop 100% at 0.723 bits on a structured fixture. Held-out pair two-hop 54.7% at 2.051 bits. The G1 control guard failed (confidence threshold although Q1 accuracy was 100%), so A is INVALID; B was NOT_RUN. Its instruction ordering cannot be interpreted as a pure role-binding test: e.g., the total orders 0<1<2<3<4<5 and 0<2<1<3<5<4 fit all *observed* pair comparisons when {1,2},{4,5},{0,3} are held out yet disagree on two held pairs. This is an inference from the task generator, not a separately executed analysis.

**Codex V171:** Native / EXACT3 / EXACT_RECORD taught NEW complete answers are 1/1/1 out of 16 in world 711 and 5/4/4 in world 712. Their lower teacher-forced loss does not imply complete-answer success; no E3 assessment was run.

**Codex V171-R1 (new corrective diagnostic):** On all taught 80, NATIVE/EXACT3/FULL are **4/4/4** in world 711 and **11/10/10** in world 712. EXACT1 was independently verified numerically. At the same saved states, the whole-parameter median relative EXACT3−EXACT1 difference is 2.74%, but within U/W/R/b recurrent parameters it is **44.67%**. Temporal credit exists; immediate gate parameters dominate the whole norm. This is **not an EXACT1-trained performance comparison**.

At fixed final native reads, independently optimized per-prefix legal [0.5,1.5] gates could obtain 47/80 and 57/80 complete answers; cap-aware outer boxes 42/80 and 51/80. These are target-informed oracles, *not attained learned policies*. The gap from actual 4/80 and 11/80 is substantial. V171-R1 also certified constrained convex per-prefix likelihood fits, but neither a common learned controller nor a causal E3 gain follows. Their special output-codebook alignment reduces rival conflicts, without eliminating shared-policy constraints.

**Corrected diagnosis:** native E1 still fails; the reader has genuine actuation restrictions on some saved sites, yet much of the actual-oracle gap could reflect controller reachability, conditioning, source dynamics, or optimization. The evidence does not select one explanation.

## 3. Claude's issue ledger — what should be retained

| Issue | Status and correction |
|---|---|
| **A1 — knowledge affects later computation** | Function remains missing in validated native core. V166 is a scaffolded BPTT witness; its query is attention-equivalent and presently excluded. V170 re-entry is non-gating evidence with INVALID verdict. |
| **A2 — fixed context trace** | Finite fading trace is limited, but no unique mandatory memory mechanism has been demonstrated. Learned recurrent alternatives remain possible. |
| **A3 — learned memory operations** | Genuinely open under the owner's *generic learning* requirement. Study markers and fixed writes can diagnose a mechanism only; they may not define the deployed learner. |
| **A4 — order/roles** | Code discriminates lag order, but structural role generalization has not been established. V170 held-token order was underdetermined by training. |
| **A5 — memory-dependent credit** | The equation dL/dq=Mᵀ(dL/dr) is valid for the historical r=Mq form; that *form* must be checked against no-attention. Reciprocal Hebbian symmetry only reproduces a transpose if birth, scaling, plasticity timing, decay and constraints match; it does not supply delayed credit automatically. |
| **A6 — structure/sample efficiency** | Re-entry shows a promising scaffolded descriptive gain, but V170 guard and unequal parameterization/training budgets preclude a general efficiency claim. |
| **T1 — local top** | Competitive deficit versus COUNT* remains; no converged local solution demonstrated. |
| **T2 — native output port** | V171-R1 differentiates some fixed-read actuation restrictions from a large attainable per-prefix oracle gap; common-controller reachability remains unresolved. |
| **T3 — birth wiring** | Imbalanced fan-out is a measurable design property, not a demonstrated dominant cause. |
| **T4 — approximate credit** | Failed KF evidence holds; causal attribution of variance or optimizer interaction is open. |
| **T5 — integration** | V171 attempted native integration but did not qualify; external successes do not automatically transfer. |
| **T6 — validity/accounting** | Preserve counting erratum, V170 INVALID and B NOT_RUN, local-vs-probe distinctions, missing original technical arrays, and version collisions. V171-R1 now adds audited diagnostics without overwriting V171. |
| **T7 — budgets** | Some arms are budget-edge, others overfit; neither limitless grid search nor one universal cap is justified. |
| **C1 — temporal credit** | Circuit-local computation is admissible in principle, but no validated memory-conditioned learner in the admissible native architecture yet. |
| **C2 — order generalization** | Need an identifiable fixture separating role placement from latent ordinal facts. |
| **C3 — digit semantics** | Generic byte learning of magnitude/value remains unproved; tally-input success uses different representation. |
| **C4 — counting deficit** | Quantified at 0.588 bits on matched V169 A→B; actual learner own prediction is the correct comparator. |
| **C5 — slow persistence** | No forgetting observed across related authors; replay not yet warranted as primary repair. |

## 4. The owner's binding goal, not a retrofit after outcomes

The user has clarified (recorded in Claude's issue ledger §7.4): **prescribe generic learning rules and generic components; the learner decides what is worth learning from input and environmental feedback.** For now, next-byte prediction *at every position* is the meta-task. Circuit-local communication and dynamic local connections are allowed; exact global BP is a reference, not the native candidate. Specific arithmetic/role rules, hand-gated study writes, fixed hop schedules and answer-site-only objectives are diagnostic scaffolds, not acceptable deployed learner features.

The standing [no-attention requirement](../NO_ATTENTION_KERNEL_REQUIREMENT_20261013.md) remains operative. Merely renaming a state-dependent key/value query as Hebbian memory or re-entry does not resolve that constraint. The exact operational boundary for normal native synaptic transmission versus attention-equivalent content-addressed reads needs an explicit equivalence audit.

## 5. Clarifying work — not an authorized experiment queue

**D1. Native action feasibility and shared-policy reachability.** V171-R1 has completed the independent-gate per-prefix LP and exact convex fits; **do not repeat that suggestion**. The remaining question is whether *one bounded shared controller*, trained by admissible rules and seeing no future answer, can realize useful gates across ordinary histories. Specify this as a new, prospective, adequately powered E1 reference; do not mislabel oracle feasibility as learnability.

**D2. Order identifiability.** Enumerate the training-consistent latent orders and test only facts logically implied by training when diagnosing role transfer. Unidentifiable held relations should be explicitly separated.

**D3. Generic objective and topology audit.** For any candidate, trace precisely the available signal **before and after** each next-byte arrival; ensure no target leakage, write-by-marker, addition-specific supervision, or hidden attention-equivalent memory query. Confirm the same rules apply to text and arithmetic.

**D4. Evaluation hygiene.** Version by `branch@commit/version` (Claude V169 ≠ Codex V169 design/V171 execution); preserve status INVALID, NOT_RUN, failed reference gate, and missing data rather than conflating these with negative primary outcomes.

**D5. Treat surprise-gated writing as a hypothesis.** Surprising random noise may not be worth storing; repeated useful information may become predictable. Test relevance selection independently before allowing surprise to stand for utility.

**Stop rule:** No new major architecture should be promoted from a loss-only improvement, free oracle, BP-probe representation gain, incorrect/underidentified task, or a reference that has not qualified. E1 complete own output, E2 retention of *relevant* learned knowledge, E3 never-taught transfer and actual CPU/feedback remain the project's decision endpoints.

## Sources

- [Claude issue ledger](https://github.com/ChangweiZhou/digitalbaby/blob/fd80e118761d5106d804f7b39f3cab40e424d5f8/minifly_note_issue_ledger_20261027/NOTE.md)
- [Claude self-review](https://github.com/ChangweiZhou/digitalbaby/blob/fd80e118761d5106d804f7b39f3cab40e424d5f8/minifly_selfreview_20261027/NOTE.md)
- [Claude V169 persistent learner](https://github.com/ChangweiZhou/digitalbaby/blob/fd80e118761d5106d804f7b39f3cab40e424d5f8/minifly_v169_native_dev_v1_20261027/REPORT.md)
- [Claude V170 re-entry](https://github.com/ChangweiZhou/digitalbaby/blob/fd80e118761d5106d804f7b39f3cab40e424d5f8/minifly_v170_reentry_dev_v1_20261027/REPORT.md)
- [Codex V171 original](https://github.com/ChangweiZhou/digitalbaby/blob/09bad998608fc95f37dff50fc54c540ac8a23e9b/minifly_v171_dev_20261026/REPORT.md)
- [Codex V171-R1 correction](https://github.com/ChangweiZhou/digitalbaby/blob/16623c73af101bbf9b017980795bfb957d444f0b/minifly_v171_r1_dev_20261026/REPORT.md)
- [Prior synthesis](README.md)

*Disposition: records corrected research interpretation. No model or experiment changed.*