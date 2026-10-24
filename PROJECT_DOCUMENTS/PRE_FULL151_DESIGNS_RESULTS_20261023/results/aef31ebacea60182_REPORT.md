# MiniFly A7 — source pooling, signed collateral and native upstream provenance

2026-09-20 · Frozen FULL_151 · Diagnostic complete; no learner promotion

**A7 resolves an ambiguity in A6 and finds a transferable average collateral component, but does not establish a safely removable component or a new persistent-learning core.** Source-specific means contain most of the useful source signal. Leaving one source out of a same-block mean does not isolate harmless interference. Holding out the whole block gives a useful offline direction on average, with substantial individual losses. The native upstream audit also shows that improving signed margin can worsen actual choices.

中文摘要：这轮有诊断进展，没有新模型性能进展。跨区块预测成分的联合离线删除提高了平均正确率，但仍伤害部分记忆；逐次更新中的余量改善也没有稳定转化为选择改善。A6 中“有用信息位于残差”的强解释需要撤回，因为按来源分别求均值后，大部分有用信息其实就在均值内。

## Design and evidence boundary

The supplied proposal was refined in three material ways before analysis: individual P/R attribution was separated from joint state removal; event-pooled and source-balanced donor means were both retained; and quantities at different stages were transported into slow-write units using captured recipient coefficients, without fitted scaling. The exact protocol is in the accompanying `DESIGN.md`.

The run uses the existing 64 development worlds arranged in eight disjoint blocks, the frozen A3 window selector, A4 K8 source rows, and A6 native states. FULL_151 is a candidate identifier, not a count of 151 neurons. A7 changes no genes, connectivity, learner rule, reader, world, or persistent state. It introduces no new connectome/specimen evidence. These are synthetic task diagnostics within the inherited anatomy-derived model.

At each Phase A checkpoint, for source c and role×context group g:

`S_cg = sum_t exp(-(T-t)/tau_s) xi_t`, `W_cg = sum_t exp(-(T-t)/tau_s)`, `P_cg = W_cg mu_donor`, `R_cg = S_cg - P_cg`.

The six donor definitions are source-specific; same-block all-source pool; leave-source-out event pool; leave-source-out balanced pool; out-of-block event pool; and out-of-block balanced pool. Designated-target rows are excluded from donor/source construction. An out-of-block fold uses the 49 non-target sources in the seven other blocks and no vector from its recipient block. The recipient supplies its own retained weight and frozen state for evaluation. This is offline prediction, not a locally available online estimate.

Every component is evaluated with the exact clipped native readout and both native choice gates. P/R Shapley attribution uses the background `native - S`. Deletion means `native - P` versus native. Joint removal evaluates `native - sum(P)` afresh; individual nonlinear effects are never added to manufacture its behavioral score. Signs are preserved throughout.

## 1. The A6 residual interpretation depends strongly on source pooling

For challenge writes, the mean cosine between retained source S and its own weighted mean is **0.99779**; mean norms are S=875.75, P=875.33 and R=56.67. The source-specific P contributes **+6.99178** to its own source's signed margin and R contributes only **+0.33921**. With A6-style pooling, P contributes **+0.53396**, while R contributes **+6.79703** to the same self population.

Thus the large useful pooled residual is largely a consequence of assigning source-specific structure to the residual. It is not evidence that a special residual-only memory mechanism has been discovered. The same-block pooled results reproduce A6 to numerical tolerance, including its independently audited joint accuracy effect.

The actual joint-removal comparison is:

| Mean used for P | All-active accuracy change, pp | Wrong→correct | Correct→wrong |
|---|---:|---:|---:|
| Source-specific | −23.36155 | 1,091 | 3,244 |
| Same-block pool | −23.57856 | 1,093 | 3,266 |
| Leave-source-out, event pooled | −23.01432 | 1,124 | 3,245 |
| Leave-source-out, balanced | −20.70313 | 1,235 | 3,143 |
| Out-of-block, event pooled | **+6.09809** | 1,294 | 732 |
| Out-of-block, balanced | **+5.83767** | 1,360 | 822 |

Each row contains 9,216 repeated fact×checkpoint decisions across 8 blocks, 64 cohort histories and 768 cohort×checkpoint cells. These are not 9,216 independent memories or trials. Both out-of-block constructions improve the block-average accuracy in all eight blocks; neither is selected as a winner. A6's approximately +6.11979 pp used equal weighting of block-pooled means. A7's +6.09809 pp event-pools donor events, while +5.83767 pp balances donor sources. The small change from A6 is a declared weighting difference, not a failed replication.

Same-block leave-source-out is not equivalent to source-independent nuisance isolation. In the challenge panel, its predicted P is harmful to its excluded source but useful to the other included sources: event pooling gives P self −0.48503 and other-source +0.49481. Removing these useful contributions jointly causes the large losses above.

## 2. Out-of-block transfer survives, with important context and tail limits

For challenge-source components, both out-of-block predictors have negative average signed P effects on self, the excluded designated target, and other source cohorts:

| Predictor | P self margin | P target margin | P other-source margin | R self margin |
|---|---:|---:|---:|---:|
| Event pooled | −0.12838 | −0.18938 | −0.19624 | +7.45937 |
| Source balanced | −0.13315 | −0.20000 | −0.20297 | +7.46414 |

For event pooling, challenge P is harmful to self and other sources on block-average margin in 8/8 blocks; deleting individual P improves their block-average choice accuracy in 8/8 blocks as well. This supports a separable **average** collateral component on this panel. It does not establish universal preservation of individual self memories.

The sign depends on context. Acquisition-source P improves average self and other-source margins in all eight blocks, so deleting it reduces those margins. Moreover, marginal choice changes and margin changes can have opposite signs. An unconditional instruction to remove all common means is unsupported.

Joint out-of-block removal retains the following harms:

| Tail measure | Event pooled | Source balanced |
|---|---:|---:|
| Cohort×checkpoint cells with accuracy loss | 21.74% | 22.01% |
| Cohort histories with net accuracy loss | 7/64 | 6/64 |
| Worst cohort-history accuracy change | −9.02778 pp | −9.72222 pp |
| Individual margin-change 10th percentile | −10.39822 | −11.35314 |
| Median margin change | +0.96699 | +1.03186 |
| 90th percentile | +12.35967 | +13.22935 |

The 732 and 822 correct→wrong decisions in the preceding table are therefore central results, not negligible exceptions. The complete signed distributions, source/destination cells and cohort histories remain available in the output archive. `SELF_FOREIGN_TAILS.json` includes quantiles down to the minimum; full cell tables retain each destination separately.

## 3. K2 provenance and native upstream state are now recoverable

The observer copies the return values of the actual `advance73` and event path, calls each original function once, and returns its result unchanged. It captures dx, raw drive u, actual allocation, and measured `w_slow = slow_post - exp(-dt/tau_s) slow_pre`, plus event metadata, timing, masks and native pre/post states. No guessed replacement update was run. The inherited path has no separate mutable eligibility trace; none was invented for this audit.

Block 0 was replayed plainly and observationally for K1, K2 forward, K2 reverse and K8 before expanding. All behavioral counts, final states and clock values matched exactly. Across all 24 observed K2/K8 branches, A6 behavioral counts and checkpoint states matched exactly; streamed D and its sparse transition-order reconstruction had zero discrepancy. The archive contains **27,648 K2 active source rows** and **16,896 selected native subevents**, including 9,216 at K8.

K2 source-group Shapley values reconstructed from the new source archive match A6 with maximum margin error 6.44e−15 and accuracy error 1.11e−16. Non-target passive floating-point residues are retained separately in transition ledgers. Regrouped source sums differ by at most approximately 3.96e−11; these reordered sums use the declared machine/state-scale tolerance rather than claiming bitwise equality. Exact unchanged operations were required to match exactly.

Archive correction: the first observation pass saved adaptation, alpha-fast and slow pre-state but omitted gamma-fast pre-state. A separate observational replay supplied the missing field in `_pre_gamma.npz` files, linked from `UPSTREAM_CAPTURE_AUDIT.json`. Each replay checked captured post-gamma, final state and behavioral identity; all passed. No scientific result or predictor used the supplemental field. The delivered capture code now saves pre-gamma directly. This execution used **60 native trajectories** including 24 completeness replays; a clean reproduction needs 36.

## 4. Cross-stage analysis gives a conditional localization, not a causal module diagnosis

One donor protocol was used throughout: hold out the entire block; group by context×delivered role×punished/unpunished presentation; preserve event and source-balanced weighting. Labels stratify the audit and never enter the learner. Primary results use the complete frozen K8 A3 windows. K2 is reported on the active subset of those windows, separately, without interpreting unmatched averages as a load effect.

The mappings into slow-write units are `P_dx = mean(dx)_donor × gain_recipient × allocation_recipient × decay`, `P_u = mean(u)_donor × allocation_recipient × decay`, and `P_w = mean(w_slow)_donor`. Recipient coefficients are captured from its actual update and validated against the native path. They are not re-evaluated under donor activity. Donor components can also extend beyond the current event's active support.

Consequently these are conditional downstream state decompositions. They do not simulate changing KC activity and allowing all feedback to respond; they cannot identify a unique upstream cause or declare the reader/teaching module defective.

K8 challenge-window results, with event pooling:

| Donor stage | P incoming margin | P foreign margin | Blocks with negative foreign P | Foreign deletion accuracy, pp | Foreign gains / losses |
|---|---:|---:|---:|---:|---:|
| dx, transported | −0.013072 | −0.000695 | 6/8 | −0.003617 | 89 / 96 |
| u, transported | −0.013607 | −0.001223 | **8/8** | −0.004650 | 86 / 95 |
| Actual slow innovation | −0.009787 | −0.000087 | 6/8 | −0.008267 | 53 / 69 |

With source balancing, foreign P is negative in 6/8, 8/8 and 5/8 blocks respectively; foreign deletion accuracy changes are −0.003100, −0.005167 and −0.005167 pp. Both constructions therefore retain the same caution: small average margin improvements do not imply better choices. For incoming facts, all three stages have negative P and positive residual R margin in all eight blocks, but deletion changes only **one** of 2,304 repeated challenge decisions from wrong to correct. Foreign counts cover 193,536 repeated fact×subevent evaluations.

The most consistent challenge foreign sign is visible in the transported raw-drive u representation. It is not a pattern appearing only after slow storage. However, dx transport already contains the recipient's teaching gain, and the stages are not causal interventions with equal realized amplitude. The data do not establish that u is where the problem originates. K2 gives mixed foreign-margin signs, further limiting any load-independent claim.

The apparent difference from the +5.84–6.10 pp Phase A result is not a contradiction: Phase A removes retained, accumulated components at later checkpoints; the cross-stage assay removes one predicted subevent innovation at its native post-event state, uses finer groups, and scores incoming/protected truths. Neither is an online trajectory with the learner repeatedly modified.

## Acceptance, decision and artifacts

`ACCEPTANCE_CHECKS.json` passes source/hash locks, unequal-count mean definitions, exclusion/indexing tests, reconstruction, exact P/R closure, native-probe invariance, A6 regression and instrumentation identity. Cross-stage maximum readout error is 1.07e−14 and Shapley margin closure error 3.55e−15. These are technical qualifications; no accuracy, margin, cosine or harm threshold selected an arm or granted scientific promotion.

**Decision: keep FULL_151 unchanged and stop at the diagnostic.** A7 supports average source-independent collateral on the reused challenge panel, rejects a universal harmless-mean interpretation, and closes the K2/raw-upstream provenance gap. It does not establish an online correction that preserves source benefit and individual choices. No optimizer, evolutionary search, turnover, memory slots or new learning policy was introduced.

The unresolved question is now narrower: how much of the accumulated offline gain can coexist with preserved individual choices under a valid online information boundary? That remains untested. The present results do not justify jumping from the u-stage sign to a new teaching formula.

Eight blocks are the reporting units. Donor sets overlap and these are previously reused development worlds, so 8/8 signs are not an independent confirmation study. Millions of readout evaluations do not enlarge that replication count. Margin units are simulated readout units, not newly measured fly physiology.

The compact result bundle contains this report, decision, final tables, figure, frozen configuration, new analysis code, qualification records and manifests. Raw ledgers/window arrays (2.43 GB, decimal) and full per-cell outputs stay separately in `CORE_RECOVERY_workstreamA7/source_collateral_v1/`; their size is required for independent source and upstream reconstruction. Parent data, model assets, historical code, environments and caches are referenced rather than copied. `PARENT_ASSETS.json` is a delivery dependency inventory; before/after integrity claims apply specifically to the preflight inventory and inherited hash locks.

![A7 diagnostic summary](/Users/pencilbard/Downloads/second_try_copy/CORE_RECOVERY_workstreamA7/source_collateral_v1/key_findings.png)
