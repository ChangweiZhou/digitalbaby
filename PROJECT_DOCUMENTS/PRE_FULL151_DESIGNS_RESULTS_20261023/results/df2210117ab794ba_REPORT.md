# V72 / Iteration 17 — Receiver observability and persistent reversal competition

14 September 2026. Completed locked development experiment; **no new learner promoted**. Keep the unchanged full 5,177-KC iteration5/bi learner instantiated by iteration10/reference, with the frozen iteration8 reader.

**本轮结论：** 按真实输出增益调整接收位置有小幅收益，但未通过全部选择保护条件；共享 MBON 邻域没有提供可靠优势。更有价值的定位来自修订：完整历史下 32/32 当时成功，一天后 22/32；省略该关联的原始写入后，一天后 30/32。次日分数差异主要落在原有状态的表达贡献，后续写入贡献接近。V72 缩小了问题范围，但没有解决持续学习，也没有证明任何新的果蝇生物机制。

## 1. What was refined before running

The supplied `V72_design_merged_refined.md` and the local `v72_review/REPORT.md` were proposals to evaluate against the actual implementation. Their hashes are retained in the lock. The final [protocol](PROTOCOL.md) was frozen after startup checks and before campaign outcomes.

| Proposed direction | Implemented refinement and reason |
|---|---|
| Synaptic headroom / homeostasis | The parent has no per-coordinate slow-weight upper/lower bound. Its **pooled outputs** clip. Intervention caps cannot establish biological dynamic-range exhaustion. No homeostatic learner was added. |
| Neighborhood information | Recover four individual MBON14 partners from the **same FlyWire 630 specimen**, rather than taking similarity of the rank-one pooled output. No cross-specimen identity joins or annotation/version replacement. |
| Oracle receiver labels | Use feasible, equal-mass exchanges and restricted-oracle exclusion regret. An optimum's allocated mass is not a safety label; exclusion can be impossible solely because of caps. |
| Old versus new memory | Opposite labels on the same pair have opposite net scores. Decompose pre-reversal state and subsequent realized writes through the actual nonlinear expression; do not label those algebraic factors independent biological memories. |
| Reuse/history traces | Correctly counted unsigned USE duplicates V71 H. ACT and decayed COUNT are offline diagnostics only. |
| Candidate search | Exactly two prespecified rules, GAIN and NEIGHBOR. Fixed ridge diagnostics cannot become candidates after results are seen. Fresh histories are conditional on all development gates. |

These changes avoid three false conclusions: a solver cap is a synaptic bound; rich connectivity automatically supplies safe allocation; and successful immediate reversal establishes durable revision.

## 2. Experiment and integrity

Sixteen existing V67 histories (6701001–6701016), loads 48/96, two presentations at NEW bout 6 or REVERSE bout 12 give **128 assigned common-prestate snapshots**. The prefix is original full-SPACE; each treatment changes one event at a prespecified 90% transport budget. The 20 arms include PARENT, NO_WRITE, PROP, BALANCED, OPP, GLOBAL, two candidates, four score shuffles per candidate and four anatomy shuffles. We evaluate isolated 300-second/day endpoints and continue the remaining episode using PARENT dynamics before the same delays. This continuation measures consequences of one intervention, **not repeated autonomous deployment**.

There are 10,240 snapshot/arm/phase/delay outcome rows; all assigned cases remain. The separate reversal block contains 32 load/history assignments, three branches, four bout checkpoints and three delays: 1,152 endpoint rows. The independent replication unit is the **history**, not a KC, pair, presentation, load or shuffle. Intervals use 5,000 paired history bootstrap draws; diagnostic feature comparisons are exploratory and unadjusted for multiple comparisons. These histories had already been studied in earlier versions, so freezing V72 does not make them fresh validation.

| Check | Result |
|---|---:|
| Startup checks | 55/55 |
| Final integrity checks | 266/266 |
| Original state / H checkpoint discrepancy | exactly 0 |
| Maximum discrepancy from V71 reference outcomes | 3.55e-15 Hz |
| Maximum summarized net transport mismatch | 3.55e-15 |
| Maximum exact-exchange versus local linear change difference | 2.67e-15 Hz |
| Expression decomposition identity error | 7.11e-15 Hz |
| Oracle nonlinear validation rejections | 0 |
| Missing assigned snapshots | 0 |
| Frozen unchanged parent assets checked | 287 |

All 2,465 small feasible exchanges remained within the corresponding linear output region to numerical resolution. Startup also checked large clipping-crossing perturbations against independent cloned model probes; exact clipped algebra was used uniformly for labels. The smallest exchange mass was 0.002446, so these derivatives were not obtained by dividing by machine-scale perturbations. Seven base LPs were **restricted-infeasible**, not numerical tolerance failures; two cases had no transport. No post-launch scientific amendment or outcome-selected rerun was used. One artificial-state startup fixture initially violated H bookkeeping and was corrected before the lock; details are in [software development log](checks/software_development.md).

Two single-threaded workers used 257 seconds of summed campaign time, about 2.2 minutes in parallel. Peak recorded process RSS was 566,853,632 bytes (541 MiB), not total simultaneous machine memory. No new data download or large BANC mother-table rescan was needed for this question.

## 3. Real anatomy supplied extra directions, but the neighborhood rule failed

The frozen FlyWire graph contains two individual MBON14 partners per hemisphere. Their KC counts sum back to the existing pooled alpha3 interface to numerical tolerance. Among KCs with positive output support there are 904 left and 866 right; the four-column normalized factor has rank two on each side, with 103/121 distinct rounded profiles. This is an operator rank, **not a memory-capacity estimate**.

The neighborhood uses shared postsynaptic partners: inverse partner-degree weighting, row normalization, then a row-stochastic affinity computed via the small factor. It includes self. It is neither physical distance nor a measured shared molecular resource. Four matched row permutations preserve hemisphere, archived KC type, positive output support and gain bins; they do not claim exact individual-degree preservation within bins.

GAIN water-fills `abs(pre_slow + raw_write) × normalized_output_gain`. NEIGHBOR water-fills the neighborhood average of the same absolute state. Both keep original donors, eligible recipients, caps, signs and per-group mass. No new mutable state is needed. Four within-group score shuffles preserve the exact priority multiset, but fixed caps mean realized allocation concentration need not match; concentration is reported.

**Worst-old-damage benefit versus BALANCED, isolated +24 h** (positive is better; Hz, paired 95% intervals):

| Load / case | GAIN | NEIGHBOR |
|---|---:|---:|
| 48 / NEW | −0.00045 [−0.00359, 0.00240] | −0.00425 [−0.00877, −0.00056] |
| 96 / NEW | 0.00306 [0.00016, 0.00648] | −0.00063 [−0.00500, 0.00475] |
| 48 / REVERSE | 0.00348 [−0.00063, 0.00767] | −0.00306 [−0.00644, 0.00011] |
| 96 / REVERSE | 0.00837 [0.00470, 0.01273] | −0.00079 [−0.00409, 0.00258] |

GAIN improves mean old damage in all four cells and lowers target actual margins by 0.0126–0.0187 Hz. Isolated target choices and PARENT-correct old losses match BALANCED. However, it fails **3/64** locked conditions: the minimum worst-damage effect at NEW/48, the pooled NEW interval against BALANCED, and continued old accuracy at REVERSE/48. That last difference is −0.0665 percentage points [−0.1995, 0], attributable to one history (6701003). PARENT-correct loss remains unchanged: the total-choice guard also counts a choice BALANCED recovered. This is a small observed harm and a failed prespecified screen, not a claim of statistically established population harm.

Pooled within history over loads, GAIN's worst-damage benefit is 0.00131 [−0.00006, 0.00279] Hz for NEW and 0.00593 [0.00316, 0.00903] for REVERSE. It beats its score shuffles in both cases. This is evidence that output-gain mapping can matter within the assay, not permission to ignore its failures.

NEIGHBOR fails **22/66** conditions. It has no clear advantage over either score shuffles or anatomical-operator shuffles, and pooled NEW worst damage is worse than BALANCED by 0.00244 [0.00035, 0.00455] Hz. Extra anatomical directions exist but this shared-partner averaging rule does not use them successfully.

![Candidate effects](figures/candidate_effects.png)

The figure is the isolated endpoint; the continued-choice failure is documented above and in [paired contrasts](results/paired_contrasts.csv). Every gate is in [progression gates](results/progression_gates.csv). Neither rule qualifies; **the fresh bank was not opened**. The accepted model is unchanged. No repeated-use safety or physiology validation is claimed for these candidates.

## 4. Safe placement is neither uniquely specified nor easily predicted here

Across 712 group opportunities, 179 have a single receiver. At full mass 299 groups have allocation freedom; at 90% mass 533 do. Exact exchange risk varies in 530 groups, which carry 91.2% of the tested transferred mass. This mass fraction is descriptive; it is not an additive attribution of worst-memory damage.

We produced 2,644 recipient records: 2,465 feasible equal-mass exchange labels and 179 no-exchange entries. Risk is the maximum harm per exchanged mass divided by protected old-choice slack, with the prespecified 0.05-Hz floor. Removing compensating mass elsewhere is essential: this is **risk relative to those compensators**, not an intrinsic danger label attached to a neuron.

The fixed-90%-budget, fixed-output-region guarded minimax LP validates 119/128 base solutions, with seven restricted-infeasible and two no-transport cases. This is not the same question as V71's maximum feasible fraction; do not replace V71's full-dose counts with these numbers. Of the 2,644 possible recipient exclusions:

| Exclusion status | Count | Interpretation |
|---|---:|---|
| Required by recipient capacity | 1,036 | Other recipients cannot carry the specified mass; no biological indispensability inference |
| Base unresolved / infeasible | 104 | No usable base optimum for the comparison |
| Validated exclusion | 1,504 | Both allocations checked against the original nonlinear model |
| Positive regret >1e-7 among validated exclusions | 256 | Removing that coordinate worsens the restricted minimax optimum |
| No material positive regret among validated exclusions | 1,248 | Alternative allocations reach essentially the same restricted objective |

Most feasible exclusions have no material regret, while a minority matter. Safety therefore does not map uniquely to one oracle allocation vector. Simultaneously excluding several zero-regret recipients need not preserve feasibility or the objective.

Four fixed ridge scopes predict `log1p(exchange risk)` with leave-one-history-out fitting, history-balanced weights and training-only standardization. The scopes are LOCAL; GLOBAL (active-population aggregate and fixed interactions); NEIGHBOR (individual-partner profiles and neighborhood field); HISTORY (H, opposing mass, ACT and COUNT). They never see old cues, old margins, association IDs, load/case/bout labels or future outcomes as inputs.

| Load / case | LOCAL R² | GLOBAL R² | NEIGHBOR R² | HISTORY R² |
|---|---:|---:|---:|---:|
| 48 / NEW | 0.008 | 0.007 | 0.010 | 0.113 |
| 48 / REVERSE | 0.025 | 0.035 | 0.051 | 0.105 |
| 96 / NEW | 0.032 | 0.027 | −0.001 | −0.024 |
| 96 / REVERSE | 0.010 | −0.010 | −0.024 | −0.086 |

These are averages of per-history R² relative to each training mean, not a single pooled regression R². Negative values mean worse prediction than that baseline. Within-group risk ranking remains weak (roughly −0.075 to 0.238 across scope/cells). Every incremental NEIGHBOR-versus-GLOBAL MSE interval includes zero. HISTORY has a positive incremental MSE interval only at REVERSE/48; that is one exploratory comparison. Mean MSE and mean per-history R² can point differently because the latter divides by each history's different baseline error.

![Information scope](figures/information_scope.png)

History carries some conditional information, but neither history nor recovered partner profiles produce a robust predictor across these conditions. This bounds the **tested features, labels and fixed ridge family**. It does not prove that all local information is insufficient, and it does not justify tuning H/OPP or fitting a richer learner after this outcome. Full ridge fits are included solely for audit.

The donor sanity check also prevents calling donors solved. Only about 35.9–41.4% of selected transferred mass lies on coordinates whose individual 10% write removal improves the current worst old endpoint. The selected donor average benefit is positive, but removal also lowers target margin; at REVERSE/96 its average benefit is below that of unselected active coordinates. This is a dose-changing, single-coordinate diagnostic. Max-endpoint ties and effects on other memories prevent treating the remaining mass as uniformly bad donors.

## 5. Reversal: substantial new expression forms, while old-prefix competition persists

Three branches replay identical cue timing: **REV** starts from the original learned history and performs 12 SPACE reversal bouts; **OMIT_REV** replays that history with only the focal association's initial writes suppressed, then performs the same reversal; **NO_REV_WRITE** retains the full history and reversal exposures but suppresses corrective plasticity. Omission preserves sensory exposure and adaptation but can alter subsequent feedback and learning. It is not globally naive, and it is not selective deletion of one molecular trace.

| Load | REV immediate → day correct | OMIT_REV immediate → day correct | Original old association correct before reversal |
|---|---:|---:|---:|
| 48 | 16/16 → 11/16 | 16/16 → 16/16 | 15/16 |
| 96 | 16/16 → 11/16 | 16/16 → 14/16 | 10/16 |

The combined 22/32 and 30/32 are descriptive totals, not 32 independent histories: both loads reuse each history. All ten delayed REV failures were originally old-choice-correct. In that subset, REV immediate success is 25/25 and day success 15/25 (10/15 at load48, 5/10 at load96). Eligibility is a reported subgroup, never a selection rule for the primary experiment. NO_REV_WRITE has 0/16 and 1/16 day reversed choices; the latter is incidental expression in a mixed history, not new acquisition.

At each endpoint, passively decay the pre-reversal fast and slow coordinates and subtract them from the actual state to get the realized subsequent-write residual. Evaluate ZERO, PREFIX-only, UPDATES-only and TOTAL with the same endpoint adaptation, and use symmetric two-factor contributions so their sum is exactly TOTAL−ZERO. This accounts for output clipping without claiming a unique causal division of nonlinear interactions.

**REV expression after 12 bouts, actual margin in the new direction (Hz):**

| Load / delay | Old-prefix contribution | Subsequent-write contribution | Total |
|---|---:|---:|---:|
| 48 / immediate | −1.749 | +16.817 | +15.068 |
| 48 / day | −3.779 | +7.853 | +4.074 |
| 96 / immediate | −2.028 | +16.842 | +14.814 |
| 96 / day | −4.452 | +8.328 | +3.876 |

The system forms strong new-direction expression immediately. At a day, that contribution is smaller and substantial opposition from the original state remains. Expression can become more negative even while the underlying weights decay: differently signed components have different dynamics and the pooled output is nonlinear. This is not growth of an independently measured old memory.

Omitting the original writes raises the day target margin by **3.721 [3.155, 4.293] Hz** at load48 and **4.461 [3.727, 5.144] Hz** at load96. The corresponding changes in old-prefix contribution are +3.679 and +4.507 Hz. Changes in subsequent-write contribution are only +0.042 [−0.185, 0.278] and −0.046 [−0.155, 0.057] Hz. Thus the measured day difference is chiefly in the carried state contribution, not a large increase in the later expression contribution.

Realized reversal slow-write mass is close but **not exactly yoked**: mean OMIT_REV−REV mass is +0.305 on a REV mean806.905 at load48, and +0.000621 on735.040 at load96. No equal-dose causal claim is based on this contrast. The initial omission also changes other associations' history; the decomposition is a model-level localization, not definitive isolation of one trace. Its size and nearly unchanged subsequent-write contribution make persistent prehistory competition a stronger lead than an immediate inability to acquire the new sign.

![Reversal expression](figures/reversal_expression.png)

The two omission cases still failing at load96 also show that original focal learning is not the only limit. Neither branch demonstrates durable learning under arbitrary future interference. The model contains no explicit gamma4 trace shift or active-zone remodeling, so this experiment cannot confirm or refute the biological forgetting paper discussed previously.

## 6. What V72 licenses next

V72 does not license another neighborhood-pressure formula, a history-trace tuning sweep, global downscaling, more KCs, or a compartment architecture on the strength of these results alone. The full network was retained; none of these failures is introduced by additional subsampling in this round.

The strongest next question is narrower: **can a circuit-accessible signal distinguish when an existing association should be revised from when shared old information should be preserved, and can an intervention reduce the opposing prehistory contribution without simply weakening all old memory or losing the new update?** V68's episode-decoding failure remains a constraint. A useful next design must compare corrective-state action and matched old-memory harm, rather than score only net reversal or initial acquisition. Candidate physiology should be mapped to an observable and action actually represented by the model before implementation; a paper analogy does not supply that map.

GAIN remains an unaccepted local calibration result, not a revised reference. It can be retained as a future control, but its failed gates should not be relaxed. NEIGHBOR and these ridge results reduce the case for spending another round on this particular receiver-pressure family. The finite exclusions also explain why oracle allocations themselves are poor training labels.

## 7. Artifacts, cost and limits

The reference mutable arrays remain **165,688 bytes**. GAIN needs no additional persistent experience state and uses existing Q. NEIGHBOR's dense four-column factor is **165,664 fixed bytes**, plus temporary normalization/work vectors. Offline H/ACT/COUNT add124,248 bytes to the audit object. These are analytical state budgets; the research runner also loads counts, permutations and diagnostics. No minimized autonomous GAIN or NEIGHBOR runtime was promoted or benchmarked, and the fixed reader remains unchanged and separately accounted in prior assets.

The compact bundle includes this report, the locked protocol/config/source, final history-level metrics, three figures, the new anatomy factor, diagnostic ridge fits, acceptance checks and a manifest. Source graphs, parent models/readers/code, raw recipient/seed outputs, snapshots, caches and previous archives are **referenced by path/hash instead of copied**. See [README](README.md) for regeneration. New generated raw data remain in the project for audit. The manifest also preserves the two proposal sources and the prior V71 archive identity.

This is a reproducible negative candidate result plus a stronger reversal diagnosis. It establishes neither a persistent-learning core nor a species-level mechanism. All original biological and synthetic-task limitations still apply.
