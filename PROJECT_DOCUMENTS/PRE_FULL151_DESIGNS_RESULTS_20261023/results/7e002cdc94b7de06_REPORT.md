# MiniFly V75 / Iteration 20 — a working computation survives the anatomical interface

14 September 2026. **All84 prespecified jobs completed; no model is promoted.** The reference remains iteration5/bi, instantiated by iteration10/full5177, with the frozen reader used by the inherited campaigns. This is a completed diagnostic iteration, not a newly qualified persistent-learning core.

**中文结论：V75 把问题定位得更具体了，但不能把它缩成“只缺一个预测误差信号”。**

- **教学支持坐标不足，不能解释这个含噪学习任务上的大部分差距。** 简单 NLMS 使用全部 KC 时为93.30%；限制到当前模型中有α教学支持的1,226个KC后，仍为92.93%。原MiniFly为62.64%。这些是16条新历史、新cue的结果，不是把旧数据再评分。
- **可用的计算在多项接口限制下仍能工作。** 同时加入适应活动、真实输出投影、输出截断、固定reader和T权重后，带全局预测误差与归一化的诊断学习器仍达到93.89%。但这个误差、归一化和每对cue一次的更新时序尚未由原DAN环路实现；它也没有原模型的快慢状态分配。因此这不是一个可直接替换的果蝇核心。
- **移除预测误差会损害表现，但并不能解释全部差距。** 四个匹配比较在含噪任务上降低5.74–9.21个百分点，均达到预先设定的实质损失判据。可是无预测误差的诊断规则仍有82.50–88.52%，明显高于原模型。还必须检查原生教学强度、归一化、快慢状态分配及更新时间。
- **额外修订的收益和代价都在新历史上复现。** 12/18/24次修订的次日成功为84/128、114/128、123/128；后续学习后为85/128、112/128、122/128。18次的平均旧准确率损失很小，但最受影响旧记忆的次日margin相对12次平均下降1.47Hz。两种额外剂量都未通过安全标准。
- **时间模型语义会改变状态，不能混同于浮点精度。** 同一解析模型只细分积分，整段历史误差约5.7e−11；改变反馈刷新时机则能明显改变容量与状态。较细步长没有修复延迟修订，连续模型状态也尚未充分收敛，不能据此替换生理拟合的事件模型。

Start with the [frozen protocol](PROTOCOL.md), [configuration](config.json), [decision record](results/decision.json), [independent audit](checks/report_audit.json), and [reproduction guide](RESULTS_README.md). Detailed final tables accompany every finding below.

## What was tested

| Block | Independent generated histories | Executed comparisons | Scope |
|---|---:|---|---|
| Learning bridge |16 |16 arms ×2 tasks,1024 observations each | Support, feature/expression interface, and error delivery |
| Dose |32 |12/18/24 bouts,4 challenge classes,2 loads and2 focal ages | Fresh resource replication, both clocks, collateral costs |
| Time |4 |9 models through full96-association histories and reversal/mixed challenges | Event-map sensitivity and an explicit continuous reconstruction |

Config, protocol, new scientific source, proposal hash, dependencies and fresh inputs were locked before scientific execution. No settings, seeds, gates or model families changed after outcomes. The run took **580.1 seconds (9.7 minutes)** with two workers. All84 jobs were retained. There are512 final bridge arm/history scores,12,288 dose endpoint rows and864 time challenge endpoint rows, plus capacity and state diagnostics.

The input generator created12,480 distinct8-of-88 PN patterns across three disjoint panels. Exact PN duplicates were excluded against the seven explicitly named prior banks and the interface calibration/evaluation sets; this is not an exclusion claim over every historical file. Not every generated pattern is presented in every assay. The bridge uses64 cue pairs per history; the dose/time campaigns train96 associations and use12 later pairs. KC overlap is retained. These are independent synthetic histories through the same specimen-derived interface, **not independent flies or natural-odor recordings**.

A history is the statistical unit. Bridge scores have16 units, dose contrasts32, time summaries4. Focal ages and loads are averaged within histories before pooled comparisons. The22 declared primary bridge contrasts use Holm-adjusted exact paired sign-flip p-values; a material loss additionally requires a mean loss of at least5pp. The95% paired bootstrap intervals are descriptive and unadjusted; sign-flip inference assumes exchangeability of paired difference signs under the null. Reversal transfer and time comparisons do not promote a candidate. Dose qualification is a prespecified intersection of benefit and safety conditions, not a weighted score or a ranking by number of gates passed.

## 1. Teaching support is sufficient for the stable-noisy benchmark

The actual inherited matrices have5,177 KCs,1,770 KCs with nonzero alpha-output projection, and **1,226 with nonzero alpha teaching support T**. All1,226 teaching-supported KCs also have an alpha-output projection. This is a property of the current reduced interface; it does not say that the biological animal has only1,226 plastic KCs.

The first comparison reproduces V74's normalized KC pair-difference computation and then restricts its writable coordinates. It keeps the same normalized feature recipe, learning rate and learned intercept. It therefore tests support without simultaneously imposing the native output or DAN calculation.

| Cue-only learner | Stable association +10% corrupted teaching | Reversal every256 observations |
|---|---:|---:|
| Native MiniFly REF |62.64% |49.94% |
| Full-KC NLMS |93.30% |57.23% |
| Teaching-supported NLMS |92.93% |56.32% |
| Teaching-supported NLMS, no intercept |94.12% |65.77% |

Support restriction changes the stable-noisy score by **−0.374pp [−0.716,−0.098]**; Holm-adjusted p=.197 across the primary family. It is a small observed cost, far short of the predeclared5pp material-loss threshold. The support-limited computation remains effective. Removing the learned intercept improves the stable score by1.19pp [0.62,1.74]; on reversal it also improves the score substantially. The intercept result is specific to this balanced pair task and normalization recipe, not a rule that bias parameters are generally harmful.

These results disfavor **insufficient writable coordinates as the main explanation for this particular stable-noisy gap**. They do not establish the delayed capacity of a1,226-neuron fly, identify a minimum neuron count, or remove the96-association campaign's interference problem. The vectors still use5,177-position indexing for auditability. Their off-support positions are exactly zero; no compact-neuron implementation is claimed. [History-level scores](results/bridge_per_history.csv), [all bridge contrasts](results/bridge_contrasts.csv).

## 2. The bridge survives several restrictions, but native teaching has not been reconstructed

The factorial is explicit about which computations it permits. All twelve arms use one alpha memory vector, no intercept, no gamma or alpha-fast plastic state, the inherited slow lifetime, and one update after each revealed pair outcome. STATIC uses binary KC activity. ADAPTED uses the inherited exogenous adaptation trajectory at the current probe. LINEAR uses the actual Q projection without the baseline. CLIPPED uses the inherited physical output bounds and the frozen sensory reader. Therefore the expression contrast combines clipping and baseline subtraction; it cannot attribute all changes to clipping alone.

GRADIENT uses the actual supported derivative of the selected output. ROUTE_ERROR distributes a scalar prediction error with the pooled T weights and an explicit global normalization. ROUTE_LABEL uses the same update direction and denominator but omits prediction subtraction. All receive the same revealed, possibly corrupted outcome; clean labels and task identity remain evaluator-only. No online weight update receives a cue ID, future outcome, historical tag or clean target.

| Activity / expression | GRADIENT, noisy / reversal | ROUTE_ERROR, noisy / reversal | ROUTE_LABEL, noisy / reversal |
|---|---:|---:|---:|
| Static / linear |94.09 /66.06% |94.15 /66.13% |87.25 /43.87% |
| Static / clipped + reader |90.12 /64.93% |91.71 /58.95% |82.50 /44.60% |
| Adapted / linear |94.29 /65.61% |94.25 /65.50% |88.52 /43.87% |
| Adapted / clipped + reader |93.86 /64.62% |**93.89 /62.44%** |87.66 /43.78% |

The fully listed ADAPTED/CLIPPED/ROUTE_ERROR bridge scores **93.89% [93.34,94.43]** on the noisy task, versus62.64% [59.81,65.48] for the native parent. Its score is not an outcome-selected tuning result; it is one fixed cell of the complete factorial. Adaptation helps some clipped arms: for GRADIENT, the activity-by-expression interaction is+3.54pp [2.48,4.72]. Under static clipped expression, about37.4% of alpha observations clip and8.7% of gradient updates are zero; the adapted gradient arm reduces these to10.4% and0.22%. A locally flat clipped gradient is a stalled update, not evidence that the memory is correct.

Removing prediction error causes the four stable-task losses below, each passing the frozen5pp/Holm criterion:

| Matched interface | ROUTE_LABEL minus ROUTE_ERROR | Holm p |
|---|---:|---:|
| Static / linear |−6.90pp [−7.98,−5.80] |.00067 |
| Static / clipped |−9.21pp [−11.17,−7.25] |.00110 |
| Adapted / linear |−5.74pp [−6.88,−4.46] |.00067 |
| Adapted / clipped |−6.23pp [−7.53,−4.84] |.00110 |

No support, adaptation, expression or route-weight contrast meets that material-loss criterion on this task, although several smaller expression effects are statistically detectable. Repeated reversal is harder: error-driven variants reach roughly59–66%, while the open-loop variants fall to about44%. The comparator has not solved continual reversal or long-delay joint-choice memory.

### The causal limit is central to interpreting this result

**ROUTE_ERROR is not the native DAN algorithm.** T constrains its coordinate support and relative distribution; the pair-level prediction error, normalization, target score scale and post-pair update timing are supplied by the engineering comparator. Pooled T weights do not preserve every original directional, temporal and compartment-specific teaching constraint. No anatomical or physiological validation establishes that neurons can compute this normalization or deliver this residual. No matched-random wiring test was run, so these scores cannot demonstrate that the actual map is uniquely advantageous.

The bridge also removes fast-state superposition and the native slow-write fraction. Relative to native MiniFly, those are jointly changed restrictions. It would be wrong to label the whole31pp gap a causal effect of prediction error: **even ROUTE_LABEL retains82.50–88.52%**, well above the native62.64%. What remains between the working restricted computation and the native model is a bundle comprising payload law and scale, normalization, fast/slow allocation, dynamic feedback and update timing. V75 narrows that bundle; it does not yet separate its members.

The physical alpha-state magnitudes also differ. On the noisy task, mean final max|slow| is169.72 for native MiniFly,9.91 for ADAPTED/CLIPPED/ROUTE_ERROR, and55.23 for its ROUTE_LABEL ablation. These are descriptive state magnitudes, not physiological bounds or proof of a scale-only cause. The original raw NLMS coefficients have a different feature/output scale and should not be numerically compared as synaptic strengths.

![All bridge arms](figures/bridge_interfaces.png)

## 3. Stronger revision works on new histories, with persistent collateral margin costs

Dose branches start at a shared REF prestate after each load. Only the number of later presentations changes. This does not change global eta or original acquisition, and it is not an autonomous policy. REV/STABLE/ANOMALY1/ANOMALY2 are generator schedules, never labels passed to a learner. Anomaly schedules supply one or two incorrect bouts followed by the original contingency; increasing their length also supplies additional correct evidence.

The primary COMMON clock pads each shorter branch to the24-bout duration before observation, then either waits or performs12 further associations. At each phase,0s,300s and one-day observations are independent clones. LAST-clock results are recorded separately immediately after each schedule and one day later. Each bout includes its trailing135s gap. A trained branch and its no-write control share the exact sensory schedule and elapsed time.

| Bouts | COMMON challenge/day correct | COMMON mixed/day correct | Challenge/day old accuracy | Mixed/day old accuracy |
|---|---:|---:|---:|---:|
|12 |84/128 |85/128 |74.70% |73.19% |
|18 |**114/128** |**112/128** |74.38% |72.69% |
|24 |**123/128** |**122/128** |73.79% |72.09% |

Relative to12 bouts, the primary18-bout intervention improves challenge/day reversal by **23.44pp [16.41,31.25]** and mixed/day reversal by **21.09pp [14.06,28.13]**. The24-bout changes are+30.47pp [21.09,40.63] and+28.91pp [20.31,38.28]. Both doses pass the two benefit gates. This prospectively confirms the V74 resource lead on new inputs and histories, including continued learning.

It does **not** confirm a free or safe repair:

| Dose minus12, REV | Challenge/day | Mixed/day |
|---|---:|---:|
|18: old accuracy |−0.315pp [−0.680,+0.041] |−0.496pp [−0.860,−0.139] |
|24: old accuracy |−0.904pp [−1.368,−0.441] |−1.102pp [−1.549,−0.640] |
|18: direct worst old margin lost relative to12 |**1.468Hz [1.326,1.617]** |**1.478Hz [1.340,1.625]** |
|24: direct worst old margin lost relative to12 |**2.997Hz [2.709,3.304]** |**3.042Hz [2.758,3.343]** |

“Direct worst” takes the largest positive12-bout-minus-longer-dose old actual-margin difference within each assignment, then averages ages/loads/histories. Victims may differ across assignments. It includes all nonfocal old associations, not just a selected initially-correct subset, and is distinct from subtracting two separately measured worst-damage values against no-write controls. Both definitions are retained. Negative or near-zero average accuracy changes cannot certify safety of the worst old margin.

At the common pre-delay endpoint,18 bouts reduce old accuracy by **2.41pp [1.67,3.12]** and24 by2.24pp [1.58,2.96], expressed as loss magnitudes. The18-bout direct worst margin loss there is5.75Hz. Considerable loss recovers with rest, but it is not absent. LAST-clock results differ because the12- and18-bout branches have not received the extra rest padding; comparing them is a different time question. Neither clock's result is silently substituted for the other.

Dose18 passes95/230 frozen resource gates; dose24 passes72/230. Both fail **all48 no-write-relative worst-damage guards and all48 direct worst-loss guards**. They also fail some old-accuracy, old-loss and mixed-new-acquisition guards. All12 STABLE/ANOMALY delayed-target guards pass at either dose. These230 checks are correlated intersection conditions, not230 independent scientific experiments. The.05Hz margin guard is a stated engineering safety requirement, not a proven biological law. The observed1–3Hz delayed costs substantially exceed it; the verdict is not a floating-point boundary artifact. [All resource gates](results/dose_gates.csv), [independent reconstruction](checks/gate_reconstruction.csv).

The retrospective minimum tested dose reaching the joint criterion at both COMMON/day endpoints is12 for79 assignments,18 for31,24 for12, and beyond the tested range for6. This is an evaluator-only resource curve, not an online selector. Immediately after the final12-bout schedule,122/128 choices are correct but only84/128 remain so after a day; after18 bouts128/128 are immediately correct yet only114/128 remain. **Current success is still not a durable stopping certificate.** No controller receives future clone outcomes or permission to choose its dose from this table.

![Dose benefit and collateral costs](figures/dose_benefit_and_cost.png)

## 4. Time semantics matter; finer subdivision did not solve delayed revision

V75 deliberately separates three cases. EVENT is the unchanged fitted transition. LEGACY_10/3/1s repeatedly calls that inherited transition while preserving physical stimulus duration; this changes several update approximations and feedback evaluation times. It is a sensitivity study, not convergence to a claimed equivalent equation.

The new continuous reconstruction explicitly defines simultaneous adaptation recovery/depletion and memory forcing/decay. HOLD fixes teaching payload at the onset of each physical cue, analytically integrates adaptation-weighted forcing and decay, and permits purely numerical subdivision of that same forcing. ODE refreshes the payload from the current state every10/3/1s and integrates each held interval analytically. HOLD and ODE differ in modeled control timing; neither is a physiological refit of EVENT. The observer remains the common inherited endpoint observer, isolating changes to training trajectories.

| Model | Load96/day capacity | Alpha observations clipped | REV challenge/day | REV mixed/day |
|---|---:|---:|---:|---:|
|EVENT |73.44% |6.45% |11/16 |11/16 |
|LEGACY_10s |73.18% |6.71% |11/16 |11/16 |
|LEGACY_3s |73.18% |6.58% |11/16 |11/16 |
|LEGACY_1s |73.18% |6.38% |11/16 |11/16 |
|HOLD_30s |48.70% |56.77% |9/16 |8/16 |
|HOLD_1s |48.70% |56.77% |9/16 |8/16 |
|ODE_10s |65.89% |20.83% |11/16 |11/16 |
|ODE_3s |72.66% |9.11% |11/16 |11/16 |
|ODE_1s |73.44% |7.10% |11/16 |11/16 |

The matched HOLD whole-history state error is at most **5.69e−11**, confirming the analytic subdivision control. By contrast, HOLD versus EVENT changes the slow vector's relative L2 norm by about100% on average. Holding an early payload is not the same modeled control as refreshing it within a cue.

Legacy1s versus3s changes slow state by about0.98% relative L2; legacy1s versus EVENT differs by16.28%. ODE3s versus10s changes slow state by33.20%; ODE1s versus3s still changes it by**9.00%**. State convergence of the continuous reconstruction is **not established** on this grid. Its capacity improves as the refresh interval becomes smaller, so coarse timing can affect behavioral scores even when a focal-choice panel is unchanged.

All legacy and refreshed-ODE variants have the same16 delayed focal decisions as EVENT in this panel, not merely the same pooled count. Their actual margins nevertheless differ: ODE1s loses about0.98Hz in challenge/day target margin relative to EVENT. Thus unchanged binary choices cannot certify identical learned states. Four histories support a compact diagnostic, not a general convergence theorem or a physiological parameter choice. No continuous variant is promoted or claimed to invalidate V50–V74. [State comparisons](results/time_state_comparisons.csv), [choice agreement](results/time_choice_agreement.csv), [capacity](results/time_capacity_summary.csv).

![Whole-history time audit](figures/whole_history_time.png)

## Costs, integrity and scientific limits

Native MiniFly retains5,177 KCs and165,688 mutable numeric bytes. The supported NLMS has41,424 weight/intercept bytes, or41,416 without an intercept; the unused last-outcome scalar counted in V74's general information interface is absent here. Static bridge dynamic-state accounting is41,440 bytes; adapted bridge accounting is82,856, including its full adaptation vector and24 bytes of counters. These are dynamic numeric-state budgets, not process RSS or full artifact sizes. The full indexed vectors are allocated, including their zeros; no hidden sparse-storage saving is claimed. Diagnostic counters, stimulus arrays and result logging are evaluation overhead.

The inherited fixed numeric interface is748,956 bytes, with an additional4,344-byte frozen reader representation in this instantiated model. Fixture encodings and features are cached for speed but are deterministic functions of the supplied PN cues and fixed matrices; the online learners do not maintain cue-indexed learned tables. ADAPTED diagnostic arms share a computation in the test harness but are each charged for the required adaptation vector. The harness's all-zero disabled-learning scratch buffers are implementation overhead. Functional global error/normalization operations and per-pair access remain explicit capabilities, not free biological assumptions.

Peak recorded worker RSS is442,187,776 bytes, about422MiB per worker, not total Mac usage. The sum of worker times is1,060.8 seconds. The raw parent, generated stimuli, per-assignment victim vectors and checkpoints remain in the project and are referenced by hash instead of copied into the compact results ZIP.

Integrity evidence includes:

- All84 atomic job receipts,490 unchanged dependency assets, locked scientific source and fresh input hashes verified.
- Complete arm/load/age/case/clock/phase/horizon grids, with no excluded assignments or numerical retries.
-96 independent NLMS anchor replays, including support and intercept variants on every bridge history and task.
- Direct event-based replay of192 observer panels, max disagreement **1.51e−14Hz**.
- Independent reconstruction of every saved dose victim's strict choice, old accuracy and direct worst-loss metric; all460 resource gate decisions reconstructed independently from final history tables.
-42 pre-run software checks and62,148 post-run assertions. Most assertions are repeated hash, input or victim bookkeeping; they are not independent biological replications.

One post-run reporting assertion initially doubled the expected dose-row count. It was corrected from768 to384 rows per history by independently enumerating the frozen grid (2 loads ×2 ages ×4 cases ×3 doses ×8 clock/phase/horizon endpoints). The actual worker specification and all scientific outputs were complete and unchanged; no scientific rerun or threshold adjustment was made.

No physiological fitting, new connectome extraction, natural-odor validation, autonomous stopping policy, memory-tag access or model promotion occurred. The bridge tests prequential forced choice on64 pairs; it is not scored on delayed joint actual/reader thresholds. Dose tests the original delayed benchmark with stronger scheduled evidence; it does not demonstrate reliable autonomous recognition of reversal. Time reconstruction diagnoses semantics while leaving physiological qualification unresolved. These distinctions prevent one block's convenient endpoint from silently replacing another's requirement.

## What this changes about the next decision

The support restriction and several interface components **do not destroy a working learning computation**. This reduces the case for starting the next round with more neurons, more connectome edges or another intuitively named local memory variable. The main remaining bridge is from pair-level normalized updates to the parent's actual teaching payload, state allocation and event timing. A useful follow-up would isolate these restrictions on a shared starting state and common output, without attributing all gains to prediction error or importing an extra learned readout unnoticed.

The positive dose replication establishes a real resource tradeoff: more revision evidence can overcome much of the delayed failure, including after subsequent learning, but carries measurable victim costs. Any eventual controller must beat a fixed-dose comparison at its **realized** dose and preserve old choices and margins; current reader sign alone is insufficient.

The time result is a boundary condition for that work. A proposed continuous rewrite must state its dynamics and feedback timing, demonstrate an adequate convergence range and revisit physiological fit before replacing EVENT. The present evidence does not justify spending another mechanism iteration on an unqualified simulator change.

**The accepted reference is preserved. V75 delivers a reproducible localization result and a confirmed revision-resource curve, not a new functioning fly core.**
