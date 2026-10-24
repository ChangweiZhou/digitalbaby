# MiniFly V74 / Iteration 19 — scales, expression, information and local correction

14 September 2026. All **493 jobs** completed. This is the post-run scientific report; the earlier runner delivery made no scientific result claims. **No learning model is promoted.** Keep the full5,177-KC iteration5/bi reference instantiated by iteration10/reference and the frozen iteration8 observer.

**中文结论：V74 不只是又一次负结果。它确认了几个足以改变研究方向的问题，但没有产生合格的新学习器。**

- **旧关联慢状态确实会阻碍修订。** 移除被实验者准确标记的旧关联慢贡献，次日修订从45/64升到60/64；继续学习后从40/64升到59/64。这个理想操作需要模型运行时没有的标签，而且会伤害部分其他记忆的幅度。
- **局部收缩有作用，但作用不够选择性。** 直接收缩能够显著增加修订成功；强收缩同时损害旧记忆。加入快状态确认后，修订收益不足，仍有旧记忆损伤。十个配置无一通过全部开发门槛，因此没有把收缩候选送入新历史确认。
- **此前担心的尺度问题是真实的。** 32条新历史确认：输出增益加倍或将KC活跃比例从5%改为10%，会让几乎全部相关输出进入饱和。离线移除最终截断能恢复大量选择，不能把这种退化都归因于存储失败。
- **修订剂量的效应很大。** 在8条开发历史的等时长比较中，12次、18次、24次修订的次日成功分别为23/32、30/32、31/32；24次也带来约1个百分点的旧记忆准确率损失。这还不是能自主选择剂量的安全策略。
- **高层测试暴露了模型与信息边界。** 稳定关联加10%错误教学的任务中，简单NLMS基线达到93.50%，MiniFly为60.18%。隐藏的随机情境则令各cue-only模型接近50%；提供情境后NLMS达到96.31%。前者提示当前学习实现有缺口，后者说明缺信息不能靠改学习率解决。

Primary evidence: [frozen protocol](PROTOCOL.md), [config](config.json), [stage decisions](results/stage_a_decision.json), [development gates](results/development_gates.csv), [final runner decision](results/decision.json), [report audit](checks/report_verification.json), [results reproduction](RESULTS_README.md).

## What ran, and what did not

The proposal's state-contraction experiment became one gated branch of the broader audit requested by the user. No new mechanism, parameter range or fallback was added after outcomes. The explicit selection rule replicated the two **largest absolute** development reversal effects, including adverse effects; it did not select the two most promising learners.

| Block | Executed scope | Disposition |
|---|---|---|
| A |29 fixed scale configurations ×8 existing development histories;16 tagged-surgery histories; partition/dose, restricted action witnesses, information tasks and physiology compatibility |301 jobs complete; surgery gate passes |
| B | CONTRACT and FAST_CONTRACT ×5 rho values ×8 development histories |80 jobs complete; no candidate passes every gate |
| C | Conditional candidate-dose yokes |Not run, because B selects no candidate |
| D | REF, double Q,10% active KCs ×32 fresh histories;16 new information-task histories |112 jobs complete; adverse scale effects replicate |

There are408 full model/history campaigns and36,576 raw challenge endpoint rows. Fresh confirmation contains7,680 globally unique PN cues, checked against the explicitly listed prior banks and interface calibration/evaluation. KC overlap remains. Development uses inspected V73 histories; information confirmation changes task histories on fixed earlier cue geometry, not novel odors. Histories are independent generated streams through one specimen, not independent flies.

Main campaigns train96 associations from zero, six bouts each; test loads48/96 and two focal ages; run REV12, STABLE12, ANOMALY1+11 and NEW6; probe immediately,300s and one day; then learn12 additional associations and probe again. Confirmation adds ANOMALY2+10 as a secondary condition. Delay probes are independent clones. These are not live serial retrieval histories.

History is the statistical unit. Focal ages are averaged before loads; quoted pooled intervals use5,000 paired history bootstrap resamples. Development and secondary contrasts are exploratory, with unadjusted95% intervals. The primary candidate would have required10pp durable-revision gains at both day endpoints, acquisition/safety guards and fixed state cost. It never reached confirmation. An empty confirmation-gates CSV denotes a skipped candidate screen, not a passed screen.

## 1. Tagged surgery locates a real source of delayed relapse

Parallel evaluator tags follow the **realized** alpha-fast/slow increments from focal original training, other prehistory, reversal and subsequent learning. Their sums recover the live states; tags never feed into dynamics. Removing a tag after training changes the present state. It is not the counterfactual trajectory that would have occurred had that training never happened, because subsequent feedback could have differed.

| Post-reversal intervention | Immediate correct | Day correct | After MIXED, day correct |
|---|---:|---:|---:|
| FULL |64/64 |45/64 |40/64 |
| Remove reversal-generated fast component |24/64 |45/64 |40/64 |
| Remove original focal slow contribution |64/64 |**60/64** |**59/64** |
| Remove reversal-generated slow contribution |63/64 |7/64 |8/64 |
| Zero all alpha-fast state |33/64 |45/64 |40/64 |

Original-focal-slow removal rescues15 of19 FULL day failures, **78.95%**. The day joint-choice gain is **23.44pp [14.06,32.81]**, with actual-margin gain **4.306Hz [4.001,4.588]**. After mixed learning it gives **29.69pp [18.75,42.19]** and4.435Hz [4.120,4.723]. Both surgery continuation conditions pass on all64 assigned cases, using16 histories.

This supports two bounded conclusions. The original carried slow contribution obstructs many revisions. The reversal also writes useful slow state: deleting reversal slow leaves only7/64 correct the next day, despite63/64 immediate success. Fast state can mask insufficient durable correction; these data do not support saying no useful slow revision was written.

The privileged operation is not harmless. Its mean worst-old positive margin damage relative to FULL is **1.159Hz [1.073,1.250]** after the challenge and1.154Hz after mixed learning. Mean old accuracy changes only+0.265pp [−0.016,0.529] and−0.198pp [−0.579,0.200], respectively. Small mean choice changes can coexist with substantial damage to individual margins. Surgery's comparator is FULL; do not numerically equate its damage with the candidate campaigns' no-write-based damage metric. [Surgery summaries and paired contrasts](final_tables/surgery_contrasts.csv).

## 2. Local contraction improves revision at the cost of other memories

The tested extra term is −r(dt)g_i s_i before slow decay, where r(dt)=1−(1−rho)^(dt/30). CONTRACT requires s_i u_i<0. FAST_CONTRACT additionally requires the **pre-update** alpha-fast state to agree with u_i. All five strengths are frozen; no new persistent array is introduced.

A scalar state contains superposed history, not a separable record of one association. Local conflict need not mean obsolete memory, and fast agreement need not mean trustworthy repetition. Existing fast state can already agree with a first anomaly. The exact-zero-fast unit test protects only that particular mathematical case.

The eight development histories contain32 reversal assignments. REF is23/32 correct after CHALLENGE/day and20/32 after MIXED/day.

| Rule | rho per30s | Challenge/day | Mixed/day | Mixed/day old accuracy change at load96* | Gates passed /23 |
|---|---:|---:|---:|---:|---:|
| CONTRACT |.01 |24/32 |21/32 |−0.82pp |19 |
| CONTRACT |.025 |24/32 |23/32 |−2.38pp |14 |
| CONTRACT |.05 |28/32 |27/32 |−5.22pp |14 |
| CONTRACT |.10 |29/32 |28/32 |−12.93pp |11 |
| CONTRACT |.20 |31/32 |27/32 |−22.03pp |8 |
| FAST_CONTRACT |.01 |23/32 |21/32 |−0.56pp |17 |
| FAST_CONTRACT |.025 |23/32 |23/32 |−1.85pp |17 |
| FAST_CONTRACT |.05 |23/32 |23/32 |−3.00pp |12 |
| FAST_CONTRACT |.10 |25/32 |23/32 |−7.27pp |10 |
| FAST_CONTRACT |.20 |25/32 |23/32 |−14.35pp |10 |

*Equally averaging the four primary challenge classes after focal averaging; absolute accuracy, not conditional accuracy among an arm-specific eligible cohort. Passing this individual mean guard does not mean zero harm or that other safety gates passed.*

Direct contraction at.05/.10/.20 meets both **development mean** revision-benefit thresholds, but fails safety. For example,.20 improves challenge/day revision by25pp [6.25,43.75], while reducing old mixed/day load96 accuracy by22.03pp and failing15 of23 gates. The smallest direct contraction fails the two revision thresholds and both challenge worst-damage guards. No direct contraction qualifies overall; it was diagnostic rather than promotion-eligible in any event.

The first active pulse of ANOMALY challenges already contracts an average9.59–14.98% of active teaching-supported coordinates across the five FAST_CONTRACT strengths. This confirms that pre-existing fast agreement does not guarantee protection of the first anomalous bout; it does not identify which pulse caused each final failure.

FAST_CONTRACT never reaches the10pp benefit requirement at either endpoint. Even its best challenge/day benefit is6.25pp; mixed/day reaches9.375pp. Every strength also fails at least some worst-old-damage guards. The fast gate neither delivers sufficient revision nor reliably protects other memories in this implementation.

Therefore **B is not a “nothing happened” result**. Strong contraction changes behavior substantially, with an adverse retention tradeoff. The failed fast gate does not prove that no local evidence-accumulation mechanism can work. Because no candidate passed, C's targeting-versus-dose comparison and fresh contraction confirmation were correctly skipped; no targeting-specific causal claim follows. [All development gates](results/development_gates.csv), [history-level behavior](final_tables/behavior_per_history.csv), [event-level mechanism summaries](final_tables/contraction_event_summary.csv).

![Tagged-state intervention and local-rule tradeoff](figures/surgery_and_contraction.png)

## 3. Two modest interface changes cause severe, independently replicated expression failure

The fixed ranking selected sparsity_0.1 and q_gain_2 because they had the largest absolute development revision effects. Both were adverse. Confirmation tests sensitivity, not a proposed improvement.

|32 fresh histories,128 reversal assignments | REF | Double output projection Q |10% active KCs instead of5% |
|---|---:|---:|---:|
| Challenge/day correct |**73/128** |14/128 |2/128 |
| Mixed/day correct |**74/128** |15/128 |2/128 |
| Load96 day capacity |71.48% |11.98% |1.56% |
| Load96 day capacity, offline final-clip removal |68.26% |62.92% |61.30% |
| Load96 alpha output observations clipped |4.12% |99.32% |99.96% |

Challenge/day paired reversal differences are **−46.09pp [−57.03,−35.94]** for double Q and **−55.47pp [−66.41,−45.31]** for10% activity. After mixed learning they are−46.09pp and−56.25pp. The fresh reference's73/128 must not be compared as a version regression against the earlier45/64: these are different histories and denominators.

An independent expression audit of all192 confirmation checkpoints shows near-universal **upper** saturation in the two changed configurations. Only changing the final offline readout expression, while retaining the learned trajectories, restores much of their capacity score. It does not completely recover the reference and is not a calibrated unclipped learner: gamma clipping and the existing frozen reader remain. Nevertheless, the very low clipped score cannot be interpreted as equally severe loss of all stored distinctions.

Increasing activity sparsity without compensating gains changes overlap, current magnitude, adaptation and writing together. Doubling Q changes both expression and feedback inputs; neither is a pure memory-capacity intervention. Primary comparisons intentionally held other factors fixed. Two fixed compensation configurations were included in development: activity×.5/Q×2 and activity×2/Q×.5 also worsened revision, so preserving a simple unadapted current product is not enough to establish dynamic equivalence. There was no output-selected rescue calibration.

These results substantiate the user's concern about seemingly small interface choices. They do **not** establish that the accepted reference is itself in this extreme saturated regime; its day/load96 clipped fraction is4.12%. [Confirmation contrasts](final_tables/pooled_contrasts.csv), [independent expression audit](final_tables/expression_per_history.csv), [capacity results](final_tables/capacity_per_history.csv).

![Confirmed scale sensitivity and saturation](figures/scale_and_saturation.png)

## 4. Dose matters more than the tested new feedback, but stronger dose is not yet a safe controller

The dose diagnostic uses32 assignments from eight inspected histories. Each branch starts at the same pre-reversal state. Shorter schedules receive REST padding to the common24-bout endpoint, then another day. Thus elapsed time is equal, but sensory exposure and number of outcome presentations intentionally differ.

| Reversal bouts | Day correct | Mean target actual margin,Hz | Mean old accuracy |
|---|---:|---:|---:|
|6 |9/32 |−1.144 |76.38% |
|9 |13/32 |1.042 |76.48% |
|12 |23/32 |3.481 |76.48% |
|18 |30/32 |8.671 |76.25% |
|24 |31/32 |13.764 |75.49% |

Relative to12 bouts,18 gives **+21.88pp [9.38,34.38]** day revision and24 gives **+25.00pp [9.38,43.75]**. The24-bout old-accuracy change is **−0.995pp [−1.853,−0.199]**;18 is−0.233pp [−0.664,0.230]. This is an effective bounded dose intervention, with a visible high-dose cost.

No24-bout autonomous strategy was confirmed on fresh histories. Its anomaly handling, mixed-learning outcome, physiology and worst-tail safety were not jointly qualified. The experimental schedule knows it is delivering reversal; the learner has not learned when to demand extra evidence. Larger dose is therefore a useful lead, not a completed persistent-learning solution. This common-prestate dose assay is not a head-to-head autonomous comparison against contraction trained from zero.

The broader eight-history screen did not find that simply multiplying all plasticity solves the problem. Eta factors.25/.5/2/4 all reduced pooled day reversal. Added feedback at1×/4×/16× the V73 added coefficient did not improve it. Halving adaptation recovery time improved development challenge/day by9.375pp and mixed/day by18.75pp, but its separate corrected physiology RMSE rose from4.196 to6.958Hz; it was not a confirmed candidate. Fraction.25 and doubled gaps had some small positive development endpoints. These findings are not a jointly optimized working region or a proof that no such region exists. [All development contrasts](results/audit_contrasts.csv), [dose contrasts](final_tables/dose_contrasts.csv).

## 5. Event partitioning changes state, but did not change this one-event choice panel

For the same30-second active event on32 matched prestates, dividing it into30 one-second events gives a mean maximum coordinate discrepancy of **0.168 slow-state units** and **1.896 fast-state units**. Mean target-margin difference from one unsplit event is about+0.219Hz immediately and+0.0445Hz after a day. Across the tested split counts1/3/10/30, target choice counts remain4/32 immediately and3/32 after a day.

Inactive-rest partitioning passes the exact-state check. Only the contraction hazard itself is normalized to compose under a fixed gate; the full inherited active-event adaptation/feedback approximation is not made partition invariant by that normalization.

This confirms a numerical modeling dependence, without demonstrating that it explains the main campaign failures. The test perturbs **one event**, not every event in a96-association trajectory. No continuous-time rewrite, convergence study of whole learning histories, or physiological refit was executed. Those remain unresolved. The extra0.5Hz metric also leaves the frozen reader threshold at least1Hz; it is a partial threshold diagnostic, not a fully relaxed joint0.5Hz decision rule. Primary thresholds were never changed. [Partition summaries](final_tables/partition_summary.csv), [paired contrasts](final_tables/partition_contrasts.csv).

![Dose and active-event partition diagnostics](figures/dose_and_partition.png)

## 6. Task information and a simple baseline expose different kinds of limitation

Each task has1,024 online observations,256 warmup and768 scored prequential predictions. New task-history confirmation uses16 seeds, with fixed earlier PN/KC cue geometry. Scores use expected forced-choice accuracy under the generated latent target law. The latent probability is available only to the evaluator. The parent reader's abstention is separately recorded; these numbers are not the main campaign's joint actual/reader memory score.

| New task histories; common cue-only information | MiniFly REF | Logistic online baseline | NLMS online baseline |
|---|---:|---:|---:|
| Stable association,10% corrupted teaching outcomes |60.18% |84.08% |**93.50%** |
| Deterministic reversals every256 observations |50.20% |50.63% |**58.47%** |
| Switching .2/.8 outcome probabilities |50.38% |50.17% |**52.80%** |
| Hidden independent balanced context |50.34% |50.01% |50.86% |

NLMS−MiniFly is **+33.32pp [31.06,35.73]** on stable associations with noisy teaching and+8.27pp [5.74,10.77] on repeated reversals. The cue-only baseline uses a fixed KC pair-difference vector and about41,432 persistent numeric bytes, compared with165,688 for MiniFly. Its update and output interface differ from the anatomical learner; it is not subject to the same physiology or DAN-route constraints. It is a useful engineering comparator showing that raw cue/outcome information is sufficient for much better performance on at least one shared-information task, not a recovered fly mechanism.

The context task is deliberately different. A fresh fair context bit switches the desired label for the same cue; without that bit, prior history cannot predict its independent current value. The50% population ceiling follows from this generator, not from classifier failure. With context supplied, NLMS reaches **96.31% [95.96,96.65]**, versus50.86% with cue alone; adding only the last revealed outcome stays near chance. The context benefit is45.44pp [44.41,46.43]. Providing context changes available information and cannot be called a same-input victory over MiniFly.

Irrelevant context/recent features actually hurt NLMS in several other tasks. More information slots alone are not a guaranteed improvement. Reversal and probability tasks remain difficult even for these two untuned baselines; their near-chance scores do not prove information-theoretic impossibility. No ideal full-history observer was implemented for those tasks. [Information summaries](results/information_summary.csv), [paired contrasts](final_tables/information_contrasts.csv).

![Task and information diagnostics](figures/task_information.png)

## 7. Restricted endpoint witnesses provide limited room, not a global impossibility proof

At32 development reversal/day endpoints, nine assignments initially fail. LPs act only on active, teaching-supported slow coordinates, retain the current clipping regions, obey explicit finite coordinate budgets, preserve correct-old thresholds and allow at most.05Hz decrease per old actual margin.

| Allowed endpoint action | Feasible /32 | Feasible among9 failures | Strict new joint-choice rescues |
|---|---:|---:|---:|
| Free signed supported change |26 |3 |2 |
| Raw-write direction |23 |0 |0 |
| Contraction direction |24 |1 |0 |

There are no solver timeouts or unresolved statuses. All feasible solutions pass independent affine/nonlinear replay tolerance checks. **Feasibility is not identical to strict choice success:** one free and one contraction witness land at a numerical decision boundary, so the strict scorer still rejects them. Their reconstruction errors are about10⁻¹⁵; no tolerance was loosened and they are not counted as rescues. This distinction avoids repeating the earlier mistake of making a numerical boundary into a biological capacity claim.

Most feasible cases were already correct. The restricted witness set offers little additional correction under these simultaneous safety constraints, but cannot rule out different budgets, trajectories, clipping-region changes, state representations or learning laws. It does not contradict tagged surgery, which has different support/access and can exceed the old-margin restrictions. [Witness summary](final_tables/witness_summary.csv).

## 8. Physiological compatibility and what remains untested

Nineteen transferable configurations were scored without refitting against both the original60-point workbook and the previously reconciled published targets. Ten interface configurations are explicitly **not identified** physiologically. No sensory or context map was invented; contraction never obtained a separate physiological qualification.

Corrected REF RMSE is4.196Hz. Eta.25/.5/2/4 gives8.741/6.687/8.104/11.870Hz; increased feedback gives4.182/4.150/4.363Hz; fraction.25 gives4.377Hz. A small conditioning-RMSE change is not sufficient biological validation, and this round does not jointly fit spacing/extinction or identify an acceptable parameter manifold. The source workbook/publication discrepancies and slow-lifetime identifiability limit remain. [Physiology results](results/physiology.csv), [explicit missing mappings](final_tables/physiology_not_identified.json), [prior source audit](../physiology_source_audit_20260913/REPORT.md).

This was a bounded audit, not a test of every foundational alternative. Still untested are a whole-history continuous-time refit; an ideal observer for all tasks; a general controllability limit; richer sensory or contextual biology; fresh confirmation of a positive dose/adaptation intervention; and an autonomous safe rule for choosing stronger revision. Structural plasticity, replay, sleep and additional teaching dimensions were not introduced or rejected.

## Integrity, cost and delivery

All493 job receipts and their output hashes verify, as do478 parent assets, the frozen scientific code, selection and fresh inputs. Every campaign has its complete prescribed endpoint grid. Development gates were independently reconstructed. All576 recorded confirmation capacity endpoints were replayed;192 checkpoint-level independent direct-step probe checks have maximum disagreement **1.33e−14Hz**. The report audit contains12,327 assertions, mostly individual hash/uniqueness checks; this count is not12,327 independent scientific tests. The34 pre-run software checks also remain available. No frozen scientific source, parameter or selection was changed during reporting.

Unattended wall time was **2,101 seconds,35.0 minutes**; worker job time sums to2,278 seconds, excluding some orchestration/startup overhead. Peak recorded worker RSS was551,469,056 bytes, about526MiB per process, not total machine memory. All candidate learners retain5,177 KCs and165,688 mutable bytes. Evaluator tags, checkpoint files and optimization buffers are not counted as deployable learner memory.

The final compact supplement includes this report, final history-level/aggregate evidence, four PNG/PDF figures, configuration, checks and new reporting code. It references the immutable runner source, parent assets, input arrays and raw job outputs by path/hash rather than duplicating them. Source datasets, raw per-seed outputs, caches, earlier archives and unchanged code are excluded. **The runner's original COMPLETED_REPORT_PENDING record is preserved as execution provenance; REPORT_DELIVERY.json records report delivery separately.**

## Implication for the next decision

V74 strengthens the case for checking operating conditions and task semantics before another anatomical mechanism search. It does not show that one global learning-rate adjustment solves persistent learning. More reversal evidence can help; indiscriminate state correction can also help the target while damaging others; and severe output saturation can disguise stored distinctions.

The next design should first decide which measured limitation it addresses: safe selection of stronger corrective evidence, expression calibration under changing activity, or a state/update interface that approaches the simple baseline on matched-information tasks. Any of these needs a frozen positive hypothesis and independent confirmation, rather than combining all apparent development improvements. The existing parent remains the reference, and the exact fast-gated contraction family is not advanced.
