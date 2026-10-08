# 108 条历史记录的自主学习目标筛选

2026年10月7日。原证据与原预算类别不改写；新增字段只用于学习信号来源和下一轮预算。108条与71项去重研究池重叠，不能相加。这里不把实现状态当作自主学习阳性，也不声称对每份历史训练代码完成了来源审计。重点信号代码及新目标见[重筛说明](AUTONOMOUS_LEARNING_RESCREEN.md)。

分类统计：`{"REFERENCE_NOT_AUTONOMOUS_PROOF": 30, "ENABLING_COMPONENT": 12, "NOT_NEXT_ROUND": 15, "PRIOR_RECIPE_STOP": 21, "PRIORITY_DESIGN": 2, "SUPERVISED_OR_SIGNAL_HOLD": 21, "CONDITIONAL_AFTER_STREAM_BASELINE": 7}`。原21项配方预算关闭保持原样；其余仍不能以未否定推断值得直接运行。

| 原ID | 机制或主张 | 原证据 | 自主目标下预算 | 理由 |
|---|---|---|---|---|
| M001 | Stateful nonlinear ELM cell (fast s, slow m, recurrent messages, internal MLP) | IMPLEMENTED; PARTIAL TASK EVIDENCE | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M002 | Scalar-channel ELM with private vector memory | IMPLEMENTED; COMPARATIVE COVERAGE INCOMPLETE | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M003 | Fixed-time-constant ELM | SHIPPED; COMPLETE RESULT NOT RECOVERED | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M004 | Linear-dendrite ELM | SHIPPED; COMPLETE RESULT NOT RECOVERED | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M005 | Simple scalar leaky recurrent cell | IMPLEMENTED/SHIPPED; PARTIAL HISTORICAL EVIDENCE | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M006 | Adaptive-threshold leaky cell (ALIF-labelled) | SHIPPED; COMPLETE RESULT NOT RECOVERED | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M007 | Scalar sparse GRU-like cell | SHIPPED; COMPLETE RESULT NOT RECOVERED | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M008 | RG-LRU / simplified Griffin recurrent model | IMPLEMENTED; LIMITED TASK LOGS | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M009 | FLM fast/slow leaky cell with sparse punctuator | IMPLEMENTED; LANGUAGE/THROUGHPUT LOGS | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M010 | Multi-timescale private neuronal state | PARTIAL POSITIVE | ENABLING_COMPONENT | 可组织感觉、过去资格、地址兼容或内部动态；本身不提供已验证的自主内容学习、保持和输出。 |
| M011 | Stubborn/adaptable heterogeneous populations | PARTIAL POSITIVE; SPECIFICITY NARROWED | ENABLING_COMPONENT | 可组织感觉、过去资格、地址兼容或内部动态；本身不提供已验证的自主内容学习、保持和输出。 |
| M012 | Separate write and forget gates | IMPLEMENTED; NOT ISOLATED DECISIVELY | ENABLING_COMPONENT | 可组织感觉、过去资格、地址兼容或内部动态；本身不提供已验证的自主内容学习、保持和输出。 |
| M013 | E/I signs, proper log-tau, homeostasis, blank-activity penalty, delay training | PARTLY IMPLEMENTED; AUDIT FLAGS | NOT_NEXT_ROUND | 没有当前合法连续自学习操作与独立收益依据；保留原条目科学范围，不据此证伪整族。 |
| M014 | Raw-node spiking LIF / richer conductance cells | EXTERNAL IMPLEMENTATIONS / PROPOSED LOCALLY | NOT_NEXT_ROUND | 没有当前合法连续自学习操作与独立收益依据；保留原条目科学范围，不据此证伪整族。 |
| M015 | Type-shared templates versus merged neurons | DESIGN CONSTRAINT | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M016 | Fixed sparse sensory/recurrent graphs and retinotopic prior | IMPLEMENTED; REFERENCE | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M017 | Prune-only / random-regrowth / guided-regrowth | IMPLEMENTED; CONTROL/AUDIT LIMITATIONS | ENABLING_COMPONENT | 可组织感觉、过去资格、地址兼容或内部动态；本身不提供已验证的自主内容学习、保持和输出。 |
| M018 | New-edge lesion and topology-only transfer | SHIPPED; FULL OUTCOMES NOT RECOVERED | NOT_NEXT_ROUND | 没有当前合法连续自学习操作与独立收益依据；保留原条目科学范围，不据此证伪整族。 |
| M019 | Global cosine-kNN graph rewiring, hubs, EMA routing | IMPLEMENTED; CAUSALITY ISSUES IN AUDIT | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M020 | Local sensory-only development | TESTED; PARTIAL POSITIVE | ENABLING_COMPONENT | 感觉本身可驱动局部开发，保留source中clean-target/observed-target与无标签局部规则的权限差异；开发表示不是学得内容的证明。 |
| M021 | Clean latent reconstruction target | IDENTIFIED PRIVILEGE | NOT_NEXT_ROUND | 没有当前合法连续自学习操作与独立收益依据；保留原条目科学范围，不据此证伪整族。 |
| M022 | Frozen, live, split-stable encoders; FF/FL/LF/LL address/content states | TESTED; TRADEOFF | ENABLING_COMPONENT | 连续感觉表示改变时保留地址/内容兼容问题；不把固定表示原配方重新开成遗漏实验。 |
| M023 | Dense affine/readout transport and anchor transport | TESTED; PARTIAL/NEGATIVE | NOT_NEXT_ROUND | 没有当前合法连续自学习操作与独立收益依据；保留原条目科学范围，不据此证伪整族。 |
| M024 | Learned down-stream adapter/recycling | TESTED; LOCALITY DISQUALIFICATION | NOT_NEXT_ROUND | 没有当前合法连续自学习操作与独立收益依据；保留原条目科学范围，不据此证伪整族。 |
| M025 | Learned operator mixtures / neural algebra / global group routing | IMPLEMENTED; COMPOSITE DEMOS | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M026 | Sinkhorn routing, attention hubs, global workspace, token MoE | IMPLEMENTED; OUTSIDE CURRENT MAINLINE | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M027 | Hebbian key/value fast matrices; residual/predictive Hebbian writes | IMPLEMENTED; MIXED | PRIORITY_DESIGN | 旧fast-weight源码实际接收label，历史结果混合。保留新的实际byte时序关联/自身误差版本为优先设计，旧版本不改名复活。 |
| M028 | Modern Hopfield memory and episodic tape | IMPLEMENTED; WORKING-MEMORY DEMOS | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M029 | Gauge/fiber semantic–surface factorization; contextual router | IMPLEMENTED; COMPOSITE/SUPERVISED | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M030 | SAC/proposal policies, learned teacher, dialogue warmup, curiosity/ASK actions | IMPLEMENTED; CONFOUNDED FOR CURRENT GOAL | REFERENCE_NOT_AUTONOMOUS_PROOF | 老师模型、语言奖励和任务teacher不能混进自主核心证据；只保留实现参照和已指出的混淆。 |
| M031 | MagmaAdamW/global optimizers; neuromorphic detach mode | IMPLEMENTED; LOCALITY LIMIT | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M032 | Chunking, parallel Hebbian scan, sparse punctuator, scaling optimizations | IMPLEMENTED; ENGINEERING | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M033 | Current-only classifier / ordinary normalized delta | TESTED BASELINE | SUPERVISED_OR_SIGNAL_HOLD | 历史结果继续有效，但没有完成不依赖任务教师的当前核心验证；需独立定义信号、时序和原生贡献。 |
| M034 | Cumulative ridge sufficient statistics / exact RLS | POSITIVE WITH LIMITS | SUPERVISED_OR_SIGNAL_HOLD | 历史结果继续有效，但没有完成不依赖任务教师的当前核心验证；需独立定义信号、时序和原生贡献。 |
| M035 | Slow-only versus combined fast+slow output | TESTED; COMPOSITION MATTERS | SUPERVISED_OR_SIGNAL_HOLD | 历史结果继续有效，但没有完成不依赖任务教师的当前核心验证；需独立定义信号、时序和原生贡献。 |
| M036 | Fast-feedback weighted ridge consolidation | SMALL SPECIFIC EFFECT; BELOW BASELINE | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M037 | Additive consolidation bonus | NEGATIVE EXACT | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M038 | Synaptic importance / neuron availability / protection | PARTIAL POSITIVE; RIGIDITY | SUPERVISED_OR_SIGNAL_HOLD | 历史结果继续有效，但没有完成不依赖任务教师的当前核心验证；需独立定义信号、时序和原生贡献。 |
| M039 | Bayesian/diagonal uncertainty; MESU-inspired gains | PARTIAL THEN INCONCLUSIVE/NEGATIVE IMPLEMENTATIONS | SUPERVISED_OR_SIGNAL_HOLD | 历史结果继续有效，但没有完成不依赖任务教师的当前核心验证；需独立定义信号、时序和原生贡献。 |
| M040 | Orthogonal projection / strict/leaky protected subspaces | TESTED TRADEOFF | NOT_NEXT_ROUND | 没有当前合法连续自学习操作与独立收益依据；保留原条目科学范围，不据此证伪整族。 |
| M041 | Two- and three-state coupled synapses | TESTED NEGATIVE SCREEN | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M042 | Feedback-gated local synaptic coupling | TESTED NEGATIVE SCREEN | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M043 | Discrete cascade depth 4/8 | TESTED TRADEOFF/NEGATIVE | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M044 | Active forgetting / surprise-triggered reopening / capacity release | TESTED MIXED/NEGATIVE | NOT_NEXT_ROUND | 没有当前合法连续自学习操作与独立收益依据；保留原条目科学范围，不据此证伪整族。 |
| M045 | Sparse distributed address memory and adaptive prototypes | TESTED; ADAPTATION COST | NOT_NEXT_ROUND | 没有当前合法连续自学习操作与独立收益依据；保留原条目科学范围，不据此证伪整族。 |
| M046 | Allocate/recycle local records | TESTED; SATURATION BIAS | NOT_NEXT_ROUND | 没有当前合法连续自学习操作与独立收益依据；保留原条目科学范围，不据此证伪整族。 |
| M047 | Quantized allocation / integer precision | TESTED; ACCOUNTING CAVEAT | NOT_NEXT_ROUND | 没有当前合法连续自学习操作与独立收益依据；保留原条目科学范围，不据此证伪整族。 |
| M048 | Separate banks, fixed vs inferred routing, hard/blended reading | PARTIAL POSITIVE REPLICATED | SUPERVISED_OR_SIGNAL_HOLD | 历史结果继续有效，但没有完成不依赖任务教师的当前核心验证；需独立定义信号、时序和原生贡献。 |
| M049 | Silent memories, context reinstatement, unlabeled reminders | PROMISING SCREEN; ASSAY LIMIT | CONDITIONAL_AFTER_STREAM_BASELINE | 先有连续流核心自主学习，再按实际缺陷启用；不因名称含内部/预测就自动发射。 |
| M050 | Bounded reservoir/recent replay, current repetition, fast-head replay | TESTED MIXED; NOT CURRENT CANDIDATE | CONDITIONAL_AFTER_STREAM_BASELINE | 先有连续流核心自主学习，再按实际缺陷启用；不因名称含内部/预测就自动发射。 |
| M051 | Recent cache retrieval, hard/soft episodic reads | TESTED MOSTLY NULL | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M052 | Spillover, local normalization, fixed-target correction, update-norm controls | TESTED; BENEFIT NARROWED | NOT_NEXT_ROUND | 没有当前合法连续自学习操作与独立收益依据；保留原条目科学范围，不据此证伪整族。 |
| M053 | Dense Hebbian/Oja encoder adaptation | TESTED TRADEOFF | ENABLING_COMPONENT | 感觉Hebb/Oja可无答案更新，但旧改坐标会破坏已存内容；原P负结果继续约束原配方，只有不同连续活动规则才待设计。 |
| M054 | Novelty/shuffled/norm gates for encoder change | NEGATIVE SPECIFICITY | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M055 | Competitive top-response plasticity, random subset, bottom-k | NO ESTABLISHED FRONTIER WIN; TASK ISSUES | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M056 | Temporal contrast / fast-minus-slow sensory channels | PARTIAL POSITIVE | ENABLING_COMPONENT | 可组织感觉、过去资格、地址兼容或内部动态；本身不提供已验证的自主内容学习、保持和输出。 |
| M057 | Variance salience / coordinate selection | TASK-SPECIFIC | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M058 | Full covariance/co-activation-directed writes | POSITIVE WITH COST/REVISION LIMITS | SUPERVISED_OR_SIGNAL_HOLD | 历史结果继续有效，但没有完成不依赖任务教师的当前核心验证；需独立定义信号、时序和原生贡献。 |
| M059 | Lag96 temporal-position-preserving byte encoding | TASK-USEFUL | ENABLING_COMPONENT | 可组织感觉、过去资格、地址兼容或内部动态；本身不提供已验证的自主内容学习、保持和输出。 |
| M060 | Four sparse compartments with local covariance | TESTED NEGATIVE | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M061 | Shared residual, local residual, absolute local values | STRONG ENGINEERING DECOMPOSITION | ENABLING_COMPONENT | 残差参考漂移是直接历史限制。自行产生预测误差作更新信号，不等于把漂移参考下的残差当作恒定可读内容。 |
| M062 | Exact dictionary versus generalizing ownership | POSITIVE ENGINEERING, UNBOUNDED | NOT_NEXT_ROUND | 没有当前合法连续自学习操作与独立收益依据；保留原条目科学范围，不据此证伪整族。 |
| M063 | Read authority: HARD/SHADOW/CORR2/SIM-MIX/fixed alpha/random | POSITIVE ENGINEERING | SUPERVISED_OR_SIGNAL_HOLD | 读权限和混合只读已有值，不生成内容学习目标。Q_HALF未采用、LINK v1有效筛选停止分别保留，不能整体归因成自主学习。 |
| M064 | Predictive Bayes arbitration, slot evidence, shuffled-PBF | PARTIAL; SPECIFICITY GATE FAIL | SUPERVISED_OR_SIGNAL_HOLD | 历史结果继续有效，但没有完成不依赖任务教师的当前核心验证；需独立定义信号、时序和原生贡献。 |
| M065 | TRACE-CORR probation and distinct-rendering admission | USEFUL BOUNDED ENGINEERING | SUPERVISED_OR_SIGNAL_HOLD | 历史结果继续有效，但没有完成不依赖任务教师的当前核心验证；需独立定义信号、时序和原生贡献。 |
| M066 | Fixed-cap slots, exact-repeat admission, shuffled-pair admission | TESTED CONTROLS | SUPERVISED_OR_SIGNAL_HOLD | 历史结果继续有效，但没有完成不依赖任务教师的当前核心验证；需独立定义信号、时序和原生贡献。 |
| M067 | Bounded superposition, exact hashing, separation ladder, load scaling | PARTIAL POSITIVE; CAPACITY LOSS | SUPERVISED_OR_SIGNAL_HOLD | 历史结果继续有效，但没有完成不依赖任务教师的当前核心验证；需独立定义信号、时序和原生贡献。 |
| M068 | Read-only hybrid versus full feedback into admission | DECISIVE DESIGN DISTINCTION | SUPERVISED_OR_SIGNAL_HOLD | 历史结果继续有效，但没有完成不依赖任务教师的当前核心验证；需独立定义信号、时序和原生贡献。 |
| M069 | Utility-based replacement; LFU-rate/LRU/random; TRACE-U | INVALID/UNDERPOWERED ASSAY | SUPERVISED_OR_SIGNAL_HOLD | 历史结果继续有效，但没有完成不依赖任务教师的当前核心验证；需独立定义信号、时序和原生贡献。 |
| M070 | Mute probationary exact traces; supported similarity retention | FIXED-HISTORY POSITIVE AUDIT | SUPERVISED_OR_SIGNAL_HOLD | 历史结果继续有效，但没有完成不依赖任务教师的当前核心验证；需独立定义信号、时序和原生贡献。 |
| M071 | Local/global Bayesian authority; zero/E2 init; fixed50; shuffled alpha | ARBITRATION SIGNAL; PRIMARY VALIDITY FAIL | SUPERVISED_OR_SIGNAL_HOLD | 历史结果继续有效，但没有完成不依赖任务教师的当前核心验证；需独立定义信号、时序和原生贡献。 |
| M072 | Trace versus mature-slot capacity relief | STRONG CONDITIONAL CAUSAL RESULT | SUPERVISED_OR_SIGNAL_HOLD | 历史结果继续有效，但没有完成不依赖任务教师的当前核心验证；需独立定义信号、时序和原生贡献。 |
| M073 | General/phase normalization, KC expansion, coding-level sweeps | TESTED; DRIFT CONFOUND | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M074 | PHASE-FROZEN versus PHASE-LIVE versus LAG | ACTUALLY TESTED | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M075 | Corroborated distributed synaptic consolidation (Candidate A) | SMALL SPECIFIC SIGNAL; ARCHITECTURE WEAKER | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M076 | 32-factor coded output with dense/modular/rewired access (Candidate B) | COMPRESSION TRADEOFF; TOPOLOGY UNSUPPORTED | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M077 | Reusable positive-pair invariance (Candidate C) | REPLICATED SYNTHETIC REPRESENTATION EFFECT | SUPERVISED_OR_SIGNAL_HOLD | 历史pair corroboration使用后来观察到的byte，有无标签成分；但配对合法性和同答案混淆仍存在，未验证当前连续流保持。 |
| M078 | Invariance plus global slow consolidation factorial | REAL SMALL GAIN; QUALIFICATION FAIL | SUPERVISED_OR_SIGNAL_HOLD | 历史结果继续有效，但没有完成不依赖任务教师的当前核心验证；需独立定义信号、时序和原生贡献。 |
| M079 | Learned overlapping pathway responsibility | NEGATIVE EXACT | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M080 | Signed protection within PAIR nuisance subspace | NEGATIVE EXACT WITH LOCAL TRADEOFF | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M081 | One-byte surprise filter on negative development | NO DETECTABLE ADDED VALUE | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M082 | Exact sparse encoding of local output values | IMPLEMENTED UNIT TEST; NOT FULL SYSTEM RESULT | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M083 | Full graph reduction and core peeling | STRUCTURAL RESULT ONLY | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M084 | Direct/indirect MBON→DAN relay distillation | STRUCTURAL RESULT ONLY | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M085 | Unsigned composed KC→MBON→DAN→KC spatial gate | STRUCTURAL/ABSTRACT NEGATIVE | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M086 | Output gain modulation; exact/compressed/shuffled anatomy | TESTED NULL | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M087 | Input-only anatomical write addressing | TESTED NO ESTABLISHED ADVANTAGE | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M088 | Joint input/readout addressing and scrambled control | TESTED NO BROAD ADVANTAGE | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M089 | KC-family/lobe bins, whole-cell overlap compartment candidates | STRUCTURAL DESCRIPTORS, NOT PHYSIOLOGY | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M090 | Raw KC↔APL scalar suppressive controller | NEGATIVE EXACT | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M091 | Aligned raw PN→KC→MBON plastic mask | LEARNS CURRENT; NO RAW SUPERIORITY ESTABLISHED | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M092 | Recorded-count heterogeneity, binary support, count shuffle | TRADEOFF; SPECIFIC ASSIGNMENT UNRESOLVED | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M093 | Unsigned linear raw MBON recurrence at radius0.25 | NEGATIVE EXACT | PRIOR_RECIPE_STOP | 保留此前有效负面、因果/资格不合格或预算停止的具体配方；自主目标不使旧失败配方原样复活。 |
| M094 | KC↔KC weak recurrence, DPM, broad modulatory feedback | STRUCTURALLY MEASURED; FUNCTIONAL TEST INCOMPLETE | ENABLING_COMPONENT | 真实递归/调制块仍缺完整合格动态学习实验。作为潜在构件，不用边数或信号名称证明内生学习。 |
| M095 | Full raw-graph functional reference followed by lesion/distillation | PROPOSED; NO RECOVERED LOCAL SUCCESS | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M096 | Bidirectional/local DAN teaching with delayed eligibility | PARTIAL PROXIES; FULL ASSAY UNESTABLISHED | PRIORITY_DESIGN | 优先明确真实观察生成的局部更新信号和资格recipient。现有监督DAN代理没有完成此验证，不借生物名称证明自主学习。 |
| M097 | Two-unit recurrent STM→LTM teaching (Fly-2) | PROPOSED; PROVENANCE WARNING | CONDITIONAL_AFTER_STREAM_BASELINE | Fly-2为提议并有来源警告；内部STM教LTM不自动增加真信息，不能当作完成的自主机制。 |
| M098 | Incremental long-vs-short information gate | PROPOSED; NOT SCIENTIFICALLY RUN | CONDITIONAL_AFTER_STREAM_BASELINE | 长短预测比可从真实后来byte计算，但须先有两个合法在线预测；窗口未来信息不能在线分配写入。 |
| M099 | Multi-horizon residual hierarchy | PROPOSED | CONDITIONAL_AFTER_STREAM_BASELINE | 多horizon目标须等真实byte到达后更新先前资格活动；单步基线未成立前不叠加整套层次。 |
| M100 | Fixed plasticity quota; competition across time | PROPOSED | CONDITIONAL_AFTER_STREAM_BASELINE | 先有连续流核心自主学习，再按实际缺陷启用；不因名称含内部/预测就自动发射。 |
| M101 | Homeostatic forgetting / targeted bidirectional extinction | SPECIFIC PROXIES TESTED; FULL CLAIM OPEN | SUPERVISED_OR_SIGNAL_HOLD | 历史结果继续有效，但没有完成不依赖任务教师的当前核心验证；需独立定义信号、时序和原生贡献。 |
| M102 | Persistent scratchpad and feedback memory | PROPOSED/RELATED IMPLEMENTATIONS | CONDITIONAL_AFTER_STREAM_BASELINE | 先有连续流核心自主学习，再按实际缺陷启用；不因名称含内部/预测就自动发射。 |
| M103 | Short-context/phase+2 slot admission | ONE-SEED DIAGNOSTIC; FULL GRID NOT RUN | NOT_NEXT_ROUND | 没有当前合法连续自学习操作与独立收益依据；保留原条目科学范围，不据此证伪整族。 |
| M104 | Class-specific write lesions on V49 | PROPOSED; NOT RUN | NOT_NEXT_ROUND | 没有当前合法连续自学习操作与独立收益依据；保留原条目科学范围，不据此证伪整族。 |
| M105 | Positive-and-negative reusable metric outside protected subspace | PROPOSED FAMILY; ONE EXACT FORM FAILED | SUPERVISED_OR_SIGNAL_HOLD | 历史结果继续有效，但没有完成不依赖任务教师的当前核心验证；需独立定义信号、时序和原生贡献。 |
| M106 | Frozen/slow residual reference, transport compensation, constraint-only slow memory | PROPOSED; RELATED NEGATIVE PRECEDENT | ENABLING_COMPONENT | 保存慢/冻结参考与合法成本，不能让外挂预测器改变后使存储残差失去原语义。 |
| M107 | Indexed retrieval (hash/tree/HNSW/ANN), exact value compaction | ENGINEERING BACKLOG | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
| M108 | Natural byte corpora, varied wrappers, nonshared transforms | PROPOSED/LARGELY DEFERRED | REFERENCE_NOT_AUTONOMOUS_PROOF | 实现、结构、反模型或范围参照保留；完整自主持续学习来源或当前核心贡献未经这些历史摘要认证。 |
