# U055归属纠正与当前加法机制重新筛选

> 版本级补充：上一轮方向排序尚未覆盖具体版本，尤其漏述了已测REPLAY-P和TOPO-G。请以[版本复核](VERSION_REGISTER_20261011/REVIEW.md)和机器可读版本索引补充判断，不能将该方向排序当作最终配方选择。


2026-10-11。目标固定：在带噪/缺失答案的外部样本下，用更少样本和计费计算，获得对**未见加法组合**的自身输出能力；旧知识保持是保护项。当前用户采用的开发默认是COV-SUPPORT，比较锚点为COV-R。此次只更新设计优先级，未运行、授权或上传新实验。

## 先纠正U055

此前将“多观察联合重估”归为U055并列为下一方向，依据不足。U055对应原账M094的KC↔KC弱递归、DPM与广义调制反馈，现有来源是结构测量与未完成的功能验证。通用的批估计、回放或鲁棒重加权都不自动属于这类电路。原始有符号/无符号递归版本的负面结果也不证明全部递归无用。U055保留为条件备选，必须先写清真正的反馈状态、电路、学习规则、终止条件和成本。

## 当前结果能支持什么

- [SAMPLE-GEN确认](../../addition_sample_generalization_dev_v2_20261011/results/CONFIRM_REPORT.md)支持完整相关的COV-R优于ID/DIAG-R；现有核心已经共享操作数约束，不能再以“首次联合利用样本”包装已有更新。
- [FEEDBACK-EFFICIENCY确认](../../addition_feedback_efficiency_dev_v1_20261011/results/CONFIRM_REPORT.md)：EXTRA低样本增量约+1.07pp，95%区间[−1.08,+3.23]，原采用门未通过；额外更新与均匀REPLAY均未获得升级资格。它们消耗更多计算，没有证成更高样本效率。不是整个选择性再激活家族的反证。
- [U042重新分析](../../addition_observation_support_dev_v1_20261011/posthoc/reanalysis_20261011/REANALYSIS.json)：BLURRY新组合曲线增量−0.83pp，探索区间[−3.79,+2.12]；旧组合终点+25.83pp。用户采用SUPPORT不等于独立确认新组合效率提高，也不等于证明它有害。
- 当前P更新随允许观察收缩，不是经校准的噪声后验。早期噪声是否造成过度自信仍未证实。训练真值静态，不能把任何大残差都当作需要遗忘的环境变化。

## 优先考虑的具体方向

| 条目 | 具体操作方向 | 为什么重新保留 | 必须分开的风险/对照 |
|---|---|---|---|
| **U047 选择性再激活** | 有界保存真实已观察输入/答案，选择不同操作数和约束方向、当前信息不足的经验，而非均匀重复或挑最大误差；不生成真值 | 直接改变相同外部样本被利用的方式；旧均匀回放失败没有测试这种选择 | 与同缓存、同内部更新预算的均匀/随机选择比较；重复经验不算新独立证据，所有读取与计算计费；原始答案含噪，选择也可能放大噪声 |
| **U034 有支持的不确定性重开** | 多条实际观察形成持续预测冲突时，只重新开放对应方向的可塑性/不确定性，保持完整相关结构；不是全局衰减 | 当前完整协方差不同于旧对角Bayes近似；可检验早期错误是否过度固化 | 当前P并非真实后验；重开可能破坏保留或学习噪声。必须定义支持条件与正定性，不用单次惊讶、不看held-out标签 |
| **U059 因果预算分配（第二顺位）** | 固定更新预算内，让重复低信息观察和新的约束方向竞争；预算只依据当前/过去状态 | 对齐单位计算获得能力，原版本尚未形成因果科学结果 | 仅减少更新不是样本效率提高；要与同预算随机/等额分配比较，不能按未来窗口分配 |
| U043 权重与真实换边协同 | 保留的结构替代方向：明确连接如何改变共享约束 | 旧单独T/S配方没有排除协同 | 当前完整相关结构有实际收益，稀疏化可能丢信息；换边与新增先验不得混算 |
| U021 稳定/灵活分区 | 仅在当前学习曲线暴露获取与保护冲突时定义分工 | 历史保持任务不是本任务的整族反证 | SUPPORT已有旧组合保护信号；分bank可能阻断共享，不先为容量扩容 |
| U055 真实递归/反馈 | 定义原来源所指电路，而不是重新命名批优化 | 未完整测试，仍有结构可能性 | 没有当前任务功能收益，不默认第一；反馈不新增外部信息，全部迭代计费 |
| U057 / U058 | 两个合法预测源的增量信息门 / 有限多horizon层级 | 保留未定义方向，未被彻底否定 | 当前没有第二预测源或长延迟短板；不通过更换任务让它们显得必要 |

这给出**2项优先设计方向、6项条件备选**，不是8个已可运行配方。U047与U034都没有新结果证明会赢；优先级来自对当前目标的契合与旧配方的区别。若要优先定义一个，我倾向U047：它可以直接与已经失败的均匀回放在相同经验与预算下比较；U034的“过度收缩”前提更弱。此判断不是运行授权。

## 其他机制如何处理

- U042/U045已在开发核心，U018保留原生E0/E1参照；三项采用记录不能合称三个已确认算术机制。
- U044、U062是稳定坐标/约束内容兼容构件。保存原始输入与观察答案可避免残差参考漂移；但“constraint-only memory”尚无新成功实现，不另行多算候选。
- U069/U070仍是表示构件。将左右角色权重直接绑定会加入交换律先验，不能仅凭名称算作KC异质性或PN采样机制；若研究它，必须明确先验并赋予公平对照。
- 不原样复活旧均匀REPLAY/EXTRA、P/T/S参数补救、SIM-MIX/LINK或单次surprise遗忘。负面或无效的范围分别保留，不能因此宣布整族死刑。
- 时序/eligibility、多时间尺度巩固、额外bank、缓存替换和读权限目前没有直接当前样本效率缺口；这是预算排序，非科学淘汰。旧“24键”“99.12%保持”“E3只是可选”的理由均不再用作当前任务结论。
- 上游RLS/full-cov（M034/M058）是当前相关更新的参照；ELM/GRU/learned optimizer等旧工程记录没有公平的本任务效能结果，不因未测就自动优先，也不以生物局部性禁令排除未来工程方案。

## 全账数量与证据边界

复核了108行历史总账、136行目录的交叉引用及71条去重保留集合。108与136相互重叠，不能相加；71本来是条件保留子集，并非全部可能机制。此次只把U034、U059从旧任务的暂缓状态重新列入当前候选；没有新增科学证伪。全部71项逐条理由和上游交叉引用见[机器可读复核](POST_U055_RESCREEN_20261011.json)。

| 当前分类 | 条数 |
|---|---:|
| ADOPTED_BASELINE | 3 |
| CONDITIONAL_CANDIDATE | 6 |
| ENABLING_COMPONENT | 8 |
| NOT_NEXT_ROUND | 46 |
| PRIORITY_DESIGN | 2 |
| RECIPE_HOLD | 4 |
| REFERENCE_ONLY | 1 |
| STOP_CURRENT_RECIPE | 1 |

3项采用、2项优先设计、6项条件备选、8项构件、52项非当前候选/暂停/参照，共71项。旧结果与来源ID不变；不是新的“剩余未被证伪变体总数”。

## 71项逐条复核

| ID | 原名称 | 来源 | 当前分类 | 当前目标下理由 |
|---|---|---|---|---|
| U001 | FE2 / Temporal contrast | C003 | NOT_NEXT_ROUND | 感觉时序/对比的旧结果不排除整族；当前相同字节组合已可进入角色特征，尚无当前低样本错误由读取时序造成的证据，不回到编码器修补。 |
| U002 | LAT4 / Latency classes | C007 | NOT_NEXT_ROUND | 感觉时序/对比的旧结果不排除整族；当前相同字节组合已可进入角色特征，尚无当前低样本错误由读取时序造成的证据，不回到编码器修补。 |
| U003 | P4 / ITDP-like | C015 | NOT_NEXT_ROUND | 当前答案在完整表达式后到来，已有pending context连接答案与输入；更长eligibility/STC须说明新增的当前信息，不能用旧延迟任务收益替代未见加法收益。 |
| U004 | E3dev / Local afferent development plus homeostasis | C016, M020 | ENABLING_COMPONENT | 输入发展/Oja为可选表示构件，旧P1/P2及失败技术P3分别保留边界；改变输入必须保证旧内容兼容。当前不作为独立低样本候选。 |
| U005 | — / Slow bounded prune/grow `S` | C035 | NOT_NEXT_ROUND | 结构/抑制并未整族被排除，但没有当前稀疏竞争导致错误的证据。需新的明确信息组织操作；不原样恢复旧S/T/APL配方或把调稀疏度当泛化机制。 |
| U006 | — / Topology × commitment K0–K3 | C039 | NOT_NEXT_ROUND | 结构/抑制并未整族被排除，但没有当前稀疏竞争导致错误的证据。需新的明确信息组织操作；不原样恢复旧S/T/APL配方或把调稀疏度当泛化机制。 |
| U007 | — / Structural eligibility | C040, C056 | NOT_NEXT_ROUND | 结构/抑制并未整族被排除，但没有当前稀疏竞争导致错误的证据。需新的明确信息组织操作；不原样恢复旧S/T/APL配方或把调稀疏度当泛化机制。 |
| U008 | Circuit-generated suppressive/decorrelating `A` | C042 | NOT_NEXT_ROUND | 结构/抑制并未整族被排除，但没有当前稀疏竞争导致错误的证据。需新的明确信息组织操作；不原样恢复旧S/T/APL配方或把调稀疏度当泛化机制。 |
| U009 | Recurrent inhibition matched for mean activity | C048 | NOT_NEXT_ROUND | 结构/抑制并未整族被排除，但没有当前稀疏竞争导致错误的证据。需新的明确信息组织操作；不原样恢复旧S/T/APL配方或把调稀疏度当泛化机制。 |
| U010 | Homeostatic rebound via KC microcircuits (#23) | C049 | NOT_NEXT_ROUND | 结构/抑制并未整族被排除，但没有当前稀疏竞争导致错误的证据。需新的明确信息组织操作；不原样恢复旧S/T/APL配方或把调稀疏度当泛化机制。 |
| U011 | A_ELIG / Multi-frame KC eligibility trace (OMNIBUS10) | C050 | NOT_NEXT_ROUND | 当前答案在完整表达式后到来，已有pending context连接答案与输入；更长eligibility/STC须说明新增的当前信息，不能用旧延迟任务收益替代未见加法收益。 |
| U012 | — / `ELIG_TRACE_PORT` | C051 | NOT_NEXT_ROUND | 当前答案在完整表达式后到来，已有pending context连接答案与输入；更长eligibility/STC须说明新增的当前信息，不能用旧延迟任务收益替代未见加法收益。 |
| U013 | — / STC tag + limited capture resource | C054 | NOT_NEXT_ROUND | 当前答案在完整表达式后到来，已有pending context连接答案与输入；更长eligibility/STC须说明新增的当前信息，不能用旧延迟任务收益替代未见加法收益。 |
| U014 | `pair_importance` | C059, M038 | NOT_NEXT_ROUND | 重要性/可用性/元可塑性可能保护旧知识，也会抑制新学习。当前必须证明从观察噪声和未见组合中获益，旧24键保持率不能直接作为本任务否定依据。暂不优先。 |
| U015 | `neuron_availability` | C061, M038 | NOT_NEXT_ROUND | 重要性/可用性/元可塑性可能保护旧知识，也会抑制新学习。当前必须证明从观察噪声和未见组合中获益，旧24键保持率不能直接作为本任务否定依据。暂不优先。 |
| U016 | `E_META` | C064 | NOT_NEXT_ROUND | 重要性/可用性/元可塑性可能保护旧知识，也会抑制新学习。当前必须证明从观察噪声和未见组合中获益，旧24键保持率不能直接作为本任务否定依据。暂不优先。 |
| U017 | Receiver availability / saturation states (V69–V71) | C069 | NOT_NEXT_ROUND | 重要性/可用性/元可塑性可能保护旧知识，也会抑制新学习。当前必须证明从观察噪声和未见组合中获益，旧24键保持率不能直接作为本任务否定依据。暂不优先。 |
| U018 | Consequence-specific DAN-like teaching | C076, M096 | ADOPTED_BASELINE | 保留原生观察学习E0/E1参照；不是当前工程COV-SUPPORT的独立加法E3证据。 |
| U019 | Delayed eligibility-gated teaching | C077, M096 | ENABLING_COMPONENT | 当前答案在完整表达式后到来，已有pending context连接答案与输入；更长eligibility/STC须说明新增的当前信息，不能用旧延迟任务收益替代未见加法收益。 |
| U020 | H / Heterogeneous compartments | C085 | RECIPE_HOLD | 原配方保持暂停：分区、双向教学、读权限或nuisance配对需要当前任务的合法信号与公平对照；同答案不等于同题，未获得这些条件不列入下一轮。 |
| U021 | — / Stable/flexible compartment split | C086 | CONDITIONAL_CANDIDATE | 稳定/灵活分区留作条件备选。当前SUPPORT的BLURRY旧组合终点98.3%是六世界探索结果，不证明永不遗忘；但暂无理由先增加存储分区而不是改善低样本新组合。 |
| U022 | V23 `active_forgetting` | C092, M044 | NOT_NEXT_ROUND | 保留旧遗忘/惊讶释放的配方负面证据，不能排除所有遗忘。当前真值静态而答案有噪声，单次冲突不足以证明旧知识过时；广泛衰减可能损害保留，不优先。 |
| U023 | `importance_surprise_release` | C094, M044 | NOT_NEXT_ROUND | 保留旧遗忘/惊讶释放的配方负面证据，不能排除所有遗忘。当前真值静态而答案有噪声，单次冲突不足以证明旧知识过时；广泛衰减可能损害保留，不优先。 |
| U024 | OMNIBUS `C_FORGET` (targeted wrong-winner decay) | C096 | NOT_NEXT_ROUND | 保留旧遗忘/惊讶释放的配方负面证据，不能排除所有遗忘。当前真值静态而答案有噪声，单次冲突不足以证明旧知识过时；广泛衰减可能损害保留，不优先。 |
| U025 | Selective downstream `R_sel` | C099, M044 | NOT_NEXT_ROUND | 保留旧遗忘/惊讶释放的配方负面证据，不能排除所有遗忘。当前真值静态而答案有噪声，单次冲突不足以证明旧知识过时；广泛衰减可能损害保留，不优先。 |
| U026 | Homeostatic forgetting / targeted bidirectional extinction | C100, M101 | NOT_NEXT_ROUND | 保留旧遗忘/惊讶释放的配方负面证据，不能排除所有遗忘。当前真值静态而答案有噪声，单次冲突不足以证明旧知识过时；广泛衰减可能损害保留，不优先。 |
| U027 | Single-DAN acquisition/erasure duality | C102, M096 | RECIPE_HOLD | 原配方保持暂停：分区、双向教学、读权限或nuisance配对需要当前任务的合法信号与公平对照；同答案不等于同题，未获得这些条件不列入下一轮。 |
| U028 | DPM-mediated consolidation | C112, M094 | NOT_NEXT_ROUND | 巩固、多时间尺度或跨流绑定需要当前收益假说；工程核心休息不衰减，当前重点是新组合低样本学习。旧任务保持数字不构成本任务整族反证，增加bank/时间常数本身不提供新约束。 |
| U029 | Spaced-training disinhibition / pERK-like gating | C115 | NOT_NEXT_ROUND | 巩固、多时间尺度或跨流绑定需要当前收益假说；工程核心休息不衰减，当前重点是新组合低样本学习。旧任务保持数字不构成本任务整族反证，增加bank/时间常数本身不提供新约束。 |
| U030 | Immediate→intermediate→slow commitment | C118 | NOT_NEXT_ROUND | 巩固、多时间尺度或跨流绑定需要当前收益假说；工程核心休息不衰减，当前重点是新组合低样本学习。旧任务保持数字不构成本任务整族反证，增加bank/时间常数本身不提供新约束。 |
| U031 | Recurrent post-event consolidation window | C119 | NOT_NEXT_ROUND | 巩固、多时间尺度或跨流绑定需要当前收益假说；工程核心休息不衰减，当前重点是新组合低样本学习。旧任务保持数字不构成本任务整族反证，增加bank/时间常数本身不提供新约束。 |
| U032 | Stubborn/adaptable heterogeneous populations | C121, M011 | NOT_NEXT_ROUND | 巩固、多时间尺度或跨流绑定需要当前收益假说；工程核心休息不衰减，当前重点是新组合低样本学习。旧任务保持数字不构成本任务整族反证，增加bank/时间常数本身不提供新约束。 |
| U033 | bayes_stable | C063, M039 | NOT_NEXT_ROUND | 置信加锁可能固定早期噪声。旧Bayes近似不等于完整Bayesian算法，也没有整族反证；本任务优先研究纠错而非加强锁定。 |
| U034 | bayes_reopening | C063, C095, M039, M044 | PRIORITY_DESIGN | 重新进入设计候选：有支持的反复预测冲突只重开相关约束方向的不确定性，不全局遗忘。旧对角近似的得失不能否定完整协方差版本。当前P不是校准后验，早期过度收缩是待检验假说而非已定位原因。 |
| U035 | context_fixed 四bank固定路由 | C081, M049 | NOT_NEXT_ROUND | 上下文分bank可隔离记忆，但本任务要求跨未见组合共享操作数信息；没有合法路由/共享规则时，新增隔离会妨碍泛化。不用隐含cohort标签路由，不原样增加bank。 |
| U036 | context_inferred 四bank推断路由 | C081, M049 | NOT_NEXT_ROUND | 上下文分bank可隔离记忆，但本任务要求跨未见组合共享操作数信息；没有合法路由/共享规则时，新增隔离会妨碍泛化。不用隐含cohort标签路由，不原样增加bank。 |
| U037 | silent_fixed 八bank固定路由 | C081, C104, M049 | NOT_NEXT_ROUND | 上下文分bank可隔离记忆，但本任务要求跨未见组合共享操作数信息；没有合法路由/共享规则时，新增隔离会妨碍泛化。不用隐含cohort标签路由，不原样增加bank。 |
| U038 | silent_inferred 八bank推断路由 | C081, C104, M049 | NOT_NEXT_ROUND | 上下文分bank可隔离记忆，但本任务要求跨未见组合共享操作数信息；没有合法路由/共享规则时，新增隔离会妨碍泛化。不用隐含cohort标签路由，不原样增加bank。 |
| U039 | CORR2 证据支持读权限 | C135, M063 | RECIPE_HOLD | 原配方保持暂停：分区、双向教学、读权限或nuisance配对需要当前任务的合法信号与公平对照；同答案不等于同题，未获得这些条件不列入下一轮。 |
| U040 | 固定 partial mixture 包括 Q_HALF | C135, M063 | REFERENCE_ONLY | Q_HALF等固定混合保留历史参照；其旧增量/因果保护失败不证明全部组合机制无效，也不作为当前算术机制收益。 |
| U041 | 细胞内部多时间尺度私有状态 | M010 | NOT_NEXT_ROUND | 巩固、多时间尺度或跨流绑定需要当前收益假说；工程核心休息不衰减，当前重点是新组合低样本学习。旧任务保持数字不构成本任务整族反证，增加bank/时间常数本身不提供新约束。 |
| U042 | 相互独立的写入门与遗忘门 | M012 | ADOPTED_BASELINE | 用户已采用SUPPORT写门。未实现独立遗忘门；不把目录家族名当已测试全部结构，不改写原DEV未晋级结果。 |
| U043 | 局部边权学习与真实换边协同 P×S | M017 | CONDITIONAL_CANDIDATE | 真实局部权重与换边协同仍作备选。旧固定图或单独换边失败不排除协同；当前COV已利用完整相关性，稀疏化可能丢约束，须说明新增共享关系且不能偷偷加入已知加法答案/角色等价先验。 |
| U044 | 稳定地址与可塑内容分离或输入坐标兼容 | M022 | ENABLING_COMPONENT | 地址稳定与内容兼容是公共构件；当前角色坐标固定。不把已经成立的接口再次包装为独立候选。 |
| U045 | 有界Hebbian key/value或predictive residual fast matrix | M027 | ADOPTED_BASELINE | 完整相关残差更新COV-R已是已确认有限E3基础，用户默认再加SUPPORT。历史RLS/full-cov证据属于参照，不能把重复实现当新机制。 |
| U046 | 有界分布式地址层与自适应原型 | M045 | NOT_NEXT_ROUND | 自适应地址可能帮助共享，但需解释当前固定角色代码的具体缺口并保护旧内容；不因旧exact-key任务通过就永久排除，也不重新转入编码器搜索。 |
| U047 | 有限选择性再激活或bounded offline O | M050 | PRIORITY_DESIGN | 优先设计：有界存储真实已观察样本，按约束方向覆盖/当前信息不足选择再激活，不按最大误差或随机均匀重复。确认中的均匀REPLAY/EXTRA未证明增量，只关闭旧配方；所有额外运算、访问和内部呈现计费，同一记录不得冒充独立证据。 |
| U048 | 真正连续活动的稳定Oja与兼容记忆的输入适应 | M053 | ENABLING_COMPONENT | 输入发展/Oja为可选表示构件，旧P1/P2及失败技术P3分别保留边界；改变输入必须保证旧内容兼容。当前不作为独立低样本候选。 |
| U049 | 历史结构化感觉输入的fast-minus-slow时间对比 | M056 | ENABLING_COMPONENT | 感觉时序/对比的旧结果不排除整族；当前相同字节组合已可进入角色特征，尚无当前低样本错误由读取时序造成的证据，不回到编码器修补。 |
| U050 | SIM-MIX 带exact-member权限的相似读出 | M063 | STOP_CURRENT_RECIPE | 关闭当前SIM-MIX/LINK配方的泛化承诺；读out近邻叠加不能替代学习规则，更不能用禁写基线下降伪造能力增加。只关闭该具体实现。 |
| U051 | Predictive Bayes arbitration 及slot evidence | M064 | NOT_NEXT_ROUND | Bayes读权限/预测仲裁需要两个合法且互补的预测源。当前不是两个输出银行相加的问题，旧authority信号不直接识别被污染答案，暂不优先。 |
| U052 | 有界utility replacement或TRACE-U | M069 | NOT_NEXT_ROUND | 有界替换是容量/缓存管理备选；当前32项缓存与角色学习瓶颈不能混同。没有当前容量拥挤证据，不先以替换修低样本泛化。 |
| U053 | 局部Bayesian读权限控制 | M071 | NOT_NEXT_ROUND | Bayes读权限/预测仲裁需要两个合法且互补的预测源。当前不是两个输出银行相加的问题，旧authority信号不直接识别被污染答案，暂不优先。 |
| U054 | Candidate C positive-pair nuisance removal | M077, M078 | RECIPE_HOLD | 原配方保持暂停：分区、双向教学、读权限或nuisance配对需要当前任务的合法信号与公平对照；同答案不等于同题，未获得这些条件不列入下一轮。 |
| U055 | 有界KC↔KC递归与广义调制反馈 | M094 | CONDITIONAL_CANDIDATE | 降为条件备选：来源M094是KC↔KC/DPM广义反馈的结构记录，尚无当前任务功能验证。多观察联合重估不自动属于U055。必须定义反馈电路、状态、学习与终止、预算以及不同于回放/批优化的操作。 |
| U056 | Fly-2 两单元STM→LTM递归教学 | M097 | NOT_NEXT_ROUND | 巩固、多时间尺度或跨流绑定需要当前收益假说；工程核心休息不衰减，当前重点是新组合低样本学习。旧任务保持数字不构成本任务整族反证，增加bank/时间常数本身不提供新约束。 |
| U057 | 增量long-vs-short信息门 | M098 | CONDITIONAL_CANDIDATE | 增量长短预测信息门仍是低优先备选；需要两个真实因果预测及增量收益，不能把残差大小等同信息量，也不能看未来标签分配预算。 |
| U058 | 多horizon residual hierarchy | M099 | CONDITIONAL_CANDIDATE | 多horizon层级仍保留但暂缓；当前反馈无长延迟缺口，不改变任务到自然语言或长时推理以让机制显得必要。 |
| U059 | 因果plasticity quota与跨时间竞争 | M100 | CONDITIONAL_CANDIDATE | 重新作为条件备选：固定因果更新预算内按已观察信息/约束方向分配塑性，而非每条重复经验同等占额。直接对齐单位算力目标，但节约更新不自动减少所需外部样本；禁止未来窗口和隐含真值选择。 |
| U060 | 有界persistent scratchpad与反馈工作状态 | M102 | NOT_NEXT_ROUND | scratchpad为未定义结构备选；当前已有合法短期上下文和有界support历史，新增状态必须说明信息如何变成可泛化约束。不自动算成新的学习机制。 |
| U061 | positive-and-negative reusable metric | M105 | NOT_NEXT_ROUND | 正负metric需可靠关系来源。带噪答案的相等/冲突都不能直接作字节身份标签；旧精确版本失败不排除整族，但当前没有合法配对来源，不优先。 |
| U062 | 冻结或慢残差参考与reference transport或constraint-only slow memory | M106, M061 | ENABLING_COMPONENT | 保留约束/参考兼容构件。当前核心存权重，不是长期保存漂移的裸残差。若U047保存经验，应存原始输入与实际观察答案，重算当前残差；联合估计须另写具体工程配方，不能默认归U055或另算独立候选。 |
| U063 | T2-OPT4E | SCREEN:参数备选 | NOT_NEXT_ROUND | 旧T/S/H/response参数备份继续不排当前轮；换参数缺少当前算术收益理由，原E3无效/技术未资格和科学负面分别保留，不把无效实验当全家族否定。 |
| U064 | T3-B40 | SCREEN:参数备选 | NOT_NEXT_ROUND | 旧T/S/H/response参数备份继续不排当前轮；换参数缺少当前算术收益理由，原E3无效/技术未资格和科学负面分别保留，不把无效实验当全家族否定。 |
| U065 | S2-R8 | SCREEN:参数备选 | NOT_NEXT_ROUND | 旧T/S/H/response参数备份继续不排当前轮；换参数缺少当前算术收益理由，原E3无效/技术未资格和科学负面分别保留，不把无效实验当全家族否定。 |
| U066 | S4-R8 | SCREEN:参数备选 | NOT_NEXT_ROUND | 旧T/S/H/response参数备份继续不排当前轮；换参数缺少当前算术收益理由，原E3无效/技术未资格和科学负面分别保留，不把无效实验当全家族否定。 |
| U067 | H-LAMBDA050 | SCREEN:参数备选 | NOT_NEXT_ROUND | 旧T/S/H/response参数备份继续不排当前轮；换参数缺少当前算术收益理由，原E3无效/技术未资格和科学负面分别保留，不把无效实验当全家族否定。 |
| U068 | RESPONSE-T-E005 | SCREEN:参数备选 | NOT_NEXT_ROUND | 旧T/S/H/response参数备份继续不排当前轮；换参数缺少当前算术收益理由，原E3无效/技术未资格和科学负面分别保留，不把无效实验当全家族否定。 |
| U069 | KC类别的表示与可塑性时窗差异 | L5 | ENABLING_COMPONENT | 表示/采样构件保留，但字节左右角色权重绑定/直接交换不变量若新加入，是额外交换律先验，不能仅凭名称归入KC类别或PN采样。若比较这种先验，必须共同赋予对照或明确是表示/先验试验。 |
| U070 | 结构化PN输入采样与输入密度 | L7 | ENABLING_COMPONENT | 表示/采样构件保留，但字节左右角色权重绑定/直接交换不变量若新加入，是额外交换律先验，不能仅凭名称归入KC类别或PN采样。若比较这种先验，必须共同赋予对照或明确是表示/先验试验。 |
| U071 | DPM cross-stream engram binding | L9 | NOT_NEXT_ROUND | 巩固、多时间尺度或跨流绑定需要当前收益假说；工程核心休息不衰减，当前重点是新组合低样本学习。旧任务保持数字不构成本任务整族反证，增加bank/时间常数本身不提供新约束。 |
