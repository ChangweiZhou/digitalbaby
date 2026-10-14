# MiniFly当前机制优先级：U055归属纠正后

> 版本级补充：上一轮方向排序尚未覆盖具体版本，尤其漏述了已测REPLAY-P和TOPO-G。请以[版本复核](VERSION_REGISTER_20261011/REVIEW.md)和机器可读版本索引补充判断，不能将该方向排序当作最终配方选择。


2026-10-11。当前开发默认COV-SUPPORT，比较锚点COV-R；目标固定为更少带噪/缺失外部样本与计费计算下的未见加法自身输出泛化，旧知识保持受保护。用户采用与科学确认分开。未运行或授权新实验。

[本次完整复核与71条理由](POST_U055_RESCREEN_20261011.md)、[原始交叉引用](POST_U055_RESCREEN_20261011.json)、[此前快照](history/pre_U055_attribution_correction_20261011/NEXT_ROUND_PRIORITY.md)。

U055原来源是KC递归/DPM/广义反馈，不能直接对应通用多观察联合重估，已撤销默认第一的推荐。优先设计U047选择性再激活和U034有支持的不确定性重开；U059因果预算分配为第二顺位。均尚未形成锁定可运行配方。

| 分类 | 数量 |
|---|---:|
| ADOPTED_BASELINE | 3 |
| CONDITIONAL_CANDIDATE | 6 |
| ENABLING_COMPONENT | 8 |
| NOT_NEXT_ROUND | 46 |
| PRIORITY_DESIGN | 2 |
| RECIPE_HOLD | 4 |
| REFERENCE_ONLY | 1 |
| STOP_CURRENT_RECIPE | 1 |

71是历史条件集合，不是全部机制。8个考虑方向含2项优先设计、6项条件备选；8个构件不算独立候选。所有历史结果和原来源ID保留，旧exact-context E1统计不作为本任务整族排除依据。

| ID | 名称 | 分类 | 当前实例状态 |
|---|---|---|---|
| U001 | FE2 / Temporal contrast | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U002 | LAT4 / Latency classes | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U003 | P4 / ITDP-like | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U004 | E3dev / Local afferent development plus homeostasis | ENABLING_COMPONENT | NO_NEW_QUALIFIED_INSTANCE |
| U005 | — / Slow bounded prune/grow `S` | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U006 | — / Topology × commitment K0–K3 | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U007 | — / Structural eligibility | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U008 | Circuit-generated suppressive/decorrelating `A` | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U009 | Recurrent inhibition matched for mean activity | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U010 | Homeostatic rebound via KC microcircuits (#23) | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U011 | A_ELIG / Multi-frame KC eligibility trace (OMNIBUS10) | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U012 | — / `ELIG_TRACE_PORT` | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U013 | — / STC tag + limited capture resource | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U014 | `pair_importance` | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U015 | `neuron_availability` | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U016 | `E_META` | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U017 | Receiver availability / saturation states (V69–V71) | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U018 | Consequence-specific DAN-like teaching | ADOPTED_BASELINE | ADOPTED |
| U019 | Delayed eligibility-gated teaching | ENABLING_COMPONENT | NO_NEW_QUALIFIED_INSTANCE |
| U020 | H / Heterogeneous compartments | RECIPE_HOLD | HELD_NOT_REHABILITATED |
| U021 | — / Stable/flexible compartment split | CONDITIONAL_CANDIDATE | PARK_NO_NEW_RETENTION_DEFICIT |
| U022 | V23 `active_forgetting` | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U023 | `importance_surprise_release` | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U024 | OMNIBUS `C_FORGET` (targeted wrong-winner decay) | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U025 | Selective downstream `R_sel` | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U026 | Homeostatic forgetting / targeted bidirectional extinction | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U027 | Single-DAN acquisition/erasure duality | RECIPE_HOLD | HELD_NOT_REHABILITATED |
| U028 | DPM-mediated consolidation | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U029 | Spaced-training disinhibition / pERK-like gating | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U030 | Immediate→intermediate→slow commitment | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U031 | Recurrent post-event consolidation window | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U032 | Stubborn/adaptable heterogeneous populations | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U033 | bayes_stable | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U034 | bayes_reopening | PRIORITY_DESIGN | SUPPORTED_DIRECTIONAL_UNCERTAINTY_REOPENING |
| U035 | context_fixed 四bank固定路由 | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U036 | context_inferred 四bank推断路由 | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U037 | silent_fixed 八bank固定路由 | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U038 | silent_inferred 八bank推断路由 | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U039 | CORR2 证据支持读权限 | RECIPE_HOLD | HELD_NOT_REHABILITATED |
| U040 | 固定 partial mixture 包括 Q_HALF | REFERENCE_ONLY | NO_NEW_QUALIFIED_INSTANCE |
| U041 | 细胞内部多时间尺度私有状态 | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U042 | 相互独立的写入门与遗忘门 | ADOPTED_BASELINE | USER_ADOPTED_SUPPORT_WRITE_GATE_DEV_EVIDENCE |
| U043 | 局部边权学习与真实换边协同 P×S | CONDITIONAL_CANDIDATE | HOLD_OLD_TOPO_RECIPES_REQUIRE_NEW_OPERATION |
| U044 | 稳定地址与可塑内容分离或输入坐标兼容 | ENABLING_COMPONENT | NO_NEW_QUALIFIED_INSTANCE |
| U045 | 有界Hebbian key/value或predictive residual fast matrix | ADOPTED_BASELINE | ADOPTED_ENGINEERED_COV_R_BOUNDED_E3 |
| U046 | 有界分布式地址层与自适应原型 | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U047 | 有限选择性再激活或bounded offline O | PRIORITY_DESIGN | BOUNDED_INFORMATION_DIVERSE_REACTIVATION |
| U048 | 真正连续活动的稳定Oja与兼容记忆的输入适应 | ENABLING_COMPONENT | NO_NEW_QUALIFIED_INSTANCE |
| U049 | 历史结构化感觉输入的fast-minus-slow时间对比 | ENABLING_COMPONENT | NO_NEW_QUALIFIED_INSTANCE |
| U050 | SIM-MIX 带exact-member权限的相似读出 | STOP_CURRENT_RECIPE | CLOSED_NEGATIVE_SCREEN |
| U051 | Predictive Bayes arbitration 及slot evidence | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U052 | 有界utility replacement或TRACE-U | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U053 | 局部Bayesian读权限控制 | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U054 | Candidate C positive-pair nuisance removal | RECIPE_HOLD | HELD_NOT_REHABILITATED |
| U055 | 有界KC↔KC递归与广义调制反馈 | CONDITIONAL_CANDIDATE | HOLD_REQUIRE_EXPLICIT_RECURRENT_CIRCUIT |
| U056 | Fly-2 两单元STM→LTM递归教学 | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U057 | 增量long-vs-short信息门 | CONDITIONAL_CANDIDATE | PARK_NO_TWO_PREDICTOR_NEED |
| U058 | 多horizon residual hierarchy | CONDITIONAL_CANDIDATE | PARK_NO_LONGER_HORIZON_NEED |
| U059 | 因果plasticity quota与跨时间竞争 | CONDITIONAL_CANDIDATE | CAUSAL_INFORMATION_BUDGET_ALLOCATION |
| U060 | 有界persistent scratchpad与反馈工作状态 | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U061 | positive-and-negative reusable metric | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U062 | 冻结或慢残差参考与reference transport或constraint-only slow memory | ENABLING_COMPONENT | NO_NEW_QUALIFIED_INSTANCE |
| U063 | T2-OPT4E | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U064 | T3-B40 | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U065 | S2-R8 | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U066 | S4-R8 | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U067 | H-LAMBDA050 | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U068 | RESPONSE-T-E005 | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U069 | KC类别的表示与可塑性时窗差异 | ENABLING_COMPONENT | NO_NEW_QUALIFIED_INSTANCE |
| U070 | 结构化PN输入采样与输入密度 | ENABLING_COMPONENT | NO_NEW_QUALIFIED_INSTANCE |
| U071 | DPM cross-stream engram binding | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
