# MiniFly mechanism priorities after U042 adoption

2026-10-11。当前开发核心采用COV-SUPPORT，由用户明确决定；COV-R保留为比较锚点。**这是开发采用，不是通过独立确认的低样本升级。** 原实验NO_ADVANCE_RETAIN_COV_R结论不变。

[采用记录](../../addition_core_v2_support_v1_20261011/ADOPTION.json)、[统计重新分析](../../addition_observation_support_dev_v1_20261011/posthoc/reanalysis_20261011/REANALYSIS.json)、[当前核心入口](../CURRENT_ADDITION_CORE.json)、[此前筛选快照](history/pre_U042_user_adoption_20261011/NEXT_ROUND_PRIORITY.md)。

71条去重记录仍在。3个采用条目分属不同证据范围：U018原生观察E0/E1，U045工程COV-R有限E3，U042用户采用的支持门。另有6个条件方向、8个构件及54个非当前候选/暂缓条目。没有新增整族证伪，也没有新实验运行授权。

| 分类 | 数量 |
|---|---:|
| ADOPTED_BASELINE | 3 |
| CONDITIONAL_CANDIDATE | 6 |
| ENABLING_COMPONENT | 8 |
| NOT_NEXT_ROUND | 48 |
| RECIPE_HOLD | 4 |
| REFERENCE_ONLY | 1 |
| STOP_CURRENT_RECIPE | 1 |

下一步仍研究未见加法的样本效率。U055多经验协调与U043实际连接调整为有条件方向；U047旧均匀重放不重启，U021/U057/U058的启动前提未满足。新增循环、状态或重复更新本身不是效能提升。

| ID | 名称 | 当前分类 | 当前实例 |
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
| U034 | bayes_reopening | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
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
| U047 | 有限选择性再激活或bounded offline O | CONDITIONAL_CANDIDATE | STOP_UNIFORM_REPLAY_FROZEN_P_AND_CURRENT_EXTRA_RECIPES |
| U048 | 真正连续活动的稳定Oja与兼容记忆的输入适应 | ENABLING_COMPONENT | NO_NEW_QUALIFIED_INSTANCE |
| U049 | 历史结构化感觉输入的fast-minus-slow时间对比 | ENABLING_COMPONENT | NO_NEW_QUALIFIED_INSTANCE |
| U050 | SIM-MIX 带exact-member权限的相似读出 | STOP_CURRENT_RECIPE | CLOSED_NEGATIVE_SCREEN |
| U051 | Predictive Bayes arbitration 及slot evidence | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U052 | 有界utility replacement或TRACE-U | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U053 | 局部Bayesian读权限控制 | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U054 | Candidate C positive-pair nuisance removal | RECIPE_HOLD | HELD_NOT_REHABILITATED |
| U055 | 有界KC↔KC递归与广义调制反馈 | CONDITIONAL_CANDIDATE | CONDITIONAL_NEW_BOUNDED_CORRECTION_CIRCUIT |
| U056 | Fly-2 两单元STM→LTM递归教学 | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
| U057 | 增量long-vs-short信息门 | CONDITIONAL_CANDIDATE | PARK_NO_TWO_PREDICTOR_NEED |
| U058 | 多horizon residual hierarchy | CONDITIONAL_CANDIDATE | PARK_NO_LONGER_HORIZON_NEED |
| U059 | 因果plasticity quota与跨时间竞争 | NOT_NEXT_ROUND | NO_NEW_QUALIFIED_INSTANCE |
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
