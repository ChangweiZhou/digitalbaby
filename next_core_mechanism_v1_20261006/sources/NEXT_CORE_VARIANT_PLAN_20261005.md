# MiniFly 下一版核心的候选与备选名册

日期：2026 年 10 月 4 日。状态：研究建议，尚未冻结为运行协议。本次没有运行新的 learner trajectories。文中的参数配置都是提议，不能当作已经验证的结果。

**2026 年 10 月 5 日更新：** GitHub 已发布新的 64 世界 ERROR 对照。它未修复四选一 Latin 表格补全，且该任务上的已教旧、新题均值比 R_center 低约 3 pp；这限制 ERROR 的用途，但没有推翻其此前 lifetime/修订收益。这里的“ERROR 强参照”仅指已有任务内参照，不是跨任务全面更优的核心。下一轮四候选名册保留，Q-HALF 的验证承诺限于现有二元关系与 lifetime；新表格任务不新增为验收要求。本次没有新测试或筛掉 P/S 救援候选，也不修改 108 总账的历史关闭计数。详见 [新结果审阅与名册更新](GITHUB_RESULT_REVIEW_20261005.md)。

下一轮建议比较四个候选配置，分别来自读出权重、输入边权可塑性和使用率驱动换边三个机制；保留 V1 正式参照和 ERROR 强参照。与此同时关闭无依据的原样重跑，把其他可能救活的配置限制在明确的备选名册内。历史的六家族、首轮十六变体和一百零八条记录使用不同计数单位，不能互相相减。

## 阶段目标

目标仍是字节读取与记忆在同一核心中工作：给定同样的新信息、教学次数和时间，学得更多、延迟与干扰后保留更多、修订错误知识，并报告有限的未教输入复用。Full151 内部结构可以调整。当前不增加自然语言、多位算术、无提示自主输出或大规模神经元扩展要求。

沿用现有 lifetime 和 reuse 两项任务，比较同一个候选成品的表现。前者是已教关联的获取、保持和修订 E1；后者有 never-reinforced probes 与 N_old 因果分支，支持的仍是该简单关系族内的 E3。任意标签回忆可以被查表解释，不能用其成功宣传 E3。V2 的正式 RETAIN_V1 结论保留；本建议不改变旧的门槛或报告。

## 本轮新增的探索性证据

从 V2 的 V1 与 ERROR 两组各 96 世界的已保存 predictions 评分，读取 384 个已完成 receipt 文件。没有调用 learner、改动原收据或生成新输入。公式为

`u = shared / 1.4911274663291492 + alpha * private / 1.3452365735750882`。

固定比较 alpha=1、alpha=.5、shared-only 与 private-only。标签仅供离线计分，不参与组合规则。lifetime 沿用首最大值输出；reuse 沿用原 ChoiceOrgan 的两 option 顺序与二元 utility 差。

| ERROR 核心的读出 | 完整私有权重 1 | 私有权重 .5 |
| --- | ---: | ---: |
| 最终旧知识正确率 | 97.05% | 97.05% |
| 最终新知识正确率 | 99.54% | 99.54% |
| 修订后一天正确率 | 96.35% | 96.09% |
| 最终未教关系 W 正确率 | 55.90% | 61.46% |
| 最终未教关系 N_old 正确率 | 46.70% | 44.62% |
| 最终未教关系 W−N_old | +9.20 pp | +16.84 pp |

W−N_old 的增幅为 +7.64 pp，其中 W 上升 +5.56 pp，N_old 下降 −2.08 pp。描述性的世界级配对正常近似区间为 [+4.78,+10.50] pp；没有多重比较确认含义。旧题和新题的观察分数相等，也不能证明未来绝对无损。

shared-only 的未教关系效应为 +31.42 pp，但其任意标签 lifetime 旧、新、修订正确率仅约 26–27%。因此不能删除 private bank 来宣布核心升级。固定 .5 是值得确认的折中，不是最优参数的证明。

**成立条件：ERROR 的教学概率 pi 继续只由 private pre-outcome values 计算。Q 只改最终读出。** 若将 pi 改为 combined output 的 softmax，这会改变学习状态，本次离线反事实评分就不再对应新代码。

证据：[离线结果](READOUT_SCREEN.json)、[评分脚本](readout_screen.py)、[原 V2 汇总](../DIGITALBABY_CLAUDE_B_D4E4578_20260929/persistent_core_v2_trial/results/SUMMARY.json)。评分经独立只读审阅；这是新探索计算的公式审查，并未重新审计原科学运行。

Sparkie 的 global-half-private 在 Latin/XOR 类任务上损害记忆且未救回 held-out 表现。它与本结果来自不同任务、核心历史和教学政策，保留两项结果，不选择性删除负面证据。历史 V37 的 partial authority 又不同：exact bound member 权重仍完整，只给 unbound similar slot 半权重。三者不可视为一个 exact variant。

## 下一轮建议名册

建议一次冻结下面的完整批次，不先发起一串额外的定位实验。当前只是选项研究，尚未授权运行这一批次。

| 角色或候选 | 具体改动 | 要回答的问题 |
| --- | --- | --- |
| V1 正式参照 | 原 R_center V1 | 保留与现任版本的比较 |
| ERROR 强参照 | 已测 private pi−y 教学，其余沿用 V2 | 避免把 ERROR 本身的巨大修订收益归给新机制 |
| ERROR + Q-HALF | 仅最终 private 读出权重 1→.5；pi 保持原定义 | 已有 shared 复用信号能否更好地表现，且不损伤保持和修订 |
| ERROR + P1-E005 | shared FE0 路径的 PN→KC Hebb 增益 .05→.005；固定支持和每 KC 入射预算 | 原 P1 的伤害是否源于过大的地址变化剂量 |
| ERROR + P1-E0005 | 同一规则，增益 .0005 | 在同一预先封顶的两档剂量内辨别有用适应与仅趋近静态 |
| ERROR + S3-RHO035355-CUECLOCK | shared FE0 路径使用率换边；目标为 `.05*2**(-.5)`；每 24 个实际 cue 记录触发，不以监督写入开关触发 | 目标与实际 coding fraction 匹配后，拓扑适应是否有行为收益 |

这是 **四个候选配置、三个新比较机制、两个参照**；控制臂和 W/N 分支另计。P 和 S 仅作用于 shared FE0 路径，private CONTENT 地址与其 ERROR 教学不变。P/S 的 Q 权重维持 1，不自动捆绑 Q-HALF，否则必须另增对应 parent 才能识别各自贡献。这些是把旧规则移植到当前核心的候选组合，不是原 Package A/B 运行的恢复。S3 的 source 按 write-enabled teach calls 计事件；本候选改成 cue clock，是明确的移植改动，不声称只调 rho。下面备选表里的 S3-RHO035355 才表示原 substrate 上只改 rho 的参数救援。

N_old 禁止的是旧反馈监督写入，不能同时悄悄停掉所有无标签输入适应。P1 和本 S3 cue-clock port 在 W/N_old 中都接收匹配的 cue-only 适应机会；实际 codes 可以因各自状态不同而不同。答案 byte 排除在适应样本之外，cue code 在反馈前取得；probe 使用 disposable clone，适应不提交到 continuing history。新审计分别记录监督权重写入、无标签边权更新和结构事件，不能用一个 learn=false 字段代替这些实际操作的检查。

S3 应有随机位置控制，预算匹配规则在新协议中明确；若使用 realized-count yoking，应标为诊断控制，不能作为自主核心候选。P 的静态零更新参照由相同 ERROR parent 提供。不得依据所测 cue 的 taught/held-out 身份选择路径或权重。

比较固定教学预算下的旧知识、新知识、修订、未教关系 W 和 N_old、以及 CPU/内存。机制增量必须比较 ERROR；候选成品同时比较 V1。采用规则需要在新协议中预先写清，保持与复用不能被一个混合均分掩盖。旧 V2 的效率门槛不追溯修改；新协议也不能把 E1 成功称为更大的 E3 能力。

先完成实现一致性、禁写真的零写入、独立 birth、clone/checkpoint、冻结 source 和资源预检，然后同一批次直接完成既有任务。预检不成为新的科学问题。n 由精度与预算确定，不在此研究备忘录中伪装成已锁定；已有 V2 在 96 世界、两任务下每 arm 约 10.6 CPU 小时，新批次还包含控制与 P/S 额外成本，不能声称是几分钟实验。

## 原实现的关闭依据

这里的关闭是停止给同一规则同一参数继续分配预算，不是数学上证明整个机制家族永远无用。

| 已测机制 | 当前证据 | 处理 |
| --- | --- | --- |
| P1 Hebb，epsilon=.05 | 有效 E1 −4.20 pp；有效 E3 −29.69 pp | 原样关闭；仅保留两档低剂量 |
| P2 Oja，epsilon=.05 | 有效 E1 −4.98 pp；有效 E3 −28.13 pp | 原样关闭；不重复整套低剂量 P2 |
| P3 novelty Oja | 当前没有可恢复的合格 technical receipt；原 Muse failure record 仍缺失 | NOT_QUALIFIED；不计科学失败，暂不启动 |
| P4 记录级时差，tau=330s | E1 −1.07 pp；E3 −.52 pp，区间很宽 | 未证明改善；留一档时常数备选 |
| T1/T2/T3 静态 PN→KC 图 | E1 未改善；T2 点估计伤害；E3 禁写分支错误 | 原样不重跑；不能把无效 E3 当负结论 |
| S1 弱边随机换边 | E1 0 pp；只有 53 个 KC 被反复操作；缺少输入权重学习 | 当前独立 S1 关闭；需要 P×S 才是新结构 |
| S2 共活动换边 | E1 −2.15 pp，区间跨零；E3 无效 | 原样关闭；低换边剂量低优先级 |
| S3 使用率换边 | E1 +1.07 pp，区间跨零；原 rho=.05 与实际 .0353553 不同 | 原样关闭；正确目标值得一次检验 |
| S4 暂试与回退 | E1 −2.64 pp，区间跨零；大量暂试与回退 | 原样关闭；低暂试预算低优先级 |
| response T 字节级时差，epsilon=.05/tau=10s | old_end 已伤害习得；final −8.40 pp [−11.51,−5.28] | 原样关闭；仅留一档低剂量 |
| H 输出 EMA 全额相减 | final +2.93 pp [−2.28,+8.14]；70 修复、55 破坏 | 未确认有效；半额相减备选 |
| J 88×88 交互库 | 对添加/乱序控制无可靠改善；形成阶段 interaction 很弱 | 原实现关闭；无独立纯 gain/寿命备选 |
| centered-J | 表示几何改善，但形成指标仍远低于冻结门槛 | 此设计关闭；属于预检失败，不额外计算一次完整科学实验 |
| B 历史活动校准代理 | development 36.33% 对 raw 39.45%；38 修复、54 破坏 | 原实现关闭；仅改 tau 缺乏救活依据 |
| REPLACE/BOTH | 修订高，但损害旧记忆；ERROR 已解决大部分修订缺陷 | 当前删除强度/触发规则不作默认升级；此轮不追加救援配置 |

response T 是逐字节输入边权适应，不是 T1–T3 图结构。H 不是 FE1 受体习惯化。J 是新增四个交互 bank，不是 R3。J 比对 FE0 也非等容量，不能把添加可写参数的效果混作相同资源下的新机制效果。

原十六项中，十五项已有 E1 科学结果，P3 仍未资格化。其中八项有有效 E3，T1/T2/T3/S1–S4 七项的 E3 无法据原错误分支判断。有效阴性、不确定、无效和技术未资格化分别保留。

证据：[A V3](../DIGITALBABY_CLAUDE_B_D4E4578_20260929/minifly_a_v3/results/REPORT.md)、[B V2](../DIGITALBABY_CLAUDE_B_D4E4578_20260929/minifly_b/REPORT.md)、[T2 初轮](../MINIFLY_THREE_MECHANISM_ROUND_20260928/results/REPORT.md)、[response](../DIGITALBABY_OTHER_RESULTS_20261002/response-mechanisms-20261001/response_mechanisms/results/final/REPORT.md)、[spectrum](../DIGITALBABY_OTHER_RESULTS_20261002/learning-spectrum-20261002/learning_spectrum/REPORT.md)。

## 十个旧机制参数救援配置

这十个是旧失败或未确认机制的参数救援配置数，另有 Q-HALF 这一个读出参数配置；合计十一项明确的参数配置，不是十一个机制家族。救援配置中的三项加上 Q-HALF 进入下一轮建议，其余七项为条件备选。T2/T3 二选一代表较弱图分离假说，不自动把两个都运行，因此这十项救援最多启用九项。

| 配置 ID | 原规则的唯一参数变化 | 保留理由与上限 |
| --- | --- | --- |
| P1-E005 | epsilon .05→.005 | 高频 KC 被反复更新；两档封顶 |
| P1-E0005 | epsilon .05→.0005 | 同上；若只趋近静态、无净收益则结束 |
| S3-RHO035355 | rho .05→`.05*2**(-.5)` | 与实际 coding fraction 匹配；一档 |
| H-LAMBDA050 | readout raw−h→raw−.5h；beta=1/16 不变 | 全校正同时修复与破坏；一档。移植到新核心时作用点另锁定 |
| RESPONSE-T-E005 | epsilon .05→.005；tau=10s 不变 | 原在习得阶段已受损；只检验降低剂量，一档 |
| P4-TAU165 | tau 330→165s；epsilon=.05 不变 | 时差积分两个记录间隔→一个，一档 |
| T3-B40 | bridge budget .2→.4 | 只有分簇过强损伤共享信号的依据时启用；与下一项择一 |
| T2-OPT4E | 图优化 proposal budget 16E→4E | 较弱扩张目标的有限检查；与上一项择一 |
| S2-R8 | 每事件最大换边 64→8 | 低优先级剂量救援；只减少伤害不算改善 |
| S4-R8 | 每事件最大暂试 64→8；trial=2 不变 | 降低暂试扰动与回退负担；只降低成本而无有效能力需如实报告 |

P1 的参数救援有具体但未证明的机制依据：旧设定根据平均约 21 次激活/KC 选择 epsilon，但部分 KC 激活可达 600 次。只读 PD 显示 P1/P2 中单一输入占 KC 入射预算超过 90% 的 KC 数约 50，对照约 8；这提示共同特征垄断或地址漂移，不能直接断言因果。

P2 不列另一套剂量扫描，是因为其 binary KC 的 Oja 项在固定每 KC 权重和归一化中消失：未触及 floor 时等价于 P1 的 epsilon/(1−epsilon)。真实 Oja 稳定作用需要更改连续活动或预算方式，应另设结构 ID。

H 备选不会自动成为下一轮必跑项：在两 option 的 utility 比较中，如果相同四维 baseline 被等量扣除，其常量部分会相消。该判断要求 h 在两个 option 间固定；若先更新 h 再读第二项，会增加顺序依赖，已经不是同一个固定 baseline 对比。H 移植需预先锁定 raw 路径和更新时点。它首先是四输出偏置校准假说，不应宣传成天然的 E3 修复器。

J 不保留单纯扩大 gain、放宽 bound 或延长寿命：原始形成阶段 interaction energy 极低，没有 clipping，寿命更长不能修复未形成的选择性。若改变表示、更新式或归一化，应明确算新架构。

## 八个仍开放的操作方向

六家族用来分类。下面八条是尚未解决的具体操作方向，成熟度不同，不能称作八个已可运行变体；也不能再声称只剩两个机制。

| 方向 | 当前状态 | 下次启动条件 |
| --- | --- | --- |
| Q 固定 partial mixture | 已有本次探索证据；Q-HALF 进入下一轮 | Fresh world 确认，writer pi 不变 |
| Q CORR2 earned authority | 当前系统没有 slot/member 与独立 context support 语义 | 明确定义模型自身的有界支持状态；不得用 evaluator taught 标签或重复教两次冒充不同 context |
| Q SIM-MIX | 当前移植未实例化，历史版本未通过资格 | 明确真实 similarity、slot 与完整 exact-member authority；低优先级 |
| E 多时刻资格 recipient | 源规则有正信号，也有保持/修订代价；byte port 未锁定 | 当前已缓存反馈前 cue，没有缺失延迟 credit；新 trace 必须提供不同的合法 recipient 信息，不只是同 x 乘 gain |
| I positive-pair nuisance removal | source 有实际操作；当前 substrate 映射未定 | 明确 96-d byte context→当前路径的移植、因果 pair 提议和 corroboration；4 答案环境中 same-outcome 不等于 same fact |
| M neuron availability | 源定义明确，有保持收益和修订成本 | 真实 capacity/干扰问题需要保护时启用；不默认锁死现已可修订的状态 |
| M recovering slow local resource | 当前只是提议 | 给出有界因果状态、恢复方程及与 incumbent allocation 的区别 |
| F C_FORGET-type local decay | source 有定义；当前符号与坐标映射待定 | ERROR 修订仍不足且错误支持可定位时启用；不能只衰减某 output 就声称 source-faithful |

FULL151 adaptive allocation、R_center 已有 shared/private 分工和已测 REPLACE 不是这八条未测方向。Candidate C、PAIR-OJA 和 PAIR-RAND 包装不自动计成三种独立操作。过去资格迹的多 frame 是同一 cue 的扰动视图，当前逐字节前缀不是同一种数据；正结果不能不经语义检查直接移植。

来源：[V37 authority](../elm_v37_retrieval_authority.py)、[V37 结果](../v37_results/SUMMARY.md)、[A_ELIG 与 C_FORGET](../elm_omnibus10.py)、[资格迹复审](../CORE_RECOVERY_REANALYSIS_AND_OPEN_EXPERIMENTS.md)、[Candidate C](../candidate_C_reusable_invariance.py)、[V45](../V45_Mac_M3/v45_results/SUMMARY.md)、[availability](../elm_fly_mechanism_tournament_v23.py)。

## 拓扑与内部结构的后续边界

T1–T3/S1–S4 主要改变 PN→KC 的静态邻接或换边。它们没有穷尽 Full151 内部拓扑；也没有实现局部计算，只是改图后仍沿用全局投影与 top-k。因此拓扑不能整体筛掉。

保留五条尚未配置的结构备选，暂不增加到下一轮候选数：

1. 真正保留稳定项的连续活动 Oja，区别于归一化后退化成 Hebb 的 P2。
2. 联合边权学习与弱边更新 P×S，区别于没有权重学习的独立 S1。
3. 表示变化时的旧记忆运输或已承诺地址冻结，解决改输入坐标而旧记忆仍留原坐标的问题。
4. 局部竞争或分区路由，区别于只改变连边但仍全局 winner selection 的 T 图。
5. 有界的 KC↔KC 或记忆反馈结构，区别于原 PN→KC 连接优化；用既有任务检验，不新增智能任务。

这些是结构设计方向，没有足够明确的方程和预算前不计为 runnable variant。此轮不把图提取、自然流、增长神经元、scratchpad 或新任务包装成额外机制。

## 分配后续实验预算

下一轮直接比较所列四候选。确认 Q-HALF 只说明固定组合在所测任务有效；P/S 若无净收益且仅靠减少改动回到静态，则关闭预先列出的参数救援，不继续无限趋零扫参。旧知识保护、修订和复用有冲突时报告权衡，不掩盖为一个总均分。

只有已确认的不同改动才进入一次联合候选比较，组合不是前面单独得分的相加。若这批仅 Q 获益，则先交付 ERROR+Q 的实验版本，并如实限定 E1/E3 范围；H、E、I 等是否继续由明确剩余短板决定。没有收益时保留当前版本，不靠再发明更难验收延长计划。

十个旧机制救援配置、Q-HALF、八个操作方向和五个结构提议不能相加作为同等级机制总数；Q-HALF 本来就属于八方向中的 Q 固定 partial mixture，部分救援又属于结构方向的较弱实现。它们是本轮研究整理后的预算名册，不是从一百零八条记录推导出的全部可能性总数。一百零八条是历史机制、组合、主张及提议记录；此前可关闭的二十条 exact recipe/claim 不代表二十个整族被证明无用。

原始总账：[108 条来源](../FULL_MECHANISM_LEDGER_SOURCE_20261002/MECHANISM_LEDGER.json)、[完整复审](../FULL_MECHANISM_LEDGER_SOURCE_20261002/FULL_PROJECT_REVIEW_AND_LEDGER.md)、[首轮 16 项](../MINIFLY_FIRST_ROUND_VARIANT_INVENTORY_20260928.md)。
