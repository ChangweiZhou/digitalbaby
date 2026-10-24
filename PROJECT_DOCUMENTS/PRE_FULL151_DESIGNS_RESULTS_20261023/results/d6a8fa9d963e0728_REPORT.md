# A11 架构表征方案：独立审计

日期：2026-09-21。审计对象：[A11_architecture_characterization_20260921.md](/Users/pencilbard/Downloads/A11_architecture_characterization_20260921.md)。范围为设计、既有源码与既有结果核对，以及预算/代数检查；**没有运行新科学世界或学习轨迹，没有实施 A11，没有修改父模型或此前报告。** 文档中的执行指令视为待审建议，不是用户授权启动实验。

**结论：值得作为独立 A11 保留，但需要重大修订后再运行。** 它与 FULL/LOW 的“在 B 上比较新增机制”不重复：A11 应回答冻结后的系统在哪些负载、输入关系和连接扰动下失效。然而当前稿有几处会把任务构造、时钟、估计器或校准失配造成的现象，解释成架构缺陷。扩充样本不能修复这些问题。

最值得保留的改进是：冻结一个父模型；暂缓没有 B 坐标迁移规则的神经元数缩放；把有益迁移与有害干扰分开；只做静态图扰动；承认阴性结果有价值；紧凑保存证据。以上不应因本次审计而撤回。

## 独立价值与已有工作的关系

| 面板 | 真正新增的信息 | 已有基础与本次判断 |
|---|---|---|
| A：固定 N 的负载×寿命 | 新父模型、K=16、连续更长经历的响应曲线 | A5 已测 FULL_151 的 K=1…8，且控制了时钟；应继承，而不是从头设计一个更混杂的负载曲线 |
| B：静态拓扑 | 固定学习规则下，对真正连接伙伴改变的敏感性 | 有价值；必须先确定改变哪张矩阵，不能复用名称误导的旧 `rewire` 钩子 |
| C：关系迁移与例外 | 尚未教学的新项目是否受益，以及例外会伤害哪些兄弟项目 | 最有新增信息，但目前输入维度、任务接口与对照都未闭合 |
| D：读写几何 | 在新父模型、关系任务和新图上的几何变化 | 应作为 A/B/C 的诊断层；A3/A4/A5 已有几何分析，不能仅凭三个秩数决定增加哪种神经元 |

我支持“机制比较完成 → 冻结一个 parent → 独立 A11”，但这是一条研究顺序，不是机制比较已完成的证据。如果 FULL 留下多个 Pareto 候选，需提前规定选哪个以及如何处理权衡；不能在 A11 上试过后再选 parent。本次也不替项目把 A10 自动标记为 NOT_RUN。

## 1. P1：Panel C 沿用了另一条模型线的输入规格

设计[第 554 行](/Users/pencilbard/Downloads/A11_architecture_characterization_20260921.md:554)要求“32 个 active PN，8 shared +24 private”，并继承 rendering noise。当前 MiniFly 的 [make_raw_rows](/Users/pencilbard/Downloads/second_try_copy/V88/minifly_rce_v87_followup_runner.py:479)实际是 **88 个输入通道中选 8 个，二值输入**；[evaluate_life](/Users/pencilbard/Downloads/second_try_copy/V88/minifly_rce_v87_followup_runner.py:730)一次编码后重复使用固定 cue。32-active 常数存在于旧 [elm_omnibus10.py](/Users/pencilbard/Downloads/second_try_copy/elm_omnibus10.py:100)，不能称为当前 MiniFly 已有语义。

32/88 的输入密度是当前 8/88 的四倍。即使 KC top-k 后活动数相近，输入到 KC 的竞争、共同支持、B 的校准分布也会改变。这可被研究，但不能偷偷作为“只加入类别结构”的任务。

还有组合约束：若四个项目的 24 个 private 通道互不相交，一个类别需 8+4×24=104 个通道，超过 88。若 private 只是“不属于那 8 个 common”，独立从剩余 80 通道抽 24 个，则两项目预期另有 24²/80=7.2 个重合；总重合约 15.2/32=47.5%，而不是单纯的 25%。这是条件代数示例，不是对实际尚未生成的世界的测量。

**修订：**默认保留 8-active，将 ρ=0.25 明确映射为 2 shared+6 private，并给出整个任务的合法支持布局。一类四项目在互斥 private 下需 26 通道，但不能由此假设三类、所有比较 cue 和所有对照都可任意完全互斥。必须冻结 PN 支持、实际 PN/KC 重合分布、参考 cue、是否有噪声及重新编码方式。结构资格不合格可以停；不能按行为好坏挑选 cue。

当前 [learn_pair](/Users/pencilbard/Downloads/second_try_copy/minifly/V82E/src/model_evo.py:315)还要求两个呈现，label 只能是 0/1，表示哪一个受到惩罚；不是一般的单 cue→任意 y 分类器。因此 A_i 到底是一对 cue，还是与另一个 reference 配对的单 cue，必须定义。需反平衡呈现位置与惩罚角色，避免共同参考 cue 的学习冒充类别迁移。[native probe](/Users/pencilbard/Downloads/second_try_copy/V88/minifly_rce_v87_followup_runner.py:367)同时检查实际 margin 与固定 reader margin；不能只挑其中较有利的一个作为 accuracy。

## 2. P1：Panel C 的对照还不能干净识别“相关结构导致迁移”

设计[第 615 行](/Users/pencilbard/Downloads/A11_architecture_characterization_20260921.md:615)让三个训练成员的 binary outcome balanced/permuted。三个二值标签无法在一个类别内精确等量；如果只是排列 R 中三个相同的 y，则没有改变任何东西。R 的类别一致性、全局结果频次、训练顺序以及 P/U 的前态必须分别说明。

三块放进同一条轨迹也不使它们自动成为配对因果对照。先发生的块改变后发生块的 fast/slow 状态、适应和偏置；三个不同 held-out cue 自身的初始 margin 也不同。预先生成世界能防止结果选样，但不能消除这些混杂。

建议优先采用**同一前态克隆的 R/P/U 三个分支**：固定公共查询项目，匹配外生时钟、教学次数和全局惩罚边际，只改变关系与结果的对应。可以用两个类别联合平衡标签，再跨类别置换，不能在三个相同标签内部做无效 permutation。冻结 cue 对的角色语义后，才锁具体构造。

所有分支先作只读 query baseline，再比较迁移增量，例如

\[
G_{rel}=(M_R^{post}-M_R^{pre})-(M_P^{post}-M_P^{pre}).
\]

U 的泄漏基线也要写出具体对象与时间。共同起点不能替代后续匹配时钟。

例外阶段的 [C_sibling](/Users/pencilbard/Downloads/A11_architecture_characterization_20260921.md:710)目前只是前后差，其中含自然衰减和刺激引起的适应。若要称为“exception teaching 造成的 collateral”，从同一例外前态增加匹配刺激但关闭写入的分支，并证明时间、适应等非塑性动态按声明运行。纯等长休息只排除时间效应，未必排除刺激效应。若只保留前后差，应称“例外阶段的总变化”，不作教学因果归因。

这会改变计算账：48 个独立世界仍是 n=48，但三分支至少是 **144 条分支轨迹**，另有例外的配对后缀。共享前缀可以节约计算和磁盘，不得继续记为 48 次完整单分支执行。若坚持原 48 条自然生活，则固定顺序反平衡、前测和标签边际，并将结论降为该混合课程下的结构关联；不能维持同样强的因果表述。

## 3. P1：Panel A 缺少寿命的操作定义，且可能丢掉 A5 已有的时钟控制

设计[第 249 行](/Users/pencilbard/Downloads/A11_architecture_characterization_20260921.md:249)只给出 K 和 L 的名称。“寿命×16”可以是同样学习后多休息、重复相同课程，或不断加入新事实；三者分别测试自然衰减、重复强化和持续学习，不能互换。

项目已有两种容易混淆的构造：

- A6 的 [t_capacity](/Users/pencilbard/Downloads/second_try_copy/WorkstreamA6_structured_drift_20260920/core_recovery_workstreamA_worlds.py:569)合并不同 cohort；fixed_per_fact 保留每个事实曝光，但增加总教学和时间。
- A5 的[负载构造和资格检查](/Users/pencilbard/Downloads/second_try_copy/WorkstreamA5_capacity_law_20260920/core_recovery_workstreamA5_capacity.py:240)使用共同 FPF8 时钟，把未启用 cohort 的 teach 逐一替为 330 秒 rest，检查逐时段对齐与 elapsed 相等。

仅“精确记录时间不同”不能把时间因素从 K 效应中分离出来。建议在九个格子内继续采用共同最大负载时钟；K 比较固定目标记忆的曝光、时龄和返回时点，其他 cohort 按预定规则启用或等长休息。这里的 K=1 不再等于原始 canonical 的物理时钟，必须改写 L 的基准定义，不能同时声称两者完全相同。

对 L，优先定义为**新事实持续进入的 episode 数**，固定同时相关人口上限及退役/返回规则。禁止把旧 12-bout 课程重复十六遍当作更长持续学习。若选择纯时间延长也合法，但明确它只测衰减。每个格子输出物理时长、独立事实数、有效相关数、每事实曝光、时龄、写入数和重访间隔；时龄不可匹配的指标应明示其复合含义。

“correct→wrong”和“worst cohort-history delta”尚无确定的比较对象。一个 parent 在九种不同世界上不能直接套用上一轮模型间 B-correct/candidate-wrong 定义。应分别冻结同世界、truth 不变、已学且相关人口的时序错误率；若比较 K，则使用嵌套世界的公共目标集合，并报告全部相关人口的总体能力。零分母记 NA，不能记零损害；未学会也不能算“保护成功”。

完整响应曲线有意义，但没有理论参照或预定操作标准时，“过早崩溃”“不是正常有限容量”不能由曲线陡峭本身证明。旧 A5 的目标/全体分解与 A4 的共同漂移、中心化干扰和非线性分解，才是这类判断的已有依据。

## 4. P1：Panel B 尚未定义干预哪张图，旧 `rewire` 不是伙伴重连

PN→KC 的 `m.B`、KC→MBON 的 `m.Q`、teaching 的 `m.T`、反馈的 `m.F/MM` 是不同对象。设计[第 390 行](/Users/pencilbard/Downloads/A11_architecture_characterization_20260921.md:390)没有指定图层，却要求学习规则、支持数和资源都不变。不同层的变动可能改变 reader、可写支持身份、教学路径以及反馈；相同支持“数量”不足以确保相同支持“坐标”。

一个具体实现陷阱：[V82E apply_phenotype](/Users/pencilbard/Downloads/second_try_copy/minifly/V82E/src/model_evo.py:95)里 `g_pnkc_rewire` 分支只置换 CSR 的 `B.data`，**没有修改 `B.indices`**。它改变权重分配，不能实现 G2/G3 的连接伙伴置换。G1、G2、G3 必须通过实际邻接变化而不是参数名字验收。

最容易解释的第一版是限定 PN→KC：锁定 PN/KC 身份、输入类型映射、Q/T/F/MM、reader 与 B baseline。用合法双边交换产生真实邻接变化；冻结禁止的自连/多边条件、分层、权重附着规则与实现算法。入/出度该严格保持就严格保持，不能用“as exactly as permits”接受未定义误差。10%/50% 按最终唯一改变的伙伴比例计算，不按交换尝试次数计算；交换可能相互抵消。

还需核对扰动在模型的**有效输入算子**上不是空操作：当前编码使用 [p[pn_type_index]](/Users/pencilbard/Downloads/second_try_copy/minifly/V82E/src/model_evo.py:124)，某些同输入类型内的交换可能只改变原始 neuron 边而不改变聚合输入。只验收边数不足以证明模型收到真正扰动。

度保持不等于加权输入强度保持。保留权重多重集合也不必保留每个 KC 的总输入或类型组成。若无法在既定预算中加相应控制，就把结论限定为“这个声明的静态扰动对冻结系统的总效应”，而不是纯伙伴身份因果效应。

冻结坐标 baseline、不随图重新校准是合理的一种问题：**冻结系统的拓扑鲁棒性**。但 PN 图变动会改变坐标被访问的分布，B 可能失配；即使两个 anchor 做 difference-in-differences，也不能单独区分 B 与拓扑的机制协同和 B 原校准分布的适用性。若 parent 含新增模块，interaction 是整个 parent 增量，不应仍称为 B 的专属作用。

## 5. P1/P2：Panel B 的世界数量与统计结构没有对应起来

13 图×10 世界×2 anchor=260 的算术正确，但[行为输出](/Users/pencilbard/Downloads/A11_architecture_characterization_20260921.md:483)同时要求 canonical 与 K8。10 个世界是每个 regime 各 10 个，还是合计 10 个？前者是 **520 轨迹**；后者如 5+5，则每 regime 只有 5 个世界，不能按 n=10 报精度。若一个配对 world 包含两段完整生命，也应把这两次计算计入。

建议同一 regime 的同一批外生世界跨所有 13 图及两个 anchor 复用。无需为每张图另生成不同世界来表示“fresh”；fresh 应指相对历史实验的新世界。图变后必须重新编码，**不能跨图复用 KC code cache**，缓存键要含图/编码器哈希。

此时是图×世界的交叉设计：按预定办法重采样独立图 seed 与世界 seed，并保留公共 native 参照和两个 anchor 的依赖。不能既使用公共世界，又把每图的世界当成完全独立嵌套样本来 bootstrap。若采用独立图内世界，则应锁另一套适合该设计的层级统计，接受较低的配对精度。

每类只有 4 张随机图，外层重复不足以给很强的图总体结论。无需因此立刻扩大；完整展示四个图效果、世界不确定性和图间范围，将结论限定到该 perturbation ensemble。更多世界不能代替更多随机图。native 只有一张这一点，原稿已处理正确。

## 6. P1：Panel D 的秩测量可能制造它正在寻找的瓶颈

设计[第 820 行](/Users/pencilbard/Downloads/A11_architecture_characterization_20260921.md:820)给“128 维 Rademacher sketch，3 个 seed”，没有定义估计器。如果把数据投到 128 维后直接计算 participation ratio，则输出秩必然 ≤128。真实 5,177 维各向同性分布也会被压到 ≤128；三个 seed 一致只能说明结果稳定，不能证明估计原空间秩准确。这里不是说所有随机迹估计都有这个上限，而是要求明确不要采用这个有上限的错误替代。

可行修订：优先利用小样本 Gram 的精确恒等式，或冻结在原空间估计 tr(C)、tr(C²) 的流式随机迹方法。后者给出公式、归一化、偏差/误差检查；不能只规定 projection dimension。用已知高秩、低秩、共同均值和退化零能量样例验收，再把原空间估计与投影空间摘要分别命名。

另有三个可比性问题：

1. d_X 用 centered covariance，d_R/d_T 却用 uncentered second moment；w 又归一化，r 未明确归一化。它们可以各自报告，不能直接把高低差称为通道瓶颈。保留幅度、共同均值与中心化残差，统一样本权重并披露方向版。
2. 样本数 n 限制实测 centered rank≤n−1。不同 K、阶段或图的样本数不同，可能产生秩随负载增长的表象。固定采样合约，同时给 n、能量、最大可测秩和零向量比例；重复事件不是独立新方向。
3. 模型有 gamma/alpha fast、slow、adaptation，未来 parent 可能有额外模块状态。`r` 对哪个状态轴求导、`w` 是 slow 还是完整状态增量、哪个 observer margin、哪一时点，都要固定。仅 slow 空间的秩不能代表完整系统。

项目已有 [A4 effective_dimension_report](/Users/pencilbard/Downloads/second_try_copy/WorkstreamA4_effdim_dh_bridge_v3_20260920/core_recovery_workstreamA4_effdim_dh.py:588)，区分 uncentered、raw centered、role/context centered 和 direction-only；[A5 README](/Users/pencilbard/Downloads/second_try_copy/WorkstreamA5_capacity_law_20260920/README.md:30)说明精确较小 Gram 侧的计算。应复用经过定义的统计量，而不是减少成三个不可比的秩数。

旧 [A4 v3 结果](/Users/pencilbard/Downloads/second_try_copy/CORE_RECOVERY_workstreamA4/effdim_dh_bridge_v3/DECISION.md:9)的 exact foreign-write effect 约 −9.90，linear 约 −5.56，nonlinear residual 约 −4.35。它不能直接预测新 B 的结果，却足以提醒：只测局部一阶读写方向，可能遗漏大量实际行为变化。小秩与行为饱和同时出现是候选线索，不是容量受该秩限制的证明。

## 7. P1/P2：动态几何公式回退到了已指出的问题

设计[第 129 行](/Users/pencilbard/Downloads/A11_architecture_characterization_20260921.md:129)把 G 定义为已经含 C_T 的 readout map，接着又用 rᵀG w；除非另加维度定义，否则混合了“状态传播”和“已读出的传播”。建议统一为：

\[
\Phi^{h}_{T\leftarrow t+1}=A_{T-1}\cdots A_{t+1},\qquad
K^{h}_{\nu\mu}=r_\nu(T,h)^\top\Phi^{h}_{T\leftarrow t+1}w_\mu(t,h).
\]

其中 h 是实际经历，w 注入在当前 event 结束后；若选择注入前，则调整乘积索引和执行次序。中途再教学、适应与 clipping 使传播依赖经历，不能只由 Δt 表示。

自记忆项也不应固定写成 I。应为

\[
K^h=D_{self}^h+A_{rel}^h+E_{nuisance}^h,
\]

在提前声明的 self/related/unrelated 掩码上分解。D_self 可随时龄衰减。若归一到对角为 1，则必须同时保存原始对角与归一因子，不能把核心遗忘现象归一化抹掉。语义相关不保证效应有益：例外教学后的错误类别传播也可能落在 related 集合里；关系掩码和按当前 truth 评估的有害方向应分别存。

36 个 replay 必须锁具体 world/checkpoint、方向、幅度、正负扰动、对照和计费方式。小扰动线性响应与真实大小写入的有限变化要分开报告；资格用的幅度检查不按行为优化。两条 ± 扰动分支不是一个计算任务，+7d 解析休息也不是七天真实 CPU。任何共用前缀/多探针省时，都应按实际重放事件与分支记账。

此外，[第 204 行](/Users/pencilbard/Downloads/A11_architecture_characterization_20260921.md:195)禁止四 namespace 共享 world，与[第 752 行](/Users/pencilbard/Downloads/A11_architecture_characterization_20260921.md:748)的 D 消费 A/B/C 轨迹冲突。修成：A/B/C 科学世界互斥，D 有独立诊断随机数，但只读复用已声明的 A/B/C 世界，并记录 lineage。无需为了字面 namespace 再生成一批生命。

## 8. P1：A12 决策图的结论强于 A11 能支持的证据

原稿正确限制了“不宣称语言/推理”，却仍在[第 1013 行起](/Users/pencilbard/Downloads/A11_architecture_characterization_20260921.md:1013)留下几条过强跳跃。

| 原结论 | 问题 | 更稳妥的 A11 输出 |
|---|---|---|
| smooth + weak topology → SCALABLE_AT_FIXED_N | 有限九格、固定 N 不证明可扩展；“weak”未定义且非显著不等于小 | 本次负载/寿命范围内的完整性能与不确定性 |
| topology changes collapse → STATIC_GEOMETRY_LIMITING | 随机重连让系统更差，只说明原结构有用或对扰动敏感 | 先区分 perturbation harm 与可复现 improvement；后者才更直接支持改图 |
| high d_X, lower d_R/d_T → 通道维数实验 | PR 差异不等于任务相关信号被丢失；仍可能是幅度、均值、时间或饱和 | 指明具体 query/write 模态的丢失和行为联系，保留候选而非定论 |
| G_rel≈0 + adequate global rank → RELATIONAL_PROPAGATION_MISSING | PN 共性可能没通过 sparse encoder；训练剂量、reader 与构造也能导致零迁移 | RELATIONAL_TRANSFER_UNRESOLVED；先定位相关信号在哪个接口消失 |
| sibling 前后下降 → RELATION_NUISANCE_SEPARATION_LIMIT | 可能只是时龄/刺激适应，尚未隔离例外写入 | 有匹配分支的额外 sibling collateral，或仅报告阶段总变化 |

尤其不能用全局 rank 高，替代“这个 held-out 类别关系在 KC 中可辨、写入可达、查询可读”的检查。一次固定 ρ 的阴性也不能排除所有共享表示。A11 可以提出一个 A12 假设，不能把未辨识的瓶颈自动命名成 recurrence 缺失。

“strong/weak/approximately zero/large collateral”若要驱动自动分支，需提前写可计算规则和精度要求；不愿指定实际意义界限时就输出连续估计并标记未决，不能用 p>0.05 代替“没有效应”。

## 9. 统计与资源：560 是现稿的计数，不是可靠执行预算

原稿 216+260+48+36=560 算术无误，但只是混合了不等成本生命与 replay 的名义数。

- Panel A 的独立抽样单位是 **24 seeds**，不是 216。bootstrap 一次抽完整 seed 的九格向量；明确 simultaneous bands 是每个 endpoint 的九格覆盖，还是所有 endpoint 的联合覆盖。
- Panel C 的独立单位是 48 world；增加 clone 不增加 n。原生 margin/choice、泄漏、例外曲线与 sibling 指标的正式区间家族需逐项列出，不能只写“固定 primary quantities”。
- 如 Panel B 每 regime 各10世界，名义总数先变为 820；若 C 改三分支，则变为 **916**，另计例外配对后缀。这是账目示例，不是建议自动扩容到该数目。
- 若 L 表示更多新 episode，且 A 的事件数粗略按 KL 增长，九格 ΣKL=271，24 seeds 是 **6,504 个 K=1 事件规模单位**。这只是解释为何轨迹条数不能代替成本，不是时间预测；共享最大时钟会产生另一张工作量账。
- [V5 资源表](/Users/pencilbard/Downloads/second_try_copy/CORE_RECOVERY_BKR/v5/audited_results/RESOURCE_LEDGER.csv:1)里 B 的 canonical/K8 平均约 5.16/62.77 秒，已不是简单八倍。不能把新长寿命 parent 的 560 项当成同成本旧生命。
- 现有 runner 一次 materialize 全部 PN→KC codes；长生命应使用唯一 cue 的位压缩/稀疏表示或有界缓存，科学上同一 cue 的编码保持一致。四进程的 RAM、编码和 probe 峰值应单列；磁盘峰值不是内存峰值。

64 MiB 持久、256 MiB 峰值磁盘可以作为记录设计约束，目前不能保证已满足。嵌套基础 cue/event 一次保存，各格只存 schedule/seed/索引差异；每图只存变换或稀疏变动；状态只保留不可重建且实际需要的 checkpoint；指标流式汇总。重放所需的 RNG 算法、版本、cue 排除清单或可重建出处必须锁住，只有 seed 不一定足够。临时文件、原子提交副本及新缓存要计入峰值。

交付包原稿写 <15 MiB；仍应遵循周某的 **<10 MB** 目标，除非证明更大科学材料不可替代。没有理由把全部原始轨迹和输入复制进结果包。

## 建议的修订顺序与放行条件

这不是要求缩减正式样本，周某偏好的大样本路线可以保留。先修设计与资格，再测算完整计划，不用行为结果决定削减或扩充。

1. **锁 parent 与可支持的结论。** 完成前序处置，固定选择规则；改写 A12 决策图，不把描述性关联当成因果瓶颈认证。
2. **锁 A/C 世界构造。** A 继承已有等时段设计，明确 L；C 回到当前 cue-pair 接口，完成合法支持布局、标签边际与共同前态分支。二者只作结构资格，不先挑表现好的世界。
3. **锁 B 干预与样本账。** 明确图层、真正伙伴重连、权重/度/有效编码验收、跨图 cache 和 regime 数量。
4. **把 D 作为嵌入式测量层。** 统一状态轴、传播公式、centered/uncentered 定义，验收原空间秩估计，冻结有限扰动与 ≤36 replay 的实际账目。
5. **最后冻结完整 n/事件/分支/存储预算。** 记录 pair/world/graph 的独立单位、区间家族、缺失与技术失败处理；资格失败不替换成另一个更容易的科学世界，科学未完成不能只汇报完成子集。

我的优先级是 **A 的正确时钟/寿命定义 + C 的干净迁移任务**；B 为后续具体改结构提供证据，D 帮助解释而非自动裁决。四个面板可以保留在同一独立 A11 中，无需发明更多机制。当前最重要的不是把 560 改成更大的数，而是保证每一条轨迹能回答它声称的问题。

## 审计产物与范围

本目录只包含报告、可重跑的只读算术/源码检查及 manifest；无数据集、父模型、旧归档、环境、缓存或原始逐 seed 输出副本。`AUDIT_CHECKS.json` 的通过表示报告引用的源码事实和算术已核对，**不表示 A11 设计已经通过，也不是新科学结果**。可按本目录 `audit_checks.py` 在项目现有 Python 中重算；manifest 引用未改变的来源与哈希。
