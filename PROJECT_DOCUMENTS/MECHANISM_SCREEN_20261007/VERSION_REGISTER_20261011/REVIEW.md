# 机制版本复核而非只筛家族

此前两次筛选主要在U编号方向层面，压缩了具体版本和组合，遗漏了已做的残差优先回放和当前权重加换边实现。本复核纠正这一粒度错误。当前默认仍为用户采用的COV-SUPPORT，COV-R是比较锚点；不新增定位实验、不启动新科学运行。

目标仍是更少带噪/缺失答案样本与计费计算下的未见加法泛化，保留旧知识。历史任意标签联想、输入编码、下一符号预测和当前加法分开；旧正负结果不是直接同榜。

## 找到的实际版本清单

| 来源 | 恢复粒度 | 数量与边界 |
|---|---|---|
| V23 ARM_MANIFEST及完整结果 | 52个明确命名的版本与对照 | 六种子筛选，状态成本不等；不是52个通过确认的家族 |
| V24 JOB_MANIFEST | 101个memory名称、具体参数、所在panel和对照 | 3498个job；38个名称与V23重合、63个其他名称。名称相同不自动说明实现相同 |
| V25 MANIFEST | 58个具体配置、地址/内容固定或可塑组合、路由/释放/门 | 348个job；包含复合、控制、精度与教学配置，不算58个独立机制 |
| V30实际终点记录 | 10个已运行方向/冻结/重排/范数版本 | screen失败、没有选择writer；未运行的practical方案不伪装成结果 |
| V31不确定性代码与结果 | 10种writer，搭配3种encoder | 有限horizon 8/32/128/512，无限、frozen/shuffle和对照；读取不足的encoder verdict void |
| 当前SAMPLE-GEN / FEEDBACK / SUPPORT | 9 / 6 / 3个实际配方 | 逐一保留DEV和CONF范围；COV-R跨实验重复，不额外算机制 |

这些是249个“程序/版本”记录，并非249个去重候选或存活项。71是旧条件保留集合，108与136是重叠的总账/目录。第一轮R/Z/P/T/S的16个操作另附原结果状态，不与上面跨底物同名字混合。完整参数、来源与结果引用在[VERSIONS.json](VERSIONS.json)；原始版本说明和已核对代码保存在[sources](sources)。这不是对全部旧notebook变体的无限展开计数。

## 同一机制的版本确实不同

| 机制方向 | 已查询具体版本 | 版本层面的判断 |
|---|---|---|
| 不确定性/重开 | V23 bayes_stable、bayes_reopening；V31 mesu_N8/N32/N128/N512、infN、frozen_sigma、shuffled_sigma；OMNIBUS E_META | V23两版本终点47.64%与38.01%，普通36.66%。V31只有observable输入够用，有限horizon结论inconclusive；不能把两种输入的void判为机制负面。E_META保护与获取的代价保留。当前完整协方差的有支持重开尚未实现，是新移植，而不是旧版本已胜出 |
| 再激活/回放 | V23 reservoir8/32、recent8/32、fastslow8/32、repeat_current；V24 budget4/8_replay；当前REPLAY-P/U、COV-REPLAY/EXTRA | V23 reservoir32终点40.55%，recent32为37.48%，普通36.66%，并有不同保持/获取代价。当前REPLAY-P已按残差选择，不是均匀版；不能再说“选择性回放未测”。信息覆盖型选择可以是新配方，但必须与残差优先及均匀同预算比较 |
| 保护/可用性 | pair_importance、strong、neuron_availability、shuffle；V24 bank4/8 fixed/local neuron/pair及release；V25 split_stable_local/release/gated | V24同E3、frozen、association、hard_b0：bank4_fixed_neuron终点66.88%，加release55.05%；并非增加释放就改善。V25稳定split局部router系列则local65.26%、release66.86%；slow_local66.88%、slow_release67.47%、slow_gated68.88%。这些均为描述历史均值，需各自父对照、区间和获取/保持指标；不可跨底物排名或声称当前算术增量 |
| 写入方向/相关结构 | V30 D/P、DX/PX、DN/PN；V34 C1/C4、C4-P；V35 H-RLS/LOCAL-R/LOCAL-V；OMNIBUS COV96；当前DIAG-R/COV-R、DIAG-I/COV-I | 多种“协方差”不是同一算法；当前COV-R确认优于DIAG-R，不等于历史方向归一化或无限记录版本都通过。当前I系列、EXTRA和完整COV每个版本分开，不重复实现现有COV-R当新机制 |
| 稳定/灵活结构与换边 | V25 FF/FL/LF/LL、split_frozen/live/stable；R/Z/P/T/S第一轮、K0–K3、当前TOPO-G/S/R | 当前TOPO-G已经同时学权重和导向换边。DEV综合分数0.1409，固定0.1318、随机0.1415；都未晋级。这关闭当前小网络/更新配方的承诺，不排除所有拓扑。不能重新命名同一P×S为“此前未测” |
| 快慢/巩固 | fast_slow2/3、feedback_consolidation、cascade4/8、fastslow_replay8/32、OMNIBUS B_CONSOL、bounded offline O | 现有耦合、深度和回放组合各有结果；无episode archive的O仍是未完整定义方案，不能当有实现的成功版本。更多时间尺度本身不提供新的算术约束 |
| 当前写门 | COV-R、COV-SUPPORT、COV-RATE；早期独立write/forget网络 | 用户采用的是SUPPORT写门实例，不是全部独立写/忘变体。RATE是诊断，不是完全逐记录匹配的门；当前主效率增量未确认，旧组合保护信号仍保留 |
| 递归 | raw MBON线性recurrence、KC↔KC/DPM、Fly-2 STM→LTM、recurrent post-event window | 它们有不同连接位置、状态及反馈信号，不能统归一个“U055联合重估”。一些是实际负面配方，一些只有来源/提案，状态必须分开 |

## 当前加法结果中的遗漏最重要

[SAMPLE-GEN DEV报告](../../../addition_sample_generalization_dev_v2_20261011/results/DEV_REPORT.md)给出同一开发样本的综合分数：COV-R 0.6892、DIAG-R 0.4724、COMPACT-ID 0.2109；REPLAY-P 0.3013、REPLAY-U 0.2892；TOPO-G 0.1409、TOPO-S 0.1318、TOPO-R 0.1415。这些是DEV而非全部独立确认。回放用compact增量底物，不能把它与完整COV的差距全部归罪于选择策略；TOPO自身小网络、参数学习和换边也需作为具体组合评判。

因此，上一轮“U047第一、U034第二”的结论降低为**待具体版本比较的暂定方向**。任何下一配方至少须标明：来自哪个已有版本；改变选择准则、底物、约束、参考还是预算；为何不是失败配方的改名；与谁在相同信息/样本/计算下比较。不能只用家族未被彻底否定作为投入理由。

真正仍可讨论的区别包括：完整相关更新中的有限不确定性恢复（区别于旧全局常量重开）、有界信息方向覆盖的再激活（区别于已测残差优先）、同一角色坐标下的稳定/灵活参数分工（区别于复制完整银行）、以及更新方向与观察支持的联合控制。这些尚是新版本设计选项，没有被这次文档复核证明有效；不会为恢复它们另开一串技术定位实验。

## 今后的筛选记账

每一条必须保留“机制→操作版本→参数配置→对照→结果适用范围→是否已实际运行”。同名称跨程序不自动合并；不同视图不算新生命；单参数剂量与实质算法改变分开；同一版本失败不替其它版本判死，也不允许靠名称变化复活相同失败操作。当前核心与所有冻结实验不变。本次未运行任何科学记录。
