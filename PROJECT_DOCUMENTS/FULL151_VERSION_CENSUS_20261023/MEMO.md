# Full151 之后：版本、结果与下一步机制复盘

2026-10-23。目标保持：在渐进热身和允许的反馈下，提高**未见加法题完整自主字节输出及自身 EOS 的样本／实际算力效率**，保护 OLD。不引入算术结构或规则，不把外接记忆、已教题复述、组件读出成功当成这个目标已经实现。当前原生搜索不使用 attention、Transformer 或 ELM。

## 先回答数量问题

按可核对、可复查的“有实际实验／技术执行记录的命名交付单元”计算，合并 GitHub 各分支后有 **53 个实验交付单元**，全部逐行列在 [CATALOG.md](CATALOG.md)，机器目录在 [EXPERIMENT_CATALOG.json](EXPERIMENT_CATALOG.json)。这包含 Full151 后的两条 Package、R_center、后续记忆／机制实验、工程化加法、独立递归参考、原生接口／绑定、warmup、memory-channel 和 V6-R1。不是只列最近几轮。

**不能把它称为 53 个独立脑模型。** 同一交付可以比较多个核心；同一核心又可能有 DEV、确认和接口资格多个交付。例如 STABLE 的 DEV／CONFIRM 分开交付但不是两种新脑；STOP-ONLY 确认也没有再造原生内核。技术周期的源码快照、同一历史的多个读出、控制分支和 dispatcher 改版不算独立学习器。表内保留了这些身份和证据等级；技术通过、科学通过、用户采纳分别记账。

另有 **40 个设计／评审目录名**未计作这 53 个已执行交付。它们有的是相应实验的设计包，有的是未实现提案，不能说另外试过 40 个模型。目录名见 JSON。

工作文件夹也扫描了：`Downloads/second_try_copy` 与 `Documents/second_try` 中找到 **187 份报告文件，按报告 SHA256 去重为 158 份不同内容**。完整索引见 [WORKFOLDER_REPORT_INDEX.json](WORKFOLDER_REPORT_INDEX.json)／[LOCAL_REPORTS.md](LOCAL_REPORTS.md)。这里包括 Full151 前身、Full151 bridge/output 系列、BYTE_CORE V9.x 和镜像副本；故这是补充历史报告普查，**不是另外 158 个 Full151 后新版本**。逐报告去重也不是证明两个执行内核完全相同。Full151 没有一个统一管理全部本地工作、源码身份和发行日期的全局注册表，因此不制造一个跨报告／跨机制的“总共 N 个独立脑”精确数字。

此前的“200 多个机制”找到了原件：**254 条具体程序／配置版本**，分为 V23 52、V24 101、V25 58、V30 10、V31 10、SAMPLE-GEN 9、FEEDBACK 6、SUPPORT 3、DIRECTION 5。完整逐版本复核在 [MECHANISMS_254.md](MECHANISMS_254.md)／[JSON](MECHANISMS_254_REVIEW.json)。另外保留 [108 项总账](SOURCE_HISTORICAL108.json)与 [136 行目录](SOURCE_CATALOGUE136.json)。这些是重叠视图，**254、108、136、53 不相加**。本次不是再用旧 71 项子集代替完整版，也没有把“未全家族排除”计成“仍有效”。

## 这些版本实际改变了什么

| 路线 | 代表证据 | 可得结论与不能继承的部分 |
|---|---|---|
| Full151 / byte-value 原生祖先 | 本地 V9.9A 64 世界最终 W−N +17.19pp [10.39,23.98]；V9.9B +9.57pp [5.09,14.05] | 有限合成关系的因果 E3 信号是真实正面证据。不是“原生系统永远只能查表”，也不是已学会加法。V9.9B 尚不能排除粗二分解释。 |
| Package A / B 与响应机制 | A 完成 832 收据；B 也完成 832，但关系禁写分支名称错误 | A 要按具体 arm／端点判断。B 的 E3 轴无效，不能把所有 B 机制判成科学负面。T/H/J 响应机制则有自己的有效 E1 结果，T 该配方显著有害。 |
| R_center / ERROR / compact / persistent V2 | R_center taught OLD 97.14%，Latin completion 因果效应 −8.20pp；V2 保留 V1 | 记住被教关联与推出新关联有明确分离。ERROR 有获取／修订价值，但不因此解决 completion；缩银行未通过可靠性全门。 |
| STABLE、COV-R、U042 | STABLE 限定 0–4 加法确认成功；COV-R 在工程化表示上确认；U042 原预定 DEV 不晋级但用户随后采用 | 正面能力有边界：数字、角色／加性及 ordinal 输出先验公开。这些不是直接叠到原生 Full151 后自动生效的机制。U042 是采纳决定，不改写成独立确认的样本效率优势。U034 当前配方失败。 |
| RAW-RULE / STREAM / KEEP / G62 等独立参考 | 有些可以 taught 100%，但 held 大多低；乘法后加法损失严重 | 原始字节任务并非所有模型都完全不能学。格式获取与规则迁移需分离；SHUF 预训练也加快乘法，不能将速度收益全归于加法知识。 |
| N151 / C151 / RC151、NATIVE-BIND | 原生完整题获取弱；V1 24 作业、V2.1 72 作业均 NO_ADVANCE；小型 2 题接口又能学会 | 不是统一的“完全收不到教学信号”，也不是已证实四则太难。接口范围、续写键冲突、状态／地址条件和负载影响获取。提高 recurrence／信用复杂度未自动解决这些问题。 |
| Claude 诊断与 CLS | 原生 fast memory 的纠错写入／adaptation reset 改善 taught memory；CLS replay 改善部分保持，未通过 unseen sums | 这是应纳入复盘的真实工作。slow e-prop、BPTT 与 replay 不是原生 MBON 已学会算术；CLS 最佳 unseen sums 13%，e-prop 0%，不能按框图宣称机制融合成功。 |
| v3 → v4 → v5.1 → R2 | 从端口到 byte identity，再到有界端到端；v5.1 有 aliasing，R2 修复上游却遇到资源上限 | 前轮部分负面定位在 E0，不应全解释为算术抽象失败。R2 6 条完成历史之外的缺失端点不能补成失败或成功。 |
| memory channel / recoverability / STOP / capacity | STD 可保留部分 payload；joint head 会破坏；STOP-only 确认改善条件停止；扩容提高 E1 但 CPU 约 3.15 倍 | 更大 decoder 不是数据支持的首选；组件成功仍远离可靠完整生成，更不是 E3 加法。扩容同时改 wiring/self-gain，不能归因于纯 N，也不证明算力效率。 |
| V6-R1 | 2 世界、8 历史、34,816 应用；OLD/NEW/held exact 均 0，自主 EOS 0；100% 截断 | 此次连获取也失败。保存态诊断发现固定噪声时换输入几乎不改 KC，而同输入换噪声地址几乎无关。当前应处理条件编码与学习信号，不能仅追加保持机制。 |

上表是解释，**完整结果目录仍然是 53 行，而不是把整个历史压成这 11 类**。本地祖先正负证据入口见下；GitHub 结果由目录链接到对应冻结报告。仅复读文档、源码元数据与既有汇总，没有运行新训练、复演轨迹或重做科学验证。

## 为什么一度“能加法”，后来既过拟合又学不动

没有一条模型连续获得这些能力再全部丢失。至少发生了三类变化：

1. **表示和任务发生变化。** 工程化加法把数值／角色和输出先验交给学习器；raw-byte 任务要求自己形成身份、顺序、组合和输出终止。前者的成功不能直接推出后者已经可解，更不能算成 Full151 自带能力。
2. **执行学习器发生变化。** generic byte bank、KEEP/G62、独立 GRU、Full151/R_center、native bind、CLS 的更新和输出路径不同。同一个项目名、连接资产或 warmup 名称不能传递机制身份。
3. **不同轮次卡在不同环节。** KEEP 的 taught 成功／held 失败接近过拟合；v5.1 的身份合并是上游表示失败；输出后缀冲突是续写条件失败；R2 还有资源未完成；V6-R1 则有严重 cue-to-noise 问题。它们不是同一种零分，也不该用统一的“生物局部学习不可能”解释。

V6-R1 的 fixed-tape 比较中，输入效应约为噪声效应的 1/70–1/87；同 tape 不同 prompt 的 KC Jaccard 约 0.98–0.99，同 prompt 不同 tape 仅约 0.01–0.02。这个保存态结果支持条件地址被随机变化淹没的定位；它不是唯一因果证明，也不保证已提出的 fan-in 初始化修订能救回泛化。下一步仍需比较完整自身生成，而不能把状态分离度升高当成 E3 胜出。

## 254 条里还有什么值得做

不是再找一个名称还没“排除”的机制大规模筛选。先用当前失败的因果位置决定能进入同一原生核心的操作，保留以下有限方向；所有实现仍只能接收字节和允许的反馈。

| 优先级与方向 | 完整清单中的具体来源 | 已有证据／反证 | 当前判断 |
|---|---|---|---|
| 1：稳定、可被输入条件化的状态／地址 | `V25/context_stable`、`V25/split_stable_base_local`、`V25/split_stable_slow_local`；完整 136 表的 FE0 / receptor trace | native suffix 冲突、adaptation 状态不匹配及 V6 噪声问题提供了明确靶点。历史 ELM 实例不是当前可直接复用的内核。 | 值得保留为通用原理。首先完成已提出的 V6 cue-scaling 对照；若无获取收益，不叠加新机制。下一方案必须保持字节置换对称，不用 operand slot 或数值代码。 |
| 2：条件正确的误差纠正／教学支持写入 | 108 表的 native teaching/error 相关条目，历史 ERROR、COV-R／U042 精确版本；Claude `fastmem.py` | ERROR 改善获取／修订；Claude native fast-memory 的纠错写入＋adaptation reset 有 taught 证据；但 COV-R 的强成功来自另一路工程表示，CREDIT 保护最近也为 null。 | 若输入条件稳定而 taught 仍弱，值得做单一 native 写入比较。纠错必须来自当时允许的反馈；scalar-only 阶段不能偷用 expected-byte loss。不是把完整 COV 预测器挂上后归为原生成功。 |
| 3：稳定／灵活分工与慢状态保护 | `V23/pair_importance`、`V23/neuron_availability`、`V23/fast_slow2`、`V23/feedback_consolidation`；V24 release 系列；V25 split-stable 系列 | 旧 association 上部分 parent 对照有保持收益，但获取／revision、reader、state 成本不同；当前 CREDIT 配方不工作。 | taught 获取及地址一致性先通过，再在同样字节历史上看 NEW 造成的 OLD 写入损失。不能靠无限银行、教师任务 ID 或额外记忆副本解决。 |
| 4：有界、真实已观察数据的再激活 | `V23/replay_reservoir8/32`、`fastslow_replay8/32`；`V24/budget4_replay`、`budget8_replay`；`SAMPLE_GEN_V2/REPLAY-P/U`、`FEEDBACK_EFFICIENCY_V1/COV-REPLAY` | 残差优先不是未测；compact 上的结果不能直接和 full COV 差距归因于选择策略。CLS replay 改善保持，但没有证实规则复用。 | 后置候选。固定缓存、存真实字节、所有重复和 CPU 计费，与同预算额外在线训练比较。它能保护经验，不能自动产生未见题规则。 |
| 保留对照，不优先重跑 | U034/SUPPORT-DIR；TOPO-G/S/R；RATE；full bank expansion；更大 joint decoder；历史 recurrence 名称 | U034 与 TOPO 的精确配方已有失败，RATE 是诊断；扩容有 E1 收益但成本高；joint decoder 会破坏 payload；已有递归配方也失败过。 | 当前没有理由改名复活。整个机制家族没有被数学排除，不构成立刻再试的理由。 |

这里没有“仍存活 73 个机制”的结论。JSON 中 73 条标成 conditional 的历史配置，只表示其操作原理与后续获取／保持有关；它们跨底物、参数、对照重复，**不是 73 个有效、兼容或待运行候选**。六条历史 anchor 行也只对应 COV-R／SUPPORT 两个实际 anchor。108／136 总账中的 ELM、attention 相关历史仅保留证据，不进入当前方案。

建议下一步只做**一个受控的原生编码修订比较**，以已经提出、尚未执行的 [V6-R1-R2](../../minifly_v6_r1_dev_20261023/NEXT_DESIGN.md) 为起点。保留 R1、匹配的冻结／NO_OLD／SH_OLD 对照，渐进 warmup 与所有训练成本；同时看 taught acquisition、真实自主 EOS、held 首次反馈前完整输出、OLD 保持。若只改善 taught，这仍是 E1；若 E3 没有因果增益，就不升级为成功。获取通过之后再从写入纠正或保护选一个操作，不把三四个机制一次叠起来。此处是建议，不授权或启动实验。

## 分支整理方式与可恢复性

新工作分支为 `codex/minifly-consolidated-20261023`，作为唯一 GitHub 分支及默认分支。原 18 个远端 tip 的 SHA、归档 tag 和兼容旧链接的同名 tag 见 [BRANCH_ARCHIVE.json](BRANCH_ARCHIVE.json)。所有原始 tip 均是新分支的祖先；不是仅以 ours merge 隐藏分歧，已实际合并各分支内容。

v4 设计 add/add 冲突保留当前冻结 `DESIGN.md`，并将 Claude 原文单独保存在 `minifly_v4_design_20261020/CLAUDE_BRANCH_ORIGINAL_DESIGN.md`；两份原始提交也完整保留。旧 branch 名同名 tag 让历史 `blob/<old-name>/...` 链接继续指向固定历史版本。tag 是归档，不是继续开发的分支。

本次 V6 发布及新增普查材料只含源码、设计、汇总结果和审核元数据；V6 的 trace、checkpoint、训练 manifest、数组与结果压缩包不上传。合并会把其他**早已存在于 GitHub 历史**的文件纳入统一树；这不是重新上传本地新原始数据，也不会清除旧公开历史。单分支整理不能视为删除过去已经发布的数据。

## 核对方法与范围限制

- GitHub：获取全部 18 个 head，比较祖先关系，合并 5 个分歧 tip；逐交付读取主报告／汇总／状态／final audit。记录所读对象 SHA256。技术根目录与后来 `results/` 中科学状态并存时，不能用较早的“science not started”覆盖较晚结果。
- 本地：两个工作根目录，报告名 `REPORT.md`、`FINAL_REPORT.md`、`FINDINGS.md`、`ROUND1_REPORT.md`，目录深度上限 5；排除依赖、vendor、reference snapshots、data、原始轨迹、checkpoint、receipt、run、archive 等树。未解包 zip、未查隐藏目录和未遍历无限深历史；所以 158 是此次可见报告内容数，不宣称历史所有 notebook 变体已穷尽。
- 254 机制版本：原参数、原结果、旧精确配方判断逐行保留；新字段是当前操作层面的保守 triage，没有重运行旧机制，也没有重新计算或纠正其多重比较。
- 53 是 GitHub 的明确交付口径；本地独有 V9.x 的合成关系工作由补充索引覆盖，不能当作已经推送的另一批原生加法版本。本地镜像和包内 vendor 不重复计作“又试了一个核心”。

本地关键证据：

- [V9.9A 报告](/Users/pencilbard/Downloads/second_try_copy/BYTE_CORE_V9_9A/REPORT.md)
- [V9.9B 报告](/Users/pencilbard/Downloads/second_try_copy/BYTE_CORE_V9_9B_RANK/REPORT.md)
- [V9.10E 教学几何诊断](/Users/pencilbard/Downloads/second_try_copy/BYTE_CORE_V9_10E_REPLAY_GEOMETRY/REPORT.md)：teaching span 并非严格无可达方向，但三个 train-label-only 估计器均未表现 cyclic transfer；hidden-label oracle 不是学习器。
- [Full151 原生祖先及迭代报告](/Users/pencilbard/Downloads/second_try_copy/minifly/REPORT.md)

这些本地路径是可复核入口，不冒充所有文件都已在 GitHub 发布；发布索引包含其内容 SHA256 与镜像位置。
