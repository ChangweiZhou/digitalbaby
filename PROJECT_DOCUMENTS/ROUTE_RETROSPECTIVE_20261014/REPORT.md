# MiniFly 研究路线与能力归属复盘

2026年10月14日。覆盖可读取的9月13日至10月14日原生 MiniFly 及相关分支，仓库来源快照为 655f515。更早 V50–V54 作为独立前史处理。

**此前所说的“三次”不是完整的漂移数量，只是三条学习器路线的概括。** 目标收缩、独立能力的错误合并、工程先验、核心替换、采用后的迁移中断、母体混用、预算比较和机制编号错配，均需要单独复盘。下文列出10个核查节点；它们有重叠，不能把10个节点当成10次互不相干的越权实验或10份无效科学结果。

研究主线实际形成了多个并列分支，没有形成一条已经证明能从原始字节和反馈学会四则混合运算的连续升级链。多份局部报告准确写明了限制；限制进入下一轮设计和口头总结时没有被持续保留，这是本次复盘的核心问题。

## 判断口径

一次模型变化本身不等于漂移。用户允许调整 Full151 的内部结构，后来也允许大幅改变模型、使用反向传播作为工程参照。需要纠正的是：目标被代理指标替代；另一个学习器的结果被归到原核心；接口升级被当作能力继承；或者比较同时改变底座、表示和训练剂量，却把差异归因于某个机制。

连续的同类诊断合并为一个事件，不按文件夹、世界数或版本号累计。A03与A04是同一加法阶段的底座和先验两个维度；A06与A07属于同一通用化迁移链。不能据此给出“全历史恰好发生了N次漂移”的精确数字。

当前目标保持为：从实际反馈中学习，以更少外部样本和实际算力获得可保留的未见四则及混合运算整串自输出能力。没有预置算术规则或人工算术表示；不使用 Transformer、attention 或 ELM。预热可以见到真实答案，随后反馈可以稀疏，最终可以只有真实0/1。这些后来的约束不反向使先前获得授权的工程参照失效。

## 更早的记录

9月13日的原生 MiniFly 已有真实来源：302个PN、88个输入通道、5177个KC和27572条PN到KC连接，带快慢状态与解剖约束的教学路径。但当时报告明确说它不是经过验证的持久苍蝇脑；干扰保持标准没有通过。[最初可执行原生来源](/Users/pencilbard/Downloads/second_try_copy/minifly/iteration5/REPORT.md:3)

同日的 V68 回顾明确写道：更早 V50–V54 的 ELM/byte-task 是不同任务和代码路线，其 FUNCTIONAL_CORE_ESTABLISHED 标签不能证明 MiniFly 恢复了苍蝇核心。可见“不同底座共用核心称呼”的风险很早就被识别。V66本身是干扰方向诊断，没有晋升新学习模型。[早期跨路线归属警告](/Users/pencilbard/Downloads/second_try_copy/minifly/v68_review/REPORT.md:13)；[V66的真实范围](/Users/pencilbard/Downloads/second_try_copy/minifly/iteration11/REPORT.md:3)

目前材料足以确认9月21日起的目标收缩；更早发现的读出、保持端点和机制归因问题，不足以一律判作未经授权的路线替换。不能为了得到更大的次数而把所有旧模型、工程观察器和失败诊断计入。

## 十个需要纠正的节点

### A01 学习目标收缩成已教关联保持

9月21日至26日，FB1/PostV5到V9.7逐渐把主要问题集中到旧的已教cue能否返回、经过新写入后还剩多少。BYTE-CORE-1以不同包装、修订和信息归一化为理由，声称标题并非记忆背诵；这些条件并没有排除“键到后果”的记忆表。所有256个有序组合都训练过的ORDER任务，也不能单靠换序测试证明发现了未见组合的规则。

这是一个连续的目标与证据收缩事件，不按V9.3、V9.4、V9.5重复计数。保持与修订是必要的有效诊断，但不能替代复用知识解决未强化输入。9月26日已有审计明确承认此事，并形成E0到E4合同；后续仍复发，说明纠偏文件没有持续约束研究决策。[历史漂移审计](/Users/pencilbard/Downloads/second_try_copy/V88_TO_V9_7_OBJECTIVE_DRIFT_AUDIT_20260926.md:7)；[当时已经存在的目标合同](/Users/pencilbard/Downloads/second_try_copy/MINIFLY_OBJECTIVE_CONTRACT.md:7)

### A02 独立模块的成功留下未解决的整合边界

V9.2的字节预测来自新增1到4字节特征预测器；原生Full151价值路线不产生这些字节概率。V9.3检查两头共存，没有证明预测表示能教会价值记忆。后来CONTENT路线记忆较好、FE0路线在关系题上有迁移，也分别属于不同配置的资格。

报告本身多次明确说“共存”或“两个基线”。这些来源直接证明路径分离，不能单靠它们指控报告伪造整合，或还原某次口头总结的具体措辞。结合此前反复出现的“又需要整合验证”争议，应登记的是阶段退出与能力继承边界：两项分别合格不能自动变成一只苍蝇同时具有两种能力。若下一轮以此为已有整体能力，便发生归属漂移。本节点不单独算作一次已经查实的报告失实。[预测器与原生路线的分离](/Users/pencilbard/Downloads/second_try_copy/BYTE_CORE_V9_2/REPORT.md:13)；[两条路线分别资格化](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/minifly_b/package/DOCS/BYTE_FULL151_E3_ASSAY_QUALIFICATION_20260928/results/REPORT.md:1)

### A03 原生Full151逐步变成独立数值学习器

10月7日最初确实测过Full151加法：Native建立九份原生存储，读取fast、slow、adapt，并用实际答案形成signed教学。不能再说“从未把Full151用于加法”。但是，这只是一个特定的九输出、共同工程表示配方；失败不能否定整个原生家族。

同一三循环中，输入由解剖投影改为角色分区投影；候选又由类别读出改为单一序数标量，并使用自己定义的残差写入。它保留了fast/slow名称，却没有执行原生Full151写入方程。10月11日保持实验明确不再运行Native；STABLE取消被动衰减。随后COV-R只保留31个权重与31乘31统计矩阵，fast/slow/adapt也不在实际运行路径中。

这些模型变化有授权，报告也有披露；需要纠正的是后来统称“原MiniFly核心升级”。相同坐标、从旧资产取模板、沿用术语，都不等于继承旧学习机制。[实际Native与Residual实现](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/addition_rule_dev_v1_20261007/learners.py:53)；[三循环中改变了什么](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/addition_rule_dev_v1_20261007/THREE_CYCLE_REPORT.md:7)；[原生对照退出](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/addition_value_retention_dev_v1_20261011/EXPERIMENT.md:15)；[COV-R的实际状态](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/addition_sample_generalization_dev_v2_20261011/cores.py:53)

### A04 工程先验限制没有持续成为能力继承条件

早期加法模型解析左右操作数，提供角色与大小输入；COV-R保留角色槽、可加分离模型与序数答案渲染。它学习权重而非直接调用加法器，留出组合成绩仍是有效的有界E3，但它已经由研究者选择了适合加法的函数类别。

后来通用字节实验仍加入ASCII数字大小，VALUE-INTERACT再解析十进制答案、提供整数辅助目标和十进制周期特征。它没有偷看未见答案，也不是直接硬编码四则求值；但它回答的是“有这些人工数值结构时如何学习”，不能代替完全端到端的问题。RAW-RULE已经明确撤回相关方案。

用户后来要求所有表示都学得，不能用这个后来约束反向宣布旧实验违规；也不能把旧先验下的成功继续作为更严格任务已经解决的证据。旧设计主动披露了先验；本项纠正的是跨轮能力继承与性能解释，不是声称这些原始报告都宣称完成了裸字节规则发现。[角色及大小表示](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/addition_rule_dev_v1_20261007/representation.py:24)；[加性模型与序数读出](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/addition_core_v2_support_v1_20261011/README.md:19)；[十进制目标与周期特征](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/arithmetic_value_interaction_dev_v1_20261012/core.py:67)；[RAW-RULE的明确撤回](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/arithmetic_feedback_rule_discovery_design_v1_20261013/EXPERIMENT.md:5)

### A05 噪声处理逐渐成为主要研究方向

带噪或缺失答案最初由用户明确选择，使用噪声本身有授权。问题是SAMPLE-GEN之后的下一问题被定义为“模糊反馈下更好处理错误答案”，随后多轮支持门、方向恢复与其他配方仍在同一加法specialist上优化。其他运算与一般表示仍没有解决。

这是优先级漂移，不是噪声实验没有价值。用户后来明确指出噪声处理是以后方向，要求转向其他运算，纠正的正是这个排序。[当时的下一步研究问题](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/addition_sample_generalization_dev_v2_20261011/NEXT_STEP_REVIEW_20261011.md:24)

### A06 用户采用的核心与下一轮实际核心断开

用户明确采用COV-SUPPORT。采用记录保留NO_ADVANCE_RETAIN_COV_R、没有独立确认增量收益、确认世界为0；这是开发基线选择，没有篡改科学门槛。

但“让U042进入核心”实际落实为打包独立ByteLearner，并没有把它移植到Full151或R_center。当前加法指针承诺以该配方和COV-R为后续比较锚点；进入通用字节任务后却使用不同学习器、不同比较锚点，也没有保留SUPPORT操作。旧采用记录还在，新实验实际底座已换，缺少迁移与能力继承合同。

不能为了机械遵守旧对照而把不兼容的标量模型强塞进新字节任务；应该明确说旧核心不满足新接口、哪些操作已移植、哪些能力尚未继承，并给出可比较的依据。我没有把这个实现范围充分讲清楚，造成了“已经加到Full151之上”的理解。[用户采用与实际打包范围](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/addition_core_v2_support_v1_20261011/ADOPTION.json:14)；[采用后的比较承诺](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/PROJECT_DOCUMENTS/CURRENT_ADDITION_CORE.json:17)；[新的比较锚点](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/arithmetic_mechanism_transfer_design_v2_20261012/EXPERIMENT.md:46)

### A07 通用输入输出升级实际另建了记忆底座

通用字节IO使用4096乘257类别塑性银行，精确上下文哈希选择16个接点，没有Full151的原生快慢/adaptation，也没有被动衰减。它在已教可变长序列上的字节与EOS获取、干扰后保持真实有效。

报告明确称其为新的工程IO参考，指针也明确“不继承旧加法E3”“不声称转换原生Full151”。因此不是代码隐瞒，但它也不是原核心加一个输入输出适配器。把此后成绩继续说成同一MiniFly的通用化，会跨越新的底座边界。24小时只推进时钟的稳定，也不是昆虫巩固机制的证据。后续四则迁移、ADAPT-2与VALUE延续此分支，不应分别再算三次离开Full151。[新哈希银行的真实能力与范围](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/generic_byte_io_v1_20261012/REPORT.md:7)；[明确未继承旧能力的指针](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/PROJECT_DOCUMENTS/CURRENT_GENERIC_IO.json:14)

### A08 基于旧设计的含义两次改变

KERNEL V3不只是给旧GRU添加一个机制。它同时改变GRU方程、初始化、读出归一化、精度、优化器、更新限制、信用范围及任务剂量。保存的复盘已经直接承认：“V3不是旧核心的受控升级”，此前简单称为“基于旧设计”不准确。

随后SOURCE-CELL寻找真实旧代码，恢复的是ANSWER-0与另一条旧ELM路线；来源表明确把R_center列为历史反馈时序和快慢写入参考，并未继承其运行时存储。这说明“旧核心”又换了指代。用户允许新结构，不等于可以省略母体选择的解释。

ELM后来被明确排除。没有证据表明禁令之后实际又运行ELM：MODEL-FEEDBACK只实例化非ELM的B24；GATED使用独立GRU/LSTM。不能把早期获授权分支倒算成违反后来禁令。[已承认的V3谱系问题](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/arithmetic_kernel_no_attention_dev_v3_20261013/ops/read_only_lineage_review_20261013/REVIEW.md:21)；[SOURCE-CELL真实母体及R_center角色](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/arithmetic_source_derived_kernel_dev_v1_20261013/SOURCE_LINEAGE.json:39)

### A09 任务剂量与证据等级混写造成表面退化

早期100%有的是精确关联记忆，有的是人工加性表示上的有限加法，有的是大量重复后的纯字节已教题。新四则及混合任务改变了输入范围、答案长度、任务数、反馈制度和每题重复次数，还同时改变了学习实现。

V3复盘记录旧ANSWER-0每字面加法题约280次呈现，新完整G0的不同运算约5.5到11.4次，混合题各4次。因此不能直接说“同一只苍蝇增加合理要求后智力消失”。这不证明只加训练就能泛化；充分重复的旧模型也没有可靠未见乘法优势。

三循环完成主要表示实现资格：接口、状态、反馈分支、梯度及审计能工作。它不等于机制有效、正式确认通过或核心可以采用。近期报告多数已明确这个区别；后续口头总结和决策也必须保留。题内状态、跨经验权重、原生快慢突触痕迹同样不能只因都叫memory就互相替代。[旧新剂量与内容获取的复盘](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/arithmetic_kernel_no_attention_dev_v3_20261013/ops/read_only_lineage_review_20261013/REVIEW.md:21)；[最新技术资格的实际结果](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/arithmetic_gated_feedback_dev_v2_20261014/REPORT.md:1)

### A10 机制编号跨版本错配污染了筛选理解

完整清单存在108、136、254、71和最初16项等重叠视图，不能相加，也不能只用短名单代替完整版。更具体的错配包括：

| 名称 | 原总账含义 | 后来实际操作与正确解释 |
| --- | --- | --- |
| U042 | 独立写入门与遗忘门，来源M012 | COV-SUPPORT只是观察支持度控制写入，没有独立遗忘门。用户采用的是SUPPORT配方，并非完整双门机制已被验证。 |
| U034 | 历史Bayes reopening | SUPPORT-DIR是另一底座上的方向恢复；它的失败不等于所有Bayes重开版本失败。 |
| U055 | KC间有界递归与广义调制反馈，来源M094 | 曾被误归为多观察联合重估，后来正式纠正。KCI写入抑制代理也不等于真实KC/DPM动态递归。 |

最新完整复核已经做了实质纠正，并保留无整族科学排除的边界；部分旧排期入口仍有混挂。不能说所有负结果和总账都无效，但也不能据旧入口的“采用”“停止”直接计算存活机制数。精确身份至少应是来源编号、实际方程版本、运行底座、任务与反馈合同。[原始71项语义](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/PROJECT_DOCUMENTS/MECHANISM_SCREEN_20261007/DEDUP_RETAINED_ITEMS.md:48)；[U055公开更正](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/PROJECT_DOCUMENTS/MECHANISM_SCREEN_20261007/POST_U055_RESCREEN_20261011.md:8)；[最新完整复核及命名纠正](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/PROJECT_DOCUMENTS/MECHANISM_SCREEN_20261007/FULL_REVIEW_MODEL_REWARD_20261013/SUMMARY.json:1)

## 不能算成新的路线漂移

用户要求四则与混合运算、任意字节输入输出、端到端学表示，是更明确的任务标准，不是解释失败的借口。用户允许改内部结构、允许工程反传参照、要求两份设计并行，也是明确授权的探索。

visible-context桥接、R_center封装、原生自主观察各有明确的窄范围目标。R_center确实保留八份独立Full151；自主观察确实从到来字节生成内部误差，更新原生快慢状态。它们本身不能一概记作“又换掉了苍蝇”。[R_center的原生底座](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/r_center_core_v1/README.md:10)；[原生自主观察基线](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/autonomous_observation_v1_20261007/README.md:11)

Package A/B的禁写分支缺陷、出生或source lock问题、运行中断属于实现与执行质量问题；必须按修正审计确定哪些端点有效。它们不自动等于路线漂移，也不提供科学否定机制的证据。最新完整总账已把技术不合格、E3无效、具体配方失败和未决区分开，不能反过来声称全部清单已被科学排除。[完整复核的证据边界](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/PROJECT_DOCUMENTS/MECHANISM_SCREEN_20261007/FULL_REVIEW_MODEL_REWARD_20261013/SUMMARY.json:1)

## 必须保留的正面证据

Full151并非从未表现泛化。V9.9A在64个新世界的未强化组合上，最终原生写入优势为+17.19个百分点，95%区间[+10.39,+23.98]；V9.9B在第二关系族为+9.57个百分点，[+5.09,+14.05]。这是有限合成关系复用，不是四则或通用推理，但足以否定“所有旧核心都只能精确查表”的概括。[原生第一关系族确认](/Users/pencilbard/Downloads/second_try_copy/BYTE_CORE_V9_9A/REPORT.md:3)；[第二关系族及其限度](/Users/pencilbard/Downloads/second_try_copy/BYTE_CORE_V9_9B_RANK/REPORT.md:3)

COV-R在明确的加性模型类上取得未教组合收益，也仍然有效。U042的用户开发采用决定保留，其增量低样本收益没有独立确认。通用哈希银行在已教字节序列上完成了IO与记忆验收。新GRU/LSTM的负面或技术结果也保留，但不能反向抹掉旧分支的能力。

## 当前实际并存的学习器

| 分支 | 实际学习状态与计算 | 仍可保留的结论 | 没有继承或没有证明 |
| --- | --- | --- | --- |
| 原生Full151及R_center相关基线 | PN到KC及原生fast、slow、adapt；R_center有shared/private多存储 | 有界关联获取、保持；不同原生配方有有限合成关系E3；原生自主观察有E0/E1确认 | R_center自身未重新确认此前所有E3；开放字节四则未完成 |
| 加法Residual、STABLE、COV-R、COV-SUPPORT | 自定义数值内容，后来31权重与协方差、支持门 | 工程加法模型内的组合泛化；COV-SUPPORT为用户采用开发基线 | 不是Full151运行时升级；没有从无人工表示的字节中发现算术 |
| 通用IO与机制迁移、ADAPT-2、VALUE | 新哈希银行及不同工程共享表示/可塑读出 | IO验收、各轮具体负结果 | 没有继承原生巩固、旧加法参数或原生E3；数值先验版本不满足后来纯字节标准 |
| RAW-RULE、STREAM、KERNEL、SOURCE-CELL、MODEL/GATED | 不同GRU/RLS或端到端GRU/LSTM；SOURCE-CELL曾另有ELM分支 | 相应资格、已教获取及负面行为证据 | 没有完整继承Full151或COV-SUPPORT；没有确认的新四则核心 |

因此，“以前能记住、以前能做加法、现在不行”不只包含一次难度上升，也包含多次底座与先验变化。它不是同一个模型的连续退化曲线。当前也没有一个已经证明统一拥有这些历史能力的新核心。

## 根因与应当执行的纠正

主要责任在设计与解释的连续性管理。局部实验往往严格冻结、审计并报告边界；进入下一轮时却把“用过旧资产”“同名机制”“同一接口”“历史成功”当作实际继承。局部文档中的限制没有变成跨版本必须核对的条件。不能用某篇报告的脚注为整体误导免责，也不应让用户为每一轮重新发现同一断点。

今后的同一份设计必须固定以下事实：

1. 明确哪份可执行学习器是母体，哪些持久状态、写入、读出与反馈路径保留，哪些换掉。替换是允许的，不能继续按未变化的旧模型归属结果。
2. 每个历史正负结果都附任务、表示先验、反馈、样本与算力预算，以及实际产生答案的分支。更换底座或接口时能力继承默认未证明。
3. 当前核心须区分用户采用、科学确认、工程参照、历史父本和实验候选；不能用多个CURRENT指针暗示它们已合并成一个系统。
4. 原生基线不适合新接口时，应明确记录不兼容和迁移方案，不能先换成独立模型，再用后者的结果解释原生核心优劣。
5. 已教题获取是必要测量，不能替代未见题终点；技术资格不能替代行为效益；内部重复与回放照样计费。
6. 使用完整机制总账的实际操作身份，停掉具体配方不等于排除整个机制家族；被禁架构与实验负面也分开。
7. 用户的新明确要求按生效时间执行。不能把合理的四则、字节输出和反馈要求归咎为研究失败的原因，也不能以回到简单题或只测噪声处理来宣布完成。

本次只新增路线复盘与可查询记录，不改变旧实验判定，不运行新轨迹，不自动发布。下一份实验设计应以这份继承审计为依据，直接服务既定未见四则与混合运算目标，不再把又一轮纯定位或整合资格当成新的研究终点。

