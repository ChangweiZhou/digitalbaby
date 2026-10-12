# 108项原总账逐项对照

原条目、原证据和处置完整保留在JSON及下方逐项说明。长期核心价值和当前加法排期分开；任何关闭只限具体已测版本。

| ID | 原机制 | 长期核心状态 | 最新范围判断 | 71子集映射 |
| --- | --- | --- | --- | --- |
| M001 | Stateful nonlinear ELM cell (fast s, slow m, recurrent messages, internal MLP) | 长期参照/混合历史证据 | 表达能力参照；部分任务依赖BPTT/向量私有状态，不能计为已验证自主持续学习。 | 未映射≠淘汰 |
| M002 | Scalar-channel ELM with private vector memory | 完整结果/比较缺失，不能判死 | 完整六细胞比较未恢复；缺结果不等于排除，保持与标量通道不同的私有状态记账。 | 未映射≠淘汰 |
| M003 | Fixed-time-constant ELM | 完整结果/比较缺失，不能判死 | 已实现/部分日志；没有足够完整对照支持删除全部时间常数方案。 | 未映射≠淘汰 |
| M004 | Linear-dendrite ELM | 完整结果/比较缺失，不能判死 | 线性树突版本仍含内部MLP；完整比较缺失，不能称已验证纯线性核心。 | 未映射≠淘汰 |
| M005 | Simple scalar leaky recurrent cell | 长期参照/混合历史证据 | 最简单标量动态参照；结构简单不自动代表低样本复用强。 | 未映射≠淘汰 |
| M006 | Adaptive-threshold leaky cell (ALIF-labelled) | 完整结果/比较缺失，不能判死 | 自适应阈值已有实现，但缺完整持续学习比较；不能只靠ALIF名称判优。 | 未映射≠淘汰 |
| M007 | Scalar sparse GRU-like cell | 完整结果/比较缺失，不能判死 | 同图GRU样实现参数量不同；缺完整结果，作为表达/成本参照。 | 未映射≠淘汰 |
| M008 | RG-LRU / simplified Griffin recurrent model | 长期参照/混合历史证据 | 门控递归有任务信号，但不是已确认持续复用或加法样本效率。 | 未映射≠淘汰 |
| M009 | FLM fast/slow leaky cell with sparse punctuator | 因果/资格/统计限制，非整族负面 | 语言/吞吐日志的Transformer比较存在NaN；不能用无效比较支持核心提升。 | 未映射≠淘汰 |
| M010 | Multi-timescale private neuronal state | 长期参照/混合历史证据 | 保留私有多时间尺度方向；它与复制多个完整银行不是同一个操作。 | U041:NOT_NEXT_ROUND |
| M011 | Stubborn/adaptable heterogeneous populations | 长期参照/混合历史证据 | 历史改善被学习率/活动门对照缩窄；没有独立异质性特异性胜利。 | U032:NOT_NEXT_ROUND |
| M012 | Separate write and forget gates | 长期仍有条件价值，须区别具体版本 | 当前只采用U042的SUPPORT写门实例；低样本提升未确认，不代表独立遗忘门已通过。 | U042:ADOPTED_BASELINE |
| M013 | E/I signs, proper log-tau, homeostasis, blank-activity penalty, delay training | 完整结果/比较缺失，不能判死 | 保留真实参数与实际dispatch范围；多个flags存在不等于全部变体科学运行过。 | 未映射≠淘汰 |
| M014 | Raw-node spiking LIF / richer conductance cells | 完整结果/比较缺失，不能判死 | 原始脉冲/电导细胞缺同底物完整对照；不为生物复杂性本身追加实验。 | 未映射≠淘汰 |
| M015 | Type-shared templates versus merged neurons | 工程组件/对照或信息边界 | 类型共享参数不得合并各神经元可变状态；是结构约束，不是已胜出学习操作。 | 未映射≠淘汰 |
| M016 | Fixed sparse sensory/recurrent graphs and retinotopic prior | 工程组件/对照或信息边界 | 固定稀疏图可作底物/对照；不能将静态拓扑或输入先验单独算新能力。 | 未映射≠淘汰 |
| M017 | Prune-only / random-regrowth / guided-regrowth | 长期仍有条件价值，须区别具体版本 | 当前TOPO-G已经联合学权重与导向换边，DEV未晋级；不同P×S操作仍须明示区别。 | U043:CONDITIONAL_CANDIDATE |
| M018 | New-edge lesion and topology-only transfer | 完整结果/比较缺失，不能判死 | 新边消融/仅拓扑迁移完整结果未恢复；不可用缺失当负面。 | 未映射≠淘汰 |
| M019 | Global cosine-kNN graph rewiring, hubs, EMA routing | 因果/资格/统计限制，非整族负面 | 早期global-kNN版本有未来信息或探测改状态问题；先标因果不合格，不能当家族科学负面。 | 未映射≠淘汰 |
| M020 | Local sensory-only development | 长期参照/混合历史证据 | 局部感觉开发有局限阳性；观察目标与干净latent目标、Hebb/Oja与BP分别记账。 | U004:ENABLING_COMPONENT |
| M021 | Clean latent reconstruction target | 工程组件/对照或信息边界 | 干净latent重建含额外信息；其成功不能证明仅凭实际噪声答案的学习。 | 未映射≠淘汰 |
| M022 | Frozen, live, split-stable encoders; FF/FL/LF/LL address/content states | 长期仍有条件价值，须区别具体版本 | FF/FL/LF/LL及冻结/活动/稳定split都已有具体版本；当前角色输入固定，不把旧结构重命名成未测。 | U044:ENABLING_COMPONENT |
| M023 | Dense affine/readout transport and anchor transport | 工程组件/对照或信息边界 | anchor/仿射运输是输入坐标兼容参照，成本与非局部性有界性待明示。 | 未映射≠淘汰 |
| M024 | Learned down-stream adapter/recycling | 因果/资格/统计限制，非整族负面 | adapter采用隐藏BP的版本局部性不合格，不是数值上等同失败。 | 未映射≠淘汰 |
| M025 | Learned operator mixtures / neural algebra / global group routing | 长期参照/混合历史证据 | 算子混合/神经代数是复合demo；没有隔离出持续核心增量。 | 未映射≠淘汰 |
| M026 | Sinkhorn routing, attention hubs, global workspace, token MoE | 长期参照/混合历史证据 | 注意力/Sinkhorn等全局路由属于不同范围；未排除，但不据名称自动排期。 | 未映射≠淘汰 |
| M027 | Hebbian key/value fast matrices; residual/predictive Hebbian writes | 长期仍有条件价值，须区别具体版本 | 当前COV-R为已确认的有限工程加法E3锚点；旧Hebb快矩阵与完整RLS不是同一实现。 | U045:ADOPTED_BASELINE |
| M028 | Modern Hopfield memory and episodic tape | 长期参照/混合历史证据 | Hopfield/episode tape的上下文重置演示不等于跨生命保留；检索成本与存储另记。 | 未映射≠淘汰 |
| M029 | Gauge/fiber semantic–surface factorization; contextual router | 长期参照/混合历史证据 | semantic/surface分解的监督复合结果不能冒充自主学得的不变性。 | 未映射≠淘汰 |
| M030 | SAC/proposal policies, learned teacher, dialogue warmup, curiosity/ASK actions | 长期参照/混合历史证据 | 奖励/预训练/教师先验与本核心获取信息预算不同，先保留参照。 | 未映射≠淘汰 |
| M031 | MagmaAdamW/global optimizers; neuromorphic detach mode | 工程组件/对照或信息边界 | 全局优化器/隐藏梯度改变学习信息与成本；局部detach名称不自动消除这些差别。 | 未映射≠淘汰 |
| M032 | Chunking, parallel Hebbian scan, sparse punctuator, scaling optimizations | 工程组件/对照或信息边界 | 吞吐/scan/分块是工程实现收益；不作为新增泛化机制。 | 未映射≠淘汰 |
| M033 | Current-only classifier / ordinary normalized delta | 长期参照/混合历史证据 | 普通delta是有用对照；其历史遗忘不排除所有低成本更新。 | 未映射≠淘汰 |
| M034 | Cumulative ridge sufficient statistics / exact RLS | 长期仍有条件价值，须区别具体版本 | 历史ridge/RLS有保持阳性，也有修订/安全失败；与当前COV-R已包含的相关统计分开。可讨论的新操作必须不是再实现现有RLS。 | 未映射≠淘汰 |
| M035 | Slow-only versus combined fast+slow output | 长期参照/混合历史证据 | slow-only与fast+slow的具体读出曾不同；简单相加可能稀释信号，不默认双路即更好。 | 未映射≠淘汰 |
| M036 | Fast-feedback weighted ridge consolidation | 停止所测具体配方，家族保留边界 | 精确反馈巩固配方有小特异性信号但总体低于父基线；停止原样采用，不排除所有巩固。 | 未映射≠淘汰 |
| M037 | Additive consolidation bonus | 停止所测具体配方，家族保留边界 | 加法式consolidation bonus未建立增量且增加统计状态；停止该具体配方。 | 未映射≠淘汰 |
| M038 | Synaptic importance / neuron availability / protection | 长期仍有条件价值，须区别具体版本 | importance/availability多版本保护旧知识与获取/修订有代价；不以旧保持替代NEW效率。 | U014:NOT_NEXT_ROUND, U015:NOT_NEXT_ROUND |
| M039 | Bayesian/diagonal uncertainty; MESU-inspired gains | 长期仍有条件价值，须区别具体版本 | V23 stable/reopening、V31 N8/32/128/512/inf/frozen/shuffle是不同版本。U034 SUPPORT-DIR退出；V31读取不足的void与observable的inconclusive不能写成整族否定。 | U033:NOT_NEXT_ROUND, U034:STOP_CURRENT_RECIPE |
| M040 | Orthogonal projection / strict/leaky protected subspaces | 长期参照/混合历史证据 | 严格/泄漏保护子空间有保持收益及僵硬代价；投影非局部成本另列。 | 未映射≠淘汰 |
| M041 | Two- and three-state coupled synapses | 长期参照/混合历史证据 | 双/三耦合状态历史结果混合，不能把所有快慢形态归同一个失败。 | 未映射≠淘汰 |
| M042 | Feedback-gated local synaptic coupling | 停止所测具体配方，家族保留边界 | feedback coupling的具体配方未优于普通；暂停该配方，不证明任意反馈回路无用。 | 未映射≠淘汰 |
| M043 | Discrete cascade depth 4/8 | 长期参照/混合历史证据 | cascade4/8已有结果，深度增加并未解决获取/修订代价；不重复做更深即更好。 | 未映射≠淘汰 |
| M044 | Active forgetting / surprise-triggered reopening / capacity release | 长期仍有条件价值，须区别具体版本 | U034支持式方向重开在合格DEV与SUPPORT的NEW J差为0，约1.18倍CPU；关闭本配方。噪声不等于规则失效，勿用惊讶自动擦除旧知识。 | U022:NOT_NEXT_ROUND, U023:NOT_NEXT_ROUND, U025:NOT_NEXT_ROUND, U034:STOP_CURRENT_RECIPE |
| M045 | Sparse distributed address memory and adaptive prototypes | 长期参照/混合历史证据 | 固定分布式地址与自适应prototype已有多容量版本；自适应成本/稳定性未解决。 | U046:NOT_NEXT_ROUND |
| M046 | Allocate/recycle local records | 长期参照/混合历史证据 | 分配/回收有早期保持与后期饱和/初始化问题；扩大容量不等于新规则。 | 未映射≠淘汰 |
| M047 | Quantized allocation / integer precision | 工程组件/对照或信息边界 | V23 quantized不是真正紧凑存储；V25 int8是另一个版本，精度/状态不能混为一谈。 | 未映射≠淘汰 |
| M048 | Separate banks, fixed vs inferred routing, hard/blended reading | 长期仍有条件价值，须区别具体版本 | 固定/推断路由、4/8 banks、hard/blended都已有试验；容量与路由信息须等额比较。 | 未映射≠淘汰 |
| M049 | Silent memories, context reinstatement, unlabeled reminders | 长期仍有条件价值，须区别具体版本 | silent/context固定与推断、reminder已有历史结果；并非所有上下文任务有可靠身份。 | U035:NOT_NEXT_ROUND, U036:NOT_NEXT_ROUND, U037:NOT_NEXT_ROUND, U038:NOT_NEXT_ROUND |
| M050 | Bounded reservoir/recent replay, current repetition, fast-head replay | 长期仍有条件价值，须区别具体版本 | reservoir/recent/fastslow各8/32及当前REPLAY-P/U和COV-REPLAY已测；残差优先并非未测。覆盖型有限再激活是新操作假说，不能原样复活回放。 | U047:PRIORITY_DESIGN |
| M051 | Recent cache retrieval, hard/soft episodic reads | 停止所测具体配方，家族保留边界 | recent hard/soft8/32多为null或很小改善；当前原样addon不优先。 | 未映射≠淘汰 |
| M052 | Spillover, local normalization, fixed-target correction, update-norm controls | 长期参照/混合历史证据 | spillover/归一化/固定target等收益被更新质量对照缩窄；保留控制而非泛化定论。 | 未映射≠淘汰 |
| M053 | Dense Hebbian/Oja encoder adaptation | 长期仍有条件价值，须区别具体版本 | P1/P2及低剂量移植未获得采用证据；活动输入Oja的兼容性新结构与旧配方分开。 | U048:ENABLING_COMPONENT |
| M054 | Novelty/shuffled/norm gates for encoder change | 停止所测具体配方，家族保留边界 | novelty/shuffle/norm specificity结果负面；P3技术不合格不是科学反证。 | 未映射≠淘汰 |
| M055 | Competitive top-response plasticity, random subset, bottom-k | 长期参照/混合历史证据 | top/bottom/random竞争没有已建立frontier胜利；缺陷任务不能用于整族否定。 | 未映射≠淘汰 |
| M056 | Temporal contrast / fast-minus-slow sensory channels | 长期参照/混合历史证据 | fast-minus-slow历史时序任务局部阳性，不足以说明当前加法失误由输入时序造成。 | U049:ENABLING_COMPONENT |
| M057 | Variance salience / coordinate selection | 停止所测具体配方，家族保留边界 | variance salience的旋转/方差对照不支持通用特异性收益，停止旧主张。 | 未映射≠淘汰 |
| M058 | Full covariance/co-activation-directed writes | 长期仍有条件价值，须区别具体版本 | full covariance既有成本/修订限制又有当前COV-R确认E3；不等同V30方向塑形或所有协方差版本。 | 未映射≠淘汰 |
| M059 | Lag96 temporal-position-preserving byte encoding | 工程组件/对照或信息边界 | Lag96工程位置编码是有效读取参照，不是独立生物学习机制。 | 未映射≠淘汰 |
| M060 | Four sparse compartments with local covariance | 停止所测具体配方，家族保留边界 | C4-P局部cov具体配方未增加未教收益；不代替完整COV的结果。 | 未映射≠淘汰 |
| M061 | Shared residual, local residual, absolute local values | 长期仍有条件价值，须区别具体版本 | shared/local residual与absolute读取有不同目标漂移；保存参考/约束含义是长期兼容问题，非单次接口得分。 | U062:ENABLING_COMPONENT |
| M062 | Exact dictionary versus generalizing ownership | 工程组件/对照或信息边界 | exact dictionary能解释E1但没有未教关系相似性；保留工程对照，不作为当前E3核心。 | 未映射≠淘汰 |
| M063 | Read authority: HARD/SHADOW/CORR2/SIM-MIX/fixed alpha/random | 长期仍有条件价值，须区别具体版本 | SIM-MIX/LINK具体读出停止；CORR2/权限控制不是同一操作，简单margin相加曾削弱迁移。 | U039:RECIPE_HOLD, U040:REFERENCE_ONLY, U050:STOP_CURRENT_RECIPE |
| M064 | Predictive Bayes arbitration, slot evidence, shuffled-PBF | 长期参照/混合历史证据 | slot/PBF只部分通过或资格失败；不要将仲裁名称当有完整验证算法。 | U051:NOT_NEXT_ROUND |
| M065 | TRACE-CORR probation and distinct-rendering admission | 长期参照/混合历史证据 | TRACE-CORR有限工程admission收益保留，完整核心门未建立。 | 未映射≠淘汰 |
| M066 | Fixed-cap slots, exact-repeat admission, shuffled-pair admission | 工程组件/对照或信息边界 | 重复/打乱admission是对照；事件身份不是 evaluator可免费提供的信息。 | 未映射≠淘汰 |
| M067 | Bounded superposition, exact hashing, separation ladder, load scaling | 长期参照/混合历史证据 | superposition/exacthash/separation/load有分别结果；哈希E1成功不证明E3。 | 未映射≠淘汰 |
| M068 | Read-only hybrid versus full feedback into admission | 长期参照/混合历史证据 | 只读混合与把结果反馈进admission的行为不同，后者可能破坏持久性。 | 未映射≠淘汰 |
| M069 | Utility-based replacement; LFU-rate/LRU/random; TRACE-U | 因果/资格/统计限制，非整族负面 | V40无效、V41单seed弱效应；utility替换未被科学排除，也未有投入优先理由。 | U052:NOT_NEXT_ROUND |
| M070 | Mute probationary exact traces; supported similarity retention | 长期参照/混合历史证据 | 有限历史audit读收益不足以算新鲜世界确认；保留工程条件。 | 未映射≠淘汰 |
| M071 | Local/global Bayesian authority; zero/E2 init; fixed50; shuffled alpha | 因果/资格/统计限制，非整族负面 | authority信号与primary validity失败同记；不能只挑fixed50阳性称总体通过。 | U053:NOT_NEXT_ROUND |
| M072 | Trace versus mature-slot capacity relief | 工程组件/对照或信息边界 | 增加成熟slot的容量缓解是因果结果，不等于一般学习规则或trace机制通过。 | 未映射≠淘汰 |
| M073 | General/phase normalization, KC expansion, coding-level sweeps | 长期参照/混合历史证据 | 编码分数/扩张/归一化有坐标漂移混淆，不能只用地址距离认领能力。 | 未映射≠淘汰 |
| M074 | PHASE-FROZEN versus PHASE-LIVE versus LAG | 长期参照/混合历史证据 | PHASE-FROZEN实际测过；不可再列为尚未运行的新变体。 | 未映射≠淘汰 |
| M075 | Corroborated distributed synaptic consolidation (Candidate A) | 停止所测具体配方，家族保留边界 | Candidate A小特异性信号而整体弱于现有配置；停止该组合原样采用。 | 未映射≠淘汰 |
| M076 | 32-factor coded output with dense/modular/rewired access (Candidate B) | 工程组件/对照或信息边界 | Candidate B压缩存在收益/代价，未证明解剖拓扑特异性。 | 未映射≠淘汰 |
| M077 | Reusable positive-pair invariance (Candidate C) | 长期仍有条件价值，须区别具体版本 | 正例对有可复用表示作用，但属于合成变换任务；不能把同答案当同对象，也不能凭此认领算术规则。 | U054:RECIPE_HOLD |
| M078 | Invariance plus global slow consolidation factorial | 停止所测具体配方，家族保留边界 | invariance+consolidation小信号未达固定门，相关扩展资格不足；不能无限靠量级挽救旧配方。 | U054:RECIPE_HOLD |
| M079 | Learned overlapping pathway responsibility | 停止所测具体配方，家族保留边界 | 重叠pathway责任分配具体架构有串扰，停止该配方。 | 未映射≠淘汰 |
| M080 | Signed protection within PAIR nuisance subspace | 停止所测具体配方，家族保留边界 | PAIR子空间signed保护具体方向损害几何且shuffle无特异性，停止该配方。 | 未映射≠淘汰 |
| M081 | One-byte surprise filter on negative development | 停止所测具体配方，家族保留边界 | one-byte surprise具体门没有新增价值，不解释为所有因果quota失败。 | 未映射≠淘汰 |
| M082 | Exact sparse encoding of local output values | 工程组件/对照或信息边界 | 精确局部值压缩单测成立；不是整个核心功能/算力可扩展证明。 | 未映射≠淘汰 |
| M083 | Full graph reduction and core peeling | 工程组件/对照或信息边界 | 图peeling结构统计不是功能消融的成功，不计能力机制。 | 未映射≠淘汰 |
| M084 | Direct/indirect MBON→DAN relay distillation | 工程组件/对照或信息边界 | MBON→DAN relay低秩蒸馏为结构结果；弱目的通道及整体损失仍未解决。 | 未映射≠淘汰 |
| M085 | Unsigned composed KC→MBON→DAN→KC spatial gate | 停止所测具体配方，家族保留边界 | unsigned composed线性空间门具体抽象负面，不是对真实回路的完整功能否定。 | 未映射≠淘汰 |
| M086 | Output gain modulation; exact/compressed/shuffled anatomy | 停止所测具体配方，家族保留边界 | 多输出gain配方小seed任务null，关闭所测标量增益操作。 | 未映射≠淘汰 |
| M087 | Input-only anatomical write addressing | 停止所测具体配方，家族保留边界 | input-only静态write mask未建立优势，不能推广为全部空间教学无效。 | 未映射≠淘汰 |
| M088 | Joint input/readout addressing and scrambled control | 停止所测具体配方，家族保留边界 | joint static input/read mask无广泛优势，真实可塑compartment未因此排除。 | 未映射≠淘汰 |
| M089 | KC-family/lobe bins, whole-cell overlap compartment candidates | 工程组件/对照或信息边界 | KC family/lobe bin互相重叠，不能当已经验证的独立生理compartments。 | 未映射≠淘汰 |
| M090 | Raw KC↔APL scalar suppressive controller | 停止所测具体配方，家族保留边界 | 标量APL匹配活动控制无干扰/保持增量；更复杂因果抑制尚非同一操作。 | 未映射≠淘汰 |
| M091 | Aligned raw PN→KC→MBON plastic mask | 长期参照/混合历史证据 | raw PN→KC→MBON学得当前但旧知识差，未建立raw superiority；不能因此否定整个connectome。 | 未映射≠淘汰 |
| M092 | Recorded-count heterogeneity, binary support, count shuffle | 工程组件/对照或信息边界 | count/binary/shuffle是特定强度分配对照，不能从计数直接推出塑性符号。 | 未映射≠淘汰 |
| M093 | Unsigned linear raw MBON recurrence at radius0.25 | 停止所测具体配方，家族保留边界 | 半径.25 unsigned MBON递归具体配方有延迟损失且无效率优势；不等于任何attractor失败。 | 未映射≠淘汰 |
| M094 | KC↔KC weak recurrence, DPM, broad modulatory feedback | 长期仍有条件价值，须区别具体版本 | KC/DPM等弱递归功能试验不完整；U055泛化联合重估不是已建立生物回路。 | U028:NOT_NEXT_ROUND, U055:CONDITIONAL_CANDIDATE |
| M095 | Full raw-graph functional reference followed by lesion/distillation | 完整结果/比较缺失，不能判死 | 完整rawgraph functional reference未有恢复的本地成功；它是候选参照，不强制重新回166k大图。 | 未映射≠淘汰 |
| M096 | Bidirectional/local DAN teaching with delayed eligibility | 长期仍有条件价值，须区别具体版本 | 当前普通到来byte可驱动学习的E0/E1基线有效；完整local DAN/延迟/双向回路尚未全部通过。 | U018:ADOPTED_BASELINE, U019:ENABLING_COMPONENT, U027:RECIPE_HOLD |
| M097 | Two-unit recurrent STM→LTM teaching (Fly-2) | 长期参照/混合历史证据 | Fly-2 STM→LTM教学为有来源提案，与旧耦合快慢不自动等同。 | U056:NOT_NEXT_ROUND |
| M098 | Incremental long-vs-short information gate | 长期仍有条件价值，须区别具体版本 | long-vs-short信息门未科学运行；需真实因果两预测器，不得看未来质量。 | U057:CONDITIONAL_CANDIDATE |
| M099 | Multi-horizon residual hierarchy | 长期仍有条件价值，须区别具体版本 | 多个horizon有提案但无完整通过实现；不能自动视为更高信息收益。 | U058:CONDITIONAL_CANDIDATE |
| M100 | Fixed plasticity quota; competition across time | 长期仍有条件价值，须区别具体版本 | 因果quota可作不同预算分配假说；acausal窗口版本不可采用，仅省写次数不是少样本泛化。 | U059:CONDITIONAL_CANDIDATE |
| M101 | Homeostatic forgetting / targeted bidirectional extinction | 长期参照/混合历史证据 | extinction/主动遗忘有局部proxy，真实反馈/修订仍开放；固定加法无规则变更不能用来选强遗忘。 | U026:NOT_NEXT_ROUND |
| M102 | Persistent scratchpad and feedback memory | 长期参照/混合历史证据 | scratchpad/tape是工作状态，不直接等于可迁移规则；完整生命周期结果缺失。 | U060:NOT_NEXT_ROUND |
| M103 | Short-context/phase+2 slot admission | 长期参照/混合历史证据 | phase+2 admission单seed是局部diagnostic，不是居民效用或规则复用确认。 | 未映射≠淘汰 |
| M104 | Class-specific write lesions on V49 | 工程组件/对照或信息边界 | class-masked lesion是oracle因果诊断，不能部署为自主学习机制。 | 未映射≠淘汰 |
| M105 | Positive-and-negative reusable metric outside protected subspace | 长期仍有条件价值，须区别具体版本 | 一个protected exact metric失败不否定所有复用metric；正负关系必须由实际观察构造，不能把标签相同偷换cue相同。 | U061:NOT_NEXT_ROUND |
| M106 | Frozen/slow residual reference, transport compensation, constraint-only slow memory | 长期仍有条件价值，须区别具体版本 | 冻结参考、transport和constraint-only slow有相关历史负面及未实现操作；与当前残差含义兼容有关，不用U034重开替代。 | U062:ENABLING_COMPONENT |
| M107 | Indexed retrieval (hash/tree/HNSW/ANN), exact value compaction | 工程组件/对照或信息边界 | 索引/压缩属于成本工程，可能长期必要，但不增加学习信息或当前NEW样本曲线。 | 未映射≠淘汰 |
| M108 | Natural byte corpora, varied wrappers, nonshared transforms | 长期参照/混合历史证据 | 自然文本/开放wrappers为后续范围；现阶段不移走有限规则学习目标。 | 未映射≠淘汰 |

## M001 — Stateful nonlinear ELM cell (fast s, slow m, recurrent messages, internal MLP)

原位置：ELM2/3/4; byte-ELM sources; scalar temporal runners

原证据状态：IMPLEMENTED; PARTIAL TASK EVIDENCE

原发现：Real stateful recurrent arithmetic, not a conventional frozen random-feature ELM; some earlier training used BPTT and larger vector-valued units.

原处置：Retain as an expressive cell reference; do not equate cell count with scalar state or claim autonomous local training established.

本次注记：表达能力参照；部分任务依赖BPTT/向量私有状态，不能计为已验证自主持续学习。

## M002 — Scalar-channel ELM with private vector memory

原位置：P4/P6; temporal_elm_core; v8–v25 audits

原证据状态：IMPLEMENTED; COMPARATIVE COVERAGE INCOMPLETE

原发现：A scalar communication channel can conceal several private coordinates and a nonlinear internal map. Temporal-task evidence exists, but the complete six-cell P4 result roster was not recovered.

原处置：Keep independent private state; recover missing comparison before declaring cell arithmetic unnecessary.

本次注记：完整六细胞比较未恢复；缺结果不等于排除，保持与标量通道不同的私有状态记账。

## M003 — Fixed-time-constant ELM

原位置：P4 fixed_tau

原证据状态：SHIPPED; COMPLETE RESULT NOT RECOVERED

原发现：Directly listed in the six-model same-graph ablation. No decisive completed result was located in this audit.

原处置：Unresolved, not pruned experimentally.

本次注记：已实现/部分日志；没有足够完整对照支持删除全部时间常数方案。

## M004 — Linear-dendrite ELM

原位置：P4 linear_dendrite

原证据状态：SHIPPED; COMPLETE RESULT NOT RECOVERED

原发现：Removes dendritic tanh while retaining the internal ELM map; not equivalent to an entirely linear cell.

原处置：Unresolved; retain exact definition.

本次注记：线性树突版本仍含内部MLP；完整比较缺失，不能称已验证纯线性核心。

## M005 — Simple scalar leaky recurrent cell

原位置：P4 leaky; P6 SHD/N-Omniglot

原证据状态：IMPLEMENTED/SHIPPED; PARTIAL HISTORICAL EVIDENCE

原发现：An intended simpler same-graph temporal baseline, not a non-temporal rate-only baseline.

原处置：Use as the smallest interpretable stateful reference, not an already demonstrated substitute in every task.

本次注记：最简单标量动态参照；结构简单不自动代表低样本复用强。

## M006 — Adaptive-threshold leaky cell (ALIF-labelled)

原位置：P4 alif

原证据状态：SHIPPED; COMPLETE RESULT NOT RECOVERED

原发现：Adaptive-threshold control listed in the cell-ablation package. Its name alone does not establish a biophysical spiking implementation or successful continual learning.

原处置：Unresolved; audit the exact cell before biological claims.

本次注记：自适应阈值已有实现，但缺完整持续学习比较；不能只靠ALIF名称判优。

## M007 — Scalar sparse GRU-like cell

原位置：P4 sparse_gru

原证据状态：SHIPPED; COMPLETE RESULT NOT RECOVERED

原发现：Same 64-cell sparse source graph and presentations were intended; parameter counts unequal and explicitly reported.

原处置：Unresolved; do not invent a ranking against ELM.

本次注记：同图GRU样实现参数量不同；缺完整结果，作为表达/成本参照。

## M008 — RG-LRU / simplified Griffin recurrent model

原位置：ELM5.ipynb

原证据状态：IMPLEMENTED; LIMITED TASK LOGS

原发现：Actual gated diagonal recurrent update exists; the notebook reports spike-prediction holdout AUC values. It is not a retention-through-intervening-learning test.

原处置：Historical engineering comparison, not a persistent-learning result. PickleLRU in this notebook is a file cache, not another neuron model.

本次注记：门控递归有任务信号，但不是已确认持续复用或加法样本效率。

## M009 — FLM fast/slow leaky cell with sparse punctuator

原位置：FLM model.ipynb

原证据状态：IMPLEMENTED; LANGUAGE/THROUGHPUT LOGS

原发现：Separate fast/slow time constants and internal MLP; sparse punctuator adds retrieval. Saved comparison has NaN Transformer perplexities, so it cannot establish superiority to that baseline.

原处置：Keep implementation evidence; no claim of reliable continual-learning or valid Transformer superiority.

本次注记：语言/吞吐日志的Transformer比较存在NaN；不能用无效比较支持核心提升。

## M010 — Multi-timescale private neuronal state

原位置：Early temporal roadmap; ELM/FLM; v25/v26 traces

原证据状态：PARTIAL POSITIVE

原发现：Multiple temporal coordinates are distinct from multiple independently stored output banks. Temporal contrast helped controlled sensory tasks.

原处置：Retain as a cell-level hypothesis; count state and do not relabel banks as neuron timescales.

本次注记：保留私有多时间尺度方向；它与复制多个完整银行不是同一个操作。

## M011 — Stubborn/adaptable heterogeneous populations

原位置：ELM16 onward; v8–v17 audits

原证据状态：PARTIAL POSITIVE; SPECIFICITY NARROWED

原发现：12 matched route/seed/order pairs gave 11 wins, about +5.4 pp all-class; later uniform/mean-rate controls recovered much of the gain. Gate affects activity updates, not directly synaptic allocation.

原处置：Keep heterogeneity as unresolved useful component; reject claims that it already solved selective long-term plasticity.

本次注记：历史改善被学习率/活动门对照缩窄；没有独立异质性特异性胜利。

## M012 — Separate write and forget gates

原位置：ELM9/11 and later notebook family; local-learning audit

原证据状态：IMPLEMENTED; NOT ISOLATED DECISIVELY

原发现：Suppressing a new write need not erase old state; learned gate networks have their own training/information costs.

原处置：Retain distinction; not equivalent to reducing all learning rates.

本次注记：当前只采用U042的SUPPORT写门实例；低样本提升未确认，不代表独立遗忘门已通过。

## M013 — E/I signs, proper log-tau, homeostasis, blank-activity penalty, delay training

原位置：P5 dynamics_repair_bench and interference_memory

原证据状态：PARTLY IMPLEMENTED; AUDIT FLAGS

原发现：Intended progressive interventions address autonomous blank-period activity. Historical audit reports dispatch issues: intended variants were not all actually exercised.

原处置：Do not treat all physiological-dynamics controls as completed negatives.

本次注记：保留真实参数与实际dispatch范围；多个flags存在不等于全部变体科学运行过。

## M014 — Raw-node spiking LIF / richer conductance cells

原位置：Digital-fly discussion and proposed reference organism

原证据状态：EXTERNAL IMPLEMENTATIONS / PROPOSED LOCALLY

原发现：No completed, same-raw-core local LIF-versus-leaky-versus-ELM retention comparison was recovered. No recovered HH/Izhikevich result either.

原处置：Not pruned. Do not add sophisticated cells merely for realism; first state exactly what dynamics are missing.

本次注记：原始脉冲/电导细胞缺同底物完整对照；不为生物复杂性本身追加实验。

## M015 — Type-shared templates versus merged neurons

原位置：Raw-core and local-learning audits

原证据状态：DESIGN CONSTRAINT

原发现：Fixed type-level parameter templates may be shared; independent root IDs and mutable states must not be silently pooled.

原处置：Preserve neuron identity; distinguish structural compression from demonstrated dynamical equivalence.

本次注记：类型共享参数不得合并各神经元可变状态；是结构约束，不是已胜出学习操作。

## M016 — Fixed sparse sensory/recurrent graphs and retinotopic prior

原位置：P0–P3; ELM; fly scaffolds

原证据状态：IMPLEMENTED; REFERENCE

原发现：Useful control shared by many experiments; later matrices and graph blocks have different populations and resource costs.

原处置：Retain stable reference topology with explicit provenance.

本次注记：固定稀疏图可作底物/对照；不能将静态拓扑或输入先验单独算新能力。

## M017 — Prune-only / random-regrowth / guided-regrowth

原位置：Fashion-MNIST P1/P2; original developmental code

原证据状态：IMPLEMENTED; CONTROL/AUDIT LIMITATIONS

原发现：Guided growth/pruning consult gradient-derived scores; early equal-label comparisons need presentation accounting.

原处置：Exact global-gradient method outside strict local-learning candidate; local structural plasticity not disproved.

本次注记：当前TOPO-G已经联合学权重与导向换边，DEV未晋级；不同P×S操作仍须明示区别。

## M018 — New-edge lesion and topology-only transfer

原位置：P0/P3

原证据状态：SHIPPED; FULL OUTCOMES NOT RECOVERED

原发现：Knock out grown edges; transfer only source_idx with weights/cell parameters reset. Designed to distinguish topology from co-adapted weights.

原处置：Unresolved where logs missing; not a proven developmental prior.

本次注记：新边消融/仅拓扑迁移完整结果未恢复；不可用缺失当负面。

## M019 — Global cosine-kNN graph rewiring, hubs, EMA routing

原位置：ELM2/3/4; byte ELM

原证据状态：IMPLEMENTED; CAUSALITY ISSUES IN AUDIT

原发现：Chunk-level future-information and evaluation-state mutation were flagged in early versions; EMA hub persistence was not a synaptic-retention demonstration.

原处置：Exact flawed versions pruned; local message passing/hubs not ruled out.

本次注记：早期global-kNN版本有未来信息或探测改状态问题；先标因果不合格，不能当家族科学负面。

## M020 — Local sensory-only development

原位置：V24 E0/E1/E2/E3; V25/V26

原证据状态：TESTED; PARTIAL POSITIVE

原发现：Separated clean-target BP, observed-target BP, random fixed, and local Hebbian/Oja regimes. Banked behavior survives local-development variants on tested streams.

原处置：Retain result with objective and information differences; no universal local-optimizer equivalence.

本次注记：局部感觉开发有局限阳性；观察目标与干净latent目标、Hebb/Oja与BP分别记账。

## M021 — Clean latent reconstruction target

原位置：V23 encoder development; V24 E0/E1 audit

原证据状态：IDENTIFIED PRIVILEGE

原发现：Clean denoising targets supplied information beyond the observed noisy sensory stream; not downstream task-label leakage, but must be charged.

原处置：Historical reproduction only for fully local/no-privileged-teacher claims.

本次注记：干净latent重建含额外信息；其成功不能证明仅凭实际噪声答案的学习。

## M022 — Frozen, live, split-stable encoders; FF/FL/LF/LL address/content states

原位置：V11–17; V25–29

原证据状态：TESTED; TRADEOFF

原发现：Moving features destabilize stored coordinates; stable address or split state helps some contrasts but no general continuing-representation solution.

原处置：Retain as compatibility evidence, not proof to freeze representations forever.

本次注记：FF/FL/LF/LL及冻结/活动/稳定split都已有具体版本；当前角色输入固定，不把旧结构重命名成未测。

## M023 — Dense affine/readout transport and anchor transport

原位置：V11–13

原证据状态：TESTED; PARTIAL/NEGATIVE

原发现：Transported and anchor methods did not recover oracle refit competence; v13 about 38.33% all-20, only part of gap closed.

原处置：Keep as nonlocal diagnostics; no solved local coordinate transport.

本次注记：anchor/仿射运输是输入坐标兼容参照，成本与非局部性有界性待明示。

## M024 — Learned down-stream adapter/recycling

原位置：V23 adapter_retained/recycled

原证据状态：TESTED; LOCALITY DISQUALIFICATION

原发现：Uses V@error and normalization derivatives: handwritten hidden-layer backprop despite NumPy implementation.

原处置：Exact candidates excluded by no-hidden-backprop constraint, not by zero numerical benefit.

本次注记：adapter采用隐藏BP的版本局部性不合格，不是数值上等同失败。

## M025 — Learned operator mixtures / neural algebra / global group routing

原位置：Liquid Baby, Genesis, ELM8–25

原证据状态：IMPLEMENTED; COMPOSITE DEMOS

原发现：Operation mixtures, intra/intergroup messages and routing bundled with multiple training mechanisms.

原处置：Not independently validated persistent-learning mechanisms.

本次注记：算子混合/神经代数是复合demo；没有隔离出持续核心增量。

## M026 — Sinkhorn routing, attention hubs, global workspace, token MoE

原位置：ELM4/8–24; Genesis family

原证据状态：IMPLEMENTED; OUTSIDE CURRENT MAINLINE

原发现：Global differentiable routing and attention often present; no valid inference that removing autograd makes communication local.

原处置：Park complete architectures; retain bounded local communication hypotheses only with explicit signals.

本次注记：注意力/Sinkhorn等全局路由属于不同范围；未排除，但不据名称自动排期。

## M027 — Hebbian key/value fast matrices; residual/predictive Hebbian writes

原位置：ELM8–25; V23 fastweights_*

原证据状态：IMPLEMENTED; MIXED

原发现：Actual online outer products exist; learned q/k/v and global norm clipping may be nonlocal. V23 exact versions did not solve retention/revision jointly.

原处置：Keep family and matched controls; no broad success claim.

本次注记：当前COV-R为已确认的有限工程加法E3锚点；旧Hebb快矩阵与完整RLS不是同一实现。

## M028 — Modern Hopfield memory and episodic tape

原位置：ELM17–19; TensorProductHippocampus ELM20–24

原证据状态：IMPLEMENTED; WORKING-MEMORY DEMOS

原发现：Retrieves context/tape contents. Reset-per-context and selected-rollout reporting do not test knowledge surviving later parameter updates.

原处置：Engineering reference only; explicit episodic store not default biological core.

本次注记：Hopfield/episode tape的上下文重置演示不等于跨生命保留；检索成本与存储另记。

## M029 — Gauge/fiber semantic–surface factorization; contextual router

原位置：ELM25; V75/V78–79; Genesis derivatives

原证据状态：IMPLEMENTED; COMPOSITE/SUPERVISED

原发现：Supervised Transformer-containing merge exists; no clean autonomous local invariance result from it.

原处置：Conceptual precedent, not evidence for a biology-based learned semantic space.

本次注记：semantic/surface分解的监督复合结果不能冒充自主学得的不变性。

## M030 — SAC/proposal policies, learned teacher, dialogue warmup, curiosity/ASK actions

原位置：Liquid Baby/Genesis; ELM8–25

原证据状态：IMPLEMENTED; CONFOUNDED FOR CURRENT GOAL

原发现：Teacher language/pretrained weights, task rewards and contextual memory can generate attractive episodes without lifetime retention.

原处置：Do not import pretrained/Transformer teacher gains into local core evidence.

本次注记：奖励/预训练/教师先验与本核心获取信息预算不同，先保留参照。

## M031 — MagmaAdamW/global optimizers; neuromorphic detach mode

原位置：ELM12–16; Genesis

原证据状态：IMPLEMENTED; LOCALITY LIMIT

原发现：Detaching states removes some temporal gradient paths, not all hidden training or all nonlocal signals. Dopamine/STDP log labels are not mechanistic proof.

原处置：Retain only honest component accounting; no claim entire model biologically local.

本次注记：全局优化器/隐藏梯度改变学习信息与成本；局部detach名称不自动消除这些差别。

## M032 — Chunking, parallel Hebbian scan, sparse punctuator, scaling optimizations

原位置：Scaling.ipynb; FLM; Genesis

原证据状态：IMPLEMENTED; ENGINEERING

原发现：Speed and memory-throughput measurements do not establish sample-efficient persistent learning.

原处置：Keep engineering techniques where causal; do not rank neurons from unequal composite benchmarks.

本次注记：吞吐/scan/分块是工程实现收益；不作为新增泛化机制。

## M033 — Current-only classifier / ordinary normalized delta

原位置：Early SHD; V20 onward

原证据状态：TESTED BASELINE

原发现：Can erase old decision competence with frozen features. Useful simple reference throughout.

原处置：Keep baseline and distinguish output competition from erased input representation.

本次注记：普通delta是有用对照；其历史遗忘不排除所有低成本更新。

## M034 — Cumulative ridge sufficient statistics / exact RLS

原位置：Early v17; V20/23; V35

原证据状态：POSITIVE WITH LIMITS

原发现：Fixed-feature history can be retained: V35 H-RLS A-after-D 95.31%. Revision/safety failed; full inverses/statistics nonlocal and larger state.

原处置：Historical reference, not a failed memory family and not a local-biological primitive.

本次注记：历史ridge/RLS有保持阳性，也有修订/安全失败；与当前COV-R已包含的相关统计分开。可讨论的新操作必须不是再实现现有RLS。

## M035 — Slow-only versus combined fast+slow output

原位置：V18/V19; V40–43

原证据状态：TESTED; COMPOSITION MATTERS

原发现：V18 combined read favored recent classes and underperformed its slow-only read. Later read-only hybrid improved engineering behavior.

原处置：Do not assume two good stores add into a good reader automatically.

本次注记：slow-only与fast+slow的具体读出曾不同；简单相加可能稀释信号，不默认双路即更好。

## M036 — Fast-feedback weighted ridge consolidation

原位置：V18

原证据状态：SMALL SPECIFIC EFFECT; BELOW BASELINE

原发现：Old-group gains over shuffled/mean-write existed, but all-class 37.88% versus v17 48.37%. Fixed-feature weighted sums do not isolate a timing mechanism.

原处置：Exact construction closed as an improvement; biological recurrent teaching remains untested by this comparison.

本次注记：精确反馈巩固配方有小特异性信号但总体低于父基线；停止原样采用，不排除所有巩固。

## M037 — Additive consolidation bonus

原位置：V19

原证据状态：NEGATIVE EXACT

原发现：Extra write mass and doubled statistics did not improve the cumulative baseline; all-class 46.93%.

原处置：Do not relaunch same additive ridge bonus.

本次注记：加法式consolidation bonus未建立增量且增加统计状态；停止该具体配方。

## M038 — Synaptic importance / neuron availability / protection

原位置：V23; V24–25

原证据状态：PARTIAL POSITIVE; RIGIDITY

原发现：Pair/neuron protection improved old retention; stronger protection impaired new acquisition and revision. Per-neuron version nearly matched per-synapse at less state.

原处置：Keep evidence and rigidity tradeoff; no blanket retirement of metaplasticity.

本次注记：importance/availability多版本保护旧知识与获取/修订有代价；不以旧保持替代NEW效率。

## M039 — Bayesian/diagonal uncertainty; MESU-inspired gains

原位置：V23 bayes_*; V31 independent run

原证据状态：PARTIAL THEN INCONCLUSIVE/NEGATIVE IMPLEMENTATIONS

原发现：V23 bayes_stable terminal47.64% versus36.66% ordinary; later adequate-task and frozen/shuffled-variance checks did not establish complete mechanism.

原处置：Do not conflate approximation with exact MESU or declare family disproved.

本次注记：V23 stable/reopening、V31 N8/32/128/512/inf/frozen/shuffle是不同版本。U034 SUPPORT-DIR退出；V31读取不足的void与observable的inconclusive不能写成整族否定。

## M040 — Orthogonal projection / strict/leaky protected subspaces

原位置：V23 projection16/48; earlier transport

原证据状态：TESTED TRADEOFF

原发现：Protects some old directions but larger protection impedes acquisition/revision; exact Gram–Schmidt/projection nonlocal.

原处置：Reference/closed exact local candidate; not proof inhibition-based separation impossible.

本次注记：严格/泄漏保护子空间有保持收益及僵硬代价；投影非局部成本另列。

## M041 — Two- and three-state coupled synapses

原位置：V23 fast_slow2/3

原证据状态：TESTED NEGATIVE SCREEN

原发现：Real bidirectional local substate coupling, not only two rates. Terminal38.43/35.61 versus ordinary36.66 with acquisition/revision issues.

原处置：Archive exact parameters; do not call coupled synapses untried or biologically refuted.

本次注记：双/三耦合状态历史结果混合，不能把所有快慢形态归同一个失败。

## M042 — Feedback-gated local synaptic coupling

原位置：V23 feedback_consolidation

原证据状态：TESTED NEGATIVE SCREEN

原发现：Terminal36.45%; no complete advantage. Surprise-modulated coupling is not delayed channel-specific reinforcement.

原处置：Do not rename as new without a changed observable operation.

本次注记：feedback coupling的具体配方未优于普通；暂停该配方，不证明任意反馈回路无用。

## M043 — Discrete cascade depth 4/8

原位置：V23 cascade4/8

原证据状态：TESTED TRADEOFF/NEGATIVE

原发现：Larger depth improved initial old recall but hurt new learning/revision; terminal35.42/25.38.

原处置：Close exact variants; retain family uncertainty.

本次注记：cascade4/8已有结果，深度增加并未解决获取/修订代价；不重复做更深即更好。

## M044 — Active forgetting / surprise-triggered reopening / capacity release

原位置：V23; V24–25; older V75

原证据状态：TESTED MIXED/NEGATIVE

原发现：Surprise-based release can damage useful memories; global decay and immediate/smoothed surprise did not supply broad improvement.

原处置：Noise, novelty and revision must be separated; no blanket equation surprise=obsolete.

本次注记：U034支持式方向重开在合格DEV与SUPPORT的NEW J差为0，约1.18倍CPU；关闭本配方。噪声不等于规则失效，勿用惊讶自动擦除旧知识。

## M045 — Sparse distributed address memory and adaptive prototypes

原位置：V23 sparse96/384_*

原证据状态：TESTED; ADAPTATION COST

原发现：Fixed larger address layer helped; adaptive versions weak, with much larger parameter state.

原处置：Do not treat as exact SDM theorem or whole-family rejection.

本次注记：固定分布式地址与自适应prototype已有多容量版本；自适应成本/稳定性未解决。

## M046 — Allocate/recycle local records

原位置：V23 allocate32/128/no_recycle

原证据状态：TESTED; SATURATION BIAS

原发现：Very high early retention can coexist with failed late acquisition. Original victim-selection/initialization flaws narrowed conclusions.

原处置：Engineering comparator; not evidence slots are mandatory or allocation generally impossible.

本次注记：分配/回收有早期保持与后期饱和/初始化问题；扩大容量不等于新规则。

## M047 — Quantized allocation / integer precision

原位置：V23 quantized_allocate128; V25 int8

原证据状态：TESTED; ACCOUNTING CAVEAT

原发现：V23 labelled quantized arm used doubled dense address storage and was not genuine total compression; V25 narrower precision tests separate.

原处置：Retain only measured state/quality tradeoffs, not label-based memory savings.

本次注记：V23 quantized不是真正紧凑存储；V25 int8是另一个版本，精度/状态不能混为一谈。

## M048 — Separate banks, fixed vs inferred routing, hard/blended reading

原位置：V23 silent/context; V24–25

原证据状态：PARTIAL POSITIVE REPLICATED

原发现：8-bank fixed route around59.09% terminal in V23 versus36.66 ordinary; later controls separate banking, routing and recent-access prior.

原处置：Useful selective-storage component; not independent proof of biological compartments or learned routing necessity.

本次注记：固定/推断路由、4/8 banks、hard/blended都已有试验；容量与路由信息须等额比较。

## M049 — Silent memories, context reinstatement, unlabeled reminders

原位置：V23 silent/context_*

原证据状态：PROMISING SCREEN; ASSAY LIMIT

原发现：Stored banks worked, but cohorts lacked guaranteed distinguishable context; reminder results do not establish silent-memory recovery mechanism.

原处置：Keep unresolved; measure recovery before answer re-exposure and false reinstatement.

本次注记：silent/context固定与推断、reminder已有历史结果；并非所有上下文任务有可靠身份。

## M050 — Bounded reservoir/recent replay, current repetition, fast-head replay

原位置：V14/14.1; V23 replay_*/repeat_current/fastslow_replay_*

原证据状态：TESTED MIXED; NOT CURRENT CANDIDATE

原发现：Some screen retention gains, serious moving-representation and stale-revision failures. All extra presentations must count.

原处置：Retain as historical controls; do not silently introduce replay or declare all biological reactivation invalid.

本次注记：reservoir/recent/fastslow各8/32及当前REPLAY-P/U和COV-REPLAY已测；残差优先并非未测。覆盖型有限再激活是新操作假说，不能原样复活回放。

## M051 — Recent cache retrieval, hard/soft episodic reads

原位置：V23 retrieve_*; early tapes

原证据状态：TESTED MOSTLY NULL

原发现：Small recent caches mostly reproduced ordinary endpoints; 32-entry variants slightly different.

原处置：Reject exact add-on claim, not retrieval generally.

本次注记：recent hard/soft8/32多为null或很小改善；当前原样addon不优先。

## M052 — Spillover, local normalization, fixed-target correction, update-norm controls

原位置：V24–25

原证据状态：TESTED; BENEFIT NARROWED

原发现：Same-array spillover alters write mass; apparent early winner did not establish generic benefit once norm/current-correction effects separated.

原处置：Do not count an unmatched write-scale gain as anatomical transfer.

本次注记：spillover/归一化/固定target等收益被更新质量对照缩窄；保留控制而非泛化定论。

## M053 — Dense Hebbian/Oja encoder adaptation

原位置：V24–29

原证据状态：TESTED TRADEOFF

原发现：Useful adaptation can damage stored representation coordinates; headroom and genuine recall needed together.

原处置：Continue only within a meaningful acquisition–retention assay.

本次注记：P1/P2及低剂量移植未获得采用证据；活动输入Oja的兼容性新结构与旧配方分开。

## M054 — Novelty/shuffled/norm gates for encoder change

原位置：V26

原证据状态：NEGATIVE SPECIFICITY

原发现：Random/rate-matched gating explained the apparent compatibility effect.

原处置：Close exact selection heuristic; not all event-level teaching ever ruled out.

本次注记：novelty/shuffle/norm specificity结果负面；P3技术不合格不是科学反证。

## M055 — Competitive top-response plasticity, random subset, bottom-k

原位置：V26–28

原证据状态：NO ESTABLISHED FRONTIER WIN; TASK ISSUES

原发现：Matched sparse/budget controls undermined competition. V27 implementations had opposite task-validity defects.

原处置：Close current claim, not biological competition as a family.

本次注记：top/bottom/random竞争没有已建立frontier胜利；缺陷任务不能用于整族否定。

## M056 — Temporal contrast / fast-minus-slow sensory channels

原位置：V26–29

原证据状态：PARTIAL POSITIVE

原发现：Survived several structured/permuted/distributed sensory controls; not a universal byte encoder.

原处置：Retain sensory mechanism; its transfer to discrete byte windows must be tested separately.

本次注记：fast-minus-slow历史时序任务局部阳性，不足以说明当前加法失误由输入时序造成。

## M057 — Variance salience / coordinate selection

原位置：V26–27

原证据状态：TASK-SPECIFIC

原发现：Benefits failed rotation/variance-matched controls; permutation alone was insufficient.

原处置：Retire generic feature-learning claim.

本次注记：variance salience的旋转/方差对照不支持通用特异性收益，停止旧主张。

## M058 — Full covariance/co-activation-directed writes

原位置：V30; V31 revision stream; V33/34

原证据状态：POSITIVE WITH COST/REVISION LIMITS

原发现：V30 overlap retention advantage about15.89pp with near-neutral transfer; quadratic statistics and revision cost. Later variants differ.

原处置：Keep precise positive evidence; not a compact local core yet.

本次注记：full covariance既有成本/修订限制又有当前COV-R确认E3；不等同V30方向塑形或所有协方差版本。

## M059 — Lag96 temporal-position-preserving byte encoding

原位置：V31–35 onward

原证据状态：TASK-USEFUL

原发现：Byte contrast encoder could be inadequate; lag-position coding exposed learnable signal.

原处置：Useful controlled byte interface; not a biologically inferred sensory code.

本次注记：Lag96工程位置编码是有效读取参照，不是独立生物学习机制。

## M060 — Four sparse compartments with local covariance

原位置：V34

原证据状态：TESTED NEGATIVE

原发现：C4-P acquired A/D but untouched-A retention after D about3.125%, matching ordinary compartment result.

原处置：Not an untried idea; exact sparse-compartment covariance construction closed.

本次注记：C4-P局部cov具体配方未增加未教收益；不代替完整COV的结果。

## M061 — Shared residual, local residual, absolute local values

原位置：V35

原证据状态：STRONG ENGINEERING DECOMPOSITION

原发现：Local residual retained29.69% versus shared7.03%; absolute local values99.22%. Stored residual semantics drift when reference predictor changes.

原处置：Retain finding; fixed encoder does not make a residual reference predictor stationary.

本次注记：shared/local residual与absolute读取有不同目标漂移；保存参考/约束含义是长期兼容问题，非单次接口得分。

## M062 — Exact dictionary versus generalizing ownership

原位置：V35–36

原证据状态：POSITIVE ENGINEERING, UNBOUNDED

原发现：Exact record memory retains exact forms but not unseen aliases; generalized ownership joins aliases under controls.

原处置：Baseline/diagnostic, not default biology or unlimited-capacity solution.

本次注记：exact dictionary能解释E1但没有未教关系相似性；保留工程对照，不作为当前E3核心。

## M063 — Read authority: HARD/SHADOW/CORR2/SIM-MIX/fixed alpha/random

原位置：V37

原证据状态：POSITIVE ENGINEERING

原发现：Hard retrieved values damage ordinary prediction; corroborated authority/fixed mixtures trade reading safely on shared histories.

原处置：Retain as read-versus-write lesson; not a new synaptic circuit.

本次注记：SIM-MIX/LINK具体读出停止；CORR2/权限控制不是同一操作，简单margin相加曾削弱迁移。

## M064 — Predictive Bayes arbitration, slot evidence, shuffled-PBF

原位置：V38

原证据状态：PARTIAL; SPECIFICITY GATE FAIL

原发现：Prediction gains existed but full planned PBF claim failed Q10; exact read histories shared.

原处置：No universal evidence arbiter established.

本次注记：slot/PBF只部分通过或资格失败；不要将仲裁名称当有完整验证算法。

## M065 — TRACE-CORR probation and distinct-rendering admission

原位置：V39 Codex

原证据状态：USEFUL BOUNDED ENGINEERING

原发现：Reduced clutter-driven permanent storage by about11x, near retained quality; formal all-gates verdict not established.

原处置：Retain benchmark and evidence principle; object admission is not a biological mechanism by itself.

本次注记：TRACE-CORR有限工程admission收益保留，完整核心门未建立。

## M066 — Fixed-cap slots, exact-repeat admission, shuffled-pair admission

原位置：V39 Codex

原证据状态：TESTED CONTROLS

原发现：Simple cap and repeated identical contexts did not match distinct corroboration under clutter.

原处置：Shows event/identity semantics mattered in that explicit-store system.

本次注记：重复/打乱admission是对照；事件身份不是 evaluator可免费提供的信息。

## M067 — Bounded superposition, exact hashing, separation ladder, load scaling

原位置：V39 Claude

原证据状态：PARTIAL POSITIVE; CAPACITY LOSS

原发现：Stationary superposition generalized early but decayed with load; exact hashing reduced transfer. Distinct experiment from Codex V39.

原处置：Keep fast predictor reference; not whole family permanently inferior to slots.

本次注记：superposition/exacthash/separation/load有分别结果；哈希E1成功不证明E3。

## M068 — Read-only hybrid versus full feedback into admission

原位置：V40 Claude; V41

原证据状态：DECISIVE DESIGN DISTINCTION

原发现：Using fast output for reading helped; allowing its current success to suppress admission harmed later durability. V40 wall hid costs.

原处置：Current correctness is not future durability; do not generalize to all fast/slow feedback forbidden.

本次注记：只读混合与把结果反馈进admission的行为不同，后者可能破坏持久性。

## M069 — Utility-based replacement; LFU-rate/LRU/random; TRACE-U

原位置：V40 Codex; V41

原证据状态：INVALID/UNDERPOWERED ASSAY

原发现：V40 task_invalid; V41 only one valid utility seed; frequency/predictability and trace restoration confounded.

原处置：Unresolved, not established and not a main biological-core proposal.

本次注记：V40无效、V41单seed弱效应；utility替换未被科学排除，也未有投入优先理由。

## M070 — Mute probationary exact traces; supported similarity retention

原位置：V42

原证据状态：FIXED-HISTORY POSITIVE AUDIT

原发现：Silencing trace reads improved loss; removing supported similarity harmed held-out recall. Four reused histories, not new replications.

原处置：Retain limited read diagnosis.

本次注记：有限历史audit读收益不足以算新鲜世界确认；保留工程条件。

## M071 — Local/global Bayesian authority; zero/E2 init; fixed50; shuffled alpha

原位置：V43

原证据状态：ARBITRATION SIGNAL; PRIMARY VALIDITY FAIL

原发现：Fixed50 strong; local evidence predicts ordinary G events. High-load slow acquisition fails; E2 init worse than zero. Identity targets scarce in G-event population.

原处置：Not a qualified persistent architecture or demonstrated target-memory classifier.

本次注记：authority信号与primary validity失败同记；不能只挑fixed50阳性称总体通过。

## M072 — Trace versus mature-slot capacity relief

原位置：V44

原证据状态：STRONG CONDITIONAL CAUSAL RESULT

原发现：D native acquisition about66.75→99.85% with slot relief, not trace relief; ~98% ever promoted, later generations often quickly evicted.

原处置：Diagnoses current slots only. Does not prove a particular replacement rule or ideal-capacity bound.

本次注记：增加成熟slot的容量缓解是因果结果，不等于一般学习规则或trace机制通过。

## M073 — General/phase normalization, KC expansion, coding-level sweeps

原位置：V36 Claude; V39

原证据状态：TESTED; DRIFT CONFOUND

原发现：Better geometric ratio with drifting coordinates did not prove or disprove stable separation benefit.

原处置：Preserve stationarity distinction; no universal ratio threshold.

本次注记：编码分数/扩张/归一化有坐标漂移混淆，不能只用地址距离认领能力。

## M074 — PHASE-FROZEN versus PHASE-LIVE versus LAG

原位置：V41 stationarity panel

原证据状态：ACTUALLY TESTED

原发现：Frozen improved live but did not beat LAG; earlier statement that this remained unrun was wrong.

原处置：Do not propose as newly missing experiment.

本次注记：PHASE-FROZEN实际测过；不可再列为尚未运行的新变体。

## M075 — Corroborated distributed synaptic consolidation (Candidate A)

原位置：Three-route screen

原证据状态：SMALL SPECIFIC SIGNAL; ARCHITECTURE WEAKER

原发现：Correct corroboration beat matched random/shuffled on selected recall contrasts but worse broad loss and worse than incumbent slot hybrid at larger load.

原处置：Close exact architecture improvement; retain causal evidence signal as limited result.

本次注记：Candidate A小特异性信号而整体弱于现有配置；停止该组合原样采用。

## M076 — 32-factor coded output with dense/modular/rewired access (Candidate B)

原位置：Three-route screen

原证据状态：COMPRESSION TRADEOFF; TOPOLOGY UNSUPPORTED

原发现：Shared factors use fewer parameters; loss/retention penalties remain. Modular support not better than degree rewire; stochastic adequacy entangles optimization.

原处置：Compression candidate, not validated fly topology or arbitrary-distribution equivalence.

本次注记：Candidate B压缩存在收益/代价，未证明解剖拓扑特异性。

## M077 — Reusable positive-pair invariance (Candidate C)

原位置：Three-route screen; V45–47

原证据状态：REPLICATED SYNTHETIC REPRESENTATION EFFECT

原发现：No evaluator identities used for pair proposals; positive relation improves new-ID geometry/probability on designed transforms and byte permutation. Delayed accuracy often low.

原处置：Retain narrow result; not general concepts or continuously adapting persistent representation.

本次注记：正例对有可复用表示作用，但属于合成变换任务；不能把同答案当同对象，也不能凭此认领算术规则。

## M078 — Invariance plus global slow consolidation factorial

原位置：V45

原证据状态：REAL SMALL GAIN; QUALIFICATION FAIL

原发现：PAIR-RAND versusRAW-RAND heldout loss −0.0582bit at256 with CI below0, below0.10 criterion; larger benefit128. CORR no additional gain, verifier event supply collapsed.

原处置：Report subthreshold positive separately from verdict. No general scaling-law conclusion from two loads.

本次注记：invariance+consolidation小信号未达固定门，相关扩展资格不足；不能无限靠量级挽救旧配方。

## M079 — Learned overlapping pathway responsibility

原位置：V46

原证据状态：NEGATIVE EXACT

原发现：Healthy8-way usage and matched51-active writes; harmful cross-talk increased and predictive feedback not better than simpler controls.

原处置：Close exact prototype/responsibility construction, not all biological recurrence or compartments.

本次注记：重叠pathway责任分配具体架构有串扰，停止该配方。

## M080 — Signed protection within PAIR nuisance subspace

原位置：V47

原证据状态：NEGATIVE EXACT WITH LOCAL TRADEOFF

原发现：Restoring8of20directions damaged desired geometry; lower some harmful interference but worse terminal loss. Correct negatives not better than shuffled.

原处置：Close exact protection construction; not all contrastive learning.

本次注记：PAIR子空间signed保护具体方向损害几何且shuffle无特异性，停止该配方。

## M081 — One-byte surprise filter on negative development

原位置：V47

原证据状态：NO DETECTABLE ADDED VALUE

原发现：Admitted~88% of conflicts; not selective on this assay. This was not a persistent plasticity-allocation test.

原处置：Do not claim contextual information gating tested generally.

本次注记：one-byte surprise具体门没有新增价值，不解释为所有因果quota失败。

## M082 — Exact sparse encoding of local output values

原位置：Engineering equivalence unit test

原证据状态：IMPLEMENTED UNIT TEST; NOT FULL SYSTEM RESULT

原发现：Unobserved output coordinates remain zero under value update; tested dense reconstruction bitwise after10k updates.

原处置：Independent engineering opportunity; no proof of whole-model equivalence or biological mechanism.

本次注记：精确局部值压缩单测成立；不是整个核心功能/算力可扩展证明。

## M083 — Full graph reduction and core peeling

原位置：fly_learning_core_results; connectivity audit

原证据状态：STRUCTURAL RESULT ONLY

原发现：10k candidate met3mass/motif criteria, missed99%weak-component seed threshold; larger graph did not fix all isolated seeds.

原处置：Does not prove no small functional core; do not minimize before functionality.

本次注记：图peeling结构统计不是功能消融的成功，不计能力机制。

## M084 — Direct/indirect MBON→DAN relay distillation

原位置：architecture_distillation reports

原证据状态：STRUCTURAL RESULT ONLY

原发现：One rank1approx loses82.08%; exactdirect+8nonnegative indirectpatterns6.73%globalerror but poor weakdestinations. Deleting relay cells ≠ preserving their effect.

原处置：Useful extraction with node provenance; not8biologicalcompartments or dynamical equivalence.

本次注记：MBON→DAN relay低秩蒸馏为结构结果；弱目的通道及整体损失仍未解决。

## M085 — Unsigned composed KC→MBON→DAN→KC spatial gate

原位置：Raw operator audit; V20–22 descendants

原证据状态：STRUCTURAL/ABSTRACT NEGATIVE

原发现：Dense normalized composed operator and small spectral radius concern a chosen linear normalization, not measured loop physiology.

原处置：Close simple spatial-protection reading; temporal/compartmental teaching not ruled out.

本次注记：unsigned composed线性空间门具体抽象负面，不是对真实回路的完整功能否定。

## M086 — Output gain modulation; exact/compressed/shuffled anatomy

原位置：V20

原证据状态：TESTED NULL

原发现：All five local arms equal final60.88% in3seedpilot; ridge retainedold but revisedpoorly.

原处置：No effect for this mean-normalized output-gain design.

本次注记：多输出gain配方小seed任务null，关闭所测标量增益操作。

## M087 — Input-only anatomical write addressing

原位置：V21

原证据状态：TESTED NO ESTABLISHED ADVANTAGE

原发现：Unsigned teaching selects active feature rows with matched current correction; not synapse-local compartment reconstruction.

原处置：Close exact input-row mask claim.

本次注记：input-only静态write mask未建立优势，不能推广为全部空间教学无效。

## M088 — Joint input/readout addressing and scrambled control

原位置：V22; V23 controls

原证据状态：TESTED NO BROAD ADVANTAGE

原发现：More specific mask did not beat ordinary robustly; sparse mask alone does not protect within its active substrate.

原处置：Close implementation; real independently signalled synaptic compartments still a different claim.

本次注记：joint static input/read mask无广泛优势，真实可塑compartment未因此排除。

## M089 — KC-family/lobe bins, whole-cell overlap compartment candidates

原位置：Raw architecture audit

原证据状态：STRUCTURAL DESCRIPTORS, NOT PHYSIOLOGY

原发现：Dominant family assignment discards mixed contacts;46overlap pairs are not46verified spatial compartments; type pooling can erase roles.

原处置：Keep provenance and denominators; do not infer memorytimescale or exclusivebanks.

本次注记：KC family/lobe bin互相重叠，不能当已经验证的独立生理compartments。

## M090 — Raw KC↔APL scalar suppressive controller

原位置：V48

原证据状态：NEGATIVE EXACT

原发现：Same51-active budget; no selective interference/persistence advantage and no superiority to rewires. Sign/gain/within-context iterations engineered.

原处置：Closes scalar profile abstraction, not APL biological function or local dendritic inhibition.

本次注记：标量APL匹配活动控制无干扰/保持增量；更复杂因果抑制尚非同一操作。

## M091 — Aligned raw PN→KC→MBON plastic mask

原位置：V49 A

原证据状态：LEARNS CURRENT; NO RAW SUPERIORITY ESTABLISHED

原发现：699PN/4552KC/104MBON;15523learnedscalars. D~89.70%, Aoldtrained~5.13%, oldheldout~1.51%. Alignment contrast CI broad.

原处置：Acquisitionwithoutpersistence under artificialbyteadaptor/codebook/allbyteupdates. Not proof rawtopology functionally irrelevant.

本次注记：raw PN→KC→MBON学得当前但旧知识差，未建立raw superiority；不能因此否定整个connectome。

## M092 — Recorded-count heterogeneity, binary support, count shuffle

原位置：V49 A; earlier raw thresholds

原证据状态：TRADEOFF; SPECIFIC ASSIGNMENT UNRESOLVED

原发现：Raw versus binary improved oldloss but worsened wholeloss; countshuffle largely matched. Count ≠ signed efficacy/plasticity/timeconstant.

原处置：Do not conclude strong=specific/slow and weak=shared/fast from countthresholds alone.

本次注记：count/binary/shuffle是特定强度分配对照，不能从计数直接推出塑性符号。

## M093 — Unsigned linear raw MBON recurrence at radius0.25

原位置：V49 B

原证据状态：NEGATIVE EXACT

原发现：Increased delayedloss~0.110bit and wholeloss~0.028; no updateefficiencybenefit or rewirespecificity.

原处置：Not a test of arbitrary nonlinear/spiking attractor networks or complete feedback dynamics.

本次注记：半径.25 unsigned MBON递归具体配方有延迟损失且无效率优势；不等于任何attractor失败。

## M094 — KC↔KC weak recurrence, DPM, broad modulatory feedback

原位置：Raw re-audits

原证据状态：STRUCTURALLY MEASURED; FUNCTIONAL TEST INCOMPLETE

原发现：Pair-row abundance differs from total counts; many weak edges and threshold dependence. No fully validated dynamic learner from these blocks.

原处置：Preserve as measured possibilities without picking a new motif merely because the previous one failed.

本次注记：KC/DPM等弱递归功能试验不完整；U055泛化联合重估不是已建立生物回路。

## M095 — Full raw-graph functional reference followed by lesion/distillation

原位置：Other-thread engineering request; Sept6 review

原证据状态：PROPOSED; NO RECOVERED LOCAL SUCCESS

原发现：Working sensory–neural–teaching–plasticity–readout system should precede aggressive reduction; no need to insist entire166kgraph initially.

原处置：Primary integration need, not another static motif tournament.

本次注记：完整rawgraph functional reference未有恢复的本地成功；它是候选参照，不强制重新回166k大图。

## M096 — Bidirectional/local DAN teaching with delayed eligibility

原位置：ELM14; local-learning review; recent brainstorms

原证据状态：PARTIAL PROXIES; FULL ASSAY UNESTABLISHED

原发现：Immediate nonnegative dopamine multiplier, unsigned routing and trace/object eligibility are not a validated timing-sensitive synaptic teacher.

原处置：Unresolved crucial interface; use explicit boundedstateandcausal signal, not a biological label pasted onto globalerror.

本次注记：当前普通到来byte可驱动学习的E0/E1基线有效；完整local DAN/延迟/双向回路尚未全部通过。

## M097 — Two-unit recurrent STM→LTM teaching (Fly-2)

原位置：Recent conversation; earlier fastslow proxies

原证据状态：PROPOSED; PROVENANCE WARNING

原发现：Specific fast/slow labels came from publications, not raw adjacency. Earlier coupledstates did run but do not reproduce this circuit.

原处置：Do not present as raw-data-deduced minimalcore or mandatory nextversion.

本次注记：Fly-2 STM→LTM教学为有来源提案，与旧耦合快慢不自动等同。

## M098 — Incremental long-vs-short information gate

原位置：PostV47 brainstorms

原证据状态：PROPOSED; NOT SCIENTIFICALLY RUN

原发现：Log predictive ratio and boundedgate were proposed; absolute surprise and informationgain differ. Future-derived mass controls are nondeployable diagnostics.

原处置：No outcome yet. Mark information provenance and coefficientmass≠actualupdate-norm.

本次注记：long-vs-short信息门未科学运行；需真实因果两预测器，不得看未来质量。

## M099 — Multi-horizon residual hierarchy

原位置：PostV47/Flycore other-thread ideas

原证据状态：PROPOSED

原发现：1/2/4/8/12 or1/8/64 horizons; no coherent boundedpersistent test in recovered record.

原处置：Not established, not entitled to become nextstep by versionchain.

本次注记：多个horizon有提案但无完整通过实现；不能自动视为更高信息收益。

## M100 — Fixed plasticity quota; competition across time

原位置：PostV47 brainstorms

原证据状态：PROPOSED

原发现：Window-wide allocations can use future information; need a causal version and separate diagnostic matching.

原处置：Unrun. Not a consequence of rawconnectome alone.

本次注记：因果quota可作不同预算分配假说；acausal窗口版本不可采用，仅省写次数不是少样本泛化。

## M101 — Homeostatic forgetting / targeted bidirectional extinction

原位置：Earlier active forgetting + later proposals

原证据状态：SPECIFIC PROXIES TESTED; FULL CLAIM OPEN

原发现：Decay exists in historical variants; an effective recurrent teaching-driven revision system has not been established.

原处置：Do not call all forgetting new or all decay impossible without it.

本次注记：extinction/主动遗忘有局部proxy，真实反馈/修订仍开放；固定加法无规则变更不能用来选强遗忘。

## M102 — Persistent scratchpad and feedback memory

原位置：Other-thread file context, earlier hubs/tapes

原证据状态：PROPOSED/RELATED IMPLEMENTATIONS

原发现：64–256-variable gated scratchpad and feedback was proposed; older actual tapes use different objects/training.

原处置：No completed cleanresult recovered; preserve statebudget and original source distinction.

本次注记：scratchpad/tape是工作状态，不直接等于可迁移规则；完整生命周期结果缺失。

## M103 — Short-context/phase+2 slot admission

原位置：Claude supplied audit, seed43101

原证据状态：ONE-SEED DIAGNOSTIC; FULL GRID NOT RUN

原发现：Keybyte candidates75.1%, framing11.7%, target13.3%; filteredgatekeeps90.2%targets versus59.7%uniformexpected. Counts are candidateadmissions, not residentcausalutility.

原处置：Useful workload insight; not validated newlearning rule or authority to return to slots.

本次注记：phase+2 admission单seed是局部diagnostic，不是居民效用或规则复用确认。

## M104 — Class-specific write lesions on V49

原位置：Latest review proposal

原证据状态：PROPOSED; NOT RUN

原发现：All-byte/no-key/no-framing/association-only oracles could localize interference; oracle masks not deployable and writecountmustbecontrolled.

原处置：A focused diagnostic on currentbaseline, not proof allocation alone will rescuelearning.

本次注记：class-masked lesion是oracle因果诊断，不能部署为自主学习机制。

## M105 — Positive-and-negative reusable metric outside protected subspace

原位置：V47 broader brainstorm

原证据状态：PROPOSED FAMILY; ONE EXACT FORM FAILED

原发现：Sameoutcome notsameidentity; nearconflict may be noise/contextshift. No fullygeneral equivalencelearner validated.

原处置：Do not close whole metricfamily or restart it without different evidencequestion.

本次注记：一个protected exact metric失败不否定所有复用metric；正负关系必须由实际观察构造，不能把标签相同偷换cue相同。

## M106 — Frozen/slow residual reference, transport compensation, constraint-only slow memory

原位置：PostV44 architecture brainstorm

原证据状态：PROPOSED; RELATED NEGATIVE PRECEDENT

原发现：V35 residual failure applies while referencepredictionsdrift; encoderstationarity alone doesnotfixit.

原处置：No new implementation success; requires explicit costed transport/frozenpredictor assumption.

本次注记：冻结参考、transport和constraint-only slow有相关历史负面及未实现操作；与当前残差含义兼容有关，不用U034重开替代。

## M107 — Indexed retrieval (hash/tree/HNSW/ANN), exact value compaction

原位置：Complexity and capacity discussions

原证据状态：ENGINEERING BACKLOG

原发现：Can reduce scan/payloadcost, doesnotbyitselfsupplybiologicallearning or representation.

原处置：Do not make the biological core a database optimization problem.

本次注记：索引/压缩属于成本工程，可能长期必要，但不增加学习信息或当前NEW样本曲线。

## M108 — Natural byte corpora, varied wrappers, nonshared transforms

原位置：Multiple designs aftersynthetic screens

原证据状态：PROPOSED/LARGELY DEFERRED

原发现：Earlierlanguage notebooks donotvalidate latestlearner; repeatedfixed14bytefixturecannotestablishgeneralbytelearning.

原处置：Keep transfer goal visible; do not report finalsyntheticassociationwin as endpoint.

本次注记：自然文本/开放wrappers为后续范围；现阶段不移走有限规则学习目标。
