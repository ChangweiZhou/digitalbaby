# 原108条记录的预算筛选

原始 M001–M108、机制名称和 evidence_status 全部保留。以下是截至2026年10月7日的后续预算判断，不追溯改写科学结果；关闭的是原配方或主张，不是整个家族。

| 原编号 | 原机制记录 | 原证据状态 | 预算处理 | 保留或停止理由 |
|---|---|---|---|---|
| M001 | Stateful nonlinear ELM cell (fast s, slow m, recurrent messages, internal MLP) | IMPLEMENTED; PARTIAL TASK EVIDENCE | 基线 对照 工程或范围外参照 | Retain as an expressive cell reference; do not equate cell count with scalar state or claim autonomous local training established. |
| M002 | Scalar-channel ELM with private vector memory | IMPLEMENTED; COMPARATIVE COVERAGE INCOMPLETE | 基线 对照 工程或范围外参照 | Keep independent private state; recover missing comparison before declaring cell arithmetic unnecessary. |
| M003 | Fixed-time-constant ELM | SHIPPED; COMPLETE RESULT NOT RECOVERED | 封存待定义或补证据 | Unresolved, not pruned experimentally. |
| M004 | Linear-dendrite ELM | SHIPPED; COMPLETE RESULT NOT RECOVERED | 封存待定义或补证据 | Unresolved; retain exact definition. |
| M005 | Simple scalar leaky recurrent cell | IMPLEMENTED/SHIPPED; PARTIAL HISTORICAL EVIDENCE | 基线 对照 工程或范围外参照 | Use as the smallest interpretable stateful reference, not an already demonstrated substitute in every task. |
| M006 | Adaptive-threshold leaky cell (ALIF-labelled) | SHIPPED; COMPLETE RESULT NOT RECOVERED | 封存待定义或补证据 | Unresolved; audit the exact cell before biological claims. |
| M007 | Scalar sparse GRU-like cell | SHIPPED; COMPLETE RESULT NOT RECOVERED | 封存待定义或补证据 | Unresolved; do not invent a ranking against ELM. |
| M008 | RG-LRU / simplified Griffin recurrent model | IMPLEMENTED; LIMITED TASK LOGS | 基线 对照 工程或范围外参照 | Historical engineering comparison, not a persistent-learning result. PickleLRU in this notebook is a file cache, not another neuron model. |
| M009 | FLM fast/slow leaky cell with sparse punctuator | IMPLEMENTED; LANGUAGE/THROUGHPUT LOGS | 基线 对照 工程或范围外参照 | Keep implementation evidence; no claim of reliable continual-learning or valid Transformer superiority. |
| M010 | Multi-timescale private neuronal state | PARTIAL POSITIVE | 有条件保留 | 保留多时间尺度状态的假说；须在相同核心、状态和信息预算下比较，不能把多 bank 当作多时间尺度。 |
| M011 | Stubborn/adaptable heterogeneous populations | PARTIAL POSITIVE; SPECIFICITY NARROWED | 有条件保留 | 异质性有部分信号，但均匀或平均速率控制解释了不少收益；只保留不能被增益控制解释的版本。 |
| M012 | Separate write and forget gates | IMPLEMENTED; NOT ISOLATED DECISIVELY | 有条件保留 | 写入和擦除是不同操作。ERROR 已改进修订，新的擦除门必须保护未修订旧知识。 |
| M013 | E/I signs, proper log-tau, homeostasis, blank-activity penalty, delay training | PARTLY IMPLEMENTED; AUDIT FLAGS | 封存待定义或补证据 | Do not treat all physiological-dynamics controls as completed negatives. |
| M014 | Raw-node spiking LIF / richer conductance cells | EXTERNAL IMPLEMENTATIONS / PROPOSED LOCALLY | 封存待定义或补证据 | Not pruned. Do not add sophisticated cells merely for realism; first state exactly what dynamics are missing. |
| M015 | Type-shared templates versus merged neurons | DESIGN CONSTRAINT | 基线 对照 工程或范围外参照 | Preserve neuron identity; distinguish structural compression from demonstrated dynamical equivalence. |
| M016 | Fixed sparse sensory/recurrent graphs and retinotopic prior | IMPLEMENTED; REFERENCE | 基线 对照 工程或范围外参照 | Retain stable reference topology with explicit provenance. |
| M017 | Prune-only / random-regrowth / guided-regrowth | IMPLEMENTED; CONTROL/AUDIT LIMITATIONS | 有条件保留 | 全局梯度指导版本不进入局部核心；仅保留有界局部权重学习与真正换边协同的新规则。 |
| M018 | New-edge lesion and topology-only transfer | SHIPPED; FULL OUTCOMES NOT RECOVERED | 封存待定义或补证据 | Unresolved where logs missing; not a proven developmental prior. |
| M019 | Global cosine-kNN graph rewiring, hubs, EMA routing | IMPLEMENTED; CAUSALITY ISSUES IN AUDIT | 停止原样投入 | 原版本有未来信息或评估状态污染，停止原样使用；这是有效性排除，非全家族科学反证。 |
| M020 | Local sensory-only development | TESTED; PARTIAL POSITIVE | 有条件保留 | 历史 local development 有部分证据；P1/P2 及低剂量 P1 已削弱当前直接移植路径，只保留不同稳定机制。 |
| M021 | Clean latent reconstruction target | IDENTIFIED PRIVILEGE | 基线 对照 工程或范围外参照 | Historical reproduction only for fully local/no-privileged-teacher claims. |
| M022 | Frozen, live, split-stable encoders; FF/FL/LF/LL address/content states | TESTED; TRADEOFF | 有条件保留 | 输入坐标变化与旧记忆不兼容是重复出现的约束；保留承诺地址冻结或记忆运输，不再仅漂移输入。 |
| M023 | Dense affine/readout transport and anchor transport | TESTED; PARTIAL/NEGATIVE | 基线 对照 工程或范围外参照 | Keep as nonlocal diagnostics; no solved local coordinate transport. |
| M024 | Learned down-stream adapter/recycling | TESTED; LOCALITY DISQUALIFICATION | 基线 对照 工程或范围外参照 | Exact candidates excluded by no-hidden-backprop constraint, not by zero numerical benefit. |
| M025 | Learned operator mixtures / neural algebra / global group routing | IMPLEMENTED; COMPOSITE DEMOS | 基线 对照 工程或范围外参照 | Not independently validated persistent-learning mechanisms. |
| M026 | Sinkhorn routing, attention hubs, global workspace, token MoE | IMPLEMENTED; OUTSIDE CURRENT MAINLINE | 基线 对照 工程或范围外参照 | Park complete architectures; retain bounded local communication hypotheses only with explicit signals. |
| M027 | Hebbian key/value fast matrices; residual/predictive Hebbian writes | IMPLEMENTED; MIXED | 有条件保留 | 旧 outer-product/残差实现不整体保留；R_center 和 ERROR 是当前参照，新增结构须改变可观察操作。 |
| M028 | Modern Hopfield memory and episodic tape | IMPLEMENTED; WORKING-MEMORY DEMOS | 基线 对照 工程或范围外参照 | Engineering reference only; explicit episodic store not default biological core. |
| M029 | Gauge/fiber semantic–surface factorization; contextual router | IMPLEMENTED; COMPOSITE/SUPERVISED | 基线 对照 工程或范围外参照 | Conceptual precedent, not evidence for a biology-based learned semantic space. |
| M030 | SAC/proposal policies, learned teacher, dialogue warmup, curiosity/ASK actions | IMPLEMENTED; CONFOUNDED FOR CURRENT GOAL | 基线 对照 工程或范围外参照 | Do not import pretrained/Transformer teacher gains into local core evidence. |
| M031 | MagmaAdamW/global optimizers; neuromorphic detach mode | IMPLEMENTED; LOCALITY LIMIT | 基线 对照 工程或范围外参照 | Retain only honest component accounting; no claim entire model biologically local. |
| M032 | Chunking, parallel Hebbian scan, sparse punctuator, scaling optimizations | IMPLEMENTED; ENGINEERING | 基线 对照 工程或范围外参照 | Keep engineering techniques where causal; do not rank neurons from unequal composite benchmarks. |
| M033 | Current-only classifier / ordinary normalized delta | TESTED BASELINE | 基线 对照 工程或范围外参照 | Keep baseline and distinguish output competition from erased input representation. |
| M034 | Cumulative ridge sufficient statistics / exact RLS | POSITIVE WITH LIMITS | 基线 对照 工程或范围外参照 | Historical reference, not a failed memory family and not a local-biological primitive. |
| M035 | Slow-only versus combined fast+slow output | TESTED; COMPOSITION MATTERS | 基线 对照 工程或范围外参照 | Do not assume two good stores add into a good reader automatically. |
| M036 | Fast-feedback weighted ridge consolidation | SMALL SPECIFIC EFFECT; BELOW BASELINE | 停止原样投入 | Exact construction closed as an improvement; biological recurrent teaching remains untested by this comparison. |
| M037 | Additive consolidation bonus | NEGATIVE EXACT | 停止原样投入 | Do not relaunch same additive ridge bonus. |
| M038 | Synaptic importance / neuron availability / protection | PARTIAL POSITIVE; RIGIDITY | 有条件保留 | 弱保护或 neuron availability 仅在真实干扰/饱和出现且可保留修订时使用；强保护原样排除。 |
| M039 | Bayesian/diagonal uncertainty; MESU-inspired gains | PARTIAL THEN INCONCLUSIVE/NEGATIVE IMPLEMENTATIONS | 有条件保留 | 历史有保持与修订权衡；不能按 Bayesian 名称豁免对固定增益/资源匹配控制的要求。 |
| M040 | Orthogonal projection / strict/leaky protected subspaces | TESTED TRADEOFF | 基线 对照 工程或范围外参照 | Reference/closed exact local candidate; not proof inhibition-based separation impossible. |
| M041 | Two- and three-state coupled synapses | TESTED NEGATIVE SCREEN | 停止原样投入 | Archive exact parameters; do not call coupled synapses untried or biologically refuted. |
| M042 | Feedback-gated local synaptic coupling | TESTED NEGATIVE SCREEN | 停止原样投入 | Do not rename as new without a changed observable operation. |
| M043 | Discrete cascade depth 4/8 | TESTED TRADEOFF/NEGATIVE | 停止原样投入 | Close exact variants; retain family uncertainty. |
| M044 | Active forgetting / surprise-triggered reopening / capacity release | TESTED MIXED/NEGATIVE | 有条件保留 | 停止全局和无条件 surprise release；只保留可区分错误与暂时不相关知识的局部规则。 |
| M045 | Sparse distributed address memory and adaptive prototypes | TESTED; ADAPTATION COST | 有条件保留 | 固定较大地址层有用但有状态成本；只能作为有容量上限的地址分离比较，不能恢复编码器轮换。 |
| M046 | Allocate/recycle local records | TESTED; SATURATION BIAS | 基线 对照 工程或范围外参照 | Engineering comparator; not evidence slots are mandatory or allocation generally impossible. |
| M047 | Quantized allocation / integer precision | TESTED; ACCOUNTING CAVEAT | 基线 对照 工程或范围外参照 | Retain only measured state/quality tradeoffs, not label-based memory savings. |
| M048 | Separate banks, fixed vs inferred routing, hard/blended reading | PARTIAL POSITIVE REPLICATED | 基线 对照 工程或范围外参照 | Useful selective-storage component; not independent proof of biological compartments or learned routing necessity. |
| M049 | Silent memories, context reinstatement, unlabeled reminders | PROMISING SCREEN; ASSAY LIMIT | 有条件保留 | 提醒试验没有证明选择性恢复。只有实际存在 dormant-return 短板时再考虑，先于答案再暴露测恢复。 |
| M050 | Bounded reservoir/recent replay, current repetition, fast-head replay | TESTED MIXED; NOT CURRENT CANDIDATE | 有条件保留 | 关闭原 replay buffer 默认升级路线；生物式有限重激活只在有独立证据的时间窗口机制下保留。 |
| M051 | Recent cache retrieval, hard/soft episodic reads | TESTED MOSTLY NULL | 停止原样投入 | Reject exact add-on claim, not retrieval generally. |
| M052 | Spillover, local normalization, fixed-target correction, update-norm controls | TESTED; BENEFIT NARROWED | 基线 对照 工程或范围外参照 | Do not count an unmatched write-scale gain as anatomical transfer. |
| M053 | Dense Hebbian/Oja encoder adaptation | TESTED TRADEOFF | 有条件保留 | P1/P2 原配方及两档 P1 移植低剂量停止；真正连续 Oja 或有地址保护的联合规则属于新结构。 |
| M054 | Novelty/shuffled/norm gates for encoder change | NEGATIVE SPECIFICITY | 停止原样投入 | Close exact selection heuristic; not all event-level teaching ever ruled out. |
| M055 | Competitive top-response plasticity, random subset, bottom-k | NO ESTABLISHED FRONTIER WIN; TASK ISSUES | 停止原样投入 | Close current claim, not biological competition as a family. |
| M056 | Temporal contrast / fast-minus-slow sensory channels | PARTIAL POSITIVE | 有条件保留 | 感官时间对比有历史正信号，但不是现在必须再换 encoder 的理由；仅在明确时序缺失时调用。 |
| M057 | Variance salience / coordinate selection | TASK-SPECIFIC | 停止原样投入 | Retire generic feature-learning claim. |
| M058 | Full covariance/co-activation-directed writes | POSITIVE WITH COST/REVISION LIMITS | 基线 对照 工程或范围外参照 | Keep precise positive evidence; not a compact local core yet. |
| M059 | Lag96 temporal-position-preserving byte encoding | TASK-USEFUL | 基线 对照 工程或范围外参照 | Useful controlled byte interface; not a biologically inferred sensory code. |
| M060 | Four sparse compartments with local covariance | TESTED NEGATIVE | 停止原样投入 | Not an untried idea; exact sparse-compartment covariance construction closed. |
| M061 | Shared residual, local residual, absolute local values | STRONG ENGINEERING DECOMPOSITION | 有条件保留 | 保留 shared/private 及中心化教学结果；learned shared prediction 对 R3 增益的贡献未获消融支持。 |
| M062 | Exact dictionary versus generalizing ownership | POSITIVE ENGINEERING, UNBOUNDED | 基线 对照 工程或范围外参照 | Baseline/diagnostic, not default biology or unlimited-capacity solution. |
| M063 | Read authority: HARD/SHADOW/CORR2/SIM-MIX/fixed alpha/random | POSITIVE ENGINEERING | 有条件保留 | Q_HALF 有有限确认增益却未通过采用；earned authority V2 的历史重放很弱，关闭该精确观察器配方。 |
| M064 | Predictive Bayes arbitration, slot evidence, shuffled-PBF | PARTIAL; SPECIFICITY GATE FAIL | 有条件保留 | 原完整 specificity gate 未通过；新支持信号必须能预测当前所需行为，不重复原证据仲裁包装。 |
| M065 | TRACE-CORR probation and distinct-rendering admission | USEFUL BOUNDED ENGINEERING | 基线 对照 工程或范围外参照 | Retain benchmark and evidence principle; object admission is not a biological mechanism by itself. |
| M066 | Fixed-cap slots, exact-repeat admission, shuffled-pair admission | TESTED CONTROLS | 基线 对照 工程或范围外参照 | Shows event/identity semantics mattered in that explicit-store system. |
| M067 | Bounded superposition, exact hashing, separation ladder, load scaling | PARTIAL POSITIVE; CAPACITY LOSS | 基线 对照 工程或范围外参照 | Keep fast predictor reference; not whole family permanently inferior to slots. |
| M068 | Read-only hybrid versus full feedback into admission | DECISIVE DESIGN DISTINCTION | 基线 对照 工程或范围外参照 | Current correctness is not future durability; do not generalize to all fast/slow feedback forbidden. |
| M069 | Utility-based replacement; LFU-rate/LRU/random; TRACE-U | INVALID/UNDERPOWERED ASSAY | 有条件保留 | 旧 task_invalid/低样本不能计淘汰，但 utility replacement 不是本阶段默认核心方向；封顶且低优先级。 |
| M070 | Mute probationary exact traces; supported similarity retention | FIXED-HISTORY POSITIVE AUDIT | 基线 对照 工程或范围外参照 | Retain limited read diagnosis. |
| M071 | Local/global Bayesian authority; zero/E2 init; fixed50; shuffled alpha | ARBITRATION SIGNAL; PRIMARY VALIDITY FAIL | 有条件保留 | 证据预测不等于有效控制；新 authority 必须跨教学行为与相对选择之间的语义缺口。 |
| M072 | Trace versus mature-slot capacity relief | STRONG CONDITIONAL CAUSAL RESULT | 基线 对照 工程或范围外参照 | Diagnoses current slots only. Does not prove a particular replacement rule or ideal-capacity bound. |
| M073 | General/phase normalization, KC expansion, coding-level sweeps | TESTED; DRIFT CONFOUND | 基线 对照 工程或范围外参照 | Preserve stationarity distinction; no universal ratio threshold. |
| M074 | PHASE-FROZEN versus PHASE-LIVE versus LAG | ACTUALLY TESTED | 基线 对照 工程或范围外参照 | Do not propose as newly missing experiment. |
| M075 | Corroborated distributed synaptic consolidation (Candidate A) | SMALL SPECIFIC SIGNAL; ARCHITECTURE WEAKER | 停止原样投入 | Close exact architecture improvement; retain causal evidence signal as limited result. |
| M076 | 32-factor coded output with dense/modular/rewired access (Candidate B) | COMPRESSION TRADEOFF; TOPOLOGY UNSUPPORTED | 基线 对照 工程或范围外参照 | Compression candidate, not validated fly topology or arbitrary-distribution equivalence. |
| M077 | Reusable positive-pair invariance (Candidate C) | REPLICATED SYNTHETIC REPRESENTATION EFFECT | 有条件保留 | 保留无标签 nuisance removal 的窄正信号；必须分开 same outcome 与 same identity，不能伪造同题配对。 |
| M078 | Invariance plus global slow consolidation factorial | REAL SMALL GAIN; QUALIFICATION FAIL | 有条件保留 | 关闭原 PAIR 加全局 CORR 的扩展承诺；仅保留 PAIR 本身的窄可复用信号，不能沿用失败 CORR。 |
| M079 | Learned overlapping pathway responsibility | NEGATIVE EXACT | 停止原样投入 | Close exact prototype/responsibility construction, not all biological recurrence or compartments. |
| M080 | Signed protection within PAIR nuisance subspace | NEGATIVE EXACT WITH LOCAL TRADEOFF | 停止原样投入 | Close exact protection construction; not all contrastive learning. |
| M081 | One-byte surprise filter on negative development | NO DETECTABLE ADDED VALUE | 停止原样投入 | Do not claim contextual information gating tested generally. |
| M082 | Exact sparse encoding of local output values | IMPLEMENTED UNIT TEST; NOT FULL SYSTEM RESULT | 基线 对照 工程或范围外参照 | Independent engineering opportunity; no proof of whole-model equivalence or biological mechanism. |
| M083 | Full graph reduction and core peeling | STRUCTURAL RESULT ONLY | 基线 对照 工程或范围外参照 | Does not prove no small functional core; do not minimize before functionality. |
| M084 | Direct/indirect MBON→DAN relay distillation | STRUCTURAL RESULT ONLY | 基线 对照 工程或范围外参照 | Useful extraction with node provenance; not8biologicalcompartments or dynamical equivalence. |
| M085 | Unsigned composed KC→MBON→DAN→KC spatial gate | STRUCTURAL/ABSTRACT NEGATIVE | 停止原样投入 | Close simple spatial-protection reading; temporal/compartmental teaching not ruled out. |
| M086 | Output gain modulation; exact/compressed/shuffled anatomy | TESTED NULL | 停止原样投入 | No effect for this mean-normalized output-gain design. |
| M087 | Input-only anatomical write addressing | TESTED NO ESTABLISHED ADVANTAGE | 停止原样投入 | Close exact input-row mask claim. |
| M088 | Joint input/readout addressing and scrambled control | TESTED NO BROAD ADVANTAGE | 停止原样投入 | Close implementation; real independently signalled synaptic compartments still a different claim. |
| M089 | KC-family/lobe bins, whole-cell overlap compartment candidates | STRUCTURAL DESCRIPTORS, NOT PHYSIOLOGY | 基线 对照 工程或范围外参照 | Keep provenance and denominators; do not infer memorytimescale or exclusivebanks. |
| M090 | Raw KC↔APL scalar suppressive controller | NEGATIVE EXACT | 停止原样投入 | Closes scalar profile abstraction, not APL biological function or local dendritic inhibition. |
| M091 | Aligned raw PN→KC→MBON plastic mask | LEARNS CURRENT; NO RAW SUPERIORITY ESTABLISHED | 基线 对照 工程或范围外参照 | Acquisitionwithoutpersistence under artificialbyteadaptor/codebook/allbyteupdates. Not proof rawtopology functionally irrelevant. |
| M092 | Recorded-count heterogeneity, binary support, count shuffle | TRADEOFF; SPECIFIC ASSIGNMENT UNRESOLVED | 基线 对照 工程或范围外参照 | Do not conclude strong=specific/slow and weak=shared/fast from countthresholds alone. |
| M093 | Unsigned linear raw MBON recurrence at radius0.25 | NEGATIVE EXACT | 停止原样投入 | Not a test of arbitrary nonlinear/spiking attractor networks or complete feedback dynamics. |
| M094 | KC↔KC weak recurrence, DPM, broad modulatory feedback | STRUCTURALLY MEASURED; FUNCTIONAL TEST INCOMPLETE | 有条件保留 | 未测试有界 KC↔KC/DPM/反馈动态不能被 PN→KC 图负结果排除；不得把静态 motif 当动态证据。 |
| M095 | Full raw-graph functional reference followed by lesion/distillation | PROPOSED; NO RECOVERED LOCAL SUCCESS | 基线 对照 工程或范围外参照 | Primary integration need, not another static motif tournament. |
| M096 | Bidirectional/local DAN teaching with delayed eligibility | PARTIAL PROXIES; FULL ASSAY UNESTABLISHED | 有条件保留 | ERROR 支持改教学目标；延迟/双向 compartment teacher 仍需独立、因果、有界规则，不凭 DAN 标签采用。 |
| M097 | Two-unit recurrent STM→LTM teaching (Fly-2) | PROPOSED; PROVENANCE WARNING | 有条件保留 | Fly-2 不是从原解剖推导的必选核心；仅作有明确新增动态与状态成本的候选。 |
| M098 | Incremental long-vs-short information gate | PROPOSED; NOT SCIENTIFICALLY RUN | 有条件保留 | 绝对 surprise 已无通用收益依据；增量信息门仍未定义有效控制规律，不自动进入运行名册。 |
| M099 | Multi-horizon residual hierarchy | PROPOSED | 有条件保留 | 多 horizon 层级缺少完整有界比较；只在能说明新增经验组织功能且不扩大任务时保留。 |
| M100 | Fixed plasticity quota; competition across time | PROPOSED | 有条件保留 | 仅保留严格因果的跨时间 quota；未来窗口预算匹配不能作为部署规则。 |
| M101 | Homeostatic forgetting / targeted bidirectional extinction | SPECIFIC PROXIES TESTED; FULL CLAIM OPEN | 有条件保留 | REPLACE/BOTH 损伤旧知识，ERROR 已解决大部分修订；局部有目标的擦除以剩余修订缺陷为启动条件。 |
| M102 | Persistent scratchpad and feedback memory | PROPOSED/RELATED IMPLEMENTATIONS | 有条件保留 | 有界工作状态可保留为结构备选，不增加开放生成或 AGI 验收；旧 tape 成功不是该候选的证据。 |
| M103 | Short-context/phase+2 slot admission | ONE-SEED DIAGNOSTIC; FULL GRID NOT RUN | 基线 对照 工程或范围外参照 | Useful workload insight; not validated newlearning rule or authority to return to slots. |
| M104 | Class-specific write lesions on V49 | PROPOSED; NOT RUN | 基线 对照 工程或范围外参照 | A focused diagnostic on currentbaseline, not proof allocation alone will rescuelearning. |
| M105 | Positive-and-negative reusable metric outside protected subspace | PROPOSED FAMILY; ONE EXACT FORM FAILED | 有条件保留 | 正负 metric 精确旧版本不复活；只保留有可信因果配对来源且匹配旧知识保持的新版本。 |
| M106 | Frozen/slow residual reference, transport compensation, constraint-only slow memory | PROPOSED; RELATED NEGATIVE PRECEDENT | 有条件保留 | reference predictor 漂移确实会破坏残差语义；冻结/运输必须显式计费并优于当前 center/ERROR。 |
| M107 | Indexed retrieval (hash/tree/HNSW/ANN), exact value compaction | ENGINEERING BACKLOG | 基线 对照 工程或范围外参照 | Do not make the biological core a database optimization problem. |
| M108 | Natural byte corpora, varied wrappers, nonshared transforms | PROPOSED/LARGELY DEFERRED | 基线 对照 工程或范围外参照 | Keep transfer goal visible; do not report finalsyntheticassociationwin as endpoint. |
