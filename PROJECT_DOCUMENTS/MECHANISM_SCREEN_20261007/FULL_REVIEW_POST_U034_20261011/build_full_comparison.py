import json,pathlib,collections,re,copy
ROOT=pathlib.Path('/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929')
BASE=ROOT/'PROJECT_DOCUMENTS/MECHANISM_SCREEN_20261007'
OUT=BASE/'FULL_REVIEW_POST_U034_20261011'
OUT.mkdir(exist_ok=True)
def read(n):return json.loads((BASE/n).read_text())
def save(n,d):(OUT/n).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
m=read('HISTORICAL_108_SCREEN.json');c=read('CATALOGUE_136_SCREEN.json');vdoc=read('VERSION_REGISTER_20261011/VERSIONS.json');v=vdoc['versions'];u=read('NEXT_ROUND_PRIORITY.json');first=read('FIRST16_SCREEN.json')
refs=collections.defaultdict(list)
for row in u:
 for ref in row['source_refs']:refs[ref].append(row)
# Explicit scope notes supplement, never overwrite, the original 108 findings/dispositions.
notes={
1:'表达能力参照；部分任务依赖BPTT/向量私有状态，不能计为已验证自主持续学习。',
2:'完整六细胞比较未恢复；缺结果不等于排除，保持与标量通道不同的私有状态记账。',
3:'已实现/部分日志；没有足够完整对照支持删除全部时间常数方案。',
4:'线性树突版本仍含内部MLP；完整比较缺失，不能称已验证纯线性核心。',
5:'最简单标量动态参照；结构简单不自动代表低样本复用强。',
6:'自适应阈值已有实现，但缺完整持续学习比较；不能只靠ALIF名称判优。',
7:'同图GRU样实现参数量不同；缺完整结果，作为表达/成本参照。',
8:'门控递归有任务信号，但不是已确认持续复用或加法样本效率。',
9:'语言/吞吐日志的Transformer比较存在NaN；不能用无效比较支持核心提升。',
10:'保留私有多时间尺度方向；它与复制多个完整银行不是同一个操作。',
11:'历史改善被学习率/活动门对照缩窄；没有独立异质性特异性胜利。',
12:'当前只采用U042的SUPPORT写门实例；低样本提升未确认，不代表独立遗忘门已通过。',
13:'保留真实参数与实际dispatch范围；多个flags存在不等于全部变体科学运行过。',
14:'原始脉冲/电导细胞缺同底物完整对照；不为生物复杂性本身追加实验。',
15:'类型共享参数不得合并各神经元可变状态；是结构约束，不是已胜出学习操作。',
16:'固定稀疏图可作底物/对照；不能将静态拓扑或输入先验单独算新能力。',
17:'当前TOPO-G已经联合学权重与导向换边，DEV未晋级；不同P×S操作仍须明示区别。',
18:'新边消融/仅拓扑迁移完整结果未恢复；不可用缺失当负面。',
19:'早期global-kNN版本有未来信息或探测改状态问题；先标因果不合格，不能当家族科学负面。',
20:'局部感觉开发有局限阳性；观察目标与干净latent目标、Hebb/Oja与BP分别记账。',
21:'干净latent重建含额外信息；其成功不能证明仅凭实际噪声答案的学习。',
22:'FF/FL/LF/LL及冻结/活动/稳定split都已有具体版本；当前角色输入固定，不把旧结构重命名成未测。',
23:'anchor/仿射运输是输入坐标兼容参照，成本与非局部性有界性待明示。',
24:'adapter采用隐藏BP的版本局部性不合格，不是数值上等同失败。',
25:'算子混合/神经代数是复合demo；没有隔离出持续核心增量。',
26:'注意力/Sinkhorn等全局路由属于不同范围；未排除，但不据名称自动排期。',
27:'当前COV-R为已确认的有限工程加法E3锚点；旧Hebb快矩阵与完整RLS不是同一实现。',
28:'Hopfield/episode tape的上下文重置演示不等于跨生命保留；检索成本与存储另记。',
29:'semantic/surface分解的监督复合结果不能冒充自主学得的不变性。',
30:'奖励/预训练/教师先验与本核心获取信息预算不同，先保留参照。',
31:'全局优化器/隐藏梯度改变学习信息与成本；局部detach名称不自动消除这些差别。',
32:'吞吐/scan/分块是工程实现收益；不作为新增泛化机制。',
33:'普通delta是有用对照；其历史遗忘不排除所有低成本更新。',
34:'历史ridge/RLS有保持阳性，也有修订/安全失败；与当前COV-R已包含的相关统计分开。可讨论的新操作必须不是再实现现有RLS。',
35:'slow-only与fast+slow的具体读出曾不同；简单相加可能稀释信号，不默认双路即更好。',
36:'精确反馈巩固配方有小特异性信号但总体低于父基线；停止原样采用，不排除所有巩固。',
37:'加法式consolidation bonus未建立增量且增加统计状态；停止该具体配方。',
38:'importance/availability多版本保护旧知识与获取/修订有代价；不以旧保持替代NEW效率。',
39:'V23 stable/reopening、V31 N8/32/128/512/inf/frozen/shuffle是不同版本。U034 SUPPORT-DIR退出；V31读取不足的void与observable的inconclusive不能写成整族否定。',
40:'严格/泄漏保护子空间有保持收益及僵硬代价；投影非局部成本另列。',
41:'双/三耦合状态历史结果混合，不能把所有快慢形态归同一个失败。',
42:'feedback coupling的具体配方未优于普通；暂停该配方，不证明任意反馈回路无用。',
43:'cascade4/8已有结果，深度增加并未解决获取/修订代价；不重复做更深即更好。',
44:'U034支持式方向重开在合格DEV与SUPPORT的NEW J差为0，约1.18倍CPU；关闭本配方。噪声不等于规则失效，勿用惊讶自动擦除旧知识。',
45:'固定分布式地址与自适应prototype已有多容量版本；自适应成本/稳定性未解决。',
46:'分配/回收有早期保持与后期饱和/初始化问题；扩大容量不等于新规则。',
47:'V23 quantized不是真正紧凑存储；V25 int8是另一个版本，精度/状态不能混为一谈。',
48:'固定/推断路由、4/8 banks、hard/blended都已有试验；容量与路由信息须等额比较。',
49:'silent/context固定与推断、reminder已有历史结果；并非所有上下文任务有可靠身份。',
50:'reservoir/recent/fastslow各8/32及当前REPLAY-P/U和COV-REPLAY已测；残差优先并非未测。覆盖型有限再激活是新操作假说，不能原样复活回放。',
51:'recent hard/soft8/32多为null或很小改善；当前原样addon不优先。',
52:'spillover/归一化/固定target等收益被更新质量对照缩窄；保留控制而非泛化定论。',
53:'P1/P2及低剂量移植未获得采用证据；活动输入Oja的兼容性新结构与旧配方分开。',
54:'novelty/shuffle/norm specificity结果负面；P3技术不合格不是科学反证。',
55:'top/bottom/random竞争没有已建立frontier胜利；缺陷任务不能用于整族否定。',
56:'fast-minus-slow历史时序任务局部阳性，不足以说明当前加法失误由输入时序造成。',
57:'variance salience的旋转/方差对照不支持通用特异性收益，停止旧主张。',
58:'full covariance既有成本/修订限制又有当前COV-R确认E3；不等同V30方向塑形或所有协方差版本。',
59:'Lag96工程位置编码是有效读取参照，不是独立生物学习机制。',
60:'C4-P局部cov具体配方未增加未教收益；不代替完整COV的结果。',
61:'shared/local residual与absolute读取有不同目标漂移；保存参考/约束含义是长期兼容问题，非单次接口得分。',
62:'exact dictionary能解释E1但没有未教关系相似性；保留工程对照，不作为当前E3核心。',
63:'SIM-MIX/LINK具体读出停止；CORR2/权限控制不是同一操作，简单margin相加曾削弱迁移。',
64:'slot/PBF只部分通过或资格失败；不要将仲裁名称当有完整验证算法。',
65:'TRACE-CORR有限工程admission收益保留，完整核心门未建立。',
66:'重复/打乱admission是对照；事件身份不是 evaluator可免费提供的信息。',
67:'superposition/exacthash/separation/load有分别结果；哈希E1成功不证明E3。',
68:'只读混合与把结果反馈进admission的行为不同，后者可能破坏持久性。',
69:'V40无效、V41单seed弱效应；utility替换未被科学排除，也未有投入优先理由。',
70:'有限历史audit读收益不足以算新鲜世界确认；保留工程条件。',
71:'authority信号与primary validity失败同记；不能只挑fixed50阳性称总体通过。',
72:'增加成熟slot的容量缓解是因果结果，不等于一般学习规则或trace机制通过。',
73:'编码分数/扩张/归一化有坐标漂移混淆，不能只用地址距离认领能力。',
74:'PHASE-FROZEN实际测过；不可再列为尚未运行的新变体。',
75:'Candidate A小特异性信号而整体弱于现有配置；停止该组合原样采用。',
76:'Candidate B压缩存在收益/代价，未证明解剖拓扑特异性。',
77:'正例对有可复用表示作用，但属于合成变换任务；不能把同答案当同对象，也不能凭此认领算术规则。',
78:'invariance+consolidation小信号未达固定门，相关扩展资格不足；不能无限靠量级挽救旧配方。',
79:'重叠pathway责任分配具体架构有串扰，停止该配方。',
80:'PAIR子空间signed保护具体方向损害几何且shuffle无特异性，停止该配方。',
81:'one-byte surprise具体门没有新增价值，不解释为所有因果quota失败。',
82:'精确局部值压缩单测成立；不是整个核心功能/算力可扩展证明。',
83:'图peeling结构统计不是功能消融的成功，不计能力机制。',
84:'MBON→DAN relay低秩蒸馏为结构结果；弱目的通道及整体损失仍未解决。',
85:'unsigned composed线性空间门具体抽象负面，不是对真实回路的完整功能否定。',
86:'多输出gain配方小seed任务null，关闭所测标量增益操作。',
87:'input-only静态write mask未建立优势，不能推广为全部空间教学无效。',
88:'joint static input/read mask无广泛优势，真实可塑compartment未因此排除。',
89:'KC family/lobe bin互相重叠，不能当已经验证的独立生理compartments。',
90:'标量APL匹配活动控制无干扰/保持增量；更复杂因果抑制尚非同一操作。',
91:'raw PN→KC→MBON学得当前但旧知识差，未建立raw superiority；不能因此否定整个connectome。',
92:'count/binary/shuffle是特定强度分配对照，不能从计数直接推出塑性符号。',
93:'半径.25 unsigned MBON递归具体配方有延迟损失且无效率优势；不等于任何attractor失败。',
94:'KC/DPM等弱递归功能试验不完整；U055泛化联合重估不是已建立生物回路。',
95:'完整rawgraph functional reference未有恢复的本地成功；它是候选参照，不强制重新回166k大图。',
96:'当前普通到来byte可驱动学习的E0/E1基线有效；完整local DAN/延迟/双向回路尚未全部通过。',
97:'Fly-2 STM→LTM教学为有来源提案，与旧耦合快慢不自动等同。',
98:'long-vs-short信息门未科学运行；需真实因果两预测器，不得看未来质量。',
99:'多个horizon有提案但无完整通过实现；不能自动视为更高信息收益。',
100:'因果quota可作不同预算分配假说；acausal窗口版本不可采用，仅省写次数不是少样本泛化。',
101:'extinction/主动遗忘有局部proxy，真实反馈/修订仍开放；固定加法无规则变更不能用来选强遗忘。',
102:'scratchpad/tape是工作状态，不直接等于可迁移规则；完整生命周期结果缺失。',
103:'phase+2 admission单seed是局部diagnostic，不是居民效用或规则复用确认。',
104:'class-masked lesion是oracle因果诊断，不能部署为自主学习机制。',
105:'一个protected exact metric失败不否定所有复用metric；正负关系必须由实际观察构造，不能把标签相同偷换cue相同。',
106:'冻结参考、transport和constraint-only slow有相关历史负面及未实现操作；与当前残差含义兼容有关，不用U034重开替代。',
107:'索引/压缩属于成本工程，可能长期必要，但不增加学习信息或当前NEW样本曲线。',
108:'自然文本/开放wrappers为后续范围；现阶段不移走有限规则学习目标。'
}
assert set(notes)==set(range(1,109))
# Broad-core relevance is independent of next addition scheduling.
engineer={15,16,21,23,31,32,47,59,62,66,72,76,82,83,84,89,92,104,107}
open_missing={2,3,4,6,7,13,14,18,95}
invalid={9,19,24,69,71}
closed_exact={36,37,42,51,54,57,60,75,78,79,80,81,85,86,87,88,90,93}
concrete={12,17,22,27,34,38,39,44,48,49,50,53,58,61,63,77,94,96,98,99,100,105,106}
macro_labels={'REFERENCE':'长期参照/混合历史证据','ENGINEERING':'工程组件/对照或信息边界','MISSING_COMPARISON':'完整结果/比较缺失，不能判死','INVALID_OR_PARTIAL':'因果/资格/统计限制，非整族负面','STOP_EXACT':'停止所测具体配方，家族保留边界','CONDITIONAL_OPERATION':'长期仍有条件价值，须区别具体版本'}
mrows=[]
for raw in m:
 n=int(raw['id'][1:]);label='ENGINEERING' if n in engineer else 'MISSING_COMPARISON' if n in open_missing else 'INVALID_OR_PARTIAL' if n in invalid else 'STOP_EXACT' if n in closed_exact else 'CONDITIONAL_OPERATION' if n in concrete else 'REFERENCE'
 mapped=refs[raw['id']]
 addition={'mapped_subset_entries':[{k:x[k] for k in ['dedup_id','name','priority_status','priority_reason']} for x in mapped], 'not_mapped_is_not_excluded':not bool(mapped)}
 if not mapped:addition['decision']='完整条目未被71子集覆盖；不能凭此排除。是否用于加法须另看操作/当前错误/样本与计算成本。'
 mrows.append({'id':raw['id'],'original':copy.deepcopy(raw),'broad_core_status':label,'broad_core_label':macro_labels[label],'updated_scope_note':notes[n],'current_addition_review':addition,'family_wide_exclusion':False,'new_run_authorized':False})
# Catalogue actual later results override old pending/untested wording only in the supplement.
cn={
2:'FE1具体集成未示可采用增量；关系下降与独立回忆上升不是一个集成能力。',
5:'FE-NI具体配方停止；不要用名称重新启动负像系列。',
7:'LAT4后验重拟合不是原先decoder的独立确认；时序输入并非当前加法已证瓶颈。',
8:'PAIR小集及16题回忆较好但per-cue与E3有局限；回忆分离≠迁移。',
9:'ROLE回忆66.4%，关系效应+34.4pp；这支持特定表示作用，未给出一个同时通过所有目标的核心。',
10:'learned h比随机改善而桥接门失败，spacing35.5%；不能忽略接口失效或将控制差异认领全桥接。',
11:'CONTENT有限精确键E1/E2/保持基线有效；spacing/prefix同键，不是新关系泛化。',
12:'P1原配方有效负面；.005/.0005移植未晋级。不以趋零扫参作新机制。',
13:'P2旧binary/normalized Oja未有采用增量；不排除不同连续输入兼容结构。',
14:'P3技术不合格，没有科学轨迹支持科学负面；本轮不修复/启动。',
19:'variance salience缺特异性控制支持，停止旧通用收益主张。',
20:'PHASE-FROZEN已实际测试；旧pending表述不得沿用为当前未测。',
23:'T1有效E1未增益；原E3禁写错误导致无效，不能排除整拓扑家族。',
24:'T2原样停止；仅4E弱优化为历史封顶备选，原E3无效。',
25:'T3原样停止；bridge .4仅历史封顶备选，原E3无效。',
30:'S1弱边SET有效E1无增益；P×S新操作不同，当前TOPO-G也已测过。',
31:'S2原样停止，8次/event仅低优先备选；原E3无效。',
32:'S3 rho cue-clock移植未晋级；rho-only未确认，不默认再扫参数。',
33:'S4原样停止，8次暂试/event仅低优先备选；原E3无效。',
38:'早期cosine-kNN未来信息/探测写状态问题属于因果不合格，不算整族负结果。',
43:'scalar APL具体操作在匹配活动控制下无增量；不推广为所有抑制反馈失败。',
52:'Z1修正后有效E3相对基线−14.6pp，区间全负；停止当前配方。',
53:'Z2无增量且taught guards弱；停止当前资源/冲突门配方。',
58:'local-policy refit具体失败；不能作为尚未测成功路径。',
60:'strong importance的保持代价损害获取/修订，旧参数不可原样采用。',
63:'stable/reopening分开；V31有限horizon仍inconclusive。当前U034 SUPPORT-DIR关闭，不能覆盖stable或全部MESU。',
65:'FULL151 adaptive allocation删除/分配损害旧基线；不自动等同当前COV-SUPPORT。',
73:'TARGET-ONLY 70.1%低于opponent89.5%；停止这个写策略替换，不称它生物忠实就获胜。',
76:'实际答案byte到来可自主驱动有限观察E0/E1；无需专设苍蝇拍，但无E3确认。',
80:'native REF类V9.9A/B有限二元关系阳性保留；不同四通道失败不能抹除阳性。',
85:'异质compartment尚须具体操作，不能默认模仿生理名称就改善样本效率。',
87:'R1有效E1/E3无可采用增量，停止原配方。',
88:'R2 corrected E3 −2.1pp区间跨零，无采用支持；不等同当前ERROR。',
89:'R3/R_center/ERROR贡献分层记录；96-world ERROR修订强，但整体V2保留V1，不可称全新核心通过。',
92:'active_forgetting已有V23/V24；不是尚未测原型，新增忘却需区别噪声与真实变更。',
93:'immediate/smoothed surprise已有具体阴性或无特异性，停止原样追加。',
95:'~~U034 SUPPORT-DIR~~退出；6个BLURRY DEV世界NEW J与SUPPORT平均相同，约1.18倍CPU；不做剂量/窗口救援。',
97:'global CONTRACT .05旧操作停止；不能据此等同所有目标定向release失败。',
98:'BR/KR/BKR原组合已停止；组合命名不能证明新版本。',
104:'silent fixed/inferred真实已有试验，不列成全部尚未运行。',
105:'context fixed/inferred真实已有试验，不与silent或当前无context加法混排。',
107:'evaluator reveal为额外信息，不能部署为模型自身熟悉度。',
109:'recent hard/soft 8/32已测多为null/小效应；原样addon停止优先投入。',
116:'没有episode archive的offline O未完整操作化；不能声称现有成功实现。U047新覆盖选择只是待测变体。',
117:'reservoir/recent/fastslow8/32均已有版本；当前REPLAY-P是残差优先，COV-REPLAY也已测。不得又声称只测均匀。',
121:'stubborn/adaptable历史收益被rate/activity controls缩窄；未证明独立特异性胜利。',
122:'V68 reversal classifier具体失败，不能用新名称当新生理机制。',
131:'四store prompt argmax为合格有限E1桥接读出；不泛称其关系泛化成功。',
132:'单store L/R读出有限关系E3阳性已保留；候选由外部呈现的边界仍在。',
133:'四store关系output v1/v0未资格化；读出失败不能替输入/核心作整族结论。',
134:'简单sum/scaled sum曾削弱FE0迁移，后验挑规则不是新确认；禁止oracle familiar标签。',
135:'CORR2/fixedmix各不同；SIM-MIX/LINK原组合停止，不把所有权限控制判死。',
136:'TRACE-CORR工程admission有条件收益，没证明生理或普通字节E3核心。'
}
crows=[]
for raw in c:
 cid=f"C{raw['source_row']:03d}";mapped=refs[cid]
 note=cn.get(raw['source_row'])
 if not note:
  note='；'.join(dict.fromkeys(x['priority_reason'] for x in mapped)) if mapped else raw['reason']+' 原条目仍完整保留；若缺最新具体试验，不能当科学排除。'
 crows.append({'id':cid,'original':copy.deepcopy(raw),'latest_scope_note':note,'mapped_subset_entries':[{k:x[k] for k in ['dedup_id','name','priority_status','priority_reason']} for x in mapped], 'broad_core_decision':'具体已测范围见最新注记；未被71映射不等于淘汰。','current_addition_decision':'；'.join(x['priority_status']+': '+x['dedup_id'] for x in mapped) if mapped else '未映射；按原具体操作保留，不从子集缺席推出失败或自动候选。','family_wide_exclusion':False,'new_run_authorized':False})
# Each version is retained with ALL contexts, parameters and evidence, not just a family name.
def version_review(raw):
 s=raw['series'];n=raw['name'];base='保留全部操作参数、任务/读取上下文和原结果。历史终点不可与当前加法NEW曲线混排。'
 if s=='V31_UNCERTAINTY':return 'MIXED_VALIDITY_INCONCLUSIVE','contrast/lagged读取资格失败的判定void；observable足够而有限8/32/128/512结论inconclusive。控制与无限horizon分别保留；U034不能替它们全部判死。'
 if s=='V30_DIRECTION':return ('CONTROL' if raw['operation'].get('control') else 'NO_SELECTED_WRITER'), 'direction screen无writer满足全部门；实用阶段未运行，不等同所有full-covariance算法失败。'
 if s=='SUPPORT_DIRECTION_V1':
  if n=='SUPPORT-DIR':return 'STOP_TESTED_RECIPE','~~U034 SUPPORT-DIR~~：合格DEV未晋级；NEW J相对SUPPORT 0pp，CPU约1.18倍。没有confirm；不以OLD改善替主目标、不参数救援。'
  if n in ('SUPPORT-RAND','SUPPORT-ISO'):return 'NONSELECTABLE_DIAGNOSTIC','有效DEV诊断但不得后验晋级；它们不是当前采纳核心。'
  return 'REFERENCE_BASELINE','本轮基线记录；COV-SUPPORT用户采用，COV-R确认锚点，重复出现不多算机制。'
 if s=='OBSERVATION_SUPPORT_V1':
  if n=='COV-SUPPORT':return 'USER_ADOPTED_LIMITED_EVIDENCE','用户采用写门；少样本NEW增量未确认，保持改善单列。采纳决定与统计优越性区分。'
  return ('NONSELECTABLE_DIAGNOSTIC' if n=='COV-RATE' else 'REFERENCE_BASELINE'),'RATE只作诊断，非逐记录完全匹配门；COV-R为锚点。'
 if s=='FEEDBACK_EFFICIENCY_V1':
  if n=='COV-R':return 'REFERENCE_BASELINE','保留已确认有限工程E3锚点。'
  if n=='COV-EXTRA':return 'STOP_TESTED_RECIPE','进入独立确认但未达采用门；主增量约1.074pp、区间跨0且计算增加；不以更多同一样本更新认领效率。'
  if n=='COV-REPLAY':return 'NONSELECTABLE_DIAGNOSTIC','已测完整COV的回放诊断；不是新未测配方，也不独立晋级。'
  return 'DEV_NO_ADOPTION','实际DEV版本保留；未成为采用配方，不能将其视为从未试过或推广整族失败。'
 if s=='SAMPLE_GEN_V2':
  if n=='COV-R':return 'CONFIRMED_BOUNDED_E3','角色可加/序位输出先验下，独立确认有限未教加法组合学习；不是unmodified Full151自主发现算术。'
  if n in ('STABLE-ID','COMPACT-ID','DIAG-R'):return 'REFERENCE_OR_COMPARATOR','有限工程加法中的已运行参照/比较；当前性能弱不代表所有记忆形式无用。'
  return 'DEV_NO_ADVANCE','TOPO-G已同时更新边权和换边；REPLAY-P已按残差选择。未晋级仅关闭当前操作/底物组合，不许改名当未测。'
 # Historical variants: preserve distinctions rather than blanket deletion.
 if n in ('ordinary','ordinary_slow','bank1') or 'scrambled' in n or 'shuffled' in n:return 'HISTORICAL_CONTROL',base+' 普通/慢率/打乱控制不作为新候选。'
 if any(t in n for t in ['replay','repeat_current']):return 'HISTORICAL_REPLAY_VERSION',base+' reservoir/recent/fastslow/预算与样本重用不同；复用不新增外部信息，所有额外更新计费。'
 if n.startswith('mesu') or n.startswith('bayes'):return 'HISTORICAL_UNCERTAINTY_VERSION',base+' stable与reopening分开；与当前支持式方向恢复及V31有限horizon不自动合并。'
 if any(t in n for t in ['bank','silent','context_']):return 'HISTORICAL_ROUTING_VERSION',base+' 固定/推断路由、银行数、hard/blended及容量/身份成本分别保留。'
 if n.startswith(('dual_','split_','shared_')):return 'HISTORICAL_COORDINATE_VERSION',base+' 冻结/活动地址与内容、稳定/释放/路由不同，不能将结果池化。'
 if any(t in n for t in ['importance','availability']):return 'HISTORICAL_PROTECTION_VERSION',base+' 保护旧知识、获取及修订的代价同时看；并非当前少样本增量证明。'
 if any(t in n for t in ['forget','surprise']):return 'HISTORICAL_FORGETTING_VERSION',base+' 因果规则变化与观察噪声不同，不默认增加forget即可提高当前加法学习。'
 if any(t in n for t in ['cascade','fast_slow','consol']):return 'HISTORICAL_CONSOLIDATION_VERSION',base+' 具体状态数/耦合/读出已保留；不能将全部时间尺度当同一个失败。'
 if any(t in n for t in ['allocate','sparse','quantized']):return 'HISTORICAL_CAPACITY_VERSION',base+' 状态/精度/适应成本与保持/获取分别比较，不能只看终点平均。'
 if n.startswith('target') or n.startswith('reward'):return 'HISTORICAL_FEEDBACK_VERSION',base+' 答案/延迟/optimism等教学信息预算与因果可用性分别记录。'
 return 'HISTORICAL_SCOPED_REFERENCE',base+' 不是因缺席71子集而排除，也不是自动可运行候选。'
vrows=[]
for raw in v:
 status,note=version_review(raw)
 vrows.append({'id':raw['version_id'],'original':copy.deepcopy(raw),'review_status':status,'review_note':note,'family_wide_exclusion':False,'new_run_authorized':False})
save('HISTORICAL_108_COMPARISON.json',mrows);save('CATALOGUE_136_COMPARISON.json',crows);save('VERSION_254_COMPARISON.json',vrows)
def esc(s):return str(s).replace('|','\\|').replace('\n','<br>')
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+''.join('| '+' | '.join(esc(a) for a in r)+' |\n' for r in rows)
mmd='# 108项原总账逐项对照\n\n原条目、原证据和处置完整保留在JSON及下方逐项说明。长期核心价值和当前加法排期分开；任何关闭只限具体已测版本。\n\n'
mmd+=table(['ID','原机制','长期核心状态','最新范围判断','71子集映射'],[(x['id'],x['original']['mechanism'],x['broad_core_label'],x['updated_scope_note'],', '.join(a['dedup_id']+':'+a['priority_status'] for a in x['current_addition_review']['mapped_subset_entries']) or '未映射≠淘汰') for x in mrows])
for x in mrows:
 raw=x['original'];mmd+=f"\n## {x['id']} — {raw['mechanism']}\n\n原位置：{raw['where']}\n\n原证据状态：{raw['evidence_status']}\n\n原发现：{raw['finding']}\n\n原处置：{raw['disposition']}\n\n本次注记：{x['updated_scope_note']}\n"
(OUT/'HISTORICAL_108_COMPARISON.md').write_text(mmd)
cmd='# 136项完整变体目录逐项对照\n\n这是2026-09-28目录的完整136行，不把当时pending/untested当当前状态。原文与后续注记分开，未被71子集映射不等于排除。\n\n'
cmd+=table(['ID','原名称','原预算状态','最新具体范围','当前子集映射'],[(x['id'],x['original']['name'],x['original']['budget_status'],x['latest_scope_note'],', '.join(a['dedup_id']+':'+a['priority_status'] for a in x['mapped_subset_entries']) or '未映射≠淘汰') for x in crows])
for x in crows:
 r=x['original'];cmd+=f"\n## {x['id']} — {r['name']}\n\n原目录位置：{r['source_path']}:{r['source_line']}\n\n原始目录行（原封保留）：\n\n```text\n{r['source_text']}\n```\n\n后续范围核对：{x['latest_scope_note']}\n"
(OUT/'CATALOGUE_136_COMPARISON.md').write_text(cmd)
vmd='# 254条程序/操作版本逐条对照\n\n254含重复基线、诊断和复合配置，不是254个独立机制。V24/V25各版本的所有上下文、参数及结果保存在每条展开内容和JSON中，不因名称相同自动合并。历史联想/修订、下一符号任务和当前加法结果不直接排名。\n\n'
vmd+=table(['版本ID','状态','本次范围核对'],[(x['id'],x['review_status'],x['review_note']) for x in vrows])
for x in vrows:
 vmd+=f"\n## {x['id']}\n\n{x['review_note']}\n\n<details><summary>完整操作、参数、上下文、来源与证据</summary>\n\n```json\n{json.dumps(x['original'],ensure_ascii=False,indent=2)}\n```\n\n</details>\n"
(OUT/'VERSION_254_COMPARISON.md').write_text(vmd)
# Verify coverage and verbatim source preservation, not a re-audit of scientific trajectories.
assert [x['id'] for x in mrows]==[x['id'] for x in m]
assert [x['id'] for x in crows]==[f"C{x['source_row']:03d}" for x in c]
assert [x['id'] for x in vrows]==[x['version_id'] for x in v]
for a,b in [(mrows,m),(crows,c),(vrows,v)]:
 assert len({x['id'] for x in a})==len(a)
 assert all(x['original']==raw for x,raw in zip(a,b))
 assert all(x['family_wide_exclusion'] is False for x in a)
assert len(mrows)==108 and len(crows)==136 and len(vrows)==254
assert vdoc['coverage_counts']==dict(collections.Counter(x['series'] for x in v))
audit={'review_date':'2026-10-11','check':'document coverage and preservation only; not scientific revalidation','counts':{'historical':108,'catalogue':136,'program_version_records':254,'older_retained_subset':71,'first_round_operations':16},'overlapping_inventories_not_added_as_unique_mechanisms':True,'all_original_rows_and_parameters_preserved':True,'unique_ids_per_inventory':True,'missing_review_rows':[],'invented_family_wide_exclusions':0,'scientific_records_executed':0,'science_files_changed':False,'publication_requested_in_this_turn':False,'source_scope':'Recovered source ledgers/version register and previously read reports; no claim every historical trajectory has now been independently audited','broad_core_status_counts':dict(collections.Counter(x['broad_core_status'] for x in mrows)),'version_scope_counts':dict(collections.Counter(x['review_status'] for x in vrows))}
save('COVERAGE_AUDIT.json',audit)
print(json.dumps(audit,ensure_ascii=False,indent=2))
