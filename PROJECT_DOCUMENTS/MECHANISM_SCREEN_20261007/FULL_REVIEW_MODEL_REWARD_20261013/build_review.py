"""Document-only review of every original row; does not import or run a learner."""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SOURCE = REPO / 'arithmetic_feedback_rule_discovery_dev_v1_20261013/ops/read_only_review_20261013/inventory/FULL_INVENTORY_REVIEW.json'
REGISTER = REPO / 'PROJECT_DOCUMENTS/MECHANISM_SCREEN_20261007/VERSION_REGISTER_20261011/VERSIONS.json'
full = json.loads(SOURCE.read_text())
versions = json.loads(REGISTER.read_text())['versions']
version_by_id = {v['version_id']: v for v in versions}

# Explicit disposition of all 108 mechanism records, rather than a keyword
# filter or an inference that absence from the 71-item subset means failure.
HISTORICAL = '''
001 EXCLUDED ELM细胞按用户要求退出；保留原技术结果，不将其改写为整族不可能。
002 EXCLUDED 私有向量ELM同样退出，不以“标量通信”名称重新纳入。
003 EXCLUDED 固定时间常数ELM退出；SOURCE-CELL的E6_FIX无优势，仅属技术世界证据。
004 EXCLUDED 线性树突ELM退出；不能换内部算术后作为非ELM新候选。
005 REFERENCE 简单泄漏递归是成本参照，尚无当前任务规则学习优势。
006 DEFERRED 阈值适应没有对应当前内容学习缺口的直接证据；不为生物逼真度增加状态。
007 PRIMARY 普通门控递归可作为D1母体；旧稀疏GRU并非当前密集GRU的等参数验证。
008 DEFERRED RG-LRU的门控递归部分可做未来成本参照；不引入完整Griffin的注意力部分，不从spike AUC推算术能力。
009 INCOMPATIBLE FLM复合版本含检索，且旧语言对照不可靠；未得到可直接继承的独立四则机制。
010 DEFERRED 通用多时间尺度不等于ELM，但增加状态本身没有当前收益；ELM实现已退出。
011 DEFERRED 旧异质性收益被平均率控制削弱，不是解决目前获取不足的优先机制。
012 PRIMARY 分开的保留与写入门是D1原则；普通GRU已有门，不把再次使用门称为新发现。
013 DEFERRED 动力学与homeostasis有未完成控制；可作数值稳定构件，不优先增加生理状态。
014 DEFERRED LIF/HH等复杂细胞没有已指出的必要功能缺口，不为真实性换细胞。
015 ENGINEERING 类型模板和独立状态是实现约束，不是算术学习候选。
016 DEFERRED 固定稀疏连接可能改善成本，视觉拓扑不移植；图稀疏不保证事件稀疏或规则能力。
017 DEFERRED 改拓扑不会自动产生信用分配；现有P/T/S负结果不支持重新优先扫换边。
018 DEFERRED 新边消融是归因工具，非已经证明有效的学习机制。
019 INCOMPATIBLE 旧cosine-kNN/hub路由有因果问题及内容检索限制；不直接复活旧版本。
020 DEFERRED 无标签afferent/Oja可调整感觉表示，未证明能提取算术关系；不替代答案/奖励信用。
021 INCOMPATIBLE 隐藏clean-latent目标不同于用户允许的真实答案反馈，不能暗中提供。
022 ENABLER D4要求表示和读出共同学习；旧地址/内容FF-FL实验不是题内encoder/decoder保留的直接证据。
023 STOP_RECIPE 旧transport未恢复oracle能力，且当前联合CE模型没有同一RLS坐标问题，不重跑旧补偿。
024 ENABLER 隐层反传目前允许；D4可借通用学习适配，但旧回收器与新题面保留必须分开登记。
025 INCOMPATIBLE 预设算术operator mixture不允许；旧复合路由演示也未隔离通用组合机制。
026 EXCLUDED 注意力/hub/Sinkhorn等复合架构不纳入，不能通过改名绕过限制。
027 DEFERRED 区分QKV检索与V23/V24无注意力快慢输出矩阵；后者旧版本无完整优势，不能一刀切称为注意力。
028 INCOMPATIBLE Hopfield内容检索不允许；顺序工作状态需按D1另立版本，不能继承episodic-tape演示成绩。
029 EXCLUDED 旧Transformer语义因子架构退出；不能作为通用学得表示已成功的证据。
030 PRIMARY D3可借奖励预期/critic思路，但预训练教师、语言能力和SAC复合演示不能继承；新critic仅学真实0/1。
031 ENABLER D4联合CE/序列奖励优化是工程构件，允许反传但不称为昆虫局部规则；当前梯度非零不等于已优化好。
032 ENGINEERING 并行扫描与吞吐优化需保持行为，不作为规则泛化机制。
033 REFERENCE 普通分类头是获取/干扰参照，不能把每次分类性能当作永久知识。
034 REFERENCE 固定特征RLS历史阳性保留；移动特征不是精确后验，数值加法版本有人工先验，不直接移植。
035 STOP_RECIPE 简单快慢输出相加已有组合失败先例，不重复堆输出存储。
036 STOP_RECIPE 固定特征加权巩固低于旧基线，不以当前标量奖励场景将其原样复活。
037 STOP_RECIPE 额外写入质量与累积统计bonus未改善，不复跑相同配方。
038 DEFERRED 保护改善旧保持但限制新获取；目前已教内容都不稳，不优先保护错误或频率策略。
039 DEFERRED 不确定性不同版本保留；V31有无效/未决结果，不能整族否定，也不优先解决噪音。
040 DEFERRED 旧投影有刚性代价；不依靠冻结更多方向来生成缺失的运算能力。
041 STOP_RECIPE 已测双/三状态耦合不能自动提升四则；原参数退出，整族结论不扩大。
042 STOP_RECIPE 已测feedback耦合无完整优势，与序列动作资格迹D2不同。
043 STOP_RECIPE cascade4/8保留获取/保持代价，不原样复跑。
044 DEFERRED surprise释放与遗忘有损害先例；最终反馈0并不表明旧知识应被删除。
045 DEFERRED 大地址层可记忆事实，尚无规则获取收益，不扩大查表底物。
046 DEFERRED slot分配解决容量与淘汰，不能替代算术学习或原始字节表示。
047 ENGINEERING 压缩需算完整状态，旧quantized命名不代表总压缩。
048 DEFERRED bank隔离有真实关联收益；当前无标签的规则学习与内容路由限制使其不是直接候选。
049 DEFERRED reinstatement缺可区别上下文控制；当前不是忘掉已学会的完整四则规则。
050 SUPPORT D5保留有界、收费、跨难度再激活；旧reservoir/recent/REPLAY-P已测，不冒称未试，更多曝光不等于规律。
051 DEFERRED 小cache几乎无增益；内容相似检索还受无注意力约束。
052 STOP_RECIPE spillover/规范化的具体收益被控制削弱，不重启写质量扫描。
053 DEFERRED Oja表示变化可破坏旧坐标，没有标签/奖励信用时不保证形成任务表示。
054 STOP_RECIPE novelty特异性被随机/率匹配控制解释，关闭旧选择启发式。
055 STOP_RECIPE 竞争plasticity未建立可靠前沿收益；不因名字像生物机制再次优先。
056 DEFERRED fast-minus-slow感觉对比的其他任务效应不是当前字节题面保留的证明。
057 STOP_RECIPE variance-salience未通过旋转等控制，退出通用学习候选。
058 DEFERRED 完整协方差在旧任务有收益，但二次状态与数值表示先验不能解决任意字节条件生成。
059 INCOMPATIBLE 手工lag/位置特征不作为本轮学得表示；顺序保留改由通用核心学习。
060 STOP_RECIPE 四稀疏compartment协方差旧保持差，不作为新想法。
061 ENABLER 残差参考漂移是D4约束；奖励预期是预测的0/1，不是重建旧四通道残差存储。
062 INCOMPATIBLE 精确字典是记忆反例，非端到端可泛化核心。
063 DEFERRED 读权限有具体工程效应；不是题面相关运算能力的来源，内容检索需排除。
064 DEFERRED Bayes arbitration未过具体资格，不以“Bayes”命名保证改进。
065 ENGINEERING corroborated admission解决显式存储clutter，不是当前主要任务。
066 REFERENCE admission对照保存，不计新机制。
067 DEFERRED 超位置/哈希存在负载与泛化取舍，当前不扩大固定地址编码。
068 ENABLER 当前成功不能直接判定未来巩固需求；作为D4/训练设计边界，不另加bank。
069 DEFERRED utility replacement证据无效/弱，不转向数据库替换问题。
070 REFERENCE 旧固定历史读诊断范围有限，不能证明新题算法获取。
071 DEFERRED 局部Bayesian权限未合格，高负载获取失败，不能当默认仲裁器。
072 REFERENCE 旧slot容量诊断不支持当前“规则缺失是容量已满”的解释。
073 DEFERRED 稀疏/规范化参数扫描有坐标漂移混杂，不能单凭几何阈值筛选算术机制。
074 STOP_RECIPE PHASE-FROZEN/LIVE/LAG已测试，不当新缺失实验。
075 STOP_RECIPE 旧distributed consolidation结构弱于incumbent，只保留小范围正确corroboration效应。
076 DEFERRED factor输出是压缩取舍，当前需要先学对多字节内容，不先增加编码复杂度。
077 DEFERRED 正配对invariance有窄任务证据；不能给算术交换关系或将同答案题强行等同。
078 STOP_RECIPE PAIR+CORR低于既定增益且事件供给弱，原样不再优先。
079 STOP_RECIPE learned pathway责任分配增加串扰，关闭确切构造。
080 STOP_RECIPE 已测signed保护损伤表示，不复跑相同操作。
081 STOP_RECIPE one-byte surprise几乎放行全部conflict，不当有效选择性学习门。
082 ENGINEERING 稀疏值压缩只有单元等价证据，与学算术无关。
083 ENGINEERING 图缩减是结构结果，不能先把功能未成立的核心继续削小。
084 DEFERRED DAN relay压缩误差与节点来源保留，不将8个pattern解释为学习四则的模块。
085 STOP_RECIPE unsigned空间反馈负面不证明时序教学不可能，但旧线性空间保护版本不重跑。
086 STOP_RECIPE mean-normalized output gain旧结果null，不作为奖励信用修复。
087 STOP_RECIPE input-only写掩码无建立优势，不复活旧mask。
088 STOP_RECIPE joint写掩码无广泛优势，稀疏mask不是学习规律。
089 ENGINEERING 解剖lobe描述不等于独立运算或记忆模块，保留结构出处。
090 STOP_RECIPE scalar APL旧负结果，不能通过多加循环次数原样重试。
091 DEFERRED raw PN-KC-MBON可获取但保持/复用弱，接口有工程codebook，不移植为原始字节答案生成器。
092 DEFERRED count异质性有取舍但shuffle匹配，不将连接数推为学习/时间常数。
093 STOP_RECIPE radius0.25 unsigned线性循环增加损失；不重复加环，也不推论所有非线性循环无效。
094 DEFERRED 原始循环结构是真实可能性但未成功实现功能；D1不用连接存在作为算法形成证明。
095 DEFERRED 完整原始图整合不是当前必要步骤，不把新核心目标变成connectome复现。
096 PRIMARY D2/D3最相关：动作/序列特异资格与真实奖励调制；旧即时非负倍率、Z1和四动作反馈不等于当前端到端序列信用。
097 DEFERRED Fly-2时序转存不是算术规则生成器，且fast/slow身份来自外部文献而非原始图。
098 DEFERRED 信息门未经科学运行；绝对惊讶或预测比率不提供正确答案，不优先限制弱学习信号。
099 DEFERRED 多horizon预测或许有额外观测信号，但旧ALL-byte训练不是普遍成功先例，当前不增加新辅助目标。
100 DEFERRED plasticity quota只能分配已有信用，不能补上缺失的信用，未来窗口方案还需因果化。
101 DEFERRED targeted extinction适用于修订而非未学会的新规律，不以错误回答直接全局遗忘。
102 PRIMARY D1的有界、学得工作状态最贴近当前缺口；禁止KV查询、人工操作数槽位和解题栈，新固定题面条件通路尚未测。
103 DEFERRED phase admission是旧显式slot工作量观察，当前不返回人工阶段掩码。
104 REFERENCE class-specific写lesion是oracle诊断，不部署为learner的教学选择器。
105 DEFERRED reusable metric不同于完整算术规律；同答案不等于同身份，不供应算术等价关系。
106 DEFERRED frozen/slow残差参考有条件性负先例，当前不另开transport修补链。
107 ENGINEERING ANN/哈希检索优化不是新学习核心，内容查询部分受禁令限制。
108 DEFERRED 自然语料与NLP留在后续，不改变当前未见四则与混合题终点。
'''

hist = {}
for line in HISTORICAL.strip().splitlines():
    n, status, reason = line.split(' ', 2)
    hist['M'+n] = dict(priority=status, reason=reason)
assert set(hist) == {f'M{i:03}' for i in range(1,109)}

DIRECTIONS = {
 'D1': {'name':'学得题面保留与独立生成工作状态','role':'PRIORITY_HYPOTHESIS',
        'historical':['M007','M012','M102'],'subset':['U042','U060'],
        'operation':'普通递归/门控保留一个学得题面状态，生成状态逐字节演化且始终受其普通仿射条件影响；不检索位置或KV。',
        'existing_limit':'旧GRU已递归，增加8次计算的旧R8也失败；本方向不是再增加次数或ELM私有状态。题面快照条件化是未测新版本。'},
 'D2': {'name':'整段实际动作的延迟资格与信用','role':'PRIORITY_HYPOTHESIS',
        'historical':['M096'],'catalogue':['C050','C051','C077'],'subset':['U011','U012','U019'],
        'operation':'在生成整串回答时保存有界动作特异信用，真实0/1到达后更新参与生成的可学习状态/参数。',
        'existing_limit':'旧Z1没有改善且修正E3较差；A_ELIG证据来自其他底物。当前答案监督已覆盖题内全部梯度，不能把失败说成32-event信用截断。V24/V25只有四动作反馈。'},
 'D3': {'name':'奖励预期调制与可学习预测基线','role':'PRIORITY_HYPOTHESIS',
        'historical':['M030','M096'],'catalogue':['C076','C128'],
        'operation':'只从实际奖励学习预期，用实际奖励与预期的差异调制动作信用；不提供正确答案或数值误差。',
        'existing_limit':'旧V24/V25已有reward-minus-chosen-value；不是新发现，也尚未证明可训练通用字节表示。旧预训练教师和SAC复合结果不继承。'},
 'D4': {'name':'表示与输出联合学习及优化一致性','role':'ENGINEERING_ENABLER',
        'historical':['M022','M024','M031','M061'],
        'operation':'在可见答案阶段联合训练categorical encoder/readout；在scalar阶段使用与真实动作和奖励一致的目标。',
        'existing_limit':'当前SOURCE-CELL已联合CE训练但获取仍弱；方向是优化/结构的配套约束，非额外已成功的昆虫机制。固定特征RLS历史成功不能继承。'},
 'D5': {'name':'收费的交错重复与有界再激活','role':'CONDITIONAL_TRAINING_SUPPORT',
        'historical':['M050'],'catalogue':['C115','C116','C117'],'subset':['U047'],
        'operation':'真实观察的重复和跨难度交错，不用模型猜测伪造答案；所有存储、选择、曝光和算力计费。',
        'existing_limit':'reservoir/recent/REPLAY-P/REPLAY-U/COV-REPLAY已测；旧纯字节模型充分重复可记已教题，却未学会可靠未见乘法。不是主机制或未试的新办法。'}
}

catalogue_primary={50:['D2'],51:['D2'],76:['D3'],77:['D2'],128:['D3']}
catalogue_enabler={21:['D4'],74:['D3','D4']}
catalogue_support={115:['D5'],116:['D5'],117:['D5']}
catalogue_controls={17,18,20,27,28,29,34,45,46,62,66,75,79,90,91,107,131,132}
catalogue_stopped={2,5,8,9,10,12,13,19,43,44,52,53,58,60,64,73,87,88,93,94,96,97,98,99}
catalogue_incompatible={6,11,22,38,80,81,82,83,84,104,105,106,109,133,134,135,136}
catalogue_note={
 2:'FE1具体版本已无集成增益，不恢复编码器轮换。',
 12:'P1原版本负面及后继screen未晋级；不因原目录写BUILD当成未测试。',
 13:'P2已测试；binary Oja不提供新的稳定语义机制。',
 14:'P3技术未合格，不是科学否定；当前也不分配修复运行预算。',
 23:'T1有效E1无增益，旧E3无效；当前不恢复同图搜索。',
 24:'T2有效E1未建立增益，旧E3无效；保留参数备选但不进入下一轮。',
 25:'T3有效E1未建立增益，旧E3无效；不按目录pending重新晋级。',
 30:'S1旧E1无增益；P×S另属新操作，不继承成功。',
 31:'S2/S2-R8是条件备选，TOPO-G相关权重/换边配方已有screen无晋级。',
 32:'S3原样停止；cue-clock匹配移植未晋级，不转为参数修补。',
 33:'S4/S4-R8仍低优先级；旧E3无效，不计整族否定。',
 50:'多帧资格值得借用，但不是当前GRU warmup缺梯度的证明。',
 51:'需要生成动作与参数的序列信用新版本，不能直接把KC地址trace移植并宣称等价。',
 52:'Z1已测无E1改善且修正E3较差；动作信用D2与此坐标EMA不是同一操作。',
 53:'Z2原配方无改善并损伤taught guards，不重新调resource阈值。',
 63:'V23/V31不确定性与SUPPORT-DIR分别处理，不用一个失败总括整族。',
 65:'native Full151 allocation是历史底物组成，当前没有其四则规则获取证明。',
 73:'target/opponent结果依底物而异；不是任意257字节生成头的直接教学政策。',
 76:'DAN式结果调制可借用；final仅可用真实0/1与学得预期，不给正确动作向量。',
 77:'资格必须记录真实生成动作，终端反馈才能回到原题和序列参数。',
 80:'V9.9A/B有限E3是真实的synthetic-choice证据，非四则整串输出资格。',
 87:'R1已测E1负差；novelty不创造正确反馈。',
 88:'R2已测E1无改善，修正E3区间跨零；不要混称新的sequence reward error。',
 89:'R3/R_center保留为旧参照，四答案/工程地址不直接适用当前任务；shared删减结果不是新规则机制。',
 117:'旧reservoir/recent、优先REPLAY-P/均匀REPLAY-U都有版本，不宣称未尝试。',
 128:'实际0/1的预测与历史反馈可借用，不能虚构奖励或继承pretrained teacher。',
 133:'仍为四通道relation organ，不满足通用字节接口。',
 134:'两store校准读法未证明解决当前内容学习，不引入独立相似检索通道。'
}

def cat_review(row):
    n=int(row['id'][1:]); old=row['original_review_row']
    if n in catalogue_primary: status, ds='PRIMARY',catalogue_primary[n]
    elif n in catalogue_enabler: status, ds='ENABLER',catalogue_enabler[n]
    elif n in catalogue_support: status, ds='SUPPORT',catalogue_support[n]
    elif n in catalogue_controls: status, ds='REFERENCE',[]
    elif n in catalogue_stopped: status, ds='STOP_RECIPE',[]
    elif n in catalogue_incompatible: status, ds='INCOMPATIBLE',[]
    else: status, ds='DEFERRED',[]
    reason=catalogue_note.get(n)
    if reason is None:
        reason={'PRIMARY':'仅借用一般信用/反馈操作；旧系统不是完整字节四则成功配方。',
                'ENABLER':'配套的表示/优化操作，不单独算新科学候选。',
                'SUPPORT':'可能帮助训练与保持，但不替代规则获取；旧配方与新课程分开。',
                'REFERENCE':'基线、对照或读出诊断保留，不计新机制。',
                'STOP_RECIPE':'已有确切配方/相关控制不足以支持再次优先投入；不扩大为家族否定。',
                'INCOMPATIBLE':'旧工程表示、固定候选输出或内容检索不能直接进入当前端到端无注意力任务。',
                'DEFERRED':'该行的生理、路由、结构或巩固操作没有直接补上当前内容获取与序列奖励信用；不因未被整族否定而排期。'}[status]
    return dict(priority=status, directions=ds,reason=reason,original_pending_not_used_as_current_verdict=True)

def vreview(v):
    name=v['name']; series=v['series']; op=v['operation']
    base=dict(priority='DEFERRED',directions=[],reason='旧关联/修订/路由配置，不是当前原始字节四则的即插即用机制；保留精确参数与上下文。')
    if series in {'SAMPLE_GEN_V2','FEEDBACK_EFFICIENCY_V1','OBSERVATION_SUPPORT_V1','SUPPORT_DIRECTION_V1'}:
        return dict(priority='INCOMPATIBLE',directions=[],reason='此具体版本依赖操作数角色、可加模型与序位/数值输出；已采用或负面结论保留，当前禁止人工算术表示，不能直接复用。')
    if series=='V30_DIRECTION':
        return dict(priority='STOP_RECIPE' if name in {'D','P'} else 'REFERENCE',directions=[],reason='原directional writer未被选中；规范化/打乱/冻结版本是对照，不是新候选。')
    if series=='V31_UNCERTAINTY':
        return dict(priority='DEFERRED',directions=[],reason='保留各horizon/frozen/shuffled原范围；contrast/lagged无效和observable未决不能总括整族。当前不以不确定性救援噪音处理为主线。')
    if series=='V25' and name.startswith('reward_'):
        return dict(priority='REFERENCE',directions=['D2','D3'],reason='源码已有四动作探索、chosen-action奖励及reward-minus-predicted-value；只借反馈/动作标记原则，不继承分类特征和bank读出，不称为首次测试0/1。')
    if series=='V25' and name.startswith('target_'):
        return dict(priority='REFERENCE',directions=['D2'],reason='延迟target/fifo/broadcast已有版本；四标签目标不是最终0/1，known_delay仅诊断，不作为学得时序归因或整串字节信用成功。')
    specs=op.get('memory_specs',[op]); families={s.get('family') for s in specs}
    kinds={s.get('kind') for s in specs}
    if 'replay' in kinds or 'replay_consolidation' in kinds or 'repeat_current' in kinds:
        return dict(priority='SUPPORT',directions=['D5'],reason='旧有界reservoir/recent/current/fastslow操作已测；可以作为收费训练对照，不自动加入快慢存储，也没有当前未见四则效率成功。')
    if 'budget24' in kinds and any(s.get('params',{}).get('alternative')=='replay' for s in specs):
        return dict(priority='SUPPORT',directions=['D5'],reason='旧等存储预算回放参照；预算等额不是新四则泛化证据，重放真实观察并计费。')
    if 'fastweights' in kinds:
        return dict(priority='STOP_RECIPE',directions=[],reason='源码是H@(V+fast)的快慢输出矩阵，没有QKV检索；不因注意力禁令剔除，但旧完整增益不足，不原样复跑或把静态输出误称学习规则推理。')
    if 'retrieval' in kinds:
        return dict(priority='INCOMPATIBLE',directions=[],reason='源码hard/soft相似缓存读为内容查询；当前无注意力及端到端表征要求下不直接复用。')
    if 'reference' in families:
        return dict(priority='REFERENCE',directions=[],reason='原 ordinary/slow/ridge/anatomy 等基线与控制保留；它们的分类终点不是未见四则自输出。')
    if kinds & {'consolidation','cascade','active_forgetting','surprise','replay_consolidation'}:
        return dict(priority='STOP_RECIPE',directions=[],reason='原耦合/阶梯/遗忘/惊讶版本已有获取—保持—修订取舍或负面结果；当前先形成内容能力，不原样增加慢存储或抑制学习。')
    if 'importance' in kinds or 'projection' in kinds:
        return dict(priority='DEFERRED',directions=[],reason='有局限的保护/释放收益仍保留；保护弱或错误策略不能自动形成新规律，先不投入强保护/投影扫描。')
    if series=='V25' and (name.startswith('dual_') or name.startswith('split_') or name.startswith('context_')):
        return dict(priority='DEFERRED',directions=['D4'],reason='地址与内容/坐标分工的关联证据保留，但这是跨学习坐标问题，不是题内固定条件状态；旧原生bank/routing表示不直接移植。')
    if kinds & {'allocation','allocation24','bank24','context_gating','silent_memory','sparse_specialization','recycling','turnover24','bayes','budget24'}:
        return dict(priority='DEFERRED',directions=[],reason='保留精确容量/保护/路由/不确定性参数和原tradeoff；底物是旧特征与固定分类输出，不能凭相似名称纳入当前字节规则学习。')
    return base

hist_to_dirs={m:[d for d,z in DIRECTIONS.items() if m in z.get('historical',[])] for m in hist}
views={}
for key,view in full['views'].items():
    rows=[]
    for row in view['rows']:
        rid=row['id']; original=row['original_review_row']
        if key=='historical108':
            decision=dict(hist[rid],directions=hist_to_dirs[rid])
        elif key=='catalogue136': decision=cat_review(row)
        elif key=='versions254': decision=vreview(version_by_id[rid])
        elif key=='subset71':
            refs=original.get('source_refs',[]); ref_dec=[]
            for ref in refs:
                if ref in hist: ref_dec.append(hist[ref])
                elif ref.startswith('C') and ref[1:].isdigit():
                    cr=next(x for x in full['views']['catalogue136']['rows'] if x['id']==ref);ref_dec.append(cat_review(cr))
            ds=[d for d,z in DIRECTIONS.items() if rid in z.get('subset',[])]
            if rid=='U045':
                decision=dict(priority='INCOMPATIBLE',directions=[],reason='注意力式KV部分被禁止；无注意力fast输出版本已有失败/取舍，不能由旧PRIORITY_DESIGN自动晋级。')
            elif rid=='U041':
                decision=dict(priority='DEFERRED',directions=[],reason='通用多时间尺度不等于ELM；ELM实例退出，其他版本没有当前收益，不优先。')
            elif ds:
                roles=[DIRECTIONS[d]['role'] for d in ds]
                status='PRIMARY' if 'PRIORITY_HYPOTHESIS' in roles else 'SUPPORT'
                decision=dict(priority=status,directions=ds,reason='按明确原来源借用一般操作，原旧实例/原优先级不自动继承；见对应D方向的操作与已有边界。')
            else:
                status='STOP_RECIPE' if any(x['priority']=='STOP_RECIPE' for x in ref_dec) else 'DEFERRED'
                decision=dict(priority=status,directions=[],reason='不按71子集旧排名自动晋级。'+(' '.join(x['reason'] for x in ref_dec) if ref_dec else '源条目未形成当前内容获取/奖励信用优势，不恢复参数备选或结构轮换。'))
            if rid=='U018':
                decision=dict(priority='REFERENCE',directions=['D3'],reason='既有A_NATIVE_OBSERVATION是有限到来字节E0/E1基线；它不是最终0/1序列学习或四则E3成功。')
        else:
            status=original.get('budget_status','DEFERRED')
            priority='STOP_RECIPE' if status=='STOP_RECIPE' else 'REFERENCE' if status=='REFERENCE' else 'DEFERRED'
            decision=dict(priority=priority,directions=[],reason=original.get('reason','')+' 当前不将此四store/KC旧配方直接移植为通用字节四则候选。')
        rows.append(dict(id=rid,original_review_row=original,current_model_reward_review=decision,
                         new_run_authorized=False,new_scientific_family_exclusion=False))
    views[key]=dict(source=view['source'],rows=rows,
                   current_priority_counts=dict(Counter(x['current_model_reward_review']['priority'] for x in rows)))

review={
 'schema':'FULL_MECHANISM_REVIEW_MODEL_AND_REWARD','date':'2026-10-13',
 'goal':'Retained exact own-output unseen four-operation and mixed arithmetic per all training samples and compute, learned from real answers during warmup and ultimately scalar 0/1 without arithmetic rules, attention or Transformers.',
 'counts':full['counts'],'counts_warning':'These are overlapping historical views, not additive or unique surviving candidates.',
 'all_original_rows_preserved':True,
 'scientifically_validated_end_to_end_unseen_four_operation_core_count':0,
 'direct_drop_in_positive_recipe_identified':False,
 'scientific_survivor_count':None,
 'directions':DIRECTIONS,
 'directions_are_not_a_survivor_census':True,
 'latest_evidence':[
   'arithmetic_source_derived_kernel_dev_v1_20261013/PREPARATION_REPORT.md',
   'arithmetic_stream_transfer_dev_v1_20261013/PREPARATION_REPORT.md',
   'arithmetic_kernel_no_attention_dev_v3_20261013/ops/read_only_lineage_review_20261013/REVIEW.md',
   'arithmetic_feedback_rule_discovery_dev_v1_20261013/ops/read_only_review_20261013/FINDINGS.md',
   'addition_sample_generalization_dev_v2_20261011/results/REPORT.md',
   'addition_feedback_efficiency_dev_v1_20261011/results/REPORT.md',
   'PROJECT_DOCUMENTS/MECHANISM_SCREEN_20261007/FIRST16_SCREEN.json',
   'PROJECT_DOCUMENTS/MECHANISM_SCREEN_20261007/VERSION_REGISTER_20261011/sources/03_elm_fly_local_memory_v24.py',
   'PROJECT_DOCUMENTS/MECHANISM_SCREEN_20261007/VERSION_REGISTER_20261011/sources/05_elm_fly_address_content_v25.py',
   'PROJECT_DOCUMENTS/MECHANISM_SCREEN_20261007/VERSION_REGISTER_20261011/sources/04_REPORT.md'],
 'clarifications':[
   'A file named elm_fly is not automatically an ELM cell: memory/update rules must be identified by operation.',
   'V23/V24 FastWeightsMemory is a decaying output matrix, not QKV attention. Its exact recipes remain low priority on evidence, not by an attention label.',
   'V24/V25 already tested scalar chosen-action reward and eligibility; four-action associative performance is not end-to-end byte sequence credit or arithmetic generalization.',
   'Current SOURCE-CELL W already had actual correct reply/EOS supervision and full frame credit; insufficient answer availability or 32-event truncation cannot explain those technical trajectories.',
   'U042 names in the historical 71 subset and the OBSERVATION_SUPPORT experiment are different operations. Likewise SUPPORT-DIR and historical bayes_reopening must not be conflated.',
   'A learned prompt snapshot/conditioned decoder is a new generic version, not an existing successful biological scratchpad or address/content split.',
   'Partial positive synthetic choices, retention and revision remain valid within their source scope. Do not convert them into four-operation capability or erase them on present-task failure.'],
 'biological_sources':[
   {'url':'https://www.nature.com/articles/s41593-020-0607-9','supports':'Recurrent sensory/output feedback to DANs and integrated predictions in larval associative learning; not arithmetic algorithms.'},
   {'url':'https://pubmed.ncbi.nlm.nih.gov/31230716/','supports':'Timing-sensitive bidirectional odor-association plasticity via dopamine receptors; not a demonstrated generic sequence credit rule.'}],
 'views':views,
 'training_executed':False,'science_sources_modified':False,'new_launch_authorized':False}

def write(name,obj):
    (HERE/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
write('FULL_REVIEW.json',review)
for k,v in views.items():write(k.upper()+'.json',v)

# Verify coverage and preserve original fields rather than silently recoding a
# frozen scientific outcome as a new exclusion. These are document assertions.
checks=[]
for k,v in views.items():
    before=full['views'][k]['rows']; after=v['rows']
    assert len(before)==len(after)==full['counts'][k]
    assert [r['id'] for r in before]==[r['id'] for r in after]
    assert len({r['id'] for r in after})==len(after)
    assert all(a['original_review_row']==b['original_review_row'] for a,b in zip(before,after))
    assert all(r['current_model_reward_review']['reason'] for r in after)
    checks.append(dict(view=k,rows=len(after),originals_preserved=True,all_rows_assessed=True))
assert {r['id'] for r in views['versions254']['rows']}==set(version_by_id)
assert hist['M001']['priority']=='EXCLUDED' and hist['M026']['priority']=='EXCLUDED'
assert vreview(version_by_id['V24/fastweights_residual'])['priority']=='STOP_RECIPE'
assert 'attention' not in vreview(version_by_id['V25/reward_existing'])['reason'].lower()
write('COVERAGE_AUDIT.json',dict(accepted=True,scope='Document coverage and exact original-row preservation, not scientific revalidation',
 checks=checks,independent_mechanism_count_not_claimed=True,training_records_executed=0,science_sources_changed=False))
write('SUMMARY.json',{k:v for k,v in review.items() if k!='views'} | {
 'priority_counts_by_view':{k:v['current_priority_counts'] for k,v in views.items()},
 'full_review':'FULL_REVIEW.json','coverage_audit':'COVERAGE_AUDIT.json'})
print(json.dumps({'accepted':True,'counts':full['counts'],'priorities':{k:v['current_priority_counts'] for k,v in views.items()},'training_records':0},ensure_ascii=False))
