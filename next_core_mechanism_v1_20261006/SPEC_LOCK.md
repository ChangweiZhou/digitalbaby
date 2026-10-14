# MiniFly 下一版持久核心机制实验 v1

日期：2026 年 10 月 6 日，America/New_York。状态：**设计已完成两路独立审阅并修订；尚未实现资格化、冻结源码或授权正式运行。**

本实验把此前的 Q/P/S 候选计划与 REL2 提议合并为一个事前封顶的计划。目标是：在相同字节经验、反馈次数和模拟时间下，提高核心的获取、保持、修订或有限复用，并检验对应能力的计算效率。新增机制与 Full151 内部结构可以改变；每项改变必须有明确 parent、接口、成本和可关闭的具体主张。

## 1. 阶段目标和验收边界

主模块继续使用已经建立的 lifetime 与简单关系 reuse 两项任务。Latin 表格补全是单独注册的能力边界模块，不能变成主模块采用候选的额外必过门槛。两模块全部事前承诺，不能因为先看到其中一项结果而决定是否运行另一项。

本轮不增加自然语言、未教算术、开放字节生成、无提示答题、十亿神经元或 AGI 要求。答案教学仍来自实际到来的答案字节。输入、反馈、输出和学习状态之间的关系必须透明，但这不是无教学信号学习的验证。

同一个候选的完整结构在不同任务的独立生命史上测试。**不宣称同一份已经训练过的状态同时掌握全部任务。** 模型不能用 evaluator 的任务、阶段、已教/未教或表格坐标标识来切换路径。

允许的证据：

| 测试 | 证据层级 | 可作的结论 |
| --- | --- | --- |
| 接口、出生、禁写、时钟与恢复 | E0 | 本次实现满足注册合同 |
| lifetime 已教旧、新、修订题 | E1 | 获取、保持、修订与干扰 |
| reuse 从不强化的符号组合 | E3 | 该简单符号类关系族内的有限复用 |
| Latin 从不强化的表格格子 | E3 | 注册四乘四表格族内的补全 |

本轮没有新的 E2 确认任务。此前相同 CONTENT key 的空格或前缀变体成功不等于新关系泛化。只有实际进入模型的新增表示和模型自己产生的答案可以记入候选表现。

## 2. 对此前证据的处理

1. V1/R_center 的获取、保持与修订基线保留；原 V2 的 `RETAIN_V1` 裁决不修改。
2. ERROR 已改善既有任务中的修订，但不能据此默认视为全面优于 V1。它在新 Latin 任务上没有修复负迁移。
3. ERROR+Q-HALF 的先前分数是离线探索；本轮必须使用新世界确认，writer 的概率来源保持不变。
4. P1 原 `.05` 配置与旧 S3 的具体实现不重跑。这里是明确的低剂量或 cue-clock 移植。
5. 旧 J、centered-J 与 REL2 都涉及交互表示，但输入、时间积分和学习器不同。REL2 是一个新工程变体，不是旧 J 的恢复，也不是整个交互家族的重新认证。
6. Muse 中通道映射、signed teacher 与实际禁写的错误不能计为有效机制阴性；本轮直接将对应反例加入资格测试。
7. 108 条总账、机制家族、参数配置和物理任务数分别记账，不互相相减。本轮失败只关闭下表规定的具体 recipe/claim。

REL2 原稿的 parent 是 CENTER。本设计明确改为 ERROR，便于在同一已修订 writer 上比较 Q/P/S/REL；CENTER 仍在两模块作为配对参照。这是事前设计修改，不将 REL+ERROR 冒称原稿的 REL+CENTER，也不沿用原稿的十二项非劣性区间或结论。改动后的名册与统计家族以下文为准。

来源保存在 [此前候选计划](sources/NEXT_CORE_VARIANT_PLAN_20261005.md)、[最新 ERROR 结果审阅](sources/GITHUB_RESULT_REVIEW_20261005.md)、[收到的 REL2 原稿](sources/MINIFLY_REL2_SCALE_V1_DESIGN.md)。原文件不修改。

## 3. 本轮完整名册

共同工程底座：12 字节 cue；固定答案字母表 `0123`；shared FE0、private CONTENT；模型预测先于答案字节；相同模拟时钟、原生 fast/slow/adaptation 和最低 ASCII argmax tie rule。

| 配置 ID | 角色 | parent | 唯一新增机制或改动 | 原生 store 数 |
| --- | --- | --- | --- | ---: |
| `CENTER` | 正式参照 | R_center V1 | 无 | 8 |
| `ERROR` | 任务内参照及独立升级候选 | CENTER | private `.25-y` 改为 `pi-y` | 8 |
| `Q_HALF` | 候选 | ERROR | 只将最终 private 权重改为 `.5` | 8 |
| `P005` | 候选 | ERROR | shared Hebb，epsilon `.005` | 8 |
| `P0005` | 候选 | ERROR | 同式，epsilon `.0005` | 8 |
| `S3_CUE` | 候选 | ERROR | 使用率驱动换边；修正 rho；cue-clock | 8 |
| `S3_RAND` | 诊断对照 | ERROR | 与 S3 实际换边次数匹配的随机换边 | 8 |
| `REL05` | 候选 | ERROR | 注册 REL2 表示，alpha 写剂量 `.5` | 12 |
| `REL10` | 主要 REL2 代表 | ERROR | 同式，剂量 `1` | 12 |
| `REL20` | 候选 | ERROR | 同式，剂量 `2` | 12 |
| `REL_PERM10` | 诊断对照 | ERROR | REL10 仅写入时置换 PN 坐标 | 12 |
| `FIRST10` | 诊断对照 | ERROR | 同额外 bank 与投影，乘法改为相加 | 12 |

名册为 **两个参照、七个新增候选配置、三个诊断对照**，涉及 Q、P、S、REL 四种本轮机制方向。ERROR 同时承担参照与升级候选角色，不重复计数。`Q_HALF` 是 ERROR 同一条新学习轨迹的注册读出策略，故共 **12 个报告配置、11 个物理学习配置**。

P/S/REL 不捆绑 Q_HALF；ERROR 的 pi 不使用 combined 或 relation 分数。没有任何候选用答案、已教身份或 evaluator 标签选择 bank 权重。未注册 Q×P、P×S 或其他组合，本轮不会自动合并赢家。

三个 REL 剂量一次列齐，不允许结果后添加 `.75`、`1.5`、另一个投影 seed、temperature 或 readout alpha。只有 1× 有专门的对齐/投影对照；低、高剂量可形成行为候选，但没有本轮专门机制归因资格。

## 4. 基础接口和出生

### 4.1 数据和状态接口

模型接口只有 observed byte、timestamp、`predict()` 和 `observe_outcome(byte, learn=...)`。正常 W 分支不接收任何科学任务或阶段 ID。N 分支的 `learn` 标志是 evaluator 施加的因果干预，不能被转化为题目身份、不同初始化或结构保护信号。

每个 store 由独立 `canonical_fresh_native()` 调用出生。允许共享已锁定的只读资产，不允许出生一个 learner 后复制为全部 store。可变 fast、slow、adapt、FE0、pending 状态、P 边权和 S 结构状态必须独立；clone 的可变状态不得反向污染原对象。

canonical B/Q、genes、运行环境、源文件与所有新映射在资格后写入 `SOURCE_LOCK.json`。遇到 B/Q 不一致应停止并解决来源问题，不得悄悄用观测 digest 替换目标 digest。

### 4.2 输出与 ERROR

沿用尺度：

```text
cS = 1.4911274663291492
cP = 1.3452365735750882
pi = softmax(private_pre_outcome_values / cP), temperature = 1
```

CENTER 的 private 系数为 `.25-y`，其他正常候选为 `pi-y`，调用真实 signed 增量接口：

```text
S_new = S_nonplastic + s * (W_native(1) - W_native(0))
```

不得用 `punishment=-s` 替代，不得把 relation 任务输出路由到 2/3 而实际读取 0/1。四输出通道始终对应 ASCII `0/1/2/3`；binary relation 也保留四个通道的正常更新。

普通输出为 `shared/cS + private/cP`；Q_HALF 为 `shared/cS + .5*private/cP`。REL/FIRST 再加 `extra/cS`。额外 bank 使用 cS 是**本设计固定的单位权重工程选择**，不是从新表示已经校准得到的最优尺度。不做标签拟合、RMS 性能选参或事后调 alpha。

## 5. P 低剂量移植

只改变 shared FE0 的 PN→KC 现有非零边权，canonical support 固定。private CONTENT 不改变。

每条完整教学记录缓存反馈前的 `h` 和各 shared store 的 KC code `x`。先产生预测、执行该记录允许的监督 value 更新，再从缓存 cue 执行一次无标签边权适应。适应在 W/N 均执行，不取当前答案 byte，不依赖监督写入开关。前一条答案仍属于原 FE0 的真实 sensory history，不另设 parser 删除它。

对于 canonical 边 e=(p,k)，令 birth 入射预算 `b_k=sum_e B0_e`，birth 平均边权 `m_k=b_k/degree_k`，`v_e=B_e/m_k`，`a_p=h[pn_type_index_p]`：

```text
v_proposed[e] = max(v[e] + epsilon*a[p]*x[k], 1e-9)
w[e] = m[k]*v_proposed[e]
B_next[e] = w[e] * b[k] / sum_{e':target(e')=k} w[e']
```

epsilon 只取 `.005` 或 `.0005`。零 degree 的 KC 不更新。无权重学习的 ERROR 是静态参照。每个 shared store 独立持有图状态；记录实际更新 L1/L2/max、每 KC 入射和、支持 digest、单边占比与地址漂移。

这不是纯参数修改：从原 teach hook 移植为一次 cue-only hook 是明确合同。反馈关闭时无标签适应继续；probe 只在 disposable clone 中执行，不提交到 continuing history。

按本公式，x 只由 FE0 与 B 生成，适应不读取 value/adaptation 状态。因此相同 bytes/clocks 的监督写入干预之间，P 图、无标签证据与操作计数必须逐位一致；完整 value 状态可以不同。该一致性是资格要求，不能把隐藏的 value 反馈当作机制效果。

## 6. S3 拓扑移植与随机对照

仅 shared 路径改变。保留原 source S3 的 support-slot、搬移边权、KC 入射边数、总边数和每 KC 权重预算；不新增 P 学习，不运输旧 value 到新地址。

固定常数：`K_EVENT=24` 完整 cue 记录、`R_MAX=64` 换边/结构事件/store、每有 canonical 入边的 KC 预留 16 个新 PN partner、证据衰减 `tau=86400s`、`rho=.05*2**(-.5)`、`rho_eps=.001`、偏差 band `log(2)`。

本轮 support pool 与 tie rank 用固定工程 domain `MINIFLY-NEXT-CORE-20261006|S3` 生成，所有世界、输出 store 和 S/Srand 共享只读候选支持；不使用 fixture world ID 来产生学习操作。原 S3 的 world-keyed pool 因此不是本次精确变体。

确定性生成：对 KC k，将非 canonical partner 的 PN p 按 `SHA256(UTF8(domain|pool|k|p))` 完整 digest 排序，取前16个，hash相等以p排序。slot按PN index升序保存；canonical slot保留原CSR数据位置和边权。KC tie用 `domain|tie_kc|k`，partner tie用 `domain|tie_partner|k|p`，相等时用canonical index。竖线为字面ASCII，整数为无前导零十进制；数组写为明确little-endian固定dtype。生成物在实现资格时一次保存并锁定，不继续调用world-keyed源Support。

在每次 cue-only hook，以缓存 `a_p,x_k` 更新：

```text
decay = exp(-elapsed_since_previous_hook / 86400)
U[k] = decay*U[k] + 1[x[k]>0]
A[p] = decay*A[p] + a[p]
nev = decay*nev + 1
dev[k] = log((U[k]/max(nev,1e-12)+.001)/rho)
```

每 24 个 hook 对有原入边、有空闲候选 PN、`abs(dev)>log(2)` 且 source 可激活的 KC 排序，按 `abs(dev)` 最大优先、birth-fixed tie 打破相等，最多考虑 64 个。低使用率：当前最低 A partner 换为最高 A 空闲 partner，仅在严格上升时搬移；高使用率相反。source 可激活 mask 明确为 `kc_side in {0,1}` 且有 canonical 入边，不向恒不参与 native coding 的 KC 追求使用率。

源编码的 finite top-k 取整与实际平均激活数必须单列报告；本轮不根据观测 rho 再调整目标。结构 hook 在 native teaching 后执行，不能改变该次答案使用的缓存地址。按注册公式，S3的图、U/A/nev、操作计数与随机游标必须在相同bytes/clocks的W/N干预之间逐位一致；native value状态不同不违反该条件。

`S3_RAND` 按同世界、分支、store、事件读取 S3 已提交的实际换边数，随机挑选合法 KC、旧 partner 和空闲 partner，并搬移同一边权；采用源CounterRng算法与固定 `MINIFLY-NEXT-CORE-20261006|S3rand|store|j` key。j是架构通道0..3，key不含world、branch或标签；相同干预历史使用相同counter stream。不得重复选择同一事件已换的 KC，不得少换后仍声称匹配。

随机对照若穷尽合法候选，停止整个S3_RAND诊断配置，记 `DIAGNOSTIC_NOT_QUALIFIED`，保留原因、位置及此前receipt，不补世界、不换算法或放宽配额；其不完整数据不能用于S3−Srand比较。该预注册诊断缺失不撤销完整S3与ERROR的行为比较，其他已锁定配置继续。若是事件错位、预算守恒错误、禁写错误或其他完整性错误，则属于全批次停止条件，不能以诊断不可用掩盖bug。

这是外部 count-yoked **诊断控制**，不是可部署的自主核心。它仅匹配换边次数与预算，不匹配具体地址扰动或实际学习影响。本轮主采用规则比较 S3 与 ERROR；S3−Srand 只作描述性归因，不能据未经校正的区间宣传 homeostatic specificity。

## 7. REL2 的可实现定义

### 7.1 输入与固定映射

明确选择 `h` 为 **88 维、非负、peak-normalized 的 FE0 PN 向量**。在预测时刻，将 source FE0 clone 推进到 t 后调用 `read()`；不能在 last cue byte 时刻直接读 live FE0，也不能把 predictor 的 128-bin feature 当作 native value input。

两个固定投影 A/B 各为 44×88，每行五个不同输入，每个非零权重为 `+1/5` 或 `-1/5`。五触点是新工程常数，接近 canonical KC 平均入度 27572/5177；不是 source 唯一规定的 fan-in。

生成方法：domain为 `MINIFLY-NEXT-CORE-20261006|REL2`。对matrix名M（字面A或B）、row r和输入index j，按 `SHA256(UTF8(domain|M|r|j|select))` 完整digest排序选最小五个；`SHA256(UTF8(domain|M|r|j|sign))` 的最后一byte与1按位与，0取+1/5、1取−1/5。整数用无前导零十进制、竖线为字面ASCII，排序相等用输入index。不得使用world、cue、坐标、答案或held-out身份。构造后保存little-endian sparse arrays与digest，全部世界、剂量和FIRST对照共用，不再换seed。

```text
a = A @ h
b = B @ h
z = a*b                         # 44维，逐元素乘法；每项在[-1,1]
q[2*r]   = max(z[r], 0)
q[2*r+1] = max(-z[r], 0)       # 非负88维，保留正负双轨
q = source bytecore._norm(q)    # 正峰值归一化；全零保持全零
xREL = source encode_sparse(relation_native_model, q)
```

native KC code 为 5177 维非负 binary，采用原 `kc_side`、active_fraction、raw>0 与 top-k/tie 规则。不硬塞新的 184 激活配额，也不宣称任意输入一定激活相同数量。零 q 不回退 FE0/CONTENT。双轨、rectification、归一化和 KC 竞争都是本工程变体的一部分。

### 7.2 额外 bank 与时序

额外 bank 是四个 independently born native Full151 store，各保留完整源F151ByteBrain wrapper。原 shared/private 的 input、writer、时钟和尺度不改变，extra 分数不参与 ERROR pi。增加四份wrapper及其固定/可变状态、每byte运算全部计入资源；不在结果后换成较轻wrapper以满足采用条件。

extra 的 value feature 在每个原字节事件前由 source FE0→q→KC 得到，原 pending interval 的 nonplastic evolution照常提交。反馈前必须缓存预测 t 对应的 h/q/每个 store 的 KC；答案进入 FE0 前完成预测，写入使用该份缓存。不得压缩整条 cue 成一条 event，漏掉原字节时钟或非塑性演化。

必须同时覆盖两条地址路径：源`_features()`返回的`assoc_x/pending_x`，以及`association_value()`内部重新产生native KC的路径。仅改pending_x会变成“q地址写、FE0地址读”，仅改读出同样非法。predictor的pooled/temporal features仍按源FE0算法产生，predictor plasticity保持`learn=False`；本变体只替换value association地址，不把q偷偷送入另一个工程预测器。

只读value仍遵循源序列：FE0 clone推进t→q/KC；native clone逐调用保留原分支：有pending时依次rest `pending_t-brain_t` 和 `t-pending_t` 两段，无pending时rest `t-brain_t` 一段，不合并、不重复，保留源定义的零时长调用；随后`observed_activity`→`expression-reader`读出。普通REL与FIRST在同一预测时刻的read q/KC必须等于答案教学前缓存的q/KC；REL_PERM仅写地址不同，是事前声明的对照例外。compute feature不新增native stimulus event：继承byte()的pending/rest提交，只有observe_outcome→teach提交注册encoded event。不得为每byte q计算另加刺激而改变adaptation、gamma或写剂量。

对答案字节 y，extra 通道 j 采用与 native shared 相同的 binary teaching `r_j=1[j != y]`，直接调用合法原生 update，而不是 private 的 ERROR signed writer。只有实际 observed answer 构造 y。

REL 写剂量：

```text
relation.fly.alpha_scale = canonical_alpha_scale * dose
dose in {.5, 1, 2}
```

它通过 `raw_event_scaled -> advance73` 的源入口直接缩放 alpha-fast/slow 塑性增量。不得改 genes、punishment、活动幅度、事件次数或 read scale 来冒充 dose。相同 prestate 的直接非塑性算法不变；后续 alpha-state 改变引发的反馈差异属于真实机制作用，不要求整个生命史的 gamma/adaptation 恒相等。

### 7.3 REL 对照

`REL_PERM10`：对缓存 PN q 使用一次固定无符号 derangement，仅在写入时置换，随后重新编码 KC。读出用普通 q。以固定 domain 的 hash-rank 得到 88 个坐标的环排列，每个位置映射到环中下一位置，保证无不动点。保持 PN 范数/非零数，**不保证 KC 重叠、实际写入范数或 fast/slow 分配相同**。全部实际写量报告。

`FIRST10`：计算 `zFIRST=(A@h+B@h)/2`，之后完全相同的双轨、norm、KC、出生、教学、dose=1 和 readout。这个对照删除了乘法，仍有后续非线性。称为“相加投影的额外 bank 对照”，不得称整个 learner 为线性模型，也不宣称有效容量严格相等。

两个对照与 REL10 保存相同矩阵和状态形状，新增固定/可变 bytes 可以精确比较。三个 REL 剂量均报告行为；只有 REL10 对这两个对照的正式比较支持本具体乘法/对齐机制的归因。1× 未获机制支持不靠 0.5× 或 2× 的分数偷换。

## 8. 两个科学模块与完整工作量

### A 主核心模块

72 个新配对世界，拟用 `61006001..61006072`。每世界所有 12 个报告配置都在两项既有任务上评分；Q_HALF 复用同世界 ERROR 轨迹分数，不另跑学习。

Lifetime 严格沿用 `persistent_core_v2_trial/v2_fixture.py` 与 compact generator：32 个旧键、32 个新键、8 个旧键改答案；384 old、384 new、96 revision，共 864 条记录。最终旧题仅计 24 个未修订键；修订题按新答案计分。六个原 checkpoint、三个一天延迟和全部模拟时间不改变。

Reuse 沿用同一 v2_fixture：12 个 old taught cue、6 个 held-out cue（3对、双方向）、6 个新 cohort cue；144 old、72 new，共216记录；两次一天延迟；原内部 ChoiceOrgan 按两 option 两种顺序产生 L/R，保持 utility 和 tie 规则。

物理规模：`11*72*2=1584` world-arm-assay jobs；`11*72*(4+2)=4752` lives。Q_HALF 的144个读出分析单元是同轨迹政策比较，不额外算独立 lives 或样本。

### B Latin 能力边界模块

64 个独立新世界，拟用 `61007001..61007064`。固定运行 `CENTER, ERROR, REL05, REL10, REL20, REL_PERM10, FIRST10` 七配置。只替换配置，不改变已接受 Round 1 的 source generator、schedule 与实际 score organ。

每世界教12格、保留4格，held-out按行/列/答案均衡；old192、new192，共384教学记录；两次86400s延迟；最终模拟时刻236160s。checkpoint为 OLD_END、AFTER_NEW、FINAL；模型正常四输出 argmax 回答，不由 evaluator 选择。

物理规模：`7*64=448` jobs，`7*64*3=1344` lives。

完整名册两模块合计 **2032 个物理 jobs、6096 条 lives**，也是science的预注册最大规模。唯一允许的diagnostic缺失是§6中S3_RAND合法候选耗尽；发生时分别列出计划、已完成、停止和未启动数，禁止声称完整12配置均完成。不影响其余配置的固定n或40个主模块主张。另有独立资格测试；样本、分支、probe和失败尝试分别计数。主模块与表格模块的 world 不混作同一任务的更多样本。

### 样本和排除名册

拟定 development IDs `61005001..61005006`；没有新的科学 pilot、地板筛选或根据 held-out 分数选择参数。正式锁定前用已知本地和远程名册建立 `WORLD_EXCLUSION_REGISTRY.json`；确认拟定名单没有旧 official/pilot/preflight 重叠。若名册冲突，在任何正式 outcome 前整体修订名单并重新锁定，不运行后替换某一世界。

主模块 n=72用于有界判定，不是统计功效保证。其依据是：在下面每个联合主张 alpha=.001 时，72世界全无严重E1退化的精确binomial上界约9.15%，可以检验10%风险带；配对SD=.15时，t下界的不确定宽度约6pp。它适合分辨中等效应，不保证辨别1–2pp。表格n=64继承原提议的规模。正式开始后不得扩n，区间宽则记不确定。

## 9. 因果分支与暴露审计

| 分支 | old 监督 value | new 监督 value | revision 监督 value |
| --- | --- | --- | --- |
| W | 开 | 开 | 开 |
| N_old | 关 | 开 | 开 |
| N_new | 开 | 关 | 开 |
| N_revision | 开 | 开 | 关 |

Lifetime使用四分支；reuse使用W/N_old；Latin使用W/N_old/N_new。严禁按分支名称的字符串包含关系猜权限；采用逐阶段显式表。禁写覆盖 shared、private、extra 的全部实际监督 value update，signed 接口也不能旁路。

感官bytes、包括到来的答案bytes、原时钟与非塑性演化一致。P/S的cue-only适应不因禁监督写而关闭。分别记录监督alpha更新、无标签边权更新和结构搬移，不能将它们合成一个“总写入”或`learn=false`字段。

No-write证据来自实际native调用和相同prestate的nonplastic参考：检查 applied plastic delta，而不是要求有衰减的fast/slow数组总差为零。计数器只能证明计划调用次数，不能替代所有通道的实际写量与目标映射检查。

Probe只使用disposable clone，first emitted answer发生在该cue反馈前；probe中的P/S适应不提交到continuing learner。Latin/reuse held-out从不强化。

必须保留的countermodel：

- Lifetime任意标签可被exact-key表解决，故只记E1。
- Reuse exact/canonical key无held-out答案；位置/频率不提供固定正确类；从先前教学学出的每symbol一bit class rule可解，是允许的窄E3解释。
- Latin exact-key无held-out答案，label frequency全局均衡；正same-row/column平均对missing label反向。外部Latin缺失符号规则可解所有4格，但属于evaluator solvability witness，不接入learner；它不证明REL2可学，也不支持抽象推理宣传。
- 当前答案替换不得影响其到来前的预测、h/q/KC或P/S缓存。

## 10. 主模块测量与采用规则

每世界记录：最终intact-old、新题、修订题W准确率；各自N分支的因果效应；reuse最终held-out W、N_old、W−N_old；old_end与后续checkpoint；reuse已教旧、新题控制。每题失败分布、最差题、地址诊断保留为解释资料，不替换注册主要指标。

### 10.1 计算与知识单位

对每配置与任务，在W生命史累计core API的process CPU：feed/predict/observe_outcome/flush，以及一次注册FINAL输出扫描或ChoiceOrgan选择。包含core内部继承的复制、hash和assert成本；外部probe clone、N分支、checkpoint、receipt audit、诊断decoder和publication另外计账。所有配置相同计时边界、相同反馈信息与调用任务。

定义 `tauL` 为该world lifetime online CPU，`tauR` 为reuse online CPU。分别报告：

```text
eta_old      = 3600 * old_accuracy / tauL
eta_new      = 3600 * new_accuracy / tauL
eta_revision = 3600 * revised_accuracy / tauL
eta_reuse    = 3600 * (heldout_W-heldout_N_old) / tauR
```

这些是固定信息/教学预算下的benchmark能力率，不是互相相加的总分。它们不包括新bank增加的知识样本；新增store/alpha dose是额外操作成本，不能声称物理写量匹配。禁止把模拟一天计作真实运行一天。

`Q_HALF`使用ERROR的学习成本，额外计自己的合法读出运算；不能重复收取一条学习轨迹，也不能将读取成本设为负。计时失效、非正tau或计时接口不一致属于技术失败，不能用epsilon修分母后继续宣布效率收益。

### 10.2 共同工程保护

ERROR的parent为CENTER；七个新增候选的parent为ERROR。满足以下条件才能进入采用候选集合：

1. lifetime旧、新、修订最终W样本均值都≥90%。
2. 三项均值相对parent都≥−3pp；old/new相对CENTER也都≥−3pp。
3. reuse W−N_old与raw W的样本均值相对parent均≥−5pp；已教old_end W≥80%、final W≥70%、new W/N≥65%。这些是既有任务的competence/保护，Latin不参与。
4. 样本总W online CPU相对parent≤1.5；状态/RSS/总预算通过。没有把大于1.5的候选事后改称“效率改善”。
5. 用世界级指示变量H：候选在任一lifetime old/new/revision上比parent低超过5pp则H=1。对H的总体概率用注册alpha的单侧Clopper–Pearson上界，必须≤10%。

第1–4项是明确的**样本工程保护**，不称为总体平均非劣性置信结论；第5项检验坏世界风险。这样区分有限样本观察与概率保证，也避免把零差异t区间折叠成“绝对无损”。

### 10.3 正式能力增量与效率增量

对每个候选，分别注册old、new、revision、reuse四个联合采用profile。一个profile通过需要共同保护通过，且对应以下全部条件成立：

- 配对世界能力差的单侧t下界>0。
- 能力差样本均值达到预定实用门槛：E1任一指标≥2pp；reuse因果效应≥5pp。ERROR相对CENTER的revision profile门槛为20pp，其他profile不变。
- 对应`eta`差的配对世界单侧t下界>0。

实用门槛是事前注册的样本工程门槛，不能写成已证实总体收益至少2/5/20pp。每项原始能力与成本单列，不能只给eta或混合平均。

另注册每个候选的一个“有限reuse E3存在”联合主张：held-out W−N_old下界>0、raw W−50%下界>0、上述taught competence通过。即使memory profile通过，E3存在主张没通过也必须报告“E1升级候选，有限E3未确认”，不把E3变成所有记忆改进的必要门槛。

包括ERROR在内共8个采用候选：32个采用profile +8个E3存在主张，**主模块家族为40个联合主张**。分配program alpha `.04`，每个联合主张alpha `.001`。每个主张用intersection–union规则，所有必要统计组件在该alpha通过才成立；主张间Bonferroni。组件界不是“全表同时覆盖95%”的区间，不另宣称未注册的组件显著性。t覆盖近似，必须披露；Clopper–Pearson部分为精确binomial界。

S3−Srand、非主要checkpoint、最差题和独立diagnostic的区间为描述性，不计为正式机制成功。候选选择受到以上完整家族约束，不私下改变m。

### 10.4 零方差和选版本

t界使用`stdev(ddof=1)`。样本方差为零时，不能自动产生point CI：有已证明的全输入结构恒等式时可用结构结果；否则，对已知有界world score差使用单侧Hoeffding保守界，记录range；对未注册有限range的eta差返回`UNRESOLVED_ZERO_VARIANCE`。H风险始终用精确binomial界。零差异保护可按10.2的工程均值与H检验处理，但不能称总体平均NI证明。

Hoeffding下界固定为`sample_mean - width*sqrt(log(1/alpha)/(2*n))`。E1配对accuracy差范围[-1,1]、width=2；单配置W−N范围[-1,1]、width=2；两个配置的W−N差范围[-2,2]、width=4；raw W−chance范围[-chance,1−chance]、width=1。不得用观察极差缩窄范围。

每个已通过profile的配置都是独立合格候选，不假设两赢家组合仍有效。先在合格候选中列出观察Pareto集合：最大化old/new/revision准确率均值与reuse因果效应均值，最小化两任务总W online CPU、W生命史峰值可变state bytes及固定架构bytes；全部方向不劣且至少一项严格较好才构成支配。从该集合按固定规则选择一个实验版本：若存在通过reuse profile及E3存在主张者，优先其世界均值绝对`eta_reuse`最高者，不按不同parent的增量混排；否则选总W online CPU最低者。并列按峰值可变bytes、再按配置ID字典序。该观察排序不形成新的显著性主张，也不证明全球最优。

终态为 `QUALIFIED_NEXT_CORE_CANDIDATE`、`PROMISING_BUT_UNRESOLVED` 或 `RETAIN_V1`。本次设计不自动替换部署代码。没有通过者就停止，不自动提高难度、扩n或再发起参数定位。

## 11. Latin 模块统计与机制结论

每剂量注册一个联合“表格E3改善”主张：

1. FINAL held-out W−N_old单侧下界>0；
2. FINAL raw W−25%单侧下界>0；
3. 该剂量的held-out因果效应−ERROR的配对下界>0；
4. old/new W样本均值均≥90%，相对ERROR样本均值均≥−5pp；
5. H表格=old或new比ERROR差超过5pp，Clopper–Pearson上界≤10%。

这是3个联合行为主张。1–3是正式改善证据；4是工程均值保护，5是坏世界风险保护。CENTER同世界分数全部报告，不用历史均值替代配对比较。14%提高到20–24%仍不能通过。

REL10另注册两个机制主张：其held-out因果效应分别优于REL_PERM10和FIRST10，世界配对单侧下界>0。Latin家族共5个主张，分配program alpha `.01`，每主张`.002`；与主模块的名义program错误预算合计`.05`。

REL10必须先通过表格行为主张且两个机制比较都通过，才记 `REL2_GEOMETRY_SUPPORTED_ON_LATIN`。仅行为通过记机制未决。REL05/REL20通过记具体剂量行为候选，不自动要求或启动另一轮匹配对照；不宣称低/高剂量机制特异性。

OLD_END/AFTER_NEW、W/N_new、per-bank分数以及decoder为描述性定位，不替换FINAL。可在CENTER的OLD_END已保存h上，以12个taught标签做固定minimum-norm decoder，分别用h/q评分4个held-out；SVD cutoff固定为`max(nrows,ncols)*machine_epsilon*sigma_max`，不加ridge或调参。它们不接入learner、不取得采用资格；失败最多限制本固定映射/decoder，不能关闭所有二阶或关系机制。

Latin终态可以为 `POSITIVE_SCOPED_E3`、`NEGATIVE_TRANSFER_REDUCED_ONLY`、`NO_REGISTERED_BENEFIT` 或 `UNRESOLVED`。不通过不撤销主模块的有效核心结果。

## 12. 三次资格循环和运行完整性

三次是同一份设计的实现资格循环，不是三套额外科学问题。仅使用development worlds，不浏览official结果选择参数。每次记录设计→独立audit→test run→修订；修改科学公式、名册、样本或采用条件须产生公开amendment并重新审查，不能标为普通bugfix。

### Cycle 1 源码和数学合同

- parent CENTER/ERROR对源；canonical每store独立birth；确认h、q、PN/KC、clock、尺度、alpha dose与本合同一致。
- A/B和S support不随world/标签改变；PN输入非负有界；cache早于当前答案。
- 双投影共同取反保持z；单投影取反交换双轨；FIRST删去乘法。错误fan-in、符号、cache时刻、额外时间推进、FE0 fallback必须被拒绝。
- Q_HALF与ERROR的continuing state逐位一致，包括writer pi、四输出参数、预测器和时钟；只能最终读出不同。
- REL额外bank不得改变parent两bank的writer/input状态；四通道真实update目标逐记录核对。
- 故意只改REL写地址或只改read地址必须被拒绝；额外逐byte刺激也必须失败。相同时刻的合法read/cache地址一致，PERM例外逐项明示。

### Cycle 2 因果与泄漏攻击

- 每阶段每分支测试实际禁写；人为复现旧`N_old_rel`名称漏禁写必须失败。
- 人为将relation target送2/3、使用`punishment=-s`、只禁某bank写、答案提前进入cache、clone共享可变数组、将held-out标记送进模型，独立auditor必须拒绝。
- 当前答案替换不改变先前预测/输入缓存；到来后四通道更新按实际字节改变。
- P/S在禁监督分支仍按cue次数适应，图/证据/游标在相同感官干预分支中逐位一致；答案byte不作额外适应样本；probe前后continuing全状态相同。
- Srand预算、REL permutation性质和各control不声称成立的norm匹配逐项记录；immutable matrices/state shapes确实匹配。

### Cycle 3 完整生命史、恢复与预算

- 每个物理配置×注册任务至少一条完整development world-job；不以“短fixture”参数被忽略后跑了全life还称两条record test。
- checkpoint kill/restart、fresh-process replay与不中断reference完全一致，包含P图、S支持/计数/随机游标、REL cache/alpha_scale、Q策略和分析游标。
- 存在已提交完整receipt时跳过，存在partial时只从同输入已提交cursor恢复；不能从头重复完成或已提交前缀。
- development恢复可重复以检查操作；official scientific failure不得自动重启。基础设施恢复须有确切停止/最后cursor证据，不能借恢复更换世界。
- 如计划8workers，用development验证4/8worker状态一致和内存安全。并发4或8在launch前锁定；全配置以预注册、均衡调度比较CPU，不依据成绩分配更快机器。
- 最终qualification、source lock、roster lock和receipt auditor共同接受后，才可以请求并执行正式launch授权。

## 13. 计算、存储和后台执行预算

旧V2的96世界两任务每配置约10.6 CPU小时；table parent64世界三分支约2.46 charged worker小时。按主模块72世界、11学习配置、其中5配置多四份store线性外推，主模块约108小时；table七配置约23小时。**约131小时是量级规划，混合了历史CPU与charged worker口径，不是已测新程序的CPU承诺。** P/S hook、projection、checkpoint、冷热启动和审计还会增加成本。

本设计总上限：全部qualification与official合计180 process CPU小时、200 charged worker小时；正式结果目录8 GiB；每worker RSS 1 GiB；主模块world-arm-assay job最多3600 worker秒，table job最多900秒。资格运行必须按最终日志/receipt政策给出逐配置预测、p95及总保守预算。不能靠降低n或删掉失败配置偷偷满足预算。

8workers理想下约16.5小时仅为131/8的算术；实际CPU竞争、恢复、审计、串行阶段和Mac性能会改变完成时间。不能承诺一小时完成。小时级状态报告是运行监控，不是interim inference；完整job/checkpoint才是合法恢复边界。

全部结果用紧凑gzip receipts保存actual output、bank scores、actual write ledger、phase counters与source identity，不落全逐字节CSV。h/q完整向量仅在development和固定FINAL probe sidecar保存，使用npz或bitpack；science每个sidecar属于注册日志量，不反馈模型。

另事前保存Latin的CENTER/W/OLD_END全部16题（12 taught＋4 held-out）的精确h侧车，供§11两个decoder使用；q由锁定映射确定性重建并核对digest。它只写一次，不反馈模型。没有该侧车则相应decoder标不可用，不能为取回向量重跑学习。

额外bank可能使REL/FIRST超出1.5×online CPU保护：资源预检必须实际测完整wrapper和最终日志政策。超出保护但仍在总运行预算内者可以完成已注册科学比较，不能获得主模块采用资格；超出总预算则整个设计停在RESOURCE_NOT_QUALIFIED，不结果后改cap、删bank或缩n。

后台supervisor与caffeinate分离终端session生命周期；caffeinate绑定supervisor PID。独占job锁、atomic/fsync commit、heartbeat与manifest用于区分活跃partial和已完成receipt。监督器完成后自动analysis、independent audit和packaging；本设计未授权GitHub发布、启动后台任务或更改其他实验。

预算不满足是 `RESOURCE_NOT_QUALIFIED`，不是机制科学失败。预算用尽或不可解释错误时停止保存现场，不打包部分数据为完整成功，不扩n，不自动延长预算或移植到另一机器继续挑结果。

## 14. 交付和明确停止

实现阶段最低交付：`SPEC_LOCK.md/json`、`SOURCE_LOCK.json`、world exclusion/roster、三循环audit/test receipts、逐bank实际写入审计、resource forecast、checkpoint/replay证据、资格 verdict。正式完成交付：世界级指标、全部profile/claim组件、逐候选保护与收益裁决、report、final independent audit和完整结果bundle。

最终报告必须回答：

1. 哪个配置在相同新信息下增加了哪一项能力，其对应效率是否提高？
2. 哪些旧知识、新学习或修订被损害；是否满足注册工程保护和坏世界风险？
3. 有限reuse与Latin表格分别达到什么证据，哪些仍未知？
4. 增加多少可变/固定bytes、每事件操作、实际监督写量、结构事件和真实CPU？
5. 可以停止投入哪些具体recipe，哪些仅资源未资格或统计未决？

Q/P/S/REL只有事前名册中的配置，没有隐藏replacement、世界替换、额外dose或更难的最终验收任务。任何结果都完成本设计的判定；未找到升级则保留V1与有效ERROR结果，不用“还要一次整合”延长本轮。

## 附录 A 关键变更与来源

设计审阅记录：两路独立审阅分别检查机制接口/源时钟与名册/判定/资源合同。已修订REL读写双路径、pending/rest调用数、P/S禁监督下图状态一致性、Srand缺失终态、OLD_END侧车和版本排序。此次只完成文档审查与静态校验，三个实现资格循环、test run和science尚未执行。

| 原设计问题 | 本设计决定 |
| --- | --- |
| REL h与normalization未实例化 | 固定88维FE0、44维有符号积、88维双轨、source norm→native KC |
| signed活动不合法 | 符号拆分为非负PN，原生event不改负活动合同 |
| 新bank scale含糊 | 明确借用cS为固定工程单位，不伪称预校准 |
| native/shared与centered teacher混淆 | extra用native binary；private继续ERROR signed |
| shuffle谎称实际update norm相等 | 仅PN几何匹配，实际写量单列 |
| 复制h不是真正独立表示capacity | FIRST采用相加投影，同budget但不声称等有效容量 |
| 没有同世界REL−parent改善比较 | 主模块对ERROR；table也注册对ERROR的E3增量 |
| 不断追加scale rescue实验 | .5/1/2一次事前注册，机制特异性仅1×有对照资格 |
| Latin变成核心新门槛 | 两模块分别判定，table失败不否决主模块 |
| 配置和世界独立性混算 | Q同轨迹评分；跨任务独立lives；2032jobs/6096lives |
| 零方差自动无损或一律堵死 | 工程均值+精确H风险；正式gain零方差用有界fallback或未决 |
| 所有阴性都排除一个机制家族 | 仅关闭exact port/参数/任务组合，not-qualified与未决分开 |

可核对的本地来源：

- [阶段目标合同](../../MINIFLY_OBJECTIVE_CONTRACT.md)与[未来核心要求](../../PERSISTENT_CORE_REQUIREMENTS.md)。
- [V1 core](../../DIGITALBABY_CLAUDE_B_D4E4578_20260929/r_center_core_v1/centered_core/core.py)、[V2 writer](../../DIGITALBABY_CLAUDE_B_D4E4578_20260929/persistent_core_v2_trial/v2_core.py)、[既有两任务](../../DIGITALBABY_CLAUDE_B_D4E4578_20260929/persistent_core_v2_trial/v2_fixture.py)。
- [P native移植源](../../DIGITALBABY_CLAUDE_B_D4E4578_20260929/minifly_a_v3/src/stores.py)、[S源](../../DIGITALBABY_CLAUDE_B_D4E4578_20260929/minifly_b/src/topo_model.py)。
- [native活动/alpha入口](../../DIGITALBABY_CLAUDE_B_D4E4578_20260929/r_center_core_v1/vendor/package/REFERENCE_SOURCE/minifly/V82E/src/model_evo.py)。
- [接受的表格结果](../../DIGITALBABY_GITHUB_UPDATES_20261005/studies/rcenter-survivor-rounds-20261003/results/report/ROUND1_REPORT.md)与[新ERROR表格结果](../../DIGITALBABY_GITHUB_UPDATES_20261005/studies/error-completion-20261004/FINAL_REPORT.md)。

本文注册的是设计选择和停止范围。源码身份、实际预检成本及正式launch资格要由实现后的独立检查填实；不能把设计完成写成science已经运行。
