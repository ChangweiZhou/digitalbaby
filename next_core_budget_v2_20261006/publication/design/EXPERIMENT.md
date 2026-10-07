# MiniFly 下一核心机制实验 v2：十小时窗口内的有限筛选与独立确认

日期：2026-10-06。状态：**新设计；未启动筛选或确认，尚须新调度与分析实现资格化。**

本文件替代 v1 的未来科学运行规模，不修改 v1 协议、资格记录、历史数据或科学代码。v1 的三轮资格已经完成：功能/因果检查通过，原定总资源门失败，正式科学轨迹为零。本次可以在没有看见正式结果的情况下重新设计预算与判定。

## 1. 唯一阶段目标

在相同字节经验、反馈次数和模拟时间下，找到一个比其 parent 更有效率、且不破坏现有获取、保持、修订和有限复用能力的具体核心配置。比较完整的 byte encoder → memory → 模型自身输出链路。

只使用已有 lifetime 和简单符号关系 reuse 两项任务。保留原题数、完整生命史、所有延迟、干扰学习、修订和禁写分支。不增加自然语言、未教算术、开放输出、无教学信号学习或规模扩张。

本轮允许三种不同结论：E1 记忆/修订工程升级；当前简单关系族内的有限 E3；某个具体对照支持的局部机制解释。三者分别报告。E1 成功不能冒称未知挑战泛化；E3 未确认不取消有效的 E1 改进。

**最大范围在这里一次封顶：全名册 8 世界筛选，最多一个候选在 48 个全新世界确认，条件性机制对照事前列齐。** 无赢家也是完成。不会追加候选、补种子、合并赢家或再要求一次整合实验。

## 2. 与 v1 的明确变更

| 项目 | v1 | v2 |
| --- | --- | --- |
| 主模块 | 所有配置 72 世界直接确认 | 所有配置 8 世界探索筛选；至多一个候选 48 世界独立确认 |
| Latin | 7 配置 × 64 世界，事前承诺 | 明确移出本轮；无 Latin 数据、结论或补跑承诺 |
| 科学主张 | 主模块 40 项，另有 Latin 5 项 | 选定配置/目标的一个采用主张、一个 E3 存在主张、最多一个机制联合主张 |
| 统计 | 主采用主张各 alpha=.001 | 采用 .03；E3 存在 .01；机制联合 .01；未使用份额不回收 |
| 最大科学规模 | 2,032 jobs / 6,096 lives | 656 jobs / 1,968 lives（REL10 最慢路径） |
| 科学规划 charged worker | 186.81 h，含 1.5×余量 | 最慢路径约 67.98 h，含 1.5×余量；硬上限 70 h |
| 每次电脑窗口 | 未设置十小时退出合同 | 第 8 h 停止派发；安全暂停；第 10 h 前退出 |

这是科学范围和统计家族的事前改版，不是原大实验的加速等价实现。原方法、模型接口和两个主任务公式继承 [v1 协议](../MINIFLY_NEXT_CORE_EXPERIMENT_20261006/MINIFLY_NEXT_CORE_MECHANISM_EXPERIMENT_V1_20261006.md) §§4–7、8A、9、10.1。若实现与这些公式不符，停止，不能按普通操作修补继续 science。

## 3. 名册：保留全部已讨论方向

| 配置 | 身份与 parent | 改变 | 是否可晋级 |
| --- | --- | --- | --- |
| CENTER | R_center V1 参照 | 原配置，8 stores | 否 |
| ERROR | parent=CENTER；亦为其余候选的参照 | private signed writer 从 `.25-y` 改为 `pi-y` | 是 |
| Q_HALF | parent=ERROR | 仅最终 private readout 权重 .5；复用 ERROR 学习轨迹 | 是 |
| P005 / P0005 | parent=ERROR | cue-only 归一化 Hebb；epsilon .005 / .0005 | 是，两种具体配置 |
| S3_CUE | parent=ERROR | cue-clock 使用率驱动换边 | 是 |
| S3_RAND | 计数配对随机换边对照 | 保持 S3 实际换边数、degree 与预算 | 否 |
| REL05 / REL10 / REL20 | parent=ERROR | 固定乘法交互表示 + 四个额外 native stores；dose .5 / 1 / 2 | 是，三种具体配置 |
| REL_PERM10 | REL10 对照 | 只在写入时固定 PN 置换 | 否 |
| FIRST10 | REL10 对照 | 相加投影替代乘法，其他流程固定 | 否 |

合计 12 个报告配置、11 个物理学习配置、8 个可晋级配置。新方向覆盖 Q、P、S、REL；不再测试 FE1。ERROR 的 writer 改动单独记账。配置数、机制家族数和 108 条历史总账分别统计，不能互相相减。

不启用新的进化变异、投影 seed 搜索或额外剂量。这次使用有限候选集的选择淘汰，充分利用已完成的功能资格。以后若采用进化算法，须另有总评估预算；不能把它作为本轮失败后的自动延长方式。Dust 不接入本轮。

## 4. 完整任务、状态和暴露合同

### Lifetime：E1

32 个旧键、32 个新键，8 个旧键改答案；384 old + 384 new + 96 revision，共 864 教学记录。旧题最终只计 24 个未修订键，修订题按新答案计。原六个 checkpoint、三个一天延迟与总时钟不变。每个 world-arm job 完整运行四条生命史。

| 分支 | old value 写入 | new value 写入 | revision value 写入 |
| --- | --- | --- | --- |
| W | 开 | 开 | 开 |
| N_old | 关 | 开 | 开 |
| N_new | 开 | 关 | 开 |
| N_revision | 开 | 开 | 关 |

### Reuse：有限 E3

12 个 old taught cue、6 个从不强化的 held-out cue（3 对、双方向）、6 个新 cohort cue；144 old + 72 new，共 216 教学记录；两次一天延迟。每 job 运行 W / N_old 两条生命史，双方仅 old value 写入不同，后续新教学相同。保留原内部两 option ChoiceOrgan 和两种呈现顺序。

全部 scored answers 由模型在本题反馈前产生；probe 用 disposable clone，不写回 continuing history。感官字节、答案字节、标签和时钟在干预间相同；P/S 无标签适应在 N 分支继续，不得冒充监督写入已关闭。unknown branch 拒绝，不使用字符串包含猜权限。

每个 native store 独立 canonical birth；共享只读资产允许，共享可变状态禁止。REL 同时替换查询/写入 association 地址，使用反馈前 cache，不改变 byte/pending/rest 时钟。Q_HALF 不改变 ERROR writer 的 pi、感官历史或 continuing state；其读出成本单独计入，不再收费一条学习轨迹。

Lifetime 的任意标签能被 exact-key 表解决，只支持 E1。Reuse 的 raw/canonical-key 表没有 withheld 答案；符号位置、频率无固定正确类，先前教学形成的单符号一 bit 类规则可以解题，属于允许的窄 E3。解析规则仅是 evaluator 的 solvability witness，不进入模型。不同任务使用同一架构但独立生命史，不宣称一份训练状态掌握全部任务。

## 5. 探索筛选：固定 8 个配对世界

拟定 world IDs：`61008001..61008008`。每个世界全 11 个物理配置 × 两任务；Q_HALF 同轨迹评分。共 **176 jobs / 528 lives**。

这些是允许读取 held-out 结果的探索世界，从所有确认统计中永久排除。development、历史 official/pilot、此次筛选和此次确认的名单须在运行前核对并写入 exclusion registry。上述数字目前只是拟定，未声称已经完成去重检查；冲突只能在筛选 outcome 前整批换号并重新锁定。

全部筛选任务和独立 receipt audit 完成之后才选择晋级配置，不依据任务完成先后、每小时估计或中途榜单提前选赢家。不进行正式 interim inference。

### 5.1 样本保护

一个配置必须同时满足以下探索样本保护，才有可晋级目标：

1. Lifetime old/new/revision 最终 W 均值各 ≥90%；各对 parent 均值差 ≥−3pp；old/new 对 CENTER 均值差 ≥−3pp。
2. Reuse 最终 raw W、W−N_old 各对 parent 均值差 ≥−5pp；old_end taught W ≥80%，final taught W ≥70%，new taught W/N 各 ≥65%。
3. 两任务总 W online process CPU / parent 相同总 CPU ≤1.5；tau 正值、计时接口一致，RSS/资源通过。
4. 8 个世界里，任一 lifetime old/new/revision 比 parent 下降 **超过 5pp** 的世界数为零。

这是探索过滤，不是总体非劣性或 10% 风险概率证明。n=8 不使用 Clopper–Pearson ≤10% 的确认门槛，不能伪造其已通过。

### 5.2 一次选择，目标一起冻结

对通过保护的配置，生成目标 `old / new / revision / reuse`。定义：

```text
A_old, A_new, A_revision = 对应 FINAL W accuracy
A_reuse                 = FINAL heldout_W - heldout_N_old
tauL / tauR             = v1 相同边界的 W online process CPU
eta_E1                  = 3600*A_E1/tauL
eta_reuse               = 3600*A_reuse/tauR
delta                   = E1 .02；reuse .05；ERROR/CENTER revision .20
C                       = mean(tauL_candidate+tauR_candidate)
                          / mean(tauL_parent+tauR_parent)
R                       = (mean(A_candidate-A_parent)/delta) / C
```

该目标须满足配对平均能力增量 ≥delta、对应配对平均 eta 增量 >0。reuse 目标还须候选 held-out W−N_old 均值 >0 且 raw W 均值 >.5。分母无效不是可以加 epsilon 修复的低分，而是技术错误。

按 R 降序；并列按 `mean(ΔA)/delta` 降序、两任务总 W CPU 升序、配置 ID 字典序、目标固定顺序 old/new/revision/reuse 排序。对每配置先保留最高项，再取全表第一项。R 只是分配确认预算的选择规则，不是新的能力总分或最优性证据。

**最多一个配置和一个主要目标晋级。** 将全部筛选数据、排序、晋级配置、parent、目标、delta、确认名册和 alpha 分配写入 `SELECTION_LOCK.json`，封印并独立复算后才生成/读取确认结果。没有合格项则 `SCREEN_COMPLETE_NO_CANDIDATE`，本轮结束。

## 6. 全新 48 世界确认，条件对照一次列齐

拟定 IDs：`61008101..61008148`。确认前完成 exclusion registry；不得混入 screening 或 development 世界。不能观察确认结果后改配置、目标、样本量、门槛或 alpha。

| 晋级配置 | 确认物理名册（两任务全部运行） | 含筛选总 jobs / lives |
| --- | --- | ---: |
| ERROR / Q_HALF | CENTER、ERROR；Q_HALF 同轨迹读出 | 368 / 1,104 |
| P005 / P0005 / REL05 / REL20 | CENTER、ERROR、所选配置 | 464 / 1,392 |
| S3_CUE | CENTER、ERROR、S3_CUE、S3_RAND | 560 / 1,680 |
| REL10 | CENTER、ERROR、REL10、REL_PERM10、FIRST10 | 656 / 1,968 |

REL05/20 没有等剂量专用置换/相加对照，因此只确认其具体行为 recipe，不宣称该剂量的乘法/对齐特异性。不能拿 1× 对照替代，也不会在结果后补建对照。

S3_RAND 必须等同世界 S3_CUE 的 committed yoke 完整存在后才能运行。合法候选池耗尽按 v1 记 `DIAGNOSTIC_NOT_QUALIFIED`：主行为仍可判定，机制主张不可用。禁写、错位、出生、source 或预算守恒 bug 属于完整性失败，不能包装成对照耗尽。

## 7. 确认统计：一个赢家，不从确认数据再挑目标

名义 program alpha=.05，预分配如下，未使用份额不回收：

| 联合主张 | alpha | 说明 |
| --- | ---: | --- |
| 选定配置在选定目标上的采用 | .03 | 只检验 SELECTION_LOCK 内一个目标 |
| 该配置的有限 E3 存在 | .01 | 与采用分别判定 |
| 已注册具体机制对照联合主张 | .01 | 仅 S3_CUE 或 REL10 有资格；其余标 NOT_REGISTERED |

筛选选择与确认世界独立，确认只产生上述最多三个主张。每主张采用 intersection–union：其所有必要统计组件在该 alpha 通过才成立；各主张的 alpha 相加为 .05。Student-t 覆盖近似，故这里是**名义 FWER 控制**，不承诺所有数据分布下精确 5%；Clopper–Pearson 风险界部分精确。

### 7.1 共同工程保护和严重退化风险

确认采用要求 §5.1 第1–3项在48世界再次通过。并定义每个独立配对世界 H：选定配置的任一 lifetime old/new/revision 比 parent 下降 >5pp 时 H=1。

在 alpha=.03 下，以单侧 Clopper–Pearson 上界检验 `Pr(H)≤.10`：h<n 时 `U=Beta.ppf(1-alpha,h+1,n-h)`，h=n 时 U=1。n=48、h=0 上界7.0449%，通过；h=1上界10.6648%，失败。8世界筛选不替代这一风险检查。共同工程保护不是总体平均非劣性置信证明。

### 7.2 选定目标的采用联合主张

共同保护通过，并同时满足：

1. 所选能力差 `A_candidate-A_parent` 的配对世界单侧下界 >0。
2. 样本能力增量 ≥冻结 delta；这是实用样本门槛，不能冒称总体增量已证实至少 delta。
3. 对应 eta 差的配对世界单侧下界 >0。
4. 对 E1 目标：候选对应 `W−N_old/new/revision` 的配对世界下界 >0，证明该目标的答案教学确实产生能力。
5. 对 reuse 目标：候选 `heldout_W−N_old` 下界 >0，raw W−.5 下界 >0，且既有 taught competence 通过；同时必须通过 §7.3 在 alpha=.01 下的 E3 存在联合主张，才可获得有限 reuse 升级标签。不能用 .03 的较宽松检验绕过单独 E3 资格。E1 目标不要求 E3 存在主张通过。

不允许从另外三个未冻结目标挑一个显著结果救回采用。其他结果完整报告为描述性。ERROR 对 CENTER 的 revision delta=.20，其他 E1=.02，reuse=.05。

### 7.3 E3 存在联合主张

在 alpha=.01 下，FINAL held-out W−N_old 下界 >0、raw W−.5 下界 >0，且 §5.1 的 reuse taught competence 满足。该主张不要求超越 parent，也不证明更广任务复用。

### 7.4 条件机制联合主张

只有主采用已通过时才允许机制成功标签；预留 alpha 不因其他主张失败而改变。

- S3_CUE：在同一个冻结主要目标上，S3_CUE−S3_RAND 配对能力下界 >0，独立 yoke/结构/预算审计通过，才支持“本 cue-driven 换边规则相对于计数匹配随机换边的作用”。它不证明所有 homeostasis 机制有效，随机对照也不保证等地址扰动。
- REL10：在同一个冻结目标上，REL10−REL_PERM10 与 REL10−FIRST10 两个配对能力下界均 >0，匹配接口/资产审计通过，才支持本表示的对齐及乘法相对于这两个对照的作用。两组件组成一个联合主张；不得挑一个通过就记机制成功。

控制比较不足时，具体行为采用仍可成立，机制解释记未决。不由确认 held-out 结果再选择一个更容易胜过 control 的目标。

### 7.5 世界级区间和零方差

样本单位是独立 world；题目、字节、record、branch 或 job 不是独立 n。非零样本方差使用 `mean - t.ppf(1-alpha,n-1)*stdev(ddof=1)/sqrt(n)`。

零方差不产生自动 point CI：只有已经证明的全输入结构恒等式才能按结构处理；其他有界分数使用 `mean - width*sqrt(log(1/alpha)/(2*n))` 的 Hoeffding 下界。E1能力差、单配置W−N的range为[-1,1]、width=2；两个配置reuse效应差为[-2,2]、width=4；raw W−.5的width=1。禁止用观察极差代替固定范围。eta 差没有注册有限range，零方差返回 `UNRESOLVED_ZERO_VARIANCE`。

n=48 能检验上述零严重退化风险带，不是功效保证。小幅收益可能未决，不能因此扩 n 或临时替换检验。

## 8. 实测预算与十小时会话合同

资源数据只使用已完成的完整 development job 的 CPU、worker wall 和文件尺寸，不使用其准确率选择候选。每 recipe 只有一次完整计时，八进程只有短 fixture 状态一致证据，故无法声称 p95 或持续八进程吞吐已证实。

| 路径 | science 点估计 charged worker | 1.5× / 8 的理想并行时间 |
| --- | ---: | ---: |
| 筛选全部 11 配置 | 12.08 h | 2.26 h |
| 筛选 + ERROR/Q_HALF 确认 | 21.57 h | 4.04 h |
| 筛选 + P005 确认 | 26.90 h | 5.04 h |
| 筛选 + REL05/20 行为确认 | ≤29.62 h | ≤5.56 h |
| 筛选 + S3_CUE/RAND 确认 | 34.18 h | 6.41 h |
| 筛选 + REL10/PERM/FIRST 确认 | 45.32 h | 8.50 h |

最慢路径 science process CPU 点估计约44.28h；science CPU ×1.5约66.42h。科学硬预算：**68 process CPU h、70 charged worker h、全部本轮交付文件合计8 GiB（含压缩包）、每worker RSS1 GiB、每个job最多3600 worker秒**。审计/分析/打包另列CPU和会话wall；已完成v1资格费用单独报告，不伪装成未来science消耗或重复收费。

最慢路径 receipts约.883GiB、最新checkpoints约1.439GiB；两者×1.5＋1GiB日志＋已有资格目录约4.67GiB。另预留1GiB用于FINAL_W_STATE/新操作证据，以及2GiB用于完整结果压缩包，合计约7.67GiB，须在8GiB以内。上述export/archive均为预留，须由最终资格实测确认，不能冒称已有导出器已经通过。结果包包含全部科学receipts、报告、锁、选择记录、确认W状态、必要源码/资产及调度日志；不重复打包恢复checkpoint、环境二进制或逐job冗长stdout，后者留在受总预算约束的本地目录。保留完整receipt和每job最新安全checkpoint；仅原子覆盖该job旧checkpoint，不删除旧实验数据、不为本轮预算冒险删证据。

### 安全停止而非十小时完成保证

一次会话wall从dispatcher启动算起，采用单调时钟：

1. 第8小时停止派发新job；不新开确认阶段。现有job继续。
2. 第9小时若仍有job，请求在下一安全完整record边界写入checkpoint后退出。checkpoint提交必须原子/fsync，保存source、fixture、branch、record cursor、累计CPU、学习状态、P/S图和随机/yoke游标。
3. 第9小时30分钟前所有worker须退出；未响应安全请求的worker由watchdog停止，记 `STOP_NEEDS_REVIEW`，保存日志和最后可靠cursor。不得把它当安全暂停自动重启。
4. supervisor在第10小时之前退出，保存队列和会话记录，caffeinate绑定supervisor。不是完成的运行不发布完整result bundle；正常安全暂停记 `PAUSED_SESSION_LIMIT`。
5. 下一次运行窗口可以在同一冻结合同下显式恢复未完成队列；完整已提交receipt跳过，partial只从已提交cursor恢复。pause不扩科学n、候选或总预算。scientific/integrity failure禁止自动恢复。

十小时是**会话退出上限**，约8.5小时是最慢路径规划值；若机器慢，可能需要第二次窗口，但不会因此增加科学范围。达到科学CPU/worker/磁盘上限且未完成则 `RESOURCE_INCOMPLETE`，停止且不自动加预算。

运行依赖锁只读源/资产与固定名册。S3_RAND只在yoke来源就绪后派发，未满足依赖的等待时间不占一个worker。按world、配置轮转均衡调度，不根据成绩给候选更快CPU。onlineW CPU用于效率比较，总job时间用于运行预算，两者不可混用。分析、审计或打包因会话截止未完成时保存其非学习游标/输出，下次只完成这些步骤；已有science绝不重跑。

## 9. 实现资格：复用已有证据，补齐本次确有变化的接口

已有功能证据绑定 v1 源码身份 `0701272ef469e110a0a5286fe7022cc2103f6cac4e2a357fce47b4c6d9a98fe2`。它不是 v2 的正式 source lock，原资源门也不会自动变成通过。

保持科学worker/模型公式时，可引用原三轮 E0 证据；**不为这次文档重跑29个完整资格jobs**。新版本仍须先落实：

- dispatcher/候选选择器/确认分析器/manifest的实现与独立静态审阅；最终 source/spec、development/screen/confirmation exclusion registry；来源资产和环境锁。
- 合成结果的选择负例：无赢家、目标并列、零tau、超过CPU保护、筛选/确认混样、确认后改目标、Q重复收费、control误晋级，必须被拒绝或按合同裁决。
- 合成统计测试：alpha分配、IUT、CP严重退化、world级配对、零方差范围和未决状态。
- development短fixture的运行中安全暂停/恢复、会话截止、锁/依赖/已有receipt跳过；与不中断scientific state一致。原证据只验证保存cursor后SIGTERM，不能冒称任意时刻SIGTERM会自己保存。
- 每个完整W生命史结束立即导出独立可加载FINAL_W_STATE，保留source/assets引用和state摘要；恢复后用只读固定probe验证一致。当前checkpoint只保留当时活动分支的learner，不能拿digest替代最终W模型。导出所有confirmation W实例；示例选择最低world ID，不以分数选最好训练状态，不合并不同world的状态。
- 实测额外导出/调度日志开销及八worker实际吞吐安全性；按最终政策重算本表。若公式改动，则重新资格化受影响路径，不能沿用旧E0为新公式背书。

只有 `V2_FUNCTIONAL_AND_RESOURCE_QUALIFIED` 后才具备正式launch资格。本文仅授权设计交付，未启动science、创建后台自动化或上传GitHub。

## 10. 完整报告、终态及历史总账

最低交付：`SCREEN_REPORT.md`、`SELECTION_LOCK.json`、确认完整receipt和W导出、`CONFIRM_REPORT.md`、逐组件采用/E3/机制裁决、独立FINAL_AUDIT、预算/会话记录及结果包。保留所有筛选配置，不能只展示赢家。

终态：`SCREEN_COMPLETE_NO_CANDIDATE`；`CONFIRMED_E1_ENGINEERING_UPGRADE`；`CONFIRMED_LIMITED_REUSE_UPGRADE`；`PROMISING_BUT_UNRESOLVED`；`RETAIN_V1`；或明确的执行/资源未完成状态。E3存在与机制主张作为独立标签附列。没有确认收益时保留V1和此前有效ERROR结果，不自动启动第三阶段。

筛选被淘汰只表示“本预算下不晋级”，不能当正式机制阴性。确认无收益只限制本配置、剂量、parent和任务，不能排除整个P/S/交互家族。资源失败、实现不合格和统计未决与有效科学阴性分别登记。108条总账中只有对应exact recipe/claim可以更新，未测试项仍未测试。

## 11. 来源和设计审查

- [原v1协议](../MINIFLY_NEXT_CORE_EXPERIMENT_20261006/MINIFLY_NEXT_CORE_MECHANISM_EXPERIMENT_V1_20261006.md)。
- [原三轮资格报告](../MINIFLY_NEXT_CORE_EXPERIMENT_20261006/THREE_CYCLE_QUALIFICATION_REPORT_20261006.md)。
- [阶段目标合同](../../MINIFLY_OBJECTIVE_CONTRACT.md)与[未来核心要求](../../PERSISTENT_CORE_REQUIREMENTS.md)。
- 资源算术来源：`DIGITALBABY_CLAUDE_B_D4E4578_20260929/next_core_mechanism_v1_20261006/results/development/cycle3/receipts/61005003_<arm>_<assay>.json.gz`，仅取cpu_s、worker_s与磁盘字节。

独立预算审阅纠正了上一讨论中的“两赢家约7小时”：若同时确认REL10和S3并保留对照，总预计10.86小时。本版固定一个赢家，保留其注册对照，最慢路径8.50小时。统计审阅要求候选与目标一起冻结、筛选不借用确认n、alpha不回收，以及E1/E3/机制分开。

本文件不声称完成新的test run或science。设计结束后只实施必要的v2调度/分析资格，不增加新的科学问题。
