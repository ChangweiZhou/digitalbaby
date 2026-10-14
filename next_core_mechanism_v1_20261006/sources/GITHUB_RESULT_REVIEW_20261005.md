# 新 ERROR 结果对核心搜索的影响

日期：2026 年 10 月 5 日。只读研究更新；没有启动实验、调整旧门槛或修改冻结科学代码。

已同步 GitHub 新提交 [3bcd6e1](https://github.com/ChangweiZhou/digitalbaby/tree/3bcd6e1388f50b274c66eb37f69feb62a99f161a/studies/error-completion-20261004)，并保留此前 Sparkie 结果供比较。原研究计划仍在 [NEXT_CORE_VARIANT_PLAN.md](NEXT_CORE_VARIANT_PLAN.md)。

## 新结果究竟测试了什么

64 个新配对世界，320001–320064。每世界比较 R_center 与 ERROR，各有 W、N_old_relation、N_new 三条 life；每条包含 192 次旧教学、192 次新教学和两次一天延迟。任务是重新标记的四乘四 Latin 表：教 12 个格子，保留 4 个格子从不强化，最终由模型四输出通道 argmax 回答。它不是此前二选一的符号类关系任务。

两核心都使用 shared FE0 与 private CONTENT。ERROR 只把 private 的教学系数从 `.25-y` 改为 `pi-y`，pi 来自反馈前 private values，temperature=1。最终 shared/private 组合权重与归一化常数相同。公开源码要求配对核心的 shared continuing state 相等。

本次源码检查确认：`N_old_relation` 确实在 old 阶段禁监督写入，`N_new` 在 new 阶段禁监督写入；clamp 覆盖实际 native teach APIs，validator 检查被禁阶段各 store 的实际写入量为零。本轮没有发现此前 Package B 的分支名称漏禁写问题。这个判断是公开源码审阅，不等于重审所有原始运行记录。

## 最终结果

| W 分支最终结果 | R_center | ERROR |
| --- | ---: | ---: |
| 已教旧题 | 98.05% | 94.92% |
| 已教新题 | 94.92% | 91.89% |
| 未教表格格子 | 15.23% | 14.06% |
| 未教 W−N_old_relation | −8.98 pp | −10.94 pp |

注册的 16 项比较使用世界级配对 t 区间与 Bonferroni 调整。ERROR 的未教准确率区间是 [8.96,19.16]%，低于 25% 机会水平；W−N_old_relation 区间为 [−17.97,−3.91] pp，旧教学在这项任务上反而损害未教输出。

ERROR−R_center 的未教准确率差为 −1.17 pp [−3.84,+1.50]，因果效应差为 −1.95 pp [−5.89,+1.99]。这两个直接比较都未确定差异，不能说 ERROR 比 R_center 更差地泛化已被证明。

ERROR 的旧、新题均值分别下降约 3.13、3.03 pp，区间分别 [−5.06,−1.19]、[−5.12,−0.93]。二者未通过“下界严格高于 −5 pp”的保持非劣性门槛；这不证明损失大于 5 pp。模型仍能学会已教题，不能将失败描述为程序完全不学习。

正式裁决是 `JOINT_REPAIR_EXCLUDED_BY_REGISTERED_BOUND`。科学完成，不继续扩 n 或用参数扫描挽救这个注册主张。

证据：[本地最终报告](../DIGITALBABY_GITHUB_UPDATES_20261005/studies/error-completion-20261004/FINAL_REPORT.md)、[公开分析](../DIGITALBABY_GITHUB_UPDATES_20261005/studies/error-completion-20261004/results/final/analysis.json)、[验证摘要](../DIGITALBABY_GITHUB_UPDATES_20261005/studies/error-completion-20261004/audits/FINAL_ANALYSIS_SUMMARY.json)、[机制源码](../DIGITALBABY_GITHUB_UPDATES_20261005/studies/error-completion-20261004/source/runtime/comparison_learner.py)。

## 搜索范围如何收窄

**关闭一个具体主张：** 在这个 fixture、时钟、四输出规则与固定温度下，只换 private 教学为 ERROR 就能修复未教表格补全。Sparkie 的 R_center 失败和此次 fresh-world 对照一致地限制了这个用途；它们不是同一批世界，不能当作重复轨迹。

**收窄 ERROR 的定位：** 保留它在现有 lifetime/修订任务中的收益；不能称它为普遍更好的核心，更不能据此默认替换 V1。后续保留 V1 正式参照和 ERROR 任务内参照，让候选相对 ERROR 的增量与相对现任版本的成品表现都可见。

**没有新排除 P/T/S、J/H 或整族机制。** 本次没有实现这些候选，也没测试 ERROR+Q-HALF。旧十六变体、参数救援名册和 108 条总账不能因为这项结果各减一。原始 15 项 E1、8 项有效 E3、7 项无效 E3、P3 未资格化的分类不变。此前“20 条 exact recipe/claim 可关闭”的历史口径不在此擅自加成 21。

**限制读出调整的承诺。** Sparkie 已测 R_center 的半私有读出没有救回表格补全；此次固定完整权重的 ERROR 也没有修复。它们降低了继续把简单读出/教学变化当作表格补全解法的优先级，但没有证明 ERROR 半权重或任何读出规则数学上都不可能有效。

## 下一轮计划的更新

仍保留已讨论的四候选、三个机制：

1. **ERROR + Q-HALF**：仅改最终 private 权重，writer pi 不变。验证目标是现有二元关系任务的有限复用与 lifetime 保持/修订；不宣传为 Latin 表格补全修复。它已有离线探索证据，可在同一 fresh ERROR trajectory 上评分，不另做一套重复学习。
2. **ERROR + P1-E005 / P1-E0005**：两档预先封顶的 shared PN→KC 适应剂量。新结果没有测试它们，资格与收益仍未知。
3. **ERROR + S3-RHO035355-CUECLOCK**：使用率目标与实际 coding fraction 对齐，明确移植为 cue clock；包含预算匹配的随机换边诊断控制。它不是旧 S3 仅改一个参数的原样重跑。

这批仍比较同一个成品在现有获取、保持、修订与有限复用任务上的表现，并计 CPU、内存与输入更新成本。新的 Latin 失败记录为能力边界，**不新增为阶段性验收要求**，也不因此发起新的形成定位系列。若更广泛未教挑战今后被列入正式目标，需要独立锁定，不能倒改当前研究范围。

本次新增能直接关闭的主张：1 项；新直接排除的上述待测参数/拓扑候选：0 项。十个旧机制参数救援、一个 Q-HALF、八个开放操作方向和五个未配置结构方向仍使用原计划的不同计数单位，不相加成“剩余机制总数”。下一轮仍未冻结、未启动。

## 证据边界

新发布包含完整 64 世界公开分析、报告、科学源码和独立验证摘要，但完整原始科学收据在本次同步提交仍标为上传中。因此当前结论据已发表分析与源码得出，不能说我们已独立验证全套原始轨迹。原运行独立审查和私有预注册是作者报告的证据；公开协议在启动后发布，不包装为同期公开预注册。
