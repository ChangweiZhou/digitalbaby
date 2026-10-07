# MiniFly budget v2：三次修订—审查—试跑报告

2026-10-06。裁决：**技术与当前资源规划资格通过；正式科学实验未启动。**

这一轮完成的是执行资格（E0）。它不证明任何候选已提高记忆、泛化或单位算力能力；E1/E3 的采用裁决仍须使用正式新世界。阶段目标保持为：固定字节接口下，对相同新知识获得更好的保留／修订能力与计算效率；有限未强化输入复用是可选目标。没有增加 NLP、规模扩张或更高智能任务。

| 循环 | 修订与审查 | 实际试跑及结果 |
|---|---|---|
| 1 | 单一候选与目标、独立确认、风险门槛、零方差规则和启动权限 | 33 项最终统计／流程／评分回归测试通过；扫描 428,670 个本地文件，排除 20,448 个历史编号，未发现拟用科学世界冲突 |
| 2 | 完整记录边界的真实信号暂停、无遗漏状态导出、篡改拒收 | 11 个物理配置均在执行中收到 SIGTERM 后保存；新进程恢复至 32-record W 结束，与不中断参考的状态和输出一致；六类篡改被拒收，已提交任务不重执行 |
| 3 | 完整生命史、评分范围、四／八 worker、实际限时暂停、强制停止计费、交付预算 | 2 组完整开发任务／6 条生命史通过状态和输出一致性检查；四／八 worker 一致；会话暂停可显式恢复；故意挂起 worker 后，在 12.14 秒的缩短测试窗口内终止，标为 STOP_NEEDS_REVIEW，CPU 被正确计费 |

## 捕获并修复的问题

1. 测试比较器误把 tuple 与 JSON 数组当成不同结果。仅统一容器类型，数值、时钟、原生状态及输出仍要求精确一致。原失败记录保留。
2. 旧题评分误混入 8 个已修订键。汇总器现按冻结的 `intact_old_items` 只计算 24 个未修订旧键；W、相应禁写分支、Q_HALF 一致。修订能力另按 8 个新答案评分。新回归测试明确构造“未修订全对、修订原答案全错”的例子，要求旧记忆分数为 1.0，而不是 0.75。**没有任何正式筛选、确认或采用裁决受此错误污染。**
3. 强制停止原先可能报告普通失败，且未正常退出的 worker 可能丢失 CPU 费用。现由单一 `wait4` 收割路径保存真实退出状态和系统 CPU；只向本调度器创建的子进程发信号。故障测试中挂起的 worker 被确认终止，消耗约 2.893 CPU 秒全部记账。
4. 最终 W 模型现在在进入后续 N 分支前独立导出，包含原生快／慢／适应状态、预测器、前端、可变图及随机／yoke 游标；无 pickle。恢复后的实际只读答案与导出前一致，不能用一个 digest 代替可加载模型。
5. 空间规划按各候选实际确认路径计入导出与归档；1 GiB／2 GiB 是规划预留，而不是单独的硬配额。预留不够时追加到总账，总硬上限仍为 8 GiB。结果包不能包含未接受或部分科学数据。

## 执行与源码覆盖

科学模型与公式继续引用此前通过资格的冻结实现，科学来源 ID 未改。第二轮后只修改了汇总器、调度器、打包预算处理与测试驱动，没有修改科学 worker、模型、出生规则、输入、生命史、样本数或候选剂量。

因此保留第二轮所有实际学习证据；修订后的汇总器与调度器重新测试，并重新执行四／八 worker 与会话暂停测试。两组已完成完整生命史只重新读取、审计和恢复只读输出，不重复学习。每轮实际执行身份、生产文件哈希和故障测试日志均保留；没有把旧收据伪装为新执行身份。旧清单在新源码下被调度器拒收，也保留了这次负面测试记录。

当前最终执行身份：`4a14b8edb220104d2c8578a4fcf351095b50ffb1f7fff36ee7775f704b030cfb`。

科学实现身份：`0701272ef469e110a0a5286fe7022cc2103f6cac4e2a357fce47b4c6d9a98fe2`。

本次有效资格目录累计提交 **2 个完整开发任务／6 条完整生命史，以及 65 个短 W 开发任务**。含修订前后的操作测试，不应当视为 65 个独立科学样本；另有两个故意挂起的故障注入任务，无完成收据。更早一次比较器失败后的短任务单独保留在 superseded 目录中。此前 v1 的 29 个完整开发任务没有全部重跑。正式科学轨迹为 **0**。

## 资源与十小时限制

| 入选路径 | 含 1.5 倍余量的 science CPU h | charged worker h | 理想八 worker 会话 h | 全交付空间 GiB |
|---|---:|---:|---:|---:|
| ERROR | 31.42 | 32.36 | 4.04 | 5.890 |
| P0005 | 39.02 | 40.26 | 5.03 | 6.456 |
| P005 | 39.13 | 40.34 | 5.04 | 6.451 |
| Q_HALF | 31.42 | 32.36 | 4.04 | 5.890 |
| REL05 | 43.24 | 44.42 | 5.55 | 6.483 |
| REL10 | 66.42 | 67.98 | 8.50 | 7.958 |
| REL20 | 43.16 | 44.32 | 5.54 | 6.484 |
| S3_CUE | 50.06 | 51.27 | 6.41 | 7.931 |

最慢路径约 8.50 小时，最紧空间预测约 **7.958 GiB**。空间余量只有约 43 MiB，属于接近上限的规划；不得据此保证所有未来世界或归档一定成功。运行时执行 CPU／worker／RSS／总磁盘硬限，超过预算停止并明确报告 RESOURCE_INCOMPLETE，不能缩减已承诺证据或悄悄增加预算。

四／八 worker 的短任务窗口实测约 24.44／18.35 秒，状态一致，但这不是持续满载的吞吐或 p95 证明。

从整个会话启动计时：8h 停派、9h 请求安全暂停、9.5h 终止不响应 worker；分析、审计与打包也检查同一会话时钟。正常暂停后的下一窗口只允许显式恢复同一输入与清单；科学失败不自动重启。正式运行时 caffeinate 绑定 supervisor。**十小时限制由暂停与停止保证，不是承诺一次窗口内全部完成。**

## 下一道门

已实现并锁定 8-world screen → 最多一个配置／一个目标 → 48 个全新世界确认，含条件机制对照。筛选无合格候选、确认 RETAIN_V1 或 PROMISING_BUT_UNRESOLVED 都是合法结束，不扩 n、不替换候选、不追加新的机制实验。

当前 `science_authorized=false`，没有启动授权文件、正式 SELECTION_LOCK 或科学收据；没有上传 GitHub。本次请求只完成三轮练习。

## 文件

- [实验协议](/Users/pencilbard/Downloads/second_try_copy/PROJECT_DOCUMENTS/MINIFLY_NEXT_CORE_BUDGET_V2_20261006/EXPERIMENT.md)
- [代码与使用说明](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/next_core_budget_v2_20261006/README.md)
- [最终资格](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/next_core_budget_v2_20261006/results/QUALIFICATION.json)
- [资源审计](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/next_core_budget_v2_20261006/results/qualification/qualification_20261006_v2_r2/RESOURCE_AUDIT.json)
- [源码锁](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/next_core_budget_v2_20261006/SOURCE_LOCK.json)
- [操作故障注入审计](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/next_core_budget_v2_20261006/results/qualification/qualification_20261006_v2_r2/OPERATIONS_AMENDMENT.json)
- [最终第三轮详细证据](/Users/pencilbard/Downloads/second_try_copy/DIGITALBABY_CLAUDE_B_D4E4578_20260929/next_core_budget_v2_20261006/results/qualification/qualification_20261006_v2_r2/CYCLE3.json)
