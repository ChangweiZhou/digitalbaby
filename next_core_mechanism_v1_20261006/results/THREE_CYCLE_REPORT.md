# MiniFly 下一核心：三轮修订—审查—试跑报告

日期：2026-10-06。范围：此前注册的 Q/P/S/REL 设计；未新增科学任务、候选机制、样本或验收门槛。

**三轮已完成。功能与因果接口测试通过，但第三轮资源门未通过，整体未取得正式运行资格。正式科学 trajectories 为 0。**

源码身份：`0701272ef469e110a0a5286fe7022cc2103f6cac4e2a357fce47b4c6d9a98fe2`。三轮最终记录绑定同一源码。`SPEC_LOCK.md` 保留原设计；本报告更新实际实现状态。仅保存 [源码快照](SOURCE_SNAPSHOT.json)，没有通过失败的资源门生成 science SOURCE_LOCK 或启动 science。

## 三轮结果

| 循环 | 修订及审查重点 | 实际试跑结果 | 裁决 |
| --- | --- | --- | --- |
| 1 | 原 CENTER/ERROR 对源；独立 canonical birth；REL 双路径、字节时钟、alpha 剂量；Q_HALF 仅读出 | CENTER/ERROR 各 8 条记录逐状态对源；11 配置独立出生及阶段禁写；额外 bank 不改变 parent 状态 | PASS |
| 2 | 实际监督写入、泄漏和错误接口攻击；P/S 无标签适应与监督写入分开 | 17 个攻击被拒绝；P/S 四配置各 32 条 W/N 图状态逐位一致 | PASS |
| 3 | 所有注册配置×任务完整运行；恢复、并发、侧车与资源 | 29 jobs、87 条完整生命史、50,832 条教学记录，全部独立审计通过；资源预测超限 | RESOURCE_NOT_QUALIFIED |

证据：[第一轮](CYCLE1.json)、[第二轮](CYCLE2.json)、[第三轮](CYCLE3.json)、[汇总资格记录](QUALIFICATION.json)。

全部完整 job 使用 development world `61005003`，覆盖配置与任务；它们不是 87 个独立科学世界。短测试另用注册 development IDs，不混入正式 n。没有根据开发准确率或 held-out 分数选择参数。

## 已落实的修订

- 恢复独立 pinned Python 环境：Python 3.11.5、numpy 2.2.6、scipy 1.14.1、numba 0.61.2、pandas 2.2.3。旧 frozen 源码和 canonical 出生资产未改写。
- 新任务模块更名为 `assays.py`，避免与继承源码的 `fixture` 模块重名。
- REL 同时替换 association 写入和查询地址，保留 predictor feature、原 byte/pending/rest 调用及零时长调用；答案进入前缓存地址。只改一条地址路径和多加逐-byte native stimulus 的错误均被拒绝。
- 禁写审计观察实际 native 调用，并与同一 prestate 的 nonplastic 转移比较。衰减造成的状态变化不被误判为写入；计数器不替代 plastic delta。
- P/S 的 cue-only 适应不随监督禁写关闭；保持每 KC 的 degree/budget，S 对照使用成功移动数 yoke，独立 counter 游标被保存。
- 修复 clone 的可变 yoke 列表共享；完整保存图、支持、计数、alpha_scale、输入上下文和恢复游标。
- 累计恢复前后的 CPU；Q_HALF 使用 ERROR 学习轨迹，额外计其最终读出成本；预算同时计 checkpoint、日志和资格开销。

细节：[修订记录](REFINEMENT_RECORD.json)、[固定投影与支持](architecture/FIXED_ARCHITECTURE.json)。中间版本 smoke/预检记录不为最终源码取得资格。

## 恢复、并发与数据审计

11 个物理配置分别比较不中断 32-record reference 与在 record 24 安全 checkpoint 后实际 SIGTERM、fresh-process 恢复到 record 32 的结果。学习状态、图、游标、输出和记录逐位一致；篡改数组、错源码身份、重新封印的非法 branch cursor 被拒绝；已有 receipt 再次调用只跳过，不重跑。

**恢复证据限于安全 checkpoint 边界终止；没有把它描述成任意 native 运算中途崩溃的恢复保证。** 共有 11 份 KILL_RECORD。恢复目录 24 个短 receipt，其中两份是 S3_RAND 的 S3_CUE yoke 来源。

四／八并发各运行 10 个非依赖配置的相同 2-record fixture，科学状态与输出一致。S3_RAND 的依赖 yoke 和恢复另行验证，未声称它参加这 10-arm 并发对照。并发测试另有 20 个短 receipt。

[侧车审计](SIDECAR_E0_AUDIT.json) 核验 3,856 条 h/q digest，且 CENTER/W/Latin/OLD_END 的 12 taught＋4 held-out 全部保存。原学习过程没有为取回向量重跑，侧车不反馈到模型。

## 实测资源与原定上限

以下是正式 2,032 jobs / 6,096 lives 的外推，不是已发生的正式消耗。

| 项目 | 实测外推或保守计划 | 原上限 | 结果 |
| --- | ---: | ---: | --- |
| science process CPU 点估计 | 121.77 h | — | 量级估计 |
| science CPU ×1.5＋本次最终资格 CPU | 184.50 h | 180 h | 超限 |
| science charged worker ×1.5＋资格上界 | 191.81 h | 200 h | 带内 |
| science receipts 点估计 | 2.55 GiB | — | 量级估计 |
| science checkpoints 点估计 | 4.78 GiB | — | 量级估计 |
| receipts/checkpoints ×1.5＋1 GiB 日志＋资格目录 | 12.17 GiB | 8 GiB | 超限 |
| 已测单进程峰值 RSS | 568.1 MiB | 1024 MiB | 带内 |

最终源码三轮资格过程计量 CPU 为 1.85 h，第三轮 wall time 为 37.4 分钟。更早的环境恢复、smoke 和中间修订重试还含少量额外开销；这只会增加总量，不会改变上述超限裁决。

每 recipe 仅一个完整 development job，不能可靠估计 job p95；1.5× 是工程规划余量，**不是置信上界**。未伪造 tail guarantee。状态尺寸字段目前是所列数组的保守记账，不能当作全部 Python 对象或 byte 精确的可变／固定资产分解；资源门使用实测 RSS 与磁盘尺寸。

存储超限很大一部分来自 completed receipts 与仍保留的完整 checkpoint 并存。正式执行前应收敛 checkpoint 保留和非科学审计/I/O 开销；若选择修改预算，必须明确修订资源合同。本轮没有靠减少 n、删配置、换世界或调科学门槛来取得 PASS。

## 结论与交付范围

本轮获得 E0 实现证据：当前代码按注册协议完成字节输入、原生学习、禁写干预和输出，恢复与并发测试通过。**它没有确认任何候选比 V1 更好，也没有取得 E1/E3 机制改善结论，更不能据此筛除整个机制家族。**

这是资格构建：worker 限制为 development IDs；正式批执行器、完整采用-profile 分析封装及正式世界排除名册尚未交付。原定资源门也未通过，所以当前不能声称已经可正式启动并自动完成 science/选版。上述工程工作仍属于同一个冻结科学目标，不需要增加新任务或新一串机制定位实验。

未自动追加第四轮；未启动正式实验；未上传 GitHub。代码和全部技术证据保留在本目录。

## 完整试跑清单

| 配置 | 任务 | 完整 lives | 教学记录 | 审计 |
| --- | --- | ---: | ---: | --- |
| CENTER | latin | 3 | 1152 | PASS |
| CENTER | lifetime | 4 | 3456 | PASS |
| CENTER | reuse | 2 | 432 | PASS |
| ERROR | latin | 3 | 1152 | PASS |
| ERROR | lifetime | 4 | 3456 | PASS |
| ERROR | reuse | 2 | 432 | PASS |
| FIRST10 | latin | 3 | 1152 | PASS |
| FIRST10 | lifetime | 4 | 3456 | PASS |
| FIRST10 | reuse | 2 | 432 | PASS |
| P0005 | lifetime | 4 | 3456 | PASS |
| P0005 | reuse | 2 | 432 | PASS |
| P005 | lifetime | 4 | 3456 | PASS |
| P005 | reuse | 2 | 432 | PASS |
| REL05 | latin | 3 | 1152 | PASS |
| REL05 | lifetime | 4 | 3456 | PASS |
| REL05 | reuse | 2 | 432 | PASS |
| REL10 | latin | 3 | 1152 | PASS |
| REL10 | lifetime | 4 | 3456 | PASS |
| REL10 | reuse | 2 | 432 | PASS |
| REL20 | latin | 3 | 1152 | PASS |
| REL20 | lifetime | 4 | 3456 | PASS |
| REL20 | reuse | 2 | 432 | PASS |
| REL_PERM10 | latin | 3 | 1152 | PASS |
| REL_PERM10 | lifetime | 4 | 3456 | PASS |
| REL_PERM10 | reuse | 2 | 432 | PASS |
| S3_CUE | lifetime | 4 | 3456 | PASS |
| S3_CUE | reuse | 2 | 432 | PASS |
| S3_RAND | lifetime | 4 | 3456 | PASS |
| S3_RAND | reuse | 2 | 432 | PASS |
