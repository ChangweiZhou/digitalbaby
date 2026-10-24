# BYTE-LEARNER-1 v8：平衡首区块工程试跑

**日期：** 2026-09-25  
**状态：** v7 的修订试跑；仍非正式 CAL/DEV/CON。保留 [v7](./BYTE_LEARNER_CAUSAL_SHARED_DESIGN_v7_20260925.md) 的 `CAUSAL_DT` 字节／教学协议、三臂、真实 FULL_151 写入、独立文本工程评分器及禁止科学确认结论的边界。

## 修订原因与证据

v7 先运行了两个工程世界各前四个 pair、六次 bout。第二个世界的四个 pair 全部是 side-0 中性，预先指定的 side-0 AUC 无定义；第一个世界只有一个 side-0 厌恶 pair。虽 `Teach` 真正写入 FULL_151，且时钟与无写入配对检查通过，但这个切样不能评价区分度。该结果保存在 `BYTE_LEARNER_TRIAL_V7_20260925/TRIAL_RESULTS.json`，不得被新试跑覆盖或当作独立复现。

## 固定的 v8 试跑

使用新工程命名空间 `BYTE-LEARNER-V8-ENG`、`TRIAL`、index 0 和 1、R2、massed。对每个世界只取**首个 acquisition block 的全部 48 个 pair，每 pair 六次 bout**，按生成器原有顺序逐字节呈现，不改变结果、顺序或时钟。首区块的 truth role 构造保证 side-0 厌恶与中性各 24 个 pair，因而重复事件各 120 条。每个世界 288 pair-trial、576 条 14-byte F3 记录。两世界仍只是工程样本，不估计正式效力。

三臂：`TRACE88 + CAUSAL_DT plastic`、同码同钟 `TRACE88 + CAUSAL_DT no-write`、`FE0 + CAUSAL_DT plastic`。每条记录结果字节前 read-only 读值，结果字节后显式 `Teach(0/1,t)`；字节与神经时间每条记录末严格对齐。报告 side-0 重复事件的描述性 AUC，以及同一 pair 的结果前厌恶／中性读值比较产生的描述性配对选择准确率与 signed margin。逐臂报告写入状态差、时钟差、预结果码一致性、教学次数和运行时。no-write 是完整的同一适配器、同一输入、`plastic=False`，不是删掉教学事件或时间推进。

文本评分器仍用 v7 已列的两份不同本地 Markdown，10k 训练字节、4k 冻结测试字节。只把它视为逐字节预测器和基线实现的工程检查。记录 manifest 文件哈希、脚本哈希、世界哈希及验收检查，不复制原始材料。

## 审计及退出条件

必须核查结果前读出时序、所有 Byte/Teach 与休眠区间恰好提交一次、同码对照、真实 plastic-vs-no-write 状态分叉，以及 AUC 的两类样本存在。另要明确 `CAUSAL_DT` 与旧 30 秒塑性剂量不同，且首区块不含 24h retention、revision、扰动或长期开放字节流。即使工程检查全部通过，仍不可据此启动 140-life 确认或宣称字节序列学习成功；正式语料、剂量合同与完整生命资格仍需另行固定。
