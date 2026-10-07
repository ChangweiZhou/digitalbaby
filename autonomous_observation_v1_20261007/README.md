# 自主观察驱动的 MiniFly 原生记忆

正式确认已完成：**ADOPT_AUTONOMOUS_OBSERVATION_BASELINE**。32 个新世界、96 条学习生命史、122,880 个学习字节，七个冻结门槛全部通过。实际总耗时约 14.5 分钟。

- [正式报告](results/confirm/REPORT.md)
- [正式结果包](results/confirm/RESULT_BUNDLE.zip)
- [统计汇总](results/confirm/SUMMARY.json)与[最终审计](results/confirm/FINAL_AUDIT.json)
- [完整世界收据](results/confirm/receipts/)与[最终状态](results/confirm/STATUS.json)
- [科学协议](EXPERIMENT.md)、[正式启动审查](ops/confirm_20261007/OPERATIONS_AUDIT.md)与[后续运行授权](ops/confirm_20261007/AUTHORIZATION.json)

模型在每个普通字节到来前预测，随后从实际字节生成内部预测误差，更新 Full151 原生快慢记忆。没有外部正确性 bit、答案槽教学或可训练旁路预测器。第二天旧内容保持 99.12%，禁旧写对照 24.32%；新内容和修订内容均 99.61%，未修订内容 89.06%。

这是 ASCII `0123`、最近四个原始字节的工程地址、四份独立原生存储的 E0/E1 自主获取、保持和修订基线。E3 未测试；精确上下文计数器也能解决此任务。不声明关系推理、任意文本或已获得可扩展的新通用核心。

顶层 REPORT.md、QUALIFICATION.json、FINAL_AUDIT.json 和 THREE_CYCLE_BUNDLE.zip 是先前的三轮 DEV 资格记录；正式结果在 results/confirm/。DEV 与隔离的调度/打包替身测试不计入正式样本。旧锁里尚未授权 science/upload 的字段记录当时状态；后来的人类运行授权和本次 publication/AUTHORIZATION.json 单独保存，不改写历史。

原生父资产已位于仓库兄弟目录 [r_center_core_v1](../r_center_core_v1/)。运行环境为 Python 3.11.5、numpy 2.2.6、scipy 1.14.1、numba 0.61.2。已完成的正式 launcher 拒绝重复启动；本次发布没有重跑实验。
