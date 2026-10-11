# 有限加法数值保持：正式确认报告

判定：**ADOPT_STABLE_LIMITED_ADDITION_V2**。64个全新确认世界，128个world-arm作业，384条训练历史，1540608训练字节。全部独立字节、数值、禁写及留出审计通过。

|规则|最终旧题|最终后续题|最终未教组合|
|---|---:|---:|---:|
|DECAY|68.63%|94.53%|98.05%|
|STABLE|100.00%|100.00%|100.00%|

三个预定配对一侧95%下界：

- old_gain_vs_decay: 均值 31.37 pp；下界 30.55 pp；PAIRED_STUDENT_ONE_SIDED_95。
- gain_no_old: 均值 75.78 pp；下界 75.14 pp；PAIRED_STUDENT_ONE_SIDED_95。
- gain_shuffled: 均值 83.40 pp；下界 81.27 pp；PAIRED_STUDENT_ONE_SIDED_95。

全部绝对与差值门槛：{"held": true, "old": true, "new": true, "gain_no_old": true, "gain_shuffled": true, "old_gain_vs_decay": true, "shortcut_gap": true, "held_vs_decay": true}。方差近零采用预定Hoeffding保守下界；没有补世界或换方法。

这是操作数0–4、答案0–8、80%正确/10%错误/10%缺失示例下的有限加法实例；模型在答案反馈前输出。完整交换轨道从教学中留出。工程数字大小、角色表示和ordinal输出先验公开，不是原生Full151发现任意算术，也不是无外部反馈学习。确认没有增加NLP、多位数、新机制或任务。所有对照、逐题和检查点均保留。24小时是模型时钟推进，不是实际等待。

若采用，范围仅为该限定任务和既定保持窗口；容量、规则变化及错误知识长期修订未经本实验认证。GitHub上传未授权。
