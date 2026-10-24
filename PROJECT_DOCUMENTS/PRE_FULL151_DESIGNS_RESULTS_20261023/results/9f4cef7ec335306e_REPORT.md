# Post-V5 FULL v3：完整结果

## 结果解释与机制对照（完成后解读，不改变预注册判定）

本轮取得了有统计支持的性能改进，但不能把收益归于路由自适应。K8 的152个世界中，B的首次返回正确率为49.986%，S_LOCAL为55.674%，配对差为+5.688个百分点，预定多重比较校正区间为[3.681, 7.696]个百分点。

固定路由对照只在预定的48个K8世界上运行。以下三臂均限于这同一子集，数值为世界平均首次返回正确率：

|B|S_LOCAL|S_FIXED|
|---:|---:|---:|
|49.002%|54.297%|73.351%|

预定的S_LOCAL−S_FIXED机制对比为−19.054个百分点，校正区间为[−22.295, −15.812]。它否定了本轮具体router适应规则有益的预期；不是所有自适应路由机制的反证。S_FIXED比B的表中差异为描述性比较，不能把48个世界扩称152个世界的正式候选验收，也未为该候选单独执行全套正式推进判定。

下一步更值得区分固定分库存储的收益来自额外容量、隔离干扰还是读写地址稳定；路由漂移破坏旧地址目前只是待测解释。当前S_LOCAL状态分配为1,159,744字节，B为165,688字节，约7倍；S_FIXED总状态分配相同，其中331,328字节路由中心固定。公共B数组和原生静态资产另外计账，因此本轮不证明等资源压缩优势，也不证明已经找到最小持续学习核心。

“没有显著保护指标损害”不等于没有旧记忆代价。K8的139/152个世界出现过B正确而S_LOCAL错误的相关已知项目决策；逐世界Q均值5.538%。这按多个checkpoint上的重复决策计数，不能解释为独立记忆永久丢失率。稳定遗忘差异区间仍跨零，风险标记保留RISK_UNRESOLVED。

## 报告恢复记录

全部720条科学/控制轨迹已在2026-09-21 13:58 UTC前完成，首次运行约49分钟。自动汇总因环境缺少matplotlib在绘图步骤中断；补装绘图库后，用原冻结runner验证全部336份world receipts并完成汇总、绘图和打包，没有重跑科学轨迹或修改冻结模型/统计规则。STATUS中的elapsed_seconds是此次恢复汇总耗时，不是完整实验耗时。额外审计遗漏了报告绘图依赖，是本次工程验收的不足。


科学世界336；主科学轨迹672，加固定路由对照48，共720。合格槽位 S；W/H 未实例化，不能视为科学阴性。

主结论：`PRIMARY_GAIN_CONFIRMED`；保护指标：`NO_CONFIRMED_GUARD_HARM`。
进入后续研究：True；风险：`RISK_UNRESOLVED`。

B 仍是冻结参考；研究候选标记不自动替换父模型，不是安全、非劣或最小核心证明。

|负载|指标|S−B pp|同步区间 pp|
|---|---|---:|---|
|canonical|P1_current_usefulness|0.653|[0.284, 1.022]|
|canonical|P5_acquisition|2.818|[2.120, 3.515]|
|canonical|P6_anomaly_recovery|0.380|[-0.201, 0.961]|
|canonical|acquisition_24h|2.417|[1.708, 3.125]|
|canonical|revision_24h|0.054|[-0.694, 0.803]|
|canonical|usefulness_7d|0.091|[-1.042, 1.223]|
|canonical|first_use_return|6.069|[3.033, 9.105]|
|canonical|stable_loss_last_day_to_final_plus7d|0.000|[-0.214, 0.214]|
|k8|P1_current_usefulness|4.586|[3.772, 5.401]|
|k8|P5_acquisition|2.405|[1.897, 2.913]|
|k8|P6_anomaly_recovery|3.561|[2.497, 4.625]|
|k8|acquisition_24h|2.265|[1.780, 2.750]|
|k8|revision_24h|2.146|[1.162, 3.130]|
|k8|usefulness_7d|2.316|[1.171, 3.462]|
|k8|first_use_return|5.688|[3.681, 7.696]|
|k8|stable_loss_last_day_to_final_plus7d|-0.055|[-0.171, 0.061]|

![关键区间](KEY_FIGURE.png)

统计：独立世界为单位，canonical/K8各 alpha=.025，M=24；paired-t 是近似。mechanism另有 alpha=.05、M=3家族。零方差保持未决。

Adaptive−fixed 路由控制的首次返回差：-19.054 pp；支持预定特异性方向：False。只有一个固定 router 初始化，结论不推广到全部初始化分布。

稳定遗忘比较 block7_day 到 final_7day，间隔七天，相对最后全局教学为第1到第8天；共同已知人口覆盖见 STABLE_SUPPORT.csv。Q/R 为重复决策的绝对配对成本，不能当独立事实样本；全部逐世界尾部保留。

模型：V24 路由启发的新移植，八套 alpha 快/慢存储，共享 gamma 与适应，选中 bank 内完整原生 B 更新；并非原 V24 多分类读出逐式恢复。额外存储完整计费；不宣称等资源优势。

科学前完成 B 原实现逐 checkpoint 等价、one-bank 恒等、被动时钟、只读 observer、clone/序列化和额外端到端试运行。见 QUALIFICATION/EXTRA_AUDIT 和 protocol manifest。

W/H来源参考：[Jiang & Litwin-Kumar](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009205)、[Aso & Rubin](https://elifesciences.org/articles/16135)。未获得当前模型的无歧义有界状态/定量解剖双策略映射，因此未实例化。
