# 第三轮冻结候选和综合资格

DEV 810201–810204 各跑同一完整生命史：W/N_OLD/N_ALL，1280 bytes/branch。行为门槛按这四个世界等权均值判断，公布各世界分布；C1/C2 不并入。16 旧键获取 ≥90%、day2 旧键损失 ≤5 pp、W−N_OLD ≥25 pp、新键 ≥80%、修订键 ≥80%、未改键 ≥85%，全 byte bpb 优于 N_ALL。所有门槛必须满足，不能只选有利项。

运行前审核确认 C2 未改旧键失败不可抹除，本轮无模型或参数改动。追加完整原生 clock 日志，修订后按现行真实标签计分。native sources 保持既有父版本；每次 checkpoint 验证父 SOURCE_LOCK 覆盖的实际依赖文件。

技术测试包括 pending prediction 中途 checkpoint vs uninterrupted 原生数组精确相等；禁写状态与直接调用 native no-plastic event 的数组相等；损坏 checkpoint、错源码、未来地址、错误 byte/误差/输出/报告、实际禁写仍写入等 negative controls 必须失败；独立 IID 概率参照与有限 context 计数反模型。

只允许 DEV，不产生 science world，未授权正式 runner/GitHub upload。完成后得出技术和行为两个独立 verdict；失败则 BLOCKED，仍算本次三轮练习已经完成。
