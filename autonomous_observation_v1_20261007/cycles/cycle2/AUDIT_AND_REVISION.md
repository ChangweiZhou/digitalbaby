# 第二轮审核和修订

独立审计 PASS，重建 8240 条真实学习及冻结探针记录。每分支 1280 个训练字节，三条实际训练生命史，CPU 24.624 s。W 旧键在获取与 day2 均为 100%；day2 N_OLD 为 12.5%，N_ALL 为 25%。新键、修订键均为 100%。

失败保留：修订后未改旧键仅 75%，低于 85% 门槛。不能称全部目标已达成。第三轮继续使用完全相同的 native update、输入、剂量、softmax 尺度和时间，检查四个预先声明的新 DEV 世界；不选参数、不减负载、不降低门槛。

实际修订：修订后的总旧集合 probe 改用 current_old 的更新后标签，revised/unchanged 仍分别报告；原 C2 旧标签 probe 原样保留，不再用它推断真实修订后准确率。N_ALL 的修订子集不必标签平衡，不能一律称 25% chance；概率均匀与固定 tie accuracy 分开写。

补上 native actual elapsed 和 committed clock 字段；加入非 pickle 源码绑定 checkpoint，覆盖块中途且 prediction 待观察时保存恢复；独立 auditor 拒绝 NaN signs，加入真实 byte/sign/删记录/禁写/未来 context/源 hash 的负控。保持主端点冻结为 day2，正式科学仍未授权。
