# 第一轮 设计 审核 试跑 修订

设计：将实际V82E phenotype及Full151事件、动态allocation和signed teaching移植为functional float64模型；所有32份记忆独立canonical出生。对照运行真实母体SequenceC16。

审核：不能把fast/slow分配写成固定比例；private signed需按Sn+s(W1-W0)处理gamma及alpha；查询使用母体5秒observer而非写入dt；两种读出不能混为同一种。

试跑：attempt1在导入前被继承源的通用model模块名冲突阻止，0记录。改用bind_kernel/bind_native唯一模块名；attempt2完成20条实际技术记录。state与读出最大差约1.42e-14，no-write/rest/256byte混合输入与257输出码本验证通过；四记录真实native-write tape梯度范数0.1030。

修订：加入非零状态下禁写、跨记录信用、有限差分和特殊机制独立检查；不把本轮非零梯度或少量技术答案当作E1/E3成绩。通过结果及执行源码见attempt2。源码快照按运行记录的SHA精确复原，过程另有SOURCE_SNAPSHOT_NOTE。
