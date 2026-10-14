# 模型与反馈联合实验结果

结论：**SHORT_TECHNICAL_PASS_NO_EFFICACY_CLAIM**。
完整作业 2；执行外部记录 7168；相关末态历史 10。

整串自身生成与EOS评分，四操作和混合题保持原权重。预热和全部后续题目计入样本面积，抽样、critic和查询计入计算。

| 固定效应 | 配对世界均值 |
|---|---:|
| J_sample | 0.000000 |
| J_compute | 0.000000 |
| scalar_increment | 0.000000 |
| warm_input_+ | -0.011628 |
| binary_+_NO_WARM | 0.000000 |
| binary_+_SH_WARM | 0.000000 |
| binary_+_shortcut | -0.156250 |
| warm_input_- | -0.061728 |
| binary_-_NO_WARM | 0.000000 |
| binary_-_SH_WARM | 0.000000 |
| binary_-_shortcut | -0.031250 |
| warm_input_* | -0.168539 |
| binary_*_NO_WARM | 0.000000 |
| binary_*_SH_WARM | 0.000000 |
| binary_*_shortcut | 0.000000 |
| warm_input_/ | -0.125000 |
| binary_/_NO_WARM | 0.000000 |
| binary_/_SH_WARM | 0.000000 |
| binary_/_shortcut | -0.031250 |
| mixed_NO_WARM | 0.000000 |
| mixed_SH_WARM | 0.000000 |
| mixed_shortcut | -0.160156 |

DENSE为信息参照；FREEZE_SCALAR从相同的过渡期末态分叉。未见题没有接受反馈。
技术世界不能提供显著性或采纳结论；启动失败不会追加dose、降低门槛或替换主要任务。
