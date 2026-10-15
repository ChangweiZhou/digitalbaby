# GATED-FEEDBACK V2 技术试跑结果

本报告只描述固定技术世界，不是正式泛化验证，也不采纳新核心。

完整作业 4；外部记录应用 442368；相关末态历史 28；完整状态档案 64。

整串输出与自身 EOS 共同计分。保持四则单步宏平均与十六种有序运算对混合题宏平均各占一半；没有强制输出格式、候选答案筛选或算术提示。

| 世界 / 模型 | WARM 未见 A | CE_ONLY 末态 A | OLD 末态 A | SAFE 末态 A | FREEZE_SAFE 末态 A | SAFE EOS |
|---|---:|---:|---:|---:|---:|---:|
| 202610142902 / G62 | 10.16% | 14.06% | 4.49% | 10.55% | 10.74% | 100.00% |
| 202610142902 / KEEP | 16.02% | 17.58% | 2.54% | 14.26% | 14.06% | 99.48% |
| 202610142903 / G62 | 13.67% | 17.38% | 6.84% | 11.91% | 12.89% | 100.00% |
| 202610142903 / KEEP | 11.13% | 13.48% | 4.49% | 12.50% | 12.89% | 99.22% |

| 固定必要效应 | 配对技术世界均值 |
|---|---:|
| J_sample_vs_KEEP_SAFE | -0.008650 |
| J_compute_vs_KEEP_SAFE | -0.014877 |
| J_sample_vs_KEEP_OLD | +0.002999 |
| J_compute_vs_KEEP_OLD | +0.004804 |
| binary_+_vs_N_WARM_SAFE | +0.062500 |
| binary_+_vs_SH_WARM_SAFE | +0.046875 |
| binary_+_vs_shortcut_envelope | -0.125000 |
| binary_-_vs_N_WARM_SAFE | +0.093750 |
| binary_-_vs_SH_WARM_SAFE | +0.078125 |
| binary_-_vs_shortcut_envelope | +0.078125 |
| binary_*_vs_N_WARM_SAFE | +0.000000 |
| binary_*_vs_SH_WARM_SAFE | +0.000000 |
| binary_*_vs_shortcut_envelope | +0.000000 |
| binary_/_vs_N_WARM_SAFE | +0.125000 |
| binary_/_vs_SH_WARM_SAFE | +0.125000 |
| binary_/_vs_shortcut_envelope | +0.125000 |
| mixed_vs_N_WARM_SAFE | +0.130859 |
| mixed_vs_SH_WARM_SAFE | +0.041016 |
| mixed_vs_shortcut_envelope | +0.033203 |
| scalar_increment | -0.005859 |

20 个必要效应均值是否全部大于零：False。这不是显著性结论。

计算面积共同上限 Cstar=96.512607 CPU 秒；由本阶段 KEEP_OLD 固定完整剂量技术作业的中位数确定。后续若获准 DEV，不得根据 DEV 结果重新设定。

CE_ONLY 获得更多答案信息，只作信息参照。N_WARM_SAFE 和 SH_WARM_SAFE 控制先前教学的作用；FREEZE 从各自 SPARSE 末态分叉。SAFE 的约束仅作用于当前实际采样前缀，不保证整体策略 KL 或未来题目不退化。

SUMMARY.json 保留每阶段已教题回忆、未见题 A、EOS、输出长度、实际奖励、critic 基线、拒绝次数、全部候选评估及实际 CPU。技术资格通过不等于 E3 通过。
