# GATED-FEEDBACK V2 技术试跑结果

本报告只描述固定技术世界，不是正式泛化验证，也不采纳新核心。

完整作业 2；外部记录应用 8192；相关末态历史 14；完整状态档案 32。

整串输出与自身 EOS 共同计分。保持四则单步宏平均与十六种有序运算对混合题宏平均各占一半；没有强制输出格式、候选答案筛选或算术提示。

| 世界 / 模型 | WARM 未见 A | CE_ONLY 末态 A | OLD 末态 A | SAFE 末态 A | FREEZE_SAFE 末态 A | SAFE EOS |
|---|---:|---:|---:|---:|---:|---:|
| 202610142901 / G62 | 0.00% | 3.32% | 0.00% | 0.00% | 0.00% | 100.00% |
| 202610142901 / KEEP | 0.00% | 3.32% | 0.00% | 0.39% | 0.00% | 7.29% |

| 固定必要效应 | 配对技术世界均值 |
|---|---:|
| J_sample_vs_KEEP_SAFE | -0.001465 |
| J_compute_vs_KEEP_SAFE | +0.000000 |
| J_sample_vs_KEEP_OLD | -0.001831 |
| J_compute_vs_KEEP_OLD | -0.002556 |
| binary_+_vs_N_WARM_SAFE | +0.000000 |
| binary_+_vs_SH_WARM_SAFE | +0.000000 |
| binary_+_vs_shortcut_envelope | -0.093750 |
| binary_-_vs_N_WARM_SAFE | +0.000000 |
| binary_-_vs_SH_WARM_SAFE | +0.000000 |
| binary_-_vs_shortcut_envelope | -0.062500 |
| binary_*_vs_N_WARM_SAFE | +0.000000 |
| binary_*_vs_SH_WARM_SAFE | +0.000000 |
| binary_*_vs_shortcut_envelope | +0.000000 |
| binary_/_vs_N_WARM_SAFE | +0.000000 |
| binary_/_vs_SH_WARM_SAFE | +0.000000 |
| binary_/_vs_shortcut_envelope | -0.031250 |
| mixed_vs_N_WARM_SAFE | +0.000000 |
| mixed_vs_SH_WARM_SAFE | +0.000000 |
| mixed_vs_shortcut_envelope | -0.128906 |
| scalar_increment | +0.000000 |

20 个必要效应均值是否全部大于零：False。这不是显著性结论。

短剂量计算面积临时上限=3.238219 CPU 秒，只用于本轮技术描述，不得充当 DEV 的 Cstar。正式共同上限只能由第三轮完整剂量 KEEP_OLD 预定规则确定。

CE_ONLY 获得更多答案信息，只作信息参照。N_WARM_SAFE 和 SH_WARM_SAFE 控制先前教学的作用；FREEZE 从各自 SPARSE 末态分叉。SAFE 的约束仅作用于当前实际采样前缀，不保证整体策略 KL 或未来题目不退化。

SUMMARY.json 保留每阶段已教题回忆、未见题 A、EOS、输出长度、实际奖励、critic 基线、拒绝次数、全部候选评估及实际 CPU。技术资格通过不等于 E3 通过。
