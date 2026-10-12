# 首轮16个具体操作的保留状态

仅描述原样配方/条件变体，不把无效E3当科学负面。完整操作来源在[原变体清单](../VERSION_REGISTER_20261011/sources/MINIFLY_FIRST_ROUND_VARIANT_INVENTORY_20260928.md)。

| 操作 | 现有状态 | 有效范围/条件 |
|---|---|---|
| R1 | STOP_RECIPE | E1负差；有效E3未显示增量。 |
| R2 | STOP_RECIPE | E1无改善；修正E3 −2.1pp，区间跨零；区别于当前ERROR。 |
| R3 | REFERENCE | 保留R_center与ERROR；shared预测本身的贡献未获focused ablation支持。 |
| Z1 | STOP_RECIPE | E1无改善；修正E3比Z0差14.6pp，区间全负。 |
| Z2 | STOP_RECIPE | 无改善且taught guards弱。 |
| P1 | STOP_RECIPE | 原规则有效负结果；.005/.0005移植版筛选未晋级，停止继续同路剂量扫描。 |
| P2 | STOP_RECIPE | 原规则有效负结果；归一化binary Oja不提供独立稳定机制。 |
| P3 | TECH_HOLD | 技术未资格化，无科学否定；暂不分配修复运行预算。 |
| P4 | CONDITIONAL | 只留tau165s一档，非原样复跑。 |
| T1 | STOP_RECIPE | 有效E1无增益；E3无效，整族未被排除。 |
| T2 | CONDITIONAL | 原样停止；只留4E较弱优化一档低优先级备选，E3原结果无效。 |
| T3 | CONDITIONAL | 原样停止；只留bridge .4一档低优先级备选，E3原结果无效。 |
| S1 | STOP_RECIPE | 独立弱边SET无E1增益；P×S是新结构，不作为旧S1复活。 |
| S2 | CONDITIONAL | 原样停止；只留每事件8次换边一档低优先级备选，E3无效。 |
| S3 | STOP_RECIPE | 原样停止；rho匹配cue-clock移植筛选未晋级。rho-only未经确认，但无充分理由优先再测。 |
| S4 | CONDITIONAL | 原样停止；只留每事件8次暂试一档低优先级备选，E3无效。 |
