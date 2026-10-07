# MiniFly budget v2 confirm

固定字节接口与既有 Full151 机制；不扩展任务、样本或候选。

| 配置 | n | old E1 | new E1 | revision E1 | reuse W | reuse W−N | W online CPU s |
|---|---:|---:|---:|---:|---:|---:|---:|
| CENTER | 48 | 0.9497 | 0.9564 | 0.3776 | 0.5486 | 0.0069 | 93.3893 |
| ERROR | 48 | 0.9774 | 0.9974 | 0.9635 | 0.5764 | 0.0382 | 94.5694 |
| Q_HALF | 48 | 0.9766 | 0.9967 | 0.9531 | 0.6111 | 0.0868 | 94.5716 |

裁决：`PROMISING_BUT_UNRESOLVED`。

E1 是教过键的回忆/保留；E3 是未强化关系的有限复用。精确键查表可解决 E1，故 E1 单独通过不证明 E3、推理或自主学习。所有测试仍采用提示输出与到来的答案字节教学。筛选结果是探索性的；确认只检验预先锁定的一组候选和目标。

完整收据/不可用标记：192；所有配置均保留。

```json
{
  "verdict": "PROMISING_BUT_UNRESOLVED",
  "candidate": "Q_HALF",
  "parent": "ERROR",
  "frozen_target": "reuse",
  "adoption": {
    "alpha": 0.03,
    "passed": false,
    "components": {
      "gain": {
        "mean": 0.04861111111111111,
        "lower": 0.018430044679967665,
        "sd": 0.10849460627251718,
        "method": "paired_Student_approximate",
        "alpha": 0.03,
        "n": 48
      },
      "efficiency_gain": {
        "mean": 8.402539047656527,
        "lower": 3.304169367962868,
        "sd": 18.32757011062138,
        "method": "paired_Student_approximate",
        "alpha": 0.03,
        "n": 48
      },
      "prior_teaching": {
        "mean": 0.08680555555555554,
        "lower": -0.0013651064710057392,
        "sd": 0.31695504475242386,
        "method": "paired_Student_approximate",
        "alpha": 0.03,
        "n": 48
      },
      "raw_above_chance": {
        "mean": 0.11111111111111106,
        "lower": 0.04061349581326956,
        "sd": 0.2534241469677889,
        "method": "paired_Student_approximate",
        "alpha": 0.03,
        "n": 48
      }
    },
    "engineering_protection": {
      "passes": true,
      "reasons": [],
      "CPU_ratio": 1.0000235973084617,
      "harms": [
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0
      ]
    },
    "harm_upper": 0.19492129870721744,
    "harm_risk_passed": false
  },
  "E3_existence": {
    "alpha": 0.01,
    "passed": false,
    "components": {
      "prior_teaching": {
        "mean": 0.08680555555555554,
        "lower": -0.02337266639770251,
        "sd": 0.31695504475242386,
        "method": "paired_Student_approximate",
        "alpha": 0.01,
        "n": 48
      },
      "raw_above_chance": {
        "mean": 0.11111111111111106,
        "lower": 0.02301716095883602,
        "sd": 0.2534241469677889,
        "method": "paired_Student_approximate",
        "alpha": 0.01,
        "n": 48
      }
    },
    "competence": true
  },
  "mechanism": {
    "alpha": 0.01,
    "registered": false,
    "available": true,
    "status": "AVAILABLE",
    "passed": false,
    "components": {}
  },
  "n": 48,
  "worlds": [
    61008101,
    61008102,
    61008103,
    61008104,
    61008105,
    61008106,
    61008107,
    61008108,
    61008109,
    61008110,
    61008111,
    61008112,
    61008113,
    61008114,
    61008115,
    61008116,
    61008117,
    61008118,
    61008119,
    61008120,
    61008121,
    61008122,
    61008123,
    61008124,
    61008125,
    61008126,
    61008127,
    61008128,
    61008129,
    61008130,
    61008131,
    61008132,
    61008133,
    61008134,
    61008135,
    61008136,
    61008137,
    61008138,
    61008139,
    61008140,
    61008141,
    61008142,
    61008143,
    61008144,
    61008145,
    61008146,
    61008147,
    61008148
  ],
  "nominal_FWER": 0.05,
  "t_coverage": "approximate"
}
```
