# E18b / E18c / E23 tables (written by e18b_e23_score.py from receipts)

## Births, final checkpoint, k = 12, main head

| Cell | Seed | add89 | sub89 | max89 | min89 | Hard | Max/min | Ckpt-mean hard | Hard curve (6 ckpts) | First head hard | CPU-s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| B2 | 73129 | 81.3 | 68.3 | 95.7 | 96.3 | **74.8** | 96.0 | 47.7 | 5.8 / 29.0 / 46.7 / 60.7 / 69.2 / 74.8 | 17.5 | 4,111 |
| B2 | 93683 | 76.3 | 63.0 | 97.3 | 96.0 | **69.7** | 96.7 | 39.6 | 2.5 / 16.3 / 33.7 / 52.8 / 62.7 / 69.7 | 16.7 | 4,107 |
| B2 | 116089 | 68.7 | 64.0 | 94.7 | 93.0 | **66.3** | 93.8 | 39.6 | 4.5 / 19.8 / 39.2 / 50.3 / 57.7 / 66.3 | 16.2 | 4,109 |
| B2-all | 73129 | 0.7 | 0.0 | 24.0 | 22.3 | **0.3** | 23.2 | 1.2 | 3.0 / 0.7 / 1.0 / 1.3 / 1.0 / 0.3 | 0.5 | 4,147 |
| B2-all | 93683 | 0.0 | 0.7 | 27.3 | 14.3 | **0.3** | 20.8 | 1.3 | 4.0 / 1.0 / 1.2 / 0.5 / 1.0 / 0.3 | 0.0 | 4,133 |
| B2-all | 116089 | 0.7 | 0.7 | 21.3 | 23.0 | **0.7** | 22.2 | 2.6 | 7.0 / 3.8 / 1.3 / 1.5 / 1.2 / 0.7 | 0.5 | 4,175 |
| B3 | 73129 | 1.0 | 1.3 | 1.3 | 0.3 | **1.2** | 0.8 | 1.1 | 1.3 / 1.3 / 1.2 / 1.2 / 0.7 / 1.2 | 1.0 | 4,954 |
| B3 | 93683 | 0.7 | 1.0 | 2.3 | 1.7 | **0.8** | 2.0 | 1.3 | 0.8 / 1.8 / 0.7 / 1.7 / 1.8 / 0.8 | 0.8 | 4,943 |
| B3 | 116089 | 0.7 | 1.0 | 1.7 | 4.0 | **0.8** | 2.8 | 0.9 | 1.8 / 0.0 / 0.7 / 1.0 / 1.0 / 0.8 | 0.7 | 4,948 |
| B3-all | 73129 | 1.7 | 0.3 | 1.0 | 1.7 | **1.0** | 1.3 | 1.1 | 1.0 / 0.8 / 0.7 / 2.0 / 1.3 / 1.0 | 1.2 | 5,033 |
| B3-all | 93683 | 1.0 | 2.3 | 2.7 | 3.0 | **1.7** | 2.8 | 0.9 | 1.2 / 0.5 / 0.7 / 1.0 / 0.7 / 1.7 | 1.2 | 5,093 |
| B3-all | 116089 | 0.0 | 0.3 | 1.3 | 1.0 | **0.2** | 1.2 | 0.7 | 1.2 / 0.7 / 0.7 / 0.8 / 0.8 / 0.2 | 1.3 | 5,149 |

## Embedding movement by role at the final checkpoint

- B2 73129: {"operator": 0.7914, "digit": 0.7151, "separator": 0.0, "equals": 0.4004, "distractor letter": 0.7021, "task code": 0.1107}; census {"operator": {"present": 4, "with_gradient": 4}, "digit": {"present": 10, "with_gradient": 10}, "equals": {"present": 1, "with_gradient": 1}, "distractor letter": {"present": 26, "with_gradient": 26}, "task code": {"present": 4, "with_gradient": 4}, "BOS/EOS": {"present": 2, "with_gradient": 0}}
- B2 93683: {"operator": 0.5449, "digit": 0.7339, "separator": 0.0, "equals": 0.3196, "distractor letter": 0.6727, "task code": 0.0998}; census {"operator": {"present": 4, "with_gradient": 4}, "digit": {"present": 10, "with_gradient": 10}, "equals": {"present": 1, "with_gradient": 1}, "distractor letter": {"present": 26, "with_gradient": 26}, "task code": {"present": 4, "with_gradient": 4}, "BOS/EOS": {"present": 2, "with_gradient": 0}}
- B2 116089: {"operator": 0.8824, "digit": 0.6867, "separator": 0.0, "equals": 0.2023, "distractor letter": 0.5399, "task code": 0.1439}; census {"operator": {"present": 4, "with_gradient": 4}, "digit": {"present": 10, "with_gradient": 10}, "equals": {"present": 1, "with_gradient": 1}, "distractor letter": {"present": 26, "with_gradient": 26}, "task code": {"present": 4, "with_gradient": 4}, "BOS/EOS": {"present": 2, "with_gradient": 0}}
- B2-all 73129: {"operator": 0.6796, "digit": 0.6259, "separator": 0.0, "equals": 0.8421, "distractor letter": 0.6199, "task code": 0.0738}; census {"operator": {"present": 4, "with_gradient": 4}, "digit": {"present": 10, "with_gradient": 10}, "equals": {"present": 1, "with_gradient": 1}, "distractor letter": {"present": 26, "with_gradient": 26}, "task code": {"present": 4, "with_gradient": 4}, "BOS/EOS": {"present": 2, "with_gradient": 0}}
- B2-all 93683: {"operator": 0.4837, "digit": 0.4824, "separator": 0.0, "equals": 0.5352, "distractor letter": 0.6851, "task code": 0.0835}; census {"operator": {"present": 4, "with_gradient": 4}, "digit": {"present": 10, "with_gradient": 10}, "equals": {"present": 1, "with_gradient": 1}, "distractor letter": {"present": 26, "with_gradient": 26}, "task code": {"present": 4, "with_gradient": 4}, "BOS/EOS": {"present": 2, "with_gradient": 0}}
- B2-all 116089: {"operator": 0.6564, "digit": 0.5931, "separator": 0.0, "equals": 0.6538, "distractor letter": 0.6359, "task code": 0.0973}; census {"operator": {"present": 4, "with_gradient": 4}, "digit": {"present": 10, "with_gradient": 10}, "equals": {"present": 1, "with_gradient": 1}, "distractor letter": {"present": 26, "with_gradient": 26}, "task code": {"present": 4, "with_gradient": 4}, "BOS/EOS": {"present": 2, "with_gradient": 0}}
- B3 73129: {"operator": 0.0, "digit": 0.7005, "separator": 0.0, "equals": 0.7892, "distractor letter": 0.0, "task code": 0.0}; census {"operator": {"present": 4, "with_gradient": 0}, "digit": {"present": 10, "with_gradient": 10}, "equals": {"present": 1, "with_gradient": 1}, "distractor letter": {"present": 26, "with_gradient": 0}, "task code": {"present": 4, "with_gradient": 0}, "BOS/EOS": {"present": 2, "with_gradient": 0}}
- B3 93683: {"operator": 0.0, "digit": 0.6797, "separator": 0.0, "equals": 0.7255, "distractor letter": 0.0, "task code": 0.0}; census {"operator": {"present": 4, "with_gradient": 0}, "digit": {"present": 10, "with_gradient": 10}, "equals": {"present": 1, "with_gradient": 1}, "distractor letter": {"present": 26, "with_gradient": 0}, "task code": {"present": 4, "with_gradient": 0}, "BOS/EOS": {"present": 2, "with_gradient": 0}}
- B3 116089: {"operator": 0.0, "digit": 0.6886, "separator": 0.0, "equals": 0.7619, "distractor letter": 0.0, "task code": 0.0}; census {"operator": {"present": 4, "with_gradient": 0}, "digit": {"present": 10, "with_gradient": 10}, "equals": {"present": 1, "with_gradient": 1}, "distractor letter": {"present": 26, "with_gradient": 0}, "task code": {"present": 4, "with_gradient": 0}, "BOS/EOS": {"present": 2, "with_gradient": 0}}
- B3-all 73129: {"operator": 0.5265, "digit": 0.7675, "separator": 0.0, "equals": 1.018, "distractor letter": 0.59, "task code": 0.119}; census {"operator": {"present": 4, "with_gradient": 4}, "digit": {"present": 10, "with_gradient": 10}, "equals": {"present": 1, "with_gradient": 1}, "distractor letter": {"present": 26, "with_gradient": 26}, "task code": {"present": 4, "with_gradient": 4}, "BOS/EOS": {"present": 2, "with_gradient": 0}}
- B3-all 93683: {"operator": 0.5025, "digit": 0.6062, "separator": 0.0, "equals": 0.48, "distractor letter": 0.6085, "task code": 0.0971}; census {"operator": {"present": 4, "with_gradient": 4}, "digit": {"present": 10, "with_gradient": 10}, "equals": {"present": 1, "with_gradient": 1}, "distractor letter": {"present": 26, "with_gradient": 26}, "task code": {"present": 4, "with_gradient": 4}, "BOS/EOS": {"present": 2, "with_gradient": 0}}
- B3-all 116089: {"operator": 0.7092, "digit": 0.8289, "separator": 0.0, "equals": 0.7108, "distractor letter": 0.5501, "task code": 0.114}; census {"operator": {"present": 4, "with_gradient": 4}, "digit": {"present": 10, "with_gradient": 10}, "equals": {"present": 1, "with_gradient": 1}, "distractor letter": {"present": 26, "with_gradient": 26}, "task code": {"present": 4, "with_gradient": 4}, "BOS/EOS": {"present": 2, "with_gradient": 0}}

## Verdicts

- P18b.1 B3-all operator and letter embeddings moved >= 0.2 row norm, all seeds: **holds**
- P18b.2 B3-all max/min >= 66 and hard >= 10, >= 2 seeds: **misses**
- P18b.3 B3-all hard >= 20 below B2-all, all seeds: **misses**
- P18b.4 |B2-all - B2| hard <= 15, >= 2 seeds: **misses**

## Validation problems

- none
