# E14S decay controls and nine-seed composition confirmation

Completed all **13 scientific jobs**, with observed exit code zero and exact command/environment/receipt validation for each. No tuning or seed substitution. The default-off checkpoint knob passes the short old/default/logged trajectory check and the full seed-400009 logged/unlogged receipt comparison.

## Main findings

Increasing decay to 1.0 hurts the BP arm too: final old-operation retention is 85.40%, 83.44%, 81.15%, versus 94.17%, 95.75%, 93.94% at 0.3. The literal <=85% prediction misses by 0.40 points in one seed; none reaches the alternate 92% threshold. This removes the basis for treating the damage as exclusive to the local rule, but does not establish equal decay sensitivity or identify a unique mechanism. BP still outperforms the local arm at matched decay 1.0, and their replay/teaching procedures differ.

The composition gate passes on the enlarged cohort: **8/9 local 0.25 seeds have held-pair means>=75%**. New means are 80.92%, 80.92%, 83.88%; every new seed has every held pair above 50%. Seed 400009 remains the one low mean, 72.04%. These are nine seeds of one source implementation, with new births and seed-derived splits for the six fresh seeds, not independent-code replication.

The failing seed shows no sampled high-accuracy acquisition followed by collapse of `+-`: all eight recorded points are <=18.67%. However, `-+` is only 1.17% at 2048 and first exceeds 60% at 6144, refuting the predicted early acquisition. Both curves rise late in the fixed dose. “Never learned” is too absolute: 1024-update sampling cannot exclude intervening transients or determine what another budget would do. The low final intermediate accuracy 19.83% also prevents localizing the failure solely to the second step.

## A Matched-decay BP control

All retention means cover the eight life operations. Composition means cover four held pair types; full scratchpad exactness is separate. Existing controls are reused and not billed as new training.

|Seed|Arm and provenance|After stage8 %|After composition %|Held final %|Held full scratchpad %|
|---|---|---|---|---|---|
|73129|BP wd1 NEW|93.21|85.40|83.08|80.04|
|73129|BP wd0.3 REUSED|95.35|94.17|89.29|87.75|
|73129|Local wd1 REUSED|90.12|79.79|54.92|48.83|
|73129|Local wd0.25 REUSED|94.58|93.75|82.29|79.54|
|93683|BP wd1 NEW|92.88|83.44|82.38|79.12|
|93683|BP wd0.3 REUSED|96.27|95.75|91.25|90.33|
|93683|Local wd1 REUSED|88.83|76.17|51.67|45.88|
|93683|Local wd0.25 REUSED|95.27|94.06|81.00|79.38|
|116089|BP wd1 NEW|92.90|81.15|79.04|75.25|
|116089|BP wd0.3 REUSED|96.21|93.94|90.75|88.62|
|116089|Local wd1 REUSED|88.73|77.60|48.29|42.67|
|116089|Local wd0.25 REUSED|95.52|94.58|85.00|83.29|

## B All nine local seeds and new BP controls

|Seed|Provenance|Final retention %|Held final mean %|Held full mean %|Mean>=75|
|---|---|---|---|---|---|
|73129|E13-A1 reused|93.75|82.29|79.54|True|
|93683|E13-A1 reused|94.06|81.00|79.38|True|
|116089|E13-A1 reused|94.58|85.00|83.29|True|
|200003|E14-A reused|94.50|84.58|82.79|True|
|300007|E14-A reused|93.46|76.67|74.58|True|
|400009|E14-A reused|94.54|72.04|70.17|False|
|500011|E14S new|92.96|80.92|79.17|True|
|600013|E14S new|94.02|80.92|77.79|True|
|700027|E14S new|92.90|83.88|81.29|True|

### Local 0.25 all nine seeds

Each pair cell is **final-answer percent / full-scratchpad percent**, including own EOS in the final-answer score. `within` is separate and excluded from the held-pair mean.

|Seed|+-|-+|><|<>|within|
|---|---|---|---|---|---|
|73129|63.17 / 60.00|76.00 / 74.33|96.00 / 92.67|94.00 / 91.17|86.67 / 84.50|
|93683|67.17 / 66.00|73.00 / 71.50|90.50 / 88.67|93.33 / 91.33|88.33 / 87.33|
|116089|71.67 / 71.00|80.33 / 79.33|92.17 / 89.83|95.83 / 93.00|88.33 / 87.17|
|200003|67.83 / 66.00|79.50 / 77.50|94.33 / 92.33|96.67 / 95.33|87.50 / 86.50|
|300007|53.17 / 52.50|72.17 / 69.17|93.17 / 91.17|88.17 / 85.50|88.67 / 87.33|
|400009|18.67 / 17.33|79.83 / 78.33|94.50 / 92.33|95.17 / 92.67|90.17 / 88.83|
|500011|71.33 / 70.50|67.33 / 65.83|93.33 / 91.17|91.67 / 89.17|86.00 / 84.67|
|600013|70.67 / 69.33|69.33 / 66.83|94.50 / 91.00|89.17 / 84.00|86.83 / 86.00|
|700027|73.33 / 71.83|74.00 / 72.50|92.83 / 89.67|95.33 / 91.17|86.00 / 84.00|

### New BP 0.3 references

Each pair cell is **final-answer percent / full-scratchpad percent**, including own EOS in the final-answer score. `within` is separate and excluded from the held-pair mean.

|Seed|+-|-+|><|<>|within|
|---|---|---|---|---|---|
|500011|84.83 / 83.83|84.00 / 83.00|96.33 / 94.50|92.50 / 91.33|89.67 / 89.17|
|600013|82.17 / 80.17|78.83 / 76.50|96.67 / 95.17|95.83 / 94.67|90.33 / 89.17|
|700027|81.83 / 80.67|84.00 / 83.67|96.17 / 93.50|97.83 / 94.83|90.33 / 89.67|

## C Composition-stage trajectories

Eight additional read-only scoring points at updates 1024 through 8192. All five panels are the original panels. Full underlying intermediate/final/full counts and all four logged lives are in E14S_COMPOSITION_CURVES.csv and raw receipts. A correct suffix is not automatically a correct complete scratchpad.

### C_400009

|Update|+-|-+|><|<>|within|
|---|---|---|---|---|---|
|1024|3.33|0.33|67.00|63.50|41.50|
|2048|2.83|1.17|31.17|82.17|63.00|
|3072|5.00|35.33|29.17|90.83|78.17|
|4096|1.17|48.83|53.50|90.83|83.00|
|5120|3.83|55.83|91.33|93.17|84.00|
|6144|9.17|63.00|90.33|94.00|86.00|
|7168|16.33|70.67|94.33|95.50|88.33|
|8192|18.67|79.83|94.50|95.17|90.17|

### B_C_500011

|Update|+-|-+|><|<>|within|
|---|---|---|---|---|---|
|1024|3.50|3.00|62.67|64.33|35.50|
|2048|16.33|13.67|80.50|76.33|53.50|
|3072|39.33|31.83|81.17|82.50|64.83|
|4096|56.67|48.67|86.83|89.33|75.67|
|5120|63.33|60.17|90.50|89.67|79.83|
|6144|69.17|65.17|91.67|90.17|81.50|
|7168|71.50|70.67|93.17|89.17|85.17|
|8192|71.33|67.33|93.33|91.67|86.00|

### B_C_600013

|Update|+-|-+|><|<>|within|
|---|---|---|---|---|---|
|1024|6.17|1.83|69.33|63.50|39.67|
|2048|27.50|1.50|82.50|77.50|55.50|
|3072|44.17|8.83|87.17|83.67|70.83|
|4096|56.00|55.50|88.50|88.67|76.17|
|5120|58.17|61.83|92.00|88.33|82.00|
|6144|64.83|66.17|93.33|89.00|84.17|
|7168|64.33|70.17|93.33|80.67|84.67|
|8192|70.67|69.33|94.50|89.17|86.83|

### B_C_700027

|Update|+-|-+|><|<>|within|
|---|---|---|---|---|---|
|1024|6.83|5.33|24.33|64.83|40.00|
|2048|35.17|25.67|66.67|82.00|58.17|
|3072|56.50|49.67|89.67|88.00|74.17|
|4096|62.00|65.00|93.00|92.00|80.33|
|5120|64.67|73.50|94.00|89.67|86.00|
|6144|68.17|72.83|93.00|92.67|81.67|
|7168|68.33|74.83|93.50|92.50|84.33|
|8192|73.33|74.00|92.83|95.33|86.00|

### Seed 400009 step-level scores

|Update|+- intermediate %|+- final %|+- full %|-+ intermediate %|-+ final %|-+ full %|
|---|---|---|---|---|---|---|
|1024|4.50|3.33|2.33|0.17|0.33|0.17|
|2048|2.00|2.83|1.17|0.67|1.17|0.50|
|3072|5.17|5.00|3.83|41.33|35.33|34.50|
|4096|0.00|1.17|0.00|54.83|48.83|47.67|
|5120|2.83|3.83|2.67|62.83|55.83|55.00|
|6144|8.83|9.17|8.17|70.83|63.00|62.33|
|7168|18.33|16.33|15.17|80.17|70.67|69.83|
|8192|19.83|18.67|17.33|84.67|79.83|78.33|

## Prospective prediction verdicts

|Prediction|Result|
|---|---|
|A BP final retention<=85 in every seed|MISS: 73129 is 85.3958%; other two pass|
|B at most one new seed has any pair<50|PASS: zero|
|B at least 7/9 local means>=75|PASS: 8/9|
|C +- never above 40 at recorded checkpoints|PASS: maximum 18.67|
|C -+ above 60 by 2048|MISS: first recorded crossing 6144|

## Per-operation scores

### e14s_A_73129_e8_life_D_73129.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|90.67|90.00|
|sub97|600|90.33|85.00|
|max97|600|96.67|95.33|
|min97|600|96.50|94.50|
|add73|600|88.17|64.17|
|sub73|600|89.83|66.33|
|max73|600|96.67|95.17|
|min73|600|96.83|92.67|
|add89|300|32.33|17.67|
|sub89|300|7.00|30.67|
|max89|300|1.33|15.67|
|min89|300|20.00|25.00|

### e8_life_D_73129.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|92.33|93.00|
|sub97|600|91.00|91.83|
|max97|600|98.17|97.83|
|min97|600|99.33|98.00|
|add73|600|90.67|88.33|
|sub73|600|92.50|88.00|
|max73|600|99.50|98.00|
|min73|600|99.33|98.33|
|add89|300|0.33|13.00|
|sub89|300|15.00|46.67|
|max89|300|96.00|46.33|
|min89|300|4.00|20.33|

### e8_life_C_73129.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|86.50|83.33|
|sub97|600|84.67|68.00|
|max97|600|96.67|89.33|
|min97|600|96.33|90.33|
|add73|600|80.00|64.83|
|sub73|600|83.17|59.17|
|max73|600|95.83|90.50|
|min73|600|97.83|92.83|
|add89|300|14.00|20.67|
|sub89|300|0.33|38.33|
|max89|300|1.67|24.33|
|min89|300|1.00|66.67|

### e13_A1_e8_life_C_73129.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|93.67|92.50|
|sub97|600|90.00|90.17|
|max97|600|98.83|96.83|
|min97|600|98.83|98.33|
|add73|600|88.83|86.17|
|sub73|600|88.17|89.50|
|max73|600|99.17|98.33|
|min73|600|99.17|98.17|
|add89|300|22.00|38.33|
|sub89|300|0.00|47.67|
|max89|300|96.00|6.67|
|min89|300|7.67|3.33|

### e14s_A_93683_e8_life_D_93683.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|93.17|91.67|
|sub97|600|92.17|83.50|
|max97|600|96.83|96.50|
|min97|600|96.33|93.83|
|add73|600|87.50|61.33|
|sub73|600|87.17|55.33|
|max73|600|95.00|94.00|
|min73|600|94.83|91.33|
|add89|300|7.00|5.67|
|sub89|300|8.67|32.67|
|max89|300|11.33|34.00|
|min89|300|11.00|27.00|

### e8_life_D_93683.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|94.50|95.17|
|sub97|600|95.33|94.33|
|max97|600|99.33|97.83|
|min97|600|99.33|99.33|
|add73|600|90.50|91.00|
|sub73|600|92.67|92.67|
|max73|600|99.33|97.50|
|min73|600|99.17|98.17|
|add89|300|0.00|24.00|
|sub89|300|41.00|47.33|
|max89|300|32.00|40.00|
|min89|300|0.00|8.33|

### e8_life_C_93683.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|86.00|81.00|
|sub97|600|83.33|64.83|
|max97|600|96.33|93.17|
|min97|600|97.67|92.50|
|add73|600|77.67|49.00|
|sub73|600|75.00|48.00|
|max73|600|97.17|89.50|
|min73|600|97.50|91.33|
|add89|300|0.33|9.00|
|sub89|300|17.00|10.33|
|max89|300|0.67|48.33|
|min89|300|3.67|24.67|

### e13_A1_e8_life_C_93683.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|93.00|94.17|
|sub97|600|91.83|93.00|
|max97|600|98.83|96.17|
|min97|600|97.83|98.17|
|add73|600|91.50|89.33|
|sub73|600|90.67|86.33|
|max73|600|99.17|97.67|
|min73|600|99.33|97.67|
|add89|300|15.33|19.33|
|sub89|300|25.67|44.67|
|max89|300|13.33|5.00|
|min89|300|14.00|24.00|

### e14s_A_116089_e8_life_D_116089.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|91.67|90.83|
|sub97|600|93.50|77.67|
|max97|600|97.33|92.33|
|min97|600|94.67|93.83|
|add73|600|86.50|54.50|
|sub73|600|87.83|56.00|
|max73|600|96.67|92.83|
|min73|600|95.00|91.17|
|add89|300|0.00|13.33|
|sub89|300|0.00|31.33|
|max89|300|0.00|25.33|
|min89|300|0.00|18.33|

### e8_life_D_116089.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|94.50|94.17|
|sub97|600|93.33|93.67|
|max97|600|99.83|95.50|
|min97|600|99.33|99.00|
|add73|600|93.17|87.83|
|sub73|600|90.83|88.33|
|max73|600|99.50|95.67|
|min73|600|99.17|97.33|
|add89|300|0.00|31.67|
|sub89|300|0.00|46.00|
|max89|300|0.33|8.33|
|min89|300|0.67|15.00|

### e8_life_C_116089.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|87.00|83.33|
|sub97|600|85.50|60.17|
|max97|600|94.00|91.83|
|min97|600|97.00|93.67|
|add73|600|76.33|50.17|
|sub73|600|80.67|58.50|
|max73|600|94.17|91.67|
|min73|600|95.17|91.50|
|add89|300|0.00|11.33|
|sub89|300|4.67|17.67|
|max89|300|0.33|43.00|
|min89|300|0.67|32.00|

### e13_A1_e8_life_C_116089.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|91.83|93.17|
|sub97|600|95.00|93.50|
|max97|600|99.17|98.83|
|min97|600|98.50|97.67|
|add73|600|90.67|89.33|
|sub73|600|91.50|90.17|
|max73|600|98.67|96.83|
|min73|600|98.83|97.17|
|add89|300|0.00|32.33|
|sub89|300|0.00|41.33|
|max89|300|0.00|16.00|
|min89|300|0.00|14.33|

### e14_A_C_e8_life_C_200003.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|91.67|92.83|
|sub97|600|88.33|91.17|
|max97|600|99.00|98.33|
|min97|600|99.33|98.67|
|add73|600|87.50|89.50|
|sub73|600|86.50|89.00|
|max73|600|98.83|97.67|
|min73|600|99.83|98.83|
|add89|300|8.00|40.00|
|sub89|300|0.00|44.67|
|max89|300|6.67|3.67|
|min89|300|12.33|9.00|

### e14_A_C_e8_life_C_300007.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|89.50|94.17|
|sub97|600|91.67|93.67|
|max97|600|98.17|98.00|
|min97|600|99.17|97.50|
|add73|600|89.50|82.67|
|sub73|600|86.67|87.83|
|max73|600|99.17|97.67|
|min73|600|99.33|96.17|
|add89|300|35.00|18.67|
|sub89|300|3.00|43.33|
|max89|300|77.00|48.33|
|min89|300|95.33|22.33|

### e14_A_C_e8_life_C_400009.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|93.33|95.17|
|sub97|600|92.17|93.17|
|max97|600|97.33|97.50|
|min97|600|99.50|97.83|
|add73|600|90.67|90.83|
|sub73|600|90.00|88.17|
|max73|600|98.17|97.00|
|min73|600|98.83|96.67|
|add89|300|29.33|34.00|
|sub89|300|0.67|43.67|
|max89|300|29.00|5.00|
|min89|300|29.67|9.67|

### e14s_B_C_500011_e8_life_C_500011.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|91.33|90.50|
|sub97|600|88.33|90.33|
|max97|600|97.17|97.00|
|min97|600|99.17|97.00|
|add73|600|89.67|87.83|
|sub73|600|89.50|88.33|
|max73|600|98.17|97.00|
|min73|600|99.50|95.67|
|add89|300|0.00|18.00|
|sub89|300|2.33|30.67|
|max89|300|0.33|4.00|
|min89|300|60.33|32.33|

### e14s_B_C_600013_e8_life_C_600013.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|91.83|90.67|
|sub97|600|88.67|91.67|
|max97|600|99.00|98.00|
|min97|600|98.33|97.83|
|add73|600|90.17|88.67|
|sub73|600|91.00|90.83|
|max73|600|98.17|96.83|
|min73|600|99.00|97.67|
|add89|300|30.00|40.00|
|sub89|300|1.33|31.33|
|max89|300|0.00|53.67|
|min89|300|0.00|6.00|

### e14s_B_C_700027_e8_life_C_700027.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|89.83|91.50|
|sub97|600|87.33|90.00|
|max97|600|96.67|97.33|
|min97|600|98.67|97.83|
|add73|600|87.33|88.33|
|sub73|600|85.33|85.33|
|max73|600|98.17|95.83|
|min73|600|99.00|97.00|
|add89|300|6.33|10.33|
|sub89|300|31.33|25.33|
|max89|300|75.67|16.67|
|min89|300|14.33|30.33|

### e14s_B_D_500011_e8_life_D_500011.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|93.50|94.00|
|sub97|600|95.17|92.50|
|max97|600|99.33|96.67|
|min97|600|97.83|97.67|
|add73|600|92.00|89.00|
|sub73|600|92.50|88.67|
|max73|600|99.50|96.67|
|min73|600|97.50|97.83|
|add89|300|2.67|12.00|
|sub89|300|3.67|35.33|
|max89|300|71.67|19.67|
|min89|300|2.67|22.67|

### e14s_B_D_600013_e8_life_D_600013.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|95.67|94.17|
|sub97|600|92.50|93.50|
|max97|600|99.67|98.17|
|min97|600|97.67|97.67|
|add73|600|92.17|88.67|
|sub73|600|93.17|89.33|
|max73|600|99.50|98.00|
|min73|600|96.00|96.67|
|add89|300|32.67|39.33|
|sub89|300|23.33|39.67|
|max89|300|0.00|6.00|
|min89|300|10.00|6.33|

### e14s_B_D_700027_e8_life_D_700027.json

|Operation|n|Stage8 exact %|Final exact %|
|---|---|---|---|
|add97|600|94.83|92.83|
|sub97|600|92.00|91.67|
|max97|600|99.50|99.17|
|min97|600|99.33|97.33|
|add73|600|90.83|87.17|
|sub73|600|87.83|84.33|
|max73|600|98.83|98.00|
|min73|600|99.67|95.67|
|add89|300|14.00|27.67|
|sub89|300|26.67|34.33|
|max89|300|97.67|10.00|
|min89|300|17.33|36.33|

## Costs and provenance

|Quantity|Value|
|---|---|
|scientific_subprocesses|13|
|births|3|
|lives|10|
|recorded_scientific_cpu_seconds_lower_bound|13810.850309|
|scientific_subprocess_wall_seconds_sum|15352.180634623859|
|campaign_wall_seconds|3245.9030310423113|
|technical_trajectories|3|
|technical_optimizer_steps_per_trajectory|20|
|technical_receipt_cpu_seconds_lower_bound|5.1646990000000015|
|total_technical_cpu|NOT_MEASURED|
|peak_RSS|NOT_MEASURED|
|new_birth_checkpoint_bytes|5659677|
|timing_limit|Receipt CPU excludes startup, serialization and final evaluations; it is a lower bound. Summed overlapping subprocess wall times are not campaign elapsed time. Technical receipt CPU is also a lower bound; initial restricted-shell attempt launched no scientific job.|

All 13 jobs used the previously qualified inference wrapper, with original forward/backward/update settings unchanged. The full logged C400009 life reproduces E14 exactly after removing only `cpu` and the added `compose_checkpoints` key. The other three logged lives are new seeds and have no unlogged full-life comparator. Logging affects elapsed CPU, not training data or the scoring panels. No life-weight checkpoints were requested; the three new birth checkpoints and curves are retained.

Seventeen prospective source/input hashes match. New birth hashes, exact per-job checks and reused-control checksums are in E14S_VERIFICATION.json; source-defined new-seed data fingerprints are in E14S_INPUT_AUDIT.json. The short technical fixture verifies model/optimizer/Torch-RNG/store/loss/score trajectory equality. Python replay Random state was not separately serialized; its noninterference evidence is source inspection plus exact trajectories and the full 400009 receipt. Peak RSS and exact total process-tree CPU were not instrumented. No external independent replication or formal checker is claimed. E16 was untouched.

