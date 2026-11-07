# E12 replay mechanics — final results

All 49 planned jobs completed: 21 new full lives, 1 default reproduction, 27 independent reminders. Three seeds in every new condition. These are development results on the existing E8 panels, not sealed independent confirmation. All original locked scientific source/settings remained unchanged through an orchestration-only repair.

The clearest positive is batch 256 at fixed example exposure: final old-operation mean 91.50–92.42%, held composition 68.62–73.08%, versus 76.17–79.79% and 48.29–54.92% at batch 64. Every seed improves both. Batch 16 instead ends near 26.5–27.1% retention and 12.2–13.3% composition. This refutes the prediction that smaller batches would help and larger batches would harm.

The intervention changes optimizer steps as well as batch size. At the unchanged learning rate and weight decay 1.0, batch 16/64/256 take 229,376 / 57,344 / 14,336 total updates. AdamW's pure multiplicative decay factor over those updates is approximately exp(-229.5), exp(-57.4), exp(-14.3), respectively (this factor is not the actual trained weight magnitude). The results cannot isolate gradient averaging from cumulative decay or optimizer-state effects. Batch256 is a promising three-seed package, not a diagnosed new learning mechanism.

A: the predicted large savings is absent in all seeds. At reminder 256, the erased learner reaches 61.92/63.29/63.42%, but the developed birth reaches 71.69/70.38/73.52%. The birth already carries general training on mod 89; it is not a random untrained model. Earlier experience with these mod 97/mod 73 tasks confers no net mean recovery advantage at this reminder dose. At 1,024 updates one seed has +2.58 points savings; the others remain slightly below birth. Neither the proposed positive-savings criterion nor the literal within 10 points-at-all-doses erasure criterion passes. Do not claim all internal information was erased or uniquely blame the output layer.

C: every seed and both replay shares lose more than 10 points on the earliest four operations before composition, while gains on the latest four remain below 5. Recency weighting hurts old skills in this deliberately nonstationary stream.

D: buffer 128 loses 14.77–21.19points at stage8 relative to 1,024. Buffer 8,192 remains within 3 points at stage8 and at the final old-operation mean in every seed. Thus retention saturates around 1,024 prompts per task in this experiment. This is buffer coverage, not a demonstration of Raviv's 25–30-exposure saturation. Composition can still change; report its table rather than treating all outcomes as saturated.

## Frozen prediction summary

| Prediction | Seed 73129 | Seed 93683 | Seed 116089 |
|---|---|---|---|
| A: large recovery advantage at reminder 256 | Refuted | Refuted | Refuted |
| B: small batch helps, large batch harms carry operations | Refuted | Refuted | Refuted |
| C: recency loses >10 points early, gains <5 late, both shares | Supported | Supported | Supported |
| D: buffer 128 loses >10 points; 8,192 within 3 of 1,024 | Supported | Supported | Supported |

## Execution correction and provenance

The first controller wrongly validated unfinished sibling receipts using a broad glob after a different job finished. It stopped six running jobs in parts B/C/D and skipped others. This was a controller failure, not a scientific failure. The stopped attempt is preserved under exp45/e12_interrupted_attempt_20261107 with hashes in E12_RESUME_LOCK.json. The repair validates only the exact job receipt.34 completed jobs were retained; 15 unfinished lives restarted from the identical birth, seeds and scientific settings. Original E12_LOCK.json and all its source/checkpoint hashes still match. No outcome-dependent tuning or extra candidate conditions occurred.

Reused controls: batch 64/uniform 1,024 at share 1/2 and uniform 1,024 at share 1/4 from E8/E9. New conditions: batch 16/256, recency shares 1/2 and 1/4, capacity 128/8,192, saved interleaved no-replay reruns and reminders. A full C seed 73129 baseline and all three interleaved A reruns match archived scientific receipts exactly after removing cpu fields only. Nine stage checkpoints exist; model/optimizer/RNG are stored, while reminders intentionally fork weights with fresh matched optimizers. E12_VERIFICATION.json records the recount and receipt hashes. No formal checker or independent replication is claimed.


Campaign status: **complete**.

Source/authorization/hashes: E12_LOCK.json. Predictions were written before gradients. Existing E8/E9 controls are reused, not new replications. Own-byte/EOS endpoints unchanged.

## A: reminder savings

| Seed | Updates | Erased | Birth | Pre-composition | Savings |
|---|---:|---:|---:|---:|---:|
|73129|64|40.69|59.77|91.71|-19.08|
|73129|256|61.92|71.69|91.29|-9.77|
|73129|1024|81.46|84.85|91.60|-3.40|
|93683|64|43.46|56.77|92.67|-13.31|
|93683|256|63.29|70.38|92.10|-7.08|
|93683|1024|84.69|82.10|91.73|+2.58|
|116089|64|41.56|53.08|91.06|-11.52|
|116089|256|63.42|73.52|91.23|-10.10|
|116089|1024|85.50|86.62|92.02|-1.12|

## B/C/D: full lives

|Part|Setting|Seed|Stage8 mean|Final mean|Held composition|CPU seconds|Origin|
|---|---|---:|---:|---:|---:|---:|---|
|B|batch 16|73129|48.62|26.98|13.25|3224.77|new|
|B|batch 64|73129|90.13|79.79|54.92|1318.37|reused|
|B|batch 256|73129|94.52|92.35|72.25|1958.30|new|
|C|recency half|73129|75.44|61.85|47.62|2118.55|new|
|C|recency quarter|73129|68.00|54.90|62.88|1764.32|new|
|C|uniform quarter|73129|80.96|70.08|61.71|851.81|reused|
|D|cap128|73129|68.94|72.10|56.58|2135.63|new|
|D|cap1024|73129|90.13|79.79|54.92|1318.37|reused|
|D|cap8192|73129|90.38|80.58|55.96|2111.72|new|
|B|batch 16|93683|57.60|27.12|12.21|3093.38|new|
|B|batch 64|93683|88.83|76.17|51.67|1318.31|reused|
|B|batch 256|93683|94.23|91.50|68.62|1979.08|new|
|C|recency half|93683|72.98|61.04|46.38|2100.16|new|
|C|recency quarter|93683|63.71|52.92|55.12|1702.69|new|
|C|uniform quarter|93683|78.06|67.08|59.38|849.68|reused|
|D|cap128|93683|69.69|71.25|54.88|2033.53|new|
|D|cap1024|93683|88.83|76.17|51.67|1318.31|reused|
|D|cap8192|93683|87.52|76.67|48.79|2043.60|new|
|B|batch 16|116089|53.54|26.48|12.21|3004.85|new|
|B|batch 64|116089|88.73|77.60|48.29|1320.68|reused|
|B|batch 256|116089|94.98|92.42|73.08|1895.42|new|
|C|recency half|116089|75.67|59.79|40.54|2004.79|new|
|C|recency quarter|116089|65.50|46.40|41.21|1764.47|new|
|C|uniform quarter|116089|79.50|61.90|47.12|763.36|reused|
|D|cap128|116089|73.96|71.98|55.04|2253.02|new|
|D|cap1024|116089|88.73|77.60|48.29|1320.68|reused|
|D|cap8192|116089|89.56|77.15|56.29|2206.52|new|

## Prediction verdicts

A seed73129: recovery>=60 and birth<=20: False.
A seed73129: within10 at all doses: False (not a proof of complete information loss).
A seed93683: recovery>=60 and birth<=20: False.
A seed93683: within10 at all doses: False (not a proof of complete information loss).
A seed116089: recovery>=60 and birth<=20: False.
A seed116089: within10 at all doses: False (not a proof of complete information loss).
B seed73129: retention within3: False; composition delta -41.67.
B seed73129: carry loss>10: False.
C recency half seed73129: early delta-25.54, late delta-3.83; predicted True; all operation deltas within5: False.
C recency quarter seed73129: early delta-22.83, late delta-3.08; predicted True; all operation deltas within5: False.
D cap128 seed73129: stage8 delta -21.19, final delta -7.69.
D cap1024 seed73129: stage8 delta +0.00, final delta +0.00.
D cap8192 seed73129: stage8 delta +0.25, final delta +0.79.
B seed93683: retention within3: False; composition delta -39.46.
B seed93683: carry loss>10: False.
C recency half seed93683: early delta-27.58, late delta-4.13; predicted True; all operation deltas within5: False.
C recency quarter seed93683: early delta-22.46, late delta-6.25; predicted True; all operation deltas within5: False.
D cap128 seed93683: stage8 delta -19.15, final delta -4.92.
D cap1024 seed93683: stage8 delta +0.00, final delta +0.00.
D cap8192 seed93683: stage8 delta -1.31, final delta +0.50.
B seed116089: retention within3: False; composition delta -36.08.
B seed116089: carry loss>10: False.
C recency half seed116089: early delta-25.71, late delta-0.42; predicted True; all operation deltas within5: False.
C recency quarter seed116089: early delta-22.79, late delta-5.21; predicted True; all operation deltas within5: False.
D cap128 seed116089: stage8 delta -14.77, final delta -5.62.
D cap1024 seed116089: stage8 delta +0.00, final delta +0.00.
D cap8192 seed116089: stage8 delta +0.83, final delta -0.46.

## Final compute accounting

Successful jobs recorded 44907.75 training/evaluation CPU seconds (12.47CPU hours), plus at least 4939CPU seconds from interrupted jobs' last printed checkpoints. The latter excludes work since the last printed stage, so the combined 49846.75CPU seconds is a lower bound. Existing clocks exclude some setup/initial scoring, as in E9.

Summed process wall time: successful 67149.46s; six interrupted jobs 8635.67s; total 75785.13s (21.05process hours). These are parallel-process sums, not elapsed time. Recorded execution elapsed is 15763.03s (4.38h), excluding downtime between the stopped controller and resumption. Technical-parity checks and final audit CPU are additional and not included in these totals. Peak RSS: NOT_MEASURED; the preparation's prospective RSS wording did not yield measurements. No new development checkpoints were trained; three E8 births reused. CPU contention and the controller restart made the initial forecast optimistic.

E11 was not altered. Path-integration follow-up remains NOT_RUN and is not fulfilled by E12.

## Limits

Savings does not localize residual information to a specific layer. Batch changes update count and cumulative weight decay. Recency tests a nonstationary task sequence. Buffer coverage differs from exposure repetition. No tuning or promotion is inferred. Path-integration follow-up remains NOT_RUN; E11 is untouched.

### Reminder seed73129, updates64
{'erased': {'add97': 13.67, 'sub97': 2.33, 'max97': 69.33, 'min97': 70.5, 'add73': 12.33, 'sub73': 3.33, 'max73': 77.33, 'min73': 76.67}, 'birth': {'add97': 32.83, 'sub97': 22.83, 'max97': 88.33, 'min97': 90.17, 'add73': 29.83, 'sub73': 29.83, 'max73': 89.83, 'min73': 94.5}, 'pre': {'add97': 87.83, 'sub97': 88.83, 'max97': 95.5, 'min97': 96.33, 'add73': 85.5, 'sub73': 86.83, 'max73': 96.0, 'min73': 96.83}}

### Reminder seed73129, updates256
{'erased': {'add97': 43.5, 'sub97': 27.5, 'max97': 90.83, 'min97': 90.0, 'add73': 34.17, 'sub73': 26.0, 'max73': 91.83, 'min73': 91.5}, 'birth': {'add97': 41.83, 'sub97': 54.0, 'max97': 97.0, 'min97': 97.83, 'add73': 39.83, 'sub73': 48.83, 'max73': 96.67, 'min73': 97.5}, 'pre': {'add97': 89.17, 'sub97': 88.0, 'max97': 96.83, 'min97': 96.83, 'add73': 81.5, 'sub73': 85.5, 'max73': 96.17, 'min73': 96.33}}

### Reminder seed73129, updates1024
{'erased': {'add97': 75.0, 'sub97': 67.0, 'max97': 93.67, 'min97': 94.67, 'add73': 64.5, 'sub73': 67.5, 'max73': 93.83, 'min73': 95.5}, 'birth': {'add97': 80.5, 'sub97': 75.0, 'max97': 95.17, 'min97': 96.33, 'add73': 70.83, 'sub73': 69.33, 'max73': 95.5, 'min73': 96.17}, 'pre': {'add97': 87.83, 'sub97': 88.67, 'max97': 96.5, 'min97': 95.83, 'add73': 84.0, 'sub73': 86.83, 'max73': 96.67, 'min73': 96.5}}

### Reminder seed93683, updates64
{'erased': {'add97': 17.67, 'sub97': 3.33, 'max97': 78.33, 'min97': 78.33, 'add73': 11.83, 'sub73': 5.0, 'max73': 77.83, 'min73': 75.33}, 'birth': {'add97': 27.67, 'sub97': 16.33, 'max97': 87.67, 'min97': 83.83, 'add73': 32.67, 'sub73': 18.5, 'max73': 95.5, 'min73': 92.0}, 'pre': {'add97': 93.5, 'sub97': 92.33, 'max97': 96.83, 'min97': 96.5, 'add73': 87.17, 'sub73': 83.5, 'max73': 94.67, 'min73': 96.83}}

### Reminder seed93683, updates256
{'erased': {'add97': 53.33, 'sub97': 23.5, 'max97': 94.67, 'min97': 91.33, 'add73': 34.67, 'sub73': 26.5, 'max73': 91.67, 'min73': 90.67}, 'birth': {'add97': 47.67, 'sub97': 42.0, 'max97': 94.67, 'min97': 94.17, 'add73': 46.67, 'sub73': 45.0, 'max73': 96.33, 'min73': 96.5}, 'pre': {'add97': 90.33, 'sub97': 91.0, 'max97': 98.17, 'min97': 96.17, 'add73': 84.33, 'sub73': 82.5, 'max73': 97.67, 'min73': 96.67}}

### Reminder seed93683, updates1024
{'erased': {'add97': 86.5, 'sub97': 80.17, 'max97': 95.83, 'min97': 95.33, 'add73': 66.0, 'sub73': 67.83, 'max73': 92.83, 'min73': 93.0}, 'birth': {'add97': 78.33, 'sub97': 70.67, 'max97': 94.67, 'min97': 94.33, 'add73': 71.67, 'sub73': 60.83, 'max73': 93.67, 'min73': 92.67}, 'pre': {'add97': 91.67, 'sub97': 91.33, 'max97': 96.83, 'min97': 96.0, 'add73': 84.0, 'sub73': 83.67, 'max73': 95.17, 'min73': 95.17}}

### Reminder seed116089, updates64
{'erased': {'add97': 19.17, 'sub97': 4.33, 'max97': 71.83, 'min97': 75.67, 'add73': 13.0, 'sub73': 7.0, 'max73': 68.67, 'min73': 72.83}, 'birth': {'add97': 31.0, 'sub97': 21.67, 'max97': 79.0, 'min97': 80.67, 'add73': 27.83, 'sub73': 22.5, 'max73': 78.17, 'min73': 83.83}, 'pre': {'add97': 90.0, 'sub97': 89.17, 'max97': 96.33, 'min97': 94.67, 'add73': 82.33, 'sub73': 84.33, 'max73': 96.17, 'min73': 95.5}}

### Reminder seed116089, updates256
{'erased': {'add97': 44.33, 'sub97': 30.17, 'max97': 91.17, 'min97': 94.5, 'add73': 29.17, 'sub73': 34.67, 'max73': 90.33, 'min73': 93.0}, 'birth': {'add97': 59.33, 'sub97': 50.83, 'max97': 95.67, 'min97': 97.67, 'add73': 44.17, 'sub73': 46.67, 'max73': 96.17, 'min73': 97.67}, 'pre': {'add97': 91.5, 'sub97': 90.33, 'max97': 94.33, 'min97': 96.67, 'add73': 83.5, 'sub73': 84.17, 'max73': 93.17, 'min73': 96.17}}

### Reminder seed116089, updates1024
{'erased': {'add97': 82.0, 'sub97': 79.67, 'max97': 96.0, 'min97': 96.33, 'add73': 74.33, 'sub73': 65.17, 'max73': 95.5, 'min73': 95.0}, 'birth': {'add97': 85.0, 'sub97': 82.5, 'max97': 95.17, 'min97': 96.33, 'add73': 75.5, 'sub73': 66.17, 'max73': 95.83, 'min73': 96.5}, 'pre': {'add97': 90.83, 'sub97': 91.33, 'max97': 97.17, 'min97': 96.67, 'add73': 82.17, 'sub73': 86.17, 'max73': 95.33, 'min73': 96.5}}

### B batch 16 seed73129
|Operation|Stage8|Final|
|---|---:|---:|
|add97|9.50|1.17|
|sub97|7.00|0.33|
|max97|85.83|52.50|
|min97|91.67|50.83|
|add73|7.00|0.50|
|sub73|10.67|0.67|
|max73|85.00|56.00|
|min73|92.33|53.83|
|add89|1.00|0.00|
|sub89|0.00|0.00|
|max89|8.67|9.00|
|min89|0.67|0.00|
{'+-': {'n': 600, 'exact': 0, 'final': 3, 'intermediate': 7}, '-+': {'n': 600, 'exact': 0, 'final': 5, 'intermediate': 6}, '><': {'n': 600, 'exact': 89, 'final': 155, 'intermediate': 214}, '<>': {'n': 600, 'exact': 80, 'final': 155, 'intermediate': 211}, 'within': {'n': 600, 'exact': 79, 'final': 107, 'intermediate': 116}}

### B batch 64 seed73129
|Operation|Stage8|Final|
|---|---:|---:|
|add97|86.50|83.33|
|sub97|84.67|68.00|
|max97|96.67|89.33|
|min97|96.33|90.33|
|add73|80.00|64.83|
|sub73|83.17|59.17|
|max73|95.83|90.50|
|min73|97.83|92.83|
|add89|14.00|20.67|
|sub89|0.33|38.33|
|max89|1.67|24.33|
|min89|1.00|66.67|
{'+-': {'n': 600, 'exact': 174, 'final': 189, 'intermediate': 300}, '-+': {'n': 600, 'exact': 202, 'final': 225, 'intermediate': 275}, '><': {'n': 600, 'exact': 374, 'final': 442, 'intermediate': 447}, '<>': {'n': 600, 'exact': 422, 'final': 462, 'intermediate': 507}, 'within': {'n': 600, 'exact': 335, 'final': 364, 'intermediate': 421}}

### B batch 256 seed73129
|Operation|Stage8|Final|
|---|---:|---:|
|add97|94.00|92.67|
|sub97|90.17|88.17|
|max97|99.00|97.17|
|min97|98.50|97.00|
|add73|86.17|83.17|
|sub73|90.17|86.67|
|max73|98.67|97.67|
|min73|99.50|96.33|
|add89|3.67|9.33|
|sub89|45.67|43.67|
|max89|12.00|39.33|
|min89|98.67|42.00|
{'+-': {'n': 600, 'exact': 293, 'final': 314, 'intermediate': 379}, '-+': {'n': 600, 'exact': 364, 'final': 379, 'intermediate': 441}, '><': {'n': 600, 'exact': 480, 'final': 517, 'intermediate': 528}, '<>': {'n': 600, 'exact': 495, 'final': 524, 'intermediate': 544}, 'within': {'n': 600, 'exact': 442, 'final': 455, 'intermediate': 498}}

### C recency half seed73129
|Operation|Stage8|Final|
|---|---:|---:|
|add97|23.33|29.17|
|sub97|47.33|17.00|
|max97|96.17|86.17|
|min97|95.17|94.17|
|add73|60.83|30.83|
|sub73|85.00|48.50|
|max73|97.33|91.17|
|min73|98.33|97.83|
|add89|5.00|4.33|
|sub89|3.67|3.33|
|max89|0.00|45.67|
|min89|0.00|40.33|
{'+-': {'n': 600, 'exact': 112, 'final': 119, 'intermediate': 338}, '-+': {'n': 600, 'exact': 118, 'final': 137, 'intermediate': 343}, '><': {'n': 600, 'exact': 383, 'final': 439, 'intermediate': 482}, '<>': {'n': 600, 'exact': 404, 'final': 448, 'intermediate': 494}, 'within': {'n': 600, 'exact': 309, 'final': 337, 'intermediate': 461}}

### C recency quarter seed73129
|Operation|Stage8|Final|
|---|---:|---:|
|add97|10.67|27.50|
|sub97|34.33|9.33|
|max97|94.50|79.17|
|min97|94.50|91.17|
|add73|32.83|21.50|
|sub73|80.00|24.83|
|max73|97.83|91.33|
|min73|99.33|94.33|
|add89|1.00|4.33|
|sub89|6.00|2.67|
|max89|0.00|34.67|
|min89|45.67|42.00|
{'+-': {'n': 600, 'exact': 217, 'final': 234, 'intermediate': 412}, '-+': {'n': 600, 'exact': 248, 'final': 260, 'intermediate': 447}, '><': {'n': 600, 'exact': 458, 'final': 503, 'intermediate': 521}, '<>': {'n': 600, 'exact': 469, 'final': 512, 'intermediate': 523}, 'within': {'n': 600, 'exact': 370, 'final': 390, 'intermediate': 497}}

### C uniform quarter seed73129
|Operation|Stage8|Final|
|---|---:|---:|
|add97|69.00|77.17|
|sub97|65.50|40.50|
|max97|94.50|89.17|
|min97|96.33|92.67|
|add73|63.00|41.33|
|sub73|65.67|36.67|
|max73|95.17|90.00|
|min73|98.50|93.17|
|add89|23.00|8.67|
|sub89|0.33|16.67|
|max89|0.00|37.33|
|min89|31.00|27.67|
{'+-': {'n': 600, 'exact': 228, 'final': 243, 'intermediate': 425}, '-+': {'n': 600, 'exact': 257, 'final': 276, 'intermediate': 411}, '><': {'n': 600, 'exact': 432, 'final': 476, 'intermediate': 515}, '<>': {'n': 600, 'exact': 447, 'final': 486, 'intermediate': 523}, 'within': {'n': 600, 'exact': 375, 'final': 403, 'intermediate': 501}}

### D cap128 seed73129
|Operation|Stage8|Final|
|---|---:|---:|
|add97|43.33|74.33|
|sub97|44.83|48.00|
|max97|90.33|89.33|
|min97|94.00|88.67|
|add73|43.33|49.67|
|sub73|47.00|45.33|
|max73|91.67|91.67|
|min73|97.00|89.83|
|add89|13.00|12.00|
|sub89|12.67|16.67|
|max89|46.33|35.00|
|min89|6.00|29.00|
{'+-': {'n': 600, 'exact': 172, 'final': 187, 'intermediate': 298}, '-+': {'n': 600, 'exact': 202, 'final': 224, 'intermediate': 316}, '><': {'n': 600, 'exact': 425, 'final': 468, 'intermediate': 507}, '<>': {'n': 600, 'exact': 431, 'final': 479, 'intermediate': 497}, 'within': {'n': 600, 'exact': 335, 'final': 356, 'intermediate': 423}}

### D cap1024 seed73129
|Operation|Stage8|Final|
|---|---:|---:|
|add97|86.50|83.33|
|sub97|84.67|68.00|
|max97|96.67|89.33|
|min97|96.33|90.33|
|add73|80.00|64.83|
|sub73|83.17|59.17|
|max73|95.83|90.50|
|min73|97.83|92.83|
|add89|14.00|20.67|
|sub89|0.33|38.33|
|max89|1.67|24.33|
|min89|1.00|66.67|
{'+-': {'n': 600, 'exact': 174, 'final': 189, 'intermediate': 300}, '-+': {'n': 600, 'exact': 202, 'final': 225, 'intermediate': 275}, '><': {'n': 600, 'exact': 374, 'final': 442, 'intermediate': 447}, '<>': {'n': 600, 'exact': 422, 'final': 462, 'intermediate': 507}, 'within': {'n': 600, 'exact': 335, 'final': 364, 'intermediate': 421}}

### D cap8192 seed73129
|Operation|Stage8|Final|
|---|---:|---:|
|add97|85.33|86.33|
|sub97|87.33|75.83|
|max97|96.67|91.83|
|min97|97.67|94.00|
|add73|79.00|53.00|
|sub73|83.00|56.83|
|max73|96.67|92.67|
|min73|97.33|94.17|
|add89|8.67|13.67|
|sub89|0.00|28.67|
|max89|7.67|32.00|
|min89|0.00|28.67|
{'+-': {'n': 600, 'exact': 185, 'final': 200, 'intermediate': 292}, '-+': {'n': 600, 'exact': 220, 'final': 242, 'intermediate': 302}, '><': {'n': 600, 'exact': 371, 'final': 429, 'intermediate': 478}, '<>': {'n': 600, 'exact': 430, 'final': 472, 'intermediate': 504}, 'within': {'n': 600, 'exact': 340, 'final': 364, 'intermediate': 410}}

### B batch 16 seed93683
|Operation|Stage8|Final|
|---|---:|---:|
|add97|27.00|3.17|
|sub97|29.83|0.50|
|max97|89.33|56.67|
|min97|86.67|47.33|
|add73|24.00|1.67|
|sub73|26.67|1.00|
|max73|88.50|54.50|
|min73|88.83|52.17|
|add89|0.67|1.67|
|sub89|0.33|0.33|
|max89|18.00|27.00|
|min89|0.00|0.00|
{'+-': {'n': 600, 'exact': 0, 'final': 1, 'intermediate': 6}, '-+': {'n': 600, 'exact': 0, 'final': 6, 'intermediate': 6}, '><': {'n': 600, 'exact': 72, 'final': 132, 'intermediate': 200}, '<>': {'n': 600, 'exact': 64, 'final': 154, 'intermediate': 187}, 'within': {'n': 600, 'exact': 71, 'final': 112, 'intermediate': 93}}

### B batch 64 seed93683
|Operation|Stage8|Final|
|---|---:|---:|
|add97|86.00|81.00|
|sub97|83.33|64.83|
|max97|96.33|93.17|
|min97|97.67|92.50|
|add73|77.67|49.00|
|sub73|75.00|48.00|
|max73|97.17|89.50|
|min73|97.50|91.33|
|add89|0.33|9.00|
|sub89|17.00|10.33|
|max89|0.67|48.33|
|min89|3.67|24.67|
{'+-': {'n': 600, 'exact': 122, 'final': 135, 'intermediate': 286}, '-+': {'n': 600, 'exact': 179, 'final': 197, 'intermediate': 271}, '><': {'n': 600, 'exact': 408, 'final': 460, 'intermediate': 481}, '<>': {'n': 600, 'exact': 392, 'final': 448, 'intermediate': 484}, 'within': {'n': 600, 'exact': 328, 'final': 348, 'intermediate': 422}}

### B batch 256 seed93683
|Operation|Stage8|Final|
|---|---:|---:|
|add97|93.00|92.17|
|sub97|93.00|89.50|
|max97|97.33|96.83|
|min97|98.33|95.67|
|add73|86.50|84.33|
|sub73|89.83|82.00|
|max73|97.17|97.17|
|min73|98.67|94.33|
|add89|28.67|7.33|
|sub89|0.00|35.67|
|max89|0.00|60.67|
|min89|95.67|26.67|
{'+-': {'n': 600, 'exact': 245, 'final': 265, 'intermediate': 360}, '-+': {'n': 600, 'exact': 324, 'final': 331, 'intermediate': 412}, '><': {'n': 600, 'exact': 512, 'final': 533, 'intermediate': 556}, '<>': {'n': 600, 'exact': 478, 'final': 518, 'intermediate': 530}, 'within': {'n': 600, 'exact': 449, 'final': 463, 'intermediate': 510}}

### C recency half seed93683
|Operation|Stage8|Final|
|---|---:|---:|
|add97|25.33|35.50|
|sub97|38.33|10.33|
|max97|93.83|91.67|
|min97|95.50|94.33|
|add73|59.67|31.67|
|sub73|77.33|37.00|
|max73|95.83|92.67|
|min73|98.00|95.17|
|add89|12.00|2.00|
|sub89|31.33|1.67|
|max89|12.33|55.67|
|min89|35.00|36.67|
{'+-': {'n': 600, 'exact': 78, 'final': 92, 'intermediate': 327}, '-+': {'n': 600, 'exact': 92, 'final': 111, 'intermediate': 251}, '><': {'n': 600, 'exact': 436, 'final': 476, 'intermediate': 517}, '<>': {'n': 600, 'exact': 384, 'final': 434, 'intermediate': 486}, 'within': {'n': 600, 'exact': 296, 'final': 321, 'intermediate': 419}}

### C recency quarter seed93683
|Operation|Stage8|Final|
|---|---:|---:|
|add97|13.17|33.83|
|sub97|28.50|4.17|
|max97|92.00|83.33|
|min97|95.67|92.33|
|add73|26.83|18.00|
|sub73|58.83|10.50|
|max73|96.67|90.50|
|min73|98.00|90.67|
|add89|5.67|1.00|
|sub89|7.00|1.33|
|max89|49.00|39.67|
|min89|49.00|40.67|
{'+-': {'n': 600, 'exact': 143, 'final': 157, 'intermediate': 415}, '-+': {'n': 600, 'exact': 167, 'final': 184, 'intermediate': 395}, '><': {'n': 600, 'exact': 463, 'final': 495, 'intermediate': 530}, '<>': {'n': 600, 'exact': 456, 'final': 487, 'intermediate': 540}, 'within': {'n': 600, 'exact': 350, 'final': 363, 'intermediate': 497}}

### C uniform quarter seed93683
|Operation|Stage8|Final|
|---|---:|---:|
|add97|59.83|77.67|
|sub97|67.67|33.00|
|max97|94.50|92.00|
|min97|97.17|91.33|
|add73|56.83|32.83|
|sub73|56.83|31.00|
|max73|94.17|90.17|
|min73|97.50|88.67|
|add89|16.33|8.00|
|sub89|1.67|10.67|
|max89|3.00|33.00|
|min89|6.33|35.00|
{'+-': {'n': 600, 'exact': 178, 'final': 193, 'intermediate': 428}, '-+': {'n': 600, 'exact': 241, 'final': 265, 'intermediate': 388}, '><': {'n': 600, 'exact': 466, 'final': 496, 'intermediate': 530}, '<>': {'n': 600, 'exact': 443, 'final': 471, 'intermediate': 535}, 'within': {'n': 600, 'exact': 377, 'final': 392, 'intermediate': 495}}

### D cap128 seed93683
|Operation|Stage8|Final|
|---|---:|---:|
|add97|45.00|75.83|
|sub97|47.50|48.17|
|max97|92.50|91.50|
|min97|94.67|93.00|
|add73|43.33|44.67|
|sub73|45.83|38.67|
|max73|93.17|89.50|
|min73|95.50|88.67|
|add89|14.00|12.67|
|sub89|20.00|21.67|
|max89|12.00|32.67|
|min89|47.33|22.33|
{'+-': {'n': 600, 'exact': 125, 'final': 135, 'intermediate': 313}, '-+': {'n': 600, 'exact': 173, 'final': 195, 'intermediate': 274}, '><': {'n': 600, 'exact': 458, 'final': 503, 'intermediate': 519}, '<>': {'n': 600, 'exact': 448, 'final': 484, 'intermediate': 521}, 'within': {'n': 600, 'exact': 329, 'final': 356, 'intermediate': 436}}

### D cap1024 seed93683
|Operation|Stage8|Final|
|---|---:|---:|
|add97|86.00|81.00|
|sub97|83.33|64.83|
|max97|96.33|93.17|
|min97|97.67|92.50|
|add73|77.67|49.00|
|sub73|75.00|48.00|
|max73|97.17|89.50|
|min73|97.50|91.33|
|add89|0.33|9.00|
|sub89|17.00|10.33|
|max89|0.67|48.33|
|min89|3.67|24.67|
{'+-': {'n': 600, 'exact': 122, 'final': 135, 'intermediate': 286}, '-+': {'n': 600, 'exact': 179, 'final': 197, 'intermediate': 271}, '><': {'n': 600, 'exact': 408, 'final': 460, 'intermediate': 481}, '<>': {'n': 600, 'exact': 392, 'final': 448, 'intermediate': 484}, 'within': {'n': 600, 'exact': 328, 'final': 348, 'intermediate': 422}}

### D cap8192 seed93683
|Operation|Stage8|Final|
|---|---:|---:|
|add97|88.00|83.83|
|sub97|86.67|70.83|
|max97|97.17|93.00|
|min97|97.00|93.17|
|add73|74.33|47.33|
|sub73|66.50|42.83|
|max73|94.33|91.67|
|min73|96.17|90.67|
|add89|10.67|15.33|
|sub89|25.67|34.00|
|max89|87.00|74.00|
|min89|1.67|22.00|
{'+-': {'n': 600, 'exact': 96, 'final': 107, 'intermediate': 235}, '-+': {'n': 600, 'exact': 144, 'final': 159, 'intermediate': 243}, '><': {'n': 600, 'exact': 431, 'final': 474, 'intermediate': 502}, '<>': {'n': 600, 'exact': 360, 'final': 431, 'intermediate': 442}, 'within': {'n': 600, 'exact': 318, 'final': 336, 'intermediate': 405}}

### B batch 16 seed116089
|Operation|Stage8|Final|
|---|---:|---:|
|add97|18.83|0.83|
|sub97|24.00|0.17|
|max97|80.50|50.83|
|min97|85.33|53.83|
|add73|19.33|1.50|
|sub73|28.50|1.83|
|max73|84.67|48.33|
|min73|87.17|54.50|
|add89|0.00|0.00|
|sub89|0.00|0.00|
|max89|2.67|19.67|
|min89|0.00|18.67|
{'+-': {'n': 600, 'exact': 0, 'final': 7, 'intermediate': 10}, '-+': {'n': 600, 'exact': 0, 'final': 5, 'intermediate': 7}, '><': {'n': 600, 'exact': 63, 'final': 132, 'intermediate': 186}, '<>': {'n': 600, 'exact': 70, 'final': 149, 'intermediate': 190}, 'within': {'n': 600, 'exact': 69, 'final': 90, 'intermediate': 107}}

### B batch 64 seed116089
|Operation|Stage8|Final|
|---|---:|---:|
|add97|87.00|83.33|
|sub97|85.50|60.17|
|max97|94.00|91.83|
|min97|97.00|93.67|
|add73|76.33|50.17|
|sub73|80.67|58.50|
|max73|94.17|91.67|
|min73|95.17|91.50|
|add89|0.00|11.33|
|sub89|4.67|17.67|
|max89|0.33|43.00|
|min89|0.67|32.00|
{'+-': {'n': 600, 'exact': 87, 'final': 100, 'intermediate': 285}, '-+': {'n': 600, 'exact': 155, 'final': 173, 'intermediate': 299}, '><': {'n': 600, 'exact': 364, 'final': 423, 'intermediate': 466}, '<>': {'n': 600, 'exact': 418, 'final': 463, 'intermediate': 501}, 'within': {'n': 600, 'exact': 281, 'final': 309, 'intermediate': 398}}

### B batch 256 seed116089
|Operation|Stage8|Final|
|---|---:|---:|
|add97|92.67|91.67|
|sub97|93.33|93.00|
|max97|98.33|96.50|
|min97|99.17|96.83|
|add73|90.50|83.00|
|sub73|88.67|87.00|
|max73|98.50|95.33|
|min73|98.67|96.00|
|add89|25.67|36.00|
|sub89|41.00|35.67|
|max89|88.00|30.33|
|min89|55.33|29.33|
{'+-': {'n': 600, 'exact': 312, 'final': 320, 'intermediate': 392}, '-+': {'n': 600, 'exact': 339, 'final': 354, 'intermediate': 414}, '><': {'n': 600, 'exact': 516, 'final': 545, 'intermediate': 548}, '<>': {'n': 600, 'exact': 504, 'final': 535, 'intermediate': 541}, 'within': {'n': 600, 'exact': 483, 'final': 499, 'intermediate': 534}}

### C recency half seed116089
|Operation|Stage8|Final|
|---|---:|---:|
|add97|29.17|26.83|
|sub97|44.17|14.83|
|max97|91.67|88.67|
|min97|95.67|92.33|
|add73|65.83|30.00|
|sub73|83.50|40.83|
|max73|97.83|92.50|
|min73|97.50|92.33|
|add89|10.67|4.67|
|sub89|13.33|2.67|
|max89|46.33|50.00|
|min89|66.00|36.67|
{'+-': {'n': 600, 'exact': 48, 'final': 54, 'intermediate': 295}, '-+': {'n': 600, 'exact': 69, 'final': 94, 'intermediate': 328}, '><': {'n': 600, 'exact': 349, 'final': 401, 'intermediate': 481}, '<>': {'n': 600, 'exact': 374, 'final': 424, 'intermediate': 484}, 'within': {'n': 600, 'exact': 244, 'final': 271, 'intermediate': 402}}

### C recency quarter seed116089
|Operation|Stage8|Final|
|---|---:|---:|
|add97|13.33|4.50|
|sub97|30.00|2.17|
|max97|91.83|82.83|
|min97|93.67|91.00|
|add73|35.67|3.67|
|sub73|67.17|8.67|
|max73|96.17|87.33|
|min73|96.17|91.00|
|add89|4.00|4.00|
|sub89|8.33|2.00|
|max89|77.00|37.33|
|min89|84.67|51.00|
{'+-': {'n': 600, 'exact': 59, 'final': 67, 'intermediate': 453}, '-+': {'n': 600, 'exact': 69, 'final': 77, 'intermediate': 470}, '><': {'n': 600, 'exact': 389, 'final': 425, 'intermediate': 523}, '<>': {'n': 600, 'exact': 394, 'final': 420, 'intermediate': 539}, 'within': {'n': 600, 'exact': 282, 'final': 298, 'intermediate': 505}}

### C uniform quarter seed116089
|Operation|Stage8|Final|
|---|---:|---:|
|add97|63.00|64.50|
|sub97|69.17|22.33|
|max97|93.17|89.00|
|min97|94.67|88.33|
|add73|57.33|32.67|
|sub73|65.67|24.17|
|max73|96.00|86.83|
|min73|97.00|87.33|
|add89|10.33|4.00|
|sub89|1.67|5.00|
|max89|35.33|43.00|
|min89|0.00|29.67|
{'+-': {'n': 600, 'exact': 106, 'final': 125, 'intermediate': 389}, '-+': {'n': 600, 'exact': 144, 'final': 160, 'intermediate': 411}, '><': {'n': 600, 'exact': 387, 'final': 422, 'intermediate': 506}, '<>': {'n': 600, 'exact': 379, 'final': 424, 'intermediate': 485}, 'within': {'n': 600, 'exact': 298, 'final': 322, 'intermediate': 458}}

### D cap128 seed116089
|Operation|Stage8|Final|
|---|---:|---:|
|add97|57.17|77.83|
|sub97|60.67|51.67|
|max97|89.83|89.17|
|min97|93.33|89.17|
|add73|51.17|42.67|
|sub73|51.50|46.33|
|max73|91.17|89.50|
|min73|96.83|89.50|
|add89|0.67|13.00|
|sub89|0.67|10.67|
|max89|1.67|32.00|
|min89|22.00|25.67|
{'+-': {'n': 600, 'exact': 148, 'final': 162, 'intermediate': 334}, '-+': {'n': 600, 'exact': 222, 'final': 241, 'intermediate': 384}, '><': {'n': 600, 'exact': 395, 'final': 448, 'intermediate': 497}, '<>': {'n': 600, 'exact': 425, 'final': 470, 'intermediate': 504}, 'within': {'n': 600, 'exact': 312, 'final': 334, 'intermediate': 450}}

### D cap1024 seed116089
|Operation|Stage8|Final|
|---|---:|---:|
|add97|87.00|83.33|
|sub97|85.50|60.17|
|max97|94.00|91.83|
|min97|97.00|93.67|
|add73|76.33|50.17|
|sub73|80.67|58.50|
|max73|94.17|91.67|
|min73|95.17|91.50|
|add89|0.00|11.33|
|sub89|4.67|17.67|
|max89|0.33|43.00|
|min89|0.67|32.00|
{'+-': {'n': 600, 'exact': 87, 'final': 100, 'intermediate': 285}, '-+': {'n': 600, 'exact': 155, 'final': 173, 'intermediate': 299}, '><': {'n': 600, 'exact': 364, 'final': 423, 'intermediate': 466}, '<>': {'n': 600, 'exact': 418, 'final': 463, 'intermediate': 501}, 'within': {'n': 600, 'exact': 281, 'final': 309, 'intermediate': 398}}

### D cap8192 seed116089
|Operation|Stage8|Final|
|---|---:|---:|
|add97|87.67|81.00|
|sub97|90.17|77.17|
|max97|95.17|92.67|
|min97|97.17|92.50|
|add73|77.50|45.00|
|sub73|77.83|47.00|
|max73|95.17|90.67|
|min73|95.83|91.17|
|add89|0.00|21.33|
|sub89|0.33|28.33|
|max89|0.00|32.33|
|min89|0.00|28.00|
{'+-': {'n': 600, 'exact': 169, 'final': 186, 'intermediate': 349}, '-+': {'n': 600, 'exact': 198, 'final': 208, 'intermediate': 269}, '><': {'n': 600, 'exact': 423, 'final': 468, 'intermediate': 513}, '<>': {'n': 600, 'exact': 451, 'final': 489, 'intermediate': 525}, 'within': {'n': 600, 'exact': 354, 'final': 379, 'intermediate': 456}}
