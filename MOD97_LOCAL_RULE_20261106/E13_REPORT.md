# E13 batch-package experiment — final results

**Complete and verified:** all 18 candidate runs, both full default reproductions, and Part D's prescribed scientific stop. Three seeds in every candidate condition. No new candidate mechanisms, tasks, panels, tuning or branches. These are development results on already-used panels, not fresh independent confirmation. E13_VERIFICATION.json records the source/receipt recount.

The strongest result is lower decay at the original batch 64: wd 0.25 produces final retention 93.75/94.06/94.58% and held composition 82.29/81.00/85.00%, compared with76.17–79.79% and48.29–54.92% at wd 1.0. This demonstrates a substantial decay effect at fixed batch size and update count. Lowering decay improves both endpoints in all three seeds without adding a mechanism.

Conversely, batch 256 with wd 4.0 collapses retention to24.48–30.06% and composition to 11.33–13.25%. These arms approximately match cumulative multiplicative decay in each direction, but the high-decay batch 256 results are much worse than batch 64/wd 1.0. Consequently, cumulative decay alone is not a complete quantitative explanation: decay per step, Adam state, gradient statistics and update count interact. The loose A2<=80 gate passes, but that does not mean it reproduced the batch 64 baseline.

Batch 256/lr0.002 retains 90.75–91.58% and composes at 75.42–77.12%. Batch 1024 retains 88.81–91.19%, but composition falls to 39.54–42.21%; increasing batch size further is not uniformly beneficial. The saturation prediction fails in every seed.

Part B transfers the unmodified E12 batch 256 package to isolated composition: held-pair final means 88.33/83.04/87.71%, full-scratchpad means 86.88/81.42/86.13%, versus E9 final means 74.96/62.33/75.88%. All three pass the predeclared >=80 held-pair MEAN gate; individual pairs may score below 80. The first two training curves still rise over their last matched-exposure interval; the third changes by -0.46 points. No later/best checkpoint is substituted.

Part C does not fully preserve the ceiling: final old-operation mean improves by +1.75/+0.67/+2.44points. The strict <2point change prediction misses in one seed. BP held composition changes from89.29/91.25/90.75% to83.83/85.75/87.42% (see raw controls for exact values). Thus the optimization effect is not exclusive to the local rule, and 'the backprop ceiling did not move' is not supported globally.

Part D: the wiring gives an instantaneous displacement-direction signal; temporal accumulation is supplied by the model. Corrected true-heading clamping still leaves return residual 0.710–0.717 of outbound distance in every native seed, above 0.30. The unchanged follow-up explicitly requires STOP in this case, so no refit was performed. Turn-only compass errors 38.1–41.1degrees are also real, but fixing heading alone cannot repair the full accumulator/encoding/readout package.

## Prediction summary

| Prediction | All-three-seed verdict |
|---|---|
| A1 final retention>=90 | Supported |
| A2 final retention<=80 | Supported, but collapse is much worse than the proposed matching baseline |
| A4 retention within3points AND composition no lower than batch 256 | Refuted |
| B held-pair mean>=80 | Supported |
| C absolute final retention change<2points | Refuted: two seeds pass, one misses |
| D clamped return residual<=0.30 | Fails all seeds; required stop delivered |

## Controls, limits and interpretation

Three executed technical cycles and a production-subprocess rehearsal preceded candidates. The E12 unfinished-sibling bug, deliberate subprocess failure isolation and duplicate-launch locking were exercised. Default E8 and E7 full scientific receipts match the archived controls exactly when cpu timing fields alone are removed. Original locked scientific source and technical artifact hashes still match.

The initial Part D nominal-phase clamp was invalid (up to12.697925degrees decoded error); its script, lock, outputs and report remain archived under PI_FOLLOWUP_CYCLE1_INVALID_CLAMP and are not admitted as ceiling evidence. The accepted diagnostic numerically inverts only the fixed geometric observer to make decoded heading equal the supplied heading (maximum error <1e-6 degree). It changes no learned gain. All prior PI summary scores reproduce within 1e-7. The accepted lock, per-episode CSV, scatter and stop report are in output/minifly-connectome-navigation.

Historical E8/E9/E12 controls are reused, not mislabeled as new runs. The historical E7 BP comparator differs in width/dose from Part B; its gap cannot establish matched-architecture closure. The joint-ground-truth BP ceiling also has different access to old targets than self-generated replay; it is an upper reference, not a pure learning-rule ablation. Old development operations not rehearsed can still be lost; neither high new-operation means nor composition averages prove general autonomous lifelong learning. No model is promoted from these results.

## Part A

|Arm|Seed|Stage8 mean|Final mean|Held composition|
|---|---:|---:|---:|---:|
|A1|73129|94.58|93.75|82.29|
|A2|73129|27.73|24.48|13.25|
|A3|73129|93.21|91.58|77.12|
|A4|73129|95.25|90.17|39.54|
|A1|93683|95.27|94.06|81.00|
|A2|93683|26.71|30.06|11.33|
|A3|93683|93.04|90.75|75.42|
|A4|93683|95.67|91.19|41.62|
|A1|116089|95.52|94.58|85.00|
|A2|116089|26.56|28.06|12.33|
|A3|116089|93.44|90.88|75.83|
|A4|116089|96.25|88.81|42.21|

## Part B

|Seed|Held final mean|Full scratchpad mean|Within final|Last interval held delta|
|---:|---:|---:|---:|---:|
|73129|88.33|86.88|90.17|+2.33|
|93683|83.04|81.42|87.17|+1.79|
|116089|87.71|86.13|89.83|-0.46|

## Part C

|Seed|Batch 64 final old mean|Batch 256 final old mean|Delta|Held composition|
|---:|---:|---:|---:|---:|
|73129|94.17|95.92|+1.75|83.83|
|93683|95.75|96.42|+0.67|85.75|
|116089|93.94|96.38|+2.44|87.42|

## Part D

Steps 1–3 delivered in ../output/minifly-connectome-navigation/PI_FOLLOWUP_REPORT.md. All three native clamped-heading return medians are 0.7099–0.7170, above 0.30; prescribed scientific STOP. No refit, no new cell types/panels. Initial invalid nominal-phase clamp preserved and corrected by fixed-observer inversion; accepted clamp error below 1e-6 degree. This does not establish a working integrator.

## Prediction checks

A1 seed73129: final retention 93.75; predicted decay-threshold True. Delta vs batch 64+13.96, vs batch 256+1.40.
A2 seed73129: final retention 24.48; predicted decay-threshold True. Delta vs batch 64-55.31, vs batch 256-67.88.
A4 seed73129: within3 retention and composition no lower: False.
A1 seed93683: final retention94.06; predicted decay-threshold True. Delta vs batch 64+17.90, vs batch 256+2.56.
A2 seed93683: final retention30.06; predicted decay-threshold True. Delta vs batch 64-46.10, vs batch 256-61.44.
A4 seed93683: within3 retention and composition no lower: False.
A1 seed116089: final retention94.58; predicted decay-threshold True. Delta vs batch 64+16.98, vs batch 256+2.17.
A2 seed116089: final retention28.06; predicted decay-threshold True. Delta vs batch 64-49.54, vs batch 256-64.35.
A4 seed116089: within3 retention and composition no lower: False.
B seed73129: held>=80 True.
B seed93683: held>=80 True.
B seed116089: held>=80 True.
C seed73129: final abs delta<2 True; stage8 delta+1.54.
C seed93683: final abs delta<2 True; stage8 delta+1.00.
C seed116089: final abs delta<2 False; stage8 delta+1.67.

## Final compute and resources

20 successful subprocesses (18 candidates, two full baseline reproductions), no failed/interrupted candidate runs. Recorded training/evaluation CPU: 28756.11 seconds (7.99 CPU hours). Summed subprocess wall: 34265.10 seconds (9.52 process hours); parallel campaign elapsed: 7887.14 seconds (2.19 hours). The last two quantities measure different things. CPU clocks follow the original scripts and exclude some startup/initial scoring. Technical-cycle work, final audit, figure generation and Part D graph setup are additional costs; Part D records 16.678 seconds for its accepted diagnostic loop. Initial invalid-clamp work is preserved and also additional. Peak RSS: NOT_MEASURED.

Model: 470,801 trainable parameters, 1,883,204 bytes of weight tensors plus 3,766,408 bytes of Adam moments per live learner before gradients/activations/store. Six existing development checkpoints reused (~11.3 MB); no new development training and no new scientific life checkpoints saved. Receipts, exact launch environments, losses, score curves and logs are preserved.

A4 reporting correction: operation stages contain 384 updates, but the inherited raw loss field divides the sum by 512. Its actual stage mean is raw_loss *512/384. Composition has512 updates and needs no correction. This aggregate never feeds an optimizer step or a prediction gate. Raw receipts are unchanged; E13_VERIFICATION.json preserves both raw/corrected means for every A4 stage. Other conditions have at least 512 updates/stage.

### A1 seed73129
|Operation|Stage8|Final|
|---|---:|---:|
|add97|93.67|92.50|
|sub97|90.00|90.17|
|max97|98.83|96.83|
|min97|98.83|98.33|
|add73|88.83|86.17|
|sub73|88.17|89.50|
|max73|99.17|98.33|
|min73|99.17|98.17|
|add89|22.00|38.33|
|sub89|0.00|47.67|
|max89|96.00|6.67|
|min89|7.67|3.33|
{'+-': {'n': 600, 'exact': 360, 'final': 379, 'intermediate': 412}, '-+': {'n': 600, 'exact': 446, 'final': 456, 'intermediate': 494}, '><': {'n': 600, 'exact': 556, 'final': 576, 'intermediate': 568}, '<>': {'n': 600, 'exact': 547, 'final': 564, 'intermediate': 569}, 'within': {'n': 600, 'exact': 507, 'final': 520, 'intermediate': 544}}

### A2 seed73129
|Operation|Stage8|Final|
|---|---:|---:|
|add97|2.17|2.50|
|sub97|1.17|1.17|
|max97|48.17|48.67|
|min97|56.33|44.17|
|add73|2.17|0.50|
|sub73|1.67|1.83|
|max73|50.83|49.67|
|min73|59.33|47.33|
|add89|1.67|2.00|
|sub89|0.33|2.33|
|max89|42.67|38.00|
|min89|0.00|34.67|
{'+-': {'n': 600, 'exact': 0, 'final': 0, 'intermediate': 4}, '-+': {'n': 600, 'exact': 0, 'final': 4, 'intermediate': 7}, '><': {'n': 600, 'exact': 69, 'final': 138, 'intermediate': 225}, '<>': {'n': 600, 'exact': 85, 'final': 176, 'intermediate': 219}, 'within': {'n': 600, 'exact': 84, 'final': 114, 'intermediate': 111}}

### A3 seed73129
|Operation|Stage8|Final|
|---|---:|---:|
|add97|90.50|91.67|
|sub97|88.17|89.83|
|max97|98.00|95.83|
|min97|98.33|95.83|
|add73|85.50|80.00|
|sub73|88.00|87.33|
|max73|98.17|95.33|
|min73|99.00|96.83|
|add89|0.67|20.67|
|sub89|20.33|38.67|
|max89|1.00|18.67|
|min89|97.00|26.67|
{'+-': {'n': 600, 'exact': 341, 'final': 357, 'intermediate': 446}, '-+': {'n': 600, 'exact': 401, 'final': 416, 'intermediate': 472}, '><': {'n': 600, 'exact': 507, 'final': 545, 'intermediate': 535}, '<>': {'n': 600, 'exact': 501, 'final': 533, 'intermediate': 539}, 'within': {'n': 600, 'exact': 462, 'final': 479, 'intermediate': 519}}

### A4 seed73129
|Operation|Stage8|Final|
|---|---:|---:|
|add97|92.00|86.67|
|sub97|92.33|85.00|
|max97|99.50|95.33|
|min97|98.83|97.33|
|add73|89.67|82.00|
|sub73|90.33|79.83|
|max73|99.50|96.50|
|min73|99.83|98.67|
|add89|28.67|35.67|
|sub89|47.67|41.00|
|max89|87.00|84.00|
|min89|98.67|68.33|
{'+-': {'n': 600, 'exact': 43, 'final': 51, 'intermediate': 76}, '-+': {'n': 600, 'exact': 32, 'final': 45, 'intermediate': 41}, '><': {'n': 600, 'exact': 291, 'final': 422, 'intermediate': 333}, '<>': {'n': 600, 'exact': 345, 'final': 431, 'intermediate': 398}, 'within': {'n': 600, 'exact': 231, 'final': 267, 'intermediate': 265}}

### A1 seed93683
|Operation|Stage8|Final|
|---|---:|---:|
|add97|93.00|94.17|
|sub97|91.83|93.00|
|max97|98.83|96.17|
|min97|97.83|98.17|
|add73|91.50|89.33|
|sub73|90.67|86.33|
|max73|99.17|97.67|
|min73|99.33|97.67|
|add89|15.33|19.33|
|sub89|25.67|44.67|
|max89|13.33|5.00|
|min89|14.00|24.00|
{'+-': {'n': 600, 'exact': 396, 'final': 403, 'intermediate': 470}, '-+': {'n': 600, 'exact': 429, 'final': 438, 'intermediate': 485}, '><': {'n': 600, 'exact': 532, 'final': 543, 'intermediate': 551}, '<>': {'n': 600, 'exact': 548, 'final': 560, 'intermediate': 574}, 'within': {'n': 600, 'exact': 524, 'final': 530, 'intermediate': 559}}

### A2 seed93683
|Operation|Stage8|Final|
|---|---:|---:|
|add97|3.17|3.83|
|sub97|1.50|1.00|
|max97|46.67|61.50|
|min97|52.00|56.50|
|add73|1.33|1.17|
|sub73|2.17|2.67|
|max73|46.33|53.83|
|min73|60.50|60.00|
|add89|0.00|0.33|
|sub89|0.00|0.33|
|max89|0.00|41.00|
|min89|0.00|54.67|
{'+-': {'n': 600, 'exact': 0, 'final': 4, 'intermediate': 11}, '-+': {'n': 600, 'exact': 0, 'final': 2, 'intermediate': 6}, '><': {'n': 600, 'exact': 68, 'final': 135, 'intermediate': 182}, '<>': {'n': 600, 'exact': 79, 'final': 131, 'intermediate': 202}, 'within': {'n': 600, 'exact': 82, 'final': 120, 'intermediate': 112}}

### A3 seed93683
|Operation|Stage8|Final|
|---|---:|---:|
|add97|92.17|91.67|
|sub97|89.00|89.17|
|max97|97.33|95.83|
|min97|97.83|93.83|
|add73|86.17|83.50|
|sub73|85.83|81.17|
|max73|97.67|96.33|
|min73|98.33|94.50|
|add89|13.00|5.67|
|sub89|11.67|10.67|
|max89|13.00|24.33|
|min89|11.67|38.00|
{'+-': {'n': 600, 'exact': 344, 'final': 359, 'intermediate': 443}, '-+': {'n': 600, 'exact': 367, 'final': 377, 'intermediate': 467}, '><': {'n': 600, 'exact': 531, 'final': 551, 'intermediate': 567}, '<>': {'n': 600, 'exact': 495, 'final': 523, 'intermediate': 548}, 'within': {'n': 600, 'exact': 479, 'final': 495, 'intermediate': 519}}

### A4 seed93683
|Operation|Stage8|Final|
|---|---:|---:|
|add97|94.00|93.83|
|sub97|92.17|86.83|
|max97|98.67|97.83|
|min97|99.33|95.83|
|add73|91.17|79.83|
|sub73|90.17|82.33|
|max73|100.00|97.33|
|min73|99.83|95.67|
|add89|23.33|2.33|
|sub89|22.33|1.33|
|max89|97.00|57.33|
|min89|100.00|89.33|
{'+-': {'n': 600, 'exact': 64, 'final': 71, 'intermediate': 107}, '-+': {'n': 600, 'exact': 55, 'final': 61, 'intermediate': 75}, '><': {'n': 600, 'exact': 381, 'final': 438, 'intermediate': 456}, '<>': {'n': 600, 'exact': 353, 'final': 429, 'intermediate': 428}, 'within': {'n': 600, 'exact': 271, 'final': 305, 'intermediate': 306}}

### A1 seed116089
|Operation|Stage8|Final|
|---|---:|---:|
|add97|91.83|93.17|
|sub97|95.00|93.50|
|max97|99.17|98.83|
|min97|98.50|97.67|
|add73|90.67|89.33|
|sub73|91.50|90.17|
|max73|98.67|96.83|
|min73|98.83|97.17|
|add89|0.00|32.33|
|sub89|0.00|41.33|
|max89|0.00|16.00|
|min89|0.00|14.33|
{'+-': {'n': 600, 'exact': 426, 'final': 430, 'intermediate': 496}, '-+': {'n': 600, 'exact': 476, 'final': 482, 'intermediate': 535}, '><': {'n': 600, 'exact': 539, 'final': 553, 'intermediate': 568}, '<>': {'n': 600, 'exact': 558, 'final': 575, 'intermediate': 572}, 'within': {'n': 600, 'exact': 523, 'final': 530, 'intermediate': 560}}

### A2 seed116089
|Operation|Stage8|Final|
|---|---:|---:|
|add97|1.83|2.83|
|sub97|1.83|2.67|
|max97|47.17|54.50|
|min97|50.83|56.17|
|add73|1.33|4.00|
|sub73|1.83|2.17|
|max73|52.67|49.00|
|min73|55.00|53.17|
|add89|0.33|1.33|
|sub89|1.00|1.00|
|max89|0.33|44.67|
|min89|3.00|41.00|
{'+-': {'n': 600, 'exact': 1, 'final': 7, 'intermediate': 13}, '-+': {'n': 600, 'exact': 0, 'final': 9, 'intermediate': 5}, '><': {'n': 600, 'exact': 73, 'final': 137, 'intermediate': 210}, '<>': {'n': 600, 'exact': 74, 'final': 143, 'intermediate': 209}, 'within': {'n': 600, 'exact': 79, 'final': 103, 'intermediate': 106}}

### A3 seed116089
|Operation|Stage8|Final|
|---|---:|---:|
|add97|90.67|90.50|
|sub97|90.50|89.50|
|max97|97.00|96.50|
|min97|98.67|95.33|
|add73|88.00|79.33|
|sub73|87.67|85.67|
|max73|96.33|94.83|
|min73|98.67|95.33|
|add89|29.33|29.33|
|sub89|26.67|25.00|
|max89|79.67|19.33|
|min89|55.67|23.67|
{'+-': {'n': 600, 'exact': 347, 'final': 355, 'intermediate': 465}, '-+': {'n': 600, 'exact': 364, 'final': 376, 'intermediate': 438}, '><': {'n': 600, 'exact': 511, 'final': 545, 'intermediate': 540}, '<>': {'n': 600, 'exact': 514, 'final': 544, 'intermediate': 544}, 'within': {'n': 600, 'exact': 494, 'final': 507, 'intermediate': 539}}

### A4 seed116089
|Operation|Stage8|Final|
|---|---:|---:|
|add97|92.83|88.00|
|sub97|94.33|87.00|
|max97|98.33|95.00|
|min97|99.17|95.33|
|add73|93.00|77.33|
|sub73|93.50|79.50|
|max73|99.67|94.83|
|min73|99.17|93.50|
|add89|34.00|25.67|
|sub89|39.33|37.33|
|max89|97.33|19.67|
|min89|97.33|72.67|
{'+-': {'n': 600, 'exact': 78, 'final': 85, 'intermediate': 135}, '-+': {'n': 600, 'exact': 48, 'final': 58, 'intermediate': 62}, '><': {'n': 600, 'exact': 349, 'final': 429, 'intermediate': 409}, '<>': {'n': 600, 'exact': 326, 'final': 441, 'intermediate': 369}, 'within': {'n': 600, 'exact': 253, 'final': 292, 'intermediate': 295}}

### B seed73129
{'+-': {'n': 600, 'exact': 473, 'final': 485, 'intermediate': 548}, '-+': {'n': 600, 'exact': 488, 'final': 495, 'intermediate': 555}, '><': {'n': 600, 'exact': 561, 'final': 572, 'intermediate': 582}, '<>': {'n': 600, 'exact': 563, 'final': 568, 'intermediate': 588}, 'within': {'n': 600, 'exact': 535, 'final': 541, 'intermediate': 574}}

### B seed93683
{'+-': {'n': 600, 'exact': 443, 'final': 455, 'intermediate': 538}, '-+': {'n': 600, 'exact': 422, 'final': 429, 'intermediate': 542}, '><': {'n': 600, 'exact': 554, 'final': 563, 'intermediate': 586}, '<>': {'n': 600, 'exact': 535, 'final': 546, 'intermediate': 576}, 'within': {'n': 600, 'exact': 508, 'final': 523, 'intermediate': 559}}

### B seed116089
{'+-': {'n': 600, 'exact': 495, 'final': 501, 'intermediate': 555}, '-+': {'n': 600, 'exact': 475, 'final': 485, 'intermediate': 550}, '><': {'n': 600, 'exact': 553, 'final': 563, 'intermediate': 581}, '<>': {'n': 600, 'exact': 544, 'final': 556, 'intermediate': 577}, 'within': {'n': 600, 'exact': 529, 'final': 539, 'intermediate': 572}}

### C seed73129
|Operation|Stage8|Final|
|---|---:|---:|
|add97|96.50|95.33|
|sub97|94.83|95.00|
|max97|99.83|96.83|
|min97|99.67|99.50|
|add73|91.33|90.50|
|sub73|93.17|92.00|
|max73|100.00|98.83|
|min73|99.83|99.33|
|add89|19.00|25.33|
|sub89|0.00|24.00|
|max89|99.00|85.00|
|min89|100.00|89.00|
{'+-': {'n': 600, 'exact': 446, 'final': 453, 'intermediate': 483}, '-+': {'n': 600, 'exact': 425, 'final': 430, 'intermediate': 457}, '><': {'n': 600, 'exact': 542, 'final': 565, 'intermediate': 554}, '<>': {'n': 600, 'exact': 559, 'final': 564, 'intermediate': 575}, 'within': {'n': 600, 'exact': 534, 'final': 541, 'intermediate': 559}}

### C seed93683
|Operation|Stage8|Final|
|---|---:|---:|
|add97|96.83|95.50|
|sub97|96.50|95.50|
|max97|99.83|99.50|
|min97|99.17|99.67|
|add73|92.33|90.17|
|sub73|93.83|92.33|
|max73|100.00|99.67|
|min73|99.67|99.00|
|add89|22.00|11.33|
|sub89|48.00|44.00|
|max89|97.33|35.33|
|min89|70.33|31.00|
{'+-': {'n': 600, 'exact': 421, 'final': 428, 'intermediate': 474}, '-+': {'n': 600, 'exact': 469, 'final': 475, 'intermediate': 517}, '><': {'n': 600, 'exact': 565, 'final': 571, 'intermediate': 581}, '<>': {'n': 600, 'exact': 570, 'final': 584, 'intermediate': 577}, 'within': {'n': 600, 'exact': 541, 'final': 554, 'intermediate': 560}}

### C seed116089
|Operation|Stage8|Final|
|---|---:|---:|
|add97|97.17|96.33|
|sub97|98.33|96.33|
|max97|100.00|98.00|
|min97|99.67|99.33|
|add73|93.67|91.00|
|sub73|94.83|92.67|
|max73|100.00|98.83|
|min73|99.33|98.50|
|add89|27.33|22.67|
|sub89|41.00|33.33|
|max89|95.33|49.33|
|min89|65.67|19.67|
{'+-': {'n': 600, 'exact': 478, 'final': 486, 'intermediate': 520}, '-+': {'n': 600, 'exact': 467, 'final': 478, 'intermediate': 502}, '><': {'n': 600, 'exact': 548, 'final': 562, 'intermediate': 575}, '<>': {'n': 600, 'exact': 547, 'final': 572, 'intermediate': 565}, 'within': {'n': 600, 'exact': 543, 'final': 553, 'intermediate': 563}}
