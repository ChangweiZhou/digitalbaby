# E12 replay mechanics

Campaign status: **partial_failed**.

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
|B|batch16|73129|Pending|Pending|Pending|Pending|new|
|B|batch64|73129|90.13|79.79|54.92|1318.37|reused|
|B|batch256|73129|94.52|92.35|72.25|1958.30|new|
|C|recency half|73129|Pending|Pending|Pending|Pending|new|
|C|recency quarter|73129|68.00|54.90|62.88|1764.32|new|
|C|uniform quarter|73129|80.96|70.08|61.71|851.81|reused|
|D|cap128|73129|68.94|72.10|56.58|2135.63|new|
|D|cap1024|73129|90.13|79.79|54.92|1318.37|reused|
|D|cap8192|73129|Pending|Pending|Pending|Pending|new|
|B|batch16|93683|Pending|Pending|Pending|Pending|new|
|B|batch64|93683|88.83|76.17|51.67|1318.31|reused|
|B|batch256|93683|Pending|Pending|Pending|Pending|new|
|C|recency half|93683|Pending|Pending|Pending|Pending|new|
|C|recency quarter|93683|Pending|Pending|Pending|Pending|new|
|C|uniform quarter|93683|78.06|67.08|59.38|849.68|reused|
|D|cap128|93683|Pending|Pending|Pending|Pending|new|
|D|cap1024|93683|88.83|76.17|51.67|1318.31|reused|
|D|cap8192|93683|Pending|Pending|Pending|Pending|new|
|B|batch16|116089|Pending|Pending|Pending|Pending|new|
|B|batch64|116089|88.73|77.60|48.29|1320.68|reused|
|B|batch256|116089|Pending|Pending|Pending|Pending|new|
|C|recency half|116089|Pending|Pending|Pending|Pending|new|
|C|recency quarter|116089|Pending|Pending|Pending|Pending|new|
|C|uniform quarter|116089|79.50|61.90|47.12|763.36|reused|
|D|cap128|116089|Pending|Pending|Pending|Pending|new|
|D|cap1024|116089|88.73|77.60|48.29|1320.68|reused|
|D|cap8192|116089|Pending|Pending|Pending|Pending|new|

## Prediction verdicts

A seed73129: recovery>=60 and birth<=20: False.
A seed73129: within10 at all doses: False (not a proof of complete information loss).
A seed93683: recovery>=60 and birth<=20: False.
A seed93683: within10 at all doses: False (not a proof of complete information loss).
A seed116089: recovery>=60 and birth<=20: False.
A seed116089: within10 at all doses: False (not a proof of complete information loss).
B seed73129: carry loss>10: False.
C recency quarter seed73129: early delta-22.83, late delta-3.08; predicted True; all operation deltas within5: False.
D cap128 seed73129: stage8 delta -21.19, final delta -7.69.
D cap1024 seed73129: stage8 delta +0.00, final delta +0.00.
D cap1024 seed93683: stage8 delta +0.00, final delta +0.00.
D cap1024 seed116089: stage8 delta +0.00, final delta +0.00.

## Actual compute

Completed subprocesses: 40. Summed wall seconds: 16294.11. Profile forecast in E12_PROFILE.json is not measured total cost.

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

### B batch64 seed73129
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

### B batch256 seed73129
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

### B batch64 seed93683
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

### B batch64 seed116089
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
