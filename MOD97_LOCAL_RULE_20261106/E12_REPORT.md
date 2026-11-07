# E12 replay mechanics

Campaign status: **prepared**.

Source/authorization/hashes: E12_LOCK.json. Predictions were written before gradients. Existing E8/E9 controls are reused, not new replications. Own-byte/EOS endpoints unchanged.

## A: reminder savings

| Seed | Updates | Erased | Birth | Pre-composition | Savings |
|---|---:|---:|---:|---:|---:|
|73129|64|Pending|Pending|Pending|Pending|
|73129|256|Pending|Pending|Pending|Pending|
|73129|1024|Pending|Pending|Pending|Pending|
|93683|64|Pending|Pending|Pending|Pending|
|93683|256|Pending|Pending|Pending|Pending|
|93683|1024|Pending|Pending|Pending|Pending|
|116089|64|Pending|Pending|Pending|Pending|
|116089|256|Pending|Pending|Pending|Pending|
|116089|1024|Pending|Pending|Pending|Pending|

## B/C/D: full lives

|Part|Setting|Seed|Stage8 mean|Final mean|Held composition|CPU seconds|Origin|
|---|---|---:|---:|---:|---:|---:|---|
|B|batch16|73129|Pending|Pending|Pending|Pending|new|
|B|batch64|73129|90.13|79.79|54.92|1318.37|reused|
|B|batch256|73129|Pending|Pending|Pending|Pending|new|
|C|recency half|73129|Pending|Pending|Pending|Pending|new|
|C|recency quarter|73129|Pending|Pending|Pending|Pending|new|
|C|uniform quarter|73129|80.96|70.08|61.71|851.81|reused|
|D|cap128|73129|Pending|Pending|Pending|Pending|new|
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

D cap1024 seed73129: stage8 delta +0.00, final delta +0.00.
D cap1024 seed93683: stage8 delta +0.00, final delta +0.00.
D cap1024 seed116089: stage8 delta +0.00, final delta +0.00.

## Actual compute

Completed subprocesses: 0. Summed wall seconds: 0.00. Profile forecast in E12_PROFILE.json is not measured total cost.

## Limits

Savings does not localize residual information to a specific layer. Batch changes update count and cumulative weight decay. Recency tests a nonstationary task sequence. Buffer coverage differs from exposure repetition. No tuning or promotion is inferred. Path-integration follow-up remains NOT_RUN; E11 is untouched.

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
