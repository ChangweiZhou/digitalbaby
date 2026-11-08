# E14 decay confirmation

Status: **prepared**; completed subprocesses 0/46 (6births,39lives,1extra default profile).

E14_LOCK.json/E14_PREDICTIONS.md are prospective. Three technical cycles precede launch; full baseline and nine logged-life parity checks are required. Fresh-seed confirmation is not an independent-code replication.

## A: fresh E8 seeds

|Seed|Arm|Stage8 mean|Final mean|Held composition|
|---|---|---:|---:|---:|
|200003|A_C|Pending|Pending|Pending|
|200003|A_D|Pending|Pending|Pending|
|300007|A_C|Pending|Pending|Pending|
|300007|A_D|Pending|Pending|Pending|
|400009|A_C|Pending|Pending|Pending|
|400009|A_D|Pending|Pending|Pending|

## A: fresh E7 seeds

|Seed|Arm|Held final mean|Full scratchpad mean|Within final|Signed BP-minus-local gap|
|---|---|---:|---:|---:|---:|
|200003|A_batch256|Pending|Pending|Pending|Pending|
|200003|A_decay025|Pending|Pending|Pending|Pending|
|200003|A_BP|Pending|Pending|Pending|Pending|
|300007|A_batch256|Pending|Pending|Pending|Pending|
|300007|A_decay025|Pending|Pending|Pending|Pending|
|300007|A_BP|Pending|Pending|Pending|Pending|
|400009|A_batch256|Pending|Pending|Pending|Pending|
|400009|A_decay025|Pending|Pending|Pending|Pending|
|400009|A_BP|Pending|Pending|Pending|Pending|

## B: standing-seed E8 curve

|Seed|Decay|Stage8 mean|Final mean|Held composition|Origin|
|---|---:|---:|---:|---:|---|
|73129|0.1|Pending|Pending|Pending|new|
|73129|0.25|94.58|93.75|82.29|reused|
|73129|0.5|Pending|Pending|Pending|new|
|73129|1.0|90.13|79.79|54.92|reused|
|93683|0.1|Pending|Pending|Pending|new|
|93683|0.25|95.27|94.06|81.00|reused|
|93683|0.5|Pending|Pending|Pending|new|
|93683|1.0|88.83|76.17|51.67|reused|
|116089|0.1|Pending|Pending|Pending|new|
|116089|0.25|95.52|94.58|85.00|reused|
|116089|0.5|Pending|Pending|Pending|new|
|116089|1.0|88.73|77.60|48.29|reused|

## B: standing-seed E7 curve

|Seed|Decay|Held final mean|Within final|Full scratchpad mean|Origin|
|---|---:|---:|---:|---:|---|
|73129|0.1|Pending|Pending|Pending|new|
|73129|0.25|Pending|Pending|Pending|new|
|73129|0.5|Pending|Pending|Pending|new|
|73129|1.0|74.96|77.67|72.04|reused|
|93683|0.1|Pending|Pending|Pending|new|
|93683|0.25|Pending|Pending|Pending|new|
|93683|0.5|Pending|Pending|Pending|new|
|93683|1.0|62.33|69.17|58.42|reused|
|116089|0.1|Pending|Pending|Pending|new|
|116089|0.25|Pending|Pending|Pending|new|
|116089|0.5|Pending|Pending|Pending|new|
|116089|1.0|75.88|79.17|71.75|reused|

## C: standing-seed diagnostic summaries

Raw-gradient cosine and actual Adam descent cosine are different. Weight tensors below; bias rows and full trajectories are in diagnostics.jsonl. Output raw cosine1 is expected by construction. True gradient is on the actual batch, including replay pseudo-labels. Cross-arm means are descriptive.

|Seed|Arm|Layer|Logged points|Mean raw cosine|Mean Adam-descent cosine|Mean Adam-step/weight|First weight norm|Last weight norm|
|---|---|---|---:|---:|---:|---:|---:|---:|

## Prediction verdicts


## Compute and limits

Completed subprocess wall sum: 0.00s; parallel elapsed is separate. Actual complete CPU/diagnostic costs pending final recount. PeakRSS:NOT_MEASURED. No tuning/seed substitution on scientific negatives. Fresh births and extra observer backward passes are compute, not free resources. Log hooks are default-off and their logged/unlogged parity is verified per completed logged job.

### Standing E7 wd1.0 seed73129
{'+-': {'n': 600, 'exact': 351, 'final': 373, 'intermediate': 509}, '-+': {'n': 600, 'exact': 359, 'final': 372, 'intermediate': 518}, '><': {'n': 600, 'exact': 500, 'final': 519, 'intermediate': 564}, '<>': {'n': 600, 'exact': 519, 'final': 535, 'intermediate': 563}, 'within': {'n': 600, 'exact': 454, 'final': 466, 'intermediate': 549}}

### Standing E7 wd1.0 seed93683
{'+-': {'n': 600, 'exact': 223, 'final': 250, 'intermediate': 452}, '-+': {'n': 600, 'exact': 236, 'final': 255, 'intermediate': 494}, '><': {'n': 600, 'exact': 489, 'final': 511, 'intermediate': 557}, '<>': {'n': 600, 'exact': 454, 'final': 480, 'intermediate': 554}, 'within': {'n': 600, 'exact': 393, 'final': 415, 'intermediate': 533}}

### Standing E7 wd1.0 seed116089
{'+-': {'n': 600, 'exact': 358, 'final': 380, 'intermediate': 517}, '-+': {'n': 600, 'exact': 370, 'final': 394, 'intermediate': 515}, '><': {'n': 600, 'exact': 490, 'final': 518, 'intermediate': 549}, '<>': {'n': 600, 'exact': 504, 'final': 529, 'intermediate': 557}, 'within': {'n': 600, 'exact': 458, 'final': 475, 'intermediate': 547}}
