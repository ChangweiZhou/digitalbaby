# E16W window-net cross-form retention

Campaign status: running. 1/10 subprocesses returned (one parity profile, nine scientific lives).

Prospective predictions: E16W_PREDICTIONS.md. New schedule: four mod97 stages, composition, four mod73 stages (nine training stages, 57,344 updates). No tuning or seed substitution.

## C_uni_73129

|Stage|Held final mean|+- final/full|-+ final/full|>< final/full|<> final/full|within final/full|
|---|---|---|---|---|---|---|
|birth|11.42|0.17/0.00|0.50/0.00|35.33/0.00|9.67/0.00|6.50/0.00|
|add97|1.00|0.17/0.00|0.83/0.00|1.17/0.00|1.83/0.00|0.17/0.00|
|sub97|0.33|0.00/0.00|1.00/0.00|0.17/0.00|0.17/0.00|0.00/0.00|

|Stage|add97|sub97|max97|min97|add73|sub73|max73|min73|
|---|---|---|---|---|---|---|---|---|
|birth|8.00|0.00|29.17|74.67|7.33|0.00|46.17|76.17|
|add97|95.17|0.17|0.17|0.00|12.67|0.00|0.50|0.00|
|sub97|91.33|92.33|0.00|0.17|1.00|2.17|0.00|0.17|

## C_bal_73129

|Stage|Held final mean|+- final/full|-+ final/full|>< final/full|<> final/full|within final/full|
|---|---|---|---|---|---|---|
|birth|11.42|0.17/0.00|0.50/0.00|35.33/0.00|9.67/0.00|6.50/0.00|
|add97|1.00|0.17/0.00|0.83/0.00|1.17/0.00|1.83/0.00|0.17/0.00|
|sub97|0.38|0.00/0.00|0.83/0.00|0.33/0.00|0.33/0.00|0.33/0.00|

|Stage|add97|sub97|max97|min97|add73|sub73|max73|min73|
|---|---|---|---|---|---|---|---|---|
|birth|8.00|0.00|29.17|74.67|7.33|0.00|46.17|76.17|
|add97|95.17|0.17|0.17|0.00|12.67|0.00|0.50|0.00|
|sub97|92.67|91.67|0.00|0.00|0.50|0.50|0.00|0.17|

## D_73129

|Stage|Held final mean|+- final/full|-+ final/full|>< final/full|<> final/full|within final/full|
|---|---|---|---|---|---|---|
|birth|11.42|0.17/0.00|0.50/0.00|35.33/0.00|9.67/0.00|6.50/0.00|
|add97|0.62|0.33/0.00|0.67/0.00|0.50/0.00|1.00/0.00|0.33/0.00|
|sub97|0.42|0.33/0.00|0.67/0.00|0.33/0.00|0.33/0.00|0.83/0.00|
|max97|1.96|0.00/0.00|0.17/0.00|0.33/0.00|7.33/0.00|0.67/0.00|

|Stage|add97|sub97|max97|min97|add73|sub73|max73|min73|
|---|---|---|---|---|---|---|---|---|
|birth|8.00|0.00|29.17|74.67|7.33|0.00|46.17|76.17|
|add97|95.50|0.00|0.00|0.00|5.67|0.00|0.00|0.00|
|sub97|96.17|93.33|0.00|0.17|1.00|6.67|0.17|0.00|
|max97|95.33|93.33|94.83|0.00|0.00|4.83|48.33|0.17|

## Interpretation limits

Uniform and per-stage balanced sampling have identical marginal stage probabilities with equal 1024-entry buffers. They differ in replacement/correlation; no intended composition-share increase is present. Report buffer admission files and replay counts. A negative balanced result cannot exclude increasing composition replay. BP uses joint ground truth and differs in optimizer decay and store use; it is a ceiling, not a rule-only control. A window/cell contrast also differs in window, distractors and schedule. Arithmetic carry pairs have much lower shortcut ceilings than comparison pairs; final-answer accuracy and full-scratchpad accuracy stay separate. No broader autonomous-learning or language claim follows.
