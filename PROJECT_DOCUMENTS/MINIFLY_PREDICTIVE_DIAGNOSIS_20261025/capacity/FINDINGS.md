# ADDRESS CAPACITY: bounded historical predictive check

The expansion improved taught payload recall in every paired world, but this does not by itself identify a general learning law. Adding the single prespecified overlap scalar **worsened** leave-world-out RMSE from 0.069778 to 0.077057.

This reanalysis used only immutable saved scalar results and definition source. It performed **zero external teaching, zero native fits and zero trajectory replays**. Eight whole paired worlds were the validation units. It did not score V147 holdback or run addition.

## Paired behavior and input geometry

| Quantity | CANON 5,177 | EXPAND 20,708 |
| --- | ---: | ---: |
| OLD before NEW payload | 53.906% | 100.000% |
| OLD final payload | 1.562% | 35.547% |
| NEW final payload | 52.734% | 100.000% |
| OLD final complete | 0.000% | 11.458% |
| NEW final complete | 9.635% | 25.521% |
| OLD final payload with passive NEW | 55.469% | 100.000% |
| Write-associated OLD payload loss: passive minus write | 53.906% | 64.453% |
| Signed OLD payload change from before NEW | -52.344% | -64.453% |
| Pre-value weighted OLD/NEW overlap | 0.079181 | 0.022850 |
| Unweighted OLD/NEW overlap | 3.573603 | 0.982403 |

The paired expansion differences are 33.984 percentage points for OLD payload and 47.266 points for NEW payload. Both value lives were averaged within world; complete output averages the three frozen STD heads and excludes empty values, as the historical reducer did. This is E1 taught association evidence: an exact-key table can satisfy the task. It says nothing directly about never-taught addition.

EXPAND retains a higher final OLD score while its absolute passive-minus-write OLD loss is larger, because it started with much stronger OLD acquisition. CANON is near the final-score floor. Thus lower cross gain cannot be equated to smaller absolute accuracy loss across these arms; early acquisition and saturation matter.

Geometry was constructed before fresh values: 44 deterministic key contexts × nine fixed slots × two sides. Every side recruits 92 cells. This makes the predictor temporally prior to OLD/NEW outcome learning, but does not make it a signed teaching effect. The weighted statistic is abs(Q)×abs(T)-weighted shared support, averaged over OLD/NEW key pairs and slots/sides. It cannot distinguish beneficial and harmful changes in the answer direction.

## Small predictive test, specified before fitting

The baseline is intercept + population-arm indicator + pre-NEW OLD payload. The enhanced model adds only mean weighted_old_new_overlap. No alternatives or hyperparameters were searched. Each of eight folds withholds the whole world, including both sizes, both lives and all heads. Statistical OLS is used only to test prediction from these historical scalar measurements; it is not a candidate learner.

Timing refinement: the root-level preliminary prediction named OLD_end. Before any outcome fitting, this subanalysis pinned OLD_rest1 in ANALYSIS_SPEC.json: after the first 3600-unit rest, immediately before NEW. These are distinct earlier checkpoints. This refines the preliminary timing; it is not a claim that the original named checkpoint was used unchanged.

| Model | LOOWO MAE | LOOWO RMSE |
| --- | ---: | ---: |
| Size + prior OLD score | 0.045535 | 0.069778 |
| Plus weighted overlap | 0.053930 | 0.077057 |

The enhanced model lowered squared error in 4/8 worlds. Predictions are unclipped; fold errors, training ranks and the enhanced geometry slope are saved in RESULTS.json. A model at this sample size is fragile. This is a retrospective check on already-known histories, not prospective confirmation or a causal test of overlap. The simple size term represents a bundled engineering intervention, not pure neuron count.

| World | CANON prior OLD | CANON final OLD | CANON weighted overlap | EXPAND prior OLD | EXPAND final OLD | EXPAND weighted overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 202610238101 | 53.12% | 0.00% | 0.078581 | 100.00% | 18.75% | 0.024741 |
| 202610238102 | 59.38% | 3.12% | 0.081335 | 100.00% | 37.50% | 0.022153 |
| 202610238103 | 43.75% | 0.00% | 0.078812 | 100.00% | 50.00% | 0.022619 |
| 202610238104 | 68.75% | 0.00% | 0.077902 | 100.00% | 37.50% | 0.023434 |
| 202610238105 | 59.38% | 3.12% | 0.079678 | 100.00% | 37.50% | 0.022010 |
| 202610238106 | 50.00% | 0.00% | 0.081336 | 100.00% | 28.12% | 0.023008 |
| 202610238107 | 53.12% | 3.12% | 0.082286 | 100.00% | 37.50% | 0.022812 |
| 202610238108 | 43.75% | 3.12% | 0.073519 | 100.00% | 37.50% | 0.022023 |

## Pre-value write-response fixture

| Quantity | CANON | EXPAND |
| --- | ---: | ---: |
| Oriented self gain | 0.938838 | 1.003204 |
| Mean absolute cross gain | 0.046232 | 0.019813 |
| Absolute cross / self | 0.049244 | 0.019750 |

The fixture contains 144 partial synthetic events per size: eight addresses × nine compartments × two coefficient signs. The reported positive orientation is −Δmargin/coefficient; the actual signed margin/coefficient is negative under this native margin convention. No semantic answer correctness is involved. The 288 fixture events preceded fresh values and were not eight independent assessment worlds. They support reduced cross response and roughly 6.9% changed self gain at construction level. They cannot predict which assessment world later retains OLD better, and their ratio must not be sold as a verified retention law.

The arm contrast changed connectivity, population and Q normalization together. Stronger self gain is a confound for a pure population/collision explanation. Compact artifacts omit event-level signed fast/slow response, clipping and target-aligned cumulative influence; no additive multi-event operator or semantic generalization claim is identified.

## Stopping controls and resource boundary

Saved response controls from the separate historical 32-world official STD cohort give final NEW complete recall 11.263% STOP versus 10.645% fixed length 1 / position-only, and OLD 0.326% versus 0.977%. Thus stopping calibration explains most gain over the zero-scoring DIRECT path; genuine EOS-memory input adds only a small NEW benefit over that stronger reference. These are not extra capacity-pilot worlds, and conditional undefined groups must not be treated as observations. No stopping head was fitted here.

DIRECT payload is a known-length scoring diagnostic on native signs. The address encoder constructs nine fixed key/slot contexts and never conditions a query on an emitted output prefix; this is not teacher forcing of answer prefixes either. Complete STOP output is decoded from the same saved fixed-slot response tensor with native payload signs and replaced EOS logits. It chooses termination, but it is not an own-generated-prefix autoregressive stream. Therefore ratios across DIRECT payload and STOP complete must not be read as P(EOS | payload) for the target own-prefix interface.

Historical full-life/readout CPU was 133.122 seconds CANON and 419.870 seconds EXPAND (3.154×), with fourfold cell-associated state and edges. These timings include probing and checkpoint/output overhead, not pure core teaching. At matched 56,448 historical record applications the pilot favored the expanded bundle on E1 accuracy, not compute efficiency. No new architecture, warmup dose, task, gate or training was changed in this reanalysis.

## What this changes

1. Keep the distinction between the temporal pre-value geometry and measured signed teaching response. The former is a possible risk proxy; the latter is closer to an influence law but lacks assessment-world variation here.
2. Retain simple early-score and engineering-arm baselines. Do not infer predictive utility from the two-arm mean contrast alone, and do not add features to rescue an unfavorable validation result.
3. The next independently authorized learning comparison should record target-oriented early changes before later retention is measured. Successful association recall and larger storage remain distinct from full own-byte/EOS unseen-addition generalization.

Source: immutable commit `9803c3d4b8e5428abc54e263c32daa2cb12d8846`. ANALYSIS_SPEC.json pins the predictors and validation unit; RESULTS.json records every source blob oid/SHA-256, scalar row, fold and resource counter. No raw arrays/checkpoints/bundles were restored.
