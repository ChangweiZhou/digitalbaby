# V147 saved-output reanalysis

Existing taught records only. No new generation, training or held scoring.

| Fit, alpha 0.03 | Correct first payload | Complete byte/EOS | EOS after one byte |
| --- | ---: | ---: | ---: |
| V147-D1 FIXED | 68.60% | 66.91% | 85.75% |
| V147-D1 LOCAL | 44.69% | 44.44% | 94.69% |
| V147-D2 FIXED | 78.74% | 77.54% | 90.34% |
| V147-D2 LOCAL | 45.41% | 44.93% | 91.55% |

For this single-payload-byte task, complete / correct-first is the conditional probability of immediate exact EOS given a correct first byte on the SAME own-generated stream. It is valid here; the known-length native payload score must not be substituted into it.
V147-D1 LOCAL: conditional immediate correct EOS = 99.46%.
V147-D2 LOCAL: conditional immediate correct EOS = 98.94%.

LOCAL has lower mean teacher bit BCE but worse complete generation in both high-rate worlds. Its representation changes; FIXED is stable and still misses readiness. Neither static code uniqueness nor mean bit loss is a sufficient success proxy.

This does not identify the dominant mechanism and does not test held generalization. All holdback boundaries remain intact.
