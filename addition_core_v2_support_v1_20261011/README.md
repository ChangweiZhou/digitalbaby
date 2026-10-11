# MiniFly addition core with observation support

COV-SUPPORT is the default development core from 11 October 2026, by the project owner’s explicit decision. It combines COV-R’s correlated residual learning with U042’s bounded observation-support write gate. COV-R remains the comparison reference.

The original experiment remains **NO_ADVANCE_RETAIN_COV_R**. The owner’s adoption is a development decision, not a claim that the prespecified low-sample upgrade gate passed. Twelve DEV worlds completed; conditional confirmation did not run.

In six BLURRY worlds, the low-sample NEW score difference was −0.83 percentage points, with paired Student 95% interval [−3.79, +2.12] and two-sided p=0.50. The BLURRY OLD held-out final difference was +25.83 points. STANDARD NEW improved by +12.08 points. These are exploratory development results, not independent confirmation of an incremental benefit.

## Use the core

Add this release directory to Python’s import path, then:

```python
from minifly_addition_core import create_core
learner = create_core()  # COV-SUPPORT by default
reference = create_core("COV-R")
```

The learner consumes `feed(byte, ticks, allowed)`. Ticks increase on each byte. It emits a prediction when `=` arrives, before the actual answer byte. A complete training record is `a+b=y` followed by newline; missing-answer records are `a+b=` followed by newline. The allowed flag governs learning when an answer actually arrives. `clone()` preserves the complete independent state; `rest(ticks)` advances time at a complete-record boundary.

Supported operand bytes are `01234ABCDEFGHIJ`; output bytes are `0` through `8`. The opaque-byte numeric mappings are learned from observed examples. The representation has engineered operand roles and an additive readout with ordinal digit output; this is a bounded engineering core rather than arithmetic emerging in unmodified Full151.

The support gate stores the last eight permitted observed answers for at most 32 ordered contexts. A conflicting answer is downweighted to one quarter only when prior observations already establish a strict majority with at least two votes. Current answers do not support themselves; both directions have separate keys. The table never supplies query answers. Missing or forbidden feedback leaves all learning state unchanged.

## Evidence and provenance

[Owner adoption](ADOPTION.json), [original experiment](../addition_observation_support_dev_v1_20261011/results/REPORT.md), [independent final audit](../addition_observation_support_dev_v1_20261011/results/FINAL_AUDIT.json), and [exploratory paired reanalysis](../addition_observation_support_dev_v1_20261011/posthoc/reanalysis_20261011/REANALYSIS.json).

The packaged learner has the qualified kernel’s operations unchanged. Its sole source adaptation is the package-relative config import; release constants and the default recipe are explicit. Adoption and packaging create no new scientific trajectories. The project objective remains improved unseen-addition ability per training sample and compute, with old knowledge retained.
