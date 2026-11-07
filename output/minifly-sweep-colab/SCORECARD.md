# MiniFly Colab sweep scorecard

The Colab run completed **99 screen jobs** and **33 smoke jobs**. This scorecard uses the screen receipts. Each of the 11 blocks used three arms and three seeds. The receipts have matching panel hashes within each block and seed and identify the same code and data manifest as the notebook. The original publication included the raw results and omitted this scorecard. [Receipt ZIP](../MiniFly_outputs_receipts.zip) · [Standalone notebook](../MiniFly_Sweep_Colab.ipynb) · [Design and decision rules](../minifly-whole-repo-audit/MINIFLY_MECHANISM_SWEEP_DESIGN_20261106.md)

## Results by block

Each triple lists seeds **284813, 516197, 873411**, in that order. Scores are counts out of **64**, except grid transport, which scores **56 moving trajectories** per seed. The continual-association rows count still-valid old associations after interference. The central-complex row counts goals reached within 20°. Other rows use the exact-correct outcome defined by their own task. Compare arms within each row.

| Block | Candidate | Reference and control | Reading |
| --- | --- | --- | --- |
| Cortical feedback | Sign feedback **60 / 51 / 59** | Random feedback **30 / 37 / 36**. Backprop **61 / 61 / 64**. | **Positive.** Clears the numerical follow-up thresholds in all seeds. Supports supervised mod-97 acquisition on held operand pairs under this feedback rule. |
| Delayed-reward actor–critic | Eligibility trace **64 / 64 / 64** | Immediate credit **0 / 0 / 0**. Shifted trace **0 / 0 / 0**. | **Positive.** The trace assigns the delayed reward to the causal earlier choice in this discrete task. |
| Central complex | Native **32 / 40 / 37** | Rewired **12 / 13 / 8**. Displaced landmark **0 / 0 / 1**. | **Consistent wiring advantage.** Median goal errors were **20.49° / 15.09° / 15.66°**, so one seed misses the ≤20° criterion. Uses hand-set gains and periodic landmark correction. |
| Grid and head-direction | Attractor sharpening **0 / 0 / 0** | Diffuse transport **56 / 56 / 56**. Reset **0 / 0 / 0**. | **Negative for this sharpening implementation.** Median errors were **35.3–36.3%** of path length with sharpening and **2.9–3.0%** with diffuse transport. The diffuse arm is an engineered displacement control. |
| Sleep replay | Blocked **53 / 41 / 40** | Interleaved **50 / 35 / 38**. Pause **50 / 36 / 38**. | **Below advancement threshold.** The blocked schedule modestly improves unconditional retention in each seed. On cases acquired by both arms, its advantage over interleaving is **4.9 / 12.5 / 3.4 percentage points**. Only one seed clears 10 points and that seed retains **39/56**, just below 70%. |
| Fast/slow learning | Coupled **37 / 37 / 37** | Single store **50 / 35 / 38**. Uncoupled **42 / 36 / 37**. | **No consistent gain at this dose.** On common acquired cases, coupled retains **33/51, 28/45, 34/50** versus **42/51, 25/45, 34/50** for the single store. The proposed advantage misses the retention rule in every seed. |
| Cerebellar temporal features | Grouped lags **2 / 0 / 5** | Ordinary **1 / 1 / 6**. Shuffled **4 / 3 / 4**. | **Unqualified.** The ordinary reference is near the floor. This run cannot resolve whether grouping lags helps a capable predictor. |
| Thalamocortical iteration | Three-pass loop **0 / 0 / 0** | Feedforward **0 / 0 / 0**. One pass **33 / 24 / 33**. | **Unqualified.** The matched reference is at the floor and the simpler arm also misses 48/64. Delayed performance is zero across all arms. |
| Working-memory gate | Learned **0 / 64 / 0** | Ungated **0 / 0 / 0**. Yoked **59 / 0 / 0**. | **Unqualified.** The learned and yoked arms succeed on different single seeds. The learned gate has no repeatable advantage. |
| Mushroom-body feedback | Closed loop **32 / 37 / 37** | Open loop **32 / 37 / 37**. Permuted **32 / 37 / 37**. | **No discriminating effect.** Acquisition, retention and correction counts tie within each seed even though reward traces can differ. The v630 contact-based gain analogue does not establish a benefit from the loop or its addressing. |
| Attention reference | Attention **64 / 64 / 64** | Register **64 / 64 / 64**. Uniform pooling **64 / 64 / 64**. | **Reference only, ceiling-limited.** The task cannot distinguish the three supervised readers at this dose. |

## Qualification and scope

The notebook ran a smoke profile and the full development screen. The published artifacts do not document completion of the design memo's full per-block qualification before screening, including a capable reference where required and exact checkpoint-restart verification. Successful job receipts establish execution and preserve the measured outcomes. The floor, ceiling, and tied comparisons above limit the conclusions those outcomes support.

The two clearest positive findings are sign feedback on held mod-97 pairs and eligibility traces on delayed discrete action credit. Diffuse grid transport is a useful positive control on its own displacement fixture. It has not been evaluated on the FlyWire outbound-and-return panel. The sweep contains neither the later fitted steering confirmation nor an experiment on E5-R scratchpad composition. [Fitted steering confirmation](../minifly-connectome-navigation/FIT_OFFICIAL_REPORT.md) · [FlyWire path-integration report](../minifly-connectome-navigation/PI_REPORT.md)
