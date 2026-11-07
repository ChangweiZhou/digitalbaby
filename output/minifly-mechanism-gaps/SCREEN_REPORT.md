# MiniFly mechanism-gap screen results

All **108 screen jobs** completed across 12 blocks, three arms and three seeds. The source hash in the run lock matches the source embedded in the notebook. Every receipt matches the lock hash, and every within-seed arm comparison has a shared held-panel hash. [Notebook](../MiniFly_Mechanism_Gaps_Colab.ipynb) · [Original receipt archive](screen_receipts.zip) · [Complete scorecard](SCREEN_SCORECARD.md) · [One-seed pilot scorecard](../MiniFly_Mechanism_Gaps_Pilot_Scorecard.md)

## Main conclusions

- **Operator binding.** The isolated executor got both steps right on **100%** of cases with correct prompts. Every arm and seed had zero exact first rewrites and zero exact second rewrites, including when supplied the correct intermediate. End-to-end exact answer and EOS was **5.2–5.9%** for attention, **5.3–6.9%** for the gated register and **7.4–9.6%** for the static reader. The failed stage is prompt rewriting. The receipts do not separate operator, operand-copy and syntax errors, and they do not report accuracy on taught operator-pair types.
- **Diagonal working state.** Online eligibility and BPTT both reached **100%** on the delayed-cue task in all seeds. No-trace arms reached **55.9%, 76.2% and 92.6%**. The task doubles the distractor gap from 8 to 16 bytes and tests retention of a single cue.
- **Separate write and read gates.** Held accuracy was **87.1–97.7%**, against **57.4–82.0%** for tied gates. The separate-gate model has an additional active set of trainable gate parameters. This comparison leaves capacity and gate separation confounded.
- **Multi-action delayed credit.** Eligibility traces reached **100%** on all three held panels, compared with **0.8–24.6%** for immediate credit and **0–19.9%** for shifted eligibility. The fixture delivers a single reward after four actions. Each action has its own cue, and the sampled held trajectories can share cue combinations with training.

## Other comparisons

| Block | Three-seed reading |
| --- | --- |
| Cerebellar timing | Localized features reduced RMSE to **0.144–0.201**, versus **0.731–0.836** for the linear features and **0.774–0.804** after scrambling. This is synthetic temporal calibration. |
| Grid graph transfer | Structured transport reached **100%**, shuffled geometry **50.4–53.9%**, reset **0%**. This shows composition of known individual displacements on the engineered torus. |
| Fan-shaped-body goal writes | Aligned writes reached **100%**, permuted writes **0%** and frozen **3.9–5.5%**. This is a context-addressed engineering proxy. |
| Mushroom-body reversal | Addressed arms scored **70.7–79.3%**, compared with **46.9–54.7%** when DAN addressing was permuted. APL-on and APL-off tied within every seed. |
| Sparse fast/slow | Coupled replay reached **57.4–61.8%** on older rules, against **51.4–54.7%** without replay. |
| Adaptive apical feedback | **95.7–96.5%**, versus **96.5–97.3%** with fixed feedback. |
| Sleep order and scaling | Ordered/scaled and shuffled/scaled were close. Ordered/unscaled was slightly higher in each seed. |
| Thalamic feedback gating | Learned, constant and open-loop arms were all at **0%** on held cross-order combinations. The comparison cannot resolve an advantage among the arms. |

The fan-shaped-body and mushroom-body blocks use engineered proxies. The grid block evaluates recombination of taught moves. The main unresolved reasoning question is whether the learner can generate a correct next-step prompt on an unseen operator pairing. A field-level audit of the saved rewrites would separate selection, copying and formatting failures.
