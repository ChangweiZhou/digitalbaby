# Scaling and reasoning in MiniFly through V223

**Research note · November 5, 2026**

MiniFly’s experiments show clear benefits from widening a factual memory and some benefits from increasing training experience. The evidence for scaling reasoning remains weak. The relevant test is whether the same learner improves on never-taught combinations as experience grows, especially when a small change in the input requires a different answer. [Research memo through V223](https://github.com/ChangweiZhou/digitalbaby/blob/8fd674c162a9d97c9eaf04b9ff1f532cf9d0cd3b/PROJECT_DOCUMENTS/MINIFLY_RESEARCH_MEMO_20261105.md)

## What the scaling experiments measured

| Experiment | Result | Implication for reasoning |
|---|---|---|
| **Recovered sparse memory** | At 512 facts and 205 active units, widening the store raised old recall from **50.2% to 91.6%** and recall under unseen renderings from **36.3% to 80.1%**. | The new rendering preserves a taught key and value. The gain establishes capacity and reuse across presentations of the same fact. [Capacity experiment](https://github.com/ChangweiZhou/digitalbaby/blob/e79732d38830b7796d3449db9e7c370ea0661c4e/minifly_capacity_experiment/RESULTS.md) |
| **V216 experience scaling** | The eight-family experiment trained to 100,000 events per arm in each family and compared fewer versus more environment instances at matched doses. Relative dose, diversity, and adaptation measures improved. The high-dose arm answered **25/256** first-response items exactly and **0/128** counterfactual pairs with both answers correct. | The gains establish useful effects of experience in this environment. The counterfactual endpoint still gives no observed gain in answering both sides of a changed case correctly. [V216 campaign](https://github.com/ChangweiZhou/digitalbaby/tree/064135ae2a4a95ca4fa7d473f180f57b024e5efc/minifly_v216_experience_pretraining_design_20261104/campaign_20261104) |
| **V223 arithmetic dose** | After 65,536 distinct addition records, exact forward and BPTT32 answered **7/1,536** and **6/1,536** held sums exactly. A no-carry shortcut answered **278/1,536**. Both learners scored **0/384** carry pairs with both answers correct. | More arithmetic exposure produced some partial prediction. Complete output and sensitivity to carries remained weak on the tested graph and schedule. [V223 official results](https://github.com/ChangweiZhou/digitalbaby/blob/e79732d38830b7796d3449db9e7c370ea0661c4e/minifly_v223_learnable_recurrent_reference/OFFICIAL_REPORT.md) |

## A positive reference for reasoning

V193 updated learned successor maps from new observations after a cyclic edit. With the readout frozen and zero new relation labels, it answered **191/192** changed held queries after 128 clean observation blocks. The fully frozen arm answered **3/192**. Under continued observation corruption, the operator answered **134/192**. Both answer classes were represented in the held cases, and the reversed pairs were held out together. The interface supplied parsed operand positions and fixed temporal strides. This establishes reuse of a learned operator when its predictive state changes, within that interface. [V193 implementation and results](https://github.com/ChangweiZhou/digitalbaby/tree/e79732d38830b7796d3449db9e7c370ea0661c4e/minifly_v193_reusable_relational_operator)

## What would establish a reasoning gain from scale

Use V193 as a positive control for structural reuse. On a fixed learner and interface, vary observation dose separately from diversity, with fresh held relation families and paired interventions where the correct answer must change. Report exact generated answers with the learner’s own termination and the fraction of pairs with both answers correct. Include a shortcut baseline that can exploit the simplest sufficient rule. Report recall of taught items separately to show whether a weak held score follows weak acquisition or appears despite strong acquisition.

The proposed V223 comparison of repeated and varied intervening records diagnoses **fact acquisition**. A reasoning scaling claim needs its own outcome on unseen combinations and counterfactuals.
