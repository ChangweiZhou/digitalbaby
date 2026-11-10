# 9.58 — E16-M2: rule or data on the two failing seeds

E16-M2 separated the two accounts of the composition failures in seeds 400009 (drifting stream) and 600013 (block stages) with two arms on the same births:
- arm `M` keeps the learner's raw-byte reservoir pipeline and swaps in the twin's backprop update;
- oracle replay keeps the sign rule and feeds it the twin's joint stream.

Results:
- **The data pipeline is not the cause (P16M2.1 holds in both seeds).** `M` held a reservoir identical to the failing learner's, the same 1,048,642 and 1,048,762 bytes and the same 55,211 and 57,921 distinct strings. It ends at 0.97 and 0.95 of the twin on held compositions, with the plus-minus pairs at 76.5–86.3, and its mod-97 operations never dip below 91.
- **Oracle replay lifts the sign rule a long way but not to the twin (P16M2.2 holds in 400009 and misses in 600013).** Compositions rise from 64.9 to 84.5 (0.90 of the twin) and from 64.7 to 78.8 (0.85). In 600013 the `+-` pair ends at 62.0, below the 65 bar.
- **The mod-97 dip follows the rule, not the data, under block stages only (P16M2.3 holds in 600013 and misses in 400009).** Under stages the dip below 70 when compositions arrive reappears with oracle data (65.7) and not in `M` (92.2). Under drift, oracle replay shows no dip at checkpoint 2 (89.3). It shows a fall to 79.2 one checkpoint later instead.

By the brief's prewritten reading the seeds split. 600013 is the rule case: the sign rule's interference persists with ideal data. 400009 is the sampling case: the rule fails on the raw reservoir but passes with oracle data. Both seeds agree on what matters for the next step: the same reservoir that backprop handles at the twin's level leaves the sign rule 9.5–14.4 points short on compositions even when the data is ideal. The remaining gap is therefore the rule's, and in 400009 sampling makes it worse. Development result on a toy harness, two seeds, one stream each. Seed 600013's comparators are scored on the pre-re-keying panels. Sources: `E16M2_REPORT.md`, `E16M2_PREDICTIONS.md`, `exp45/SUMMARY_E16M2.md`, the four receipts and final weights.
