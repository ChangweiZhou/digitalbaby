# Reasoning scaling three-cycle report

**November 6, 2026 · Engineering complete · Scientific roster not run**

| Cycle | What the audit changed | Executed result |
|---|---|---|
| **1. Benchmark and controls** | The original matched diamond pairs are invariant to swapping the opcode meanings. Added a polarity panel with identical prompts whose correct answers flip under opposite taught laws. Randomized locations within environments so increased dose can bring fresh records. | Eight technical families passed the graph witness and exposure checks. A deliberately injected held diamond was rejected. The strongest tested surface shortcut got **8/32** structural pairs in each family. In every fixture all **32** original structural pairs retained their gold answers after polarity reversal, while all **64** individual polarity answers changed on identical prefixes. Both diversity arms yielded **10,000 distinct records** in a generator-only check. [Cycle 1 receipt](CYCLE1_TEST.json) |
| **2. Learner and output path** | Kept the original width-64 V216 NORMAL learner and added the width-128 second hidden layer with expanded fixed feedback and matched output fan-in scaling. | Width 64 matched the original engine bit for bit over **16 updates**. Width 128 completed **16 updates**, with independently recomputed nonzero coordinates agreeing for the first hidden layer, second hidden layer, and output. Mutating the target preserved the earlier sensory features. Both widths generated native outputs through the 257-class head without mutating learner state. [Cycle 2 receipt](CYCLE2_TEST.json) |
| **3. Lifetime and restart** | Rehearsed A, B, and A′ with a matched no-edit branch. After a concurrent Python restart attempt hit an OpenMP shared-memory error, separated checkpoint creation and restore into sequential processes. | All four width-by-diversity arms completed the single-family rehearsal with **7,680 training events**, **376,320 training target symbols**, and two additional verification updates. The width-128 checkpoint was **705,954 bytes**. A fresh process recovered it and matched the next-update state hash exactly. Each edited and no-edit pair shared the same pre-fork state and A′ prefixes. [Cycle 3 receipt](CYCLE3_TEST.json.gz) |

## Rehearsal scores

The scores below are from one technical family at 1,536 A events, followed by 128 B events and 128 A′ events or matched no-edit events. They check the scoring and continuation paths. The full proposal uses 10,000 and 100,000 A checkpoints and fresh paired families.

| Width / environments | Exact taught A | Exact held structure A | Both structural answers | Correct under both laws across matched A′ branches |
|---|---:|---:|---:|---:|
| 64 / 4 | 2/32 | 1/64 | 0/32 | 0/64 |
| 64 / 32 | 3/32 | 1/64 | 0/32 | 0/64 |
| 128 / 4 | 4/32 | 1/64 | 0/32 | 0/64 |
| 128 / 32 | 2/32 | 1/64 | 0/32 | 0/64 |

The last column scores the same held prefix against the reversed law in the edited branch and the original law in the no-edit branch. It was separately reduced from the saved output bytes using the evaluator's two opcode interpretations. Complete taught acquisition was weak at the engineering dose, so these scores give no basis for ranking scaling settings or claiming reasoning or revision.

## Resources and launch status

The final rehearsal used **49.1 worker CPU seconds** plus the separately recorded restore check. Peak single-process RSS was about **349 MiB**. The late-training sample projects roughly **6.3 CPU hours for training alone** on the proposed roster. Full evaluation, generator and cache behavior at scale, concurrent workers, and output retention remain to be profiled. The estimate supplies an order of magnitude for admission.

The failed overlapping restart attempt completed three arms and reached the fourth post-B checkpoint before the fresh process aborted. That attempt applied 7,425 updates including one in-process verification update. Earlier smoke and numerical checks applied 100 source smoke updates and three 48-update width checks including the original-engine controls. The receipt table above refers to the finalized cycles. Python 3.9 compatibility required removing an unsupported keyword from the initial draft before the model checks.

**Disposition:** The revised design and bounded engineering harness are ready for review. Scientific launch still requires fresh locked families, the full exposure and shortcut checks, prespecified polarity and retention margins, measured resource admission, and independent review. This exercise used a self-audit and separate evaluator code. Independent review remains pending.

[Revised design](REVISED_DESIGN.md) · [Reproduction instructions](README.md)
