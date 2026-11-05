# V217 — MiniFly Round 23–24 fresh-seed reproduction

Implemented from the supplied Claude session export and completed all 150 planned jobs. This is a token-level state-law reproduction. No GLN integration, robust-role replacement, operation-grammar search or scaffold removal is claimed.

## Findings

The main conditional pattern reproduces on fresh seeds. The full Round 23 learner averages 80.2% object accuracy, above the pointer control's 73.9% and snapshot control's 49.1%. Under disjoint vocabulary, prior experience raises object accuracy from 60.4% to 98.1% and person accuracy from 76.3% to 91.0%. The paired object gain is 37.7 points, with a descriptive seed-bootstrap interval of 32.3–43.4 points.

Vocabulary reuse remains a weakness. With the later span detector, object accuracy falls from 60.4% to 54.6% and person accuracy from 76.7% to 65.0%. The object accuracy difference alone is uncertain (95% interval −17.6 to +5.6 points). With the earlier dispersion detector, object accuracy modestly increases while the native probability score worsens. Thus the detailed failure depends on the detector and metric; broad robust transfer has not been established.

C's experienced object accuracy ranges from 98.1% to 99.0% across spans 800/1500/3000. The success is broadly insensitive here, but the original claim of identical span results does not reproduce exactly on these new seeds. The next useful assay is robust role detection on vocabulary reuse, with proper causal normalized scoring and separate frozen-rule controls. This reproduction does not implement those proposed changes.

## Protocol

- Recovered the original MiniWorld generator, Round 23 dispersion learner, and both Round 24 detector versions by replaying literal source edits. Exported shell commands were not executed. Hashes are in RECOVERY.json.
- Round 23 uses ten new world seeds 23011–23020, 25 worlds × 60 questions per seed, and all five original arms (50 jobs).
- Round 24 uses ten new seeds 24011–24020. History RNG is 100 + seed, test-world RNG 600 + seed and test-stream RNG 500 + seed, following the original offsets. Each paired k0/k24 arm sees the exact same fresh-world text. Each history has 24 worlds × 60 questions. Test worlds contain 80 questions and held-out movement templates; get/drop templates remain shared.
- A reproduces the original unrestricted vocabulary. B rejects within-world case-insensitive homographs but permits cross-world reuse. C rejects content-word reuse across all worlds and collisions with function words. B is tested under the original dispersion detector and the later case-folded span detector. C spans 800, 1500 and 3000 are retained, without selecting a winner.
- PYTHONHASHSEED=241105 is fixed because the recovered generator uses unordered sets and the shuffle arm samples a set-derived list. The learner's original shuffle RNG=0 is preserved. There is no hyperparameter search.
- Streams and exact world vocabularies are saved in data/ and DATA_MANIFEST.json. Per-question traces record predictions before feedback. Online learning continues after each answer in both history and test.

## Round 23 results

Each cell is the equal-weight mean over ten independent lifetime seeds. The original memo reported 89% person and 76% object accuracy for full on one stream.

| Arm | Person accuracy | Object accuracy | Person raw score | Object raw score |
|---|---:|---:|---:|---:|
| full | 87.7% | 80.2% | 1.30 | 1.74 |
| ptronly | 87.7% | 73.9% | 1.26 | 1.95 |
| snaponly | 87.9% | 49.1% | 1.26 | 3.27 |
| shufsig | 6.9% | 5.9% | 6.38 | 6.33 |
| noupd | 59.1% | 0.0% | 2.77 | 5.93 |

## Round 24 results

The table below scores exactly the 80 fresh-world questions. Raw scores are the recovered “answer bits” calculation and carry the caveats below.

| Vocabulary / detector | Object accuracy, k0 → k24 | Person accuracy, k0 → k24 | Object raw score | Person raw score |
|---|---:|---:|---:|---:|
| A, dispersion | 55.9% → 68.8% | 75.6% → 74.8% | 2.09 → 2.81 | 1.66 → 2.99 |
| B, dispersion | 60.4% → 66.1% | 76.7% → 73.3% | 1.80 → 3.03 | 1.61 → 3.11 |
| B, span | 60.4% → 54.6% | 76.7% → 65.0% | 1.80 → 3.81 | 1.61 → 3.78 |
| C, span | 60.4% → 98.1% | 76.3% → 91.0% | 1.82 → 0.59 | 1.66 → 1.26 |

Paired seed confidence intervals and original boundary-inclusive results are in SUMMARY.json. These intervals measure variation among these synthetic world seeds; they do not establish natural-language generalization.

## Audit findings

1. **The original transfer score includes one historical question.** It counts one-based line indices with `LI >= TESTFROM`, where TESTFROM is the number of history lines. The final history line is an answer, so k24 scores 81 rather than 80 questions. Recovered files remain unchanged. The headline table is recomputed from the saved traces using `line > history_lines`. Native 81-question scores remain available for comparison.
2. **The reported bits are not a validated code length.** `known=recentp or {a}` uses the current gold answer when no recent typed place exists. The 0.85 candidate mixture loses mass when there are no candidates, and the per-token epsilon escape does not define a normalized distribution over all possible answers. The original calculation is retained only as a diagnostic for matching Claude's report. It cannot establish byte compression or a rigorous advantage over log2(6).
3. **Top-1 is computed before answer feedback.** Instrumentation reproduces the native output exactly. Replacing an answer leaves that query's predicted answer and operation evidence unchanged. The score caveat does not directly inject the answer into top-1 selection.
4. **There is an explicit QA interface.** The source recognizes `?`, takes the final question word as the entity, and updates slot typing and operation reliability from the revealed answer. “No QA pipeline” is too broad a description of this source.
5. **Substantial semantics remain supplied.** The four CO/OWN/SNAP/PTR operations, direct entity-keyed state, per-attribute non-destructive updates, and whitespace/event boundaries are implemented by hand. The learner chooses among these operations and learns slot evidence. It does not discover the repertoire or entity-state architecture.
6. **Negative controls are limited.** `ptronly` retains CO and OWN; `snaponly` also retains CO and OWN. `noupd` disables PLACE writes but continues LAST writes, co-mention candidates and learning. Shuffled signatures do not shuffle the slot types used to calculate event participants. They are the recovered controls, not complete isolation of all mechanisms.
7. **Transfer remains online.** The experiment does not freeze all learned rules during test and does not isolate operation-learning transfer from slot-typing transfer. World-disjoint C is a restricted synthetic condition; movement template holdout does not include unseen get/drop verbs.

## Reproduction

Use Python with NumPy installed, then run from any directory:

```sh
python -B run.py
python -B check.py
python -B summarize.py
```

run.py pins the hash seed in a child process, generates the saved streams/specifications and executes the recovered learners in fresh processes. It resumes existing result files. To rerun from scratch, use a separate extracted copy and remove only its generated results directory. CHECKS.json records protocol and instrumentation checks. No official roster beyond this declared replication was run.
