#!/usr/bin/env python3
"""V39 earned consolidation. Run: python3 elm_v39_earned_consolidation.py --install-deps
Eight shared-base streams, six independent memory branches each. No automatic follow-up.
Resume skips verified completed seeds; an interrupted seed restarts deterministically.
"""
from __future__ import annotations
import argparse, collections, copy, csv, hashlib, importlib, io, json, math
import os, platform, subprocess, sys, time, zipfile
from pathlib import Path
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
np=None
student_t=None
D=96
BANKS=8
OUTPUTS=256
RATE=.6
TEMP=.1
PREFIX=b'~|~|'
HEADERS=(b'@',b'#',b'?')
SEPARATORS=(b':',b'=',b'>')
MAIN_ARMS=('EXACT-V','PRED-V')
FLY_ARMS=()
TAU=.85
VERSION='v39.1.0'
SEEDS=tuple(range(39101,39109))
ENGINEERING_SEEDS=(973901,973902)
POLICIES=('LEGACY-U','ANCHOR2-U','FULL-CAP','TRACE-CORR','TRACE-EXACT2','SHUFFLED-PAIR')

REFERENCE_SHA256='319bbee9c823ffedb5849fffb87fd8adb2dcbcc0fc8fb547d943763f8bcc1b8c'
PROTOCOL="# V39 — Earned Consolidation Under Hard Capacity\n## From one-shot exact traces to bounded generalizable object memory\n\n**Status:** design / preregistration only. No V39 scientific outcomes have been observed.\n\n**Proposed scientific seeds:** `39101`–`39108`  \n**Independent base streams:** 8  \n**Memory policies:** 6 branches per seed, sharing the same base-learning trajectory  \n**Primary mechanism:** `TRACE-CORR` — probationary exact trace → corroborated two-anchor slot  \n**Primary read rule:** locked V37 `CORR2`  \n**Hard capacities:** `T = 512` probationary traces; `C = 384` consolidated slots  \n**No PBF as a load-bearing mechanism. No parameter sweep. No automatic follow-up.**\n\n---\n\n# 0. Decision in one sentence\n\nV37 established that a memory must **earn authority** before generalizing to unseen similar inputs. V38 showed that graded authority can improve utility further, but the slot-specific predictive-evidence mechanism failed its preregistered shuffled-evidence specificity veto.\n\nV39 therefore returns to the strongest scientifically supported variable — **cross-context corroboration** — and asks a new question:\n\n\\[\n\\boxed{\\text{Does a one-shot experience need a full long-term slot before it has earned reusable identity?}}\n\\]\n\nThe primary hypothesis is:\n\n> **One surprising experience deserves immediate exact recall, but only a second distinct, predictively confirmed rendering deserves expensive, similarity-addressable long-term consolidation.**\n\nThe intended architecture is therefore\n\n\\[\n\\boxed{\n\\text{shared predictive base}\n+\n\\text{bounded probationary exact traces}\n+\n\\text{bounded corroborated object slots}\n}\n\\]\n\nrather than the V38 regime in which essentially every eligible context immediately becomes a permanent full slot.\n\n---\n\n# 1. Why generic eviction is not the right first capacity experiment\n\nV38 makes the source of memory growth unusually clear.\n\nAcross the eight V38 `PRED-V` seeds at `after_D`:\n\n- mean total slots: **1460.5**;\n- mean target/object slots: **63.75**;\n- mean nontarget-only slots: **1396.75**;\n- mean nontarget multi-context slots: **232.75**;\n- mean nontarget singleton slots: **1164.0**;\n- nontarget singletons as a fraction of all slots: **79.7%**;\n- mean exact member contexts: **1862.4**;\n- mean incremental episodic payload (values + address state + PBF ledger): **7.16 MB**;\n- mean total payload including the fixed representation and shared base: **9.51 MB**;\n- mean operation proxy: about **3.40 × 10^9** multiply-add units.\n\nThe dominant waste is therefore not primarily:\n\n> “important memories live forever and need a better eviction priority.”\n\nIt is:\n\n> **“a one-off base error immediately allocates the same expensive long-term object machinery as a repeatedly useful identity.”**\n\nA generic LRU/LFU tournament would treat the symptom after the allocation has already happened. V39 instead tests whether most of those allocations should never occur.\n\n---\n\n# 2. What V39 inherits, what it rejects, and what it is allowed to change\n\n## 2.1 Frozen scientific substrate\n\nV39 freezes the following successful or settled choices:\n\n- the byte-level causal next-byte task;\n- the V36–V38 A→B→C→D workload and its revision/noise structure;\n- the shared base learner and its learning rule;\n- lag96 address geometry;\n- similarity threshold\n  \\[\n  \\tau=0.85;\n  \\]\n- absolute local value semantics (`V`);\n- PRED-V style **pre-update predictive confirmation**;\n- no replay;\n- no task ID, key ID, alias ID, wave ID, revision flag, noise flag, or evaluator metadata in the learner;\n- V37 `CORR2` as the **primary deployable read-authority rule** for an unbound similarity hit on a consolidated slot.\n\n## 2.2 V38 PBF is not promoted to a foundational primitive\n\nV38 produced an excellent engineering policy (`PBF`) but failed Q10: its slot-specific evidence assignment did not beat the shuffled-evidence control by the preregistered materiality threshold.\n\nTherefore V39 must **not** use PBF evidence to decide:\n\n- which trace is promoted;\n- which slot is evicted;\n- which memory is protected;\n- which write is accepted.\n\nPBF may be reported as an optional read-only engineering diagnostic only if this adds negligible complexity. It has no qualification role.\n\n## 2.3 V39 changes one scientific variable\n\nV39 changes **admission/consolidation state semantics**:\n\n- a first eligible context becomes a bounded probationary trace;\n- a full similarity-addressable slot is allocated only after cross-context corroboration.\n\nEverything else should be held as fixed as possible.\n\n---\n\n# 3. Engineering preflight: remove representation waste before measuring capacity\n\nA capacity experiment is uninterpretable if the measured “memory requirement” is mostly duplicate Python state.\n\nThe V38 implementation stores several quantities redundantly:\n\n- `member_vectors[c]`, although lag96 address vectors are deterministic from the 12-byte context;\n- `local[c]`, which duplicates a reference to the slot value already recoverable from membership;\n- both `sums[j]` and normalized `prototypes[j]`;\n- a permanent `eligible` set even though, for the PRED-V path, a newly eligible context is immediately allocated/bound on the same event;\n- full slot/member containers whose Python overhead is separate from their semantic content.\n\nV39 therefore begins with a **mandatory lossless-compaction preflight**.\n\n## 3.1 `COMPACT-LEGACY` representation\n\nImplement the original unbounded PRED-V semantics with no scientific change, but:\n\n1. reconstruct member lag96 vectors from the stored 12-byte context when needed;\n2. remove `local` as a separate state table;\n3. store the unnormalized prototype accumulator only and normalize on lookup;\n4. derive eligibility from resident episodic state where mathematically equivalent rather than retaining a duplicate permanent set;\n5. keep the 256-way value vector in float64 for V39 — do **not** introduce quantization as another variable.\n\n## 3.2 Preflight equivalence gate P0\n\nBefore any V39 science seed runs, `COMPACT-LEGACY` must reproduce the legacy V38-style PRED-V trajectory on at least two full verification seeds.\n\nRequired equivalence:\n\n- same base predictions at every event;\n- same tentative candidate identities and similarities to numerical tolerance;\n- same PRED-V confirmation decisions;\n- same allocations and bindings;\n- same slot values to numerical tolerance;\n- same target acquisition/retention/revision curves;\n- same CORR2 predictions and losses;\n- same final semantic state hash after canonical serialization.\n\nIf P0 fails, the V39 scientific run does not start.\n\n**This compaction is engineering hygiene, not the V39 scientific mechanism.**\n\n---\n\n# 4. The bounded memory primitives\n\nV39 introduces two explicit state types.\n\n## 4.1 Probationary exact trace\n\nA trace represents one context that has earned local memory because the shared base was wrong, but has **not** yet earned generalizable object identity.\n\nA trace stores only bounded state:\n\n- exact 12-byte context;\n- absolute float64 256-way local value;\n- exact hit count;\n- last-use / insertion counter;\n- no separately stored lag96 vector (reconstruct from context);\n- no prototype;\n- no similarity-generalization authority.\n\nHard cap:\n\n\\[\nT=512.\n\\]\n\nWhen the trace pool is full, evict the least-recently-used trace; ties break by oldest insertion order.\n\nEviction intentionally forgets that one-shot memory. There is **no hidden tombstone or unbounded eligibility table** preserving it.\n\n### Trace read semantics\n\n- exact context hit → trace value has full absolute authority;\n- merely similar context → trace is a **proposal only** and has zero current read authority.\n\nThus V37's support-1 lesson is preserved:\n\n\\[\n\\boxed{\\text{one experience may own itself, but may not yet generalize.}}\n\\]\n\n## 4.2 Consolidated two-anchor slot\n\nA slot represents an identity that has earned generalizable memory.\n\nA consolidated slot stores:\n\n- at most **two distinct exact anchor contexts**;\n- one float64 256-way absolute value;\n- one 96-dimensional prototype accumulator / prototype state;\n- access count;\n- last-use counter;\n- no additional permanent exact members beyond the two anchors.\n\nHard cap:\n\n\\[\nC=384.\n\\]\n\nA consolidated slot has CORR2 support by construction: it exists only after two distinct contexts have established the identity.\n\n### Why two anchors are sufficient for this round\n\nThe V36–V38 target geometry already shows the relevant structure:\n\n- first learned rendering ↔ second learned rendering similarity is just above \\(\\tau\\);\n- the held-out third rendering is below threshold against either raw rendering alone;\n- the prototype formed by the two learned renderings places the held-out alias above threshold and gives the successful unseen-alias transfer.\n\nThus two anchors are not an arbitrary compression number. They are the **minimum state already demonstrated to generate the desired identity basin**.\n\nAdditional similar contexts may reinforce the slot value when causally confirmed, but they do not enlarge the exact-member table or prototype state in V39.\n\n## 4.3 Consolidated-slot eviction\n\nThe same fixed policy is used by every bounded arm:\n\n1. evict the slot with the smallest access count;\n2. break ties by oldest last use;\n3. break any remaining ties by oldest allocation index.\n\nThis is `LFU→LRU`, deliberately simple and evaluator-blind.\n\nV39 does **not** claim this is the optimal eviction mechanism. It is common infrastructure so that admission policies can be compared under the same downstream pressure.\n\n---\n\n# 5. Primary mechanism: TRACE-CORR\n\nFor each observed 12-byte context \\(c\\), prediction happens before the true next byte \\(y\\) is revealed.\n\n## 5.1 Exact hit\n\n### Exact probationary trace\n\nIf \\(c\\) is in the trace table:\n\n- predict with the trace's absolute value;\n- after observing \\(y\\), update that trace with the unchanged V-rule;\n- increment exact hit / recency metadata;\n- **do not promote merely because the same exact context recurred**.\n\n### Exact consolidated anchor\n\nIf \\(c\\) is one of a slot's two anchors:\n\n- predict with the slot's absolute value;\n- after observing \\(y\\), update the slot value exactly as V37/V38 would update an exact bound member;\n- this preserves one-shot true revision on the learned anchor renderings.\n\n## 5.2 Unbound context: candidate search\n\nIf there is no exact hit, compute the nearest candidate over:\n\n1. consolidated-slot prototypes;\n2. probationary trace addresses reconstructed from their contexts.\n\nOnly candidates with cosine similarity \\(\\ge\\tau\\) are retained.\n\nIf both pools contain candidates, use the single highest-similarity candidate. Ties are resolved deterministically by state type then allocation index, fixed before science runs.\n\n## 5.3 Prediction-time authority\n\n### Candidate is a consolidated slot\n\nThe slot has already earned two-anchor support, so locked `CORR2` applies:\n\n\\[\np(y\\mid c)=p_j(y).\n\\]\n\n### Candidate is a probationary trace\n\nSupport is one, so it has **zero generalized authority**:\n\n\\[\np(y\\mid c)=p_B(y\\mid c).\n\\]\n\nThe trace candidate exists only for the post-outcome corroboration test.\n\n### No candidate\n\nUse the shared base.\n\n## 5.4 Post-outcome state transition\n\nCompute, from frozen pre-update state:\n\n- base top-1;\n- candidate top-1 if a candidate exists.\n\nThen update the shared base exactly as before.\n\n### Case A — base was correct\n\nDo not create or promote episodic memory for this previously unbound context.\n\nThe base already explains the event; V39 does not spend scarce long-term state on it.\n\n### Case B — base was wrong; trace candidate exists and predicted \\(y\\)\n\nThis is the primary consolidation event.\n\nRequire the current context to be distinct from the trace's exact context.\n\nPromote:\n\n\\[\n\\text{trace}(c_1)\n+\n(c_2,y)\n\\longrightarrow\n\\text{slot}(c_1,c_2).\n\\]\n\nProcedure:\n\n1. remove \\(c_1\\) from the trace pool;\n2. allocate a two-anchor slot (evicting one full slot first only if \\(C\\) is already reached);\n3. initialize the slot value from the trace value;\n4. apply the current observation's unchanged V-update once;\n5. set the prototype from the two distinct anchor addresses;\n6. store exactly the two anchor contexts;\n7. initialize access metadata deterministically.\n\n### Case C — base was wrong; consolidated slot candidate predicted \\(y\\)\n\nThe candidate identity was useful, but V39 does not add another permanent member.\n\n- reinforce/update the slot value using the unchanged V-rule;\n- update access metadata;\n- do not change the two-anchor identity state.\n\n### Case D — base was wrong; no candidate or candidate predicted the wrong byte\n\nCreate a new probationary exact trace for \\(c\\).\n\nIf \\(T\\) is full, evict one trace first by the fixed trace-LRU rule.\n\nThis is the only admission path for a new unsupported experience.\n\n---\n\n# 6. Why this is a genuine consolidation hypothesis rather than “smaller cache”\n\nTRACE-CORR makes a strong structural prediction.\n\nA useful arbitrary object in the current task has two distinct learned renderings that share a target consequence. The first rendering should become a trace. The second rendering should be invisible to that trace at prediction time, but if the trace's pre-update value correctly predicts the second rendering's surprising target, the pair should consolidate.\n\nBy contrast:\n\n- a singleton transient context never supplies cross-context corroboration;\n- repeated exact noise may strengthen exact recall but does not establish a reusable identity neighborhood;\n- generic contexts for which the base already predicts correctly do not earn new episodic state;\n- a wrong similarity match fails pre-update predictive confirmation and does not consolidate.\n\nThus the proposed state transition is:\n\n\\[\n\\boxed{\n\\text{surprising one-shot experience}\n\\rightarrow\n\\text{probationary exact trace}\n\\rightarrow\n\\text{distinct-context predictive corroboration}\n\\rightarrow\n\\text{generalizable long-term slot}\n}\n\\]\n\nThis directly reuses the scientifically validated V37 distinction between support 1 and support 2.\n\n---\n\n# 7. V39 memory policies\n\nAll policies share the same data stream and shared-base trajectory within each seed. Auxiliary memory state branches independently.\n\n## 7.1 LEGACY-U — unbounded compact legacy reference\n\n`COMPACT-LEGACY` PRED-V semantics with locked CORR2 read authority and no capacity limit.\n\nPurpose:\n\n- fresh-seed reference for target quality and bpb;\n- denominator for memory/operation reduction;\n- not deployable as a bounded learner.\n\n## 7.2 ANCHOR2-U — unbounded two-anchor representation diagnostic\n\nImmediate full-slot allocation as in legacy, but every slot stores at most two exact anchors and fixed two-anchor identity state.\n\nNo slot cap.\n\nPurpose:\n\n- isolate whether the two-anchor state restriction itself changes target behavior before capacity pressure is introduced.\n\nIf ANCHOR2-U materially damages target persistence or held-out revision relative to LEGACY-U, V39 cannot attribute later effects cleanly to admission and the primary consolidation claim is void.\n\n## 7.3 FULL-CAP — hard cap without probationary admission\n\nSame two-anchor slot representation as ANCHOR2-U, but:\n\n- every first base-miss immediately allocates a full slot;\n- slot cap \\(C=384\\);\n- no trace tier;\n- common `LFU→LRU` slot eviction.\n\nThis is the primary “just cap/evict the old architecture” control.\n\n## 7.4 TRACE-CORR — primary deployable V39 arm\n\nThe two-tier policy in Section 5:\n\n- trace cap \\(T=512\\);\n- slot cap \\(C=384\\);\n- distinct-context predictive confirmation required for promotion;\n- common `LFU→LRU` full-slot eviction;\n- CORR2 read authority for consolidated slots.\n\n## 7.5 TRACE-EXACT2 — recurrence control\n\nSame two-tier state and same caps as TRACE-CORR, except promotion is earned by the **second exact occurrence of the same probationary context**, not by a distinct context.\n\nThe promoted slot receives recurrence credit but only one unique anchor prototype.\n\nPurpose:\n\n\\[\n\\boxed{\\text{Does generic recurrence/frequency explain the benefit, or is cross-context identity corroboration necessary?}}\n\\]\n\nThis arm should preserve many frequently repeated exact memories but is predicted to generalize poorly to held-out aliases whose similarity requires the two-rendering prototype.\n\n## 7.6 SHUFFLED-PAIR — nondeployable resource-matched control\n\nThis control receives the same **promotion count and promotion event times** as TRACE-CORR for each seed, supplied by a shadow TRACE-CORR schedule that cannot affect the control's predictions.\n\nAt each scheduled promotion, instead of promoting the causally matched trace, select a random resident trace from the same exact-hit-count class and nearest feasible age quartile; if the matched stratum is empty, fall back deterministically to the full trace pool.\n\nPair that random trace with the current context and execute the same allocation/value-update budget.\n\nThus SHUFFLED-PAIR matches approximately:\n\n- number of full slots created;\n- promotion timing;\n- trace removals;\n- value update count;\n- hard capacities.\n\nBut it deliberately destroys **which prior experience is deemed the corroborating identity**.\n\nThis is the decisive causal control for the consolidation story. It is evaluator-assisted and explicitly nondeployable.\n\n---\n\n# 8. Capacity-stress tail\n\nThe ordinary A→B→C→D stream already creates more singleton nontarget slots than the proposed trace capacity. Nevertheless V39 should include a fixed, compact stress tail so that boundedness is tested deliberately rather than incidentally.\n\nAppend the tail **after the existing `after_D` checkpoint**. No old target item is replayed for learning during this tail.\n\nThe tail uses the same 14-byte framing family and no learner-visible type labels.\n\n## 8.1 `SINGLETON` clutter\n\nGenerate **768** new unique framed contexts, each presented exactly once with an arbitrary target from the same 24-byte target alphabet.\n\nPurpose:\n\n- force the probationary trace pool to turn over;\n- make immediate-full-allocation policies consume/evict long-term slots;\n- test whether one-off experience can be forgotten without destroying consolidated objects.\n\n## 8.2 `EXACT-REPEAT` clutter\n\nGenerate **128** additional unique contexts, each presented exactly twice with the same target, separated in time.\n\nNo alternate rendering is ever shown.\n\nPurpose:\n\n- distinguish generic recurrence from cross-context corroboration;\n- give TRACE-EXACT2 a fair class of events it is designed to consolidate.\n\n## 8.3 Schedule\n\nUse a fixed two-pass schedule:\n\n- pass 1: all 768 singleton rows plus first occurrences of the 128 repeated contexts, seed-shuffled;\n- pass 2: second occurrences of the 128 repeated contexts, interleaved with an additional deterministic permutation of remaining singleton-style filler windows if needed for spacing.\n\nThe exact generator, row count, and stream SHA are locked in the run manifest before science execution.\n\n## 8.4 Read-only post-tail probes\n\nAfter the tail, probe without learning:\n\n- untouched A keys;\n- revised A keys at their final true values;\n- B/C/D target keys;\n- known learned renderings;\n- held-out alias 2;\n- a sample of stress-tail singleton and exact-repeat contexts;\n- base-only counterfactual.\n\nThis creates a direct persistence-through-intervening-experience endpoint.\n\n---\n\n# 9. Primary hypotheses\n\n## H1 — admission, not merely eviction, is the dominant capacity lever\n\nTRACE-CORR should preserve substantially more old target memory than FULL-CAP under the same consolidated-slot capacity because one-off contexts remain in the bounded trace tier instead of immediately evicting long-term objects.\n\n## H2 — one-shot exact learning survives probation\n\nA target's first eligible rendering should be recalled on its first exact recurrence from the trace tier without waiting for consolidation.\n\nThus the mechanism must not solve capacity by giving up sample efficiency.\n\n## H3 — cross-context corroboration earns unseen-alias transfer\n\nAfter a second distinct learned rendering confirms the first trace, TRACE-CORR should consolidate a two-anchor slot and restore held-out alias transfer close to the unbounded reference.\n\n## H4 — exact recurrence is not equivalent to identity corroboration\n\nTRACE-CORR should beat TRACE-EXACT2 on held-out alias transfer while remaining competitive on exact recurrence and whole-stream utility.\n\n## H5 — the identity-specific pairing matters beyond promotion rate\n\nTRACE-CORR should beat SHUFFLED-PAIR despite matched promotion timing/count and nearly matched state occupancy.\n\nIf SHUFFLED-PAIR matches TRACE-CORR, the scientific consolidation claim is rejected.\n\n## H6 — boundedness is literal\n\nAt no event may TRACE-CORR exceed:\n\n\\[\n\\boxed{512\\text{ traces},\\quad384\\text{ slots},\\quad768\\text{ stored slot anchors}.}\n\\]\n\nNo hidden eligibility, tombstone, replay, or unbounded member table may exist outside these caps.\n\n---\n\n# 10. Qualification gates\n\nAll thresholds below are fixed before V39 results are observed.\n\n## Q0 — engineering equivalence\n\nP0 `COMPACT-LEGACY` equivalence passes. Otherwise no scientific interpretation.\n\n## Q1 — two-anchor representation adequacy\n\nANCHOR2-U versus LEGACY-U:\n\n- A-after-D loss no worse than **2 pp**;\n- held-out revision @4 loss no worse than **5 pp**;\n- whole-stream bpb no worse by more than **0.05**.\n\nFailure voids the bounded admission comparison because the representation restriction itself is damaging the mechanism.\n\n## Q2 — target acquisition under TRACE-CORR\n\nMean target acquisition after the standard learning exposures:\n\n\\[\n\\ge 90\\%.\n\\]\n\nEvery seed must remain \\(\\ge 80\\%\\).\n\n## Q3 — old persistence after D\n\nFor untouched A targets before the capacity-stress tail:\n\n\\[\n\\text{TRACE-CORR A-after-D}\\ge90\\%\\text{ mean},\n\\]\n\nwith every seed \\(\\ge80\\%\\).\n\n## Q4 — persistence after clutter\n\nAfter the full capacity-stress tail, on the union of stable A/B/C/D target objects excluding deliberately obsolete labels:\n\n\\[\n\\text{current-target accuracy}\\ge90\\%\\text{ mean},\n\\]\n\nwith every seed \\(\\ge80\\%\\).\n\nThis is the main lifelong-retention endpoint.\n\n## Q5 — revision remains functional\n\nOn the existing true-revision keys:\n\n- exact anchor revision @4 \\(\\ge90\\%\\);\n- known other learned rendering @4 \\(\\ge90\\%\\);\n- held-out alias @4 \\(\\ge90\\%\\);\n- obsolete-value probability after 8 revision presentations \\(\\le0.10\\).\n\n## Q6 — whole-stream utility\n\nTRACE-CORR whole-stream bpb must be no more than **0.10 bpb worse** than LEGACY-U on the same seed.\n\nThe same 0.10 noninferiority margin applies to nontarget bpb.\n\n## Q7 — literal capacity\n\nFor every event in every seed:\n\n- trace count \\(\\le512\\);\n- consolidated-slot count \\(\\le384\\);\n- stored consolidated anchor contexts \\(\\le768\\);\n- no auxiliary context-address table exceeds the sum of the explicitly bounded resident contexts;\n- no replay buffer exists;\n- no historical eviction log is required for prediction or learning.\n\nAny violation fails V39 regardless of accuracy.\n\n## Q8 — semantic memory reduction\n\nPeak incremental episodic semantic payload for TRACE-CORR must satisfy both:\n\n1. absolute bound \\(\\le 2.30\\) MiB, excluding the unchanged shared base/fixed encoders and evaluator logs;\n2. \\(\\le40\\%\\) of LEGACY-U's peak incremental episodic payload on the same seed.\n\nReport actual Python container overhead separately; it cannot be used to hide an unbounded structure.\n\n## Q9 — admission beats immediate allocation under the same slot cap\n\nTRACE-CORR − FULL-CAP on post-clutter current-target accuracy:\n\n- paired mean advantage \\(\\ge5\\) percentage points;\n- paired 95% lower interval \\(>0\\).\n\nIf FULL-CAP is already noninferior and equally bounded, probationary admission has not earned its complexity.\n\n## Q10 — distinct-context corroboration beats exact recurrence\n\nTRACE-CORR − TRACE-EXACT2 on held-out alias accuracy after consolidation/revision:\n\n- paired mean advantage \\(\\ge10\\) percentage points;\n- paired 95% lower interval \\(>0\\).\n\nAt the same time, TRACE-CORR exact-recurrence accuracy must be noninferior within **5 pp**.\n\nThis is the primary mechanism-specific gate.\n\n## Q11 — identity-specific promotion beats shuffled pairing\n\nTRACE-CORR must beat SHUFFLED-PAIR on at least one of:\n\n- post-clutter current-target accuracy by \\(\\ge10\\) pp;\n- held-out alias accuracy by \\(\\ge10\\) pp;\n- whole-stream bpb by \\(\\ge0.10\\);\n\nand the corresponding paired interval must exclude zero in TRACE-CORR's favor.\n\nIf Q11 fails, the claim that the correct corroborating identity matters is rejected.\n\n## Q12 — one-shot trace sample efficiency\n\nFor target contexts that become probationary traces:\n\n- first exact recurrence top-1 accuracy \\(\\ge90\\%\\) mean;\n- no more than one external target presentation is required to create the exact trace.\n\n## Q13 — immediate E1→E2 consolidation transfer\n\nFor a fresh logical target:\n\n- after E1: exact E1 recurrence is remembered, but held-out alias has no required generalized authority;\n- after the first distinct E2 confirmation: the slot exists immediately;\n- the first subsequent held-out alias probe has top-1 accuracy \\(\\ge90\\%\\) mean.\n\nThis prevents a hidden many-shot consolidation mechanism.\n\n---\n\n# 11. Evaluator-only admission diagnostics\n\nThese diagnostics may use key/type metadata **only after the run**. They cannot influence the learner.\n\nReport:\n\n- fraction of original 64 target logical keys that reach consolidated slots;\n- number of target traces evicted before consolidation;\n- singleton clutter trace admission rate;\n- singleton clutter consolidation rate;\n- exact-repeat clutter consolidation rate;\n- nontarget consolidated slots;\n- target consolidated slots;\n- promotion precision: fraction of consolidated target pairs belonging to the same evaluator-known logical key;\n- cross-key false promotion count;\n- slot evictions by target/nontarget evaluator class;\n- trace residence-time distribution;\n- slot access-count distribution;\n- peak and terminal occupancy of each tier.\n\nThe expected TRACE-CORR signature is:\n\n\\[\n\\boxed{\n\\text{high target consolidation}\n+\n\\text{near-zero singleton consolidation}\n+\n\\text{large trace churn}\n+\n\\text{stable consolidated target slots}\n}\n\\]\n\nbut qualification is determined by Q gates, not by visually pleasing diagnostics.\n\n---\n\n# 12. Resource model and hard upper bound\n\nV39 must report semantic payload from first principles, not only `sys.getsizeof`.\n\nWith float64 values and no cached trace vectors, a conservative payload budget is approximately:\n\n### One probationary trace\n\n- context: 12 bytes;\n- 256 float64 value: 2048 bytes;\n- counters/metadata: at most 24 bytes.\n\nBudget:\n\n\\[\n\\le 2084\\text{ bytes/trace}.\n\\]\n\nAt \\(T=512\\):\n\n\\[\n\\le 1.02\\text{ MiB}.\n\\]\n\n### One consolidated slot\n\n- two anchor contexts: 24 bytes;\n- 96 float64 prototype accumulator: 768 bytes;\n- 256 float64 value: 2048 bytes;\n- counters/metadata: at most 32 bytes.\n\nBudget:\n\n\\[\n\\le 2872\\text{ bytes/slot}.\n\\]\n\nAt \\(C=384\\):\n\n\\[\n\\le 1.05\\text{ MiB}.\n\\]\n\nAllowing bounded index arrays/maps, the preregistered semantic episodic cap is therefore:\n\n\\[\n\\boxed{2.30\\text{ MiB}.}\n\\]\n\nThe unchanged shared base and fixed encoders are reported separately and do not count toward Q8, because V39 specifically tests the episodic-memory growth mechanism.\n\n---\n\n# 13. Operation accounting\n\nThe current unbounded PRED-V lookup cost grows approximately linearly with slot count. V39 should report operation proxies for:\n\n- base prediction/update;\n- consolidated-slot similarity scan;\n- probationary-trace similarity scan;\n- address reconstruction for traces;\n- value scoring;\n- allocation/eviction operations.\n\nDo **not** claim sublinear retrieval in V39. The bounded scan is still linear in the fixed capacities:\n\n\\[\nO((T+C)D),\n\\]\n\nbut because \\(T+C=896\\) is constant, lookup cost no longer grows with lifetime experience.\n\nApproximate-nearest-neighbor indexing / LSH is explicitly deferred until a bounded semantic memory has first been established.\n\n---\n\n# 14. Hard implementation invariants\n\nV39 is invalid if any invariant fails.\n\n### I1 — one shared base trajectory per seed\n\nAll memory policies in a seed see the same base pre-update distribution and the same base update. Auxiliary policy predictions cannot change base learning.\n\n### I2 — prediction precedes outcome\n\nTrace/slot candidate choice and prediction use only pre-outcome state. Predictive confirmation is evaluated only after the true byte arrives.\n\n### I3 — no evaluator metadata\n\nNo memory policy can access logical key, alias, target/nontarget class, stress-tail type, wave name, noise/revision status, or probe type.\n\n### I4 — no hidden unbounded eligibility\n\nTRACE-CORR may remember an unsupported context only if that context is currently resident in the bounded trace tier or is an explicit consolidated anchor.\n\nAn evicted trace is genuinely forgotten.\n\n### I5 — only two exact slot anchors\n\nA consolidated slot cannot accumulate an unbounded member set. Additional confirmed similar contexts may reinforce the value but do not become permanent exact members in V39.\n\n### I6 — probes are read-only\n\nAll evaluator probes leave base, trace, slot, counters, recency, and RNG state unchanged.\n\n### I7 — deterministic eviction\n\nGiven the same state and event, every deployable arm's eviction choice is deterministic.\n\nOnly SHUFFLED-PAIR has a dedicated preregistered RNG.\n\n### I8 — matched control schedule cannot leak into deployable arms\n\nTRACE-CORR never reads SHUFFLED-PAIR or evaluator state. SHUFFLED-PAIR may read the shadow TRACE-CORR promotion schedule only because it is a nondeployable causal control.\n\n### I9 — resume determinism\n\nResume must reject source/protocol/plan hash mismatch and reproduce uninterrupted outputs exactly.\n\n---\n\n# 15. Required outputs — deliberately compact\n\nPrevious rounds produced unnecessarily large archives. V39 should default to aggregate scientific outputs and avoid per-byte CSV dumps.\n\nRequired:\n\n- `SUMMARY.md`\n- `REPORT.json`\n- `SUMMARY.csv`\n- `PROTOCOL.md`\n- `RUN_MANIFEST.json`\n- `STATUS.json`\n- `STATE_COSTS.json`\n- `OCCUPANCY_CURVES.csv` — checkpoints/event blocks, not every byte\n- `ADMISSION_SUMMARY.csv`\n- `EVICTION_SUMMARY.csv`\n- `E1_E2_CONSOLIDATION.csv`\n- `RETENTION_BY_WAVE.csv`\n- `POST_CLUTTER_RETENTION.csv`\n- `REVISION_CURVES.csv`\n- `PAIRED_CONTRASTS.csv`\n- `CONTROL_AUDIT.json`\n- `COMPACTION_EQUIVALENCE.json`\n- one self-contained executable `elm_v39_earned_consolidation.py`\n- artifact hashes.\n\nPer-event traces are **off by default**. An optional `--audit-events` flag may emit a compressed detailed log for one specified seed, but this is not part of the scientific archive.\n\nTarget archive size: **well below 20 MB**.\n\n---\n\n# 16. Engineering tests before science\n\n`--self-test` must verify at least:\n\n1. compact legacy state is semantically equivalent on synthetic sequences;\n2. first base miss creates one trace and no full slot in TRACE-CORR;\n3. exact trace recurrence gets full local authority;\n4. similar support-1 trace has zero current authority;\n5. distinct similar context + pre-update predictive confirmation promotes exactly once;\n6. promotion removes the source trace and creates exactly one two-anchor slot;\n7. same exact recurrence alone never promotes TRACE-CORR;\n8. TRACE-EXACT2 does promote after the specified exact recurrence;\n9. consolidated slot has full CORR2 authority on a new similar context;\n10. exact anchor contradiction updates the whole slot value immediately;\n11. a third confirmed similar context does not increase permanent anchor/member count;\n12. trace pool never exceeds 512 under overflow;\n13. slot pool never exceeds 384 under overflow;\n14. slot eviction removes both anchor lookup entries and all slot state;\n15. evicted trace leaves no eligibility/tombstone residue;\n16. probes do not alter recency counters;\n17. SHUFFLED-PAIR matches the primary promotion count/timing on a synthetic schedule;\n18. SHUFFLED-PAIR changes identity assignment;\n19. all deployable arms share identical base hashes;\n20. deterministic resume/restart reproduces hashes.\n\n`--verify-full` should additionally run one complete engineering seed and verify:\n\n- P0 compact-legacy equivalence;\n- all hard capacities at every event;\n- no leaked evaluator metadata;\n- no output artifact exceeds the configured logging budget.\n\n---\n\n# 17. Decision tree after V39\n\n## Outcome A — TRACE-CORR passes Q0–Q13\n\nInterpretation:\n\n\\[\n\\boxed{\\text{corroborated consolidation establishes a bounded persistent associative memory on this workload.}}\n\\]\n\nThe architecture has now separated:\n\n1. shared statistical prediction;\n2. one-shot exact episodic traces;\n3. reusable object-level long-term memory;\n4. corroborated read authority.\n\n**Next round:** deliberately exceed the number of genuinely reusable objects that the consolidated capacity can hold. That is the first scientifically legitimate round for testing eviction/consolidation priority among *useful* memories rather than filtering transient clutter.\n\n## Outcome B — TRACE-CORR preserves targets but Q9 fails because FULL-CAP matches it\n\nInterpretation: the trace tier is not scientifically necessary at this capacity/workload. The benefit comes from the fixed cap/common eviction or engineering compaction.\n\nDo not elaborate probationary consolidation.\n\n## Outcome C — TRACE-CORR beats FULL-CAP but Q10 fails because TRACE-EXACT2 matches it\n\nInterpretation: recurrence/frequency is enough; cross-context corroboration has not earned a mechanistic role in admission.\n\nUse the simpler recurrence-based admission rule in the next capacity stress.\n\n## Outcome D — SHUFFLED-PAIR matches TRACE-CORR\n\nInterpretation: correct identity-specific pairing is not established as causal. Reject the corroboration-consolidation story despite any good headline memory numbers.\n\n## Outcome E — target traces are evicted before E2\n\nInterpretation: the trace tier is too small **for this fixed schedule**, not proof that the consolidation idea is wrong.\n\nDo not tune `T` post hoc. Report failure and redesign the task/capacity relationship in the next preregistered round.\n\n## Outcome F — two-anchor representation itself fails Q1\n\nInterpretation: exact-member state contains important information beyond the minimal two-rendering prototype. Do not proceed with capacity claims; first identify which additional member statistic is necessary.\n\n---\n\n# 18. What V39 may claim if successful\n\nA successful V39 would establish, on the controlled byte workload, that:\n\n- one-shot local learning does not require permanent full-slot allocation;\n- a bounded trace tier can preserve immediate exact recall;\n- distinct-context predictive corroboration can promote selected experiences into reusable object memory;\n- the promoted memory preserves old knowledge through substantial intervening transient experience;\n- held-out alias transfer and rapid revision survive the bounded state transition;\n- total episodic state and candidate-scan cost are bounded independently of lifetime stream length;\n- matched cap/recurrence/shuffled-pair controls support the claim that admission structure, not merely fewer allocations, matters.\n\nIt would **not** establish:\n\n- optimal eviction among more useful memories than capacity can hold;\n- robustness to one-off false revision;\n- natural-language semantic identity;\n- sublinear memory retrieval;\n- large-scale language-model competitiveness;\n- biological equivalence to mushroom-body consolidation.\n\nThe intended claim is deliberately narrower:\n\n\\[\n\\boxed{\n\\text{Immediate exact memory can be cheap and temporary; reusable identity memory must be earned.}\n}\n\\]\n\n---\n\n# 19. Why V39 should be this round\n\nThe project has finally reached a point where “capacity” can be decomposed rather than attacked by another heuristic sweep.\n\nThe sequence of established questions is now:\n\n\\[\n\\begin{aligned}\n\\text{V35:}&\\quad \\text{Who owns a memory?}\\\\\n\\text{V36:}&\\quad \\text{Which renderings belong to one object?}\\\\\n\\text{V37:}&\\quad \\text{When may that object memory speak?}\\\\\n\\text{V38:}&\\quad \\text{Can graded confidence improve authority? Yes functionally; specific PBF mechanism not established.}\\\\\n\\textbf{V39:}&\\quad \\mathbf{\\text{When does an experience deserve scarce long-term memory at all?}}\n\\end{aligned}\n\\]\n\nThe V38 state audit gives a direct empirical reason to ask this now: nearly four-fifths of full slots are nontarget singletons.\n\nIf V39 succeeds, the next capacity question becomes much harder but also much cleaner: **what to forget when all remaining memories have already earned consolidation?**\n\nThat is the point at which eviction should become the research problem.\n\n---\n\n# 20. Provenance and locked inputs\n\nV39 design is based on the completed V37/V38 program state and, specifically, the following V38 artifacts:\n\n- `SUMMARY.md` / `REPORT.json` — V38 mechanism verdict and qualified/failing gates;\n- `STATE_COSTS.json` — unbounded slot/member/payload/operation growth;\n- `ADDRESSING.json` — target vs nontarget slots and multi-context counts;\n- `PROTOCOL.md` — locked V38 substrate and causal restrictions;\n- `elm_v38_predictive_evidence.py` — exact PRED-V allocation/member/value representation audited for state redundancy.\n\nKey V38 facts frozen into this design:\n\n- PBF is an engineering-positive but mechanism-unqualified result because Q10 failed;\n- CORR2 remains the strongest clean deployable authority reference;\n- mean after-D PRED-V slot count is 1460.5;\n- mean target slots are 63.75;\n- mean nontarget singleton slots are 1164.0 (~79.7% of all slots);\n- mean incremental episodic payload is ~7.16 MB;\n- storage remains unbounded in V38.\n\nNo V39 threshold may be changed after V39 scientific outcomes are observed.\n"

def dependencies(install=False):
    global np, student_t
    try:
        np = importlib.import_module('numpy')
        student_t = importlib.import_module('scipy.stats').t
    except ImportError:
        if not install:
            raise SystemExit('NumPy and SciPy are required. Rerun with --install-deps.')
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'numpy', 'scipy'], check=True)
        np = importlib.import_module('numpy')
        student_t = importlib.import_module('scipy.stats').t

def encode_json(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False)

def sha(x):
    return hashlib.sha256(x if isinstance(x, bytes) else encode_json(x).encode()).hexdigest()

def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    with tmp.open('w') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def write_json(path, obj):
    atomic(path, json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + '\n')

def mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else None

def stat(values):
    """Seeds are the sampling units. Missing values prevent interval claims."""
    vals = [None if x is None else float(x) for x in values]
    good = [x for x in vals if x is not None and math.isfinite(x)]
    out = {'values': vals, 'n': len(good), 'expected_n': len(vals),
           'mean': mean(good), 'low': None, 'high': None}
    if len(good) >= 2 and len(good) == len(vals):
        half = float(student_t.ppf(.975, len(good)-1) * np.std(good, ddof=1) / math.sqrt(len(good)))
        out.update(low=out['mean']-half, high=out['mean']+half)
    return out

def softmax(scores):
    logits = scores / TEMP
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / exp.sum()

def loss(scores, target):
    logits = scores / TEMP
    maximum = float(np.max(logits))
    return (maximum + math.log(float(np.exp(logits-maximum).sum())) - float(logits[target])) / math.log(2)

class Encoder:
    """Only past raw bytes are accepted. No state is fitted or reset by metadata."""
    def __init__(self, seed, kind='last_fd'):
        if kind not in ('last_fd', 'v31', 'lag'):
            raise ValueError(kind)
        self.kind = kind
        dims = 24 if kind == 'v31' else 32
        rr = np.random.default_rng(int(seed) + 710003)
        self.embedding = rr.normal(size=(256, dims))
        self.embedding /= np.linalg.norm(self.embedding, axis=1, keepdims=True)
        # Exact V31 RNG order for its reference. Other encoders share these centres.
        ref = np.random.default_rng(int(seed) + 710003)
        ref.normal(size=(256, 24))
        self.centres = ref.normal(size=(8, 96))
        self.centres /= np.linalg.norm(self.centres, axis=1, keepdims=True)
        self.fast = {}
        self.slow = {}
        for n in range(1, 13):
            ages = np.arange(n-1, -1, -1, dtype=float)
            for table, tau in ((self.fast, 2.), (self.slow, 6.)):
                weights = np.exp(-ages/tau)
                table[n] = weights / weights.sum()
        self.bit_code = 2.*np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).astype(float)-1.

    def encode(self, history):
        h = bytes(history)[-12:]
        if not h:
            return np.zeros(D), 0
        ints = np.frombuffer(h, dtype=np.uint8)
        if self.kind == 'lag':
            raw = np.zeros(D)
            raw[-8*len(h):] = self.bit_code[ints].ravel()
        else:
            e = self.embedding[ints]
            fast = self.fast[len(h)] @ e
            slow = self.slow[len(h)] @ e
            if self.kind == 'v31':
                raw = np.concatenate((e[-1], fast, slow, 2*fast-slow))
                active = np.argpartition(np.abs(raw), -24)[-24:]
                z = np.zeros(D)
                z[active] = raw[active]
                raw = z
            else:
                raw = np.concatenate((e[-1], fast, fast-slow))
        norm = float(np.linalg.norm(raw))
        z = raw/norm if norm else raw
        return z, int(np.argmax(self.centres @ z))

def direction(z, inverse):
    zz = float(z @ z)
    if zz <= 1e-15:
        return np.zeros_like(z), 0., 0., 0., False
    q0 = z/zz
    if inverse is None:
        return q0, 1., 1., zz, False
    pz = inverse @ z
    zpz = float(z @ pz)
    if not math.isfinite(zpz) or zpz <= 0:
        raise FloatingPointError('Covariance is not positive definite on the current code.')
    q = pz/zpz
    raw_ratio = float(np.linalg.norm(q) * math.sqrt(zz))
    perpendicular = q-q0
    perpendicular -= z*(float(z @ perpendicular)/zz)
    length = float(np.linalg.norm(perpendicular))
    limit = math.sqrt((1/RATE**2-1)/zz)
    capped = length > limit
    if capped:
        perpendicular *= limit/length
    q = q0+perpendicular
    return q, float(np.linalg.norm(q)*math.sqrt(zz)), raw_ratio, zpz, capped

class Learner:
    def __init__(self, writer):
        if writer not in ('ordinary', 'full', 'full_h2048'):
            raise ValueError(writer)
        self.writer = writer
        self.weights = np.zeros((BANKS, D, OUTPUTS))
        self.inverse = None if writer == 'ordinary' else np.repeat(np.eye(D)[None], BANKS, axis=0)
        self.decay = 2.**(-1/2048) if writer == 'full_h2048' else 1.
        self.writes = np.zeros(BANKS, dtype=np.int64)
        self.max_correction_error = 0.

    def scores(self, z, bank):
        return z @ self.weights[bank]

    def predict(self, z, bank):
        return softmax(self.scores(z, bank))

    def observe(self, z, bank, actual_byte):
        scores = self.scores(z, bank)
        if not np.isfinite(scores).all():
            raise FloatingPointError('Nonfinite prediction.')
        pre_loss = loss(scores, actual_byte)
        correct = int(np.argmax(scores) == actual_byte)
        residual = -scores
        residual[actual_byte] += 1.
        inv = None if self.inverse is None else self.inverse[bank]
        q, ratio, raw_ratio, zpz, cap = direction(z, inv)
        self.weights[bank] += RATE * np.outer(q, residual)
        if float(z@z) > 1e-15:
            error = abs(float(z@q)-1.) * RATE * float(np.max(np.abs(residual)))
            self.max_correction_error = max(self.max_correction_error, error)
            if error > 1e-9:
                raise AssertionError('Write failed current-code correction.')
        if inv is not None:
            pz = inv @ z
            inv -= np.outer(pz, pz)/(self.decay+float(z@pz))
            inv /= self.decay
        self.writes[bank] += 1
        return pre_loss, correct, {'q_ratio': ratio, 'q_ratio_uncapped': raw_ratio,
                                  'zpz': zpz, 'cap': float(cap)}

    def geometry(self):
        if self.inverse is None:
            return None
        rows = []
        for b, p in enumerate(self.inverse):
            eigp = np.linalg.eigvalsh((p+p.T)/2)
            if eigp.min() <= 0 or not np.isfinite(eigp).all():
                raise FloatingPointError('Nonpositive covariance spectrum.')
            eigc = 1/eigp
            ridge = self.decay**int(self.writes[b])
            data_eigs = np.maximum(eigc-ridge, 0.)
            def rank(eigs):
                if eigs.sum() <= 1e-10:
                    return 0.
                probs = eigs/eigs.sum()
                probs = probs[probs > 0]
                return float(np.exp(-np.sum(probs*np.log(probs))))
            rows.append({'bank': b, 'writes': int(self.writes[b]), 'trace_P': float(eigp.sum()),
                         'effective_rank_C': rank(eigc), 'effective_rank_data': rank(data_eigs),
                         'min_eigen_P': float(eigp.min()), 'max_eigen_C': float(eigc.max()),
                         'condition_C': float(eigp.max()/eigp.min()), 'remaining_ridge': ridge})
        total = int(self.writes.sum())
        return {'banks': rows, 'weighted_data_effective_rank':
                sum(r['effective_rank_data']*r['writes'] for r in rows)/max(total, 1),
                'trace_P': sum(r['trace_P'] for r in rows)}

    def state_hash(self):
        return sha(self.weights.tobytes() + (b'' if self.inverse is None else self.inverse.tobytes()) + self.writes.tobytes())

    def costs(self, encoder):
        w = self.weights.nbytes
        p = 0 if self.inverse is None else self.inverse.nbytes
        return {'output_weights_bytes': w, 'protection_bytes': p,
                'protection_over_output_ratio': p/w,
                'fixed_receptors_bytes': encoder.embedding.nbytes if encoder.kind != 'lag' else 0,
                'routing_centres_bytes': encoder.centres.nbytes,
                'write_counter_bytes': self.writes.nbytes,
                'history_bytes': 12, 'asymptotic_write_cost': 'O(96*256+96^2)' if p else 'O(96*256)',
                'evaluator_and_temporary_arrays_excluded': True}

def context(key, alias):
    return PREFIX+HEADERS[alias]+key+SEPARATORS[alias]

def make_stream(seed, preflight=False, smoke=False):
    rr = np.random.default_rng(seed+810007)
    count, repeats = (4, 4) if smoke else (16, 32)
    waves = 1 if preflight else 4
    key_alphabet = np.frombuffer(b'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', dtype=np.uint8)
    keys = []
    while len(keys) < count*waves:
        key = bytes(rr.choice(key_alphabet, size=6))
        if key not in keys:
            keys.append(key)
    labels = rr.choice(np.frombuffer(b'abcdefghijklmnopqrstuvwx', dtype=np.uint8), size=len(keys)).astype(int)
    nrev = 1 if smoke else 4
    revised = list(range(nrev))
    noisy = list(range(nrev, 2*nrev))
    untouched = list(range(2*nrev, count))
    changed = {i: int(ord('a')+(int(labels[i])-ord('a')+1+int(rr.integers(23)))%24) for i in revised}
    rows = []
    checkpoints = []
    render_counts = collections.Counter()
    def add(i, target, kind, exposure, wave, alias):
        render_counts[(i, alias, kind)] += 1
        rows.append({'key_id': int(i), 'target': int(target), 'kind': kind, 'exposure': exposure,
                     'wave': wave, 'alias': alias,
                     'rendering_exposure': render_counts[(i, alias, kind)],
                     'raw': context(keys[i], alias)+bytes([target])+b'\n'})
    for wave in range(waves):
        cohort = chr(65+wave)
        for exposure in range(1, repeats+1):
            for i in rr.permutation(np.arange(wave*count, (wave+1)*count)):
                add(int(i), int(labels[i]), 'acquisition', exposure, cohort, (exposure-1)%2)
            if exposure in (1,2,4,8,16,32):
                checkpoints.append({'record_count': len(rows), 'name': f'{cohort}_acq_{exposure}',
                                    'type': 'acquisition', 'wave': cohort, 'exposure': exposure})
        checkpoints.append({'record_count': len(rows), 'name': f'after_{cohort}', 'type': 'wave', 'wave': cohort})
        if wave == 1:
            for i in noisy:
                wrong = ord('a')+(int(labels[i])-ord('a')+1)%24
                add(i, wrong, 'one_off_noise', 1, cohort, 0)
            checkpoints.append({'record_count': len(rows), 'name': 'after_noise', 'type': 'noise', 'wave': cohort})
        if wave == 2:
            for exposure in range(1, 9):
                for i in rr.permutation(revised):
                    add(int(i), changed[int(i)], 'revision', exposure, cohort, 0)
                if exposure in (1,2,4,8):
                    checkpoints.append({'record_count': len(rows), 'name': f'revision_{exposure}',
                                        'type': 'revision', 'wave': cohort, 'exposure': exposure})
    raw = b''.join(row['raw'] for row in rows)
    assert all(len(row['raw']) == 14 for row in rows)
    assert len(set(keys)) == len(keys)
    assert all(b'?' not in row['raw'] and b'>' not in row['raw'] for row in rows)
    return {'seed': seed, 'keys': keys, 'labels': labels, 'revised': revised, 'noisy': noisy,
            'untouched': untouched, 'revisions': changed, 'rows': rows, 'raw': raw,
            'checkpoints': checkpoints, 'keys_per_wave': count, 'repeats': repeats,
            'raw_sha256': sha(raw)}

class RunLock:
    def __init__(self, root):
        self.path = root/'.run.lock'
        self.handle = None
    def __enter__(self):
        import fcntl
        self.handle = self.path.open('a+')
        try:
            fcntl.flock(self.handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            raise RuntimeError('Another process is writing this run directory.')
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(str(os.getpid()))
        self.handle.flush()
        return self
    def __exit__(self,*args):
        import fcntl
        fcntl.flock(self.handle.fileno(),fcntl.LOCK_UN)
        self.handle.close()

def csv_output(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=keys);writer.writeheader()
    for r in rows:writer.writerow({k:encode_json(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    atomic(path,out.getvalue())

class ByteModel:
    """Causal interface only; membership has no evaluator key/alias metadata."""
    def __init__(self,seed,arm):
        assert arm in MAIN_ARMS+FLY_ARMS
        self.arm=arm;self.stage='II';self.encoder=Encoder(seed,'last_fd')
        self.address_encoder=Encoder(seed,'lag');self.base=Learner('full');self.frozen=False
        self.eligible=set();self.members={};self.member_vectors={};self.slot_members=[]
        self.values=[];self.local={};self.sums=[];self.prototypes=np.zeros((128,D));self.drift=[]
        self.bind_events=[];self.aux_updates=0;self.locality_checks=0;self.ops=0;self.shared=None
        self.sparse=SparseCode(seed) if arm in FLY_ARMS else None
        self.output=np.zeros((4096,OUTPUTS)) if self.sparse else None
    def base_scores(self,h):
        z,b=self.encoder.encode(h);return self.base.scores(z,b)
    def lookup(self,c):
        c=bytes(c)[-12:]
        if c in self.members:return self.members[c],1.,None
        if self.arm not in ('SIM-R','SIM-V','PRED-V') or not self.values:return None,None,None
        sims=self.prototypes[:len(self.values)]@self.address_encoder.encode(c)[0]
        j=int(np.argmax(sims));s=float(sims[j]);second=float(np.partition(sims,-2)[-2]) if len(sims)>1 else None
        return (j if s>=TAU else None),s,None if second is None else s-second
    def scores(self,h):
        c=bytes(h)[-12:]
        if self.sparse:
            ix,v=self.sparse.encode(c,self.arm=='HASH-O');return v@self.output[ix]
        sb=self.base_scores(c);j,_,_=self.lookup(c)
        if j is None:return sb
        v=np.frombuffer(self.values[j],dtype=np.float64)
        return sb+v if self.arm.endswith('-R') else v
    def probabilities(self,h):return softmax(self.scores(h))
    def bind(self,c,j):
        assert c not in self.members and c in self.eligible
        k=self.address_encoder.encode(c)[0]
        new=j is None
        if new:
            j=len(self.values);self.values.append(np.zeros(OUTPUTS).tobytes())
            self.slot_members.append(set());self.sums.append(np.zeros(D))
            if j==len(self.prototypes):self.prototypes=np.vstack((self.prototypes,np.zeros_like(self.prototypes)))
        old=self.prototypes[j].copy()
        self.members[c]=j;self.member_vectors[c]=k.copy();self.slot_members[j].add(c)
        self.sums[j]=self.sums[j]+k
        n=np.linalg.norm(self.sums[j]);self.prototypes[j]=self.sums[j]/n if n else 0.
        self.drift.append(float(np.linalg.norm(self.prototypes[j]-old)) if not new else 0.)
        self.local[c]=self.values[j]
        self.bind_events.append((c,j,new))
        return j
    def observe(self,h,y):
        c=bytes(h)[-12:];z,b=self.encoder.encode(c);sb=self.base.scores(z,b)
        j,sim,margin=self.lookup(c);used=self.scores(c).copy()
        existed=c in self.eligible;bound=c in self.members
        pre_loss=loss(used,y);correct=int(np.argmax(used)==y)
        base_loss=loss(sb,y);base_correct=int(np.argmax(sb)==y)
        # Pre-update candidate confirmation is retained before any writer changes state.
        confirmed=j is not None and int(np.argmax(np.frombuffer(self.values[j],dtype=np.float64)))==y
        if not self.frozen:self.base.observe(z,b,y)
        if not base_correct:self.eligible.add(c)
        change=0.;local_norm=0.;touched=0
        if self.sparse:
            ix,v=self.sparse.encode(c,self.arm=='HASH-O');vv=float(v@v)
            if vv>1e-15:
                error=-used;error[y]+=1.;delta=RATE*np.outer(v/vv,error)
                self.output[ix]+=delta;change=float(np.linalg.norm(delta))
            self.aux_updates+=1;self.ops+=205*OUTPUTS
        elif c in self.eligible:
            if not bound:
                candidate=j
                if self.arm=='PRED-V' and not confirmed:candidate=None
                j=self.bind(c,candidate)
            old=np.frombuffer(self.values[j],dtype=np.float64)
            if self.arm.endswith('-R'):
                residual=-used;residual[y]+=1.;value=old+RATE*residual
            else:
                value=(1-RATE)*old;value[y]+=RATE
            if not np.isfinite(value).all():raise FloatingPointError('Nonfinite local value')
            self.values[j]=value.tobytes()
            for member in self.slot_members[j]:self.local[member]=self.values[j]
            touched=1;self.aux_updates+=1;change=float(np.linalg.norm(value-old));local_norm=float(np.linalg.norm(value))
        self.ops+=D*OUTPUTS+D*D+len(self.values)*D+OUTPUTS*touched
        return pre_loss,correct,{'base_bpb':base_loss,'base_accuracy':float(base_correct),
            'eligible_before':float(existed),'eligible_after':float(c in self.eligible),
            'newly_eligible':float(not existed and c in self.eligible),'local_vectors_touched':float(touched),
            'aux_change_norm':change,'aux_value_norm':local_norm,'tentative_candidate':float(not bound and j is not None),
            'candidate_confirmed':float(confirmed)}
    def eligibility_hash(self):return sha([c.hex() for c in sorted(self.eligible)])
    def state_hash(self):
        data=self.base.state_hash().encode()+self.eligibility_hash().encode()+bytes([self.frozen])
        data+=b''.join(len(c).to_bytes(1,'little')+c+j.to_bytes(8,'little')+self.member_vectors[c].tobytes() for c,j in sorted(self.members.items()))
        data+=b''.join(self.values)+self.prototypes.tobytes()+b''.join(s.tobytes() for s in self.sums)
        if self.output is not None:data+=self.output.tobytes()
        return sha(data)
    def costs(self):
        fixed=sum(a.nbytes for enc in (self.encoder,self.address_encoder) for a in [enc.embedding,enc.centres,enc.bit_code,*enc.fast.values(),*enc.slow.values()])
        if self.sparse:fixed+=self.sparse.contacts.nbytes+self.sparse.weights.nbytes
        base=self.base.weights.nbytes+self.base.inverse.nbytes+self.base.writes.nbytes
        values=sum(len(v) for v in self.values);address=sum(len(c)+8+v.nbytes for c,v in self.member_vectors.items())+self.prototypes.nbytes+sum(s.nbytes for s in self.sums)
        sizes=[len(s) for s in self.slot_members]
        overhead=sys.getsizeof(self.members)+sys.getsizeof(self.local)+sys.getsizeof(self.eligible)+sum(sys.getsizeof(s) for s in self.slot_members)
        return {'shared_base_bytes':base,'local_value_bytes':values,'slot_address_payload_bytes':address,
            'exact_member_contexts':len(self.members),'slots':len(self.values),'mean_members':mean(sizes),'max_members':max(sizes,default=0),
            'fixed_representation_bytes':fixed,'output_coefficient_bytes':0 if self.output is None else self.output.nbytes,
            'total_payload_bytes':base+values+address+fixed+(0 if self.output is None else self.output.nbytes),
            'python_container_overhead_estimate':overhead,'operation_proxy_multiply_add_units':self.ops,
            'bounded':bool(self.sparse),'cost_notes':'Payloads counted exactly; Python overhead approximate. Evaluator logs excluded; exact eligibility grows even for fly counterfactual accounting.'}

IMPLEMENTATION_NOTES = '''
## Locked implementation choices (before science)
- Ties: slot before trace, then oldest allocation index. Similarity uses float64.
- Slot access counts start at 1; promotion starts at source exact-hit count + 1.
  One access is counted on an exact hit, authoritative slot read, or confirmed write;
  at most one increment per event. Lookup/probes never touch counters.
- TRACE-EXACT2 promotes on the second exact presentation, including admission.
  Its single unique anchor has CORR2 support 1. A later distinct confirmed context
  can fill anchor 2 (as ANCHOR2-U); subsequent identity state is frozen.
  Similar traces never cause promotion in TRACE-EXACT2.
- SHUFFLED-PAIR follows TRACE-CORR online after all predictions are frozen. Source
  matching uses exact hit count, then nearest age quartile (rank of insertion age
  among resident traces); ties prefer the lower quartile. Empty hit class falls
  back to the entire eligible trace pool. Current context is excluded as source.
  If the current context is already an anchor, its owning slot is removed first;
  if it is a trace, that trace is removed. These extra removals are reported.
  No resident source means a missed promotion, never a fabricated trace. Any
  schedule mismatch makes Q11 invalid (fails qualification).
- Stress tail: 768 singleton and 128 repeated new uppercase/digit six-byte keys,
  alias-0 framing. First pass shuffles 896 rows; second pass shuffles 128 repeats.
  No optional fillers: exactly 1024 rows / 14336 bytes. Smoke scales to 24+8*2.
- Whole/nontarget bpb gates include the tail; pre-tail bpb is separately reported.
  Q6, Q8 and Q10 recurrence margins must hold on each paired seed. Other level
  gates use means and the explicitly specified per-seed floors.
- Q10/Q11 held-out endpoint is revision@4 alias 2. Q4/Q9 use all 64 current
  objects, both learned aliases, after clutter; alias-2 accuracy is separate.
- Q12 denominator: first subsequent exact acquisition presentation for every
  target context admitted as a trace (including intervening eviction/promotion).
  Q13 uses first exposure of E2, not later successful consolidation; includes
  failures to confirm or consolidate. Immediate E2 existence is separately gated.
- Completed seed checkpoints are atomic/checksummed. Interrupted seeds restart;
  no pickle or hidden live-state checkpoint is required. Source, protocol, plan,
  stream hashes and engineering P0 evidence are locked. No scientific results
  are used for implementation decisions.
'''

class Memory:
    """Causal resident state only. Inputs: raw context, frozen base scores, byte.

    No evaluator fields and no permanent cached per-context address vectors.
    Slot/trace ids are monotonically increasing scalar counters, not tombstones.
    """
    def __init__(self, policy, seed, trace_cap=512, slot_cap=384):
        self.policy=policy; self.enc=Encoder(seed,'lag')
        self.trace_cap=trace_cap; self.slot_cap=slot_cap
        self.traces={}; self.slots={}; self.anchors={}; self.serial=0; self.clock=0
        self.rng=np.random.default_rng(seed+390009) if policy=='SHUFFLED-PAIR' else None
        self.ops=collections.Counter(); self.peak={'traces':0,'slots':0,'anchors':0,'semantic_bytes':0}
        self.two_tier=policy in ('TRACE-CORR','TRACE-EXACT2','SHUFFLED-PAIR')
        self.bounded=policy not in ('LEGACY-U','ANCHOR2-U')

    def address(self,c): return self.enc.encode(c)[0]

    def trace_matrix(self):
        # Temporary vectorized reconstruction, never stored between events.
        cs=list(self.traces)
        if not cs:return np.empty((0,D))
        padded=b''.join(c.rjust(12,b'\0') for c in cs)
        a=self.enc.bit_code[np.frombuffer(padded,dtype=np.uint8)].reshape(len(cs),D).copy()
        for i,c in enumerate(cs):
            if len(c)<12:a[i,:D-8*len(c)]=0.
        norms=np.linalg.norm(a,axis=1,keepdims=True)
        return np.divide(a,norms,out=np.zeros_like(a),where=norms!=0)

    def lookup(self,context_bytes,base_scores):
        c=bytes(context_bytes)[-12:]; exact=False; typ=None; ident=None; sim=None
        if c in self.anchors:typ='slot';ident=self.anchors[c];exact=True;sim=1.
        elif c in self.traces:typ='trace';ident=c;exact=True;sim=1.
        else:
            z=self.address(c); best=-float('inf')
            if self.slots:
                ids=list(self.slots); sums=np.array([self.slots[j]['sum'] for j in ids])
                norms=np.linalg.norm(sums,axis=1,keepdims=True)
                prot=np.divide(sums,norms,out=np.zeros_like(sums),where=norms!=0)
                sims=prot@z;k=int(np.argmax(sims));best=float(sims[k]);typ='slot';ident=ids[k]
            if self.two_tier and self.traces:
                sims=self.trace_matrix()@z;k=int(np.argmax(sims));s=float(sims[k])
                if s>best:best=s;typ='trace';ident=list(self.traces)[k]
            sim=None if best==-float('inf') else best
            if sim is None or sim<TAU:typ=None;ident=None
        item=self.slots[ident] if typ=='slot' else self.traces[ident] if typ=='trace' else None
        support=len(item['anchors']) if typ=='slot' else 1 if typ else 0
        authority=exact or (typ=='slot' and support>=2)
        scores=item['value'] if authority else base_scores
        return {'c':c,'type':typ,'id':ident,'exact':exact,'sim':sim,'support':support,
                'candidate_top':int(np.argmax(item['value'])) if item is not None else None,
                'base_top':int(np.argmax(base_scores)),'scores':scores,'authority':authority}

    def _id(self):
        n=self.serial;self.serial+=1;return n

    def _drop_slot(self,j,events,reason='capacity'):
        s=self.slots.pop(j)
        for c in s['anchors']:del self.anchors[c]
        events.append({'event':'slot_eviction','contexts':[c.hex() for c in s['anchors']],
                       'reason':reason,'age':self.clock-s['created'],'hits':s['hits']})

    def _drop_trace(self,c,events,reason='capacity'):
        t=self.traces.pop(c)
        events.append({'event':'trace_removal','context':c.hex(),'reason':reason,
                       'age':self.clock-t['created'],'hits':t['hits']})
        return t

    def _new_slot(self,cs,value,events,hits=1):
        if self.bounded and len(self.slots)>=self.slot_cap:
            j=min(self.slots,key=lambda j:(self.slots[j]['hits'],self.slots[j]['last'],j))
            self._drop_slot(j,events)
        j=self._id();acc=np.zeros(D)
        for c in cs:
            assert c not in self.anchors and c not in self.traces
            acc=acc+self.address(c); self.anchors[c]=j
        self.slots[j]={'anchors':list(cs),'sum':acc,'value':value.copy(),'hits':hits,
                       'last':self.clock,'created':self.clock}
        events.append({'event':'slot_allocation','contexts':[c.hex() for c in cs]})
        return j

    def _new_trace(self,c,y,events):
        if len(self.traces)>=self.trace_cap:
            victim=min(self.traces,key=lambda c:(self.traces[c]['last'],self.traces[c]['id']))
            self._drop_trace(victim,events)
        value=np.zeros(OUTPUTS);value[y]=RATE
        self.traces[c]={'value':value,'hits':1,'last':self.clock,'created':self.clock,'id':self._id()}
        events.append({'event':'trace_admission','context':c.hex()})

    @staticmethod
    def update(item,y):
        v=(1-RATE)*item['value'];v[y]+=RATE;item['value']=v

    def promote(self,c,source,y,events,recurrence=False):
        t=self._drop_trace(source,events,'promotion')
        cs=[source] if recurrence else [source,c]
        j=self._new_slot(cs,t['value'],events,t['hits']+1)
        self.update(self.slots[j],y)
        events.append({'event':'promotion','contexts':[x.hex() for x in cs],
                       'source_hits':t['hits'],'source_age':self.clock-t['created']})

    def shuffled_promote(self,p,y,schedule,events):
        c=p['c'];pool=[x for x in self.traces if x!=c]
        if not pool:
            events.append({'event':'missed_schedule'});return False
        matching=[x for x in pool if self.traces[x]['hits']==schedule['source_hits']]
        if matching:
            by_age=sorted(pool,key=lambda x:(self.clock-self.traces[x]['created'],self.traces[x]['id']))
            quart={x:min(3,4*i//len(by_age)) for i,x in enumerate(by_age)}
            dist=min(abs(quart[x]-schedule['source_quartile']) for x in matching)
            q=min(quart[x] for x in matching if abs(quart[x]-schedule['source_quartile'])==dist)
            choices=[x for x in matching if quart[x]==q]
        else:choices=pool;events.append({'event':'shuffle_stratum_fallback'})
        source=choices[int(self.rng.integers(len(choices)))]
        if c in self.anchors:self._drop_slot(self.anchors[c],events,'shuffle_current_collision')
        if c in self.traces:self._drop_trace(c,events,'shuffle_current_collision')
        self.promote(c,source,y,events)
        events.append({'event':'schedule_match','changed_pair':source.hex()!=schedule['contexts'][0]})
        return True

    def observe(self,p,y,schedule=None):
        self.clock+=1;events=[];c=p['c'];typ=p['type'];j=p['id']
        self.ops['slot_similarity']+=0 if p['exact'] else len(self.slots)*D
        self.ops['trace_similarity']+=0 if p['exact'] else len(self.traces)*D
        self.ops['trace_address_reconstruction']+=0 if p['exact'] else len(self.traces)*D
        self.ops['value_scoring']+=OUTPUTS
        if self.policy=='SHUFFLED-PAIR' and schedule is not None:
            if self.shuffled_promote(p,y,schedule,events):
                self.check();return events
        if p['exact']:
            item=self.slots[j] if typ=='slot' else self.traces[j]
            if typ=='trace' and self.policy=='TRACE-EXACT2' and item['hits']>=1:
                self.promote(c,c,y,events,recurrence=True)
            else:
                self.update(item,y);item['hits']+=1;item['last']=self.clock
        else:
            # Count authoritative uses, even when the base explains the byte.
            if typ=='slot' and p['authority']:
                self.slots[j]['hits']+=1;self.slots[j]['last']=self.clock
            if p['base_top']!=y:
                confirmed=p['candidate_top']==y
                if typ=='slot' and confirmed:
                    s=self.slots[j]
                    if self.policy=='LEGACY-U' or len(s['anchors'])<2:
                        s['anchors'].append(c);self.anchors[c]=j;s['sum']=s['sum']+self.address(c)
                        events.append({'event':'anchor_binding','contexts':[x.hex() for x in s['anchors']]})
                    self.update(s,y)
                    if not p['authority']:s['hits']+=1;s['last']=self.clock
                elif typ=='trace' and confirmed and self.policy=='TRACE-CORR':
                    by_age=sorted(self.traces,key=lambda x:(self.clock-self.traces[x]['created'],self.traces[x]['id']))
                    quart=min(3,4*by_age.index(j)//len(by_age))
                    self.promote(c,j,y,events)
                    events[-1]['source_quartile']=quart
                elif self.two_tier:self._new_trace(c,y,events)
                else:
                    v=np.zeros(OUTPUTS);v[y]=RATE;self._new_slot([c],v,events)
        self.ops['allocation_eviction']+=sum(e['event'] in ('slot_allocation','slot_eviction','trace_admission','trace_removal') for e in events)
        self.check();return events

    def cost(self,overhead=False):
        nt=len(self.traces);ns=len(self.slots);na=len(self.anchors)
        # Conservative fixed-width serialized fields, plus explicit index entries.
        # Trace has 4 uint64 fields (including id), rather than the design's 3;
        # charged honestly and still beneath the preregistered overall cap.
        payload=16+nt*(12+2048+32)+ns*(2048+768+32)+sum(len(c) for c in self.anchors)+8*(nt+na)
        r={'traces':nt,'slots':ns,'anchors':na,'semantic_bytes':payload,
           'operation_proxy':dict(self.ops),'bounded':self.bounded}
        if overhead:
            seen=set()
            def size(x):
                if id(x) in seen:return 0
                seen.add(id(x));v=sys.getsizeof(x)
                if isinstance(x,dict):v+=sum(size(k)+size(y) for k,y in x.items())
                elif isinstance(x,(list,tuple)):v+=sum(size(y) for y in x)
                return v
            r['python_resident_bytes_estimate']=size((self.traces,self.slots,self.anchors))
        return r

    def check(self):
        if self.bounded:
            assert len(self.traces)<=self.trace_cap and len(self.slots)<=self.slot_cap
            assert len(self.anchors)<=2*self.slot_cap
        # Cardinality equality plus mutation checks avoid an expensive global per-byte scan.
        assert not (self.policy in ('FULL-CAP','ANCHOR2-U','LEGACY-U') and self.traces)
        cost=self.cost()
        for k in self.peak:self.peak[k]=max(self.peak[k],cost[k])

    def semantic(self):
        return {'slots':[{'id':j,'anchors':[c.hex() for c in s['anchors']],
                           'sum':s['sum'].tobytes().hex(),'value':s['value'].tobytes().hex()}
                          for j,s in self.slots.items()]}

    def state_hash(self):
        return sha({'semantic':self.semantic(),'traces':[(c.hex(),t['value'].tobytes().hex(),t['hits'],t['last'],t['created'],t['id']) for c,t in self.traces.items()],
                    'metadata':[(j,s['hits'],s['last'],s['created']) for j,s in self.slots.items()],
                    'serial':self.serial,'clock':self.clock,'rng':None if self.rng is None else self.rng.bit_generator.state})


def legacy_semantic(m):
    return {'slots':[{'id':j,'anchors':[c.hex() for c in m.members if m.members[c]==j],
                       'sum':m.sums[j].tobytes().hex(),'value':m.values[j].hex()} for j in range(len(m.values))]}


def compaction_preflight(seed,smoke=False):
    legacy=ByteModel(seed,'PRED-V');compact=Memory('LEGACY-U',seed)
    enc=Encoder(seed,'last_fd');base=Learner('full');h=bytearray();stream=make_stream(seed,smoke=smoke)
    max_sim=0.;digest=hashlib.sha256();probe_checks=0
    checkpoint_bytes={cp['record_count']*14 for cp in stream['checkpoints']}
    for n,y in enumerate(stream['raw']):
        c=bytes(h);z,b=enc.encode(c);sb=base.scores(z,b)
        assert np.array_equal(sb,legacy.base_scores(c))
        p=compact.lookup(c,sb);j,sim,_=legacy.lookup(c)
        assert (p['id'] if p['type'] else None)==j
        if j is not None:
            err=abs(p['sim']-sim);max_sim=max(max_sim,err);assert err<1e-12
        exact=c in legacy.members
        old=np.frombuffer(legacy.values[j],dtype=np.float64) if j is not None else None
        ref=old if exact or (j is not None and len(legacy.slot_members[j])>=2) else sb
        assert np.array_equal(p['scores'],ref)
        assert p['candidate_top']==(None if old is None else int(np.argmax(old)))
        digest.update(p['scores'].tobytes())
        compact.observe(p,y);legacy.observe(c,y);base.observe(z,b,y)
        if n%2048==0 or n==len(stream['raw'])-1:
            assert compact.semantic()==legacy_semantic(legacy)
            assert base.state_hash()==legacy.base.state_hash()
            assert set(compact.anchors)==legacy.eligible
        if n+1 in checkpoint_bytes:
            before=compact.state_hash()
            for key in stream['keys']:
                for alias in (0,1,2):
                    pc=context(key,alias);psb=legacy.base_scores(pc)
                    pp=compact.lookup(pc,psb);pj,_,_=legacy.lookup(pc)
                    pv=np.frombuffer(legacy.values[pj],dtype=np.float64) if pj is not None else None
                    expected=pv if pc in legacy.members or (pj is not None and len(legacy.slot_members[pj])>=2) else psb
                    assert np.array_equal(pp['scores'],expected)
                    probe_checks+=1
            assert before==compact.state_hash()
        h.append(y)
        if len(h)>12:del h[0]
    return {'seed':seed,'full_length':not smoke,'events':len(stream['raw']),'passed':True,
            'max_similarity_error':max_sim,'semantic_sha256':sha(compact.semantic()),
            'prediction_sha256':digest.hexdigest(),'base_sha256':base.state_hash(),
            'stream_sha256':stream['raw_sha256'],'reference_source_sha256':REFERENCE_SHA256,
            'read_only_checkpoint_alias_probes':probe_checks}


def workload(seed,smoke=False):
    s=make_stream(seed,smoke=smoke);s['standard_rows']=len(s['rows']);s['standard_sha256']=s['raw_sha256']
    rr=np.random.default_rng(seed+390081);single_n,repeat_n=(24,8) if smoke else (768,128)
    used=set(s['keys']);keys=[];alphabet=np.frombuffer(b'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',dtype=np.uint8)
    while len(keys)<single_n+repeat_n:
        k=bytes(rr.choice(alphabet,size=6))
        if k not in used:used.add(k);keys.append(k)
    targets=rr.choice(np.frombuffer(b'abcdefghijklmnopqrstuvwx',dtype=np.uint8),size=len(keys)).astype(int)
    tail=[]
    for i in rr.permutation(len(keys)):
        tail.append({'raw':context(keys[i],0)+bytes([targets[i]])+b'\n','kind':'singleton' if i<single_n else 'exact_repeat',
                     'target':int(targets[i]),'tail_id':int(i),'exposure':1})
    for i in rr.permutation(np.arange(single_n,len(keys))):
        tail.append({'raw':context(keys[i],0)+bytes([targets[i]])+b'\n','kind':'exact_repeat',
                     'target':int(targets[i]),'tail_id':int(i),'exposure':2})
    s['rows']+=tail;s['raw']=b''.join(r['raw'] for r in s['rows']);s['raw_sha256']=sha(s['raw'])
    s['tail_probes']=[{'context':context(k,0),'target':int(y),'kind':'singleton' if i<single_n else 'exact_repeat'} for i,(k,y) in enumerate(zip(keys,targets))]
    s['tail_rows']=len(tail);s['tail_sha256']=sha(b''.join(r['raw'] for r in tail))
    assert len({r['raw'][:12] for r in tail})==single_n+repeat_n
    assert len(tail)==single_n+2*repeat_n
    return s

class Aggregate:
    def __init__(self):self.n=0;self.total=0.;self.correct=0
    def add(self,scores,y):self.n+=1;self.total+=loss(scores,y);self.correct+=int(np.argmax(scores)==y)
    def result(self):return {'n':self.n,'bpb':self.total/self.n if self.n else None,'accuracy':self.correct/self.n if self.n else None}

class Observer:
    """Evaluator-only, never passed into Memory. No event history is needed to learn."""
    def __init__(self,stream):
        self.labels={context(k,a).hex():i for i,k in enumerate(stream['keys']) for a in (0,1,2)}
        self.tail={r['context'].hex():r['kind'] for r in stream['tail_probes']}
        self.counts={p:collections.Counter() for p in POLICIES};self.residence={p:collections.Counter() for p in POLICIES}
        self.schedule_count=0;self.schedule_matched=0;self.changed=0;self.control_extra=0
        self.consolidated={p:set() for p in POLICIES}
    def category(self,contexts):
        if any(c in self.labels for c in contexts):return 'target'
        for kind in ('singleton','exact_repeat'):
            if any(self.tail.get(c)==kind for c in contexts):return kind
        return 'nontarget'
    def events(self,policy,events):
        for e in events:
            cs=e.get('contexts',[e.get('context','')]);cat=self.category(cs);name=e['event']
            self.counts[policy][name+':'+cat]+=1
            if name=='promotion':
                ids=[self.labels.get(c) for c in cs]
                if any(i is not None for i in ids):
                    pure=len(ids)==2 and ids[0] is not None and ids[0]==ids[1]
                    self.counts[policy]['target_pairs']+=1;self.counts[policy]['pure_target_pairs']+=int(pure)
                    self.counts[policy]['cross_key_false_promotions']+=int(len(set(i for i in ids if i is not None))>1)
                    if pure:self.consolidated[policy].add(ids[0])
                if policy=='TRACE-CORR':self.schedule_count+=1
            if name=='trace_removal':
                self.counts[policy]['trace_removal_'+e['reason']+':'+cat]+=1
                bucket=min(20,int(math.log2(max(e['age'],1))))
                self.residence[policy][str(bucket)]+=1
            if name=='slot_eviction':self.counts[policy]['slot_eviction_'+e['reason']+':'+cat]+=1
            if name=='schedule_match':self.schedule_matched+=1;self.changed+=int(e['changed_pair'])
            if e.get('reason')=='shuffle_current_collision':self.control_extra+=1


def trial39(seed,smoke=False,observers=True):
    stream=workload(seed,smoke);enc=Encoder(seed,'last_fd');base=Learner('full')
    memories={p:Memory(p,seed) for p in POLICIES};obs=Observer(stream)
    groups=collections.defaultdict(Aggregate);probes=[];emergence=[];occupancy=[];costs=[]
    h=bytearray();pending={p:{} for p in POLICIES};admitted={p:set() for p in POLICIES}
    recurrences={p:[] for p in POLICIES};checkpoints=[];evidence_hash=hashlib.sha256()
    cps=collections.defaultdict(list)
    for cp in stream['checkpoints']:cps[cp['record_count']].append(cp)

    def probe(cp,kind,ids,aliases=(0,1,2),revised=False):
        if not observers:return
        rows=[]
        for p,m in memories.items():
            for alias in aliases:
                a=Aggregate();oldp=[];targetp=[]
                for i in ids:
                    c=context(stream['keys'][i],alias);z,b=enc.encode(c);sb=base.scores(z,b)
                    packet=m.lookup(c,sb);y=stream['revisions'].get(i,int(stream['labels'][i])) if revised else int(stream['labels'][i])
                    a.add(packet['scores'],y);prob=softmax(packet['scores']);targetp.append(float(prob[y]));oldp.append(float(prob[int(stream['labels'][i])]))
                rows.append(dict(seed=seed,policy=p,checkpoint=cp,kind=kind,alias=alias,
                                 **a.result(),target_probability=mean(targetp),obsolete_probability=mean(oldp)))
        probes.extend(rows)
        for alias in aliases:
            a=Aggregate()
            for i in ids:
                z,b=enc.encode(context(stream['keys'][i],alias));y=stream['revisions'].get(i,int(stream['labels'][i])) if revised else int(stream['labels'][i]);a.add(base.scores(z,b),y)
            probes.append(dict(seed=seed,policy='BASE',checkpoint=cp,kind=kind,alias=alias,**a.result()))

    def checkpoint(name):
        if not observers:return
        for p,m in memories.items():
            cost=m.cost(overhead=True);costs.append(dict(seed=seed,policy=p,checkpoint=name,peak=dict(m.peak),**cost))
            count_target=sum(any(c.hex() in obs.labels for c in s['anchors']) for s in m.slots.values())
            obs.counts[p]['resident_target_slots_'+name]=count_target
            obs.counts[p]['resident_nontarget_slots_'+name]=len(m.slots)-count_target
            hist=collections.Counter(str(min(20,int(math.log2(max(s['hits'],1))))) for s in m.slots.values())
            costs[-1]['slot_access_log2_histogram']=dict(hist)
        checkpoints.append({'checkpoint':name,'base_hash':base.state_hash(),
                            'memory_hashes':{p:m.state_hash() for p,m in memories.items()},'probes_read_only':True})

    for number,row in enumerate(stream['rows'],1):
        for pos,y in enumerate(row['raw']):
            c=bytes(h);z,b=enc.encode(c);sb=base.scores(z,b)
            packets={p:m.lookup(c,sb) for p,m in memories.items()} # NO outcome argument
            if observers:
                category='target' if pos==12 else 'nontarget'
                for p,packet in packets.items():
                    for cat in ('whole',category,'standard_whole' if number<=stream['standard_rows'] else 'tail_whole'):
                        groups[p,cat].add(packet['scores'],y)
                for cat in ('whole',category):groups['BASE',cat].add(sb,y)
            if pos==12 and row['kind']=='acquisition' and observers:
                for p,m in memories.items():
                    if c in pending[p]:
                        admission_exposure=pending[p].pop(c)
                        recurrences[p].append({'accuracy':int(np.argmax(packets[p]['scores'])==y),
                            'admission_exposure':admission_exposure,'resident_trace':c in m.traces})
            base.observe(z,b,y) # exactly one shared base update, independent of all branches
            schedule=None
            for p in POLICIES:
                events=memories[p].observe(packets[p],y,schedule if p=='SHUFFLED-PAIR' else None)
                if p=='TRACE-CORR':schedule=next((e for e in events if e['event']=='promotion'),None)
                if observers:
                    obs.events(p,events)
                    if pos==12 and row['kind']=='acquisition':
                        for e in events:
                            if e['event']=='trace_admission' and e['context']==c.hex() and c not in admitted[p]:
                                admitted[p].add(c);pending[p][c]=row['rendering_exposure']
            if schedule is not None:evidence_hash.update(encode_json((number,pos,schedule)).encode())
            h.append(y)
            if len(h)>12:del h[0]
            if observers and pos==12 and row['kind']=='acquisition' and row['exposure'] in (1,2):
                i=row['key_id'];c0,c1=[context(stream['keys'][i],a) for a in (0,1)]
                for p,m in memories.items():
                    before=m.state_hash();z0,b0=enc.encode(c0);z2,b2=enc.encode(context(stream['keys'][i],2))
                    ex=m.lookup(c0,base.scores(z0,b0));held=m.lookup(context(stream['keys'][i],2),base.scores(z2,b2))
                    j0,j1=m.anchors.get(c0),m.anchors.get(c1)
                    emergence.append(dict(seed=seed,policy=p,key_id=i,phase='E1' if row['exposure']==1 else 'E2',
                        exact_accuracy=int(np.argmax(ex['scores'])==y),heldout_accuracy=int(np.argmax(held['scores'])==y),
                        consolidated_pair=j0 is not None and j0==j1,heldout_authority=held['authority'],
                        immediate_trace=c0 in m.traces,byte_count=(number-1)*14+pos+1))
                    assert before==m.state_hash()
        if observers and (number%128==0 or number==len(stream['rows'])):
            for p,m in memories.items():occupancy.append(dict(seed=seed,policy=p,row=number,**{k:v for k,v in m.cost().items() if k not in ('operation_proxy','bounded')}))
        if observers:
            for cp in cps.get(number,[]):
                before=base.state_hash(),{p:m.state_hash() for p,m in memories.items()}
                name=cp['name'];k=stream['keys_per_wave'];wi=ord(cp['wave'])-65
                if cp['type']=='acquisition':probe(name,'current',range(wi*k,(wi+1)*k))
                elif cp['type']=='wave':
                    probe(name,'untouched_A',stream['untouched']);probe(name,'current',range(wi*k,(wi+1)*k))
                    if cp['wave']=='A':probe(name,'all_A',range(k))
                    probe(name,'revised_A',stream['revised'],revised=cp['wave']=='D')
                elif cp['type']=='revision':probe(name,'revision',stream['revised'],revised=True)
                elif cp['type']=='noise':probe(name,'noise_A',stream['noisy'])
                assert before==(base.state_hash(),{p:m.state_hash() for p,m in memories.items()})
                if cp['type'] in ('wave','revision'):checkpoint(name)
    if observers:
        before=base.state_hash(),{p:m.state_hash() for p,m in memories.items()}
        probe('after_clutter','current_all',range(len(stream['keys'])),revised=True)
        probe('after_clutter','untouched_A',stream['untouched'])
        probe('after_clutter','revised_A',stream['revised'],revised=True)
        for wi in (1,2,3):
            k=stream['keys_per_wave'];probe('after_clutter','cohort_'+chr(65+wi),range(wi*k,(wi+1)*k))
        for kind in ('singleton','exact_repeat'):
            selected=[r for r in stream['tail_probes'] if r['kind']==kind][:32]
            for p,m in memories.items():
                a=Aggregate()
                for r in selected:
                    z,b=enc.encode(r['context']);a.add(m.lookup(r['context'],base.scores(z,b))['scores'],r['target'])
                probes.append(dict(seed=seed,policy=p,checkpoint='after_clutter',kind=kind,alias=0,**a.result()))
        assert before==(base.state_hash(),{p:m.state_hash() for p,m in memories.items()})
        checkpoint('after_clutter')
    result={'seed':seed,'smoke':smoke,'stream_sha256':stream['raw_sha256'],'standard_sha256':stream['standard_sha256'],
        'tail_sha256':stream['tail_sha256'],'byte_events':len(stream['raw']),'target_presentations':len(stream['rows']),
        'original_logical_keys':len(stream['keys']),'tail_unique_contexts':len(stream['tail_probes']),
        'shared_base_updates':int(base.writes.sum()),'base_hash':base.state_hash(),
        'final_hashes':{p:m.state_hash() for p,m in memories.items()},
        'groups':{p:{c:a.result() for (pp,c),a in groups.items() if pp==p} for p in (*POLICIES,'BASE')},
        'probes':probes,'emergence':emergence,'occupancy':occupancy,'costs':costs,'checkpoints':checkpoints,
        'recurrences':{p:{'n':len(rs),'accuracy':mean(x['accuracy'] for x in rs),
            'max_admission_exposure':max((x['admission_exposure'] for x in rs),default=None),
            'pending_without_recurrence':len(pending[p]),'first_trace_contexts':len(admitted[p])} for p,rs in recurrences.items()},
        'admission':{p:dict(v) for p,v in obs.counts.items()},'trace_residence_log2_histogram':{p:dict(v) for p,v in obs.residence.items()},
        'target_consolidation_fraction':{p:len(v)/len(stream['keys']) for p,v in obs.consolidated.items()},
        'control':{'primary_promotions':obs.schedule_count,'matched_promotions':obs.schedule_matched,
                   'changed_pairs':obs.changed,'current_context_extra_removals':obs.control_extra,
                   'schedule_sha256':evidence_hash.hexdigest(),'timing_count_match':obs.schedule_count==obs.schedule_matched,
                   'deployable':False},
        'peak':{p:dict(m.peak) for p,m in memories.items()},
        'shared_base_bytes':base.weights.nbytes+base.inverse.nbytes+base.writes.nbytes,
        'fixed_encoder_bytes':sum(x.nbytes for e in (enc,memories['LEGACY-U'].enc) for x in (e.embedding,e.centres,e.bit_code,*e.fast.values(),*e.slow.values())),
        'base_operation_proxy':len(stream['raw'])*(D*OUTPUTS*2+D*D*3)}
    return result


def metric(r,p,checkpoint,kind,aliases=(0,1),field='accuracy'):
    rows=[v for v in r['probes'] if v['policy']==p and v['checkpoint']==checkpoint and v['kind']==kind and v['alias'] in aliases]
    if not rows or any(v.get(field) is None for v in rows):return None
    n=sum(v['n'] for v in rows)
    return sum(v[field]*v['n'] for v in rows)/n if n else None


def metrics(r,p):
    e2=[x for x in r['emergence'] if x['policy']==p and x['phase']=='E2']
    acq=[metric(r,p,'after_'+w,'current') for w in 'ABCD']
    return {'acquisition':None if any(x is None for x in acq) else mean(acq),
        'A_acquisition':metric(r,p,'after_A','all_A'),
        'A_after_D':metric(r,p,'after_D','untouched_A'),
        'post_clutter':metric(r,p,'after_clutter','current_all'),
        'post_clutter_heldout':metric(r,p,'after_clutter','current_all',(2,)),
        'revision_exact4':metric(r,p,'revision_4','revision',(0,)),
        'revision_known4':metric(r,p,'revision_4','revision',(1,)),
        'revision_heldout4':metric(r,p,'revision_4','revision',(2,)),
        'obsolete8':metric(r,p,'revision_8','revision',(0,1,2),'obsolete_probability'),
        'whole_bpb':r['groups'][p]['whole']['bpb'],'nontarget_bpb':r['groups'][p]['nontarget']['bpb'],
        'standard_bpb':r['groups'][p]['standard_whole']['bpb'],
        'exact_recurrence':r['recurrences'][p]['accuracy'],
        'E2_immediate_heldout':mean(x['heldout_accuracy'] for x in e2),
        'E2_immediate_pair':mean(x['consolidated_pair'] for x in e2),
        'peak_semantic_bytes':r['peak'][p]['semantic_bytes']}


def report39(records,expected,p0,smoke):
    complete={r['seed'] for r in records}==set(expected)
    levels={p:{k:stat([metrics(r,p)[k] for r in records]) for k in metrics(records[0],p)} for p in POLICIES} if records else {}
    contrasts={}
    def contrast(other,k,benefit='higher'):
        name='TRACE-CORR_minus_'+other+'__'+k
        vals=[]
        for r in records:
            a,b=metrics(r,'TRACE-CORR')[k],metrics(r,other)[k]
            vals.append(None if a is None or b is None else (a-b)*(1 if benefit=='higher' else -1))
        contrasts[name]=stat(vals);contrasts[name]['positive_favors']='TRACE-CORR'
        return contrasts[name]
    def level(k,threshold,floor=None):
        v=levels['TRACE-CORR'][k]
        return v['mean'] is not None and v['n']==len(expected) and v['mean']>=threshold and (floor is None or min(v['values'])>=floor)
    def adv(other,k,margin,benefit='higher'):
        c=contrast(other,k,benefit);return c['mean'] is not None and c['mean']>=margin and c['low'] is not None and c['low']>0
    def pair_limit(p,q,k,margin,every=False):
        diffs=[]
        for r in records:
            a,b=metrics(r,p)[k],metrics(r,q)[k]
            if a is None or b is None:return False
            diffs.append(a-b)
        return bool(diffs) and (max(diffs) if every else mean(diffs))<=margin
    gates={}
    if complete:
        gates['Q0']=len(p0)>=2 and all(x['passed'] and x['full_length'] for x in p0)
        gates['Q1']=pair_limit('LEGACY-U','ANCHOR2-U','A_after_D',.02) and pair_limit('LEGACY-U','ANCHOR2-U','revision_heldout4',.05) and pair_limit('ANCHOR2-U','LEGACY-U','whole_bpb',.05)
        gates['Q2']=level('acquisition',.90,.80)
        gates['Q3']=level('A_after_D',.90,.80)
        gates['Q4']=level('post_clutter',.90,.80)
        gates['Q5']=all(level(k,.90) for k in ('revision_exact4','revision_known4','revision_heldout4')) and levels['TRACE-CORR']['obsolete8']['mean'] is not None and levels['TRACE-CORR']['obsolete8']['mean']<=.10
        gates['Q6']=all(pair_limit('TRACE-CORR','LEGACY-U',k,.10,True) for k in ('whole_bpb','nontarget_bpb'))
        gates['Q7']=all(r['peak']['TRACE-CORR']['traces']<=512 and r['peak']['TRACE-CORR']['slots']<=384 and r['peak']['TRACE-CORR']['anchors']<=768 for r in records)
        gates['Q8']=all(r['peak']['TRACE-CORR']['semantic_bytes']<=2.30*1024**2 and r['peak']['TRACE-CORR']['semantic_bytes']<=.40*r['peak']['LEGACY-U']['semantic_bytes'] for r in records)
        gates['Q9']=adv('FULL-CAP','post_clutter',.05)
        gates['Q10']=adv('TRACE-EXACT2','revision_heldout4',.10) and pair_limit('TRACE-EXACT2','TRACE-CORR','exact_recurrence',.05,True)
        # Evaluate all contrasts; no short-circuit hides alternatives.
        specificity=[adv('SHUFFLED-PAIR','post_clutter',.10),adv('SHUFFLED-PAIR','revision_heldout4',.10),adv('SHUFFLED-PAIR','whole_bpb',.10,'lower')]
        gates['Q11']=any(specificity) and all(r['control']['timing_count_match'] for r in records)
        gates['Q12']=level('exact_recurrence',.90) and all(r['recurrences']['TRACE-CORR']['max_admission_exposure']==1 for r in records)
        gates['Q13']=level('E2_immediate_heldout',.90) and level('E2_immediate_pair',1.)
    else:gates={f'Q{i}':None for i in range(14)}
    return {'version':VERSION,'complete':complete,'smoke':smoke,'completed_seeds':[r['seed'] for r in records],
        'expected_seeds':list(expected),'shared_base_trajectories':len(records),'memory_branches':6*len(records),
        'qualified':complete and not smoke and all(gates.values()),'gates':gates,'levels':levels,'contrasts':contrasts,
        'mechanism_interpretable':complete and not smoke and gates['Q0'] and gates['Q1'],
        'interval_method':'paired Student t, seed unit, two-sided 95%, unadjusted; Q11 is the preregistered disjunction',
        'caveat':'Engineering/smoke runs cannot qualify a scientific mechanism. Failed Q1 voids admission attribution.'}


def package_outputs(root,records,rep,manifest,p0,source):
    write_json(root/'REPORT.json',rep);write_json(root/'RUN_MANIFEST.json',manifest)
    write_json(root/'STATUS.json',{'complete':rep['complete'],'completed_seeds':rep['completed_seeds'],'expected_seeds':rep['expected_seeds']})
    atomic(root/'PROTOCOL.md',PROTOCOL+'\n'+IMPLEMENTATION_NOTES)
    atomic(root/'elm_v39_earned_consolidation.py',source)
    write_json(root/'COMPACTION_EQUIVALENCE.json',{'runs':p0,'passed':len(p0)>=2 and all(x['passed'] and x['full_length'] for x in p0)})
    write_json(root/'CONTROL_AUDIT.json',{'seeds':[dict(seed=r['seed'],**r['control']) for r in records],
        'shared_base':'one physical base per seed; base update called once per byte',
        'no_evaluator_inputs':'Memory.lookup(c,base_scores); Memory.observe(packet,y,schedule for SHUFFLED only)',
        'capacity_enforced_every_event':True,'prequential_prediction':True})
    write_json(root/'STATE_COSTS.json',{'seeds':[{'seed':r['seed'],'peak':r['peak'],'costs':r['costs'],
        'shared_base_bytes':r['shared_base_bytes'],'fixed_encoder_bytes':r['fixed_encoder_bytes'],
        'base_operation_proxy':r['base_operation_proxy']} for r in records],
        'notes':'Semantic payload charges explicit indices and counters. Python resident estimate includes array payload, not just overhead. Temporary reconstruction arrays excluded from persistent payload; no cached trace vectors.'})
    rows=[dict(seed=r['seed'],policy=p,**metrics(r,p)) for r in records for p in POLICIES]
    csv_output(root/'SUMMARY.csv',rows)
    for name,key in [('OCCUPANCY_CURVES.csv','occupancy'),('E1_E2_CONSOLIDATION.csv','emergence')]:csv_output(root/name,[v for r in records for v in r[key]])
    allprobes=[v for r in records for v in r['probes']]
    csv_output(root/'RETENTION_BY_WAVE.csv',[v for v in allprobes if v['checkpoint'].startswith('after_') and v['checkpoint']!='after_clutter'])
    csv_output(root/'POST_CLUTTER_RETENTION.csv',[v for v in allprobes if v['checkpoint']=='after_clutter'])
    csv_output(root/'REVISION_CURVES.csv',[v for v in allprobes if v['kind']=='revision'])
    admissions=[];evictions=[]
    for r in records:
        for p,counts in r['admission'].items():
            for event,count in counts.items():
                target=evictions if 'eviction' in event or 'removal' in event else admissions
                target.append(dict(seed=r['seed'],policy=p,event=event,count=count))
            admissions.append(dict(seed=r['seed'],policy=p,event='fraction_target_keys_consolidated',count=r['target_consolidation_fraction'][p]))
    csv_output(root/'ADMISSION_SUMMARY.csv',admissions);csv_output(root/'EVICTION_SUMMARY.csv',evictions)
    csv_output(root/'PAIRED_CONTRASTS.csv',[dict(contrast=k,**v) for k,v in rep['contrasts'].items()])
    lines=['# V39 earned consolidation', '',f"Completed {len(records)}/{len(rep['expected_seeds'])} shared-base seeds; {6*len(records)} memory branches.",
        f"Scientific qualification: {'PASS' if rep['qualified'] else 'NOT ESTABLISHED' if rep['complete'] else 'PENDING'}.",
        'Engineering/smoke output is not scientific evidence.' if rep['smoke'] else '',
        '', '| Policy | A after D | After clutter | Held-out revision @4 | Whole bpb |', '|---|---:|---:|---:|---:|']
    for p,l in rep['levels'].items():
        fmt=lambda k: 'n/a' if l[k]['mean'] is None else f"{l[k]['mean']:.4f}"
        lines.append('| '+p+' | '+' | '.join(fmt(k) for k in ('A_after_D','post_clutter','revision_heldout4','whole_bpb'))+' |')
    lines+=['','Gates: '+', '.join(k+'='+('pending' if v is None else 'pass' if v else 'fail') for k,v in rep['gates'].items()),
        '', 'Accuracy is a fraction. Gates and all paired seed values are in REPORT.json. Failed Q1 voids admission attribution.',
        'Resume verifies completed seed records; an interrupted seed restarts from its beginning.']
    atomic(root/'SUMMARY.md','\n'.join(lines)+'\n')
    # Archive only intended aggregate products and small completed-seed records.
    files=sorted(p for p in root.rglob('*') if p.is_file() and p.name not in ('.run.lock','ARTIFACT_HASHES.json') and '.tmp-' not in p.name)
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write_json(root/'ARTIFACT_HASHES.json',hashes);files.append(root/'ARTIFACT_HASHES.json')
    archive=root.with_name(root.name+'_results.zip');tmp=archive.with_name(archive.name+'.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as zf:
        for p in files:zf.write(p,p.relative_to(root))
    os.replace(tmp,archive)
    if archive.stat().st_size>=20*1024**2:raise RuntimeError('Archive exceeds 20 MiB logging budget')
    return archive


def self_test39():
    checks=[]
    def ok(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    seed=973910;c0=context(b'ABCDEF',0);c1=context(b'ABCDEF',1);c2=context(b'ABCDEF',2)
    base=np.zeros(OUTPUTS);base[2]=1.;y=100
    m=Memory('TRACE-CORR',seed)
    p=m.lookup(c0,base);m.observe(p,y)
    ok('first miss creates trace only',len(m.traces)==1 and not m.slots)
    ok('exact trace authority',m.lookup(c0,base)['authority'] and np.argmax(m.lookup(c0,base)['scores'])==y)
    p=m.lookup(c1,base);ok('support one remains shadow',p['type']=='trace' and not p['authority'] and np.array_equal(p['scores'],base))
    m.observe(m.lookup(c0,base),y);ok('exact recurrence never promotes TRACE-CORR',len(m.traces)==1 and not m.slots)
    events=m.observe(m.lookup(c1,base),y)
    ok('distinct confirmation promotes once',sum(e['event']=='promotion' for e in events)==1)
    ok('promotion removes trace and stores two anchors',not m.traces and len(m.slots)==1 and len(m.anchors)==2)
    ok('held-out consolidated authority',m.lookup(c2,base)['authority'])
    old=m.state_hash();m.lookup(c2,base);ok('probe counters and RNG unchanged',old==m.state_hash())
    m.observe(m.lookup(c2,base),y);ok('third context not permanent member',len(m.anchors)==2 and c2 not in m.anchors)
    m.observe(m.lookup(c0,base),101);ok('anchor contradiction revises entire slot',np.argmax(m.lookup(c1,base)['scores'])==101)
    m2=Memory('TRACE-EXACT2',seed);m2.observe(m2.lookup(c0,base),y);m2.observe(m2.lookup(c0,base),y)
    ok('EXACT2 second exact occurrence promotes',len(m2.slots)==1 and not m2.traces and len(m2.anchors)==1)
    ok('recurrence credit not false CORR2 support',not m2.lookup(c1,base)['authority'])
    bad=Memory('TRACE-CORR',seed);bad.observe(bad.lookup(c0,base),y);bad.observe(bad.lookup(c1,base),101)
    ok('wrong pre-update prediction rejects promotion',len(bad.traces)==2 and not bad.slots)
    correctbase=np.zeros(OUTPUTS);correctbase[y]=1
    n=Memory('TRACE-CORR',seed);n.observe(n.lookup(c0,correctbase),y)
    ok('base-correct unbound context allocates nothing',not n.traces and not n.slots)
    # Actual hard caps, using allocation primitives to isolate overflow from routing.
    n=Memory('TRACE-CORR',seed);events=[]
    first=(0).to_bytes(12,'big')
    for i in range(520):n.clock+=1;n._new_trace(i.to_bytes(12,'big'),y,events);n.check()
    ok('512 trace overflow enforced',len(n.traces)==512 and first not in n.traces)
    ok('no hidden eligibility or tombstones',not hasattr(n,'eligible') and not hasattr(n,'evicted') and not n.anchors)
    s=Memory('FULL-CAP',seed);events=[]
    for i in range(390):
        s.clock+=1;s._new_slot([(2*i).to_bytes(12,'big'),(2*i+1).to_bytes(12,'big')],np.zeros(OUTPUTS),events);s.check()
    ok('384 slot overflow enforced',len(s.slots)==384 and len(s.anchors)==768)
    ok('eviction cleans both anchor indices',first not in s.anchors and (1).to_bytes(12,'big') not in s.anchors)
    ok('every retained index resolves',all(c in s.slots[j]['anchors'] for c,j in s.anchors.items()))
    same=Memory('FULL-CAP',seed);events2=[]
    for i in range(390):
        same.clock+=1;same._new_slot([(2*i).to_bytes(12,'big'),(2*i+1).to_bytes(12,'big')],np.zeros(OUTPUTS),events2);same.check()
    ok('deterministic LFU LRU eviction',s.state_hash()==same.state_hash())
    tc=Memory('TRACE-CORR',seed);sh=Memory('SHUFFLED-PAIR',seed)
    tc.observe(tc.lookup(c0,base),y)
    sh.observe(sh.lookup(context(b'UVWXYZ',0),base),y)
    packet=tc.lookup(c1,base);control=sh.lookup(c1,base)
    schedule=next(e for e in tc.observe(packet,y) if e['event']=='promotion')
    ev=sh.observe(control,y,schedule)
    ok('shuffled schedule same promotion count',sum(e['event']=='promotion' for e in ev)==1)
    ok('shuffled identity assignment changed',next(e for e in ev if e['event']=='schedule_match')['changed_pair'])
    ok('worst-case payload below cap',512*(12+2048+32+8)+384*(2048+768+32+2*(12+8))<=2.30*1024**2)
    comp=compaction_preflight(seed,True);ok('synthetic compact legacy equivalence',comp['passed'])
    # Records are JSON not executable pickle. Atomic seed checkpoint integrity.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/'record.json';data={'seed':seed,'hash':s.state_hash()};write_json(path,{'data':data,'sha256':sha(data)})
        loaded=json.loads(path.read_text());ok('atomic restart exact record hash',loaded['data']==data and loaded['sha256']==sha(loaded['data']))
        loaded['data']['seed']+=1;ok('corruption detected',loaded['sha256']!=sha(loaded['data']))
        identity={'source':'a','protocol':'b'}
        try:validate_identity(identity,dict(identity,source='c'))
        except RuntimeError:ok('changed source rejected',True)
        else:ok('changed source rejected',False)
    return {'passed':True,'count':len(checks),'checks':checks,'compact_smoke':comp}


def validate_identity(existing,expected):
    if existing!=expected:raise RuntimeError('Source/protocol/plan/stream identity mismatch. Use a new output directory.')


def record_write(path,data,identity):
    envelope={'identity_sha256':sha(identity),'data':data,'sha256':sha(data)};write_json(path,envelope)


def record_read(path,identity):
    x=json.loads(path.read_text())
    if x.get('identity_sha256')!=sha(identity) or x.get('sha256')!=sha(x.get('data')):
        raise RuntimeError('Corrupt or mismatched completed record: '+path.name)
    return x['data']


def main39(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--install-deps',action='store_true');ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--verify-full',action='store_true',help='two full P0 seeds plus full observer invariance; no science')
    ap.add_argument('--smoke',action='store_true',help='two reduced engineering seeds, all six policies')
    ap.add_argument('--output',type=Path);ap.add_argument('--max-seeds',type=int,help='pause after this many NEW completed seeds')
    args=ap.parse_args(argv);dependencies(args.install_deps)
    if args.max_seeds is not None and args.max_seeds<0:ap.error('--max-seeds must be nonnegative')
    caffeinate=None
    if platform.system()=='Darwin':
        try:caffeinate=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        except FileNotFoundError:print('caffeinate unavailable; continuing.',flush=True)
    try:
        source=Path(__file__).read_text();tests=self_test39()
        print(f"Startup checks passed: {tests['count']}",flush=True)
        if args.self_test and not args.verify_full:return 0
        seeds=(973911,973912) if args.smoke else SEEDS
        mode='verification' if args.verify_full else 'smoke' if args.smoke else 'science'
        root=args.output or Path('elm_v39_'+mode)
        root.mkdir(parents=True,exist_ok=True)
        plans=[]
        if not args.verify_full:
            for seed in seeds:
                s=workload(seed,args.smoke)
                plans.append({'seed':seed,'standard_sha256':s['standard_sha256'],'stream_sha256':s['raw_sha256'],
                              'tail_sha256':s['tail_sha256'],'byte_events':len(s['raw']),'tail_rows':s['tail_rows']})
        identity={'version':VERSION,'source_sha256':sha(source.encode()),'protocol_sha256':sha((PROTOCOL+IMPLEMENTATION_NOTES).encode()),
            'reference_sha256':REFERENCE_SHA256,'policies':list(POLICIES),'mode':mode,'plans':plans,
            'caps':{'trace':512,'slot':384,'anchors_per_slot':2},'engineering_seeds':list(ENGINEERING_SEEDS)}
        with RunLock(root):
            manifest_path=root/'RUN_MANIFEST.json'
            if manifest_path.exists():validate_identity(json.loads(manifest_path.read_text()),identity)
            else:write_json(manifest_path,identity)
            p0path=root/'P0_CHECKPOINT.json'
            if p0path.exists():p0=record_read(p0path,identity)
            else:
                p0=[]
                for seed in ENGINEERING_SEEDS:
                    print(f'Full compaction equivalence: seed {seed}',flush=True)
                    p0.append(compaction_preflight(seed))
                    print(f'P0 seed {seed} passed',flush=True)
                record_write(p0path,p0,identity)
            if len(p0)!=2 or not all(r['passed'] and r['full_length'] for r in p0):raise RuntimeError('P0 gate failed')
            if args.verify_full:
                print('Full engineering seed: six branches with observers',flush=True)
                a=trial39(973903,False,True)
                print('Full engineering seed: independent replay without observers',flush=True)
                b=trial39(973903,False,False)
                assert a['base_hash']==b['base_hash'] and a['final_hashes']==b['final_hashes']
                assert a['control']['schedule_sha256']==b['control']['schedule_sha256']
                verification={'self_test':tests,'p0':p0,'observer_invariance':True,'engineering_seed':973903,
                              'base_hash':a['base_hash'],'final_hashes':a['final_hashes'],'peak':a['peak'],
                              'control':a['control'],'science_run':False,'source_sha256':identity['source_sha256']}
                write_json(root/'VERIFICATION.json',verification)
                rep=report39([a],(973903,),p0,True);archive=package_outputs(root,[a],rep,identity,p0,source)
                print('Full verification passed. '+str(archive),flush=True);return 0
            records=[];new=0
            for seed in seeds:
                path=root/'completed'/f'{seed}.json'
                if path.exists():
                    r=record_read(path,identity)
                    if r['seed']!=seed or r['stream_sha256']!=next(p['stream_sha256'] for p in plans if p['seed']==seed):raise RuntimeError('Seed/stream mismatch')
                    records.append(r);print(f'Reused seed {seed}',flush=True);continue
                if args.max_seeds is not None and new>=args.max_seeds:continue
                print(f'Running seed {seed}: one base, six branches',flush=True)
                r=trial39(seed,args.smoke);record_write(path,r,identity);records.append(r);new+=1
                print(f'Completed {len(records)}/{len(seeds)} seeds',flush=True)
                rep=report39(records,seeds,p0,args.smoke);package_outputs(root,records,rep,identity,p0,source)
            rep=report39(records,seeds,p0,args.smoke);archive=package_outputs(root,records,rep,identity,p0,source)
            print(f"{'Complete' if rep['complete'] else 'Paused'}: {len(records)}/{len(seeds)} seeds; {new} new. {archive}",flush=True)
            return 0
    finally:
        if caffeinate is not None:caffeinate.terminate()

if __name__=='__main__':
    try:raise SystemExit(main39())
    except KeyboardInterrupt:
        print('\nInterrupted. Completed seeds are safe; rerun the same command to resume.',file=sys.stderr)
        raise SystemExit(130)
