#!/usr/bin/env python3
"""V40 utility under scarcity. Run: python3 elm_v40_utility_under_scarcity.py --install-deps
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

VERSION="v40.1.0"
SEEDS=tuple(range(40101,40109))
ENGINEERING_SEEDS=(974001,974002)
POLICIES=("RANDOM","LRU","LFU-RATE","UTILITY-RATE","SHUFFLED-UTILITY","TRACE-U")
REFERENCE_SHA256='229acfe264f40a0647aede5b7301a05d09c4ea81743748a17216a1de544103e9'
PROTOCOL="# V40 — Utility Under Scarcity\n## Which persistent memories deserve continued existence under a hard long-term budget?\n\n**Status:** design / preregistration only. No V40 scientific outcomes have been observed.\n\n**Proposed scientific seeds:** `40101`–`40108`  \n**Independent streams:** 8  \n**Primary substrate:** V39 `TRACE-CORR`, frozen  \n**Primary scientific variable:** consolidated-slot replacement priority only  \n**Hard capacities:** `T = 512` probationary exact traces; `C = 384` consolidated two-anchor slots  \n**Total logical identities:** `N = 768` (2× consolidated-slot capacity)  \n**Learned renderings per identity:** 2; third rendering remains read-only held-out  \n**Primary candidate:** `UTILITY-RATE`  \n**Decisive control:** `SHUFFLED-UTILITY`  \n**No PBF as a load-bearing mechanism. No capacity sweep. No utility-temperature/decay sweep. No automatic follow-up.**\n\n---\n\n# 0. Decision in one sentence\n\nV39 established that most one-shot experiences should not immediately consume expensive long-term object slots: a bounded probationary trace plus distinct-context predictive corroboration preserved almost all target behavior under heavy transient clutter while reducing incremental episodic state by roughly an order of magnitude.\n\nV40 asks the next question, which V39 deliberately did not test:\n\n\\[\n\\boxed{\n\\text{What happens when the number of genuinely corroborated, reusable objects exceeds the slot budget?}\n}\n\\]\n\nThe primary hypothesis is:\n\n> **When long-term capacity is genuinely scarce, persistence should track the realized marginal predictive value of each memory, not merely its age, recurrence count, or arbitrary identity.**\n\nThe proposed replacement score is the memory's causal rate of bits saved relative to the shared base:\n\n\\[\n\\boxed{\nU_j(t)\n=\n\\frac{G_j(t)}\n     {t-t_j^{\\rm birth}+1},\n\\qquad\nG_j(t)\n=\n\\sum_{\\tau\\le t,\\ \\tau\\in j}\n\\left[\n\\ell_B(\\tau)-\\ell_j(\\tau)\n\\right].\n}\n\\]\n\nHere:\n\n- \\(\\ell_B\\) is the pre-update base-only log loss on the actually observed byte;\n- \\(\\ell_j\\) is the pre-update loss of the deployed memory prediction for slot \\(j\\);\n- positive gain means the memory saved predictive bits;\n- negative gain means the memory was actively harmful;\n- dividing by slot age produces a parameter-free utility **rate** in bits saved per stream event.\n\nWhen a new corroborated object must consolidate and all \\(C=384\\) slots are occupied, `UTILITY-RATE` evicts the slot with the lowest \\(U_j\\).\n\n---\n\n# 1. What V40 is — and is not\n\n## 1.1 V40 is a genuine scarcity experiment\n\nV39's main stressor was transient clutter. `TRACE-CORR` succeeded largely because most clutter never earned consolidation.\n\nThat was the correct first bounded-memory problem, but it did **not** answer the harder case in which:\n\n\\[\nN_{\\rm legitimate\\ consolidated\\ objects} > C.\n\\]\n\nV40 therefore makes every competing identity capable of:\n\n- one-shot exact memory;\n- distinct-context corroboration;\n- two-anchor consolidation;\n- later recurrence;\n- held-out alias transfer.\n\nThe learner cannot solve V40 simply by refusing to admit one-off noise.\n\n## 1.2 V40 is not an eviction-rule zoo\n\nOnly replacement priority changes.\n\nThe following remain identical across all deployable bounded arms:\n\n- byte-level causal next-byte prediction;\n- shared base learner;\n- lag96 address representation;\n- \\(\\tau=0.85\\);\n- absolute local value semantics;\n- pre-update predictive confirmation;\n- V37 `CORR2` read authority;\n- V39 probationary exact traces;\n- V39 distinct-context promotion;\n- V39 two-anchor consolidated slots;\n- trace cap \\(T=512\\);\n- slot cap \\(C=384\\);\n- demotion semantics after slot eviction;\n- no replay;\n- no evaluator identity metadata;\n- no task/wave/importance labels.\n\nV40 does **not** compare new encoders, write rules, admission gates, confidence temperatures, trace sizes, slot sizes, or capacity levels.\n\n## 1.3 V40 does not include the phase-stationarity closure\n\nClaude's V39 reopened a separate representation question: high separation was confounded with nonstationary phase normalization.\n\nThat closure should be run separately as a tiny V39.2 experiment:\n\n\\[\n\\text{S2-lag-c12}\n\\quad\\text{vs}\\quad\n\\text{frozen-phase-c12}\n\\quad\\text{vs}\\quad\n\\text{live-phase-c12}.\n\\]\n\nIts result may matter later for indexing or superposed overflow, but it is **not** a V40 variable.\n\n---\n\n# 2. Why raw LRU/LFU are insufficient hypotheses\n\nLong-term memory capacity should not necessarily be spent on whatever was seen most recently or most frequently.\n\nConsider two equally recurrent contexts:\n\n- slot \\(a\\): the shared base already assigns the correct byte probability \\(0.95\\);\n- slot \\(b\\): the shared base assigns \\(0.02\\), while the object memory assigns \\(0.95\\).\n\nThe recurrence count is the same, but their marginal episodic value is radically different.\n\nThe actual bit savings are approximately:\n\n\\[\nr_a\n=\n-\\log_2(0.95)+\\log_2(0.95)\n\\approx0,\n\\]\n\nwhile\n\n\\[\nr_b\n=\n-\\log_2(0.02)+\\log_2(0.95)\n\\approx5.6\\ {\\rm bits}.\n\\]\n\nA lifelong learner should prefer \\(b\\) even if \\(a\\) is marginally more recent.\n\nThis motivates the V40 distinction:\n\n\\[\n\\boxed{\n\\text{frequency of use}\n\\neq\n\\text{marginal value of remembering}.\n}\n\\]\n\n---\n\n# 3. The learner-visible memory hierarchy\n\nV40 inherits the V39 hierarchy exactly.\n\n## 3.1 Shared predictive base\n\nThe base operates continuously and is never capacity-limited by V40.\n\nIts pre-update distribution is always available as the counterfactual:\n\n\\[\np_B(y\\mid c).\n\\]\n\nThe base does not receive information about memory priority, hidden recurrence class, or evaluator utility.\n\n## 3.2 Probationary exact trace\n\nA first surprising experience earns immediate exact memory but not reusable identity.\n\nA trace stores the same compact state as V39:\n\n- exact 12-byte context;\n- absolute local value;\n- exact-hit metadata;\n- recency metadata.\n\nHard cap:\n\n\\[\nT=512.\n\\]\n\nExact hit:\n\n\\[\np(y\\mid c)=p_{\\rm trace}(y).\n\\]\n\nSimilarity-only hit:\n\n\\[\np(y\\mid c)=p_B(y\\mid c).\n\\]\n\nA support-1 trace may propose a candidate for post-outcome corroboration but has zero generalized authority.\n\n## 3.3 Consolidated two-anchor slot\n\nA second **distinct** context that:\n\n1. similarity-matches the trace;\n2. arrives while the base is wrong;\n3. is correctly predicted pre-update by the trace value,\n\npromotes the identity to one consolidated slot.\n\nThe slot contains:\n\n- exactly two anchor contexts;\n- one prototype accumulator / prototype;\n- one absolute 256-way value;\n- access metadata;\n- replacement-policy metadata.\n\nHard cap:\n\n\\[\nC=384.\n\\]\n\nNo additional permanent member table is allowed.\n\n---\n\n# 4. Common demotion semantics\n\nV40 must distinguish:\n\n\\[\n\\text{loss of generalized ownership}\n\\]\n\nfrom:\n\n\\[\n\\text{complete erasure of exact experience}.\n\\]\n\nTherefore every bounded replacement policy uses the same deterministic demotion rule.\n\nWhen consolidated slot \\(j\\) is evicted:\n\n1. remove its similarity-addressable slot state;\n2. copy its two exact anchors back into the probationary trace pool as two exact traces carrying the current absolute value;\n3. if the trace pool exceeds \\(T=512\\), apply the unchanged V39 trace-LRU rule until the cap is restored;\n4. destroy the evicted slot's replacement ledger;\n5. retain no tombstone, historical score, hidden membership record, or replay state.\n\nThus an evicted memory degrades:\n\n\\[\n\\boxed{\n\\text{generalizable slot}\n\\rightarrow\n\\text{exact-only trace}\n\\rightarrow\n\\text{forgotten}.\n}\n\\]\n\nIf two distinct traces for that logical identity later re-establish predictive corroboration, the object may consolidate again.\n\nRe-consolidation starts a **fresh** replacement ledger. No old utility score survives eviction.\n\nThis common demotion mechanism is infrastructure, not a V40 scientific variable.\n\n---\n\n# 5. Causal utility measurement\n\n## 5.1 Reward is computed only after the outcome is observed\n\nFor every event on which consolidated slot \\(j\\) supplied the deployed object-memory prediction, record from the frozen pre-update state:\n\n\\[\n\\ell_B\n=\n-\\log_2 p_B(y\\mid c),\n\\]\n\n\\[\n\\ell_j\n=\n-\\log_2 p_j^{\\rm deployed}(y\\mid c),\n\\]\n\nand\n\n\\[\n\\boxed{\nr_j=\\ell_B-\\ell_j.\n}\n\\]\n\nPositive \\(r_j\\):\n\n> retaining the slot saved predictive bits.\n\nNegative \\(r_j\\):\n\n> the slot hurt prediction relative to base-only.\n\nNo reward is available before observing \\(y\\).\n\nThe reward therefore contains no future information and no evaluator label.\n\n## 5.2 Which events update the ledger?\n\nUpdate the ledger only when the slot had actual deployed authority:\n\n- exact anchor hit;\n- confirmed CORR2 similarity retrieval from a consolidated slot.\n\nDo not credit a slot for:\n\n- merely being the nearest candidate while support/authority blocks its prediction;\n- read-only evaluator probes;\n- outcomes after the slot has already been evicted;\n- hypothetical predictions never used by the learner.\n\nThis makes the ledger a record of **realized marginal deployed utility**, not a counterfactual fantasy about every nearby context.\n\n## 5.3 Initial utility at consolidation\n\nA newborn slot would otherwise have age \\(1\\) and zero reward.\n\nAt the E2 consolidation event, the first trace's pre-update prediction and the base prediction are both already available.\n\nSeed the new slot's gain ledger with:\n\n\\[\nG_j(t_{\\rm birth})\n=\n\\ell_B(E2)-\\ell_{\\rm trace}(E2).\n\\]\n\nThis is exactly the causal predictive-confirmation event that earned consolidation.\n\nNo arbitrary prior bonus is added.\n\n## 5.4 Utility-rate score\n\nAt stream event \\(t\\):\n\n\\[\n\\boxed{\nU_j(t)\n=\n\\frac{G_j(t)}\n     {t-t_j^{\\rm birth}+1}.\n}\n\\]\n\nThe denominator advances even when the slot is not accessed.\n\nConsequences:\n\n- useful but dormant memories gradually lose priority;\n- repeatedly useful memories maintain a high rate;\n- frequently accessed memories that add little beyond the base remain low-value;\n- harmful memories can become negative;\n- no exponential-decay coefficient or horizon is introduced.\n\nTie-breaking:\n\n1. smaller \\(U_j\\);\n2. older last use;\n3. older allocation index.\n\nAll ties are deterministic.\n\n---\n\n# 6. V40 workload: rolling useful-memory overload\n\nA naive experiment that first presents 768 identical-looking memories and only later reveals which 384 matter would be a clairvoyance test.\n\nV40 instead makes usefulness observable **online before and during scarcity**.\n\n## 6.1 Identity population\n\nGenerate:\n\n\\[\nN=768\n\\]\n\nunique logical identities.\n\nEach identity has:\n\n- learned rendering \\(a_0\\);\n- learned rendering \\(a_1\\);\n- never-trained held-out rendering \\(a_2\\);\n- one arbitrary target byte from the same 24-byte target alphabet used in V38/V39.\n\nIdentities are divided evaluator-only into six introduction cohorts:\n\n\\[\nK_1,\\ldots,K_6,\n\\qquad\n|K_k|=128.\n\\]\n\nThe cohort identifier is never exposed to the learner.\n\n## 6.2 Admission pair for every identity\n\nFor each cohort:\n\n1. present every \\(a_0\\) once in seed-shuffled order;\n2. after an interleaving gap, present every \\(a_1\\) once;\n3. require the ordinary TRACE-CORR mechanism to create traces and then consolidate identities.\n\nNo special admission path exists for V40.\n\nEvery identity is evaluator-intended to be capable of earning a slot from its two learned renderings.\n\n## 6.3 Recurrence classes\n\nAt stream generation, identities are assigned evaluator-only recurrence schedules.\n\nUse four equal classes:\n\n- `HOT`: 192 identities;\n- `WARM`: 192;\n- `COOL`: 192;\n- `FADING`: 192.\n\nThe class is never stored in the learner state.\n\nThe fixed service schedule is:\n\n| Class | Service opportunities after introduction |\n|---|---|\n| HOT | 4 per service epoch |\n| WARM | 2 per service epoch |\n| COOL | 1 every second service epoch |\n| FADING | 2 per epoch for the first two eligible epochs, then 0 |\n\nWithin each service epoch:\n\n- recurrence rows are seed-shuffled;\n- learned aliases \\(a_0,a_1\\) alternate;\n- no \\(a_2\\) rendering is trained;\n- no old item is replayed merely to protect it — recurrence is the workload itself.\n\nThis creates:\n\n- high continuing demand;\n- medium demand;\n- sparse demand;\n- memories whose usefulness genuinely expires.\n\n## 6.4 Rolling schedule\n\nThe stream is:\n\n\\[\nK_1\n\\rightarrow S_1\n\\rightarrow\nK_2\n\\rightarrow S_2\n\\rightarrow\nK_3\n\\rightarrow S_3\n\\rightarrow\nV\n\\rightarrow\nK_4\n\\rightarrow S_4\n\\rightarrow\nK_5\n\\rightarrow S_5\n\\rightarrow\nK_6\n\\rightarrow S_6\n\\rightarrow\nS_7\n\\rightarrow\nS_8.\n\\]\n\nHere \\(S_k\\) is a recurrence/service epoch over all identities introduced so far.\n\nAfter \\(K_3\\), exactly 384 identities have had the opportunity to consolidate, matching the slot cap.\n\nBefore \\(K_4\\) starts, run a **valuation pass** \\(V\\):\n\n- one learned-rendering recurrence for every currently consolidated identity;\n- identical exposure count across those identities;\n- no new identities.\n\nThis guarantees that every incumbent slot has at least one post-consolidation causal utility observation before the first intended capacity-forced replacement.\n\nThe first genuine slot-scarcity event should therefore occur during \\(K_4\\), not before the learner has any evidence.\n\n## 6.5 Why the recurrence schedule is not an importance label\n\nThe learner observes only ordinary byte events.\n\nIt is never told:\n\n- recurrence class;\n- future recurrence count;\n- whether an object will fade;\n- evaluator utility;\n- whether the current slot is in the future top quartile.\n\nRecurrence history becomes informative only because the environment actually recurs.\n\nThis is the causal information an online learner is allowed to exploit.\n\n---\n\n# 7. Task-validity screens\n\nThese gates are evaluated before replacement-policy claims.\n\nIf a screen fails, report the workload result but do not interpret policy rankings.\n\n## P0 — consolidation validity\n\nIn the unbounded reference:\n\n- at least **95%** of the 768 intended identities must consolidate by their E2 event;\n- exact learned-rendering accuracy after consolidation \\(\\ge90\\%\\);\n- held-out \\(a_2\\) accuracy after consolidation \\(\\ge90\\%\\).\n\nOtherwise the workload is testing admission failure rather than scarcity.\n\n## P1 — genuine capacity pressure\n\nIn every bounded arm:\n\n- slot occupancy reaches exactly \\(C=384\\);\n- at least **192 slot replacement events** occur after scarcity begins;\n- no arm can avoid pressure merely by failing to acquire.\n\nThe unbounded reference must exceed:\n\n\\[\n1.75C=672\n\\]\n\nsimultaneously resident consolidated identities at some point.\n\n## P2 — pre-scarcity observability\n\nBefore the first intended replacement:\n\n- every incumbent consolidated slot has at least one post-birth deployed-memory reward observation;\n- no replacement policy has access to future recurrence information.\n\nThis gate prevents V40 from becoming a clairvoyance benchmark.\n\n## P3 — utility heterogeneity exists\n\nUsing the **unbounded reference only** and evaluator-side post hoc summaries, compute each slot's realized future marginal bits saved from scarcity onset onward.\n\nRequire:\n\n- 90th percentile future utility at least **2×** the median;\n- interquartile range \\(>0\\);\n- at least 20% of slots with positive future cumulative gain;\n- at least 20% with future gain less than half the 80th-percentile value.\n\nThis gate says there is something meaningful for a replacement policy to discriminate.\n\nIt does **not** expose those future values to deployable learners.\n\n---\n\n# 8. Replacement policies\n\nAll bounded policies share the same admission, demotion, trace policy, base trajectory and capacities.\n\nOnly the victim-selection score changes.\n\n## 8.1 `RANDOM`\n\nAt every forced slot replacement:\n\n- choose a uniformly random resident consolidated slot;\n- deterministic RNG seeded from science seed and replacement index.\n\nPurpose:\n\n- capacity floor / sanity control.\n\n## 8.2 `LRU`\n\nEvict the consolidated slot with the oldest last deployed use.\n\nTie-break by oldest allocation index.\n\nPurpose:\n\n- standard recency baseline.\n\n## 8.3 `LFU-RATE`\n\nMaintain deployed-use count \\(H_j\\).\n\nScore:\n\n\\[\nF_j(t)\n=\n\\frac{H_j(t)}\n     {t-t_j^{\\rm birth}+1}.\n\\]\n\nEvict the smallest \\(F_j\\).\n\nThis is frequency per unit stream time rather than raw cumulative count, so very old objects are not automatically immortal.\n\nPurpose:\n\n- strongest simple recurrence-only comparator.\n\n## 8.4 `UTILITY-RATE` — primary V40 arm\n\nUse:\n\n\\[\nU_j(t)\n=\n\\frac{\\sum r_j}\n     {t-t_j^{\\rm birth}+1}.\n\\]\n\nEvict the smallest \\(U_j\\).\n\nPurpose:\n\n- test whether **marginal predictive value** is a better persistence signal than recency or recurrence frequency.\n\n## 8.5 `SHUFFLED-UTILITY` — decisive specificity control\n\nThis control computes the **same real per-slot utility ledgers** as `UTILITY-RATE`.\n\nAt every forced replacement event:\n\n1. collect the current resident slots;\n2. bin them by:\n   - deployed-use count: `0–1`, `2–3`, `4–7`, `8+`;\n   - age quartile within the current resident population;\n3. within each non-singleton stratum, deterministically permute the current \\(U_j\\) scores among slot identities using a seed-specific shadow RNG;\n4. singleton strata retain their own score only because no same-stratum alternative exists;\n5. evict according to the **shuffled** score.\n\nThus the control preserves, as closely as possible:\n\n- current score distribution;\n- age distribution;\n- frequency distribution;\n- number and timing of replacement events;\n- memory budget;\n- score scale and dispersion.\n\nIt destroys:\n\n\\[\n\\boxed{\\text{which slot earned which utility history}.}\n\\]\n\nIf `SHUFFLED-UTILITY` matches `UTILITY-RATE`, slot-specific realized utility has not earned its mechanistic interpretation.\n\nThis arm is evaluator-assisted and nondeployable.\n\n## 8.6 `TRACE-U` — unbounded reference\n\nSame TRACE-CORR admission/read/update semantics with no trace or slot capacity limit.\n\nPurpose:\n\n- regret reference;\n- task-validity analysis;\n- evaluator-only future-utility ceiling construction.\n\nIt is not a bounded competitor.\n\n## 8.7 `ORACLE-FUTURE` — optional nondeployable ceiling\n\nFrom the fully realized fixed stream and `TRACE-U` trajectory, an evaluator may compute each currently resident identity's **future** marginal bits saved.\n\nAt each replacement event, `ORACLE-FUTURE` evicts the slot with lowest realized remaining future value.\n\nThis arm is strictly descriptive:\n\n- no qualification gate requires beating it;\n- it cannot be called a learner;\n- its only purpose is to estimate how much room remains above causal policies.\n\nIf implementing it complicates the runner materially, omit it.\n\n---\n\n# 9. Primary endpoint: predictive regret under scarcity\n\nOnce useful memories exceed capacity, 100% retention is impossible by construction.\n\nThe primary endpoint is therefore not raw accuracy.\n\nLet scarcity begin at the first event where a new consolidated slot requires a victim.\n\nFor bounded policy \\(m\\):\n\n\\[\n\\boxed{\nR_m\n=\n\\sum_{t\\ge t_{\\rm scarcity}}\n\\left[\n\\ell_m(t)-\\ell_U(t)\n\\right],\n}\n\\]\n\nwhere:\n\n- \\(\\ell_m(t)\\) is the policy's prequential log loss;\n- \\(\\ell_U(t)\\) is `TRACE-U` loss on the same byte event and same seed.\n\nReport both:\n\n\\[\nR_m^{\\rm object}\n\\]\n\nover object target bytes only, and\n\n\\[\nR_m^{\\rm whole}\n\\]\n\nover the full byte stream.\n\nNormalize for presentation as:\n\n\\[\n\\bar R_m\n=\nR_m / N_{\\rm object\\ events}\n\\]\n\nin **excess bits per object event**.\n\nThe qualification endpoint is:\n\n\\[\n\\boxed{\\bar R^{\\rm object}.}\n\\]\n\nLower is better.\n\nA negative value is allowed: a bounded policy may occasionally outperform the unbounded reference by avoiding harmful stale retrieval.\n\n---\n\n# 10. Secondary endpoints\n\n## 10.1 Utility-decile survival\n\nUsing evaluator-only realized future utility from `TRACE-U`, divide logical identities into deciles.\n\nAt the final checkpoint report for every policy:\n\n- fraction retaining a consolidated slot;\n- fraction retaining at least one exact trace;\n- learned-rendering accuracy;\n- held-out alias accuracy.\n\nThis reveals whether a policy preferentially preserves the genuinely valuable tail.\n\n## 10.2 Recurrence-class survival\n\nFor `HOT/WARM/COOL/FADING`, report:\n\n- consolidated survival;\n- exact-only survival;\n- held-out \\(a_2\\) accuracy;\n- regret contribution.\n\nThis is descriptive; recurrence class is never learner-visible.\n\n## 10.3 Demotion and recovery\n\nTrack:\n\n- slot \\(\\to\\) trace demotions;\n- trace evictions;\n- re-consolidations;\n- number of object events needed to regain held-out transfer after re-consolidation.\n\nAll replacement policies share the same recovery machinery.\n\n## 10.4 Ordinary collateral utility\n\nReport:\n\n- nontarget bpb;\n- whole-stream bpb;\n- candidate-hit rate on nontarget contexts.\n\nA replacement policy may not improve target regret by causing widespread false memory authority.\n\n## 10.5 Final held-out return\n\nAfter \\(S_8\\), perform read-only probes for all 768 identities:\n\n- \\(a_0\\);\n- \\(a_1\\);\n- never-trained \\(a_2\\).\n\nDo not learn from these probes.\n\nReport exact and held-out accuracy by future-utility decile and recurrence class.\n\n---\n\n# 11. Qualification gates\n\nAll thresholds are locked before V40 science outcomes are observed.\n\n## Q0 — V39 substrate invariants\n\nOn two full engineering seeds before science:\n\n- TRACE-CORR predictions and consolidation transitions reproduce the locked V39 implementation;\n- hard capacities and canonical semantic-state accounting match;\n- probes are read-only;\n- branch policies share identical base state.\n\nFailure stops V40.\n\n## Q1 — task-validity screens\n\nP0–P3 in Section 7 all pass.\n\nFailure yields:\n\n\\[\n\\texttt{task\\_invalid}\n\\]\n\nand replacement-policy gates are not interpreted.\n\n## Q2 — literal boundedness\n\nFor every event and every bounded arm:\n\n\\[\n\\text{traces}\\le512,\n\\qquad\n\\text{slots}\\le384,\n\\qquad\n\\text{anchors}\\le768.\n\\]\n\nNo hidden tombstones, unbounded eligibility tables, replay logs, future-value tables or historical member lists may affect deployable predictions.\n\nFailure voids V40 regardless of accuracy.\n\n## Q3 — acquisition remains valid\n\nAcross all 768 intended identities:\n\n- \\(\\ge90\\%\\) mean exact learned-rendering accuracy after consolidation opportunity;\n- every seed \\(\\ge80\\%\\).\n\nReplacement cannot win by failing to learn newcomers.\n\n## Q4 — UTILITY-RATE beats LRU on object regret\n\nDefine:\n\n\\[\n\\Delta_{U-LRU}\n=\n\\bar R_U^{\\rm object}\n-\n\\bar R_{LRU}^{\\rm object}.\n\\]\n\nRequired:\n\n- paired mean \\(\\le -0.05\\) bits/object-event;\n- paired 95% upper bound \\(<0\\).\n\n## Q5 — UTILITY-RATE beats LFU-RATE\n\nSame requirement:\n\n\\[\n\\Delta_{U-LFU}\\le -0.05\n\\]\n\nmean and paired upper interval \\(<0\\).\n\nIf LFU-RATE matches UTILITY-RATE, the extra base-relative utility ledger has not earned its complexity.\n\n## Q6 — slot-specific utility beats shuffled utility\n\nPrimary specificity veto:\n\n\\[\n\\Delta_{U-SHUFFLED}\n=\n\\bar R_U^{\\rm object}\n-\n\\bar R_{SHUFFLED}^{\\rm object}.\n\\]\n\nRequired:\n\n- paired mean \\(\\le -0.05\\) bits/object-event;\n- paired 95% upper bound \\(<0\\).\n\nIf Q6 fails:\n\n\\[\n\\boxed{\\text{the slot-specific utility mechanism is not established}.}\n\\]\n\nEven if UTILITY-RATE has the best raw score, the mechanistic claim is rejected.\n\n## Q7 — high-value survival advantage\n\nOn evaluator-only top-quartile future-utility identities:\n\nUTILITY-RATE must beat both LRU and LFU-RATE on final held-out \\(a_2\\) accuracy by:\n\n- mean advantage \\(\\ge10\\) percentage points against at least one comparator;\n- paired 95% lower interval \\(>0\\).\n\nThis is an interpretability gate: lower regret should correspond to keeping valuable generalized memories, not merely shifting errors elsewhere.\n\n## Q8 — whole-stream collateral noninferiority\n\nRelative to the best of LRU/LFU-RATE:\n\n- whole-stream bpb no worse by more than **0.05**;\n- nontarget bpb no worse by more than **0.05**.\n\nUTILITY-RATE cannot qualify by improving object-memory regret while degrading ordinary prediction.\n\n## Q9 — graceful degradation survives\n\nAmong objects whose consolidated slot was evicted:\n\n- exact learned-rendering accuracy from the demoted trace tier must remain \\(\\ge70\\%\\) at the first read-only post-eviction probe before trace eviction;\n- held-out alias accuracy should drop materially relative to exact accuracy.\n\nThe second condition is descriptive and verifies that demotion really means:\n\n\\[\n\\text{generalization lost before exact memory}.\n\\]\n\n## Q10 — recovery is functional\n\nFor evicted identities that subsequently receive two distinct learned-rendering events and re-consolidate:\n\n- held-out \\(a_2\\) accuracy after re-consolidation \\(\\ge90\\%\\) mean;\n- no evaluator identity metadata is used.\n\nThis ensures replacement does not irreversibly destroy the V39 identity mechanism.\n\n---\n\n# 12. Decision logic\n\n## Outcome A — utility-based persistence established\n\nRequirements:\n\n- Q0–Q10 pass;\n- especially Q4, Q5 and Q6.\n\nConclusion:\n\n> Realized marginal predictive value is a causally supported persistence signal under genuine memory scarcity.\n\nCarry `UTILITY-RATE` forward as the default consolidated-slot replacement rule.\n\n## Outcome B — recurrence is enough\n\nPattern:\n\n- LFU-RATE matches UTILITY-RATE;\n- shuffled utility may also match.\n\nConclusion:\n\n> The environment's value structure is adequately captured by recurrence rate; base-relative utility accounting is unnecessary.\n\nAdopt the simpler LFU-RATE rule.\n\n## Outcome C — recency is enough\n\nLRU matches or beats utility/frequency.\n\nConclusion:\n\n> Under this workload, recent use dominates long-run utility accounting.\n\nDo not add a utility ledger.\n\n## Outcome D — utility appears best but fails shuffled specificity\n\nUTILITY-RATE has the best raw regret but Q6 fails.\n\nConclusion:\n\n> Adaptive score dispersion may help, but the claim that each slot's own causal utility history deserves retention is not established.\n\nDo not make slot-specific utility load-bearing.\n\n## Outcome E — no causal policy approaches the oracle\n\nIf the optional oracle is dramatically better while all causal policies cluster together:\n\n> The workload contains value that becomes knowable too late for the tested local statistics.\n\nThe next question becomes richer predictive survival models, not another threshold sweep.\n\n## Outcome F — task invalid\n\nIf P0–P3 fail, redesign the scarcity stream before interpreting replacement rules.\n\nNo mechanism search is allowed on an invalid scarcity task.\n\n---\n\n# 13. What V40 must not do\n\nV40 must not add:\n\n- phase/frozen-phase representation changes;\n- superposed overflow memory;\n- new admission rules;\n- PBF authority changes;\n- revision-confidence mechanisms;\n- utility decay/horizon sweeps;\n- capacity sweeps;\n- trace-size sweeps;\n- learned eviction networks;\n- replay;\n- task IDs or evaluator-known importance labels;\n- post-hoc threshold tuning.\n\nThose are possible later questions, but they destroy causal attribution here.\n\n---\n\n# 14. Compute plan\n\nThe expensive shared base trajectory is generated once per seed.\n\nWithin a seed:\n\n1. construct the fixed V40 stream;\n2. run the shared base once;\n3. branch auxiliary memory policies from identical initial state;\n4. use the same byte events and same base predictions for:\n   - RANDOM;\n   - LRU;\n   - LFU-RATE;\n   - UTILITY-RATE;\n   - SHUFFLED-UTILITY;\n   - TRACE-U reference.\n\nThus:\n\n\\[\n8\\text{ base streams}\n\\]\n\nand at most:\n\n\\[\n8\\times6=48\n\\]\n\nmemory-policy branches.\n\nNo separate large grid.\n\nIf runtime is materially larger than V39, first run:\n\n- two engineering seeds;\n- then four science seeds as a completion checkpoint;\n- then all eight without changing any thresholds.\n\nPartial science results must not alter the protocol.\n\n---\n\n# 15. Required output files\n\nKeep the archive compact.\n\nRequired:\n\n- `SUMMARY.md`\n- `REPORT.json`\n- `MANIFEST.json`\n- `PROTOCOL.md`\n- `TERMINAL.csv`\n- `REGRET_BY_POLICY.csv`\n- `UTILITY_DECILES.csv`\n- `RECURRENCE_CLASS_SURVIVAL.csv`\n- `EVICTION_SUMMARY.csv`\n- `DEMOTION_RECOVERY.csv`\n- `OCCUPANCY_CURVES.csv`\n- `SHUFFLED_UTILITY_AUDIT.json`\n- `STATE_COSTS.json`\n\nDo **not** dump full per-byte prediction logs.\n\nFor diagnostics, aggregate online into the files above.\n\nThe shuffled-control audit must verify at each seed:\n\n- same number of replacement events as UTILITY-RATE;\n- identical capacity;\n- identical current utility-score multiset before shuffling, to numerical tolerance;\n- substantial reassignment among slot identities;\n- no use of future information.\n\n---\n\n# 16. Interpretation limits\n\nEven a successful V40 would establish only:\n\n- one synthetic byte-stream scarcity regime;\n- one specific TRACE-CORR object-memory substrate;\n- one hard capacity point;\n- one family of causal replacement statistics.\n\nIt would **not** establish:\n\n- optimal human-like forgetting;\n- optimal capacity scaling;\n- natural-language long-term memory;\n- biological equivalence;\n- that utility rate is universally optimal;\n- that the fly uses such a ledger.\n\nThe scientific claim is deliberately narrower:\n\n\\[\n\\boxed{\n\\text{Under genuine online memory scarcity, does a memory's realized marginal predictive value causally improve which persistent objects survive?}\n}\n\\]\n\n---\n\n# 17. If V40 succeeds\n\nThe most interesting V41 would then be architectural rather than another replacement-policy tournament:\n\n\\[\n\\boxed{\n\\text{precise utility-managed slots}\n+\n\\text{bounded superposed overflow for the evicted long tail}.\n}\n\\]\n\nThat would finally merge the two positive V39 branches:\n\n- TRACE-CORR / earned explicit ownership;\n- bounded superposed memory.\n\nThe hypothesis would be:\n\n> high-value identities retain precise slots; lower-value evicted identities decay into a fixed-size distributed long-tail store rather than disappearing entirely.\n\nBut V40 must first establish what deserves the precise tier.\n"
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

def validate_identity(existing,expected):
    if existing!=expected:raise RuntimeError('Source/protocol/plan/stream identity mismatch. Use a new output directory.')

def record_write(path,data,identity):
    envelope={'identity_sha256':sha(identity),'data':data,'sha256':sha(data)};write_json(path,envelope)

def record_read(path,identity):
    x=json.loads(path.read_text())
    if x.get('identity_sha256')!=sha(identity) or x.get('sha256')!=sha(x.get('data')):
        raise RuntimeError('Corrupt or mismatched completed record: '+path.name)
    return x['data']

V39Memory=Memory

IMPLEMENTATION_NOTES = '''
# Frozen implementation resolutions
- Stream time is the observed byte count, including nontarget bytes.
- Each cohort has 32 identities of each class (2 in smoke), randomly assigned.
  COOL receives one service on its 2nd, 4th, ... eligible epoch. FADING's
  first two eligible epochs each have two services. Recurrence aliases alternate
  independently of admission, starting at a0; valuation also advances this counter.
- The interleaving gap is one shuffled service epoch for already introduced
  identities, using the current epoch schedule. K1 uses one newline as its gap.
  Gap services count as actual workload services but not a new eligible epoch.
- V is one recurrence for ALL first 384 introduced identities, in fixed shuffled
  order. Conditioning the shared input on branch-specific residency would leak
  memory policy into the workload. P2 still checks every actual incumbent.
- The common scarcity onset for paired regret is the first replacement in ANY
  bounded branch. Whole and object excess loss use that same byte boundary.
- E2 gain seeds G from the frozen trace and base; H starts at zero, since the
  support-1 trace did not deploy generalized authority on E2. Postbirth H counts
  all deployed uses, even harmful ones. E2 credit is not counted again.
- Demoted anchors get fresh trace insertion ids, last=now, hits=1, equal copies
  of the slot value. Old ledger is destroyed. Promotion removes both anchors
  from the trace pool if a demotion recreated the current context mid-allocation.
- V39 exact-trace hits remain exact-only: two demoted exact anchors do NOT
  automatically re-consolidate. Recovery is measured when ordinary inherited
  corroboration occurs; an empty recovery sample is not a pass for Q10.
- Independent eviction causes divergent residency, admission and replacement
  schedules. SHUFFLED-UTILITY permutes its OWN current ledger within strata;
  its score multiset is preserved exactly. Cross-arm counts/timing/multisets
  are audited, not forced. If these fail, the design's strict matched-control
  audit is unmet and mechanistic qualification is withheld. No shadow admission.
- Quartiles are stable rank quartiles of age, ties by allocation id. Permutations
  may include fixed points. Substantial reassignment means >=50% of slots in
  nonsingleton strata change score-source identity, pooled over evictions.
- P0/P1/P2/P3 are required per seed. P0's resident count counts intended identities
  with both learned anchors in the same slot, not incidental nontarget slots.
- Future utility is accumulated in TRACE-U after common scarcity onset. For
  a slot with multiple intended identities among anchors, credit is divided
  equally among them; nontarget-only slots are excluded from identity deciles.
  Decile/top-quartile ties use identity index, evaluator only.
- Q7 requires positive paired mean against BOTH comparators, with >=10pp and
  positive lower 95% bound against at least one. Q8 uses the better comparator
  separately for each aggregate loss. Q9/Q10 require nonempty per-seed samples.
- All CIs are two-sided paired Student-t intervals across eight seeds. No
  partial or smoke results can qualify. No optional future oracle is included.
- Q0 verifies two full historical V39 streams under a compatibility replacement
  mode; V40 demotion/utility changes are tested separately with forced overflow.
  That mode is used only by engineering checks, never scientific branches.
- Resume is at completed-seed boundaries; an interrupted seed restarts. Scientific
  seeds are fixed, with no sweep or automatic second experiment.
'''

class ScarcityMemory(V39Memory):
    def __init__(self, policy, seed, trace_cap=512, slot_cap=384, compatibility=False):
        super().__init__('TRACE-CORR',seed,trace_cap,slot_cap)
        self.replacement=policy;self.compatibility=compatibility
        self.bounded=policy!='TRACE-U'
        if not self.bounded:self.trace_cap=2**63-1;self.slot_cap=2**63-1
        self.rng=np.random.default_rng(seed+400009)
        self.audit=collections.Counter()
        self._birth_gain=0.

    def priority(self,j):
        s=self.slots[j];age=self.clock-s['created']+1
        return s['gain']/age

    def victim(self):
        ids=sorted(self.slots)
        if self.compatibility:
            return min(ids,key=lambda j:(self.slots[j]['hits'],self.slots[j]['last'],j))
        if self.replacement=='RANDOM':return ids[int(self.rng.integers(len(ids)))]
        if self.replacement=='LRU':return min(ids,key=lambda j:(self.slots[j]['last'],j))
        if self.replacement=='LFU-RATE':
            scores={j:self.slots[j]['uses']/(self.clock-self.slots[j]['created']+1) for j in ids}
        else:scores={j:self.priority(j) for j in ids}
        if self.replacement=='SHUFFLED-UTILITY':
            before=sorted(scores.values())
            byage=sorted(ids,key=lambda j:(self.clock-self.slots[j]['created'],j))
            quart={j:min(3,4*k//len(ids)) for k,j in enumerate(byage)}
            bins=collections.defaultdict(list)
            for j in ids:
                h=self.slots[j]['uses'];b=0 if h<2 else 1 if h<4 else 2 if h<8 else 3
                bins[(b,quart[j])].append(j)
            for group in bins.values():
                self.audit['stratum_slots']+=len(group)
                if len(group)==1:self.audit['singleton_slots']+=1;continue
                perm=self.rng.permutation(len(group));old=[scores[j] for j in group]
                self.audit['permutable_slots']+=len(group)
                for k,j in enumerate(group):
                    scores[j]=old[int(perm[k])]
                    self.audit['reassigned_slots']+=int(perm[k]!=k)
            assert sorted(scores.values())==before
            self.audit['multiset_checks']+=1
        return min(ids,key=lambda j:(scores[j],self.slots[j]['last'],j))

    def _drop_slot(self,j,events,reason='capacity'):
        s=self.slots[j]
        saved=[(c,s['value'].copy()) for c in s['anchors']]
        super()._drop_slot(j,events,reason)
        if self.compatibility:return
        for c,v in saved:
            if c in self.traces:self._drop_trace(c,events,'demotion_collision')
            while len(self.traces)>=self.trace_cap:
                c0=min(self.traces,key=lambda x:(self.traces[x]['last'],self.traces[x]['id']))
                self._drop_trace(c0,events)
            self.traces[c]={'value':v,'hits':1,'last':self.clock,'created':self.clock,'id':self._id()}
            events.append({'event':'demotion','context':c.hex()})
        self.audit['demoted_slots']+=1

    def _new_slot(self,cs,value,events,hits=1):
        if self.bounded and len(self.slots)>=self.slot_cap:
            self.audit['replacements']+=1
            self._drop_slot(self.victim(),events)
        # A wrong authoritative retrieval can admit a trace resembling an old slot.
        # Demoting that slot must not leave an exact trace shadowing its new anchor.
        for c in cs:
            if c in self.traces:self._drop_trace(c,events,'new_anchor_cleanup')
        j=super()._new_slot(cs,value,events,hits)
        self.slots[j].update(gain=self._birth_gain,uses=0)
        self.audit['allocations']+=1
        return j

    def observe(self,p,y,schedule=None):
        # Base and candidate losses were frozen by the caller, before either write.
        if p['type']=='slot' and p['authority']:
            s=self.slots[p['id']];s['gain']+=p['base_loss']-loss(p['scores'],y);s['uses']+=1
        self._birth_gain=0.
        if p['type']=='trace' and not p['exact'] and p['base_top']!=y and p['candidate_top']==y:
            self._birth_gain=p['base_loss']-loss(self.traces[p['id']]['value'],y)
        events=super().observe(p,y,schedule)
        self._birth_gain=0.
        return events

    def cost(self,overhead=False):
        r=super().cost(overhead)
        # G float64 + deployed-use uint64; birth already counted in V39 metadata.
        r['semantic_bytes']+=16*len(self.slots)+128
        r['utility_ledger_bytes']=16*len(self.slots)
        r['accounting_notes']='Includes ledger and fixed RNG/counters; no learner history tables. Temporary lookup arrays excluded.'
        return r

    def state_hash(self):
        return sha({'substrate':super().state_hash(),
                    'ledger':[(j,s['gain'],s['uses']) for j,s in self.slots.items()],
                    'replacement':self.replacement})


def stream40(seed,smoke=False):
    rr=np.random.default_rng(seed+400081);cohort=8 if smoke else 128;n=6*cohort
    alphabet=np.frombuffer(b'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ',dtype=np.uint8)
    keys=[];seen=set()
    while len(keys)<n:
        k=bytes(rr.choice(alphabet,size=6))
        if k not in seen:keys.append(k);seen.add(k)
    labels=rr.choice(np.frombuffer(b'abcdefghijklmnopqrstuvwx',dtype=np.uint8),size=n).astype(int)
    classes=[]
    for _ in range(6):classes.extend(rr.permutation(['HOT','WARM','COOL','FADING']*(cohort//4)).tolist())
    counts=np.zeros(n,dtype=int);rows=[];checkpoints=[]
    def add(i,a,stage):
        rows.append({'i':int(i),'alias':int(a),'stage':stage,'raw':context(keys[i],a)+bytes([int(labels[i])])+b'\n'})
    def service(k,limit,stage):
        batch=[]
        for i in range(limit):
            age=k-i//cohort
            cl=classes[i]
            times=4 if cl=='HOT' else 2 if cl=='WARM' else int(age%2==0) if cl=='COOL' else 2*int(age<=2)
            batch.extend([i]*times)
        for i in rr.permutation(batch):
            add(i,int(counts[i]%2),stage);counts[i]+=1
    for k in range(1,9):
        if k<=6:
            for i in rr.permutation(np.arange((k-1)*cohort,k*cohort)):add(i,0,f'K{k}_E1')
            if k==1:rows.append({'i':None,'alias':None,'stage':'gap','raw':b'\n'})
            else:service(k,(k-1)*cohort,f'K{k}_gap')
            for i in rr.permutation(np.arange((k-1)*cohort,k*cohort)):add(i,1,f'K{k}_E2')
            checkpoints.append((len(rows),f'K{k}'))
        service(k,min(k,6)*cohort,f'S{k}')
        checkpoints.append((len(rows),f'S{k}'))
        if k==3:
            for i in rr.permutation(3*cohort):add(i,int(counts[i]%2),'V');counts[i]+=1
            checkpoints.append((len(rows),'V'))
    raw=b''.join(r['raw'] for r in rows)
    assert all(r['alias']!=2 for r in rows)
    return dict(keys=keys,labels=labels,classes=classes,rows=rows,raw_sha256=sha(raw),
                byte_events=len(raw),cohort=cohort,checkpoints=dict(checkpoints))


def probe40(m,enc,base,key,y):
    out=[]
    for a in (0,1,2):
        c=context(key,a);z,b=enc.encode(c);p=m.lookup(c,base.scores(z,b))
        out.append(int(np.argmax(p['scores'])==y))
    return out


def resident_identity(m,key):
    a=context(key,0);b=context(key,1)
    return a in m.anchors and b in m.anchors and m.anchors[a]==m.anchors[b]


def trial40(seed,smoke=False,observers=True):
    started=time.time();s=stream40(seed,smoke);n=len(s['keys']);cap=3*s['cohort'];tcap=32 if smoke else 512
    enc=Encoder(seed,'last_fd');base=Learner('full')
    memories={p:ScarcityMemory(p,seed,tcap,cap) for p in POLICIES}
    bycontext={context(k,a):i for i,k in enumerate(s['keys']) for a in (0,1)}
    agg={p:collections.Counter() for p in POLICIES};future=np.zeros(n)
    class_regret={p:collections.Counter() for p in POLICIES}
    acquired={p:[] for p in POLICIES};held_acq={p:[] for p in POLICIES}
    e2_consolidated=0;peak_intended=0;scarcity=None;p2={p:None for p in POLICIES[:-1]}
    eviction_times={p:hashlib.sha256() for p in POLICIES}
    eviction_counts={p:0 for p in POLICIES}
    demotion={p:[] for p in POLICIES};recovery={p:[] for p in POLICIES}
    # Evaluator history never reaches a Memory method.
    evicted={p:{} for p in POLICIES};evictionstats={p:collections.Counter() for p in POLICIES}
    occupancy=[];h=bytearray();clock=0;control_same=0;control_comparisons=0
    for rowno,row in enumerate(s['rows'],1):
        ident=row['i'];stage=row['stage']
        for pos,y in enumerate(row['raw']):
            clock+=1;c=bytes(h);z,b=enc.encode(c);sb=base.scores(z,b);bl=loss(sb,y)
            preds={p:m.lookup(c,sb) for p,m in memories.items()}
            losses={p:loss(v['scores'],y) for p,v in preds.items()}
            object_event=ident is not None and pos==12
            forcing=[]
            for p,m in memories.items():
                v=preds[p]
                promotes=(v['type']=='trace' and not v['exact'] and v['base_top']!=y and v['candidate_top']==y)
                if m.bounded and promotes and len(m.slots)>=cap:
                    forcing.append(p)
                    if p2[p] is None:
                        p2[p]={'all_observed':all(vv['uses']>=1 for vv in m.slots.values()),
                               'unobserved':sum(vv['uses']==0 for vv in m.slots.values()),
                               'first_replacement_byte':clock,'stage':stage}
            if forcing and scarcity is None:scarcity=clock
            if 'SHUFFLED-UTILITY' in forcing:
                a=memories['UTILITY-RATE'];bmem=memories['SHUFFLED-UTILITY']
                av=sorted(a.priority(j) for j in a.slots);bv=sorted(bmem.priority(j) for j in bmem.slots)
                control_comparisons+=1
                control_same+=int(len(av)==len(bv) and np.allclose(av,bv,rtol=0,atol=1e-12))
            if scarcity is not None:
                pu=preds['TRACE-U'];mu=memories['TRACE-U']
                if pu['type']=='slot' and pu['authority']:
                    ii=sorted({bycontext[x] for x in mu.slots[pu['id']]['anchors'] if x in bycontext})
                    for i in ii:future[i]+=(bl-losses['TRACE-U'])/len(ii)
            all_events={}
            for p,m in memories.items():
                v=preds[p];v['base_loss']=bl
                a=agg[p];a['all_n']+=1;a['all_loss']+=losses[p]
                typ='object' if object_event else 'nontarget'
                a[typ+'_n']+=1;a[typ+'_loss']+=losses[p]
                a[typ+'_correct']+=int(np.argmax(v['scores'])==y)
                if not object_event:
                    a['nontarget_candidates']+=int(v['type'] is not None and not v['exact'])
                if scarcity is not None:
                    diff=losses[p]-losses['TRACE-U'];a['whole_regret']+=diff
                    if object_event:
                        a['scarce_object_n']+=1;a['object_regret']+=diff
                        class_regret[p][s['classes'][ident]]+=diff
                if object_event and ident in evicted[p]:
                    old=evicted[p][ident];old['aliases'].add(row['alias']);old['events']+=1
                events=m.observe(v,y);all_events[p]=events
                for e in events:
                    if e['event']=='slot_eviction':
                        eviction_counts[p]+=1;eviction_times[p].update(clock.to_bytes(8,'little'))
                        ii={bycontext.get(bytes.fromhex(x)) for x in e['contexts']}-{None}
                        for i in ii:
                            evicted[p][i]={'at':clock,'aliases':set(),'events':0}
                            evictionstats[p]['identity_evictions']+=1
                    elif e['event']=='trace_removal' and e['reason']=='capacity':evictionstats[p]['trace_evictions']+=1
                    elif e['event']=='promotion':
                        ii={bycontext.get(bytes.fromhex(x)) for x in e['contexts']}-{None}
                        for i in ii:
                            old=evicted[p].get(i)
                            if old and len(old['aliases'])>=2 and resident_identity(m,s['keys'][i]):
                                ans=probe40(m,enc,base,s['keys'][i],int(s['labels'][i])) if observers else [0,0,0]
                                recovery[p].append({'heldout':ans[2],'object_events':old['events']})
                                del evicted[p][i]
            if any(e['event'] in ('promotion','slot_eviction') for e in all_events['TRACE-U']):
                peak_intended=max(peak_intended,sum(resident_identity(memories['TRACE-U'],k) for k in s['keys']))
            # Base is learned exactly once from the same pre-update state for all branches.
            base.observe(z,b,y)
            if observers:
                for p,events in all_events.items():
                    m=memories[p]
                    for e in events:
                        if e['event']!='slot_eviction':continue
                        ii={bycontext.get(bytes.fromhex(x)) for x in e['contexts']}-{None}
                        for i in ii:
                            cs=[context(s['keys'][i],a) for a in (0,1)]
                            available=[x for x in cs if x in m.traces]
                            if not available:continue
                            exact=mean(int(np.argmax(m.traces[x]['value'])==int(s['labels'][i])) for x in available)
                            held=probe40(m,enc,base,s['keys'][i],int(s['labels'][i]))[2]
                            demotion[p].append({'exact':exact,'heldout':held,'anchors_tested':len(available)})
            if object_event and stage.endswith('_E2'):
                e2_consolidated+=int(resident_identity(memories['TRACE-U'],s['keys'][ident]))
                if observers:
                    for p,m in memories.items():
                        before=m.state_hash();ans=probe40(m,enc,base,s['keys'][ident],int(s['labels'][ident]))
                        assert before==m.state_hash()
                        acquired[p].append(mean(ans[:2]));held_acq[p].append(ans[2])
            h.append(y)
            if len(h)>12:del h[0]
        if rowno in s['checkpoints']:
            name=s['checkpoints'][rowno]
            mu=memories['TRACE-U'];resident=sum(resident_identity(mu,k) for k in s['keys'])
            peak_intended=max(peak_intended,resident)
            for p,m in memories.items():
                occupancy.append({'seed':seed,'policy':p,'checkpoint':name,'byte':clock,**m.cost()})
            print(f'  seed {seed} {name}: byte {clock}/{s["byte_events"]}',flush=True)
    terminal=[];order=sorted(range(n),key=lambda i:(future[i],i));ranks={i:k for k,i in enumerate(order)}
    for p,m in memories.items():
        before=m.state_hash()
        for i,k in enumerate(s['keys']):
            ans=probe40(m,enc,base,k,int(s['labels'][i])) if observers else [0,0,0]
            slot=resident_identity(m,k);trace=any(context(k,a) in m.traces for a in (0,1))
            terminal.append({'seed':seed,'policy':p,'identity':i,'class':s['classes'][i],
                             'future_gain':float(future[i]),'decile':min(9,10*ranks[i]//n)+1,
                             'top_quartile':ranks[i]>=3*n//4,'slot':int(slot),'trace':int(trace),
                             'exact_only':int(trace and not slot),'exact':mean(ans[:2]),'heldout':ans[2]})
        assert before==m.state_hash()
    q=np.quantile(future,[.25,.5,.75,.8,.9])
    p3=bool(q[4]>=2*q[1] and q[2]>q[0] and np.mean(future>0)>=.2 and np.mean(future<.5*q[3])>=.2)
    shuffle=memories['SHUFFLED-UTILITY'].audit
    return {'seed':seed,'smoke':smoke,'stream_sha256':s['raw_sha256'],'byte_events':clock,'objects':n,
            'scarcity_byte':scarcity,'base_hash':base.state_hash(),
            'final_hashes':{p:m.state_hash() for p,m in memories.items()},
            'aggregates':{p:dict(a) for p,a in agg.items()},'class_regret':{p:dict(a) for p,a in class_regret.items()},
            'acquisition':{p:mean(a) for p,a in acquired.items()},'heldout_acquisition':{p:mean(a) for p,a in held_acq.items()},
            'p0_e2_fraction':e2_consolidated/n,'peak_intended_unbounded':peak_intended,'p2':p2,
            'p3':{'passed':p3,'q25':float(q[0]),'median':float(q[1]),'q75':float(q[2]),'q80':float(q[3]),'q90':float(q[4]),
                  'positive_fraction':float(np.mean(future>0)),'low_fraction':float(np.mean(future<.5*q[3]))},
            'terminal':terminal,'occupancy':occupancy,'demotion':demotion,'recovery':recovery,
            'eviction_stats':{p:dict(v) for p,v in evictionstats.items()},'eviction_counts':eviction_counts,
            'costs':{p:m.cost(True) for p,m in memories.items()},'peaks':{p:m.peak for p,m in memories.items()},
            'shuffle':{'within_branch_multiset_exact':True,**dict(shuffle),
                'cross_arm_score_multiset_checks':control_comparisons,'cross_arm_score_multiset_matches':control_same,
                'same_replacement_count':eviction_counts['UTILITY-RATE']==eviction_counts['SHUFFLED-UTILITY'],
                'same_replacement_timing':eviction_times['UTILITY-RATE'].hexdigest()==eviction_times['SHUFFLED-UTILITY'].hexdigest(),
                'same_capacity':True,'future_information_used':False},
            'elapsed_seconds':time.time()-started}

def historical_preflight(seed,smoke=False):
    ref=V39Memory('TRACE-CORR',seed)
    new=ScarcityMemory('UTILITY-RATE',seed,compatibility=True)
    enc=Encoder(seed);base=Learner('full');other=Learner('full');s=make_stream(seed,smoke=smoke)
    h=bytearray();digest=hashlib.sha256();probes=0
    for t,y in enumerate(s['raw'],1):
        c=bytes(h);z,b=enc.encode(c);sb=base.scores(z,b)
        assert np.array_equal(sb,other.scores(z,b))
        a=ref.lookup(c,sb);v=new.lookup(c,sb)
        for k in ('c','type','id','exact','sim','support','candidate_top','base_top','authority'):assert a[k]==v[k],k
        assert np.array_equal(a['scores'],v['scores'])
        digest.update(v['scores'].tobytes());v['base_loss']=loss(sb,y)
        assert ref.observe(a,y)==new.observe(v,y)
        base.observe(z,b,y);other.observe(z,b,y)
        if t%4096==0 or t==len(s['raw']):
            assert ref.semantic()==new.semantic()
            assert set(ref.traces)==set(new.traces)
            for x in ref.traces:
                for k in ref.traces[x]:
                    if k=='value':assert np.array_equal(ref.traces[x][k],new.traces[x][k])
                    else:assert ref.traces[x][k]==new.traces[x][k]
            assert ref.cost()['semantic_bytes']+16*len(new.slots)+128==new.cost()['semantic_bytes']
            before=new.state_hash()
            for k in s['keys']:
                probe40(new,enc,base,k,int(s['labels'][s['keys'].index(k)]));probes+=3
            assert before==new.state_hash() and base.state_hash()==other.state_hash()
            print(f'  Q0 seed {seed}: {t}/{len(s["raw"])} bytes',flush=True)
        h.append(y)
        if len(h)>12:del h[0]
    return {'seed':seed,'passed':True,'full_length':not smoke,'events':len(s['raw']),
            'prediction_sha256':digest.hexdigest(),'base_sha256':base.state_hash(),'read_only_probes':probes,
            'reference_sha256':REFERENCE_SHA256}


def self_test40():
    checks=[]
    def ok(name,condition=True):
        if not condition:raise AssertionError(name)
        checks.append(name)
    sb=np.zeros(256);sb[0]=1.;y=ord('a');key=b'ABC123';c0=context(key,0);c1=context(key,1)
    m=ScarcityMemory('UTILITY-RATE',974010,8,2)
    def step(m,c,y):
        p=m.lookup(c,sb);p['base_loss']=loss(sb,y);e=m.observe(p,y);return p,e
    p,e=step(m,c0,y);ok('E1 exact trace',c0 in m.traces and not m.slots)
    p=m.lookup(c1,sb);ok('support1 has no authority',p['type']=='trace' and not p['authority'])
    expected=loss(sb,y)-loss(m.traces[c0]['value'],y)
    step(m,c1,y);j=m.anchors[c0]
    ok('distinct confirmation promotes',m.anchors[c1]==j)
    ok('birth gain is frozen trace difference',m.slots[j]['gain']==expected)
    ok('birth is not deployed-use count',m.slots[j]['uses']==0)
    before=m.state_hash();m.lookup(context(key,2),sb);ok('lookup read only',before==m.state_hash())
    old=m.slots[j]['gain'];v=m.lookup(c0,sb);g=loss(sb,y)-loss(v['scores'],y)
    step(m,c0,y);ok('gain observes frozen deployed prediction',m.slots[j]['gain']==old+g and m.slots[j]['uses']==1)
    m._drop_slot(j,[]);ok('two exact demoted anchors',c0 in m.traces and c1 in m.traces and not m.slots)
    step(m,c0,y);step(m,c1,y);ok('exact recurrence does not secretly promote',not m.slots)
    ok('demoted values retain target',int(np.argmax(m.traces[c0]['value']))==y)
    for policy in POLICIES[:-1]:
        mm=ScarcityMemory(policy,974010,4,2)
        for i in range(8):
            k=f'{i:06d}'.encode();cs=[context(k,0),context(k,1)]
            v=np.zeros(256);v[y]=.6;mm.clock+=1;mm._birth_gain=float(i)
            mm._new_slot(cs,v,[]);mm.check()
        ok(policy+' hard bounds',len(mm.slots)==2 and len(mm.traces)<=4 and len(mm.anchors)==4)
        ok(policy+' cleanup',not(set(mm.anchors)&set(mm.traces)))
    mm=ScarcityMemory('SHUFFLED-UTILITY',974010,8,16)
    for i in range(16):
        v=np.zeros(256);mm._birth_gain=float(i);mm._new_slot([context(f'{i:06d}'.encode(),0)],v,[])
    for _ in range(10):mm.victim()
    ok('shuffle preserves multiset',mm.audit['multiset_checks']==10)
    ok('shuffle reassigns',mm.audit['reassigned_slots']>0)
    u=ScarcityMemory('TRACE-U',974010,1,1)
    for i in range(4):u._new_trace(context(f'{i:06d}'.encode(),0),y,[])
    ok('unbounded trace reference',len(u.traces)==4)
    s=stream40(974010,True);ok('stream deterministic',s['raw_sha256']==stream40(974010,True)['raw_sha256'])
    ok('heldout never trained',all(r['alias']!=2 for r in s['rows']))
    ok('one E1 and E2 each',sum(r['stage'].endswith('_E1') for r in s['rows'])==48 and sum(r['stage'].endswith('_E2') for r in s['rows'])==48)
    ok('valuation uniform',len([r for r in s['rows'] if r['stage']=='V'])==24)
    # Verify different scores, recency and frequency really select different victims.
    mm=ScarcityMemory('UTILITY-RATE',974010,8,2)
    for i in range(2):mm._new_slot([context(f'{i:06d}'.encode(),0)],np.zeros(256),[])
    ids=list(mm.slots);mm.clock=10
    mm.slots[ids[0]].update(gain=-1.,uses=8,last=10)
    mm.slots[ids[1]].update(gain=4.,uses=1,last=1)
    ok('utility includes harmful gain',mm.victim()==ids[0]);mm.replacement='LRU';ok('LRU selects oldest',mm.victim()==ids[1])
    mm.replacement='LFU-RATE';ok('LFU-rate selects least rate',mm.victim()==ids[1])
    return {'passed':True,'count':len(checks),'checks':checks}


def report40(records,seeds,p0,smoke):
    complete=len(records)==len(seeds);interpretable=complete and not smoke
    def val(r,p,name):
        a=r['aggregates'][p]
        if name=='regret':return a.get('object_regret',0)/max(1,a.get('scarce_object_n',0)) if r['scarcity_byte'] else None
        return a.get(name+'_loss',0)/max(1,a.get(name+'_n',0))
    def paired(a,b,name):return stat([None if val(r,a,name) is None or val(r,b,name) is None else val(r,a,name)-val(r,b,name) for r in records])
    def supported(st,sign,threshold):
        return bool(st['n']==len(seeds) and st['mean'] is not None and
                    (st['mean']<=threshold and st['high'] is not None and st['high']<0 if sign<0 else
                     st['mean']>=threshold and st['low'] is not None and st['low']>0))
    task=[]
    for r in records:
        cap=24 if smoke else 384
        p0v=(r['p0_e2_fraction']>=.95 and (r['acquisition']['TRACE-U'] or 0)>=.9 and (r['heldout_acquisition']['TRACE-U'] or 0)>=.9)
        p1v=all(r['peaks'][p]['slots']==cap and r['eviction_counts'][p]>=(12 if smoke else 192) for p in POLICIES[:-1]) and r['peak_intended_unbounded']>1.75*cap
        p2v=all(x is not None and x['all_observed'] and x['stage'].startswith('K4') for x in r['p2'].values())
        task.append({'seed':r['seed'],'P0':p0v,'P1':p1v,'P2':p2v,'P3':r['p3']['passed']})
    gates={};gates['Q0']=len(p0)==2 and all(x['passed'] and x['full_length'] for x in p0)
    gates['Q1']=bool(complete and all(all(x[k] for k in ('P0','P1','P2','P3')) for x in task))
    gates['Q2']=all(all(r['peaks'][p]['traces']<=(32 if smoke else 512) and r['peaks'][p]['slots']<=(24 if smoke else 384) and r['peaks'][p]['anchors']<=2*(24 if smoke else 384) for p in POLICIES[:-1]) for r in records) and complete
    gates['Q3']=bool(complete and all(mean(r['acquisition'][p] for r in records)>=.9 and all((r['acquisition'][p] or 0)>=.8 for r in records) for p in POLICIES[:-1]))
    contrasts={}
    for q,p in (('Q4','LRU'),('Q5','LFU-RATE'),('Q6','SHUFFLED-UTILITY')):
        contrasts[q]=paired('UTILITY-RATE',p,'regret');gates[q]=supported(contrasts[q],-1,-.05)
    surv={}
    for p in ('LRU','LFU-RATE'):
        diffs=[]
        for r in records:
            a=mean(x['heldout'] for x in r['terminal'] if x['policy']=='UTILITY-RATE' and x['top_quartile'])
            b=mean(x['heldout'] for x in r['terminal'] if x['policy']==p and x['top_quartile'])
            diffs.append(a-b)
        surv[p]=stat(diffs)
    gates['Q7']=bool(complete and all(x['mean'] is not None and x['mean']>0 for x in surv.values()) and any(supported(x,1,.1) for x in surv.values()))
    collateral={name:{p:stat([val(r,p,name) for r in records]) for p in ('UTILITY-RATE','LRU','LFU-RATE')} for name in ('all','nontarget')}
    gates['Q8']=bool(complete and all(v['UTILITY-RATE']['mean']<=min(v['LRU']['mean'],v['LFU-RATE']['mean'])+.05 for v in collateral.values()))
    dg={};rg={}
    for p in POLICIES[:-1]:
        dg[p]=stat([mean(x['exact'] for x in r['demotion'][p]) for r in records])
        rg[p]=stat([mean(x['heldout'] for x in r['recovery'][p]) for r in records])
    gates['Q9']=bool(complete and all(v['n']==len(seeds) and v['mean']>=.7 for v in dg.values()))
    gates['Q10']=bool(complete and all(v['n']==len(seeds) and v['mean']>=.9 for v in rg.values()))
    audit=all(r['shuffle']['same_replacement_count'] and r['shuffle']['same_replacement_timing'] and
              r['shuffle']['cross_arm_score_multiset_matches']==r['shuffle']['cross_arm_score_multiset_checks'] and
              r['shuffle'].get('permutable_slots',0)>0 and
              r['shuffle'].get('reassigned_slots',0)/r['shuffle']['permutable_slots']>=.5 for r in records) and complete
    verdict='engineering_only' if smoke else 'incomplete' if not complete else 'task_invalid' if not gates['Q1'] else 'qualified' if all(gates.values()) and audit else 'not_established'
    return {'version':VERSION,'complete':complete,'science':not smoke,'verdict':verdict,'completed_seeds':len(records),'expected_seeds':len(seeds),
            'task_screens':task,'gates':gates,'policy_gates_interpretable':interpretable and gates['Q1'],
            'strict_shuffled_matching_audit_passed':audit,'regret_contrasts':contrasts,'high_value_survival_contrasts':surv,
            'collateral':collateral,'demotion_exact':dg,'recovery_heldout':rg,
            'per_policy':{p:{'object_regret':stat([val(r,p,'regret') for r in records]),'whole_bpb':stat([val(r,p,'all') for r in records]),
                              'nontarget_bpb':stat([val(r,p,'nontarget') for r in records])} for p in POLICIES},
            'interpretation':'Failure to beat a comparator is not equivalence. No automatic promotion of LRU/LFU from a nonsignificant contrast.'}


def package40(root,records,rep,identity,p0):
    write_json(root/'REPORT.json',rep);atomic(root/'PROTOCOL.md',PROTOCOL+'\n'+IMPLEMENTATION_NOTES)
    atomic(root/Path(__file__).name,Path(__file__).read_text())
    terminal=[x for r in records for x in r['terminal']];csv_output(root/'TERMINAL.csv',terminal)
    regret=[];evictions=[];recovery=[];grouped=[];deciles=[]
    for r in records:
        for p in POLICIES:
            a=r['aggregates'][p];den=a.get('scarce_object_n',0)
            regret.append({'seed':r['seed'],'policy':p,'scarcity_byte':r['scarcity_byte'],
                           'object_events_after_scarcity':den,'object_regret_bits':a.get('object_regret',0),
                           'excess_bits_per_object':a.get('object_regret',0)/den if den else None,
                           'whole_regret_bits':a.get('whole_regret',0),'whole_bpb':a['all_loss']/a['all_n'],
                           'nontarget_bpb':a['nontarget_loss']/a['nontarget_n'],
                           'nontarget_candidate_hit_rate':a.get('nontarget_candidates',0)/a['nontarget_n']})
            evictions.append({'seed':r['seed'],'policy':p,'slot_replacements':r['eviction_counts'][p],**r['eviction_stats'][p]})
            dd=r['demotion'][p];rr=r['recovery'][p]
            recovery.append({'seed':r['seed'],'policy':p,'demotion_probes':len(dd),'demotion_exact':mean(x['exact'] for x in dd),
                             'demotion_heldout':mean(x['heldout'] for x in dd),'reconsolidations':len(rr),
                             'recovery_heldout':mean(x['heldout'] for x in rr),'object_events_to_recovery':mean(x['object_events'] for x in rr)})
            for field,values,out in [('class',('HOT','WARM','COOL','FADING'),grouped),('decile',range(1,11),deciles)]:
                for v in values:
                    rows=[x for x in r['terminal'] if x['policy']==p and x[field]==v]
                    out.append({'seed':r['seed'],'policy':p,field:v,'n':len(rows),**{k:mean(x[k] for x in rows) for k in ('slot','trace','exact_only','exact','heldout')},
                                **({'regret_bits':r['class_regret'][p].get(v,0)} if field=='class' else {})})
    for name,rows in [('REGRET_BY_POLICY',regret),('EVICTION_SUMMARY',evictions),('DEMOTION_RECOVERY',recovery),('RECURRENCE_CLASS_SURVIVAL',grouped),('UTILITY_DECILES',deciles),('OCCUPANCY_CURVES',[x for r in records for x in r['occupancy']])]:
        csv_output(root/(name+'.csv'),rows)
    write_json(root/'SHUFFLED_UTILITY_AUDIT.json',{'per_seed':[{'seed':r['seed'],**r['shuffle']} for r in records],
               'strict_matching_passed':rep['strict_shuffled_matching_audit_passed'],
               'note':'Within-branch permutations are exact; independent policy trajectories need not remain cross-arm matched.'})
    write_json(root/'STATE_COSTS.json',{str(r['seed']):{'costs':r['costs'],'peaks':r['peaks']} for r in records})
    write_json(root/'Q0_VERIFICATION.json',p0)
    text='# V40 Utility Under Scarcity\n\nVerdict: **'+rep['verdict']+'**\n\n'
    text+=f"Completed {len(records)}/{rep['expected_seeds']} seeds; science={rep['science']}.\n\n"
    text+='| Policy | Excess bits/object | Whole bpb | Nontarget bpb |\n|---|---:|---:|---:|\n'
    for p,d in rep['per_policy'].items():
        fmt=lambda x:'n/a' if x['mean'] is None else f"{x['mean']:.5f}"
        text+=f"| {p} | {fmt(d['object_regret'])} | {fmt(d['whole_bpb'])} | {fmt(d['nontarget_bpb'])} |\n"
    text+='\nTask screens precede policy interpretation. Missing recovery observations do not pass. See REPORT.json and PROTOCOL.md for gates and implementation resolutions.\n'
    atomic(root/'SUMMARY.md',text)
    names=['SUMMARY.md','REPORT.json','PROTOCOL.md','TERMINAL.csv','REGRET_BY_POLICY.csv','UTILITY_DECILES.csv','RECURRENCE_CLASS_SURVIVAL.csv','EVICTION_SUMMARY.csv','DEMOTION_RECOVERY.csv','OCCUPANCY_CURVES.csv','SHUFFLED_UTILITY_AUDIT.json','STATE_COSTS.json','Q0_VERIFICATION.json',Path(__file__).name]
    names += [str(x.relative_to(root)) for x in sorted((root/'completed').glob('*.json'))] if (root/'completed').exists() else []
    manifest={'run':identity,'file_sha256':{name:sha((root/name).read_bytes()) for name in names}}
    write_json(root/'MANIFEST.json',manifest);names.append('MANIFEST.json')
    dest=root.parent/(root.name+'_results.zip');tmp=dest.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in names:z.write(root/name,name)
    with zipfile.ZipFile(tmp) as z:
        assert z.testzip() is None
        for name,digest in manifest['file_sha256'].items():assert sha(z.read(name))==digest
    os.replace(tmp,dest);return dest


def main40(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--install-deps',action='store_true');ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--verify-full',action='store_true')
    ap.add_argument('--output',type=Path);ap.add_argument('--max-seeds',type=int)
    args=ap.parse_args(argv);dependencies(args.install_deps)
    if args.max_seeds is not None and args.max_seeds<0:ap.error('--max-seeds must be nonnegative')
    caffeinate=None
    if platform.system()=='Darwin':
        try:caffeinate=subprocess.Popen(['caffeinate','-i','-w',str(os.getpid())],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        except FileNotFoundError:pass
    try:
        tests=self_test40();print(f"Startup checks passed: {tests['count']}",flush=True)
        if args.self_test:return 0
        mode='verification' if args.verify_full else 'smoke' if args.smoke else 'science'
        seeds=(974011,974012) if args.smoke else SEEDS
        root=args.output or Path('elm_v40_'+mode);root.mkdir(parents=True,exist_ok=True)
        plans=[{'seed':seed,'stream_sha256':stream40(seed,args.smoke)['raw_sha256']} for seed in seeds]
        identity={'version':VERSION,'source_sha256':sha(Path(__file__).read_bytes()),'reference_sha256':REFERENCE_SHA256,
                  'protocol_sha256':sha((PROTOCOL+IMPLEMENTATION_NOTES).encode()),'mode':mode,'seeds':list(seeds),'plans':plans,
                  'policies':list(POLICIES),'trace_cap':32 if args.smoke else 512,'slot_cap':24 if args.smoke else 384}
        with RunLock(root):
            mp=root/'RUN_MANIFEST.json'
            if mp.exists():validate_identity(json.loads(mp.read_text()),identity)
            else:write_json(mp,identity)
            pp=root/'Q0_CHECKPOINT.json'
            if pp.exists():p0=record_read(pp,identity)
            else:
                p0=[]
                for seed in ENGINEERING_SEEDS:
                    print(f'V39 substrate equivalence: {seed}',flush=True)
                    p0.append(historical_preflight(seed,smoke=args.smoke))
                record_write(pp,p0,identity)
            assert len(p0)==2 and all(x['passed'] and (args.smoke or x['full_length']) for x in p0)
            if args.verify_full:
                a=trial40(974003,False,True);b=trial40(974003,False,False)
                assert a['base_hash']==b['base_hash'] and a['final_hashes']==b['final_hashes']
                write_json(root/'VERIFICATION.json',{'startup':tests,'p0':p0,'full_observer_invariance':True,'science_run':False})
                print('Full verification passed.',flush=True);return 0
            records=[];new=0
            for seed,plan in zip(seeds,plans):
                path=root/'completed'/f'{seed}.json'
                if path.exists():
                    r=record_read(path,identity)
                    assert r['seed']==seed and r['stream_sha256']==plan['stream_sha256']
                    records.append(r);print(f'Reused seed {seed}',flush=True);continue
                if args.max_seeds is not None and new>=args.max_seeds:continue
                r=trial40(seed,args.smoke);record_write(path,r,identity);records.append(r);new+=1
                rep=report40(records,seeds,p0,args.smoke);package40(root,records,rep,identity,p0)
                print(f'Completed {len(records)}/{len(seeds)} seeds',flush=True)
            rep=report40(records,seeds,p0,args.smoke);dest=package40(root,records,rep,identity,p0)
            print(f"{rep['verdict']}: {len(records)}/{len(seeds)} seeds, {new} new; {dest}",flush=True)
            return 0
    finally:
        if caffeinate is not None:caffeinate.terminate()

if __name__=='__main__':
    raise SystemExit(main40())
