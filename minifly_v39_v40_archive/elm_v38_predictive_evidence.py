#!/usr/bin/env python3
"""V38: Predictive Evidence Arbitration. One file, 16 fixed scientific trajectories.
python3 elm_v38_predictive_evidence.py --install-deps
--self-test: engineering checks only; --smoke: four small engineering trajectories.
Resume reuses completed trajectories; interrupted trajectories restart deterministically.
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
VERSION='v38.1.0'
SEEDS=tuple(range(38101,38109))
ENGINEERING_SEEDS=(973801,973802)



from array import array


REFERENCE_SHA256='919c5e778baceb5f1949ef4ed3a03fa464017df04dfb80b0c18c57e51f972804'

PROTOCOL="# V38 — Predictive Evidence Arbitration\n## From corroboration count to causal confidence\n\n**Status:** design / preregistration only. No V38 scientific outcomes have been observed.\n\n**Proposed scientific seeds:** `38101`–`38108`  \n**Stateful trajectories:** 16 total = 8 × `PRED-V` + 8 × `EXACT-V`  \n**Primary mechanism:** slot-local predictive Bayes-factor authority (`PBF`)  \n**No automatic follow-up. No parameter sweep. No new encoder. No new write rule.**\n\n---\n\n## 0. Decision in one sentence\n\nV37 established that **memory authority must be selective**, but it also exposed that raw corroboration count is too coarse: a support-2 candidate can be either the exact situation where held-out alias transfer should be trusted or a generic nontarget cluster where hard retrieval is harmful.\n\nV38 therefore asks:\n\n\\[\n\\boxed{\\text{Can a slot earn generalization authority from how much better it predicts new contexts than the shared base?}}\n\\]\n\nThe primary policy accumulates a causal, label-free, slot-local log predictive evidence score. A second rendering counts as strong corroboration only when the memory predicted an outcome that the base did **not** already predict equally well.\n\n---\n\n# 1. Why the first V38 proposal is rejected\n\nThe earlier proposal was to introduce a stable value plus a provisional revision branch and require two distinct contexts before a contradiction could rewrite the whole slot. That is a plausible future mechanism, but it is not the best next experiment for three reasons.\n\n### 1.1 It attacks a downstream problem before finishing the V37 mechanism\n\nV37 qualified `CORR2`, but `CORR2` is not yet a satisfactory endpoint. It grants full authority to every unbound candidate whose slot has support ≥2. V37's own conditional results show that this is still badly overconfident on ordinary text.\n\nWeighted across the eight V37 seeds:\n\n| tentative context class | base bpb | HARD / CORR2 hard-read bpb | fixed α=.50 bpb |\n|---|---:|---:|---:|\n| nontarget, support 1 | **1.164** | 5.300 | 1.458 |\n| nontarget, support 2 | **0.623** | **2.356** | **0.543** |\n| nontarget, support 3+ | 0.601 | 0.752 | **0.301** |\n\nThe support-2 row is decisive. `CORR2` treats all support-2 neighborhoods as trustworthy, yet hard retrieval there is almost four times the base log loss.\n\nAt the same time, support 2 is exactly the point at which the real held-out target alias becomes useful:\n\n| E2 held-out probe after the second trained rendering joins | accuracy | target probability |\n|---|---:|---:|\n| SHADOW | 3.9% | 0.015 |\n| α=.50 | 91.0% | 0.481 |\n| CORR2 / HARD | **100%** | **0.946** |\n\nThus the unresolved question is **not** whether two observations matter. It is why some two-member neighborhoods deserve almost full authority and others deserve little or none.\n\n### 1.2 Provisional revision would be confounded with slower updating\n\nA two-hit or two-context revision rule could improve one-off-noise resistance simply by delaying writes. Without a validated confidence variable, a positive result would be difficult to distinguish from a generic learning-rate / confirmation-delay effect—the same class of confound that killed several earlier gating mechanisms.\n\n### 1.3 It would throw away V37's clean causal experimental structure\n\nV37's strongest methodological feature is that many read policies share **one identical PRED-V state trajectory**. A revision-policy experiment would branch the learner state and immediately reintroduce write-path confounding. V38 should exploit the clean read-only counterfactual structure one more time while the remaining question is still about authority.\n\n**Therefore V38 does not change revision semantics.** One-off-noise behavior remains a diagnostic only.\n\n---\n\n# 2. What V37 established and what remains open\n\n## 2.1 Established\n\nV37 completed 16/16 fresh stateful trajectories and passed state-path invariance.\n\n- `HARD`: 100% held-out revision, but whole bpb ≈ **2.306**.\n- `SHADOW`: whole bpb ≈ **1.290**, but 0% held-out revision.\n- `CORR2`: **100% held-out revision**, whole bpb ≈ **1.388**, and qualified under all V37 gates.\n- fixed `α=.50`: **87.5% held-out revision**, whole bpb ≈ **1.328**, also qualified.\n- `RAND-MATCH`, with the same number of hard tentative reads as CORR2, was materially worse. Thus support count contains real information.\n\nThe architectural decomposition is now:\n\n\\[\n\\boxed{\\text{proposal} \\neq \\text{binding} \\neq \\text{authority}}\n\\]\n\nand this decomposition is retained.\n\n## 2.2 Still open\n\nV37 does **not** establish that support count is the right confidence variable.\n\nTwo facts coexist:\n\n1. `CORR2` achieves the strongest held-out transfer because support 2 can justify high authority on real aliases.\n2. fixed `α=.50` achieves better whole-stream utility because full authority on many support-2 nontarget candidates is excessive.\n\nThis is the V38 target:\n\n\\[\n\\boxed{\\text{distinguish informative corroboration from redundant or accidental corroboration.}}\n\\]\n\n---\n\n# 3. Core idea: predictive evidence, not event count\n\nSuppose an unbound context \\(c_t\\) tentatively retrieves slot \\(j\\) before the next byte \\(y_t\\) is known.\n\nLet\n\n- \\(p_B(y\\mid c_t)\\): frozen pre-update shared-base distribution;\n- \\(p_j(y)\\): frozen pre-update slot distribution;\n- \\(m_j\\): number of exact members already in slot \\(j\\);\n- \\(E_j\\): evidence accumulated from **previous tentative uses of this slot**.\n\nAfter the true byte is observed, compute the prequential log predictive advantage\n\n\\[\n\\Delta E_t\n=\n\\log p_j(y_t)-\\log p_B(y_t).\n\\]\n\nThen update only the auxiliary evidence ledger:\n\n\\[\nE_j \\leftarrow E_j + \\Delta E_t.\n\\]\n\nNo learner weight, slot value, prototype, member set, eligibility state, or binding decision depends on \\(E_j\\).\n\nInterpretation:\n\n- \\(\\Delta E_t \\gg 0\\): the slot predicted something that the base found surprising; this is strong evidence that the similarity match carries reusable memory.\n- \\(\\Delta E_t \\approx 0\\): both systems already knew the outcome; the match adds little evidence of a special object-level relation.\n- \\(\\Delta E_t < 0\\): the base explained the outcome better; this is evidence **against** granting that slot broad authority over similar contexts.\n\nThis is the key refinement over CORR2:\n\n> A second context is not automatically one unit of corroboration. Its evidential value depends on what it predicted beyond the base.\n\n---\n\n# 4. Why this is particularly well matched to V37\n\nThe PRED-V permanent-binding rule already requires a candidate slot's **pre-update top-1** to equal the subsequently observed byte before a new context may join the slot.\n\nBut top-1 equality is a very weak test of evidence strength.\n\nExample:\n\n- slot assigns 0.55 to byte `A`;\n- base assigns 0.95 to byte `A`.\n\nPRED-V calls this a successful confirmation because both top-1 predictions are `A`. Yet the observation is evidence **for the base, not for the slot**:\n\n\\[\n\\Delta E=\\log(0.55)-\\log(0.95)<0.\n\\]\n\nBy contrast, suppose a second rendering of an arbitrary stored association yields:\n\n- slot assigns 0.98 to byte `Q`;\n- base assigns 0.01 to byte `Q`.\n\nThen\n\n\\[\n\\Delta E \\approx \\log 98 \\gg 0.\n\\]\n\nBoth events increment support from 1 to 2 under CORR2. V38 treats them radically differently.\n\nThis is exactly the distinction needed to reconcile the V37 support-2 target and nontarget results.\n\n---\n\n# 5. Fixed substrate: do not reopen solved components\n\nV38 freezes the qualified V37 substrate:\n\n- V36/V37 byte workload;\n- causal next-byte residual teaching signal;\n- shared base learner;\n- lag96 address geometry;\n- fixed threshold \\(\\tau=0.85\\);\n- PRED-V proposal rule;\n- PRED-V permanent binding / allocation rule;\n- slot prototype updates;\n- absolute slot values;\n- exact-member hard ownership;\n- no replay;\n- no task IDs, logical-key IDs, alias IDs, or evaluator metadata in the learner;\n- no capacity limit yet;\n- no revision protection yet;\n- no covariance / RLS / MESU / plasticity selection / new encoder.\n\nThe only new deployable state is **one scalar \\(E_j\\) per slot**.\n\n---\n\n# 6. Common underlying learner trajectory\n\nFor each V38 seed, run one physical `PRED-V` learner trajectory.\n\nAll V38 authority policies see identical, frozen pre-observation quantities at each byte event:\n\n- base distribution;\n- candidate slot distribution;\n- candidate identity;\n- candidate similarity;\n- candidate support;\n- exact-bound / tentative status.\n\nAfter the true byte arrives:\n\n1. score every policy;\n2. update each policy's auxiliary evidence state according to its own predefined rule;\n3. execute the **single unchanged PRED-V observe/bind/write operation**;\n4. verify the underlying PRED-V state hash is policy-independent.\n\nThus policy differences cannot alter:\n\n- future base learning;\n- slot values;\n- prototype geometry;\n- membership;\n- eligibility;\n- allocation count.\n\nThis keeps V38 a read-authority experiment, not a hidden write-rule experiment.\n\n---\n\n# 7. Authority policies\n\nFor every exact bound context, all deployable policies retain **full absolute ownership** exactly as V37.\n\nFor no-hit contexts, all policies use the base.\n\nPolicies differ only on **tentative unbound candidates**.\n\n## 7.1 CORR2 — locked V37 primary reference\n\n\\[\nq_{\\mathrm{CORR2}}=\n\\begin{cases}\n1,&m_j\\ge2,\\\\\n0,&m_j<2.\n\\end{cases}\n\\]\n\nPrediction:\n\n\\[\np=(1-q)p_B+q p_j.\n\\]\n\nNo changes from V37.\n\n## 7.2 ALPHA-0.50 — locked simple reference\n\nFor every tentative candidate:\n\n\\[\nq_{0.5}=0.5.\n\\]\n\nThis arm was diagnostic in V37 but unexpectedly qualified. V38 treats `0.50` as a **locked candidate**, not as a tuned value. No alpha grid is rerun.\n\n## 7.3 PBF — primary V38 policy\n\nInitialize on slot allocation:\n\n\\[\nE_j=0.\n\\]\n\nBefore the current outcome is known:\n\n\\[\nq_j=\n\\begin{cases}\n0,&m_j<2,\\\\[2mm]\n\\sigma(E_j)=\\dfrac{1}{1+e^{-E_j}},&m_j\\ge2.\n\\end{cases}\n\\]\n\nThen\n\n\\[\np_{\\mathrm{PBF}}=(1-q_j)p_B+q_jp_j.\n\\]\n\nAfter the outcome is observed, if the context was tentative for slot \\(j\\), update\n\n\\[\nE_j\\leftarrow E_j+\n\\left[\\log p_j(y_t)-\\log p_B(y_t)\\right],\n\\]\n\nusing the **pre-update** distributions frozen before observation.\n\nImportant details:\n\n- evidence is updated whether or not PRED-V ultimately binds the context;\n- failed tentative matches therefore contribute negative evidence when appropriate;\n- exact bound repetitions do **not** update \\(E_j\\), because they test memorization, not generalization of the slot neighborhood;\n- no decay, cap, temperature, learning rate, or threshold is introduced;\n- `log_softmax` is used directly, so no arbitrary epsilon is required;\n- the current event never benefits from its own outcome; \\(E_j\\) affects predictions only from the next event onward.\n\nThe `m_j<2` shadow gate is retained for one reason only: V37 already causally established that singleton speculative authority is catastrophic, and a singleton has not yet generated an out-of-member successful binding event from which reusable identity can be inferred.\n\n## 7.4 SLOT-EVID — base-blind evidence control\n\nThis control asks whether the important variable is genuinely **advantage over the base**, rather than simply the slot being confident/correct.\n\nMaintain\n\n\\[\nA_j\\leftarrow A_j+\n\\left[\\log p_j(y_t)-\\log(1/256)\\right]\n\\]\n\non the same tentative events, and use\n\n\\[\nq_j=0\\quad(m_j<2),\\qquad q_j=\\sigma(A_j)\\quad(m_j\\ge2).\n\\]\n\nThe uniform reference is fixed by the 256-byte alphabet; no fitted quantity is introduced.\n\nIf SLOT-EVID matches PBF, the shared-base comparison is unnecessary and the mechanism should be simplified.\n\n## 7.5 SHUFFLED-PBF — decisive specificity control, nondeployable\n\nThis control preserves the **entire marginal distribution and timing of PBF evidence**, but destroys which slot earned which evidence.\n\nAfter each tentative event:\n\n1. compute the same \\(\\Delta E_t\\) as PBF;\n2. update the PBF ledger normally for the primary arm;\n3. for the control, deterministically shuffle the current evidence values among slots **within the same current support bin** (`1`, `2`, `3+`) using a seed-fixed RNG before the next event.\n\nThus SHUFFLED-PBF preserves, at each step and support class:\n\n- number of slots;\n- multiset of evidence values;\n- total evidence;\n- evidence update timing;\n- support distribution;\n- underlying learner state.\n\nIt destroys only the mapping\n\n\\[\n\\text{this slot} \\leftrightarrow \\text{this slot's predictive history}.\n\\]\n\nSHUFFLED-PBF is evaluator-side and nondeployable. It exists solely to answer:\n\n\\[\n\\boxed{\\text{Does slot-specific predictive history matter, or only the global authority distribution?}}\n\\]\n\nIf SHUFFLED-PBF matches PBF, the mechanism is rejected regardless of headline performance.\n\n---\n\n# 8. No post-hoc authority search\n\nV38 deliberately does **not** rerun:\n\n- the V37 alpha frontier;\n- a threshold sweep on \\(E_j\\);\n- evidence decay rates;\n- evidence clipping;\n- temperatures on \\(\\sigma(E)\\);\n- similarity-weighted evidence;\n- support thresholds 2/3/4;\n- hybrid CORR2 × alpha rules.\n\nThe central claim must survive in its parameter-free form or be rejected.\n\n---\n\n# 9. Scientific workload\n\nUse eight fresh science seeds:\n\n\\[\n38101,38102,38103,38104,38105,38106,38107,38108.\n\\]\n\nFor each seed:\n\n1. one `PRED-V` state trajectory carrying all read/evidence policies in parallel;\n2. one `EXACT-V` trajectory as the exact-ownership reference.\n\nTotal:\n\n\\[\n\\boxed{16\\text{ stateful trajectories}.}\n\\]\n\nThe evidence policies do not multiply expensive learner runs.\n\nNo V37 science seed is reused for qualification.\n\n---\n\n# 10. Primary hypotheses\n\n## H1 — V37 residual failure is localized to coarse support confidence\n\nOn **tentative nontarget support-2 events**, CORR2 hard retrieval must remain materially worse than the base on the fresh seeds.\n\nThis is a replication gate for the problem V38 claims to solve.\n\nIf support-2 hard retrieval is no longer harmful on fresh seeds, V38's mechanistic premise is not replicated and PBF interpretation is downgraded.\n\n## H2 — predictive evidence improves the CORR2 utility/transfer conjunction\n\nRelative to CORR2, PBF should reduce nontarget and whole-stream log loss while preserving held-out alias revision.\n\nPrimary paired contrasts:\n\n\\[\n\\Delta_{\\text{PBF-CORR2}}\\text{ nontarget bpb}<0,\n\\]\n\n\\[\n\\Delta_{\\text{PBF-CORR2}}\\text{ whole bpb}<0,\n\\]\n\nwith held-out revision noninferior.\n\n## H3 — PBF improves on the simple α=.50 compromise\n\nRelative to fixed α=.50, PBF should recover the missing held-out transfer while remaining whole-stream noninferior.\n\nThis is what distinguishes a learned local confidence variable from a globally reduced memory gain.\n\n## H4 — slot-specific evidence is causal\n\nPBF must beat SHUFFLED-PBF on the primary utility/transfer conjunction.\n\nFailure of H4 retires PBF as a mechanism even if PBF's raw metrics are attractive.\n\n## H5 — evidence over the shared base matters\n\nPBF should outperform SLOT-EVID. If SLOT-EVID matches it, the architecture is simplified to base-blind slot confidence and no claim is made for comparative predictive evidence.\n\n---\n\n# 11. Qualification gates for PBF\n\nAll gates are evaluated on fresh seeds only.\n\n## Q0 — invariance / causality\n\nMandatory engineering gate.\n\nAt every checkpoint, the physical PRED-V learner must have identical hashes regardless of read/evidence policy for:\n\n- base;\n- slot values;\n- prototypes and sums;\n- member sets/vectors;\n- eligibility;\n- full learner state.\n\nEvery prediction and evidence increment must use distributions frozen **before** the current byte is revealed.\n\nAny violation voids the run.\n\n## Q1 — acquisition and persistence retained\n\nUsing the PBF read policy:\n\n- A acquisition ≥ 90% mean;\n- A-after-D ≥ 90% mean;\n- D-current ≥ 90% mean;\n- no science seed below 75% on A-after-D.\n\n## Q2 — exact and known-alias revision retained\n\nAfter four true revision presentations:\n\n- exact revised alias ≥ 90%;\n- known trained alias ≥ 90%.\n\n## Q3 — held-out generalization remains strong\n\nAfter four true revision presentations:\n\n\\[\n\\boxed{\\text{held-out alias accuracy}\\ge 90\\%}\n\\]\n\nwith no seed below 75%.\n\nThe threshold is intentionally stricter than V37's original 50% gate because V37 CORR2 already established 100% on this workload. V38 is a refinement, not a regression to the old minimum.\n\n## Q4 — obsolete value suppressed\n\nAfter eight true revision presentations:\n\n\\[\nP(\\text{obsolete byte})\\le 0.10.\n\\]\n\n## Q5 — whole-stream utility\n\nPBF whole-stream bpb must be no more than **0.05 bpb worse** than its base-only counterfactual:\n\n\\[\n\\Delta_{\\rm base}^{\\rm whole}\\le0.05.\n\\]\n\nThis tightens V37's +0.10 qualification margin because V37 has now shown that positive whole-stream utility is achievable.\n\n## Q6 — nontarget utility\n\nPBF nontarget bpb must be no more than **0.05 bpb worse** than base:\n\n\\[\n\\Delta_{\\rm base}^{\\rm nontarget}\\le0.05.\n\\]\n\n## Q7 — binding purity unchanged\n\nBecause PBF is not allowed to alter binding, PRED-V must again show:\n\n- evaluator-known different-key target binding rate <5%;\n- final impure target-context fraction <5%.\n\nAny deviation from the underlying PRED-V reference indicates an implementation bug.\n\n## Q8 — comparative improvement over CORR2\n\nPBF must improve nontarget bpb over CORR2 by at least **0.10 bpb in mean paired difference**, with the paired 95% interval excluding zero, while held-out accuracy is noninferior within **5 percentage points**.\n\nThis is the main V38 mechanism gate.\n\n## Q9 — comparative improvement over α=.50\n\nPBF must be whole-stream noninferior to α=.50 within **0.05 bpb**, while improving mean held-out accuracy by at least **5 percentage points**.\n\nThe interval for the held-out contrast is reported; failure to exclude zero is not by itself fatal because eight-seed Bernoulli endpoints are coarse, but the direction and materiality threshold are fixed in advance.\n\n## Q10 — shuffled-evidence veto\n\nPBF must beat SHUFFLED-PBF on at least one of:\n\n- whole-stream bpb by ≥0.10;\n- nontarget bpb by ≥0.10;\n- held-out accuracy by ≥10 pp;\n\nwith the paired 95% interval excluding zero on that endpoint, and must not be materially worse on the other two primary endpoints.\n\nIf SHUFFLED-PBF matches PBF, **PBF is rejected as slot-specific confidence**.\n\n---\n\n# 12. Evidence diagnostics — mechanistic, not just endpoint scoring\n\nV38 must explain *why* PBF succeeds or fails.\n\n## 12.1 Evidence calibration by support\n\nFor tentative events, report separately for support 1, support 2, and support 3+:\n\n- count;\n- mean / quantiles of \\(E_j\\) before prediction;\n- mean authority \\(q_j\\);\n- base bpb;\n- slot-only bpb;\n- PBF bpb;\n- fraction where slot beats base in log loss.\n\n## 12.2 Evidence deciles\n\nWithin tentative support-2 nontarget events, bin by **pre-event** PBF authority decile.\n\nFor each decile report:\n\n\\[\nR=\\log p_j(y)-\\log p_B(y)\n\\]\n\nand PBF/base log loss.\n\nA useful confidence variable should order future reliability: high-authority bins should have higher subsequent slot advantage than low-authority bins.\n\nNo evaluator label enters the authority calculation; labels are used only to summarize calibration after the event.\n\n## 12.3 E1 → E2 held-out emergence\n\nRepeat the V37 emergence probe.\n\nAt E1 and E2 report for the held-out alias:\n\n- source slot support;\n- source slot accumulated evidence;\n- PBF authority;\n- base probability of target;\n- slot probability of target;\n- PBF probability of target;\n- top-1 accuracy.\n\nThe desired signature is:\n\n- E1: support 1, authority 0;\n- E2: support 2, strongly positive evidence, authority near 1 for genuine logical aliases.\n\nThis is the direct test of the proposed mechanism.\n\n## 12.4 Generic support-2 counterexample audit\n\nFor nontarget support-2 candidates, record the same quantities.\n\nThe central expected contrast is:\n\n\\[\nE_j(\\text{true alias support 2})\n>\nE_j(\\text{generic support 2})\n\\]\n\nin a distributional, not evaluator-hardcoded, sense.\n\n---\n\n# 13. One-off noise remains diagnostic only\n\nV38 does **not** change slot values or revision writes, so the known V37 one-off-noise failure is expected to remain.\n\nNevertheless record, without qualification consequences:\n\n- evidence state before a one-off wrong observation;\n- exact/known/held-out probabilities immediately before and after;\n- hypothetical contradictory log evidence\n  \\[\n  \\log p_B(y_{\\rm wrong})-\\log p_j(y_{\\rm wrong});\n  \\]\n- how many independent contradictory contexts would be required to offset the slot's accumulated generalization evidence if the same log-evidence principle were later used for revision.\n\nThis is **not** a V38 mechanism. It is the preregistered diagnostic that decides whether V39 should reuse predictive evidence for write/revision arbitration.\n\n---\n\n# 14. Resource accounting\n\nReport separately:\n\n- original shared-base bytes;\n- slot value bytes;\n- member/prototype address bytes;\n- PBF evidence bytes;\n- evidence bytes / slot;\n- total payload;\n- slots;\n- exact members;\n- operation proxy excluding evaluator-only controls;\n- operation proxy including deployable PBF arithmetic.\n\nExpected incremental persistent state:\n\n\\[\n8\\text{ bytes per slot}\n\\]\n\nfor float64 \\(E_j\\), negligible relative to V37's 256-way slot value vector and address payload.\n\nThe run remains explicitly **unbounded**. Capacity is not claimed solved.\n\n---\n\n# 15. Hard implementation invariants\n\nV38 is invalid if any invariant fails.\n\n### I1 — one physical PRED-V learner\n\nNo authority/evidence policy has a private copy of base weights, slot values, prototypes, membership, eligibility, or bindings.\n\n### I2 — evidence cannot influence binding\n\nPRED-V candidate confirmation and allocation use the exact V37 rule and never read \\(E_j\\), \\(A_j\\), PBF authority, or policy output.\n\n### I3 — evidence is prequential\n\nThe prediction uses \\(E_j^{(t-1)}\\). The current outcome updates \\(E_j\\) only after all policies have been scored.\n\n### I4 — exact ownership unchanged\n\nFor a bound exact context, all deployable policies return the absolute slot value with authority 1. Evidence affects only unbound tentative reads.\n\n### I5 — probes are read-only\n\nHeld-out alias probes, revision probes, calibration probes, and noise probes cannot update evidence or learner state.\n\n### I6 — SHUFFLED-PBF has identical evidence marginals\n\nAt each audited checkpoint and support bin, the sorted multiset of SHUFFLED-PBF evidence values must equal the sorted multiset of PBF evidence values to numerical tolerance.\n\nOnly slot assignment differs.\n\n### I7 — no evaluator metadata\n\nThe learner/policy cannot access logical key ID, alias number, target/nontarget flag, wave name, revision status, noise status, or probe type.\n\n---\n\n# 16. Required outputs\n\nThe V38 runner should emit at minimum:\n\n- `SUMMARY.md`\n- `REPORT.json`\n- `SUMMARY.csv`\n- `PROTOCOL.md`\n- `RUN_MANIFEST.json`\n- `STATUS.json`\n- `STATE_INVARIANCE.json`\n- `STATE_COSTS.json`\n- `AUTHORITY_BY_SUPPORT.csv`\n- `EVIDENCE_BY_SUPPORT.csv`\n- `EVIDENCE_DECILES.csv`\n- `EVIDENCE_TRAJECTORIES.csv`\n- `HELDOUT_E1_E2.csv`\n- `PBF_PAIRED_DELTAS.csv`\n- `SHUFFLED_PBF_AUDIT.json`\n- `REVISION_CURVES.csv`\n- `NOISE_DIAGNOSTIC.json`\n- per-seed job JSONs\n- one self-contained executable `elm_v38_predictive_evidence.py`\n\nThe final archive must contain hashes for source, protocol, run plan, and every emitted artifact.\n\n---\n\n# 17. Engineering tests before science seeds\n\n`--self-test` must verify at least:\n\n1. PBF starts with zero authority on a singleton tentative candidate.\n2. A positive tentative predictive advantage increases that slot's future authority.\n3. A negative predictive advantage decreases that slot's future authority.\n4. The current outcome cannot change the current prediction retroactively.\n5. Exact bound contexts remain full absolute ownership regardless of evidence.\n6. Evidence does not alter PRED-V binding/allocation.\n7. Repeated probes do not update evidence.\n8. PBF and SHUFFLED-PBF have identical evidence multisets at checkpoints.\n9. SHUFFLED-PBF assigns evidence to different slots.\n10. Underlying learner hashes match an uninstrumented V37-style PRED-V replay.\n11. `CORR2` and `ALPHA-0.50` reproduce their frozen definitions exactly.\n12. A synthetic high-Bayes-factor second rendering drives PBF authority close to 1 without any fitted gain.\n13. A synthetic redundant confirmation where base is more confident than slot does not increase trust.\n14. Resume rejects mismatched source/protocol/plan hashes.\n\n`--verify-full` should independently replay at least one full engineering seed without the observer panel and match learner hashes and HARD losses exactly.\n\n---\n\n# 18. Decision tree after V38\n\n## Outcome A — PBF passes Q1–Q10\n\nInterpretation:\n\n\\[\n\\boxed{\\text{slot-local prequential evidence is a validated confidence primitive.}}\n\\]\n\nFreeze proposal, binding, absolute ownership, and PBF authority.\n\n**Next round:** use the same evidence principle for **write/revision arbitration**, not a hand-coded two-hit rule. A contradiction should first accumulate evidence against the stable value; only sufficient independent predictive evidence should rewrite the object globally. Capacity can then use the same confidence ledger for admission/consolidation.\n\n## Outcome B — PBF beats CORR2 but not α=.50\n\nInterpretation: the extra evidence state is not justified; a globally bounded memory contribution is sufficient on this workload.\n\nFreeze `α=.50` only after its fresh-seed V38 replication. Move to revision/capacity with the simpler read rule.\n\n## Outcome C — PBF matches CORR2 but does not improve utility\n\nInterpretation: support count captures most useful confidence available from this representation. Do not tune evidence. Move on to bounded capacity / revision.\n\n## Outcome D — SHUFFLED-PBF matches PBF\n\nInterpretation: the apparent benefit comes from the marginal authority distribution or generic regularization, not slot-specific predictive history.\n\n**Reject PBF.** Do not carry the evidence ledger forward.\n\n## Outcome E — PBF loses held-out transfer because evidence is systematically too weak\n\nDo **not** add a temperature or tune a gain. The likely missing variable is identity geometry / evidence quality, not calibration scale.\n\nThe next experiment should test a separate identity-address representation (the surviving lesson from the independent V36 separation branch), while keeping the predictive base unchanged.\n\n---\n\n# 19. What V38 is allowed to claim if successful\n\nA successful V38 would establish, on the current controlled workload, that:\n\n1. persistent absolute memories can remain distinct from a shared predictive learner;\n2. similarity can propose candidate object identity;\n3. binding can be confirmed causally from subsequent sensory outcome;\n4. retrieval authority can be calibrated from **slot-specific prequential evidence relative to the base** rather than a fixed support count;\n5. the resulting evidence state improves the transfer/utility conjunction under a decisive shuffled-evidence control.\n\nIt would **not** establish:\n\n- bounded lifelong capacity;\n- robustness to one-off false revisions;\n- natural-language semantics;\n- biological equivalence to the fly mushroom body;\n- a general replacement for Transformers;\n- scaling laws.\n\nThe exact intended claim is narrower and more valuable:\n\n\\[\n\\boxed{\\text{A memory should earn trust by explaining new experience better than the system that would otherwise predict it.}}\n\\]\n\n---\n\n# 20. Why this round is worth doing before capacity or revision\n\nThe project has repeatedly suffered when a downstream mechanism was built before the variable it should respond to had been identified.\n\nV37 has now given a rare opportunity to avoid that mistake. We know:\n\n- support count is informative;\n- support count is not sufficient;\n- a global 0.5 authority level is surprisingly competitive;\n- the remaining error is concentrated in exactly the support-2 ambiguity where “corroboration” can mean either real cross-rendering identity or generic redundant structure.\n\nPredictive evidence provides a single causal quantity that could later serve three roles:\n\n1. **read authority** — V38;\n2. **revision confidence** — future round;\n3. **admission/consolidation priority under fixed capacity** — future round.\n\nIf V38 falsifies it, we learn this before entangling it with writes or eviction. If it survives, subsequent rounds can be built around one validated confidence primitive rather than a collection of unrelated heuristics.\n\nThat is the reason V38 should be this experiment.\n"

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

def exposure_accounting(stream):
    raw=stream['raw'];rows=stream['rows']
    windows={raw[max(0,p-12):p] for p in range(len(raw))}
    return {'unique_logical_keys':len(stream['keys']),'trained_target_renderings':len({r['raw'][:12] for r in rows}),
        'heldout_target_renderings':len(stream['keys']),'distinct_raw_12byte_contexts':sum(len(c)==12 for c in windows),
        'distinct_contexts_including_startup':len(windows),
        'context_target_pairs':len({(raw[max(0,p-12):p],raw[p]) for p in range(len(raw))}),
        'target_context_target_pairs':len({r['raw'][:13] for r in rows}),
        'target_events':len(rows),'byte_events':len(raw),'initial_presentations_per_mapping':stream['repeats'],
        'per_key':[{'key_id':i,'acquisition':sum(r['key_id']==i and r['kind']=='acquisition' for r in rows),
        'noise':sum(r['key_id']==i and r['kind']=='one_off_noise' for r in rows),
        'revision':sum(r['key_id']==i and r['kind']=='revision' for r in rows),
        'rendering_presentations':[sum(r['key_id']==i and r['alias']==a for r in rows) for a in (0,1)]} for i in range(len(stream['keys']))]}

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

def distribution(vals):
    vals=[float(v) for v in vals]
    return {'n':len(vals),'mean':mean(vals),'quantiles':dict(zip(['min','p01','p05','p25','p50','p75','p95','p99','max'],map(float,np.quantile(vals,[0,.01,.05,.25,.5,.75,.95,.99,1])))) if vals else {},'values':vals}

def slot_diagnostics(model,stream):
    lookup={context(k,a):i for i,k in enumerate(stream['keys']) for a in (0,1)}
    keys_by_slot=collections.defaultdict(set);target_contexts=collections.Counter();false=0;bindings=0
    false_contexts=set();seen=collections.defaultdict(set)
    for c,j,new in model.bind_events:
        if c in lookup:
            key=lookup[c];bindings+=1
            if seen[j]-{key}:false+=1;false_contexts.add(c)
            seen[j].add(key);keys_by_slot[j].add(key);target_contexts[j]+=1
    same=[];slots_per=[];held=[];similarities=[];margins=[];target_slot_ids=set(keys_by_slot)
    acquired=[]
    for i,k in enumerate(stream['keys']):
        js=[model.members.get(context(k,a)) for a in (0,1)]
        if any(j is not None for j in js):
            same.append(float(js[0] is not None and js[0]==js[1]));slots_per.append(len({j for j in js if j is not None}));acquired.append(i)
            q=model.address_encoder.encode(context(k,2))[0]
            if model.values:
                ss=model.prototypes[:len(model.values)]@q;j=int(np.argmax(ss))
                held.append(float(i in keys_by_slot[j]));similarities.append(float(ss[j]))
                if len(ss)>1:margins.append(float(ss[j]-np.partition(ss,-2)[-2]))
    presentation_counts=collections.Counter(r['raw'][:12] for r in stream['rows'])
    weighted_den=sum(presentation_counts[c] for c in model.members if c in lookup)
    # Event-level false bindings use only earlier target members. Final impurity is separate.
    impure=sum(target_contexts[j] for j,ks in keys_by_slot.items() if len(ks)>1)
    return {'slots':len(model.values),'members_per_slot':[len(x) for x in model.slot_members],
        'target_slots':len(target_slot_ids),'nontarget_only_slots':len(model.values)-len(target_slot_ids),
        'logical_keys_per_slot':[len(keys_by_slot[j]) for j in range(len(model.values))],
        'slots_per_acquired_logical_key':slots_per,'target_bindings':bindings,'different_key_false_bindings':false,
        'different_key_binding_rate':false/bindings if bindings else 0.,
        'final_impure_target_context_fraction':impure/bindings if bindings else 0.,
        'presentation_weighted_false_binding_rate':sum(presentation_counts[c] for c in false_contexts)/weighted_den if weighted_den else 0.,
        'same_key_merge_rate':mean(same) if same else 0.,'heldout_nearest_slot_identity_accuracy':mean(held),
        'nearest_similarity':distribution(similarities),'top1_top2_margin':distribution(margins),
        'prototype_drift':{'mean':mean(model.drift),'max':max(model.drift,default=0.),'sum':sum(model.drift)},
        'nontarget_identity_false_merges':None,'nontarget_note':'No ground-truth logical identity exists for generic non-target contexts; reported as unidentifiable, not zero.',
        'nontarget_multi_context_slots':sum(len(s)>1 and j not in target_slot_ids for j,s in enumerate(model.slot_members)),
        'binding_trace':[{'context_hex':c.hex(),'slot':j,'allocated':new} for c,j,new in model.bind_events]}

def component_hashes(m):
    return dict(base=m.base.state_hash(),values=sha(b''.join(m.values)),
        prototypes=sha(m.prototypes.tobytes()+b''.join(s.tobytes() for s in m.sums)),
        members=sha([(c.hex(),j,m.member_vectors[c].tobytes().hex()) for c,j in sorted(m.members.items())]),
        eligibility=m.eligibility_hash(),full=m.state_hash())

class Stats:
    def __init__(self):self.n=0;self.sums=collections.Counter();self.den=collections.Counter()
    def add(self,row):
        self.n+=1
        for k,v in row.items():
            if v is not None:self.sums[k]+=float(v);self.den[k]+=1
    def result(self):return dict(n=self.n,**{k:self.sums[k]/self.den[k] for k in self.sums})

def add_event(groups,scored,meta,category,row):
    support='3+' if meta['support']>=3 else str(meta['support'])
    for p,v in scored.items():
        augmented=dict(v,tentative_rate=float(meta['tentative']),
            nonzero_tentative_authority=float(meta['tentative'] and v['authority']>0),
            tentative_authority_weight=v['authority'] if meta['tentative'] else 0.)
        for key in ('whole',category,'wave_'+row['wave']+'_'+category):groups[(p,key)].add(augmented)
        if meta['tentative']:groups[(p,category+'_support_'+support)].add(augmented)
        if category=='target':
            groups[(p,f"{row['kind']}_{row['wave']}_{row['exposure']}")].add(augmented)
            groups[(p,f"render_{row['kind']}_{row['wave']}_{row['rendering_exposure']}")].add(augmented)

def probe_mean(r,p,cp,kind,field='accuracy',aliases=(0,1)):
    return mean(e['scores'][p][field] for e in r['probes'] if e['checkpoint']==cp and e['kind']==kind and e['alias'] in aliases)

def policy_metrics(r,p):
    g=r['groups'][p];base=r['groups']['BASE'];m={key+'_bpb':g[key]['bpb'] for key in ('whole','target','nontarget')}
    m.update(base_whole_bpb=base['whole']['bpb'],base_nontarget_bpb=base['nontarget']['bpb'],
        whole_delta_base=m['whole_bpb']-base['whole']['bpb'],nontarget_delta_base=m['nontarget_bpb']-base['nontarget']['bpb'],
        A_initial=probe_mean(r,p,'after_A','all_A'),D_current=probe_mean(r,p,'after_D','current'),
        false_merge_rate=r['addressing']['different_key_binding_rate'],final_impurity=r['addressing']['final_impure_target_context_fraction'])
    for w in 'ABCD':m['A_after_'+w]=probe_mean(r,p,'after_'+w,'untouched_A')
    for ex in (1,2,4,8):
        for a in range(3):
            m[f'revision_{ex}_alias_{a}']=probe_mean(r,p,f'revision_{ex}','revision',aliases=(a,))
            m[f'obsolete_{ex}_alias_{a}']=probe_mean(r,p,f'revision_{ex}','revision','obsolete_probability',(a,))
    for e in ('E1','E2'):m[e+'_heldout_accuracy']=probe_mean(r,p,e,'emergence',aliases=(2,))
    return m

POLICIES=('CORR2','ALPHA-0.50','PBF','SLOT-EVID','SHUFFLED-PBF')
IMPLEMENTATION_NOTES={
 'shuffle':'After each learner observation that changes evidence, allocation or support, rebuild the control as a permutation of current PBF values within post-observation support bins. Never independently accumulate a shuffled ledger. Dedicated RNG; probes never advance it. Exact repetitions leave the permutation unchanged.',
 'Q10':'Other-endpoint material harm: >0.05 bpb in whole/nontarget or >0.05 loss in heldout accuracy. Intervals are paired two-sided Student-t 95%, seeds as units; endpoint OR is preregistered, unadjusted.',
 'H1':'Material replication: support-2 nontarget HARD minus BASE >=0.10 bpb and paired lower interval >0; missing cells cannot pass. Failure downgrades interpretation, separate from Q gates.',
 'H5':'Same specificity endpoints/margins as Q10 against SLOT-EVID; absence of superiority is inconclusive, not evidence of equivalence.',
 'quantiles':'Exact within-seed empirical evidence quantiles. Authority deciles are fixed probability bins [0,.1),...,[.9,1], not fitted equal-frequency bins.',
 'noise':'Offset count is ceil(max(E,0)/contradictory_log_advantage) when positive; hypothetical equal-strength independent contexts only, never a revision rule.',
 'state':'One physical learner, separate evidence ledger. HARD and BASE retained solely for engineering/diagnostics. Evidence has exactly one float64 per slot per deployable evidence policy.',
 'emergence':'Actual support reported, not forced to 1/2; missing emergence is not imputed.',
 'resume':'Atomic completed-job recovery; interrupted trajectory restarts deterministically.'}

def log_softmax(scores):
    a=scores/TEMP;a=a-np.max(a)
    return a-math.log(float(np.exp(a).sum()))

def sigmoid(x):
    if x>=0:return 1/(1+math.exp(-x))
    e=math.exp(x);return e/(1+e)

class Evidence:
    """Only pre-event distributions and learner structural support enter this ledger."""
    def __init__(self,seed):
        self.E=array('d');self.A=array('d');self.shuffled=np.zeros(0)
        self.rng=np.random.default_rng(seed+3801701)
        self.events=0;self.shuffles=0;self.audit_count=0;self.changed_assignments=0
    def synchronize(self,m,changed=False):
        n=len(m.values);added=n-len(self.E)
        if added<0:raise AssertionError('Slots cannot disappear')
        self.E.extend([0.]*added);self.A.extend([0.]*added)
        if not (changed or added):return
        e=np.asarray(self.E);self.shuffled=e.copy()
        bins=np.array([min(len(s),3) for s in m.slot_members])
        for b in (1,2,3):
            ix=np.flatnonzero(bins==b)
            self.shuffled[ix]=e[self.rng.permutation(ix)]
        self.shuffles+=1
        self.changed_assignments+=int(np.count_nonzero(self.shuffled!=e))
        self.audit(m)
    def update(self,packet,y):
        meta=packet[2]
        if not meta['tentative']:return False
        j=meta['slot'];d=float(meta['log_slot'][y]-meta['log_base'][y])
        self.E[j]+=d;self.A[j]+=float(meta['log_slot'][y])+math.log(256)
        if not math.isfinite(self.E[j]) or not math.isfinite(self.A[j]):raise FloatingPointError('Evidence is nonfinite')
        self.events+=1
        return True
    def audit(self,m):
        if len(self.E)!=len(m.values):raise AssertionError('Ledger size mismatch')
        bins=np.array([min(len(s),3) for s in m.slot_members]);e=np.asarray(self.E)
        hashes={}
        for b in (1,2,3):
            ix=np.flatnonzero(bins==b);a=np.sort(e[ix]);s=np.sort(self.shuffled[ix])
            if not np.array_equal(a,s):raise AssertionError('Shuffled evidence marginal mismatch')
            hashes[str(b)]=dict(slots=len(ix),sha256=sha(a.tobytes()))
        self.audit_count+=1
        return hashes
    def state_hash(self):return sha(self.E.tobytes()+self.A.tobytes()+self.shuffled.tobytes()+encode_json(self.rng.bit_generator.state).encode())

def read_panel(m,h,ledger=None):
    """No outcome or evaluator metadata. Returned distributions are frozen copies."""
    c=bytes(h)[-12:];lb=log_softmax(m.base_scores(c));j,s,margin=m.lookup(c)
    bound=c in m.members;tentative=j is not None and not bound
    support=0 if j is None else len(m.slot_members[j])
    lj=lb.copy() if j is None else log_softmax(np.frombuffer(m.values[j],dtype=np.float64))
    evidence=0. if j is None or ledger is None else ledger.E[j]
    names=POLICIES+('HARD',) if m.arm=='PRED-V' else ('EXACT-V',)
    weights={};logs={}
    for p in names:
        if j is None:a=0.
        elif bound:a=1.
        elif p=='HARD':a=1.
        elif p=='CORR2':a=float(support>=2)
        elif p=='ALPHA-0.50':a=.5
        elif support<2:a=0.
        elif p=='PBF':a=sigmoid(evidence)
        elif p=='SLOT-EVID':a=sigmoid(ledger.A[j])
        else:a=sigmoid(ledger.shuffled[j])
        weights[p]=a
        logs[p]=lb.copy() if a==0 else lj.copy() if a==1 else np.logaddexp(math.log1p(-a)+lb,math.log(a)+lj)
    logs['BASE']=lb;weights['BASE']=0.
    meta=dict(tentative=tentative,bound=bound,support=support,similarity=s,slot=j,evidence=evidence,
        log_base=lb,log_slot=lj,log_probs=logs)
    return {p:np.exp(v) for p,v in logs.items()},weights,meta

def score_panel(packet,y,old=None):
    probs,weights,meta=packet
    scored={p:dict(bpb=-float(meta['log_probs'][p][y])/math.log(2),accuracy=float(np.argmax(v)==y),
        target_probability=float(v[y]),obsolete_probability=None if old is None else float(v[old]),authority=weights[p]) for p,v in probs.items()}
    # Never serialize distribution vectors into per-event/probe output.
    public={k:v for k,v in meta.items() if not k.startswith('log_')}
    public.update(base_target_probability=float(probs['BASE'][y]),slot_target_probability=float(math.exp(meta['log_slot'][y])))
    return scored,public

class EvidenceDiagnostics:
    def __init__(self):
        self.groups=collections.defaultdict(Stats);self.quantiles=collections.defaultdict(list)
        self.deciles=collections.defaultdict(Stats);self.noise=[];self.trajectories=[]
    def event(self,packet,y,category):
        _,w,m=packet
        if not m['tentative']:return
        b=min(m['support'],3);lb=float(m['log_base'][y]);lj=float(m['log_slot'][y]);r=lj-lb
        row=dict(evidence=m['evidence'],authority=w['PBF'],base_bpb=-lb/math.log(2),slot_bpb=-lj/math.log(2),
            PBF_bpb=-float(m['log_probs']['PBF'][y])/math.log(2),slot_beats_base=float(r>0),predictive_advantage=r)
        for cat in ('all',category):
            self.groups[(cat,b)].add(row);self.quantiles[(cat,b)].append(m['evidence'])
        if category=='nontarget' and b==2:self.deciles[min(9,int(w['PBF']*10))].add(row)
    def checkpoint(self,name,m,e):
        # Per-slot at major wave checkpoints only, not per-presentation logs.
        for j,x in enumerate(e.E):self.trajectories.append(dict(checkpoint=name,slot=j,support=len(m.slot_members[j]),evidence=x,authority=0 if len(m.slot_members[j])<2 else sigmoid(x)))
    def result(self):
        support=[]
        for (c,b),s in sorted(self.groups.items()):
            q=np.quantile(self.quantiles[c,b],[0,.1,.25,.5,.75,.9,1]).tolist()
            support.append(dict(category=c,support='3+' if b==3 else str(b),**s.result(),**dict(zip(('E_min','E_p10','E_p25','E_median','E_p75','E_p90','E_max'),q))))
        return dict(support=support,deciles=[dict(decile=i,authority_low=i/10,authority_high=(i+1)/10,**self.deciles[i].result()) for i in range(10)],trajectories=self.trajectories,noise=self.noise)


def trial(seed,arm,smoke=False):
    start=time.perf_counter();stream=make_stream(seed,smoke=smoke);m=ByteModel(seed,arm)
    ledger=Evidence(seed);diagnostics=EvidenceDiagnostics();shuffle_checks=[]
    groups=collections.defaultdict(Stats);probes=[];events=[];checks=[];costs=[];emergence={};history=bytearray()
    support_hist=collections.Counter()
    cps=collections.defaultdict(list)
    for cp in stream['checkpoints']:cps[cp['record_count']].append(cp)
    def probe(name,kind,ids,aliases=(0,1,2),revised=False,extra=None,corrupted=None):
        for i in ids:
            y=stream['revisions'][i] if revised and i in stream['revisions'] else int(stream['labels'][i])
            for alias in aliases:
                # Freeze probabilities before scoring against evaluator-only labels.
                packet=read_panel(m,context(stream['keys'][i],alias),ledger)
                scores,meta=score_panel(packet,y,int(stream['labels'][i]))
                if corrupted is not None:
                    for p,v in packet[0].items():scores[p]['corrupt_target_probability']=float(v[corrupted])
                probes.append(dict(checkpoint=name,kind=kind,key_id=i,alias=alias,meta=meta,scores=scores,**(extra or {})))
    for number,row in enumerate(stream['rows'],1):
        for pos,y in enumerate(row['raw']):
            if pos==12 and row['kind']=='one_off_noise':
                before=(component_hashes(m),ledger.state_hash());probe('noise_before_'+str(row['key_id']),'noise', [row['key_id']],corrupted=y);probe('noise_others_before_'+str(row['key_id']),'noise_others',stream['untouched'])
                assert before==(component_hashes(m),ledger.state_hash())
            packet=read_panel(m,history,ledger)  # actual byte not passed to read policy
            scores,meta=score_panel(packet,y)
            category='target' if pos==12 else 'nontarget'
            if meta['tentative']:support_hist[(category,meta['support'])]+=1
            add_event(groups,scores,meta,category,row)
            if arm=='PRED-V':
                diagnostics.event(packet,y,category)
                if pos==12 and row['kind']=='one_off_noise':
                    contradiction=float(packet[2]['log_base'][y]-packet[2]['log_slot'][y])
                    E=packet[2]['evidence']
                    diagnostics.noise.append(dict(key_id=row['key_id'],evidence_before=E,
                        support=meta['support'],contradictory_log_evidence=contradiction,
                        hypothetical_contexts_to_offset=0 if E<=0 else math.ceil(E/contradiction) if contradiction>0 else None,
                        note='Equal-strength hypothetical independent contradictions; no evidence update on exact hits'))
            evidence_changed=ledger.update(packet,y) if arm=='PRED-V' else False
            members_before=len(m.members)
            reference_loss,reference_correct,_=m.observe(history,y)
            hard=scores['HARD' if arm=='PRED-V' else 'EXACT-V']
            if abs(reference_loss-hard['bpb'])>1e-10 or reference_correct!=hard['accuracy']:
                raise AssertionError('HARD prediction did not reproduce V36')
            ledger.synchronize(m,evidence_changed or len(m.members)!=members_before)
            history.append(y)
            if len(history)>12:del history[0]
            if pos==12:
                if row['kind']=='one_off_noise':
                    before=(component_hashes(m),ledger.state_hash());probe('noise_after_'+str(row['key_id']),'noise',[row['key_id']],corrupted=y);probe('noise_others_after_'+str(row['key_id']),'noise_others',stream['untouched'])
                    assert before==(component_hashes(m),ledger.state_hash())
                if row['kind']=='acquisition':
                    i=row['key_id'];c0,c1=[context(stream['keys'][i],a) for a in (0,1)]
                    j0,j1=m.members.get(c0),m.members.get(c1)
                    phase=emergence.setdefault(i,set())
                    e=None
                    if 'E1' not in phase and j0 is not None and j1 is None:e='E1'
                    elif 'E2' not in phase and j0 is not None and j0==j1:e='E2'
                    if e:
                        before=(component_hashes(m),ledger.state_hash());phase.add(e)
                        probe(e,'emergence',[i],(2,),extra=dict(bytes_observed=(number-1)*14+pos+1,
                            source_evidence=ledger.E[j0],source_slot=j0,source_support=len(m.slot_members[j0]),
                            clean_support_transition=(len(m.slot_members[j0])==(1 if e=='E1' else 2))))
                        assert before==(component_hashes(m),ledger.state_hash())
        for cp in cps.get(number,[]):
            before=(component_hashes(m),ledger.state_hash());name=cp['name'];k=stream['keys_per_wave'];wi=ord(cp['wave'])-65
            if cp['type']=='acquisition':probe(name,'current',range(wi*k,(wi+1)*k))
            elif cp['type']=='wave':
                probe(name,'untouched_A',stream['untouched']);probe(name,'current',range(wi*k,(wi+1)*k))
                if cp['wave']=='A':probe(name,'all_A',range(k))
                probe(name,'noise_A',stream['noisy']);probe(name,'revised_A',stream['revised'],revised=cp['wave']=='D')
            elif cp['type']=='revision':probe(name,'revision',stream['revised'],revised=True)
            elif cp['type']=='noise':probe(name,'noise_A',stream['noisy'])
            after=(component_hashes(m),ledger.state_hash())
            if before!=after:raise AssertionError('Read policy mutated state')
            checks.append(dict(checkpoint=name,components=before[0],evidence_sha256=before[1],all_policy_hashes_identical=True,read_only=True))
            shuffle_checks.append(dict(checkpoint=name,bins=ledger.audit(m)))
            if cp['type']=='wave':
                cost=m.costs();increment=8*len(ledger.E) if arm=='PRED-V' else 0
                cost.update(PBF_evidence_bytes=increment,evidence_bytes_per_slot=8 if arm=='PRED-V' else 0,
                    total_with_PBF_bytes=cost['total_payload_bytes']+increment,
                    operation_proxy_with_PBF=cost['operation_proxy_multiply_add_units']+ledger.events*5,
                    evidence_operation_notes='Increment proxy: two log-probability lookups, subtraction, addition, sigmoid; vector scoring and shuffled/SLOT-EVID controls excluded.')
                costs.append(dict(checkpoint=name,**cost))
                if arm=='PRED-V':diagnostics.checkpoint(name,m,ledger)
    address=slot_diagnostics(m,stream)
    # Compact final diagnostics; no per-context or per-byte traces in result files.
    for key in ('binding_trace','members_per_slot','logical_keys_per_slot','slots_per_acquired_logical_key'):address.pop(key,None)
    for key in ('nearest_similarity','top1_top2_margin'):address[key].pop('values',None)
    grouped={p:{g:a.result() for (pp,g),a in groups.items() if pp==p} for p in sorted({p for p,g in groups})}
    return dict(seed=seed,arm=arm,smoke=smoke,stream_sha256=stream['raw_sha256'],groups=grouped,
        probes=probes,checks=checks,random_matching={},evidence_diagnostics=diagnostics.result(),
        shuffled_audit=dict(checkpoints=shuffle_checks,audited_events=ledger.audit_count,shuffles=ledger.shuffles,changed_assignments=ledger.changed_assignments,all_marginals_exact=True),costs=costs,addressing=address,
        counts=exposure_accounting(stream),final_components=component_hashes(m),
        candidate_support_histogram=[dict(category=c,support=s,count=n) for (c,s),n in sorted(support_hist.items())],
        emergence_coverage=dict(keys=len(stream['keys']),E1=sum('E1' in x for x in emergence.values()),E2=sum('E2' in x for x in emergence.values())),
        elapsed_seconds=time.perf_counter()-start)

def qualification_gates(levels,contrasts,valid):
    v=lambda k:levels[k]['mean']
    minimum=lambda k:min(levels[k]['values']) if all(x is not None for x in levels[k]['values']) else -math.inf
    neg=lambda s:s['high'] is not None and s['high']<0
    pos=lambda s:s['low'] is not None and s['low']>0
    def specificity(d):
        gains=(d['whole_bpb']['mean']<=-.1 and neg(d['whole_bpb'])) or (d['nontarget_bpb']['mean']<=-.1 and neg(d['nontarget_bpb'])) or (d['revision_4_alias_2']['mean']>=.1 and pos(d['revision_4_alias_2']))
        safe=d['whole_bpb']['mean']<=.05 and d['nontarget_bpb']['mean']<=.05 and d['revision_4_alias_2']['mean']>=-.05
        return gains and safe
    c=contrasts['PBF - CORR2'];a=contrasts['PBF - ALPHA-0.50'];s=contrasts['PBF - SHUFFLED-PBF']
    return dict(Q0=valid,Q1=v('A_initial')>=.9 and v('A_after_D')>=.9 and v('D_current')>=.9 and minimum('A_after_D')>=.75,
        Q2=v('revision_4_alias_0')>=.9 and v('revision_4_alias_1')>=.9,
        Q3=v('revision_4_alias_2')>=.9 and minimum('revision_4_alias_2')>=.75,
        Q4=v('obsolete_8_alias_0')<=.1,Q5=v('whole_delta_base')<=.05,Q6=v('nontarget_delta_base')<=.05,
        Q7=v('false_merge_rate')<.05 and v('final_impurity')<.05,
        Q8=c['nontarget_bpb']['mean']<=-.1 and neg(c['nontarget_bpb']) and c['revision_4_alias_2']['mean']>=-.05,
        Q9=a['whole_bpb']['mean']<=.05 and a['revision_4_alias_2']['mean']>=.05,
        Q10=specificity(s))

def report(records,seeds,smoke):
    by={(r['seed'],r['arm']):r for r in records};complete=len(by)==2*len(seeds)
    rows=[];invariance=[]
    for seed in seeds:
        if (seed,'PRED-V') in by and (seed,'EXACT-V') in by:
            a,b=by[seed,'PRED-V'],by[seed,'EXACT-V']
            ok=len(a['checks'])==len(b['checks'])
            for ca,cb in zip(a['checks'],b['checks']):
                ok=ok and ca['checkpoint']==cb['checkpoint'] and all(ca['components'][k]==cb['components'][k] for k in ('base','eligibility'))
            if not ok:raise AssertionError('PRED-V/EXACT-V base or eligibility mismatch')
            invariance.append(dict(seed=seed,base_and_eligibility_identical=True,policy_state_shared=True,checkpoints=len(a['checks'])))
    for r in records:
        for p in r['groups']:
            if p!='BASE':rows.append(dict(seed=r['seed'],policy=p,trajectory=r['arm'],**policy_metrics(r,p)))
    levels={};qualification={};hyp={};contrasts={}
    if complete:
        ix={(r['seed'],r['policy']):r for r in rows}
        for p in POLICIES+('HARD','EXACT-V'):
            levels[p]={k:stat([ix[s,p][k] for s in seeds]) for k in rows[0] if k not in ('seed','policy','trajectory')}
        for p in ('CORR2','ALPHA-0.50','SHUFFLED-PBF','SLOT-EVID'):
            contrasts['PBF - '+p]={k:stat([ix[s,'PBF'][k]-ix[s,p][k] for s in seeds]) for k in ('whole_bpb','nontarget_bpb','target_bpb','A_after_D','revision_4_alias_2')}
        valid=len(invariance)==len(seeds) and all(r['shuffled_audit']['all_marginals_exact'] and all(c['read_only'] for c in r['checks']) for r in records)
        gates=qualification_gates(levels['PBF'],contrasts,valid)
        qualification['PBF']=dict(gates=gates,qualified=all(gates.values()) and not smoke)
        h1=[]
        for seed in seeds:
            g=by[seed,'PRED-V']['groups'];h=g['HARD'].get('nontarget_support_2');b=g['BASE'].get('nontarget_support_2')
            h1.append(h['bpb']-b['bpb'] if h and b else None)
        rep=stat(h1);neg=lambda s:s['high'] is not None and s['high']<0
        c=contrasts['PBF - CORR2'];a=contrasts['PBF - ALPHA-0.50'];blind=contrasts['PBF - SLOT-EVID']
        h5=(blind['whole_bpb']['mean']<=-.1 and neg(blind['whole_bpb'])) or (blind['nontarget_bpb']['mean']<=-.1 and neg(blind['nontarget_bpb'])) or (blind['revision_4_alias_2']['mean']>=.1 and blind['revision_4_alias_2']['low'] is not None and blind['revision_4_alias_2']['low']>0)
        h5=h5 and blind['whole_bpb']['mean']<=.05 and blind['nontarget_bpb']['mean']<=.05 and blind['revision_4_alias_2']['mean']>=-.05
        hyp=dict(H1=dict(paired_hard_minus_base=rep,replicated=rep['low'] is not None and rep['low']>0 and rep['mean']>=.1),
            H2=dict(supported=gates['Q8'] and neg(c['whole_bpb'])),H3=dict(supported=gates['Q9']),H4=dict(supported=gates['Q10']),H5=dict(supported=h5,note='Nonsuperiority does not establish equivalence.'))
        qualification['PBF']['mechanism_supported']=qualification['PBF']['qualified'] and hyp['H1']['replicated'] and h5
    return dict(version=VERSION,status='complete' if complete else 'partial',scientific=not smoke,
        completed_trajectories=len(records),expected_trajectories=2*len(seeds),seeds=list(seeds),levels=levels,qualification=qualification,
        hypotheses=hyp,paired_contrasts=contrasts,invariance=invariance,per_seed=rows,implementation_notes=IMPLEMENTATION_NOTES,
        roles=dict(stateful=MAIN_ARMS,deployable=('CORR2','ALPHA-0.50','PBF','SLOT-EVID','EXACT-V'),nondeployable=('SHUFFLED-PBF',),diagnostics=('BASE','HARD')),
        limitations=['Unbounded storage','One-off corruption intentionally uncorrected','No authority tuning','Engineering outcomes cannot qualify science'])

def summary_text(rep):
    lines=['# V38 — Predictive Evidence Arbitration','',f"Status: {rep['status']}; {rep['completed_trajectories']}/{rep['expected_trajectories']} trajectories.",
        'Scientific run.' if rep['scientific'] else 'ENGINEERING ONLY — not scientific evidence.','',
        '| Policy | A after D | Exact @4 | Known @4 | Heldout @4 | Whole bpb | Nontarget bpb |','|---|---:|---:|---:|---:|---:|---:|']
    for p,l in rep['levels'].items():
        vals=[f"{100*l[k]['mean']:.2f}%" for k in ('A_after_D','revision_4_alias_0','revision_4_alias_1','revision_4_alias_2')]
        vals += [f"{l[k]['mean']:.4f}" for k in ('whole_bpb','nontarget_bpb')]
        lines.append('| '+p+' | '+' | '.join(vals)+' |')
    if rep['qualification']:
        q=rep['qualification']['PBF'];lines+=['',f"PBF qualified: {q['qualified']}. Full comparative mechanism supported: {q['mechanism_supported']}.",
            'Failed gates: '+', '.join(k for k,v in q['gates'].items() if not v)]
    lines+=['','One unchanged PRED-V trajectory carries all read policies. SHUFFLED-PBF is nondeployable.',
        'Paired 95% Student-t intervals use seeds as sampling units. No post-hoc selection or follow-up.',
        'Storage remains unbounded; noise/revision protection remains unresolved.','']
    return '\n'.join(lines)


def save_outputs(root,records,rep,identity,source):
    write_json(root/'REPORT.json',rep);atomic(root/'SUMMARY.md',summary_text(rep))
    csv_output(root/'SUMMARY.csv',rep['per_seed'])
    support=[];usage=[];retention=[];revision=[];emergence=[];noise=[];acquisition=[];costs=[];states=[]
    for r in records:
        for p,g in r['groups'].items():
            for name,v in g.items():
                if '_support_' in name:support.append(dict(seed=r['seed'],trajectory=r['arm'],policy=p,group=name,**v))
                if name in ('target','nontarget'):
                    tentative=v['tentative_rate'];usage.append(dict(seed=r['seed'],trajectory=r['arm'],policy=p,category=name,
                        tentative_count=round(v['n']*tentative),nonzero_fraction_given_tentative=v['nonzero_tentative_authority']/tentative if tentative else None,
                        mean_weight_given_tentative=v['tentative_authority_weight']/tentative if tentative else None,**v))
                if name.startswith(('acquisition_','render_acquisition_','revision_')):
                    acquisition.append(dict(seed=r['seed'],trajectory=r['arm'],policy=p,group=name,**v))
        for e in r['probes']:
            for p,v in e['scores'].items():
                out=dict(seed=r['seed'],trajectory=r['arm'],policy=p,checkpoint=e['checkpoint'],kind=e['kind'],key_id=e['key_id'],alias=e['alias'],
                    evidence=e['meta']['evidence'],base_target_probability=e['meta']['base_target_probability'],slot_target_probability=e['meta']['slot_target_probability'],candidate_support=e['meta']['support'],tentative=e['meta']['tentative'],candidate_similarity=e['meta']['similarity'],**v)
                if e['kind']=='emergence':
                    out.update({k:e[k] for k in ('source_slot','source_support','source_evidence','clean_support_transition','bytes_observed')});emergence.append(out)
                elif e['kind']=='revision':revision.append(out)
                elif e['kind'].startswith('noise'):noise.append(out)
                elif e['checkpoint'].startswith('after_'):retention.append(out)
                else:acquisition.append(out)
        costs.append(dict(seed=r['seed'],trajectory=r['arm'],data=r['costs']))
        states.append(dict(seed=r['seed'],trajectory=r['arm'],checks=r['checks'],final_components=r['final_components'],random_matching=r['random_matching'],emergence_coverage=r['emergence_coverage']))
    for name,rows in [('AUTHORITY_BY_SUPPORT',support),('NONTARGET_AUTHORITY',usage),('RETENTION_BY_WAVE',retention),('REVISION_CURVES',revision),('HELDOUT_E1_E2',emergence),('ACQUISITION_CURVES',acquisition)]:csv_output(root/(name+'.csv'),rows)
    # Key-paired E2-E1 emergence, including clean support-1/2 subset.
    deltas=[]
    for r in records:
        for p in r['groups']:
            e1={e['key_id']:e for e in r['probes'] if e['kind']=='emergence' and e['checkpoint']=='E1'}
            e2={e['key_id']:e for e in r['probes'] if e['kind']=='emergence' and e['checkpoint']=='E2'}
            for clean in (False,True):
                ids=[i for i in e1.keys()&e2.keys() if not clean or e1[i]['source_support']==1 and e2[i]['source_support']==2]
                deltas.append(dict(seed=r['seed'],trajectory=r['arm'],policy=p,clean_support_1_to_2=clean,n_keys=len(ids),
                    accuracy_delta=mean(e2[i]['scores'][p]['accuracy']-e1[i]['scores'][p]['accuracy'] for i in ids),
                    target_probability_delta=mean(e2[i]['scores'][p]['target_probability']-e1[i]['scores'][p]['target_probability'] for i in ids)))
    csv_output(root/'CORROBORATION_PAIRED_DELTAS.csv',deltas)
    write_json(root/'CANDIDATE_SUPPORT_HISTOGRAM.json',[dict(seed=r['seed'],trajectory=r['arm'],data=r['candidate_support_histogram']) for r in records])
    write_json(root/'STATE_INVARIANCE.json',dict(paired_trajectories=rep['invariance'],records=states))
    write_json(root/'NOISE.json',noise);write_json(root/'STATE_COSTS.json',costs)
    write_json(root/'EXPOSURES.json',[dict(seed=r['seed'],trajectory=r['arm'],**r['counts']) for r in records])
    write_json(root/'ADDRESSING.json',[dict(seed=r['seed'],trajectory=r['arm'],**r['addressing']) for r in records])
    for filename,key in [('EVIDENCE_BY_SUPPORT','support'),('EVIDENCE_DECILES','deciles'),('EVIDENCE_TRAJECTORIES','trajectories')]:
        csv_output(root/(filename+'.csv'),[dict(seed=r['seed'],**x) for r in records if r['arm']=='PRED-V' for x in r['evidence_diagnostics'][key]])
    csv_output(root/'PBF_PAIRED_DELTAS.csv',[dict(contrast=c,endpoint=k,**{a:b for a,b in v.items() if a!='values'}) for c,d in rep['paired_contrasts'].items() for k,v in d.items()])
    write_json(root/'SHUFFLED_PBF_AUDIT.json',[dict(seed=r['seed'],trajectory=r['arm'],**r['shuffled_audit']) for r in records])
    write_json(root/'NOISE_DIAGNOSTIC.json',dict(probability_probes=noise,evidence=[dict(seed=r['seed'],**x) for r in records for x in r['evidence_diagnostics']['noise']]))
    paths=[p for p in root.rglob('*') if p.is_file() and not p.name.startswith('.') and '.tmp-' not in p.name and p.name!='ARTIFACT_HASHES.json']
    write_json(root/'ARTIFACT_HASHES.json',{str(p.relative_to(root)):sha(p.read_bytes()) for p in paths})
    archive=Path(str(root)+'.zip');temp=Path(str(archive)+'.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths+[root/'ARTIFACT_HASHES.json']:z.write(p,p.relative_to(root))
    os.replace(temp,archive)
    return archive

def validate_manifest(existing,expected):
    if existing!=expected:raise RuntimeError('Resume rejected: source, protocol, plan or environment changed')

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-deps',action='store_true');parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--verify-full',action='store_true',help='Full-length independent engineering replay; no scientific seeds')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--out-dir','--out',type=Path)
    parser.add_argument('--max-new-jobs',type=int,help='Pause at a completed trajectory boundary')
    args=parser.parse_args(argv)
    if args.max_new_jobs is not None and args.max_new_jobs<0:parser.error('--max-new-jobs must be nonnegative')
    dependencies(args.install_deps);caffeine=None
    if sys.platform=='darwin':
        try:caffeine=subprocess.Popen(['caffeinate','-dimsu','-w',str(os.getpid())])
        except FileNotFoundError:pass
    try:
        if args.self_test or args.verify_full:
            print(json.dumps(self_test(args.verify_full),indent=2));return
        root=(args.out_dir or Path('v38_smoke' if args.smoke else 'v38_results')).resolve();root.mkdir(parents=True,exist_ok=True)
        source=Path(__file__).read_bytes();seeds=ENGINEERING_SEEDS if args.smoke else SEEDS
        plan=dict(seeds=list(seeds),arms=list(MAIN_ARMS),policies=list(POLICIES),smoke=args.smoke,
            streams={str(s):make_stream(s,smoke=args.smoke)['raw_sha256'] for s in seeds},
            implementation_notes=IMPLEMENTATION_NOTES,resume='Completed trajectories only; interrupted trajectory deterministically restarts')
        identity=dict(source_sha256=sha(source),protocol_sha256=sha(PROTOCOL.encode()),plan_sha256=sha(plan))
        manifest=dict(version=VERSION,identity=identity,plan=plan,reference_v37_sha256=REFERENCE_SHA256,
            environment=dict(python=sys.version,numpy=np.__version__,scipy=importlib.import_module('scipy').__version__,platform=platform.platform(),blas_threads=1))
        with RunLock(root):
            path=root/'RUN_MANIFEST.json'
            if path.exists():
                validate_manifest(json.loads(path.read_text()),manifest)
            else:
                if any(root.glob('jobs/*.json')):raise RuntimeError('Orphan jobs without manifest')
                write_json(path,manifest)
            atomic(root/Path(__file__).name,source.decode());atomic(root/'PROTOCOL.md',PROTOCOL)
            jobs=root/'jobs';jobs.mkdir(exist_ok=True)
            allowed={f'{s}_{a}.json' for s in seeds for a in MAIN_ARMS}
            if any(p.name not in allowed for p in jobs.glob('*.json')):raise RuntimeError('Foreign cached job')
            records=[];new=0;reused=0;stop=False
            for seed in seeds:
                for arm in MAIN_ARMS:
                    p=jobs/f'{seed}_{arm}.json'
                    if p.exists():
                        e=json.loads(p.read_text());r=e['result']
                        if e['identity']!=identity or e['sha256']!=sha(r) or (r['seed'],r['arm'],r['smoke'],r['stream_sha256'])!=(seed,arm,args.smoke,plan['streams'][str(seed)]):raise RuntimeError('Corrupt cached trajectory '+p.name)
                        if r['counts']['byte_events']!=r['groups']['BASE']['whole']['n']:raise RuntimeError('Invalid exposure counts')
                        reused+=1
                    else:
                        if args.max_new_jobs is not None and new>=args.max_new_jobs:stop=True;break
                        print(f'Starting {seed} {arm}',flush=True);r=trial(seed,arm,args.smoke)
                        write_json(p,dict(identity=identity,sha256=sha(r),result=r));new+=1
                        print(f"Completed {seed} {arm} in {r['elapsed_seconds']:.1f}s",flush=True)
                    records.append(r)
                    report(records,seeds,args.smoke) # fail immediately on paired state mismatch
                if stop:break
            rep=report(records,seeds,args.smoke)
            write_json(root/'STATUS.json',dict(status=rep['status'],completed=len(records),new=new,reused=reused))
            archive=save_outputs(root,records,rep,identity,source)
            print(f"{rep['status']}: {new} new, {reused} reused. {archive}",flush=True)
    finally:
        if caffeine is not None:caffeine.terminate()

def self_test(full=False):
    tests=[]
    def check(n,v):
        if not v:raise AssertionError(n)
        tests.append(n)
    m=ByteModel(ENGINEERING_SEEDS[0],'PRED-V');e=Evidence(123)
    c=context(b'ABCDEF',0);ca=c[:-1]+b';';held=c[:-1]+b'8'
    m.eligible.add(c);j=m.bind(c,None);v=np.zeros(256);v[97]=.8;m.values[j]=v.tobytes();e.synchronize(m)
    packet=read_panel(m,held,e)
    check('singleton PBF shadow',packet[2]['tentative'] and packet[1]['PBF']==0)
    e.update(packet,97);m.eligible.add(ca);m.bind(ca,j);e.synchronize(m,True)
    check('positive advantage future authority',read_panel(m,held,e)[1]['PBF']>.5)
    frozen=read_panel(m,held,e);old=frozen[0]['PBF'].copy()
    fake=copy.deepcopy(frozen);fake[2]['log_slot'][98]=-10.;fake[2]['log_base'][98]=-.01
    prev=e.E[j];e.update(fake,98);e.synchronize(m,True)
    check('negative advantage reduces future trust',e.E[j]<prev and read_panel(m,held,e)[1]['PBF']<frozen[1]['PBF'])
    check('current distribution remains frozen',np.array_equal(old,frozen[0]['PBF']))
    exact=read_panel(m,c,e);prev=e.E.tobytes();e.update(exact,98)
    check('exact ownership and no exact evidence',all(exact[1][p]==1 for p in POLICIES) and prev==e.E.tobytes())
    before=(component_hashes(m),e.state_hash())
    for _ in range(3):read_panel(m,held,e)
    check('repeated probes read only',before==(component_hashes(m),e.state_hash()))
    p=read_panel(m,held,e)
    check('CORR2 frozen definition',p[1]['CORR2']==1 and np.array_equal(p[0]['CORR2'],p[0]['HARD']))
    check('alpha half frozen definition',np.allclose(p[0]['ALPHA-0.50'],.5*p[0]['BASE']+.5*p[0]['HARD'],atol=1e-15))
    e.E[j]=0;fake=copy.deepcopy(p);fake[2]['log_slot'][97]=math.log(.98);fake[2]['log_base'][97]=math.log(.001)
    e.update(fake,97);e.synchronize(m,True)
    check('high Bayes second rendering near one',read_panel(m,held,e)[1]['PBF']>.998)
    e.E[j]=0;fake[2]['log_slot'][97]=math.log(.55);fake[2]['log_base'][97]=math.log(.95);e.update(fake,97)
    check('redundant confirmation reduces trust',e.E[j]<0)
    for i in range(8):
        cc=bytes([i])*12;m.eligible.add(cc);m.bind(cc,None)
    e.synchronize(m);e.E[:]=array('d',range(len(e.E)));e.synchronize(m,True);e.audit(m)
    check('shuffled exact multisets',e.audit_count>0)
    check('shuffled different assignments',e.changed_assignments>0)
    check('finite log probabilities extreme scores',np.isfinite(log_softmax(np.arange(256.)*1000)).all())
    runs=[]
    for arm in MAIN_ARMS:
        r=trial(ENGINEERING_SEEDS[0],arm,not full)
        model=ByteModel(ENGINEERING_SEEDS[0],arm);h=bytearray();totals=[0.,0.,0.]
        stream=make_stream(ENGINEERING_SEEDS[0],smoke=not full)
        for row in stream['rows']:
            for pos,y in enumerate(row['raw']):
                l,_,_=model.observe(h,y);totals[0]+=l;totals[1 if pos==12 else 2]+=l
                h.append(y)
                if len(h)>12:del h[0]
        check(arm+' unchanged full learner state and binding',r['final_components']==component_hashes(model))
        p='HARD' if arm=='PRED-V' else arm
        for k,total,n in zip(('whole','target','nontarget'),totals,(len(stream['raw']),len(stream['rows']),13*len(stream['rows']))):
            check(arm+' historical '+k+' loss',abs(r['groups'][p][k]['bpb']-total/n)<1e-11)
        runs.append(r)
    rep=report(runs,ENGINEERING_SEEDS[:1],True)
    check('paired common base eligibility',len(rep['invariance'])==1)
    # Test every absolute threshold independently on constructed, passing data.
    levels={k:dict(mean=1.,values=[1.,1.]) for k in ('A_initial','A_after_D','D_current','revision_4_alias_0','revision_4_alias_1','revision_4_alias_2')}
    levels.update({k:dict(mean=0.,values=[0.,0.]) for k in ('obsolete_8_alias_0','whole_delta_base','nontarget_delta_base','false_merge_rate','final_impurity')})
    contrasts={}
    for p in ('CORR2','ALPHA-0.50','SHUFFLED-PBF'):
        contrasts['PBF - '+p]={k:dict(mean=.2 if k=='revision_4_alias_2' else -.2,low=.1 if k=='revision_4_alias_2' else -.3,high=.3 if k=='revision_4_alias_2' else -.1) for k in ('whole_bpb','nontarget_bpb','revision_4_alias_2')}
    check('all synthetic qualification gates pass',all(qualification_gates(levels,contrasts,True).values()))
    for k,bad in [('A_initial',.89),('A_after_D',.89),('D_current',.89),('revision_4_alias_0',.89),('revision_4_alias_1',.89),('revision_4_alias_2',.89),('obsolete_8_alias_0',.11),('whole_delta_base',.051),('nontarget_delta_base',.051),('false_merge_rate',.05),('final_impurity',.05)]:
        v=copy.deepcopy(levels);v[k]['mean']=bad
        check('reject '+k,not all(qualification_gates(v,contrasts,True).values()))
    for k in ('A_after_D','revision_4_alias_2'):
        v=copy.deepcopy(levels);v[k]['values']=[.74,1.]
        check('reject seed floor '+k,not all(qualification_gates(v,contrasts,True).values()))
    check('reject failed invariance',not qualification_gates(levels,contrasts,False)['Q0'])
    for p,gate in [('CORR2','Q8'),('ALPHA-0.50','Q9'),('SHUFFLED-PBF','Q10')]:
        d=copy.deepcopy(contrasts)
        for v in d['PBF - '+p].values():v.update(mean=0.,low=0.,high=0.)
        check('reject null '+gate,not qualification_gates(levels,d,True)[gate])
    # Manifest identity protects source, protocol and plan; real recovery is tested via CLI.
    identity=dict(source_sha256='s',protocol_sha256='p',plan_sha256='r')
    for k in identity:
        other=dict(identity);other[k]+='changed'
        rejected=False
        try:validate_manifest(dict(identity=other),dict(identity=identity))
        except RuntimeError:rejected=True
        check('resume rejects '+k,rejected)
    return dict(passed=True,checks=tests,full_length=full,engineering_seed=ENGINEERING_SEEDS[0],independent_replay_components={r['arm']:r['final_components'] for r in runs})

if __name__=='__main__':main()
