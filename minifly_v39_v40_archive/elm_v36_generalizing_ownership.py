#!/usr/bin/env python3
"""V36: Generalizing Ownership.
Run: python3 elm_v36_generalizing_ownership.py --install-deps
48 main scientific jobs; 16 fly/hash jobs only if the geometry gate passes.
--smoke uses engineering seeds; --verify runs engineering observer tests; scientific seeds are not used.
"""
from __future__ import annotations
import argparse, ast, base64, collections, copy, csv, hashlib, importlib, io
import json, math, os, platform, subprocess, sys, time, zipfile, zlib
from pathlib import Path
for _key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
np=None
student_t=None
D=96
BANKS=8
OUTPUTS=256
RATE=0.6
TEMP=0.1
PREFIX=b'~|~|'
HEADERS=(b'@',b'#',b'?')
SEPARATORS=(b':',b'=',b'>')

VERSION='v36.1.0'
SEEDS=tuple(range(36101,36109))
PREFLIGHT_SEEDS=tuple(range(963601,963605))
MAIN_ARMS=('EXACT-R','EXACT-V','SIM-R','SIM-V','PRED-V','FROZEN-R')
FLY_ARMS=('FLY-O','HASH-O')
TAU=.85
PROTOCOL="# V36 — Generalizing Ownership\n## Refined design after V35 and the second fly re-audit\n\n## Status\n\n**Design / preregistration only. No V36 scientific outcomes have been observed.**\n\nV36 follows the completed V35 result.\n\nThe central V35 finding was not merely that exact local memory retains. The decisive causal contrast was:\n\n\\[\n\\text{LOCAL-R}: s=s_B+r_c\n\\]\n\nversus\n\n\\[\n\\text{LOCAL-V}: s=v_c.\n\\]\n\nBoth arms used:\n\n- the same V32 temporal base;\n- the same exact raw 12-byte address;\n- the same eligibility schedule;\n- the same number of local vectors;\n- the same 0.6 local update scale.\n\nYet LOCAL-R retained roughly 30% of untouched A while LOCAL-V retained roughly 99%.\n\nTherefore the large LOCAL-V − LOCAL-R gap is **not an addressing or capacity effect**. It isolates whether owned state is interpreted relative to a drifting shared base or owns the absolute prediction.\n\nAt the same time, LOCAL-V transferred essentially no revision to the other rendering of the same logical key.\n\nV36 therefore asks:\n\n\\[\n\\boxed{\n\\textbf{Can absolute local ownership be made to generalize across causally similar contexts\nwithout giving up its resistance to unrelated overwrite?}\n}\n\\]\n\nThis is now more urgent than bounded capacity.\n\nBounded capacity is deferred until we know whether the owned state can represent a *logical object* rather than only an exact byte string.\n\n---\n\n# 1. Empirical constraints inherited from V35\n\nThe following V35 conclusions are treated as fixed evidence.\n\n## 1.1 Exact local absolute ownership works\n\nLOCAL-V achieved approximately:\n\n- 99% untouched-A retention after D;\n- 100% D-current acquisition;\n- immediate exact revision after a contradiction;\n- no collateral damage to unrelated exact local memories;\n- whole-stream prediction better than the shared base.\n\nThis establishes exact-context ownership as a real anti-overwrite primitive.\n\nIt does **not** establish a bounded lifelong architecture.\n\n## 1.2 Additive local residuals remain vulnerable to base drift\n\nLOCAL-R and LOCAL-V had identical exact addressing and capacity.\n\nThe sole architectural difference was:\n\n\\[\ns_B+r_c\n\\quad\\text{versus}\\quad\nv_c.\n\\]\n\nThe V35 base-drift diagnostics showed that the shared base score on old contexts moved substantially after later waves.\n\nTherefore V36 includes a frozen-base LOCAL-R control to causally verify that this drift explains the LOCAL-R failure.\n\nThat control is diagnostic only and is not a deployable candidate.\n\n## 1.3 Exact ownership does not create logical-object identity\n\nFor true revision of rendering 0:\n\n- LOCAL-V revised rendering 0 rapidly;\n- the already learned alternate rendering remained essentially at the obsolete value;\n- the held-out rendering did not inherit the new value.\n\nThis failure is mechanical under exact addressing.\n\nThe result separates:\n\n\\[\n\\boxed{\\text{storage ownership}}\n\\]\n\nfrom:\n\n\\[\n\\boxed{\\text{addressing / identity formation}}.\n\\]\n\n## 1.4 The alias endpoint is task-valid\n\nThe V35 current-fact minimum-norm oracle can transfer revision to the never-trained alias substantially above the live learners.\n\nThus failure of held-out alias transfer is not an intrinsic impossibility of the workload.\n\nV36 may use alias transfer as a genuine endpoint.\n\n---\n\n# 2. The missing 2×2\n\nV35 tested only the left column:\n\n| value semantics | exact address | similarity / logical address |\n|---|---:|---:|\n| additive residual on shared base | LOCAL-R | **untested** |\n| absolute locally owned value | LOCAL-V | **untested** |\n\nV36 fills the right column.\n\nThe principal live causal comparison is therefore:\n\n\\[\n\\boxed{\n\\text{exact vs similarity-linked addressing}\n\\times\n\\text{residual vs absolute value semantics}\n}\n\\]\n\nwith the same shared base.\n\n---\n\n# 3. What the fly re-audit contributes — and what it does not\n\nThe additional connectome analysis motivates a **continuum of overlap** rather than exact identity versus global sharing.\n\nThe useful structural idea is:\n\n> related inputs should reactivate substantially overlapping local substrate, while unrelated inputs should overlap much less.\n\nThis could naturally supply:\n\n- retention through low unrelated overlap;\n- generalization through partial related overlap;\n- revision through reactivation of the same local substrate.\n\nHowever, V36 does **not** assume that simple width plus sparsity automatically creates this regime.\n\n## 3.1 V34 already tested a nearby sparse-expansion hypothesis\n\nV34 used a fixed 768-unit expansion with global Top48 sparsity.\n\nIts fixed code produced approximately:\n\n\\[\nJ_{\\rm same}\\approx0.349\n\\]\n\nfor the two trained renderings and\n\n\\[\nJ_{\\rm different}\\approx0.079,\n\\]\n\na separation ratio of only about:\n\n\\[\n4.4\\times.\n\\]\n\nThe global sparse ordinary arm learned well but retained only about 6% of untouched A after D.\n\nThus:\n\n\\[\n\\boxed{\n\\text{sparse expansion at a few-fold separation can delay interference without producing persistence}\n}\n\\]\n\nis already established.\n\nV36 will not repeat V34 under a larger width unless a read-only representation gate first demonstrates a qualitatively stronger separation regime.\n\n## 3.2 The current byte workload contains large common syntax\n\nA learned record context is:\n\n\\[\n\\texttt{~|~|} + \\text{header} + \\text{6-byte key} + \\text{separator}.\n\\]\n\nTwo renderings of the same key differ only in header and separator and share 10 of 12 raw causal bytes.\n\nBut different logical keys also share substantial fixed syntax.\n\nTherefore a random sparse expansion cannot be assumed to behave like odor patterns whose unrelated sensory-channel overlap is near zero.\n\nA useful fly-like code for byte data may ultimately need recurrent/statistical preprocessing that discounts ubiquitous syntax.\n\nThat is not added to V36.\n\n---\n\n# 4. Non-negotiable causal interface\n\nEvery deployable V36 candidate receives only:\n\n\\[\n\\text{past bytes}\n\\rightarrow\n\\text{prediction}\n\\rightarrow\n\\text{actual next byte}\n\\rightarrow\n\\text{online update}.\n\\]\n\nNo learner receives:\n\n- wave ID;\n- task ID;\n- logical-key ID;\n- alias ID;\n- target-position ID;\n- revision flag;\n- corruption flag;\n- semantic memory ID;\n- future information.\n\nLogical identity and alias metadata remain evaluator-only.\n\nThere is no replay of untouched A.\n\nThere is no reset at wave or record boundaries.\n\n---\n\n# 5. Fixed workload\n\nReuse the exact V35 / V34 multi-wave generator:\n\n\\[\nA\\rightarrow B\\rightarrow C\\rightarrow D.\n\\]\n\nPer full trajectory:\n\n- 64 logical keys;\n- 16 keys per wave;\n- 32 initial target presentations per key;\n- two alternating learned renderings;\n- one never-trained alias;\n- four one-off corrupt A keys;\n- four disjoint truly revised A keys;\n- eight untouched A keys;\n- 2,084 target events;\n- 29,176 total next-byte prediction events.\n\nUnique examples and total presentations must be reported separately.\n\n---\n\n# 6. Fresh scientific seeds\n\nUse:\n\n\\[\n\\boxed{36101,\\ldots,36108}\n\\]\n\nfor the main ownership/addressing block.\n\nUse separate fixed engineering/preflight seeds:\n\n\\[\n\\boxed{963601,\\ldots,963604}.\n\\]\n\nBefore science, audit both blocks against all accessible prior manifests and embedded source seed lists.\n\nIf any collision is found before outcomes exist, replace the entire affected block with the next unused consecutive block and lock it.\n\n---\n\n# 7. Shared base — held fixed\n\nAll main V36 arms use the exact source-locked V35 Stage-II base:\n\n> the V32 current temporal full-covariance causal byte learner.\n\nThe implementation is copied from the executing V35 source, including:\n\n- current temporal representation;\n- fixed eight-bank router;\n- full covariance writer;\n- historical 0.6 update scale;\n- historical direction cap;\n- softmax temperature 0.1.\n\nAs in V35, auxiliary output must **never** affect base learning.\n\nFor every main arm and seed:\n\n- base prediction before auxiliary combination is identical;\n- base update is identical;\n- base state hashes must be identical at every checkpoint.\n\nA separate BASE trajectory is unnecessary.\n\nThe runner records counterfactual base-only loss and accuracy from every arm.\n\nAny base-state mismatch invalidates the experiment.\n\n---\n\n# 8. Private address representation\n\nUse the exact V35 lag96 address code:\n\n\\[\nk(c)\\in\\mathbb R^{96}.\n\\]\n\nNo encoder learning.\n\nNo retuning.\n\nBefore science report on the V36 streams:\n\n- same-key cosine distributions for rendering 0/1 and 0/2;\n- different-key distributions;\n- full tails, not merely means;\n- nearest-neighbor ranks;\n- false-neighbor rates at the prespecified similarity threshold.\n\n---\n\n# 9. Main arm roster\n\nSix live arms are run on the eight fresh scientific seeds:\n\n| Arm | Addressing | Value semantics | Role |\n|---|---|---|---|\n| **EXACT-R** | exact 12-byte context | additive residual \\(s_B+r_c\\) | V35 LOCAL-R replication |\n| **EXACT-V** | exact 12-byte context | absolute local \\(v_c\\) | V35 LOCAL-V replication |\n| **SIM-R** | similarity-linked slot | additive residual \\(s_B+r_j\\) | missing 2×2 cell |\n| **SIM-V** | similarity-linked slot | absolute local \\(v_j\\) | primary missing cell |\n| **PRED-V** | similarity + predictive confirmation | absolute local \\(v_j\\) | self-supervised identity candidate |\n| **FROZEN-R** | exact context | additive residual | nondeployable base-drift causal control |\n\nThus:\n\n\\[\n6\\times8=\\boxed{48\\text{ main scientific trajectories}}.\n\\]\n\nNo capacity bound or eviction is imposed in the main block.\n\n---\n\n# 10. Common V35 eligibility rule\n\nTo preserve comparability, retain V35's parameter-free local-state eligibility concept.\n\nThe base-only prediction is computed before any auxiliary update.\n\nAfter observing byte \\(y\\):\n\n- if the base top-1 prediction was wrong, the exact current context becomes eligible for auxiliary state;\n- once eligible it remains eligible.\n\nFor SIM/PRED arms, similarity retrieval can make a tentative prediction before the exact context has permanent membership; permanent binding is specified below.\n\nThe base-error rule is **not** interpreted as a revision detector or special biological mechanism.\n\nReport target-eligibility coverage as in V35.\n\n---\n\n# 11. EXACT-R\n\nIdentical in meaning to V35 LOCAL-R.\n\nEach eligible exact context \\(c\\) owns:\n\n\\[\nr_c\\in\\mathbb R^{256}.\n\\]\n\nPrediction:\n\n\\[\ns=s_B+r_c.\n\\]\n\nUpdate:\n\n\\[\nr_c^+\n=\nr_c\n+\n0.6(y-s).\n\\]\n\nOnly \\(r_c\\) changes.\n\nAll other local states are bit-for-bit unchanged.\n\n---\n\n# 12. EXACT-V\n\nIdentical in meaning to V35 LOCAL-V.\n\nEach eligible exact context owns:\n\n\\[\nv_c\\in\\mathbb R^{256}.\n\\]\n\nPrediction:\n\n\\[\ns=v_c.\n\\]\n\nUpdate:\n\n\\[\nv_c^+\n=\n0.4v_c+0.6y.\n\\]\n\nThe shared base continues learning independently but does not affect the prediction of an owned exact context.\n\n---\n\n# 13. Similarity slot representation\n\nSIM-R, SIM-V, and PRED-V store local **slots**.\n\nA slot \\(j\\) contains:\n\n- a set of exact member contexts \\(M_j\\);\n- the lag96 vector of each member;\n- a normalized prototype \\(p_j\\), equal to the normalized mean of member lag vectors;\n- one value state:\n  - residual \\(r_j\\) for SIM-R;\n  - absolute \\(v_j\\) for SIM-V/PRED-V.\n\nNo logical-key ID is stored.\n\nNo alias ID is stored.\n\nNo evaluator metadata enters the slot.\n\n## 13.1 Fixed similarity threshold\n\nUse:\n\n\\[\n\\boxed{\\tau=0.85}.\n\\]\n\nThis threshold is selected **before V36** from V35's already observed lag96 geometry.\n\nIt is not tuned on V36 outcomes.\n\nThe V36 report must display the entire V36 same/different similarity distributions so the threshold's operating characteristics are transparent.\n\n---\n\n# 14. Similarity retrieval\n\nFor current exact context \\(c\\):\n\n### Exact member hit\n\nIf \\(c\\in M_j\\) for some slot, retrieve that slot directly.\n\n### Unbound context\n\nOtherwise compute:\n\n\\[\ns_j=p_j^\\top k(c)\n\\]\n\nfor every slot.\n\nLet:\n\n\\[\nj^*=\\arg\\max_j s_j.\n\\]\n\nIf:\n\n\\[\ns_{j^*}\\ge\\tau,\n\\]\n\nthe slot is a **tentative similarity candidate** for the current prediction.\n\nOtherwise there is no auxiliary retrieval and prediction is base-only.\n\nThis tentative lookup is causal and requires no identity label.\n\n---\n\n# 15. SIM-R permanent binding\n\nSIM-R tests similarity address with residual value semantics, without predictive confirmation.\n\nWhen a previously unbound exact context becomes eligible:\n\n- if a tentative candidate with similarity \\(\\ge\\tau\\) exists, permanently bind the context to that slot;\n- otherwise allocate a new slot.\n\nThe context's lag vector is added to the slot and the prototype is recomputed.\n\nPrediction for a retrieved slot:\n\n\\[\ns=s_B+r_j.\n\\]\n\nOnly the retrieved slot's residual updates.\n\nThis arm intentionally exposes the failure mode of false similarity merges.\n\n---\n\n# 16. SIM-V permanent binding\n\nSame address rule as SIM-R.\n\nThe only difference is absolute ownership.\n\nPrediction:\n\n\\[\ns=v_j.\n\\]\n\nUpdate:\n\n\\[\nv_j^+\n=\n0.4v_j+0.6y.\n\\]\n\nThus SIM-V fills the previously missing cell:\n\n\\[\n\\boxed{\\text{similarity-linked address + absolute owned value}.}\n\\]\n\nThis is the primary V36 arm.\n\n---\n\n# 17. PRED-V — predictive-confirmed similarity ownership\n\nPRED-V tests a stricter autonomous identity rule:\n\n\\[\n\\boxed{\n\\text{similar context}\n+\n\\text{compatible predictive consequence}\n}\n\\]\n\nrather than similarity alone.\n\nFor an unbound context:\n\n1. find tentative candidate \\(j^*\\) using the same \\(\\tau=0.85\\);\n2. use the candidate's current absolute value for the tentative prediction;\n3. observe the true next byte \\(y\\);\n4. if the candidate's **pre-update top-1 byte equals \\(y\\)**, permanently bind the new exact context to that slot;\n5. otherwise allocate a new slot when the context satisfies the common eligibility rule.\n\nOnce contexts are bound to a slot, later contradictory outcomes update that one shared slot and therefore propagate revision to all linked members.\n\nNo global revision classifier is used.\n\nThe rule asks only whether a proposed identity link predicted the observed consequence correctly.\n\n## 17.1 Held-out alias probe\n\nA never-trained held-out alias is never permanently bound by evaluator probing.\n\nAt a read-only probe it may tentatively retrieve the most similar slot if \\(s\\ge\\tau\\).\n\nThis permits genuine held-out transfer measurement without modifying learner state.\n\n---\n\n# 18. FROZEN-R — base-drift causal control\n\nFROZEN-R is intentionally nondeployable.\n\nIt is identical to EXACT-R except:\n\n- after the end of wave A, the shared base state is frozen permanently;\n- exact residual memories continue learning causally.\n\nThe A boundary is evaluator metadata and is therefore used only in this diagnostic control.\n\nFROZEN-R cannot qualify as an architecture.\n\nIts sole purpose is to test:\n\n\\[\n\\boxed{\n\\text{Does removing shared-base drift restore additive local residual persistence?}\n}\n\\]\n\nIf FROZEN-R rises close to EXACT-V while EXACT-R remains poor, the V35 base-drift interpretation is causally confirmed.\n\n---\n\n# 19. Main primary endpoints\n\nAlways report absolute levels.\n\n## Persistence\n\n\\[\n\\boxed{\\text{untouched A after D}}\n\\]\n\naveraged over the two trained renderings.\n\nAlso report after A/B/C.\n\n## Current learning\n\n\\[\n\\boxed{D_{\\rm current}}.\n\\]\n\n## Exact revision\n\nFor revised rendering 0:\n\n- prequential accuracy after 1/2/4/8 changed occurrences;\n- probe accuracy;\n- obsolete-target probability.\n\n## Known-alias revision\n\nRendering 1 of the same logical key.\n\nThis becomes a **primary V36 addressing endpoint**.\n\n## Held-out alias transfer\n\nNever-trained rendering 2.\n\nThis is also primary, because V35 oracles established that the task can express this property.\n\n## Prediction\n\nReport:\n\n- whole-stream bpb;\n- target bpb;\n- non-target bpb;\n- base-only counterfactual bpb.\n\n## Sample efficiency\n\nReport:\n\n- first target occurrence;\n- second logical-key occurrence;\n- first exact recurrence;\n- first recurrence after slot creation;\n- number of observations before two aliases share one slot.\n\n---\n\n# 20. Addressing diagnostics\n\nFor SIM/PRED arms report:\n\n- slots allocated by wave;\n- exact contexts per slot;\n- target versus non-target slots;\n- number of logical keys per slot, evaluator-only;\n- number of slots per logical key;\n- same-key merge rate;\n- different-key false-merge rate;\n- false merge rate weighted by presentations;\n- held-out nearest-slot accuracy;\n- nearest similarity;\n- top1–top2 similarity margin;\n- slot prototype drift;\n- revision propagation across linked members.\n\nLogical-key metadata is diagnostic only.\n\n---\n\n# 21. Prespecified main contrasts\n\nUse paired two-sided 95% Student-\\(t\\) intervals across eight seeds.\n\n## V1 — fresh replication of absolute anchoring\n\n\\[\n\\boxed{\\text{EXACT-V}-\\text{EXACT-R}}\n\\]\n\non untouched A after D.\n\nThis replicates V35's large R3 effect.\n\nStrong support requires:\n\n- both arms validly acquire A and D;\n- paired lower bound \\(>0\\);\n- mean gain \\(\\ge40\\) pp.\n\n## V2 — frozen-base mechanism check\n\n\\[\n\\boxed{\\text{FROZEN-R}-\\text{EXACT-R}}.\n\\]\n\nPrimary diagnostic interpretation:\n\n- if FROZEN-R is within 10 pp of EXACT-V on A-after-D, base drift explains most of the EXACT-R failure;\n- if it remains far below EXACT-V, the additive-residual explanation is incomplete.\n\nThis is not an architectural qualification gate.\n\n## V3 — does similarity addressing buy alias transfer under absolute ownership?\n\n\\[\n\\boxed{\\text{SIM-V}-\\text{EXACT-V}}\n\\]\n\non known-alias revised accuracy after four changed outcomes.\n\nRequire:\n\n- mean gain \\(\\ge30\\) pp;\n- paired lower bound \\(>0\\);\n- SIM-V untouched A after D \\(\\ge85\\%\\).\n\nThis prevents calling alias transfer a success if it destroys ownership.\n\n## V4 — does absolute ownership still matter under similarity addressing?\n\n\\[\n\\boxed{\\text{SIM-V}-\\text{SIM-R}}\n\\]\n\non A-after-D.\n\nThis completes the 2×2.\n\n## V5 — does predictive confirmation repair false similarity merges?\n\n\\[\n\\boxed{\\text{PRED-V}-\\text{SIM-V}}\n\\]\n\nPrimary mechanism outcomes:\n\n- lower different-key false-merge rate;\n- no more than 5 pp loss in known-alias merge rate;\n- no more than 5 pp loss in A-after-D;\n- whole-stream bpb noninferior within 0.10 bits/byte.\n\nIf SIM-V already has negligible false merges, PRED-V may be unnecessary; report that rather than forcing a positive claim.\n\n---\n\n# 22. Candidate qualification\n\nA V36 **generalizing ownership candidate** must satisfy all:\n\n### Q1 — acquisition\n\n- A-after-A mean \\(\\ge90\\%\\);\n- minimum seed A-after-A \\(\\ge75\\%\\);\n- D-current mean \\(\\ge90\\%\\).\n\n### Q2 — strict persistence\n\n\\[\n\\boxed{A_{\\rm afterD}\\ge90\\%}.\n\\]\n\n### Q3 — exact revision\n\nAfter four changed outcomes:\n\n\\[\n\\text{exact revised accuracy}\\ge90\\%.\n\\]\n\nAfter eight:\n\n\\[\nP(\\text{obsolete target})\\le0.10.\n\\]\n\n### Q4 — known-alias revision\n\nAfter four changed outcomes:\n\n\\[\n\\boxed{\\text{known-alias revised accuracy}\\ge70\\%}.\n\\]\n\n### Q5 — held-out alias transfer\n\nAfter four changed outcomes:\n\n\\[\n\\boxed{\\text{held-out alias revised accuracy}\\ge50\\%}.\n\\]\n\n### Q6 — whole-stream utility\n\nMean whole-stream bpb may be no worse than the base-only counterfactual by more than:\n\n\\[\n0.10\\text{ bits/byte}.\n\\]\n\n### Q7 — no major false identity collapse\n\nEvaluator-known different-key permanent merges must remain below:\n\n\\[\n\\boxed{5\\%}\n\\]\n\nof target-context bindings.\n\nReport non-target false merges separately.\n\nPassing Q1–Q7 establishes a promising **unbounded generalizing ownership mechanism**, not a bounded lifelong architecture.\n\n---\n\n# 23. Graded verdicts\n\nUse the V35 vocabulary:\n\n- `qualified`\n- `supported_below_qualification`\n- `favorable_unresolved_precision`\n- `no_mean_advantage`\n- `adverse`\n- `invalid_acquisition`\n\nAdd:\n\n- `addressing_invalid_false_merges`\n- `alias_task_invalid` only if the new V36 oracle/task check unexpectedly fails\n- `fly_separation_gate_failed`\n\nDo not compress a 20–50 pp supported effect into a binary failure.\n\n---\n\n# 24. Fly separation preflight\n\nThe additional fly hypothesis is tested first as **representation geometry**, not immediately as a learning claim.\n\nUse the exact same lag96 input.\n\nConstruct a fixed KC-like code:\n\n\\[\n96\\rightarrow4096.\n\\]\n\nFor each KC:\n\n- choose exactly 5 distinct lag96 coordinates;\n- fixed seeded Gaussian contact weights;\n- normalize the five weights once;\n- compute tanh drive;\n- retain the top 5% by absolute activity:\n\n\\[\nK=\\boxed{205}\n\\]\n\nactive units.\n\nNo learning in the code.\n\nThe numerical choices are engineering abstractions inspired by the connectome scale, not a claim of biological identity.\n\n## 24.1 Fly-code representation gate\n\nOn the four fixed preflight seeds report, separately for 0/1 and 0/2 aliases:\n\n\\[\nJ_{\\rm same},\n\\qquad\nJ_{\\rm different},\n\\qquad\nR_J=\\frac{J_{\\rm same}}{J_{\\rm different}}.\n\\]\n\nThe fly-separation hypothesis is considered instantiated only if both alias pairs satisfy:\n\n\\[\n\\boxed{J_{\\rm same}\\ge0.30}\n\\]\n\n\\[\n\\boxed{J_{\\rm different}\\le0.04}\n\\]\n\nand\n\n\\[\n\\boxed{R_J\\ge10}.\n\\]\n\nAlso report full distributions and code utilization.\n\nIf any condition fails:\n\n\\[\n\\boxed{\\texttt{fly_separation_gate_failed}}\n\\]\n\nand **no fly learning trajectories run**.\n\nThis is a legitimate scientific outcome.\n\nIt means simple widening/sparsifying the present lag code does not reproduce the proposed fly overlap regime on this byte workload.\n\n---\n\n# 25. Conditional fly learning block\n\nOnly if the representation gate passes, run two additional arms on the same eight scientific seeds:\n\n## FLY-O\n\n- 4096-dimensional 5%-active code above;\n- one global ordinary score matrix;\n- ordinary normalized 0.6 delta update;\n- no covariance;\n- no RLS;\n- no local slots;\n- no semantic labels.\n\n## HASH-O\n\nMatched control:\n\n- same 4096-dimensional width;\n- exactly 205 active features;\n- same 4096×256 learned score matrix;\n- same 0.6 ordinary update;\n- active set deterministically generated from the exact raw 12-byte context hash;\n- independent exact contexts therefore have near-random sparse overlap;\n- no logical or alias identity enters the hash.\n\nHASH-O asks whether low overlap alone gives retention while destroying alias transfer.\n\nThe comparison isolates whether the **structured overlap** of the fly-like code adds useful generalization beyond sparse capacity.\n\nConditional work:\n\n\\[\n2\\times8=16\\text{ trajectories}.\n\\]\n\nThus maximum V36 workload is:\n\n\\[\n48+16=\\boxed{64\\text{ trajectories}}.\n\\]\n\n---\n\n# 26. Conditional fly hypothesis\n\nThe strong version of the proposed fly explanation predicts:\n\n### F1 — retention\n\nIf \\(R_J\\ge10\\), FLY-O should retain substantially more untouched A than V34's lower-separation sparse ordinary lineage.\n\nFor the fresh run require at least:\n\n\\[\nA_{\\rm afterD}\\ge50\\%\n\\]\n\nbefore calling it practically interesting.\n\n### F2 — structured overlap versus hash\n\nCompared with HASH-O, FLY-O should show substantially greater:\n\n- known-alias revision;\n- held-out alias transfer;\n\nwhile retaining current D.\n\n### F3 — ordinary writer sufficiency\n\nIf FLY-O reaches:\n\n- A-after-D \\(\\ge75\\%\\);\n- D-current \\(\\ge90\\%\\);\n- exact revision \\(\\ge90\\%\\);\n- known-alias revision \\(\\ge70\\%\\);\n\nwith no covariance/local ownership,\n\nthen low-overlap distributed coding becomes a serious alternative mainline.\n\nIf the representation gate passes but FLY-O still catastrophically forgets:\n\n\\[\n\\boxed{\n\\text{the hypothesis that separation alone makes an ordinary writer persistent is falsified on this workload}\n}\n\\]\n\nand the project should retain local ownership as the mainline.\n\n---\n\n# 27. Important interpretation limit on sparsity\n\nA nonzero different-key overlap is not equivalent to permanent ownership.\n\nWith a fixed shared score matrix, even a small collision probability accumulates across an indefinitely long sequence.\n\nTherefore a successful finite four-wave FLY-O result would establish:\n\n\\[\n\\boxed{\\text{strong interference suppression over this horizon}}\n\\]\n\nnot asymptotic lifelong persistence.\n\nLonger streams and capacity saturation remain required.\n\n---\n\n# 28. Why V36 does not bound the local store yet\n\nV35 LOCAL-V stored roughly 1,800 exact contexts.\n\nThat is clearly too large for a lifelong system.\n\nHowever, bounding exact LOCAL-V before solving alias identity would optimize a memory system that treats two renderings of one logical fact as unrelated items.\n\nV36 therefore sequences the problems:\n\n1. establish ownership;\n2. establish autonomous cross-context identity;\n3. then bound / evict / consolidate.\n\nIf SIM-V or PRED-V succeeds, V37 should become the bounded-capacity round.\n\nIf both fail, the next problem is addressing/representation, not eviction.\n\n---\n\n# 29. Noise versus revision remains separate\n\nAfter one contradictory outcome, no causal learner can know with certainty whether:\n\n- the world has permanently changed;\n- or one noisy observation occurred.\n\nV36 therefore does not add a global noise/revision classifier.\n\nReport:\n\n- direct damage to the contradicted slot;\n- unrelated-slot damage;\n- whether linked aliases inherit the corruption;\n- recovery after later correct observations if available.\n\nA linked-memory mechanism may spread a one-off corruption to aliases. That cost must be visible.\n\nIf this becomes the dominant failure after V36, a future round should test **slot-local provisional contradiction state**, not a global familiarity gate.\n\n---\n\n# 30. Mandatory invariants\n\nFor main arms:\n\n- shared base hashes identical across all deployable main arms for a seed;\n- exact lag encoder fixed;\n- probes read-only;\n- unrelated exact local states unchanged unless they intentionally share the same similarity slot;\n- slot membership changes only by the prespecified causal binding rule.\n\nFor PRED-V:\n\n- link confirmation uses only the candidate's pre-update prediction and the subsequently observed byte;\n- evaluator identity metadata must never enter the rule.\n\nFor FROZEN-R:\n\n- explicitly label wave-boundary freezing as noncausal diagnostic intervention.\n\nFor conditional fly arms:\n\n- identical stream;\n- matched width, active count, output coefficient count and writer between FLY-O and HASH-O.\n\n---\n\n# 31. State and sample accounting\n\nReport for every arm:\n\n- shared-base bytes;\n- local value bytes;\n- slot/address bytes;\n- number of exact member contexts;\n- number of slots;\n- mean members per slot;\n- maximum members per slot;\n- auxiliary state growth by wave;\n- fixed representation bytes;\n- output coefficient bytes;\n- operation-count proxy.\n\nDo not call SIM/PRED memory bounded.\n\nDo not count aliases or repeated presentations as new unique facts.\n\n---\n\n# 32. Main interpretation tree\n\n## A. FROZEN-R ≈ EXACT-V\n\nBase drift is causally confirmed as the reason additive exact residual ownership failed.\n\nThe full-vector anchored residual is then merely a reparameterization of absolute ownership and should not become another mainline mechanism.\n\n## B. SIM-V preserves but does not transfer\n\nSimilarity clustering failed to create useful shared identity.\n\nInspect threshold/tails/slot purity; do not immediately change the writer.\n\n## C. SIM-V transfers but loses retention through false merges\n\nThe missing primitive is selective identity binding.\n\nPRED-V becomes the main result.\n\n## D. PRED-V retains and transfers\n\nThe architecture becomes:\n\n\\[\n\\boxed{\n\\text{shared statistical base}\n+\n\\text{absolute owned slot}\n+\n\\text{similarity/predictive-equivalence addressing}\n}\n\\]\n\nV37 should then bound capacity.\n\n## E. PRED-V preserves known aliases but not held-out aliases\n\nStorage and learned alias binding are functioning, but one-shot unseen generalization remains insufficient.\n\nThe next problem is representation-level invariance.\n\n## F. Fly separation gate fails\n\nDo not run a large sparse-code learning experiment.\n\nInterpretation:\n\n> simple random expansion of the current byte representation does not reproduce the fly-like overlap geometry; the fly's sensory preprocessing cannot be skipped.\n\n## G. Fly gate passes and FLY-O succeeds\n\nDistributed low-overlap coding becomes a viable alternative to explicit slot ownership.\n\nA longer-horizon comparison against PRED-V becomes necessary.\n\n## H. Fly gate passes and FLY-O fails\n\nSeparation alone is not sufficient; retain item-local ownership as the mainline.\n\n---\n\n# 33. Out of scope\n\nV36 does not test:\n\n- bounded slot counts;\n- eviction policies;\n- utility-based retention;\n- memory compression;\n- forgetting-factor RLS;\n- MESU;\n- covariance variants;\n- KC compartment counts;\n- APL homeostasis;\n- learned representation plasticity;\n- replay;\n- semantic IDs;\n- Transformers;\n- natural text scaling.\n\nThose remain downstream.\n\n---\n\n# 34. Execution contract\n\nOne self-contained Python file and one command.\n\nSuggested name:\n\n```text\nelm_v36_generalizing_ownership.py\n```\n\nSuggested invocation:\n\n```bash\npython3 elm_v36_generalizing_ownership.py --install-deps\n```\n\nRequirements:\n\n- NumPy/SciPy only unless strictly necessary;\n- one BLAS thread;\n- automatic macOS `caffeinate`;\n- process lock;\n- deterministic resume;\n- atomic per-job writes;\n- fixed source/protocol/plan hashes;\n- engineering and scientific seed separation;\n- compact outputs;\n- no per-byte model dumps.\n\nThe conditional fly block runs automatically only if the locked representation gate passes.\n\n---\n\n# 35. Required artifact package\n\n- executing `.py`;\n- locked V36 protocol;\n- `RUN_MANIFEST.json`;\n- `PLAN.json`;\n- `REPORT.json`;\n- `SUMMARY.md`;\n- `SUMMARY.csv`;\n- per-seed compact records;\n- retention-by-wave;\n- acquisition curves;\n- exact/known/held-out revision curves;\n- obsolete-target curves;\n- base-drift control results;\n- lag similarity distributions;\n- slot-link diagnostics;\n- false-merge diagnostics;\n- state/resource accounting;\n- noise diagnostics;\n- fly representation gate;\n- FLY/HASH results if gate passes;\n- observer-effect audit;\n- artifact checksums.\n\n---\n\n# 36. Central V36 falsifiable hypotheses\n\n### H1 — anchoring\n\n\\[\n\\boxed{\n\\text{the exact LOCAL-R deficit is caused predominantly by shared-base drift}\n}\n\\]\n\ntested by FROZEN-R.\n\n### H2 — missing 2×2 cell\n\n\\[\n\\boxed{\n\\text{similarity-linked absolute ownership combines LOCAL-V persistence with cross-rendering transfer}\n}\n\\]\n\ntested by SIM-V.\n\n### H3 — autonomous identity\n\n\\[\n\\boxed{\n\\text{predictive confirmation reduces false similarity merges while preserving useful alias binding}\n}\n\\]\n\ntested by PRED-V.\n\n### H4 — fly separation\n\n\\[\n\\boxed{\n\\text{if a fixed sparse expanded code actually reaches a ≥10× same/different overlap separation,\nan ordinary shared delta writer may become substantially more persistent}\n}\n\\]\n\ntested only if the representation gate is met.\n\nV36 is designed to decide these questions without yet mixing in bounded capacity or another family of write rules.\n"
REFERENCE_SHA256='493180017e55442f28bb7837171469090d8f99faf4735b988f9510cd306bf470'
SEED_AUDIT=[{'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/BACKEND_PREFLIGHT.json', 'sha256': '34a377771e7ddbf246c0454eac15d4543805f9794752f874a79c4df602e4cd5b', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/INVOCATION.json', 'sha256': '29ace5c5cece95d16e258a93024d3344f3b12760f3d9412f36e01be9cb2304d4', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/PROMOTION_DECISION.json', 'sha256': 'f954c18e31ebd8edf5c726a3fbc550d5c18c816a79c7e4bcb5b0f9f98d146a27', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/RUN_MANIFEST.json', 'sha256': 'e3853842ba8f6f9db4ef33b7e7e03ad4dab2ec87652444075c28d00e88163e69', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/SAMPLE_PLAN_o404.json', 'sha256': '75ea01ad20faf7dd5a9db1522ec1494ab331c7e3c723bc180e11159a02bda29e', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/SAMPLE_PLAN_o505.json', 'sha256': 'b090048284ff1f7b07cd2b27a8b0dec7c85dfc2fb225ef5c6f02d06acbc4aa1f', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s17_o404_09c08c6961d1/ANCHOR_AUDIT.json', 'sha256': '2b1666d95ccc4c395984f7e7609745e662f689e4045bccd30f78f3fbda927a31', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s17_o404_09c08c6961d1/BOUNDARIES.json', 'sha256': '23ad5c82fad0cc74a55d9bde5a581611624f16776164c5646b6165440377312f', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s17_o404_09c08c6961d1/COSTS.json', 'sha256': '66def15caaf3ce7cfa718a7840bf74165f5500827dde00f507cdb59f38aaf61d', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s17_o404_09c08c6961d1/summary.json', 'sha256': '38615f6f1d148e948528c2a4619a8464e2bcc765f322283bbca11ec87a48751c', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s17_o505_af9e66cc95b7/ANCHOR_AUDIT.json', 'sha256': '9c85c37dd05500987959ecc09ee7a33bcd42e7bbc59f870110be9d5a9b72a896', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s17_o505_af9e66cc95b7/BOUNDARIES.json', 'sha256': '676e3b1119e3883074834eae98e285439caeb74ffb69d5e880370267cc7af32f', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s17_o505_af9e66cc95b7/COSTS.json', 'sha256': '66def15caaf3ce7cfa718a7840bf74165f5500827dde00f507cdb59f38aaf61d', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s17_o505_af9e66cc95b7/summary.json', 'sha256': 'a116e47d932f99fa09b9a7549cc377e5d73d4e3bd22755f56a85654729855160', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s29_o404_59b890b01bd2/ANCHOR_AUDIT.json', 'sha256': '2b1666d95ccc4c395984f7e7609745e662f689e4045bccd30f78f3fbda927a31', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s29_o404_59b890b01bd2/BOUNDARIES.json', 'sha256': '7362fee0ad5c5e1ce70611f02eb81769d847415c72e7b892c67cb71ff947774d', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s29_o404_59b890b01bd2/COSTS.json', 'sha256': '66def15caaf3ce7cfa718a7840bf74165f5500827dde00f507cdb59f38aaf61d', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s29_o404_59b890b01bd2/summary.json', 'sha256': '7c6d354dffd100dfc3c416a9155356b3484246c3d34cd764b11f2078ac622e20', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s29_o505_e5adc5b9108c/ANCHOR_AUDIT.json', 'sha256': '9c85c37dd05500987959ecc09ee7a33bcd42e7bbc59f870110be9d5a9b72a896', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s29_o505_e5adc5b9108c/BOUNDARIES.json', 'sha256': 'f1952829fcce7795943c07a2a3b7cdc6ad214f5e9b8872c309e5814473e5df8e', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s29_o505_e5adc5b9108c/COSTS.json', 'sha256': '66def15caaf3ce7cfa718a7840bf74165f5500827dde00f507cdb59f38aaf61d', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s29_o505_e5adc5b9108c/summary.json', 'sha256': '58dbe8cfdeb651795a3ce5ede1d32cb97630916ce3bea18bccfc0190abb87b60', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s43_o404_fb1549b34d8b/ANCHOR_AUDIT.json', 'sha256': '2b1666d95ccc4c395984f7e7609745e662f689e4045bccd30f78f3fbda927a31', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s43_o404_fb1549b34d8b/BOUNDARIES.json', 'sha256': '4cb41f93c7c3a3335ee2a7daf9ba83df6f170aea6e99d9367df93a4a10e09505', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s43_o404_fb1549b34d8b/COSTS.json', 'sha256': '66def15caaf3ce7cfa718a7840bf74165f5500827dde00f507cdb59f38aaf61d', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s43_o404_fb1549b34d8b/summary.json', 'sha256': '9cafe3cba422f665c75aec917885662d07a6bef2d7b619092579bbb825046135', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s43_o505_3e314427ec8b/ANCHOR_AUDIT.json', 'sha256': '9c85c37dd05500987959ecc09ee7a33bcd42e7bbc59f870110be9d5a9b72a896', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s43_o505_3e314427ec8b/BOUNDARIES.json', 'sha256': 'b48d8e5a7354eec383505cea6ab0df9fdd58bbfe0da352eab42c92c8dd143edd', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s43_o505_3e314427ec8b/COSTS.json', 'sha256': '66def15caaf3ce7cfa718a7840bf74165f5500827dde00f507cdb59f38aaf61d', 'collisions': []}, {'source': 'project_sources/07-elm_anchor_transport_results.zip!elm_anchor_transport_runs/runs/anchor_shared_s43_o505_3e314427ec8b/summary.json', 'sha256': '792029a14fabbcd2946f314e1b93252971cadd954f3d41de9b5b33dde7c34b1c', 'collisions': []}, {'source': 'project_sources/09-elm_fast_feedback_consolidation_v18_results.zip!V18_DECISION.json', 'sha256': '35b970ba4750022c8a5281d610686f49df8c4d4a96e4008cca7b76e8c078d1cb', 'collisions': []}, {'source': 'project_sources/09-elm_fast_feedback_consolidation_v18_results.zip!SUMMARY.json', 'sha256': 'd90342f888a44b6aa6117748037caa59441f7b22dd8aecf24df5e5fc07386fb8', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_data/shd/train_61b0b25af064.json', 'sha256': 'e81f49b91d5a5024aac7b0575921563742d42b50fbc0a3ac238a42af01e244d5', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_data/shd/test_a3c52a9ae2db.json', 'sha256': 'a40905a8424194eaf00ef87dc35fbf0169c2b6b43d8742fbeb02b02b93ed29eb', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/PLAN.json', 'sha256': '40525981cef8658c129fdf111f963dfc5e8b2d4e4bc7454847e94f8938c6d55b', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/ENVIRONMENT.json', 'sha256': 'c62fa1b885e07d742f46a36f724d206f23d4674400d784457a0a97ca92d9898b', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/RUN_MANIFEST.json', 'sha256': '207ad7f299b279ad7cbd56bd6cd4af52116953cb7657ef0d2750d89a6f50b634', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_n1000_s7_d101_2efa4fc55ada05b4/summary.json', 'sha256': '8601db9d5a4b44a5af480660604d7eabc84428b7f02e675b4221dfe9d5a04aa3', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_n1000_s7_d101_2efa4fc55ada05b4/config.json', 'sha256': '24bf061eee0a78822b3eea27d48a8ab9e212f0a593ba409df9f6c815abadf6ec', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_n1000_s7_d101_2efa4fc55ada05b4/diagnostics.json', 'sha256': '94c3ed3f9cc1df7eeb77a81c4f9c7798bf75be66f3311662de75a2f4ae8d1ed4', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_n1000_s7_d101_2efa4fc55ada05b4/probe_predictions.json', 'sha256': '73ecc7f6629ed02380e053fb7563042cecff2b0e2e80c02900da7a11975a5bf9', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_n1000_s7_d101_2efa4fc55ada05b4/endpoint_predictions.json', 'sha256': '8f339416bb56668313058f0df6a0035da0190c484959573f472bd17e90a79a6b', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_n1000_s7_d101_2efa4fc55ada05b4/status.json', 'sha256': '52253e460a825fb3ef2ca7c705312bc7ac2c824474b588b271ce4ce8a0c925c8', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_gated_n200_s7_d101_4f4cd537cd34010a/summary.json', 'sha256': '8cd9f3a610cda565c28e55076366f113180968bb0f29ecb77bb0dc16f47f9069', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_gated_n200_s7_d101_4f4cd537cd34010a/config.json', 'sha256': 'd462ebd6d96039bcf0894ed0dc239fa070bc120c0681190a7bf3a47c04b80b9f', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_gated_n200_s7_d101_4f4cd537cd34010a/diagnostics.json', 'sha256': '9eb864b8071fa10430aada017b803d88882fd373f4c0bf762b78b3377e801452', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_gated_n200_s7_d101_4f4cd537cd34010a/probe_predictions.json', 'sha256': '299d7b944cf8e6338c75599dfd17937b75b6208fceb05552e38092899809848a', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_gated_n200_s7_d101_4f4cd537cd34010a/endpoint_predictions.json', 'sha256': 'b94e2867205fbbc792639aca19a4f05ebef95cc81d61feb41ba50eb99d81dd8c', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_gated_n200_s7_d101_4f4cd537cd34010a/status.json', 'sha256': '4a462876fe3299c9551f3f5be7ac492b5e93c5fc376d4d4650d47e974ffa0b5a', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_n1000_s7_d101_8acf373458820bd6/summary.json', 'sha256': '37d9951bcf627ff8a4b10fdf68dd17b4ba672a2d005c8ed1016e4808fe6d4239', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_n1000_s7_d101_8acf373458820bd6/config.json', 'sha256': '41bc0773a2fb0e834f45fc98612776b74e60b8808b33a72935287191735c98c0', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_n1000_s7_d101_8acf373458820bd6/diagnostics.json', 'sha256': 'e1c9f2a5d141abb6a3bfc0c726832c29e3cedfe10536a6f6191958aab6173ffa', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_n1000_s7_d101_8acf373458820bd6/probe_predictions.json', 'sha256': '7000365fb4d747daa15abaca964c036565c2c0d9097b8cd74a961b576e6cdb60', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_n1000_s7_d101_8acf373458820bd6/endpoint_predictions.json', 'sha256': '553e1046134d26a9538da2fb95a3a7f217ca367096a8d012e0d8628fed5d2477', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_n1000_s7_d101_8acf373458820bd6/status.json', 'sha256': 'a0b86a056b86ce1ebb42a3690311befcadb535f94bdae6c17007c4896ad3ee88', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_gated_n200_s7_d101_f9ff666bb03d0c0d/summary.json', 'sha256': '2079b0d0ee19636ecc0e003f831fc86082e11194efd5ad19829a1fe9c9b24f73', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_gated_n200_s7_d101_f9ff666bb03d0c0d/config.json', 'sha256': '96b0eeb11e525248d76176f51b3bb68773a3c058e2377de0901cc8be3d7abef4', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_gated_n200_s7_d101_f9ff666bb03d0c0d/diagnostics.json', 'sha256': 'be6a741d2adce33cced2e97b5bdbbf981825033fe8f63b5b4506da651440ac2b', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_gated_n200_s7_d101_f9ff666bb03d0c0d/probe_predictions.json', 'sha256': 'd94ae2d7b71cb6bf04b88a47f7c3eada405cea107f12f8574ba610cf5a598c8e', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_gated_n200_s7_d101_f9ff666bb03d0c0d/endpoint_predictions.json', 'sha256': '224aa4d6997b024f3bbafb6e64dd4e391acb73f44a7d35ed13b07845de09eca6', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_gated_n200_s7_d101_f9ff666bb03d0c0d/status.json', 'sha256': '4ecd92881d96ab5aeb01252ebe447baa3e31faf3c60bf3be035af0e5d9d8c81c', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_gated_n1000_s7_d101_856618390b8493ff/summary.json', 'sha256': '5a6f7f7ff27bd2af8d33d0bdbb1cf7d246b476e1192500960f56edde720f00a5', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_gated_n1000_s7_d101_856618390b8493ff/config.json', 'sha256': 'b8198117e17524a6c001a071097bfbda8953f53188b205224e7d32e3bec71531', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_gated_n1000_s7_d101_856618390b8493ff/diagnostics.json', 'sha256': '65b1da1875fe45c3096bef3237edc45607fb3eae66be73718c3d37115bec11e7', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_gated_n1000_s7_d101_856618390b8493ff/probe_predictions.json', 'sha256': 'd29a738ac84ade66611a2e4ec012bb0c32613a1c3c663674eb47f5a93471c46b', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_gated_n1000_s7_d101_856618390b8493ff/endpoint_predictions.json', 'sha256': 'c97e021da8de48cdea134cb71efc462e2530759c57247a0cf1bdbac129a84b1d', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_gated_n1000_s7_d101_856618390b8493ff/status.json', 'sha256': 'c68003af636508ed314b63f9f20bee381d29aecda93e92ae432f6d336385e2f1', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_bank_n1000_s7_d101_950b3d7967f1bcbd/summary.json', 'sha256': 'd180e11c13e691647ecb9b2c19ad01f75720c1c43037fcf7ec28ce458ca098ec', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_bank_n1000_s7_d101_950b3d7967f1bcbd/config.json', 'sha256': 'b458e46e584229d6fe1c9b2b6731b1b9cde5e8d2a9b2257c95bf7b1592d762d1', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_bank_n1000_s7_d101_950b3d7967f1bcbd/diagnostics.json', 'sha256': '27e1dda5d7a8b77f64080d26e77536ca9eaa3c386385078ee01d28ece9a2ed2b', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_bank_n1000_s7_d101_950b3d7967f1bcbd/probe_predictions.json', 'sha256': '571c97d81f2775748ace9958a6363924781641da3fc0721df9fc85faf99dee78', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_bank_n1000_s7_d101_950b3d7967f1bcbd/endpoint_predictions.json', 'sha256': '350ed6bb64f244e730e4c4e470760e9f627913970187c4bf19a209df13302a2c', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_bank_n1000_s7_d101_950b3d7967f1bcbd/status.json', 'sha256': 'f213d98462aa742bdf27362a19b1f3f1d47eab83746f710f48b21e6c134fe062', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_n200_s7_d101_71e58f427e5174df/summary.json', 'sha256': '0e83cdd65b3a8e9eb5648c9ee3b29ce42dcf6d9d36f8afab1d800a5e85059130', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_n200_s7_d101_71e58f427e5174df/config.json', 'sha256': '7eff7f8318515e01b0fc849fabf8e4e3504b7c45a0104df42264b5079e374570', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_n200_s7_d101_71e58f427e5174df/diagnostics.json', 'sha256': '268f1f3579b5dd7bd652d14c56ba27f31c09dfdff31d525c1b2a1c11159e896a', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_n200_s7_d101_71e58f427e5174df/probe_predictions.json', 'sha256': '69a77a076eb5b60a3ccaffeb43d514f79f212c6a47fc15704cee7335b1135829', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_n200_s7_d101_71e58f427e5174df/endpoint_predictions.json', 'sha256': '3c2685d79e0cff130dc58f01ba7988b1016c52c3c2dc027c6343d21915a4562a', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_n200_s7_d101_71e58f427e5174df/status.json', 'sha256': '9a9f1c39d1e1189cb3be05a4c25bd29bb4871e1e9b23814d89930bbe0f232929', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_rate_n200_s7_d101_f94eb597ac0f8086/summary.json', 'sha256': '5a3831e9d201fe1d406861a1727f91daad1be0fa060d72768788690b5839d544', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_rate_n200_s7_d101_f94eb597ac0f8086/config.json', 'sha256': '353a6dc4b778201663202437cd4fd0c3abd87ee6685dc36a2e87699796fcb061', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_rate_n200_s7_d101_f94eb597ac0f8086/diagnostics.json', 'sha256': 'b5cb320104f2d2a95d774733770c604f6d098ccd62564e77e74c520b7407da0f', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_rate_n200_s7_d101_f94eb597ac0f8086/probe_predictions.json', 'sha256': '85cf5843fb737c12167fa1565f87849f2fa9ca28fcc5f26e8f8e42ef070f0f9f', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_rate_n200_s7_d101_f94eb597ac0f8086/endpoint_predictions.json', 'sha256': '30e83a9ffcf37b699f29f62597f4fc91dd094e83c1f128afc3825e8694987a72', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_rate_n200_s7_d101_f94eb597ac0f8086/status.json', 'sha256': '55c557f58aefdb68ad22b59be727331eca4e49d70b0cec6bd9af3204868ba1f2', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_n200_s7_d101_84c36c2894abc7c1/summary.json', 'sha256': '1298d627493fc99b667ee4fac3b071074558c5a4c66ba0456205b7b4ff03a870', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_n200_s7_d101_84c36c2894abc7c1/config.json', 'sha256': '850b13b8400143a88c33e61bb7615970362dc1bf2955ca84434a3994a372baec', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_n200_s7_d101_84c36c2894abc7c1/diagnostics.json', 'sha256': '37275643049fb3ee917763c289634a7c2961c2b5568df46652805430d36b6e2c', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_n200_s7_d101_84c36c2894abc7c1/probe_predictions.json', 'sha256': '807099112d896d89c2bc0d9730a41c237c18b803bf173b0bcae5f3b55425ad2e', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_n200_s7_d101_84c36c2894abc7c1/endpoint_predictions.json', 'sha256': '160d55112bbb876f2f909e1bc5e7d5f76b3c2a2bfe296ecf79cd6b82113e4837', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_n200_s7_d101_84c36c2894abc7c1/status.json', 'sha256': '6f56bb25f8553f52cfc757ba07e1313408c43801d6b36b78098e4106c13c6998', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_gated_n1000_s7_d101_bb6e94a557a58318/summary.json', 'sha256': 'f771aaf2f2a1dc431a0e166644344f60206ab79c26ac14bfcc0de1e215241e59', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_gated_n1000_s7_d101_bb6e94a557a58318/config.json', 'sha256': '4d2589f445f827819379dbb87aa345d6009e76aafbcb6de334f478f342efd5f8', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_gated_n1000_s7_d101_bb6e94a557a58318/diagnostics.json', 'sha256': 'a1fdf53fc4b19c42bb7cf99fb9805354da86f197e8b150bc9d1a2dae572d2c2a', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_gated_n1000_s7_d101_bb6e94a557a58318/probe_predictions.json', 'sha256': '40935ae2f6c83864791c1e0c7bc97bbadcfbe098ff766f44cd70e7109f91a7bf', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_gated_n1000_s7_d101_bb6e94a557a58318/endpoint_predictions.json', 'sha256': '47f99e5a368ee8986b7f8a71414e1edacb06dcd6eaf810b01fd56050d00f29aa', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_gated_n1000_s7_d101_bb6e94a557a58318/status.json', 'sha256': '0836224cf91b0415d95b883dd75f828fd9e76a83c956488d4b694cd76131f944', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_gated_n1000_s7_d101_b615f00bde1448fd/summary.json', 'sha256': '55ddb698ddfac535e81afcb231ef848a1ab62dea3661257710f87c65c8d372a8', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_gated_n1000_s7_d101_b615f00bde1448fd/config.json', 'sha256': '2814d6de3384d588b6fb19eb6d6cdcbdb166736d75952ca3cf2c3cd31c03144d', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_gated_n1000_s7_d101_b615f00bde1448fd/diagnostics.json', 'sha256': '14e4ba1f74e2f1590c348b2db21f9285c440f07bff9587cba83d5ededfac8c6b', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_gated_n1000_s7_d101_b615f00bde1448fd/probe_predictions.json', 'sha256': '930ddfe5d78cb25fbe4586d863fb7c66f0c6c411e0e22e55fd0f975ccbe6007a', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_gated_n1000_s7_d101_b615f00bde1448fd/endpoint_predictions.json', 'sha256': '9ad92ceb29a7c600ecd5614ccb9c06b90d8651623fb866c0a6928285576c36eb', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_gated_n1000_s7_d101_b615f00bde1448fd/status.json', 'sha256': 'bab89df4114c4c521c89d77c3f13d2fb9beb941c86763f3b95abeb30d4cebed2', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_bank_n200_s7_d101_3c8add9de3fbd38f/summary.json', 'sha256': '1f2d86bce81631b1e1bb66f8231bfb31dd854310cacc700748d7bb720fcdcc07', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_bank_n200_s7_d101_3c8add9de3fbd38f/config.json', 'sha256': '8e30b2ce9cd9f39638045608ee1ff871364b37ca9747a2894bcc1e868ce57973', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_bank_n200_s7_d101_3c8add9de3fbd38f/diagnostics.json', 'sha256': '5d92fec3b86b7f62da3333d6bd94abc26deaae80944a3df3d828e500045e623a', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_bank_n200_s7_d101_3c8add9de3fbd38f/probe_predictions.json', 'sha256': 'fb06de5f7c97ce58a8c9136db834424c1955edd658d3a40a442001a2c773a2fb', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_bank_n200_s7_d101_3c8add9de3fbd38f/endpoint_predictions.json', 'sha256': 'ea3b45d6a2af556eb39c49cb3689a3d6235523c3e71efcece4b2cd8da211e42b', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_bank_n200_s7_d101_3c8add9de3fbd38f/status.json', 'sha256': 'de451d4f373bd99bc41dfae44ed8a66101b9c75806574c9a8bb32bf1f0e1cac3', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_n1000_s7_d101_3e74a92164d329bd/summary.json', 'sha256': '81b618e926a63029456d6bcb14cdf81cc741d0fb754b080f63bf6cb1499d3cdd', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_n1000_s7_d101_3e74a92164d329bd/config.json', 'sha256': '8181ac2f12411dc654c179ad7be17a0f285de9b01ec6d1ab06affdc66cca1fb0', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_n1000_s7_d101_3e74a92164d329bd/diagnostics.json', 'sha256': 'e35afbf6b3a0c520977aa93b3a8b4dd0675d1b8d1de7c5f00edf89011d2e84e3', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_n1000_s7_d101_3e74a92164d329bd/probe_predictions.json', 'sha256': 'ec1360f3803f4f68ee5a555615a64d5e88f932899ed846a32a38e7b6415e5c66', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_n1000_s7_d101_3e74a92164d329bd/endpoint_predictions.json', 'sha256': '980e2f4e0b83c4912267fa731539cd5ee3fa945f1ec37f8630dbc861e0b5cd2c', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_n1000_s7_d101_3e74a92164d329bd/status.json', 'sha256': '0b8ee579c516b9bd21d8d2628e1789b544eeee0db2517cdaaf8f9d966d860928', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_n200_s7_d101_742c2a1b3f882ede/summary.json', 'sha256': '893853f134796acd15e1f5c017c69743ab1983e9513f6e4fa9ec711ad80ceb12', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_n200_s7_d101_742c2a1b3f882ede/config.json', 'sha256': '1e3e3dc62abeb71c243a068234feb6b2e66a06d64d5c7794f491b116fb15345b', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_n200_s7_d101_742c2a1b3f882ede/diagnostics.json', 'sha256': '7934c4451667fc82956cbcabda833f345b406b984fda5e0163f504f7ac3b27e5', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_n200_s7_d101_742c2a1b3f882ede/probe_predictions.json', 'sha256': 'b789f70bb3feec36a756546ab49c253b38acca08f8a45900f4230696cd849351', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_n200_s7_d101_742c2a1b3f882ede/endpoint_predictions.json', 'sha256': '051c2011627af5877097bc3098eaefb1e60896be4687b12b8cc5ca7e4cc50c34', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_n200_s7_d101_742c2a1b3f882ede/status.json', 'sha256': '4a413612d97a9af63bb34559ca0978ee93b40f373ccce5694f89e2aa7926a008', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_gated_n1000_s7_d101_ff89777a6b32f5b7/summary.json', 'sha256': '0bda24769e467d95aaaf25c486c3b06644595f9e86a8825eb81b28bed0d857f5', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_gated_n1000_s7_d101_ff89777a6b32f5b7/config.json', 'sha256': '227cdc8b9d1e6a5cfa4bc4a96ebba5044e69204dce63e4c934473cc0bdaf7344', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_gated_n1000_s7_d101_ff89777a6b32f5b7/diagnostics.json', 'sha256': '8c0b4e42b3129074dc8ddb0c7d7dfa1917faef3cff31e5b9fec63b9947cfc54d', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_gated_n1000_s7_d101_ff89777a6b32f5b7/probe_predictions.json', 'sha256': '1355f1fa2c9f65a6478489bc3db105c93c44c2f508327b6d5f7fb5a7d81cf926', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_gated_n1000_s7_d101_ff89777a6b32f5b7/endpoint_predictions.json', 'sha256': '2c768adcbbd8cd60948b88c61fc8a4145e7069ade03819d412ebdecd4ac25908', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_gated_n1000_s7_d101_ff89777a6b32f5b7/status.json', 'sha256': 'bfadca69eb5e2224e2a30a746b6965afaf061f949664d38ee523c86881525395', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_gated_n200_s7_d101_dc1e15d5611545e2/summary.json', 'sha256': 'e6da2d69e5f8c59e352c1ba957c0beda06af1f79fb55f945b43fdf22a8c4ab31', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_gated_n200_s7_d101_dc1e15d5611545e2/config.json', 'sha256': '11a8570ecda2d8efb44f7868d503b92de76035f7940f08947178ffb43b80bfba', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_gated_n200_s7_d101_dc1e15d5611545e2/diagnostics.json', 'sha256': 'd77d51f3183dad1987d8b5382accea1a43f764284bfa1161a80782b0aca571f7', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_gated_n200_s7_d101_dc1e15d5611545e2/probe_predictions.json', 'sha256': '3deb7ec1ef2eace816761766147fc37651153c69318dfb05bfb244d417f647b6', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_gated_n200_s7_d101_dc1e15d5611545e2/endpoint_predictions.json', 'sha256': '53be226ffc6699923ce41e4d38114fa513b922e451686861ca00874f55747c49', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_shared_gated_n200_s7_d101_dc1e15d5611545e2/status.json', 'sha256': '4605f3f5293b7adb714d35f27a8d8322262036e0c16e498587e4d3627ee41978', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_n1000_s7_d101_8b412b6e9b99aafd/summary.json', 'sha256': '5932eeacf730497f4bd0eac1b8ab66c76146c30071dd3d096bc622b378557bbc', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_n1000_s7_d101_8b412b6e9b99aafd/config.json', 'sha256': 'b99d9ba4874b3e41577c42974d5c633937d6f6ac9bee9be77056d1b9cd5d50db', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_n1000_s7_d101_8b412b6e9b99aafd/diagnostics.json', 'sha256': 'aea258135e6b0575eb616f726c0e394b25223f3335802ccdefe9ec8e3adbc720', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_n1000_s7_d101_8b412b6e9b99aafd/probe_predictions.json', 'sha256': 'd0583786132a14396f4ac60735c52a0c0c3330edf98f9b573ab64fc37110d635', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_n1000_s7_d101_8b412b6e9b99aafd/endpoint_predictions.json', 'sha256': '3939f18272c16cc8a28689aa44c3d9c158af108ca0be08259e84143f4e3fde7d', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_n1000_s7_d101_8b412b6e9b99aafd/status.json', 'sha256': '412dcc345e14ed357b7c09ac9083eb5b88889c4010243f3b8dea26c14f26e8ff', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_gated_n200_s7_d101_5770d5d1564ca98f/summary.json', 'sha256': 'ee535018947faefbdcd117e6165821a9e55b2e7280a261234937a22d467a445c', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_gated_n200_s7_d101_5770d5d1564ca98f/config.json', 'sha256': '233d98f5bb0e619a8c7c448bf817959684bfde70c08286edc4d6b7ffae0c2212', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_gated_n200_s7_d101_5770d5d1564ca98f/diagnostics.json', 'sha256': '7027f9d720ece6dcec640ae775ea4a6eefa5e5c0a506af8576781e35281a3f34', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_gated_n200_s7_d101_5770d5d1564ca98f/probe_predictions.json', 'sha256': '7cbdf04978bb1898149c62760540a5602133e7074eefa339e087d320a86d9b3c', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_gated_n200_s7_d101_5770d5d1564ca98f/endpoint_predictions.json', 'sha256': 'e7afe9149a25a36d815689a00503fadc61bba074cb8a622b642a4e1ca49d6306', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_gated_n200_s7_d101_5770d5d1564ca98f/status.json', 'sha256': '9ca3c124ff3f6e2fd7cd3902b18d987aa512dd58136a8f4e1d0068a712f06079', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_rate_n1000_s7_d101_16f5ceff00a0dafa/summary.json', 'sha256': 'ca0895eb0e2d2a635a00437d5beb5d84910915d64e20f8d7a17910b57a83b7a5', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_rate_n1000_s7_d101_16f5ceff00a0dafa/config.json', 'sha256': '6381b20c4aef00815f57f73d6fe9ef8a59bc8fa9bd8c329c75fa525c29e390f7', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_rate_n1000_s7_d101_16f5ceff00a0dafa/diagnostics.json', 'sha256': '613e4fc16601ac9977ca1aa61e5349c035189865f008148f1f9751e1f82f9923', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_rate_n1000_s7_d101_16f5ceff00a0dafa/probe_predictions.json', 'sha256': '3009555ee66c933da6efb3d5473cc85f0c56d3990b1d43abf65e91b82dad3c61', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_rate_n1000_s7_d101_16f5ceff00a0dafa/endpoint_predictions.json', 'sha256': '1c8758d60fb75e943dd4a3949e6a9d247385f679fd88a5d3007cdbbab956003e', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_rate_n1000_s7_d101_16f5ceff00a0dafa/status.json', 'sha256': '6e8f35a3828f9d79a52522d445a7fd88df4ea62639ccd3414b6fb08a1c111a14', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_n200_s7_d101_312a32d4eb0d76c2/summary.json', 'sha256': '2f4811f7d67eefc9d576c9ae1a6c8157e49f2bcd0319032eb46ba425144e2798', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_n200_s7_d101_312a32d4eb0d76c2/config.json', 'sha256': '21aad58782fc9e0c1e301350f06ac003402c9dc8f2c1fcf849012da3097dd808', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_n200_s7_d101_312a32d4eb0d76c2/diagnostics.json', 'sha256': '3949ff6bc496191483da2817416ebf4cca8385b11db4b7fbb85302e2ac45755e', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_n200_s7_d101_312a32d4eb0d76c2/probe_predictions.json', 'sha256': '09ad83dbc0ea97b40ae4daf1b84cf9b5cb3e15e63bd3b63f23fa4f48bceea9a8', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_n200_s7_d101_312a32d4eb0d76c2/endpoint_predictions.json', 'sha256': 'd23ea96475c48d363c920baffe6714e9341fb75b42c43fb1f11ac5a18c5defd2', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_elm_shared_n200_s7_d101_312a32d4eb0d76c2/status.json', 'sha256': 'e54b41b8c79b65e544fc29d2d5758f3692b6bce464f7b44da4d18f643977576b', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_n200_s7_d101_7f2170b8df4113b4/summary.json', 'sha256': 'd8d6087f8487ff5ca015be5787de9e00b1fffc69cc181300269316b555e23422', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_n200_s7_d101_7f2170b8df4113b4/config.json', 'sha256': '828d748e36ed563ea8181f0824c978236c3207748c487c9bf1a808bc7b5e1212', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_n200_s7_d101_7f2170b8df4113b4/diagnostics.json', 'sha256': 'a317a9b24796ebd72ea0e9678f298429a4fd2d0d6a971255751a0eaebc82635e', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_n200_s7_d101_7f2170b8df4113b4/probe_predictions.json', 'sha256': '1b8310d0c427ae1076e9392ee28612d093a711a06c15da7b12b6c5dbd9f766ca', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_n200_s7_d101_7f2170b8df4113b4/endpoint_predictions.json', 'sha256': '9fcdf61db2124f6e850ab888db9144bfb46ff5f103bb4009c628d7e3bb731531', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/shd_bank_n200_s7_d101_7f2170b8df4113b4/status.json', 'sha256': '84696f04a8d8ce180d1de34aa42918b52631cb89b8baf863f6e1f43f12c7cf20', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_n1000_s7_d101_e5e1e470a5fcef06/summary.json', 'sha256': '78c8cc43172c27228123bc8fa17587f99294b62f60127237a4e0d717fc085822', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_n1000_s7_d101_e5e1e470a5fcef06/config.json', 'sha256': 'bc860e6ff9e40359e3215e269a2e545ff428e398750f750fb1587f42130989a0', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_n1000_s7_d101_e5e1e470a5fcef06/diagnostics.json', 'sha256': '53eb0d74b40bc4ccdfc96d76f2c3b01e7560eb5cf55fe20a530502b7c45cf902', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_n1000_s7_d101_e5e1e470a5fcef06/probe_predictions.json', 'sha256': '398402d80f7861fee9f12e847515a062cbcffe38addd7455fa0e6ed981dad3f0', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_n1000_s7_d101_e5e1e470a5fcef06/endpoint_predictions.json', 'sha256': '13e2dbfb7dc03d9e6d2a3ba010fc7618d430370cc8d7394ffab344cea249948a', 'collisions': []}, {'source': 'project_sources/04-Archive.zip!elm_next_runs/runs/memory_elm_n1000_s7_d101_e5e1e470a5fcef06/status.json', 'sha256': '43c602c43b3db9a6d21b8e586130c2ddfafa730d46dba02456d0aadab583dcf8', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/BACKEND_PREFLIGHT.json', 'sha256': 'b25a65d192320d5269be0f5d528973dbb58bc191889b4816168ae60fae6d5cb9', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/INVOCATION.json', 'sha256': 'a86c65fa87c77cf7a99ecff4b6110a5ec08eccf34bfa5f546577cfcb0953f630', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/PROMOTION_DECISION.json', 'sha256': '6abdbc8494b43f9d801192fd3bd6763d1b9a63b55705ddd85ff22d294cf3223d', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/RUN_MANIFEST.json', 'sha256': '6032ba91e451789bed2ebf2dded455ab879df7c989ff54216bc61d455cdc3696', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/SAMPLE_PLAN_o404.json', 'sha256': '75ea01ad20faf7dd5a9db1522ec1494ab331c7e3c723bc180e11159a02bda29e', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/SAMPLE_PLAN_o505.json', 'sha256': 'b090048284ff1f7b07cd2b27a8b0dec7c85dfc2fb225ef5c6f02d06acbc4aa1f', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s17_o404_e9e62aa09a76/BOUNDARIES.json', 'sha256': 'c665c2a5e64125b7af86069bb30d5bc08361385a38b6f0c8fd6f0a3715c757da', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s17_o404_e9e62aa09a76/COSTS.json', 'sha256': '0ebfd7da057f4b840cc687641728e87f6fc3cca1e1d3fd8dc741ce291567aafc', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s17_o404_e9e62aa09a76/RESERVOIR_FINAL.json', 'sha256': '19decacfe26f9fbd29d587a1bb34405fd007f13103caadde1303f65b51a41b77', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s17_o404_e9e62aa09a76/summary.json', 'sha256': '078c52e08057e49b74cad3781cd0e29a416d819387e6f166bd8148012e892609', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s17_o505_7f4df726585e/BOUNDARIES.json', 'sha256': 'd986f5e49b6fbddf5aba679e3b07b4ef9d14c98bcc1e90f4164f0efe00b116fa', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s17_o505_7f4df726585e/COSTS.json', 'sha256': '48b83c473894580e17b0f19801779f432b01646290dc50fd56080dc65af32cb9', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s17_o505_7f4df726585e/RESERVOIR_FINAL.json', 'sha256': '180c00a24640bf74a9317654d4dcd387322049fa26dde326c698b5e9b456f6c5', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s17_o505_7f4df726585e/summary.json', 'sha256': '3c031df7b2716b6359626c9ec5e80b5f46034d491c4f8742ece62f514cc5b54b', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s29_o404_4e6286010044/BOUNDARIES.json', 'sha256': 'a0076ce10c7e35af296e4aba96de63c0b7293cda3187460a0078d5c717711917', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s29_o404_4e6286010044/COSTS.json', 'sha256': '9db44cb76ec2aeafb39d2765641708ef1d1f4b8042d3a4fe99588d261c24d342', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s29_o404_4e6286010044/RESERVOIR_FINAL.json', 'sha256': 'cc5350b691aebb4356b997e024de7022ff5feada12d5d84023aa887cb28e3d3b', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s29_o404_4e6286010044/summary.json', 'sha256': '3361fbb9063e494e9cc5cb4c1c3a23c16954f42068a5782a1397f0f7da474722', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s29_o505_c4e9babbb674/BOUNDARIES.json', 'sha256': '35e5232a4d6e4fca6eb38b9837835b9721323dc0e24f5be6f6ec6f8c8e4249b9', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s29_o505_c4e9babbb674/COSTS.json', 'sha256': '1db64d69ac90510cb0c21d9e8b4dca73630213491e00ff4c7f49a8770eda2984', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s29_o505_c4e9babbb674/RESERVOIR_FINAL.json', 'sha256': '98df0330d61f277d5c9d3742a17ea5c48987379bb5b9e94a7e515b266fc6e286', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s29_o505_c4e9babbb674/summary.json', 'sha256': '8447afbc6f7fbdb62f63cecae29383a55625cb90631e737296840d72061b44be', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s43_o404_64247f66fbf3/BOUNDARIES.json', 'sha256': 'c270ca6c0f583d9bb9e2cc0f19e98ada8f7b01fa521f61741a9fb75976a69d8e', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s43_o404_64247f66fbf3/COSTS.json', 'sha256': 'e7771cf967f14048c6b0371b2e0857d2718628a2e81e649e957a916a6d36683a', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s43_o404_64247f66fbf3/RESERVOIR_FINAL.json', 'sha256': 'e95aa7ad49b30a43ce48e3d38d97d1f543714755fb20badcae83ebd20a6fd653', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s43_o404_64247f66fbf3/summary.json', 'sha256': 'd380a934428accf9e22f64d9db1ecaadec744999c6bd0e95ac8a1a0c82f640f5', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s43_o505_feb930cb716e/BOUNDARIES.json', 'sha256': 'cf6230dfef5883c7552a6cf5ac107027bb56201ccbc1c544850ebcbd00176b9e', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s43_o505_feb930cb716e/COSTS.json', 'sha256': 'e27bd000081450ea4ef4e579a6b689b1895f1790254475874b998a08d41fb2b0', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s43_o505_feb930cb716e/RESERVOIR_FINAL.json', 'sha256': '56145524f4b53c55fe91754cbcc6e48cd7f7205619ca2e82c386612d29e5586f', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_coupled_s43_o505_feb930cb716e/summary.json', 'sha256': 'a7c1c6c61b54c8745054a03d6a71a4822594502cdaa6902e55fd681f6cb30c9c', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s17_o404_d6947b45267e/BOUNDARIES.json', 'sha256': 'bd1606084f1aecc4d64dcd8e7c311c26a33e0602ad9777163bdbc8a56fff9efb', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s17_o404_d6947b45267e/COSTS.json', 'sha256': '60342a31ce6ab736b4a3e24dede42ec01d387952d9de95b0136385b722a895b1', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s17_o404_d6947b45267e/RESERVOIR_FINAL.json', 'sha256': 'a012c908733b0161da5845e080f4f003c6978697fb685a2de5edc99535d71630', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s17_o404_d6947b45267e/summary.json', 'sha256': '843e36c1139e7f4305c2579c8630c06235c16b83421a836cdaa865d1a3d8c56e', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s17_o505_f71d6e4bc0ca/BOUNDARIES.json', 'sha256': '5fba0970c88a2e22d2c57f640c71225301119e5c8d55e7cad3f94ffd6b80d8b7', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s17_o505_f71d6e4bc0ca/COSTS.json', 'sha256': '7da6159cc496801330bac30c15a360439aa5bc5e4079e4f92566504efa248de2', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s17_o505_f71d6e4bc0ca/RESERVOIR_FINAL.json', 'sha256': 'fe306285fb23ecf00b59b0a0ae2244133f1821ca6e16df2556ad32abe8938015', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s17_o505_f71d6e4bc0ca/summary.json', 'sha256': '07a81c324f09f1f2ca37cce7735f28489d559bc14d5a46ae76a2acbdafc9b579', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s29_o404_d72d0b35e3d2/BOUNDARIES.json', 'sha256': 'a382787f782f39387a133db11d4b447fc19ede3f8c3107a7e708993cdc285cd1', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s29_o404_d72d0b35e3d2/COSTS.json', 'sha256': 'deed42ac79cade8dc2d9d147386967829533a85656c179219eb4d36035f19e90', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s29_o404_d72d0b35e3d2/RESERVOIR_FINAL.json', 'sha256': '00a34f87dcf3fdb4d20704283d44eeea201d70e203fbff0236f6ade19e120a8a', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s29_o404_d72d0b35e3d2/summary.json', 'sha256': '421915871a36043c2de487a451191db5b08699cfc48af7d985d2aa84a58f0230', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s29_o505_29b7c48b2a69/BOUNDARIES.json', 'sha256': '0d309d5dd48b04405dda011283ce8c21904a0efef5dfd893993e39f026a2d80b', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s29_o505_29b7c48b2a69/COSTS.json', 'sha256': '3933f859278af59c6a77da3bf44cde63762563eb5e9935d893eddd45db17ce05', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s29_o505_29b7c48b2a69/RESERVOIR_FINAL.json', 'sha256': '935443626fa2918394ff6a4bc2fe8b6e2abb1cba3a53c77a0b5f7e187ed25bf8', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s29_o505_29b7c48b2a69/summary.json', 'sha256': 'ff29287dcae2ebc148773731fcd9b915fc3453a2e0166e0b143ecae3c22214df', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s43_o404_470f324f883f/BOUNDARIES.json', 'sha256': 'f38099c203386a967690759c49ffe9375d43cf82f9a1e155c47e79529c74a6fd', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s43_o404_470f324f883f/COSTS.json', 'sha256': '0b77f59db439b943da9a5b5c358654da4430eca55330936d8d9ff7c797de0cef', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s43_o404_470f324f883f/RESERVOIR_FINAL.json', 'sha256': 'b5eadf932f597911406f54d9ee7d3d0f0bf1777a8c07cc5ee382e29b411cb4ec', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s43_o404_470f324f883f/summary.json', 'sha256': '9c29ec5de5917c9a80a1f9fe88d0c052afc0ea66dcfe1b8ab1fbda6a6ce1e333', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s43_o505_6328cb4d903c/BOUNDARIES.json', 'sha256': '815984cd14b1171457a35396883c2ec95ff94b9f371262fd5d4256a5d6d4ab25', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s43_o505_6328cb4d903c/COSTS.json', 'sha256': 'e7dc3c84589e2a68e0eb8a98a677b3384c8d0cf69a44c3c66060fbd73b8b4000', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s43_o505_6328cb4d903c/RESERVOIR_FINAL.json', 'sha256': 'fd04405a40d13da74eab1c23c91e0839720f1a9519a9eb4be1b8a830ee90bb5f', 'collisions': []}, {'source': 'project_sources/08-elm_bounded_replay_results.zip!elm_bounded_replay_runs/runs/bounded_replay_shadow_s43_o505_6328cb4d903c/summary.json', 'sha256': '9a88e027c7ba801d6943abad188619af6f172aca58aa6edacdbfac4424fb02f8', 'collisions': []}, {'source': 'project_sources/16-fly_architecture_distillation.py.zip!fly_architecture_distillation.py', 'sha256': 'cd7a65133e2a2a2e91a964b3bc03431954ac641687855c2f5eff6e87475ac7be', 'collisions': []}, {'source': 'project_sources/10-elm_additive_feedback_consolidation_v19_results.zip!V19_DECISION.json', 'sha256': '7150a0e31011613cda31659f7bbd92bcf9f0e0b2dddf729a6123b4d6c425a50d', 'collisions': []}, {'source': 'project_sources/05-RUN_MANIFEST.json', 'sha256': 'f970d66afb208b3596a386b6f0456d9160bd501a3c382f38f24c830f9cc3705c', 'collisions': []}, {'source': 'project_sources/15-elm_fly_v20_20260906_152006_26173_results.zip!CONFIG.json', 'sha256': '6eab1eccd7c5d8c8baf9cdc1b0e46e22868535a9450374b2fa41e70d768fca7a', 'collisions': []}, {'source': 'project_sources/15-elm_fly_v20_20260906_152006_26173_results.zip!COSTS.json', 'sha256': '49f518acb9ec83614f7d7d1e309932040caa7b80de03d1b910b7ebf1a7125ba5', 'collisions': []}, {'source': 'project_sources/15-elm_fly_v20_20260906_152006_26173_results.zip!INTERPRETATION.json', 'sha256': '076c2d70f3e51192b573d8ba7cd356c71132cbce7c69d87c3d5b1da4cf5c6929', 'collisions': []}, {'source': 'project_sources/15-elm_fly_v20_20260906_152006_26173_results.zip!VERIFICATION.json', 'sha256': 'e46bee4fd2761fb493845626fe6952420c46ad37572ce51d9191ebcef22d2462', 'collisions': []}, {'source': 'project_sources/11-fly_learning_core_results.zip!fly_learning_core_results/DECISION.json', 'sha256': 'b69d19c51923c3e18c0988fcdcad7812dbc3d073b4388a90f292e54a910d3e48', 'collisions': []}, {'source': 'project_sources/11-fly_learning_core_results.zip!fly_learning_core_results/RUN_MANIFEST.json', 'sha256': '26fec53d09afb4002c9ece2d551ff3071d5adc3ba89e16038efdf8ff8326839f', 'collisions': []}, {'source': 'project_sources/12-fly_core_connectivity_audit_20260906_131800.zip!fly_core_connectivity_audit_20260906_131800/ARTIFACT_HASHES.json', 'sha256': '68ca3b299039a58949cd0b163d9c93198f4d053cbdd8c08c061f296c5d87f1a2', 'collisions': []}, {'source': 'project_sources/12-fly_core_connectivity_audit_20260906_131800.zip!fly_core_connectivity_audit_20260906_131800/DECISION.json', 'sha256': '927adc3b04cb078c76426389ed0a832945ac8351fd2228332ab4332e87746f85', 'collisions': []}, {'source': 'project_sources/12-fly_core_connectivity_audit_20260906_131800.zip!fly_core_connectivity_audit_20260906_131800/REFERENCE_CHECKS.json', 'sha256': '15e665e27f9781b0a890a2128b443b0988396021c90a53fa27d3db7af4349f2f', 'collisions': []}, {'source': 'project_sources/12-fly_core_connectivity_audit_20260906_131800.zip!fly_core_connectivity_audit_20260906_131800/RUN_MANIFEST.json', 'sha256': '898840b06acb74cad8cb6a79a4bea93853fda48314713aff960f47c1d744c749', 'collisions': []}, {'source': 'reference/elm_v35_shared_solver_ownership.py', 'sha256': '493180017e55442f28bb7837171469090d8f99faf4735b988f9510cd306bf470', 'collisions': []}, {'source': 'embedded:v32.py', 'sha256': 'aa3c903796b4d23fdcaccb2a80aa376af71999c5ef7a03fa2b7c95c3f3b6e626', 'collisions': []}, {'source': 'embedded:v33.py', 'sha256': 'fe45c72d7bc34acafe7f36e714859bd292a447ab8c6af573b58246ec424df500', 'collisions': []}]
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

def lower_gate(s, threshold, strict=False):
    if s['low'] is None:
        return 'inconclusive'
    if (s['low'] > threshold) if strict else (s['low'] >= threshold):
        return 'pass'
    if (s['high'] <= threshold) if strict else (s['high'] < threshold):
        return 'adverse'
    return 'inconclusive'

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

class Aggregate:
    def __init__(self):
        self.n = 0
        self.total_loss = 0.
        self.correct = 0
        self.diag = collections.Counter()

    def add(self, l, a, d):
        self.n += 1
        self.total_loss += l
        self.correct += a
        self.diag.update(d)

    def result(self):
        return {'n': self.n, 'bpb': self.total_loss/self.n if self.n else None,
                'accuracy': self.correct/self.n if self.n else None,
                **{k: v/self.n for k,v in self.diag.items()}}

def view_average(probes, key='accuracy', aliases=('0','1')):
    vals = [probes[a][key] for a in aliases]
    return None if any(v is None for v in vals) else mean(vals)

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

def evaluate(model,stream,ids,expected=None,old=None):
    out={}
    for alias in range(3):
        rows=[]
        for i in list(ids):
            h=context(stream['keys'][i],alias);y=int(stream['labels'][i]) if expected is None else int(expected[i])
            p=model.probabilities(h)
            l=-math.log2(max(float(p[y]),1e-300)) if model.arm=='DICT' else loss(model.scores(h),y)
            bank=model.encoder.encode(h)[1]
            trained={model.encoder.encode(context(stream['keys'][i],a))[1] for a in (0,1)}
            rows.append({'accuracy':float(np.argmax(p)==y),'bpb':l,'target_probability':float(p[y]),
                         'obsolete_probability':None if old is None else float(p[int(old[i])]),
                         'route_match':bank in trained})
        def summary(rs):
            return {'n':len(rs),**{k:mean(r[k] for r in rs if r[k] is not None) for k in
                   ('accuracy','bpb','target_probability','obsolete_probability')}}
        out[str(alias)]=summary(rows)
        if alias:
            out[str(alias)]['route_conditioned']={str(match):summary([r for r in rows if r['route_match']==match]) for match in (True,False)}
            out[str(alias)]['route_match_fraction']=mean(float(r['route_match']) for r in rows)
            out[str(alias)]['route_note']='Fixed base-router diagnostic; local exact addressing and shared auxiliary matrix do not use this router.' if model.stage=='II' else 'Match to either acquired rendering bank; does not condition the strict primary endpoint.'
    return out

def eligibility_snapshot(model,stream,seen,seen_targets,wave):
    n=(ord(wave)-64)*stream['keys_per_wave']
    target={context(stream['keys'][i],a) for i in range(n) for a in (0,1)}
    untouched={context(stream['keys'][i],a) for i in stream['untouched'] for a in (0,1)}
    ratio=lambda keys:len(keys & model.eligible)/len(keys) if keys else None
    return {'total_eligible':len(model.eligible),'trained_target_coverage':ratio(target),
        'untouched_A_coverage':ratio(untouched),'seen_context_coverage':ratio(seen),
        'eligible_seen_target_count':len(model.eligible&seen_targets),
        'eligible_other_count':len(model.eligible-seen_targets),
        'base_hash':model.base.state_hash(),'eligibility_hash':model.eligibility_hash()}

def old_state(model,stream):
    out={}
    for i in stream['untouched']:
        for a in (0,1):
            c=context(stream['keys'][i],a)
            out[c]=(model.base_scores(c).copy(),model.scores(c).copy(),model.local.get(c))
    return out

def drift_summary(model,old):
    base=[];final=[];res=[];unchanged=[]
    for c,(b,s,v) in old.items():
        base.append(float(np.mean((model.base_scores(c)-b)**2)))
        final.append(float(np.mean((model.scores(c)-s)**2)))
        if v is not None:
            now=model.local.get(c);unchanged.append(now==v)
            res.append(float(np.mean((np.frombuffer(now,dtype=np.float64)-np.frombuffer(v,dtype=np.float64))**2)))
    return {'base_score_rms_drift':math.sqrt(mean(base)),'final_score_rms_drift':math.sqrt(mean(final)),
        'local_vector_rms_drift':None if not res else math.sqrt(mean(res)),
        'all_acquired_untouched_vectors_bit_identical':all(unchanged),'acquired_local_vectors':len(unchanged)}

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

def metrics(r):
    m={'whole_bpb':r['whole_stream']['bpb'],'target_bpb':r['target_stream']['bpb'],'nontarget_bpb':r['nontarget_stream']['bpb']}
    s=r['snapshots'];g=r['groups']
    m['A_initial_accuracy']=view_average(s['after_A']['untouched_A'])
    initial_loss=view_average(s['after_A']['untouched_A'],'bpb');m['A_initial_bpb']=initial_loss
    for w in 'ABCD':
        pr=s['after_'+w]['untouched_A'];m['A_after_'+w]=view_average(pr);m['A_bpb_after_'+w]=view_average(pr,'bpb')
        m['A_heldout_after_'+w]=pr['2']['accuracy']
        for chance,denom in [(1/256,'256'),(1/24,'24')]:
            m['retention_fraction_'+w+'_'+denom]=(m['A_after_'+w]-chance)/(m['A_initial_accuracy']-chance) if m['A_initial_accuracy']>=.9 else None
        m['loss_retention_fraction_'+w]=(8-m['A_bpb_after_'+w])/(8-initial_loss) if m['A_initial_accuracy']>=.9 and 8-initial_loss>.5 else None
        for ex in (1,2,4,8,16,32):m[f'{w}_acquisition_{ex}']=g.get(f'acquisition_{w}_{ex}',{}).get('accuracy')
        m[w+'_first_exact_repeat']=g['render_acquisition_'+w+'_2']['accuracy']
        if w!='A':m[w+'_current']=view_average(s['after_'+w]['cohorts'][w])
    m['new_second_prequential']=mean(m[w+'_acquisition_2'] for w in 'BCD')
    for ex in (1,2,4,8):
        for a in ('0','1','2'):
            pr=s[f'revision_{ex}']['revised_A'][a]
            m[f'revision_{ex}_alias_{a}']=pr['accuracy'];m[f'obsolete_{ex}_alias_{a}']=pr['obsolete_probability']
        m[f'revision_{ex}_prequential']=g[f'revision_C_{ex}']['accuracy']
    m['noise_damage']=mean(view_average(n['before'])-view_average(n['after']) for n in r['noise_events'])
    m['noise_unrelated_damage']=mean(view_average(n['others_before'])-view_average(n['others_after']) for n in r['noise_events'])
    if r['stage']=='II':
        m['eligibility_A']=r['eligibility']['after_A']['untouched_A_coverage']
        m['first_exact_recurrence_after_creation']=g.get('first_exact_recurrence_after_creation',{}).get('accuracy')
        m['base_whole_bpb']=r['whole_stream']['base_bpb']
    return m

def level(records,arm,key,seeds):
    ix={(r['seed'],r['arm']):r for r in records}
    return stat([ix.get((seed,arm),{}).get('metrics',{}).get(key) for seed in seeds])

def difference(records,a,b,key,seeds):
    va=level(records,a,key,seeds)['values'];vb=level(records,b,key,seeds)['values']
    return stat([None if x is None or y is None else x-y for x,y in zip(va,vb)])

def acquisition(records,arm,seeds):
    a=level(records,arm,'A_initial_accuracy',seeds);d=level(records,arm,'D_current',seeds)
    complete=a['n']==len(seeds) and d['n']==len(seeds)
    return {'passed':bool(complete and a['mean']>=.9 and min(a['values'])>=.75 and d['mean']>=.9),
            'A_after_A':a,'D_current':d,'minimum_seed_A':min(a['values']) if complete else None}

def graded(effect,valid=True,qualified=False,coverage=True):
    if not coverage and not qualified:return 'ownership_unresolved_due_to_eligibility'
    if not valid:return 'invalid_acquisition'
    if qualified:return 'qualified'
    if effect['high'] is not None and effect['high']<0:return 'adverse'
    if effect['low'] is not None and effect['low']>0:return 'supported_below_qualification'
    if effect['mean'] is not None and effect['mean']>0:return 'favorable_unresolved_precision'
    return 'no_mean_advantage'

def csv_output(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=keys);writer.writeheader()
    for r in rows:writer.writerow({k:encode_json(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    atomic(path,out.getvalue())

def run_trial(seed,arm,smoke=False,diagnostics=True):
    started=time.perf_counter();stream=make_stream(seed,False,smoke);model=ByteModel(seed,arm)
    history=bytearray();overall=Aggregate();target=Aggregate();non=Aggregate()
    groups=collections.defaultdict(Aggregate);cps=collections.defaultdict(list)
    for cp in stream['checkpoints']:cps[cp['record_count']].append(cp)
    snapshots={};noise=[];oracles={};eligibility={};old=None;drift={};geometry={}
    oracle=None
    seen=set();seen_targets=set();pending=set();counts=collections.Counter();errors=collections.Counter()
    eligible_at_occurrence=collections.defaultdict(list);activation_errors=[]
    readonly=0;k=stream['keys_per_wave'];link_times={};key_presentations=collections.Counter()
    for number,row in enumerate(stream['rows'],1):
        if diagnostics and row['kind']=='one_off_noise':
            nb=evaluate(model,stream,[row['key_id']]);ob=evaluate(model,stream,stream['untouched'])
        for pos,y in enumerate(row['raw']):
            c=bytes(history);was=c in model.eligible
            if diagnostics:
                seen.add(c);counts[c]+=1
                if pos==12:seen_targets.add(c)
            if oracle is not None:
                z,b=model.encoder.encode(c);oracle.observe(z,b,y,pos==12)
            if diagnostics and pos==12 and row['kind']=='one_off_noise':
                immediate_before=evaluate(model,stream,[row['key_id']],old=stream['labels'])
                before_local=model.local.get(c);before_eligible=was
            l,a,d=model.observe(history,y)
            overall.add(l,a,d);groups['all_'+row['wave']].add(l,a,d)
            if pos==12:
                target.add(l,a,d);groups['target_'+row['wave']].add(l,a,d)
                groups[f"{row['kind']}_{row['wave']}_{row['exposure']}"].add(l,a,d)
                groups[f"render_{row['kind']}_{row['wave']}_{row['rendering_exposure']}"].add(l,a,d)
            else:non.add(l,a,d)
            if diagnostics and model.stage=='II':
                errors[c]+=int(not d['base_accuracy'])
                if c in pending:
                    if pos==12:groups['first_exact_recurrence_after_creation'].add(l,a,d)
                    pending.remove(c)
                if not was and c in model.eligible:
                    pending.add(c);activation_errors.append(errors[c])
                if pos==12 and row['kind']=='acquisition':
                    eligible_at_occurrence[str(row['rendering_exposure'])].append(float(c in model.eligible))
            if diagnostics and pos==12 and row['kind']=='one_off_noise':
                immediate_after=evaluate(model,stream,[row['key_id']],old=stream['labels'])
                changed_local=model.local.get(c)
                local_delta=None if before_local is None or changed_local is None else float(np.linalg.norm(np.frombuffer(changed_local,dtype=np.float64)-np.frombuffer(before_local,dtype=np.float64)))
                wrong_after=float(model.probabilities(c)[y])
            history.append(y)
            if len(history)>12:del history[0]
        if diagnostics:
            key_presentations[row['key_id']]+=1
            cs=[context(stream['keys'][row['key_id']],a) for a in (0,1)]
            js=[model.members.get(c) for c in cs]
            if js[0] is not None and js[0]==js[1] and str(row['key_id']) not in link_times:
                link_times[str(row['key_id'])]=key_presentations[row['key_id']]
        if diagnostics and row['kind']=='one_off_noise':
            noise.append({'key_id':row['key_id'],'before':nb,'after':evaluate(model,stream,[row['key_id']]),
                'others_before':ob,'others_after':evaluate(model,stream,stream['untouched']),
                'target_immediate_before':immediate_before,'target_immediate_after':immediate_after,
                'wrong_target_probability_immediately_after':wrong_after,'eligible_before':before_eligible,
                'local_vector_change_norm':local_delta})
        if arm=='FROZEN-R' and any(cp['name']=='after_A' for cp in cps.get(number,[])):
            model.frozen=True
        if not diagnostics:continue
        for cp in cps.get(number,[]):
            state=model.state_hash();s=dict(cp,bytes_observed=number*14);wi=ord(cp['wave'])-65
            if cp['type']=='acquisition':s['current']=evaluate(model,stream,range(wi*k,(wi+1)*k))
            else:
                revised=cp['type']=='revision' or cp['wave']=='D'
                s['untouched_A']=evaluate(model,stream,stream['untouched'])
                s['noise_A']=evaluate(model,stream,stream['noisy'])
                s['revised_A']=evaluate(model,stream,stream['revised'],stream['revisions'] if revised else None,stream['labels'] if revised else None)
                s['cohorts']={chr(65+w):evaluate(model,stream,range(w*k,(w+1)*k)) for w in range(1,wi+1)}
                if oracle is not None:oracles[cp['name']]=oracle.checkpoint(model,stream,cp['wave'],revised)
                if cp['type']=='wave' and arm!='DICT':geometry[cp['wave']]=model.base.geometry()
            if model.stage=='II':
                eligibility[cp['name']]=eligibility_snapshot(model,stream,seen,seen_targets,cp['wave'])
                if cp['name']=='after_A':old=old_state(model,stream)
                if cp['type']=='wave' and old is not None:drift[cp['wave']]=drift_summary(model,old)
            if state!=model.state_hash():raise AssertionError('Observer changed learned state')
            if cp['type'] in ('wave','revision'):
                s['addressing']=slot_diagnostics(model,stream)
                s['costs']=model.costs()
            readonly+=1;snapshots[cp['name']]=s
    r={'seed':seed,'arm':arm,'stage':model.stage,'smoke':smoke,'diagnostics':diagnostics,
        'stream_sha256':stream['raw_sha256'],'state_sha256':model.state_hash(),
        'base_state_sha256':model.base.state_hash(),'eligibility_sha256':model.eligibility_hash(),
        'whole_stream':overall.result(),'target_stream':target.result(),'nontarget_stream':non.result(),
        'groups':{name:g.result() for name,g in groups.items()},'snapshots':snapshots,'oracles':oracles,
        'noise_events':noise,'covariance_geometry':geometry,'eligibility':eligibility,'base_drift':drift,
        'target_eligibility_by_exact_occurrence':{e:mean(v) for e,v in eligible_at_occurrence.items()},
        'base_errors_at_activation':{'n':len(activation_errors),'mean':mean(activation_errors),'max':max(activation_errors,default=0)},
        'costs':model.costs(),'counts':exposure_accounting(stream),'readonly_checks':readonly,
        'elapsed_seconds':time.perf_counter()-started}
    r['metrics']=metrics(r) if diagnostics else {}
    if diagnostics:
        r['addressing']=slot_diagnostics(model,stream)
        r['metrics']['false_merge_rate']=r['addressing']['different_key_binding_rate']
        r['metrics']['same_key_merge_rate']=r['addressing']['same_key_merge_rate']
        r['metrics']['base_bpb_delta']=r['metrics']['whole_bpb']-r['metrics']['base_whole_bpb']
        r['alias_link_curves']=link_times
        r['sample_efficiency']=dict(first_target=groups['acquisition_A_1'].result(), second_logical_key=groups['acquisition_A_2'].result(), first_exact_recurrence=groups['render_acquisition_A_2'].result(), first_recurrence_after_creation=groups['first_exact_recurrence_after_creation'].result())
    return r


class SparseCode:
    """Fixed sensory code. Exact raw bytes are the only hash input."""
    def __init__(self,seed):
        rr=np.random.default_rng(seed+3604096)
        self.contacts=np.array([rr.choice(96,5,replace=False) for _ in range(4096)])
        self.weights=rr.normal(size=(4096,5))
        self.weights/=np.linalg.norm(self.weights,axis=1,keepdims=True)
        self.encoder=Encoder(seed,'lag')
    def encode(self,c,hashed=False):
        if hashed:
            rr=np.random.default_rng(int.from_bytes(hashlib.sha256(bytes(c)[-12:]).digest()[:16],'little'))
            ix=np.sort(rr.choice(4096,205,replace=False))
            # Independent signed amplitudes preserve a unit-norm score code without shared bias.
            vals=(2*rr.integers(0,2,205)-1)/math.sqrt(205)
        else:
            k=self.encoder.encode(c)[0]
            u=np.tanh(np.sum(self.weights*k[self.contacts],axis=1))
            # Stable ordering makes ties deterministic.
            ix=np.sort(np.argsort(-np.abs(u),kind='stable')[:205]);vals=u[ix]
            norm=np.linalg.norm(vals)
            vals=vals/norm if norm else np.zeros(205)
        return ix,vals

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


def geometry_preflight(seed):
    stream=make_stream(seed);enc=Encoder(seed,'lag');code=SparseCode(seed)
    keys=stream['keys'];z=np.array([[enc.encode(context(k,a))[0] for a in range(3)] for k in keys])
    active=np.zeros((len(keys),3,4096),dtype=np.int16)
    for i,k in enumerate(keys):
        for a in range(3):active[i,a,code.encode(context(k,a))[0]]=1
    lag={};fly={};passed=True
    for a in (1,2):
        sims=z[:,0]@z[:,a].T;mask=~np.eye(len(keys),dtype=bool)
        same=np.diag(sims);diff=sims[mask];ranks=1+np.sum(sims>same[:,None]+1e-14,axis=1)
        lag[str(a)]={'same':distribution(same),'different':distribution(diff),'nearest_neighbor_ranks':ranks.tolist(),
            'false_neighbor_pair_rate_at_tau':float(np.mean(diff>=TAU)),
            'wrong_nearest_rate_at_tau':float(np.mean((np.argmax(sims,axis=1)!=np.arange(len(keys)))&(np.max(sims,axis=1)>=TAU)))}
        overlap=active[:,0]@active[:,a].T;j=overlap/(410-overlap)
        samej=np.diag(j);diffj=j[mask];s=mean(samej);d=mean(diffj);ratio=s/d if d else None
        ok=s>=.30 and d<=.04 and (d==0 or ratio>=10)
        passed &= ok
        fly[str(a)]={'same':distribution(samej),'different':distribution(diffj),'ratio':ratio,'passed':bool(ok)}
    f=active.mean(axis=(0,1));ordered=np.sort(f);n=len(f)
    gini=float((2*np.sum(np.arange(1,n+1)*ordered)/(n*ordered.sum()))-(n+1)/n) if ordered.sum() else 0.
    # Evaluator-only current-fact minimum-norm oracle, never used in learning.
    revised=dict(enumerate(map(int,stream['labels'])));revised.update(stream['revisions'])
    X=z[:,:2].reshape(-1,96);Y=np.eye(256)[np.repeat([revised[i] for i in range(len(keys))],2)]
    W=np.linalg.pinv(X,rcond=1e-12)@Y
    oracle={str(a):float(np.mean([np.argmax(z[i,a]@W)==revised[i] for i in stream['revised']])) for a in range(3)}
    return {'seed':seed,'lag':lag,'fly':fly,'passed':bool(passed),'code_utilization':{'frequencies':f.tolist(),'mean':mean(f),'maximum':float(f.max()),'gini':gini,'never_active_fraction':float(np.mean(f==0))},'current_fact_oracle_revised_accuracy':oracle}


def preflight():
    rows=[geometry_preflight(s) for s in PREFLIGHT_SEEDS]
    # Conservative completion: every fixed preflight seed must meet both pair gates.
    gate=all(r['passed'] for r in rows)
    return {'seeds':rows,'fly_gate_passed':gate,'verdict':'fly_separation_gate_passed' if gate else 'fly_separation_gate_failed',
        'gate_aggregation':'Both alias pairs on every one of four preflight seeds; no outcome tuning.',
        'alias_task_oracle_valid':mean(r['current_fact_oracle_revised_accuracy']['2'] for r in rows)>1/24,
        'hash_completion':'Exact-context SHA256 -> seeded Top205 without replacement, independent signed equal amplitudes, unit norm. Fixed and not tuned.'}


def lag_stream_diagnostics(seeds,smoke=False):
    rows=[]
    for seed in seeds:
        stream=make_stream(seed,smoke=smoke);enc=Encoder(seed,'lag')
        z=np.array([[enc.encode(context(k,a))[0] for a in range(3)] for k in stream['keys']])
        pairs={}
        for a in (1,2):
            sims=z[:,0]@z[:,a].T;mask=~np.eye(len(z),dtype=bool);same=np.diag(sims)
            pairs[str(a)]={'same':distribution(same),'different':distribution(sims[mask]),
                'nearest_neighbor_ranks':(1+np.sum(sims>same[:,None]+1e-14,axis=1)).tolist(),
                'false_neighbor_pair_rate_at_tau':float(np.mean(sims[mask]>=TAU))}
        rows.append({'seed':seed,'stream_sha256':stream['raw_sha256'],'pairs':pairs})
    return rows


def invariant_audit(records):
    checked=0
    for seed in sorted({r['seed'] for r in records}):
        rows=[r for r in records if r['seed']==seed and r['arm'] in MAIN_ARMS and r['arm']!='FROZEN-R']
        if len(rows)>1:
            for field in ('base_state_sha256','eligibility_sha256','stream_sha256'):
                if len({r[field] for r in rows})!=1:raise AssertionError('Causal invariant mismatch '+field)
            for cp in rows[0]['eligibility']:
                for field in ('base_hash','eligibility_hash'):
                    if len({r['eligibility'][cp][field] for r in rows})!=1:raise AssertionError('Checkpoint base/eligibility mismatch')
            checked+=1
        frozen=[r for r in records if r['seed']==seed and r['arm']=='FROZEN-R']
        if frozen:
            f=frozen[0]
            assert len({x['base_hash'] for name,x in f['eligibility'].items() if name=='after_A' or not name.startswith('A_')})==1
    return {'passed':True,'paired_seed_sets_checked':checked,'frozen_exception':'Boundary freezing is a noncausal evaluator intervention, excluded from deployable base equality.'}


def build_report(records,seeds,flight):
    invariants=invariant_audit(records);arms=MAIN_ARMS+(FLY_ARMS if flight['fly_gate_passed'] else ())
    complete=len(records)==len(seeds)*len(arms);levels={};qual={}
    for arm in arms:
        levels[arm]={k:level(records,arm,k,seeds) for k in ('A_initial_accuracy','A_after_B','A_after_C','A_after_D','D_current','revision_4_alias_0','revision_4_alias_1','revision_4_alias_2','obsolete_8_alias_0','whole_bpb','target_bpb','nontarget_bpb','base_bpb_delta','false_merge_rate','same_key_merge_rate')}
        l=levels[arm];val=lambda k:l[k]['mean']
        full=all(x['n']==len(seeds) for x in l.values())
        if not full:qual[arm]={'verdict':'incomplete'};continue
        q={'Q1':acquisition(records,arm,seeds)['passed'],'Q2':val('A_after_D')>=.90,
           'Q3':val('revision_4_alias_0')>=.90 and val('obsolete_8_alias_0')<=.10,
           'Q4':val('revision_4_alias_1')>=.70,'Q5':val('revision_4_alias_2')>=.50,
           'Q6':val('base_bpb_delta')<=.10,'Q7':val('false_merge_rate')<.05}
        valid=arm!='FROZEN-R' and arm not in FLY_ARMS
        qualified=all(q.values()) and valid and flight['alias_task_oracle_valid']
        control={'EXACT-R':'EXACT-V','EXACT-V':'EXACT-R','SIM-R':'EXACT-R','SIM-V':'EXACT-V','PRED-V':'SIM-V','FROZEN-R':'EXACT-R','FLY-O':'HASH-O','HASH-O':'FLY-O'}[arm]
        effect=difference(records,arm,control,'A_after_D',seeds)
        verdict='alias_task_invalid' if not flight['alias_task_oracle_valid'] else 'addressing_invalid_false_merges' if not q['Q7'] else graded(effect,q['Q1'],qualified)
        qual[arm]={'gates':q,'qualified':qualified,'verdict':verdict,'deployable_main_candidate':valid,
            'grading_retention_control':control,'grading_retention_effect':effect}
    contrasts={}
    for name,a,b,key,threshold in [('V1','EXACT-V','EXACT-R','A_after_D',.40),('V2','FROZEN-R','EXACT-R','A_after_D',0.),('V3','SIM-V','EXACT-V','revision_4_alias_1',.30),('V4','SIM-V','SIM-R','A_after_D',0.),('V5','PRED-V','SIM-V','A_after_D',0.)]:
        effect=difference(records,a,b,key,seeds);valid=acquisition(records,a,seeds)['passed'] and acquisition(records,b,seeds)['passed']
        support=effect['low'] is not None and effect['low']>0 and effect['mean']>=threshold and valid
        if name=='V3':support=support and (levels[a]['A_after_D']['mean'] or 0)>=.85
        contrasts[name]={'candidate':a,'control':b,'endpoint':key,'effect_fraction':effect,'prespecified_large_effect_support':bool(support),'verdict':graded(effect,valid)}
    frozen_vs_absolute=difference(records,'FROZEN-R','EXACT-V','A_after_D',seeds)
    contrasts['V2']['frozen_minus_absolute']=frozen_vs_absolute
    contrasts['V2']['within_10pp']=None if frozen_vs_absolute['n']!=len(seeds) else abs(frozen_vs_absolute['mean'])<=.10
    v5={k:difference(records,'PRED-V','SIM-V',k,seeds) for k in ('false_merge_rate','same_key_merge_rate','A_after_D','whole_bpb')}
    v5full=all(v['n']==len(seeds) for v in v5.values())
    contrasts['V5']['mechanism_effects']=v5
    contrasts['V5']['mean_mechanism_conditions_met']=bool(v5full and v5['false_merge_rate']['mean']<0 and v5['same_key_merge_rate']['mean']>=-.05 and v5['A_after_D']['mean']>=-.05 and v5['whole_bpb']['mean']<=.10)
    contrasts['V5']['interval_supported_conditions']=bool(v5full and v5['false_merge_rate']['high']<0 and v5['same_key_merge_rate']['low']>=-.05 and v5['A_after_D']['low']>=-.05 and v5['whole_bpb']['high']<=.10)
    conditional={}
    if flight['fly_gate_passed']:
        conditional={'FLY_minus_HASH':{k:difference(records,'FLY-O','HASH-O',k,seeds) for k in ('A_after_D','D_current','revision_4_alias_0','revision_4_alias_1','revision_4_alias_2')}}
        if levels['FLY-O']['A_after_D']['n']==len(seeds):
            l=levels['FLY-O'];v=lambda k:l[k]['mean']
            conditional.update(F1_practically_interesting=v('A_after_D')>=.50,F3_ordinary_sufficiency=v('A_after_D')>=.75 and v('D_current')>=.90 and v('revision_4_alias_0')>=.90 and v('revision_4_alias_1')>=.70)
    return {'version':VERSION,'status':'complete' if complete else 'partial','scientific':seeds==SEEDS,'completed':len(records),'expected':len(seeds)*len(arms),
        'levels':levels,'qualification':qual,'contrasts':contrasts,'causal_invariants':invariants,'fly_preflight_verdict':flight['verdict'],
        'alias_task_verdict':'valid' if flight['alias_task_oracle_valid'] else 'alias_task_invalid',
        'conditional_fly':conditional,'limitations':['Ownership store is unbounded.','FROZEN-R cannot qualify.','Seeds are statistical units; no parameter tuning or additional stages.','False binding rates use earlier permanent target members; final impurity reported separately.']}


def summary_text(report):
    lines=['# V36 — Generalizing Ownership','',f"Status: {report['status']}; {report['completed']}/{report['expected']} trajectories.",'', '| Arm | A acquired | A after D | D current | Known revision 4 | Held-out revision 4 |','|---|---:|---:|---:|---:|---:|']
    def fmt(v):return '—' if v is None else f'{100*v:.2f}%'
    for a,l in report['levels'].items():lines.append('| '+a+' | '+' | '.join(fmt(l[k]['mean']) for k in ('A_initial_accuracy','A_after_D','D_current','revision_4_alias_1','revision_4_alias_2'))+' |')
    lines+=['',report['fly_preflight_verdict'],'','Intervals, V1–V5, all qualification gates, sample accounting, and limitations are in REPORT.json.','FROZEN-R is a nondeployable boundary intervention; all local slot arms are unbounded.']
    return '\n'.join(lines)+'\n'


def self_tests():
    checks=[]
    for arm in MAIN_ARMS:
        m=ByteModel(PREFLIGHT_SEEDS[0],arm);c=context(b'ABC123',0);other=context(b'XYZ999',0)
        m.observe(c,97);m.observe(other,98)
        before=list(m.values);j=m.members[c];old=np.frombuffer(before[j],dtype=np.float64).copy();used=m.scores(c).copy()
        m.observe(c,99);expected=old+RATE*(np.eye(256)[99]-used) if arm.endswith('-R') else .4*old+.6*np.eye(256)[99]
        assert np.allclose(np.frombuffer(m.values[j],dtype=np.float64),expected,atol=1e-15)
        assert all(v is m.values[i] for i,v in enumerate(before) if i!=j)
        h=m.state_hash();m.probabilities(context(b'ABC123',2));assert h==m.state_hash()
        assert all(np.allclose(m.prototypes[i],np.mean([m.member_vectors[c] for c in mem],axis=0)/np.linalg.norm(np.mean([m.member_vectors[c] for c in mem],axis=0))) for i,mem in enumerate(m.slot_members))
        checks.append(arm+' value/locality/probe/prototype')
    # Force high similarity from two one-bit different exact histories; verify pre-update confirmation.
    c=bytes([64])*12;c2=bytes([64])*11+bytes([65])
    for y,merged in ((97,True),(98,False)):
        m=ByteModel(PREFLIGHT_SEEDS[0],'PRED-V');m.observe(c,97)
        assert m.lookup(c2)[0]==0
        m.base=Learner('full')  # Ensure common base-error eligibility independently of the candidate.
        m.observe(c2,y);assert (m.members[c2]==m.members[c])==merged
    checks.append('PRED-V accepts correct pre-update consequence; rejects contradictory candidate')
    # Frozen state is unaffected by subsequent bytes while residual learning continues.
    m=ByteModel(PREFLIGHT_SEEDS[0],'FROZEN-R');m.observe(c,97);m.frozen=True;h=m.base.state_hash();m.observe(c,98);assert h==m.base.state_hash()
    checks.append('Frozen base invariant')
    code=SparseCode(PREFLIGHT_SEEDS[0])
    assert all(len(set(row))==5 for row in code.contacts)
    for hashed in (False,True):
        ix,v=code.encode(c,hashed);assert len(set(ix))==205 and abs(v@v-1)<1e-12
        ix2,v2=code.encode(c,hashed);assert np.array_equal(ix,ix2) and np.array_equal(v,v2)
    checks.append('Fly/hash deterministic width, distinct contacts, active count, unit norm')
    stream=make_stream(PREFLIGHT_SEEDS[0]);assert len(stream['rows'])==2084 and len(stream['raw'])==29176
    checks.append('Historical full workload counts')
    return {'passed':True,'checks':checks}


def engineering_verification(full=False):
    tests=self_tests();rows=[]
    for arm in MAIN_ARMS+FLY_ARMS:
        a=run_trial(PREFLIGHT_SEEDS[0],arm,not full,True)
        b=run_trial(PREFLIGHT_SEEDS[0],arm,not full,False)
        assert a['state_sha256']==b['state_sha256'] and a['whole_stream']==b['whole_stream']
        rows.append(a);print('Observer invariant passed:',arm,flush=True)
    invariant_audit(rows)
    return dict(tests,full_length=full,observer_arms=list(MAIN_ARMS+FLY_ARMS),engineering_seed=PREFLIGHT_SEEDS[0],
        final_hashes={r['arm']:r['state_sha256'] for r in rows})


class Runner:
    def __init__(self,root,smoke=False,max_new=None):
        self.root=Path(root);self.smoke=smoke;self.max_new=max_new;self.records=[];self.new=0;self.reused=0
        self.seeds=PREFLIGHT_SEEDS[:2] if smoke else SEEDS
        self.source=Path(__file__).read_bytes()
        self.plan={'seeds':list(self.seeds),'main_arms':list(MAIN_ARMS),'conditional_arms':list(FLY_ARMS),
            'preflight_seeds':list(PREFLIGHT_SEEDS),'smoke':smoke,'streams':{str(s):make_stream(s,smoke=smoke)['raw_sha256'] for s in self.seeds},
            'conditional_rule':'All representation thresholds on each of four fixed preflight seeds; no fly jobs on failure.',
            'tau':TAU,'boundary_resume':'Completed trajectories reused; interrupted trajectory restarts deterministically.'}
        self.identity={'source_sha256':sha(self.source),'protocol_sha256':sha(PROTOCOL.encode()),'plan_sha256':sha(self.plan)}
    def setup(self):
        self.root.mkdir(parents=True,exist_ok=True);path=self.root/'RUN_MANIFEST.json'
        env={'python':sys.version,'numpy':np.__version__,'scipy':importlib.import_module('scipy').__version__,'platform':platform.platform()}
        manifest=dict(self.identity,environment=env,version=VERSION,reference_sha256=REFERENCE_SHA256,seed_audit=SEED_AUDIT,blas_threads=1)
        if path.exists():
            if json.loads(path.read_text())!=manifest:raise RuntimeError('Resume rejected: source, protocol, plan, or numerical environment changed')
            if json.loads((self.root/'PLAN.json').read_text())!=self.plan:raise RuntimeError('Corrupt plan')
        else:
            if any(self.root.glob('jobs/*.json')):raise RuntimeError('Orphan jobs without manifest')
            write_json(path,manifest);write_json(self.root/'PLAN.json',self.plan)
        atomic(self.root/Path(__file__).name,self.source.decode());atomic(self.root/'PROTOCOL.md',PROTOCOL)
        (self.root/'jobs').mkdir(exist_ok=True)
        self.flight=self.cached('PREFLIGHT.json',preflight)
        self.cached('LAG_SIMILARITY.json',lambda:lag_stream_diagnostics(self.seeds,self.smoke))
        self.verification=self.cached('OBSERVER_AUDIT.json',lambda:engineering_verification(False))
        self.arms=MAIN_ARMS+(FLY_ARMS if self.flight['fly_gate_passed'] else ())
        allowed={f'{s}_{a}.json' for s in self.seeds for a in self.arms}
        for p in (self.root/'jobs').glob('*.json'):
            if p.name not in allowed:raise RuntimeError('Foreign or ineligible cached trajectory: '+p.name)
    def cached(self,name,compute):
        p=self.root/name
        if p.exists():
            e=json.loads(p.read_text())
            if e['identity']!=self.identity or e['sha256']!=sha(e['result']):raise RuntimeError('Corrupt cache '+name)
            return e['result']
        r=compute();write_json(p,{'identity':self.identity,'result':r,'sha256':sha(r)});return r
    def run(self):
        for seed in self.seeds:
            for arm in self.arms:
                p=self.root/'jobs'/f'{seed}_{arm}.json'
                if p.exists():
                    e=json.loads(p.read_text());r=e['result']
                    if e['identity']!=self.identity or e['sha256']!=sha(r) or r['seed']!=seed or r['arm']!=arm or r['smoke']!=self.smoke:raise RuntimeError('Corrupt or mismatched job '+p.name)
                    if r['stream_sha256']!=self.plan['streams'][str(seed)]:raise RuntimeError('Wrong stream')
                    standard=metrics(r)
                    if any(r['metrics'].get(k)!=v for k,v in standard.items()):raise RuntimeError('Inconsistent cached metrics')
                    if r['counts']['byte_events']!=r['whole_stream']['n']:raise RuntimeError('Inconsistent byte counts')
                    self.reused+=1
                else:
                    if self.max_new is not None and self.new>=self.max_new:return False
                    print(f'Starting {seed} {arm}',flush=True);r=run_trial(seed,arm,self.smoke)
                    write_json(p,{'identity':self.identity,'result':r,'sha256':sha(r)});self.new+=1
                    print(f"Completed {arm}, {r['elapsed_seconds']:.1f}s",flush=True)
                self.records.append(r);invariant_audit(self.records)
                write_json(self.root/'STATUS.json',{'status':'running','completed':len(self.records),'new':self.new,'reused':self.reused})
        return True
    def finish(self,complete):
        report=build_report(self.records,self.seeds,self.flight)
        if not complete:report['status']='paused'
        write_json(self.root/'REPORT.json',report);atomic(self.root/'SUMMARY.md',summary_text(report))
        csv_output(self.root/'SUMMARY.csv',[dict(seed=r['seed'],arm=r['arm'],**r['metrics']) for r in self.records])
        for filename,key in [('NOISE','noise_events'),('BASE_DRIFT','base_drift'),('STATE_COSTS','costs'),('EXPOSURES','counts'),('SLOT_LINKS','addressing'),('ALIAS_LINK_CURVES','alias_link_curves'),('SAMPLE_EFFICIENCY','sample_efficiency')]:
            write_json(self.root/(filename+'.json'),[{'seed':r['seed'],'arm':r['arm'],'data':r[key]} for r in self.records])
        for typ,filename,field in [('wave','RETENTION_BY_WAVE','untouched_A'),('revision','REVISION_AND_OBSOLETE_CURVES','revised_A'),('acquisition','ACQUISITION_CURVES','current')]:
            rows=[]
            for r in self.records:
                for cp in r['snapshots'].values():
                    if cp['type']==typ:
                        for a,pr in cp[field].items():rows.append(dict(seed=r['seed'],arm=r['arm'],checkpoint=cp['name'],alias=a,**pr))
            csv_output(self.root/(filename+'.csv'),rows)
        write_json(self.root/'STATUS.json',{'status':report['status'],'completed':len(self.records),'new':self.new,'reused':self.reused})
        paths=[p for p in self.root.rglob('*') if p.is_file() and not p.name.startswith('.') and '.tmp-' not in p.name and p.name!='ARTIFACT_HASHES.json']
        write_json(self.root/'ARTIFACT_HASHES.json',{str(p.relative_to(self.root)):sha(p.read_bytes()) for p in paths})
        temp=self.root.with_suffix('.zip.tmp');archive=self.root.with_suffix('.zip')
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in paths+[self.root/'ARTIFACT_HASHES.json']:z.write(p,p.relative_to(self.root))
        os.replace(temp,archive)
        print(f"{report['status']}: {self.new} new, {self.reused} reused; {archive}",flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install-deps',action='store_true');parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--verify',action='store_true',help='Full-length engineering observer audit only; no science')
    parser.add_argument('--out',type=Path);parser.add_argument('--max-new-jobs',type=int)
    args=parser.parse_args(argv);dependencies(args.install_deps)
    if args.max_new_jobs is not None and args.max_new_jobs<0:parser.error('--max-new-jobs must be nonnegative')
    caffeine=None
    if sys.platform=='darwin':
        try:caffeine=subprocess.Popen(['caffeinate','-dimsu','-w',str(os.getpid())])
        except FileNotFoundError:print('caffeinate not available; continuing',file=sys.stderr)
    try:
        if args.verify:
            result=engineering_verification(True);print(json.dumps(result,indent=2));return
        root=args.out or Path('v36_smoke' if args.smoke else 'v36_results');root.mkdir(parents=True,exist_ok=True)
        with RunLock(root):
            runner=Runner(root,args.smoke,args.max_new_jobs);runner.setup()
            complete=runner.run();runner.finish(complete)
    finally:
        if caffeine is not None:caffeine.terminate()

if __name__=='__main__':main()
