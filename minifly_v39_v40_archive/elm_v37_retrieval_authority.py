#!/usr/bin/env python3
"""V37: Corroborated Retrieval Authority. One file, 16 fixed scientific trajectories.
python3 elm_v37_retrieval_authority.py --install-deps
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
VERSION='v37.1.0'
SEEDS=tuple(range(37101,37109))
ENGINEERING_SEEDS=(973701,973702)


REFERENCE_SHA256='e44443a37994812f64c190d77b83d8da3872ccfb26488e4ebc27fd24ccbad3ef'

PROTOCOL="# V37 — Corroborated Retrieval Authority\n## Focused design after the two V36 branches\n\n**Status:** design / preregistration only. No V37 scientific outcomes have been observed.\n\n---\n\n## 0. Decision in one sentence\n\nV37 will **not** change the successful V36 ownership or similarity-binding mechanism. It will hold the PRED-V learner state fixed and test only the missing variable:\n\n\\[\n\\boxed{\\text{When is a merely similar, not-yet-confirmed memory allowed to control prediction?}}\n\\]\n\nThe primary candidate is **corroborated authority**: an exact member always owns its prediction, but an unseen similar context may generalize from a slot only after that slot has been independently confirmed by at least two distinct exact contexts.\n\n---\n\n# 1. Evidence inherited from V36\n\nTwo different experiments were called V36 and must not be conflated.\n\n## 1.1 Generalizing-ownership V36\n\nThe six-arm ownership experiment completed 48/48 trajectories.\n\n| Arm | A acquired | A after D | D current | Known-alias revision @4 | Held-out revision @4 |\n|---|---:|---:|---:|---:|---:|\n| EXACT-R | 100.00% | 29.69% | 100.00% | 25.00% | 12.50% |\n| EXACT-V | 100.00% | 98.44% | 100.00% | 0.00% | 12.50% |\n| SIM-R | 100.00% | 27.34% | 100.00% | 65.62% | 21.88% |\n| SIM-V | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |\n| PRED-V | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |\n| FROZEN-R | 100.00% | 100.00% | 100.00% | 0.00% | 0.00% |\n\nThe causal interpretation is now strong:\n\n1. **absolute ownership fixes base-drift forgetting**;\n2. **similarity-linked slots add cross-rendering identity**;\n3. the combination gives essentially perfect persistence + revision + held-out alias transfer on the target task.\n\nThe failure is Q6, whole-stream utility.\n\n| Arm | whole-stream bpb | delta vs its base-only counterfactual | nontarget bpb |\n|---|---:|---:|---:|\n| EXACT-V | **1.244** | **-0.217** | **1.291** |\n| PRED-V | 2.280 | +0.819 | 2.427 |\n| SIM-V | 2.792 | +1.330 | 2.977 |\n\nThus exact ownership itself is not the problem. The damage appears when an **unbound context is allowed to retrieve a merely similar slot before that identity relation has been established**.\n\nPRED-V already moved in the correct direction: compared with SIM-V, it improved whole-stream bpb by about **0.511 bits/byte** with no loss in persistence or alias revision. But V36 predictive confirmation occurs only **after** the true next byte is observed, so it cannot protect the prediction that has already been made.\n\n## 1.2 Separation V36\n\nThe independent separation branch successfully changed representation geometry:\n\n| code | separation ratio |\n|---|---:|\n| raw | 1.43x |\n| norm | 1.70x |\n| expanded | 2.60x |\n| phase-normalized | 5.71x |\n| phase-expanded | 5.61x |\n\nBut stronger separation was not monotonically better for prediction. The most separated codes were worse than raw, while moderate expansion gave only a modest improvement.\n\nTherefore V37 treats the following as a closed conclusion for the present workload:\n\n\\[\n\\boxed{\\text{maximizing same/different separation is not, by itself, the solution.}}\n\\]\n\nThe useful architectural implication is narrower: a code useful for **identity/addressing** need not be the same code used for **general next-byte prediction**. That dual-code idea is retained as a contingency, but it is not manipulated in the primary V37 round.\n\n---\n\n# 2. What is fixed and will not be reopened\n\nV37 freezes the following V36 choices:\n\n- the V36 workload and causal next-byte teaching signal;\n- the shared base learner;\n- lag96 similarity geometry;\n- fixed similarity threshold\n  \\[\n  \\tau=0.85;\n  \\]\n- absolute slot value semantics;\n- PRED-V permanent-binding rule;\n- slot prototype update;\n- exact-member hard ownership;\n- no replay;\n- no task IDs or logical-key IDs in the learner;\n- no covariance, RLS, MESU, novelty gating, competitive plasticity, or new write rule.\n\nV37 is specifically **not** another search over how weights are written.\n\n---\n\n# 3. The mechanistic decomposition V36 exposed\n\nFor a context \\(c\\), separate three decisions that V36 partially conflated:\n\n1. **proposal** — which slot, if any, is similar to \\(c\\)?\n2. **binding** — after observing the consequence, should \\(c\\) become a permanent member of that slot?\n3. **authority** — before observing the consequence, how much should the proposed slot alter the current prediction?\n\nPRED-V already gives us a causal proposal and binding rule. V37 leaves both unchanged and manipulates only step 3.\n\nThis produces a cleaner causal experiment than creating another memory architecture.\n\n---\n\n# 4. Core V37 insight: one learner trajectory, many read policies\n\nThe PRED-V binding rule is based on the candidate slot's **pre-update internal prediction**, not on which externally reported read policy was used.\n\nTherefore all V37 authority policies can share, for each seed:\n\n- exactly the same base state at every event;\n- exactly the same slots;\n- exactly the same prototypes;\n- exactly the same permanent bindings;\n- exactly the same local values;\n- exactly the same eligibility events.\n\nOnly the reported predictive distribution changes on a **tentative, unbound similarity hit**.\n\nThis means V37 can score several authority policies counterfactually during one underlying PRED-V trajectory instead of training a separate learner for every policy.\n\nThat is both cheaper and scientifically cleaner.\n\n---\n\n# 5. Common state evolution: PRED-V unchanged\n\nFor an unbound context \\(c\\):\n\n1. find the nearest slot \\(j^*\\) by prototype cosine similarity;\n2. if\n   \\[\n   s_1=p_{j^*}^{\\top}k(c) < \\tau,\n   \\]\n   there is no tentative candidate;\n3. otherwise retain \\(j^*\\) as a tentative candidate;\n4. compute the candidate slot's internal pre-update prediction;\n5. observe the true next byte \\(y\\);\n6. if the candidate's pre-update top-1 byte equals \\(y\\), permanently bind \\(c\\) to that slot;\n7. otherwise allocate a new slot when the unchanged V36 eligibility rule requires it.\n\nOnce an exact context is permanently bound, it receives full absolute ownership exactly as in V36.\n\nNo authority policy below is allowed to alter this state evolution.\n\n---\n\n# 6. V37 authority policies\n\nLet\n\n- \\(p_B(y\\mid c)\\) be the shared-base predictive distribution;\n- \\(p_j(y)\\) be the candidate slot distribution;\n- \\(s_1\\) be nearest-slot similarity.\n\nFor exact-member hits, **every deployable policy uses \\(p_j\\) with full authority**.\n\nPolicies differ only for an **unbound context with a tentative candidate**.\n\n## 6.1 HARD — V36 PRED-V replication\n\n\\[\np_{\\mathrm{HARD}}=p_j.\n\\]\n\nThis is the V36 behavior and is expected to reproduce excellent target memory with poor whole-stream utility.\n\n## 6.2 SHADOW — diagnostic no-authority control\n\n\\[\np_{\\mathrm{SHADOW}}=p_B.\n\\]\n\nThe tentative slot is still used internally for the unchanged PRED-V binding test after \\(y\\) arrives, but it has **zero authority over the current prediction**.\n\nThis is deployable in principle but is a diagnostic here because a never-seen held-out alias cannot benefit at prediction time.\n\nIf SHADOW restores nontarget/whole-stream utility while HARD does not, the V36 Q6 failure is causally localized to speculative read authority.\n\n## 6.3 CORR2 — primary V37 candidate: corroborated authority\n\nA slot may generalize to a previously unseen context only if it already contains at least two **distinct exact member contexts** before the current event:\n\n\\[\n|M_j|\\ge 2.\n\\]\n\nThen\n\n\\[\np_{\\mathrm{CORR2}}=\n\\begin{cases}\n p_j, & |M_j|\\ge2,\\\\\n p_B, & |M_j|<2.\n\\end{cases}\n\\]\n\nWhy two?\n\n- one context establishes an episodic memory but gives no independent evidence that its similarity neighborhood corresponds to a reusable identity;\n- a second distinct context that independently passes the unchanged predictive-confirmation rule is the minimum possible corroboration;\n- V36 trains two renderings before evaluating the third held-out rendering, so \\(|M_j|=2\\) is sufficient in principle for genuine unseen-alias transfer;\n- the threshold is structural, not tuned on V37 outcomes.\n\nCORR2 embodies the hypothesis:\n\n\\[\n\\boxed{\\text{exact ownership is immediate; generalization authority must be earned.}}\n\\]\n\n## 6.4 SIM-MIX — secondary continuous-authority policy\n\nFor a tentative candidate, define the parameter-free excess-similarity weight\n\n\\[\n\\alpha(c)=\\operatorname{clip}\\left(\\frac{s_1-\\tau}{1-\\tau},0,1\\right).\n\\]\n\nMix probabilities rather than logits:\n\n\\[\np_{\\mathrm{SIMMIX}}=(1-\\alpha)p_B+\\alpha p_j.\n\\]\n\nThis has two useful properties:\n\n1. a candidate exactly at the existing V36 threshold has zero predictive authority;\n2. only an exact similarity of 1 obtains full authority.\n\nIt introduces no fitted scale beyond the already fixed \\(\\tau=0.85\\).\n\nSIM-MIX is secondary. V37 is not an alpha-tuning round.\n\n## 6.5 Fixed-alpha authority frontier — diagnostic only\n\nBecause all read policies share the same state path, also report:\n\n\\[\n\\alpha\\in\\{0,0.10,0.25,0.50,1.00\\}\n\\]\n\nwith\n\n\\[\np_\\alpha=(1-\\alpha)p_B+\\alpha p_j\n\\]\n\non tentative candidates only.\n\nThis is not used to select a post-hoc winning architecture. It answers a basic identifiability question:\n\n> Is there already a simple global authority level at which held-out transfer and Q6 coexist, or is context-selective authority genuinely necessary?\n\n## 6.6 RAND-MATCH — nondeployable matched-usage control\n\nFor each seed, score an evaluator-side random authority mask that grants HARD tentative authority on exactly the same number of tentative events as CORR2, but chooses those events uniformly at random.\n\nThe learner state remains unchanged.\n\nRAND-MATCH is not a candidate architecture. Its role is the same anti-shortcut role that random gates and bottom-k controls played in earlier rounds:\n\n\\[\n\\boxed{\\text{Does CORR2 help because corroboration identifies useful reads, or merely because it reads memory less often?}}\n\\]\n\n---\n\n# 7. Fresh scientific run\n\nUse eight fresh seeds:\n\n\\[\n37101,37102,37103,37104,37105,37106,37107,37108.\n\\]\n\nFor each seed run only:\n\n1. one **PRED-V state trajectory** with all read policies scored in parallel;\n2. one fresh **EXACT-V** trajectory as the exact-ownership reference.\n\nThus the stateful scientific workload is only:\n\n\\[\n2\\times8=\\boxed{16\\text{ trajectories}}.\n\\]\n\nThe authority panel does not multiply training runs because it is read-only with respect to learner state.\n\nNo automatic follow-up experiment is run.\n\n---\n\n# 8. Hard implementation invariants\n\nV37 is invalid if any of these fail.\n\n## I1 — state-path identity across authority policies\n\nAt every checkpoint all PRED-derived read policies must have identical hashes for:\n\n- shared base;\n- slot values;\n- prototypes;\n- member sets;\n- eligibility state.\n\n## I2 — prediction precedes observation\n\nEvery authority policy is scored before the true next byte is exposed.\n\n## I3 — held-out alias probes are read-only\n\nThey never create bindings or alter support counts.\n\n## I4 — CORR2 uses only causal member count\n\nThe learner may count exact contexts in a slot, but may not know logical key, alias number, wave, target/non-target status, or evaluator labels.\n\n## I5 — RAND-MATCH cannot affect learning\n\nIt is scoring-only and clearly marked nondeployable.\n\n---\n\n# 9. Primary endpoints\n\nRetain all V36 endpoints so the successful properties cannot silently disappear.\n\n## 9.1 Persistence\n\n\\[\nA_{\\mathrm{afterD}}\n\\]\n\nfor untouched A.\n\n## 9.2 Current learning\n\n\\[\nD_{\\mathrm{current}}.\n\\]\n\n## 9.3 Exact revision\n\nAt 1/2/4/8 changed occurrences:\n\n- revised-target accuracy;\n- obsolete-target probability.\n\n## 9.4 Known-alias revision\n\nRendering 1 after revision of rendering 0.\n\n## 9.5 Held-out alias transfer\n\nNever-trained rendering 2, read-only.\n\nThis is the endpoint that distinguishes SHADOW from a genuinely generalizing authority rule.\n\n## 9.6 Whole-stream utility\n\nReport:\n\n- whole-stream bpb;\n- target bpb;\n- nontarget bpb;\n- base-only counterfactual bpb;\n- EXACT-V reference bpb.\n\n## 9.7 Authority usage\n\nFor each policy report separately on target and nontarget events:\n\n- tentative-candidate rate;\n- fraction receiving nonzero memory authority;\n- mean authority weight;\n- candidate-slot support distribution;\n- bpb conditional on support 1, 2, 3+;\n- accuracy conditional on support 1, 2, 3+.\n\n---\n\n# 10. New mechanistic endpoint: authority emergence\n\nV37 should explicitly measure the transition from episodic ownership to generalizable identity.\n\nFor each logical key, probe the never-trained rendering 2 at two read-only times:\n\n### E1 — after only one trained rendering is established\n\nThe relevant slot has, at most, one corroborated exact member.\n\nCORR2 should **not** generalize hard from it.\n\n### E2 — after the second trained rendering has independently joined the slot\n\nThe slot now has \\(|M_j|\\ge2\\).\n\nCORR2 may generalize to rendering 2.\n\nReport the change in held-out revised-target probability and accuracy:\n\n\\[\n\\Delta_{\\mathrm{corroboration}}=\n\\text{heldout after support 2}-\\text{heldout after support 1}.\n\\]\n\nThis is the cleanest direct test of the V37 mechanism.\n\n---\n\n# 11. Prespecified hypotheses\n\nUse paired two-sided 95% Student-t intervals across the eight fresh seeds. Absolute levels are always reported.\n\n## P1 — reproduce the V36 authority problem\n\nHARD should retain the V36 target-memory success:\n\n- A-after-D \\(\\ge90\\%\\);\n- D-current \\(\\ge90\\%\\);\n- known-alias revision @4 \\(\\ge70\\%\\);\n- held-out revision @4 \\(\\ge50\\%\\).\n\nIts whole-stream bpb is expected to remain materially worse than the base-only counterfactual.\n\nFailure to reproduce the target-memory behavior invalidates the round as a test of authority.\n\n## P2 — speculative authority causes the collateral damage\n\nCompare SHADOW against HARD on nontarget bpb and whole-stream bpb.\n\nStrong support requires:\n\n- paired SHADOW-HARD nontarget bpb difference < 0 with upper interval bound < 0;\n- same direction for whole-stream bpb;\n- identical learner-state hashes.\n\nThis is a causal read-path test, not a new learner claim.\n\n## P3 — corroboration can recover utility without giving up unseen-alias transfer\n\nCORR2 is the primary candidate.\n\nIt must satisfy the full qualification gates in Section 12.\n\nThe critical conjunction is:\n\n\\[\n\\boxed{Q5\\ \\text{held-out transfer} + Q6\\ \\text{whole-stream utility}.}\n\\]\n\nV36 had Q5 but not Q6; EXACT-V had Q6 but not Q5.\n\n## P4 — corroboration is informative, not merely less memory use\n\nCompare CORR2 with RAND-MATCH at matched tentative hard-read count.\n\nSupport requires CORR2 to show at least one of:\n\n- materially better held-out alias accuracy;\n- materially lower target bpb;\n\nwith no worse nontarget bpb.\n\nIf RAND-MATCH matches CORR2 on both target and nontarget endpoints, the support-count rule is merely an authority-rate reduction and should be retired.\n\n## P5 — fixed authority may or may not be enough\n\nThe fixed-alpha panel is diagnostic.\n\nIf a broad interval of constant \\(\\alpha\\) values simultaneously satisfies Q5 and Q6, there is no evidence that a special contextual authority rule is needed; prefer the simpler fixed mixture.\n\nIf no constant \\(\\alpha\\) can satisfy both while CORR2 can, context-selective authority is supported.\n\n---\n\n# 12. V37 qualification gates\n\nA deployable V37 candidate must satisfy all gates on fresh seeds.\n\n## Q1 — acquisition\n\n- A-after-A mean \\(\\ge90\\%\\);\n- minimum seed A-after-A \\(\\ge75\\%\\);\n- D-current mean \\(\\ge90\\%\\).\n\n## Q2 — strict persistence\n\n\\[\nA_{\\mathrm{afterD}}\\ge90\\%.\n\\]\n\n## Q3 — exact revision\n\nAfter four changed outcomes:\n\n\\[\n\\text{exact revised accuracy}\\ge90\\%.\n\\]\n\nAfter eight:\n\n\\[\nP(\\text{obsolete target})\\le0.10.\n\\]\n\n## Q4 — known-alias revision\n\nAfter four changed outcomes:\n\n\\[\n\\text{known-alias revised accuracy}\\ge70\\%.\n\\]\n\n## Q5 — held-out alias transfer\n\nAfter four changed outcomes:\n\n\\[\n\\boxed{\\text{held-out revised accuracy}\\ge50\\%.}\n\\]\n\n## Q6 — whole-stream utility\n\nMean whole-stream bpb may be no worse than the base-only counterfactual by more than:\n\n\\[\n\\boxed{0.10\\text{ bits/byte}.}\n\\]\n\n## Q6b — collateral-utility guard\n\nMean nontarget bpb may also be no worse than base-only by more than:\n\n\\[\n\\boxed{0.10\\text{ bits/byte}.}\n\\]\n\nThis prevents target gains from hiding broad collateral damage.\n\n## Q7 — no major target identity collapse\n\nEvaluator-known different-key permanent target merges remain below:\n\n\\[\n5\\%.\n\\]\n\n## Q8 — causal/state invariance\n\nAll hard implementation invariants I1-I5 pass.\n\nPassing Q1-Q8 establishes an **unbounded generalizing ownership + selective retrieval mechanism** on this workload. It does not establish bounded lifelong capacity.\n\n---\n\n# 13. Noise remains diagnostic, not a V37 optimization target\n\nRetain the V36 one-off corruption probe unchanged.\n\nDo not add a contradiction detector, majority vote, confidence window, or provisional revision buffer in this round.\n\nReason: once a context is a confirmed exact member, V37 authority policies intentionally give it full ownership. Therefore they should not solve the already observed fact that one erroneous observation can corrupt a shared slot.\n\nReport this failure honestly and leave it for the next mechanism round if V37 qualifies.\n\n---\n\n# 14. Capacity remains deferred one more round\n\nV36 local stores are unbounded, and PRED-V still allocated many slots. That is not acceptable for a lifelong system.\n\nHowever, bounding capacity before fixing Q6 would confound two separate questions:\n\n1. can the memory decide **when it is allowed to speak**?\n2. can the memory remain useful under a fixed storage budget?\n\nV37 answers (1) only.\n\nIf a V37 policy passes Q1-Q8, the next round should finally combine:\n\n- bounded slot capacity / eviction or consolidation;\n- the unchanged qualified authority rule;\n- the existing one-off-noise diagnostic.\n\nIf no V37 policy passes Q5+Q6, do **not** proceed to capacity.\n\n---\n\n# 15. Contingency if V37 fails\n\nThe merged V36 evidence leaves one specific fallback rather than a broad architecture search.\n\nIf:\n\n- SHADOW repairs Q6, proving speculative retrieval is the cause;\n- but neither fixed mixing nor CORR2 can preserve held-out transfer while meeting Q6;\n\nthen the next hypothesis is **dual representation**:\n\n\\[\n\\boxed{\n\\text{predictive code for ordinary next-byte modeling}\n\\quad+\\quad\n\\text{separate identity code used only for memory addressing}\n}\n\\]\n\nThe independent V36 separation branch motivates this because aggressive normalization increased identity separation while damaging ordinary prediction. That is exactly the pattern expected when one code is being forced to serve incompatible objectives.\n\nThis dual-code contingency should not be added to V37 unless the authority-only test fails.\n\n---\n\n# 16. What is explicitly out of scope\n\nDo not add in V37:\n\n- new plasticity selection rules;\n- novelty gates;\n- competitive / top-k / bottom-k learning;\n- MESU or covariance protection;\n- replay;\n- capacity eviction;\n- phase-normalized or KC-expanded code in the primary learner;\n- contradiction persistence logic;\n- task IDs;\n- semantic/logical-key labels;\n- a parameter sweep chosen post hoc for qualification.\n\nThe round is successful scientifically even if CORR2 fails, provided the read-authority causal decomposition is cleanly measured.\n\n---\n\n# 17. Required outputs\n\nThe single V37 runner should write:\n\n- `SUMMARY.md`\n- `REPORT.json`\n- `AUTHORITY_FRONTIER.csv`\n- `AUTHORITY_BY_SUPPORT.csv`\n- `RETENTION_BY_WAVE.csv`\n- `REVISION_CURVES.csv`\n- `HELDOUT_CORROBORATION_CURVE.csv`\n- `NONTARGET_AUTHORITY.csv`\n- `STATE_INVARIANCE.json`\n- `NOISE.json`\n- `STATE_COSTS.json`\n- `RUN_MANIFEST.json`\n\nThe report must distinguish:\n\n- deployable policies: HARD, SHADOW, CORR2, SIM-MIX, fixed-alpha policies;\n- nondeployable diagnostic control: RAND-MATCH;\n- stateful learner trajectory: PRED-V;\n- read-only counterfactual policies scored from that trajectory.\n\n---\n\n# 18. Execution constraint\n\nShip V37 as one self-contained executable Python file and one command, with a smoke/self-test mode.\n\nTarget interface:\n\n```bash\npython3 elm_v37_retrieval_authority.py --self-test\npython3 elm_v37_retrieval_authority.py --smoke\npython3 elm_v37_retrieval_authority.py --out-dir v37_results\n```\n\nThe scientific run should use the eight fixed fresh seeds in Section 7 and should not automatically launch a follow-up experiment based on intermediate results.\n\n---\n\n# 19. Expected interpretation tree\n\n### Outcome A — CORR2 passes Q1-Q8\n\nThe project has, on this workload:\n\n1. persistent absolute ownership;\n2. autonomous cross-context identity;\n3. held-out generalization;\n4. selective retrieval without broad predictive harm.\n\nProceed next to bounded capacity and noise-vs-revision.\n\n### Outcome B — a fixed-alpha mixture passes, CORR2 adds nothing\n\nPrefer the simpler mixture. Retrieval authority is a quantitative blending problem, not a special corroboration mechanism.\n\n### Outcome C — SHADOW fixes Q6, but every generalizing policy fails Q5 or Q6\n\nThe problem is confirmed to be speculative retrieval, but raw lag96 similarity cannot distinguish useful from harmful tentative reads well enough.\n\nProceed to the dual-code identity-address hypothesis motivated by the independent V36 separation result.\n\n### Outcome D — SHADOW does not materially improve Q6\n\nOur V36 diagnosis was incomplete. Damage is coming from permanent binding / exact-member use or another state-path effect, not merely current tentative authority. Stop and audit before proposing another mechanism.\n\n---\n\n# 20. Why this is the right V37\n\nV20-V35 repeatedly changed learning rules. V36 finally isolated a mechanism that works on the target memory problem and a separate failure that lives on the read path.\n\nV37 should exploit that decomposition rather than reopen the search space.\n\nThe proposed experiment is unusually strong because the main policies share an identical learner trajectory. Therefore a difference in bpb or held-out transfer cannot be blamed on different training, different slot allocation, different write budgets, or different base drift.\n\nThe central falsifiable question is simply:\n\n\\[\n\\boxed{\n\\textbf{Can a memory own confirmed experiences absolutely while requiring corroboration before generalizing that ownership to novel similar inputs?}\n}\n\\]\n"

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

POLICIES=('HARD','SHADOW','CORR2','SIM-MIX','ALPHA-0','ALPHA-0.10','ALPHA-0.25','ALPHA-0.50','ALPHA-1')
ALPHAS={'ALPHA-0':0.,'ALPHA-0.10':.1,'ALPHA-0.25':.25,'ALPHA-0.50':.5,'ALPHA-1':1.}
IMPLEMENTATION_NOTES={
 'random_control':'Uniform without replacement over all tentative stream events per seed; exactly matches CORR2 count. Separate uniform mask over all tentative evaluator probes, also count-matched. No stratification by labels. Nondeployable; pre-observation distributions are frozen, offline mask cannot influence learning.',
 'emergence':'E1 is first acquired rendering-0 binding while rendering 1 is unbound; E2 is first same-slot binding of both trained renderings. Record actual total support; no forced support=1 assumption. E1 slots already supported by unrelated contexts are flagged. Missing E1/E2 and noise/revision probes are never imputed.',
 'P4':'Design leaves materiality unspecified. Frozen implementation: target bpb improvement >=0.10 or heldout @4 accuracy improvement >=0.10, paired CI excluding zero, plus nontarget mean noninferiority with zero margin. Report component intervals; random control is never qualified.',
 'P5':'Report Q5+Q6 for every fixed alpha, no automatic selection or follow-up. A grid cannot establish a continuous interval.',
 'Q1':'A acquisition uses all A keys, with untouched-A acquisition separately reported.',
 'Q7':'Require both event-level different-key binding rate and final impure target-context fraction below 5%.',
 'invariance':'One physical PRED-V state. Read panel/probes cannot write it; checkpoint pre/post component hashes verified. Independently replay uninstrumented engineering controls in --self-test.'}

def component_hashes(m):
    return dict(base=m.base.state_hash(),values=sha(b''.join(m.values)),
        prototypes=sha(m.prototypes.tobytes()+b''.join(s.tobytes() for s in m.sums)),
        members=sha([(c.hex(),j,m.member_vectors[c].tobytes().hex()) for c,j in sorted(m.members.items())]),
        eligibility=m.eligibility_hash(),full=m.state_hash())

def read_panel(m,h):
    """No actual byte or evaluator metadata enters this function."""
    c=bytes(h)[-12:];pb=softmax(m.base_scores(c));j,s,margin=m.lookup(c)
    bound=c in m.members;tentative=j is not None and not bound
    support=0 if j is None else len(m.slot_members[j])
    pj=pb if j is None else softmax(np.frombuffer(m.values[j],dtype=np.float64))
    names=POLICIES if m.arm=='PRED-V' else ('EXACT-V',)
    weights={}
    for policy in names:
        if j is None:a=0.
        elif bound:a=1.
        elif policy=='HARD':a=1.
        elif policy=='SHADOW':a=0.
        elif policy=='CORR2':a=float(support>=2)
        elif policy=='SIM-MIX':a=float(np.clip((s-TAU)/(1-TAU),0.,1.))
        else:a=ALPHAS[policy]
        weights[policy]=a
    probs={p:(1-a)*pb+a*pj for p,a in weights.items()};probs['BASE']=pb;weights['BASE']=0.
    return probs,weights,dict(tentative=tentative,bound=bound,support=support,similarity=s,slot=j)

def score_panel(packet,y,old=None):
    probs,weights,meta=packet
    scored={p:dict(bpb=-math.log2(max(float(v[y]),1e-300)),accuracy=float(np.argmax(v)==y),
        target_probability=float(v[y]),obsolete_probability=None if old is None else float(v[old]),
        authority=weights[p]) for p,v in probs.items()}
    return scored,meta

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


def random_mask(entries,seed,offset):
    ix=[i for i,e in enumerate(entries) if e['meta']['tentative']]
    count=sum(e['meta']['tentative'] and e['meta']['support']>=2 for e in entries)
    rr=np.random.default_rng(seed+offset)
    chosen=set(map(int,rr.choice(ix,count,replace=False))) if count else set()
    assert len(chosen)==count
    return chosen,dict(tentative_events=len(ix),corr2_hard_reads=count,random_hard_reads=len(chosen))

def trial(seed,arm,smoke=False):
    start=time.perf_counter();stream=make_stream(seed,smoke=smoke);m=ByteModel(seed,arm)
    groups=collections.defaultdict(Stats);probes=[];events=[];checks=[];costs=[];emergence={};history=bytearray()
    support_hist=collections.Counter()
    cps=collections.defaultdict(list)
    for cp in stream['checkpoints']:cps[cp['record_count']].append(cp)
    def probe(name,kind,ids,aliases=(0,1,2),revised=False,extra=None,corrupted=None):
        for i in ids:
            y=stream['revisions'][i] if revised and i in stream['revisions'] else int(stream['labels'][i])
            for alias in aliases:
                # Freeze probabilities before scoring against evaluator-only labels.
                packet=read_panel(m,context(stream['keys'][i],alias))
                scores,meta=score_panel(packet,y,int(stream['labels'][i]))
                if corrupted is not None:
                    for p,v in packet[0].items():scores[p]['corrupt_target_probability']=float(v[corrupted])
                probes.append(dict(checkpoint=name,kind=kind,key_id=i,alias=alias,meta=meta,scores=scores,**(extra or {})))
    for number,row in enumerate(stream['rows'],1):
        for pos,y in enumerate(row['raw']):
            if pos==12 and row['kind']=='one_off_noise':
                before=component_hashes(m);probe('noise_before_'+str(row['key_id']),'noise', [row['key_id']],corrupted=y);probe('noise_others_before_'+str(row['key_id']),'noise_others',stream['untouched'])
                assert before==component_hashes(m)
            packet=read_panel(m,history)  # actual byte not passed to read policy
            scores,meta=score_panel(packet,y)
            category='target' if pos==12 else 'nontarget'
            if meta['tentative']:support_hist[(category,meta['support'])]+=1
            add_event(groups,scores,meta,category,row)
            if arm=='PRED-V':
                # Only scalar pre-update scores retained temporarily for RAND-MATCH.
                events.append(dict(meta=meta,hard=scores['HARD'],shadow=scores['SHADOW'],category=category,
                    row={k:row[k] for k in ('wave','kind','exposure','rendering_exposure')}))
            reference_loss,reference_correct,_=m.observe(history,y)
            hard=scores['HARD' if arm=='PRED-V' else 'EXACT-V']
            if abs(reference_loss-hard['bpb'])>1e-10 or reference_correct!=hard['accuracy']:
                raise AssertionError('HARD prediction did not reproduce V36')
            history.append(y)
            if len(history)>12:del history[0]
            if pos==12:
                if row['kind']=='one_off_noise':
                    before=component_hashes(m);probe('noise_after_'+str(row['key_id']),'noise',[row['key_id']],corrupted=y);probe('noise_others_after_'+str(row['key_id']),'noise_others',stream['untouched'])
                    assert before==component_hashes(m)
                if row['kind']=='acquisition':
                    i=row['key_id'];c0,c1=[context(stream['keys'][i],a) for a in (0,1)]
                    j0,j1=m.members.get(c0),m.members.get(c1)
                    phase=emergence.setdefault(i,set())
                    e=None
                    if 'E1' not in phase and j0 is not None and j1 is None:e='E1'
                    elif 'E2' not in phase and j0 is not None and j0==j1:e='E2'
                    if e:
                        before=component_hashes(m);phase.add(e)
                        probe(e,'emergence',[i],(2,),extra=dict(bytes_observed=(number-1)*14+pos+1,
                            source_slot=j0,source_support=len(m.slot_members[j0]),
                            clean_support_transition=(len(m.slot_members[j0])==(1 if e=='E1' else 2))))
                        assert before==component_hashes(m)
        for cp in cps.get(number,[]):
            before=component_hashes(m);name=cp['name'];k=stream['keys_per_wave'];wi=ord(cp['wave'])-65
            if cp['type']=='acquisition':probe(name,'current',range(wi*k,(wi+1)*k))
            elif cp['type']=='wave':
                probe(name,'untouched_A',stream['untouched']);probe(name,'current',range(wi*k,(wi+1)*k))
                if cp['wave']=='A':probe(name,'all_A',range(k))
                probe(name,'noise_A',stream['noisy']);probe(name,'revised_A',stream['revised'],revised=cp['wave']=='D')
            elif cp['type']=='revision':probe(name,'revision',stream['revised'],revised=True)
            elif cp['type']=='noise':probe(name,'noise_A',stream['noisy'])
            after=component_hashes(m)
            if before!=after:raise AssertionError('Read policy mutated state')
            checks.append(dict(checkpoint=name,components=before,all_policy_hashes_identical=True,read_only=True))
            if cp['type']=='wave':costs.append(dict(checkpoint=name,**m.costs()))
    rand={}
    if arm=='PRED-V':
        chosen,rand['stream']=random_mask(events,seed,3701701)
        for i,e in enumerate(events):
            v=e['hard'] if not e['meta']['tentative'] or i in chosen else e['shadow']
            add_event(groups,{'RAND-MATCH':v},e['meta'],e['category'],e['row'])
        chosen,rand['probes']=random_mask(probes,seed,3701702)
        for i,e in enumerate(probes):
            e['scores']['RAND-MATCH']=e['scores']['HARD'] if not e['meta']['tentative'] or i in chosen else e['scores']['SHADOW']
    address=slot_diagnostics(m,stream)
    # Compact final diagnostics; no per-context or per-byte traces in result files.
    for key in ('binding_trace','members_per_slot','logical_keys_per_slot','slots_per_acquired_logical_key'):address.pop(key,None)
    for key in ('nearest_similarity','top1_top2_margin'):address[key].pop('values',None)
    grouped={p:{g:a.result() for (pp,g),a in groups.items() if pp==p} for p in sorted({p for p,g in groups})}
    return dict(seed=seed,arm=arm,smoke=smoke,stream_sha256=stream['raw_sha256'],groups=grouped,
        probes=probes,checks=checks,random_matching=rand,costs=costs,addressing=address,
        counts=exposure_accounting(stream),final_components=component_hashes(m),
        candidate_support_histogram=[dict(category=c,support=s,count=n) for (c,s),n in sorted(support_hist.items())],
        emergence_coverage=dict(keys=len(stream['keys']),E1=sum('E1' in x for x in emergence.values()),E2=sum('E2' in x for x in emergence.values())),
        elapsed_seconds=time.perf_counter()-start)

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

def qualify(levels,valid):
    v=lambda k:levels[k]['mean']
    qs=dict(Q1=v('A_initial')>=.9 and min(levels['A_initial']['values'])>=.75 and v('D_current')>=.9,
        Q2=v('A_after_D')>=.9,Q3=v('revision_4_alias_0')>=.9 and v('obsolete_8_alias_0')<=.1,
        Q4=v('revision_4_alias_1')>=.7,Q5=v('revision_4_alias_2')>=.5,
        Q6=v('whole_delta_base')<=.1,Q6b=v('nontarget_delta_base')<=.1,
        Q7=v('false_merge_rate')<.05 and v('final_impurity')<.05,Q8=valid)
    return dict(gates=qs,qualified=all(qs.values()))

def report(records,seeds,smoke):
    by={(r['seed'],r['arm']):r for r in records};complete=len(by)==2*len(seeds)
    rows=[];invariance=[]
    for seed in seeds:
        if (seed,'PRED-V') in by and (seed,'EXACT-V') in by:
            a,b=by[seed,'PRED-V'],by[seed,'EXACT-V']
            pairs=zip(a['checks'],b['checks']);ok=len(a['checks'])==len(b['checks'])
            for ca,cb in pairs:
                ok=ok and ca['checkpoint']==cb['checkpoint'] and all(ca['components'][k]==cb['components'][k] for k in ('base','eligibility'))
            if not ok:raise AssertionError('PRED-V/EXACT-V base or eligibility mismatch')
            invariance.append(dict(seed=seed,base_and_eligibility_identical=True,policy_state_shared=True,checkpoints=len(a['checks'])))
    for r in records:
        for p in r['groups']:
            if p=='BASE':continue
            rows.append(dict(seed=r['seed'],policy=p,trajectory=r['arm'],**policy_metrics(r,p)))
    levels={};qualification={};hyp={};contrasts={}
    if complete:
        indexed={(r['seed'],r['policy']):r for r in rows}
        names=POLICIES+('RAND-MATCH','EXACT-V')
        for p in names:
            levels[p]={k:stat([indexed[s,p][k] for s in seeds]) for k in rows[0] if k not in ('seed','policy','trajectory')}
            if p!='RAND-MATCH':qualification[p]=qualify(levels[p],len(invariance)==len(seeds))
        def diff(a,b,k):return stat([indexed[s,a][k]-indexed[s,b][k] for s in seeds])
        for a,b in [('SHADOW','HARD'),('CORR2','HARD'),('CORR2','RAND-MATCH'),('CORR2','EXACT-V'),('SIM-MIX','EXACT-V')]:
            contrasts[a+' - '+b]={k:diff(a,b,k) for k in ('whole_bpb','nontarget_bpb','target_bpb','A_after_D','revision_4_alias_2')}
        hard=levels['HARD'];p1=all(hard[k]['mean']>=threshold for k,threshold in [('A_after_D',.9),('D_current',.9),('revision_4_alias_1',.7),('revision_4_alias_2',.5)])
        shadow=contrasts['SHADOW - HARD'];rand=contrasts['CORR2 - RAND-MATCH']
        negative=lambda x:x['high'] is not None and x['high']<0
        positive=lambda x:x['low'] is not None and x['low']>0
        hyp=dict(P1=dict(target_replication=p1,hard_delta_base=hard['whole_delta_base']),
            P2=dict(supported=negative(shadow['nontarget_bpb']) and negative(shadow['whole_bpb'])),
            P3=qualification['CORR2'],
            P4=dict(supported=((rand['target_bpb']['mean']<=-.1 and negative(rand['target_bpb'])) or (rand['revision_4_alias_2']['mean']>=.1 and positive(rand['revision_4_alias_2']))) and rand['nontarget_bpb']['mean']<=0,
                note=IMPLEMENTATION_NOTES['P4']),
            P5=dict(grid_Q5_Q6_pass=[p for p in ALPHAS if qualification[p]['gates']['Q5'] and qualification[p]['gates']['Q6']],diagnostic_only=True))
        # Qualification must not survive failure to replicate its inherited mechanism.
        for p,q in qualification.items():q['round_valid']=p1;q['qualified']=q['qualified'] and p1 and not smoke
    return dict(version=VERSION,status='complete' if complete else 'partial',scientific=not smoke,
        completed_trajectories=len(records),expected_trajectories=2*len(seeds),seeds=list(seeds),
        levels=levels,qualification=qualification,hypotheses=hyp,paired_contrasts=contrasts,
        invariance=invariance,per_seed=rows,implementation_notes=IMPLEMENTATION_NOTES,
        roles=dict(stateful=('PRED-V','EXACT-V'),deployable=POLICIES+('EXACT-V',),nondeployable=('RAND-MATCH',),base='Counterfactual only'),
        limitations=['Unbounded storage','One-off corruption intentionally uncorrected','Engineering smoke outcomes do not qualify science'])

def summary_text(rep):
    lines=['# V37 — Corroborated Retrieval Authority','',f"Status: {rep['status']}; {rep['completed_trajectories']}/{rep['expected_trajectories']} stateful trajectories.",
        'Scientific run.' if rep['scientific'] else 'ENGINEERING SMOKE ONLY — not scientific evidence.','',
        '| Policy | A acquired | A after D | D current | Known @4 | Heldout @4 | Whole bpb | Delta base | Qualified |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for p,lev in rep['levels'].items():
        vals=[f"{100*lev[k]['mean']:.2f}%" for k in ('A_initial','A_after_D','D_current','revision_4_alias_1','revision_4_alias_2')]
        vals += [f"{lev[k]['mean']:.4f}" for k in ('whole_bpb','whole_delta_base')]
        lines.append('| '+p+' | '+' | '.join(vals)+' | '+str(rep['qualification'].get(p,{}).get('qualified','diagnostic'))+' |')
    lines+=['','All read policies share one PRED-V state. EXACT-V is a separate matched reference.',
        'RAND-MATCH is an offline, count-matched evaluator control, never a candidate.',
        'Student-t intervals and absolute levels are in REPORT.json. No automatic follow-up or post-hoc selection.',
        'Storage remains unbounded; noise and capacity are unresolved.',
        'Emergence rows include actual slot support and a clean-transition flag; missing transitions are not imputed.','']
    return '\n'.join(lines)


def save_outputs(root,records,rep,identity,source):
    write_json(root/'REPORT.json',rep);atomic(root/'SUMMARY.md',summary_text(rep))
    csv_output(root/'SUMMARY.csv',rep['per_seed']);csv_output(root/'AUTHORITY_FRONTIER.csv',rep['per_seed'])
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
                    candidate_support=e['meta']['support'],tentative=e['meta']['tentative'],candidate_similarity=e['meta']['similarity'],**v)
                if e['kind']=='emergence':
                    out.update({k:e[k] for k in ('source_slot','source_support','clean_support_transition','bytes_observed')});emergence.append(out)
                elif e['kind']=='revision':revision.append(out)
                elif e['kind'].startswith('noise'):noise.append(out)
                elif e['checkpoint'].startswith('after_'):retention.append(out)
                else:acquisition.append(out)
        costs.append(dict(seed=r['seed'],trajectory=r['arm'],data=r['costs']))
        states.append(dict(seed=r['seed'],trajectory=r['arm'],checks=r['checks'],final_components=r['final_components'],random_matching=r['random_matching'],emergence_coverage=r['emergence_coverage']))
    for name,rows in [('AUTHORITY_BY_SUPPORT',support),('NONTARGET_AUTHORITY',usage),('RETENTION_BY_WAVE',retention),('REVISION_CURVES',revision),('HELDOUT_CORROBORATION_CURVE',emergence),('ACQUISITION_CURVES',acquisition)]:csv_output(root/(name+'.csv'),rows)
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
    paths=[p for p in root.rglob('*') if p.is_file() and not p.name.startswith('.') and '.tmp-' not in p.name and p.name!='ARTIFACT_HASHES.json']
    write_json(root/'ARTIFACT_HASHES.json',{str(p.relative_to(root)):sha(p.read_bytes()) for p in paths})
    archive=Path(str(root)+'.zip');temp=Path(str(archive)+'.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths+[root/'ARTIFACT_HASHES.json']:z.write(p,p.relative_to(root))
    os.replace(temp,archive)
    return archive


def self_test(full=False):
    """Engineering-only independent replay checks; never consumes scientific seeds."""
    tests=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        tests.append(name)
    m=ByteModel(ENGINEERING_SEEDS[0],'PRED-V');c=context(b'ABCDEF',0);ca=c[:-1]+b';';held=c[:-1]+b'8'
    m.eligible.add(c);j=m.bind(c,None);v=np.zeros(256);v[97]=.8;m.values[j]=v.tobytes()
    before=component_hashes(m);probs,w,meta=read_panel(m,held)
    check('support one candidate',meta['tentative'] and meta['support']==1)
    check('CORR2 suppresses unsupported read',w['CORR2']==0 and np.array_equal(probs['CORR2'],probs['BASE']))
    check('probability mixture formula',np.allclose(probs['ALPHA-0.25'],.75*probs['BASE']+.25*probs['HARD']))
    check('fixed endpoints',np.array_equal(probs['ALPHA-0'],probs['SHADOW']) and np.array_equal(probs['ALPHA-1'],probs['HARD']))
    for _ in range(3):read_panel(m,held)
    check('probe cannot corroborate',before==component_hashes(m))
    m.eligible.add(ca);m.bind(ca,j);probs,w,meta=read_panel(m,held)
    check('support two grants authority',meta['support']==2 and w['CORR2']==1 and np.array_equal(probs['CORR2'],probs['HARD']))
    check('exact ownership all policies',all(a==1 for p,a in read_panel(m,c)[1].items() if p!='BASE'))
    # Matching must not privilege targets or alter frozen pre-observation predictions.
    examples=[dict(meta=dict(tentative=i<8,support=2 if i%3==0 else 1)) for i in range(10)]
    chosen,audit=random_mask(examples,1,2)
    check('random exact count',len(chosen)==3 and all(i<8 for i in chosen))
    rr=np.random.default_rng(17);a=ByteModel(ENGINEERING_SEEDS[0],'PRED-V');b=ByteModel(ENGINEERING_SEEDS[0],'PRED-V')
    h=bytearray()
    for y in rr.integers(0,256,64):
        packet=read_panel(a,h);a.observe(h,int(y));b.observe(h,int(y));h.append(int(y))
        if len(h)>12:del h[0]
    check('panel observer independent random byte replay',component_hashes(a)==component_hashes(b))
    runs=[]
    for arm in MAIN_ARMS:
        r=trial(ENGINEERING_SEEDS[0],arm,not full)
        m=ByteModel(ENGINEERING_SEEDS[0],arm);h=bytearray();total=0.;target=0.;non=0.
        stream=make_stream(ENGINEERING_SEEDS[0],smoke=not full)
        for row in stream['rows']:
            for pos,y in enumerate(row['raw']):
                l,_,_=m.observe(h,y);total+=l
                if pos==12:target+=l
                else:non+=l
                h.append(y)
                if len(h)>12:del h[0]
        check(arm+' independent full state replay',r['final_components']==component_hashes(m))
        policy='HARD' if arm=='PRED-V' else arm;g=r['groups'][policy]
        check(arm+' historical losses',abs(g['whole']['bpb']-total/len(stream['raw']))<1e-11 and abs(g['target']['bpb']-target/len(stream['rows']))<1e-11 and abs(g['nontarget']['bpb']-non/(13*len(stream['rows'])))<1e-11)
        runs.append(r)
    rep=report(runs,ENGINEERING_SEEDS[:1],not full)
    check('paired base eligibility across arms',len(rep['invariance'])==1)
    pred=next(r for r in runs if r['arm']=='PRED-V')
    for names in [('HARD','ALPHA-1'),('SHADOW','ALPHA-0')]:
        check('policy endpoint identity '+str(names),pred['groups'][names[0]]==pred['groups'][names[1]])
    # Exact members and no-hit distributions must agree across every deployable policy.
    check('exact read policy identity in probes',all(len({e['scores'][p]['target_probability'] for p in POLICIES})==1 for e in pred['probes'] if not e['meta']['tentative']))
    synthetic={k:dict(mean=1.,values=[1.,1.]) for k in ('A_initial','D_current','A_after_D','revision_4_alias_0','revision_4_alias_1','revision_4_alias_2')}
    synthetic.update({k:dict(mean=0.,values=[0.,0.]) for k in ('obsolete_8_alias_0','whole_delta_base','nontarget_delta_base','false_merge_rate','final_impurity')})
    check('qualification accepts all required gates',qualify(synthetic,True)['qualified'])
    for field,bad in [('A_initial',.89),('D_current',.89),('A_after_D',.89),('revision_4_alias_0',.89),('revision_4_alias_1',.69),('revision_4_alias_2',.49),('obsolete_8_alias_0',.11),('whole_delta_base',.11),('nontarget_delta_base',.11),('false_merge_rate',.05),('final_impurity',.05)]:
        altered=copy.deepcopy(synthetic);altered[field]['mean']=bad
        check('qualification rejects '+field,not qualify(altered,True)['qualified'])
    altered=copy.deepcopy(synthetic);altered['A_initial']['values']=[.74,1.]
    check('qualification rejects bad acquisition seed',not qualify(altered,True)['qualified'])
    check('qualification rejects failed invariance',not qualify(synthetic,False)['qualified'])
    return dict(passed=True,checks=tests,full_length=full,engineering_seed=ENGINEERING_SEEDS[0],
        independent_replay_components={r['arm']:r['final_components'] for r in runs})


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
        root=(args.out_dir or Path('v37_smoke' if args.smoke else 'v37_results')).resolve();root.mkdir(parents=True,exist_ok=True)
        source=Path(__file__).read_bytes();seeds=ENGINEERING_SEEDS if args.smoke else SEEDS
        plan=dict(seeds=list(seeds),arms=list(MAIN_ARMS),policies=list(POLICIES),smoke=args.smoke,
            streams={str(s):make_stream(s,smoke=args.smoke)['raw_sha256'] for s in seeds},
            implementation_notes=IMPLEMENTATION_NOTES,resume='Completed trajectories only; interrupted trajectory deterministically restarts')
        identity=dict(source_sha256=sha(source),protocol_sha256=sha(PROTOCOL.encode()),plan_sha256=sha(plan))
        manifest=dict(version=VERSION,identity=identity,plan=plan,reference_v36_sha256=REFERENCE_SHA256,
            environment=dict(python=sys.version,numpy=np.__version__,scipy=importlib.import_module('scipy').__version__,platform=platform.platform(),blas_threads=1))
        with RunLock(root):
            path=root/'RUN_MANIFEST.json'
            if path.exists():
                if json.loads(path.read_text())!=manifest:raise RuntimeError('Resume rejected: source, protocol, plan or environment changed')
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

if __name__=='__main__':main()
