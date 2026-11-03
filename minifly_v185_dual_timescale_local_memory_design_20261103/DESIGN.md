# MiniFly V185 — Compact dual-timescale, conflict-aware local memory

**Identity:** V185-DTLM-D1  
**Date:** 2026-11-03 (America/New_York)  
**Status:** \`DESIGN_ONLY\`; no new model, numerical qualification, scientific record, continuation, adopted core or official checker result.  
**Parent / research anchor:** \`ChangweiZhou/digitalbaby\`, \`codex/minifly-consolidated-20261023@9203640ce5b65907fef41c893e456c999d23c66e\`. V184-R1 rejected the exact H-to-output residual package; see \`minifly_v184_h_residual_deployment_dev_20261102/campaign/DECISION.json\`, \`SUMMARY.json\`, \`GENERATION_COMPACT.json\` and \`HRES/A2_ALIGNMENT_COMPACT.json\`.  
**Owner instruction:** “The V184 is done. Activate the skill and iterate.” Design/registration only; **not** authority to run training, change old budgets/gates, import a different adopted core or start confirmation.

## 1. Research decision: stop another prose-credit tweak

V181–V184 are an actual 37,313-weight compact, local, 257-way predictive lineage, **not** proof of generic persistent learning. The latest V184 on opened M1 gave A2-held 4.582233 bits (NORMAL), 4.612251 (norm sham), 4.634836 (H residual). HRES-minus-NORMAL +0.052604 bits, paired-record CI [0.040242,0.064919]. Exchanging its H residual across position-matched held records *improved* loss (aligned 4.634332, exchanged 4.618981). HRES failed conditional/retention/EOS guards and produced only 1 own EOS across 96 continuations. Its source ran as registered; rejection is not a pretext for a fifth alpha/hidden-credit experiment.

A separate historically **real** native-memory lineage offers a more relevant mechanism target. Claude's V172 \`DELTA-H\` learned 50/64 taught OLD full answers but retained only 23/50 after learning 16 NEW; the benchmark \`N2B\` achieved only 10/80 final (same development world). Claude V174's equal-slot REPLAY preserved 32/41, 35/47 and 21/33 formerly correct OLD, versus BLOCKED 19/41, 19/47 and 7/33, but total net gain was only +2/+2/+7, not the prespecified +8. Those are **opened development results from a different 16-bank native memory**, **not** a validated mini-model and **not** directly comparable NLL to V184. They justify investigating interference-aware *local synaptic stabilization*, not importing an uncounted 16-bank model.

**Question:** In a *genuinely compact* learner that has acquired own complete answers, does making slow synaptic writes selective against contradictory prior synaptic evidence preserve still-valid knowledge through NEW learning **without** preventing NEW acquisition or eventual legitimate revision?

Two mutually exclusive testable outcomes:
- **Localized metaplasticity helps:** a conflict-aware slow writer beats both a full-write writer and an equal-total-slow-dose but unselective writer on previously acquired OLD retention, while keeping new acquisition and revision; retain the exact architecture for a fresh-world review.
- **No attributable benefit:** it fails one of the explicit acquisition, retention, dose-control, new-learning, EOS or revision guards; reject this exact V185 mechanism. A failed sensor/readout **feasibility** gate means \`NOT_TESTABLE_AS_SPECIFIED\`, **not** a biological negative.

The revised direction is a *mechanism experiment*, not a fitted-probe experiment, a larger learner, or a promise that E3 is close. One high-quality negative with a proper positive acquisition reference is more useful than running another Gutenberg output shortcut.

## 2. New operative learner: one fixed sensory circuit, two small writable timescales

V185 is **independent of V181's learned H/T network**, not a continuation of its checkpoints. Its support uses the exact original fixed native FE-PERM→KC sensory machinery as a **single** 88-receptor, 5,177-KC birth, with an explicitly counted fixed feature projection. It does not instantiate Full151's sixteen value banks, a pretrained encoder, an evaluator head, a second teacher, a Transformer, attention, supplied operand/digit/carry features or an externally controlled EOS.

**Fixed, generic pre-readout:** reuse the V181 \`I_{jr}\` incidence construction (\`j=0,…,63\`, \`r=0,…,255\`, deterministic 5,177 permutation). On each incidence edge draw a **fixed** sign \(\sigma_{jr}\in\{-1,+1\}\) from \`SHA256('V185|projection-sign|j|r')\` using its low bit; do not fit/select signs from data. For the 5,177-component native **binary** KC code \(x\),

\[
h_j(x)=\tanh\!\left(\frac{\sum_{r=0}^{255}\sigma_{jr}\,x_{I_{jr}}}{3}\right)\quad (j=0,\ldots,63).
\]

These 64 features are part of the real learner, **not** a supplied task parser or a fitted H representation. They may lose information; C0 **must** establish their empirical representational sufficiency for the locked finite diagnostic before treating any writer result as interpretable. Count every fixed sign, index, receptor and native code array in the actual model memory.

**Writable state and output:** two float32 local output-synapse arrays \(\ F,S\in\mathbb{R}^{257\times64}\) (fast and slow) and float32 \(b\in\mathbb{R}^{257}\) (class bias), initialized exactly zero, are the only learned arrays. Let

\[
\ell_c(x)=\sum_{j=0}^{63}(F_{cj}+S_{cj})h_j(x)+b_c,\qquad
p_c=\operatorname{softmax}_{257}(\ell)_c.
\]

Real EOS is class 256. The model predicts **before** it receives the observed outcome; only then may the target \`y\` drive a synapse-local update. In generation it emits its **own** argmax bytes until it independently emits EOS or reaches the disclosed cap; no teacher intervention or forced EOS at cap.

Exactly \(2(257\cdot64)+257=\mathbf{33,153}\) trainable scalars, including both learned timescales and biases, versus the previous compact 37,313. Fixed sensor arrays, fast/slow mutable weights and per-life clocks are physical memory; no optimizer moments exist for this SGD-like rule. Any additional learned variable requires re-design and a new resource census. This proposal makes **no runtime memory assertion** before actual measurement; the frozen model-specific **8 MiB inference**, **1 MiB full checkpoint** and **40,000 learned-scalar** gates apply.

## 3. One mechanism and matched controls

At each genuine observed byte/EOS target \`y\`, first evaluate \`p\` and the output features \`h\`. Define the ordinary output-local error-corrective write direction (the only teacher):

\[
d_{cj}=\frac{(\mathbf1_{c=y}-p_c)h_j}{1+\sum_{k=0}^{63}h_k^2},
\qquad e_c=\mathbf1_{c=y}-p_c.
\]

All updates use the **pre-update** \(F,S,b,h,p\); no gradient crosses a learned hidden layer or enters preceding time, and no hidden prompt/label class may select a rule. Fixed coefficients for all arms: \(\eta_F=0.25,\eta_S=0.25,\eta_b=0.03,\lambda_F=1/1024\) per **teaching** target (no decay from a read-only panel). First apply the same proposed fast/bias rules to every arm:

\[
F^{+}=(1-\lambda_F)F+\eta_F d,\qquad b^{+}=b+\eta_b e.
\]

The slow writer differs **only** by a prefeedback-state, postsynaptic target-error gate. Let the candidate's synapse-wise conflict and release flags be

\[
c_{cj}=\mathbf1_{S_{cj}d_{cj}<0},\qquad
r_{cj}=\mathbf1_{F_{cj}S_{cj}<0\ \wedge\ |F_{cj}|>2|S_{cj}|}.
\]

The **release** condition permits a sufficiently strong, persistently contradictory fast trace to revise the slow trace, rather than freezing an obsolete outcome forever. With fixed \(\gamma=0.10\), define

\[
g_{cj}=\begin{cases}
1,&c_{cj}=0\ \text{or}\ r_{cj}=1\\
0.10,&c_{cj}=1\ \text{and}\ r_{cj}=0.
\end{cases}
\]
\[
S^{+}_{cj}=S_{cj}+\eta_S\,a_{cj}\,d_{cj}.
\]

The three primary scientific arms run as separate complete, uninterrupted learner states from *the exact same born weights, sensor and example order*:

| Arm | Slow multiplier \(a_{cj}\) | Scientific role |
|---|---|---|
| **FULL** | \(1\) | Ungated continuous local writing |
| **UNIFORM** | \(u=\frac{\sum_{cj}|d_{cj}|g_{cj}}{\sum_{cj}|d_{cj}|}\), with \(u=1\) if denominator zero | **Same proposed slow-write L1 per event at its own current state**, but no synapse-selective placement |
| **SELECTIVE** | \(g_{cj}\) | Local conflict-and-release metaplasticity |

\`UNIFORM\` computes \(g\) from **its own** pre-update state after the true target is revealed and uses only the global scalar \(u\); it does **not** borrow the other arm's data, states or private labels. \(\sum|\eta_Sud|=\sum|\eta_Sgd|\) at the same state is exact algebra, **not** a claim of cumulative matched writes after histories diverge. This control distinguishes spatial selectivity from generic reduction of slow learning. Count actual per-branch fast/slow proposed and realized L1/L2, signed reversals, gate fraction, release fraction and EOS-class writes. The selected fraction or \(\gamma\) is **never** adapted to held results; no dose grid.

The mechanism is generic across all 257 classes and all experienced raw contexts. It knows neither A/B/C phase nor which facts are “OLD,” “NEW,” or “corrected.” Such names belong only to the experiment generator and reducer.

## 4. Locked first scientific question: true E1 with interference and legitimate correction

**Scope:** A new, disclosed **development-only** deterministic byte-key→two-byte-value ecology, selected *before implementation* by \`SHA256('V185|DEV|20261103')\`; keys and answers are independently generated with labeled deterministic streams. No human-supplied digit, operand or positional feature enters the model, and no evaluator-fitted readout supplies its output. This is a deliberately narrow *associative* memory test: the exact-key lookup ceiling is 100%, but **success cannot show reuse of a never-taught relation (E3)**. Do not dress an E1 result up as an arithmetic rule.

- **A / OLD:** 64 unique raw ASCII key prompts (6–10 nonsemantically generated bytes plus a common terminator); each has a frozen independently assigned exactly **two-byte** ASCII answer and a real EOS. Present all 64 in 16 fixed hash-seeded chronological passes (1,024 physical full-record presentations), no best-checkpoint selection.
- **B / NEW:** 16 distinct unique prompts/answers, not equivalent byte renderings of an OLD prompt. Present 8 passes (128 records) continuing the **same** mutable model state and clock. OLD correct before B and retained after B are scored by **complete own two-byte response plus own EOS**, with immediate no-feedback probes; B acquisition is scored the same way.
- **C / EDIT vs NOEDIT:** from the **identical B8 full state** within each arm, two genuine continuations: 8 predetermined OLD keys receive **new two-byte values** (different from the former value, generated by a locked salt independent of model performance) for 4 passes; NOEDIT sees the same 8 keys with the original values for 4 passes. This is a legitimate update to an **arbitrary symbolic mapping**; unlike arithmetic \`a+b\`, it does not make a true mathematical sum “change.” The other 72 previously learned key/value relations remain still-valid. The model receives the ordinary prompt and observed answer, **not** an edit bit, stage ID or explicit protection list. Score corrected new value, obsolete old value, same-key stable values and B retention separately.

Per arm: **1,024 + 128 + 32 + 32 = 1,216** optimizer-record applications (the C/NOEDIT children are correlated). Across three arms: **3,648** records / **10,944** true byte/EOS training targets (each response always has exactly two payload bytes and one EOS). Add three separate B-only-from-birth reference lives at 128 records apiece (**384** records / **1,152** targets) to identify prior-A help versus interference. **Grand scientific total = 4,032 optimizer-record presentations / 12,096 targets.** Reserve at most 256 further physical technical training records, up to 768 target exposures; **4,288 total presentations ceiling** for this plan, within the binding 5,000 cap. Every read-only site, forward probe, generated byte, failed fixture, CPU second and full copy is reported separately.

Required full states per primary arm: birth, A8, A16, B4, B8, EDIT4, NOEDIT4 (**21** complete checkpoints); plus three B-only birth/B8 reference states (**6** checkpoints). Full state includes fast+slow matrices, b, time/clock, any actual RNG and the frozen projection/sensor digest. All A/B/C panels are read-only clones that cannot alter the learner or its sensory adaptation. Key/answer histories are *experimental observations*, **not** a hidden direct-answer lookup at inference. No external replay or teacher labels are read by the candidate at inference.

All evaluation records are generated and **source-locked before any C2 neural exposure**. Freeze the precise byte alphabet, disjoint-key determinism, salts, response length, all pass orders, correction IDs, source and runtime hashes, pre-feedback timing and full old/new equivalence classes in an implementation spec prior to technical qualification. Do not invent a new seed or edit selection after viewing learner responses.

## 5. C0/C1: cheap falsifiers, source integrity and mini admission

**Data/input capability before fit:**
1. Verify exact native \`88→5,177\` birth digest and authentic FE-PERM code parity with the single-source parent. Verify no extra native value bank, no learned target-specific feature and no input/prompt keyed answer cache. Key prompts/answers must be unique where specified; correction must be well-formed and preserve all other labels.
2. Confirm the **fixed 64-feature output code** distinguishes all staged answer sites sufficiently. An external, frozen read-only separability/oracle test may fit a readout **only as a C0 witness**, never a candidate or future evaluation answerer. Require a full deterministic **80/80** exact two-byte+EOS training ceiling before calling the writer comparison qualified. If not, \`REPRESENTATION_CEILING_FAILED\`; do not enlarge H, change projections, select salts, or count the resulting failure as negative evidence about metaplasticity. The plain exact-key table oracle is a *shortcut* and reported as such.
3. At source birth verify true 257-class logits, distinct per-record target sites, the actual EOS prediction, local output-gradient sign and numerical first-order loss decrease for tiny step; check \(\sum |ud|=\sum |gd|\) at the **same state**, including d=0, zero-S, conflict, fast-release and opposite signs. Check \(\gamma=0.1\) and the absence of stage labels in candidate calls.
4. Assert all three main arms start with exact same born model, example order and raw observation interfaces. Verify complete state roundtrip, deterministic equality on restored continuations and EDIT/NOEDIT identical fork ancestors; no hidden optimizer-state resets.
5. Profile all arms and forks using the actual complete roster and outputs **before** scientific fit. Binding hard constraints: ≤40,000 learned scalars; ≤8 MiB single deployed inference-state bytes incl native sensor, fixed projection and *both* writable banks; ≤1 MiB **complete** mutable checkpoint; ≤5,000 physical training/qualification record presentations; ≤600,000 supervised target exposures; ≤900 aggregate CPU seconds; ≤1,800 execution-wall seconds; ≤1 GiB process RSS; ≤256 MiB complete outputs *including archive, traces, review copies, runtime and reserve*. Any measurement outside limits yields \`RESOURCE_NOT_ADMITTED\`, with no silent compression/omission or expanded cap.

These C0 conditions are mandatory **engineering feasibility**, not a claimed scientific success. The historical formal MRK-1 checker is not installed; no phantom check or seal is permitted.

## 6. Predeclared learning gates, negative outcomes and interpretation

Primary is the **entire final A→B→C life**, not an intermediate “best OLD” snapshot. All outcomes are complete own bytes plus independently emitted EOS; teacher-forced site NLL, top1, EOS NLL/Brier and length/truncation are companion diagnostics only.

- **E1 acquisition qualification:** at A16, SELECTIVE and both controls must each correctly generate at least **40/64** OLD two-byte+EOS answers; otherwise report \`ACQUISITION_REFERENCE_UNQUALIFIED\` and all later outcomes. Do not stop training after observing this failure, replace the data, select a best checkpoint or call an apparent retention fraction proof of prevention.
- **Selective retention:** for each arm, report OLD items correct *both* at A16 and B8, divided by its actual A16-correct set, plus absolute retained counts and paired item IDs. Require SELECTIVE's **absolute retained OLD count** at B8 to exceed both FULL and UNIFORM by **≥8 items**, while its A16 acquired OLD count is no more than 2 below either control. A high retention *percentage* from acquiring very few facts is not enough.
- **New learning / net lifetime:** SELECTIVE's B8 NEW complete-correct score must be within **2 items** of the better control. Require SELECTIVE's B8 total correctly recalled among the 80 taught items to exceed **both** controls by at least **8**, and also exceed its own A16 OLD-correct count by at least **8** (new learning must more than pay for forgetting). Report B-only controls but do not use them to patch a failed gate.
- **Legitimate C correction:** after EDIT4, on the 8 changed keys the model must produce the **new** two-byte+EOS answer for at least **6/8**, and its new-vs-obsolete log-probability odds must increase relative to NOEDIT4 on those same 8 keys. Candidate must preserve at least **90% of previously correct still-valid** OLD/NEW keys in EDIT4 relative to B8, and report same controls in NOEDIT4. Obsolete answers are **never** counted as protected competence. Repeated conflicting writes should eventually unlock the slow trace; a protected-but-uncorrectable memory does not pass.
- **EOS / source:** report true EOS and every truncated/empty/incorrect output. No forced termination, substring/string-repair match, role/digit parser, pretrained head, best checkpoint, evaluator-readout replacement or uncharged replay can satisfy a success. Complete-correct count is **zero** if EOS is absent.

A single development ecology supports \`BOUNDED_DEV_POSITIVE\` at most; item bootstrap CIs may be shown but 64 correlated prompts are **not** independent worlds, and no threshold is certified generalization. **Reject the exact package** if acquisition passes but the retention/control/correction/own-EOS vector fails. If acquisition or representational reference fails, do **not** infer a biological negative for the slow-gate mechanism; emit the honest narrower failure, stop, and rethink architecture. A positive gets only \`RETAIN_EXACT_PACKAGE_FOR_PROSPECTIVE_REVIEW\`, **not** adoption or E3.

**Not tested or claimed:** arithmetic abstraction, never-taught addition, causal E3 relation transfer, broad learning across real prose, biological mechanism validation, sealed confirmation or an adopted core. If this compact causal-memory experiment works, a later separately approved E3 design must use real held relation/orbit controls, positive no-prior teaching and exact own EOS; a table memorizer is only an E1 component.

## 7. Research-loop closure and authorization

\`\`\`text
Anchor: ChangweiZhou/digitalbaby codex/minifly-consolidated-20261023@9203640ce5b65907fef41c893e456c999d23c66e; latest completed V184-R1.
Measured incident: HRES 4.634836 vs NORMAL 4.582233 held bits; aligned
                   H feature 4.634332 vs exchanged 4.618981; own EOS 1/96.
Mechanistic target: preserve competence acquired from legitimately observed
                   byte/answer feedback while continuing NEW and correcting C.
Opposing explanations: synapse-local conflict protection versus mere globally
                       weaker slow writing; FULL/UNIFORM/SELECTIVE test them.
Run in this turn: GitHub source/skill inspection and DESIGN ONLY.
Scientific training in V185: 0. Local numeric qualification: 0.
Data exposure in V185: 0 (prospective seed only).
Disposition now: V185-DTLM-D1 proposal; NOT_RUN, no adopted core.
Next decision: owner-authorized C0 implementation/technical source & physical
               census, then separate launch of a frozen scientific life.
\`\`\`

The five integrated MiniFly skills are **guidance**, not an executable MRK-1 check: the older dispatcher and required workers are missing; independent cross-family review remains pending. This design does **not** overwrite V172/V174/V181–V184 reports or activate a historical KB finding.
