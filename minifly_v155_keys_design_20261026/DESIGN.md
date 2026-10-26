V155 DESIGN (proposal; no code; nothing run under it)

# V155 design: can MiniFly learn its own keys, without back-propagation? (26 October 2026)

**Status:** a design for the owner's review. It needs an owner grant (stages and caps, §9) before any code. It follows the research-loop stages: S1 lock, C0 testbed, C1 technical, C2 dev, C3 rehearsal, and S6 official only if granted.

**Basis:**
* the headroom check: `minifly_v155_headroom_check_20261026/REPORT.md`, verdict ROOM_BUT_NOT_COUNTING;
* V153: `minifly_v153_bits_dev_v1_20261025/REPORT.md`;
* the literature review: `minifly_literature_review_20261025/REVIEW.md`, §5–6;
* the S0 erratum's three checks: `minifly_v148_explore_s0_20261025/EXPLORE.md`.

## 1. Why this question, and why now
**What the evidence says** (development results, *Thursday*, 40 KB taught):
* **The output side cannot reach the next gain.**
  * Every readout MiniFly can use is linear in its KC code.
  * The best such readout of the frozen random code scores 2.963 bits/symbol, which is PPM-2 level.
  * The class-bias recommendation after V153 would at best approach that number. It is withdrawn: it could not exceed counting and learns nothing that carries over.
* **The representation side has room.**
  * Keys learned by back-propagation, reading the same receptor trace, reach 2.699: 0.26 better than the frozen keys (line-bootstrap SD 0.016).
  * That reference is dense. MiniFly's keys are sparse: each of 5,177 KCs is wired to about 5 of 302 projection neurons. How much of the 0.26 fits the native wiring is unknown.
* **Counting is not beatable at 40 KB by learners of this kind.**
  * Even given the last four symbols explicitly, the back-propagation learner only ties PPM-4 (+0.02 ± 0.02).
  * "Beat the best counting rung on the same text at 40 KB" is therefore the wrong yardstick at this data size. §3 replaces it.
* **Nothing in MiniFly today learns anything that could carry over.**
  * Keys and front end are fixed at birth. Only the KC-to-compartment weights learn, and they store "this context → this next symbol": a smoothed counting table.
  * Pre-training that carries over to later language tasks needs a learned representation. The keys are the place for it, and the literature review found that learning them locally is plausible: CLAPP-type predictive rules, and FlyVec on text.

**The question:** can a local, label-free rule learn MiniFly's input-to-KC keys so that
1. the learned KC code predicts held-out text better than the frozen random code does, and
2. keys learned on one text help on a different text when little of the second text is available?

This is the adopted step 2 ("predictive (CLAPP-type) against competitive Hebbian learning of the input-to-KC connections"), sharpened by the check.

## 2. What stays fixed, and what changes
**Fixed in every arm:**
* the front end: V153's cell P81 (scrambling on, fade 0.807 per byte), read at the native time;
* the projection-neuron mapping;
* the 5,177 KCs, with top-92 per side active (184);
* thresholds at zero (homeostasis off, as in V149/V153);
* the class definition;
* the splits;
* the evaluators.

**The only thing that changes between arms** is the PN-to-KC weight matrix B (302 × 5,177).

| Arm | Keys | Learning |
|---|---|---|
| FROZEN | native birth weights | none (= the check's T2) |
| L-COMP | native wiring, weights learned | competitive Hebbian rule (§4.1) |
| L-PRED | native wiring, weights learned | predictive three-factor rule (§4.2) |
| REF-N | native wiring, weights learned | back-propagation on next-symbol loss: reference only |
| REF-D | dense, all 302 × 5,177 weights learned | back-propagation on next-symbol loss: reference only |

**Rules for the arms:**
* The local arms keep the native wiring: weights change only on existing synapses.
* Weights stay non-negative, as excitatory PN→KC synapses should.
* Each KC's total input weight is rescaled after every update to its birth value (synaptic scaling). Without that, the per-side top-k competition would be decided by weight growth rather than by what each KC encodes. The same rescaling applies to both local arms and to REF-N.

**Back-propagation never feeds MiniFly.** REF-N and REF-D only measure how much a learned key matrix could gain under each wiring. Their training uses a differentiable stand-in for top-k:
* activity = ReLU(pre-activation − the per-side 92nd value), with the 92nd value held constant in the backward pass;
* a linear readout on top.

Every arm, references included, is then scored the same way: R1 on the **binary** top-k code its keys produce. All numbers are therefore comparable with the frozen 2.963.

## 3. Yardsticks (replacing "beat counting at 40 KB")
Two measures, both decided before any outcome:

1. **Same-text capture.** On the taught text's held-out lines, how much of the native-wiring headroom a local rule captures:

   capture = (FROZEN − arm) / (FROZEN − REF-N), in R1 held-out bits.

   It is reported with a line-bootstrap SD. It is meaningful only if REF-N's headroom passes gate G0 (§5).

2. **Carry-over.**
   * Keys learned on text A are frozen.
   * A fresh readout (R1, and the local ONLINE readout) is trained on only **10 KB of text B**, a different author.
   * It is scored on B's held-out lines.

   carry-over = FROZEN_B − arm_B (bits; positive means the learned keys help on B).

   Counting baselines are reported beside it, not used as pass marks:
   * PPM-4 on B's 10 KB alone;
   * PPM-4 on A's 40 KB followed by B's 10 KB ("pre-trained counting").

**Small data is kept.**
* Text A stays at 40 KB.
* The local rules may take several passes; every pass is counted as learner records.
* "More text" is the alternative yardstick for the owner (§9, D1). This design does not assume it.

**Checking the yardsticks against the erratum:**

| Check | How V155 meets it |
|---|---|
| Cost against vocabulary | The keys sit on the input side. Their size (302 × 5,177) does not grow with the output vocabulary. |
| Sharing | A learned code lets contexts with similar futures (L-PRED) or similar inputs (L-COMP) share KCs, and the carry-over yardstick measures sharing directly. |
| Can it exceed counting? | Not at 40 KB on the same text (§1). The carry-over test is where a learned representation can beat counting, because counting cannot reuse structure. That is stated as an open question, not a promise. |

## 4. The two local rules
Notation:
* q_t: the 302-dimensional PN input at position t;
* x_t: the binary KC code (top-92 per side of Bᵀq_t);
* S_j: the existing synapses of KC j.

Updates are online, one position at a time, in text order. Each pass over the 40 KB counts as about 705 learner records (lines).

### 4.1 L-COMP: competitive Hebbian (instar on the winners)
For every active KC j (x_t,j = 1) and every synapse i ∈ S_j:

ΔB_ij = η · (q̂_t,i − B_ij / ‖B_·j‖₁), where q̂_t = q_t / ‖q_t‖₁.

This is followed by non-negativity and synaptic scaling.

* **What it does:** each winning KC moves its incoming weights towards the inputs it wins on. KCs come to tile the inputs that actually occur, instead of random directions.
* **Precedents:** the Krotov–Hopfield / FlyVec family, and SoftHebb in its hard winner-take-all form.
* **Locality:** no label and no prediction signal; everything a synapse needs is local.
* **Known risk: winner monopoly** (a few KCs win everything).
  * It is measured, not assumed: the entropy of KC usage and the fraction of KCs never active are reported every pass.
  * Pre-stated remedy, used only if C2 shows monopoly: switch on FastMemory's own homeostatic thresholds, in **all** arms at once, so the contrast stays one factor.

### 4.2 L-PRED: predictive three-factor rule (CLAPP-type, single layer)
A prediction matrix W (5,177 × 5,177, starting at zero) scores whether one code follows another: s(a, b) = bᵀ W a.

At each position t, the rule looks at two pairs:
* the **positive pair**: (x_t, x_{t+1}), consecutive codes;
* the **negative pair**: (x_t, x_n), with x_n the code at a random earlier position of the taught text (a replayed code), drawn with a fixed seed.

A scalar modulator γ is set as follows:
* γ = +1 for the positive pair if s(x_t, x_{t+1}) < 1;
* γ = −1 for the negative pair if s(x_t, x_n) > −1;
* otherwise γ = 0 (hinge, margin 1).

For the code b of the pair (b = x_{t+1} or x_n), its input q, and every active KC j of b with i ∈ S_j:

ΔB_ij = η · γ · (W x_t)_j · q̂_i, and ΔW = η_W · γ · b x_tᵀ.

This is again followed by non-negativity and synaptic scaling of B.

* **What it does:** contexts whose next codes are similar are pushed towards codes that predict each other. That is the route by which contexts with similar futures come to share KCs, which counting cannot do.
* **Precedents:** CLAPP (Illing et al. 2021); CLAPP++ (Zihan et al. 2026, Eq. 8).
* **Locality:**
  * three factors: a broadcast scalar γ; a dendritic prediction (W x_t)_j available at KC j; and presynaptic activity;
  * no error is sent back through any weight matrix;
  * for a single layer, this update equals back-propagation of the layer's own local loss (Zihan et al.). A gradient-cosine check is therefore unnecessary for this layer, and is not planned.
* **Surrogate for the derivative of a binary top-k unit:** only active units of b update (ρ' = x). This choice is stated, not tuned.
* **Cost:** W x_t only needs W's columns at x_t's 184 active units, and the W update touches 184 × 184 entries. That is cheap per position.
* **Known risks:**
  * W grows without bound: it is clipped to [−1, 1];
  * the rule ignores the input structure L-COMP captures.

Both are reported. Neither is tuned on held-out text.

### 4.3 Learning rates and passes
* η (and η_W for L-PRED) are chosen on **calibration** bits of the ONLINE readout, from a grid of three values per rule, on the dev text only.
* They are then fixed for the rehearsal and any main run.
* Passes: 10, with the code scored after passes 1, 3 and 10, so that the course of learning is visible. The pass count is not selected on held-out text.

## 5. Gates and decision rules (fixed here, before any code)
**G0, the native-wiring headroom (C2 dev, then confirmed in C3).** Read REF-N against FROZEN, then REF-D against REF-N.

| Result | Meaning | Next |
|---|---|---|
| FROZEN − REF-N ≥ 0.08 (about 5 SDs) | the native wiring leaves room | run the local arms |
| FROZEN − REF-N < 0.08, but FROZEN − REF-D ≥ 0.08 | the native wiring is the limit, not the weights | stop before the local arms; the owner decides whether rules may grow new synapses (D3) |
| both < 0.08 | the binary top-k code itself is the limit at this size | stop; report; no key-learning round |

Back-propagation sets an upper reference here. A local rule is not expected to beat REF-N.

**Outcome rules for the local arms** (rehearsal and main, if granted; one rule per arm):

| Label | Condition |
|---|---|
| LEARNS | capture ≥ 0.30 **and** arm better than FROZEN by ≥ 3 line-bootstrap SDs on the same text |
| CARRIES | carry-over ≥ 0.05 bits on text B, ≥ 3 SDs |
| NEITHER | otherwise; reported as a negative result for that rule as specified |

**Interpretation, fixed in advance:**
* LEARNS and CARRIES: the next round integrates the rule into FastMemory, so that its keys learn while it reads, and tests it there.
* LEARNS only: the representation fits the text but does not transfer. The next round examines why, with no new mechanism added.
* CARRIES only: the transfer is real but small on the same text. It is reported as such.
* NEITHER for both rules: the next step is the owner's. Options: more text, new synapses, or a different rule family. None is chosen automatically.

## 6. Predictions (recorded before any code; my record on such predictions is poor)
| | Prediction | My probability |
|---|---|---|
| P1 | G0 passes: REF-N headroom ≥ 0.08 | 0.6 |
| P2 | L-PRED: LEARNS | 0.3 |
| P3 | L-COMP: LEARNS | 0.25 |
| P4 | at least one rule CARRIES | 0.3 |
| P5 | L-COMP shows winner monopoly in C2 without homeostasis | 0.5 |

## 7. Texts and splits
All texts come from the same pinned NLTK Gutenberg archive (SHA-256 2d3c3ab5…d611).
* Text A: 40 KB taught, 5 KB calibration, 10 KB held-out.
* Text B: 10 KB taught (readout only), 5 KB calibration, 10 KB held-out.
* Both use V148's record and split rules.

| Stage | Text A (keys learned) | Text B (carry-over) | Status of the texts |
|---|---|---|---|
| C2 dev | chesterton-thursday | edgeworth-parents | both used before (dev): Thursday ties G0 to the headroom check |
| C3 rehearsal | austen-emma | chesterton-ball | both unused |
| S6 official (only if granted) | austen-persuasion | melville-moby_dick | both unused; Persuasion was reserved for V153's main run, which was not run |

austen-sense stays unused as a spare. Each text's SHA-256 is pinned at S1 lock.

## 8. Stages, checks and costs
**Implementation economy.** The front end is fixed, so each text's PN inputs q_t are computed once with MiniFly and cached. Codes for any key matrix are then top-k of Bᵀq_t. C1 checks that this cached path reproduces MiniFly's own `encode` bit for bit on the frozen keys.

| Stage | Content | Estimated CPU-h |
|---|---|---|
| C0 testbed | A known structure that a correct rule must find (§8.1); FROZEN parity with the check's 2.963 on *Thursday* | 0.2 |
| C1 technical | Cached-path parity; each rule's update matches its formula on a small fixture; non-negativity and scaling hold; L-PRED's hinge loss falls on a 2 KB fixture; REF-N/REF-D gradients checked by finite differences; determinism | 0.2 |
| C2 dev | G0 on Thursday; learning-rate grid for both rules (ONLINE calibration bits); 10 passes; carry-over to Edgeworth; monopoly diagnostics | 1.2 |
| C3 rehearsal | Fixed settings on Emma → Ball; G0 confirmed; all outcome rules applied as in a main run | 1.0 |
| S6 official | Persuasion → Moby-Dick, only with a separate grant and a cross-agent review | 1.0 |

* **R1 fits dominate the cost** (about 7 minutes each at 40 KB). Learning-rate selection uses the cheap ONLINE readout. R1 is fitted only for chosen settings and the scored passes.
* **Learner records:** C2 has 2 rules × 3 rates × 10 passes × about 705 lines = about 42,300. C3 has 2 × 10 × 705 = about 14,100.

### 8.1 C0 testbed: a structure a correct rule must find
A synthetic byte stream in which two disjoint sets of 8 bytes are interchangeable: every byte of a set is followed by the same distribution. Correct behaviour:
* L-PRED should make codes of contexts ending in bytes from the same set more similar than codes from different sets.
* L-COMP need not, because its objective is not predictive.

This tests that the predictive rule does what it claims before any text is used. If L-PRED fails it, the verdict is CANNOT_TEST_CLAIM_AS_SPECIFIED, and the design returns to the owner.

## 9. Owner decisions needed before any code
* **D1, yardstick and data.**
  * Recommended: same-text capture plus carry-over at 40 KB, with counted passes (§3).
  * Alternative: allow more taught text (for example 160 KB) and keep "beat counting" as the yardstick. It would need a different design and larger caps.
* **D2, caps for C0–C3:**
  * proposed: 3 CPU-h, 60,000 learner records and 3 test cycles;
  * plus 1.5 CPU-h and 15,000 records for S6 only if separately granted.
* **D3, wiring.** If G0 finds that the native wiring is the limit, may a later round let the rules grow new PN→KC synapses? Not needed for V155 itself, which stops there.
* **D4, references.** Back-propagation references (REF-N, REF-D) are used as instruments only, as in the check. Is that acceptable?

## 10. Out of scope, and coordination
**Out of scope:**
* native integration (keys learning inside FastMemory while it reads);
* the 16-compartment output port;
* the native writer;
* retention;
* longer or multiple timescales in the front end;
* arithmetic.

**Coordination with Codex:**
* Codex's V154 proposes deriving what a native write does to the 16 expressed values. That addresses the native writer, which is complementary and not duplicated here.
* Codex's objective is unseen addition. V155's is text representation and carry-over. The owner holds that split.

## 11. Guarding against my earlier mistakes
| Earlier mistake | Guard in this design |
|---|---|
| Option A and the class bias: choosing by the largest number in front of me, not the program's aim | Every arm is checked against the erratum's three checks (§3), and the class-bias recommendation is withdrawn |
| FE0 attribution confounded fade with scrambling | Only the key matrix differs between arms; homeostasis, if needed, changes in all arms at once |
| A reference that never learned (V152), and an undertrained baseline (COLA) | REF-N/REF-D gradients are checked in C1; G0 stops the round if the references show no headroom |
| Predictions stated with confidence, then wrong | Probabilities are recorded in §6. The C0 testbed checks the predictive rule's claim before any text |
| Settings tuned on held-out text | Rates are tuned on calibration in C2 only and then fixed; passes are fixed at 1, 3 and 10 |
| Bits on the same text taken as the whole aim | Carry-over to a second author is a primary yardstick |
| CPU projection from `ps` | Child CPU is measured by the attempt wrapper against the cap, as in the check |

## 12. Pre-lock refinements (26 October 2026, after the owner's go-ahead, before any code or outcome)
These changes are made at the S1 lock. No V155 code existed and no V155 outcome had been read when they were written. Sections 1–11 above are the version the owner saw. Where they differ, this section governs.

**12.1 Stage placement (the skill requires the testbed before learner code).**
* **C0 testbed = gate G0.** It is evaluator-only, and no local-rule code exists yet. On *Thursday* (V153 split) it runs:
  * FROZEN parity: R1 must reproduce 2.963 within 0.005;
  * REF-N and REF-D;
  * a **floor**: REF-N trained on the taught text with its targets permuted (fixed seed), for the same number of epochs as the chosen REF-N, then scored by R1 against the true targets.
* **C0 verdict:**
  * CONTINUE if all three hold: parity; FLOOR ≥ FROZEN − 0.03 (a pipeline that gains from shuffled targets is broken); and FROZEN − REF-N ≥ 0.08.
  * Otherwise one of: CANNOT_TEST_CLAIM_AS_SPECIFIED (the G0 rows of §5, with their stated next steps), or INVALID (floor or parity failed).
* **C1 technical:** the local rules and their fixtures, including the synthetic interchangeable-bytes fixture of §8.1, which moves here from C0.
* **C2:** no longer contains G0. **C3** still confirms G0 on its fresh text.

**12.2 Passes and records.**
* Passes are reduced to **4** (codes scored after passes 1, 2 and 4), instead of 10 (1, 3, 10).
* The rate grid is compared at pass 2 on ONLINE calibration bits. The chosen run of each rule then continues to pass 4.
* Reason: to stay within the kit's draft T1 limit of 20,000 learner records per delivery.
  * C2: 3 rates × 2 rules × 2 passes × 705 lines, plus 2 rules × 2 further passes × 705 = about 11,280.
  * C3: 2 rules × 4 passes × the rehearsal text's lines = about 5,600.
  * C1 fixtures: at most 1,000.

**12.3 Accounting.**
* **Learner records** are line presentations to a key-learning rule (passes × taught lines, per run).
* L-PRED's negative replays are counted separately: they are earlier taught inputs replayed, not new text.
* Readout and reference fits (R1, ONLINE, REF-N, REF-D, FLOOR) are evaluator presentations, reported separately. In V155, ONLINE evaluates a code; it is not an arm, unlike in V153.

**12.4 Readouts.**
* **ONLINE:** V153's online delta rule (η 0.01, 3 passes, binary code plus empty-state indicator). It is used for rate selection (calibration bits) and reported on held-out lines after passes 1, 2 and 4 for the chosen runs.
* **R1:** V153's settings (L2 grid 100, 30, 10; 300 iterations), fitted on the pass-4 code of the chosen runs, and on FROZEN, REF-N, REF-D and FLOOR.
* **Carry-over** (text B, 10 KB taught): R1 with L2 grid 300, 100, 30, 10, plus ONLINE.

**12.5 The rules, exactly.**
Definitions:
* c_j is the birth column sum of KC j;
* S_j is its birth synapses (KCs with no synapse never change);
* q̂^(j) is q restricted to S_j and normalised to sum 1 over S_j (no update when that sum is 0).

L-COMP:
* For each active j: B_Sj,j ← B_Sj,j + η · (c_j · q̂^(j) − B_Sj,j).
* This is a convex step, so each column keeps sum c_j and stays non-negative without separate rescaling.

L-PRED: the score is normalised, s(a, b) = b̃ᵀ W ã with ã = a/‖a‖₁, so s ∈ [−1, 1], and the hinge margin is 0.5. At each position t whose code x_t is non-empty and whose next position lies in the same line:
* **Positive pair:** b = x_{t+1}; γ = +1 if s < 0.5, else 0.
* **Negative pair:** the input of a position drawn uniformly (seeded) from positions already presented in this run with a non-empty code, re-encoded with the current keys; γ = −1 if s > −0.5, else 0.
* **Update order:** both γ are computed from the current B and W. Then, for each pair with γ ≠ 0, the key update B_ij ← B_ij + η · c_j · γ · (W ã_t)_j · q̂^(j)_i is applied to every active j of b and every i ∈ S_j. Then W ← clip(W + η · γ · b aᵀ, −1, 1), with binary a = x_t.
* **After the B update:** B is clipped at 0, and each touched column is rescaled to sum c_j. A column whose sum reaches 0 is reset to its birth weights, and the reset is counted.

Rates: η ∈ {0.01, 0.03, 0.1} for both rules. L-PRED uses one shared rate for B and W.

**12.6 References.**
* **Training:** torch, one thread, seed 0. Adam with learning rate ∈ {1e-3, 3e-3}, batch 256, at most 20 epochs, stopping after 3 epochs without improvement in the calibration cross-entropy of the reference's own linear readout.
* **Initialisation:** birth weights; readout at zero.
* **Top-k stand-in:** for each side, a_j = ReLU(pre_j − τ_side), where τ_side is that side's 92nd-largest pre-activation (held constant in the backward pass).
* **After every optimiser step:** B ≥ 0. REF-N is restricted to S_j. Columns are rescaled to c_j, and a zero column is reset to birth.
* **REF-N** is computed sparsely, on the existing synapses only. **REF-D** is dense, with all 302 × 5,177 entries learnable.
* The learning rate is chosen on calibration cross-entropy. The chosen B's binary top-k code is then scored by R1.

**12.7 Winner monopoly.**
* Measure: KC-usage entropy over taught positions in the last pass, normalised by log 5,177.
* Monopoly is declared if it falls below 0.8 × FROZEN's.
* The cycle decision is then "revise": the homeostatic thresholds are added to every arm, as one new design (k + 1), within the caps.

**12.8 Caps.**
* The owner's go-ahead named no numbers, so the defaults apply: 2 CPU-h, 5,000 learner records, 3 test cycles.
* C0 and C1 fit within them. Caps for C2 and C3 will be requested with the costs measured in C0 and C1.
