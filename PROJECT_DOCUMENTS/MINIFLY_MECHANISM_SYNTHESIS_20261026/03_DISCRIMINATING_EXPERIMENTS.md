# Memo 03 — Next experiments: identify the missing operation, avoid another mechanism carousel

**Status:** Proposals needing separate design approval. No experiment is authorized by publishing this memo.

## Priority 0: finish the already positive evidence without overclaiming

### E0. V167 independent carry-over

**Question:** Does the locally trained representation improve *more than a BP evaluator can see*?

Use the already prepared second pair Emma → The Ball and the Cross; retain the documented Gutenberg-footer exclusion and source hashes. Under fixed rules and fair exposure:
- Compare LOCAL, LOCAL-SHUF, direct KC and BP-H positive reference.
- Score both a frozen feature representation with a newly trained **local** B readout and the BP-probe reference.
- Record representation-learning presentations, B-probe training information and total CPU separately.
- Predeclare one carry-over criterion and do not tune on held-out B.

**Interpretation:** If local-source/local-probe scores retain the gain, a complete local feedforward transfer result is supported on that setting. If only BP evaluation accesses the information, distinguish representation from its learned deployment. This is not an E3 arithmetic claim.

## Priority 1: inspect V168's learning signal, not its next version number

### E1. Controlled gradient/optimizer microscope

**Question:** Does KF-RTRL fail because its gradient estimate is too noisy, because the optimizer converts noise into poor updates, or because short memory-conditioned influence is lost?

Use fixed saved states where available; reconstruct only permissible source-locked diagnostic states, explicitly declaring any missing original optimizer state.

On identical query episodes obtain:

1. Exact BPTT gradient and exact TBPTT-3 gradient.
2. The exact memory/query-specific carried component and corresponding KC memory factors.
3. KF estimator over a preregistered small number of signs, with mean, variance, cosine, norm and alignment of the *query-weight* component separately.
4. Per-step contributions by lag 0,1,2 and >=3; compare to the exact short-window gradient.
5. Actual post-clipping gradients and optimizer steps **from cloned identical AdamW state**; test infinitesimal and one-step objective changes on untouched diagnostic examples.

**Discriminating predictions:**
- Good mean, large scatter, and improved averages with independent draws -> variance likely dominates.
- Good raw estimates but bad applied steps -> clipping/Adam scaling problem.
- Missing query-weight credit even in mean -> implementation or structural approximation flaw (despite the exact expectation fixtures).
- Exact TBPTT-3 itself cannot learn under a newly proposed schedule -> the reference is not qualified; do not use it as a ceiling.

Keep optimization searches narrow and grounded in E1 results rather than an unbounded learning-rate or sample-count sweep.

### E2. Three-step deterministic short-trace reference

Implement an independent numerical reference for the exact recent-influence recursion and compare **analytically or via finite differences** to TBPTT-3 on small episodes. Respect that the V168 recurrent system carries both h and r and the output has a direct r dependency.

Use V166's factored M_t to see whether short memory-conditioned query influence can be represented cheaply. Report actual parameter memory, factor ranks, FLOPs and wall/CPU, and nonlocal communications required.

Only if the short-path reference can learn at the fixed target budget should a compressed/local trace be attempted.

**Important:** “Online” and “synapse-local” are distinct constraints. An exact short-window trace might be forward-only yet depend on dense Jacobians and inter-neuron feedback.

### E3. Fixed-rank variance ablation (conditional on E1)

If variance emerges as the leading failure, compare predetermined numbers of factors/draws (e.g., 1 versus 4) at **matched total compute and exposure**, with a correctly qualified TBPTT-3 positive control, and independent initialization/worlds. Do not allow a large unregistered parameter/rate grid to turn a failed estimator into an outcome-picked result.

## Priority 2: clarify state sufficiency and compute-with-memory

### E4. Required-history distinction

Construct controlled **task histories whose current cues/recent suffixes match but which require different future responses**. Evaluate:

- Whether a qualified read-only diagnostic decoder can recover the relevant earlier distinction from the internal state;
- Whether altering the relevant memory association causally changes the intermediate query and final output;
- Whether matched unrelated memory perturbations leave behavior largely intact;
- Whether the distinction remains available under distractors and new learning.

This tests *availability*, *actual use*, and *retention* separately. Evaluator labels and oracle states do not enter native learning.

### E5. Architecture-only extension after learned two-hop

Only after a non-BPTT learner qualifies on V166's same two-hop task, vary one scaffold at a time:

- Additional composition depth (three hops without retraining versus retrained).
- Control of query timing rather than fixed Q/WAIT/ANS schedule.
- Learning which events to write instead of hand-gated writes.
- Replace simplified 512-KC key generator with genuine MiniFly code.
- Only then generic byte input and native autonomous output/EOS.

Do not simultaneously replace the key code, learning rule, controller, memory writer and score.

## Priority 3: investigate V163's auxiliary objective only if still relevant

### E6. Nuisance-controlled contrastive pairs

V163's local pair discrimination is ~99.7%, but downstream carry-over remains worse than direct KC. Hypothesis: adjacent states share input history, creating a shortcut.

Test a cheap overlap/suffix-only classifier as a nuisance baseline; compare original negatives with negatives matched for short-history and KC overlap. Hold representation architecture and evaluation consistent. Verify a positive reference still learns the revised contrast before blaming the local rule. Independently compare features learned by BP-H via controlled subspace intervention, not only coefficient cosine.

Do **not** choose a new objective using already opened held-out performance.

## A mathematical stop/go board

| Established fact | Follow-up authorized by *scientific reasoning* (still needs procedural approval) | Stop/redirect if |
|---|---|---|
| V165/V167 local task feedback helps learned features | Repeat on second pair with local B probe | Positive effect vanishes when evaluated locally/independently |
| V166 memory loop trained with BPTT composes two associations | Derive non-BPTT short credit reference | Qualified reference does not reproduce within restricted setup |
| V168 three-step gradients suffice in tested task | Preserve 2-step carried influence exactly, then compress | Trace cannot match TBPTT-3 even in small fixtures |
| V168 KF-RTRL unbiased but ineffective at budget | Examine variance and optimizer preconditioning | Negative effect explained by code/numerical inconsistency |
| V159/V162 G redundant on tested text | Condition information/use study | No incremental signal beyond already available KC history |
| V157 legacy native addition still fails E3 | Keep unchanged E3 endpoint while isolating mechanisms | Proposed learner solves only recitation, not novel relations |

## Anti-overclaim and artifact discipline

- The architecture-sufficiency result is V166 **BPTT synthetic** success only, not proof of universal architecture necessity/sufficiency.
- KF-RTRL is not a biologically local rule merely because it is forward/online. Its exact Jacobians and 64×512 influence factors violate a strict synapse-local interpretation.
- A positive gradient-cosine result does not itself establish a useful update after clipping/Adam.
- Unbiasedness is an expectation identity, not a sample-efficiency theorem.
- E0 finite feasibility, E1 taught own-answer acquisition, E2 relevant retention, E3 novel relation transfer remain separate.
- Keep the latest committed artifact and the frozen old versions; do not alter current core adoption pointers.
- Require exact source+asset hashes, original data receipts, technical failures, no-hidden-answer compliance, and compute accounting before interpreting any new experiment.
- No new full 150-mechanism campaign. Consult [the 44-paper bibliography](../LOCAL_LEARNING_LITERATURE_20261025/PAPERS.md) and [150-row human mechanism catalog](https://github.com/ChangweiZhou/digitalbaby/blob/7c2a4d94791adbf2b3ac8516b53acf1695d3cc0c/human_learning_mechanisms_20261023/CATALOG.md) when a missing *computational operation* has been identified.

**Highest-value immediate pair of actions:** complete a second-pair, locally evaluated V167 replication and run a fixed-state V168 query-credit microscope. Both are narrower and more falsifiable than another end-to-end architecture redesign.
