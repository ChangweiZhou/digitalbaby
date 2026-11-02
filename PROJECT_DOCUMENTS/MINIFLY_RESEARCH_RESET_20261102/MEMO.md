# MiniFly Research Program Correction — 2026-11-02

**Status: ACTIVE RESEARCH DECISION, DOCUMENTATION ONLY.** This note corrects the experimental-prioritization failure identified after V175–V178. It does not undo any source lock, retract a valid negative result, modify Full151 or authorize training.

## 1. What failed in the research process

The last four development experiments did not form a progression of one local, continuing learner:
- V175 tested coactivity-based inhibition, not the native writer's functional interference.
- V176 trained representations and then imposed an unvalidated fixed bridge and a fresh native value bank, so a failure of the final port did not establish failure of feature learning or persistent adaptation.
- V177 trained a new recurrent architecture under BPTT; its reference did not qualify and local candidates were never trained.
- V178 faithfully reproduced V164's dense BPTT delayed-copy reference but stopped at an arbitrarily sparsified ungated intermediate. Failure of sparse ungated recurrence cannot logically veto an untested gated variant, and its delayed-copy result is not a continual-learning outcome.

This was **goal substitution plus constraint accumulation**: decisions moved from persistent useful learning to probe accuracy, native bridging, BPTT trainability and a specific sparse topology. These intermediate questions were treated as necessary serial gates without demonstrating their logical dependence. Repeated failure has not falsified all local learning.

## 2. Exact operational correction

The active development reference is V167/V169's **fully local H/T/own-prediction learner**. This is not a claim that it beats counting or solves arithmetic. V169's own continued predictions from A→B improved on both A and B; its fully local model still lagged tuned counting by 0.59 bits. The historical model had an A-specific OTHER output mapping and restarted AdamW at the B transition. These limitations motivate **one** controlled continuous-life experiment, V179, with a full 257-class output and explicit optimizer-continuation and edited-feedback controls.

Starting now, prospective core-learning designs must include a **lifetime capability ledger** covering acquisition, retained still-valid prediction, legitimate revision and future learning/reuse from a single continuing parameter state. Component experiments can occur separately, but their scores do not promote the learner or force architectural dependencies. BPTT, evaluator probes and count models remain diagnostic external references.

## 3. Governance decisions

**Do:** preserve authentic source/seed hashes and each failure verdict, keep the fixed no-attention/no-arithmetic-features requirement, compare a candidate's own predictions, and account for prior experience, repeated exposures, optimizer state and CPU. A reusable mechanism must be trained and deployed by the same operative learner; no fitted evaluator substitutes for it. Never mask EOS or change task semantics after an outcome.

**Do not:** automatically allocate a new version after a failed component gate; freeze the program into an arbitrary sparse ring; infer all forgetting from a text shift; restore a best checkpoint before a new learning phase and call it continuous life; or demand that every native candidate pass through Full151's 16-dimensional output code. These restrictions are not logically implied by “local learning”.

V179 runs a **development** test on already-opened text pairs. It does not establish independent generalization, E2 factual retention or E3 never-taught addition. Any newly proposed mechanism after V179 must state which observed lifetime failure it addresses, the minimal causal control, the relevant *own-output* endpoint, and the fresh prospective evaluation protocol. The later project's never-taught addition criterion remains unchanged.

## 4. Evidence snapshots and disposition

- V167: fully local learned representation can improve transfer read through an evaluator probe; own top remains weaker.
- V169: B-only 3.917 bits; A→B 3.443 bits; A-after-B 3.215 vs A-before-B 3.273 bits; COUNT* B 2.819 vs replay learner 3.407. This is bounded, not independent or conclusive.
- V175: final complete association totals BASE 43/34 vs intervention 42/37 on two worlds; both capability/mechanism screens failed.
- V176: both feature-probe spaces gain, but held complete native output improved by only one item in one pair and zero in the other.
- V177: 0/704 exact-eight continuations on each world; BPTT reference did not qualify, local variants not run.
- V178: dense V164 reference 1.151629 bits; sparse 2.478 calibration bits at epoch 10, 1.839 at epoch 20. Gated and guard variants not run.

**Action:** register V179-CLLR-D1 as design only, and preserve the V175–V178 reports as unmodified history. The adopted core remains unchanged. A non-architectural baseline test may fail to show the desired capabilities; its negative result must be reported without generating another ungrounded mechanism by default.

See the accompanying V179 design and source links for exact controls and interpretation.
