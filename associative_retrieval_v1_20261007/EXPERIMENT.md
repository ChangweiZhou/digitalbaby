# Bounded associative retrieval — persistent-core experiment v1

7 October 2026. The three-cycle exercise develops and qualifies this design. Formal science is not authorized by the exercise. The final qualification verdict and resource estimates are recorded in `THREE_CYCLE_REPORT.md` and `QUALIFICATION.json`.

## Objective and scope

Improve the existing bounded byte-memory core by learning how its shared representation can retrieve useful private memories. Keep acquisition, delayed retention and revision strong; test modest reuse on the existing never-taught binary relation. No new intelligence requirement, encoder replacement, natural-text task, autonomous stopping or output alphabet is added.

The only candidate is **LINK**, a 64-entry learned association table connecting observed shared FE0 KC activity to observed CONTENT KC activity. This is an engineered memory operation. Ledger precedents M045/M048/M061–63 and M094 motivate it but do not prove it works. It is not a claim that the fly's anatomical KC network implements this table.

## Frozen mechanism

All eight Full151 stores receive independent canonical births. Shared FE0 and private last-four-visible-byte CONTENT inputs, scales, timing and original ERROR teaching remain fixed.

- An entry stores a sparse shared code and coherent private code, each sorted/unique with at most 256 active KC indices. It stores no answer, identity label, cue string, class, world, stage or familiarity flag.
- Capacity is exactly 64 entries. A newly observed private code replaces the oldest insertion when full. Repeated private codes update the paired shared observation without refreshing insertion age. No unbounded history is retained in the learner.
- Prediction retrieves at most four prior entries by shared-code cosine overlap. Ties use oldest insertion, then slot. Only positive scores participate; weights sum to one. An empty table contributes zero.
- Each selected private activity vector is read through the existing private Full151 stores at current model time, using disposable native copies. There is no extra teaching or elapsed model time. The retrieved four-channel values are weight-averaged.
- Let `u_ERROR = shared/shared_scale + private/private_scale`. The emitted LINK byte is the first argmax of `u_ERROR + 0.5*(retrieved−mean(retrieved))/private_scale`. The half-gain is fixed, not fitted. Original private probabilities continue to determine ERROR writes; retrieved values never enter the teacher.
- A completed record commits the cached pre-answer pair, using cue activity only. This occurs identically with native teaching on or off. Queries, two-option probes and unfinished records do not commit associations. The extra state is disclosed as exposure learning, not as feedback-dependent memory.

## Conditions and exact trajectory sharing

**ERROR** is the existing core; **LINK** adds correct learned retrieval; **PERM** uses the identical learned table/query weights but permutes private target KC indices with a fixed label-blind bijection preserving side, encoder eligibility and writable-support strata. Capacity and read operation counts match. Individual Q/T weights and realized value magnitudes need not match.

These are three reporting conditions on one native trajectory, because neither association outputs nor the chosen byte changes the original private-only write probabilities, input sequence or feedback. Native parity is tested, not assumed. This saves native trajectories while retaining the integrated LINK run and actual teacher audit. Baseline cost excludes the extra operation; candidate and PERM costs include their respective lookup/read operation. Diagnostic sidecar computation and audit overhead are separately charged. Eight stores are independent within every native life; sharing a proven trajectory across readout conditions is explicitly declared.

## Exposure, outputs and causal controls

Reuse unchanged lifetime and wrapped relation fixtures from `persistent_core_v2_trial/v2_fixture.py`, imported through the immutable parent runtime.

Lifetime: 32 old and 32 disjoint new arbitrary labelled keys, followed by eight corrected old keys. Four histories W/N_old/N_new/N_revision have the same bytes and clocks; only the declared native supervisory stage differs. Final old accuracy uses the 24 intact old keys, never the eight obsolete labels. Acquisitions, day gaps, original checkpoints, new learning and revision remain unchanged. Exact/canonical-key lookup can solve this E1 task.

Relation: twelve old taught orientations, six never-taught orientations (three reversed pairs), and six disjoint new taught orientations. W and N_old have identical sensory histories and cue-only association updates; old native teaching is suppressed in N_old and actual native transitions are checked. Old-end, day-one, post-new and final probes are preserved. The internal ChoiceOrgan receives two options and emits L/R in both balanced orders before feedback. Neither arm is given a held-out answer. Taught E1 and held-out E3 are reported separately.

An exact-key table cannot determine held-out relation answers. Exposure-independent symbol/frequency biases are checked by N_old; a simple class rule learned from old experience is valid limited E3. No requirement to outperform a simple learned class rule is introduced. Native value-writing controls are essential: the association table itself contains no answer, although it supplies learned retrieval geometry.

## Three technical cycles

1. Design/audit sparse-address, capacity, tie, FIFO, empty-query and restoration semantics; unit tests and native smoke.
2. Refine causal/trajectory/cost accounting; native birth and ERROR parity, actual no-write and reader-input tests, future-answer leakage tests, clone and hostile receipt cases.
3. Complete both existing development tasks with all branches and checkpoints, independently replay every query/update, fresh-process resume and committed-receipt skip guard, then measure cost and footprint. One development world is not independent scientific evidence. Technical scores cannot select parameters.

Development IDs: 71008001–71008003. The shipped worker rejects science IDs. A safe-record resume fixture is not arbitrary mid-record crash recovery. No automatic restart after a scientific failure is permitted.

## Prospective bounded science plan

Screen IDs 71009001–71009008, then at most 64 fresh confirmation IDs 71010001–71010064. Fixed candidate/target; no dose search or sample extension. Screen: 16 physical world-assay jobs / 48 native lives / 31,104 exposure records. Confirmation: 128 physical jobs / 384 lives / 248,832 records. Maximum: **144 physical jobs / 432 native lives / 279,936 records**, with three readout conditions. Disposable probe copies and partial checkpoints are counted separately.

Screen advances only if final LINK W old/new/revision E1 are each at least 90%, mean loss versus ERROR is at most 2pp for each, at most one of eight worlds has any E1 loss over 10pp, raw held-out W is above .5, own W−N_old is positive, causal gain versus ERROR is at least 5pp, LINK−PERM causal gain is positive, and measured operational CPU ratio is at most 1.25. These are budget-selection rules, not tests proving a mechanism absent when they fail. No replacement candidate is chosen.

Confirmation n=64 is fixed. A prior paired-gain SD near .11 implies an approximate two-sided 95% half-width of 2.7pp; own prior-teaching SD near .32 implies about 7.8pp. This design can remain inconclusive for a small effect. SD assumptions come from prior results, not the development test; power is not guaranteed.

Primary target: final held-out causal reuse gain `LINK(W−N_old)−ERROR(W−N_old)`. Adoption requires mean gain at least 5pp and positive one-sided .03 lower bound, positive raw W improvement lower bound, positive capability-per-operational-CPU gain lower bound, plus .01 joint evidence that LINK's own W−N_old is positive and raw W is above .5. Report W, N_old, old-end and final effects separately. Matched knowledge/feedback budgets are unchanged.

Protect final intact-old/new/revision E1: observed mean losses no more than 2pp and an exact-binomial one-sided .03 upper bound below .10 for the fraction of worlds with any E1 loss over 10pp. Operational CPU includes teaching and final queries; LINK/ERROR ratio must be at most 1.25. Report fixed/mutable/temporary state, charged worker/CPU, cold start and external audit separately. A positive LINK−PERM .01 bound supports the registered correspondence mechanism; this diagnostic never supplies a fallback winner or recycles alpha.

Use paired world-level Student approximate lower bounds for nonzero variance. For zero variance use the preregistered Hoeffding bound with actual theoretical range (gain width 4; own-effect width 2; raw accuracy width 1), or report unresolved; no zero-width interval from observed equality. Define capability efficiency as `1000*(W−N_old)/max(W operational CPU seconds,1.0)` and compare LINK minus ERROR per world. Its theoretical gain range is [-2000,2000], width 4000. The one-second floor is declared in advance, not fitted to measurements; measured complete lives are much longer. Operational deployment cost uses W; all causal-control lives remain charged to experiment resources. Formal analysis implementation must match this lock; it is not a way to change the scientific target.

## Resource and stopping contract

New mutable association arrays: 131,840 bytes; fixed diagnostic permutation: 20,708 bytes. Inherited native state, Python objects, caches, query matrices and clone buffers are additional and measured separately. Query scans at most 64 sparse codes and reads at most four coherent patterns; no history-sized table or dense N×N learned matrix. Inherited Full151 still performs global projection/sorting/state updates: this experiment does not claim a scalable sparse core.

For a formal launch, use measured complete-job wall/CPU/disk with a 1.5× planning margin, maximum 8 GiB experiment storage, 1 GiB measured RSS per worker, and an eight-worker operational target only after launcher qualification. Keep only active whole-record checkpoints; retain complete receipts and failed-job evidence. Each session stops dispatch at nine hours and pauses safely before the ten-hour outer limit. No automatic resume or retry. The margin is not a hard future-runtime guarantee.

An E1 feasibility loss over 10pp or teacher-API cost over 1.5× in the fixed development world blocks promotion to science pending a documented design decision; it does not permit tuning on held-out scores. Final science may legitimately retain ERROR. Report the limited evidence actually earned and end this experiment without adding a new task or another integration round.
