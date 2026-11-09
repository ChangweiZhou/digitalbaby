# MiniFly design proposal: an on-demand temporal read head

**Status:** DESIGN ONLY — no scientific result, no implementation claim, no architecture adoption.  
**Date:** 2026-11-08 (America/New_York).  
**Target:** `codex/minifly-consolidated-20261023`.  
**Scope:** bounded, addressable historical reading as a supplement to the E11/E16 recurrent cell and E16-R3 raw-byte replay. This is a separate Phase-B hypothesis, **not** permission to change running E16-S/L/F/RC or the Phase-A gate.

## 1. Problem and hypothesis

The E11 cell processes a byte window and maintains a 512-dimensional diagonal-gated working state. In `exp11_cell.py::forward`, the state starts as `torch.zeros(B,C)` for each call; E16's long life is a *sequence of training examples and persistent parameter updates*, not one persistent recurrent inference session. The `exp45.Store` is an exact recent-byte-window -> next-byte-logit lookup, not an indexed autobiographical reader. E16-R3's reservoir stores observed prompt/answer bytes verbatim for **training replay**; its uniform sample operation is not inference-time retrieval.

Consequently, the model must currently compress intra-example information into the fixed-size hidden state; it cannot deliberately revisit an earlier observation on demand. The hypothesis is that **a small learned read controller plus an external addressable episodic log** can recover information that is no longer in the window or recurrent state, while preserving the existing local-learning philosophy.

A successful result would be *selective recovery of genuinely old, previously observed evidence*, not a merely larger prompt, a supervised key-value answer store, or replay into weights. No claim that a finite memory can reproduce an infinite history.

## 2. Proposed architecture and memory contract

Keep the existing E11 byte encoder, `z1`, projection `p`, diagonal cell, readout, local update and replay path unchanged for the first experiment. Add the following separable infrastructure:

1. **Append-only episode log** `E`, entries `(episode_id, t_start, t_end, raw_bytes, source_id, optional metadata)`. Raw observations, not regenerated correct answers. A monotone byte offset gives a *stable temporal address*. Store boundaries and time addresses even if the model never learns them. The log can be disk-backed; establish a strict byte budget and count indexes/metadata in memory usage.
2. **Sparse retrieval index** `I` mapping a learned or deliberately simple key to candidate *addresses*. Start with an explicit associative key (e.g. entity/event cue), hash/inverted map, or small embedding; do **not** count an oracle index as learned retrieval. Record index recall separately from downstream answer quality. A direct `read_at(i, width)` reads a raw slice around byte offset `i`.
3. **Read controller**, attached to current `h_t`, current byte features and optionally the question: outputs (a) an optional read request, (b) a query key or temporal offset, (c) a confidence score/budget estimate, (d) up to `K` candidate addresses.
4. **Re-encoding path**: fetch a *bounded* piece of old raw bytes, encode it with the already shared byte encoder, and merge it into a **separate temporary read workspace** or a gated residual of current state. Do not set `h_t <- h_i`; that would discard the current task and can import unrelated past dynamics. If a historical `h_i` is stored as an optimization, preserve provenance and still compare with raw-byte re-encoding.
5. **Write/consolidation path** remains separate: reading old evidence may improve the present answer; its eventual use to update slow parameters is a distinct, controlled experiment.

Conceptual flow:

```text
stream byte x_t -> shared byte encoder -> recurrent state h_t -> read controller
                                             |                |
                                      current computation  SKIP or SEEK(query/time)
                                                              |
                                  indexed episodic log E -> candidate offsets
                                                              |
                                    bounded raw-byte fetch -> shared encoder
                                                              |
                                             gated merge into temporary workspace
                                                              |
                                          continue reasoning / produce byte answer

background learning: observed raw episode -> bounded replay reservoir -> local updates
```

**Important separation:** the replay reservoir is a *sampling policy* over past training examples; the addressable log is a *persistent, stable-address data structure*. They can reuse the same underlying raw bytes, but a uniformly sampled, replaceable Algorithm-R slot is **not a stable historical address**. Eviction must invalidate references or use a versioned indirection table. Avoid silently conflating these two structures or reporting a single 1.05 MB figure if the index and raw episode log add memory.

## 3. Minimal mathematical specification

Let `h_t` be current working state, `x_t` current input, `E_t` external historical episodes, and `theta` the learner parameters.

```math
q_t = Q_theta(h_t,x_t),             u_t = sigmoid(G_theta(h_t,x_t)),
A_t = Retrieve(I_t,q_t,K),          b_t = ReadRaw(E_t,A_t,width=w),
r_t = Encode_theta(b_t),            h_t^+ = h_t + u_t * Merge_theta(h_t,r_t).
```

Optionally support a distinct `SeekTime(theta_t)` path when an explicit offset or previously stored pointer is available. For a factual query, query by content first and then follow a temporal pointer or nearby byte offsets. Multi-hop read is allowed only under a budget `K <= K_max` and total bytes read `<= B_read`.

Reading should be an *action* with a cost, not always-on attention. One possible action objective is expected change in predictive risk relative to reading cost:

```math
Delta_t = E[loss(answer_without_read) - loss(answer_with_read) | state_t]
          - lambda_reads * (#reads) - lambda_bytes * (bytes_read).
```

This is a definition of the proposed training/selection criterion, not an available unbiased target in deployment. Do not train a fake "uncertainty" oracle using the evaluation answers.

## 4. Trigger and credit assignment

**Why entropy alone is insufficient.** A confidently wrong model may never ask to read, whereas a correctly uncertain model may over-read. Measure the *value of information*, i.e. whether a read improves a subsequent answer at acceptable cost.

First-stage controller baseline: a binary `READ/SKIP` gate with a separately evaluated retriever. Develop using paired counterfactual episodes (same learner, query and frozen parameters; answer with versus without a candidate read), logged exclusively on training/development episodes. Compare to: always-read, never-read, random-read and oracle-read. At evaluation, neither oracle address nor oracle READ decision is available to the learned arm.

For lifetime-local learning, candidate mechanisms are (in order): a local supervised broadcast learning signal for the gate, reward-modulated eligibility traces on read actions, and only then a learned content-key encoder. Do not infer a successful *local rule* just because a BP-developed retriever works. If an address is discrete, train the controller with an explicit local action-learning objective or restricted candidate reranking; record how exploration and rewards are supplied.

Correctness of a retrieval action is a separate metric from downstream answer accuracy. A retrieved substring must be able to affect *later* computations, rather than only add a final logit.

## 5. Complexity and scaling

Let total archived history size be `M` bytes, `N` indexed records, embedding dimension `d`, `K` returned candidates and `w` bytes per retrieved candidate.

- Direct known-offset access: `O(1)` address lookup **plus I/O of O(w) bytes** under random-access storage assumptions; physical latency is not necessarily constant.
- Exhaustive semantic similarity: `O(Nd)` per query; exact all-pairs attention to all past tokens is not the proposal.
- Inverted/hash index: expected query complexity depends on postings length and indexing quality. Approximate nearest-neighbor may improve typical query cost but provides no universal sublinear worst-case guarantee.
- Read/merge neural cost: approximately `O(K*w*P_step)` in a simple byte re-encoding implementation, plus indexing.
- Persistent storage: at least `Omega(M)` if every byte must remain exactly retrievable; indexes and metadata are additional. A bounded memory gives an explicit forgetting/eviction contract.
- Normal streaming forward cost **without retrieval** remains linear in stream length at fixed architecture, assuming incremental recurrent inference; current `respond` implementation may recompute prefixes and should not be cited as a streaming benchmark.

Report all of: neural FLOPs, number of actual reads, average/p95 retrieval latency, bytes returned, index memory, log memory, replay memory and total RAM/disk. No claim of unlimited-context memory for fixed storage.

## 6. Experiment ladder: one-factor-at-a-time controls

### Phase 0 — plumbing / oracle ceiling (not evidence of learned retrieval)

Freeze a chosen existing E11/E16 checkpoint. Give each input a unique episode ID and byte offset. Make two lossless retrieval APIs: (a) `read_at(i,w)`, (b) `read_by_key(key,K)`. Verify round-trip bytes, stable addresses, index recall, on/off parity with baseline, context isolation and no test targets in index entries. First score oracle address plus read/merge integration. If oracle reading fails, fix the integration; do not train the controller.

### Phase 1 — long-gap episodic recall

Training stream: arbitrary factual assignments such as `Alice assigned project R17 to Bob`, then distractors; later ask `Who was assigned R17?`. Use many randomly generated names, projects and roles, so a model cannot answer by memorizing a fixed set of pairings. Split entities, keys, assignment tuples, times and rendered forms; include new names and unseen syntactic templates. Do not let held-out facts or their answers re-enter training as subproblems.

Compare matched arms:

| Arm | External archive | Address selection | Merge |
| --- | --- | --- | --- |
| A | none | none | baseline E11 |
| B | same archive | random eligible historical address | same merge |
| C | same archive | oracle correct address | same merge, **upper bound** |
| D | same archive | learned key/time read head | same merge |
| E | same archive | always-read, same budget | same merge |
| F | same archive | learned head, **scrambled fetched content** | same merge |

All arms should have identical neural capacity, training examples, update counts and available RAM budget where feasible; clearly identify inherent archive differences. `C` must never be reported as an autonomous or local-learning success.

Sweep gaps `10^2,10^3,10^4,10^5,10^6` bytes and report exact answer, address recall@K, evidence precision, READ rate, false-positive read rate, skipped-needed rate, cost and latency. Counterbalance distance so time position alone does not predict the answer. Evaluate once on frozen sealed panels.

### Phase 2 — mutation and temporal precedence

Present different values for the same key at different times: `R17 -> Bob`, later `R17 -> Carol`. Ask for *latest*, *earliest*, and *value at time i*. Include contradictory distractors and delayed corrections. Require retrieved evidence and answer agreement; an exact-key lookup returning one arbitrary old value should fail these tasks.

### Phase 3 — multi-hop re-use

Create questions that need **two separate historical addresses** and an intermediate computation (e.g., retrieve who owned R17 at time i, then retrieve that person's assigned laboratory in a later episode). Score both addresses, intermediate byte answer, final answer, and wrong-evidence ablations. Teach neither held-out question pair nor its final answer.

### Phase 4 — continuous life, no reset

Only after Phases 0–3, switch to a true online API `step(x_t,h_t,E_t)` carrying state and persistent log across input calls. Evaluate identical histories both as one contiguous stream and split into arbitrary calls, with invariance of outputs/addresses. Separate learned *knowledge in theta* from one-off facts retrieved from E. Use an explicit, bounded archive and a longer nonstationary sequence than any training history.

## 7. Pre-registered decision gates (draft; lock before launching)

Use three original seeds only for engineering smoke tests; use **at least three fresh seeds** for science after code and thresholds are frozen. Proposed Phase-1 gates:

1. Baseline (A) cannot solve the longest gap by local window or repetition shortcuts; preregister a quantitative floor based on held-out data.
2. Oracle read (C) demonstrates that the evidence and merge path are sufficient: >=95% exact answers at all gaps up to `10^5` bytes, otherwise stop and repair integration.
3. Learned read (D) reaches >=85% exact and >=90% evidence recall@K at `10^5` gap, and is no more than 10 percentage points below (C) at the same gap, in **every fresh seed**.
4. Random-address and scrambled-evidence arms (B,F) must be at least 30 points below (D); otherwise mechanism attribution fails.
5. On clearly no-read-needed control questions, D must skip >=80% of the time; otherwise this is always-on retrieval rather than selective control.
6. No held-out episode, query, target, or near-duplicate appears in archive *before it is presented in the evaluation timeline*. Audit the entire timeline, including any replay and auxiliary teaching steps.
7. Report persistent bytes and the realized retrieval budget; if the budget is exceeded or the same test is retried with a changed threshold, mark the preregistered run as failed.

These are **prospective targets, not achieved results**. For Phase 2/3 define independent gates *before* adding them. Avoid reading the current E16-R3 gate as evidence for these new tasks.

## 8. Known threats / failure signatures

- **Address oracle leakage:** human/fixture gives exact offset at evaluation, or the query uniquely encodes a sortable training ID; evaluate a learned mode without either.
- **Repetition disguised as reasoning:** direct record contains the answer; evaluate separately factual recall, temporal selection, and genuine two-hop composition.
- **Index coverage leakage:** retriever index built over held-out future episodes or normalized with hidden labels; timestamp each write and freeze splits.
- **Confident wrong without READ:** entropy-only controller misses the case; measure conditional read value and false-skip rates.
- **Store/replay confusion:** `exp45.Store` logit lookup, raw history index and training reservoir are three separate capabilities; ablate each.
- **Runaway read loop:** cap `K`, bytes, wall latency and retries per question.
- **Unbounded cache disguised as constant memory:** log external storage bytes as a resource; compare equal persistent-memory budgets.
- **Reset confound:** `forward` currently zero-initializes state; a no-reset claim requires a new streaming API and split-call parity test.
- **Locality confound:** an end-to-end BP-trained read controller is only a feasibility ceiling until lifetime-local training is demonstrated.

## 9. Integration constraints and explicit next steps

1. Preserve E16-R3, E16-S/L/F/RC and E16-A scientific code and predictions. No modifications to their recorded gates or concurrent jobs as part of this document.
2. Build the raw log and `read_at` in a **new experimental module**; test stable addresses with evictions, restarts and arbitrary call boundaries.
3. Add read/skip and fixed-key retriever for Phase 0, **without** learned queries. Check learned-model parity with the old path when no reading occurs.
4. Run oracle, random and scrambled-read arms before developing the learned head. Only then add a locally trainable controller.
5. Include the exact-key `Store` ablation (E16-A) as a separate factor: avoid crediting the read head for a benefit supplied by old key-logit bias.
6. Only if the head passes temporal-selection and multi-hop tests should it be wired into the standard E16 learner and used for natural-text pilots.

**Non-goals:** reproducing Transformer full attention, making every old hidden activation queryable, changing the PN–KC–MBON architecture by fiat, claiming general reasoning, or eliminating finite-memory limits.

## 10. Prior work and originality boundary

This is a project proposal, not a claim that a read head is novel: Neural Turing Machines (Graves et al., 2014), Differentiable Neural Computers (Graves et al., 2016), and retrieval-augmented language models already study external memory and learned addressing. The potential contribution is a *causally audited, low-cost, locally plastic controller integrated with an online recurrent learner and experience consolidation*, not the concept of external retrieval itself.

**Source anchors:**
- `MOD97_LOCAL_RULE_20261106/exp11_cell.py` (`CellNet`, `forward`, `respond`, local trace).
- `MOD97_LOCAL_RULE_20261106/exp45.py` (`Store` exact-key logit memory).
- `MOD97_LOCAL_RULE_20261106/exp16_integrated.py` (E16-R2/R3 reservoir and replay distinctions).
- `MOD97_LOCAL_RULE_20261106/E45_REPORT.md` (E16-R3 observed-byte results and qualifications).
- `MOD97_LOCAL_RULE_20261106/E16V_REPORT.md` (source/receipt audit limitations).

**Acceptance criterion for the design document:** reviewers should be able to implement a head without treating an oracle, extra historical storage, or full BP during life as proof of an autonomous persistent-learning mechanism.
