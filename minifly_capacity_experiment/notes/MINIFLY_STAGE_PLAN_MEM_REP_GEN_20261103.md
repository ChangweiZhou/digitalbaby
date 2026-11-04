# MiniFly persistent learning by stage: memorization → representation → generalization

*2026-11-03. This builds on `MINIFLY_WHY_STUCK_20261103.md` and the companion review (Addenda 1–3).*

**Constraints:** no backpropagation and no attention, including content-addressed key–value reads. Everything else may change.

**Evidence type:** historical locked reports, plus my local development reruns of the recovered source, 1–2 seeds each. Nothing here is a confirmation. Every proposed experiment is DESIGN_ONLY and NOT_RUN, and needs owner authorization.

---

## 0. Stage definitions

The project's own E-scale is used to keep the stages testable.

| Stage | Question | Pass looks like | Main enemy |
|---|---|---|---|
| **1. Memorization** | Can a taught fact be stored and kept while newer facts are learned, and revised when it changes? | E1 own-output recall; E2 retention after interference; correction of obsolete facts | Write interference on shared state; drift of a shared base |
| **2. Representation** | Is the stored fact attached to an *identity* rather than one exact surface form, so it is reused across renderings, positions and phrasings? | The fact is recalled from a held-out rendering, one-shot, without breaking retention | Codes that separate (good for recall) versus codes that share (good for reuse); code drift over time |
| **3. Generalization** | Can the learner answer **never-taught** relations derived from taught ones (order, roles, cycles, arithmetic)? | E3: whole-relation holdouts beat lookup, frequency and shortcut controls | A lookup-like store has no inductive bias for structure |

---

## 1. Memorization — status: solved in principle; the remaining issue is resources

**Evidence:**
- **Exact or own-addressed state retains well:**
  - V35 LOCAL-V: 99.2% retention.
  - Visible-context bridge into FULL151: 512/512, −0.78 pp after interference.
  - The FE-PERM line in V181–V186.
- **Shared plastic state fails, and the cause is known:**
  - Writes from later learning, not decay (iteration 9: REST 32/32, no-write 32/32, full learning 24/32).
  - Writing on every byte: target-only writes 23% vs all-byte 3% (V52).
  - Base drift: V35 LOCAL-R 29.7%.
- **New this session:** a *fixed sparse superposed store* (lag-96 code → N KC units, K active → local delta-rule readout) also memorizes. Its capacity grows with N:

| facts | N | retention after 3 waves |
|---|---|---|
| 128 | 4,096 | 59–69% |
| 128 | 16,384 | 100% |
| 512 | 65,536 | 94.5% |

  These are recovered-code runs in `v39_recovered_runs/`, one seed per cell.

**What is left:**
1. **The capacity law C(N, K).** One grid with 4 seeds, quantized storage. This decides the resource contract; the V181 cap of 37k parameters and 8 MiB rules out the working size.
2. **Label-free write allocation,** to stop spending capacity on syntax bytes. Arms: error-gated writes; recall-gated consolidation (Lindsey & Litwin-Kumar 2024); and a count-matched random control. If neither gate beats the random control, close the gating class.

---

## 2. Representation — status: half-solved. Surface invariance works; semantic invariance does not

**What worked:**

| Result | Mechanism | Limit |
|---|---|---|
| V39 superposed store: one-shot held-out rendering E1 ≈ 1.0; held-out after D 89% at 512 facts with N=65,536 | Shared sparse features: alias 0 writes alias 2's neighbourhood in proportion to same-key overlap | Only renderings that share most bytes with a trained one (header/separator changes). Surface similarity, not identity |
| V37 CORR2 slots: 100% held-out alias after corroboration | Similarity-linked slots (content-addressed, now excluded) | Needs the alias link to be *observed* (E1 = 0.8%); not allowed under the no-attention rule |
| V45 PAIR-OJA: a learned invariant subspace from corroborated pairs; held-out −0.058 bits, transfers to fresh identities | Local Hebbian/Oja learning of an equivalence subspace | It raised cross-talk inside a *single shared* matrix; the gain shrank with load |
| Bridge Step 15 ROLE code: +34 pp relation transfer, 66% capacity | Fixed early/late role traces of adjacent bytes | A fixed, hand-chosen code; capacity below gate |
| V9.2 hashed short-context predictor: beat Markov-2 on held-out authors (2.81 vs 3.00 bits) | Many overlapping exact-context features with a local delta rule | Statistical generalization only; needs ≥100 kB of data |

**What failed and why (this is the stage's key lesson):**
- **Codes trained to predict the next byte erase identity.** Bridge Step 5: KC Hamming distance 67 → 9 as predicted distributions converge. A learned GRU lost position information. The predictive bridge harmed E3 by 22 pp.
- **Codes that keep adapting break old memories.** V26 live encoder: old facts 33%. FE1 habituation: −0.044 on delayed probes. V40 phase codes drifted to 62% self-overlap.
- **Maximum separation is not the goal.** Retention tracks different-key overlap; transfer tracks same-key overlap (V39). Stationarity beat a high separation ratio.
- **A change in position or wrapper breaks a pure lag code.** The key is bound to byte position, so a shifted rendering looks like a different fact. Hash codes are worse still (H4 shift access 24%), and the FE0 shift test also failed (E2 +4 pp).

**Diagnosis:** the project has a stationary code that generalizes across *small surface edits*, and a store that can hold it. What is missing is a code in which **the same entity in a different position or wrapper activates overlapping units**, learned without labels and without drifting after facts are stored.

**Proposed work, in order:**

**R1 — measure the representation gap directly (C0, read-only, cheap).**
- On the scaled V39 store, add held-out probe families of increasing difficulty:
  - (a) new header or separator (already passes);
  - (b) the key shifted by 1–3 bytes;
  - (c) the key embedded in a longer, variable wrapper;
  - (d) the key's bytes case-changed or with a spelling variant.
- Measure the same-identity overlap (Jaccard) of the code for each family, and its accuracy.
- **Falsifier/decision:** if (b) and (c) fail while (a) passes, the bottleneck is *position binding* in the front end, not the store, and the next step is R2.

**R2 — a multi-scale, position-tolerant front end (fixed, no learning).**
- Add to the lag code a bag-of-recent-n-grams component: hashed byte bigrams and trigrams within the last 12–24 bytes, without absolute position.
- This is the V9.2 feature family (which generalized on prose) plus the fly's "many PNs per KC" mixing.
- Arms: lag-only; n-gram-only; lag + n-gram; and a hash control that destroys input similarity.
- **Pass:** shifted and wrapper probes rise by ≥20 pp with retention within 5 pp of lag-only at matched N.
- **Falsifier:** if a position-tolerant code costs retention, because unrelated keys now share n-grams, the trade-off is intrinsic to a fixed code and R3 is needed.

**R3 — learned invariance with consolidation-safe timing.**
- Revive V45 PAIR-OJA. Learn the equivalence subspace from label-free corroborated pairs: two contexts followed by the same rare outcome.
- Apply it **only to the address that new writes use**, and freeze it per memory epoch, so that stored facts are never re-addressed (the drift lesson). Concretely: two code generations, with old facts kept readable through the frozen old code. Retiring the old generation is a later question.
- **Falsifier:** no gain on R1 families (b)–(d) beyond R2, or gains only with retention loss.

**R4 — keep the predictor and the memory separate.**
- Use one code for next-byte prediction and a separate stationary code for memory addressing (the V37 dual-code contingency, never run).
- Never feed predictive features into the memory address (Step 5, the learned-h bridge).

---

## 3. Generalization (E3) — status: narrow positives only. This is the real frontier

**What worked:**
- **Rank and class transfer from a few taught pairs:**
  - Native V9.9A: +17.2 pp.
  - V9.9B: +9.6 pp.
  - A shared NLMS head on fixed KC codes, V9.10C: **77.5%** on never-taught rank pairs, persisting after interference.
  - Z0 E3 fixture: +38 pp.
  - FE0 single store: +21.9 pp.
- **Role coding helps:** ROLE code +34.4 pp vs FE0 +17.7 pp.

**What failed:**
- **Cyclic relations** (A>B>C>A): taught 97.6%, withheld 51%. V9.10E showed this is "pair-specific fitting without useful inductive bias". The oracle can solve it within the span of the taught examples, but ridge and label-sum cannot.
- **Arithmetic:** the native writer reached 0–2/32 on held-out operand pairs. Even a *supervised* GRU with 512 labels scored 5.5–10.9% (constant-zero level) on its test set. At the project's data scale this is a task and data problem, not just a local-learning one.
- **Pair-separating codes:** taught pairs recalled 90%, held-out transfer −6 pp (Steps 11–13).

**Diagnosis:**
- Every E3 success so far is *scalar*: a linear head learns a one-dimensional ordering, or class membership, that is shared across items.
- Structures that need **binding a value to a role** — "X is left of Y", the carry digit in addition, cyclic order — do not fit a linear readout over item features.
- Under the no-backprop, no-attention rules, structure must come from:
  - (i) representations that already factor roles and fillers;
  - (ii) composition of several local learners, where one learner's output becomes another's input;
  - (iii) iterative use of the store over time steps (recurrence), in place of attention.

**Proposed work, in order:**

**G1 — engineered positive control first (required by the 1+2 design and the E3 skill).**
- Before testing any native learner, confirm that each E3 task is solvable at the chosen data budget by a *known-capable non-native reference*. Examples: a symbolic or relational learner, and an NLMS head over oracle role-filler features.
- If the reference fails at the budget, the task is mis-sized. This is the arithmetic lesson: change the budget, not the learner.

**G2 — role-filler binding code + shared error-corrective head, on relations.**
- Representation: a fixed binding of each element to its role (first or second argument), built from the ROLE traces or a tensor-like sparse conjunction (KC conjunctions of role-PN × item-PN). The fly MB already computes sparse conjunctions.
- Learner: one shared NLMS head (V9.10C), trained locally.
- Tasks: rank (should pass, as before) → **cyclic** → an asymmetric relation with an exception.
- Controls: item-only code; a shuffled role assignment; lookup, frequency and no-prior-teaching models.
- **Pass:** cyclic held-out ≥ 75% (scalar ceiling 66.7%).
- **Falsifier:** cyclic stays near 50% with role codes, meaning a single linear head over a fixed code cannot represent it. Go to G3.

**G3 — composition by chaining local learners (recurrence instead of attention).**
- Learner 1 maps (item, role) → an intermediate code, for example the "successor" of an item.
- That output re-enters the sensory stream, and learner 2 maps the intermediate code to the answer.
- Each learner is a local delta rule. Credit is assigned because each stage has its own observable target: successors are observed in the stream. This is not backprop.
- Test: transitive and cyclic relations, then 1-digit addition expressed as repeated succession ("count on").
- **Falsifier:** if chained learners don't beat G2 on held-out relations, sequential composition is not providing structure. Report it and stop E3 work until the representation stage is solved on text.

**G4 — arithmetic only after G1–G3 pass.**
- Use the E3-design skill strictly: complete own bytes plus EOS, whole-relation holdouts (unseen operand pairs and orientations), the shortcut ladder (operand-marginal, nearest-neighbour, caches), sealed final sets.
- Set the data budget from the G1 reference, not from history.

---

## 4. How the stages connect (avoid serial-prerequisite drift)

- **Shared substrate:** one reference learner for all three stages. It combines:
  - the sparse superposed store (memorization);
  - a position-tolerant, stationary front end (representation, R2);
  - a separate predictor;
  - a shared error-corrective head over role codes (generalization, G2).
- **Stages can run in parallel on synthetic tasks.** They join only on natural text.
- Do **not** block G2 on R3, or R2 on the capacity law. Each has its own falsifier.
- **Every arm** gets a matched-budget control and a shuffled control (the V27/V37 standard), plus the "just make it bigger" control (`MINIFLY_WHY_STUCK` §5).

## 5. Priority list

| # | Experiment | Stage | Cost | Decides |
|---|---|---|---|---|
| 1 | Capacity law C(N,K), 4 seeds | Memorization | Low–medium (CPU hours) | Resource contract |
| 2 | R1 probe families on the scaled store | Representation | Low, read-only | Whether position binding is the bottleneck |
| 3 | G1 reference controls for rank/cyclic/1-digit addition | Generalization | Low | Whether tasks are well-sized |
| 4 | Label-free write allocation | Memorization | Medium | Capacity per byte; closes or opens gating |
| 5 | R2 n-gram front end | Representation | Medium | Position-tolerant identity |
| 6 | G2 role binding + NLMS head | Generalization | Medium | Whether relational structure is expressible |
| 7 | Natural-text joint test | All | Higher | Whether any of this survives real data |
| 8 | G3 chained learners, then G4 arithmetic | Generalization | Higher | Compositional generalization |

## 6. Confidence and gaps

- **Memorization:** high for the mechanism; medium for the scaling (one seed per cell).
- **Representation:** the surface-level result is reproduced. Semantic and position invariance is unknown; it was never tested on the scaled store.
- **Generalization:** the narrow positives are archival single-family results. Nothing beyond rank/class has worked, and the proposed G2/G3 have no in-project precedent.
- **Open owner decisions:**
  - The size contract.
  - Whether engineered fixed codes (lag, n-gram, role conjunction) are acceptable under the biological-plausibility constraint (C8).
  - Authorization for items 1–3.
