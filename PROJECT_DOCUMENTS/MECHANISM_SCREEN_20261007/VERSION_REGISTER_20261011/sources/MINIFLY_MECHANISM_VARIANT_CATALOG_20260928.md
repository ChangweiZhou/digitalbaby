# MiniFly mechanism and variant catalogue for the byte→Full151 platform

**Date:** 2026-09-28
**Compiled by:** Claude. Read-only compilation: no experiment was run for this document, and no frozen design, source lock or result was changed.
**Source-check correction:** Codex, 2026-09-28. The percentages affected by this correction were checked against their direct reports; the corrections distinguish prespecified verdicts from descriptive scores and historical implementations from current-platform qualification. This documentary revision did not change any experiment or frozen source.
**Status:** reference catalogue. It is **not** a run protocol and **not** a priority ranking.

## Sources

| Tag | Source |
|---|---|
| [50+ §…] | `MiniFly_50plus_Mechanisms_Pruned_Codex_Summary_2026-09-26.md`, section cited |
| [INV] | `MINIFLY_FIRST_ROUND_VARIANT_INVENTORY_20260928.md` (the current 16 variants) |
| [V2] | `MINIFLY_FIRST_ROUND_17_MECHANISM_PROGRAM_V2_20260928.md` |
| Byte-line reports | Cited per row by folder name. This correction rechecked the affected figures and verdicts against the direct reports. |
| Literature | §16, with primary links. Available full texts and a source/access manifest are in `MINIFLY_LITERATURE_20260928/`; an abstract-only entry is not a saved full paper. |
| Source-check record | `MINIFLY_MECHANISM_CATALOG_INDEPENDENT_AUDIT_20260928.md`; direct report/source qualifications are now included in the affected rows. |

---

## 0. How to read this catalogue

**The platform, stage by stage:**

```text
bytes → front end (88 PN) → fixed PN→KC expansion (5,177 KC, top-5% sparse)
      → Full151 native stores (fast/slow/adapt; KC→output writes)
      → readout (four-store argmax at "=", or L/R choice organ)
      ← teaching (arriving answer byte, or valence bit)
```

**Implementability codes (column "Impl"):**

| Code | Meaning |
|---|---|
| **READY** | Code has been exercised on the stated byte→Full151 platform. This describes implementation availability, **not** passage of a scientific gate; see each row's verdict. |
| **SOURCE** | Code exists in a historical or separate substrate, but a port to the current four-output/choice platform and its qualification have not been established. |
| **BUILD** | A new state or rule inside the existing platform. Needs an exact spec (equations, budgets, clocks) before coding. |
| **ARCH** | An architecture package: an extra store, route or population. Needs a capacity-matched baseline [V2 §3]. |
| **DIAG** | A diagnostic or oracle, not a deployable learner [50+ §12 rule 10]. |
| **OUT** | Outside the platform, or not a persistent-memory question [50+ §5.3]. |

**Evidence column:**
- Statuses and numbers are quoted from the sources.
- "Untested here" means no qualifying run on the stated byte→Full151 task is documented. A source-locked variant that is producing science receipts is marked **in execution**, not as a positive result.
- Lineage warning [50+ §10]: the BANC V50, FlyWire FULL_151 and BYTE_CORE V9.x lines are different organisms. Their scores are not directly comparable.
- E0 (code/interface), E1 (directly taught association), E2 (untaught presentation of a taught relation), and E3 (never-reinforced relation) are different evidence levels under `MINIFLY_OBJECTIVE_CONTRACT.md`. Scores from distinct levels or output organs do not constitute one integrated learner.

**Rules carried over from [50+ §0, §12]:**
- A failed exact variant does not close its family.
- CLOSED EXACT variants are not rerun unchanged.
- The learner never receives task IDs, labels, future relevance or evaluator identity.

---

## 1. Sensory front end (PN level)

This locus is **not** one of the five sites in [INV]. FE1 was added as F1 in [V2].

| ID | Variant | Operation | Impl | Evidence on this platform |
|---|---|---|---|---|
| FE0 | Receptor trace | 88-channel trace, τ = 10 s | READY | Wrapped relation test: held-out W−N +17.2 pp [+9.8, +24.6] (`BYTE_FULL151_E3_ASSAY_QUALIFICATION_20260928`). 16-fact recall 52.34% (Step 13); 49.6% in the bridge, where the spacing form scored 44.5% (`FULL151_VISIBLE_CONTEXT_BRIDGE_20260927`). |
| FE1 | Channel habituation | Input R[b]·(1−h); κ = 0.3, τh = 1,800 s | READY | **FEAS-1:** in-stream `ΔI(AUC)=+0.0506`; separate absolute time-stratified `η` is positive before 24 h and negative afterward **for both FE1 and FE0**. This is not a reversal of the AUC contrast. Fresh-key `ΔFK(AUC)=+0.0045`; delayed unchanged-return `R=−0.0120 [−0.0347,+0.0107]` unresolved; `NOT_SELECTED` (`BYTE_CORE_FEAS_1_20260923/results/REPORT.md`). **FE1 case V3:** `NO_DEMONSTRATED_INCREMENT`; relation θ −9.64 pp [−17.14,−2.13] in a one-store choice task; 16-fact recall +13.67 pp [+9.22,+18.12] over FE0 in a separate four-store task. These are not one integrated ability (`MINIFLY_FE1_CASE_V3_20260928/results/REPORT.md`). |
| FE2 | Temporal contrast | fast (5 s) − 0.5 × slow (30 s), rectified | SOURCE | BYTE-CORE-1 exploratory learner: η 0.0001, accuracy 0.5050 (n = 16). No current-platform qualification. |
| FE3 | Habituation plus temporal contrast | FE2 with FE1-style habituation | SOURCE | BYTE-CORE-1 exploratory learner: η 0.0002, accuracy 0.5065 (n = 16). No current-platform qualification. |
| FE-NI | Negative image | Subtractive, pattern-level; no decay in silence | SOURCE | FEAS-1: A1 −0.0006 [−0.0038,+0.0026], **FUTILE** in that assay; no current-platform qualification. It was the design's direct test of the "common-component" account. |
| FE-LAG | Engineering lag code | Last 12 bytes → 88 channels | SOURCE | FEAS-1: A1 +0.0279 [+0.0176,+0.0383], SELECTABLE there, never confirmed on the current platform. Engineering, not biological. |
| LAT4 | Latency classes | Four PN latency classes | SOURCE | V9 line: native byte-ORDER plasticity behaviour passed in 3/3 fresh worlds. The **prespecified** KC target decoder failed at 0.481. KC decoding 0.93–0.98 versus FE0 at most 0.68 came from a **post-hoc decoder refit** and still needs fresh-world confirmation (`BYTE_READ_ORDER_ITER2_20260925/REPORT.md`). |
| PAIR | Fading ordered pairs | FE0 plus hashed ordered-pair contacts, τ = 30 s | READY | Step 13 E1: 16-fact overall recall 90.04%, but weakest cue `1+1=` was 12/32; the **prespecified uniform per-cue gate failed**. Step 12 limited E3: relation W−N at old end +3.13 pp versus FE0 +23.96 pp. These are distinct assays. |
| ROLE | Role traces | FE0 plus "earlier byte" and "later byte" traces | READY | Step 15: E1 16-fact recall 66.4%, **below the frozen 70% gate**; limited E3 relation W−N +34.4 pp versus FE0 +17.7 pp in a separate one-store task. Not a qualified single-route compromise. |
| H_TRAINED / H_RANDOM | Learned GRU state | 64-d GRU `h` (next byte + 4-byte reconstruction), projected to PN; random-weight twin | READY (weights frozen) | E1 taught recall 73.0% / 57.8%; E2 spacing form 35.5% / 51.4%. The trained route **failed the prespecified full bridge gate** despite improved canonical recall (`FULL151_LEARNED_H_BRIDGE_V2_20260927/REPORT.md`). |
| CONTENT | Visible-context address | Hash of the last 4 visible bytes → 24 PN contacts. Falls back to FE0 below 4 visible bytes. | READY | E1 16 taught facts: 100%; two untaught E2 spacing/prefix forms: 100%, but both map to the taught key by design. At 32 distinct keys, E1 old-cue persistence after four new cohorts and a day was 99.22%; write-enabled versus no-new-write difference −0.78 pp [−1.30,−0.26] (`FULL151_VISIBLE_CONTEXT_PERSISTENCE_20260927/REPORT.md`). In the separate wrapped limited-E3 assay, final W−N was −1.0 pp [−8.2,+6.1]: **no reliable final transfer demonstrated, but zero transfer was not proven** (`BYTE_FULL151_E3_ASSAY_QUALIFICATION_20260928/results/REPORT.md`). The exact-key code does not deliberately preserve graded cue similarity. |

**Descriptive lesson from the FE1 case (hypothesis, not proven):** front-end state that changes between teaching and a delayed probe may move the KC address of a taught cue. The FE1 case's lessons file states this as a hypothesis, because cross-time addresses were not saved.

---

## 2. Input-side plasticity on fixed PN→KC adjacency (M10) [50+ §P1-C; INV P]

| ID | Variant | Operation | Impl | Evidence / status |
|---|---|---|---|---|
| P1 | Bounded Hebb | Co-activity changes existing PN→KC edge weights under a fixed weight/row budget | BUILD | Untested here [INV] |
| P2 | Oja | Hebb with a stabilising term, on the same adjacency and budget | BUILD | Untested here |
| P3 | Novelty-gated Oja | P2, updated only when local novelty is high. The novelty must not come from answers. | BUILD | Untested here |
| P4 | ITDP-like | Weight change by PN/KC timing difference. Not KC→output eligibility. | BUILD | Untested here. Original-list #2 marked P1. |
| E3dev | Local afferent development plus homeostasis | V24 `E3` | BUILD | History: "local sensory development was separated from privileged clean-target development" [50+ §6] |
| — | Norm gate | Input-scale/weight-budget control. Applies to **all** P variants [INV]. | control | — |
| — | Shuffled / rate-matched gate | Causal controls for the gated P variants | control | — |
| — | Variance salience | Task-specific selection | BUILD | Deprioritised as a generic rule [50+ §P1-C] |
| — | PHASE-FROZEN / PHASE-LIVE / LAG | Stationarity panel | DIAG | — |
| — | V24 E0 / E1 | Clean-target / observed-target backprop development | DIAG | Historical controls, not local candidates |
| — | V53/V54 INVARIANT-CUE, XOR-OFF, phase-Z, reliability learning | BANC V50 lineage | DIAG / BUILD | Lineage warning |

---

## 3. PN→KC topology (M6) [50+ §P1-C; INV T, S]

**Static: fixed at birth.**

| ID | Variant | Operation | Impl | Evidence / status |
|---|---|---|---|---|
| T1 | Hierarchical modular graph | Multi-level modules with limited cross-level edges. Module assignment is label-free. | BUILD | "Theoretical proposal, not validated" |
| T2 | Bounded-degree expander | Fixed degree bound with high neighbourhood expansion | BUILD | Source-locked and producing science receipts in the current R2/Z1/T2 round; **final scientific verdict pending**. Technical graph qualification is not a behavioural result (`MINIFLY_THREE_MECHANISM_ROUND_20260928/results/SOURCE_LOCK.json`, `MINIFLY_THREE_MECHANISM_ROUND_20260928/t2_spec.md`). |
| T3 | Sparse-bridge graph | Local clusters plus budgeted cross-cluster bridges | BUILD | Same |
| G0–G3 | A11 panel | Native, weight shuffle, 10% rewire, 50% rewire | BUILD | "Designed, not fully executed" |
| — | Support count / placement controls | [50+ §9 step 6] | BUILD | — |
| — | Topology-only transfer | Weights/state reset | DIAG | Historical |
| — | New-edge lesion | — | DIAG | Historical |

Each T variant needs its own random-graph control, matched in KC degree, edge count, weight histogram and readout sparsity [V2].

**Dynamic: real rewiring during life.**

| ID | Variant | Operation | Impl | Evidence / status |
|---|---|---|---|---|
| S1 | SET-like | Prune weak edges, regrow new partners from a fixed pool, keep the edge count | BUILD | "Proposed/open" |
| S2 | Co-activity guided | Prune/grow driven by PN–KC co-activity | BUILD | Same |
| S3 | Homeostatic | Prune/grow toward an activity or usage target | BUILD | Same |
| S4 | Sample then stabilise | Trial edges kept only after sustained local evidence | BUILD | Same |
| — | FIXED / RANDOM | Controls with matched rewiring counts | control | — |
| — | Slow bounded prune/grow `S` | — | BUILD | Open |
| — | Prune-only development; random regrowth | Historical | BUILD | — |
| — | Gradient-guided regrowth | Nonlocal | DIAG | — |
| — | Cosine-kNN rewiring / hubs / EMA routing | — | BUILD | "Early locality/causality issues" |
| — | Topology × commitment K0–K3 | — | BUILD | Only topology-blind K0 tested, as an exact prototype |
| — | Structural eligibility | M1/M6 boundary | BUILD | Open |

**Not rewiring** [50+ §12 rules 5–6]: `g_pnkc_rewire` (row/weight reassignment), `g_pnkc_weight` (weight perturbation), `g_support` (writable-support expansion).

---

## 4. KC sparsity and inhibition (M5) [50+ §P2-D, §7 #9, #23]

| Variant | Impl | Status |
|---|---|---|
| Hard/static top-k (current platform) | READY | CORE; "hard sparse coding remains useful" |
| Circuit-generated suppressive/decorrelating `A` | BUILD | Proposed |
| V48 scalar KC↔APL controller | — | **CLOSED EXACT** (negative) |
| Competitive top-response plasticity | BUILD | Open |
| Random-subset matched control; bottom-k control | control | — |
| Local normalisation / spillover controls | control | — |
| `g_sparsity` coding-fraction sweeps | BUILD | Historical evolution gene |
| Recurrent inhibition matched for mean activity | BUILD | Open |
| Homeostatic rebound via KC microcircuits (#23) | BUILD | P1/P2 boundary; use abstract resource control |

Stated trigger for this family: "only if topology/generalization studies show that adaptive inhibition can preserve relational geometry better than hard thresholding" [50+ §P2-D].

---

## 5. Write eligibility and recipient selection (M1) [50+ §P1-A; INV Z]

| ID | Variant | Operation | Impl | Evidence / status |
|---|---|---|---|---|
| A_ELIG | Multi-frame KC eligibility trace (OMNIBUS10) | Bridges the cue→consequence delay | BUILD (port) | "Robustly improved both old and new held-out loss" (other substrate) |
| — | `ELIG_TRACE_PORT` | Source-faithful port, with `ELIG_ADDR_SHUFFLE` as its control | BUILD | Designed, not run |
| Z1 | Coordinate-local activity/teaching trace | Bounded trace per writable KC→output coordinate | BUILD | Source-locked and producing science receipts in the current R2/Z1/T2 round; **final scientific verdict pending** (`MINIFLY_THREE_MECHANISM_ROUND_20260928/results/SOURCE_LOCK.json`). |
| Z2 | Coordinate-local conflict/resource gate | Uses local fast/slow conflict and available resource | BUILD | Untested here |
| — | STC tag + limited capture resource | Separate local tag and scarce consolidation resource | BUILD | P1 abstraction of STC/Orb2/mTOR/ERK stories |
| — | Evolved local rule `Z_i` | A **search method**, not a mechanism, until it yields a rule [INV] | — | Untested |
| — | Structural eligibility | Where structural change may occur | BUILD | Open |
| — | V69–V71 PROP/OPP/BALANCED receiver allocation | Restricted safe-witness solvers | DIAG | "No decisive deployable selector" |
| — | Local-policy refit | — | — | Failed validation |

Requirement added by V9.8: eligibility must keep useful overlap between related inputs, not just suppress overlap [50+ §P1-A].

---

## 6. Resource budgeting and metaplasticity (M9) [50+ §P1-D]

| Variant | Impl | Status |
|---|---|---|
| `pair_importance` | BUILD | Retention benefit, revision/acquisition cost |
| `pair_importance_strong` | — | Not to revisit unchanged |
| `neuron_availability` | BUILD | Similar benefit at lower state cost |
| `pair_importance_shuffled` | control | Address control |
| `bayes_stable` / `bayes_reopening` | BUILD | Partial retention, new-learning cost |
| `E_META` | BUILD | Old retention up, new acquisition "substantially worse" (OMNIBUS) |
| Adaptive allocation in FULL_151 | READY (in substrate) | Deleting it was harmful (Stage 3A) |
| Fixed/equal-budget controls | control | Showed some allocation gains were budget/gain effects |
| Plasticity quota / competition across time | BUILD | Proposed; must be causal online |
| `M` slow local resource state | BUILD | Open |
| Receiver availability / saturation states (V69–V71) | BUILD | Open |
| Mitochondrial budgeting | BUILD | Abstraction only |

This family is **not** in the current 16-variant inventory [INV] or the older 17-item V2 draft.

---

## 7. Teaching and error quality (control axis) [50+ §P1-E]

| Variant | Impl | Status / byte-line evidence |
|---|---|---|
| Ordinary local target write | READY | Current |
| Opponent teaching on the four-store bridge (target + 3 non-target writes) | READY | Current bridge policy |
| TARGET-ONLY vs ALL writes (V52) | READY | Byte bridge, Step 14: target-only 70.1% vs opponent 89.5%. Opposite direction to V52's BANC-line result; the lineages differ. |
| Normalised / error-based teaching | BUILD | "Large teaching-normalization gain in one late line" |
| Full-KC NLMS / global-error reference | DIAG | High ceiling; oracle |
| Consequence-specific DAN-like teaching | BUILD | Open |
| Delayed eligibility-gated teaching | BUILD | Links to §5 |
| B slow→fast baseline reassignment | BUILD | "Repeat positive evidence"; incumbent component |
| Class/role-masked write lesions | DIAG | — |

Rule [50+ §P1-E]: do not credit a memory mechanism for repairing a teaching defect.

---

## 8. Shared vs private memory and compartment heterogeneity [50+ §P1-B; INV R]

| ID | Variant | Impl | Evidence / status |
|---|---|---|---|
| — | Native shared REF-like state | SOURCE | V9.8 exploratory limited E3; V9.9A confirmed +17.19 pp on held-out pair choices in one synthetic relation family, and V9.9B confirmed +9.57 pp in a second. V9.9B's stronger graded-beyond-one-bit gate failed. These are **separate V9.x-lineage** results, not current four-output-platform qualification (`BYTE_CORE_V9_9A/REPORT.md`, `BYTE_CORE_V9_9B_RANK/REPORT.md`). |
| — | 4-bank / 8-bank, fixed or inferred routing | BUILD | History |
| — | FB2 / FB4 / FB8 isolated alpha stores | BUILD | "Capacity/isolation confounded" |
| — | GEOM4 | BUILD | "Expensive router" |
| — | HASH4 / HASH8 | SOURCE (V9.7) | "Not a general persistent-learning core after V9.8"; historical V9.x implementation, not a qualified port to the current platform. |
| H | Heterogeneous compartments | ARCH | Small number of classes with different learning/retention rules |
| — | Stable/flexible compartment split | ARCH | Conceptual |
| R1 | Novelty-gated private write | ARCH | Untested here [INV] |
| R2 | Prediction-error-gated private write | ARCH | Needs outcome feedback. Source-locked and producing science receipts in the current R2/Z1/T2 round; **final scientific verdict pending** (`MINIFLY_THREE_MECHANISM_ROUND_20260928/results/SOURCE_LOCK.json`). |
| R3 | Signed-residual private store | ARCH | Needs a rewritten (signed) teaching interface [INV] |
| — | BROADCAST8 | DIAG | — |
| — | Canonical-bank forced read | DIAG | — |

**Byte-line readout evidence** from saved single-store relation values, read-only:
- **Raw sum** FE0 + CONTENT: held-out W 50.52%, N 45.83%, so W−N +4.69 pp, against FE0 alone +17.2 pp [V2 §1]. The raw-sum writing arm was near chance; the positive difference partly reflects the depressed N baseline. This is an **offline recombination** of saved single-route values, not an integrated learner result.
- **Post-hoc 0.02 weight:** taught 97.40%, held-out W 66.15% vs N 49.48% (`BYTE_FULL151_BRIDGE_STAGE_EXIT_DECISION_20260928.md`). The weight was chosen after seeing the data, so these are exploratory offline readouts requiring fresh validation before any integrated claim.
- **V2 readout:** V2 fixes the R readout as a label-free, per-store scale calibration.

---

## 9. Selective forgetting and reopening (M2) [50+ §P2-A]

**Variants:**
- V23 `active_forgetting`
- `surprise_immediate`, `surprise_smoothed`
- `importance_surprise_release`
- `bayes_reopening`
- OMNIBUS `C_FORGET` (targeted wrong-winner decay)
- `CONTRACT_0.05` / global `R`
- `BR`, `KR`, `BKR`
- Selective downstream `R_sel`
- Homeostatic forgetting / targeted bidirectional extinction
- Rac1/Cofilin actin-remodelling abstraction
- Single-DAN acquisition/erasure duality
- D2-like anti-persistence gate

All are BUILD.

**Status:** the global unconditional variants are CLOSED EXACT as core mechanisms. `C_FORGET` is a trade-off, not a family-level negative. `R_sel` is open.

**Trigger:** "replicated evidence that stale slow state blocks revision or that a selective erase policy could separate obsolete from dormant-useful traces."

---

## 10. Latent memory and reinstatement (M3) [50+ §P2-B]

**Variants:**
- `silent_fixed`, `silent_inferred`
- `context_fixed`, `context_inferred`
- `P_REINSTATE` (correct / wrong / no reminder)
- K-probation evaluator reveal
- ACTIVE↔LATENT `L` state
- Recent-cache hard/soft retrieval controls
- Forgotten-but-stored engram abstraction

All are BUILD or DIAG.

**Status:** P2, open. "Banked stores work as storage, but reminder assays did not establish selective reinstatement."

**Trigger:** poor dormant return that a few renewed exposures rapidly restore.

---

## 11. Offline consolidation (M4) [50+ §P2-C]

**Variants:**
- Sleep-dependent DAN reactivation
- DPM-mediated consolidation
- Sleep/wake MBON microcircuit gating
- Circadian consolidation window
- Spaced-training disinhibition / pERK-like gating
- Bounded offline `O` without an episode archive
- Replay reservoir8/32; recent replay8/32; fastslow_replay8/32
- Immediate→intermediate→slow commitment
- Recurrent post-event consolidation window
- CREB/mTOR/ERK/Orb2 persistence, abstracted as commitment/capture

All are BUILD.

**Status:**
- Generic replay buffers are not preferred.
- One global K commitment clock is CLOSED EXACT.
- `B_CONSOL` was harmful in OMNIBUS.
- "Sleep solves persistence" is not currently supported.

**Trigger:** "delay-only branch loses transfer while interference-free rest produces reproducible improvement not explained by ordinary state recovery."

**Byte-line note:** in the persistence run, old-cue accuracy recovered from 97.17% to 99.22% over the final 24 h with no teaching (`FULL151_VISIBLE_CONTEXT_PERSISTENCE_20260927`). That report does not attribute the recovery to any mechanism.

---

## 12. Internal-state and motivational gating (M7) [50+ §P2-E]

**Variants:**
- Stubborn/adaptable heterogeneous populations
- V68 active-population reversal classifier
- `V` bounded causal internal-state learning gate
- Incremental long-vs-short information gate
- Octopamine/metabolic/sugar-state gate
- D1/D2 balance
- Cholinergic single-trial-LTM gate
- History/motivation dopamine integration
- Sleep/circadian modulator
- Stress/age state

All are BUILD.

**Status:** P2, deferred. "Information in state is not enough; the control law matters" (V68: AUROC about 0.951, but the chosen control policy harmed learning).

---

## 13. Readout and expression

This is not a separate family in [50+], but the integration question depends on it.

| Variant | Impl | Evidence |
|---|---|---|
| Four-store argmax at the `=` prompt | READY | Bridge and persistence runs (§1) |
| L/R two-option choice organ (single store) | READY | Step 10: 68.75% vs 46.88% on never-taught pairs (`FULL151_E3_OUTPUT_STEP10_INTERNAL_CHOICE_20260927`) |
| Four-store relation organ, v1 − v0 on channels 1/0 | BUILD | New in [V2]; not yet qualified |
| Calibrated two-store sum | BUILD | [V2 §3]; a new, unqualified hypothesis |
| Read authority `CORR2` / fixed mixtures | BUILD | V36–V49: "storage and expression are separable" |
| `TRACE-CORR` corroborated probation | BUILD | Useful in explicit-memory systems, "not a biological core result" |

---

## 14. Do not rerun unchanged [50+ §5]

**CLOSED EXACT:**
- Global K / THREE commitment
- Global R / `CONTRACT_0.05` as a complete core
- `B_CONSOL`
- `D_LOCAL`
- V48 scalar APL
- V34 four sparse compartments with local covariance
- `cascade8` (and `cascade4` as-is)
- Exact projection16/48 protected subspaces
- V19 additive consolidation bonus
- V23 feedback-consolidation
- Strong pair-importance
- Exact surprise-immediate/smoothed forgetting
- Exact hashing as the **entire** memory
- Unsigned linear MBON recurrence at the tested gain
- V20–V22 raw output-gain/input-addressing variants

**DEPRIORITISED:**
- Unbounded dictionaries and exact slots
- Replay archives as the default solution
- RLS / full covariance (keep as oracles only)
- Global orthogonalisation
- Attention / Sinkhorn / MoE composites
- Transformer teacher paths
- Learned global adapters that need hidden backprop
- Hash/tree/HNSW retrieval

**OUT OF SCOPE:**
- Ring attractors and navigation (LAL steering, UpWind integration, central complex)
- Direct motor-valence pathways
- Literal Arc capsid transport
- Literal gap-junction models, unless communication becomes the bottleneck

---

## 15. Mapping to the 16-variant inventory and the current executed subset [INV; V2]

| Programme item | Catalogue section | Status |
|---|---|---|
| F1 (FE1), outside the 16 | §1 | Tested in FE1 case V3: `NO_DEMONSTRATED_INCREMENT`; **excluded** from the current three-mechanism round. It belonged to the older 17-item V2 draft. |
| R1, R3 | §8 | Untested here |
| R2 | §8 | In current source-locked execution; final scientific verdict pending |
| Z1 | §5 | In current source-locked execution; final scientific verdict pending |
| Z2 | §5 | Untested here |
| P1–P4 | §2 | Untested here |
| T1, T3 | §3 | Untested here |
| T2 | §3 | In current source-locked execution; final scientific verdict pending |
| S1–S4 | §3 | Untested here |

The 16-item inventory is `R1–R3 + Z1–Z2 + P1–P4 + T1–T3 + S1–S4`. This is a **2026-09-28 source-check snapshot, not a live status table**. The source-locked experiment has three candidates (R2, Z1, T2) and their controls, not all 16. Its `SOURCE_LOCK.json` and committed world-arm receipts establish execution, **not effectiveness**; consult its final audited report for any later verdict.

**Families in [50+] with no representative among the 16** (descriptive, not a recommendation):
- M9 resource budgeting (§6)
- M5 sparsity and inhibition (§4)
- M2, M3, M4 and M7 (§9–§12), which are P2 families gated by their own triggers
- The teaching axis (§7), which is a control axis
- Readout (§13)

---

## 16. Literature scan: possibly omitted mechanisms (revised 2026-09-28 after reading)

### Method and access

- **Sources read:** abstracts, plus main text wherever it was freely available.
- **Saved copies:** obtainable open-access full texts, author manuscripts and preprints are in `MINIFLY_LITERATURE_20260928/`; consult its `SOURCES_MANIFEST.md` for exact files, primary URLs and access limits. A saved full text does not mean every section was reviewed.
- **Full text unavailable to this archive:** only the primary abstract was used, whether the publisher marks the article open access or not. Access status and exact failures are recorded in `MINIFLY_LITERATURE_20260928/SOURCES_MANIFEST.md`.
- **Read depth per item:**
  - **FT** = open full text available and its cited claims checked.
  - **AM** = PMC-hosted manuscript available and its cited claims checked.
  - **ABS** = abstract checked; full text not relied on for the cited claim.
- **2026-09-28 source check:** relevant claims in the newly saved PMC manuscripts for L2, L3, L6 and L7 were checked directly. This claim-specific check does not imply an exhaustive rereading of every section. Chan 2026 and Kropf 2026 remain abstract-only in this catalogue.
- **Coverage:** compares each item with [50+] §3–§7.
- **Relevance:** Claude's judgement of how each item touches a *known* MiniFly problem. The reason is given each time.
- **Sketches:** every "untested sketch" is a hypothesis, not a recommendation.

**Known MiniFly problems the relevance judgements refer to:**
- Separation vs sharing: recall against transfer, §1.
- Byte order: §1, LAT4 row.
- Integrating two routes: §8, §13.
- State changes between teaching and a delayed probe: the FE1 case.
- Learning without an external teacher: the long-term goal.

### Summary

| # | Mechanism | Read | Coverage in [50+] | MiniFly relevance |
|---|---|---|---|---|
| L1 | Learned cue becomes a teacher (second-order conditioning) | FT | Absent | Medium: **after prior externally reinforced first-order learning**, a learned cue may drive a second association; the resulting memory is transient. This is not teacher-free learning from the outset. |
| L2 | KC–KC axo-axonic suppression of plasticity | AM | Absent | Low–medium: binary top-k may already approximate it (unchecked) |
| L3 | Order-dependent sign of plasticity | AM | Partial | Low at the current byte timescale |
| L4 | Slow antagonistic signal (NO) and subtype-specific plasticity duration | FT | Partial | Low–medium: a recency compartment, not a persistence aid |
| L5 | KC classes differ in plasticity duration and in representation | FT + AM + ABS | Absent (KC side) | High: a biological template for keeping a separating and a categorising code in one system |
| L6 | Non-associative novelty/familiarity compartment | AM + ABS | Partial (novelty gate unspecified) | Medium: a concrete novelty source for R1/P3; familiarity lasts under an hour |
| L7 | Structured PN→KC sampling; input density sets overlap | AM + FT + ABS | Partial | High: directly about the discrimination vs categorisation trade-off |
| L8 | KC population clock | ABS | Partial | Medium: second-scale timing matches the byte clock |
| L9 | Cross-stream engram binding via DPM | FT (preprint) | Absent | High: structurally the two-route integration problem |
| L10 | Lateral inhibition plus spike-frequency adaptation (model) | FT (preprint) | Partial | Low: millisecond mechanisms, far below the byte clock |

---

### L1. A learned cue becomes a teacher: second-order conditioning

**Source:** Yamada et al., eLife 2023. [Link](https://elifesciences.org/articles/79042). FT: `Yamada2023_eLife_second_order_PMC9937650.*`

**What it shows** (adult flies, appetitive):
- The α1 compartment, "responsible for long-lasting appetitive memory", acts as the teacher.
- The cholinergic interneuron SMP108 "forms an excitatory pathway from MBON-α1 to DANs in other compartments". After pairing, "responses to the paired odor were selectively potentiated".
- Silencing SMP108 impaired second-order conditioning.

**Limits:**
- The resulting memory is short-lived: "second-order memory decayed within a day and was highly susceptible to extinction", and it "peaked at the third training and declined subsequently".
- Drosophila **larvae** showed no evidence of second-order conditioning or sensory preconditioning, though they did show conditioned inhibition. FT: `LearnMem2024_larval_higher_order_PMC11199949.*`

**Coverage:** absent as a mechanism. [50+] treats teaching as a control axis (§7; #25).

**Relevance:** medium. After an initial externally reinforced first-order association, it is a biological route for a *learned* value to generate later teaching without a new answer byte. The resulting second-order memory is transient; this does not establish wholly teacher-free learning.

**Untested sketch:** a store's learned value for a cue drives teaching writes to other stores.

**Impl:** BUILD. First check whether Full151's existing feedback pathway `m.F` (required per [50+ §2.2]) already carries an MBON→DAN-like route. **Not checked.**

---

### L2. KC–KC axo-axonic suppression of plasticity

**Source:** Manoim et al., Curr Biol 2022. [Link](https://www.cell.com/current-biology/fulltext/S0960-9822(22)01451-8). AM: PMC9613607.

**What it shows:**
- ">80% of local synaptic inputs to the KC axons" come from other KCs. Each γ KC synapses with about 190 other KCs, mostly within its own subtype.
- Through mAChR-B, active KCs suppress calcium and **dopamine-evoked cAMP** in neighbouring KCs.
- Dopamine alone raises cAMP in non-activated KCs; this suppression counteracts that.
- Knocking mAChR-B down made an **unpaired** odour aversive, i.e. non-specific learning.

**Limits:**
- RNAi gave no temporal control.
- Effects appeared mainly in γ KCs.

**Coverage:** absent. APL (§4) is global inhibition of activity, not lateral suppression of plasticity.

**Relevance:** low–medium. The mechanism removes writes at weakly or non-activated KCs. If Full151's writes are already gated by binary top-k activity, the platform may approximate it; this was **not checked**. Any added value would come with graded KC activity.

**Untested sketch:** a KC's write eligibility is reduced by its neighbours' activity. This is a Z-family variant.

**Impl:** BUILD. Needs a KC–KC neighbour graph, which was **not checked** in Full151.

---

### L3. The sign of plasticity depends on timing order

**Source:** Handler et al., Cell 2019. [Link](https://www.cell.com/cell/fulltext/S0092-8674(19)30611-7). AM: PMC9012144.

**What it shows:**
- In the γ4 compartment, forward pairing (odour before DAN) depresses and backward pairing potentiates.
- "Shifting the relative timing of an odor and reinforcement by <1 sec can switch the valence."
- DopR1 (Gαs → cAMP) acts as a coincidence detector; DopR2 (Gαq → ER Ca²⁺) responds to backward order.
- Associations reverse trial by trial across 50 trials.

**Limits:** the authors say this reversible plasticity "must co-exist with" mechanisms for longer-term memory.

**Coverage:** partial (#20; P2-A duality).

**Relevance:** low for current tasks. MiniFly bytes arrive about every 2.1 s (DT = 30/14 s) and the teacher always follows its cue, so sub-second order has no direct analogue. Its potential relevance is to revision (M2).

**Untested sketch:** a Z timing kernel with an order-dependent sign.

**Impl:** BUILD.

---

### L4. A slow antagonistic co-transmitter, and plasticity duration that differs by KC subtype

**Sources:**
- Aso et al., eLife 2019. [Link](https://elifesciences.org/articles/49257). FT: `Aso2019_eLife_nitric_oxide_PMC6948953.*`
- Yamada et al., J Physiol 2024. [Link](https://physoc.onlinelibrary.wiley.com/doi/full/10.1113/JP285745). FT: `Yamada2024_JPhysiol_cyclic_nucleotide_PMC10557778.*`

**What they show:**
- **Aso 2019:** nitric oxide from some DANs (PPL1-γ1pedc; PAM-γ5/β′2a) has an effect that "develops slowly, requires longer training than dopamine-dependent memory, and shortens memory retention".
  - It needs soluble guanylate cyclase (Gycβ100B) in KCs.
  - Knocking NOS down "prolonged the retention".
  - The authors' interpretation: memories become "specialized for predicting the value of odors based only on recent events".
- **Yamada 2024:**
  - cAMP-induced LTD at KC output synapses "additionally requires simultaneous KC activation".
  - cGMP paired with KC activation "induces slowly developing LTP".
  - At MBON-γ1pedc, LTD from γ KCs "lasted at least for 30 min", while α/β KC LTD returned to baseline "after ~10 min".

**Coverage:** partial (`H` heterogeneous compartments; fast/slow stores are CORE).

**Relevance:** low–medium. It builds a *recency* compartment, which shortens retention; that is the opposite of the persistence goal, but a candidate for the flexible side of a stable/flexible split.

**Untested sketch:** writes that carry a delayed opposite-sign component; plasticity duration set per KC subpopulation.

**Impl:** ARCH.

---

### L5. KC classes differ in plasticity duration and in representational structure

**Sources:**
- Yamada 2024 (above, FT): γ vs α/β LTD duration at the same MBON.
- Chan et al., Cell Reports 2026, ABS: "food-related odor representations formed by αβ and, in part, α′β′ KCs were separated from representations of odors with other ethological relevance"; γ KCs showed no such grouping; biased PN input to αβ explains this in models.
- Zheng et al., Curr Biol 2022, AM (claim checked against `Zheng2022_CurrBiol_structured_sampling_PMC9413950.*`): over-convergent, mostly food-responsive PN types "preferentially co-arborize and connect with dendrites of αβ and α′β′ KC subtypes".
- Bouzaiane et al., Cell Reports 2015, FT (author-hosted article archived and relevant claim checked): six discrete aversive-memory components; γ KC memory is retrieved by the M6 output pathway.

**Coverage:** absent as a KC-side split. [50+] compartments are output-side.

**Relevance:** high. In the fly, one KC population supports categorisation, through biased input, while another does not. That mirrors MiniFly's measured separation-vs-sharing conflict within a single fly.

**Untested sketch:** split the KC pool into sub-populations, each with its own PN→KC sampling rule (random vs group-biased), plasticity duration and readout.

**Impl:** ARCH. The split must be matched in total KC count and state.

---

### L6. A non-associative novelty/familiarity compartment

**Sources:**
- Hattori et al., Cell 2017. [Link](https://www.cell.com/cell/fulltext/S0092-8674(17)30479-8). AM: PMC5806120.
- Yamauchi, bioRxiv 2026, ABS: a model.

**What they show:**
- **Hattori 2017:**
  - MBON-α′3 responses to a novel odour fall ">50% by the second exposure" and ">80% within three to seven repetitions".
  - The drop is odour-specific.
  - It persists at 20 min but returns to novel levels "after 1 hr".
  - It needs coincident odour and PPL1-α′3 dopamine, with no reward or punishment.
- **Yamauchi 2026:** in kernel-perceptron models, this forgetting "is essential for reducing error when learning occurs within a brain of limited capacity".

**Coverage:** partial. Novelty appears as a *gate* (R1, P3), but no novelty source is specified.

**Relevance:** medium. It gives R1/P3 a concrete, label-free novelty signal.
- Familiarity recovers within about an hour, so across MiniFly's 24 h rests old cues would read as novel again.
- For a write *gate*, that means repeated private writes rather than the address mismatch seen with FE1. The consequence differs, but the timescale issue is the same.

**Untested sketch:** a fast, non-associatively depressing store whose output gates private writes.

**Impl:** BUILD.

---

### L7. Structured PN→KC sampling, and input density as the overlap control

**Sources:**
- Zheng 2022 (AM, claim checked): the structured PN–KC network degraded discrimination compared with a random network, "except when all signal flowed through the overconvergent, primarily food-responsive PN types".
- Chan 2026 (ABS, above).
- Ahmed et al., Curr Biol 2023 (AM, claim checked against `Ahmed2023_CurrBiol_input_density_PMC10529417.*`): "input density, but not cell number, tunes neuronal odor selectivity". Animals with more input per KC "show increased overlap in Kenyon cell odor responses and become worse at odor discrimination".
- Elkahlah et al., eLife 2020 (FT, `Elkahlah2020_eLife_sparse_wiring_PMC7028369.*`): "Kenyon cells produce fixed distributions of dendritic claws while presynaptic processes are plastic"; sparse odour responses survived reduced cell repertoires.

**Coverage:** partial (T1–T3, `g_sparsity`, variance salience).

**Relevance:** high.
- Biased sampling buys categorisation at the cost of discrimination, which is the MiniFly trade-off.
- Inputs per KC directly set response overlap, the variable behind crowding.
- The original catalogue did not incorporate the platform's PN→KC census. The current T2 technical specification checked the instantiated Full151 graph: `302×5177`, `27,572` edges, PN row degree `1–590`, KC column degree `0–17`, with `323` KCs receiving no PN edge (`MINIFLY_THREE_MECHANISM_ROUND_20260928/t2_spec.md`). This graph check is not evidence that T2 improves behaviour.

**Untested sketch:**
- T4: group-biased static sampling chosen by a label-free rule.
- The number of inputs per KC as an explicit, matched parameter of every T variant.

**Impl:** BUILD.

---

### L8. A KC population clock

**Source:** Kropf et al., Curr Biol 2026. [Link](https://www.cell.com/current-biology/fulltext/S0960-9822(25)01685-9). ABS only **in this archive**. The publisher marks it open access, but direct full-text retrieval from this host returned HTTP 403; see `MINIFLY_LITERATURE_20260928/SOURCES_MANIFEST.md`. Do not infer that the article is paywalled.

**What it shows:**
- The KC ensemble "generates a continuous representation of time from odor onset to odor offset and beyond", with neurons peaking at "odor-specific delays".
- Dopamine depresses concurrently active KC→MBON synapses, "creating notches of depression".
- Flies reproduce intervals of "several seconds".
- The authors call the mushroom body a "cerebellum-like adaptive filter".

**Coverage:** partial (PN-level temporal contrast/lag: FE2, FE3, FE-LAG).

**Relevance:** medium. The several-second timescale is comparable to a few MiniFly bytes (about 2.1 s each). PN-level contrast (FE2/FE3) showed no learner signal in BYTE-CORE-1, but this is a KC-level mechanism.

**Untested sketch:** heterogeneous KC time constants or delays that carry position within a line.

**Impl:** BUILD.

---

### L9. Binding two co-active KC streams through DPM (preprint)

**Source:** Okray et al., arXiv 2604.28007, April 2026. [Link](https://arxiv.org/abs/2604.28007). FT: `Okray2026_arXiv2604.28007_multisensory_engram.*`

**What it shows:**
- Visual input reaches separate KC populations (γd and αβp).
- Multisensory training improved memory "even when each sensory modality was tested alone".
- Visual γd KCs were needed for enhanced *olfactory* recall.
- The serotonergic DPM neuron forms compartment-specific branches that could "bridge" olfactory γm and visual γd KCs; APL makes synapses along those branches.
- **Required elements:**
  - DPM output during both learning and retrieval;
  - γ KC output during multisensory (but not olfactory-only) learning;
  - the 5-HT2A receptor in γd KCs;
  - a dopamine receptor in APL.
- Prior multisensory training also improved later learning with the same odour.

**Limits:**
- Preprint.
- The bridge is inferred from connectomics plus behavioural genetics.
- **Internal inconsistency:** the abstract names DopR1 in APL, but the main-text experiment knocks down **Dop2R**.
- The authors state that multisensory learning "involves different plasticity rules" from olfactory-only learning.

**Coverage:** absent. [50+] mentions DPM only for consolidation.

**Relevance:** high as a hypothesis. The preprint proposes a *possible* DPM-mediated circuit that binds co-active KC populations under reinforcement, supported by connectomics and interventions; it does not establish that bridge as the unique causal mechanism. The structural analogy to MiniFly's CONTENT + FE0 integration problem is an inference, not an empirical result about the model. In MiniFly both routes see the same bytes, so co-activation is automatic.

**Untested sketch:** reinforcement-gated excitatory bridges from one KC population to another, through a hub, formed only while a global-inhibition gate is released.

**Impl:** ARCH.

---

### L10. Lateral inhibition plus spike-frequency adaptation (model only)

**Source:** Li et al., arXiv 2510.21315, 2025. [Link](https://arxiv.org/abs/2510.21315). FT: `Li2025_arXiv2510.21315_LI_SFA.*`

**What it shows:**
- A spiking model of fly olfaction with a 5 ms lateral-inhibition trace and a 50 ms adaptation time constant, under Gaussian or OU input noise.
- Lateral inhibition helps at low and medium noise but "may reverse under higher-noise conditions".
- Adaptation helps at all noise levels.

**Coverage:** partial (§4).

**Relevance:** low. These are millisecond spiking mechanisms, while MiniFly steps about 2.1 s per byte, so there is no direct analogue.

**Impl:** OUT at the current timescale.

---

### Read and judged not to add a new MiniFly variant

- **Bouton-level plasticity heterogeneity** ([eNeuro 2023](https://www.eneuro.org/content/10/10/ENEURO.0275-23.2023); FT `eNeuro2023_bouton_plasticity_PMC10616905.*`). LTD is presynaptic and compartment-specific. Boutons strongly activated by the paired odour show calcium depression, while weakly responding boutons show potentiation. This is an activity-graded sign rule; it applies to MiniFly only if KC activity is graded rather than binary.
- **"Representation learning in cerebellum-like structures"** ([Rudelt et al., arXiv 2511.10261](https://arxiv.org/abs/2511.10261); ABS). A review: plasticity inside expansion layers matters, and the interplay of non-associative and associative plasticity "is not well understood". Supports keeping the P/S families; no new variant.
- **Fly-CL** ([arXiv 2510.16877](https://www.arxiv.org/abs/2510.16877); ABS). An ML method on "a nearly-frozen pretrained model", not a biological mechanism. OUT.

---

## 17. What this catalogue does not do

- It does not rank or recommend. Priorities, specs, controls and budgets belong to the frozen programme ([V2] and its successors).
- It does not change the conclusions of any completed experiment.
- The §16 sketches are hypotheses. Before any of them enters a programme, it needs its own evidence check: read the full prior literature and project reports, including controls that tested the claimed mechanism.
