# BYTE-CORE-FEAS-1 (v4) — Can a deployable, biologically motivated byte front end improve FULL_151?

**Namespace:** `BYTE-CORE-FEAS-1`
**Date:** 23 September 2026
**Status:** design for review. **Not locked. No code written. Nothing run.**

**Authority:** this is the only executable FEAS-1 design. `DESIGN_DRAFT.md` (v1), `DESIGN_v2.md` and `DESIGN_v3.md` are history and must not be executed.
**Implements:** `ROADMAP_20260923.md` §3.1, which was revised together with this document.

**Parents (read-only):**
- `BYTE_CORE_1_20260922`: the frozen round.
- `BYTE_CORE_1_ERRATUM_20260923`: post-hoc re-analysis, cited as "erratum §n".
  - §12 (day-probe η) and §13 (delayed retention) were added for this design.
  - 19 self-tests pass, and the frozen round is unchanged.

---

## R. Response to the 23 Sep review of v3

| # | Finding | Change in v4 |
|---|---|---|
| 1 | A2 (the decoupled contrast) cannot certify "cue-specific" learning. It cuts only the consequence-byte path. It does not exclude use of timing, presentation order, wrappers or other context, and it does not test dependence on learner writes. | **Claims narrowed to what each control establishes** (§1). **New timing-preserving cue-destruction control:** every decoupled arm also runs on the frozen `unpaired_world`, which destroys cue→consequence contingency and keeps timing, order, keys, wrappers, within-trial pairing and consequence frequencies. The difference-in-differences C (§8) is required for adoption. **Write dependence** is shown by argument, with a no-write check that has a positive control (H6). What remains unestablished, own-cue vs partner-cue learning under R2 pairing, is stated as a non-claim. |
| 2 | Adoption allowed a near-zero gain alongside a retention loss of up to 0.02 (≈ 40–45% of FE0's day-probe η). | **No loss margin anywhere.** Automatic adoption now requires the retention gain's CI to lie above 0. A confirmed gain whose retention is unresolved or worse is labelled as such and **goes to the owner**; it is never a default (§9.5). **The retention measure is re-chosen from data** (§8.3). The v3 day-probe measure hides a loss that frozen data show on delayed returns (erratum §13): FE1 −0.044 [−0.075, −0.013] (DEV) and −0.040 [−0.076, −0.004] (CAL). |
| 3 | Q12 required cross-world key disjointness, which fails by chance: ≈ 1.5 coincidences are expected. | **Replaced by a world-identity rule** (§4.2): seeds and world hashes are never reused; keys are unique within a world (frozen audit); cross-world key coincidences are counted and reported. |
| 4 | The S2 bridge would misattribute failure. | The roadmap bridge now tests learning for FE0 and the candidate separately. It separates confirmed failure from "not determined", fixes n with a precision argument, and pre-specifies a fallback for each case (roadmap §3.2b). |
| 5 | The candidate scope did not match the question. | The question now says **biologically motivated** (the owner's standing preference; constraint C8). FE-LAG is evaluated under **identical rules** as an engineering benchmark. Its status is always reported, and it shapes how `NO_CANDIDATE_SELECTED` is read (§9.6). |
| — | FE-NI's `read()` depended on a p̂ defined in `feed()`; empty-read counting sat inside `read()`. | `read()` recomputes everything from the current state and changes nothing. Empty reads are counted by an external observer (§6.2, Q3). |
| — | The A11 pointer was stale. | The roadmap points to A11 v3.2 (the newest file), which is not cleared to run. |

**Also changed.** **FE1-NI is dropped.** Frozen data show FE1's habituation carries a multi-day retention cost (erratum §13). FE1-NI inherits that habituation, and the factorial question it served does not decide the front end. It is deferred, not dismissed.

---

## 0. Acceptance contract

| Item | Contract |
|---|---|
| **Question** | Does one explicit, deployable, **biologically motivated** byte front end improve the frozen FULL_151 on R2/F3 under a prespecified readout, where the gain (a) survives removal of consequence information from the front end, (b) depends on the world's cue→consequence contingencies, and (c) also improves retention of unchanged pairs across ≥ 24 h? |
| **Decides** | Whether the byte line's default front end changes automatically. Outcomes that are not clean improvements go to the owner. It does **not** decide S2 readiness (roadmap §3.2b). |
| **Stream** | BYTE-CORE-1 world: V88 schedule plus amendment A10 (1,824 trials, ≈ 27 simulated days). R2 with reserved bytes 0xF1 / 0xF0. Frozen F3 rendering. Massed, clock ×1, episode 30 s, gap 135 s. Fresh namespaces (§4). |
| **Interfaces (§3)** | Front ends receive only (byte, time). `read()` is a pure function of state. The innate detector's event reaches only the frozen learner, identically in every arm. The code is taken when a detector byte arrives, before that byte is integrated (Q2). |
| **Learner** | FULL_151 `151ebd9e2d054ec9ec14`, frozen; adapter `S0-EPISODE`, unchanged; frozen reader Q. |
| **Stream conditions** | **I** intact. **CD**: the front end sees random substitutes for consequence bytes. **CD-UNP**: CD on the frozen `unpaired_world`. **NW** (calibration only): learner writes disabled. |
| **Arms** | **Reference:** FE0. **Candidates:** FE1, FE-NI. Reference and candidates each run under I, CD and CD-UNP. **Engineering benchmark:** FE-LAG, under I and UNP; its CD ≡ I at F3, verified in Q4. **Fixtures:** XR and NI-READLOCKED. |
| **Readout** | p(aversive) = σ(a·s + c), s = −v. One fitting procedure for every arm, on intact CAL lives, frozen before development. CD, CD-UNP and NW arms inherit their intact parent's readout. |
| **Decision quantities (§8)** | **A1:** intact Δη. **A2:** Δη under CD. **C:** contingency difference-in-differences in AUC, CD vs CD-UNP. **R:** Δη on delayed returns of unchanged pairs. All are per world, paired vs FE0. |
| **Automatic adoption** | Confirmed A1, A2, C and R lower bounds all > 0. Any other confirmed-gain pattern goes to the owner. |
| **Development** | 16 worlds, plus one pre-set extension of 16. Used only for selection. The screening line 0.02 applies to A1. |
| **Confirmation** | One frozen candidate on untouched worlds, with n fixed in advance (32–64). It runs automatically only if automatic adoption is reachable at that n (§9.4). |
| **Budget** | Always ≤ 288 lives. Ceiling ≤ 816 lives, a limit and not a target (§10). |

---

## 1. What this round can and cannot establish

Each claim is licensed only by the control named.

| Claim | Licensed by | Meaning |
|---|---|---|
| **Intact gain** | A1 | The complete strategy (front end + frozen learner + declared readout) predicts delivered consequences better than FE0 on the actual stream. |
| **Not via consequence bytes** | A2 | The gain remains when no consequence information reaches the front end. The learner still learns from the true consequences. |
| **Contingency-dependent** | C | The candidate's advantage in discriminating consequences is larger when cue→consequence contingencies exist than when they are destroyed. Timing, order, keys, wrappers, within-trial pairing and consequence frequencies are all preserved in the destroyed version. So the advantage does not come from any of those alone. |
| **Write-dependent** | Argument + H6 | Under CD, front-end input is independent of the true consequences. With writes disabled, the learner's response is a fixed function of front-end output. So a CD strategy without writes cannot beat chance except by accident, and any CD advantage must pass through learner writes. H6 checks this empirically, with a positive control. |
| **Retention not traded away** | R | Delayed (≥ 24 h) returns of pairs whose consequence has not changed are predicted better than with FE0. |
| **Not established** | — | Whether the learning is about the **current cue's own identity** or about its **paired partner**. Under R2 every trial pairs one aversive with one neutral cue, and a learner can use a partner's learned value. C keeps this "pair-structure" learning, because it too depends on contingency. Separating the two needs worlds with independent consequences (S2-ID at R8; roadmap §3.2). FEAS-1 does not claim it. |

---

## 2. Evidence this design uses

| Evidence | Value | Source |
|---|---|---|
| FULL_151 + FE0 learns above chance on F3, weakly | In-stream AUC 0.613 (no-learning 0.496). η 0.0047 with the frozen decoder; 0.0263 with the CAL-fitted bias decoder | erratum §2, PH-1 |
| Learned common offset | 98.9% of F3 events below the no-learning copy | erratum §3 |
| Exploratory in-stream gains (DEV, n = 16) | FE1 +0.0254 [0.0178, 0.0330] (SD 0.0143). FE-LAG +0.0295 [0.0235, 0.0355] (SD 0.0112) | erratum §5, §12 |
| **Retention differs by horizon** | 24-h day probes of recently acquired pairs: FE1 +0.0266 [+0.0034, +0.0498]. Delayed (≥ 24 h) returns of unchanged pairs (40 per life): FE1 −0.0438 [−0.0745, −0.0131] (DEV), −0.0396 [−0.0756, −0.0037] (CAL). FE-LAG +0.011 (DEV), −0.003 (CAL), both with CIs spanning 0 | erratum §12, §13 |
| Precision of R | Per-world SD of ΔR ≈ 0.043–0.082 | erratum §13 |
| FE0 on the retention subset | η 0.0138 [0.0065, 0.0212] (CON) | erratum §13 |
| Frozen unpaired control | FE0 F3 UNPAIRED: AUC 0.4948 | erratum §2 |
| Read timing in the frozen adapter | `present` locates the reserved byte, feeds bytes up to it, advances to its arrival, reads, teaches, then feeds the rest | `bytecore.py` 949–970 |
| Frozen world audit | Keys unique within a world; no cross-world check | `bytecore.py` `audit_world` |

**Context only (not acceptance controls under this readout):** frozen oracle and no-learning numbers.

---

## 3. Interfaces and causality

### 3.1 What each component receives

| Component | Receives | Never receives |
|---|---|---|
| **Front end** | (byte, time) for every byte, in arrival order | The detector event; record boundaries; identity, role, key position, wrapper id; tags; truth |
| **Innate detector** (fixed; part of the animal) | Each arriving byte | Anything else |
| **Learner** (frozen) | The detector event, and the front end's output at the detector moment | Truth, identity, role, wrapper id, tags |
| **Evaluator only** | Delivered and committed truth, identities, roles, tags, schedule | — |

### 3.2 Detector event (declared; inherited)

- It fires on the arrival of 0xF1 (US on) and of 0xF0 (US off), so once per record, neutral records included.
- This is BYTE-CORE-1's S0 convention (§3.3, §4.2, amendment A1). It is not a fly finding, since there is no detector for "no shock".
- It is shared identically by every arm, and only through the learner.

### 3.3 Read moment

- At the arrival time t_c of the detector byte, the front end is advanced to t_c and read **before** that byte is integrated.
- The frozen adapter finds that position by scanning the record. This is lookahead in code, not in the information used.
- **Q2** requires a byte-by-byte driver to reproduce the adapter bit for bit, in every arm. On each arrival the driver:
  1. lets the detector inspect the byte;
  2. if it fires, commits the snapshot from before the byte as the code, and teaches;
  3. feeds the byte.

### 3.4 Read purity

`read()` computes its output from the current state and modifies nothing: no cached quantities, no counters. **Q3** inserts extra reads at random times and requires every later output and the final state to be bit-identical. NI-READLOCKED (the v2 form) must fail Q3.

---

## 4. Stream and worlds

### 4.1 World generator and namespaces

Worlds come from the frozen `create_world(ns, block, index, "R2", "massed")`, seeded as `BYTE-CORE-1|<ns>|<block>|<index>`. UNP worlds come from the frozen `unpaired_world(w)`:
- whole trials' deliveries are permuted across trials;
- timing, order, keys, wrappers, schedule tags, within-trial pairing (one aversive per trial) and consequence frequency are unchanged;
- cue→consequence contingency is destroyed.

| Namespace | Block | Indices | Use |
|---|---|---|---|
| `FEAS1-ENG` | `Q` | 0–1 | Technical fixtures |
| `FEAS1-CAL` | `K` | 0–7 | Readout and NLMS-rate fitting; checks H2–H6 |
| `FEAS1-DEV` | `D` | 0–15, plus 16–31 for the extension | Development (selection) |
| `FEAS1-CON` | `V` | 0 … n_V − 1 | Confirmation. Generated only after `CONFIRM_PLAN.json` is hashed |

### 4.2 World identity (replaces v3's Q12)

1. **No reuse of seeds or worlds.** Namespace strings are new, so seeds are distinct by construction. Q12 verifies that every FEAS-1 world hash differs from every other FEAS-1 world hash and from every BYTE-CORE-1 world hash recorded in its job files.
2. **Within-world key uniqueness,** by the frozen `audit_world`, on every world.
3. **Cross-world key coincidences are counted and reported, not failed.**
   - Keys are 6 characters over 36 symbols (36⁶ ≈ 2.2 × 10⁹). With ~396 keys per world, about 1.5 coincidences with BYTE-CORE-1 worlds are expected by chance.
   - Such a coincidence carries no information. Every life starts from re-initialized front-end and learner state, no data or state passes between lives, and a learner never sees another world.

### 4.3 Format and life

- **Format:** F3 only, via the frozen `render_record` (14 bytes; wrapper alternating per presentation; 30/14 s spacing). At read time, a 12-byte window holds exactly the current context.
- **Every life:**
  - 3,648 scored consequence events;
  - 13 day checkpoints, 12 of them with relevant-pair probes (320 in total);
  - 40 delayed returns of unchanged pairs (erratum §13; the count is fixed by the schedule, test T10);
  - windows W1–W13 between checkpoints: 576, 288, 288, 288, 288, 288, 144, 336, 144, 360, 144, 360, 144 events.

---

## 5. Learner, adapter, stream conditions

**Learner and adapter (frozen).** Each record:
1. The front end is fed the context bytes.
2. At the detector moment the code is taken (§3.3) and encoded live with `encode_sparse`, and v is recorded as the prediction.
3. Teaching is one 30 s event with that code and punishment = [0xF1].
4. The remaining bytes are fed, then the 135 s gap follows.

The mutable learner state is 165,664 bytes.

| Condition | What changes | Everything else |
|---|---|---|
| **I** (intact) | Nothing | — |
| **CD** (consequence-decoupled) | The front end is fed c′ in place of each consequence byte. c′ is drawn i.i.d. uniform from {0xF1, 0xF0} from a world-seeded stream (`…\|cd\|…`) independent of schedule and consequences. | Detector, teaching, learner, newline and probes are unchanged; probes contain no consequence bytes. Byte frequencies are preserved. |
| **CD-UNP** | CD on `unpaired_world(w)`. | As CD. Contingency is destroyed; within-trial pairing is kept. |
| **NW** (CAL only) | Learner writes disabled (frozen `write=False`). | As its parent condition. |

**Expected, and checked:**
- FE-LAG: I ≡ CD in every `ev_*` and `pr_*` array. At F3, its window never holds a previous consequence at read time (Q4).
- FE0: per-world |η(I) − η(CD)| < 0.001, since the previous consequence's residue at the next read is e^(−16.5) ≈ 7 × 10⁻⁸ (Q4, H5).

---

## 6. Front ends

Every front end:
- receives only (byte, time);
- keeps all state in counted arrays;
- has no learned parameters;
- has a pure `read()`;
- is identical across worlds.

The receptor table **R** (256 × 88) is frozen: 6 channels per byte, amplitudes U[0.4, 1.0], seed `BYTE-CORE-1|RECEPTOR|v1`.

### 6.1 Arms

| Arm | Role | Definition | State | Basis (evidence label) |
|---|---|---|---:|---|
| **FE0** | Reference; frozen `bytecore.FE0` | p ← p·e^(−Δt/10 s) + R[b]; `read()` = p/max p | 88 floats | Hallem & Carlson 2006 (R) |
| **FE1** | Candidate; frozen `bytecore.FE1` | Input R[b]⊙(1 − h); per byte h ← clip(h + 0.3(1 − h)⊙R[b], 0, 1); h decays with τh = 1,800 s; p as FE0 | 176 floats | Short-term odorant-selective habituation (Das et al. 2011, C+) |
| **FE-NI** | Candidate; new | FE0 integration plus an exposure-driven negative image m (§6.2) | 176 floats | Long-term habituation through glomerulus-selective potentiated inhibition (Das et al. 2011, C+); negative-image proposal (Ramaswami 2014, T) |
| **FE-LAG** | Engineering benchmark; frozen `bytecore.FELAG`; evaluated under identical rules, never adopted (C8) | V51 lag code of the last 12 bytes; signed 8-contact adaptor into 88 channels; tanh; rectified; normalized | 12 bytes | Delay lines; not claimed biological |

### 6.2 FE-NI, exactly

**State:** p ∈ ℝ^88 (as FE0), and m ∈ ℝ^88 with m₀ = 0.

**`advance(t)`:** p decays with τp = 10 s. m does not change, so it does not decay in silence.

**`feed(b, t)`**, in order:
1. `advance(t)`.
2. Compute the sample p̂ = p / max p (the zero vector if p = 0).
3. m ← m + (p̂ − m) / N_m.
4. p ← p + R[b].

**`read()`**, pure, computed from the current p and m only:
1. p̂ = p / max p (the zero vector if p = 0).
2. o = max(p̂ − m, 0).
3. Return o / max o, or the zero vector if max o = 0.

**Empty reads.** An all-zero output yields an empty KC code, so no write occurs. Empty reads are counted by the **external recording wrapper**, which inspects returned vectors, never inside `read()`.

**Parameters** (derived, not tuned; Appendix A):
- N_m = 1,792 samples, i.e. 128 records × 14 bytes.
- **Self-cancellation:** 12 massed records of one pair contribute 9.0% of m.
- **Convergence:** 95% after ≈ 383 records, which falls within block 0 (576 records).
- **Subtraction weight:** 1.

**What m sees.** The exposure pattern at every byte arrival. One sample per record is the read-time pattern itself. One sample (at the newline's arrival) contains the consequence byte's drive, which CD neutralizes.

**Expected weakness, stated in advance.** Averaging over positions within a record under-cancels components that appear late. The separator, for example, is present in only 2 of 14 samples.

**Mechanistic contrast with FE1** (tested through R): FE1's habituation decays during rests, so its state at a delayed return differs from its state during learning. FE-NI's m does not decay in silence (§13).

### 6.3 Fixtures (never candidates)

| Fixture | Definition | Used in |
|---|---|---|
| **XR** (deliberate shortcut) | FE0 plus q ∈ ℝ^88, decaying with τ = 600 s. On arrival of a reserved byte, q += R[b] **after** the read. `read()` = normalize(p + q), pure. The previous consequence enters the next read with weight ≈ 0.76. It deliberately uses knowledge of the reserved-byte class. | H5, H6 (CAL) |
| **NI-READLOCKED** | v2's FE-NI: m is updated inside `read()` | Q3, which it must fail |

**Not candidates, and not dismissed:**
- **FE1-NI** (see §R);
- **the read-locked negative image**, which would need the detector event inside a front end;
- **the output-driven negative image;**
- **a sparser receptor table;**
- **kernels tied to slot geometry;**
- **FE-SURP.**

---

## 7. Declared readout

1. **Score:** s_e = −v_e.
2. **Model:** p_e(aversive) = σ(a·s_e + c).
3. **Fitting data:** all scored events of the arm's 8 intact CAL lives, pooled. Each life has 3,648 events, so worlds carry equal weight.
4. **Fitting method:**
   - maximize the mean Bernoulli log-likelihood (natural log) against the delivered byte;
   - the problem is convex, so it is solved by Newton's method (IRLS) from (0, 0) until max |gradient| < 10⁻¹⁰;
   - inside the objective only, probabilities are clipped to [10⁻¹², 1 − 10⁻¹²];
   - one function serves every arm.
5. **Freeze:** (a, c) goes to `READOUT.json`, hashed before any DEV world is generated.
6. **Inheritance:** CD, CD-UNP, UNP and NW arms use their intact parent's (a, c).
7. **Use:** every η uses the frozen (a, c) with clip ε = 10⁻³. This covers in-stream events, retention returns, windows, day probes and the final probe; nothing is recalibrated.
8. **No online correction.** A readout that loses usability as the learner changes is a property of the strategy.

The CODE-NLMS learning rate (reference only) is chosen in the same phase over {0.1, 0.3, 1, 2}.

---

## 8. Outcomes

### 8.1 η (BYTE-CORE-1 §5.2)

- For each scored item: p̃(y) = clip(p if y is aversive, else 1 − p; ε, 1 − ε).
- Bits saved: s = log₂ p̃(y) + 1. Relevant information: i = 1.
- η = Σs / Σi, per world.
- In-stream truth is the **delivered** byte, anomalies included.
- A Δη of 0.02 is 2 bits saved per 100 bits of relevant consequence information. It is not 2 percentage points of accuracy.

### 8.2 Decision quantities (per world, paired vs FE0 on identical worlds)

| ID | Definition | Currency | Why this currency |
|---|---|---|---|
| **A1** | η(c, I) − η(FE0, I), over all scored events | η | The owner's metric on the actual stream |
| **A2** | η(c, CD) − η(FE0, CD) | η | Same metric, with consequence input to the front end removed |
| **C** | [AUC(c, CD) − AUC(FE0, CD)] − [AUC(c, CD-UNP) − AUC(FE0, CD-UNP)] | AUC | UNP lives run under readouts fitted on contingent worlds, so their η carries readout mismatch (erratum §2 caveat); AUC is readout-free |
| **R** | η(c, I) − η(FE0, I) over the 40 delayed returns of unchanged pairs per life (§8.3) | η | Persistent learning, in the owner's metric |

### 8.3 Why this retention measure

- **R is the owner's quantity on retention events.** It uses the in-stream, prequential η with the same readout and the same truth, restricted to delayed demand. The subset is cues returning ≥ 24 h after their previous presentation whose consequence has not changed: evaluator tags `stable` and `react`.
- **Excluded events measure flexibility, not retention.** These are revisions, anomalies and misleading reminders, where the old memory should be wrong. They remain inside A1.
- **The v3 measure hid a trade-off.** v3's measure was 24-h day probes of recently acquired pairs. In frozen data it shows FE1 **gaining** (+0.027), while delayed returns show FE1 **losing** (−0.044 and −0.040 in two panels; erratum §12, §13).
- **Day probes stay as a reported horizon.** They are read-only, use a readout that over-predicts on probes (mean p ≈ 0.6), and cover only the 1-day horizon of new pairs.
- **Selection uses evaluator tags only.** Nothing reaches the learner.
- **Precision is low.** With 40 events per life, the per-world SD of R is ≈ 0.04–0.08, so a retention gain is confirmable only if it is large (§9.4). The design states this rather than substituting a margin.

### 8.4 Reported, with no decision role

| ID | Quantity |
|---|---|
| R1 | In-stream AUC (discrimination; not calibration) |
| R2 | Native paired accuracy |
| R3 | Retention profile: η and AUC on returns stratified by time since last presentation (< 1 h, 1–24 h, 24 h–7 d, ≥ 7 d) and by tag; 24-h day-probe η and AUC; final 7-day probe η and AUC |
| R4 | Windows W1–W13 and day checkpoints: η, observed aversive frequency, unconditional mean p, AUC, n |
| R5 | Offset share (learned minus no-learning copy below 0), per window |
| R6 | Path(x) = η(x, I) − η(x, CD); the split of A1 into A2 and the remainder; the UNP terms of C |
| R7 | Side-1 first-presentation AUC (descriptive; C decides) |
| R8 | CODE-NLMS reference on the same codes |
| R9 | Live interface statistics, from the external wrapper: active KCs; Jaccard between consecutive presentations of a cue; Jaccard between the two cues of a trial; empty reads |
| R10 | Resources: front-end state bytes, learner mutable bytes, wall time |
| R11 | F4 day probes: η and AUC |

---

## 9. Decision rules

**Statistics.**
- The world is the unit; arms are paired on identical worlds.
- For each contrast: per-world differences, their mean and SD, and a 95% two-sided t interval with df = n − 1.
- Development is selection, so there is no multiplicity adjustment. Confirmation tests one frozen candidate.

### 9.1 Development (n = 16)

For each candidate c ∈ {FE1, FE-NI}:

| Status | Condition |
|---|---|
| **SELECTABLE** | A1 mean ≥ 0.02 **and** A1 lower > 0 **and** A2 lower > 0 **and** C lower > 0 **and** R mean ≥ 0 |
| **TRADE_OFF_IN_DEVELOPMENT** | A1, A2 and C as for SELECTABLE, but R upper < 0. **Not confirmed automatically;** reported to the owner |
| **FUTILE** | A1 upper < 0.02, or A2 upper < 0, or C upper < 0 |
| **INCONCLUSIVE** | Anything else |

FE-LAG is evaluated with the same rules. For it, A2 ≡ A1 (Q4), and C uses FE-LAG's I and UNP arms. Its status is reported as an engineering benchmark and never enters selection.

### 9.2 Extension (still development)

- **Runs only if** no candidate is SELECTABLE at n = 16 and at least one is INCONCLUSIVE.
- **Covers** FE0 and every INCONCLUSIVE candidate (I, CD, CD-UNP) on DEV 16–31.
- **Afterwards,** §9.1 is re-applied at n = 32. Anything still INCONCLUSIVE becomes **NOT_SELECTED**.
- There is no second extension.

### 9.3 Selection

- Among SELECTABLE candidates, choose the one with the **largest A2 lower bound**. A tie goes to the smaller state.
- The selection is reported as: "intact gain supported, estimate meets the screening line 0.02; gain survives consequence decoupling; contingency-dependent; retention estimate not negative". It is never reported as an established gain of any size.
- `SELECTION.json` is hashed.
- If nothing is selected, the outcome is `NO_CANDIDATE_SELECTED` (§9.6).

### 9.4 Confirmation plan (fixed before worlds exist)

**Size.** n_V is the smallest n ≥ 32 that gives ≥ 95% power for each of A1, A2 and C to have a lower bound > 0, capped at 64:

n ≥ ((t₀.₉₇₅,ₙ₋₁ + t₀.₉₅,ₙ₋₁) · SD / Δ_plan)², iterated in n.

- **SD** is the development per-world SD at the final development n.
- **Δ_plan** is half the development mean (a guard against selection bias).
- Joint power across A1, A2 and C is ≥ 0.85 by the union bound.
- If n_V > 64, the label is `NOT_CONFIRMABLE_THIS_ROUND`, and nothing runs.

**Worked example (A1 only):** with SD 0.0143 and Δ_plan 0.0127, the formula gives n ≈ 18, so n_V = 32.

**Reachability of automatic adoption.**
- Compute the power, at n_V, for R's lower bound to exceed 0, planning at half the development mean of R (0 if that mean is ≤ 0).
- **If that power is ≥ 0.80,** confirmation runs automatically.
- **Otherwise** the runner writes `CONFIRM_PLAN.json` with status `AWAITING_OWNER` and stops. Automatic adoption would be unreachable, so a confirmation could at best settle the gain and bound the retention change, and spending on that is the owner's decision.
- **Example:** with SD_R = 0.05 and n_V = 64, power 0.80 needs Δ_plan ≳ 0.018, i.e. a development mean R ≳ 0.036.

`CONFIRM_PLAN.json` records the candidate, n_V, the power per condition, the readout hash and the six arms (candidate and FE0, each under I, CD and CD-UNP). It is hashed before any CON world is generated.

### 9.5 Confirmation outcome classes (95% CIs; one candidate)

Let **G** mean: A1 lower > 0, A2 lower > 0 and C lower > 0.

| Class | Condition | Consequence |
|---|---|---|
| **`IMPROVES`** | G and R lower > 0 | The byte line's default front end changes automatically. It must still pass the S2 bridge |
| **`GAIN_RETENTION_UNRESOLVED`** | G, and R's CI contains 0 | **Owner decision.** The report states that the data allow a retention change anywhere between R lower and R upper |
| **`TRADE_OFF`** | G and R upper < 0 | **Owner decision.** Not a default |
| **`GAIN_NOT_ATTRIBUTABLE`** | A1 lower > 0, but A2 or C lower ≤ 0 | Not adopted. The report names which attribution failed |
| **`NOT_CONFIRMED`** | A1 lower ≤ 0 | Not adopted |

Wording is always added, reporting each estimate against the screening line and giving A1's lower bound. There is no loss margin in any class.

### 9.6 Round-level readings

- **`NO_CANDIDATE_SELECTED`** means no biologically motivated candidate in this set met §9.1.
  - If FE-LAG **would have been** SELECTABLE, the report says that a deployable byte/time front end can meet the rules, and that the biologically motivated candidates are what fall short.
  - Promoting an engineering front end would be the owner's decision; constraint C8 allows it only as a labelled reference.
  - If FE-LAG also fails, the rules are not met by any tested deployable front end.
- **`TRADE_OFF_IN_DEVELOPMENT`** goes to the owner with the development evidence. It is not confirmed automatically.

---

## 10. Phases, roster, budget

| Phase | Arms × worlds | Lives | Runs |
|---|---|---:|---|
| **Q** Fixtures | Q-roster below | 24 | Always |
| **K** Calibration and instrument checks | Intact FE0, FE1, FE-NI, FE-LAG × CAL 0–7 (32). XR under I, CD and NW (24). FE0 under CD (8). FE0, FE1 and FE-NI under CD + NW (24) | 88 | Always |
| **D** Development | FE0, FE1 and FE-NI, each under I, CD and CD-UNP (9 arms); FE-LAG under I and UNP (2) × DEV 0–15 | 176 | Always |
| **X** Extension | FE0 and ≤ 2 INCONCLUSIVE candidates, each under I, CD and CD-UNP × DEV 16–31 | ≤ 144 | Only under §9.2 |
| **V** Confirmation | Candidate and FE0, each under I, CD and CD-UNP × CON 0 … n_V − 1 | 192–384 | Only under §9.4 |

**Q-roster (24 lives):**

| Lives | Checks |
|---:|---|
| 9 | **Q1 identity:** FE0 and FE1 through the FEAS path and the frozen path, on ENG 0–1 (8). Plus 1 failing variant. |
| 2 | **Q1 UNP:** FE0 on `unpaired_world`, through both paths. |
| 8 | **Q2 causal driver:** FE0, FE1, FE-NI and FE-LAG (4). Frozen-path runs for FE-NI and FE-LAG (2). Two failing variants (2). |
| 3 | **Q4:** FE-LAG under CD (1); FE0 under CD on ENG 0–1 (2). |
| 1 | **Q6:** failing probe variant. |
| 1 | **Q11:** empty-read positive control. |

Q3 drives the front ends alone and costs no lives.

**Totals.**

| Case | Lives | CPU at 4–9 s per life |
|---|---:|---|
| Always | 288 | ≈ 19–43 CPU-min |
| With extension | ≤ 432 | — |
| Ceiling | ≤ 816 | ≈ 54–122 CPU-min |

Wall time is minutes with 8 workers.

**Growth over v3** (232 always, ≤ 616): CD-UNP arms throughout, calibration-phase NW checks, and FE-LAG's UNP arm. Dropping FE1-NI offsets part of it. BYTE-CORE-1 ran 3,856 lives.

---

## 11. Instrument checkpoint (after Q and K; before any DEV world)

Any halt writes `CHECKPOINT.json` and stops the runner. No DEV world is generated until a written amendment has been reviewed.

| ID | Halt if |
|---|---|
| H1 | Any Q check fails, or any negative or positive control misbehaves. |
| H2 | A readout does not transfer across CAL halves. The test is a 2-fold cross-fit (worlds 0–3 ↔ 4–7). It fails for any intact arm if either condition holds: \|held-out η − in-sample η\| > 0.01, or held-out mean p lies outside 0.50 ± 0.05. |
| H3 | FE-NI empty-read rate on CAL > 1%. |
| H4 | FE0's CAL in-stream η has lower bound ≤ 0 (frozen CAL: 0.0243). |
| H5 | **Decoupling fails.** Either Path(XR) has lower bound ≤ 0 with n = 8 (the known path is not removed), or any CAL world has \|Path(FE0)\| ≥ 0.001 (an arm without a path is disturbed). |
| H6 | **The write-dependence check fails.** Either the positive control fails (XR under NW, intact stream: the CI of AUC − 0.5 includes 0), or any of FE0, FE1 and FE-NI under CD + NW has a CI of AUC − 0.5 that excludes 0 (an unexpected write-free predictive path). |

---

## 12. Qualification (phase Q)

- **Negative control:** a defect the check must catch.
- **Positive control:** a known effect a diagnostic must register.
- Results are computed from archived runs; no flag is a literal.

| ID | Check | Negative control | Positive control |
|---|---|---|---|
| Q1 | **Execution identity.** The FEAS path (in-memory arm registration, learner factory, recording wrapper, condition switches off) reproduces the frozen `run_life` bit for bit (all `ev_*` and `pr_*` arrays, final digest): FE0 and FE1 on ENG 0–1, and FE0 on `unpaired_world` | A wrapper that adds one active KC must break identity | — |
| Q2 | **Causal read moment** (§3.3): the byte-by-byte driver ≡ the frozen adapter, for FE0, FE1, FE-NI and FE-LAG on ENG 0 | (a) Snapshot after integrating the detector byte; (b) snapshot at the previous byte's arrival. Each must break identity | — |
| Q3 | **Read purity.** Each front end (FE0, FE1, FE-NI, FE-LAG, XR) is driven alone on ENG 0's byte stream. 1,000 extra `read()` calls at random times leave all scheduled outputs and the final state bit-identical, and the wrapper's empty-read counter is unchanged by extra reads | NI-READLOCKED must fail | — |
| Q4 | **CD and UNP machinery.** Re-seeding substitutes changes them but not the world hash; the substitute frequency is within binomial limits. Detector and teaching logs match between I and CD. FE-LAG I ≡ CD in all `ev_*`/`pr_*` arrays (ENG 0). FE0 per-world \|Δη\| < 0.001 (ENG 0–1). `unpaired_world` keeps within-trial pairing, keys, tags and timing (structural diff) | A CD variant feeding the true byte must fail independence. An UNP variant that permutes sides within trials must fail the pairing check | Path(XR) > 0 (H5) |
| Q5 | **No annotation reaches a front end.** New classes reject `set_oracle`; the learner accepts only byte records | An FE-NI variant exposing `set_oracle(key_offset)` must be flagged | The frozen FE-TAIL4 is flagged |
| Q6 | **Probes are read-only.** The persistent digest (learner, plus front-end p, h, m, q) is unchanged across every probe | A probe that reads the live front end must raise | — |
| Q7 | **Readout fitting.** Refuses non-CAL worlds; a shuffled-label fit gives η ≤ 0.002 on ENG | Injecting a DEV world must raise | Reproduces the erratum fit on frozen CAL FE0 F3 data (a = 0.06543, c = −0.8256) within 10⁻³ |
| Q8 | **Metric and decision fixtures.** η, AUC and paired accuracy on known inputs. The development-status and confirmation-class functions return pre-computed labels on synthetic per-world inputs, covering every class and boundary | A constant-offset synthetic stream gives flat window means | A planted drift is tracked by the window means |
| Q9 | **R7 detector** (descriptive) | Frozen FE0 F1, FE-TAIL4 F3 and FE-ID F3 (CON) stay quiet | Frozen V50-REF F1 (0.759) and FE3 F3 (0.569) fire |
| Q10 | **Retention and day-probe measures.** On frozen files, reproduce erratum §12 (FE1 − FE0 day-probe Δη +0.0266, DEV) and §13 (FE1 − FE0 R −0.0438, DEV; 40 events per life) | Shuffled labels give η ≤ 0 | The reproductions themselves |
| Q11 | **Empty reads.** FE-NI < 1% on ENG | — | A variant with m₀ = 1 on every channel must exceed 1% |
| Q12 | **World identity (§4.2).** All FEAS-1 world hashes are distinct from each other and from every BYTE-CORE-1 world hash; `audit_world` passes; cross-world key coincidences are counted and reported | A world built with a reused BYTE-CORE-1 namespace and index must be rejected. A world with a duplicated key must be rejected by `audit_world` | — |
| Q13 | **Source lock** over new files, hashes of the frozen imports, and the frozen `SOURCE_LOCK.json` | An appended comment must raise | — |
| Q14 | **Frozen folders unchanged** (before and after each phase) | A one-byte change in a scratch copy must be detected | — |
| Q15 | **Output manifest** | Removing an output must fail packaging | — |

---

## 13. Predictions, written before any run

| Arm | A1 | A2, C | R and other quantities |
|---|---|---|---|
| FE0 | — | — | In-stream η ≈ 0.026; R-subset η ≈ 0.014; day-probe η ≈ 0.045; Path ≈ 0 |
| FE1 | ≈ +0.025 (frozen) | A2 ≤ A1. C > 0 expected: its intact AUC gain is ≈ 0.05 | **R ≈ −0.04** (frozen −0.044 DEV, −0.040 CAL). Expected status **`TRADE_OFF_IN_DEVELOPMENT`**: not confirmed automatically, owner decides |
| FE-NI | Positive if the common-component account holds; no magnitude predicted | A2 ≈ A1 (the consequence sample in m is 1 of 14 per record). C > 0 if A1 > 0 | R not below FE0. **If R < 0 as for FE1, the rest-recovery account of FE1's loss (§6.2) is contradicted.** Offset share below FE0's 0.99 |
| FE-LAG (benchmark) | ≈ +0.03 (frozen) | A2 ≡ A1; C unknown | R ≈ 0 (frozen +0.011 DEV, −0.003 CAL, CIs spanning 0), so likely **INCONCLUSIVE** on R |
| XR | — | — | Path(XR) > 0. Under NW on the intact stream, AUC ≠ 0.5 |
| CD + NW arms | — | — | AUC ≈ 0.5 |

**Falsifiers for FE-NI's rationale.**
- The offset share falls toward 0.5 while A1 and A2 have upper bounds ≤ 0. Then the offset is a symptom, not the cause.
- R < 0 with a CI excluding 0. Then no-decay adaptation does not protect retention.

---

## 14. Limits of any result

- **Complete strategies.** Results say nothing, taken alone, about the information content of a code.
- **Own-cue vs partner-cue learning is not separated** (§1).
- **Scope:** R2/F3, frozen FULL_151, this world family, biologically motivated candidates. No claim about other rungs, formats, V50 or general raw-byte learning.
- **Not S2 readiness** (roadmap §3.2b).
- **Mechanism readings are observations:** offset, windows, NLMS, and FE1 vs FE-NI on R.
- **The detector convention** (§3.2) is shared by all arms and is not a biological claim.

---

## 15. Implementation plan (after review; no code now)

**Files, all new, in `BYTE_CORE_FEAS_1_20260923/`:**

| File | Contents |
|---|---|
| `feas1.py` | FE-NI, XR and NI-READLOCKED classes. A learner factory around the frozen `F151Learner`. The external recording wrapper, which reads returned codes and outputs and counts empty reads. Condition switches for CD, UNP (via the frozen `unpaired_world`) and NW (via the frozen `write=False`). In-memory registration of arm specs in the imported `ARMS` dict, so the frozen `run_life` runs unchanged (disclosed; Q1). |
| `feas1_readout.py` | IRLS fitter, CAL-only guard, cross-fit (H2). |
| `feas1_analyze.py` | §8 metrics, §9 rules, report. |
| `feas1_qualify.py` | Q1–Q15, including the byte-causal driver and the read-purity harness. |
| `run_feas1.py` | Phase runner. It is resumable, writes each gating file before the next phase, writes a completion marker, and packages the results itself. |
| `package_feas1.py` | Idempotent packaging, with the manifest check. |

The frozen `bytecore.py` is imported read-only, with `sys.dont_write_bytecode` set and snapshot checks (Q14).

**Gating sequence:**

```
Q → QUALIFICATION.json
K → READOUT.json (hashed) → CHECKPOINT.json (H1–H6; halt here if any)
D → SELECTION_D16.json (hashed)
[X → SELECTION_D32.json (hashed)]                      only under §9.2
SELECTION.json (hashed)
CONFIRM_PLAN.json (hashed; status RUN or AWAITING_OWNER)
[V]                                                   only if RUN
DECISION.json → REPORT.md → MANIFEST.json → package
```

**Outputs (`results/`):**
- **Gating files:** `QUALIFICATION.json`, `CHECKPOINT.json`, `READOUT.json`, `SELECTION_D16.json`, `SELECTION_D32.json` (if run), `SELECTION.json`, `CONFIRM_PLAN.json`.
- **Cells and contrasts:** `DEV_CELLS.csv`, `DEV_CONTRASTS.csv` (A1, A2, C, R, Path), `CONFIRM_CELLS.csv` and `CONFIRM_CONTRASTS.csv` (if run).
- **Profiles and diagnostics:** `RETENTION_PROFILE.csv`, `WINDOWS.csv`, `CHECKPOINTS.csv`, `XRECORD.csv`, `NLMS.csv`, `LIVE_INTERFACE.csv`, `WORLD_IDENTITY.csv` (including key coincidences), `RESOURCES.csv`.
- **Decision and package:** `DECISION.json`, `REPORT.md`, `MANIFEST.json`, `qualification/`, `jobs/`.

**Environment of record:** V88 venv (Python 3.11.5, numpy 2.2.6, scipy 1.14.1, numba 0.61.2).

---

## 16. Known risks

| Risk | Handling |
|---|---|
| R is imprecise (40 events per life) | Stated in advance. Automatic confirmation only when reachable (§9.4); otherwise the owner decides |
| FE-NI in exposure-driven form may be too weak to test the common-component account sharply | Stated (§6.2) |
| CD alters the front end's input history as well as removing information | Frequencies are preserved; FE-LAG and FE0 identity checks bound the disturbance (Q4, H5) |
| UNP η is miscalibrated under contingent-world readouts | C uses AUC; UNP η is reported only |
| The XR positive controls could be too weak at n = 8 | H5 and H6 halt the round rather than proceed without sensitivity |
| Winner's curse | Untouched confirmation; Δ_plan halved |
| In-memory `ARMS` registration | Q1 bit identity; frozen files untouched (Q14) |

---

## 17. Deferred (roadmap §3.3)

- FE1-NI.
- The timing-vs-framing split.
- The variable-filler format.
- The clock time gate, including FE0's collapse at ×2.
- Wrapper invariance (V52 H-I).
- The read-locked and output-driven negative images.
- A sparser receptor table.
- Oracle arms on new worlds.
- Own-cue vs partner-cue separation, which needs independent-consequence worlds.

---

## 18. Changes from v3

| v3 | v4 |
|---|---|
| A2 described as certifying "cue-specific" learning | Claims narrowed (§1). A timing-preserving cue-destruction control (CD-UNP) and the C contrast are required. Write dependence is argued and checked (H6). Own-cue vs partner-cue is a stated non-claim |
| Retention: 24-h day-probe Δη, non-inferior at margin 0.02 | Retention R on delayed returns of unchanged pairs, chosen from data (erratum §13). No margin: automatic adoption requires R > 0, and other patterns go to the owner |
| A single adoption label | Five outcome classes (§9.5); `TRADE_OFF_IN_DEVELOPMENT` at selection |
| Confirmation always run if n_V ≤ 64 | Automatic only when `IMPROVES` is reachable; otherwise `AWAITING_OWNER` |
| Q12: cross-world key disjointness | World-identity rule; coincidences counted, not failed |
| FE-LAG "never selectable", unexplained | The question states the biological scope; FE-LAG is evaluated under identical rules as a benchmark; §9.6 readings |
| FE-NI `read()` used a p̂ from `feed()`; counter inside `read()` | `read()` recomputes and is pure; external counter |
| FE1-NI candidate | Dropped (retention evidence; §R) |
| Budget: 232 always, ≤ 616 ceiling | 288 always, ≤ 816 ceiling |

---

## Appendix A — FE-NI parameters

- 14 samples per record; N_m = 128 × 14 = 1,792.
- Self-cancellation: 1 − (1 − 1/1792)¹⁶⁸ = 0.0896.
- Convergence: ln 0.05 / ln(1 − 1/1792) = 5,367 samples ≈ 383 records.

## Appendix B — Planning inputs (erratum §5, §12, §13)

| Quantity | Value |
|---|---|
| SD of A1-type Δη (DEV) | FE1 0.0143; FE-LAG 0.0112 |
| SD of R-type Δη | FE1 0.0576 (DEV), 0.0430 (CAL); FE-LAG 0.0815 (DEV), 0.0698 (CAL) |
| FE0 R-subset η | 0.0138 (CON) |
| FE0 day-probe η | 0.048 (CON) |
| Life wall time | 4–9 s |

## Appendix C — Literature (abstracts checked on Europe PMC, 23 Sep 2026)

| Paper | Used for | Label |
|---|---|---|
| Das et al., PNAS 2011 (PMID 21795607) | Odorant-selective short-term and long-term habituation via glomerulus-selective potentiation of inhibition onto PNs; long-term habituation reduces odorant-evoked PN calcium | C+ |
| Ramaswami, Neuron 2014 (PMID 24945768) | Proposal: excitation creates an inhibitory negative image of itself | T |
| Roselli, Ramaswami & Aranha, Bio-protocol 2024 (PMID 39525971) | Long-term habituation after 4-day exposure lasts > 24 h (basis for no decay of m) | Protocol citing prior work |
| Hallem & Carlson, Cell 2006 | Receptor table (inherited) | R |
