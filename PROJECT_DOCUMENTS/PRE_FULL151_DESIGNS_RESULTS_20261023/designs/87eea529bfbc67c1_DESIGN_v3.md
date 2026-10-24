# BYTE-CORE-FEAS-1 (v3) — Can a deployable byte front end improve FULL_151?

**Namespace:** `BYTE-CORE-FEAS-1`
**Date:** 23 September 2026
**Status:** design for review. **Not locked. No code written. Nothing run.**
**Authority:** this is the only executable FEAS-1 design. `DESIGN_DRAFT.md` (v1) and `DESIGN_v2.md` (v2) are superseded, kept unedited as history, and must not be executed.
**Implements:** `ROADMAP_20260923.md` §3.1 (roadmap revised the same day).
**Parents (read-only):**
- `BYTE_CORE_1_20260922`: frozen round; its preregistered decisions stand.
- `BYTE_CORE_1_ERRATUM_20260923`: post-hoc re-analysis, cited as "erratum §n". §12 was added for this revision; 17 self-tests pass.

---

## R. Response to the 23 Sep review of v2

| # | Finding | Change in v3 |
|---|---|---|
| 1 | **Blocker.** The front-end contract did not match read timing. The frozen adapter scans the record for the reserved byte and then calls `read()`, and v2's FE-NI updated long-term state inside `read()`. That gave FE-NI a per-record sampling pulse that FE0 and FE1 never use. | FE-NI is redefined so its state changes only on byte arrival (§5.2). Every front end's `read()` must now be free of side effects (Q3); the v2 form is kept only as a fixture that Q3 must reject. The innate detector's event is declared to reach the learner only, never a front end (§2). A byte-causal driver must reproduce the frozen adapter bit for bit for every arm, with two failing variants (Q2). |
| 2 | **Blocker.** A cross-record shortcut could enter the main gain. R8 gated only in development, and only above 0.5; UNPAIRED does not split the gain vs FE0. | New **consequence-decoupled (CD)** intervention (§4.2). The front end receives a random substitute for each consequence byte, while the learner's detector receives the true one. Adoption now requires the gain to survive decoupling (A2) as well as on the intact stream (A1), in both development and confirmation. R8 is descriptive only. A deliberately leaky fixture (XR) shows the intervention removes such a path (H5). UNPAIRED is dropped. |
| 3 | Retention was judged on point estimates only. | Retention is now **non-inferiority**: day-probe Δη has a 95% CI lower bound > −0.02. The margin, its rationale and its relative size are stated (§8.3). The sample size is planned for this condition too (§10.3). |
| 4 | 0.02 was a screening convention being used as an adoption threshold. The sample-size rule was not built for the stated claim. | 0.02 is renamed the **screening line**, used only for selection. Adoption requires a confirmed positive, cue-specific gain plus retention non-inferiority; why that suffices is argued in §8.2. The sample size gives ≥ 95% power per adoption condition (§10.3). The claim "gain ≥ 0.02 confirmed" is reported only if met, with its power stated. |
| 5 | The roadmap's S2 bridge had no preset rules, and its version references were stale. | The roadmap now pre-specifies the bridge: stream, metrics, pass lines and failure actions. It names v3 as authoritative and states that A11 v3 is not cleared to run. |

---

## 0. Acceptance contract

| Item | Contract |
|---|---|
| **Question** | Does one explicit, deployable byte front end improve the frozen FULL_151 on R2/F3 under a prespecified usable readout? The gain must be cue-specific (it must survive decoupling), and retention must be non-inferior. |
| **Decides** | Which front end the byte line uses by default for the S2 bridge. It does **not** decide readiness for S2 (roadmap §3.2b). |
| **Stream** | BYTE-CORE-1 world (§3): V88 schedule plus amendment A10; R2 with reserved bytes 0xF1/0xF0; frozen F3 rendering; massed density; clock ×1; episode 30 s; gap 135 s; fresh namespaces. |
| **Interfaces (§2)** | A front end receives only (byte, time); its `read()` has no side effects. The innate detector's event reaches only the frozen learner, identically for every arm. The code is taken when a detector byte arrives, before that byte is integrated; Q2 proves this is causal. |
| **Learner** | FULL_151 `151ebd9e2d054ec9ec14`, frozen; adapter `S0-EPISODE` unchanged; frozen reader Q. |
| **Arms** | FE0 (reference). Candidates FE1, FE-NI and FE1-NI. Each is run **intact** and **consequence-decoupled (CD)**. FE-LAG is an engineering baseline, intact only, never selectable. Fixtures: XR (calibration only), and the v2 read-locked FE-NI (qualification only). |
| **Readout** | p(aversive) = σ(a·s + c), with s = −v. Same family, calibration budget, fitting procedure, clipping and world weighting for every arm. Fitted on intact CAL lives and frozen before development worlds exist. CD arms use their intact parent's readout (§6). |
| **Adoption conditions** | **A1**: intact in-stream Δη vs FE0 > 0. **A2**: cue-specific Δη (CD arms) vs FE0-CD > 0. **A3**: day-probe Δη vs FE0 > −0.02 (non-inferiority). All are judged by 95% CI lower bounds, per world and paired. |
| **Development** | 16 worlds, plus one pre-set 16-world extension. Used only to **select** at most one candidate; the 0.02 screening line applies to A1 and A2 here. |
| **Confirmation** | One frozen candidate vs FE0 on untouched worlds. n is fixed before those worlds exist, in the range 32–64, with ≥ 95% power per condition. |
| **Stop** | No selectable candidate → stop. Required n > 64 → stop. Any halt condition H1–H5 → stop before development. |
| **Budget** | Always ≤ 232 lives. Ceiling ≤ 616 lives; this is a limit, not a target (§9). |

---

## 1. Evidence used

| Evidence | Value | Source |
|---|---|---|
| FULL_151 + FE0 learns above chance on F3, weakly | In-stream AUC 0.613 (no-learning 0.496); η 0.0047 with the frozen decoder, 0.0263 with the CAL-fitted bias decoder | erratum §2, PH-1 |
| A learned common offset | 98.9% of F3 events fall below the no-learning copy | erratum §3 |
| Exploratory leads (DEV, n = 16) | In-stream Δη vs FE0: FE1 +0.0254 [0.0178, 0.0330] (SD 0.0143); FE-LAG +0.0295 [0.0235, 0.0355] (SD 0.0112) | erratum §5, §12 |
| Day-probe η under the in-stream readout | FE0 0.043 (DEV), 0.048 (CON). FE1 − FE0 = +0.0266 [+0.0034, +0.0498] (SD 0.0436). Mean predicted P(aversive) on day probes ≈ 0.59–0.61 for FE0 | erratum §12 |
| The frozen proxy vetoed FE1 using a state FE1 never has while learning | Measured after the final 7-day rest | `bytecore.py` `interface_metrics` |
| Cross-record cue in window learners on 8-byte records; weak signs in FE1 and FE3 | V50-REF F1 side-1 first-presentation AUC 0.759; FE3 0.569 (DEV); FE1 0.518 [0.498, 0.538] (DEV) | erratum §10 |
| Read timing in the frozen adapter | `present` locates the reserved byte, feeds bytes up to it, advances to its arrival time, calls `read()`, then teaches, then feeds the rest | `bytecore.py` lines 949–970 |

Frozen oracle and no-learning numbers are **context only**. They are not acceptance controls under this readout.

---

## 2. Interfaces and causality

### 2.1 What each component receives

| Component | Receives | Never receives |
|---|---|---|
| World | — | — |
| **Front end** | (byte, time) for every byte, in arrival order | The detector event; record boundaries; identity, role, key position, wrapper id; evaluator truth |
| **Innate detector** (fixed; part of the animal) | Each arriving byte | Anything else |
| **Learner** (frozen FULL_151) | The detector event, plus the front end's output at the detector moment | Evaluator truth, identity, role, wrapper id |
| Evaluator only | Delivered and committed truth, identities, roles, schedule tags | — |

### 2.2 The detector event (declared, inherited)

- It fires when 0xF1 arrives (US on) and when 0xF0 arrives (US off). It therefore fires once per record, neutral records included.
- This is BYTE-CORE-1's S0 convention (§3.3, §4.2, amendment A1), not a fly finding: a fly has no detector for "no shock".
- It gives the **learner** a consequence-onset time on every record. Every arm receives it identically, and only through the frozen learner.
- **No front end receives it.** v2's FE-NI did, through read-locked updates; that form is withdrawn.

### 2.3 Read moment and its causal equivalent

- The code is taken at the arrival time t_c of the detector byte: the front end is advanced to t_c and read **before** that byte is integrated.
- The frozen adapter implements this by scanning the record for the reserved position. That is a lookahead in the code, not in the information used.
- The causal equivalent is a byte-by-byte driver:
  1. on each byte's arrival, the detector inspects the byte first;
  2. if it fires, the snapshot of the state from before that byte is committed as the code, and teaching follows;
  3. then the byte is fed.
- **Q2** requires this driver to reproduce the frozen adapter bit for bit, for every arm.
- Byte times are set by the world's emission schedule (a 30 s episode divided evenly over the record length). The learner receives them as (byte, time).

### 2.4 Read purity

`read()` must be a pure function of the state. **Q3** inserts extra reads at random times and requires every later output and the final state to be bit-identical. The v2 read-locked FE-NI must fail Q3.

---

## 3. Stream and worlds

**World generator.** The frozen `create_world(ns, block, index, "R2", "massed")`, seeded as `BYTE-CORE-1|<ns>|<block>|<index>`. Each world passes the frozen `audit_world`. Keys are audited disjoint across all FEAS-1 and BYTE-CORE-1 namespaces (Q12).

| Namespace | Block | Indices | Use |
|---|---|---|---|
| `FEAS1-ENG` | `Q` | 0–1 | Technical fixtures |
| `FEAS1-CAL` | `K` | 0–7 | Readout and NLMS-rate fitting; instrument checks H2–H5 |
| `FEAS1-DEV` | `D` | 0–15, plus 16–31 for the extension | Development (selection) |
| `FEAS1-CON` | `V` | 0 … n_V − 1 | Confirmation; generated only after `CONFIRM_PLAN.json` is hashed |

**Format: F3 only**, frozen `render_record`.
- 14 bytes: `~|~|` + header[w] + key with XOR mask[w] on key bytes 0–1 + separator[w] + consequence + newline.
- The wrapper alternates per presentation of a cue.
- Bytes are spaced 30/14 s apart.
- At read time, a 12-byte window holds exactly the current context (erratum §10).

**Every life:**
- 3,648 scored consequence events (1,824 trials).
- 13 day checkpoints; 12 of them carry relevant-pair probes, 320 per life in total.
- Windows W1–W13 between checkpoints hold 576, 288, 288, 288, 288, 288, 144, 336, 144, 360, 144, 360, 144 events.

**Probes** are those of the frozen `run_life` and are read-only: immediate, day, final, reactivation, revision curve, plus F4 day probes (wrapper 2).

---

## 4. Learner, adapter, and the decoupling intervention

### 4.1 Frozen learner and adapter

**FULL_151.** Genome and locked objects are as in BYTE-CORE-1 §4.2. Mutable state is 165,664 bytes.

**`S0-EPISODE`.** For each record:
1. The front end is fed the context bytes at their times.
2. At the detector moment, the code is taken (§2.3) and encoded live with `encode_sparse`, and the reader-relative response v is recorded. This is the prediction.
3. Teaching is one 30 s event with that code and punishment = [0xF1].
4. The remaining bytes are fed.
5. The 135 s gap follows.

### 4.2 Consequence-decoupled (CD) arms

**What changes.** In a CD life, the front end is fed a substitute consequence byte c′ in place of the true one.
- c′ is drawn i.i.d. uniformly from {0xF1, 0xF0}, from a world-seeded stream (`…|cd|…`) independent of the schedule and of consequences.
- Everything else is identical: the detector and the teaching event (both from the true byte), the learner, the newline byte, and every probe (probes contain no consequence bytes).
- Byte frequencies are preserved. Only the information about **which** consequence occurred is removed from the front end's input.

**Why.** In the intact stream, the consequence byte is fed to the front end after the read. Anything it leaves in slow front-end state can inform later reads:
- FE1's habituation h;
- FE-NI's image m;
- any trace.

That is the cross-record path (erratum §10). In a CD life this path carries no information, while cue→consequence teaching through the detector is untouched.

**Contrasts, all per world and paired:**

| Contrast | Definition | Role |
|---|---|---|
| **A1** | η(c) − η(FE0) | Intact gain: the owner's stream |
| **A2** | η(c-CD) − η(FE0-CD) | Cue-specific gain: no consequence information passes through any front end |
| **Path(x)** | η(x) − η(x-CD) | Descriptive. How much of arm x's own η depends on consequence information through its front end, including shortcut use |
| **Share** | A2 / A1 and Path(c) − Path(FE0) | Descriptive: how the candidate's gain splits |

**What CD does not remove.** The learner's own cross-record effects, such as generalization from the just-taught partner cue through shared KCs, are untouched. They are the same kind of effect in every arm.

**Expected, and checked:**
- FE-LAG: intact ≡ CD bit for bit, because at F3 its window never holds a previous consequence at read time (Q4).
- FE0: |η(FE0) − η(FE0-CD)| < 0.001 per world, because the residue of the previous consequence at the next read is e^(−16.5) ≈ 7 × 10⁻⁸ (Q4, H5).

---

## 5. Front ends

Every front end:
- receives only (byte, time);
- keeps all state in counted arrays;
- has no learned parameters;
- has a pure `read()`;
- is identical across worlds.

The receptor table **R** ∈ ℝ^{256×88} is frozen: for each byte, 6 channels chosen without replacement, amplitudes U[0.4, 1.0], seed `BYTE-CORE-1|RECEPTOR|v1`.

### 5.1 Arms

| Arm | Status | Definition | State | Basis (label) |
|---|---|---|---:|---|
| **FE0** | Reference; frozen `bytecore.FE0` | p ← p·e^(−Δt/10 s) + R[b]; read p/max p | 88 floats | Hallem & Carlson 2006 (R) |
| **FE1** | Candidate; frozen `bytecore.FE1` | Input R[b]⊙(1 − h). Per byte, h ← clip(h + 0.3(1 − h)⊙R[b], 0, 1). h decays with τh = 1,800 s. p as FE0 | 176 floats | Short-term, odorant-selective habituation (Das et al. 2011, C+) |
| **FE-NI** | Candidate; **new in v3** | FE0 integration, plus an exposure-driven negative image m (§5.2) | 176 floats | Long-term habituation via glomerulus-selective potentiated inhibition (Das et al. 2011, C+). Negative-image proposal (Ramaswami 2014, T) |
| **FE1-NI** | Candidate; new in v3 | FE1 integration, plus the same m | 264 floats | Both parents |
| **FE-LAG** | Engineering baseline; frozen `bytecore.FELAG`; never selectable | V51 lag code of the last 12 bytes, random signed 8-contact adaptor into 88 channels, tanh, rectified, normalized | 12 bytes | Delay lines; not claimed biological |

### 5.2 FE-NI, exactly (FE1-NI is identical, using FE1's p and h)

**State:** p ∈ ℝ^88 as FE0, and m ∈ ℝ^88 with m₀ = 0.

**`advance(t)`:** p decays (and h, for FE1-NI). m does not change: there is no decay in silence.

**`feed(b, t)`**, on every byte arrival, in this order:
1. `advance(t)`.
2. **Sample.** p̂ = p / max p, the zero vector if p = 0. This is the pattern the front end shows at the byte's arrival, before the byte.
3. **Adapt.** m ← m + (p̂ − m)/N_m.
4. **Integrate b.** p += R[b] for FE-NI; for FE1-NI, FE1's gated input and its h update.

**`read()`**, pure: o = max(p̂ − m, 0). The output is o / max o, or the zero vector if max o = 0; such reads are counted, and they produce an empty KC code and no write.

**Parameters,** derived rather than tuned (Appendix A):
- **N_m = 1,792 samples**, i.e. 128 records × 14 bytes.
  - Massed acquisition exposes one pair for 12 records, i.e. 168 samples; these make up 1 − (1 − 1/1792)¹⁶⁸ = 9.0% of m, so a cue barely cancels itself.
  - m reaches 95% of a constant component after 5,367 samples (≈ 383 records), inside block 0 (576 records).
- **Subtraction weight 1.**

**What m sees.** The exposure pattern sampled at every byte arrival: activity-driven, as in the negative-image proposal and in exposure-induced habituation. One of the 14 samples per record is the read-time pattern itself. The pattern just after the consequence byte (sampled at the newline's arrival) also enters m, with weight 1/14 per record. So m carries a small consequence-dependent component; A2 and Path measure what it is worth.

**Expected weakness, stated before any run.** Averaging over within-record positions cancels components present throughout the record, such as the constant prefix, more than components that appear late: the separator arrives at slot 11 and is present in only 2 of 14 samples. FE-NI therefore removes the read-time common component less completely than the withdrawn read-locked form would have.

**Not candidates:** the read-locked negative image (it needs the detector event inside the front end, §2.2), the output-driven negative image, a sparser receptor table, kernels tied to slot geometry, and FE-SURP (roadmap §3.3).

### 5.3 Fixtures (never candidates)

- **XR (deliberate shortcut)** is used in CAL only, to show that the CD intervention removes a consequence path (H5).
  - Definition: FE0 plus q ∈ ℝ^88. q decays with τ = 600 s. When a reserved byte arrives, q += R[b], after the read. `read()` = normalize(p + q).
  - It carries earlier consequences into later reads: the previous one enters with weight e^(−165/600) ≈ 0.76 at the next read.
  - It deliberately uses knowledge of the reserved-byte class.
- **NI-READLOCKED** is the v2 FE-NI. It is used in Q3 only, as the negative control that must fail read purity.

---

## 6. Declared readout

1. **Score.** s_e = −v_e for each scored event.
2. **Model.** p_e(aversive) = σ(a·s_e + c).
3. **Fitting data.** All scored events of the arm's 8 intact `FEAS1-CAL` lives are pooled. Each life has 3,648 events, so worlds are weighted equally.
4. **Fit.**
   - Maximize the mean Bernoulli log-likelihood (natural log) against the **delivered** byte.
   - The problem is convex, so it is solved by Newton's method (IRLS) from (0, 0) until max |gradient| < 10⁻¹⁰.
   - Probabilities are clipped to [10⁻¹², 1 − 10⁻¹²] inside the objective only.
   - One function serves every arm.
5. **Freeze.** The (a, c) values are written to `READOUT.json` and hashed before any DEV world is generated.
6. **Inheritance.** CD arms use their intact parent's (a, c). Decoupling is an intervention on the stream, not a new strategy.
7. **Use.** Every later η uses the frozen (a, c) with clip ε = 10⁻³. This covers in-stream events, windows, day probes and the final probe; nothing is recalibrated.
8. **No online correction.** A readout that becomes less usable as the learner changes is a property of the tested strategy. For example, day-probe mean p ≈ 0.6 for FE0 (erratum §12).

The CODE-NLMS learning rate (reference only) is chosen in the same phase over {0.1, 0.3, 1, 2}.

---

## 7. Outcomes

### 7.1 η (BYTE-CORE-1 §5.2), per world

- For each scored event: p̃_e(y_e) = clip(p_e if y_e is aversive, else 1 − p_e; ε, 1 − ε).
- Bits saved: s_e = log₂ p̃_e(y_e) + 1. Relevant information fed per event: i_e = 1.
- η = Σ s_e / Σ i_e.
- In-stream truth is the **delivered** byte, anomalies included.
- Day-probe η uses the 320 relevant-pair day probes, scored against **committed** truth (frozen `TruthTimeline`), because probes have no delivered byte.
- **Units.** A Δη of 0.02 is 2 bits saved per 100 bits of relevant consequence information fed. It is not 2 percentage points of accuracy.

### 7.2 Decision quantities

| ID | Quantity | Arms |
|---|---|---|
| **A1** | In-stream Δη vs FE0 | Intact |
| **A2** | In-stream Δη vs FE0-CD | CD |
| **A3** | Day-probe Δη vs FE0 | Intact |

### 7.3 Reported, with no decision role

| ID | Quantity |
|---|---|
| R1 | In-stream AUC. Measures discrimination, not calibration; pooled AUC can move with time-varying offsets |
| R2 | Native paired accuracy |
| R3 | Day-probe AUC; day-probe η of the CD arms |
| R4 | Final 7-day probe η and AUC |
| R5 | For W1–W13 and each day checkpoint: η, observed aversive frequency, unconditional mean of p_t, AUC, n |
| R6 | Offset share (learned minus no-learning copy below 0), per window |
| R7 | Path(x), A2/A1, and Path(c) − Path(FE0), with CIs |
| R8 | Side-1 first-presentation AUC. Descriptive only; the decision uses A2 whatever R8 shows |
| R9 | CODE-NLMS reference on the same codes (a reference learner, not a measure of information) |
| R10 | Live interface statistics: active KCs; Jaccard between consecutive presentations of a cue; Jaccard between the two cues of a trial; empty reads |
| R11 | Resources: front-end state bytes, learner mutable bytes, wall time |
| R12 | F4 day probes: η and AUC |

---

## 8. What each threshold decides

### 8.1 The screening line (0.02), development only

- This is BYTE-CORE-1's series convention δ (§6.3), used here **only** to decide whether an estimate justifies buying a confirmation.
- It is not derived from a requirement, so it is not called "materiality" in this design.
- In context: FE0's in-stream η under this readout is ≈ 0.026, so 0.02 would be a near doubling of FE0.

### 8.2 Adoption

Adoption requires, on untouched worlds, A1 > 0, A2 > 0 and A3 > −0.02, each judged by its 95% CI lower bound. A positive estimate of any size is sufficient, for four reasons:
1. **What is decided.** The decision is the byte line's default front end for the next bridge. It is reversible and makes no claim that the byte interface is solved.
2. **Cost.** Every candidate is fixed, has no learned parameters, and costs at most 264 floats of state.
3. **FE0 has no standing as a default** beyond its selection by the misapplied proxy (§1). Two deployable front ends with learner data, FE1 and FE-LAG, exceeded it in exploratory data (erratum §5); only the temporal-contrast FE2 and FE3 did worse. So a switch needs a reliable improvement, not a large one.
4. **A further gate.** The adopted front end must still pass the S2 bridge before S2 on bytes (roadmap §3.2b).

The estimate's size against the screening line is always reported (§10.4).

### 8.3 Retention margin (A3)

- **Tolerable loss:** 0.02 in day-probe η, judged by the 95% CI lower bound of the per-world Δη.
- **Rationale:** a retention loss is tolerated only up to the size of the in-stream gain that motivates selection. Both are measured in bits saved per relevant bit.
- **Relative size, stated openly:** this is ≈ 40–45% of FE0's day-probe η (0.043–0.048; erratum §12). It is lenient in relative terms, and the day-probe Δ with its CI is always reported.

---

## 9. Phases, roster and budget

| Phase | Arms × worlds | Lives | Runs |
|---|---|---:|---|
| **Q** Fixtures | See Q-roster below | 24 | Always |
| **K** Calibration and instrument checks | Intact FE0, FE1, FE-NI, FE1-NI, FE-LAG × CAL 0–7 (40). XR and XR-CD × CAL 0–7 (16). FE0-CD × CAL 0–7 (8) | 64 | Always |
| **D** Development | FE0, FE1, FE-NI and FE1-NI, each intact and CD × DEV 0–15 (128). FE-LAG intact × DEV 0–15 (16) | 144 | Always |
| **X** Extension | FE0 and each INCONCLUSIVE candidate, intact and CD × DEV 16–31 | ≤ 128 | Only under §10.2 |
| **V** Confirmation | Candidate, candidate-CD, FE0 and FE0-CD × CON 0 … n_V − 1, n_V ∈ [32, 64] | 128–256 | Only if a candidate is selected and n_V ≤ 64 |

**Q-roster (24 lives):**

| Check | Lives |
|---|---:|
| Q1 identity: FE0 and FE1 through the FEAS and frozen paths, 2 worlds each | 8 |
| Q1 failing variant | 1 |
| Q2 byte-causal driver: 5 arms on ENG 0 | 5 |
| Q2 frozen-path runs for FE-NI, FE1-NI and FE-LAG | 3 |
| Q2 failing variants | 2 |
| Q4 FE-LAG-CD on ENG 0 | 1 |
| Q4 FE0-CD on ENG 0–1 | 2 |
| Q6 probe failing variant | 1 |
| Q11 empty-read positive control | 1 |

Q3 (read purity) drives the front ends alone, without the learner, and costs no lives.

**Totals.**

| Scope | Lives |
|---|---:|
| Always | 232 |
| With extension | ≤ 360 |
| Ceiling (with extension and confirmation) | ≤ 616 |

- At 4–9 s per FULL_151 episode life, the always-run part is ≈ 15–35 CPU-min and the ceiling is ≈ 40–90 CPU-min.
- With 8 workers, wall time is minutes.
- Most of the growth over v2 (394) comes from running CD arms in development and confirmation, which blocker 2 requires.
- BYTE-CORE-1 ran 3,856 lives.

---

## 10. Decision rules

**Statistics.**
- The world is the unit, and arms are paired on identical worlds.
- Each contrast is a per-world difference, summarized by its mean, its SD and a 95% two-sided t interval with df = n − 1.
- Development is selection, so there is no multiplicity adjustment. Confirmation tests one frozen candidate.

### 10.1 Development (n = 16)

For each c ∈ {FE1, FE-NI, FE1-NI}:

| Status | Condition |
|---|---|
| **SELECTABLE** | **S1:** A1 mean ≥ 0.02 and CI lower bound > 0.<br>**S2:** A2 mean ≥ 0.02 and CI lower bound > 0.<br>**S3:** A3 CI lower bound > −0.02. |
| **FUTILE** | A1 or A2 has CI upper bound < 0.02, or A3 has CI upper bound < −0.02. |
| **INCONCLUSIVE** | Neither of the above. |

FE-LAG is reported on the same terms, but it is never selectable.

### 10.2 Extension (still development)

- **Runs only if** no candidate is SELECTABLE at n = 16 **and** at least one is INCONCLUSIVE.
- **Arms:** FE0 and every INCONCLUSIVE candidate, intact and CD, on DEV 16–31.
- **At n = 32,** §10.1 is re-applied, and anything not SELECTABLE becomes NOT_SELECTED.
- There is no second extension. Looking twice is acceptable because this is selection; the untouched confirmation carries the inference.

### 10.3 Selection and confirmation plan

**Selection.**
- The SELECTABLE candidate with the **largest A2 lower bound** is chosen; a tie goes to the smaller state.
- It is reported as "positive intact and cue-specific gain supported; estimates meet the screening line; retention non-inferior at margin 0.02". Never as "a gain of at least 0.02 established".
- `SELECTION.json` is hashed. If there is no candidate, the outcome is `NO_CANDIDATE_SELECTED` and the round stops after reporting.

**Confirmation size.** n_V is the smallest n ≥ 32 that gives ≥ 95% power to **each** adoption condition at planning values, using

n ≥ ((t₀.₉₇₅,ₙ₋₁ + t₀.₉₅,ₙ₋₁) · SD / (Δ_plan + margin))², iterated in n.

- **SD** is the development per-world SD of the contrast, at the final development n.
- **Δ_plan** is half the development mean, to allow for selection bias. For A3, a negative mean is used unhalved.
- **Margin** is 0 for A1 and A2, and 0.02 for A3.
- The joint power of the three conditions is ≥ 0.85 by the union bound.
- If n_V > 64, the outcome is `NOT_CONFIRMABLE_THIS_ROUND`, and nothing runs.

**Worked example with frozen FE1 numbers** (erratum §5, §12; A2 is unknown until development):

| Condition | SD | Δ_plan | Required n | n used |
|---|---:|---:|---:|---:|
| A1 | 0.0143 | 0.0127 | 17.7 | 32 |
| A3 | 0.0436 | 0.0133 | 23.9 | 32 |

`CONFIRM_PLAN.json` records the candidate, n_V, the readout hash and the four arms. It is hashed, and only then are CON worlds generated.

### 10.4 Confirmation claims (95% CIs, one candidate)

| Claim | Conditions |
|---|---|
| **Adopt** — `FRONT_END_IMPROVES_OVER_FE0` | A1 lower bound > 0, **and** A2 lower bound > 0, **and** A3 lower bound > −0.02 |
| Added wording: "estimates meet the screening line" | Adopt, and A1 and A2 means ≥ 0.02 |
| Added wording: "gain ≥ 0.02 confirmed" | Adopt, and A1 and A2 lower bounds > 0.02. Not powered at n ≤ 64 unless the true gain is ≳ 0.027 (SD 0.0143); stated as such |
| Always reported | R7, the split of the gain: A2/A1 and Path(c) − Path(FE0) |

### 10.5 Outcome labels

| Label | Meaning | Next (roadmap) |
|---|---|---|
| `FRONT_END_IMPROVES_OVER_FE0` | Adoption conditions met | Default front end for the S2 bridge. **Not** S2 readiness |
| `NOT_CONFIRMED` | Selected, but a condition failed; the report names which | FE0 stays the default |
| `NOT_CONFIRMABLE_THIS_ROUND` | n_V > 64 | FE0 stays the default; the development estimate is reported as selection evidence only |
| `NO_CANDIDATE_SELECTED` | Nothing met §10.1 | Stop; the owner decides the next step |

---

## 11. Instrument checkpoint (after Q and K, before any DEV world)

A halt writes `CHECKPOINT.json` and stops the runner. No DEV world is generated until a written amendment has been reviewed.

| ID | Halt if |
|---|---|
| H1 | Any Q check fails, or any negative or positive control misbehaves |
| H2 | 2-fold readout cross-fit on CAL (fit on worlds 0–3, test on 4–7, and the reverse). For any intact arm: \|held-out η − in-sample η\| > 0.01, or held-out mean p outside 0.50 ± 0.05. This is an engineering check; untouched worlds carry the real evidence |
| H3 | FE-NI or FE1-NI empty-read rate > 1% on CAL |
| H4 | FE0's CAL in-stream η has a CI lower bound ≤ 0; frozen CAL gave 0.0243 |
| H5 | **The decoupling intervention fails.** Either Path(XR) has a CI lower bound ≤ 0 at n = 8 (it does not remove a known consequence path), or some CAL world has \|Path(FE0)\| ≥ 0.001 (it disturbs an arm without a path) |

---

## 12. Qualification (phase Q)

**Negative control:** a defect the check must catch. **Positive control:** a known effect a diagnostic must register. Results are computed from archived runs; no flag is a literal.

| ID | Check | Negative control | Positive control |
|---|---|---|---|
| Q1 | **Execution identity.** The FEAS path reproduces the frozen `run_life` bit for bit for FE0 and FE1 on ENG 0–1: every `ev_*` and `pr_*` array and the final digest. The path is in-memory arm registration + learner factory + recording subclass, with CD off | A recording subclass that adds one active KC must break identity | — |
| Q2 | **Causal read moment.** The byte-causal driver of §2.3 reproduces the frozen adapter bit for bit for FE0, FE1, FE-NI, FE1-NI and FE-LAG on ENG 0 | (a) A snapshot taken after integrating the detector byte must break identity. (b) A snapshot taken at the previous byte's arrival must break identity | — |
| Q3 | **Read purity.** Each front end (FE0, FE1, FE-NI, FE1-NI, FE-LAG, XR) is driven alone on the ENG 0 byte stream. Extra `read()` calls at 1,000 random times must leave every scheduled output and the final state bit-identical | NI-READLOCKED must fail | — |
| Q4 | **CD machinery.** Re-seeding the substitute stream changes the substitutes but not the world hash, and substitute frequency is within binomial limits of 0.5. The detector and teaching logs of intact and CD lives are identical. FE-LAG intact ≡ CD bit for bit (ENG 0). FE0 has \|Δη\| < 0.001 per world (ENG 0–1) | A CD variant that feeds the true consequence byte must fail the independence check | Path(XR) > 0, tested in K (H5) |
| Q5 | **No annotation reaches a front end.** New classes reject `set_oracle`; the learner accepts only byte records | An FE-NI variant exposing `set_oracle(key_offset)` must be flagged | The frozen FE-TAIL4 is flagged |
| Q6 | **Probes are read-only.** The persistent digest (learner fast/slow/adapt/elapsed; front-end p, h, m, q) is unchanged across every probe | A probe that reads the live front end must raise | — |
| Q7 | **Readout fitting.** Refuses non-CAL worlds; a shuffled-label fit gives η ≤ 0.002 on ENG | Injecting a DEV world must raise | On frozen CAL FE0 F3 data it reproduces the erratum fit (a = 0.06543, c = −0.8256) within 10⁻³ |
| Q8 | **Metric and decision fixtures.** η: uniform p gives 0. AUC: perfect 1, reversed 0, constant 0.5. Paired accuracy ≈ 0.5 on shuffled scores. The decision function, fed synthetic per-world differences with known means and SDs, returns the pre-computed SELECTABLE / FUTILE / INCONCLUSIVE and adopt / not-adopt labels, including the boundary cases | A constant-offset synthetic stream keeps R5 window means flat | A planted offset drift is tracked by the R5 window means |
| Q9 | **R8 detector** (descriptive) | Frozen FE0 F1, FE-TAIL4 F3 and FE-ID F3 (CON) stay quiet | Frozen V50-REF F1 (0.759) and FE3 F3 (0.569) fire |
| Q10 | **Day-probe measures** | Shuffled committed truth gives AUC ≈ 0.5 and η ≤ 0 | On frozen files the code reproduces erratum §11 and §12: FE-ID vs FE0 day-probe ΔAUC +0.266 (CON); FE1 vs FE0 day-probe Δη +0.0266 (DEV) |
| Q11 | **Empty reads.** Rate < 1% on ENG for FE-NI and FE1-NI | — | A variant with m₀ = 1 on every channel must exceed 1% |
| Q12 | **Worlds.** The frozen `audit_world` passes; keys are disjoint across all FEAS-1 and BYTE-CORE-1 namespaces | A world with one key copied from a BYTE-CORE-1 CON world must be rejected | — |
| Q13 | **Source lock.** New files, hashes of the frozen imports, and verification of the frozen `SOURCE_LOCK.json` | An appended comment must raise | — |
| Q14 | **Frozen folders unchanged.** Snapshots of `BYTE_CORE_1_20260922/` and the erratum, before and after each phase | A one-byte change in a scratch copy must be detected | — |
| Q15 | **Output manifest.** Every output in §15 exists before packaging | Removing one must fail packaging | — |

---

## 13. Predictions, written before any run

| Arm | A1 | A2 | A3 and other quantities |
|---|---|---|---|
| FE0 | — | — | In-stream η ≈ 0.02–0.03. Day-probe η ≈ 0.04–0.05 with mean p ≈ 0.6. Path(FE0) ≈ 0. Offset share ≈ 0.99 |
| FE1 | ≈ +0.025 (frozen +0.0254 [0.0178, 0.0330]) | ≤ A1. ≈ A1 if its gain is cue-specific; smaller if its habituated consequence channels carry a shortcut (frozen R8 0.518 [0.498, 0.538] allows one) | ≈ +0.027 (frozen [0.003, 0.050]), so non-inferiority is likely. At the selection boundary overall |
| FE-NI | Positive if the common-component account holds; no magnitude predicted | Close to A1 (m's consequence component is 1/14 of each record's samples) | Offset share below FE0's 0.99. Weaker than a read-locked form would be (§5.2) |
| FE1-NI | ≥ the larger of FE1 and FE-NI if their mechanisms are complementary; about equal to the larger if redundant | — | The interaction contrast A1(FE1-NI) − A1(FE1) − A1(FE-NI), with its CI, is descriptive |
| FE-LAG | ≈ +0.03 (frozen +0.0295) | Not run; intact ≡ CD at F3 by construction | Never selectable |
| XR | — | — | Path(XR) clearly > 0 |

**What falsifies FE-NI's rationale.** Its offset share moves toward 0.5, but A1 and A2 both have CI upper bounds ≤ 0. Then the offset is a symptom, not the cause of FE0's weakness.

---

## 14. What results can and cannot claim

- **Complete strategies.** The comparison is front end + frozen learner + declared readout. It says nothing on its own about the information in a code.
- **Cue-specificity.** CD removes consequence information entering through the front end. It does not remove the learner's own cross-record effects, which are the same kind in every arm. A2 therefore certifies "not a front-end shortcut". It does not certify "free of every sequential effect".
- **Scope.** R2/F3, frozen FULL_151, this world family. Nothing is claimed about other rungs, formats, V50 or general raw-byte learning.
- **Not S2 readiness** (roadmap §3.2b).
- **Mechanism readings are observations.** An FE-NI win does not prove that the offset caused FE0's loss. The factorial comparison, NLMS and window summaries are descriptive.
- **The detector convention** (§2.2) is a property of the tested system shared by all arms. It is not a biological claim.

---

## 15. Implementation plan (after review; no code now)

**New files in `BYTE_CORE_FEAS_1_20260923/`:**

| File | Contents |
|---|---|
| `feas1.py` | FE-NI, FE1-NI, XR and NI-READLOCKED classes. A learner factory around the frozen `F151Learner`. A recording subclass (R10) that reads the returned code without altering it. The CD subclass, which substitutes the consequence byte fed to the front end only. In-memory registration of arm specs in the imported `ARMS` dict, so the frozen `run_life` runs unchanged (disclosed; Q1) |
| `feas1_readout.py` | IRLS fitter, CAL-only guard, cross-fit (H2) |
| `feas1_analyze.py` | §7 metrics, §10 rules, report |
| `feas1_qualify.py` | Q1–Q15, including the byte-causal driver (Q2) and the read-purity harness (Q3) |
| `run_feas1.py` | Phase runner. Resumable; writes each gating file before the next phase; writes a completion marker; packages itself (no pid watcher) |
| `package_feas1.py` | Idempotent; checks the manifest (Q15) |

The frozen `bytecore.py` is imported read-only, with `sys.dont_write_bytecode` and snapshot checks (Q14).

**Gating sequence:**

```
Q → QUALIFICATION.json
K → READOUT.json (hashed) → CHECKPOINT.json (H1–H5; halt here if any)
D → SELECTION_D16.json (hashed)
[X → SELECTION_D32.json (hashed)]                     only under §10.2
SELECTION.json (hashed)
[CONFIRM_PLAN.json (hashed) → V]                      only if selected and n_V ≤ 64
DECISION.json → REPORT.md → MANIFEST.json → package
```

**Outputs (`results/`):**
- `QUALIFICATION.json`, `CHECKPOINT.json`, `READOUT.json`
- `DEV_CELLS.csv`, `DEV_CONTRASTS.csv` (A1, A2, A3, Path), `SELECTION_D16.json`, `SELECTION_D32.json` (if run), `SELECTION.json`
- `CONFIRM_PLAN.json`, `CONFIRM_CELLS.csv`, `CONFIRM_CONTRASTS.csv` (if run)
- `WINDOWS.csv`, `CHECKPOINTS.csv`, `XRECORD.csv`, `NLMS.csv`, `LIVE_INTERFACE.csv`, `RESOURCES.csv`
- `DECISION.json`, `REPORT.md`, `MANIFEST.json`
- `qualification/`, `jobs/`

The report is generated from the full metric table and flags any endpoint that was computed but not shown.

**Environment of record:** V88 venv (Python 3.11.5, numpy 2.2.6, scipy 1.14.1, numba 0.61.2).

---

## 16. Known risks

| Risk | Handling |
|---|---|
| FE1 sits at the selection boundary | Pre-set rules; one extension; confirmation may end `NOT_CONFIRMABLE_THIS_ROUND` |
| Winner's curse | Untouched confirmation; Δ_plan halved |
| FE-NI in exposure-driven form may be too weak to test the common-component account sharply | Stated in advance (§5.2). The read-locked form is excluded on interface grounds, not on evidence |
| CD changes the front end's input history (random consequences) as well as removing information | Byte frequencies are preserved. FE-LAG and FE0 identity checks bound disturbance where there is no path (Q4, H5) |
| The retention margin is lenient relative to FE0's day-probe η | Stated (§8.3). Day-probe Δ and its CI are always reported |
| The readout over-predicts on day probes | Part of the tested strategy; reported (R5) |
| In-memory patching of `ARMS` | Q1 bit identity; frozen files untouched (Q14) |

---

## 17. Deferred (roadmap §3.3)

- The timing-vs-framing split.
- The variable-filler format.
- The clock ×0.5/×2 time gate and localization of FE0's collapse at ×2.
- Wrapper invariance (V52 H-I).
- The read-locked and output-driven negative images.
- A sparser receptor table.
- Oracle arms on new worlds.

---

## 18. Changes from v2

| v2 | v3 |
|---|---|
| FE-NI updated m inside `read()`, which put the detector's timing inside the front end | Byte-arrival updates; pure `read()` for every front end; Q2 causal equivalence; Q3 read purity |
| Front-end contract "(byte, time) only" was not enforced | Interfaces declared (§2); the detector event reaches the learner only |
| R8 gated only in development, above 0.5 | CD intervention; A2 is required in development and confirmation; R8 is descriptive |
| UNPAIRED control in confirmation | Replaced by the CD arms, which split the gain vs FE0 |
| Retention: day-probe mean ≥ FE0 | Non-inferiority, CI lower bound > −0.02, with the margin argued |
| 0.02 called "materiality" and used as an adoption line | Screening line for selection only; adoption rationale in §8.2 |
| n_V formula powered only for a positive gain | ≥ 95% power per adoption condition at halved development estimates |
| No positive control for the shortcut intervention | XR fixture (H5) |
| Budget: 138 always, ≤ 394 ceiling | 232 always, ≤ 616 ceiling |

---

## Appendix A — FE-NI parameters

- **Samples per record:** 14 (one per byte arrival). N_m = 128 records × 14 = 1,792 samples.
- **Self-cancellation:** 1 − (1 − 1/1792)¹⁶⁸ = 0.0896 (12 records × 14 samples).
- **Convergence:** ln 0.05 / ln(1 − 1/1792) = 5,367 samples ≈ 383 records. Block 0 has 576 records.
- **Consequence content:** one sample in 14 (at the newline's arrival) contains the consequence byte's drive.

## Appendix B — Planning inputs (frozen data; erratum §5, §12)

| Quantity | Value |
|---|---|
| Per-world SD of in-stream Δη vs FE0 (DEV) | FE1 0.0143; FE-LAG 0.0112 |
| Per-world SD of day-probe Δη vs FE0 | FE1 0.0436 (DEV), 0.0402 (CAL); FE-LAG 0.0680 |
| FE0 day-probe η | 0.043 (DEV), 0.048 (CON) |
| FULL_151 episode life, wall time | 4–9 s |

## Appendix C — Literature (abstracts checked on Europe PMC, 23 Sep 2026)

| Paper | Used for | Label |
|---|---|---|
| Das et al., PNAS 2011 (PMID 21795607) | Odorant-selective short- and long-term habituation via glomerulus-selective potentiation of inhibitory synapses onto PNs; long-term habituation reduces odorant-evoked PN calcium. Basis of FE1 (short-term) and FE-NI (long-term, exposure-driven) | C+ |
| Ramaswami, Neuron 2014 (PMID 24945768) | Proposal: excitation creates an inhibitory negative image of itself; familiar input is transmitted less | T |
| Roselli, Ramaswami & Aranha, Bio-protocol 2024 (PMID 39525971) | Long-term habituation after 4-day exposure lasts > 24 h; basis for no decay of m | Protocol citing prior work |
| Hallem & Carlson, Cell 2006 | Receptor table (inherited) | R |
