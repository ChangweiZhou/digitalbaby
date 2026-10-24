# BYTE-CORE-FEAS-1 — Stage 1 byte-interface feasibility screen for FULL_151

**Namespace:** `BYTE-CORE-FEAS-1`
**Date:** 23 September 2026
**Status:** **DRAFT for review** by the owner and Codex. Not locked. No code has been written. Nothing has been run. No trajectories exist.
**Parents (read-only):**
- `BYTE_CORE_1_20260922`: the frozen S0–S1 round; its preregistered decisions stand.
- `BYTE_CORE_1_ERRATUM_20260923`: a post-hoc re-analysis of that round. Cited below as "erratum §n".
**Scope:** FULL_151 (`151ebd9e2d054ec9ec14`) with a byte front end, rung R2 only. The learner is frozen.

---

## 0. What this round decides

**Primary question (with a stopping rule).** Is there a deployable, label-free, biologically inspired byte front end that meets two requirements?
- FULL_151 must discriminate R2 byte cues under the V51 serialization (F3) **materially better than FE0**.
- It must do so **well enough that FULL_151 is off the floor** for S2.

**Supporting questions (descriptive, answered on the same lives):**

- **L — localization.** Which part of a record costs FE0 its discrimination? The candidates are record timing, extra bytes in the integration window, the constancy of those bytes, and wrapper variation.
- **C — clock.** Is the collapse at clock ×2 (erratum: AUC 0.507, down from 0.614) a learner effect or a front-end effect?

**This round does not:**
- re-decide any S0–S1 rule (D-B `FRONT_END_IS_BINDING` stands);
- run V50-REF, R8/R24, the streaming adapter or interleaved density;
- start S2;
- tune any parameter;
- use any frozen confirmation world.

---

## 1. Evidence this design builds on

All numbers are in the frozen results or the erratum package; nothing below is new data.

| # | Finding | Value | Source | Scope and uncertainty |
|---|---|---|---|---|
| E1 | FE0 learns something on F3, far below the oracles | In-stream AUC 0.613; NOLEARN 0.496; FE-TAIL4 0.772; FE-ID 0.845 (CON, n = 128) | erratum §2, PH-1 | Oracles are not deployable; they bound the problem |
| E2 | Most of FE0's loss appears with fixed framing | F1→F2 ΔAUC −0.145; F2→F3 −0.010 | erratum PH-3, PH-4 | F1→F2 changes content, length and byte timing together |
| E3 | With framing, FULL_151 carries a **learned** common offset | Share of cues below 0 (learned minus no-learning copy): F1 0.736, F2 0.985, F3 0.989; FE-ID 0.497 | erratum §3 | That the offset **causes** the loss is a hypothesis |
| E4 | The frozen decoder undersold FULL_151; AUC is immune to readout bias | F3 η: 0.0047 frozen, 0.0263 with bias+temperature fitted on CAL | erratum §2, §8 | Adding a bias was a post-hoc choice |
| E5 | FE1 recovers most of FE0's F2 loss, then pays at F3 | CAL AUC F1/F2/F3: FE1 0.771/0.729/0.664; FE0 0.761/0.611/0.606 | `ERRATUM_CELLS.csv` (CAL rows) | n = 8 CAL worlds. Once framing is handled, wrappers matter: this is V52's H-I territory |
| E6 | FE1 and FE-LAG gain on F3 | ΔAUC vs FE0: +0.052 [0.037, 0.067] and +0.061 [0.048, 0.074] | erratum §5 | Exploratory (n = 16); never confirmed |
| E7 | Byte timing alone costs a key-only code little | FE-TAIL4's input differs between F1 and F2 only in byte timing: F2 − F1 ΔAUC +0.003 [−0.006, +0.013] | erratum PH-13 (CAL, n = 8) | Shown only for the 4-byte key tail |
| E8 | Window learners on 8-byte records read the previous record's consequence | V50-REF, F1, side-1 **first-presentation** AUC 0.759 [0.750, 0.769] vs 0.47–0.50 at F2/F3. FE-LAG at F1: +0.089 in-stream, −0.104 on day probes | erratum §10, §11 | FE0 and the oracles show none. FE1 shows none clearly (DEV 0.518 [0.498, 0.538]). FE3 does, at F2/F3 |
| E9 | Day-probe AUC is a leak-free check, but noisier | Per-world SD of Δ vs FE0: 0.04–0.08 (day probes) vs 0.02–0.04 (in-stream) | erratum §11 | Day probes use wrapper 0 at F3 |
| E10 | The clock matters, but the cause is unknown | FE0 F3 AUC ×0.5 / ×1 / ×2 = 0.608 / 0.614 / 0.507 | `ERRATUM_CELLS.csv` (CON-E, n = 32) | ×2 changes the learner's teaching dose and the front end's timing together. The preregistered `CLOCK_INSENSITIVE` (on η) was floor-limited |
| E11 | The frozen proxy scored FE1 in a state it never has while learning | `interface_metrics` drives the front end through the whole life and measures **after the final 7-day rest**; FE1's habituation (τh = 1,800 s) has then decayed to ≈ 0 | `bytecore.py` `interface_metrics`; `LIFE` | Explains why the proxy vetoed FE1 |
| E12 | FE0's F3 codes carry more than FULL_151 uses | CODE-NLMS (online logistic with a bias term) on the same codes: η 0.079 | frozen `MILESTONE_TABLE.csv` | The bias term absorbs any common offset |

**FE0's timing geometry, computed from the constants** (τp = 10 s; 14-slot record at 2.143 s per slot; read at slot 12):
- Weights at the read moment: separator 0.81; key bytes 0.65 down to 0.22; header and prefix 0.18 down to 0.08.
- Framing is 37% of the input mass at F2, and the separator alone is 21%.
- This is why recency emphasis (FE2, FE3) failed: it amplifies the separator.

---

## 2. What went wrong in S0–S1, and what this draft changes

| S0–S1 design error | Consequence | This draft |
|---|---|---|
| No feasibility or futility gate before confirmation | 3,072 confirmation lives on a floor-level interface | 16-world screen first. Only candidates that pass go to confirmation. Explicit stop |
| The largest sample size, driven by one contrast (C-2a), was applied to all 24 cells (cap 128) | Precision bought where no decision needed it | Sample size per candidate from the screen SD: default 32, cap 64 |
| Front end chosen by a geometry proxy measured after a 7-day rest (E11), contrary to V52 §13 | FE1 vetoed; D-1 never run | No proxy gate; the learner outcome decides. Geometry is recorded live, at read time, as description only |
| Offset seen in smoke tests (amendment A9), but the decoder was left bias-free | η collapsed; the headline undersold FULL_151 | Primary statistic is AUC. η is reported with two CAL-fitted decoders and decides nothing |
| Format axis confounded length, spacing and content | Loss could not be localized | Time-matched 14-slot ladder (§3.1) |
| Verdicts issued at the floor (D-C, D-D, D-E) | Uninformative labels | Floor rule (§7.4); the clock block localizes instead of labelling |
| Q7 ran on FE-ID and accuracy, with a literal `True` as the broken-variant flag | Qualification did not test the designed claim | Each check's failing variant is run, archived and read back (§10) |
| Missing `INTERFACE_TABLE.csv`; fragile pid-based packaging watcher | Incomplete handoff | The packager checks outputs against the list in §11; the runner writes its own completion marker |
| Cross-record cue at F1 went unnoticed | V50-REF's F1 row inflated (E8) | Window learners run only where record length ≥ window; cross-record diagnostic per arm; day-probe check |

**Why V52's `NO_DEPLOYABLE_INTERFACE_CANDIDATE` did not trigger a gate.**
- I designed S0–S1 as a milestone-calibration round and put the front-end question second: Block D, with one candidate chosen by proxy.
- V52 had already shown that the interface binds for V50. The right design would have made the front-end screen the first block, with a go/no-go decision before any milestone confirmation.
- I also declared engineering smoke results off-limits for design changes. That rule stopped the offset seen before launch (A9) from changing the decoder or the plan.

---

## 3. World

Inherited unchanged from BYTE-CORE-1:
- §3.4 relevance schedule, with amendment A10: 1,824 trials, about 27 simulated days;
- reserved R2 bytes 0xF1 (aversive) / 0xF0 (neutral) from A1, and the innate detector;
- episode 30 s, gap 135 s; rests of 86,400 s and 604,800 s.

Worlds come from the frozen `create_world` under new namespaces: `FEAS1-ENG`, `FEAS1-CAL`, `FEAS1-DEV`, `FEAS1-CON`. Keys and seeds are audited disjoint from every BYTE-CORE-1 namespace.

### 3.1 Time-matched format ladder

Every 14-slot format uses the same timing:
- slot spacing 30/14 s = 2.143 s;
- **key bytes at slots 5–10**;
- consequence at slot 12 (the read moment);
- newline at slot 13.

| Format | Slots 0–4 | Slots 5–10 | Slot 11 | Status |
|---|---|---|---|---|
| **F1** (bridge to the frozen round) | 8-slot record: key at 0–5, consequence at 6, newline at 7; spacing 3.75 s | — | — | frozen |
| **TS** key + silence | silent | key | silent | new |
| **TV** key + variable filler | fresh uniform draws from the framing class | key | fresh draw | new |
| **F2** fixed framing | `~\|~\|@` | key | `:` | frozen, unchanged |
| **F3** wrappers | `~\|~\|` + header[w] | key with XOR mask[w] on bytes 0–1 | separator[w] | frozen, unchanged |

**Framing class** (14 bytes): `~ | @ # ? ! % & : = > ; / +`.
- These are exactly the bytes F2/F3 draw their framing from.
- They are disjoint from the key alphabet (`0-9A-Z`) and from the reserved bytes.
- TV draws are fresh at every presentation and every probe. They come from a world-seeded stream that is independent of the schedule and the consequences.

**Silent slots:** no byte is fed, but time still advances.

**What each step isolates:**

| Step | Contrast | Isolates |
|---|---|---|
| L1 | TS − F1 | Record timing: spacing, key position, delay to the read |
| L2 | TV − TS | Extra, varying bytes in the integration window |
| L3 | F2 − TV | Constancy of those bytes (a common component) |
| L4 | F3 − F2 | Wrapper variation (V52 H-I) |

**Window-learner constraint (E8).** A TS record feeds only 8 bytes, so a 12-byte window would reach into the previous record. FE-LAG therefore runs only at TV, F2 and F3, where its window equals the current context. No V50 arm runs in this round.

---

## 4. Arms

### 4.1 Candidates (eligible for confirmation)

| | **FE1** (frozen) | **FE-NI** (new) | **FE1-NI** (new) |
|---|---|---|---|
| Observable inputs | byte, time | byte, time, sample moments | byte, time, sample moments |
| Mutable state | p, h: 176 floats | p, m: 176 floats | p, h, m: 264 floats |
| Update rule | Frozen FE1: channel gain habituation, κ = 0.3, τh = 1,800 s | FE0 integration; negative image m of the sampled pattern (below) | FE1 integration, then the same negative image |
| Biological basis | Das et al. 2011 (C+) | Das et al. 2011 (C+); Ramaswami 2014 negative-image proposal (T) | Both |
| Parameters | Frozen defaults | τp = 10 s (FE0); N_m = 128 from the schedule; w = 1 | As the two parents |
| Leakage checks | FQ4, FQ6, cross-record diagnostic, day probes | FQ4, FQ5, FQ6, FQ10, cross-record diagnostic, day probes | Same as FE-NI |

**FE-NI, defined exactly:**

1. Integration is FE0: p ← p·e^(−Δt/τp) + R[b] at each byte.
2. At a sample, the sampled pattern is p̂ = p / max p (FE0's read).
3. The output is o = [p̂ − m]₊, renormalized to max 1. If every entry is 0, the output is the zero vector. `encode_sparse` then yields an empty KC code and no write occurs; such reads are counted.
4. After the output is taken, update m ← m + (p̂ − m) / N_m. There is no decay in silence; long-term habituation in Drosophila persists for more than 24 h.
5. **Sample moments** are once per record, when the learner reads its code: the innate detector has met a reserved byte, and that byte has not yet been fed.
   - The learner already receives the detector's output under constraint C4.
   - Probes read a clone, so the live m never changes during a probe (FQ6).
6. What FE-NI never sees: identity, role, key position, record boundary, wrapper id, and the consequence of the current record. m is formed only from pre-consequence samples (FQ5).

**Why N_m = 128.**
- **Self-cancellation.** Massed acquisition presents the same two cues for 12 consecutive samples. With N_m = 128 those samples make up at most 1 − (127/128)¹² = 9% of m, so a cue barely cancels itself.
- **Convergence.** m reaches 95% of a constant component after 382 samples. That is inside block 0 (576 samples).
- **Cost.** Codes drift during those first 382 samples. The acquisition-block profile shows the cost.

**Why full, input-driven subtraction (w = 1).** It tests the common-component hypothesis at full strength.
- In the more faithful form, inhibition scales with PN output. That form cancels a constant component only partly: the residual is c/(1 + w).
- It is deferred. It is worth testing only if the full form passes.
- The read-locked sampling is a modelling choice and is disclosed as such.

**Why FE1-NI.** Together with FE0, FE1 and FE-NI it completes a 2 × 2 factorial: habituation off/on × negative image off/on.
- The two mechanisms act at different levels. FE1 gates channel input in proportion to byte frequency. FE-NI subtracts the average sampled pattern.
- The interaction shows whether they remove the same component (redundant) or different ones (complementary).
- This matters for H-I. FE1's frequency-graded habituation partly suppresses wrapper bytes (each appears in half the records); a subtracted mean does not.

**Not in this round, and not dismissed:**
- **FE-SURP.** Dropped as defined: it is not invariant (Codex, 22 Sep).
- **Sparser receptor table (k = 3).** Parked. The TS step with FE-NI tests the shared key-alphabet component first.
- **Temporal-selection kernels.** Excluded. A kernel that peaks on the key slots would be tuned to the fixture's geometry, which is evaluator knowledge of key position.
- **Output-driven negative image.** Deferred, as above.

### 4.2 References, oracles and floor

| Arm | Role | Formats (screen) |
|---|---|---|
| FE0 | Reference for every contrast | F1, TS, TV, F2, F3 |
| FE-LAG | Engineering byte-only baseline; not promotable (C8) | TV, F2, F3 |
| FE-TAIL4 | Key-localization oracle; its F3 value sets the absolute bar | F1, F3 (TS only in FQ2) |
| FE-ID | Identity oracle, format-free by construction | F3 |
| FE0-NOLEARN | Floor | F3 |
| CODE-NLMS shadow | Online logistic with a bias term on each arm's own KC codes; learning rate chosen on CAL | every F3 cell |

### 4.3 Predictions, written before any run

| Arm | TS, vs FE0 | F2, vs FE0 | F3, vs FE0 | Own F3 − F2 | Share of cues below 0 at F2 | Other |
|---|---|---|---|---|---|---|
| FE0 ladder | — | — | — | ≈ −0.01 (E2) | ≈ 0.98 | \|L1\| < 0.03 (E7, key tail only); L2 + L3 ≈ −0.14, split unknown |
| FE1 | ≈ +0.01 (CAL F1 +0.010) | ≈ +0.10 (CAL +0.117) | ≈ +0.05 (CAL +0.057, DEV +0.052) | ≈ −0.065 (CAL) | ≈ 0.81 (CAL) | — |
| FE-NI | > 0 (removes the shared key-alphabet component) | ≥ +0.10, and FE-NI(F2) ≥ FE-NI(TS) − 0.03 | > 0 | ≤ −0.03 (wrapper residual survives the mean) | 0.4–0.6 | Gain at TV smaller than at F2. Gain for FULL_151 larger than for NLMS on the same codes |
| FE1-NI | — | — | ≥ max(FE1, FE-NI) − 0.03 if complementary; ≈ max if redundant | — | — | — |
| FE-LAG | — | ≈ +0.09 (CAL) | ≈ +0.07 (CAL) | — | — | — |

**What falsifies FE-NI's rationale.** Suppose FE-NI moves the share of cues below 0 toward 0.5 but F2 AUC does not rise. Then the offset is a symptom, not the cause of the loss (§8).

---

## 5. Measurement

| Role | Statistic | Definition |
|---|---|---|
| **Primary** | In-stream AUC per world | Over every scored consequence event (prequential; the prediction is recorded before the byte arrives). Invariant to any readout bias or temperature |
| Behavioural check | Native paired accuracy | Which cue of a trial is punished; ties count 0.5 |
| Leak-free check | Day-probe AUC per world | Relevant pairs at day checkpoints, 24 h after their last presentation |
| Cross-record diagnostic | Side-1 first-presentation AUC, paired vs FE0 | Flagged if its CI lies above 0. FE0 cannot carry a record to the next read: the previous record's bytes decay by at least e^(−16) |
| Representation vs learnability | NLMS AUC on the same codes (F3) | NLMS's bias absorbs any common offset |
| Secondary | η with two decoders, fitted on FEAS1-CAL (F3) | (a) temperature only, the frozen form; (b) bias + temperature. Also ρ = η_b(candidate)/η_b(FE-ID), descriptive |
| Mechanism | Offset | Share of cues whose learned-minus-no-learning response is below 0 (erratum §3) |
| Mechanism | Live interface statistics, at each read, in the life | Active KCs; Jaccard between consecutive presentations of the same cue (different wrappers at F3); Jaccard between the two cues of a trial; share of active KCs in common with the life-mean code; empty reads (NI arms) |
| Resources | Per arm | Front-end state bytes; wall time |

**No decision in this round uses η.** Later stages return to η as the headline, with the decoder form fixed before they run.

---

## 6. Blocks, worlds and budget

| Block | Content | Worlds | Lives |
|---|---|---|---:|
| **A** Qualification | FQ1–FQ11 with failing variants | 2 `FEAS1-ENG` worlds + fixtures | ≈ 30 |
| **K** Calibration | F3: FE0, FE1, FE-NI, FE1-NI, FE-LAG, FE-TAIL4, FE-ID. Decoders and NLMS learning rate | 8 `FEAS1-CAL` | 56 |
| **S** Screen | FE0 × 5 formats (80); FE1, FE-NI, FE1-NI × 4 formats (192); FE-LAG × 3 (48); FE-TAIL4 × 2 (32); FE-ID (16); FE0-NOLEARN (16) | 16 `FEAS1-DEV` (0–15) | 384 |
| **C** Clock | F3 at clock ×2: FE0, FE-TAIL4, FE-ID. The ×1 values come from S | same 16 | 48 |
| **X** Extension (conditional) | For each inconclusive candidate, plus FE0 and FE-TAIL4, at F3 | 16 more `FEAS1-DEV` (16–31) | ≤ 80 |
| **V** Confirmation (conditional) | Per passing candidate (≤ 2): the candidate and its UNPAIRED control; FE0; FE-TAIL4 | n = 32–64 fresh `FEAS1-CON` | ≤ 384 |

**Totals:**
- **Always run:** about 518 lives, ≈ 0.9 CPU-h. Frozen FULL_151 episode lives took 4–9 s each.
- **Worst case:** about 982 lives, ≈ 1.7 CPU-h.
- For scale, S0–S1 ran 3,856 lives, including V50 lives at 20–40 s each.

**Sizing, from frozen pilot data (erratum §5, §11):**
- The per-world SD of in-stream ΔAUC vs FE0 is 0.024–0.028. At n = 16 this gives a 95% half-width of about 0.015.
- The day-probe Δ SD is 0.049–0.084, giving a half-width of about 0.026–0.045. That is why the screen uses day probes only as a sign check.

---

## 7. Decision rules

Every rule below is on F3, uses 95% two-sided paired t intervals across worlds, and pairs arms on the same worlds.

### 7.1 Screen (n = 16)

A candidate **PASSES** if all three hold:

| Criterion | Rule |
|---|---|
| S1 — material gain | ΔAUC vs FE0: lower CI > **m = 0.03** |
| S2 — absolute bar | Mean AUC ≥ **the midpoint of FE0 and FE-TAIL4** on the same worlds (≈ 0.69 in CON) |
| S3 — leak-free sign | Day-probe ΔAUC vs FE0 > 0 **and** Δ paired accuracy > 0 (point estimates) |

The other outcomes:

| Outcome | Condition | Final? |
|---|---|---|
| **FUTILE** | ΔAUC upper CI < m, or day-probe ΔAUC upper CI < 0 | Yes |
| **INSUFFICIENT** | S1 holds but S2 fails: a real gain that leaves FULL_151 below half of the oracle gap | Yes; reported, not promoted |
| **INCONCLUSIVE** | Anything else | Goes to one pre-specified extension |

**Extension (Block X).** An INCONCLUSIVE candidate gets one extension: +16 worlds, then the same rules at n = 32. If it is still INCONCLUSIVE, it becomes **UNRESOLVED** and is not promoted.

**Why these bars.** Both are judgments; review is invited (§13).
- **m = 0.03:**
  - it is twice the screen's expected half-width, so noise alone cannot pass it;
  - it is about one fifth of the FE0 → FE-TAIL4 gap (0.159 in CON);
  - it is below FE1's and FE-LAG's exploratory gains, so the rule could have passed them on relative grounds.
- **The midpoint of FE0 and FE-TAIL4:** FE-TAIL4 is FE0 given perfect key localization. A front end that recovers less than half of that gap leaves FULL_151 near the floor on which S0–S1's time verdicts proved uninformative. FE1 and FE-LAG sat below it in exploratory data (0.665 and 0.674), so this bar is deliberately stricter than their evidence.

The screen decision is written to `SCREEN_DECISION.json` and hashed **before** any extension or confirmation world is generated.

### 7.2 Confirmation (Block V; fresh worlds)

**Who goes.** At most 2 candidates. If more than 2 pass, the two with the highest screen lower CI of ΔAUC go.

**Sample size, per candidate.** n = max(32, ⌈(t·SD/h)²⌉), using the screen SDs.
- For the in-stream criterion, h = (Δ̂ − m)/2.
- For the day-probe criterion, h = Δ̂_day/2.
- The cap is 64. If the required n exceeds it, the label is `NOT_CONFIRMABLE_THIS_ROUND` (the effect is too close to the margin) and the candidate is not run.

**Intervals.** CIs are taken at level 1 − 0.05/k, where k is the number of candidates confirmed.

A candidate is **CONFIRMED** only if all four hold:

| Criterion | Rule |
|---|---|
| C1 | ΔAUC vs FE0: lower CI > m |
| C2 | Day-probe ΔAUC vs FE0: lower CI > 0 |
| C3 | AUC − midpoint(FE0, FE-TAIL4): lower CI > 0 |
| C4 | Candidate − its UNPAIRED control, ΔAUC: lower CI > m (the gain requires pairing) |

### 7.3 Round outcome and stopping

| Label | Condition | Consequence |
|---|---|---|
| `CANDIDATE_CONFIRMED` | At least 1 candidate confirmed | It becomes S2's primary front end |
| `CANDIDATE_NOT_CONFIRMED` | Passed the screen, failed confirmation | No promotion; the report states which criterion failed |
| `NO_CANDIDATE_PASSES_SCREEN` | Nothing passes after the extension | **Stop.** No confirmation is run. The scope is these three candidates only. The ladder readings (§8) choose the next question |

Nothing is added after results are seen. Any new candidate goes into a new design on fresh worlds.

### 7.4 Floor rule

- No verdict about a factor (timing, clock, format) is issued for an arm whose reference cell is within 0.05 AUC of NOLEARN. That arm's results are reported as floor-limited.
- A ladder step is called **material** if |Δ| ≥ 0.03 and its CI excludes 0.

---

## 8. Readings, pre-written

**Localization (FE0 ladder, and each candidate's ladder):**

| Observation | Reading | Next question |
|---|---|---|
| L3 material and the largest step; share below 0 jumps from TV to F2; FE-NI restores F2 | A constant common component drives the loss | The output-driven negative image. If the best candidate's L4 is material, wrapper invariance (V52 H-I proper) |
| L3 material; FE-NI moves the share toward 0.5, but AUC does not rise | The offset is a symptom, not the cause | Learner side (α3 on overlapping codes); an S2/S3 question, not a front-end one |
| L2 material and the largest | Extra bytes in the window hurt whether or not they are constant | Temporal selection without position knowledge; NI is not expected to help |
| L1 material | Record timing itself matters | The byte clock becomes a mandatory factor in later rounds |
| Best candidate's L4 ≥ 0.03 | Wrappers cost discrimination once framing is handled | V52 H-I: link co-varying surface forms of one identity without labels |

**Representation vs learnability (F3, candidate − FE0):**
- **Gain for FULL_151 but not for NLMS on the same codes:** the information was already in the code; the candidate made it learnable for α3.
- **Gain for both:** the candidate changed what the code carries.

**Clock block (×2 vs ×1):**
- **FE-ID also collapses:** learner kinetics under the doubled teaching dose; the front end is not implicated.
- **FE-ID holds, FE0 and FE-TAIL4 collapse:** front-end timing (τp relative to byte spacing).
- **Only FE0 collapses:** integration dominated by framing under slower bytes.

---

## 9. What this means for S2–S5

- The **S2 draft** (22 Sep) stays unlocked. Its FE0 end-to-end run stays deferred until this round ends, as Codex recommended.
- **If a candidate is confirmed:** S2 uses it as the primary front end, with FE-ID as the capacity diagnostic.
- **If none is:** S2's FE-ID output-capacity question can run as its own small design, because it does not depend on the byte interface. Nothing about raw-byte performance may be claimed from it.
- **S2's readout needs a bridge check.** S2 reads "plastic minus no-plastic copy". That difference does **not** remove the offset: the offset persists in it (erratum §3). S2 needs either the CAL-fitted bias decoder or a front end that removes the offset.
- **Frozen results that stand regardless:**
  - the FE-ID ceiling on FULL_151 (AUC 0.845);
  - V50-REF's F2/F3 rows (its F1 row carries the cross-record cue);
  - the necessity of the innate route (erratum PH-7).
- **Undecided until the interface is solved:**
  - for FULL_151: streaming (D-C), density (D-D) and clock (D-E), apart from what Block C localizes;
  - any S3/S4 benefit on bytes.

---

## 10. Qualification (Block A)

Every check runs its failing variant. The variant's failure is archived, and the qualification summary is computed from those archives, never from constants.

| ID | Check | Failing variant that must be caught |
|---|---|---|
| FQ1 | **Timed-adapter bridge.** On F2 and F3, the new timed adapter and life driver reproduce the frozen `run_life` bit for bit for FE0 on 2 ENG worlds: every `ev_raw`, every `pr_raw`, and the final state digest | Slot spacing 30/13 must break identity |
| FQ2 | **Time matching.** A byte-time log shows key byte k at t0 + (5 + k)·30/14 and the read at t0 + 12·30/14, for TS, TV, F2 and F3. FE-TAIL4's whole-life digest is identical at TS and F3 | F1 must fail the log audit; FE-TAIL4 at F1 must give a different digest |
| FQ3 | **Filler.** TV filler bytes lie only in the framing class: never a key-alphabet or reserved byte. Re-seeding the filler stream changes the filler but not the world hash | A filler drawn from the cue's own key must be rejected |
| FQ4 | **Front-end API.** FE-NI and FE1-NI accept only (byte, time) and the sample call; passing any annotation raises | An FE-NI subclass that exposes `set_oracle(key_offset)` must be flagged, as FE-TAIL4 is |
| FQ5 | **Pre-consequence adaptation.** FE-NI's m trajectory differs between a world and its UNPAIRED twin by < 10⁻⁴. The bound covers the previous consequence byte's residue in p at the next read, 165 s later: e^(−16.5) ≈ 7 × 10⁻⁸ of its amplitude | Updating m after the consequence byte is fed must exceed the bound |
| FQ6 | **Probes are read-only.** The persistent-state digest (learner, plus the front end's p, h and m) is unchanged across every probe | A probe that samples the live front end must be caught |
| FQ7 | **CAL-only fitting.** Decoder and NLMS fits refuse any non-CAL world | Injecting a DEV world must raise |
| FQ8 | **Metric fixtures.** AUC gives 1 when perfect, 0 when reversed, 0.5 when constant. Paired accuracy and day-probe AUC give ≈ 0.5 on shuffled labels. The cross-record detector fires on the frozen V50-REF F1 lives and stays quiet on frozen FE0 F1 | The fixtures are the failing cases |
| FQ9 | **Source lock.** New files, plus hashes of the frozen files imported read-only | Appending a comment to a locked file must raise |
| FQ10 | **Empty reads.** FE-NI's empty-output rate is < 1% on ENG worlds | A variant with m initialized to 1 on every channel must exceed it |
| FQ11 | **Output manifest.** Every output in §11 exists before packaging | Removing one must fail the package step |

---

## 11. Implementation, provenance, outputs

**Folder.** `BYTE_CORE_FEAS_1_20260923/`. The frozen `bytecore.py` is imported read-only:
- `sys.dont_write_bytecode`;
- a before/after snapshot of the frozen directory, as in the erratum.

**New code:**
- timed rendering for TS and TV;
- a timed adapter subclassing the frozen FULL_151 learner;
- FE-NI and FE1-NI;
- `run_life_timed`, a copy of the frozen driver with timed rendering and live interface statistics. FQ1 proves it identical to the frozen driver on F2/F3.

**Runner order:**
1. A → K → S → C.
2. Screen decision, written and hashed.
3. X, if needed, then the decision again.
4. V, if anything passed.
5. Decision, report, package.

Jobs are resumable: a job with valid outputs is skipped. The runner writes a completion marker and packages the results itself. The report is generated from the full metric table.

**Outputs (`results/`):**
- `SCREEN_CELLS.csv`, `SCREEN_CONTRASTS.csv`, `LADDER.csv`
- `CLOCK.csv`, `XRECORD.csv`, `DAYPROBE.csv`, `NLMS.csv`
- `LIVE_INTERFACE.csv`, `DECODER_FITS.json`, `SCREEN_DECISION.json`
- `EXTENSION.csv` (if run), `CONFIRM_CELLS.csv` and `CONFIRM_CONTRASTS.csv` (if run)
- `DECISION.json`, `REPORT.md`, `MANIFEST.json`
- `qualification/`, `jobs/`

**Environment of record.** The V88 venv: Python 3.11.5, numpy 2.2.6, scipy 1.14.1, numba 0.61.2.

**Checks.** Only focused ones: the qualification checks and probe-level runs. No repository-wide build or global report.

---

## 12. Not claimed; known risks

- FE-NI is an idealized form of a biological proposal. Its read-locked sampling and full subtraction are modelling choices.
- AUC measures discrimination, not calibrated bits. The owner's quantity (ability gained per unit of relevant information) remains the headline for later stages.
- m = 0.03 and the midpoint bar are judgments, stated as such.
- Slot geometry is fixed across the 14-slot formats. The candidates have no position parameters, but FE0's exponential kernel already weights by recency. A position-shift probe tier is not included (§13).
- The day probes at F3 use wrapper 0 only. Held-out wrapper probes (F4) remain descriptive.
- FE-NI's codes drift during its first ~382 samples. This is reported in the acquisition profile, not corrected.
- A pass here establishes a front end for FULL_151 on this byte world only. It says nothing about V50, other rungs, or general raw-byte learning.

---

## 13. Open questions for review

1. Accept **m = 0.03** and the **midpoint absolute bar**, or set them differently?
2. **Confirmation:** automatic after the hashed screen decision, or a manual checkpoint?
3. **Clock block C** (48 lives): run it now, or defer it?
4. **Second new front end:** FE1-NI (the factorial), or instead the output-driven negative image, or a sparser receptor table (k = 3)?
5. Add a **position-shift probe tier** (framing length varied at probes) to test reliance on slot geometry?

---

## Appendix A — Pilot quantities used for sizing (frozen data; erratum package)

| Quantity | Value | Source |
|---|---|---|
| SD of in-stream ΔAUC vs FE0, F3 | FE1 0.028; FE-LAG 0.024 (DEV, n = 16) | erratum §11 |
| SD of day-probe ΔAUC vs FE0, F3 | FE1 0.049; FE-LAG 0.084 (DEV); 0.042–0.056 (CAL) | erratum §11 |
| FE0 / FE-TAIL4 in-stream AUC, F3 | 0.613 / 0.772 (CON); 0.606 / 0.768 (CAL) | erratum §2; `ERRATUM_CELLS.csv` |
| FULL_151 episode life, wall time | 4–9 s | frozen `JOB_INDEX.csv` |

## Appendix B — Literature (abstracts checked on Europe PMC, 23 Sep 2026)

| Paper | Used for | Label |
|---|---|---|
| Das S, …, Ramaswami M. *Plasticity of local GABAergic interneurons drives olfactory habituation.* PNAS 2011 (PMID 21795607) | Odorant-selective short- and long-term habituation. It arises from glomerulus-selective potentiation of inhibitory synapses onto projection neurons, and long-term habituation reduces odorant-evoked calcium in PNs. Basis of FE1 and FE-NI | C+ |
| Ramaswami M. *Network plasticity in adaptive filtering and behavioral habituation.* Neuron 2014 (PMID 24945768) | Proposal: a pattern of excitation creates an inhibitory "negative image" of itself through target-specific scaling of inhibition, so familiar input is transmitted less. The form of FE-NI | T (proposal) |
| Roselli C, Ramaswami M, Aranha MM. Bio-protocol 2024 (PMID 39525971) | Long-term habituation is induced by 4-day exposure and lasts more than 24 h. Basis for no decay of m in silence | Protocol citing prior work |

The other entries are reused unchanged from BYTE-CORE-1 Appendix B.
