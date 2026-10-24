# BYTE-CORE-1 — Byte-level persistent-learning instrument, milestone calibration, and FULL_151 byte conversion (S0–S1)

**Namespace:** `BYTE-CORE-1`
**Date:** 22 September 2026
**Status:** design for review — no code has been written; nothing has been run
**Milestones (fixed):** `V50-REF` (V50 core as run in V51) and `FULL_151` (`151ebd9e2d054ec9ec14`)
**Supersedes:** nothing. Prior rounds, results and locks are unchanged and are referenced read-only.

---

## 0. Decisions this design implements

Fixed by the project owner (22 Sep 2026):

1. The project is **byte level**. Raw bytes enter; there is no tokenizer. The instrument measures the persistent learning performance of the core — **ability gained from relevant experience**, not blind memorization. Large generalization/extrapolation is not expected at fly level and is tested later.
2. **V50 and FULL_151 are the two fixed milestones.** FULL_151 is treated as a half-finished line whose byte-level problem must be resolved.
3. The target quantity is **the amount of ability gained per unit of relevant information fed in during a given time period.** Topology will later change with time, so time is decisive; how to weight time is decided here (§5.4).
4. Anything sensible that no experiment has dismissed stays available. Mechanisms should be biologically inspired.

Decided in this design (with reasons in the referenced sections):

| Item | Decision | Section |
|---|---|---|
| What counts as competence | Predicting the **consequence** of currently presented cues. Framing regularities are a separate account. Random key bytes are not scored. | §5.1 |
| Headline metric | Prequential relevant compression η (bits saved per bit of relevant consequence information fed) | §5.2 |
| Time weighting | None inside η. Time enters through the schedule (delays, feed density, clock) and through mandatory time-stratified reporting. | §5.4 |
| Truth used for the headline | The byte actually delivered. Committed truth is used only for diagnostics, through one tested utility. | §5.3 |
| How FULL_151 meets bytes | Staged conversion; this round runs S0 (unchanged learner + fixed byte front end) and S1 (biologically inspired front ends) | §4, §8 |
| Constraints | §2 | §2 |

---

## 1. Why this round exists — established facts it builds on

All numbers below are from archived project files.

**V50 on bytes collapsed.** V51 (`V51_Mac_M3/v51_results/FIRST_RETURN.csv`, 16 seeds, load 128, `V50-REF`):

| Task | First-return correct (mean of 32) | Bits per byte |
|---|---:|---:|
| CUE8 (cues fed directly as PN patterns) | 30.69 (95.9%) | 0.99 |
| BYTE256 (14-byte records) | 1.56 (4.9%) | 7.31 |
| BYTE256, no-learning | 0.00 | 8.00 |
| BYTE256, unpaired | 0.13 | 8.28 |

**V52 located the failure in the interface and the write schedule, not in missing information** (`V52_refined_byte_interface_decomposition_design.md` §1; `V52_Mac_M3/v52_results/SUMMARY.md`):

- The byte code carried key identity (wrapper-0 → wrapper-2 nearest-neighbour retrieval 91.0% at BYTE128, 87.4% at BYTE256) but had low margin: same/different KC-code overlap ratio 2.25 on bytes versus 17.3 on CUE8.
- Phase statistics drifted; XOR on key bytes 0–1 aliased identities.
- Target-only writes beat all-byte writes by +0.199 [0.147, 0.251]. Removing framing writes alone: 0.000; removing key writes alone: +0.035 [−0.001, 0.072].
- Nondeployable oracles show the memory can use a better code (target-only writes: shipped code 31.5%, phase-Z + XOR-off 86.8%, invariant key-tail 96.6%).
- Verdict `NO_DEPLOYABLE_INTERFACE_CANDIDATE`. Hypothesis H-I (label-free discovery of an invariant identity code) was never pursued.

**FULL_151 has never seen bytes.** From code and the Stage 3A record:

- Input is a static PN pattern per cue (V88 worlds: 8 of 88 channels), encoded once per life by `encode_sparse` (FlyWire PN→KC matrix `B`, per-side top-k at 3.54%).
- One teaching event (`learn_pair`) is: cue A for 30 s with punishment co-presented if A is the punished cue, 135 s gap, then the same for cue B (330 s per pair).
- Learning and readout live in one compartment (α3): 1,770 writable KCs (1,226 + 544 support-expanded), 4 of 326 DANs and 6 of 97 MBONs used. Teaching is a scalar punishment. **It is a one-bit valence learner.**
- The readout is a paired choice: the punished cue's reader-relative response is lower.
- Mutable learner state 165,664 B; fast_tau 3,375.62 s; slow_tau 12,319,263 s (≈142.6 days).

**Audit issues this round is designed to avoid repeating** (project audit, 22 Sep 2026): committed-vs-delivered truth errors (G1, three occurrences), reuse of one development panel for confirmation (C1), uncertifiable retention bounds (B4), inconsistent materiality (B8), missing per-event archives (F2/F3), no common instrument for V50 and FULL_151 (A1/B6), codes cached once per life (D11).

---

## 2. Constraints (decided)

| # | Constraint | Reason |
|---|---|---|
| C1 | Raw bytes only; no tokenizer, no learned embedding table trained by gradient | Fixed by the owner |
| C2 | Learning is local: synapse-local eligibility × a broadcast modulatory signal computed by the circuit | The fly's KC→MBON rule (dopamine-gated, compartment-specific) |
| C3 | No backprop through layers; no Transformers | Fixed |
| C4 | The learner receives only the timed byte stream (and, for FULL_151 at S0, the output of a fixed innate reinforcer detector, §3.3). No labels, identities, roles, record boundaries, relevance tags, wrapper ids or evaluator truth. | Label-free learning |
| C5 | Bounded state. Every persistent or transient array in front end and learner is counted. Later, plastic edges will count as state. | Bounded core |
| C6 | No storage and replay of past inputs. Offline phases (not in this round) may use internally generated activity only. | Fixed |
| C7 | The mapping from output neurons to predictions is a fixed decoder with one calibrated scalar temperature per arm (§5.5). No trained readout. | Learning must live in the core |
| C8 | Mechanisms are tied to a fly finding with an evidence label (Appendix B). Engineering devices (oracles, NLMS, lag lines) are allowed only as labelled references. | Owner's preference for biological inspiration |

---

## 3. The byte world

### 3.1 Stream structure

The world emits a sequence of **timed bytes**. Bytes arrive only during **episodes**; between episodes there is silence. A silence gap is a sensory fact (no input), not a label.

Each episode carries one **record**: cue bytes, then one **consequence byte**, then a newline byte `0x0A`. Episodes are grouped as in V88: a **trial** presents two records (cue A, then cue B), each followed by a 135 s silent gap.

| Timing constant | Value | Source |
|---|---:|---|
| Episode duration | 30 s | FULL_151 cue event (`learn_pair`) |
| Post-episode gap | 135 s | `learn_pair` |
| Byte spacing inside an episode | 30 s / record length (bytes evenly spaced) | This design |
| Day rest | 86,400 s | V88 `LIFE` |
| Final rest | 604,800 s | V88 `LIFE` |

Holding the episode at 30 s for every format keeps FULL_151's teaching dose (`dt`) constant, so formats differ only in the cue code.

### 3.2 Record formats (input axis)

Byte constants are reused from V51 (`elm_v51_core_scope.py`, sha256 `99e69d9b…316c`).

| Format | Cue bytes | Wrappers | Length incl. consequence + newline |
|---|---|---|---:|
| **F1** key only | 6 key bytes | none | 8 |
| **F2** fixed framing | `~\|~\|` + header[0] + key + separator[0] | wrapper 0 only, no XOR | 14 |
| **F3** V51 serialization | `~\|~\|` + header[w] + key with XOR mask[w] on bytes 0–1 + separator[w] | training families w ∈ {0,1}, alternating per presentation | 14 |
| **F4** held-out wrapper (generalization tier) | as F3 with w = 2 | read-only probes only | — |

- **Keys:** 6 bytes drawn from `0-9A-Z` (V51), unique within a world.
- **Headers:** `@ # ? ! % &`. **Separators:** `: = > ; / +`. **XOR masks:** `(00,00) (11,22) (33,55) (0F,3C) (12,34) (21,43)`.
- F4 is reported but gates nothing in this round (owner decision 1).

### 3.3 Consequence alphabets (output axis)

| Rung | Consequence bytes | Structure | Learners in this round |
|---|---|---|---|
| **R2** | `y` (aversive), `z` (neutral) | Paired: in every trial exactly one of A/B gets `y` (V88 role semantics) | V50-REF, FULL_151 (S0/S1), references |
| **R8** | `a`–`h` | Independent consequence per cue | V50-REF, references |
| **R24** | `a`–`x` (V51 target set) | Independent consequence per cue | V50-REF, references |

**Innate reinforcer bytes (R2 only).** The bytes `y` and `z` are reserved: they never appear as key, header, separator or other consequence bytes. For FULL_151 at S0, a fixed **innate reinforcer detector** maps `y` → punishment on, `z` → punishment off, exactly as the fly's shock pathway is innate rather than learned.

- This is a fixed property of the world and of the animal, identical in every world. It is not per-task information.
- V50-REF receives `y`/`z` as ordinary bytes and must learn to predict them.
- Control `US-ROUTE-CUT` (§4.4) removes the innate route to show it is the teaching channel.

R8/R24 have no innate route; they need self-supervised consequence learning, which FULL_151 cannot express until stage S2 (§8).

### 3.4 Relevance schedule

The schedule ports V88 (`V88/minifly_rce_v87_followup_runner.py`, `LIFE` and `create_world`, sha256 `49b19beb…d435`) with pairs of byte cues in place of pairs of PN patterns.

| Element | Value (V88) |
|---|---|
| Acquisition blocks (pairs) | 48, 24, 24, 24, 12, 12, 12, 12 (168 trained pairs) |
| Acquisition | 6 consecutive trials per pair (massed), then immediate probe, 24 h rest, day probe |
| Challenges | after blocks 0, 3, 4, 5, 6 (5 challenges) |
| Challenge groups | 4 stable, 4 anomaly (wrong for 1 or 2 of the first cycles), 4 revision pairs per challenge |
| Challenge cycles | 12 interleaved cycles over all challenge targets |
| Reactivation | 6 episodes: 2 dormant stable pairs return in each of challenges 2, 3, 4; probed at exposures 0, 1, 2, 4 |
| Untrained pairs | 24 (never taught) |
| Final | 7-day rest, final probe |

**Additions in BYTE-CORE-1** (all audited at world creation):

1. **Novel-matched recovery control (`NOVEL-MATCH`).** In challenges 2, 3 and 4, add 2 never-seen pairs that follow exactly the reactivated pairs' exposure schedule and probes. Recovery is reported as reactivated minus novel-matched, at equal exposure. This closes the gap found on 22 Sep: V88's P7 has no exposure-matched novel control.
2. **Misleading reminder (`MISLEAD`).** In challenges 3 and 4, one dormant, previously learned pair (not revised, not reactivated) returns with its consequence flipped on cycle 1 only (R8/R24: a different random consequence), then correct for cycles 2–12. This measures false recovery. Basis: misleading reminders create false memories (Yang et al., *Nat Neurosci* 2026).
3. **Near-neighbour probes (`NEIGHBOUR`).** 24 never-taught cues, each copying a trained key with 2 of key bytes 2–5 replaced by random alphabet characters. Read-only probes at every day checkpoint; descriptive specificity measure (§5.6).
4. **Held-out wrapper probes (`F4`)** on F3 worlds at every day checkpoint; descriptive.

For R8/R24 the same schedule applies, with consequences drawn independently per cue. Anomalies deliver a different random consequence; revisions change it permanently.

**Committed truth (diagnostics only).** For revision pairs, the committed truth switches at the **first** revision presentation. For anomaly pairs it never switches. This deliberately differs from V88's bookkeeping, where `change_truth` fired after revision teaching (the defect class G1). The headline does not use committed truth at all (§5.3).

---

## 4. Arms

### 4.1 Milestone V50-REF (frozen)

- The V50 raw-BANC core exactly as run in V51 byte mode (`elm_v51_core_scope.py`, sha256 `99e69d9b…316c`):
  - 699 PN, 4,552 KC, 104 MBON;
  - one learned scalar per raw KC→MBON edge (15,523 floats ≈ 124 kB);
  - context = last 12 bytes, 8-bit ±1 code per byte after a fixed byte permutation, random 8-contact adaptor into PN space, 4 microframes, softmax over outcome codewords at τ = 0.25;
  - writes on every byte (V51 all-byte schedule).
- Its archived feature cache is kept as in V51 and counted separately as engineering state.
- V50 has no time-dependent persistent state, so rests only advance the clock. Qualification Q5 confirms this; if it proves false, the actual behaviour is recorded.
- A new world driver feeds the byte stream; the core code path is unchanged (Q5).

### 4.2 Milestone FULL_151 at S0 (learner frozen; byte front end added)

**Learner.** Genome `151ebd9e2d054ec9ec14` with values:

```
[0.164547, 0.75, -0.740848, -0.394336, 0.048862, 0.535677, 0.17006, -0.04835,
 0.30465, 0.848126, -0.176815, -0.047073, 0.300916, 1.0, -0.5, 0.12099, 0.0]
```

Locked sources and objects:

| Object | sha256 |
|---|---|
| `V82E/src/model_evo.py` | `ccdeaab99a889e6a74e47f95d641bba79ed95597b8bd902181e1c535942e25d7` |
| `V82E/src/common_evo.py` | `595211c5136bf5c13392d7390ce6c55ecb30f99fe21a537374eb3794cee68ad7` |
| PN→KC matrix `B` (installed explicitly, not redrawn) | `32a3726cb25a8255c1aef5696ff40aa303138c5d061a598004d2087d4da5f964` |
| Frozen reference reader `Q` | `1aa98dbc6c7424b3f0ddc8cb2af6e02a4d405f718d2de0d829bde05f3575f202` |

**Adapter `S0-EPISODE` (milestone form).** Per record:

1. The front end (§4.3) processes the cue bytes in real time.
2. Immediately before the consequence byte, the front end's 88-channel PN vector p is encoded live with `encode_sparse` (same `B`, same top-k) to the KC code x.
3. **Prediction (read-only).** The 5-second observer used in V88/`read()` gives the reader-relative response v = mean over output columns of (expression − reader prediction) for this single cue, in Hz. With the sign convention of V88's paired probe (punished cue lower): p(`y`) = σ(−v / T), T per §5.5. The native paired choice is also recorded.
4. **Teaching.** The innate detector reads the consequence byte. The learner receives one 30 s event with code x and punishment = 1 if `y`, then the 135 s gap. This reproduces `learn_pair` exactly, with byte-derived codes.

Episode segmentation uses only the silence gap. The moment "immediately before the consequence byte" is defined by the innate detector (reserved bytes), not by a record-boundary label.

**Adapter `S0-STREAM` (diagnostic).** The learner receives one event per byte (`dt` = byte spacing) with the live code, and punishment only during the `y` byte. This asks what FULL_151 does with an unsegmented stream; S2/S3 will need it.

### 4.3 Front ends (stage S1)

All front ends are fixed, have no learned parameters, map each byte into FULL_151's 88 PN channels, and are identical across worlds. Their state is counted (C5).

**Shared receptor table.** R ∈ ℝ^{256×88}: for each byte value, 6 channels without replacement with amplitudes U[0.4, 1.0], drawn once from the global seed `BYTE-CORE-1|RECEPTOR|v1`.

Basis: odorant receptor responses are sparse and combinatorial, with rich temporal dynamics (Hallem & Carlson, *Cell* 2006); KCs spike only when several input claws are coactive (Gruntman & Turner, *Nat Neurosci* 2013).

| Front end | Definition | Biological basis | State |
|---|---|---|---|
| **FE0** receptor blend | p ← p·e^{−Δt/τp} + R[b] at each byte; τp = 10 s; p normalized to max 1 at read time | Receptor code + PN integration (above) | 88 floats |
| **FE1** + habituation | Channel response R[b]⊙(1−h); per byte h ← h + κ(1−h)⊙R[b]; between bytes h ← h·e^{−Δt/τh}; κ = 0.3, τh = 1,800 s | Odor-selective, glomerulus-specific habituation via potentiated LN GABA inhibition (Das et al., *PNAS* 2011) | 176 floats |
| **FE2** + temporal contrast | Two traces τf = 5 s, τs = 30 s; p = [p_f − 0.5·p_s]₊ | Temporal dynamics of receptor responses (Hallem & Carlson 2006); project precedent M056 (partial positive) | 176 floats |
| **FE3** | FE1 + FE2 | as above | 264 floats |
| **FE-LAG** (engineering reference) | V51's last-12-byte lag code, projected into 88 channels by a random 8-contact signed adaptor, tanh, rectified | Delay lines; not claimed biological | 12-byte window |
| **FE-ID** (oracle) | Each key identity → a random 8-of-88 binary pattern (V88 native cue), wrapper-invariant | Evaluator-only; defines the interface ceiling on FULL_151's own substrate | — |
| **FE-TAIL4** (oracle) | FE0 applied to key bytes 2–5 only | Evaluator-only; V52 invariant-tail analogue | 88 floats |

Parameters (k = 6, τp, κ, τh, τf, τs, 0.5) are frozen defaults chosen from the timing constants (§3.1), not by search. Any change requires a written amendment before Block D.

### 4.4 Controls

| Control | Definition | Applied to |
|---|---|---|
| `NOLEARN` | Writes disabled; predictions still recorded | Each learner, R2/F3 (+ V50 R24/F3) |
| `UNPAIRED` | Consequence bytes permuted across the world's teaching events (V88 unpaired semantics) | Each learner, R2/F3 (+ V50 R24/F3) |
| `US-ROUTE-CUT` | FULL_151 S0: the innate detector is removed; `y`/`z` enter only as receptor inputs | FULL_151 FE0, R2/F3 |

### 4.5 References (evaluator-side; not learners, no claims)

| Reference | Definition | Use |
|---|---|---|
| `IDREF` | Given true cue identity: per-identity Beta (R2) or Dirichlet (R8/R24) with an exponential forgetting factor λ per exposure. λ and the output clip are fitted on calibration worlds. | Task-level scale |
| `CODE-NLMS` | Online logistic (R2) or softmax (R8/R24) regression with normalized step on the **same KC code** the learner sees (FULL_151 5,177-dim; V50 4,552-dim). Step fitted on calibration worlds. Global error (as in V75). | Interface-limited scale |
| `MARKOV2` | V51's order-2 byte-count model, as archived | Trivial byte-statistics baseline; regularity account |

---

## 5. Measurement

### 5.1 What is scored

- **Scored events:** every consequence byte in the stream. The world only presents currently relevant cues, so every presentation is a demand: acquisition, stable, anomaly, revision, reactivation, novel-matched and misleading-reminder presentations.
- **Why framing and keys are excluded from the headline:** in F3, 7 of 14 bytes are framing (learnable syntax) and 6 are random key characters (predicting them is rote memorization). Whole-stream bits per byte would be about 13/14 syntax and rote strings.
- **Why the headline is not memorization:**
  - retired cues never return and cost nothing when forgotten;
  - every presentation is re-rendered (F3 alternates wrappers), so strings do not repeat exactly;
  - the score is normalized by information fed;
  - interference, revision, anomalies, dormancy and return are all in the stream.

### 5.2 Headline: prequential relevant compression

For each scored event e with delivered consequence y_e over alphabet Y, the learner's prediction p̃_e is recorded **before** the byte is revealed. It is renormalized over Y and clipped to [ε, 1−ε] with ε = 10⁻³. Then:

- bits saved s_e = log₂ p̃_e(y_e) + log₂|Y|
- relevant information fed i_e = log₂|Y|
- **η_W = Σ_{e∈W} s_e / Σ_{e∈W} i_e** over a window W, computed per world

η = 0 means nothing learned; η < 0 means confident errors (following anomalies, false recovery); the achievable maximum is below 1 because first presentations cannot be predicted. Prediction accuracy (argmax correct) is reported beside η because it does not depend on decoder calibration.

**Reading:** bits saved accumulate only after ability is acquired, only while it is retained, and only when it is demanded. η therefore measures ability gained × how long it is used, per unit of relevant information.

**Normalized forms reported:** η / η(`CODE-NLMS`, same codes), and η(`CODE-NLMS`) / η(`IDREF`).

**Windows reported:** whole life; each acquisition block; each challenge; the final segment.

**Secondary (not in η):**
- Unrenormalized η for V50-REF (credits learning the consequence alphabet).
- Read-only checkpoint probes (immediate / day / final 7-day) as a retention account, in bits relative to uniform.

### 5.3 Truth

- The headline scores against the **delivered** byte, including anomalies. No bookkeeping truth enters η.
- Diagnostic endpoints (acquisition, maintenance, revision success, anomaly resistance, recovery) use committed truth (§3.4) through one shared utility with fixture tests (Q3).

### 5.4 Time

**No time weighting inside η.** Every scored event counts the same whenever it occurs.

- Discounting later events rewards cramming; up-weighting long delays rewards hoarding.
- The schedule already decides when ability is demanded.
- Unweighted sums add across windows, which later rounds need once topology changes over time.

**Time enters through three channels instead:**

1. **Schedule.** Delays are real physical time (§3.1) and act on the learner's kinetics.
2. **Factors (Block E).**
   - *Feed density:* the same events per acquisition block delivered massed (V88 native: 6 consecutive trials per pair) versus interleaved (cycles across the block's pairs). Information fed is identical.
   - *Clock:* episode and gap durations ×0.5 and ×2, with rests unchanged.
3. **Mandatory time-stratified profiles:**
   - η by number of prior presentations of the cue (acquisition profile);
   - η by time since the cue's last presentation: < 1 h, 1–24 h, 24 h–7 d, ≥ 7 d (retention profile);
   - η by life window (drift profile; later the topology-change profile).

### 5.5 Decoders and calibration

- Each arm has one scalar temperature T: FULL_151 on the reader-relative response v; V50-REF on its logits, starting from its native τ = 0.25.
- T is fitted to maximize η on **calibration worlds** (namespace `CAL`, 8 worlds, disjoint from all others), then frozen for every later use of that arm.
- Accuracy is reported beside η for every arm, so calibration cannot create an effect alone.

### 5.6 Diagnostics (descriptive unless named in §6)

| Diagnostic | Definition |
|---|---|
| Acquisition | η and accuracy over the first 1, 2, 4, 6 presentations of new pairs |
| Maintenance | η on stable pairs in challenges |
| Revision | Committed-truth accuracy on revision pairs at cycles 1, 2, 4, 8, 12 and after 24 h |
| Anomaly resistance | η on anomaly pairs after their wrong cycles, versus stable pairs |
| Recovery | Reactivated minus `NOVEL-MATCH`, at exposures 0, 1, 2, 4, in bits and accuracy |
| False recovery | `MISLEAD` pairs: bits on cycles 2–12 versus anomaly pairs on the same cycles |
| Proactive vs retroactive | Proactive: acquisition η of a new block versus the first block. Retroactive: day-probe loss on pairs from earlier blocks after a new block, versus after rest only. (Zhao et al., *eLife* 2023: the two differ in interval dependence and molecular pathway.) |
| Specificity | R2: mean p̃(`y`) on neighbours of aversive-trained cues minus on neighbours of neutral-trained cues. R8/R24: probability mass on the source cue's consequence. The 24 untrained pairs serve as the unrelated baseline. |
| Storage vs expression (FULL_151) | V78 trace-fate method at every day checkpoint: slow projection versus a no-write branch, de-clip rescue, fast-state removal |
| Regularity account | Bits saved on framing bytes (V50-REF, `MARKOV2`) |
| Held-out wrappers (F4) | η and accuracy on read-only wrapper-2 probes |

### 5.7 Interface metrics (front ends; computed without learning)

All computed on streams in the target format, on FULL_151's own `B` and top-k:

| Metric | Definition |
|---|---|
| J_same | Mean Jaccard of KC codes of the same identity across training wrappers |
| J_diff | Mean Jaccard across different identities |
| Margin M | J_same / J_diff |
| Retrieval | 1-NN identity retrieval, wrapper 0 gallery → wrapper 1 query (F4 query reported separately) |
| Stationarity S | Jaccard between codes of the same identity-and-wrapper at the start and end of a life-length stream |
| Framing share Φ | Fraction of active KCs unchanged when all key bytes are replaced by a fixed filler |

### 5.8 Resources

Reported per arm, in separate accounts, never summed into one headline:

- mutable learner bytes;
- front-end state bytes;
- fixed anatomy;
- observer/reader;
- engineering caches;
- write L1 and event counts;
- wall time.

---

## 6. Blocks, contrasts and statistics

### 6.1 Blocks

| Block | Content | Worlds |
|---|---|---|
| **A** | Qualification (§7) | fixtures + 2 engineering worlds |
| **B** | Interface metrics for FE0–FE3, FE-LAG, FE-TAIL4, FE-ID on F1–F3 (+F4); V50-REF's own front end as reference | 16 streams per front end × format |
| **C** | Milestone calibration: V50-REF on R2/R8/R24 × F1–F3; FULL_151 `S0-EPISODE` with FE0 (R2 × F1–F3), FE-ID (R2 × F1, F3), FE-TAIL4 (R2 × F3); `S0-STREAM` with FE0 (R2 × F1, F3); controls; references | pilot 8 (`PIL`), then n confirmation worlds (`CON`) |
| **D** | FULL_151 front-end screen at R2/F3 | 16 development worlds (`DEV`); confirmation of one pre-chosen front end versus FE0 on fresh `CON` worlds |
| **E** | Time factors on R2/F3 for V50-REF and FULL_151 with the Block-D front end: density (massed/interleaved), clock ×0.5/×2 | 32 fresh worlds per condition |

**Front-end choice without outcome selection.** The front end carried into Block D confirmation is chosen **from Block B interface metrics alone**: largest M subject to S ≥ 0.8 and retrieval ≥ FE0's. The choice is written down before any Block D learner result is inspected. Block D development results for all front ends are reported, not used for choice.

### 6.2 Sample size

- Unit of inference: the world. Arms are paired on identical worlds.
- The Block C pilot (8 worlds) estimates the between-world SD of η per cell.
- Confirmation n per contrast = max(32, ⌈(1.96·SD / 0.01)²⌉), capped at 128. This targets a 95% half-width of about δ/2.
- n is computed and written before confirmation worlds are generated.

### 6.3 Primary contrasts (one family, Holm-adjusted, 95% paired t intervals)

| ID | Contrast | Purpose |
|---|---|---|
| C-1a | V50-REF η − `NOLEARN`, R2/F3 | Milestone learns on bytes at all |
| C-1b | FULL_151 S0-FE0 η − `NOLEARN`, R2/F3 | Same |
| C-2a/b/c | FULL_151 S0-FE0 − V50-REF, R2 at F1, F2, F3 | Where the milestones stand on a common instrument (descriptive, no winner) |
| D-1 | FULL_151 with chosen front end − FE0, R2/F3 | Does a biological front end help FULL_151 on bytes |
| E-1a/b | Interleaved − massed, per milestone, R2/F3 | Time density |
| E-2a/b | Clock ×2 − ×0.5, per milestone, R2/F3 | Clock sensitivity |

- **Materiality:** δ = 0.02 η (2 bits saved per 100 relevant bits fed). It is fixed for all BYTE-CORE rounds unless amended in writing (audit B8).
- An endpoint with zero estimated variance is reported **unresolved** unless it is proved structural (audit B9).
- Everything else is descriptive, with intervals.

### 6.4 Namespaces

World seeds derive from `BYTE-CORE-1|<namespace>|<block>|<index>` with namespaces `ENG`, `CAL`, `PIL`, `DEV`, `CON`.

- Keys are audited disjoint within a world and across namespaces.
- `CON` worlds are never reused for choice in later rounds (audit C1).

---

## 7. Qualification (Block A)

Each check must be able to fail. Each has a deliberately broken variant that must be caught; the broken variant's failure is archived.

| ID | Check | Negative control that must fail |
|---|---|---|
| Q1 | Prediction is recorded before the consequence byte is revealed | Driver variant that reveals the byte first |
| Q2 | The learner API accepts only (byte, time) plus the innate detector output; evaluator annotations cannot reach it | Passing an annotation field must raise |
| Q3 | Committed-truth utility: fixtures for anomaly, revision (switch at first revision presentation), reactivation, misleading reminder | V88-style commit-after-teaching implementation must fail the revision fixture |
| Q4 | FULL_151 identity: with FE-ID codes on a V88-shaped schedule, the `S0-EPISODE` event sequence and final state are **bit-identical** to V88's `evaluate_life` path with the same PN rows | Adapter variant with a changed gap duration must break identity |
| Q5 | V50 identity: the new driver with the archived V51 world reproduces V51 BYTE256 load-128 V50-REF first return (mean 1.5625/32) and bpb exactly on seeds 50901–50916; rests do not change V50 state | A one-bit change to V50's τ must break reproduction |
| Q6 | Probes are read-only: full persistent-state hash unchanged across every probe | Probe variant that writes must be caught |
| Q7 | Innate route matters: `US-ROUTE-CUT` η ≤ `NOLEARN` η + δ on an engineering world | Positive control: intact route above `NOLEARN` + δ |
| Q8 | World audits: schedule counts, 6 reactivation episodes, 6 `NOVEL-MATCH` pairs, 2 `MISLEAD` pairs, reserved-byte exclusivity, key uniqueness, namespace disjointness | Corrupted world must be rejected |
| Q9 | Source locks enforced at every stage start (hashes in §4.1, §4.2) | Appending a comment to a locked file must raise |
| Q10 | Per-event rows are serialized for every world (prediction, delivered, committed truth, kind, exposure count, time since last presentation) | Missing row must fail the manifest check |

---

## 8. Decision rules after this round

This round promotes nothing. It produces the milestone table and decides the next stage.

| Rule | Condition (on `CON` worlds) | Consequence |
|---|---|---|
| D-A | Always | Publish the milestone table: η, accuracy and components per learner × rung × format, with ceilings |
| D-B | Interface efficiency ρ = η(FULL_151, chosen biological front end) / η(FULL_151, FE-ID), R2/F3 | ρ ≥ 0.5: design S2 next. 0.25 ≤ ρ < 0.5: one front-end round (V52 H-I, label-free invariance) before S2. ρ < 0.25: the front end is the binding constraint for FULL_151, as it was for V50 in V52 |
| D-C | η(S0-STREAM) < η(S0-EPISODE) − δ on F3 | S2/S3 must address within-record, continuous-time teaching before any self-supervised rung (R8, R24) |
| D-D | \|E-1\| ≥ δ for either milestone | Feed density becomes a mandatory factor in all later rounds |
| D-E | \|E-2\| ≥ δ for either milestone | The byte clock is a first-class parameter, to be fixed from biology rather than tuned |
| D-F | Always | V50-REF's R8/R24 rows set the byte-level reference that S2 must meet at equal or lower mutable state |

---

## 9. Roadmap after this round (interfaces locked, designs not yet frozen)

Each stage answers one question, uses fresh confirmation worlds, and is compared against V50-REF and FULL_151@S0 on this instrument.

| Stage | Change | Question | Biological basis (evidence label, App. B) | Entry condition |
|---|---|---|---|---|
| **S2** output | K real compartments. Per-compartment KC support is read from the FlyWire v630 graph (97 MBONs in 36 types, 326 DANs in 25 named types in the annotation table). Each compartment runs FULL_151's α3 rule and is taught one bit of a fixed consequence code. Fixed decoder; rungs R8 → R24 | Is output capacity, not the rule, what stops FULL_151 at bytes? | Aso et al. 2014 (34 MBONs of 21 types; valence encoded by the ensemble) — C+; Aso & Rubin 2016 (20 DAN types; distinct compartments with distinct rules) — C+ | D-B passed |
| **S3** teaching gate | Each compartment's DAN signal = consequence code minus its own prediction via MBON→DAN feedback, so predicted bytes teach little | Does prediction-dependent teaching remove V52-type write interference and close part of the 30.7 pp NLMS gap (V75)? | Felsenberg 2017 (C+); McCurdy 2021 (C); Cervantes-Sandoval 2017 (C); Hattori 2017 (C+); Bennett 2021 (T) | S2 run; D-C resolved |
| **S4** durable-write gate | The slow store's share is earned by repeated consistent evidence (today every event writes ≥ 8.7% of its alpha update to slow: allocation range [0.0866, 0.7385]) | Does earned commitment reduce tail harms and anomaly damage? | Huang 2024 (C+); Cell Rep 2023 cholinergic LTM gate (C); Lindsey & Litwin-Kumar 2024 (T) | S3 run |
| **S5** | Re-evolve the genes on this instrument with a widened gene box (4 FULL_151 genes sit at bounds); then time-varying topology | Operating point; structural change over time | — | S4 run |

**Other mechanisms, not dismissed and available** (no experiment has closed them in these forms):

- revision by opposing traces or retrieval suppression (Felsenberg 2018; Sabandal 2021);
- active protection (Zhang 2018);
- forgetting that tracks volatility (Brea 2014, T);
- compartment-specific rules (Aso & Rubin 2016);
- offline phases: renormalization; lowering the forgetting drive (Berry 2015); internally generated reactivation;
- homeostatic potentiation (Jiang & Litwin-Kumar 2021, T);
- state gating (Francés 2026);
- local rather than global APL inhibition (Lin 2014; the scalar V48 form alone was closed);
- upstream PN→KC plasticity (needs the live encoding introduced here);
- bounded structural plasticity;
- second-order conditioning;
- a second NO/cGMP channel.

**Closed by experiment, in their exact tested forms only:** C_FORGET (OMNIBUS10); CONTRACT_0.05 at load 48; the scalar KC↔APL controller (V48); unsigned linear MBON recurrence at radius 0.25 (V49); ELIG-WRITE on bytes (V51); V52 deployable encoders E0–E2; additive consolidation bonus (M037); four sparse compartments with local covariance (M060); coupled-synapse screens (M041, M042); recipient redistribution within α3 (V69, V72); each Stage 3A deletion.

---

## 10. What this round does not claim

- No promotion, minimality or biological-fit claim.
- The innate reinforcer bytes are an interface convention modelled on innate US detection, not a finding.
- FE0–FE3 are biologically inspired front ends, not reconstructions of fly olfaction; their parameters are defaults, not fits.
- η values on different rungs reflect different tasks. Cross-rung comparisons are descriptive.
- F4 (held-out wrappers) is a descriptive generalization tier only.
- V50-REF's absence of time dependence (if confirmed by Q5) means rests cannot cause it to forget; its losses are interference only.

### 10.1 Known risks

| Risk | Why it matters | Handling |
|---|---|---|
| Clock mismatch | FULL_151's kinetics were evolved in the PN world; a wrong byte clock shifts every result | Block E clock factor (×0.5 / ×2); rule D-E |
| Habituation of the cue itself | During massed acquisition a key recurs every 330 s, so FE1/FE3 also habituate the key's own channels, not only framing | Visible as FE1 < FE0 on F1 (no framing); reported, not tuned away |
| Per-cue decoding of a paired learner | FULL_151 was evolved on paired choices; its single-cue response may be poorly calibrated | Per-arm temperature on `CAL` worlds; accuracy and the native paired choice reported beside η |
| V50's feature cache | An engineering cache of recent contexts, not biological state | Kept as archived for milestone fidelity; counted separately; flagged in every table |
| Innate reserved bytes | A convention that makes R2 label-free for FULL_151 but has no self-supervised analogue | Limited to R2; R8/R24 require S2/S3 self-supervision |
| Life length | One life spans about 27 days of simulated time (13 day-rests, the 7-day final rest, and about 6.6 days of trials); topology change will need longer lives | Deferred to S5; window profiles are already additive |

---

## 11. Deliverables and provenance

```
BYTE_CORE_1_20260922/
  DESIGN.md                       this file (frozen at launch; amendments appended, never edited in place)
  SOURCE_LOCK.json                hashes of all locked sources and objects, environment
  PROTOCOL_LOCK.json              constants, namespaces, δ, n rule, front-end parameters
  qualification/                  Q1–Q10 results, including the failing negative-control variants
  worlds/                         world manifests (seeds, keys, schedules, audits)
  events/                         per-world per-event rows (Q10)
  results/
    MILESTONE_TABLE.csv
    INTERFACE_TABLE.csv
    PROFILES_TIME.csv             acquisition / retention / window profiles
    DIAGNOSTICS.csv
    CONTRASTS.csv                 primary family with Holm-adjusted intervals
    RESOURCES.csv
    DECISION.json                 rules D-A … D-F evaluated
    REPORT.md
```

- Launch uses the project root `/Users/pencilbard/Downloads/second_try_copy`. No prior directory is written to.
- Environment of record, matching the FULL_151 locks: Python 3.11.5, numpy 2.2.6, scipy 1.14.1, numba 0.61.2.

---

## 12. Parameters frozen by this design (for review before launch)

| Parameter | Value | Rationale |
|---|---|---|
| Episode / gap | 30 s / 135 s | FULL_151 `learn_pair` |
| Rests | 86,400 s / 604,800 s | V88 |
| Reserved R2 bytes | `y` aversive, `z` neutral | Outside V51 key and target alphabets |
| R8 / R24 alphabets | `a`–`h` / `a`–`x` | V51 target set |
| Receptor table | 6 of 88 channels per byte, U[0.4, 1.0], seed `BYTE-CORE-1\|RECEPTOR\|v1` | Sparse combinatorial code |
| FE0 | τp = 10 s | About one third of the episode |
| FE1 | κ = 0.3, τh = 1,800 s | Framing bytes (every record) stay habituated across the 165 s cycle; a key seen about hourly recovers |
| FE2 | τf = 5 s, τs = 30 s, weight 0.5 | Fast-minus-slow contrast within an episode |
| ε | 10⁻³ | Bounded log loss |
| δ | 0.02 η | Materiality, all BYTE-CORE rounds |
| Worlds | `CAL` 8, `PIL` 8, `DEV` 16, `CON` n per §6.2 (32–128) | §6.2 |
| Front-end choice rule | max M s.t. S ≥ 0.8 and retrieval ≥ FE0 | §6.1 |

---

## Appendix A — Mapping to the 22 Sep audit

| Audit issue | How this design addresses it |
|---|---|
| A1, B6 no common instrument for V50 and FULL_151 | Both milestones run on the same worlds and metric (Block C) |
| B1 generalization unmeasured | F4 held-out wrapper tier, descriptive |
| B2 byte input unmeasured since V52 | The whole instrument is byte level |
| B4 uncertifiable retention | Retention appears in η through returning demand, with paired world-level intervals instead of simultaneous Hoeffding bounds |
| B5 retired-fact rule | Explicit: retired cues never return, so forgetting them is free; returning cues are scored |
| B8 inconsistent materiality | δ fixed across rounds |
| B9 zero-variance endpoints | Reported unresolved unless proved structural |
| C1 panel reuse | `DEV` for choice, fresh `CON` for confirmation, never reused for choice |
| D11 codes cached once per life | Live encoding in every adapter |
| F2, F3 missing per-event archives | Q10 per-event rows |
| G1 truth-timing defect | Headline scored on delivered bytes; committed truth through one tested utility with a failing V88-style fixture |
| G3 vacuous checks | Every qualification check has a negative control that must fail |

## Appendix B — Literature used, with evidence labels

Labels: **R** convergent experimental support; **C+** causal with follow-up; **C** causal, limited follow-up; **T** computational model. Every entry below was checked against its original abstract (Europe PMC) on 22 Sep 2026; claims are used only at abstract level unless stated.

| Paper | What is used here | Label |
|---|---|---|
| Hallem & Carlson 2006, *Cell* | Sparse, combinatorial receptor responses with informative temporal dynamics → receptor table | R |
| Gruntman & Turner 2013, *Nat Neurosci* | KCs require coactive inputs across claws → integration before KC coding | C+ |
| Das et al. 2011, *PNAS* | Odor-selective habituation via glomerulus-specific LN GABA inhibition → FE1 | C+ |
| Lin et al. 2014, *Nat Neurosci* | KC–APL feedback keeps codes sparse and decorrelated; needed for fine discrimination → top-k, local inhibition candidate | C+ |
| Aso et al. 2014, *eLife* | ~2,000 KCs converge on 34 MBONs of 21 types; ensemble encodes valence → S2 | C+ |
| Aso & Rubin 2016, *eLife* | 20 DAN types, distinct compartments, distinct learning rules; timing can reverse memory sign → S2, timing caution | C+ |
| Felsenberg et al. 2017, *Nature* | Re-evaluation depends on prediction accuracy via MBON→DAN pathways → S3 | C+ |
| McCurdy et al. 2021, *Nat Commun* | Omitted punishment encoded as reward through a cholinergic relay → S3 | C |
| Cervantes-Sandoval et al. 2017, *eLife* | KC→DAN reciprocal synapses, positive feedback required for learning → S3 | C |
| Hattori et al. 2017, *Cell* | α′3 MBON responses suppressed by familiarity, dopamine-dependent → S3 (predicted input teaches less) | C+ |
| Huang et al. 2024, *Nature* | Fast compartments weaken inhibitory MBON feedback to PPL1-α′2α2/α3; with further conditioning these DANs gate LTM → S4 | C+ |
| Cell Rep 2023 (PMID 37590142) | α/β cholinergic neurons block LTM after weak single training → S4 | C |
| Zhao et al. 2023, *eLife* | Proactive and retroactive interference differ in interval dependence and molecules → §5.6 | C |
| Yang et al. 2026, *Nat Neurosci* | Silent traces recovered by reminders; misleading reminders create false memories → `MISLEAD` | C |
| Jacob & Waddell 2020, *Neuron* | Spacing forms an additional opposite-valence safety memory → density factor | C |
| Bennett et al. 2021, *Nat Commun* | DAN prediction errors from MBON feedback in a model → S3 | T |
| Lindsey & Litwin-Kumar 2024, *eLife* | Recall-gated consolidation shields long-term memory from spurious changes; spacing effects → S4 | T |

Details used from paper bodies rather than abstracts (Huang 2024's spacing window, Aso & Rubin's per-compartment capacities, the Jiang & Litwin-Kumar homeostatic term) come from the project's earlier paper-level memos and are not relied on in this round.

---

## Amendments before launch (22 Sep 2026)

Written after implementation and engineering smoke tests, **before** the source lock and before any calibration, pilot or confirmation world was run. Engineering smoke results (2 worlds per block, scratch directory outside the project) were used only to find defects; none of those numbers is reported as a result.

| # | Amendment | Reason |
|---|---|---|
| A1 | Reserved R2 bytes are **0xF1 (aversive) and 0xF0 (neutral)**, not `y`/`z` | V51's XOR mask 0x22 maps key byte `X` (0x58) to `z` (0x7A), so `z` could appear inside a key. Bytes ≥ 0x80 cannot arise from the key alphabet, framing or any XOR mask used here. Audited per world (Q8). |
| A2 | Q5 reproduces V51's **byte** runs (seeds 51601–51616; checked on 51601 and 51602), not V50's cue seeds 50901–50916 | The design text named the wrong seed set. The reproduction is exact (loss sum equal to all printed digits). |
| A3 | Storage-vs-expression diagnostic this round = **fast-state-removal rescue** of wrong/correct day-probe choices on the relevant set | De-clip rescue and the matched no-write branch of V78 need model internals not exposed by the frozen API; deferred, not dropped. |
| A4 | D-1 (chosen front end vs FE0) is confirmed on the **Block C confirmation worlds** | Those worlds are fresh relative to the front-end choice, which used only Block B streams; FE0's arm is shared. |
| A5 | Block B streams use namespace `DEV`, block `B`, indices 1000–1015 | Disjoint from Block D development worlds. |
| A6 | Fitted-on-`CAL` quantities: decoder temperature T per (parent arm, rung, format) on a log grid (FULL_151: 141 points, 0.05–10⁴; V50: 121 points, 0.005–20); NLMS normalized step ∈ {0.1, 0.3, 1, 2}; IDREF forgetting λ ∈ {0.5, 0.7, 0.8, 0.9, 0.95, 1.0} per rung. Controls and Block E conditions inherit their parent cell's T. | Makes §5.5 operational. A T at the top of the grid means the arm carries no usable per-cue signal (decoder ≈ 50/50). |
| A7 | Contrast statistics: paired t, unadjusted 95% intervals, **Holm-adjusted p-values** across the 10-contrast family; "resolved" = Holm p < 0.05 | Holm adjusts tests, not intervals. |
| A8 | Only the reserved R2 bytes are exclusive. R8/R24 letters can occur inside XOR'd keys, as in V51. | Keeps the V51 target alphabet. |
| A9 | FULL_151's per-cue score is the reader-relative response v (expression − frozen-reader prediction), p(aversive) = σ(−v/T); the baseline-relative response is recorded too. The **native paired choice** (which cue of a trial is punished; ties = 0.5) is reported beside η for both milestones. | Engineering smoke showed FULL_151's per-cue responses carry a common offset, so a zero-threshold per-cue decoder undersells it; the paired choice is FULL_151's native readout. The pre-registered η decoder is unchanged. |
| A10 | World additions as implemented: `NOVEL-MATCH` = 6 extra never-acquired pairs (2 per challenge 2–4); `MISLEAD` drawn challenge-fresh from acquired pairs; day probes also cover the 24 untrained pairs, 24 neighbours and, on F3, wrapper-2 (F4) probes of the relevant set. One life = 1,824 trials, ≈ 27 simulated days. | Makes §3.4 operational. |
| A11 | Retroactive diagnostic this round = final 7-day probe of all trained pairs; proactive = acquisition η by block. The "versus rest only" comparator is deferred. | Needs a no-learning branch per block. |
| A12 | The V50-REF clock contrast (E-2a) is a **structural zero**: V50-REF has no time-dependent state (Q5), so it is reported as such rather than as unresolved. | §6.3 zero-variance rule. |
| A13 | Streaming adapter (`S0-STREAM`) delivers each byte as its own event; the punishment coincides with the reinforcer byte's own code. | As designed; noted because smoke tests showed it changes what is associated with the US. |
