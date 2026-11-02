# MiniFly V175 — Learned Local Pattern Separation (LPS-16) to reduce write interference

**Version:** V175 / LPS-16-D1  
**Date:** 2026-11-01  
**State:** DESIGN_ONLY; no implementation, training, science run, or adoption  
**Lineage:** WCFI-V1's native Full151/FE-PERM/DELTA writer; cross-checked against separately developed V172–V174  
**Owner's target:** generic local learning, persistent acquisition and revision, and eventually unseen transfer per sample and CPU; arithmetic is a diagnostic task, not a hand-coded objective.

## 0. Decision and evidentiary boundary

Test exactly one candidate architectural mechanism: **sparse plastic lateral inhibition between Kenyon cells (KCs)**, learned without answers, labels, task IDs, phases, model-output gradients, or past-example lookup. Its purpose is to reduce harmful coactivation of otherwise distinct input codes, thereby reducing the destructive effects of later native value writes.

This is **not** an SNN or an attention mechanism. Keep binary sparse KC activity, the native fast/slow synaptic memory, C16 own-byte/EOS output, and the existing signed DELTA teacher. It does not implement Alex Wang's key/value softmax, global covariance inverse, MLP, replay, or a plasticity-protection rule. Local consolidation/protection is a distinct possible successor experiment, not an extra V175 arm.

Reason to test: Wang's associative-memory exposition explains how nonorthogonal codes create crosstalk; this is a mechanism hypothesis for MiniFly, **not** a transfer of random-key capacity formulas to highly correlated binary KCs. Past measured evidence:

- WCFI-V1, two opaque-association development worlds: a strong DELTA writer acquired most NEW facts but reduced final OLD complete answers from 50/64 before NEW to 33/64 and 25/64 under its D-high arms; matched write-inhibited observation and equal-time idle controls were identical at final output in each world. Do not call all forgetting synaptic overwrite without intervention.
- V172 (Claude, 2026-10-27): native DELTA improved taught recall relative to N2B, but NEW caused marked OLD loss.
- V174 (Claude, 2026-10-28): replay produced +15 and +17 final complete answers over matched blocked observations in the development worlds, yet the prespecified net-acquisition gate failed; one-sided KC homeostasis did not improve retention. The replay arm used extra actual writes.
- An earlier capacity experiment quadrupled KCs (5,177 to 20,708), improving E1 recall at a roughly 3.15x measured CPU cost. V175 holds KC count fixed.

**Question:** can an online, cue-only, locally learned inhibitory circuit improve *both* later OLD survival and total final complete answers, at fixed native writer, feedback, exposure, KC count, and comparable measured compute?

**Scope of any outcome:** E0 encoder/learning correctness, E1 taught-association acquisition, E1-R later retention. No E2 general relevance selection, no E3 never-taught arithmetic/general reasoning, and no core promotion can be established by this experiment.

## 1. Frozen model and controls

Use WCFI-V1's source-locked Full151 native bank, FE-PERM, and C16 output (256 payload symbols plus EOS=256); follow the checked clone, write/no-write and prospective-time read conventions. In particular:

- 16 canonical native compartments share the **same** presynaptic KC code; KC wiring `B`, native fast/slow/adaptation equations, native reader, 257-class C16 codebook, and vocabulary softmax remain unchanged.
- FE-PERM receptor trace decay 0.95 per byte, fresh adaptation reset at record start, original elapsed native clocks, side-specific KC top-k eligibility and native active fraction remain unchanged.
- Homeostasis OFF, `theta=0` and no adaptive theta updates; the V172/174 homeostasis results are contextual references, not a concurrent treatment.
- DELTA/D-high writer at gain 1, with output vector `v`, target-code row `C_y`, and observed-target count `n`:

  \[
  a_j(v,y,n)=w(n)\frac{\tanh((v_j-\overline v)/2)-C_{yj}}{2},
  \qquad
  w(n)=\frac1{16}+\frac{15}{16}\min(n/1024,1).
  \]

  Count payload and EOS target arrivals, including write-inhibited diagnostics; at a native target the 16 `a_j` values are passed to the unchanged signed teacher. Do not silently alter signs, dosage, reset, coefficient clipping, or prospective read timing. This is the *same* writer in both longitudinal arms.
- Each arm is a separate, matched canonical birth for a given world; immutable native assets and dataset streams agree, but mutable memory is never shared between arms.
- **BASE**: exactly the old encoder and writer; no lateral-inhibition state or dynamic changes.
- **LPS-16**: replace only the per-side KC code-selection computation by `Section 2` below; all other rules are identical. No hindsight, replay, extra value writer, global optimizer, task parser, answer decoder, or attention/key-value table.

This comparison is the **effect of adding one learned inhibitory circuit**, not a pure comparison of learned versus frozen inhibition. A later distinct design must compare plastic inhibition with a strength-matched frozen-inhibition circuit to isolate plasticity itself.

## 2. Fully specified LPS-16 candidate

### 2.1 Sparse, fixed local topology

Let `N` be the original 5,177 KCs, partitioned by the native `kc_side` array; use the existing `B` and per-side `k_s=ceil(active_fraction*N_s)`. Build one graph shared by all 16 native compartments:

1. For each side, independently permute KC indices with a deterministic PCG64 generator seeded from SHA256 of `V175|topology|side`. These seeds are topology-only, not output-/task-dependent.
2. In the permuted cyclic order, connect each KC to offsets ±1 through ±8, symmetrically and without duplicates; this gives degree 16 **within the same side**, no cross-side edges, and exactly `8N` undirected edges when side sizes exceed 16.
3. Store one shared nonnegative float32 strength `L_{ij}=L_{ji}` per undirected edge; all start at zero. Topology and coefficient precision are frozen, with no learned edges outside this graph.

This sparse graph is an engineered microcircuit, **not** a claim that the fly connectome has exactly these KC–KC synapses. Its purpose is to test a biologically plausible class of local competition with an explicit computational budget.

### 2.2 One-pass inhibitory competition

For a receptor trace `p` at a native encoding instant, compute ordinary raw KC drive

\[
u_j = (B^\top p_{\mathrm{PN}})_j-\theta_j.
\]

Let `E_s={j: kc_side_j=s and u_j>0}`; use the native deterministic per-side descending-score ordering, breaking ties by increasing canonical KC index. Define `TopK_s(q;E_s)` to select the first `min(k_s,|E_s|)` eligible cells ranked by `q`, yielding a binary indicator. Compute

\[
z^{(0)}=\operatorname{TopK}(u;E),\qquad
d_j=\sum_{k:(j,k)\in G}L_{jk}z^{(0)}_k,\qquad
z=\operatorname{TopK}(u-\gamma_s d;E).
\]

Use **one** synchronous competition pass, not an iterative solver or dynamic selection of rounds. Keeping eligibility `E` based on *pre-inhibition positive drive* ensures identical per-side population counts to BASE even if the inhibited rank changes; the identity of active KCs, not their total number, is the intervention. At zero inhibition `L=0`, output must be bitwise identical to the native encoder. If a side has fewer than `k_s` eligible cells, both arms use that smaller number and disclose it.

For physical scale, determine one fixed `gamma_s` per side **before any main run**, using 64 unlabeled, independent five-byte prompts in the technical calibration world. Evaluate their first-answer-site native `u`; let `sigma_s` be the median across prompts of the within-side standard deviation of positive `u_j`, and set

\[
\gamma_s=\tfrac14\sigma_s.
\]

No target, score, future key, or retrieval value can influence calibration. Nonfinite or zero `sigma_s` aborts the technical stage. Store the exact calibrated scalars and source/seed receipts. **No outcome-based gamma grid or online normalization.**

### 2.3 Cue-only, edge-local anti-Hebbian update

For each completed physical observed record, cache exactly **one** `z^(0)` sampled at the first answer-site encoder context after the prompt but *before* the first answer target is observed. Keep `L` fixed throughout that record (including answer and EOS), so teaching never uses a code that changes because of a mid-record inhibitory update.

After the EOS teaching event and before the next record, update **every fixed graph edge** exactly once:

\[
L_{ij}\leftarrow
\min\left(1,\ (1-\delta)L_{ij}+\eta\,z_i^{(0)}z_j^{(0)}\right),
\quad
\delta=\frac1{2048},\quad\eta=\frac1{32}.
\]

The value is a local coactivation trace that raises later inhibitory competition between frequently coactivated neurons. It **does not** use correctness, output label, desired answer, OLD/NEW status, or a learned value. Use exactly this one rule: no free reliability estimator, winner-history table, running covariance inverse, or task-specific penalty. Apply the same cue-only update to actual observed records even when the value write is inhibited; otherwise a write/no-write control would also change representation learning. Read-only probes and evaluator clones do **not** update `L`, the clock, teacher counters, or native values.

A pure comparison against the old encoder is obtained by disabling LPS and its updates in BASE; the technical fixture must also verify that `eta=0,L=0` is identical to BASE. Clones/checkpoints deep-copy `L`, not just the native fast/slow arrays.

The fixed symmetric side graph has approximately 41,416 edge strengths (~162 KiB of float32 values), plus fixed index arrays and transient scratch space; report measured resident state, clones, CPU, and temporary allocations rather than equating this theoretical storage count to total runtime memory.

## 3. Two development worlds; exposure and splits frozen

Use the WCFI opaque one-byte association fixture, not arithmetic rules. Each world samples eight distinct cue bytes, two shared suffix bytes, and sixteen reply symbols from one seeded byte permutation; select 80 distinct three-cue prompts, all carrying the same two-byte suffix. Balance the 64 OLD response labels across the 16 symbols and assign each of 16 NEW prompts a distinct response symbol. Exactly one payload byte plus EOS is taught per record. Regenerate data and pass permutations from `SHA256("V175|world|stream")`; do not reuse opened WCFI outcomes, replace difficult worlds, select more separable prompts, or pass labels into the KC network.

| Reserved world | Role | Exposure |
|---|---|---|
| 202611017501 | Technical/calibration only | 64 unlabeled gamma-calibration prompts and ≤32 technical full records |
| 202611017511 | Main W1 | BASE and LPS-16 |
| 202611017512 | Main W2 | BASE and LPS-16 |

Confirm this namespace remains unused before implementation and source-lock data hashes before any scored main history. World IDs are reservations, **not** assertions that these worlds have been run or qualified.

Per arm/world, OLD is `64*16 = 1,024` full records (16 passes, first 8 fixed orders repeated); NEW is `16*8 = 128` further records (8 frozen orders). Do not insert replay, blocked auxiliary records, additional warmup, repeated pretraining, or matched-copy training histories. Four longitudinal lives consume `2 worlds * 2 arms * 1,152 = 4,608` full records / `9,216` target sites. Record effective information exposure and wall/CPU separately. Include independent technical probes and calibration in the full resource ledger.

Panels: baseline initial, after OLD pass 8, after OLD pass 16 (common boundary for analysis, but **not** a shared mutable state), after NEW pass 4, final NEW pass 8, and final rest equal to 128 native steps. Every scored response is generated by the model's **own** bytes and its own EOS within four decisions; report payload and EOS teacher-forced accuracy, but never substitute a fitted external head for the primary outcome.

## 4. Three-cycle execution plan and hard preflight

**C0 — design/static and task feasibility (no main training):**

- Freeze literal native parent/source commit and hash, WCFI generator port, codebook, topology seed, all three world seeds, exact writer signs, schedules, EOS and above design thresholds. Separate-source audit must verify that label/answer bytes never affect `L` or its topology.
- Inspect unlabeled calibration from `202611017501`; freeze `gamma_s` before opening any main scores.
- On each fresh main world's *initial* KC codes, an **evaluator-only** ridge/lookup ceiling should recall ≥72/80 taught pairs and a position-frequency floor ≤8/80. Record exact duplicate-code/conflicting-label counts. These are testbed feasibility gates, not training components or a claim of native learning.
- No use of Alex Wang's large-capacity formula as a performance gate: random-unit-key assumptions do not hold here.

**C1 — technical qualification (≤32 full records):**

- BASE reproduces the pinned WCFI D-high native input/output, state transitions and time to numerical/bitwise parity as appropriate. With LPS effectively disabled, encoder `z` equals native `TopK` for all tested inputs; same KC counts on both sides when enabled.
- Deterministic seeds/ties; reciprocal strengths and side separation; graph degree 16; valid bounds `0≤L≤1`; no target leakage; one update per actual completed record; `L` fixed during a record; empty-input and EOS fixtures.
- Read-only panels do not mutate anything; independently cloned/checkpoint-restored lives give identical code, logits, output and next update; write-disabled target events **still undergo native passive decay**, so never assert that all fast/slow values remain unchanged.
- Check writer coefficient bounds/sign, native fixed-condition read, matched clocks and counters, bit-exact reproducibility, runtime dependency closure, and actual extra topology/clone CPU. A 1.5x measured full-roster CPU forecast including panels/shadows must fit the execution envelope. If not, STOP and revise prospectively; no silent cut of worlds or passes.

**C2 — fixed development test:**

- Run both worlds, exactly two arms each, all four full lives and their fixed panels. Keep checkpoints, source/hash receipts, completed/failing attempts, own-response rows, KC-code snapshots, time, native fast/slow movements, and true resource counts.
- Predeclare eight OLD sentinel prompts per world by a label-blind hash. At **eight fixed NEW write events per world** (first NEW pass, fixed ordinal indices 1,3,5,7,9,11,13,15), apply separately charged clone-only zero-write versus written-target interventions at the **same** source state, sensory code, clock and prospective read condition. Report margin changes for all eight sentinels, plus the self-target margin. These shadow observations never update a live learner. Record actual passive decay in the zero-write arm and prohibit hindsight selection of sentinels or events.
- No retuning, replacement arm/world, extra seed, new data split, or outcome-driven version extension inside C2.

## 5. Frozen metrics and decision rule

For world `w` and arm `a` define:

- `O0(a,w)`: number of OLD items correctly emitted just before NEW (/64).
- `O1(a,w)`: final OLD correct (/64); `N1(a,w)`: final NEW correct (/16).
- `T(a,w)=O1+N1`: final total complete recall (/80).
- `R(a,w)`: fraction of items correct at OLD boundary that are **still** correct after NEW; undefined if `O0=0` (must report, not substitute zero).
- `J(a,w)`: mean OLD×NEW prompt-code Jaccard at the OLD boundary (`64*16` cross pairs). Evaluate both on the arm's *frozen* actual encoder state at the identical prospective prompt time.
- `S(a,w)`: mean per-OLD-prompt Jaccard from OLD-pass-8 to OLD-pass-16 own-code snapshots, with the same record start convention.
- `D(a,w)`: average sentinel target-margin damage from an actual NEW write versus the matched no-write native event, measured on same-state clones. It is an intervention diagnostic, not an automatic causal mediation estimate.

**P1 (primary capability engineering screen):** in **each** world, `T(LPS)-T(BASE)≥8` complete items, `N1(LPS)≥N1(BASE)-2`, and `R(LPS)≥R(BASE)-0.05` when both fractions are defined. These safeguards prevent an apparent total improvement obtained merely by refusing to learn NEW or destroying a formerly retained OLD fraction. Also report `O0`, `O1` and `N1` and their intersections; do not hide worse acquisition behind a conditional survival statistic.

**P2 (mechanistic screen):** in **each** world, `J(BASE)-J(LPS)≥0.05` and `S(LPS)≥0.85`; there must be at least one independently verified code difference under nonzero `L`, and no new conflicting identical codes among the 80 taught cues at the measured final state. Population counts must remain identical to the original per-side top-k eligibility rule. These are *operational falsifiers*, not a mathematical claim that Jaccard equals true synaptic interference.

**P3 (cost/safety screen):** no >1 GiB worker RSS, ≤2 CPU-hours charged, ≤5,000 full learner records including technical, technical read-only/restore/parity checks pass, and the experimental treatment must not use a larger KC population, teacher information to form codes, or a new readout. Report capability per measured operational CPU; a performance gain purchased with extra CPU is not automatically an efficiency gain.

**Decision:** "PROMISING_DEV" only if P1, P2 and P3 all pass on both fresh worlds. Otherwise record the exact FAILED, INACTIVE, INVALID, or UNRESOLVED gate, with full negative evidence. A promising *development* result merely warrants a separately authorized frozen confirmation design; it **does not** adopt the core, establish statistical significance, justify attention, or imply E3 transfer.

Uncertainty: report every paired item result and both world-level differences. Bootstrap over items may describe within-world variation, but item observations from the same trained life are dependent; **two worlds do not support a general performance claim or a valid 64-world-level confidence interval**. Preserve any evaluator-only oracle as such; do not claim a fitted decoder was learned by MiniFly.

## 6. Mechanism-specific interpretations and stop cases

| Observation | Interpretation allowed | Next decision |
|---|---|---|
| LPS reduces overlap, preserves key stability, and improves complete retention/net totals | Good development evidence for an *inhibitory-circuit package* reducing practical interference | Propose independent fixed-inhibition/matched-compute and no-write controls before confirmation |
| Lower overlap, no memory benefit | Jaccard reduction insufficient, or value writer/readout remains bottleneck | Stop this formulation; inspect shadow cross-write effects and readout, do not auto-increase gamma |
| More correct final facts, but no overlap reduction | Benefit could be conditioning or another encoding effect, not the hypothesized decorrelation | Do not call it mechanistically confirmed |
| OLD loss improves only because NEW learning collapses | Protective freezing disguised as retention | P1 fails |
| Code drift `S<0.85`, collisions, or identical-record inconsistency | Plastic address instability | P2 fails; no retuning on opened worlds |
| Improved E1 but worse CPU-normalized capability | Accuracy/cost tradeoff; no efficiency improvement | Report both, do not claim sample/compute superiority |
| Strong E1 with zero unseen-rule gain | Better associative memory, not a rule learner | Keep E3 as an independently prospective objective |

A later **distinct** experiment may test local confidence-based protection or replay against LPS-16, but must not combine them post hoc into V175. A later E3 proposal must learn its number/role/rule representations from ordinary byte feedback and test never-taught relations with a matched no-teaching control. No arithmetic features, oracle gates, study/query markers, output imitation, or new external answer decoder may be inserted into the present kernel.

## 7. Resources, provenance, and status contract

Proposed development envelope follows the latest branch's default planning contract: **2 CPU-hours, 5,000 learner full records, 3 test cycles**, subject to C1 preflight and a separately recorded wall/RSS bound; no standing permission is inferred to start even this capped run. Reserve ≤32 technical full records and 4,608 main full records; all clone-only site events, probes, constructor runs and failed attempts are additionally charged in their own counters. Do not convert source/testbed records to "free" information or exclude evaluator time from compute-efficiency comparisons.

All future code must pin exact source-ref/blob SHA, runner/runtime closure, dependency versions, output codebook, topology seed, gamma-calibration inputs, data generator/salt, fresh world manifest, fixed panel schema, and full operation ledger **before** C2. If original source is missing from a container, stop; do not silently reconstruct a substitute native fly. The design's current links are provenance, not a future implementation's source lock.

**No execution, source-kernel edit, current-adoption-pointer edit, E3 evaluation, or confirmation is authorized by the existence of this design.** Version V175 is allocated once. Minor corrective design revisions must be marked V175-R1/R2 with a change log; a different intervention needs a new available experiment number after checking all live branches.

### Provenance

- Alex Wang, *You Could Have Invented Transformers from Memory*, 2026-10-15: https://alexwang.ai/posts/associative-memory/
- WCFI native writer intervention and controls: https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_writer_interference_dev_v1_20261027/REPORT.md
- Native KC implementation: https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/minifly_cls_dev_20261017/fastmem.py
- Separate V172 native E1 report: https://github.com/ChangweiZhou/digitalbaby/blob/claude/charming-clarke-u9g850/minifly_v172_native_e1_dev_v1_20261027/REPORT.md
- Separate V174 replay report: https://github.com/ChangweiZhou/digitalbaby/blob/claude/charming-clarke-u9g850/minifly_v174_replay_dev_v1_20261028/REPORT.md
- Standing no-attention restriction: https://github.com/ChangweiZhou/digitalbaby/blob/codex/minifly-consolidated-20261023/PROJECT_DOCUMENTS/NO_ATTENTION_KERNEL_REQUIREMENT_20261013.md
