# V45 — Invariance-Guided Distributed Consolidation
## Can learned reusable representation make a fixed-size, slot-free slow synaptic memory persist under continual learning?

**Status:** locked design / implementation specification only  
**Date:** 11 September 2026  
**Target platform:** Apple Silicon Mac M3, macOS, Python 3.11+  
**Scientific results observed before this document:** V44 and the completed Candidate A/B/C route screens only. No V45 outcome has been observed.

---

# 0. Research constitution — no drift

V45 is constrained by the original project objective:

> Build a **bounded-state, non-Transformer, online persistent learner** that learns from sequential experience, retains useful knowledge through intervening learning, generalizes beyond exact presentations, revises obsolete knowledge, and does not require replay or full retraining.

V45 does **not** optimize the current explicit-slot system. Permanent object slots are not the candidate long-term substrate in this round.

The scientific question is fixed:

\[
\boxed{
\text{Can reusable learned invariance reduce interference enough for a bounded distributed slow memory,}
\\
\text{and does corroborated durable writing add independent value once that representation is improved?}
}
\]

The current slot hybrid is carried only as a **non-qualifying engineering reference**. It cannot make a V45 mechanism pass or fail.

V45 may not add, tune, or search over:

- LFU/LRU/age/utility slot replacement;
- larger permanent slot capacity;
- factor-coded output storage from Candidate B;
- new learned compartments or routing pathways;
- recurrent whole-fly dynamics;
- replay;
- online adaptation of the learned invariance during A→B→C→D;
- new slow-write rules beyond the inherited Candidate-A delta update;
- a learning-rate sweep;
- a read-arbitration sweep;
- identity, wrapper, alias, cohort, target-position, or revision labels inside the deployable learner.

If V45 fails, these exclusions are not silently relaxed mid-run.

---

# 1. Evidence entering V45

## 1.1 Candidate A: corroboration is informative, but the tested shared slow store still interferes

At total identity load 256, Candidate A produced the following completed 16-seed means:

| arm | whole bpb ↓ | delayed held-out accuracy ↑ | D acquisition | slow state |
|---|---:|---:|---:|---:|
| `CORR` | 3.1615 | 65.84% | 100% | 3.684 MB |
| `RANDOM-MATCH` | 3.1108 | 63.57% | 100% | 3.684 MB |
| `ORDINARY-r0.6` | 2.9750 | 56.03% | 100% | 3.684 MB |
| `SHUFFLED-CONTEXT` | 3.3029 | 56.49% | 100% | 3.684 MB |
| `SLOT-HYBRID` reference | 2.5439 | 82.67% | 99.95% | 2.183 MB |

Correct corroboration therefore carried content-specific information beyond exact write-count matching and shuffled content, but the single fixed distributed slow matrix did not match the incumbent explicit-memory reference.

**V45 interpretation:** preserve the corroboration question; challenge the representation under which those durable writes interfere.

Candidate A source SHA256:  
`8be7d40d94f246db6fda0465cc8d15bbd3941573b60473e9cdf3583a240675f3`

## 1.2 Candidate C: learned invariance transferred to fresh identities and survived byte permutation

Candidate C learned a rank-20 nuisance/equivalence subspace using only causal, pre-outcome similarity proposals followed by outcome corroboration. Development identities and test identities were disjoint.

In the arbitrary byte-permutation condition, 16-seed means were:

| arm | geometry margin ↑ | E1 bpb ↓ | E2 bpb ↓ | delayed bpb ↓ |
|---|---:|---:|---:|---:|
| `RAW` | 0.1584 | 6.5231 | 5.5764 | 7.1156 |
| **`PAIR-OJA`** | **0.2612** | **6.1257** | **5.1624** | **6.9707** |
| `SHUFFLED-PAIR-OJA` | 0.2204 | 6.3109 | 5.3940 | 7.1055 |
| `RANDOM-SUBSPACE` | 0.1602 | 6.6036 | 5.7263 | 7.1970 |

Thus the learned relation between experiences affected future representations and predictions for previously unseen identities; generic dimension removal did not explain the result.

**V45 interpretation:** test whether this representation changes the interference scaling of a fixed distributed slow store.

Candidate C source SHA256:  
`0fa0f13bbfbe9f295b94a4f917f475971409e62aaa5eaa353561f323b0391220`

## 1.3 V44 remains diagnostic context, not the V45 mechanism

V44 localized the high-load failure of the current slot hybrid to mature-slot residency pressure at load 256. That result does **not** establish slots as the final architecture. V45 therefore asks whether a different durable substrate can avoid creating the same object-residency problem.

V44 design SHA256:  
`7e7926e5ec47f90bbd94717bdb23e5f6daebc077088a568592682ec87f3be3fb`

---

# 2. One primary hypothesis, two separable factors

V45 changes two things that already have independent evidence:

1. **representation:** RAW versus learned PAIR-OJA invariance;
2. **durable-write selection:** exact write-count-matched RANDOM versus causally CORROBORATED writes.

The primary scientific core is therefore a 2×2 factorial:

| representation | matched random durable writes | corroborated durable writes |
|---|---|---|
| RAW | `RAW-RAND` | `RAW-CORR` |
| PAIR-OJA | `PAIR-RAND` | **`PAIR-CORR`** |

Two representation-specificity controls use the same corroborated schedule:

- `SHUFFLED-PAIR-CORR`
- `RANDOM-SUBSPACE-CORR`

Two high-write references are secondary, not primary gates:

- `RAW-ORDINARY`
- `PAIR-ORDINARY`

One incumbent reference is descriptive only:

- `SLOT-HYBRID-REF`

Total scientific arms per persistent trajectory: **8 distributed arms + 1 non-qualifying slot reference**.

No arm changes its mechanism as load changes.

---

# 3. Critical causal isolation: the write schedule is representation-independent

This is a hard design requirement.

If PAIR-OJA were allowed to change which events qualify for corroborated slow writing, then

\[
\text{representation} \to \text{different event schedule} \to \text{different memory}
\]

would be inseparable from

\[
\text{representation} \to \text{different interference geometry}.
\]

Therefore each seed/load first constructs **one master durable-write schedule** from a fixed RAW verifier and the shared BASE trajectory.

### 3.1 Master CORR schedule

Reuse Candidate A's causal discipline:

1. make the prediction before observing byte \(y_t\);
2. RAW verifier state is bounded and contains no evaluator identity labels;
3. after \(y_t\) arrives, a verification event can authorize durable writing;
4. the resulting event index is appended to the master `CORR` schedule.

This schedule is then applied at exactly those event indices to:

- `RAW-CORR`,
- `PAIR-CORR`,
- `SHUFFLED-PAIR-CORR`,
- `RANDOM-SUBSPACE-CORR`.

The same observed outcome \(y_t\) is written in every arm; only the representation under that event differs.

### 3.2 Master RANDOM-MATCH schedule

Construct a nondeployable matched control from BASE-miss events that were available to the causal slow learner, with exactly the same number of writes as CORR.

To prevent trivial differences in timing or stream phase, match the CORR count within each:

- cohort/wave;
- record byte position (0…13);
- coarse temporal block of 256 byte events.

If a stratum is too small, merge only with the adjacent temporal block within the same wave and byte position. Record every merge.

The final schedule is identical for `RAW-RAND` and `PAIR-RAND`.

### 3.3 ORDINARY schedule

`RAW-ORDINARY` and `PAIR-ORDINARY` write on every inherited BASE-miss event at the fixed Candidate-A rate 0.6. They answer whether learned representation makes selective consolidation unnecessary. They do not replace the primary matched-schedule contrast.

---

# 4. Development phase: learn representation, then freeze it

V45 does **not** solve continual coordinate drift. It first asks whether a useful learned representation can improve subsequent bounded continual memory at all.

## 4.1 Fresh development identities

For each scientific seed:

- 192 development identities;
- each identity appears under wrappers 0–3;
- 2 development repeats;
- 256 possible next-byte outcomes are used, not only the 24 association labels;
- no development identity appears in persistent cohorts A–D;
- development presentations count toward sample/compute accounting.

At 14 bytes per record this is:

\[
192\times 4\times 2=1536\text{ records}
\]

or 21,504 raw byte events per seed before the A–D experiment.

## 4.2 Learner-visible information

The development learner receives only the byte stream and its own state. It is not given:

- identity IDs;
- wrapper IDs;
- true pair labels;
- target-position flags;
- latent clean keys;
- evaluator nuisance directions.

## 4.3 Pair verifier

Use Candidate C's pre-outcome top-K proposal discipline:

1. before the next byte is known, produce a small similarity-ranked candidate set from a fixed-capacity trace buffer;
2. after the outcome arrives, corroborate or reject candidates using the observed byte;
3. accepted pair differences update PAIR-OJA;
4. evaluator identity is used only later to measure false-pair rate.

Fixed development parameters inherited from Candidate C:

- rank = 20;
- trace cap = 1024;
- Oja/Sanger update rate = 0.06;
- proposal threshold = 0.58;
- top-K candidate set = 12.

No parameter is tuned on V45 outcomes.

## 4.4 Four representation states

Produce and freeze:

- `U_RAW`: zero/unused rank-20 placeholder;
- `U_PAIR`: genuine corroborated-pair Oja/Sanger subspace;
- `U_SHUFFLED`: same update count/timing/outcome-stratum discipline but wrong pair source;
- `U_RANDOM`: fixed random orthonormal rank-20 subspace.

All arms allocate the same rank-20 matrix bytes, including RAW, so representation-state accounting is exact rather than approximate.

After development:

- the development verifier is destroyed/reset;
- BASE, FAST, persistent verifier and all slow weights are freshly initialized;
- only the frozen representation matrix survives into A–D.

This makes development → future persistence a true transfer test.

---

# 5. Byte representation and nuisance manipulation

V45 uses the **arbitrary byte-permutation encoding as the primary scientific world**, because Candidate C already demonstrated the effect in both ordinary and permuted encodings and the permuted condition removes convenient ASCII/binary semantics.

Let \(r(c)\in\mathbb R^{96}\) be the normalized 12-byte bit code after a fixed public permutation of the 256 byte identities.

For a frozen rank-20 matrix \(U\):

\[
\tilde r_U(c)=r(c)-U^\top U r(c),
\qquad
r_U(c)=\frac{\tilde r_U(c)}{\|\tilde r_U(c)\|}.
\]

RAW uses \(U=0\).

The sparse slow code then uses the **same fixed projection wiring in every arm**:

- slow units = 1024;
- contacts per unit = 12;
- active units = 51 (~5%);
- same contact indices and fixed projection weights within a seed across every arm.

Thus representation arms differ in \(U\), not in random wiring.

### Wrapper discipline

The primary V45 experiment deliberately keeps the six-wrapper family already validated in Candidate C rather than simultaneously inventing a new nuisance generator. Wrappers change header/separator bytes and apply fixed surface transforms to part of the key while leaving sufficient latent identity evidence.

- development: wrappers 0–3;
- persistent acquisition: wrappers 0 and 1;
- held-out generalization probes: wrappers 4 and 5.

A qualitatively new transformation family is **not** added to V45 because doing so would confound the new representation-memory interaction with a second unvalidated robustness question. If V45 qualifies, cross-family transformation robustness becomes a follow-up.

---

# 6. Persistent A→B→C→D workload

After development, all learner state except frozen \(U\) is reset.

The persistent workload preserves the established four-wave discipline:

- cohorts A, B, C, D;
- 8 acquisition presentations per identity;
- no replay of previous cohorts for learning;
- acquisition alternates trained wrappers 0 and 1;
- held-out probes use wrappers 4 and 5, never used for persistent training;
- one-off corruption after B;
- revision of a subset of A after C;
- read-only probes at preregistered checkpoints.

Scientific loads:

- **primary:** total identity load 256 = 64 identities per wave;
- **secondary scaling check:** total identity load 128 = 32 identities per wave.

V45 does not run load 64 as a scientific arm; prior A/B/C work already established the low-load regime. This saves Mac runtime and centers inference on the regime where persistent interference matters.

Fresh V45 scientific seeds:

`45501 … 45516`

These do not overlap V44 or Candidate A/B/C science seeds.

---

# 7. FAST, BASE, and the bounded SLOW substrate

## 7.1 FAST

FAST is the stationary 4096-unit, 5%-active superposed layer used in Candidate A:

- 4096 units;
- 12 contacts/unit;
- 205 active units;
- fixed projection;
- online local output update rate 0.6.

FAST is shared across all distributed arms within a seed/load because its physical trajectory does not depend on those arms.

## 7.2 BASE

The inherited fixed BASE learner is shared across arms and used only for the master verifier/admission discipline and ordinary reference schedule. No distributed SLOW arm may influence BASE.

## 7.3 SLOW

Every distributed arm has its own independent fixed-size matrix:

\[
W_S\in\mathbb R^{1024\times256}.
\]

On a slow write with sparse normalized code \((I,v)\), scores are

\[
s=v^\top W_S[I],
\]

and the inherited Candidate-A delta update is

\[
W_S[I]\leftarrow W_S[I]
+\eta\,(v/\langle v,v\rangle)\otimes(e_y-s),
\qquad \eta=0.6.
\]

No new write equation is introduced.

## 7.4 Readout

All distributed arms use the same fixed read rule:

\[
p_{\rm deployed}=0.5\,p_{FAST}+0.5\,p_{SLOW}.
\]

No learned arbitration is allowed.

`SLOT-HYBRID-REF` uses its inherited deployed semantics and is reported separately. It is not part of any V45 mechanism gate.

---

# 8. Fixed-state requirement

V45 is a bounded-memory experiment. Persistent learner state may not grow with lifetime or load.

Per distributed arm, charged slow-side state includes:

1. `W_S` (1024×256 float64 = 2,097,152 bytes);
2. fixed slow projection indices and weights;
3. the bounded persistent RAW verifier (cap fixed at 512);
4. one rank-20 representation matrix, whether used or placebo;
5. all persistent counters/metadata required for future behavior.

The implementation must report exact serialized semantic bytes and peak transient bytes separately.

**Hard state gate:** distributed slow-side semantic state must be ≤ **3.75 MiB per arm** and must not increase when load changes from 128 to 256 except for occupancy inside already allocated bounded structures.

FAST and BASE state are also reported, but because they are identical and shared across all distributed arms, they are not multiplied by the number of counterfactual arms in architecture-level cost comparisons.

A positive result obtained by expanding units, trace capacity, rank, or output width is invalid.

---

# 9. Endpoints

## 9.1 Primary persistent endpoint

At `after_D`, score all intended identities from A–D using held-out wrappers 4 and 5.

Primary outcome:

\[
\boxed{\texttt{heldout\_after\_D\_bpb}}
\]

Lower is better.

Accuracy is reported alongside bpb but bpb is primary because Candidate C showed that representation improvements can increase true-target probability before they necessarily change argmax.

## 9.2 Acquisition

For every cohort, at acquisition checkpoints and `after_D`, score trained wrappers 0 and 1.

Report:

- accuracy;
- bpb;
- fraction of intended identities represented in the denominator (must always be 100%; no survivor-only metric may substitute).

## 9.3 Matched-delay cohort retention

To avoid comparing A after three later waves with D immediately after learning, report each cohort at:

- immediately after its acquisition;
- after one subsequent cohort where available;
- after the final D endpoint.

These are descriptive secondary endpoints but required for interpretation.

## 9.4 Revision

Measure revised A identities after revision exposures 1, 2, 4 and 8 under:

- trained rendering;
- held-out wrapper.

A model cannot qualify by preserving obsolete A at the cost of revision.

## 9.5 Whole-stream prediction

Report:

- whole bpb;
- target-position bpb/accuracy;
- non-target bpb.

## 9.6 Development transfer geometry

Before persistent learning begins, on fresh A–D identities measure:

\[
M=\mathbb E[\mathrm{sim}(same\ identity,heldout\ wrapper)]
-\mathbb E[\mathrm{sim}(different\ identity,same\ outcome)].
\]

This is a manipulation check, not a substitute for persistence.

---

# 10. Direct interference assay

V45 adds one mechanism-level endpoint: how much a slow durable write changes predictions for previously learned held-out associations.

Running a full probe bank after every write would dominate Mac runtime, so the assay is deterministically subsampled.

For each seed/arm:

- choose at most **128 slow-write events** using a predeclared SHA256 hash of `(seed,event_index)`;
- maintain a fixed evaluator-only probe bank of 48 prior-cohort identity/rendering pairs when available;
- compute probe true-target loss immediately before and immediately after that SLOW write without changing learner state.

For sampled write \(t\):

\[
I_t=\frac1{|P_t|}\sum_{q\in P_t}
[\ell_q(W^+)-\ell_q(W^-)].
\]

Report separately:

- all prior-cohort probes;
- same-outcome/different-identity probes;
- different-outcome/different-identity probes;
- evaluator-only same-identity alternate-rendering probes when such a relation exists.

A negative value means the write helped the old probe bank; positive means interference.

Identity labels used to stratify this audit are evaluator-only and never influence updates.

---

# 11. Predeclared gates

Paired Student 95% intervals use seed as the sampling unit. Repeated events, identities and wrappers within a seed are not independent replicates.

## G0 — implementation and causal firewall

Must all pass:

- deterministic self-test;
- probes are read-only;
- distributed arms have independent `W_S` arrays;
- shared BASE/FAST trajectories are bit-identical regardless of distributed arm order;
- CORR event indices are identical across all CORR arms;
- RANDOM event indices are identical across RAW/PAIR and exactly match CORR write counts under the declared strata;
- no evaluator metadata reaches learner state;
- resume produces the same final hash as uninterrupted execution.

Failure: `INVALID_IMPLEMENTATION`.

## G1 — development validity

Across all 16 seeds:

- genuine accepted pair updates: mean ≥ 300 per seed;
- evaluator-only false-pair fraction: mean ≤ 5%;
- PAIR-OJA subspace remains finite and rank 20 after orthonormalization.

Failure: `INVALID_DEVELOPMENT_SIGNAL`.

## G2 — representation manipulation is real on fresh identities

On the pre-persistent geometry panel:

- `PAIR` − `RAW` margin ≥ **+0.05**, paired lower 95% bound > 0;
- `PAIR` − `SHUFFLED` margin ≥ **+0.02**, paired lower bound > 0;
- `PAIR` − `RANDOM-SUBSPACE` margin ≥ **+0.05**, paired lower bound > 0.

Failure: `REPRESENTATION_EFFECT_NOT_REPLICATED`.

## G3 — continued acquisition remains valid

At load 256, both `PAIR-RAND` and `PAIR-CORR` must have after-D trained-wrapper D-cohort accuracy ≥ 0.90 in at least 14/16 seeds, and the pooled seed mean must be ≥ 0.90.

Failure: `ACQUISITION_TRADEOFF`.

## G4 — representation reduces persistent loss under a matched write schedule

Primary contrast:

\[
\Delta_R=
\texttt{PAIR-RAND}-\texttt{RAW-RAND}
\]

on `heldout_after_D_bpb` at load 256.

Required:

- mean ≤ **−0.10 bits**;
- paired upper 95% bound < 0;
- D acquisition difference lower bound ≥ −0.02;
- whole-bpb difference mean ≤ +0.10 bits.

Pass means learned representation changes persistent memory behavior, not merely geometry.

## G5 — corroboration adds value once representation is improved

Primary contrast:

\[
\Delta_C=
\texttt{PAIR-CORR}-\texttt{PAIR-RAND}.
\]

Required on load-256 `heldout_after_D_bpb`:

- mean ≤ **−0.10 bits**;
- paired upper bound < 0;
- D acquisition lower-bound difference ≥ −0.02;
- whole-bpb mean difference ≤ +0.10 bits.

If G4 passes and G5 fails, the simpler learned-representation + matched/ordinary distributed learner takes priority; corroboration is not promoted as a core component.

## G6 — representation specificity

`PAIR-CORR` must beat **both**:

- `SHUFFLED-PAIR-CORR`;
- `RANDOM-SUBSPACE-CORR`

on load-256 `heldout_after_D_bpb` by at least **0.05 bits**, with paired upper bounds below zero.

Failure after G4/G5 evidence: `GENERIC_DIMENSION_REDUCTION_EFFECT` or `PAIR_SPECIFICITY_NOT_ESTABLISHED`.

## G7 — direct interference support

For the deterministic sampled-write assay, compare `PAIR-RAND` against `RAW-RAND`.

Supporting criterion:

- mean prior-cohort interference \(I_t\) is lower under PAIR in paired seed summaries;
- target relative reduction ≥ 20% when RAW mean interference is positive;
- paired difference upper bound < 0.

G7 is **mechanism-supporting, not a hard qualification gate**. Persistence can be real even if this finite probe bank does not localize the mechanism.

## G8 — bounded state

All distributed arms must satisfy the 3.75 MiB slow-side state cap and must remain fixed-size across load 128→256.

Failure: `UNBOUNDED_OR_BUDGET_VIOLATION`.

## G9 — slot reference is descriptive only

`SLOT-HYBRID-REF` reports the performance/cost gap to the incumbent. It cannot enter G2–G8 and cannot veto a V45 mechanism result.

---

# 12. Factorial interpretation

The factorial interaction on the primary endpoint is

\[
I=(PAIR\!\! -\! CORR-PAIR\!\! -\! RAND)
-(RAW\!\! -\! CORR-RAW\!\! -\! RAND),
\]

implemented algebraically as

\[
I=(\mathrm{PAIR\mbox{-}CORR}-\mathrm{PAIR\mbox{-}RAND})
-(\mathrm{RAW\mbox{-}CORR}-\mathrm{RAW\mbox{-}RAND}).
\]

Because lower bpb is better, a negative interaction means corroboration is more beneficial under learned invariance.

The interaction is descriptive. V45 does **not** require synergy: additive independent gains are scientifically sufficient.

---

# 13. Verdict vocabulary

After G0/G1/G2/G8:

- **`QUALIFIED_SLOT_FREE_CORE`** — G3, G4, G5 and G6 all pass; learned invariance and corroboration independently support bounded distributed persistent learning.
- **`REPRESENTATION_CORE_SUPPORTED`** — G3, G4 and G6 pass; G5 fails. Learned representation is promoted; corroboration is not required.
- **`CORROBORATION_ONLY`** — G5 passes but G4 fails. Candidate C does not solve distributed interference in this continual setting.
- **`PAIR_SPECIFICITY_FAILED`** — apparent PAIR advantage is matched by shuffled/random representation controls.
- **`ACQUISITION_TRADEOFF`** — preservation is bought by failing to learn D.
- **`NO_PERSISTENT_GAIN`** — neither representation nor corroboration clears its persistent endpoint.
- **`UNBOUNDED_OR_BUDGET_VIOLATION`** — fixed-state requirement fails.
- **`INVALID_DEVELOPMENT_SIGNAL`**, **`REPRESENTATION_EFFECT_NOT_REPLICATED`**, **`INVALID_IMPLEMENTATION`** — corresponding hard screen failures.

No post-hoc relabeling creates a stronger verdict.

---

# 14. Secondary scaling interpretation

Load 128 is not used to rescue a failed load-256 primary claim.

It answers only:

- does the direction of the primary contrast already appear before severe interference?;
- does a positive load-256 result come from a pathological high-load corner?;
- is state fixed while identity load doubles?

If load 256 fails, a favorable load 128 result is reported as `LOWER_LOAD_ONLY` and does not qualify the architecture.

---

# 15. Mac M3 implementation plan

V45 must be implementable as one self-contained Python runner plus macOS launchers.

Recommended filename:

`elm_v45_invariance_guided_consolidation.py`

## 15.1 Dependencies

Only:

- Python 3.11+
- NumPy ≥ 1.26
- SciPy ≥ 1.11

No PyTorch, JAX, GPU runtime, database, or network access is required.

## 15.2 Thread discipline

Before importing NumPy, set:

```bash
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
OMP_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

Scientific parallelism is across independent seed/load jobs, not BLAS threads.

## 15.3 Multiprocessing

Use macOS-safe `spawn`, never fork-dependent mutable state.

Default:

- `V45_WORKERS=4` on M3/M3 Pro systems with adequate RAM;
- permit override to 2 or 3 workers;
- no nested multiprocessing.

Each seed/load worker runs **all arms together**. This is essential for:

- exact pairing;
- one shared byte stream;
- one shared BASE/FAST physical trajectory;
- one master CORR/RANDOM schedule;
- avoiding repeated front-end computation.

The representation-development job runs once per seed and is reused by both loads.

## 15.4 Expected job graph

Development:

- 16 seed jobs.

Persistent science:

- 16 seeds × load 256 = 16 jobs;
- 16 seeds × load 128 = 16 jobs.

Total top-level jobs: **48**, not one job per arm.

Within a worker, eight distributed `W_S` matrices require about 16 MiB total for learned slow weights. FAST, BASE, fixed projections, verifier state, probes and Python overhead remain comfortably below typical M3 memory limits; the implementation must nevertheless measure peak RSS during smoke/full engineering tests rather than assume it.

## 15.5 Caffeinate launchers

Ship:

- `RUN_V45_M3.command`
- `RUN_V45_PRIMARY_M3.command`
- `STATUS_V45.command`

Recommended full launcher behavior:

1. create/use `.venv_v45` only if NumPy/SciPy are unavailable;
2. export one-thread BLAS variables;
3. run `--self-test`;
4. run `--engineering-check` on non-scientific seeds;
5. `caffeinate -dimsu` development stage;
6. `caffeinate -dimsu` load-256 primary stage;
7. only after primary files are complete, run load-128 scale stage;
8. aggregate and create `v45_results_upload.zip` excluding temporary checkpoints.

`RUN_V45_PRIMARY_M3.command` stops after step 6 and produces a primary upload bundle. Running the full launcher later reuses the completed primary jobs.

Closing a MacBook lid may still suspend it unless macOS is in a supported clamshell configuration; `caffeinate` only prevents ordinary idle sleep.

---

# 16. Resume and interruption safety

V45 must be restartable by rerunning the same command.

## Identity

Every cached artifact carries an identity hash over:

- source SHA256;
- this design SHA256;
- version string;
- seed;
- load/stage;
- all scientific parameters;
- representation-development artifact hash;
- major NumPy/SciPy versions relevant to deterministic reuse.

Mismatched cache files are rejected, never silently reused.

## Atomic outputs

- write to `.tmp`;
- `fsync` where practical;
- atomic rename;
- keep previous checkpoint generation until the new one validates.

Persistent jobs checkpoint at least:

- after each cohort;
- every 128 records inside a cohort.

Checkpoint state includes every future-relevant RNG and learner state. Resume equivalence is a self-test requirement.

---

# 17. Computational optimizations allowed without changing science

These are implementation optimizations, not mechanisms:

1. **Share immutable sparse projection arrays** across arms within a seed.
2. **Batch score arms with the same representation** using `einsum`/matrix operations.
3. Cache immutable encoding of repeated 12-byte contexts after representation freeze.
4. BASE and FAST are physically computed once per byte and shared.
5. Master CORR/RANDOM schedules are generated once per seed/load.
6. Direct-interference auditing is capped at 128 deterministic sampled writes per arm/seed.
7. Development U matrices are cached once per seed and reused at load 128/256.

No optimization may cause two arms to share learned `W_S` state.

---

# 18. Self-tests and engineering checks before scientific execution

Required self-tests:

1. context is exactly 12 bytes;
2. byte-permutation encoding is deterministic and bijective;
3. `U_PAIR`, `U_SHUFFLED`, `U_RANDOM` are rank 20 and orthonormal after freeze;
4. RAW projection is exactly identity/no nuisance removal;
5. same sparse projection contacts/weights are used across representation arms;
6. CORR schedules are bit-identical across all CORR arms;
7. RANDOM schedules match CORR counts in every declared stratum;
8. probe calls do not change state hashes;
9. one slow write changes only that arm's `W_S`;
10. development verifier receives no evaluator identity metadata;
11. persistent verifier resets after development;
12. all persistent learner state except frozen U resets after development;
13. no permanent slot objects exist in distributed candidate arms;
14. fixed-state cap is enforced;
15. interrupted/resumed smoke run hashes equal uninterrupted run hashes;
16. arm ordering does not change outcomes.

Engineering-only seeds must be disjoint from science seeds.

Engineering check should run:

- 2 seeds;
- reduced development identities;
- load 64 or a reduced stream;
- every scientific arm;
- one forced interruption/resume cycle;
- peak RSS and wall-clock logging.

Engineering outcomes never enter scientific summaries.

---

# 19. Required outputs

At minimum:

- `MANIFEST.json`
- `PROTOCOL.md`
- `GATES.json`
- `SUMMARY.md`
- `REPORT.json`
- `DEVELOPMENT_AUDIT.csv`
- `REPRESENTATION_GEOMETRY.csv`
- `TERMINAL.csv`
- `CHECKPOINT_METRICS.csv.gz`
- `PERSISTENCE_CONTRASTS.csv`
- `FACTORIAL_CONTRASTS.csv`
- `STATE_COSTS.csv`
- `SAMPLE_BUDGET.csv`
- `WRITE_SCHEDULE_AUDIT.csv`
- `INTERFERENCE_SUMMARY.csv`
- `INTERFERENCE_EVENTS.csv.gz`
- `REVISION.csv`
- `RESUME_AUDIT.json`
- `FIREWALL_AUDIT.json`

Final launcher creates:

`v45_results_upload.zip`

Temporary checkpoints, large redundant arm states and virtual-environment files are excluded.

---

# 20. Interpretation rules

1. **Geometry is not success.** G2 only validates the manipulation; G4–G6 decide whether it matters for persistent learning.
2. **Retention without acquisition is failure.** G3 prevents reduced writing from masquerading as persistence.
3. **PAIR must beat shuffled/random subspaces.** Otherwise the result is generic dimensionality reduction.
4. **Corroboration must beat matched random writes.** Otherwise the result is write-budget control, not evidence selection.
5. **The slot reference cannot veto a mechanism.** It is an incumbent cost/performance benchmark only.
6. **No unbounded state is allowed.** The experiment is explicitly about bounded persistent learning.
7. **Development cost is charged.** PAIR-OJA does not receive free pretraining.
8. **U is frozen during A–D.** A positive V45 result does not yet solve continual representation drift.
9. **No post-hoc parameter tuning.** If the locked Candidate-A/C parameters fail, V45 reports failure.
10. **No architecture merger with Candidate B.** Factor coding remains a separate compression question until the core architecture qualifies.

---

# 21. What each outcome means

### G4 + G5 + G6 pass

Evidence supports a bounded, slot-free core in which past experience changes future representation and causal corroboration selects durable shared-synaptic learning.

Next question is continual representation update without coordinate drift—not slots.

### G4 passes, G5 fails

Learned reusable invariance is the important mechanism; selective corroborated consolidation is unnecessary under the improved representation. Prefer the simpler distributed learner.

### G5 passes, G4 fails

Corroboration remains useful, but Candidate C's representation result does not solve the persistent-interference problem. Do not promote PAIR-OJA into the core architecture.

### G4/G5 apparent gains but G6 fails

The benefit is generic projection/dimensionality reduction, not learned equivalence. Retire the strong representation-learning interpretation.

### Acquisition fails

The architecture preserves history by refusing or failing to learn new knowledge. It does not qualify.

### Everything fails despite valid development

A single shared distributed matrix with this fixed local writer is insufficient even after learned invariance and event selection. The next architecture branch should test **learned overlapping slow pathways**, motivated by the fly's shared representation plus differentiated output/teaching pathways—not return to permanent slots.

---

# 22. One-sentence decision

V45 asks whether **past experience can teach a bounded distributed learner how to represent future experience so that durable synaptic learning remains useful without permanent object slots**.

That question is the research target. Everything else in this document exists to prevent it from drifting into a different one.
