# V52 — Byte Interface Decomposition, Mac M3 execution specification

**Experiment namespace:** `V52-BYTE-INTERFACE`  
**Version:** refined V52.0  
**Date:** 12 September 2026  
**Reference learner:** frozen `V50-REF` carried through V51  
**Platform:** Apple Silicon Mac M3; one self-contained Python file; interruption-safe cached jobs

## 1. Scientific decision

V52 asks:

> Why does the bounded fly-like V50/V51 associative core still work on direct CUE8 identities but fail on the serialized BYTE256 interface?

The round does **not** search another memory architecture. It decomposes the interface problem in the fixed causal order

\[
\boxed{
\text{identity information}
\to
\text{stationary representation}
\to
\text{synaptic usability}
\to
\text{write interference}
\to
\text{deployable all-byte behaviour}
}
\]

A later-level result is not interpreted when an earlier implementation/validity condition fails.

## 2. V51 facts carried forward, and one V51 correction

V51 established that the same downstream circuit can still learn direct cues at loads where the byte task is very weak. V51 also showed that `ELIG-WRITE` is harmful on BYTE256, so V52 uses only the original V50/V51 reference write rule.

The byte code is not information-free: post-hoc wrapper-to-wrapper identity retrieval is high despite a much poorer same/different support-overlap ratio than CUE8. Therefore V52 must distinguish **decodability** from **usability by the fixed KC→MBON rule**.

### 2.1 V51 probe-cohort correction

The V51 BYTE generator defines:

- revised A identities = first quarter;
- one-off-noise A identities = second quarter;
- truly untouched A identities = second half.

However, several V51 probes later labelled `A_untouched...` use the **first half**, which is exactly revised + noisy. V52 does not inherit this mismatch.

V52 scientific retention uses

```text
A_true_untouched = cohort A \ (revised ∪ noisy)
```

and separately emits

```text
A_legacy_firsthalf_...
```

for regression against V51. This correction changes measurement only; it does not change training.

## 3. Frozen biological substrate

Every deployable V52 candidate preserves:

- 699 PNs;
- 4,552 KCs;
- 104 MBONs;
- 22,417 PN→KC directed pairs;
- 15,523 KC→MBON directed pairs;
- raw BANC count-normalized PN→KC weights;
- leaky KC equation with `rho = 0.75`;
- exactly 228 active KCs (5%);
- binary anatomical KC→MBON support;
- one float64 learned scalar per KC→MBON edge;
- `eta = 0.5`;
- 104-dimensional consequence codebook;
- hard/unattenuated readout;
- no slots, replay, eligibility accumulation, consolidation, metaplasticity, learned routing, APL mechanism, MBON recurrence, new neuron family, or new memory state.

Only the fixed **byte sensory interface before the existing V51 byte→PN adaptor** may change in deployable arms.

## 4. Official byte workload

Every deployable arm retains V51 serialization:

```text
PREFIX(4) + HEADER(1) + KEY(6) + SEPARATOR(1) + TARGET(1) + NEWLINE(1)
```

Record length = 14 bytes.

Wrapper surface variation changes:

- header;
- separator;
- XOR mask on the first two key bytes.

Key bytes 2–5 remain invariant across wrappers.

Training uses wrappers 0/1. Held-out probes use wrapper 2. XOR remains enabled for every deployable candidate.

No deployable learner receives record boundaries, wrapper IDs, key IDs, byte-role labels, cohort IDs, or target-position labels.

## 5. Mac M3 calibration correction

The earlier draft used a 2000-byte calibration prefix. That is unsafe for a phase-conditioned encoder because

\[
2000 \bmod 14 = 12.
\]

If calibration and science are treated causally as consecutive streams, the phase origin would shift by 12 bytes. Resetting it to zero would silently reveal alignment.

V52 therefore uses **144 complete dummy records**:

\[
144\times14=\boxed{2016\text{ calibration bytes}}.
\]

The first complete record is a burn-in. Phase statistics are fit on the following 143 complete cycles, giving exactly 143 full-context observations per discovered phase when the period is 14.

Calibration:

- uses separate dummy identities;
- has the same wrapper 0/1 alternation and byte alphabet;
- causes zero KC→MBON writes;
- is explicitly charged in sample accounting;
- is an integer number of records, so science can start at phase zero without a hidden phase reset.

## 6. Fundamental-period discovery

On calibration bytes, compute byte-value autocorrelation for lags 2–40. Let \(r_{\max}\) be the largest finite value and choose

\[
\widehat P
=
\min\{\ell:r(\ell)\ge0.8r_{\max}\}.
\]

The smallest qualifying lag is used because wrapper alternation can make the 28-byte harmonic slightly larger than the true 14-byte fundamental.

There is no hard-coded fallback. For a deployable phase arm, period discovery is valid only if all screen seeds return 14.

## 7. Frozen phase statistics

For each prediction event on calibration, form the exact V51 96-dimensional `context_bits` vector. By phase \(\phi=t\bmod\widehat P\), fit

\[
\mu_{\phi,k},\qquad \sigma_{\phi,k}.
\]

After calibration these statistics are immutable. Variance floor = \(10^{-6}\).

## 8. Deployable encoder roster — complete minimal transform ladder

The earlier draft omitted the simplest important case: removing phase-constant coordinates without centering or scaling. V52 includes it.

### E0 — `V51-LAG`

Exact V51 interface. No phase transform.

### E1 — `PHASE-MASK-FROZEN`

At each phase, coordinates with calibration variance at or below the floor are zeroed; all other raw signed bits are unchanged.

This asks whether merely removing phase-constant boilerplate is enough.

### E2 — `PHASE-CENTER-FROZEN`

For live coordinates,

\[
x'_k=x_k-\mu_{\phi,k}.
\]

### E3 — `PHASE-Z-FROZEN`

For live coordinates,

\[
x'_k=(x_k-\mu_{\phi,k})/\sigma_{\phi,k}.
\]

All four feed the **same V51 96→699 random adaptor**, raw PN→KC graph, leaky KCs, and 228-KC top-k code.

No historical V36/V39 random expansion and no KC sparsity sweep are included; those would change the current organism rather than isolate interface preprocessing.

## 9. Nondeployable representation oracles — complete transform × XOR check

To avoid testing XOR only under one arbitrary normalization, the entire transform ladder is repeated with XOR disabled:

- `XOR-OFF`;
- `PHASE-MASK-XOR-OFF`;
- `PHASE-CENTER-XOR-OFF`;
- `PHASE-Z-XOR-OFF`.

These can never become production candidates.

### O5 — `INVARIANT-TAIL4`

A fixture-aware oracle retaining only the four key bytes invariant across wrappers. The mask is phase-aware on the rolling 12-byte history; it does not assume the target event is the only context ever encoded.

A strong result means the downstream store can use a clean invariant identity subspace. It does **not** solve how a learner discovers that subspace.

## 10. Output-alphabet diagnostic — an omitted V51 confound now closed

CUE8 and BYTE256 differ not only in sensory interface but also in output alphabet. BYTE target values occupy only the 24 letters `a`–`x`, but V51 predicts through 256 codewords.

To prevent a failed invariant-tail/256 run from being misdiagnosed as a memory failure, V52 adds two nondeployable target-only diagnostics:

- `V51-LAG | TARGET24`;
- `INVARIANT-TAIL4 | TARGET24`.

They use the same 104 MBONs and local update but a 24-codeword consequence alphabet restricted to target letters.

Thus V52 contains the diagnostic 2×2:

| representation | 256-output target-only | 24-output target-only |
|---|---:|---:|
| V51-LAG | yes | yes |
| invariant-tail4 | yes | yes |

If invariant-tail/256 is weak but invariant-tail/24 is strong, the next problem is the output interface, **not** the KC→MBON memory.

## 11. Seed firewall

Engineering:

```text
52671–52674
```

Screen:

```text
52501–52516
```

Science:

```text
52601–52616
```

Science seeds are not inspected until `CANDIDATE_FREEZE.json` is written.

## 12. Block A — geometry, decodability, and stationarity

No synaptic learning. Loads 128 and 256.

All nine representation arms are measured.

### 12.1 Same-identity geometry

Do **all three** wrapper pairs, not only wrapper 0→2:

\[
(0,1),\;(0,2),\;(1,2).
\]

Report Jaccard and cosine for each pair, plus mean and minimum same-identity overlap.

### 12.2 Different-identity floor

Report different-identity overlap separately within wrappers 0, 1, 2 and across wrapper pairs 0–1, 0–2, 1–2.

This prevents one wrapper from hiding a cross-talk problem.

### 12.3 Retrieval

Wrapper 2 is queried against:

- wrapper-0 gallery;
- wrapper-1 gallery;
- max similarity over training wrappers 0/1;
- normalized centroid of wrapper 0/1.

Report cosine and Jaccard retrieval where defined.

### 12.4 Two stationarity tests

1. **Target-context stationarity:** re-encode an identical target context with frozen statistics.
2. **Actual-time recurrence stationarity:** find exact rolling 12-byte contexts that recur in the actual training stream and compare their codes at the phases where they really recur.

The second test catches a period/harmonic error that matched-phase tests can hide.

## 13. Critical change: geometry is not a behavioral veto

The earlier draft allowed a cross-talk/Jaccard threshold to decide whether a phase arm reached target-only learning. That is too strong.

**Every stationary deployable arm receives Block B**, even if its Jaccard ratio does not meet a preferred target.

Representation validity requires only:

- period 14 on all screen seeds;
- target-context Jaccard self-overlap ≥ 0.999;
- actual-time recurrence self-overlap ≥ 0.95;
- combined training-wrapper cosine identity retrieval no more than 5 percentage points below V51-LAG.

`J_same`, `J_diff`, ratio, and cosine margin remain causal diagnostics, not sufficient statistics for learnability.

## 14. Block B — target-only synaptic-usability ceiling

Use the exact V50/V51 `Memory` rule and outcome codebook.

All stream contexts still exist, but KC→MBON writes are allowed only at target events. At a target event the preceding 12 bytes are exactly the serialized surface context, so the screen can process target events directly without spending Mac M3 time computing irrelevant non-target predictions.

All nine representation arms are tested at load 128, plus the two TARGET24 diagnostics.

Required endpoints:

- prequential target loss and accuracy by exposure;
- A wrapper 0, wrapper 1, and wrapper 2 after A;
- true untouched A first-return after B;
- B after B;
- one-off-noise before/after;
- true untouched A after C;
- revision 1/2/4/8;
- true untouched A after D;
- legacy V51 first-half A after D;
- B/C/D after D;
- revised-A after D;
- synapse reset and within-MBON shuffle;
- update counts/norms.

Define

\[
U_{TO}=\min(A_{true,afterD,heldout},D_{afterD,heldout}).
\]

## 15. Deployable candidate gate

A deployable phase candidate must be representation-valid and satisfy, on 16 screen seeds:

\[
A_{true,afterD,heldout}\ge0.40,
\]

\[
D_{afterD,heldout}\ge0.60,
\]

\[
U_{TO}^{cand}-U_{TO}^{lag}\ge0.10
\]

with paired 95% lower bound > 0, and target prequential loss may worsen by at most 0.05 bit.

If multiple phase candidates pass, choose highest mean \(U_{TO}\). If within 0.02, prefer lower interface complexity:

```text
MASK < CENTER < Z
```

The result and hashes are frozen before science.

If none pass:

```text
NO_DEPLOYABLE_INTERFACE_CANDIDATE
```

This means only that this frozen phase-transform family is insufficient.

## 16. Block C — write-interference decomposition

This is where the earlier design needed a second decisive control.

Run on V51-LAG always, and on the selected deployable encoder if one exists.

Policies:

### P0 — `ALL`

Every byte writes, exactly as V51.

### P1 — `TARGET-ONLY`

Exactly one target write per record.

### P2 — `NO-FRAMING`

Key + target bytes write.

### P3 — `NO-KEY`

Framing + target bytes write.

### P4 — `RANDOM-MATCH`

Exactly one write per record, at a deterministic uniformly sampled byte position. This matches write count and record timing but destroys semantic allocation.

### P5 — `TARGET-UNPAIRED`

Exactly the **same target contexts, target phase, and write count** as TARGET-ONLY, but the target teacher is permuted within cohort/exposure while preserving marginals.

This closes the hole in RANDOM-MATCH: RANDOM-MATCH changes both semantic relevance and context phase; TARGET-UNPAIRED destroys only context→outcome pairing.

### P6 — `NOLEARN`

Uniform learned-state reference.

Hard accounting assertion:

```text
updates(TARGET-ONLY)
 = updates(RANDOM-MATCH)
 = updates(TARGET-UNPAIRED)
 = number_of_records
```

## 17. Block C interpretation

Primary endpoint = true untouched A held-out after D.

- `TARGET-ONLY >> RANDOM-MATCH`: which events write matters beyond write count.
- `TARGET-ONLY >> TARGET-UNPAIRED`: the gain depends on correct target pairing at the exact same target contexts.
- `NO-FRAMING > ALL`: framing writes interfere.
- `NO-KEY > ALL`: key writes interfere.
- candidate target-only strong but candidate ALL weak: endogenous write allocation is the remaining bottleneck.

No evaluator-assisted policy is deployable.

## 18. Block D — fresh-seed all-byte science

Runs only after candidate freeze.

At load 128, independently run the same stream for:

- `V51-LAG / ALL`;
- `V51-LAG / UNPAIRED`;
- selected candidate / `ALL`;
- selected candidate / `UNPAIRED`;
- `NOLEARN` references.

`UNPAIRED` is V51-style: non-target bytes still learn normally; target teachers are permuted within cohort/exposure.

All real deployable learners update on every byte.

Science preserves:

- framing/key/target and whole-stream metrics;
- target prequential curves;
- sustained 80% exposure crossing with right censoring;
- first return;
- true untouched and legacy A probes;
- B/C/D retention;
- noise;
- revision;
- reset/shuffle localization;
- Markov2 baseline;
- sample/state/runtime accounting.

Define

\[
U_{ALL}=\min(A_{true,afterD,heldout},D_{afterD,heldout}).
\]

## 19. Science verdicts

Interface gain:

\[
U_{ALL}^{cand}-U_{ALL}^{lag}\ge0.10
\]

with paired lower 95% bound > 0, and candidate whole-stream bpb no worse by more than 0.10.

Separately report a stronger functional operating point:

\[
A_{true,afterD,heldout}\ge0.25,
\qquad
D_{afterD,heldout}\ge0.40.
\]

Candidate must also beat its matched unpaired control; otherwise:

```text
PAIRING_NOT_CAUSAL
```

## 20. Conditional load-256 extension — both triggers implemented

Run full science at load 256 only if load-128 science satisfies either:

1. interface \(U_{ALL}\) improvement ≥ 0.10 with lower CI > 0; **or**
2. candidate minus baseline D-heldout bpb ≤ −0.50 with upper CI < 0;

and whole-stream bpb is no worse by more than 0.10.

This prevents a costly high-load run after a weak effect while retaining the originally intended strong probability/loss improvement route.

## 21. Decision logic including output-alphabet branch

### Invariant-tail/256 strong

Downstream memory can exploit clean invariant identity. Do not reopen memory.

### Invariant-tail/256 weak, invariant-tail/24 strong

The 256-way output interface is the relevant remaining bottleneck. Do not reopen memory.

### Invariant-tail/256 and invariant-tail/24 weak

Only here does V52 earn the right to reopen KC→MBON memory/readout architecture.

### Target-only strong, ALL weak

Plasticity allocation is the next problem.

### XOR-off strong, XOR-on weak

Wrapper invariance is a major burden; changing the fixture is still not a solution.

### Phase geometry improves but target-only does not

Jaccard/cosine geometry is not the usable variable for this local readout. Do not tune overlap indefinitely.

## 22. Why the fixture is not automatically changed

A strong invariant-tail oracle with weak deployable phase preprocessing identifies a legitimate unresolved operation:

> discover a stable identity subspace from repeated serialized surface forms without evaluator labels.

Changing the record format at that point would remove the problem instead of solving it.

## 23. Why CUE512 remains separate

CUE512 is a capacity-cliff experiment. V52 is an interface-causality experiment. V52 keeps only V50/V51 cue regression in self-test and does not spend science seeds on CUE512.

## 24. Statistics

Seeds are inferential units.

Primary contrasts use paired seed differences with:

- Student 95% CI;
- bootstrap 95% CI;
- raw seed values.

Byte events, identities, exposures, and probes are not independent replicates.

Exposure-to-criterion remains right-censored after exposure 8; no fabricated ninth exposure.

## 25. Mac M3 execution design

The script is self-contained and includes the embedded BANC graph inherited from V51.

BLAS/OpenMP threads are forced to one per process.

Recommended worker policy:

- geometry/target-only screen: up to 4 workers;
- full-stream interference/science: internally capped at 2 workers to avoid Mac M3 memory pressure.

Every independent job is atomically cached as gzip JSON. Cache identity includes source hash, design hash, graph hash, block, seed, load, encoder, and candidate-freeze hash when applicable.

Interrupted runs restart from completed jobs rather than from the beginning.

## 26. One-command and staged execution

Self-test:

```bash
python3 elm_v52_byte_interface.py --self-test
```

Recommended complete run:

```bash
caffeinate -dimsu python3 elm_v52_byte_interface.py \
  --stage all \
  --workers 4 \
  --out-dir v52_results
```

Safer staged run:

```bash
caffeinate -dimsu python3 elm_v52_byte_interface.py --stage screen  --workers 4 --out-dir v52_results
caffeinate -dimsu python3 elm_v52_byte_interface.py --stage science --workers 4 --out-dir v52_results
caffeinate -dimsu python3 elm_v52_byte_interface.py --stage load256 --workers 4 --out-dir v52_results
python3 elm_v52_byte_interface.py --stage aggregate --out-dir v52_results
```

`load256` refuses to run if the load-128 extension gate has not cleared.

## 27. Required implementation tests

Before screen evidence is interpreted:

1. V50/V51 cue fixture reproduces bit-for-bit.
2. `V51-LAG` V52 code reproduces V51 byte KC vectors bit-for-bit.
3. calibration length is phase-neutral (2016 mod 14 = 0).
4. period discovery returns 14 on engineering seeds without fallback.
5. each phase receives exactly 143 calibration contexts.
6. frozen phase statistics do not mutate after calibration.
7. every representation produces exactly 228 active KCs.
8. full transform × XOR-off roster is present.
9. TARGET24 mapping is exactly `a..x -> 0..23`.
10. true untouched A is disjoint from revised and noise cohorts.
11. target-only, random-match, and target-unpaired write counts match exactly.
12. class metric counts and loss sums reconcile to whole-stream totals.
13. probes are read-only.
14. reset/shuffle probes use identical sensory realizations.
15. multiprocessing uses spawn and deterministic seeded RNGs.
16. science cache identity contains the frozen-candidate hash.

Any failure is

```text
IMPLEMENTATION_INVALID
```

## 28. Required output families

Screen:

```text
CALIBRATION_AUDIT.csv
GEOMETRY_ALL.csv
GEOMETRY_JACCARD.csv
GEOMETRY_COSINE.csv
IDENTITY_RETRIEVAL.csv
STATIONARITY.csv
TARGET_ONLY_PREQUENTIAL.csv
TARGET_ONLY_RETENTION.csv
TARGET_ONLY_CEILING.csv
TARGET_ONLY_LOCALIZATION.csv
OUTPUT_ALPHABET_DIAGNOSTIC.csv
SCREEN_CANDIDATE_GATES.csv
CANDIDATE_FREEZE.json
PLASTICITY_DECOMPOSITION.csv
PLASTICITY_CONTRASTS.csv
PLASTICITY_BYTE_CLASS_METRICS.csv
PLASTICITY_UPDATE_COUNTS.csv
```

Science:

```text
PREQUENTIAL_ASSOCIATION.csv.gz
ACQUISITION_CURVES.csv
EXPOSURES_TO_CRITERION.csv
FIRST_RETURN.csv
COHORT_RETENTION.csv
BYTE_ASSOCIATION_PROBES.csv
BYTE_CLASS_METRICS.csv
NOISE_RESISTANCE.csv
REVISION_CURVES.csv
REVISED_AFTER_D.csv
STATE_LOCALIZATION.csv
MARKOV_BASELINE.csv
UPDATE_NORMS.csv
STATE_COSTS.csv
SAMPLE_BUDGET.csv
RUNTIME.csv
BYTE_ACCOUNTING_ASSERTIONS.json
SCIENCE_TERMINAL.csv
SCIENCE_DECISION.json
```

Load-256 outputs use the `L256_` prefix.

## 29. Final interpretation vocabulary

Use only labels consistent with the measured branch, including:

```text
IMPLEMENTATION_INVALID
NO_DEPLOYABLE_INTERFACE_CANDIDATE
GEOMETRY_GAIN_NOT_SYNAPTICALLY_USABLE
INVARIANCE_DISCOVERY_BOTTLENECK
OUTPUT_INTERFACE_BOTTLENECK
REPRESENTATION_AND_WRITE_INTERFERENCE
PLASTICITY_ALLOCATION_BOTTLENECK
BYTE_INTERFACE_GAIN_ESTABLISHED
INTERFACE_GAIN_BELOW_FUNCTIONAL_THRESHOLD
BYTE_INTERFACE_FUNCTIONAL
PAIRING_NOT_CAUSAL
DOWNSTREAM_MEMORY_LIMIT_REOPENED
```

## 30. Non-claims

V52 does not establish that phase is a biological fly variable, that the byte fixture is biologically natural, that XOR-off or invariant-tail is deployable, that nearest-neighbour identity decoding implies local-synaptic usability, that Jaccard is a sufficient statistic, that success at load 128 establishes scaling, or that byte-interface repair alone yields a general persistent learner.

The round succeeds if it localizes the missing operation sharply enough that V53 is forced by evidence rather than chosen from another menu of mechanisms.
