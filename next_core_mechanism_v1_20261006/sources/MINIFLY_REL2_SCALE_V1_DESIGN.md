# MiniFly REL2-Scale v1 — Frozen E3 Experiment Design

**Status:** design ready for independent audit; no official science is authorized until all three qualification cycles pass.

## 1. Scientific question and evidence level

Primary question:

> Can a bounded second-order conjunctive shared representation, trained only from observed outcomes and without task/row/column identities, convert MiniFly's currently harmful held-out transfer into positive persistent E3 transfer?

The primary claim, if successful, is **E3 transfer within the registered 4×4 Latin-table family only**. It is not E4, not a general reasoning claim, and not evidence that the literal fly implements this mechanism.

A second, non-promotional question is whether the incumbent shared representation already contains information that a legal generic decoder can exploit.

## 2. Frozen parent and world

Use the exact frozen `R_center` parent from the accepted table-completion experiment:

- shared FE0 bank;
- private CONTENT bank;
- centered private teaching `1/4 - y`;
- same sensory route, timing, native scales, state budget and output tie rule;
- same old/new teaching schedule and delay schedule;
- same held-out table-completion generator.

Do not alter the previous experiment or reuse its official worlds.

Each science world must be a fresh relabeling from the same generator. Twelve of sixteen table cells are taught. Four balanced cells, one per row/column and one per output, are never reinforced. The new/interference block and both delays remain exactly as in the accepted Round 1 protocol.

The official world roster is frozen and hashed before science and must be disjoint from every prior official, pilot and preflight roster.

## 3. New mechanism: REL2

### 3.1 Input

Let `h(x)` be the exact normalized shared FE0 cue representation available immediately before the outcome byte and used by the native shared bank. The source audit must identify the exact object and prove that extracting it does not mutate learner state.

### 3.2 Fixed conjunctive transform

Construct one additional bounded relation feature vector `q(x)` of the same dimensionality as `h(x)`.

Use two fixed signed sparse projections `A` and `B`, generated once from a design seed and reused in every world. Their fan-in per output must equal the nearest source-native FE0 fan-in; no new fan-in hyperparameter is introduced.

```text
z(x) = (A h(x)) ⊙ (B h(x))
q(x) = native_feature_normalize(z(x))
```

`native_feature_normalize` must be an already existing source-equivalent normalization routine. If no such routine exists, the design returns for audit rather than inventing a fitted normalization constant.

`A` and `B` are fixed architectural state. They never depend on a world, cue identity, table coordinate, output, held-out status or evaluator label.

Interpretation: REL2 is a bounded second-order/dendritic-conjunction representative of the relational-representation family. It is not claimed to be a literal fly microcircuit.

### 3.3 Relation value bank

Add one value bank with the same value-store/update primitive as the native shared bank, but driven by `q(x)`.

It receives exactly the same **observed** four-way outcome teaching signal as `R_center`. It receives no prediction of the held-out answer and no row/column/table identity.

The relation bank is updated on every ordinary taught event, including later new/interference events. It gets no task flag that would allow it to protect old relation events selectively.

Normal prediction is

```text
u = shared_normalized + private_normalized + relation_normalized
```

where the relation bank uses the same native bank-specific scaling semantics as the other banks. No fitted readout alpha is introduced.

## 4. Prespecified scale bracket

The only candidate parameter varied in science is relation-write dose:

- `REL_0.5`: 0.5 × the source-native relation write;
- `REL_1.0`: 1.0 ×;
- `REL_2.0`: 2.0 ×.

The multiplier acts on the relation-bank value write only. It does not change `h`, `q`, native shared/private writes, read scaling, timing or state size.

There is **no** intermediate/post-hoc dose, learning-rate search, temperature search, alpha sweep or sample-size adaptation.

`REL_1.0` is the primary mechanism representative. `REL_0.5` and `REL_2.0` are prespecified scale-rescue arms.

## 5. Science arms

Run six candidate/control configurations on every official world:

| Arm | Description |
|---|---|
| `CENTER` | Frozen `R_center`, no relation bank |
| `REL_0.5` | REL2 at 0.5× write dose |
| `REL_1.0` | REL2 at 1.0× write dose |
| `REL_2.0` | REL2 at 2.0× write dose |
| `REL_SHUFFLE_1.0` | Same state, timing, update norm and relation bank as `REL_1.0`, but a fixed signed permutation is applied to `q` on writes only and not on reads |
| `LIN_EXTRA_1.0` | Capacity control: same extra bank/state/update law as REL2, but driven by a second copy of the ordinary first-order `h(x)` rather than `q(x)` |

`REL_SHUFFLE_1.0` tests aligned conjunctive write/read geometry. `LIN_EXTRA_1.0` tests whether an improvement is merely due to adding another bank/capacity.

No eligibility, forgetting, metaplasticity, topology, replay, row parser, table parser or CORR2 mechanism is added in this round.

## 6. Causal branches

Within every arm and world, run the same three matched branches:

- `W`: all ordinary value writes enabled;
- `N_old`: old-table plastic value writes disabled in **all** banks, including relation/extra banks, while sensory input, clocks and nonplastic evolution continue;
- `N_new`: new/interference plastic value writes disabled in all banks.

No held-out query receives feedback before its scored response.

All probes use disposable clones and must leave the continuing learner state byte-identical.

## 7. Schedule and checkpoints

Reuse the accepted Round 1 schedule exactly:

1. old teaching block;
2. `OLD_END` probe;
3. first 86,400 s delay;
4. new/interference teaching block;
5. `AFTER_NEW` probe;
6. second registered delay;
7. `FINAL` probe.

The primary E3 endpoint is the **FINAL** held-out response after intervening learning and delay.

`OLD_END` and `AFTER_NEW` are prespecified localization endpoints, not alternate primary endpoints.

## 8. Required saved outputs

For every probe save, before any feedback:

- shared four-channel score;
- private four-channel score;
- relation/extra-bank score when present;
- actual combined score;
- selected answer;
- correctness;
- branch;
- checkpoint;
- cue index;
- state digest before and after the disposable probe.

For technical/diagnostic storage also save the exact `h(x)` and `q(x)` probe vectors or a lossless binary sidecar sufficient to reconstruct them. These vectors are not passed back to the learner.

Record per arm/world:

- mutable-state bytes;
- fixed architectural bytes;
- total update L1/L2/max movement by bank;
- write counts;
- CPU/wall time;
- peak RSS.

## 9. Read-only localization diagnostics

These diagnostics never feed the learner and cannot qualify a candidate.

At `OLD_END`, on the frozen `CENTER` trajectory:

### `DEC_LIN`

Use the twelve taught cells only. Fit a minimum-norm four-output linear least-squares decoder from their saved `h(x)` vectors to observed one-hot outcomes. No ridge/grid/validation is allowed. Score the four held-out cells.

### `DEC_REL2`

Repeat using `q(x)` instead of `h(x)`.

### `LATIN_CERT`

External positive control only: reconstruct the four missing labels from the twelve observed table entries using the declared Latin-table constraint. It must solve all four; it is not a learner arm.

Interpretation:

- `DEC_LIN` succeeds: existing representation is usable by a generic legal readout; prioritize structured composition next.
- `DEC_REL2` succeeds but integrated REL2 fails: relation representation is available but online bounded teaching/read integration is the bottleneck.
- both fail: this second-order feature family is insufficient; move to a genuine learned equivariant/constraint mechanism, not a gain sweep.

A successful diagnostic decoder is never reported as native MiniFly E3 success.

## 10. Exposure/shortcut audit

The report must explicitly verify:

- none of the four scored held-out table cells was reinforced;
- first held-out answers precede any feedback;
- no learner object receives row, column, table, world, held-out or evaluator identity;
- exact-key lookup has no held-out answer;
- global label-frequency lookup is tied by construction and cannot exceed the 25% balanced baseline systematically;
- positive same-row/same-column interpolation is anti-aligned with the held-out label on this fixture;
- the external Latin completion rule is kept outside the learner.

## 11. Official sample and inference unit

Run **64 fresh independent paired worlds**.

The world is the sole independent unit. Cues, probes, branches and repeated presentations are not additional replicates.

Every world receives every arm. No world replacement, outcome-driven stopping, extension or sample-size adaptation is permitted.

## 12. Primary behavioral family

For each of `REL_0.5`, `REL_1.0`, `REL_2.0`, compute four world-level contrasts:

1. `E3_FINAL = accuracy(W, heldout, FINAL) - accuracy(N_old, heldout, FINAL)`;
2. `RAW_FINAL = accuracy(W, heldout, FINAL) - 25%`;
3. `OLD_NI = final exact-old W accuracy - CENTER final exact-old W accuracy`;
4. `NEW_NI = final new W accuracy - CENTER final new W accuracy`.

These 12 contrasts are one frozen inferential family. Use the same world-level Student-t/Bonferroni simultaneous-interval machinery as Round 1, generalized to `m=12`.

A dose **passes the integrated E3 gate** only if all four simultaneous lower bounds satisfy:

```text
E3_FINAL > 0
RAW_FINAL > 0
OLD_NI > -5 percentage points
NEW_NI > -5 percentage points
```

Raw improvement from 14% to 20–24% is a reduction in harm, not completion success.

## 13. Mechanism-specificity family

Two separate primary mechanistic contrasts, corrected together:

1. `E3_FINAL(REL_1.0) - E3_FINAL(REL_SHUFFLE_1.0)`;
2. `E3_FINAL(REL_1.0) - E3_FINAL(LIN_EXTRA_1.0)`.

Mechanistic specificity is supported only if both simultaneous lower bounds are `> 0`.

If `REL_1.0` passes behavior but not specificity, report behavioral improvement without claiming that conjunctive relational geometry is the identified cause.

If only `REL_0.5` or `REL_2.0` passes, label it `SCALE_RESCUE_CANDIDATE`; do **not** claim mechanism specificity until that exact frozen dose receives its own matched shuffle/capacity confirmation on fresh worlds.

## 14. Terminal decision rules

- `REL_1.0` passes behavior + both specificity contrasts: **ADVANCE_REL2**.
- `REL_1.0` passes behavior only: **ADVANCE_BEHAVIOR_MECHANISM_UNRESOLVED**.
- only 0.5× or 2× passes: **SCALE_RESCUE_CANDIDATE**, fresh exact confirmation required.
- none passes but held-out harm is reduced: **NEGATIVE_TRANSFER_REDUCED_ONLY**.
- none passes and no meaningful reduction: **REL2_BRACKET_CLOSED** for this fixture and dose range. Do not add 0.75×, 1.5×, new temperature, extra rank or more worlds afterward.
- `DEC_LIN`/`DEC_REL2` outcomes determine the next *new protocol* as described above; they never rescue the current preregistration.

## 15. Three qualification cycles before science

### Cycle 1 — source/science contract

Must verify:

- exact frozen parent/source hash;
- exact `h(x)` source object;
- extraction is read-only;
- relation bank uses the native shared-bank update primitive;
- scale multiplier touches relation writes only;
- fixture/schedule byte-for-byte matches the accepted generator;
- official roster is fresh and locked.

### Cycle 2 — leakage/causal adversarial tests

Must verify:

- held-out labels cannot affect any learner state before scoring;
- `A/B` projections have no world/evaluator dependence;
- `REL_SHUFFLE` preserves feature/write norms while breaking write/read coordinate alignment;
- `LIN_EXTRA` matches REL state size and update primitive;
- W/N_old/N_new differ only at the intended value-write intervention;
- probes and diagnostics are non-mutating;
- outcome permutation tests produce the expected state/score changes and no hidden table shortcut appears.

### Cycle 3 — recovery/resource/exact replay

Must verify:

- full-life technical run;
- checkpoint kill/restart and exact continuation;
- fresh-process deterministic replay;
- exact receipt/readback;
- peak RSS < 768 MiB per worker;
- runtime < 900 s per world-arm attempt;
- all state/fixed-byte accounting complete.

Technical/preflight worlds are permanently excluded from official inference. No performance-based parameter selection is allowed in qualification.

## 16. Reporting

The final report must separate:

- integrated native E3;
- E1 old/new preservation;
- scale characterization;
- mechanistic specificity;
- read-only localization diagnostics;
- resource/state cost;
- failures/interruptions/retries.

The strongest permissible positive statement is:

> A bounded second-order conjunctive shared-value mechanism improved persistent held-out completion within the frozen 4×4 Latin-table family under the registered dose and controls.

Do not call this general reasoning, E4, a complete persistent-learning core, or proof of a literal fly mechanism.
