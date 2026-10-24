# V39 recommendation: bounded superposed memory

Written 9 September 2026, after `v36_results` (Claude, 56 trajectories, source SHA256
`0e2cbef0…aa66ec`), `v36_results` (ChatGPT, 48 trajectories), `v37_results` (16) and
`v38_results` (16).

**The V36–V38 arc solved the behaviour and paid for it with unbounded storage and an
O(N) read scan. Every question that is still scientifically open is on the other side
of a budget. The bounded arm is already written — `FLY-O` and `HASH-O` in
`elm_v36_generalizing_ownership.py` — and was gated off by a preflight threshold this
workload cannot satisfy by construction. V39 should correct that gate and run the
capacity experiment.**

Intervals throughout are paired Student 95% intervals over independent seed
differences, unadjusted for multiplicity.

---

## 1. What the new results settled

### 1.1 The V35 four-way wall broke

V35 ended with no arm holding more than two of {retention, revision, alias transfer,
prediction}: LOCAL-V at 99.22% retention and **0.0%** alias transfer, SHARED-R at 7.03%
and 28.1%. ChatGPT's V36 `SIM-V` and `PRED-V` hold all four — 100% acquired, 100% after
D, 100% current, 100% known revision, 100% held-out revision.

The mechanism is on the **read** side. Every write-side mechanism this program has
tested (output gain modulation, learning-rate cascades, novelty and norm gating,
variance salience, competitive plasticity ×3, compartments ×4, MESU metaplasticity)
failed its matched control. The first thing to pass all gates is a retrieval policy
over an append-only store. That inversion is the arc's real finding.

### 1.2 Two mechanism claims are on their control's boundary

| contrast | endpoint | value |
|---|---|---|
| V37 CORR2 − RAND-MATCH | held-out alias | **+0.1562 [+0.0007, +0.3118]** |
| V38 PBF − SHUFFLED-PBF | held-out alias | **+0.2188 [−0.0418, +0.4793]** |

The count-matched random evaluator already reaches 84.4% held-out transfer; the
shuffled-evidence control reaches 77.4%. CORR2's decisive, non-marginal advantage over
random is prediction, not transfer: whole bpb −0.1728 [−0.1936, −0.1521], and CORR2
beats the base predictor (−0.0668) where RAND-MATCH is worse than it (+0.1060).

V38's Q10 failure is a **power** failure, not a mechanism failure. PBF is the best
point on the frontier — it keeps 100% transfer and buys back 0.125 bpb of nontarget
prediction that CORR2 gives away:

| policy | E2 held-out | whole bpb vs base | nontarget vs base |
|---|---:|---:|---:|
| EXACT-V | — (0%) | −0.2191 | −0.1522 |
| **PBF** | **100%** | **−0.1970** | **−0.1251** |
| SHUFFLED-PBF | 77.4% | −0.1437 | −0.0677 |
| SLOT-EVID | 100% | −0.1424 | −0.0662 |
| ALPHA-0.50 | 92.6% | −0.1411 | −0.0491 |
| CORR2 | 100% | −0.0723 | +0.0092 |

From the reported intervals, sd ≈ 0.31 on PBF−SHUFFLED and ≈ 0.19 on CORR2−RAND-MATCH.
**n = 16 resolves both** if the variance holds. See §7.

### 1.3 Transfer requires an observed alias link

`E1_heldout_accuracy` is +0.0080 [−0.0064, +0.0224] for every policy; `E2` is 100%.
Held-out transfer is exactly zero until the second *trained* rendering binds to the
same slot, and total thereafter. The system will not infer that two renderings denote
one thing until it watches them co-occur. Similarity routes; evidence merges.

### 1.4 The price

V38's own state accounting, PRED-V trajectory:

| checkpoint | slots | mean members | payload | multiply-adds |
|---|---:|---:|---:|---:|
| after A | 354 | 1.305 | 4.0 MB | 458 M |
| after B | 733 | 1.307 | 5.8 MB | 1,163 M |
| after C | 1,150 | 1.307 | 8.1 MB | 2,147 M |
| after D | 1,504 | 1.328 | 9.5 MB | 3,453 M |

Consolidation happens but is weak: mean 1.33 members per slot, max 7, and slot count
still grows linearly (≈ +380 per wave). EXACT-V, with no consolidation at all, reaches
1,998 slots — PRED-V saves 25% of the store and none of the growth rate. **64 facts
cost 9.5 MB and 3.5 G multiply-adds.** The `len(self.values)*D` term in
`ByteModel.observe` is an O(N) scan per byte; compute rose 7.5× for a 4.2× slot
increase.

Both reports list unbounded storage as a limitation. V38's protocol already names the
next step: *"admission/consolidation priority under fixed capacity — future round."*

### 1.5 Separation is not the constraint in a scanning architecture

My V36 moved the separation ratio **1.433 → 5.708** (+4.275 [+4.007, +4.543]) with
phase-conditioned normalisation and label-free period discovery. Codes were stable
across the run (`ratio_before` ≈ `ratio_after`), so this is not a drift artefact.
`phase_expand` reached same 0.499 / different 0.089, structurally close to the fly's
0.417 / 0.032.

Behaviour moved the wrong way:

| arm | separation | target acc | target bpb |
|---|---:|---:|---:|
| raw/delta | 1.433 | 0.7502 | 2.113 |
| expand/delta | 2.596 | 0.8001 | 1.857 |
| phase_norm/delta | 5.708 | 0.7184 | 2.617 |
| phase_expand/delta | 5.614 | 0.7021 | 2.403 |
| raw/local_v | 1.433 | **0.9239** | 3.062 |
| phase_expand/local_v | 5.614 | **0.9213** | 3.109 |

Under `local_v` separation is simply irrelevant (0.9239 → 0.9213 across a 3.9× ratio
change). Under plain delta it costs 3–5 pp. **This falsifies "separation is the binding
constraint" for the scanning architecture at 64 keys, and only there.** §3 argues the
scope of that falsification is exactly the point.

---

## 2. Why the fly gate was mis-specified

ChatGPT's V36 preflight required, on both alias pairs:

$$J_{\rm same}\ge0.30,\qquad J_{\rm different}\le0.04,\qquad R_J\ge10.$$

Measured across four seeds: R_J **4.01–4.91**, J_diff **0.096–0.129**. Gate failed;
`fly_separation_gate_failed`; no fly trajectories ran.

The gate is unreachable on this workload regardless of code quality. The context is

```
PREFIX(4) + HEADER(1) + key(6) + SEPARATOR(1) = 12 bytes
```

so two different keys at the same alias share **6 of 12 bytes by construction**. My V36
measured the ceiling independently: expansion alone tops out at 2.60×, and the best
code I could build reaches 5.71×. The fly's 13.5× is not achievable here by any
encoder. The preflight therefore measured the **workload**, not the hypothesis, and
discarded a fully implemented arm on that basis.

This is the same class of error as V31's P0 and V27's headroom screen — a gate that
looks principled and is actually testing a property of the task. The correct
specification is relative (does the manipulation move separation?) plus a reported
achievable ceiling, which is what my V36's P1 did.

**Correction to my own §1.5 framing:** I initially reported to Mr. Zhou that
`max_members` was 1 at every checkpoint and nothing was ever consolidated. That is the
EXACT-V trajectory. PRED-V consolidates weakly (mean 1.33, max 7). The substantive
point — unbounded linear growth — survives; the specific claim was wrong.

---

## 3. The hypothesis V39 tests

**Code separation is not an independent design goal. It is what bounded superposed
storage requires, and nothing else does.**

The two architectures fail differently:

| | slot store (V35–V38) | superposed store (fly) |
|---|---|---|
| where memory lives | one value vector per slot | one weight matrix, all facts on top of each other |
| how two facts interfere | only if the read scan confuses them | always, in proportion to code overlap |
| cost of a new fact | +1 slot, +D scan per read | zero |
| what separation buys | better routing | **directly less interference** |

At 64 keys with 1,504 slots, routing is trivial, so separation buys nothing — which is
exactly what my V36 measured. In a fixed-size superposed store there are no slots to
scan; every write lands on every other, and interference *is* code overlap.

This predicts a specific, falsifiable interaction:

> Retention should be a steep increasing function of separation **under capacity
> pressure**, and flat without it.

It also explains the connectome number instead of merely copying it. The BANC v888
PN→KC query gave different-odour overlap 0.031 and same-odour 0.417 (13.5×) at 5%
coding, falling to ≈2.1× at 50% coding. If the hypothesis holds, the fly's sparse
coding level is not an aesthetic choice — it is the operating point a fixed KC
population needs in order to superpose without interference. That is a claim we can
test on our own workload, at our own achievable separations, without needing 13.5×.

---

## 4. Design

The bounded arm exists. `SparseCode` (`elm_v36_generalizing_ownership.py:580`) builds a
fixed 96 → 4096 code, 5 contacts per KC, seeded Gaussian contact weights, tanh drive,
top-5% by |activity| (K = 205). `ByteModel` already routes it to a fixed
`output[4096, 256]` matrix with a delta rule on the active set, reports
`'bounded': True`, and charges `205*OUTPUTS` operations per byte. `HASH-O` is the
matched random-hash control at identical K. Only `FLY_ARMS=()` stands in the way.

### Axis 1 — capacity pressure: load, not store size

Vary the number of facts at fixed store size, the standard associative-memory way.
Changing the store size would move separation and capacity together and confound them.

$$P \in \{64,\ 128,\ 256,\ 512\}\ \text{facts}$$

built by extending `make_stream`'s `count` (keys per wave) at fixed `repeats = 32`, so
every fact receives identical training regardless of load.

### Axis 2 — separation: coding level

$$K/N \in \{5\%,\ 10\%,\ 25\%,\ 50\%\}\qquad (K \in \{205,\ 410,\ 1024,\ 2048\},\ N = 4096)$$

This is the connectome's own knob and the one for which we have measured separations.
Report the achieved separation ratio for each level rather than assuming it.

### Axis 3 — structure control: FLY-O vs HASH-O

At matched K and matched sparsity, structured KC wiring (5 contacts, tanh drive) versus
a random hash of the exact context. This is the matched control the program's history
demands. If they are equal, sparsity — not connectome structure — is doing the work,
and we say so.

### Grid

| block | arms | cells | purpose |
|---|---|---:|---|
| A | FLY-O, load × coding | 16 | primary interaction |
| B | HASH-O, load × coding at 5% and 50% | 8 | structure control |
| C | PRED-V slot store, load ∈ {64, 128} | 2 | architecture reference |
| D | EXACT-V slot store, load ∈ {64, 128} | 2 | no-consolidation reference |

28 cells × 16 seeds = **448 trajectories**. The superposed arms are cheap
(O(K · OUTPUTS) per byte, constant in load). The slot arms are not — their cost grows
as O(N_slots · D) per byte and becomes infeasible above load 128. **That infeasibility
is a reportable endpoint, not an obstacle:** report multiply-adds per byte for both
architectures at every load they run, and the extrapolated crossover.

### Block E — does capacity pressure produce transfer? (secondary)

`E1`/`E2` are defined in V37/V38 by slot-binding events, which do not exist in a
superposed store. Redefine them by **stream position** — E1 at the last byte before the
first exposure of alias 1 for that key, E2 after it — so the same checkpoints are
computable for both architectures. Then ask whether E1 held-out accuracy, pinned at
0.8% in V38, rises with load.

If forced merging under budget produces transfer the unbounded store never achieves,
that is the most interesting result available here: **forgetting pressure as the source
of generalisation.** If E1 stays at zero across all loads, we learn that transfer
strictly requires an observed alias link in both architectures, and we stop asking.

---

## 5. Predeclared gates

Ordered. A failure at any level voids the levels beneath it — the V31 P0 lesson.

**G0 — acquisition validity.** Every cell must reach ≥90% accuracy on wave A
immediately after wave A at load 64. A cell that never learns cannot be scored. Failing
cells are reported `invalid_acquisition` and excluded from the interaction estimate.

**G1 — the separation axis is real.** Measured separation ratio must span ≥2× between
the 5% and 50% coding levels, paired interval excluding zero. Otherwise
`separation_axis_void` and Blocks A/B are uninterpretable.

**G2 — capacity pressure exists.** At the lowest separation (50% coding), retention at
load 512 must be at least 10 pp below retention at load 64, paired upper bound below
zero. If load hurts nobody, there is no pressure and the design is void:
`no_capacity_pressure`. This is V27's headroom screen applied to capacity, and it is
the gate most likely to fail.

**G3 — primary endpoint, the interaction.** Difference-in-differences on retention:

$$\Delta\Delta = \big[R(5\%, P{=}512) - R(50\%, P{=}512)\big] - \big[R(5\%, P{=}64) - R(50\%, P{=}64)\big]$$

Required: **≥ +10 pp with paired 95% lower bound above zero.** Seeds are the sampling
unit.

**G4 — structure control.** FLY-O vs HASH-O at matched K, all loads. Reported, not
required. If equal within ±5 pp, the verdict is `sparsity_effect_not_structure`: the
capacity result stands, the connectome wiring claim does not.

**G5 — cost accounting.** State bytes and multiply-adds per byte for every arm at every
load, plus the crossover load. Descriptive; no threshold.

### Verdict vocabulary

V35/V36 vocabulary — `qualified`, `supported_below_qualification`,
`favorable_unresolved_precision`, `no_mean_advantage`, `adverse`,
`invalid_acquisition` — plus `separation_axis_void`, `no_capacity_pressure`,
`sparsity_effect_not_structure`.

Do not compress a graded effect into a binary.

---

## 6. What each outcome teaches

| outcome | reading |
|---|---|
| G3 passes, G4 shows FLY-O > HASH-O | separation is a capacity requirement **and** connectome structure beats random sparsity. Strongest available result. |
| G3 passes, FLY-O ≈ HASH-O | separation-by-sparsity is a real capacity requirement; the wiring is not special. Still resolves §1.5 and justifies the fly's coding level. |
| G2 fails | 512 facts is not enough pressure for a 4096×256 store. Rerun at higher load or smaller N before concluding anything. Cheap to detect early. |
| G3 fails with G2 passing | capacity hurts everyone equally; separation is not what protects a superposed store. The connectome analogy is then dead in both architectures and we stop paying attention to it. |
| Block E shows E1 rising with load | forgetting pressure produces generalisation. New, and the most valuable thing here. |

Every branch is publishable to Mr. Zhou as a result. None requires the hypothesis to be
true.

---

## 7. Power

V38 ran 8 seeds with 32 held-out items, quantising the endpoint to 3.125 pp steps; both
decisive controls landed on their interval boundary as a direct consequence. V39 uses
**16 seeds**, and load ≥ 128 raises the held-out item count with it.

Separately and independently of V39, the two boundary contrasts in §1.2 should be
re-run at 16 seeds. `SEEDS` is a module constant
(`elm_v38_predictive_evidence.py:27`, `tuple(range(38101,38109))`), not a flag, so this
is a one-line edit and a rerun into a fresh output directory — the existing 8
trajectories are reused by the resume logic, so only 8 are new. Do it concurrently; it
does not interact with V39.

Also replace Q10's three-endpoint OR with one pre-registered primary. The OR let the
coarsest endpoint drive the verdict.

### 7.1 Runtime and staging

The superposed update is `output[ix] += delta` with `delta` of shape (K, 256), so cost
per byte is O(K · 256) and the 50% coding cells are ~10× the 5% cells. At load 512 the
acquisition stream alone is 512 × 32 × 14 ≈ 229 kB, giving roughly 1.2 × 10¹¹
multiply-adds per trajectory at K = 2048. Block A is dominated by its four
high-coding / high-load cells.

Stage the run so a failed gate is cheap to discover:

1. `--smoke` (4 keys, 4 repeats, 2 seeds) — wiring only, no verdicts.
2. **G2 first**, at 50% coding, loads 64 and 512, 16 seeds. This is the gate most
   likely to fail and it costs a small fraction of the grid. If there is no capacity
   pressure, stop and re-scope before running Blocks A/B.
3. Full grid only after G2 passes.

`HASH-O`'s active-set size is hardcoded (`rr.choice(4096, 205, replace=False)`) and
must be parametrised by K for Axis 2 to be matched between the two structures.

---

## 8. Not in scope

New write rules; further arbitration variants (V38 shows PBF adds exactly
+0.0000 [0, 0] over CORR2 on every behavioural endpoint); selection heuristics of any
kind — that class has failed matched controls six times; code geometry *inside the
scanning architecture*, which §1.5 closes; language modelling; hierarchical or
multi-horizon architectures; retuning anything in the closed list.

**Scaling beyond load 512 and real byte corpora stay deferred** — now behind capacity
rather than behind task validity. If G2 or G3 fails, the right next move is a real
corpus with the unbounded store and honest cost accounting, not another mechanism
hunt.

---

## 9. Provenance

- `v36_results` (Claude) — 56 trajectories, 7 arms × 8 seeds, source SHA256
  `0e2cbef0…aa66ec`. `TERMINAL.csv`, `GATES.json`.
- `v36_results` (ChatGPT) — 48 trajectories. `PREFLIGHT.json` (four seeds, fly gate),
  `PROTOCOL.md` §24–24.1, `SUMMARY.md`.
- `v37_results` — 16 trajectories, 11 read policies on one PRED-V state.
  `REPORT.json` paired contrasts and qualification.
- `v38_results` — 16 trajectories. `REPORT.json` levels and qualification,
  `STATE_COSTS.json`, `PROTOCOL.md` §Q8–Q10.
- `elm_v36_generalizing_ownership.py:580` (`SparseCode`), `:603` (`ByteModel`),
  `elm_v38_predictive_evidence.py:248` (`context`/`make_stream`), `:348` (`ByteModel`),
  `:24` (`FLY_ARMS=()`).
- BANC v888 PN→KC overlap query (5% coding: 0.417 / 0.031; 50% coding: ≈2.1×).
- `V35_design.md`, `v28_experiment_recommendation.md`, `v31_results/REPORT.md`.

Corrections recorded here: the `max_members` claim in §2; and the scope of the
separation falsification in §1.5, which holds for the scanning architecture at 64 keys
and is the premise of §3 rather than a general result.
