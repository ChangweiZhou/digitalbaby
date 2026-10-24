# V40 recommendation: complementary two-layer memory

Written 9 September 2026, after two independent V39 runs: `elm_v39_science_results`
(ChatGPT, 8 seeds, 48 memory branches) and `v39_results` (Claude, 26 cells × 16 seeds
= 416 trajectories, source SHA256 `0f9577b4…a78d0`).

**The two V39s solved different halves of the same problem and neither half works
without the other. Admission-controlled slots bound storage 11.3× but cannot
generalize to an unseen rendering before it has been seen twice. A bounded superposed
store generalizes one-shot at 0.998 but collapses under load. V40 should test whether
one memory can hold both, using the read rule V37 already validated to arbitrate
between them.**

Intervals are paired Student 95% intervals over independent seed differences,
unadjusted for multiplicity.

---

## 1. What V39 settled

### 1.1 Bounded storage works, and the mechanism is admission control

ChatGPT's `TRACE-CORR` keeps one-shot exact traces on probation and promotes to a
permanent slot only after a second **distinct, predictively confirmed** rendering.
Hard caps: 512 traces, 384 slots.

| policy | A after D | post-clutter | post-clutter held-out | held-out rev @4 | whole bpb | peak store |
|---|---:|---:|---:|---:|---:|---:|
| LEGACY-U (unbounded) | 1.0000 | 0.9199 | 0.9062 | 0.9375 | 2.4136 | **22.56 MB** |
| **TRACE-CORR** | 1.0000 | 0.9082 | 0.9062 | 0.9375 | 2.4688 | **2.00 MB** |
| FULL-CAP (cap, no admission rule) | 1.0000 | 0.5000 | 0.3945 | 0.9375 | 2.5137 | 1.05 MB |
| TRACE-EXACT2 | 1.0000 | 0.5469 | 0.0352 | **0.0000** | 2.4009 | 2.10 MB |
| SHUFFLED-PAIR | 0.5781 | 0.5068 | 0.0273 | 0.1250 | 2.5233 | 2.00 MB |

11.3× reduction for 1.2 pp of post-clutter accuracy and 0.055 bpb. The clutter phase
decides it: LEGACY-U goes 4.03 → 22.56 MB absorbing irrelevant material; TRACE-CORR
goes 1.83 → 2.00 MB. Controls are decisive — vs FULL-CAP +0.4082 [+0.3615, +0.4549],
vs SHUFFLED-PAIR +0.4014 [+0.3836, +0.4191] post-clutter and +0.8125 [+0.7157, +0.9093]
held-out.

**This is the specificity veto that sank V38's PBF, and it passes.** Moving the
mechanism from evidence-weighting to admission control is what fixed it.

Q12 and Q13 failed on threshold strictness, not mechanism: Q12's `exact_recurrence` is
1.0000 (passes ≥0.90) and fails only on `all(max_admission_exposure == 1)`, where 6 of 8
seeds are truly one-shot; Q13's `E2_immediate_heldout` is 0.9375 (passes ≥0.90) and
fails on a companion term requiring exactly 1.000, achieved 0.9336. The headline
"NOT ESTABLISHED" understates its own result.

### 1.2 Bounded superposition works and is third

| architecture | store | wall (§2) | ops/byte |
|---|---:|---:|---:|
| TRACE-CORR (bounded slots + admission) | **2.00 MB** | 0.03 | bounded |
| PRED-V (unbounded slots) | 9.1 MB | 0.027 | 117,652, grows |
| S2-lag-c12 (bounded superposed) | 8.8 MB | **0.936 @32**, 0.092 @256 | 158,208, constant |

Superposition genuinely works and at load 32 is indistinguishable from an unbounded
slot store (PRED-V − S2-lag-c12 = +0.0312 [−0.0512, +0.1137]), but by load 64 it is
behind (+0.1289 [+0.0752, +0.1826]) and it costs 4× the memory of admission-controlled
slots. Constant compute is its only structural advantage, and TRACE-CORR's caps bound
its scan too.

### 1.3 The separation ratio is not the lever, and stationarity dominates it

| front end | arm | ratio | wall @32 | same-context self-overlap over time |
|---|---|---:|---:|---:|
| lag | HASH-O | 1.02 | 0.000 | **1.0000** |
| lag | S1-lag-c2 | 1.49 | 0.000 | **1.0000** |
| lag | **S2-lag-c12** | **2.70** | **0.936** | **1.0000** |
| phase | S3-phase-c5 | 6.13 | 0.281 | 0.62–0.74 |
| phase | S4-phase-c12 | 6.77 | 0.422 | 0.62–0.74 |
| phase | S5-phase-c12-k82 | **11.01** | 0.477 | 0.62–0.74 |

`S2-lag-c12` beats the highest-ratio arm at every load — +0.5156 [+0.3805, +0.6508] at
load 32, +0.0732 [+0.0463, +0.1002] at 256 — despite a quarter of its separation ratio.
Within a stationarity class higher ratio is better; across classes stationarity wins.
The phase normaliser's running statistics never stop updating, so a fact written at
t = 5,000 is read at t = 29,000 through a code sharing 62% of its active units with the
one it was written under.

`S5-phase-c12-k82` reached ratio **11.011 [10.247, 11.775]**, J_diff **0.032**, J_same
**0.350** — clearing all three of ChatGPT's V36 preflight conditions (≥10, ≤0.04, ≥0.30)
on the workload where that gate was declared unreachable. Clearing it bought nothing.

**Two corrections to my own earlier claims.** The V39 design doc §2 said the gate was
unreachable by any encoder; it was reachable, and reaching it did not help. And V36's
conclusion that "separation rose 4× and behaviour fell, so separation is not the
constraint" was confounded: the codes that raised separation were also non-stationary,
and V36 checked that the *ratio* was stable, never that the *code* was.

### 1.4 A gate bug worth recording

V39's G2 selected the best arm across all arms; PRED-V won it, but the slot arms run at
`SLOT_LOADS` only, so the check against the highest load found nothing and the gate
voided itself. The staged screen passed earlier only because no slot arms were in that
stage. Corrected in `reanalyse_v39.py` without rerunning trajectories; the corrected
verdict is `qualified`. **G3 still tested the wrong arm** — it selects by measured ratio,
so it scored S4 and returned a technically-true result while S2 was better. Left as
preregistered; changing it after seeing data would be post-hoc selection.

---

## 2. The finding that should drive V40

V38 concluded that held-out transfer requires an observed alias link, with
`E1_heldout_accuracy` pinned at 0.0080 [−0.0064, +0.0224] for every policy. **That is a
property of the slot architecture, not of memory.**

| architecture | E1 held-out (one-shot, before any alias link) |
|---|---:|
| PRED-V (slots, similarity routing) | 0.016 – 0.027 |
| EXACT-V (slots, exact routing) | 0.016 – 0.027 |
| HASH-O (superposed, no input similarity) | 0.002 |
| **S2-lag-c12 (superposed)** | **0.998** |
| **S1-lag-c2 (superposed)** | **1.000** |
| S4-phase-c12 (superposed) | 0.949 |

A slot store must *bind* before it can generalize, and binding is what E1 is defined to
precede. A superposed store has no binding event: writing alias 0 writes alias 2's
neighbourhood automatically, in proportion to same-key overlap. ChatGPT's own Q13 —
"immediate E1→E2 consolidation transfer" — failed for precisely this reason, and the
other architecture solves it for free.

### The wall, as one number

Define the V35 four-way wall as a single scalar over the three
architecture-discriminating components, all measured on the untouched wave-A cohort so
they share a delay:

$$\text{wall} = \min(\text{retention},\ \text{delayed held-out transfer},\ \text{E1 one-shot transfer})$$

| best of each family | retention | delayed | E1 | **wall** |
|---|---:|---:|---:|---:|
| PRED-V@64 (slots, unbounded) | 1.000 | 1.000 | 0.027 | **0.027** |
| TRACE-CORR (slots, bounded) | 1.000 | 0.906 | ~0.03 | **~0.03** |
| S2-lag-c12@32 (superposed) | 0.984 | 0.938 | 0.998 | **0.936** |
| S2-lag-c12@64 | 0.922 | 0.898 | 0.999 | **0.871** |
| S2-lag-c12@128 | 0.645 | 0.492 | 0.995 | **0.488** |
| S2-lag-c12@256 | 0.197 | 0.092 | 0.985 | **0.092** |

**No architecture holds the wall above load 64.** Slots are capped at ~0.03 by E1 at
any load; superposition holds 0.94 at load 32 and decays to 0.09 by 256. The failure
modes are exactly complementary, which is what makes a two-layer design worth testing
rather than assuming.

---

## 3. Hypothesis

> A fast superposed layer supplies immediate generalization; a slow
> admission-controlled slot layer supplies bounded retention; a corroboration-gated
> read rule decides which speaks. The pair holds the wall at loads where neither
> single layer does.

This is complementary-learning-systems structure arrived at empirically rather than
borrowed, and every component is already validated:

- the superposed layer is V39's `S2-lag-c12` (lag96 → 4096 KC, 12 contacts, K = 205);
- the slot layer is V39's `TRACE-CORR` (probationary traces, two-hit admission, caps);
- the read rule is V37's `CORR2`, whose only change is arbitrating three sources instead
  of two.

### Read rule

For each byte, in order:

1. if the exact context holds a **corroborated** slot (support ≥ 2), use the slot value —
   this is V37's earned authority, unchanged;
2. else use the **superposed** layer's prediction — this is where E1 generalization comes
   from, and it is available from the first exposure;
3. else the shared base predictor.

Nothing new is invented. The claim under test is that the *composition* holds properties
neither part holds alone.

---

## 4. Design

Build on `elm_v39_earned_consolidation.py`, which already carries the base learner,
traces, slots, admission, eviction and the V37 read policy. Adding one superposed layer
to it is far less code than porting admission control into the V39 capacity harness, and
it keeps the bounded-storage result byte-comparable.

### Block A — architectures (primary)

| arm | layers | purpose |
|---|---|---|
| `SUPER` | superposed only | V39 `S2-lag-c12`, the generalization corner |
| `SLOTS` | base + traces + slots | V39 `TRACE-CORR`, the retention corner |
| **`HYBRID`** | base + superposed + traces + slots | the mechanism under test |
| `HYBRID-SHUFFLE` | as HYBRID, superposed output permuted across contexts within support bins | destroys the layer's content, keeps its timing and magnitude |
| `HYBRID-RANDOM` | as HYBRID, arbitration decided at random at matched rates | the V37 `RAND-MATCH` analogue |
| `HYBRID-FROZEN-SUPER` | as HYBRID, superposed layer frozen after wave A | separates "generalizes" from "keeps learning" |

`HYBRID-SHUFFLE` and `HYBRID-RANDOM` are nondeployable controls and can never qualify.

### Block B — load (the axis the primary contrast is measured on)

$$P \in \{64,\ 128,\ 256,\ 512\}$$

The hybrid's claim is specifically about loads where superposition collapses, so scale
is not a separate round. This also retires "scaling untested since V20" as a side
effect. Note ChatGPT's slot cap already binds occasionally at P = 64 (peak 384 of 384 in
some seeds, mean 219); at P = 512 it must bind constantly, so the bounded claim is
tested where it matters for the first time.

### Block C — stationarity (cheap, resolves the V39 confound)

Two front ends for the superposed layer: `lag` (stationary by construction) and `phase-frozen`
(phase-conditioned normalisation whose statistics are **fitted on a warmup prefix and then
frozen**). Still label-free and causal. Report same-context self-overlap over time for
both; a front end whose self-overlap falls below 0.95 is flagged non-stationary and its
behavioural cells are reported but excluded from the primary contrast.

This is the experiment V36 and V39 both intended and neither ran: whether a *stationary*
code above ratio 2.70 beats `S2-lag-c12`.

### Block D — sustained noise (secondary)

Corroboration's entire justification, tested so far with exactly one one-off corruption.
Sweep a corruption rate $\varepsilon \in \{0,\ 0.02,\ 0.05,\ 0.10\}$ applied to target
bytes. The specific vulnerability is stated in advance: **a two-hit admission rule is
defeated by any error that occurs twice**, and at $\varepsilon = 0.10$ over 32 exposures
a repeated error is likely. Report the admission false-positive rate against
$\varepsilon$. Runs only if Block A produces a qualified hybrid.

---

## 5. Predeclared gates

Ordered; a failure voids the levels beneath it.

**G0 — acquisition validity, at every load.** Every cell entering a contrast must reach
≥ 0.90 wave-A acquisition at the load where it is compared. Cells below it are
`invalid_acquisition` and enter no contrast. (V39's G0 caught an arm at 0.184 that the
lowest-load-only version would have admitted.)

**G1 — the gap exists.** Both single layers must fail in the predicted complementary way
at the primary load: `SLOTS` E1 ≤ 0.10, and `SUPER` wall at least 0.30 below its own
value at P = 64, each with a paired interval excluding zero. If either layer already
holds the wall, there is no gap to fill and the design is void — `no_complementary_gap`.

**G2 — primary.** At the largest load where `SUPER`, `SLOTS` and `HYBRID` all pass G0,
`HYBRID` must beat **both** single layers on `wall` by ≥ **0.10** with paired 95% lower
bounds above zero.

**G3 — specificity.** `HYBRID` must beat `HYBRID-SHUFFLE` and `HYBRID-RANDOM` on `wall`
by ≥ **0.10** with lower bounds above zero. If the shuffled layer matches, the hybrid is
an ensembling effect and not a generalization mechanism — this is V38's Q10 applied to
the new mechanism, and V39 showed the veto is passable when the mechanism is real.

**G4 — cost.** Total payload and ops/byte for every arm at every load, plus whether the
slot and trace caps bound. `HYBRID` must stay bounded in both. Descriptive threshold:
report, do not gate, except that an unbounded hybrid cannot qualify.

**G5 — stationarity.** Same-context self-overlap over time for each front end.
Descriptive; drives the exclusion rule in Block C.

**Prediction endpoint** (`whole_bpb`, `nontarget_bpb`) reported for every arm against the
base predictor. `HYBRID` must be non-inferior to `SLOTS` within 0.10 bpb; a hybrid that
buys the wall by wrecking prediction has not solved anything.

### Verdict vocabulary

`qualified`, `supported_below_qualification`, `favorable_unresolved_precision`,
`no_mean_advantage`, `adverse`, `invalid_acquisition`, plus `no_complementary_gap` and
`ensembling_not_mechanism` (G3 fails).

---

## 6. What each outcome teaches

| outcome | reading |
|---|---|
| G2 and G3 pass | Two-layer memory holds the full V35 wall under bounded storage at loads neither layer survives. The strongest result the program could produce, and the first architecture claim rather than a mechanism claim. |
| G2 passes, G3 fails | The gain is ensembling two predictors, not complementary memory. Still useful, must be named honestly, and closes the CLS analogy. |
| G2 fails with G1 passing | The gap is real and this composition does not close it. Then the read rule is the suspect, and the next question is arbitration, not architecture. |
| G1 fails | A single layer already holds the wall at high load. Report it and stop — the whole premise dissolves and the program's remaining questions are noise and real data. |
| Block C shows a stationary high-ratio code beating S2 | Separation is the lever after all, and V36 and V39 both under-measured it because their high-separation codes drifted. |
| Block D shows admission collapsing at ε = 0.05 | The two-hit rule is not noise-robust and bounded storage is not solved for real streams. |

Every branch is reportable. None requires the hypothesis to be true.

---

## 7. Power and staging

**16 seeds.** V38 ran 8 and both of its decisive controls landed on their interval
boundary; V39 at 16 resolved contrasts of ±0.07 cleanly.

Stage so that the cheapest voiding failure is found first:

1. `--self-test` — wiring, and byte-identity of the workload with V39.
2. **G1 first**, at the primary load, `SUPER` and `SLOTS` only. If there is no
   complementary gap, stop before building the hybrid.
3. Block A at the primary load.
4. Block B load sweep, then Block C.
5. Block D only on a qualified hybrid.

Runtime: V39's superposed arms cost ~117–158 k ops/byte, constant in load
(3.1 s at P = 64, 31 s at P = 256 per trajectory); the slot arms grow but are capped.
The hybrid is approximately the sum. Budget ~3–4 hours for Blocks A–C at 16 seeds.

---

## 8. Not in scope

New write rules; further arbitration variants (V38: PBF adds +0.0000 [0, 0] over CORR2
on every behavioural endpoint); selection heuristics — that class has failed matched
controls six times; code geometry as a standalone lever, which §1.3 closes; retuning
anything in the closed list.

**Real byte corpora stay deferred**, now behind architecture rather than behind capacity
or task validity. If G1 or G2 fails, the right next move is tinyshakespeare with the
better single layer and honest cost accounting, rather than another architecture.

---

## 9. Provenance

- `elm_v39_science_results` (ChatGPT) — 8 seeds, 48 memory branches, `REPORT.json`
  levels/contrasts/gates, `STATE_COSTS.json` checkpoint storage, `PROTOCOL.md` Q0–Q13,
  gate arithmetic read from `elm_v39_earned_consolidation.py:975`.
- `v39_results` (Claude) — 416 trajectories, 26 cells × 16 seeds, source SHA256
  `0f9577b4…a78d0`. `TERMINAL.csv`, `GATES_CORRECTED.json`, `REPORT_CORRECTED.md`.
- Stationarity measurement: same-context active-set self-overlap at t = 5k/15k/29k,
  `lag` 1.0000 versus `phase` 0.62–0.74.
- `v39_experiment_recommendation.md`, `v38_results/REPORT.json` (E1/E2),
  `V35_design.md`, BANC v888 PN→KC overlap query.

Corrections recorded here: the V39 design doc's claim that the fly gate was unreachable
(§1.3); V36's separation conclusion, which was confounded by code drift (§1.3); and
V39's G2 arm-selection bug and G3 arm choice (§1.4).
