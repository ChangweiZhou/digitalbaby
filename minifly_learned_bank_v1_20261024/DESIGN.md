# LEARNED-BANK V1: can a locally trained feature layer make the Full151 memory generalise?

24 October 2026. Design document, written before any code for this experiment. It will be refined in three cycles (refine, audit, test run, exercise), then frozen and run. Settings marked **(provisional)** may change during the cycles; every change is recorded in `cycles/`. Nothing is frozen until the cycle-3 lock.

## 1. Question

Condition 1 from the emergence discussion: an associative memory with a fixed random expansion and a learned readout can only interpolate between stored examples (kernel regime). This experiment asks one narrow question:

> If a feature layer upstream of a native Full151 bank learns with a local three-factor rule, does the memory's kernel change in a task-aligned way, and does the bank answer never-taught problems better than the same architecture with the layer frozen?

It is a first, deliberately small step (not "all in"). No recurrence, no working memory, no replay, no multi-byte answers and no stopping are tested here.

## 2. What is kept, added and excluded

Kept from Full151 / R_center:
* Canonical newborn Full151 native stores (`minifly_a_v3`, unchanged native equations: fast/slow/adaptation, native allocation, signed teaching via the store's own `teach_signed`).
* The R_center two-bank layout: a fixed **shared bank** plus a second bank whose address comes from a different stream. Here the second stream is learned instead of R_center's hand-chosen "last four non-space bytes".
* Deterministic order-keeping receptor front end FE-PERM (the v0 front end used in the G1 memory gate: fixed permutation of the 88 receptor channels, then add the byte's receptor pattern; decay 0.95 per byte step), and KC adaptation reset at record start.

Added:
* **Learned feature layer** feeding a **learned bank**: receptor trace → hidden layer (learned) → 88-channel virtual receptor vector → canonical PN→KC wiring → native stores.
* **Opponent output coding**: each of the 8 answer bits uses two identically born compartments (bit-on, bit-off); the bit's margin is their difference, so the common native offset cancels without a reference read or a codebook.

Excluded: attention, Transformers, ELM, backpropagation through the native store or through time, supplied numeric features or arithmetic roles, codebooks, noise in the answer path.

## 3. Architecture

Per record (prompt bytes then one answer byte), brain time advances 30/14 s per event, as in v0.

**Front end.** FE-PERM trace `p` (88-d), normalised with the native `_norm`. Read once at the answer event.

**Shared bank S.** `x_S = topk_side(B^T p[pn])` with v0's one-sided KC homeostasis. 16 compartments (8 bits × on/off).

**Learned layer.** `a = W1 p̂ + b1`, `h = relu(a)`, H = 128 hidden units **(provisional)**. Virtual receptor `v = W2 h` with W2 fixed random. `x_L = topk_side(B^T v[pn])`, always exactly the native per-side active count (largest drive wins; no positivity requirement), stable tie-break by index. 16 compartments.

**Readout.** Bank margin for bit j: `m_j = z_{j,on} − z_{j,off}` (native value reads on clones; reads never change state). Emitted bit = `m_j > 0`. The system answer uses the summed margins of S and L. Each bank's own answer is also recorded.

**Native teaching (per bank, independent).** `e_j = (tanh(m_j/T) − y_j)/2`, T = 2, y_j = ±1 the observed answer bit. Compartment on gets `s = a(n)·e_j`, off gets `−a(n)·e_j` through native `teach_signed(b=0, s)`. Each bank is taught on its own margin so the learned bank keeps receiving error until it solves the task itself.

**Learned-layer rule (three-factor, local).** At each answer teaching event:

`ΔW1 = −η(n) [ (G e_L) ⊙ 1(a > 0) ] p̂ᵀ − η(n) λ W1`, `Δb1 = −η(n) (G e_L) ⊙ 1(a > 0)`

Pre-synaptic activity `p̂`, post-synaptic gate `1(a>0)`, and a broadcast third factor `G e_L`, where `e_L` is the learned bank's own 8-channel native teaching residual and G (H×8) is fixed random feedback. This is direct feedback alignment: no derivative is taken through the KC top-k or the native store. λ is a small weight decay (synaptic-scaling-like) **(provisional)**.

**Warmup.** Native strength `a(n) = min(1, 1/16 + (15/16)·n/1024)` over completed records; learned-layer rate `η(n) = η0·min(1, n/1024)` **(provisional η0)**. The native ceiling is the v0 level (1.0), not Codex's 1/16: earlier analysis showed the 1/16 ceiling keeps the native rule in its near-additive regime. Both arms in every comparison share the same schedule.

**Determinism.** No stochastic units. The same prompt always produces the same code at a given moment.

## 4. Tasks

Prompts `a+b=` with single digits a, b ∈ 0…9 (raw bytes). One answer byte.

* **T-MOD (target):** answer = last digit of a+b (the column rule of addition without the carry). Requires an interaction between a and b; a fixed random kernel should not generalise.
* **T-COPY (control):** answer = a. A first-order function; a fixed kernel with position-sensitive input should generalise. Checks the measurement chain.

**Splits.** The 55 unordered pairs {a,b} are partitioned per world: 15 held-out orbits (both orientations never taught), 40 training orbits (both orientations taught). Every digit must appear in at least four training orbits. Held-out answers never reach the learner.

**Schedule.** Interleaved, reshuffled each pass over the training ordered pairs; P passes **(provisional, about 60-80)**; one simulated hour of rest before the final probe.

## 5. Arms

| Arm | Learned layer | Role |
|---|---|---|
| BASE | none (shared bank only) | Full151-style baseline |
| FROZEN | present, W1 fixed at birth | random-feature control |
| LEARN | present, W1 trained by the three-factor rule | candidate |
| SHUF | as LEARN, training answers deranged across training pairs (fixed) | leakage/structure control |
| ORACLE | learned bank receives a fixed random 88-d code per answer class instead of `v` | plumbing positive control only (evaluator-supplied; never a candidate) |

All arms share birth, front end, data order and schedules within a world. T-COPY runs BASE, FROZEN, LEARN.

## 6. Measurements

* Held-out and taught exact accuracy (system answer and per-bank answer) at birth and every 10 passes, read-only.
* **Kernel meter:** Jaccard matrix of learned-bank KC codes over all 100 prompts; centred kernel-target alignment with the answer classes, for all prompts and for held-out prompts. Also mean same-answer minus different-answer Jaccard.
* **Input-identity gate** at birth (V6 lesson): distinct prompts must give distinct codes in every bank; same prompt read twice must give identical codes.
* **Drift:** Jaccard between successive checkpoints of each training prompt's learned-bank code; taught accuracy over time (old facts must not be lost to drift).
* Native write magnitudes, hidden-unit activity, weight norms, CPU per record.

## 7. Predictions and decision rule (to be frozen at the cycle-3 lock)

* T-COPY: all three arms well above chance on held-out (else stop: measurement chain broken).
* ORACLE on T-MOD: held-out near 100% (else stop: native readout cannot express the answer even with ideal features).
* BASE and FROZEN on T-MOD: held-out near chance (10%).
* **Condition 1 supported in this toy** if, in every main world, LEARN's learned-bank held-out exact exceeds FROZEN and SHUF, and LEARN's held-out kernel-target alignment rises above its birth value while FROZEN's does not.
* If the kernel does not move: credit signal failure. If the kernel moves but held-out does not improve: features change in an unhelpful direction (points to conditions 2-3). Either way no tuning on the main worlds.

## 8. Worlds, tuning and budget

* Development worlds (seeds D1-D3) for the cycles' test runs and any tuning of H, η0, λ, P.
* Four fresh main worlds (seeds M1-M4) touched only by the frozen protocol.
* Cost estimate from a timing probe: about 0.1 s per record for 16 native compartments; this design uses 32, so roughly 0.2 s per record. A main history of about 6,000 records is about 20 minutes; the main T-MOD roster (5 arms × 4 worlds) is about 7 CPU-hours, run on 4 cores in parallel. The cycles measure the real cost before the lock.

## 9. Three cycles before the run

Each cycle: refine the design or code, audit it (code read-through, unit checks against the native equations, claim check), test run on development worlds, then exercise the full pipeline end to end at small scale. Cycle notes and results go in `cycles/`.

1. **Cycle 1:** implement banks, opponent readout, three-factor rule, meters; unit checks (native parity of a bank with homeostasis off, opponent cancellation at birth, no-write invariance of reads, identity gate); tiny test run.
2. **Cycle 2:** development-world test runs of all arms at reduced passes; tune provisional settings on development worlds only; measure cost.
3. **Cycle 3:** fix whatever cycles 1-2 found, run the exact main protocol on one development world as a dress rehearsal, then lock settings and seeds.

After the lock: run the main worlds, write the report with all arms and all worlds, push results.

## 10. Out of scope

Multi-digit numbers, carries across columns, stopping, recurrence, working memory, replay and sleep, worked examples. These follow only if condition 1 is supported here.
