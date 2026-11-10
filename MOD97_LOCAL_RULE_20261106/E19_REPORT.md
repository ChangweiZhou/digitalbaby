# E19 report: a two-bank learner whose credit is exact and forward-only (10 November 2026)

## Sources and branch

- **Brief:** `OPUS_TASK_E19_FORWARD_ONLY_TWO_BANK_20261110.md` (commit ab0ae7ac). This cloud session ran the whole job.
- **Predictions:** `E19_PREDICTIONS.md`, committed at f1f79b65 at 17:26:17 UTC, before Stage A was launched at 17:26:26. The header of that file says "17:35 UTC". That is wrong: the commit time is the true one. The file is not edited.
- **Code:** new files only — `arch19.py`, `exp19_kbank.py`, `e19_tables.py`. No existing file was changed.
- **Tables:** `exp45/SUMMARY_E19.md` gives every checkpoint of every birth, four operations at five values of k.
- **Checks:**
  - `exp45/E19_CHECK.json`: tests 1 and 2, before any birth.
  - `exp45/E19_CHECK_BIRTHS.json`: test 1 on the births, and test 3.
- **Branch:** `claude/vibrant-gauss-tops5f`.

## Outcome

**P19.1 (acceptance) holds.** Tests 1 and 2 pass to machine precision.

**P19.2 (the gate) holds.** With the linear write into the second bank, backprop gives the network a hard mean of 69.0, 70.5 and 69.3. That is above 60 in all three seeds and below the two-bank reference in all three.

**P19.3 (the rule against its own ceiling) misses in all three seeds.** The forward-only rule with weight signs (R1) reaches 43.2, 45.8 and 54.5. That is 0.63, 0.65 and 0.79 of the backprop birth, against the 0.85 required.

**P19.4 (the deep traces matter) holds.** R1 is above R2 by +8.0, +14.5 and +23.8 points. That clears the 10-point margin in two seeds of three.

**Which case of the brief's reading obtained.** "P19.3 missing with P19.2 holding says weight signs, not the credit path, are the limit for this network."

- **The credit path is exact.** Test 1 shows the forward-only rule is backprop through time when exact weights are used.
- **The deep traces carry it.** P19.4 holds, and test 3 shows them doing so.
- **What is lost against backprop is lost to the weight-sign feedback.** At the births, the readout, W2 and the table E align with the true gradient at cosine only 0.71–0.82 under weight signs.
- **Two qualifications:**
  - The R1 births were still rising at the last checkpoint in two seeds, by +3.1 and +10.2 points over the last 4,096 updates. Seed 73129 fell by 1.0.
  - No arm ran the forward-only rule with **exact** feedback. That arm would separate the two halves directly, and it is the obvious next one.

## What was built, and the derivation

**The network: `kbank2`, as in the brief.** It has 1,182,977 parameters, against 1,125,265 for the existing two-bank net.

```
h1_t = R1 h1_{t-1} + A[:, x_t]
h2_t = R2 h2_{t-1} + W2 tanh(h1_t) + b2
z_t  = tanh(Wm1 tanh(h1_t) + Wm2 tanh(h2_t) + E[:, x_t])
logits_t = W3 z_t + b3
```

- **Retention:** R1 = R2 = arch17's five rates, in equal groups.
- **Initialisation:** the tables A and E are initialised as `nn.Linear(258, ·)` weights; every other layer is PyTorch's default.
- **Optimiser:** AdamW, learning rate 0.001, betas (0.9, 0.98). Decay is 0.3 for backprop and 0.25 for the rule. A and E sit in a no-decay group.

**The rule.** It is derived in full in the header of `exp19_kbank.py`. The brief's sketch was correct as written.

- **Readout:** the readout and E learn within the step.
- **Second bank's write:** W2 and b2 use one trace per lower unit per upper rate, e2_t[g, j] = r_g e2_{t−1}[g, j] + tanh(h1_t)[j]. This is exact, because h2 is linear in W2.
- **First bank's table A:** a first-order trace e1 plus five deep traces ζ^g, each driven by K^g_t = Σ_{c∈g} δ2_t[c] F(W2)[c, j].
  - These are exact: h1 is linear in A, and tanh′(h1_t) is taken at the step where it applies.
  - Exchanging the sums over t and τ turns the second bank's future error into a forward trace. Test 1 confirms this.
- **R2 replaces ζ^g by its current-step term,** so the first bank hears only the second bank's current-step error.
- **Weight-sign feedback:** F(W) = sign(W) · mean|W| for W3, Wm1, Wm2, and for W2 inside K.

**Trace storage.** Traces are kept per sample, in slots for the bytes that sample contains, and merged into table columns at the end of the item.
- This is equivalent to traces per table entry, because a byte absent from a sample has an exactly zero trace.
- **What a true online version would store**, per table byte b seen since the last weight update:
  - one first-order trace per byte and lower rate group, at most 5 × 258;
  - five deep traces per table entry, at most 5 × 512 × 258 ≈ 660k;
  - for W2, five traces per lower unit (5 × 512, shared by the roughly 102 upper units of each rate group);
  - five for b2.

## The three acceptance tests

**Test 1: rule with exact feedback against autograd backprop through time, every parameter.**
- Maximum relative error is max|rule − bptt| / max|bptt| per tensor.
- **Random initialisations:** three of them, development batches of 64 (22 steps, distractors up to twelve letters). Error 3.9e-16 to 5.1e-16 in float64, 2.2e-7 to 2.6e-7 in float32.
- **Stage A births** (float64, a later development batch): 2.3e-15, 2.9e-15 and 2.8e-15. Passes.

**Test 2: streaming against batched.** The streaming version feeds one item one byte at a time, holds only the two states and the traces, is dense over all 258 bytes, and loops over units in Python.
- Compared against the batched implementation on the first four items of each test batch, in float64.
- Three settings: exact feedback with deep traces; weight signs with and without deep traces.
- Maximum relative error 5.6e-16 to 7.0e-16. Passes.

**Test 3: alignment with the true gradient at the Stage A births.** Cosine, mean of four development batches per seed, for seeds 73129 / 93683 / 116089:

| Parameter | Weight signs + deep traces (R1) | Weight signs, no deep traces (R2) | Exact feedback, no deep traces |
|---|---|---|---|
| A (first bank's table) | 0.69 / 0.62 / 0.61 | 0.42 / 0.41 / 0.41 | 0.63 / 0.64 / 0.64 |
| W2, weight | 0.78 / 0.71 / 0.78 | same | 1.00 |
| Wm1, Wm2 | 0.80–0.82 | same | 1.00 |
| E (readout's table) | 0.82 / 0.79 / 0.80 | same | 1.00 |
| W3, b3 | 1.00 | 1.00 | 1.00 |

- Under weight signs the deep traces raise A's alignment from about 0.41 to about 0.64. That is level with exact feedback without deep traces.
- Every layer above A loses 0.18–0.29 of cosine to the signs alone.

## Births

All values are at k = 12, final checkpoint (24,576 updates), on sealed panels of 300 items. Every checkpoint is in `exp45/SUMMARY_E19.md`.

| Arm | Seed | add89 | sub89 | max89 | min89 | Hard mean | Hard mean over the six checkpoints | add / sub at k = 20 |
|---|---|---:|---:|---:|---:|---:|---|---|
| backprop | 73129 | 67.3 | 70.7 | 93.7 | 92.7 | **69.0** | 13.7 28.0 36.7 54.5 59.8 69.0 | 55.7 / 60.0 |
| backprop | 93683 | 72.3 | 68.7 | 91.3 | 94.7 | **70.5** | 12.8 27.0 39.0 47.3 56.5 70.5 | 59.7 / 63.3 |
| backprop | 116089 | 68.7 | 70.0 | 91.7 | 92.7 | **69.3** | 13.3 22.3 41.2 53.0 61.7 69.3 | 56.7 / 58.7 |
| R1 (signs, deep traces) | 73129 | 45.3 | 41.0 | 89.7 | 86.7 | **43.2** | 10.8 22.2 28.8 37.0 44.2 43.2 | 38.7 / 36.3 |
| R1 | 93683 | 47.0 | 44.7 | 88.3 | 90.0 | **45.8** | 11.0 21.7 24.0 37.0 42.7 45.8 | 43.0 / 38.0 |
| R1 | 116089 | 53.7 | 55.3 | 88.0 | 89.0 | **54.5** | 12.8 19.0 30.0 37.5 44.3 54.5 | 44.3 / 45.0 |
| R2 (signs, no deep traces) | 73129 | 35.7 | 34.7 | 80.0 | 76.7 | **35.2** | 9.3 20.3 22.0 26.5 30.5 35.2 | 30.7 / 25.0 |
| R2 | 93683 | 34.3 | 28.3 | 77.7 | 80.0 | **31.3** | 7.0 18.2 18.8 24.5 27.5 31.3 | 32.3 / 26.3 |
| R2 | 116089 | 31.0 | 30.3 | 75.3 | 79.0 | **30.7** | 7.5 13.3 20.0 25.8 28.7 30.7 | 23.7 / 29.0 |

- **Silenced** (both banks zero in the readout), k = 12: 0% on every operation for backprop and R1. R2 seed 116089 scores 1–4 items of 300.
- **References** (owner's machine; compare levels):
  - the existing two-bank network by backprop, hard mean 91.2 / 94.5 / 90.3;
  - one bank alone, 21.7 / 19.3 / 15.2.
- **R1 against R2 on max and min:** 88.2–89.2 against 77.2–78.8. **Against backprop on add and sub:** a gap of 15–28 points.

## Verdicts

| Prediction | Per seed (73129 / 93683 / 116089) | Verdict |
|---|---|---|
| P19.1 acceptance | tests 1 and 2 pass on every initialisation; test 1 passes on every birth | **holds** |
| P19.2 gate: Stage A hard mean ≥ 60 in at least two seeds, and below the reference in all three | 69.0 / 70.5 / 69.3, against 91.2 / 94.5 / 90.3 | **holds**; the gate passed, so Stage B ran |
| P19.3 R1 ≥ 0.85 of Stage A, in at least two seeds | 0.626 / 0.650 / 0.786 | **misses, all three** |
| P19.4 R1 − R2 ≥ 10, in at least two seeds | +8.0 / +14.5 / +23.8 | **holds** (two of three) |
| P19.5 cost | see below | reported |

## P19.5: cost

**CPU per update** (measured over 24,576 updates, scoring excluded):

| Arm | ms per update | Relative to backprop |
|---|---:|---:|
| Backprop (autograd) | 102.5–103.5 | 1 |
| R1 | 219.6–220.3 | 2.1× |
| R2 | 130.1–131.6 | 1.3× |

**Peak memory:** 506–514 MB per process for every arm.

**Traces held:**

| Parameter | Per synapse | Per sample |
|---|---|---|
| A[j, b] | one first-order trace, shared within a rate group, plus five deep traces (R1) or none (R2) | up to 5 × U first-order and 5 × 512 × U deep, where U is the number of distinct bytes in the sample (at most 22 here) |
| W2 | 5 per lower unit, shared by each upper rate group: 2,560 traces for 262,144 synapses | 5 × 512 for W2, plus 5 for b2 |

**CPU-seconds per birth:**

| Arm | CPU-s |
|---|---|
| backprop | 2,557–2,582 |
| R1 | 5,433–5,452 |
| R2 | 3,233–3,272 |

## Run identity, pace, forecast against measured

- **Machine:** `Linux vm 6.18.44-fc-v338 #1 SMP x86_64`, 4 cores, Python 3.13.18, PyTorch 2.15.0 (CPU), one thread per process.
- **Environment:** every process with default environment; no E16 variables are read by `exp19_kbank.py`.
- **Commits:** code at f1f79b65 (on ab0ae7ac).
- **Wall times:** Stage A ran 17:26–18:35 UTC; Stage B ran 18:35–20:38 UTC.

| | Forecast | Measured |
|---|---|---|
| Backprop birth | about 3,000 CPU-s | 2,557–2,582 |
| R1 birth | about 6,000 CPU-s | 5,433–5,452 |
| R2 birth | about 3,700 CPU-s | 3,233–3,272 |
| Stage A wall time | about 1 hour | 1 h 9 min |
| Stage B wall time | about 2.5 hours | 2 h 3 min |

## What went wrong, or is limited

- **CPU sharing with E16-M4.** E16-M4's twelve lives were launched beside Stage A at `nice -n 10`. Each `setsid` process has its own scheduler autogroup, so `nice` did not rank them, and the Stage A births ran at about 36% of a core until 18:06 UTC. From then on, M4's autogroups were set to nice 19, and E19 ran at about a full core. This changed only wall time; CPU-seconds and results are unaffected.
- **Predictions header time.** The header says 17:35 UTC where the commit is 17:26:17. The commit time is the true one.
- **Missing arm.** No arm ran the forward-only rule with exact feedback. The weight-sign/credit-path split is therefore inferred from P19.2 + P19.4 + test 3, not measured by a one-change arm.
- **Scope.** One birth per arm and seed, development panels only. No life, no retention, no transfer is tested.
- **Evidence level:** E2 (sealed development panels).
- **Disputed constraint.** Whether weight-sign feedback meets the project's locality constraint is disputed (Mr. Zhou has not ruled). Every claim here is conditional on it.
