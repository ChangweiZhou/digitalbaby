# Can the fly's own route keep byte order and learn from it? — design (fixed before any run)

**Date:** 25 September 2026. Its SHA-256 is recorded in `RESULTS.json`.

**Goal**, as the owner set it:
- **(2)** the fly's state keeps the order of recent bytes;
- **(3)** its plasticity can use that order to predict.

Exact per-byte identity through dedicated cells is **not** a goal, because one cell per byte does not scale. V9's ORDER pass does not count for the same reason: it comes from engineered one-cell-per-pair (hashed) cells.

## Task and gate: V9's ORDER, reused unchanged

The stream functions, scorer and baselines are imported from `BYTE_CORE_V9/experiment.py` (hash recorded): `order_stream`, `order_swap_stream`, `ngram_evaluate`, `model_score` and `coverage`.

- **Stream:** a continuous stream of 4-byte records `[0xFF, 'A'+a, 'A'+b, 'a'+((a+3b) mod 16)]`, with a, b ∈ 0…15.
  - Given either byte alone, the target is uniform, so only the ordered pair predicts it.
  - Byte i arrives at i × 30/14 s.
- **Training:** 256 passes over all 256 ordered pairs (262,144 bytes). Learning happens online at every byte.
- **Test:** a fresh stream of 16 passes (4,096 targets).
  - All weights are frozen; transient state is reset once at the start and then evolves causally.
  - Each target is scored before it arrives.
- **Worlds:** 201, 202, 203. Stream seeds are 1,000,000 + 100·world + 1 for training and + 2 for test, disjoint from V9's worlds 0, 11, 12 and 101–103.
- **Gate, per arm and world:** V9's `acceptance_checks` rule for ORDER, restated:
  - both training and test cover every pair exactly (256 and 16 times);
  - macro accuracy ≥ 0.65 and target bits ≤ 2.0;
  - target bits ≤ the target-slot last-byte Markov baseline (`oracle_last1`) − 1.0;
  - lower bits and higher macro accuracy than the arm's plasticity-off twin;
  - swapping a and b in the frozen test stream costs ≥ 0.1 bits on the 224 pairs whose target changes under the swap.

## Arms (the same streams in every world)

| Arm | Route |
|---|---|
| **FLY_FE0** | FE0 → 88 PN → native FULL_151 PN→KC projection with its native sparsity → **all 5,177 KCs** → 256 plastic output synapses |
| **FLY_TRACE88** | The same route, with the v6 TRACE88 front end (multi-timescale traces; `v8_trial.Trace88`, tested equal to the v6 encoder) |
| FLY_FE0_frozen, FLY_TRACE88_frozen | Identical twins with plasticity off |
| **V9_POSITIVE** | V9's `F151ByteBrain(temporal=True, order_via_native=True)`, the candidate that passed V9 DEV. It only shows the task and gate can be passed |

The fly-route arms have no 128-bin pooling, no last-byte or pair cells, and no engineered ordered residual.

**Output rule:** V9's local rule, unchanged.
- The KC input vector is the active-KC indicator scaled to unit L2 norm, like V9's pooled vector.
- p = softmax(bias + xW).
- W ← W − 0.03·x(p − onehot(b))ᵀ, and bias ← bias − 0.001·(p − onehot(b)).
- The learning rate is V9's and **is not tuned**.
- The FULL_151 value memory is not involved.

## Decision

- **Per arm:** PASS if the gate holds in all 3 worlds.
- **Reading verdict:** *the fly's own route keeps and uses byte order* if **FLY_FE0 or FLY_TRACE88** passes. Each front end is reported separately.
- **V9_POSITIVE:** must pass, otherwise the task or harness is suspect.

## Diagnostic, run only for a fly-route arm that fails in any world

This locates where order is lost. It runs on world 201, using that arm's front end. States are recorded at the target slot, just before the target arrives:
- the 88-PN vector;
- the 5,177-KC code.

**Decoders:** offline multinomial logistic regressions, fitted by L-BFGS with L2 penalty λ = 1e-4 on the mean log-loss. They are trained on the last 16,384 training target slots and tested on the 4,096 test slots.

| Level | Decoded |
|---|---|
| PN | **a** (first symbol) and **b** (second symbol), 16 classes each. Knowing which symbol is first is the order information |
| KC | a, b, and the target y |

y is not decoded from PN: it depends on the interaction of a and b, which a linear PN decoder cannot represent even when order is present.

**Controls:**
- NC: y with shuffled labels at KC must be near 1/16.
- PC: a and b from V9's engineered pair and last-byte cells should be near 1.

**Reading the diagnostic:**
- a and b at chance from PN → the input organ loses order.
- a and b decodable from PN but not KC → the KC coding loses it.
- y decodable from KC but the online arm fails → the learning rule or rate is the problem, not reading.

## Checks

- **From V9's scorer:**
  - every target is scored before the byte is seen, and `predict` equals the score used;
  - parameters are unchanged during test scoring;
  - coverage is exact.
- **Fly-route inputs:** the fly-route predictor's inputs are exactly the 5,177 native KCs.
- **Frozen twins:** their parameters stay at zero.

## Not in scope

- memory efficiency (ignored, as instructed);
- natural text;
- the identity task;
- teaching and value memory.
