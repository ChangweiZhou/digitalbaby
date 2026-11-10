# E19 predictions: a two-bank learner whose credit is exact and forward-only (10 November 2026, 17:35 UTC, written before Stage A)

## Source and branch

- **Brief:** `OPUS_TASK_E19_FORWARD_ONLY_TWO_BANK_20261110.md` (commit ab0ae7ac). The whole job, Stages A and B, is run by this cloud session, which also runs E16-M4 at lower CPU priority on the same container.
- **Branch:** `claude/vibrant-gauss-tops5f`, fast-forwarded to `codex/minifly-consolidated-20261023` at ab0ae7ac.
- **New files only:** `arch19.py` and `exp19_kbank.py`. No existing file is edited.

## Run identity

- **Machine:** `Linux vm 6.18.44-fc-v338 #1 SMP x86_64`, 4 cores.
- **Software:** Python 3.13.18, PyTorch 2.15.0 (CPU), one thread per process.

## Pre-launch checks

- **No birth files exist.** As of 17:35 UTC there is no `exp45/e19_dev_kbank2_*` file.
- **Acceptance tests 1 and 2 passed before any birth.** Results are in `exp45/E19_CHECK.json`.

| Test | What it compares | Data | Maximum relative error |
|---|---|---|---|
| Test 1 | The forward-only rule with exact feedback against autograd's backprop through time, on every parameter | Three random initialisations, development batches of 64 with distractors up to twelve letters | 5.1e-16 (float64), 2.6e-7 (float32) |
| Test 2 | A byte-at-a-time streaming implementation, holding only the two states and the traces, dense over all 258 bytes, against the batched implementation | The first four items of each batch, float64; for the exact rule with deep traces, and for weight signs with and without deep traces | 7.0e-16 |

- **Relative error** is max |rule − reference| / max |reference|, per parameter tensor.
- **Parameter count:** 1,182,977, against 1,125,265 for the existing two-bank network.

## What was built

The network and rule are as in the brief, derived in `exp19_kbank.py`'s header. The brief's sketch was correct as written.

- **Traces are stored per sample.** Each sample has a slot for each byte it contains, and the slots are merged into table columns at the end of the item. This is equivalent to traces per table entry, because a byte absent from a sample has a trace of exactly zero.
- **What a true online version would store** for a first-bank table entry A[j, b], per byte b seen since the last weight update:
  - one first-order trace per byte and lower rate group (5 × 258 at most);
  - one deep trace per entry and upper rate (5 × 512 × 258 at most);
  - for W2, five traces per lower unit and one per upper rate for the bias.
- **Tables.** A and E are initialised as `nn.Linear(258, ·)` weights (uniform ±1/√258) and sit in a no-decay group.
- **Weight-sign feedback** is sign(W) · mean|W| in place of W3, Wm1, Wm2, and W2 inside K. Nothing else changes.
- **Silencing:** at scoring, both banks' contributions to the readout are set to zero.

## Dry runs

Seed 73129, 512 updates; files are in `exp45/archive_e19_dry/`. The three ran together in 140 s of wall time.

| Arm | CPU per update | Peak RSS |
|---|---:|---:|
| Backprop | 116 ms | 0.5 GB |
| R1 | 239 ms | 0.5 GB |
| R2 | 145 ms | 0.5 GB |

## Runs

Seeds 73129, 93683, 116089. Standard development: 24,576 updates, batches of 64, sealed panels at k = 0, 4, 8, 12, 20 every 4,096 updates.

- **Stage A:** `python3 exp19_kbank.py dev bptt <seed>`. Backprop through time (autograd), AdamW lr 0.001, decay 0.3, tables without decay.
- **Gate:** Stage B runs only if the Stage A hard mean (mean of add89 and sub89 at k = 12, final checkpoint) is at or above 60 in at least two of the three seeds.
- **Stage B:**
  - `dev r1 <seed>`: the forward-only rule with weight signs and deep traces;
  - `dev r2 <seed>`: the same with the deep traces removed.
  - Both use decay 0.25.
- **After Stage A:** `python3 exp19_kbank.py check_births` repeats test 1 on the births and makes test 3 (alignment).

## Predictions (copied from the brief; not to be changed)

- **P19.1 (acceptance)** Tests 1 and 2 pass.
- **P19.2 (the gate)** Stage A's hard mean is at or above 60 in at least two of the three seeds, and below the existing two-bank reference in all three. Held at about 50%: the second bank now accumulates at every step, distractor letters included, with no gate to refuse a write.
- **P19.3 (the rule against its own ceiling)** If Stage B runs: R1's hard mean is at or above 0.85 of the Stage A birth of the same seed in at least two of the three seeds.
- **P19.4 (the deep traces matter)** R1's hard mean is at least 10 points above R2's in at least two of the three seeds.
- **P19.5 (cost)** Reported, not predicted: CPU-milliseconds per update for backprop, R1 and R2; traces held per synapse and per sample; peak memory.

**References on file (owner's machine; compare levels, not digits).** Hard means at k = 12, final checkpoint:
- the existing two-bank network by backprop: 91.2 / 94.5 / 90.3;
- one bank alone: 21.7 / 19.3 / 15.2.

## Forecast, from the dry runs

The forecast is in CPU-seconds; scoring adds about 120 CPU-s per birth.

| Birth | Forecast per birth |
|---|---:|
| Backprop | about 3,000 |
| R1 | about 6,000 |
| R2 | about 3,700 |

- **Stage A:** three births at normal priority beside E16-M4's niced lives, about 1 hour of wall time.
- **Stage B:** six births, about 2.5 hours, set by R1.
