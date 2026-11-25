# lt7 status (25 Nov 2026)

`lt7` is a local tissue built on the template of the cerebral cortex. The header of `lt7.c` maps each component to its biological source and to the code.

## History

- `lt5` was the first design and is kept for the record.
- `lt6` was based on the cerebellum and is abandoned: the goal is the cerebrum.

## Changes since the first push of lt7

| Change | Why | Evidence (in `investigations/`) |
|---|---|---|
| Per-part metabolic budgets | Connection counts come from competition for each cell's resources; there are no count caps | cost levels off after the network develops |
| No clock-based synapse decay (`lam` = 0); synapses are withdrawn only by competition; output budget 5, so the budget binds | A decay driven by time erased useful synapses that were rarely refreshed, which caused late-life degradation | in `late_probe_results.jsonl`, freezing learning did not stop the decline but removing the decay did; early learning is faster (3.10 vs 3.47 at 32k bytes) |
| `scale` means a larger sheet at the same density | In the brain, area grows while density and connection reach stay fixed. Packing cells more densely diluted each output cell's input | the scaling grid's size term went from +0.117 to +0.033 per doubling |
| Operation counter `lt7_visits` | Lets cost be measured as work instead of wall time | operations per byte grow with exponent 1.00 in size; wall time grows with exponent 1.37 because of cache misses |

## Battery v2 (`../battery2/`)

| Item | Result |
|---|---|
| Scoring check (`life.py` oracle) | passes |
| net13 positive control on the separated grid | size term −0.027 ± 0.009, data term −0.160 |
| lt7 baseline on the English + addition life (`life_lt7_s1.json`; earlier lt7 version, with fast decay) | no addition learned: real and control give the same answer loss. English recovers after interference |
| lt7 scaling grid (`grid_lt7.jsonl`, same-density scale; earlier version, with decay 2e-5) | size term +0.033 ± 0.009, data term −0.197 |

## Checks for the current default

| Check | Result |
|---|---|
| test purity | pass |
| requirements audit | pass |
| 7-byte cycle | pass |
| learned connections reach the output | pass |
| cost (operations) | operation count is linear in size; the wall-time check still reports a fail |
| memory 41 bytes back | **fail** |
| relay benefit | **fail** |

## Open problems

1. Under bounded memory, new text displaces older knowledge. Loss drifts up after about 128k bytes. This is the continual-learning question itself. One attempt, protecting consolidated synapses from competition, made it worse.
2. A bigger network is not yet better: the size term is still positive.
3. No addition is learned, the relay gives no benefit, and nothing is carried 40 or more bytes back.

The battery-v2 lt7 numbers above come from the earlier version and should be re-measured on the current default.
