# E32 predictions and the nine registered strengths (main worker, 14 November 2026; pushed before any full birth)

**Brief:** `OPUS_TASK_E32_EVERY_BYTE_BIRTH_SWITCHES_20261114.md`, revision 4 (a1583c23).

**Runner:** `exp32_births.py`, main worker.

**Sources:** the default branch at a1583c23, merged into the worker's branch. `exp16_integrated.py` and every other source equal what E28 recorded; the hashes are in `E32_LAMBDA_MAIN.json`.

## The nine strengths (check 7, P4; computed without training)

**Definition (brief, arms P4):**
- `lam = ratio × (mean task norm) / (mean term norm at lam = 1)`, at the weights of E28 S-all row 4,096 (`e28_dev_S-all_<seed>_row4096.pt`, hash checked).
- The means are over batches 4,097 to 4,224 of the birth stream. The stream's digest after 4,096 batches equals E28's row-4,096 digest, which is asserted.
- Each norm is one joint norm over rows 256–511 of `weight_ih_l0` and `weight_hh_l0` and the same entries of both biases.
- The task norm is taken as the sweep computes it.

Full record: `E32_LAMBDA_MAIN.json`.

| Seed | Mean task norm | Mean term norm at lam 1 | P4a (0.25) | P4b (0.05) | P4c (0.01) |
|---|---|---|---|---|---|
| 73129 | 0.0747757616097607 | 0.9066939671880865 | 0.02061769580359661 | 0.004123539160719322 | 0.0008247078321438645 |
| 93683 | 0.05825484093858196 | 0.8485100666731051 | 0.017163862641898706 | 0.003432772528379741 | 0.0006865545056759483 |
| 116089 | 0.057865426007193355 | 0.8224646759180856 | 0.017589030781960457 | 0.0035178061563920916 | 0.0007035612312784183 |

Each P4 birth recomputes `lam` at its own weights after update 4,096 (which must have E28's row-4,096 hash) and asserts it equals the value above.

**Batch 4,097 alone, beside the reviewer's numpy reader (brief, check 7):**

| Seed | Task norm, main | Reviewer | Relative difference | Term norm at lam 1, main | Reviewer | Relative difference |
|---|---|---|---|---|---|---|
| 73129 | 0.036740466 | 0.036740 | 1.3e-5 | 0.890970115 | 0.890970 | 1.3e-7 |
| 93683 | 0.077113332 | 0.077113 | 4.3e-6 | 0.836491392 | 0.836491 | 4.7e-7 |
| 116089 | 0.093319743 | 0.093320 | 2.8e-6 | 0.849241398 | 0.849241 | 4.7e-7 |

All agree well inside the expected 1e-3. The reviewer's figures are given to six significant figures.

## Predictions (the coordinator's, from the brief, copied unchanged)

| | Prediction | Chance |
|---|---|---|
| P32.1 | H moves accuracy at 40 letters | 45% |
| P32.2 | G moves accuracy at 40 letters | 30% |
| P32.3 | K moves accuracy at 40 letters | 30% |
| P32.4 | K is within 15 points of H at 40 letters in at least two of three seeds | 45% |
| P32.5 | H has keepers at 12 letters | 60% |
| P32.6a | P4a has keepers at 12 letters | 55% |
| P32.6b | the same for P4b | 40% |
| P32.6c | the same for P4c | 20% |
| P32.7 | at least one P4 strength moves accuracy at 20 letters | 30% |
| P32.8 | at least one P4 strength moves accuracy at 40 letters | 20% |
| P32.9 | no arm does harm | 65% |
| P32.10 | the rerun ceiling keeps at least 50 keepers at 12 letters at every saved row from 8,192 on, in all three seeds | 60% |

The endpoints, marks and readings are the brief's, fixed there. No full birth starts until the second worker's check records agree with the main worker's.
