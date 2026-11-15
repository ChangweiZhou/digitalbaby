# E32 check 7, second worker (A): records and comparison (14 November 2026)

- **Code.** `exp32_second_A.py`, written from brief revision 4 (`a1583c23`) alone. I did not read worker B's `exp32_births.py`, and E31's files are not imported.
- **Host.** Linux x86-64, Python 3.13.18, torch 2.15.0+cpu, one thread.

## Dry births, seed 73129, 2,048 updates

| arm | `tensors_sha256` | main worker (df8bcf8f) | | CPU s |
|---|---|---|---|---|
| floor (validation) | `b49561cf…` | E28's dry receipt `b49561cf…` | equal; data digest equal | 243 |
| G | `ad7114bfe44974e889709931e7c1ccadac5ae9d2b835cd8319cc62f01060386a` | `ad7114bf…` | **equal** | 244 |
| H | `7fdaa9ddf66fa2fd8d86502ea42888bda3e8cab2e8fc137f16b15d40b0c1a4dc` | `7fdaa9dd…` | **equal** | 227 |
| K | `e1433252a2cef41069df76565b9ab0396824be0f9b75f9fd7a0bcdfc6c3cfd5e` | `e1433252…` | **equal** | 234 |

K's update identity holds to 7e-9, 1.5e-8 and 1.9e-8 at updates 1, 10 and 100. At the end, cos(Bhh, W) = 0.873 and ‖Bhh − W‖/‖W‖ = 0.504.

## P4 strengths, no training

The strengths are computed at E28's S-all row-4,096 weights (hash checked against the receipt) over batches 4,097 to 4,224 (stream digest checked against E28's row 4,096).

| seed | P4a | P4b | P4c | largest relative difference from the main worker |
|---|---|---|---|---|
| 73129 | 0.0206176843 | 0.00412353686 | 0.000824707372 | 5.6e-7 |
| 93683 | 0.0171638601 | 0.00343277202 | 0.000686554404 | 1.5e-7 |
| 116089 | 0.0175890262 | 0.00351780523 | 0.000703561046 | 2.6e-7 |

Batch 4,097 alone, against the reviewer's reference:

| seed | task norm | reviewer | term norm (λ = 1) | reviewer |
|---|---|---|---|---|
| 73129 | 0.0367404 | 0.036740 | 0.890970 | 0.890970 |
| 93683 | 0.0771133 | 0.077113 | 0.836491 | 0.836491 |
| 116089 | 0.0933196 | 0.093320 | 0.849241 | 0.849241 |

Relative differences are at most 1.2e-5, which is the precision of the reviewer's printed digits.

**Agreement.** All three hashes are equal and the nine strengths agree to 5.6e-7 relative, well inside the 1e-5 bar. Record: `E32_CHECK7_COMPARE_A.json`.
