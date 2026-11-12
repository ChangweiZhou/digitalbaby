# E26 stage 2: the six births (full dose, 24,576 updates)

Written by verify_births.py from the receipts. Exact answers out of 300.

## Checks

- S_73129: {'admit_e26': True, 'complete': True, 'optimizer_steps_24576': True, 'weights_sha256_matches_file': True, 'tensors_sha256_matches_file': True, 'sources_equal_360e1b8d': True, 'six_rows_4096_to_24576': True, 'finite_losses_every_row': True, 'data_digest_last_row_equals_receipt': True, 'weight_decay_read_back': [0.25, 0.0]}
- B_73129: {'admit_e26': True, 'complete': True, 'optimizer_steps_24576': True, 'weights_sha256_matches_file': True, 'tensors_sha256_matches_file': True, 'sources_equal_360e1b8d': True, 'six_rows_4096_to_24576': True, 'finite_losses_every_row': True, 'data_digest_last_row_equals_receipt': True, 'weight_decay_read_back': [0.25, 0.0]}
- data_digest_S_equals_B_73129: True
- S_93683: {'admit_e26': True, 'complete': True, 'optimizer_steps_24576': True, 'weights_sha256_matches_file': True, 'tensors_sha256_matches_file': True, 'sources_equal_360e1b8d': True, 'six_rows_4096_to_24576': True, 'finite_losses_every_row': True, 'data_digest_last_row_equals_receipt': True, 'weight_decay_read_back': [0.25, 0.0]}
- B_93683: {'admit_e26': True, 'complete': True, 'optimizer_steps_24576': True, 'weights_sha256_matches_file': True, 'tensors_sha256_matches_file': True, 'sources_equal_360e1b8d': True, 'six_rows_4096_to_24576': True, 'finite_losses_every_row': True, 'data_digest_last_row_equals_receipt': True, 'weight_decay_read_back': [0.25, 0.0]}
- data_digest_S_equals_B_93683: True
- S_116089: {'admit_e26': True, 'complete': True, 'optimizer_steps_24576': True, 'weights_sha256_matches_file': True, 'tensors_sha256_matches_file': True, 'sources_equal_360e1b8d': True, 'six_rows_4096_to_24576': True, 'finite_losses_every_row': True, 'data_digest_last_row_equals_receipt': True, 'weight_decay_read_back': [0.25, 0.0]}
- B_116089: {'admit_e26': True, 'complete': True, 'optimizer_steps_24576': True, 'weights_sha256_matches_file': True, 'tensors_sha256_matches_file': True, 'sources_equal_360e1b8d': True, 'six_rows_4096_to_24576': True, 'finite_losses_every_row': True, 'data_digest_last_row_equals_receipt': True, 'weight_decay_read_back': [0.25, 0.0]}
- data_digest_S_equals_B_116089: True

## Hard mean (add89 and sub89 at k=12) through the birth

| step | S 73129 | B 73129 | S 93683 | B 93683 | S 116089 | B 116089 | S−B 73129 | S−B 93683 | S−B 116089 | S−B mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096 | 209.0 | 188.0 | 220.0 | 253.0 | 229.5 | 237.5 | +21.0 | -33.0 | -8.0 | -6.7 |
| 8192 | 267.5 | 264.0 | 275.5 | 284.5 | 276.0 | 266.5 | +3.5 | -9.0 | +9.5 | +1.3 |
| 12288 | 281.0 | 269.0 | 289.0 | 276.0 | 282.5 | 282.5 | +12.0 | +13.0 | +0.0 | +8.3 |
| 16384 | 272.5 | 283.0 | 282.0 | 296.5 | 291.0 | 278.0 | -10.5 | -14.5 | +13.0 | -4.0 |
| 20480 | 293.0 | 252.5 | 291.5 | 285.5 | 289.0 | 275.0 | +40.5 | +6.0 | +14.0 | +20.2 |
| 24576 | 294.0 | 295.5 | 294.5 | 296.0 | 299.5 | 281.0 | -1.5 | -1.5 | +18.5 | +5.2 |

## Loss (training batch at the checkpoint)

| step | S 73129 | B 73129 | S 93683 | B 93683 | S 116089 | B 116089 |
|---|---|---|---|---|---|---|
| 4096 | 0.1968 | 0.2299 | 0.1807 | 0.1353 | 0.1529 | 0.1044 |
| 8192 | 0.0582 | 0.0519 | 0.0430 | 0.0334 | 0.0451 | 0.0464 |
| 12288 | 0.0416 | 0.0392 | 0.0280 | 0.0246 | 0.0273 | 0.0357 |
| 16384 | 0.0286 | 0.0309 | 0.0207 | 0.0208 | 0.0180 | 0.0266 |
| 20480 | 0.0263 | 0.0316 | 0.0164 | 0.0178 | 0.0147 | 0.0247 |
| 24576 | 0.0207 | 0.0295 | 0.0131 | 0.0162 | 0.0134 | 0.0226 |

## End of birth, all operations and k (exact /300)

| birth | add89 k0 | add89 k4 | add89 k8 | add89 k12 | add89 k20 | sub89 k0 | sub89 k4 | sub89 k8 | sub89 k12 | sub89 k20 | max89 k0 | max89 k4 | max89 k8 | max89 k12 | max89 k20 | min89 k0 | min89 k4 | min89 k8 | min89 k12 | min89 k20 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S 73129 | 290 | 295 | 295 | 293 | 294 | 289 | 294 | 294 | 295 | 294 | 299 | 299 | 299 | 299 | 299 | 294 | 296 | 295 | 295 | 292 |
| B 73129 | 295 | 295 | 295 | 295 | 294 | 294 | 297 | 296 | 296 | 295 | 300 | 300 | 300 | 300 | 300 | 288 | 292 | 292 | 292 | 289 |
| S 93683 | 294 | 296 | 296 | 296 | 296 | 290 | 295 | 295 | 293 | 292 | 295 | 296 | 296 | 295 | 293 | 297 | 298 | 298 | 297 | 297 |
| B 93683 | 297 | 297 | 297 | 298 | 298 | 291 | 295 | 294 | 294 | 295 | 298 | 297 | 297 | 297 | 297 | 292 | 295 | 294 | 293 | 293 |
| S 116089 | 298 | 300 | 300 | 300 | 300 | 300 | 300 | 299 | 299 | 299 | 296 | 293 | 294 | 294 | 293 | 287 | 292 | 291 | 291 | 291 |
| B 116089 | 278 | 284 | 284 | 284 | 279 | 279 | 281 | 279 | 278 | 276 | 292 | 295 | 296 | 297 | 298 | 295 | 294 | 293 | 293 | 294 |

## Silenced readout at k=12, end of birth

| birth | add89 | sub89 | max89 | min89 |
|---|---|---|---|---|
| S 73129 | 5 | 3 | 0 | 8 |
| B 73129 | 0 | 0 | 0 | 0 |
| S 93683 | 0 | 0 | 0 | 0 |
| B 93683 | 0 | 0 | 0 | 0 |
| S 116089 | 0 | 0 | 0 | 0 |
| B 116089 | 3 | 6 | 4 | 6 |

## CPU seconds

| birth | total | updates | scoring |
|---|---|---|---|
| S 73129 | 3278 | 3228 | 50 |
| B 73129 | 3070 | 3020 | 51 |
| S 93683 | 3282 | 3232 | 50 |
| B 93683 | 3070 | 3019 | 51 |
| S 116089 | 3274 | 3223 | 51 |
| B 116089 | 3068 | 3018 | 50 |
