## Runs (updates to 80: first of three consecutive rows at or above 80; level: mean of the last three rows)

| Rule | Arm | Updates to 80 on A | Updates to 80 on R | Level A | Level R | Level length 40 | Read | Losses finite | Resumed at |
|---|---|---|---|---|---|---|---|---|---|
| S | one | 13568 | 17664 | 92.3 | 87.8 | 90.8 | read | True | 9216 |
| S | x3 | 15360 | 18432 | 89.4 | 86.8 | 86.7 | read | True | – |
| S | wd0 | 11264 | 13312 | 89.2 | 86.3 | 89.6 | read | True | 16384, 21504 |
| S | x3-wd0 | 15104 | 22272 | 83.5 | 78.5 | 82.4 | not read | True | 16384, 21504 |
| B | one | 10496 | 11008 | 95.4 | 94.2 | 95.2 | read | True | 13312 |
| B | x3 | 9728 | 10240 | 93.8 | 93.5 | 92.3 | read | True | – |
| B | wd0 | 8704 | 11776 | 90.2 | 88.0 | 91.3 | read | True | 16384, 21504 |
| B | x3-wd0 | 9728 | 13568 | 89.6 | 85.5 | 91.1 | read | True | – |

## Learning, in the words of `describe` (candidate against reference; 24,064 as the last possible start)

| Rule | Candidate | Reference | A: updates | A: words | R: updates | R: words |
|---|---|---|---|---|---|---|
| S | x3 | one | 15360 / 13568 | about the same | 18432 / 17664 | about the same |
| S | wd0 | one | 11264 / 13568 | about the same | 13312 / 17664 | about the same |
| S | x3-wd0 | one | 15104 / 13568 | about the same | 22272 / 17664 | about the same |
| S | x3-wd0 | x3 | 15104 / 15360 | about the same | 22272 / 18432 | about the same |
| S | x3-wd0 | wd0 | 15104 / 11264 | about the same | 22272 / 13312 | more |
| B | x3 | one | 9728 / 10496 | about the same | 10240 / 11008 | about the same |
| B | wd0 | one | 8704 / 10496 | about the same | 11776 / 11008 | about the same |
| B | x3-wd0 | one | 9728 / 10496 | about the same | 13568 / 11008 | about the same |
| B | x3-wd0 | x3 | 9728 / 9728 | about the same | 13568 / 10240 | about the same |
| B | x3-wd0 | wd0 | 9728 / 8704 | about the same | 13568 / 11776 | about the same |

## The far cells as a whole (the two diagnostics at the last row; the forms of E47's brief, by `use_form`)

| Rule | Arm | A plain / far at zero / far at mean | R plain / far at zero / far at mean | Fall A zero, mean | Fall R zero, mean | Form |
|---|---|---|---|---|---|---|
| S | one | 93.3 / 77.5 / 84.8 | 86.3 / 77.7 / 80.6 | 15.8, 8.5 | 8.6, 5.8 | partly |
| S | x3 | 90.2 / 75.3 / 82.2 | 85.7 / 69.4 / 81.3 | 14.8, 8.0 | 16.3, 4.4 | partly |
| S | wd0 | 89.7 / 49.5 / 85.0 | 85.7 / 50.0 / 83.3 | 40.2, 4.7 | 35.7, 2.4 | a constant is used |
| S | x3-wd0 | 85.7 / 43.5 / 76.3 | 78.8 / 47.0 / 72.5 | 42.2, 9.3 | 31.8, 6.3 | partly |
| B | one | 96.2 / 78.8 / 90.2 | 94.7 / 78.9 / 88.7 | 17.3, 6.0 | 15.8, 6.0 | partly |
| B | x3 | 94.7 / 60.2 / 60.0 | 92.8 / 55.9 / 65.6 | 34.5, 34.7 | 36.9, 27.2 | used |
| B | wd0 | 91.2 / 52.2 / 76.2 | 88.6 / 43.8 / 75.7 | 39.0, 15.0 | 44.8, 12.9 | used |
| B | x3-wd0 | 90.2 / 39.0 / 48.5 | 86.2 / 38.9 / 50.1 | 51.2, 41.7 | 47.3, 36.1 | used |

## The size of the weights by ring (mean absolute recurrent weight into the cells of a group) at the first row, at update 4,096 and at the last row

| Rule | Arm | Update | Interface | Ring 1 | Ring 2 | Ring 3 |
|---|---|---|---|---|---|---|
| S | one (from its weight files) | 0 | 0.152 | 0.150 | 0.151 | 0.150 |
| S | one (from its weight files) | 24576 | 0.198 | 0.178 | 0.132 | 0.145 |
| S | one | 4096 | not recorded by the trial | | | |
| S | x3 | 0 | 0.455 | 0.451 | 0.452 | 0.449 |
| S | x3 | 4096 | 0.243 | 0.201 | 0.176 | 0.169 |
| S | x3 | 24576 | 0.213 | 0.154 | 0.058 | 0.024 |
| S | wd0 | 0 | 0.152 | 0.150 | 0.151 | 0.150 |
| S | wd0 | 4096 | 0.262 | 0.238 | 0.204 | 0.178 |
| S | wd0 | 24576 | 0.404 | 0.346 | 0.287 | 0.237 |
| S | x3-wd0 | 0 | 0.455 | 0.451 | 0.452 | 0.449 |
| S | x3-wd0 | 4096 | 0.514 | 0.474 | 0.462 | 0.455 |
| S | x3-wd0 | 24576 | 0.574 | 0.519 | 0.483 | 0.468 |
| B | one (from its weight files) | 0 | 0.152 | 0.150 | 0.151 | 0.150 |
| B | one (from its weight files) | 24576 | 0.179 | 0.234 | 0.257 | 0.234 |
| B | one | 4096 | not recorded by the trial | | | |
| B | x3 | 0 | 0.455 | 0.451 | 0.452 | 0.449 |
| B | x3 | 4096 | 0.241 | 0.217 | 0.188 | 0.180 |
| B | x3 | 24576 | 0.178 | 0.241 | 0.247 | 0.223 |
| B | wd0 | 0 | 0.152 | 0.150 | 0.151 | 0.150 |
| B | wd0 | 4096 | 0.267 | 0.264 | 0.255 | 0.259 |
| B | wd0 | 24576 | 0.409 | 0.387 | 0.361 | 0.348 |
| B | x3-wd0 | 0 | 0.455 | 0.451 | 0.452 | 0.449 |
| B | x3-wd0 | 4096 | 0.509 | 0.475 | 0.459 | 0.452 |
| B | x3-wd0 | 24576 | 0.580 | 0.519 | 0.485 | 0.467 |

## The condition of the conditional run (under back-propagation)

- x3 B: not read False; updates to 80 on A 9728 against the trial's 10496 (about the same); qualifies: False
- x3-wd0 B: not read False; updates to 80 on A 9728 against the trial's 10496 (about the same); qualifies: False
- the conditional run: not made

## Predictions P53.2 to P53.5, P53.8 to P53.10 and P53.12 (P53.1 from the checks; P53.6, P53.7, P53.11 are the reviewer's)

- P53.2 (wd0 reaches 80 on A under both rules): true
- P53.3 (wd0 against one: about the same on A under both rules): true
- P53.4 (x3-wd0 reaches 80 on A under back-propagation): true
- P53.5 (x3-wd0 reaches 80 on A under the sweep): true
- P53.8 (no new run is fewer than the trial's on updates to 80 on A): true ([])
- P53.9 (x3 reaches 80 on A under back-propagation): true
- P53.10 (x3 reaches 80 on A under the sweep): true
- P53.12 (in x3 the mean absolute weight into ring 3 at update 4,096 is at least 0.30 under both rules): false (S 0.169, B 0.180)
