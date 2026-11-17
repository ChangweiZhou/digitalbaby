# E47 trial tables (computed from the receipts by e47_trial_read.py)

| Run | A80 (3 rows) | R50 | R80 (3 rows) | R90 | final A | final R | len12 | len40 (last 3) | use of far cells | CPU s/update |
|---|---|---|---|---|---|---|---|---|---|---|
| one S | 13568 | 9728 | 17664 | None | 93.3 | 86.3 | 93.0 | 90.8 | partly | 0.397 |
| one B | 10496 | 6400 | 11008 | 17152 | 96.2 | 94.7 | 94.7 | 95.2 | partly | 0.398 |
| place B | 10496 | 6400 | 11264 | 15360 | 95.8 | 95.2 | 97.0 | 96.6 | partly | 0.440 |
| E42 F10 S | 13312 | 9472 | 15104 | 23296 | 92.2 | 88.5 | 91.7 | 89.1 | — | 0.105 |
| E42 F10 B | 8704 | 6400 | 9472 | 19200 | 91.8 | 93.3 | 91.3 | 92.6 | — | 0.102 |
| E42 P8 S | 7680 | 6400 | 8448 | 12288 | 95.7 | 92.5 | 96.0 | 96.0 | — | 0.138 |
| E42 P8 B | 6400 | 5120 | 7168 | 8960 | 98.0 | 96.6 | 97.3 | 96.6 | — | 0.135 |
| E42 G S | 5632 | 3840 | 6656 | 13824 | 98.0 | 95.8 | 98.3 | 96.4 | — | 0.103 |
| E42 G B | 5120 | 3840 | 6400 | 7936 | 98.5 | 98.2 | 98.3 | 94.6 | — | 0.097 |

## Scores at the end, plain and with the far cells held

| Run | A plain / zero / mean | R plain / zero / mean | len12 plain / zero / mean | len40 plain / zero / mean | form |
|---|---|---|---|---|---|
| one S | 93.3 / 77.5 / 84.8 | 86.3 / 77.7 / 80.6 | 93.0 / 84.7 / 86.3 | 90.0 / 79.0 / 80.3 | partly |
| one B | 96.2 / 78.8 / 90.2 | 94.7 / 78.9 / 88.7 | 94.7 / 81.3 / 89.7 | 95.3 / 81.0 / 89.0 | partly |
| place B | 95.8 / 88.8 / 93.3 | 95.2 / 89.8 / 93.2 | 97.0 / 88.7 / 94.7 | 96.7 / 88.3 / 94.3 | partly |

## Comparisons (E41's four cases on updates; lengths: mean of the last three rows at 40 letters)

| Comparison | reference | candidate | updates | multiply-adds | len40 difference |
|---|---|---|---|---|---|
| one S vs E42 F10 S: A80_three | 13312 | 13568 | about the same | about the same | +1.7 (within 15 points) |
| one S vs E42 F10 S: R80_three | 15104 | 17664 | about the same | about the same | +1.7 (within 15 points) |
| one S vs E42 G S: A80_three | 5632 | 13568 | more | more | -5.7 (within 15 points) |
| one S vs E42 G S: R80_three | 6656 | 17664 | more | more | -5.7 (within 15 points) |
| one B vs E42 F10 B: A80_three | 8704 | 10496 | about the same | about the same | +2.7 (within 15 points) |
| one B vs E42 F10 B: R80_three | 9472 | 11008 | about the same | about the same | +2.7 (within 15 points) |
| one B vs E42 G B: A80_three | 5120 | 10496 | more | more | +0.7 (within 15 points) |
| one B vs E42 G B: R80_three | 6400 | 11008 | more | about the same | +0.7 (within 15 points) |
| place B vs E42 F10 B: A80_three | 8704 | 10496 | about the same | about the same | +4.0 (within 15 points) |
| place B vs E42 F10 B: R80_three | 9472 | 11264 | about the same | about the same | +4.0 (within 15 points) |
| place B vs E42 G B: A80_three | 5120 | 10496 | more | more | +2.0 (within 15 points) |
| place B vs E42 G B: R80_three | 6400 | 11264 | more | about the same | +2.0 (within 15 points) |
| place B vs E42 P8 B: A80_three | 6400 | 10496 | more | — | +0.0 (within 15 points) |
| place B vs E42 P8 B: R80_three | 7168 | 11264 | more | — | +0.0 (within 15 points) |
| place B vs one B: A80_three | 10496 | 10496 | about the same | about the same | +1.3 (within 15 points) |
| place B vs one B: R80_three | 11008 | 11264 | about the same | about the same | +1.3 (within 15 points) |

## Cosines between the sweep's and the exact gradient of the recurrent weights (first batch)

| Run | start: interface / far / all | end: interface / far / all | far non-zero share start B / S | end B / S |
|---|---|---|---|---|
| one S | 0.758 / 0.621 / 0.757 | 0.611 / 0.527 / 0.597 | 1.000 / 1.000 | 1.000 / 1.000 |
| one B | 0.758 / 0.621 / 0.757 | 0.658 / 0.473 / 0.643 | 1.000 / 1.000 | 1.000 / 1.000 |
| place B | 0.758 / 0.626 / 0.758 | 0.617 / 0.443 / 0.614 | 0.900 / 0.900 | 0.900 / 0.900 |

## Mean magnitude of the recurrent weights

| Run | start interface / far / scale | end interface / far / scale |
|---|---|---|
| one S | 0.1515 / 0.1503 / 0.1506 | 0.1985 / 0.1497 / 0.1619 |
| one B | 0.1515 / 0.1503 / 0.1506 | 0.1789 / 0.2426 / 0.2266 |
| place B | 0.1515 / 0.1503 / 0.1506 | 0.1757 / 0.1226 / 0.1359 |

## Predictions

- **P47.1** Stage 0, checks 1 and 2 pass under both rules: **yes**
- **P47.2** B place reaches 80 on R (three rows): **yes**
- **P47.4** B one: far cells used: **no**
- **P47.5** S one: far cells used: **no**
- **P47.6** place: cosine at the end lower for far cells than for interface cells by 0.1 or more: **yes**
- **P47.7** B one: updates to 80 on R about the same as or more than E42's F10 (not fewer): **yes**
