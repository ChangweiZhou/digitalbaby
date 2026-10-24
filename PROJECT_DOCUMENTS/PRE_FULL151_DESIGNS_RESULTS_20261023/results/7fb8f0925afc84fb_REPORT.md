# V36 results

**Source SHA256** `0e2cbef0e58d46c235716852f35285289183f3c4dae7009c9e3b924735aa66ec`

Label-free. Workload imported unchanged from V35.
Writer is plain delta unless the arm says `local_v`.

## P1 — did the manipulation move separation?

_Verdict: **interpretable**_

At least one code raised the separation ratio by the required factor; behavioural contrasts are interpretable.

| arm | separation ratio | gain vs raw | moved | sufficient |
|---|---|---|:-:|:-:|
| expand/delta | +2.596 [+2.372, +2.820] | +1.163 [+0.944, +1.383] | True | False |
| norm/delta | +1.699 [+1.682, +1.715] | +0.266 [+0.256, +0.275] | True | False |
| phase_expand/delta | +5.614 [+5.231, +5.998] | +4.182 [+3.798, +4.566] | True | True |
| phase_expand/local_v | +5.614 [+5.231, +5.998] | — | — | — |
| phase_norm/delta | +5.708 [+5.435, +5.980] | +4.275 [+4.007, +4.543] | True | True |
| raw/delta | +1.433 [+1.425, +1.440] | — | — | — |
| raw/local_v | +1.433 [+1.425, +1.440] | — | — | — |

## Behaviour (paired vs raw/delta)

| arm | target accuracy | target bpb | whole bpb |
|---|---|---|---|
| expand/delta | +0.0498 [+0.0433, +0.0564] | -0.2561 [-0.2795, -0.2327] | -0.0754 [-0.0864, -0.0645] |
| norm/delta | +0.0402 [+0.0369, +0.0435] | -0.2905 [-0.2997, -0.2814] | +0.0251 [+0.0206, +0.0296] |
| phase_expand/delta | -0.0481 [-0.0565, -0.0397] | +0.2897 [+0.2340, +0.3454] | +0.3512 [+0.3192, +0.3831] |
| phase_expand/local_v | +0.1711 [+0.1651, +0.1770] | +0.9959 [+0.9497, +1.0421] | +2.3608 [+2.3409, +2.3807] |
| phase_norm/delta | -0.0318 [-0.0431, -0.0206] | +0.5040 [+0.3974, +0.6107] | +2.8268 [+2.7253, +2.9283] |
| raw/local_v | +0.1736 [+0.1680, +0.1793] | +0.9490 [+0.9047, +0.9933] | +2.3377 [+2.3234, +2.3520] |

## Reference points

| quantity | value |
|---|---:|
| fly KC overlap, same odour (real wiring, 5% coding) | 0.417 |
| fly KC overlap, different odour | 0.032 |
| **fly separation ratio** | **13.5×** |
| lag96 same / different | 0.854 / 0.599 |
| **lag96 separation ratio** | **1.43×** |
| V35 LOCAL-V retention / known-alias | 99.22% / **0.0%** |
| V35 SHARED-R retention / known-alias | 7.03% / 28.1% |

## Interpretation limits

- P1 is a hard gate. If separation did not move, nothing below it
  means anything; that is the V31 lesson applied to the code.
- Cosine on dense codes and Jaccard on the fly binary sets are
  different metrics. Fly numbers are the motivation, not a target
  to be matched numerically.
- Expansion ALONE was measured at 2.64× before this file was written;
  normalisation is doing most of the work and must be credited so.
- Different keys share 6 of 12 context bytes by construction
  (PREFIX + HEADER + SEPARATOR), so 13.5× is unreachable here.
  The question is whether the achievable rise changes behaviour.
- Plain delta is used deliberately. A win here is a claim about the
  code, not a new write rule.
