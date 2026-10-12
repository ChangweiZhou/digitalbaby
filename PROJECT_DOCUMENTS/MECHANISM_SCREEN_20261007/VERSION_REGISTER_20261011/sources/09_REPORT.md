# V31 results

**Source SHA256** `028f30c6ce834bdbb37894bd6fa942b27b792c194ba7621237547e038f41cb1b`

Label-free: the only teaching signal is the next-symbol prediction residual.
Lower bits/symbol is better; uniform is 4.000.

## Encoder adequacy (P0)

| encoder | plain bits on A | gain over uniform | adequate |
|---|---|---|---|
| contrast | +4.0873 [+4.0612, +4.1134] | -0.0873 [-0.1134, -0.0612] | **False** |
| lagged | +3.9962 [+3.9792, +4.0131] | +0.0038 [-0.0131, +0.0208] | **False** |
| observable | +3.5839 [+3.5115, +3.6562] | +0.4161 [+0.3438, +0.4885] | **True** |

## contrast

_Code carries no task information. All mechanism verdicts VOID._


## lagged

_Code carries no task information. All mechanism verdicts VOID._


## observable

_Task eligible: **True**_

| gate | value | pass |
|---|---|---|
| headroom | +0.4357 [+0.2950, +0.5763] | True |
| tension | +0.4428 [+0.3666, +0.5190] | True |
| recall dependence | +0.3812 [+0.3096, +0.4528] | True |

| arm | P1 retention vs plain | P2 vs frozen-sigma | P3 vs shuffled-sigma | verdict |
|---|---|---|---|---|
| mesu_N128 | +0.2652 [+0.2110, +0.3194] | +0.0183 [+0.0071, +0.0295] | +0.0793 [+0.0624, +0.0961] | **inconclusive** |
| mesu_N32 | +0.3770 [+0.3103, +0.4436] | +0.1301 [+0.1055, +0.1546] | +0.1911 [+0.1576, +0.2245] | **inconclusive** |
| mesu_N512 | +0.2719 [+0.2233, +0.3205] | +0.0249 [+0.0032, +0.0467] | +0.0859 [+0.0651, +0.1068] | **inconclusive** |
| mesu_N8 | +0.4242 [+0.3478, +0.5006] | +0.1773 [+0.1400, +0.2146] | +0.2383 [+0.1929, +0.2837] | **inconclusive** |

**P4 (eq 19, infinite horizon ossifies): True** -- retention advantage +0.1656 [+0.1384, +0.1927], late-B cost +0.1234 [+0.0998, +0.1470]

## Interpretation limits

- P0 is a hard gate. An inadequate code voids every verdict beneath it;
  running write rules on a code that cannot express the task is how V27
  produced uninterpretable results.
- `fixed_rate_effect` means a frozen or misassigned variance matches the
  learned one: a per-weight rate schedule, not metaplasticity. Retire it.
- `observable` is a diagnostic ceiling, not a proposed architecture.
- Retention is scored against a never-trained generic source; a small
  recall margin means the metric captures only generic symbol statistics.
- Synthetic streams with a controlled shift. No scaling or language claim.
