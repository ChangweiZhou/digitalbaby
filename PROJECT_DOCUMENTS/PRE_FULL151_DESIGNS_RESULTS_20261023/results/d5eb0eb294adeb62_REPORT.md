# OMNIBUS10 results

Version `omnibus10.1.1`
Scientific roster: **True**
Completed conceptual trajectories: **3920**

## Overall five-factor effects on delayed A held-out retention

Positive values mean lower (better) bits/consequence when the factor is ON.

| factor | improvement (bits) | 95% CI | Holm p | classification |
|---|---:|---:|---:|---|
| A_ELIG | +0.0371 | [+0.0136, +0.0605] | 0.02109 | INDEPENDENT_POSITIVE |
| B_CONSOL | -0.0065 | [-0.0152, +0.0023] | 0.2692 | CONDITIONAL_POSITIVE |
| C_FORGET | -0.0437 | [-0.0568, -0.0307] | 3.791e-05 | CONDITIONAL_POSITIVE |
| D_LOCAL | -1.3655 | [-1.5293, -1.2017] | 2.258e-10 | CONDITIONAL_POSITIVE |
| E_META | +0.0147 | [+0.0104, +0.0190] | 3.155e-05 | TRADEOFF |

## Cell contrasts

Positive values mean the left-described cell family has lower delayed-A loss.

| contrast | improvement | 95% CI | Holm p |
|---|---:|---:|---:|
| STATEFUL_vs_STATIC | +0.0640 | [-0.0090, +0.1369] | 0.2439 |
| LEAK_HOM_vs_STATIC | +0.1621 | [+0.0900, +0.2343] | 0.001433 |
| ALIF_HOM_vs_LEAK_HOM | -0.0155 | [-0.0290, -0.0021] | 0.1048 |
| ELM_HOM_vs_LEAK_HOM | +0.0038 | [-0.0014, +0.0091] | 0.2692 |
| LEAK_HET_vs_LEAK_HOM | -0.2002 | [-0.2663, -0.1342] | 8.594e-05 |
| ALIF_HET_vs_ALIF_HOM | -0.1329 | [-0.1905, -0.0753] | 0.001296 |
| ELM_HET_vs_ELM_HOM | -0.2323 | [-0.3021, -0.1626] | 3.791e-05 |
| HET_vs_HOM | -0.1885 | [-0.2483, -0.1286] | 6.255e-05 |

## Interpretation

- The exact factorial mechanisms are ranked by paired seed-level effects; a null result closes only the tested construction.
- Silent reinstatement and read authority are counterfactual read probes and are not extra training trajectories.
- COV96 is a bounded nonlocal covariance oracle, not a candidate biological core.
- `ELM_HOM/HET` are fixed ELM-like references, not the unrecovered historical P4 ELM implementation.
