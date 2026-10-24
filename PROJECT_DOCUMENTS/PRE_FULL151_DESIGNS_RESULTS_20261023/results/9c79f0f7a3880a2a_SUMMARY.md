# V34 — Sparse Expanded Compartment Memory

Status: complete; 64/64 trajectories.

Promising V34 candidate: **False**.

| Arm | A after A | A after D | D current | Whole bpb | Target bpb | Non-target bpb |
|---|---:|---:|---:|---:|---:|---:|
| H-O | 100.00% | 7.03% | 100.00% | 2.1481 | 1.1857 | 2.2221 |
| H-P | 100.00% | 15.62% | 100.00% | 1.8060 | 0.7786 | 1.8851 |
| G-D-O | 78.12% | 2.34% | 78.91% | 3.2121 | 4.6500 | 3.1015 |
| G-S-O | 100.00% | 6.25% | 100.00% | 1.5935 | 0.6336 | 1.6673 |
| C1-O | 100.00% | 1.56% | 100.00% | 2.4056 | 1.3591 | 2.4861 |
| C4-O | 100.00% | 3.12% | 100.00% | 1.7632 | 0.6683 | 1.8474 |
| C4-P | 100.00% | 3.12% | 100.00% | 1.5933 | 0.5602 | 1.6728 |
| Dict | 100.00% | 100.00% | 100.00% | 2.8628 | 0.8176 | 3.0201 |

## Preregistered contrasts

| Gate | Contrast | A-after-D gain, pp [95% CI] | Verdict |
|---|---|---:|---|
| P1 | G-S-O − G-D-O | 3.91 [-2.90, 10.71] | invalid_acquisition |
| P2 | C4-O − C1-O | 1.56 [-0.86, 3.98] | below_effect_threshold |
| P3 | C4-P − C4-O | 0.00 [-2.79, 2.79] | below_effect_threshold |
| P4 | C4-P − H-P | -12.50 [-28.05, 3.05] | below_effect_threshold |

## Revision after four changed labels

| Arm | Exact | Known alias | Held-out alias |
|---|---:|---:|---:|
| H-O | 100.00% | 68.75% | 78.12% |
| H-P | 100.00% | 68.75% | 78.12% |
| G-D-O | 100.00% | 100.00% | 100.00% |
| G-S-O | 100.00% | 100.00% | 96.88% |
| C1-O | 100.00% | 40.62% | 34.38% |
| C4-O | 100.00% | 87.50% | 84.38% |
| C4-P | 100.00% | 75.00% | 71.88% |
| Dict | 0.00% | 0.00% | 0.00% |

Acquisition validity requires mean A ≥90%, minimum seed A ≥75%, and mean D ≥90%. Each gate also requires its mean retention effect, a positive paired bound, and D-current noninferiority. Whole-stream loss is diagnostic. REPORT.json separates invalid acquisition, adverse results and unresolved precision.

Exactly eight fixed arms; no outcome-dependent follow-up. These are fresh seeds for the post-V33 architecture hypothesis. Covariance can alter more than 48 rows within selected compartments. Costs and utilization are reported explicitly.
