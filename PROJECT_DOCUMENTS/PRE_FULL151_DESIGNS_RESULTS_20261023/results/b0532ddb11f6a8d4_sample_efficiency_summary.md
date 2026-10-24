# D-ELM sample-efficiency sweep

The adaptive x-axis/sample count includes the separate architecture-label pool. With the default sweep this is exactly 128 additional unique labeled examples; rewiring revisits them at each structural update.

## Endpoint behavior by unique-label budget

| Weight labels | Variant | Seeds | Total unique labels | Test acc | Delay-16 acc | Delay-24 acc |
|---:|---|---:|---:|---:|---:|---:|
| 250 | adaptive | 3 | 378 | 13.38% ± 5.85% | 10.00% ± 0.00% | 10.00% ± 0.00% |
| 250 | fixed | 3 | 250 | 13.30% ± 5.72% | 10.00% ± 0.00% | 10.00% ± 0.00% |
| 500 | adaptive | 3 | 628 | 32.62% ± 1.38% | 29.57% ± 2.27% | 25.35% ± 3.46% |
| 500 | fixed | 3 | 500 | 32.49% ± 1.55% | 29.98% ± 2.68% | 25.40% ± 3.60% |
| 1000 | adaptive | 3 | 1128 | 67.79% ± 1.09% | 65.43% ± 1.78% | 63.05% ± 1.69% |
| 1000 | fixed | 3 | 1000 | 66.55% ± 2.02% | 64.56% ± 2.68% | 62.31% ± 2.91% |
| 2000 | adaptive | 3 | 2128 | 75.08% ± 0.83% | 73.81% ± 0.37% | 70.42% ± 1.74% |
| 2000 | fixed | 3 | 2000 | 74.71% ± 0.74% | 73.20% ± 0.99% | 70.64% ± 1.97% |
| 5000 | adaptive | 3 | 5128 | 80.18% ± 0.16% | 75.24% ± 2.25% | 72.76% ± 1.12% |
| 5000 | fixed | 3 | 5000 | 79.62% ± 0.45% | 74.61% ± 3.94% | 69.50% ± 2.19% |

## Presentation-efficiency thresholds (individual runs)

`total presentations` counts repeated weight-training examples plus every labeled architecture example used at a rewiring event.

| Run | Best val | Presentations to 70% | to 75% | to 80% | to 85% |
|---|---:|---:|---:|---:|---:|
| adaptive_n250_seed7 | 10.32% | — | — | — | — |
| fixed_n250_seed7 | 10.32% | — | — | — | — |
| adaptive_n250_seed17 | 9.98% | — | — | — | — |
| fixed_n250_seed17 | 9.98% | — | — | — | — |
| adaptive_n250_seed29 | 20.06% | — | — | — | — |
| fixed_n250_seed29 | 19.92% | — | — | — | — |
| adaptive_n500_seed7 | 31.48% | — | — | — | — |
| fixed_n500_seed7 | 31.00% | — | — | — | — |
| adaptive_n500_seed17 | 34.58% | — | — | — | — |
| fixed_n500_seed17 | 34.44% | — | — | — | — |
| adaptive_n500_seed29 | 32.20% | — | — | — | — |
| fixed_n500_seed29 | 32.26% | — | — | — | — |
| adaptive_n1000_seed7 | 67.66% | — | — | — | — |
| fixed_n1000_seed7 | 66.12% | — | — | — | — |
| adaptive_n1000_seed17 | 67.50% | — | — | — | — |
| fixed_n1000_seed17 | 65.02% | — | — | — | — |
| adaptive_n1000_seed29 | 67.98% | — | — | — | — |
| fixed_n1000_seed29 | 67.66% | — | — | — | — |
| adaptive_n2000_seed7 | 74.90% | 26,640 | — | — | — |
| fixed_n2000_seed7 | 75.00% | 26,000 | 56,000 | — | — |
| adaptive_n2000_seed17 | 76.28% | 22,512 | 38,768 | — | — |
| fixed_n2000_seed17 | 75.70% | 22,000 | 42,000 | — | — |
| adaptive_n2000_seed29 | 75.62% | 22,512 | 44,768 | — | — |
| fixed_n2000_seed29 | 75.12% | 24,000 | 54,000 | — | — |
| adaptive_n5000_seed7 | 81.04% | 25,128 | 35,256 | 85,768 | — |
| fixed_n5000_seed7 | 80.58% | 25,000 | 35,000 | 100,000 | — |
| adaptive_n5000_seed17 | 81.10% | 20,128 | 30,256 | 90,768 | — |
| fixed_n5000_seed17 | 80.72% | 20,000 | 35,000 | 90,000 | — |
| adaptive_n5000_seed29 | 80.28% | 20,128 | 35,256 | 105,768 | — |
| fixed_n5000_seed29 | 80.06% | 20,000 | 35,000 | 125,000 | — |

## Reading the sweep

Prefer the architecture that reaches an equivalent behavioral region with fewer unique labels. Endpoint accuracy is secondary. Delay-16/24 columns test whether the same sample budget also purchases a more persistent memory trace.

Do not compare adaptive and fixed runs by weight labels alone: adaptive runs receive a separate labeled architecture set and that cost is shown explicitly above.
