**Iteration 3: interventions before anatomical integration**

Declared before executing the new comparisons, 13 September 2026.

Primary source: Rachad et al., Cell Reports 44, 115593 (2025), deposited as mmc5.pdf. Author code commit 3cb2686e0f57f19cb79e5e166d5303794cd3023c. Secondary constraints: Weber et al. larval teaching pathways and Sun et al. adult structural plasticity. Specimens and developmental stages remain separate.

1. Execute the original functions for the 18 published stage-by-cell blocking conditions in each of the two circuit versions. Import only function definitions from run.py to avoid its plotting/demo side effects. Do not modify the source. Preserve raw zero-denominator readouts as undefined and also record the authors' plotted-zero convention.
2. Construct an independent implementation that groups KCs only when their complete permitted odor activation vectors and initial weights agree. Retain group multiplicity in every current sum and per-neuron weights in plasticity. Require rate and weight agreement with the original 2,000-KC model within 1e-10 across all conditions. This is conditional exact model reduction, not anatomical neuron subsampling.
3. Evaluate a deliberately coarser odor-only grouping and matched-budget uniform sampling with inverse-inclusion weighting; report their errors. These are compression controls, not connectome rewiring nulls.
4. Challenge the model with 1, 3, 6 and 9 second-order trials and a novel odor alone after punishment training. Report whether it captures the observed nonmonotonic trial-count trend and whether its hypothetical global cotransmitter state causes changes at never-punished inputs. These are out-of-scope stress tests of the published fit, not new animal data.
5. Audit FlyWire v630 retention of PPL101, PPL103, MBON11, MBON13 and MBON02, their specified feedback paths, and the KC populations feeding gamma1 versus alpha-prime-2. Preserve the distinction between alpha2 and alpha-prime-2. Do not import hemibrain neurons or synapses into FlyWire.
6. Update the learning contract from these results. Do not implement a new biochemical structural-plasticity or cotransmitter mechanism as if it were established by the papers. Keep all earlier negative results and reproduction discrepancies.
