# BANC neuron-level learning-core peeling report

Script version: `1.1.0`
Dataset: `doi:10.7910/DVN/7WTH1N`

## Non-negotiable representation invariant

**One BANC v888 root ID is one graph node. No cell type is merged.** Cell-type annotations are used only as biological labels / seed definitions / summaries.

## Biological seed counts

| category | individual neurons |
|---|---:|
| KC | 4,552 |
| AL_PN | 699 |
| MB_DAN | 317 |
| MBON | 104 |
| DPM | 2 |
| APL | 1 |

## Candidate structural cores

| scenario | nodes | seed-edge mass | PPR mass | seed in giant weak CC | min 3-hop motif retention | pass |
|---|---:|---:|---:|---:|---:|---|
| semantic_only | 5,675 | 0.323 | 0.562 | 0.875 | 0.247 | False |
| olfactory_networks | 8,841 | 0.840 | 0.743 | 0.889 | 0.326 | False |
| olfactory_plus_superior | 13,663 | 0.933 | 0.853 | 0.903 | 0.913 | False |
| ppr_budget_7000 | 7,000 | 0.848 | 0.828 | 0.901 | 0.946 | False |
| ppr_budget_8000 | 8,000 | 0.887 | 0.871 | 0.904 | 0.981 | False |
| ppr_budget_10000 | 10,000 | 0.934 | 0.920 | 0.907 | 0.990 | False |
| ppr_budget_12000 | 12,000 | 0.957 | 0.947 | 0.908 | 0.993 | False |
| ppr_budget_15000 | 15,000 | 0.976 | 0.970 | 0.909 | 1.000 | False |

## Decision

Smallest tested PPR budget passing: **None**
10,000-neuron structural hypothesis supported: **False**

If a <=10k PPR core passes, the next round should instantiate that exact individual-neuron subgraph with shared type-level dynamics but independent neuron states. If only >10k cores pass, do not force the 10k target; inspect which network/semantic shells enter between 10k and the first passing budget. If no tested budget passes, expand the budget ladder or reconsider the seed/universe definitions before dynamics experiments.

## What this experiment does *not* claim

A structurally retained subgraph is not yet a functioning learning system. The next dynamics experiment must keep every selected neuron as an independent stateful unit; sharing type-level parameter templates is allowed, node merging is not.

## Files

- `RUN_MANIFEST.json`: exact Dataverse file IDs/checksums and run parameters
- `SCENARIO_DIAGNOSTICS.csv`: graph-retention metrics
- `MOTIF_REACHABILITY.csv`: directed 1/2/3-hop motif preservation
- `NEURON_LEDGER.csv`: one row per individual neuron in the discovery universe
- `SEMANTIC_COMPOSITION.csv`: biological category composition
- `NETWORK_COMPOSITION.csv`: CNS-network composition
- `CELL_TYPE_SPLIT_DIAGNOSTICS.csv`: confirms same-type neurons may be split by keep/delete decisions
- `GLOBAL_SUPERCLASS_COUNTS.csv`: metadata / proofreading inventory
- `DECISION.json`: machine-readable structural decision
