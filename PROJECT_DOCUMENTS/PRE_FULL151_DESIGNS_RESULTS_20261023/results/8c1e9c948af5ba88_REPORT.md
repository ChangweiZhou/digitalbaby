# Fly learning-core connectivity audit

Version: 2.0.0

**Structural audit only. No neural training or learning-performance claim.**

## Decision

ORIGINAL_99PCT_GATE_UNATTAINABLE_IN_THRESHOLD5_UNIVERSE

Audit weak-edge and universe-boundary exclusions before revising the connectivity criterion. Increasing the node budget inside this same universe cannot satisfy the original 99% gate.

The previous decision is preserved verbatim in DECISION.json. The 99% gate is not relaxed.

## Fixed-membership threshold comparison

| Threshold | Graph | Nodes | Seeds in legacy giant | Full-graph seed ceiling | Isolated seeds | Selection-induced isolates | Retained seed pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | full_universe | 31,043 | 98.960% | 98.960% | 48 | 0 | 100.0000% |
| 1 | ppr_budget_10000 | 10,000 | 98.890% | 98.960% | 59 | 11 | 99.8576% |
| 1 | ppr_budget_15000 | 15,000 | 98.907% | 98.960% | 49 | 1 | 99.8932% |
| 3 | full_universe | 31,043 | 96.793% | 96.793% | 181 | 0 | 100.0000% |
| 3 | ppr_budget_10000 | 10,000 | 96.617% | 96.793% | 192 | 11 | 99.6362% |
| 3 | ppr_budget_15000 | 15,000 | 96.740% | 96.793% | 185 | 4 | 99.8908% |
| 5 | full_universe | 31,043 | 90.925% | 90.925% | 515 | 0 | 100.0000% |
| 5 | ppr_budget_10000 | 10,000 | 90.696% | 90.925% | 528 | 13 | 99.4967% |
| 5 | ppr_budget_15000 | 15,000 | 90.925% | 90.925% | 515 | 0 | 100.0000% |

## Reading the diagnostics

- Legacy giant: largest weak component by total neurons, matching v1.1.
- Full-graph seed ceiling: most seeds in any full-graph component divided by all seeds. No induced subgraph retaining all seeds can exceed this.
- Selection-induced isolate: seed has a non-self neighbor in the full graph at this threshold, but none in the fixed core.
- Retained seed pairs: fraction of seed pairs connected in the full graph that remain connected in the core. This avoids component-ranking switches.
- All mandatory seeds are retained, including unproofread seeds. Self-loop-only seeds are distinguished from seeds with no incident edge.
- Thresholds are applied to original edgelist row counts before duplicate aggregation, exactly as in v1.1.
- Self edges are excluded from neighbor/strength columns and reported separately.
- External partner categories describe edges crossing the original discovery universe; they do not show that these edges reconnect components.
- Core memberships come from the previous ledger. PPR and motif retention are not rerun or claimed at new thresholds.

## Files

- SCENARIOS.csv: full graph and both fixed cores at all three thresholds.
- SEED_EXCEPTIONS.csv: exact IDs and degrees for seeds outside the legacy giant, isolated, or separated from their full-component largest seed fragment.
- SEED_GROUPS.csv: all seeds summarized by biological category, hemisphere and proofreading status.
- EXTERNAL_BOUNDARY_SUMMARY.csv: excluded partner categories and synaptic mass; incoming/outgoing directions refer to the seed.
- SEED_FRAGMENTATION.csv: full components split into multiple seeded fragments by selection.
- SEEDED_COMPONENTS.csv: sizes and seed counts of every component containing a seed.
- REFERENCE_CHECKS.json: numerical reproduction of the archived threshold-5 core metrics.
- RUN_MANIFEST.json and DECISION.json: immutable input provenance, settings and limited interpretation.

Reference checks passed: 20/20.
Seed exception rows: 2,302 (the same seed can appear in multiple thresholds/scenarios).

No experiment input data, full ledger, or neural checkpoint is included in the results ZIP.
