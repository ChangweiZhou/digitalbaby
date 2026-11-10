# Experiment verdict (fill from immutable receipts)

Experiment ID: `<id>` | Prediction lock: `<SHA/time>` | Code: `<SHA>` | Data/held panels: `<hashes>`

## Preregistered outcome (never overwrite)

| Prediction | Literal criterion | Seed-by-seed result | Verdict | Notes |
|---|---|---|---|---|
| `<P1>` | `<...>` | `<...>` | SUPPORTED / REFUTED / MIXED / INCONCLUSIVE / INVALIDATED | `<...>` |

## Matched and unmatched comparisons

| Candidate | Comparator | Identical stream, labels, optimizer budget? | Metric | Inference allowed |
|---|---|---|---|---|
| `<...>` | `<...>` | `<yes/no + differences>` | `<...>` | `<causal bound/diagnostic only>` |

## Mechanism inspection

- Learned/acquired then forgotten vs never acquired:
- Input embedding reachability (per token class), no-gradient decay:
- Layer/state norms, write/read gates, trace-vs-exact error:
- Held intermediate-step exposure and shortcut ceilings:
- Whole worked answer versus family means:

## Provenance, cost, and integrity

- Seed count: independent `<N>`; identical computational reruns `<N>` (do not add)
- CPU-seconds and peak RSS per arm:
- Stored raw bytes, replay + index overhead, transient pending:
- All bugs/errata and invalidated readings:
- Process/machine lock + exit statuses:

## Decision

What remains established:

What is **not** established:

Next minimal causal test, prewritten falsifier, estimated cost and authorization status:
