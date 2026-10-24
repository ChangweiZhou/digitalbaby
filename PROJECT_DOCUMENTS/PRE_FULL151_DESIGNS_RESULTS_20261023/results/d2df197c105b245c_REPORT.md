# Full151 bridge step 2 — gain 0.05 does not clear the preservation gate

All 128/128 fresh worlds 9201–9328 completed, with no failure receipt.
Technical world 9199 reproduced the frozen FE0 route and the step-1
gain-0.05-online route exactly (maximum pair-margin error zero). The separate
receipt reducer checked fixture exclusions, source hashes, write counts,
clock metadata and all published choice arithmetic. This is receipt-level
audit, not an independent re-execution of native synapse dynamics.

| Immediate endpoint | FE0 | Gain-0.05 bridge |
|---|---:|---:|
| Taught W choice | 92.45% | 88.15% |
| Held-out W choice | 74.74% | 73.18% |
| Held-out N choice | 49.48% | 52.08% |
| Held-out W−N | +25.26 pp | +21.09 pp |

The predeclared conjunctive gate **failed / remained unresolved**:

| Gate quantity | Estimate | One-sided 95% lower bound | Required lower bound |
|---|---:|---:|---:|
| `D_T`: bridge minus FE0 taught W | −4.30 pp | −6.31 pp | greater than −5 pp |
| `D_H`: bridge minus FE0 held-out W−N | −4.17 pp | −10.11 pp | greater than −10 pp |
| `G_H`: bridge held-out W−N | +21.09 pp | +15.06 pp | greater than 0 |

Thus smaller gain restores a positive native teaching signal on never-taught
pairings, but it does **not** establish preservation of FE0's existing ability
under the margins fixed before this run. The pilot's apparent held-out gain
advantage did not replicate. Repeating or retuning on these 128 worlds would
turn them into development data and cannot repair this confirmation. The
result concerns immediate synthetic E1/E3 two-choice behavior, not delay,
byte-order-specific benefit, arithmetic, or autonomous output.

The next architecture should keep an untouched FE0 value branch and place
the temporal input in a separately testable branch. This is a structural
proposal, not a result of the current experiment. See [DESIGN.md](DESIGN.md)
and [results/SUMMARY.json](results/SUMMARY.json).
