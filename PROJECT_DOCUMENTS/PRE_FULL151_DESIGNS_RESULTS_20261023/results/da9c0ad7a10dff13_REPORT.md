# Full151 bridge step 1 — exploratory engineering result

The repaired qualification passed exact FE0 and gain-0.5-online parent parity
on new technical world 9098. All 8/8 exploratory worlds 9101–9108 completed;
the receipt reducer passed. These worlds only selected the next candidate and
must not enter a confirmation estimate. The earlier v1 qualification attempt
ran no pilot world because of a JSON-list versus Python-tuple source-check bug;
its technical receipt and failure explanation remain in the v1 folder.

| Route | Immediate taught W−N | Immediate held-out W−N | Taught W accuracy |
|---|---:|---:|---:|
| FE0, online predictor | +43.75 pp | +16.67 pp | 89.58% |
| Bridge 0.5, online | +2.08 pp | +25.00 pp | 56.25% |
| Bridge 0.5, frozen | +33.33 pp | +16.67 pp | 93.75% |
| Bridge 0.05, online | +41.67 pp | +29.17 pp | 87.50% |
| Bridge 0.05, frozen | +41.67 pp | +29.17 pp | 91.67% |

For gain 0.05, the predictor signal changed the tested KC code on a mean
11.75/12 taught cues with online updating and 12/12 with the frozen predictor.
The gain-0.05 online route was selected for a fresh-world preservation test
because it retained online predictor learning while restoring the immediate
taught write effect in this small pilot. The pilot is too small to establish
held-out superiority or strict FE0 preservation.

See [DESIGN.md](DESIGN.md), [results/TECHNICAL.json](results/TECHNICAL.json)
and [results/PILOT_SUMMARY.json](results/PILOT_SUMMARY.json). This is E1/E3
engineering evidence on an externally prompted two-choice synthetic task;
there is no arithmetic or autonomous output result.
