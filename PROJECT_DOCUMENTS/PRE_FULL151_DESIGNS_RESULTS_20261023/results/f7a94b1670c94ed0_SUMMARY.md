# V54 — context-dependent teaching

Completed primary seeds: 16 / 16
Verdict: **`NO_CONTEXT_TEACHING_GAIN`**

| arm | U | old A | new D | whole bpb | revision8 | gain mean |
|---|---:|---:|---:|---:|---:|---:|
| `ALL` | 0.2773 | 0.2773 | 0.7734 | 3.8511 | 0.8984 | 1.0000 |
| `TARGET-ONLY` | 0.9023 | 0.9062 | 0.9902 | 7.9228 | 0.9688 | 0.0714 |
| `CONTEXT-GAIN-1` | 0.1875 | 0.3633 | 0.1953 | 4.3517 | 0.1875 | 0.7322 |
| `CONTEXT-GAIN-2` | 0.3770 | 0.4023 | 0.4668 | 4.2257 | 0.3516 | 0.7800 |
| `CONST-BLOCK-MATCH` | 0.1875 | 0.1875 | 0.5703 | 4.0747 | 0.7578 | 0.7322 |
| `SHUFFLED-BLOCK-MATCH` | 0.1719 | 0.1719 | 0.5801 | 4.0853 | 0.6719 | 0.7322 |
| `SURPRISE-BLOCK-MATCH` | 0.2539 | 0.2539 | 0.8223 | 4.2282 | 0.9062 | 0.7322 |

## Primary contrasts

- CONTEXT-GAIN-1 − ALL, U: {'n': 16, 'mean': -0.08984375, 'sd': 0.10877813333417093, 'low': -0.14779530053377954, 'high': -0.03189219946622045}
- CONTEXT-GAIN-1 − CONST-BLOCK-MATCH, U: {'n': 16, 'mean': 0.0, 'sd': 0.08912187348419766, 'low': -0.0474796780987063, 'high': 0.0474796780987063}
- CONTEXT-GAIN-1 − SHUFFLED-BLOCK-MATCH, U: {'n': 16, 'mean': 0.015625, 'sd': 0.090571104663684, 'low': -0.032626756009577644, 'high': 0.06387675600957765}
- CONTEXT-GAIN-1 − SURPRISE-BLOCK-MATCH, U: {'n': 16, 'mean': -0.06640625, 'sd': 0.09606516343087124, 'low': -0.11758496581779665, 'high': -0.01522753418220335}

- Update-norm confound: **True**

A positive primary result is conditional on the nondeployable INVARIANT-CUE representation. Raw-interface transfer is a separate stage.
