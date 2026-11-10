# E19: the coordinating session's independent check of the worker's forward-only rule (10 November 2026, 13:40 EST)

Code read at f1f79b65 on `claude/vibrant-gauss-tops5f` (`arch19.py`, `exp19_kbank.py`), before any E19 birth reported. `e19_coordinator_check.py` writes its own forward pass from the brief's four equations (one-hot byte through the tables, plain tensor operations, not `arch19.forward`), differentiates it by autograd in float64 on development batches with every weight scaled by 3, and compares with the worker's `rule_grads`.

| Init seed | Rule against my autograd, max relative error over all parameters | Control arm (no deep traces): error on A | Control arm: error on every other parameter | Finite differences on three entries of A |
|---:|---:|---:|---:|---|
| 11 | 5.5e-16 | 0.873 | 5.5e-16 | 6.6e-09, 2.2e-07, 8.2e-09 |
| 22 | 5.7e-16 | 1.002 | 4.7e-16 | 4.1e-08, 2.1e-09, 7.7e-09 |
| 33 | 5.8e-16 | 0.776 | 5.7e-16 | 2.2e-07, 6.6e-09, 4.3e-08 |

Reading. The rule with exact weights is backprop through time to rounding on a forward pass the worker did not write. The control arm differs from the true gradient on the first bank's table alone, which is the one difference the brief asked for. By reading: the batched rule loops forward over steps and holds the two states and the traces; its only use of the whole item is to size the per-sample trace array to the bytes the item contains, and the streaming version (the worker's test 2) holds dense traces and sees one byte at a time. The optimizer matches the harness (`betas=(.9, .98)`, tables without decay). Nothing here is a result about learning.
