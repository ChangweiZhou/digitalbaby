# V211C · Object-event interference diagnostic

**Disposition** Direct-path diagnostic completed. The Round 17 object learner and archived per-world receipts were absent from the inspected exports, workspace and published commit search. This run uses the unchanged recovered V211B learner. It cannot reproduce Round 17's event-time enrichment, second recall, or object answers.

Eight paired worlds ran with two vocabulary renderings each. There were 512 packets, 1,536 scored read-only probes, 512 restoration checks and 3,200 prior feedback questions. The paired runs held movement facts, event order, query timing and learner settings fixed. Pickup and drop sentences were controlled text perturbations. The harness did not simulate possession. No target feedback was supplied.

| Vocabulary | Exact before | Exact after other person's event | Exact after own event | Correct candidate present after own event |
|---|---:|---:|---:|---:|
| Syllable names | 252/256 | 244/256 | 0/256 | 5/256 |
| Low-overlap names | 256/256 | 256/256 | 0/256 | 10/256 |

The vocabulary intervention reduced mean summed pairwise entity-only hashed-row overlap from 199 to 12.25 per world (93.84%). Yet the actual query still shared about 57 to 58 address rows with its own person's object event, versus about 12 to 13 with the other person's. The low-overlap vocabulary did not rescue same-person availability. Its paired difference was +1.95 percentage points with 95% interval −8.11 to +12.02. It improved exact retrieval after the other person's event by +4.69 points, with 95% interval −3.09 to +12.46. Treat the latter as descriptive.

The direct path fails primarily before ranking. After a same-person object event, the valid location is absent on 497 of 512 probes. When present, it still never wins. The failure survives a large reduction in cross-entity lexical overlap. In the inherited writer, the object's sentence can replace rows shared with the person's movement. The same- versus other-person intervention exposes this displacement under an unchanged physical location. The Round 17 enrichment may change where the loss occurs, so the conclusion does not extend to that implementation without its source.

Next isolate preservation of still-valid location state when a newer event about the same person arrives. Establish candidate access before training a richer selector or delayed-credit mechanism. Obtain the exact Round 17 source and receipts for any archival reproduction or three-path audit.

Reproduce with Python and NumPy from this directory:
```sh
PYTHONHASHSEED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 python -B run.py
```
