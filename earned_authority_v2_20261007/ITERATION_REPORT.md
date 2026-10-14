# Earned authority V2 — iteration outcome

7 October 2026. **The revised rule is technically sound but not worth a large confirmation run on present evidence.**

The original design's weakness was substantive: its binary behavior largely duplicated Q_HALF, and its word “earned” described no learning mechanism. V2 replaces it with a bounded past-feedback observer: the model keeps a 64-key reinforcement-support cache and separate helpful/harmful half-weight counts for supported and unsupported addresses. It sees no held-out answers or evaluator familiarity flags. All causal no-write histories suppress this added learning too.

The revision was audited independently and prototyped before any native run. Tests passed. Replaying archived trajectories then showed that the selector mostly retains ERROR, rather than discovering when less private authority helps reuse.

| Final binary causal prior-learning effect | Result |
|---|---:|
| ERROR | +3.82 pp |
| Q_HALF | +8.68 pp |
| Learned V2 | +4.17 pp |

V2 changes only one of 288 W held-out responses in the 48 archived binary worlds. Historical W taught retention/acquisition/revision remain exactly ERROR. The one available Latin development world has zero causal held-out effect. No universal inference follows from that single world.

The mismatch is now explicit: ordinary feedback rewards single-cue digit prediction, whereas the binary reuse output is a relative two-option choice. Legal feedback can train the observer, but its usefulness for that different action has not been established. This global help/hurt calibration retained ERROR on these histories. A fixed half-weight on unfamiliar addresses retains the known binary gain, but that is an ablation chosen in advance, not successful learned authority or evidence of Latin reuse.

## Verification delivered

- 19 prototype tests passed; 12 malformed observer snapshots and 8 tampered actual-write receipt cases rejected. These receipt checks are not new native write-path experiments.
- Exact ERROR/Q_HALF parity on 191,664 archived teacher records and 77,464 query rows.
- Independent arithmetic/counter/cache audit: 5,040 records, 1,624 query rows, 41 observer snapshots.
- 141,888 permitted observer updates; 49,776 disabled updates remained disabled.
- Synthetic 65-key eviction and uninterrupted/resumed observer-state equality checked.
- Actual Python observer approximately 3.6 KiB at full support capacity; logical packed layout 313 bytes. No native-system speedup claim.

## What did not happen

No formal science, new native trajectory, source-parent modification, repeated completed life, result-based threshold tuning, GitHub publication or core promotion. This is **new observer learning on historical records**, not zero learning and not fresh confirmation. The native worker/birth/checkpoint integration was not requalified because the replay did not justify building and launching it.

The main deliverables are [EXPERIMENT.md](EXPERIMENT.md), [RULE_LOCK.json](RULE_LOCK.json), [QUALIFICATION.json](QUALIFICATION.json), [replay/SUMMARY.json](replay/SUMMARY.json) and the independent audits in `evidence/`. The stop is intentional: a correctly implemented weak mechanism should not consume another large run merely because it can execute.
