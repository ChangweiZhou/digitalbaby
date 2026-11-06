# MiniFly ordered byte front end — three engineering cycles

**Disposition: engineering rehearsal complete. Hold the full run.** Both candidate graph panels admit a shortcut that solves every primary item. The three front ends ran through the planned update and checkpoint paths with the same 37,313 trainable scalars.

## Cycle 1 — Challenge the graph benchmark

Eight engineering blocks each contained 32 prompt pairs. Every pair shared the same byte counts and final eight bytes. Its two answers disagreed, and swapping the taught meaning of X and Y flipped both answers. The graph oracle passed all 256 quartets. Teaching witnesses uniquely identified the active opcode and rejected a two-hop cutoff. The two held rooted motifs were non-isomorphic and their path-length signature differed from all A teaching motifs.

A fixed serialized edge slot solved at most 13/32 quartets in any block. A topology-aware rule did much better. Ignore opcodes when identifying the unique shortest path, then read the opcode on its only edge shared with a longer path. This single opcode solved **32/32 in every block**, including the reversed motif. The original panel cannot clear the proposed four-item margin over this baseline.

I also tested one eight-edge motif with three-hop routes. Its 12 matched pairs retain equal byte counts, equal endpoint opcode counts, and the same final eight bytes. Exhaustive truth tables given oracle edge roles and active opcode meaning found maxima of **2/12** with one edge and **2/12** with two edges. A three-edge rule, parity on the active bits of one designated route, solved **12/12**. That same parity rule answered only **145/256** correctly when all assignments on the motif were tested. The best three-edge truth table for that broader set answered **199/256** correctly. Selecting only the flip pairs created the perfect parity shortcut. The eight-edge panel was evaluated separately and was never fed to the learner.

## Cycle 2 — Verify the encoder and update path

The harness reproduced native stock state after each of 24 updates and reproduced stock native generation on 32 held inputs. The ordered encoder separated all 32 held pairs. The order-removed encoder mapped both members of every pair to the same pre-answer input, and its active coordinate count matched the ordered encoder at each byte step tested. Both alternative encoders executed 24 real local updates. Changing the answer byte left all pre-answer features intact in both arms.

## Cycle 3 — Rehearse teaching, interference, editing and restart

One engineering block exercised three front ends under two opposite teaching histories. Each life received 1,024 A records and 64 B records, then forked into 64 edit and 64 matched no-edit records. The edit and no-edit branches shared literal prefixes and began at the same state. The ordered checkpoint reproduced the next update's full state hash bit for bit in a fresh process, including optimizer state and RNG. The 7,296 rehearsal training events took 85 CPU seconds, with roughly 337 MiB peak RSS for the macOS worker. The extra restart comparison used two updates. These measurements cover the small rehearsal and its scoring.

| Front end | Taught A exact | Held A exact | Held A correct first byte | Held A own EOS | Held A all four answers exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock | 1/64 | 0/128 | 2/128 | 67/128 | 0/32 |
| Ordered | 7/64 | 17/128 | 21/128 | 127/128 | 0/32 |
| Order removed | 9/64 | 0/128 | 0/128 | 116/128 | 0/32 |

The held counts pool both teaching histories. A quartet requires both prompts to be correct under both histories. The ordered arm produced more individually correct held answers and stopped at EOS more often, with weak taught acquisition and zero joint items. On the held A′ probes the ordered edit branch had 10/128 exact under the new law, and its no-edit branch had 11/128 exact under the old law. Those are separate endpoints. The rehearsal used the original six-edge panel and a single shared birth and was sized to exercise the implementation.

## What to change before admission

Keep the three front ends and the learner fixed. Replace the primary held panel with a mixture of rooted graph families and preregister matched flip pairs alongside invariant and route-interaction cases. Include assignments where the same designated route has identical active bits but another route changes the answer. Also include cases where that designated route changes and the answer stays fixed. On small motifs, enumerate every opcode assignment before selecting prompts and record the selected set’s accuracy for topology-aware one-, two-, and three-link truth tables, the parity shortcut, and the earlier controls. Use the resulting shortcut ceiling and the exact graph oracle to decide whether the locked endpoint is feasible before scheduling learner runs. The small engineering scores do not decide whether changing encoders is worthwhile at the full dose.

The full-dose run, fitted readout, resource admission, and independent review remain pending. The rehearsal generator uses engineering renderings, so its results do not qualify the proposed fixed 32-instance teaching schedule.

## Receipts and reproduction

- [Cycle 1 structural checks](CYCLE1_TEST.json) and [adversarial shortcut audit](SHORTCUT_AUDIT.json).
- [Cycle 2 encoder checks](CYCLE2_TEST.json) and [Cycle 3 raw native outputs and restart](CYCLE3_TEST.json.gz).
- [Engineering harness](qualify.py) and [shortcut audit runner](shortcut_audit.py).

From the project root, use the PyTorch environment and pass the local archive directory containing the V216 and V181 sources. The runners verify the archived dependency checksums against the V216 lock.

```sh
python -B output/minifly-ordered-front-end/qualify.py "$archive" 1
python -B output/minifly-ordered-front-end/shortcut_audit.py
python -B output/minifly-ordered-front-end/qualify.py "$archive" 2
scratch=$(mktemp -d)
python -B output/minifly-ordered-front-end/qualify.py "$archive" 3 "$scratch/ordered.pt"
python -B output/minifly-ordered-front-end/qualify.py "$archive" resume "$scratch/ordered.pt"
```
