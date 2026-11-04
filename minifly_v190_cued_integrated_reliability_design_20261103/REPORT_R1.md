# V190-R1 development result

Verdict: DID_NOT_PASS. Three worlds, 3504 training presentations and 44136 byte/EOS target sites. Both correction branches counted.

After-B scores are complete byte plus own EOS (byte-only in parentheses). Each relation has 64 held pairs. Each fact wrapper has 32 taught A keys.

| World | PREQ comparison / parity | REPEAT comparison / parity | UNIFORM comparison / parity | Cued facts, wrappers 2 / 3 |
|---|---|---|---|---|
| 0 | 47/64 (47) / 36/64 (36) | 47/64 (47) / 36/64 (36) | 39/64 (39) / 40/64 (40) | 18/32 (18) / 5/32 (5) |
| 1 | 40/64 (40) / 39/64 (39) | 44/64 (44) / 39/64 (39) | 37/64 (37) / 39/64 (39) | 10/32 (10) / 2/32 (2) |
| 2 | 46/64 (46) / 39/64 (39) | 42/64 (42) / 40/64 (40) | 38/64 (38) / 32/64 (32) | 11/32 (11) / 7/32 (7) |

This is a reconstructed local development test. The raw-byte frame has registered cyclic symbols and an explicit task cue. See RECONSTRUCTION.md. The original Round 4 implementation remains unavailable. The fixture-derived CPU projection is approximate.

Saved decisions and receipts record both correction branches, single-expert scores, mixed loss and evaluator-only references. This experiment does not establish arithmetic transfer.
