# V191 · Recovered Round 4 replay versus V190-R1

The unmodified recovered relation program reproduced the original console summary. A second run added read-only per-pair logging to both relations, and its original stdout matched byte for byte. V190-R1 values come from its saved B-checkpoint receipts. V190-R1 was not retrained.

| Held relation after parity training | Round 4 correct first byte | V190-R1 correct first byte | Round 4 correct byte plus predicted newline | V190-R1 correct byte plus native EOS |
|---|---:|---:|---:|---:|
| Comparison | 140/144 (97.2%) | 133/192 (69.3%) | 140/144 (97.2%) | 133/192 (69.3%) |
| Parity | 143/144 (99.3%) | 114/192 (59.4%) | 143/144 (99.3%) | 114/192 (59.4%) |

First-byte exactness is scored identically. For the two-token comparison, both predictions must match the answer and the implementation's prescribed end symbol. The original performs one separate bigram-table lookup on its own answer to predict the following newline. It lacks the 257th native EOS class and an autonomous completion decoder, so its strict native-EOS result is unavailable. Every correct first byte in the V190-R1 panels was followed by native EOS. Changing the endpoint to byte-only does not explain the relation gap.

The held rosters differ. Round 4 has six seeds on 12 cyclic symbols, with 24 held pairs per relation at distances 2 and 10. V190-R1 has three worlds on 16 relabeled symbols and 64 held pairs per relation at distances 3, 6, 10, and 13. Both original parity held distances are even. Its 99.3% therefore tests the same-parity class only, which a constant same-parity shortcut would score at 100%. V190-R1 balances both parity classes. The original also uses 800 relation presentations per seed versus 640 per world in V190-R1. These are descriptive cross-roster comparisons, not a matched causal estimate for map updates or reliability weighting.

The original fact program uses a 256-byte vocabulary and only scores its first answer byte. It has no native EOS fact result. The portable replay changed only the absolute import paths for the original fact program and its helper. The original addition programs were also replayed. The byte addition implementation reached 0–10% held first-byte accuracy across its panels and 0% answer-plus-newline. The symbolic centered-conjunction half-table score was 11.3%, versus 21.0% with shuffled structure. The two sparse holdouts remained at 100% with both real and shuffled structure.

See COMPARISON.json for counts, the relation logs for the exact scored outputs, and the familiarity JSON files for the original fact metrics. The historical recovered files remain intact.
