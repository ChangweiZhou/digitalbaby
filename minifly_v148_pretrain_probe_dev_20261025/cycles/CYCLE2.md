# Cycle 2: dev test runs (25 Oct 2026)

Development text: *The Adventures of Buster Bear* (`burgess-busterbrown.txt`), at reduced size. Prefixes 0, 2.5, 5 and 10 KB; calibration 2 KB; held-out 4 KB. Nothing was scored on the main text.

## Run c2_dev (single pass): `results/c2_dev/RESULTS.json`

| Taught | MiniFly held bits/byte | MiniFly held accuracy | MiniFly taught (last 5 KB) bits/byte | Unigram held | Bigram held | PPM-3 held |
|---:|---:|---:|---:|---:|---:|---:|
| 0 KB | 8.006 | 0.000 | – | 8.006 | 8.006 | 8.006 |
| 2.5 KB | 7.035 | 0.107 | 6.080 | 4.537 | 3.666 | 3.454 |
| 5 KB | 7.001 | 0.103 | 6.217 | 4.496 | 3.592 | 3.227 |
| 10 KB | 6.945 | 0.117 | 6.110 | 4.468 | 3.539 | 3.006 |

**Reading.** MiniFly is worse than the unigram counter on held-out text, and also on the text it was taught. One pass barely stores the text: the G1 memory gate needed 24 passes for 84–88% recall of 80 facts (`minifly_cls_dev_20261017/REPORT.md`). Its greedy generations are mostly non-text bytes.

## Run c2_dev_passes (repetition axis added): `results/c2_dev_passes/RESULTS.json`
This is the same continuous history, followed by passes 2 and 4 over the full 10 KB. Two diagnostics were added: the probability mass on symbols never seen in the taught text, and how often the argmax is such a symbol.

| Checkpoint | Held bits/byte | Held accuracy | Taught first / last 5 KB bits/byte | Taught last 5 KB accuracy | Held mass on never-seen symbols | Argmax never-seen rate |
|---|---:|---:|---:|---:|---:|---:|
| 10 KB, pass 1 | 6.945 | 0.117 | 6.49 / 6.11 | 0.222 | 0.659 | 0.199 |
| 10 KB, pass 2 | 6.975 | 0.115 | 6.25 / 5.83 | 0.254 | 0.647 | 0.229 |
| 10 KB, pass 4 | 7.029 | 0.116 | 6.16 / 5.82 | 0.280 | 0.672 | 0.268 |

**Reading.**
* Repetition helps taught text slightly and does not help held-out text.
* About two-thirds of the probability goes to bytes that never occur in the taught text. The fixed 257-class C16 output code has no notion of which symbols occur, so this costs well over a bit per byte before context plays any role.

A smoke test of a further diagnostic shows the size of this effect. Renormalising the native scores over symbols seen in training, with its own temperature fitted on the calibration block, lowered held bits/byte from 7.18 to 4.96 on a 0.5 KB toy run. The smoke results were discarded.

## Decisions for the protocol (taken before the rehearsal and before any main-text scoring)
1. **Add the repetition axis.** Passes 1, 2 and 4 over the full 40 KB train pool are evaluated after the nested single-pass prefixes. This separates storing from generalising.
2. **Keep the primary metric unchanged:** held bits/byte, native 257-way softmax, one temperature fitted on calibration.
3. **Add two diagnostics, labelled as such:**
   * the never-seen-symbol mass and argmax rate;
   * bits/byte renormalised over symbols seen in training, with its own calibration temperature.

   The diagnostics say how much of the gap comes from the output code. They are not primary.
4. **The registered predictions (design section 7) stay as written before any code.** The dev data already suggest that prediction 1 ("between bigram and PPM order 3") will fail on the worse side. That expectation is recorded here, separately; the predictions are not edited.

## Cost
* Training: 0.036 CPU-s per event. A full evaluation checkpoint on 10 KB held-out plus calibration and taught slices takes about 210 CPU-s on this reduced protocol.
* Full protocol (40 KB, passes 1, 2 and 4, six checkpoints): about 2 CPU-hours per text.
