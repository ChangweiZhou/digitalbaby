# E25: how well the sign sweep follows the true gradient in a GRU, an LSTM and the two-bank network. Predictions, written before any saved file is read

11 November 2026, 18:33 EST (clock). Coordinator. A measurement on E20-B's saved weights; nothing is trained. Diagnostic, evidence level E2. It was suggested by the audit session (its note of 11 November, 4.2) and adopted in review 9.101.

## Why

The working rule sends the error back through weight signs wherever it crosses a weight matrix. In the two-bank network a step back in time crosses none. In a GRU or an LSTM it crosses the recurrent matrix at every step, about twenty times in an item. No rule-trained GRU exists. Before one is born, measure what the sweep's update would be on networks that already compose.

## What is measured

`e25_alignment.py`, on the births and end-of-life files of E20-B's G0 (two-bank), G1 (GRU) and G2 (LSTM), seeds 73129, 93683 and 116089. For each file, on 256 items per panel, under answer targets and under every-byte targets:

- for every parameter tensor, the cosine between the sweep's gradient and the true gradient;
- under answer targets, for every state, the cosine between the sweep's error and the true error at that state, by steps back from `=`.

Birth files are read on mod-89 operations at k = 12. End-of-life files are read on mod-97 operations at k = 12 and on new instances of taught compositions. A cosine of 1 means the sweep's update points where the gradient points; 0 means it carries nothing of it.

The GRU and LSTM steps are written out so that each linear map can run exactly or by signs. The gates' own nonlinearities and products are differentiated exactly, as in the harness's cell. One scale per whole matrix, as the rule does for every layer.

## What I knew when writing this

- E20-B's results (review 9.101): the GRU and the LSTM compose from memory; the two-bank network does not.
- This script's fixtures, on untrained networks only (`exp45/E25_FIXTURES_<seed>.json`, three seeds, all pass): the written-out steps equal the library layers in output (5e-7) and in every gradient (2e-6, relative); the sweep leaves the forward pass and the top layer's gradient unchanged; the two-bank exact path equals the harness's own backprop exactly.
- On untrained networks, the state-error cosine at `=` and twelve steps back: GRU 0.73 to 0.74, falling to 0.52 to 0.57; LSTM 0.73 to 0.75, falling to 0.47 to 0.58; first bank 0.72 to 0.74, falling to 0.59 to 0.64; second bank 0.74 to 0.75, staying at 0.73 to 0.76.
- No saved E20-B file has been read by this script.

## Predictions. End-of-life files, answer targets, mod-97 operations at k = 12. "Holds" means in at least two of three seeds

- **P25.1** In the GRU the state-error cosine twelve steps back is at most half its value at `=`. Confidence about 60%.
- **P25.2** In the two-bank network the first bank's state-error cosine twelve steps back is at least 0.8 of its value at `=`. Confidence about 65%.
- **P25.3** Twelve steps back, the GRU's state-error cosine is below the first bank's. Confidence about 65%.
- **P25.4** Twelve steps back, the LSTM's cell-state cosine is above the GRU's state cosine by at least 0.10. Confidence about 50%.
- **P25.5** For the first layer's weights (`l1.weight`, the deepest matrix all three networks share), the GRU's cosine is below the two-bank network's by at least 0.10. Confidence about 55%.

## Readings, fixed now (the first two are the audit session's)

- **The GRU's alignment at the longest gaps is at or above the two-bank network's** (P25.3 misses): a GRU birth under the sweep is worth its compute, at a reduced dose first.
- **It falls toward zero within a few steps:** the sweep as it stands is not expected to train a GRU through time. Remedies, cheapest first: keep the weight-free part of the step back (the gate value) and drop the terms that cross the recurrent matrix; learned feedback weights for the recurrent matrices, which Mr. Xu's ruling of 11 November allows; a finer scale than one per matrix. The measurement is repeated on whichever is tried before any birth.
- **In between** (lower than the two-bank network but well above zero at twelve steps): a reduced-dose birth is the test, and this measurement does not decide it.
- **The LSTM is clearly better than the GRU** (its cell-state cosine twelve steps back above the GRU's by 0.15 or more in all three seeds): that outweighs the 2.5-point margin by which the GRU was selected, and the candidate is reconsidered.
- **Every-byte targets:** reported beside the answer targets, with no prediction. If the per-matrix cosines are much lower under every-byte targets in all three networks, that is a lead for E24's reading and nothing more.

## What this cannot say

- Alignment at one point in training is not learning. A network trained by the sweep moves to where its own feedback is better aligned; these files were trained by backprop and have had no such chance. A low cosine here is a warning; a high one is a licence to try.
- Three seeds, one size, end-of-life and birth files only.
- Nothing about composition, retention or the target test.
