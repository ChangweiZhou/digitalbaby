# MiniFly recurrent state and worked examples qualification

**Three engineering cycles passed with the revised rendering.** The rehearsal exercised all 24 histories with a reduced dose. The native scores below are diagnostics from that dose.

## Cycle 1 Fixtures and controls

The teaching and held streams were checked in two engineering blocks under both laws and both phases. Each fixture had 512 distinct feedback prefixes and 64 presentation keys with two unreinforced edge-row orders each. The oracle passed 1,024 checks on literal renderings of the full held mask sets under opposite laws. Shuffling preserved the multiset of trace targets and avoided pairing a teaching key with its own trace.

The first renderer used a fixed semantic order for the actor introduction rows. That produced exactly three distinct worked traces per fixture, and 1,762 of 4,096 shuffled targets were byte-identical to the true target. The expected count under random pairing across different keys was about 1,796. The overlap came from the small trace vocabulary.

The revised renderer randomizes actor introduction order across teaching keys and fixes it across all renderings of each key. The E2 probe still changes only edge-row order. The revision increased the number of distinct traces per fixture to 17–25 and reduced shuffled matches to **1,124/4,096**. Matching nonzero traces fell from **750** to **112**. The all-zero trace remains common and should be reported separately from informative cases. [Fixture receipt](CYCLE1.json) · [Trace audit](SHUFFLE_AUDIT.json)

## Cycle 2 Model and loss parity

Both variants started with identical tensors and 33,281 trainable scalars. Changing the auxiliary response bytes changed the supervised worked-example loss and left the answer-only masked loss bitwise unchanged. Future target bytes did not change earlier logits within the checked tolerance. Both paths completed finite updates and produced distinct outputs from their respective context calculations. [Model receipt](CYCLE2.json)

## Cycle 3 Training and restart

Each history received 128 A pairs followed by 16 B pairs. The 24 histories used **3,456 updates and 425,088 delivered symbols**, plus two restart verification updates. The ordered-window and recurrent models both scored read-only probes before and after B. A fresh process reloaded the recurrent worked-example checkpoint and matched the uninterrupted next-update model hash. The checkpoint includes optimizer state. The rehearsal used 14.1 worker CPU seconds including its probes. [Rehearsal and restart receipt](CYCLE3.json)

| Model | Teaching | A taught exact | A held pairs both exact | After B taught A exact | After B held A pairs both exact |
| --- | --- | ---: | ---: | ---: | ---: |
| window | answer | 4/32 | 4/32 | 0/32 | 0/32 |
| window | shuffled | 4/32 | 4/32 | 6/32 | 6/32 |
| window | worked | 4/32 | 4/32 | 6/32 | 4/32 |
| recurrent | answer | 3/32 | 3/32 | 5/32 | 4/32 |
| recurrent | shuffled | 4/32 | 4/32 | 2/32 | 3/32 |
| recurrent | worked | 3/32 | 3/32 | 1/32 | 1/32 |

Each cell pools eight already-exposed keys per history over two births and two laws. The held columns require both fresh row-order variants to be exact with native EOS. This sample was used to verify the scoring path. The full 2,048 and 8,192 A doses, model scoring on the exhaustive E3 panel, and the matched inconsistent-law branch remain pending.

## Reproduction

From the project root, use the PyTorch environment. Run the final revised renderer with these commands.

```sh
runner=output/minifly-recurrent-worked-examples/qualify_worked.py
scratch=$(mktemp -d)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -B "$runner" 1
python -B "$runner" 2
python -B "$runner" 3 "$scratch/resume.pt"
python -B "$runner" resume "$scratch/resume.pt"
```

[Engineering source](qualify_worked.py) · [Experiment design](../minifly-whole-repo-audit/MINIFLY_RECURRENT_WORKED_EXAMPLES_EXPERIMENT_DESIGN_20261106.md)
