# MiniFly V9.3 — joint byte/value diagnostic

## Result

**The locked V9.3 diagnostic passed.** The same V9.2 checkpoint continued to
learn raw bytes while its inherited Full_151 value pathway acquired rewarded
associations. In four fresh 48-pair acquisition worlds, the immediate paired
choice rate was **66.7%** with value writes versus **51.6%** with identical
bytes and sham value writes (mean world difference **+15.1 percentage
points**). Both held-out author streams remained below the previously locked
train-only Markov-2 loss on full and content bytes after the acquisition
block. All clock, source, copy-purity, checkpoint, and exact-route checks
passed.

The primary order-value test also passed. With the same suffix byte after an
ordered two-byte cue, 12 rewarded bouts produced **87.5%** paired choice
versus **45.8%** in the same-stream no-value-write arm. The paired world effect
was **+41.7 percentage points [95% t interval +21.7, +61.7]** across 16 fresh
three-symbol worlds. The simple-cue positive control passed. The delayed
order condition began at 50.0% before teaching; thus its main result is not
an initial orientation bias. Its analog neutral-minus-aversive margin improved
by **0.0716 Hz [0.0490, 0.0943]** versus sham at bout 12. The margin is
small, and readout-noise robustness remains untested.

This is a **two-head coexistence and native order-binding result**, not
integration of their learned representations. V9.2's engineered predictor
updates separate weights; value learning comes from the inherited
FE0→PN→KC fast/slow route. Predictor-off and joint arms had exactly identical
native states and value scores throughout. The learned predictor weights
therefore supplied none of the measured value improvement.

## What was tested

The [design](DESIGN.md) was locked before the full run. Internal hostile audits
found and repaired initial ambiguities in the causal claim, reward-balanced
order task, continuation probes, source locking, and decision gates. The
corrected out-of-set smoke run passed before any of the four coexistence or
16 order worlds ran. The parent V9.2 model/source and Full_151 topology were
not modified.

The experiment loaded the already trained V9.2 text checkpoint and continued
from its saved byte history and clock. It did not reset between records or
between the text and reward streams. Each reward record contained 12 visible
cue bytes, one outcome byte and a newline at 30/14-second byte intervals;
records started 165 seconds apart. Reward was an external `Teach` bit. The
ordered-value task kept the outcome byte identical for every reward, so that
byte did not reveal the bit. The primary one-byte-delay task also kept the
last **cue** byte identical across orientations. Its cyclic three-symbol
reward assignment balanced labels by first and second key byte and reversed
the label when key order was swapped.

The frozen text probe likewise continued from copied live history without a
hidden reset. The common pre-block scores were 2.8097/2.7217 full-stream BPB
on Doyle/Stevenson. Across the four coexistence worlds, post-block scores
averaged **2.8337/2.7495** BPB, an increase of about **0.024/0.028**. All
four worlds remained better than the locked Markov-2 values **3.0010/2.9256**;
content-byte checks also passed. The identical byte stream without value
writes produced bit-identical predictor state. This slight loss is therefore
predictor drift from reading the synthetic acquisition stream, not reward
interference with the predictor.

| Order task at 12 bouts | Value write | Same-stream no value write | Paired effect, 95% interval |
| --- | ---: | ---: | ---: |
| Simple cue | 100.0% | 56.3% | +43.8 pp [+16.4, +71.1] |
| Adjacent ordered bytes | 87.5% | 60.4% | +27.1 pp [+9.7, +44.5] |
| Ordered bytes + common suffix | 87.5% | 45.8% | **+41.7 pp [+21.7, +61.7]** |

At the passive +24-hour readout, the primary delayed-order choice was 75.0%
versus 45.8% sham, paired **+29.2 pp [+11.0, +47.4]**. This extrapolates
the native state through time on copies; it does **not** expose it to later
learning. The delayed ordered cues had distinct pre-reward KC codes in all
48 reversed pairs, but they were very similar: mean Jaccard overlap about
0.936 and mean Hamming distance about 12 of 5,177 KCs. Code difference alone
would not have proved learnability; the rewarded-value comparison supplies
that behavioral evidence for this small trained task.

## Interpretation and next decision

We do **not** need to enlarge the model or modify its topology merely to make
it associate rewards with these short, repeatedly trained ordered-byte cues.
The inherited FE0 path already does that while useful byte prediction remains
available. This reduces the immediate search space: a new memory mechanism
is not a prerequisite for basic byte-order/value binding.

It does not settle the persistent-learning core. Each order world trained all
six orientations 12 times; unseen-symbol generalization was not tested. The
assay uses one fixed fly with varying generated triples, four engineering
coexistence worlds, and a small load. It does not test protection of older
native memories while new byte-conditioned associations arrive, the A11/full
life, noisy choice, or a predictor→value bridge. The native delayed-order
margin is modest, so the next test should assess robustness and interference
before spending memory on a larger topology. A bridge would be a separate
versioned mechanism experiment with its own route-off and shuffled controls.

## Audit trail and files

- `final/ACCEPTANCE_CHECKS.json`: machine-readable locked verdict.
- `final/coexistence.csv`, `final/order_aggregate.csv`,
  `final/order_worlds_bout12.csv`: final metrics.
- `final/FIGURE.png`: key comparison.
- `results/SUMMARY_FULL.json` and 20 per-world receipts: complete scores and
  exact checks; `results/SOURCE_LOCK.json` includes Full_151 native B/Q,
  environment, protocol, and source hashes.
- `joint_assay.py`, `test_experiment.py`, `summarize.py`: executable assay,
  qualification tests, and result renderer. The V9.2 checkpoint and datasets
  are referenced in `MANIFEST.json`, not duplicated.
