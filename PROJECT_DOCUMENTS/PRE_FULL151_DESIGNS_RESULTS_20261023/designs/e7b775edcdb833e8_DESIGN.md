# MiniFly byte core V9.2 — bounded context extension

**Lock scope:** This is a versioned successor to V9.1, selected only with the
V9.1 DEV result and the Carroll DEV excerpt. The inherited FULL_151 source and
A11 anchors remain unchanged. The fresh controlled worlds 101–103 and the two
held-out authors are reserved for CONFIRM; their scores did not select this
architecture or its rates.

## Why this change

V9.1 passed two balanced 256-cue identity worlds and two 256-pair order worlds,
but the author-separated DEV prose score was 3.177 bit/raw byte versus a
train-only Markov-2 score of 3.015. A train-only context check on the same DEV
file found Markov-3/4 scores of 2.714/2.648, while fifth- and eighth-order
scores rose to 2.669/2.761 as exact contexts became sparse. This isolates a
modest, testable 3–4-byte context opportunity; it does not call for a large
generic language model.

V9.2 retains the unmodified FE0→PN→KC and inherited fast/slow value route in
one checkpoint. It replaces V9.1's predictor with 256 last-byte cells and two
fixed hash banks each for the preceding 2, 3 and 4 raw bytes. The banks have
256, 512 and 512 cells respectively, totaling 2,816 presynaptic rows and 256
output cells. Each incoming byte is predicted before being fed; observed-byte
prediction error updates only active presynaptic rows. The local rates are
0.03 for last-byte and pair rows, 0.10 for three- and four-byte rows, and
0.001 for output bias. The 0.10 choice came from a four-value DEV-only pilot
(0.03, 0.10, 0.30, 1.00); all four values and scores are saved in
`PILOT_CONTEXT.json`. Predictive mutable storage is 2,884,608 bytes, plus the
inherited FULL_151 value state. These temporal cells and the 256-output
teaching rule are engineering additions, not fly-connectome observations.

## Assay and decisions

Use V9's unmodified causal generators, full-factorial balance, raw-byte
timing, train-only n-gram references, frozen evaluation checks, and controlled
acceptance thresholds. No per-record reset, task/slot input, hidden reward or
future byte enters the model. The model receives ordinary bytes one at a time;
the scorer alone knows target positions. DEV uses worlds 11/12 and Carroll;
CONFIRM uses worlds 101/102/103 and separately scores Doyle and Stevenson
from the same frozen trained checkpoint. The Austen/Wells training stream
mixing and source hashes are inherited from V9.

Controlled CONFIRM succeeds only if all three fresh worlds satisfy the V9
identity and order gates, including the order-swap counterfactual and matched
no-predictive-write control. Prose usefulness follows the earlier
`BYTE_CORE_V9/TEXT_DECISION_RULE.md`: on **each** held-out author, full-stream
and Unicode-letter/number byte BPB must both beat train-only Markov-2 and the
identical no-write model. Report 8-gram and author-specific failures equally.
An empty or partial CONFIRM result has no success label.

The pure temporal DEV pilot reached 2.843 BPB on Carroll (versus V9.1 3.177,
Markov-2 3.015) and retained the controlled ID/ORDER task thresholds. This is
model selection evidence, **not** an independent result. Run the full V9.2
brain, its checkpoint and F3 regression in DEV before CONFIRM. If this full
model misses, change the architecture under a new version rather than tuning
against the reserved CONFIRM data.

Even a confirmed pass would establish a functional engineered raw-byte
sequence route on this narrow corpus, not general NLP capability or a
fly-derived prediction mechanism. The inherited fast/slow value synapses do
not generate the byte probabilities. A later joint-task preservation assay
must establish whether this fork is a viable replacement for Full_151.
