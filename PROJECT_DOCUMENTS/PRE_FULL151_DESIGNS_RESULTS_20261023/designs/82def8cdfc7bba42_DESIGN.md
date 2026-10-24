# V9.3 — joint-byte interface and sequence-value localization

**Status: design lock before implementation and confirm results.** This is a
diagnostic experiment on the already accepted V9.2 model, not a new learning
mechanism or a replacement for Full_151. It asks (A) whether byte prediction
and inherited value learning can run in one uninterrupted checkpoint without
unexpected interference, and (B) whether the inherited FE0→PN→KC value route
can bind reward to byte **order**. V9.2's predictor is not routed into value
memory; no result here will be described as transfer from the new byte reader
to the old value circuit.

## Locked parent and source boundary

- Load `../BYTE_CORE_V9_2/model_v92_text.npz` (SHA-256
  `ac79d7f4dcc28a131049b93d6b8390a2423bbe4208e43df1cb2d4052857b11c9`).
  It is trained on the V9.2 Austen/Wells stream only. Do not retrain or tune it
  on the two V9.2 held-out authors.
- The training chronology **continues** at the next byte after the saved
  pending text byte. No `reset_stream()` or `rest()` on the main learner, no
  per-record state clear. Feed each record's visible 14 bytes at the inherited
  `DT=30/14 s` spacing and start records 165 s apart. Advance the native clock
  through the gap with `native.flush`; the V9.2 last-four-byte predictor has
  no elapsed-time decay, which is an explicit model limitation.
- `Teach` follows the outcome byte at that same time. Frozen probes operate
  on checkpoint copies; the main checkpoint's state digest must be unchanged.
  Primary text probes continue from the copied current history and clock
  without reset. A reset-on-copy score is secondary V9.2 comparability only.

## Block A: coexistence and exact-route audit

Use four fresh F3/R2 worlds, namespace `BYTE-CORE-V9-3`, block `COEX`, indices
101–104. Construct each with `bytecore.create_world(namespace, block, index,
"R2", "massed")`; the old `v8.schedule()` has a different hard-coded
namespace and must not be used. Select and validate its first acquisition
block: 48 pairs × 6 bouts, both cues per pair, 12 context bytes,
outcome byte, visible newline, 165-s record starts. The stream, timestamps,
reward labels, and random order are identical in three arms, each starting
from the same text checkpoint:

| arm | byte predictor update | native Teach write |
| --- | --- | --- |
| joint | on, every byte | on |
| native-sham | on, every byte | off |
| predictor-sham | off, every byte | on |

Before and after training, frozen copies continue the **exact current history
and clock, without reset**, into 20,000 raw bytes separately from Doyle and
Stevenson, with full-stream/content BPB. Each author branches from a separate
copy. These authors were already inspected in V9.2, so this is a regression,
not fresh independent confirmation. Score the shared pre-state once, then
post-joint and post-predictor-sham per world. Native-sham's predictor state
must equal joint bitwise; spot-check one full native-sham text score, then
infer the redundant scores by the exact invariant. Immediately after the
block and after 24 h of **read-only native-time extrapolation** on copies,
probe both cues of each pair from separate copies of the same state; score
neutral-minus-aversive value margin and paired choice. No 24-h `rest()` or
history clear occurs on the main stream. These four worlds are an engineering
regression, not a population confirmation of the old retention effect.

The following are exact invariants, not statistical claims:

1. `joint.native` and `predictor-sham.native` remain bit-identical after every
   record; their value probes agree exactly. Predictor updates therefore do
   not change native teaching.
2. `joint` and `native-sham` have bit-identical predictor weights, bias, and
   byte probabilities on the same stream. Reward writes therefore do not
   change byte prediction.
3. The original V9.2 checkpoint is unmodified by all frozen text and value
   probes. Save/load of a joint trained checkpoint reproduces both heads.

If an invariant fails, stop: it is an adapter/timing defect, not biological
interference. Text BPB change shared by joint and native-sham is synthetic
stream drift; value change shared by joint and predictor-sham is inherited FE0
behavior. Report full absolute text scores and choice rates, not only deltas.
First require the no-reset *pre* continuation score on each author to beat
the exact train-only Markov-2 full-stream and content values in the saved
V9.2 confirmation receipt. If this fails, label the comparison
**baseline-incompatible**, report post-minus-pre, and make no operational
coexistence claim. Conditional on this eligibility, an **operational
byte-retention pass** requires post-joint full-stream and content BPB on each
author in **every** Block A world to remain below those same Markov-2 values,
all exact and clock checks to pass, and the pooled immediate
joint-minus-native-sham choice effect over the four worlds to be positive
(an engineering sign check, not inference). If joint and native-sham both
lose the text advantage, classify it as predictor drift under the synthetic
stream and do not claim practical coexistence. The four-world value results
are an engineering regression; report their mean paired
joint-minus-native-sham choice and every world without a population claim.
The prior FE0 retention experiment is a scale reference, not a fresh control.

## Block B: can native value memory use ordered bytes?

Use 16 fresh worlds, namespace `BYTE-CORE-V9-3`, block `ORDER-VALUE`, indices
201–216. Before running, each world deterministically samples three distinct
ASCII uppercase bytes from `A`–`Z` using seed `20260926 + index`. Label the
three bytes in draw order `(a,b,c)`. There are six ordered cues and three
unordered pairs. The aversive outcome (`Teach(1)`) is assigned to the cyclic
orientations `(a,b),(b,c),(c,a)`; reverse orientations receive `Teach(0)`.
Each symbol appears equally often in the first and last key position under
both outcomes. Every reversed pair has opposite outcome.

Each 12-byte cue is made only of visible raw bytes; train all six cues once
per bout in deterministic seeded random order for 12 bouts (seed
`20260926 + 10000*index + bout`). Each record is 12 context bytes, a constant
outcome byte `!`, and a visible newline at byte times `0..13` times
`DT=30/14 s`; record starts are 165 s apart. The first record begins at the
saved checkpoint's `native.pending_t + DT`. Both cue
positions use the *same* fixed 10-byte filler and constant outcome byte `!`
followed by visible newline; raw outcome bytes do not disclose reward. The
conditions differ only in cue position:

- **proximal:** 10 fixed filler bytes, then ordered `(x,y)`;
- **one-byte delay:** 9 fixed filler bytes, `(x,y)`, then one fixed suffix byte.

The fixed filler/suffix is ASCII space (`0x20`). In the delay condition the
last context byte is identical for every cue. The proximal condition tests
the most favorable FE0 recency geometry; the one-byte-delay condition asks
whether that order survives a common suffix. These are separate checkpoints
from the same saved parent, not a hidden task-mode signal to one learner.
A two-cue positive control uses 11 spaces followed by `a` or `b`, with
`a` aversive (`Teach(1)`) and `b` neutral (`Teach(0)`), and 12 presentations
per cue. `Teach` supplies an **external** reward bit while the outcome byte is
constant; this tests byte-cue association under reinforcement, not reward
learning from bytes alone. It checks that this exact
byte/timing/Teach harness can learn a simple cue difference.

For each condition and world run the three arms in Block A. Probe before
reward at the first record start, after 6 bouts and after 12 bouts at the
next record-start clock immediately following the completed bout, and at that
last clock plus 86,400 s by read-only native-time extrapolation. Probe cues
from separate checkpoint copies with no outcome or
teaching. Record KC-code Hamming/Jaccard separation for every reversed pair
at the actual pre-reward, 6-bout, and 12-bout probe states; also record native
value margin, paired choice (tie = 0.5), predictor
digests, clock checks, and probe purity. A code collision is a measured
representation failure, not a reason to drop a world. If a condition fails,
do not alter byte positions, alphabet, teaching dose, or thresholds in V9.3.

## Decisions and hostile-audit constraints

For each unordered pair, the margin is `value(neutral orientation) −
value(aversive orientation)`; paired choice is 1 for positive margin, 0.5 for
zero, and 0 for negative. A world choice rate is the mean over its three
unordered pairs (one pair for the simple-cue control). Use 95% t-intervals
over the 16 paired world differences (joint minus native-sham). These
intervals quantify variation across generated byte triples for one fixed
fly/checkpoint, not across biological specimens or model seeds. A valid assay
requires all source hashes, clock/accounting,
no-leakage, probe-purity, and exact-route invariants to pass. The simple-cue
positive control must have joint choice mean above 0.5 with 95% CI lower
bound above 0.5 **and** joint-minus-native-sham choice difference with 95% CI
lower bound above 0 after 12 bouts. Otherwise the order result is labeled
**assay-insensitive**.

No order pass may be emitted if source, clock, route, or probe-purity checks
fail; validity failure is a separate outcome. Primary order endpoint is the
one-byte-delay condition at 12 bouts: joint
choice CI lower bound > 0.5 and paired joint-minus-native-sham choice CI lower
bound > 0, conditional on the positive control. Proximal at 12 bouts and the
24-h extrapolations are secondary localization endpoints. Report all absolute
choices/margins,
including no-write and predictor-sham arms. If delayed order fails but
proximal succeeds, the supported finding is **position-sensitive value
binding**; code separation can help diagnose but cannot uniquely prove a
temporal-retention cause. If both order conditions fail while the simple cue
control passes, the bottleneck lies somewhere in native ordered-cue
representation, reward write/readout, or interference; raw code difference
alone is not proof of value-decodable geometry. If all pass, existing FE0 can
bind short byte order to reward;
this still does not show learned V9.2 predictor features driving value.

Do not enlarge the model, modify topology, add a protection mechanism, tune
learning rates, or promote A11/Full_151 on the basis of this experiment. A
predictor→value bridge would require a separately versioned fork, matched
route-off and shuffled-projection controls, and its own fresh worlds.

## Run discipline and compact output

First run structural/unit and one out-of-set smoke world (index 900) for
accounting only. Do not select parameters based on smoke performance. Then
run Blocks A and B with four workers and atomic per-world receipts so an
interruption can resume without mixing source versions. Preserve scripts,
source hashes, concise world metrics, final tables/figure, `REPORT.md`, and
`MANIFEST.json` in this folder. Reference unchanged V9.2 checkpoint and
source data; do not duplicate them. Target bundle <10 MB.
