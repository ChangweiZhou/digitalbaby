# Experiment 3 (rung 3): working state for multi-step computation

**6–7 November 2026 · 48 runs plus an archived first attempt · three seeds · predictions P1–P4 stated before launch**

**Question.** Which structure lets a learner hold one operand across a run of distractor bytes, and can a local life rule use that structure?

**Answer.** A gated recurrent state (GRU) does it: developed by backprop, it answers 84–94% of held sums at every distractor length it was trained on and 82–88% at twice that length. An ungated state degrades with length; a stateless window is at the floor beyond its window, by construction. In life, the local rule works on this structure **only if credit flows through time**: the sign rule with credit through time reaches 68–86% on a held-out modulus, the same rule without it collapses to 4–13%, and even the true gradient without credit through time loses 25 points and fails on long runs. The within-step local rule that sufficed for stateless networks in Experiments 1 and 2 is not enough for a recurrent one. The biological reading is that the life rule for a working state needs synaptic eligibility traces.

## Task and setup

Records are `<code> a a + D1 … Dk b b = c c EOS`, with k uppercase-letter distractors between the operator and the second operand, k drawn uniformly in 0..12 during training and fixed at 0, 2, …, 12, 14, 16, 20 at test (14–20 is extrapolation). Development on mod 97 and mod 89 (codes `r`, `q`), half the pairs of each; life on held-out mod 73 (code `s`), 25% of pairs taught. Sealed panels: 300 held pairs at each k with seeded distractors.

Three architectures share the byte embedding, a 10-byte window front end, a hidden readout and the 257-way head: `win` (no state, 62,097 parameters), `rnn` (Elman tanh state, 78,609), `gru` (gated state, 142,737). Every linear map runs through one function that uses the true weight forward and, in local mode, the sign of the weight scaled to its mean magnitude backward: sign-symmetric feedback for any architecture. "No credit through time" means the previous state is detached at each step.

Development: backprop with full BPTT, AdamW lr 0.001, decay 0.3, 24,576 updates. Life: 12,288 updates, AdamW lr 0.001, decay 1.0 for sign-rule arms and 0.3 for backprop arms.

**Objective change and archived attempt.** The first run trained on every position of the record, as the earlier rounds did. With up to 12 unpredictable distractor letters per record (ln 26 nats each), the answer signal was swamped: the window and Elman nets never left their initial loss, every within-step arm stayed at 1–2%, and only the GRU learned. A k=0-only check confirmed the harness itself was fine (window net 87% held after 2,048 updates). All runs were repeated with the loss restricted to the two answer bytes and EOS in every arm. The first attempt is archived under `exp3/fullloss_attempt/`.

Qualification (`exp3/QUALIFY.json`): the autograd sign-feedback path reproduces the manual local rule of the earlier rounds exactly on all seven parameter tensors of the window net; the window net's output is provably unchanged by the first operand once k ≥ 6; distractor letters are disjoint from digits, operators and codes; every architecture trains finitely in both modes.

## Development: sealed held exact by k on mod 97, final, three seeds

| Arch | k=0 | k=4 | k=6 | k=8 | k=12 | k=14 | k=16 | k=20 |
|---|---|---|---|---|---|---|---|---|
| win | 54 · 0 · 69 | 31 · 0 · 37 | 1 · 0 · 1 | 1 · 0 · 1 | 1 · 1 · 2 | 1 · 0 · 2 | 1 · 1 · 1 | 1 · 0 · 2 |
| rnn | 67 · 72 · 64 | 77 · 87 · 72 | 71 · 86 · 70 | 65 · 81 · 68 | 56 · 73 · 65 | 50 · 68 · 61 | 45 · 61 · 59 | 34 · 48 · 49 |
| **gru** | **84 · 88 · 86** | **89 · 93 · 92** | **88 · 94 · 90** | **88 · 93 · 90** | **87 · 93 · 89** | 86 · 91 · 88 | 86 · 88 · 87 | 84 · 84 · 82 |

## Life on held-out mod 73, 25% taught: sealed held exact by k, final, three seeds

| Arch | Arm | k=0 | k=4 | k=8 | k=12 | k=16 | k=20 |
|---|---|---|---|---|---|---|---|
| gru | warm, backprop with BPTT (ceiling) | 86 · 91 · 88 | 87 · 90 · 87 | 88 · 91 · 87 | 86 · 91 · 86 | 83 · 91 · 85 | 82 · 91 · 84 |
| gru | random, backprop with BPTT | 85 · 79 · 87 | 90 · 86 · 87 | 90 · 85 · 87 | 87 · 85 · 85 | 86 · 84 · 82 | 85 · 83 · 80 |
| **gru** | **warm, sign rule, credit through time** | **74 · 70 · 68** | **86 · 83 · 82** | **83 · 82 · 81** | **81 · 78 · 75** | 78 · 75 · 65 | 71 · 71 · 49 |
| gru | warm, true gradient, no credit through time | 63 · 60 · 59 | 64 · 66 · 66 | 69 · 67 · 65 | 61 · 58 · 62 | 43 · 50 · 54 | 29 · 40 · 34 |
| gru | warm, sign rule, no credit through time | 13 · 4 · 7 | 11 · 5 · 9 | 11 · 7 · 8 | 11 · 6 · 7 | 10 · 4 · 5 | 7 · 3 · 6 |
| gru | random, sign rule, no credit through time | 1 · 1 · 1 | 2 · 2 · 1 | 1 · 1 · 2 | 1 · 1 · 2 | 1 · 2 · 2 | 1 · 2 · 1 |
| rnn | random, backprop with BPTT | 33 · 72 · 73 | 32 · 74 · 81 | 31 · 72 · 77 | 29 · 66 · 74 | 26 · 58 · 70 | 26 · 47 · 65 |
| rnn | warm, sign rule, credit through time | 1 · 52 · 49 | 1 · 70 · 68 | 1 · 67 · 71 | 1 · 58 · 64 | 1 · 50 · 53 | 1 · 38 · 42 |
| rnn | warm, sign rule, no credit through time | 31 · 7 · 6 | 25 · 8 · 7 | 27 · 7 · 8 | 23 · 7 · 7 | 19 · 7 · 6 | 15 · 7 · 7 |
| win | random, backprop | 14 · 11 · 6 | 6 · 6 · 3 | 0 · 0 · 0 | 0 · 0 · 0 | 0 · 0 · 0 | 0 · 0 · 0 |
| win | warm or random, sign rule | ≤10 | ≤6 | ≤2 | ≤1 | ≤1 | ≤2 |

Shortcut floors on the sealed panel: frequency 0.7–1.0%, second operand alone 0%, exact lookup 0%, no-wrap half 49–51%. Full tables with k=2, 6, 10 are in `exp3/SUMMARY.md`.

## Scorecard

| Prediction | Outcome |
|---|---|
| P1 Window net at the floor once the first operand leaves the window | **Held**, in development and life, and verified by construction (zero logit change at k ≥ 6). |
| P2 Developed GRU ≥80% at k ≤ 12; Elman <50% at k ≥ 10 | **GRU held** (84–94%). Elman degrades but more gracefully than predicted (62–78% at k=10, 34–49% at k=20). |
| P3 Warm GRU keeps ≥70% under the within-step local rule; random birth <30% at k ≥ 6 | **First half refuted**: the within-step sign rule collapses the warm GRU to 4–13%. Second half held (1–3%). |
| P4 GRU extrapolates to ≥50% at k 14–20 | **Held** in development (82–91%) and for every life arm with credit through time (49–91%). |

## What the diagnostics say

Two things are removed when the life rule is "within-step sign feedback": the true backward weights, and the gradient through the previous state. Separating them on the warm GRU:

| Removed | Held at k ≤ 12 | Held at k = 20 |
|---|---|---|
| Nothing (backprop, BPTT) | 86–92% | 82–91% |
| Backward weights only (sign rule, credit through time) | 68–86% | 49–71% |
| Credit through time only (true gradient, state detached) | 58–69% | 29–40% |
| Both | 4–13% | 3–7% |

Each costs on its own, and together they compound into failure. The sign approximation alone costs 5–15 points, as in the stateless experiments. Truncating credit at one step costs 20–30 points at trained lengths and most of the extrapolation, because learning to hold an operand across a run is exactly a credit-through-time problem. With both removed the recurrent weights receive a signal that is both misdirected and shortsighted, and the readout cannot compensate.

The Elman net shows the same ordering where it learns, with one seed dead under every sign-rule arm (a repeat of the random-birth instability seen in Experiment 2).

## What this settles

1. **The structure to borrow is a gated working state.** A GRU developed by backprop holds an operand across distractor runs and extrapolates to twice the trained length. The ungated state does not; the stateless window cannot.
2. **The within-step local rule is not enough for working state.** It sufficed for stateless networks in Experiments 1 and 2. For a recurrent state it fails even from a developed birth. Any life rule for the working-state rung must carry credit through time.
3. **Sign feedback with credit through time is a usable life rule.** 68–86% on a held-out modulus at trained lengths, 5–15 points under backprop, with the backward weights replaced by their signs. What it still needs, to be local in time as well as in space, is an eligibility trace: a per-synapse memory of recent presynaptic and postsynaptic activity that the error signal multiplies when it arrives. That is the e-prop construction, and it is the mechanism biology uses at exactly this point (dopamine acting on a synaptic eligibility trace). The next experiment is to implement the trace and see how much of the 68–86% survives.
4. **Answer-only loss is required once the stream contains unpredictable bytes.** Training on every position buried the signal under distractor noise and froze two of three architectures. The project's own designs already use answer-only loss; this is why.

## Evidence level and limits

Development result, three seeds, one harness, no independent confirmation. Parameter counts differ across architectures (the GRU has 2.3× the window net's), so the architecture comparison is not parameter-matched; it is a comparison of what each structure can do at a size that costs minutes. The GRU's extrapolation was tested to 20 distractors only. The Elman net's results are seed-sensitive.

## Reproduction

```sh
python3 exp3_state.py qualify
for arch in win rnn gru; do for s in 73129 93683 116089; do python3 exp3_state.py develop $arch $s; done; done
for arch in win rnn gru; do for arm in W R B; do for s in 73129 93683 116089; do python3 exp3_state.py life $arch $arm $s; done; done; done
for arm in T F G; do for s in 73129 93683 116089; do python3 exp3_state.py life gru $arm $s; done; done
for s in 73129 93683 116089; do python3 exp3_state.py life rnn F $s; done
python3 summarize_exp3.py
```
