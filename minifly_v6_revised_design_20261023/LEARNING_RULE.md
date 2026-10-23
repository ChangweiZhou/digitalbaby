# Local recurrent eligibility candidate for V6 R1

This is an unimplemented rate-neuron/local-stochastic-learning proposal. It is not an e-prop implementation, an established fly mechanism or a proven way to learn arithmetic.

## State and initialization

Let H=88, h in [-1,1]^88, and x a one-hot vector over 259 observed symbols: 256 arbitrary bytes, BOS, end-of-input and observed EOS. Define `u = concat(x, h_previous, 1)`, of dimension 348. There are two weight matrices, `Wv` and `Wg`, each 88×348. Their columns over raw bytes are learned sensory embeddings; no ordinal distances or semantic classes are supplied.

Initialize h to zero. Draw nonbias entries of both matrices independently from `Normal(0, 0.05²/348)` with the paired birth seed. Content biases are zero; gate biases are `log(0.1/0.9)`. The latter is a generic initial tendency to retain state, not an assigned neural role. All neurons have identical rules. Reset h, two eligibility matrices and two accumulated-gradient matrices at BOS. Retain weights, native value state and one scalar reward baseline across records.

For each newly observed symbol, using the weights frozen at that record's start:

```text
mu_i = Wv_i · u
p_i  = sigmoid(Wg_i · u)
g_i  ~ Bernoulli(p_i)
v_i  ~ Normal(mu_i, sigma²), sigma=0.1
h_i  = (1-g_i) * h_previous_i + g_i * tanh(v_i)
```

These per-neuron gates neither rank memory items nor normalize across workspace slots. The same transition is reused for every prompt and reply position. There is no positional carry state, pointer, queue, content lookup or arithmetic action vocabulary. Only the available byte and previous neural state determine the transition distribution.

## Local score traces

For a transition that consumed an observed symbol, accumulate:

```text
Ev_i += g_i * (v_i-mu_i)/sigma² * u
Eg_i += (g_i-p_i) * u
```

Use no trace decay inside the bounded record: lambda=1. Clearing at BOS bounds its lifetime; maximum input/output lengths bound its magnitude and cost. A proposal with a closed gate is unused, so its content score is set to zero. The Gaussian and Bernoulli score identities must be checked independently before use in a native learner. Store current traces, not a history of states or a backward computation tape.

After h has been formed, it defines the native PN input and current KC code. A future observed byte supplies feedback to the prediction already made at that code. For actual payload byte b, y is its eight transport bits encoded as ±1 plus EOS-off; for observed output EOS, only EOS-on is eligible. There is no prediction loss for BOS/end-of-input protocol controls.

With native neutral-subtracted margins m and temperature 2, compute

`L = mean_over_eligible_channels(logaddexp(0, -y_c*m_c/2))`

`r = -min(L,20)/20`.

This bounded signal is a function of the actually supplied token and the native prediction. It contains no numeric difference between predicted and correct sums. It is also richer than a whole-answer 0/1 judgement; this first design does not claim scalar-only learning.

Let b be a scalar moving baseline from **previous completed records**, initially zero. Freeze `b_start=b` at BOS, independently of every random action in the new record. Accumulate

```text
Gv += (r-b_start) * Ev
Gg += (r-b_start) * Eg
```

Only traces of already consumed symbols are eligible for the current prediction's reward. Do not first consume the current target and then credit its own state transition. Current native teaching uses the same pre-target code and original gradual residual; the observed token subsequently advances h for the next prediction. All target/reward calculations occur outside the symbol-to-code operation.

The baseline update is disabled along with controller/native plasticity in NO_OLD's OLD phase. The ignored matrices in FROZEN are storage-matched diagnostics and never influence its actions. No evaluation feedback updates any baseline, trace or weight.

At record completion, update the baseline with `b += (1/1024)*(mean_record_reward-b)`. Holding it fixed during the record prevents a baseline that depends on an earlier action from being subtracted against that action's accumulated score. Divide the combined G by the number of actual feedback events, then clip its joint Frobenius norm to at most one. Apply ascent:

`eta(n) = 0.001 * (1/16 + (15/16)*min(n/1024,1))`

`Wv <- Wv + eta(n)*Gv`, `Wg <- Wg + eta(n)*Gg`.

There is no weight decay or extra optimizer state in this initial candidate. The maximum combined update norm per record is eta(n). This bounds the intervention and preserves gradual development; its numerical constants are engineering hypotheses, not physiology or tuned arithmetic optima. No learning-rate search is conducted on held answers. Publish gradient clipping frequency, trace norms, gate entropy, baseline, actual weight movement and native write statistics.

The delayed record-end update keeps controller weights fixed while that record is observed. Native value state is still updated causally at observed events. Joint learning can change addresses between records: measure that drift explicitly and do not claim native association stability from stationary surrogate-key results.

## Generation and interpretation

During a scored answer, use the same stochastic transition law with learning disabled on a private transient view. Native payload bits and EOS signs produce every token. Do not substitute a deterministic mean controller after stochastic training and quietly treat the two as equivalent. Three prespecified evaluation random tapes are averaged, with no best-tape selection or majority-vote decoder. Generated output never becomes externally confirmed teaching.

Reset adaptation once according to the inherited record protocol, not differently for teacher and generated prefixes. Advance private query physiology consistently with its elapsed native time. A prototype may use one private native snapshot per generated sequence; copying it on every output byte is not an unreported optimization. Charge snapshot/decay/encoding costs and verify the persistent original is unchanged afterward.

The local score approach can carry credit through the probability of past state transitions without differentiating through hard KC competition. That is a design rationale, not a theorem about this full training procedure. Clipping, the evolving native readout, finite samples and high-variance delayed effects can prevent improvement. Finite-difference scores and a tiny enumerable stochastic test validate implementation only; performance must come from the full controlled addition assay.

No chunking, replay, scalar-reward actor critic, value network, prediction-error decoder or auxiliary supervised representation loss is added when this rule struggles. A replacement rule would be a new version with its own stated reason and test.
