# Exact special case and limits of the proposed theory

This is a source-specific E0 derivation, not a unified native learning law. No ridge/ELM decoder or other candidate is introduced.

For one V147 query context let x be its binary KC code, phi=x/sqrt(K), K=sum(x), and m_j=2 theta_j dot phi. The code is normalized, so its inner product with another phi is the cosine kernel in the memorandum. One observed signed target t_j has e_j=[tanh(m_j/2)-t_j]/2, and the executed naming update is

`theta'_j = theta_j - alpha s(n) e_j phi_s`.

With representation fixed during this event, the exact immediate query-margin effect is

`Delta m_j(q) = -2 alpha s(n) e_j <phi_s,phi_q>`

`             = alpha s(n) [t_j-tanh(m_j(s)/2)] K(q,s)`.

The nine output bits include EOS. This equation concerns margins at a fixed query/prefix, not full generated strings. The warmup factor uses completed prior records, as in the original source. The disposable numerical fixtures use already-warmed saved states, so they do not experimentally qualify the ramp again.

When LOCAL also changes the representation, the exact two-term identity is

`Delta m_j(q) = -2 alpha s(n) e_j <phi_s,phi_q>`

`             + 2 theta'_j dot (phi'_q-phi_q)`.

The first term uses pre-update teacher and query codes; the second uses post-update naming weights. Using phi'_q means this is a decomposition of an observed update, not an independent forecast of a future code jump. Numerical equality does not establish that drift typically harms learning, or that these few queries represent the population.

## Signed effects and query groups

For an evaluable target bit, define useful influence as `t_j(q) Delta m_j(q)`. A positive value moves the margin toward the correct target; an absolute-value cross-gain does not say whether the movement helps. Keep directly taught, related never-taught and protected OLD queries separate. Minimizing all cross-query effects can also suppress desired transfer.

For native stores, a documented margin convention must be oriented to an actual target before interpreting self/cross gain. The historical Capacity fixture used `-Delta margin/coefficient`; it is not already a correct-answer-oriented teaching effect.

## Native and temporal limits

Full151 includes state-dependent native dynamics. Its response cannot be replaced by the V147 formula merely because the same PN/KC asset is used. Finite write/no-write effects include downstream changes and need not add across events. The Jacobian product in the proposal is a local differentiable approximation; its reception/survival/access factors are not separately identified by the product alone.

A matched counterfactual must preserve sensory observations, clock, RNG/tape, counters and subsequent exposure, suppressing only the stated plastic increment. Endpoint W/S_NEW scores identify net NEW-write effects under that schedule; they do not reconstruct each event's operator.

## Output identity

For V147's single-payload-byte assay, the saved own-generation events give exactly

`P(complete) = P(correct first payload) P(immediate exact EOS | correct first payload)`.

This is an accounting identity within one generation protocol. Capacity DIRECT payload instead uses evaluator-known length on fixed key/slot responses, and its STOP output uses the saved slot tensor. Their ratio is not this conditional own-prefix probability.

## Prediction boundary

A final-query Delta is already a behavioral causal outcome. Using it to predict the same final outcome is circular. Final label alignment also consumes evaluation information. Forecasts must use earlier/calibration-only measurements and be compared with current margin/accuracy, elapsed time and later-write count baselines. All historical aggregates here were known; grouped retrospective validation is not independent prospective confirmation.
