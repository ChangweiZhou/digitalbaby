# Memo 02 — A mathematical jigsaw: operations, representations, learning and credit

**Scope:** The equations here are either reductions of committed code or explicitly marked deductions / proposed measurements. They are not new experiments.

## 1. What should count as a sufficient architecture?

Let H denote observed history; s(H) the full accessible state of the candidate; u a permissible future continuation. Distinct histories must remain distinguishable **only insofar as** the task requires different future behavior:

    s(H1) = s(H2)  =>  P(Y | H1,u) = P(Y | H2,u) for relevant u,

as a necessary condition for a deterministic state-based predictor (with fixed current parameters).

For a fixed trained representation Z=f_theta(X), data processing gives I(Y;Z | theta) <= I(Y;X | theta). Better learned features do not magically add information from X; they can make information linearly accessible, better conditioned, or invariant to nuisance variation.

For frozen Z, a useful decomposition of cross-entropy risk under the *same* data distribution is:

    R(q) = H(Y|H) + I(Y;H|Z) + approximation_error(F,Z) + optimization_excess(F,q).

This requires the evaluator's prediction function class F and an appropriate conditioning convention. The fitted external R1 is **not** H(Y|Z), and its failure does not prove information is missing.

## 2. Before adding another layer, reduce its actual algebra

At evaluation the V159 auxiliary predictor is linear in KC activity:

    r_t = M [x_t;1],    z_{t+1} = rho z_t + r_t.

For fixed learned M and no inter-line state, substitute:

    z_t = M_x sum_{j<t} rho^(t-1-j) x_j + b sum_{j<t} rho^(t-1-j).

This supplies a learned projection of a fading history, not a loop where **retrieved facts change subsequent internal queries**. V162's calibration selection of S alone is compatible with redundancy: marginal I(Y;Z)>0 does not imply conditional I(Y;Z|X)>0. In a finite fitted model the latter is not proven zero, even when no measured gain appears.

Before any new module: document whether it adds a genuinely state-dependent operation, merely recodes existing information, or changes optimization/regularization only.

## 3. What the successful V166 loop computes

V166's committed state equations, up to naming, are:

    h_t = tanh(W_x x_t + W_h h_{t-1} + W_r r_{t-1} + b_h)
    q_t = ReLU(W_q h_t + b_q)
    r_t = M_t q_t
    logits_t = W_o h_t + W_or r_t + b_o.

The essential edge is r_{t-1} -> h_t. It makes learned episodic content influence the *next query*. For idealized square memory, f(f(a)) corresponds to M^2 e_a; the bilinear interactions (M^2)_{ca}=sum_b M_{cb}M_{ba} express composition. V166 itself uses rectangular KC-like memory and learned transformations, not a literal square M^2.

The result is a bounded *computational witness*, not proof that human anatomy must be copied or that a fly-local learner can acquire the controller. Input markers, fixed study writes and BPTT training are important scaffolding.

## 4. Temporal credit required by the loop

Let D_t = diag(1-h_t^2) and P_{t-1} the active query-ReLU mask. The time-step state Jacobian, for the fixed memory within the relevant step, is

    J_t = D_t [W_h + W_r M_{t-1} P_{t-1} W_q].

The W_r M P W_q term couples distinct neural units through currently stored facts. V168's exact gradient controls confirm that the query at the first hop needs credit crossing two subsequent transitions; TBPTT-3 recovers ~97% of BPTT's observed Q2 gain, TBPTT-2 ~4.5%.

Exact recurrent adjoints satisfy

    lambda_t = grad_{h_t}(loss_t) + J_{t+1}^T lambda_{t+1},

with any direct memory-output dependence included consistently. The immediate loss term alone cannot generally teach a query whose consequence is assessed two steps later.

Forward sensitivities instead satisfy

    E_t = J_t E_{t-1} + B_t,
    B_t = partial h_t / partial theta.

This provides a non-BPTT *reference identity*, not automatically a synapse-local algorithm.

## 5. Why an unbiased gradient estimator can still fail

V168's KF-RTRL traces are unbiased in technical enumeration and reproduce exact BPTT when replaced with exact influence matrices. The trained KF model nonetheless remains close to TBPTT-1.

For unbiased ghat=g+noise and locally beta-smooth loss:

    E[L(theta - eta ghat)] <= L(theta) - eta ||g||^2
                              + beta eta^2/2 (||g||^2 + tr Sigma).

Thus a useful expected gradient may lose to large variance at finite step size. Actual AdamW and norm clipping are nonlinear operations:

    E[clip(ghat)] != clip(E[ghat]).

The report measured KF-BPTT gradient cosine ~0.23–0.44 for single estimates at initialization, versus ~0.39 for the biased TBPTT-1 estimator; four KF draws improved the mean cosine to ~0.59. These facts make variance and optimizer interaction plausible, **not yet separately identified causes**.

## 6. A candidate exact short-horizon influence baseline (not yet tested)

Because V168's decisive credit horizon is short, propose first checking the fixed-parameter sensitivity:

    E_t^[3] = B_t + J_t B_{t-1} + J_t J_{t-1} B_{t-2},

with direct query-output and augmented-state effects accounted for exactly. This is the forward sensitivity counterpart to preserving recent parameter influences for the appropriate TBPTT-3 window.

If each B_t is (or can be factored into a sum of) Kronecker terms, keep the **three recent terms** instead of compressing all history to a single stochastic factor. This gives a small deterministic *numerical reference*, not proof of a biologically local implementation.

**Potential special structure:** V166's memory has at most eight written cue/value pairs per line,

    M_t = sum_{i in S_t} e_{b_i} k(a_i)^T /16.

Hence rank(M_t)<=8, and the memory-mediated Jacobian term W_r M_t P_t W_q also has rank<=8. This is an algebraic bound from the implemented write equation, not a measured algorithm speedup. The ordinary W_h recurrence may be full rank, and low-rank memory influence alone does not make the complete RTRL sensitivity low rank.

Factor M_t=U_t V_t^T to test whether short memory-conditioned query influences can be propagated with less variance and fewer operations. Do not overclaim locality: a compact factorization may still require nonlocal feedback communication.

## 7. The representation-learning problem is different

V163 SSL-h features improved over a random new layer, but remained below the direct-KC baseline. V165/V167 improved by broadcasting a correctly matched output error, rather than optimizing an easy temporal contrastive task. Source data: V163 contrastive pair separation ~99.7%, alignment cosine ~0.019; V165 error-feedback alignment rose ~0.48/0.59/0.65 across three epochs.

**Hypothesis, not settled fact:** the SSL task can exploit mechanically overlapping fading histories and need not preserve distinctions needed for later output. Test nuisance-matched negative pairs and causal interventions on BP-H feature subspaces before inventing another self-supervised objective.

**Implementation caveat:** V163's gradient cosine compares nonidentical sets of positions (next-byte samples versus adjacent pairs) and raw gradients rather than actual Adam-preconditioned steps. Treat it as a clue, not a complete explanation.

## 8. Memory-conditioned feedback, not generic error amplification

For r=Mq, exact error to q includes

    partial L / partial q = M^T (partial L / partial r).

A fixed random feedback map trained successful feedforward features in V165/V167; it cannot, as a fixed linear operator, equal M^T d for every possible changing episodic M and d. That does not establish impossibility for a learned contextual controller, but explains why spatial DFA success alone does not solve the V166 query learner.

## 9. Additional measurement cautions

- V164 COLA computes time statistics from loss-normalized errors and denominators containing fixed eps. If raw errors scale by c, a ratio S/(Q+eps) becomes c^2 S/(c^2 Q+eps); numerical invariance must be checked before assigning a near-zero alpha entirely to task statistics.
- V168's KF is unbiased as a raw estimator; test actual **post-clipping, post-Adam** update alignment, not merely raw estimator cosine.
- A free evaluator readout over full-rank taught features demonstrates only finite interpolation; it does not show that the native writer can reach those weights or learn new relations.
- Not all weak transfer means old knowledge should be protected. The target is **relevance-selective retention**, not preservation of all previous examples.

## 10. Primary code anchors

- [V159 predictor](https://github.com/ChangweiZhou/digitalbaby/blob/7c2a4d94791adbf2b3ac8516b53acf1695d3cc0c/minifly_v159_pfeed_dev_v1_20261026/v159_pf.py)
- [V166 forward loop and episodic memory](https://github.com/ChangweiZhou/digitalbaby/blob/7c2a4d94791adbf2b3ac8516b53acf1695d3cc0c/minifly_v166_memloop_dev_v1_20261026/v166_lib.py)
- [V167 fully local source updates](https://github.com/ChangweiZhou/digitalbaby/blob/7c2a4d94791adbf2b3ac8516b53acf1695d3cc0c/minifly_v167_local_dev_v1_20261026/v167_local.py)
- [V168 KF-RTRL trace](https://github.com/ChangweiZhou/digitalbaby/blob/7c2a4d94791adbf2b3ac8516b53acf1695d3cc0c/minifly_v168_kfloop_dev_v1_20261026/v168_kf.py)
- [V168 technical identity tests](https://github.com/ChangweiZhou/digitalbaby/blob/7c2a4d94791adbf2b3ac8516b53acf1695d3cc0c/minifly_v168_kfloop_dev_v1_20261026/results/technical/CHECKS.json)
