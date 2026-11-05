# MiniFly research note: Reassessing the U-turn and proposing end-to-end structural plasticity

**Date:** 2026-11-05  
**Status:** Research proposal and interpretation. No structural self-improvement claim established; no experiment or implementation authorized by this note.  
**Repository context:** DigitalBaby / MiniFly; relationship to FULL151-derived sensing, R_center, V181/V216, and earlier post-U-turn mechanisms.

## Executive assessment

The earlier U-turn **conceptually and experimentally disentangled** memorization, representation, and generalization on bounded assays. It did **not** establish that three independently successful modules had become a single persistent learner. The earlier interpretation overstated this distinction.

In particular, memory and representation are interacting components of computation, whereas **generalization is primarily a behavioral property** measured on unseen cases, not a separately solved module. Subsequent integration experiments revealed the dependence of memory accessibility on representation coordinates and event-addressing, and the dependence of answer generation on interfaces and supervision.

A promising next direction is not another task-specific QA module, and not indiscriminate free-form code rewriting. It is **a fixed learning program capable of slowly changing its own realized computational graph under a bounded resource budget**, alongside faster working-state, associative-memory and synaptic-weight updates.

The central principle:

> **Fix the structural-learning rule and resource budget; allow the instantiated network connectivity to evolve as a consequence of experience.**

This is an experimentally testable architectural proposal, not evidence that MiniFly can already improve its own structure.

## 1. What the first U-turn actually established

Three meanings of separation must not be conflated:

1. **Conceptual:** memorization, representation construction and transfer are different explanatory problems. Established as useful research decomposition.
2. **Experimental:** bounded interventions on representation, memory write/read and relational readout can identify component-specific benefits. Several positive and negative comparisons support this.
3. **Integrated and persistent:** a single continuously learning system coordinates these processes through changing environments, learning representations and memory states. **Not established.**

Examples:
- V193's reused relational readout after structural-map changes supports a bounded interface, not complete independence under continued corruption.
- V206 improves causal access by retaining earlier context, but its anchor and write timing are supplied.
- V211C finds that a new object event about the same person can displace access to a still-valid location on the recovered direct memory path, even when unrelated cross-entity lexical collisions are reduced. That directly exposes **address/update interference**, not necessarily loss of the underlying observation from all conceivable storage.
- V216 uses a fixed FULL151-derived FE-PERM/PN-to-KC sensory front end and V181's small learning network, **not** the complete FULL151 learning machinery or R_center's original eight-store centered-write mechanism. Successes across those lineages are not automatically cumulative capabilities of one learner.

Corrected claim: the first U-turn isolated important failure mechanisms and discovered useful primitives. It did **not** solve their mutual coordination or establish generalized lifelong learning.

## 2. Why representation and memory become coupled again

Even the simple readout `y_hat = W phi(x)` changes as

```
Delta y_hat = Delta W phi + W Delta phi + Delta W Delta phi.
```

If a learned representation changes coordinates from `phi` to `Q phi`, preserving the same output function requires `W_new = W Q^{-1}` when `Q` is invertible. Merely retaining the memory weights does not retain the mapping from observations to answers. Noninvertible changes are harder.

In associative address systems, new features can change which events collide, which entries are retrieved, and which writes overwrite still-valid information. Thus the goal is **controlled co-adaptation across learning timescales**, not perfect independence.

The error exposed by the architecture history is different learners being built and independently scored: these do not show that one persistent model learned all the capabilities.

## 3. Distinguish four kinds of change

Define four different roles:

- `h_t`: short-lived inferred/working state for the current environment;
- `M_t`: fast, revisable associative memory of useful observations;
- `theta_t`: reusable computational parameters for prediction, transition and readout;
- `A_t`: sparse connectivity and allocation of computational resources.

A candidate joint loop is

```
h[t+1]     = F(theta[t], A[t], h[t], observation[t])
M[t+1]     = U_M(M[t], h[t], observation[t], observed_feedback[t])
theta[t+1] = U_theta(theta[t], local_traces[t], observed_feedback[t])
A[t+1]     = U_A(A[t], sampled_structural_choice, local_utility, observed_feedback_history)
```

These are **interface requirements**, not a complete proven algorithm. Each actual update must specify its available information, update times, and causal loss signal.

Important distinction: **the world's relation graph is not the learner's synaptic graph.** Acquiring and releasing an object should ordinarily update `h_t` or `M_t`, not spawn a permanent synapse for each named object/person pair. Slow structural adaptation should change a generic computation that becomes useful across many episodes.

## 4. Proposed minimal architecture

Maintain a stable baseline sensory/address route—possibly the validated FULL151-derived FE-PERM/KC encoder—as an initial reference coordinate system for episodic memory. Add a bounded adaptive sparse recurrent route, with the same fixed input/output and observation interface in all controls. Use a learned ordinary readout; there must not be an evaluator-supplied task selector.

This is *not* a proposal to restore the complete FULL151 or all R_center stores. R_center is a historically verified bounded taught-association system with a restricted prompt and output interface; it is a candidate/reference, not an assumed solution for autonomous byte/world-state generation.

Keep neuron count and **number of active synapses** fixed. Permit sparse one-edge-out/one-edge-in modifications in one predeclared recurrent connection family. A fixed topology control and a random-rewire control must have the **same recurrent pathway, parameters, and compute budget**.

A stable memory coordinate path prevents arbitrary reinterpretation of previously written keys during exploratory structural rewiring; it does *not* itself solve semantic write selection or address collisions. Those remain separately testable problems.

## 5. Concrete structural-learning hypothesis

For a structural slot of postsynaptic neuron `j`, sample presynaptic source `I` from candidate set `C_j` using

```
pi(j,i) = exp(alpha(j,i)) / sum_{k in C_j} exp(alpha(j,k)).
```

Include the present source as an explicit option. Log the chosen proposal **before** observing the outcomes that judge it. After a bounded subsequent experience window, update structural preference scores by a score-function estimator:

```
Delta alpha(j,i) = eta_s * (R - b) * (1[I=i] - pi(j,i)).
```

Here `R` is **observed future utility**, e.g., negative loss on actual predictions or lawful action consequences; `b` is a causally available baseline. The proposal is sparse and the utility window is bounded. The learner cannot access hidden simulator graphs, correct answers before observation, privileged event labels, or held-out validation scores as a live learning signal.

This is a legitimate proposed local stochastic structural update, not proof of an efficient or low-variance credit rule. Important risks:
- utility signal confounded by unrelated weight updates during the trial;
- very high estimator variance when many synapses change at once;
- preference for easy byte-format prediction if that dominates the loss;
- loss of rare but still-useful competencies through turnover;
- degradation during exploratory rewiring.

Start with one structural slot/family and sufficiently infrequent proposals; compare with an identically resourced random policy. Record the actual number of changed edges and any structural search overhead.

Fast synaptic weights may separately use eligibility-based local plasticity, `Delta w_ji = -eta_w L_j e_ji`, but writing this generic equation is not an implementation. Eligibility traces and feedback factors must be grounded in the forward dynamics and real prediction error. E-prop and reward-modulated rewiring are methodological precedents, not validated recipes for this MiniFly architecture.

## 6. One end-to-end continual lifetime

One learner continually:
1. receives observations, updates working state, and preserves still-valid memory;
2. predicts or acts **before** feedback is available;
3. receives real observable consequences;
4. updates fast associative memory and reusable weights locally;
5. occasionally samples and evaluates bounded structural changes against subsequent evidence.

No external "memorize now", "learn representation now", or "generalize now" task switch is delivered to the model. The functions remain **analytically distinguishable** because their state variables and timescales differ, but coexist in one chronological learning process.

Initially prefer externally scheduled experiences so all candidates see identical evidence and structural effects are not confounded with different policies; autonomous action selection can be tested in a later controlled study.

Measure loss and accuracy **during** the entire life and on matched read-only forks, rather than only at a final benchmark. Forgetting an obsolete fact is correct; preserving still-valid relevant knowledge and retaining new-task acquisition ability are the targets.

## 7. Required controls and scientific endpoints

Three indispensable arms (same neuron budget, active edges, causal inputs, local weight rule, memory access and output interface):

| Arm | Weight plasticity | Structural rewiring |
| --- | --- | --- |
| FIXED | Yes | Never |
| RANDOM-REWIRE | Yes | Matched rewiring frequency/budget; unrelated to observed utility |
| LEARNED-REWIRE | Yes | Experience-dependent sampled structural policy |

Also compare a **frozen-post-pretraining topology** fork to continued structural adaptation, without changing the initial learned weights. This distinguishes the quality of the previously acquired graph from the value of ongoing structural plasticity.

Primary measures:
- prequential predictive loss or real decision consequences, recorded before feedback;
- improvement in new environment learning curves and genuine forward transfer;
- retention of still-valid earlier skills/facts, separately from obsolete facts;
- ability to learn late in the lifetime (plasticity);
- compute, model bytes, active edges, actual synapse turnover and structural search budget.

The test family must not consist solely of repeated encodings of the same storybook. Test both new entity identities under shared transition rules and **new transition laws requiring adaptation**. Never infer transfer from better performance on fixed prompt templates or mostly nonanswer byte compression.

Scaling should run **in parallel** with mechanism diagnostics: a negative short run is not an automatic veto, but a positive large run must exceed FIXED/RANDOM controls and cannot be established by simply adding more capacity.

## 8. Boundaries and staged decision

This is **not yet a verified self-improving architecture**. The proposed structural score-function may not produce enough useful credit signal to outperform random rewiring. Structural novelty is not itself intelligence. A continuously changing graph might just amplify the problems seen in V212/V213.

A first deliverable should specify:
- the exact candidate recurrent connection family and local state/update rule;
- stable memory-address compatibility across a connection swap;
- how performance signals are localized in time and how structural proposals are isolated from ongoing parameter changes;
- a small causal experiment and matched fixed/random controls;
- an experience scaling curve with meaningful consequences, not only routine record bytes.

If LEARNED-REWIRE does not improve cumulative loss, held-structure transfer or late-life plasticity under matched resources, do not adopt it merely because it produces a dynamically changing network. Report the negative result.

**Suggested next decision, not execution authorization:** Evaluate an end-to-end fixed-rule structural-plasticity design in a small shared-world dynamics task, then increase experience and environment diversity with the same learner program.

## References and source scope

Related methodological precedents (not direct validations):
- Kappel et al. (2018), *A dynamic connectome supports the emergence of stable computational function of neural circuits*, eNeuro: https://www.eneuro.org/content/5/2/ENEURO.0301-17.2018
- Bellec et al. (2020), *A solution to the learning dilemma for recurrent networks of spiking neurons*, Nature Communications (e-prop): https://www.nature.com/articles/s41467-020-17236-y
- Mocanu et al. (2018), *Scalable training of artificial neural networks with adaptive sparse connectivity inspired by network science*: https://arxiv.org/abs/1707.04780
- Dohare et al. (2024), *Loss of Plasticity in Deep Continual Learning*, Nature: https://www.nature.com/articles/s41586-024-07711-7

Prior project anchors:
- [Experience-driven strategy](MINIFLY_EXPERIENCE_DRIVEN_CONTINUAL_LEARNING_STRATEGY_20261104.md)
- [V213/V214 mechanism and transfer review](MINIFLY_V213_V214_MECHANISM_AND_TRANSFER_REVIEW_20261104.md)
- [FULL151 version census](FULL151_VERSION_CENSUS_20261023/MEMO.md)
- [V211C interference result](../minifly_v211c_object_interference_diagnostic_20261104/REPORT.md)
- [V216 campaign](../minifly_v216_experience_pretraining_design_20261104/campaign_20261104/REPORT.md)
