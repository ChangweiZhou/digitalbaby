# Review of the supplied generic-stream proposal

Status: source/document review only, 13 October 2026. No prototype or learning run.

The useful proposal is to test an unchanged generic recurrent byte learner on arithmetic-relevant observed streams, predict payload bytes throughout records, and make repeated exposure and regularization explicit. Engineering backprop is an appropriate feasibility reference under the user's existing authorization to change the core. It cannot be described as a local fly mechanism.

Corrections incorporated in the design:

| Supplied assertion or suggestion | Independent judgment and resulting design |
| --- | --- |
| Every previous core is an exact-context lookup | False for RAW-RULE's GRU and older FE0/role/REF routes. The current acquisition failure is acknowledged; A is retained. |
| Shared state is the only route to generalization | Shared learned representations are useful but this is not a theorem excluding other learners. No exhaustive mechanism exclusion. |
| Small capacity and forgetting make rules win | A hypothesis. A small GRU may memorize, weight decay may hurt acquisition/retention, and bias strength matters. Fixed decay contrast, no guarantee. |
| 0–9 is paradoxically easier | Unestablished here. Use 0–9 as the proposed larger table, with its cost/scope disclosed. |
| Repetition is permitted because it is data | Yes, but every presentation, byte and update is charged. Distinct information and total experience are separate. No “70 samples” claim after 17,920 presentations. |
| Predict every byte | Every observed payload/EOS, except input-only transport controls. Compare with a matched answer-only generic mask. No evaluator target or generated-guess training. |
| Counting can teach number order | Valid data hypothesis. Counting is explicit, paid arithmetic-relevant feedback; no isolated no-counting causal claim. |
| Addition speeds multiplication | A hypothesis requiring N_ADD and SH_ADD, matched multiplication doses, and total-prefix cost. Conditional transfer can be expensive overall. |
| A minutes-long prototype precedes safeguards | Timing is unknown. Design a finite exploratory DEV and measure a full technical job first; basic write/exposure/IO safeguards precede any training science. |
| Backprop reference, later local rules | Accepted as an engineered reference; no vague local-rule arm. Future local update equation must be frozen and matched. |
| Addition→multiplication completes the target | It does not. Design A retains four-operation testing; B records the route back to subtraction/division and then mixed expressions. |

Primary sources checked:

- [Power et al., Grokking](https://arxiv.org/abs/2201.02177): abstract-symbol binary operations using small transformers. Extensive optimization and dataset-size dependence make repetition a reasonable variable; they do not establish this decimal GRU curriculum.
- [Liu et al., Omnigrok](https://arxiv.org/abs/2210.01117): analyzes conditions associated with late generalization and its induction/removal. No blanket regularization guarantee.
- [Author code, Grokking in recurrent networks with attractive and oscillatory dynamics](https://github.com/keith-murray/grokking-recurrent-networks): modular a+b+c=0 mod 3. Recurrent structure learning is possible in that setting; task and output differ substantially here. The linked OpenReview forum was browser-challenge restricted during this check; this review relies on the authors' stated repository scope, not an unread paper.

The design is static. Architecture, optimizer, full-budget cost, source isolation, causal masks, long-record gradients and independent reducer remain implementation obligations. The 12-test confirmation family is prospective; no historical gate/result is changed.
