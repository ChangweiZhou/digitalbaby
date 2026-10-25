# Root cause of the unseen-addition failures: dev-only diagnostics (25 Oct 2026)

**Status: DEV_NOT_EVIDENCE.** These are evaluator-side diagnostics on 8 fresh development splits (`RC-D1`..`RC-D8`, the LEARNED-BANK orbit-split generator) with 2–10 seeds each. No main world was used, no native store, no own-byte/EOS generation. Answers are 19 sum classes. Each step was chosen as the cheapest test that could falsify the leading explanation, following the research-loop kit's explore stage. Every number below comes from the `.log` file next to its script, produced by running the committed scripts from this folder.

## Question
Why does every MiniFly route fail on held-out single-digit addition (V143 mod-10, V144–V147)? This includes exact-gradient references and the designs I proposed.

## Steps and results
| Step | Script | Question | Result |
|---|---|---|---|
| A | `a_determinacy.py` | Do the 40 taught orbits determine the 15 held answers? | Under an additive model (answer = u(a) + v(b)), 15/15 are determined in every split, each by a one-step rectangle witness. The data are sufficient given that bias. |
| B | `b_capacity.py` | Does limiting capacity create a preference for the rule? | No. One-hot symbol inputs, hidden width 1–256, exact gradient (diagnostic) and DFA (local): held ≤ 0.09. Orbit-consistent shuffled labels are learned about as fast as true labels: faster in most cells, with true labels ahead only at widths 1–2 under exact gradient at the last checkpoint (taught 0.34 vs 0.24; 0.61 vs 0.48), where held is still ≤ 0.09. Per-item error-driven learning shows no useful preference for the rule. |
| C | `c_search.py` | Width 1 started near the rule | Confounded: a single ReLU unit with a 19-way softmax cannot fit even from a near-rule start (taught ≤ 0.57). Superseded by C′. |
| C′ | `c2_bottleneck.py` | Clean 1-D additive state s = u(a) + v(b) with a fixed tuned bank and a delta readout | Error-driven credit from random starts memorises (held 0.08–0.46; shuffled labels largely fitted). A local *relational* rule (pull the states of taught prompts with the same answer together, push different answers apart) reaches held 0.78–0.80 from random starts and fits shuffled labels only partly (0.62–0.71). |
| D | `d_ablate.py` | Does the relational result need the 1-D constraint, or a number code? | No. With symbol-factorised codes it reaches held 0.80 in 1-D additive, 8-D additive and 32-D nonlinear (ReLU) forms. Through the FE-PERM `read()` trace with a linear map: held 0.43, taught 0.71. |
| E | `e_frontend.py` | Why is the FE-PERM trace worse? | `FEPerm.read()` returns `p / max(p)`. The raw trace `p` is exactly affine in the two operands (residual 0.0000); max-normalisation adds a 0.2260 non-additive residual. Relational rule on raw `p`: taught 1.00, held 0.80. On `read()`: taught 0.71, held 0.43. |
| F | `f_relu_raw.py` | Through a ReLU-32 layer (V143-style) on the trace? | Raw `p`: held 0.68 (shuffled taught 0.85). `read()`: held 0.39. |

**The 0.80 ceiling.** About 80% of held orbits have a sum value that occurs somewhere in training (`a_determinacy.log`). A lookup keyed by the converged state cannot name a sum it never saw.

## Interpretation (hypothesis supported by these dev diagnostics, not a confirmed finding)
1. **Learning signal.** All routes so far learn only from per-item output error: the store delta rule, the three-factor rule, DFA and the exact-gradient references. On a Latin-square task, memorising satisfies that objective as well as the rule does, and step B shows no preference. Generalising requires that taught prompts with the same answer come to share internal state. A relational (same-answer) signal supplies that requirement. On per-operand factorised inputs it forces the additive structure, so held prompts inherit it (steps C′ and D). The human analogue is acquired equivalence; the nearest catalog row is SYS "Retrieval-mediated memory integration" (B).
2. **Interface.** FE-PERM's max-normalisation distorts the exactly additive raw trace. Every MiniFly route reads the normalised trace (steps E and F).
3. **Excluded as primary causes by these diagnostics:**
   * insufficient data (A);
   * capacity (B);
   * a missing number-line or magnitude code (D: symbols suffice once the signal is relational);
   * lack of downscaling (V146).

**Supporting overlap measurement (canonical PN→KC code, V143 FROZEN path, 100 prompts).**

| Prompt pairs | Mean Jaccard overlap |
|---|---:|
| Sharing an operand (always different answers) | 0.384 |
| Same answer (never share an operand) | 0.261 |
| Other pairs | 0.259 |

Overlap is therefore concentrated among pairs that must have different answers, and same-answer pairs get no help. This is consistent with slow acquisition and no transfer under per-item error learning.

## Not shown
* Own-byte/EOS generation, the native store, main worlds.
* A MiniFly-specified local mechanism for comparing the current episode with a recent one (state and answer).
* Sensitivity of the push-term settings.
* With a ReLU layer, shuffled labels are fitted fairly well (0.85), so the preference is weaker there.

## Proposed findings (for the other agent family to verify and activate)
1. On single-digit addition with symbol inputs, per-item error-driven learning (exact gradient and DFA, widths 1–256) learns orbit-consistent shuffled labels about as fast as true labels and reaches held ≤ 0.09 (dev, class outputs).
2. A local same-answer relational rule on per-operand factorised codes reaches the sum-coverage ceiling (held 0.78–0.80) and fits shuffled labels only partly (dev, class outputs, nearest-centroid naming).
3. FE-PERM `read()` max-normalisation turns an exactly additive raw trace (residual 0.0000) into a non-additive one (residual 0.2260), halving the relational rule's held accuracy (0.80 → 0.43, linear map).
