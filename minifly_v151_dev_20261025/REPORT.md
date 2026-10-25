# V151 — native teaching-response operator: three-cycle results

25 October 2026. **Completed: native MiniFly executed the local teaching directions correctly in this bounded assay. B's weaker effective dose was real and a calibrated scalar largely removed it. This did not establish useful addition generalization or solve native acquisition/retention.**

No new core is adopted. No sequential training history, V150 trajectory replay, language campaign, or addition assessment was run. The three cycles are engineering diagnosis, bounded counterfactual testing, and independent artifact audit/revision—not three independent scientific replications.

## 1. Main result: the native response is working locally

Every scored A/B/B_MATCHED intervention reduced the same-context target loss: 24/24 sites per condition, including all 16 sites on the four trained A/B-final states. These are repeated probes on six anchors, not 24 independent worlds. The native compartment response was diagonal, positive in the expected signed-teaching convention, and very accurately predicted by the measured operator.

Across the 24 scored source sites, measured self gains ranged from 0.3832 to 0.9546. The largest compartment-to-compartment self-gain spread within a site was 0.000290. Off-diagonal response was zero. The largest discrepancy between the two perturbation scales was 4.45e-12. Even the full teaching interventions were predicted in raw native output to within **1.77e-13**. Loss predictions retained ordinary softmax curvature: the largest trained-site first-order loss residual was about 0.00218 bits.

Thus the tested native circuit was not generally unable to execute a descent signal, and its response had not collapsed in the sampled trained states. This clears a specific concern from the V150 audit; it does not establish behavior for arbitrary states, longer histories or new relations.

## 2. Matching B's displacement removes its large local disadvantage

Gain multipliers were selected only from two prespecified calibration sources per anchor, then locked before any scored intervention. They ranged **1.9635–1.9980**. Buster's empty-prefix source was a genuine null, so each Buster match used one effective non-null calibration source. No source was replaced.

Mean change in immediate target loss, in bits (negative is improvement; four scored sources per anchor):

| Frozen parent state | A_COMMON | B_COMMON | B_MATCHED |
|---|---:|---:|---:|
| Bryant newborn | −0.009782 | −0.004895 | −0.009780 |
| Bryant A-final | −0.143800 | −0.073103 | −0.144724 |
| Bryant B-final | −0.146610 | −0.074363 | −0.147313 |
| Buster newborn | −0.014483 | −0.007245 | −0.014460 |
| Buster A-final | −0.204164 | −0.105457 | −0.206463 |
| Buster B-final | −0.212207 | −0.108736 | −0.213289 |

On the four trained parents, equal-source mean improvements were **0.17670 bits for A, 0.09041 for B, and 0.17795 for matched B**. Matching restored almost all of B's nominal-dose disadvantage. Matched B had a slightly larger mean improvement than A in each trained parent, and at 14/16 trained sites. Its aggregate advantage over A was only about 0.00125 bits (0.7% of A's loss reduction).

Measured matched-B/A displacement ratios at the trained scored sites ranged **0.9912–1.0136**. Calibration transported closely, but not perfectly: the small B-versus-A advantage cannot be treated as an exact equal-displacement or statistically confirmed superiority result. No gain was retuned on scored outcomes.

**This supports a dose explanation for the one-event gap. It does not establish that gain was the principal cause of V150's entire training failure.** A long learning history changes the competing memory state and query distribution.

## 3. Better probability did not repair trained-state answers

Before intervention, none of the 24 selected source targets was the model's top choice. After one intervention, each condition selected the correct next class on **8/24** sources: the eight newborn cases. The trained states remained **0/16 correct**, despite improving their target probabilities on all 16. The two newborn EOS sources became correct EOS choices; none of the four trained EOS sources did.

These are single-symbol responses immediately after feedback, not complete own-byte/EOS sequences. They show that a functioning write and a lower loss are insufficient for the current trained state to retrieve the taught answer as its own winner. The remaining problem cannot be summarized as a dead teaching interface, but this assay does not separate initial under-acquisition from subsequent interference or forgetting.

## 4. Cross-context effects and the empty-input null

For trained-state matched B:

- Resetting/re-prompting the same source still improved loss at every site; the mean change was −0.19637 bits under that distinct timing/adaptation path.
- Same-target peers improved in 10/16 cases; four changes were zero and two were tiny increases of about 5.5e-11 bits. The mean change was −0.01006 bits.
- Different-target peers were harmed in 10/16 cases; their mean change was +0.000535 bits. Some other peers benefited, so this is not uniformly harmful transfer.

Peer targets and byte prefixes were chosen from already taught training data before native testing. Their KC overlap was often small; selecting them by same/different next symbol does not make them semantic neighbors. These effects are address-sharing/interference diagnostics, not rule learning or E3 generalization. Exact-key or frequency mechanisms remain valid countermodels for taught-cue improvement.

The three Buster primary anchors all had zero response at the prespecified empty-prefix calibration source: FE-PERM and the frozen positive-activation encoder had no active KC there. The gradient was not reversed; no active address was available. This is a concrete interface limitation, not an explanation of every nonempty addition prompt or the entire V150 loss gap.

## 5. Code-only versus native-only changes

The ancillary eight-case C-final factorial used the saved learned code, frozen except in its declared code-update branches. Mean immediate self-loss changes:

| C-final parent | Native only | Code only | Joint |
|---|---:|---:|---:|
| Bryant | −0.074074 | −0.000553 | −0.074625 |
| Buster | −0.108359 | −0.000566 | −0.108921 |

Mean projected code steps were 0.00150 and 0.00161 in Frobenius norm. Projection changed the raw step measurably; its largest norm difference was 0.000844. Mean self interaction was only 2.0e-6 and 3.3e-6 bits. The code step helped slightly at this dose, while the native write supplied most immediate improvement.

This is direct one-step evidence, stronger than reasoning from historical net drift alone. It does not prove learned code is necessary, diagnose a unique cold-start failure, or establish that increasing code rate/norm is safe. V150's better BIAS control remains relevant. Per-position RMS/span/probability bounds are retained in the raw local factorial records; no bound on all positions is inferred from an average.

## 6. The three cycles and independent audit

1. **Design/interface audit:** frozen V151 roster, sources and assets were checked. Before any target, numerical tolerances and gain floors were recorded. Finite-difference output gradients agreed to 6.99e-10; all 257 classes were mathematically reachable. Clone isolation, no target feeding, exact zero/passive agreement and native sign checks passed. A separately paid cross-process native resume preserved the state and warmup counter.
2. **Bounded test:** all 36 primary operator sites, 12 calibration sites, 24 scored sites and eight C-factorial sites completed. All six gains were locked before scored interventions. No fitted native inverse/operator, additional training pass, coefficient clipping, changed gate, replacement site or retry was used.
3. **Artifact audit/revision:** an independent reducer reconstructed finite-difference matrices, delivered coefficients, losses, class choices and factorial effects from saved vectors. Largest reduction disagreement: 5.56e-17. Five corrupted-roster fixtures were rejected. Event ledgers reconciled the exposure/API counts. Locked sources and all six original V150 checkpoint hashes remained unchanged. The revision is REVISED_DESIGN.md.

No scientific or technical execution failed, and no learning was retried. Runtime is authoritative in STATUS.json; the original design remains a design-delivery snapshot. The generic arithmetic design pointer was not changed.

## 7. Counts and compute

| Item | Actual |
|---|---:|
| Operator pulse branches | 2,304 |
| All event branches, including controls/preflight | 2,550 |
| Real-target applications | **126** |
| Primary calibration / scored target applications | 24 / 72 |
| C-factorial / technical target applications | 24 / 6 |
| Nonzero-coefficient compartment interventions | 4,194 |
| Signed API calls, including zero components | 40,208 |
| Native internal plastic evaluations | 8,388 |
| Native vector readouts | 10,142 |
| Passive prefix bytes/events | 52,058 |
| Record control events | 15,186 |
| Full sequential training histories / addition records | **0 / 0** |

A nonzero coefficient is an attempted compartment intervention, not a count of nonzero synaptic changes: the empty-input pulses produced no plastic effect. The 0/1 evaluations inside the signed interface are internal native computations, not extra externally observed answers. Prefix queries are sensory-only on disposable clones; their compute and repeated bytes are charged, not called free learning.

The supervised session used **265.953 summed child CPU seconds**, 271.531 wall seconds, and about **354.5 MiB peak RSS**, below the 1,800 CPU / 2,700 wall / 1 GiB limits. Independent audit and reporting reducers used about 0.38 additional measured CPU seconds in their bodies; startup, authoring and small shell inspections were not fully timed. No claim is made that these figures measure all project energy.

Prompt processing accounted for about 211.44 CPU seconds; native readouts 23.18, intervention calls 14.33, top-level clone calls 3.00, and state hashing 1.71. The known minimum mutable-array copy traffic was **55.16 GB**, excluding temporary copies hidden inside inherited clone methods. These are evaluator costs, not a scalable production architecture or proof of superior learning efficiency.

Raw vectors, per-branch records, technical checkpoints and logs remain local. Compact design, source, provenance, audit and result files are prepared for review. Nothing was pushed in this exercise.

## 8. Decision

**The native teaching interface is locally functional; the approximately twofold dose mismatch is confirmed and correctable for the sampled one-event responses. Trained own-answer retrieval remains inadequate.**

Retain the old core as the comparison, adopt neither a new learned code nor a calibrated controller from these results, and stop this text diagnostic. A separately designed next comparison should test whether correcting B's dose improves acquisition and retention through a finite addition learning history, then evaluate never-taught addition only if acquisition is meaningful. It must retain gradual warmup, generic byte/EOS output, local rules, matched exposure/compute accounting and the exclusion of arithmetic rules. No such run is authorized or executed here.
