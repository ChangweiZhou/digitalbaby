# U045 content core development results

Three design–audit–test–revision cycles are complete. **HOLD_B_CURRENT_INSTANCE**: the candidate is technically executable, but its delayed new-content accuracy in the final two development worlds was 87.5%, versus 100% for the adopted native baseline A. One B world scored 75%. This is a development warning, not a confirmatory estimate or adoption result.

Completed work comprises **four DEV worlds, 32 training histories and 32,000 training bytes; zero science worlds**. The first DEV world used a shortened history. The final two worlds contributed 44,800 audited rows; all four main receipts contributed 70,080 rows. Synthetic unit and checkpoint fixtures are separate from those counts.

- [Three-cycle findings](THREE_CYCLE_REPORT.md), [qualification](QUALIFICATION.json) and [final development audit](FINAL_AUDIT.json).
- [Experimental protocol](EXPERIMENT.md), [design review](AUDIT_OF_DESIGN.md) and [source lock](SOURCE_LOCK.json).
- [Cycle records](cycles/), including all four compressed world receipts, source snapshots, the negative first cycle, and the resolved auditor and reducer failures.
- [Publication record](PUBLICATION.json).

B replaces the content writer/readout with a bounded local predictive-residual matrix. It retains A's fixed input coordinates, but it does not preserve all native Full151 plasticity. Its speed advantage includes shared encoding and avoidance of inherited clone/API costs. Neither speed nor exact-key recall establishes an addition rule, held-out transfer, or a generally superior persistent core.

## Dependencies and archived fixtures

Keep this folder within a checkout of `ChangweiZhou/digitalbaby`. Imports use the sibling [autonomous-observation baseline](../autonomous_observation_v1_20261007/) and its sibling [native assets](../r_center_core_v1/); no separate large asset copy is required. The recorded environment is Python 3.11.5, numpy 2.2.6, scipy 1.14.1 and numba 0.61.2.

Files under `cycles/*/preflight/` and `postflight/` are technical checkpoint fixtures. Intentionally malformed NPZ files test rejection; they are not failed science trajectories. JIT caches and Python bytecode are excluded from publication. Source snapshots and failure records are retained unchanged. This publication does not rerun trajectories.

The current code admits only the declared DEV worlds. The prospective 32-world science roster in the protocol was not executed and remains on hold. The user's subsequent V2 milestone is discussed separately in [the addition proposal](../PROJECT_DOCUMENTS/V2_ADDITION_MILESTONE_20261007.md); it does not change these results or their frozen source identity.
