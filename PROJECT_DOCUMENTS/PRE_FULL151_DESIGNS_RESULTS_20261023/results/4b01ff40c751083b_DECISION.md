# Workstream A3 geometry audit — decision record

Qualified blocks: **8**; all qualified: **True**.

This runner intentionally emits **no automatic mechanism verdict**. No threshold was invented to label geometry as good/bad.

## Block-mean descriptive quantities

- `rho_exact`: `0.11070782014809075`
- `rho_linear`: `0.11068290783839055`
- `destructive_cross_fraction`: `0.35356711335676627`
- `code_abs_cos_offdiag_mean`: `0.06471693610133666`
- `support_overlap_offdiag_mean`: `0.11767091306686961`
- `u_proto_abs_cos_diff_fact_mean`: `0.057961521671439246`
- `w_proto_abs_cos_diff_fact_mean`: `0.05676372965315095`
- `foreign_to_target_exact_mean`: `-0.00024816701380828545`

## Interpretation order

1. Inspect static sparse-code overlap relative to the independent-support null.
2. Compare raw-update (`u`) prototype overlap with actual slow-write (`w`) prototype overlap.
3. Inspect exact slow-only cross effects and `rho_exact`; use `r^T w` only to diagnose linearity/clipping.
4. Inspect foreign-to-target effects as a local bridge to the previously established D/H result.
5. Only after those facts are reviewed decide whether representation, write geometry, reader alignment, or temporal dynamics deserves the next experiment.

The run does not authorize a new trace, topology, slot/chamber, selector, or policy family by itself.
