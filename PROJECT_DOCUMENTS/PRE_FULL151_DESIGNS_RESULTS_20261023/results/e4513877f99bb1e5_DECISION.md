# Workstream A4 — quantitative bridge record

Qualified blocks: **1**; all qualified: **False**.

No automatic mechanism verdict is emitted.

## Block-mean checkpoint-weighted quantities

- `exact_phi_D`: `-11.16882630974606`
- `linear_phi_D`: `-6.127501396557381`
- `nonlinear_residual`: `-5.041324913188677`
- `raw_centering.mean_contribution`: `-5.5604388778241365`
- `raw_centering.actual_centered`: `-0.5670625187331486`
- `raw_centering.independent_rms`: `2.64483161853566`
- `raw_centering.temporal_order_z`: `-2.458362963659849`
- `role_canonical_centering.mean_contribution`: `-0.0029953102598722684`
- `role_canonical_centering.actual_centered`: `-6.124506086297415`
- `role_context_centering.mean_contribution`: `-5.849006603017881`
- `role_context_centering.actual_centered`: `-0.2784947935394044`

## Interpretation order

1. Verify exact prior-D/H regression and D-source reconstruction.
2. Compare exact symmetric D attribution with the two-baseline linear bridge.
3. Compare raw, role-canonicalized, and role×context common-mode contributions.
4. Compare actual centered accumulation with the independent RMS and temporal-order statistic.
5. Use effective read/write dimension only as a measured geometric quantity, never as a hard-coded target.
6. Select no intervention until the preceding decomposition has been reviewed block by block.
