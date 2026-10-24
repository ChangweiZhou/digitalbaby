# Workstream A4 — quantitative bridge record

Qualified blocks: **8**; all qualified: **True**.

No automatic mechanism verdict is emitted.

## Block-mean checkpoint-weighted quantities

- `exact_phi_D`: `-9.902222116818454`
- `linear_phi_D`: `-5.556602952500803`
- `nonlinear_residual`: `-4.345619164317652`
- `raw_centering.mean_contribution`: `-5.078316948471901`
- `raw_centering.actual_centered`: `-0.47828600402877525`
- `raw_centering.independent_rms`: `2.7571674437981835`
- `raw_centering.temporal_order_z`: `-2.163057249717708`
- `role_canonical_centering.mean_contribution`: `-0.04949652460002765`
- `role_canonical_centering.actual_centered`: `-5.5071064279006485`
- `role_context_centering.mean_contribution`: `-5.435472491647089`
- `role_context_centering.actual_centered`: `-0.12113046085358639`

## Interpretation order

1. Verify exact prior-D/H regression and D-source reconstruction.
2. Compare exact symmetric D attribution with the two-baseline linear bridge.
3. Compare raw, role-canonicalized, and role×context common-mode contributions.
4. Compare actual centered accumulation with the independent RMS and temporal-order statistic.
5. Use effective read/write dimension only as a measured geometric quantity, never as a hard-coded target.
6. Select no intervention until the preceding decomposition has been reviewed block by block.
