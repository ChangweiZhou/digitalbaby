# Byte value persistence V9.6: source-defined B transport and fixed-address storage

Frozen before prospective evaluation: 2026-09-26. This is a constructive comparison, not a new audit of V9.5.

## Parent and candidates

Use the V9.5 four-cohort byte world generator, feeding schedule, native reader, teaching dose, and all old/new probes unchanged. All arms start from the same V9.2 predictor/value checkpoint and see identical bytes, rewards, and times. Only the native value transition differs.

1. **REF:** exact V9.5 native value learner. A single engineering world must reproduce its archived pair margins exactly.
2. **B_TRANSFER:** transport the historical A8/B local baseline correction with its original frozen PN-calibrated vector (no refit): after the FULL_151 native event, let `u=rawalpha`, `p=allocation_fraction`, `h=adapted_eligibility(pre_adapt,x,dt)`, `c=p*h*b`. Replace raw slow/fast writes by `rawslow-c` and `rawfast+c`, patching post-decay states by `-c*ds` and `+c*df`. This is a transfer test; the byte parent's existing non-neutral allocation is *not* the B correction. The source vector's 5,177 float64 values and its file SHA are frozen and charged as fixed parameters.
3. **FB4_TRANSFER:** B_TRANSFER plus four alpha stores using the historical fixed eight-centre KC router and its frozen four-bank grouping at seed 2026092101. An active event selects its bank from the current 5,177-dimensional KC code; readout selects by the queried KC code. Nonselected banks passively decay once per elapsed interval. No world identity, cohort label, reward, future query, or evaluator label enters routing. This comparator changes both storage volume and isolation, so any gain is not attributed to either alone.

All arms retain the original independent engineered byte predictor and update it on the same stream. Predictor identity across write/no-write branches and across candidates is a validity check, not evidence of predictor-to-value integration.

## Scientific comparison

Engineering-only world 401 is used for REF reproduction, B arithmetic, bank one-store identity, clone/query purity, and resource checks. No candidate parameter is selected by its performance. After those pass, evaluate **80 untouched worlds 701–780**, paired across the three arms. The reserved index range is disjoint from V9.5 science (401–656). Each arm includes V9.5's WWWW and matched NNNN histories; W/N same-prestate shadows and WWWN/NNNW factorial branches remain available. Technical failures abort the prospective run.

Primary displayed vector for every candidate-minus-REF comparison: absolute all-write old choice at D4; new-cohort choice at each E_j and D4; old/new analog margins; old-pair loss counts. Show paired 95% t intervals and signs by world for each component. The own WWWW−NNNN contrast is diagnostic and cannot substitute for absolute candidate-minus-REF performance. Do not declare a winner from an unplanned scalar score or a merely-above-chance new-learning gate. Report predictor checks, exact no-write baselines, route occupancy, collision frequency, additional persistent bytes, and runtime.

Sample size is intended to distinguish large (~10 percentage-point) shifts, not establish 1-point noninferiority. If paired SD were the V9.5 write-harm SD of 22.4 pp, 80 worlds would give an approximate 95% half-width of 5 pp; candidate-minus-parent SD is not known in advance. Every estimate and interval is reported, including null or harmful results. No post hoc sample increase or parameter tuning on 701–780.

## Frozen source distinctions and safety checks

- Historical B operation: `WorkstreamA8_baseline_transfer_20260920/engine.py` and `WorkstreamBKR_v5_20260920/model_bkr.py`. Frozen vector: `CORE_RECOVERY_workstreamA8/baseline_transfer_v1/calibration/frozen.npz`. Historical adapted-eligibility equation comes from `local_payload_balance_candidate_20260920.zip`; the implementation copies that exact equation and records the archive hash.
- Historical bank semantics: `WorkstreamFB1_v3_2_20260921/fb_model.py` and `fb_common.py`. Routing centers and mapping are source-derived; no byte-world fitting.
- Byte parent: `BYTE_CORE_V9_5/run/lifetime_assay.py`, `BYTE_CORE_V9_4/interference_assay.py`, `BYTE_CORE_V9_2/brain_v92.py`, and inherited native FULL_151 code. V9.5's frozen source/receipts are not modified.
- Future receipts treat three pair margins and their identities as canonical. Recompute choice, mean margin, noisy choice, and pair losses before aggregation. Include synthetic corruption regression tests from `BYTE_CORE_V9_5/audit_20260926`.
- Stop rather than interpret if REF does not reproduce the archived world, B violates the native write identity, the one-bank wrapper fails B identity on a nonzero state, a query changes model state, predictors diverge on the same stream, or source hashes change.

This experiment tests a concrete transferred operation and a resource-expanding comparator. It cannot establish fly physiology, isolate capacity from routing, or complete PN-cue A11.
