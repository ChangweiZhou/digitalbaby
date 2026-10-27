# Cycle 1 — design and implementation audit

The user authorized the bounded three-cycle exercise on 27 October 2026. The original design remains intact.

1. Candidate learning uses the two specified generic error vectors, gradual warmup and four fixed gains. Counterfactual gain matching is restricted to discarded evaluator clones.
2. Native `values()` is prospective. The implementation will compare written and inhibited teacher calls at identical event time and then use the same prospective query time/code on each. It will independently reconstruct that reader and check equality against `FastMemory.values()` before admission.
3. OBS executes signed teacher calls with `write=False`, including predictions, observations, counter increments and identical clocks. IDLE uses measured elapsed seconds from the common OLD checkpoint, never record counts.
4. Common OLD is one history per world. Main forks are correlated continuations. Probe clones never re-enter learning. All exact mappings are reinforced: evidence is E0/E1/E1-R, never E3 or arithmetic generalization.
5. The original design leaves generator salt/hash names, scalar bracket samples, numerical floor and resource safety multiplier unspecified. The execution spec fixes these before model queries: SHA256 WCFI-V1 namespaces; endpoints and midpoint followed by bounded bisection (20 evaluations total); absolute dose floor from zero-write residual plus 1e-12 floating-point guard; conservative timing admission at 1.5 times measured full-roster estimate, within original caps. No efficacy threshold is added.
6. Generation uses exactly four decisions and reports CAP if no EOS. It does not inherit the older wrapper's extra fifth decision.
7. A failed address preflight, calibration or resource admission terminates the planned science with preserved receipts. No difficult world, dose or failed trajectory is replaced.

No code changes are made to historical native sources, arithmetic designs or adoption pointers.
