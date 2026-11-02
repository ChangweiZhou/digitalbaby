# V176 operational revision — asset restoration before admission

The original model, data, controls, budgets and scientific gates remain unchanged. The observed failure is preflight availability only.

Add an explicit preparation stage P−1 before a future C0: recover original corpus/source files exclusively from f4146856c3241d94f7fa948e0b7d30fa0b869bc1; save exact bytes in an execution-local asset directory; verify every corpus SHA256 from the frozen SPEC and every pinned source blob; record recovery provenance and runtime closure. Do not normalize text, change encodings or fall back to another edition. No model construction or training occurs during recovery.

A later C0 may use only verified recovered assets. The previous failed preflight remains preserved, C1/C2 are still unexecuted, and no scientific parameter is revised. This operational clarification is not a successful cycle or a new global experiment version.
